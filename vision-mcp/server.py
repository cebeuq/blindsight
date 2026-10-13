"""vision-mcp: the blindsight "ask the image" MCP server.

The text model is text-only, so imgfix (behind the router) replaces every image/document block in a request with a
stable text reference ([image img_<16hex> | ...] / [pdf pdf_<16hex> | ...]) and stores the bytes in the shared /store
volume. The text model then calls the ask_image tool here; we load the stored image (or render a PDF page), optionally
crop/zoom a region, downscale to <= 1600 px, and ask the vision model (any OpenAI-compatible chat/completions
endpoint) a targeted question. Store layout (written by imgfix, read-only here): /store/img/<id>.<png|jpg|gif|webp>,
/store/pdf/<id>.pdf, /store/meta/<id>.json.

Transport: MCP streamable HTTP, stateless, JSON responses, on :8102 at /mcp. GET /health -> 200 when /store is readable.
Tool failures come back as plain "ask_image error: ..." strings (never exceptions) so the text model can read and react
to them.

Env: VISION_MODEL (required), VISION_BASE_URL (with or without /v1), VISION_API_KEY (optional Bearer token, never
logged), VISION_EXTRA_BODY (optional JSON object merged into every vision request; it may not set
model/messages/stream), VISION_TIMEOUT (60), STORE_DIR (/store), ASK_CONCURRENCY (4), DECODE_CONCURRENCY (2),
MCP_ALLOWED_HOSTS (Host-header allowlist; "*" = off), HOST, PORT (8102), LOG_LEVEL (INFO). A bad VISION_MODEL /
VISION_EXTRA_BODY stops the server at startup with a clear error (exit 2).
"""

import base64
import io
import json
import logging
import math
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any

import anyio
import anyio.to_thread
import httpx2
import uvicorn
from mcp.server.mcpserver import MCPServer
from mcp.server.transport_security import TransportSecuritySettings
from mcp_types import ToolAnnotations
from PIL import Image, ImageOps
from pydantic import Field
from starlette.requests import Request
from starlette.responses import PlainTextResponse

log = logging.getLogger("vision-mcp")
Image.MAX_IMAGE_PIXELS = None  # Pillow's open-time bomb check would reject huge JPEGs before draft(); Config.max_pixels guards decode

ID_RE = re.compile(r"(img|pdf)_[0-9a-f]{16}")  # used with fullmatch: no path tricks, no trailing newline
URL_ID_RE = re.compile(r"url_[0-9a-f]{16}")  # remote images the router never fetched
IMG_EXTS = ("png", "jpg", "gif", "webp")  # router writes img/<id>.<ext> with ext from the media type
IMG_FORMATS = ("PNG", "JPEG", "GIF", "WEBP")  # the only Pillow decoders tried on stored bytes (no TIFF/PSD/BMP/...)
POPPLER_MEM = 768 << 20  # address-space cap per poppler process (prlimit --as), so one PDF cannot OOM the container

MAX_EDGE = 1600  # long edge sent to the vision model
ZOOM_EDGE = 1024  # a cropped region smaller than this is upscaled (max ZOOM_MAX x) so the vision model gets more tokens on it
ZOOM_MAX = 4.0
PDF_DPI = 110  # whole-page render; a region renders at up to PDF_DPI * ZOOM_MAX for real detail instead of upscaling
MAX_QUESTION = 4000  # chars; longer question/context is truncated (keeps the vision prompt bounded)
MAX_CONTEXT = 4000

SYSTEM_PROMPT = (
    "You are the eyes of another AI assistant that cannot see images. You receive one image (sometimes a zoomed crop "
    "or a rendered PDF page) and a question about it. Rules:\n"
    "- Answer only from what is visible in this image. Never guess, infer hidden content, or fill in what is typical.\n"
    "- Quote visible text exactly as written (same spelling, case, punctuation, numbers), in quotes.\n"
    '- If something asked about is not in the image, say "[not visible]". If it is there but cannot be read reliably, '
    'say "[illegible]" (for partly readable text, quote the readable part and mark the rest [illegible]).\n'
    "- The caller's context may be wrong or incomplete. Trust the image over it and point out any contradiction.\n"
    "- Be concise and specific: answer the question first, then only the details needed to support it."
)

TOOL_DESCRIPTION = (
    "Look at an image (or a PDF page) from this conversation and answer a question about it. You cannot see images "
    "yourself: each one appears in the conversation as a text reference like "
    '[image img_0123456789abcdef | 1280x720 png | short caption], and each PDF as [pdf pdf_0123456789abcdef | N pages] '
    "followed by its extracted text. Pass that id as image_id and ask one specific, neutral question (e.g. "
    '"What does the error dialog say, word for word?", "List the column headers of the table"). Do not put the answer '
    "you expect into the question or context: a separate vision model answers from the pixels only and may be steered "
    "by hints. Follow-up questions about the same image are fine and encouraged; use region to zoom into small "
    "details. For exact long text of a PDF, use the extracted text already in the conversation; use page only for "
    "layout, figures, tables, handwriting or scanned pages."
)

# Tool parameter schemas. Like TOOL_DESCRIPTION they are part of the text model's prompt (tool list), so keep them
# byte-stable: any change invalidates the text model's prefix cache for every running session.
ImageIdArg = Annotated[str, Field(description='Id from the conversation, e.g. "img_0123456789abcdef" or "pdf_0123456789abcdef".')]
QuestionArg = Annotated[str, Field(description="One specific, neutral question about what is visible. Do not include the answer you expect.")]
ContextArg = Annotated[str, Field(description="Optional background that helps interpret the image (what the user is doing). "
                                               "Facts only, no expected answers.")]
PageArg = Annotated[int | None, Field(description="PDF page number (1-based, default 1). Only for pdf_ ids.")]
RegionArg = Annotated[str | None, Field(description='Optional zoom: "x0,y0,x1,y1" as fractions 0..1 of width/height from the '
                                                     'top-left, e.g. "0,0,0.5,0.5" = top-left quarter.')]


DEFAULT_ALLOWED_HOSTS = "127.0.0.1:*,localhost:*,vision-mcp:*"
RESERVED_BODY_KEYS = ("model", "messages", "stream")  # VISION_EXTRA_BODY may not set these


class ConfigError(Exception):
    """A startup configuration problem, reported to the operator before the server starts."""


def parse_extra_body(s: str | None) -> dict[str, Any]:
    """VISION_EXTRA_BODY -> dict merged into every vision request ("" -> {}). Raises ConfigError: not JSON, not an
    object, or a reserved key (the request's model/messages/non-streaming are fixed)."""
    s = (s or "").strip()
    if not s:
        return {}
    try:
        d = json.loads(s)
    except ValueError as e:
        raise ConfigError(f"VISION_EXTRA_BODY is not valid JSON ({e}); expected an object like "
                          '{"chat_template_kwargs":{"enable_thinking":false}}') from None
    if not isinstance(d, dict):
        raise ConfigError(f"VISION_EXTRA_BODY must be a JSON object, got {type(d).__name__}")
    bad = sorted(k for k in d if k in RESERVED_BODY_KEYS)
    if bad:
        raise ConfigError(f"VISION_EXTRA_BODY may not set {', '.join(bad)} (fixed by blindsight)")
    return d


@dataclass
class Config:
    store: Path = Path("/store")
    vision_base_url: str = "http://host.docker.internal:8001/v1"
    vision_model: str = ""  # required in from_env (no sensible default)
    vision_api_key: str = ""  # optional Bearer token (e.g. vLLM --api-key, hosted APIs); never logged
    vision_timeout: float = 60.0
    extra_body: dict[str, Any] = field(default_factory=dict)  # VISION_EXTRA_BODY, merged into every request
    concurrency: int = 4  # simultaneous ask_image calls (bounds vision model load)
    decode_concurrency: int = 2  # simultaneous image decodes/renders (bounds memory: ~0.5 GB worst case each)
    max_pixels: int = 50_000_000  # refuse to decode larger images (after JPEG draft-scaling). Shared contract:
    # MUST equal imgfix MAX_PIXELS, so every image that gets an [image img_...] reference can be inspected here.
    pdf_max_pixels: int = 40_000_000  # cap rendered page size (huge page formats get a lower dpi)
    allowed_hosts: list[str] = field(default_factory=list)  # Host-header allowlist (DNS-rebinding guard); [] = off

    @classmethod
    def from_env(cls) -> Config:
        """Raises ConfigError (missing VISION_MODEL, bad VISION_EXTRA_BODY)."""
        model = os.environ.get("VISION_MODEL", "").strip()
        if not model:
            raise ConfigError("VISION_MODEL is not set (the model name your vision server serves)")
        hosts = os.environ.get("MCP_ALLOWED_HOSTS", "").strip() or DEFAULT_ALLOWED_HOSTS  # unset/empty: guard on
        allowed = [h.strip() for h in hosts.split(",") if h.strip()] or DEFAULT_ALLOWED_HOSTS.split(",")
        return cls(
            store=Path(os.environ.get("STORE_DIR", "/store")),
            vision_base_url=os.environ.get("VISION_BASE_URL", "").strip() or cls.vision_base_url,
            vision_model=model,
            vision_api_key=os.environ.get("VISION_API_KEY", "").strip(),
            vision_timeout=float(os.environ.get("VISION_TIMEOUT", cls.vision_timeout)),
            extra_body=parse_extra_body(os.environ.get("VISION_EXTRA_BODY", "")),
            concurrency=int(os.environ.get("ASK_CONCURRENCY", cls.concurrency)),
            decode_concurrency=int(os.environ.get("DECODE_CONCURRENCY", cls.decode_concurrency)),
            allowed_hosts=[] if hosts == "*" else allowed,
        )


class AskError(Exception):
    """A problem to report back to the caller as text (unknown id, bad region, vision model down, ...)."""


# ---- ids, store, region ---------------------------------------------------------------------------------------


def valid_id(image_id: str) -> bool:
    return isinstance(image_id, str) and ID_RE.fullmatch(image_id) is not None


def find_image(store: Path, image_id: str) -> Path | None:
    """Stored file for a *valid* id: img/<id>.<ext> (first known ext that exists) or pdf/<id>.pdf."""
    if image_id.startswith("pdf_"):
        p = store / "pdf" / f"{image_id}.pdf"
        return p if p.is_file() else None
    for ext in IMG_EXTS:
        p = store / "img" / f"{image_id}.{ext}"
        if p.is_file():
            return p
    return None


def parse_region(region: str) -> tuple[float, float, float, float]:
    """'x0,y0,x1,y1' fractions of width/height (0..1, top-left origin). Lenient on brackets/spaces."""
    parts = [p for p in re.split(r"[,\s]+", region.strip().strip("()[]{}").strip()) if p]
    try:
        vals = tuple(float(p) for p in parts)
    except ValueError:
        vals = ()
    if len(vals) != 4 or not all(math.isfinite(v) and 0.0 <= v <= 1.0 for v in vals):
        raise AskError(f'region must be "x0,y0,x1,y1" with fractions 0..1 (e.g. "0.5,0,1,0.5" = top-right quarter); got {region!r}')
    x0, y0, x1, y1 = vals
    if not (x0 < x1 and y0 < y1):
        raise AskError(f"region needs x0 < x1 and y0 < y1; got {region!r}")
    return x0, y0, x1, y1


def crop_box(w: int, h: int, r: tuple[float, float, float, float]) -> tuple[int, int, int, int]:
    """Pixel box (left, top, right, bottom) covering the fractional region: outward-rounded, clamped, >= 1 px.
    The 1e-9 slack keeps float noise (0.3*1000 = 300.00000000000006) from adding a pixel."""
    x0, y0, x1, y1 = r
    left, top = math.floor(x0 * w + 1e-9), math.floor(y0 * h + 1e-9)
    right, bottom = math.ceil(x1 * w - 1e-9), math.ceil(y1 * h - 1e-9)
    left, top = min(max(left, 0), w - 1), min(max(top, 0), h - 1)
    right, bottom = min(max(right, left + 1), w), min(max(bottom, top + 1), h)
    return left, top, right, bottom


# ---- image preparation (blocking; runs in a worker thread) --------------------------------------------------------


def to_rgb(im: Image.Image) -> Image.Image:
    """Upright RGB. Transparency is flattened onto white: a plain RGBA->RGB drop (what servers do) turns
    black-on-transparent screenshots into black-on-black."""
    ImageOps.exif_transpose(im, in_place=True)
    if im.mode in ("RGBA", "LA", "PA") or (im.mode == "P" and "transparency" in im.info):
        im = im.convert("RGBA")
        bg = Image.new("RGB", im.size, (255, 255, 255))
        bg.paste(im, mask=im.getchannel("A"))
        return bg
    return im if im.mode == "RGB" else im.convert("RGB")


def fit(im: Image.Image, region: tuple[float, float, float, float] | None) -> Image.Image:
    """Crop to region (zooming small crops up to ZOOM_EDGE), then downscale so the long edge <= MAX_EDGE."""
    if region is not None:
        im = im.crop(crop_box(im.width, im.height, region))
        long_edge = max(im.size)
        if long_edge < ZOOM_EDGE:
            s = min(ZOOM_MAX, ZOOM_EDGE / long_edge)
            im = im.resize((max(1, round(im.width * s)), max(1, round(im.height * s))), Image.Resampling.LANCZOS)
    if max(im.size) > MAX_EDGE:
        s = MAX_EDGE / max(im.size)
        im = im.resize((max(1, round(im.width * s)), max(1, round(im.height * s))), Image.Resampling.LANCZOS)
    return im


def png_bytes(im: Image.Image) -> bytes:
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


def load_image(path: Path, region, max_pixels: int) -> bytes:
    """Stored raster image -> PNG bytes ready for the vision model (first frame of animations)."""
    try:
        with Image.open(path, formats=IMG_FORMATS) as im:
            if im.format == "JPEG":
                # Decode big JPEGs (phone photos) at 1/2..1/8 scale in the DCT domain, keeping at least the pixels the
                # (possibly cropped) output needs. Conservative for rotated images: uses the smaller region side.
                need = math.ceil(MAX_EDGE / (min(region[2] - region[0], region[3] - region[1]) if region else 1.0))
                im.draft(im.mode, (need, need))
            if im.width * im.height > max_pixels:
                raise AskError(f"image is too large to inspect ({im.width}x{im.height})")
            im.load()
            return png_bytes(fit(to_rgb(im), region))
    except AskError:
        raise
    except Exception as e:  # truncated/corrupt/unsupported data
        raise AskError(f"could not decode the stored image ({type(e).__name__})") from e


def _run(cmd: list[str], timeout: float) -> subprocess.CompletedProcess[bytes]:
    try:
        return subprocess.run(["prlimit", f"--as={POPPLER_MEM}", "--", *cmd], capture_output=True, timeout=timeout,
                              env={"PATH": os.environ.get("PATH", ""), "LANG": "C"})
    except subprocess.TimeoutExpired as e:
        raise AskError(f"{cmd[0]} timed out after {timeout:g}s on the stored PDF") from e


def pdf_info(path: Path, page: int) -> tuple[int, float, float]:
    """(page count, page width pt, page height pt) via pdfinfo (the ranged call fails for out-of-range pages, so the
    page count comes from a plain call first)."""
    r = _run(["pdfinfo", str(path)], 30)
    m = re.search(r"^Pages:\s+(\d+)", r.stdout.decode("utf-8", "replace"), re.M)
    if r.returncode != 0 or not m:
        raise AskError("could not read the stored PDF (damaged or password-protected)")
    pages = int(m.group(1))
    if not 1 <= page <= pages:
        raise AskError(f"page {page} is out of range: this PDF has {pages} page{'s' if pages != 1 else ''}")
    out = _run(["pdfinfo", "-f", str(page), "-l", str(page), str(path)], 30).stdout.decode("utf-8", "replace")
    s = re.search(rf"^Page\s+{page} size:\s+([\d.]+) x ([\d.]+) pts", out, re.M)
    w, h = (float(s.group(1)), float(s.group(2))) if s else (612.0, 792.0)  # assume US letter if unreported
    return pages, w, h


def pdf_dpi(w_pt: float, h_pt: float, region, max_pixels: int) -> int:
    """PDF_DPI for a whole page; up to PDF_DPI*ZOOM_MAX for a region (real detail); capped by the pixel budget."""
    dpi = PDF_DPI
    if region is not None:
        dpi = PDF_DPI * min(ZOOM_MAX, 1.0 / max(region[2] - region[0], region[3] - region[1]))
    cap = 72.0 * math.sqrt(max_pixels / max(w_pt * h_pt, 1.0))
    return max(10, int(min(dpi, cap)))


def render_pdf_page(path: Path, page: int, region, max_pixels: int) -> bytes:
    """PDF page -> PNG bytes ready for the vision model (pdftoppm, crop box = what viewers show)."""
    _, w, h = pdf_info(path, page)
    dpi = pdf_dpi(w, h, region, max_pixels)
    r = _run(["pdftoppm", "-f", str(page), "-l", str(page), "-r", str(dpi), "-cropbox", "-png", "-singlefile", str(path)], 60)
    if r.returncode != 0 or not r.stdout:
        raise AskError(f"could not render page {page} of the stored PDF")
    try:
        with Image.open(io.BytesIO(r.stdout), formats=("PNG",)) as im:
            im.load()
            return png_bytes(fit(to_rgb(im), region))
    except Exception as e:
        raise AskError(f"could not decode the rendered PDF page ({type(e).__name__})") from e


def prepare(cfg: Config, image_id: str, page: int | None, region) -> bytes:
    path = find_image(cfg.store, image_id)
    if path is None:
        raise AskError(
            f"unknown image_id {image_id!r}: nothing stored under that id. Use an id exactly as it appears in an "
            "[image img_...] or [pdf pdf_...] reference in this conversation."
        )
    if not os.access(path, os.R_OK):  # e.g. the writer created it 0600 under another uid
        raise AskError(f"{image_id} is stored but not readable by vision-mcp (check /store file permissions)")
    if image_id.startswith("pdf_"):
        return render_pdf_page(path, page or 1, region, cfg.pdf_max_pixels)
    if page not in (None, 1):
        raise AskError("page applies only to pdf_ ids; an image has a single page")
    return load_image(path, region, cfg.max_pixels)


# ---- vision model -------------------------------------------------------------------------------------------------


def chat_url(base: str) -> str:
    """Accept VISION_BASE_URL with or without the OpenAI '/v1' suffix."""
    base = base.rstrip("/")
    return f"{base}/chat/completions" if base.endswith("/v1") else f"{base}/v1/chat/completions"


def user_prompt(image_id: str, question: str, context: str, page: int | None, region) -> str:
    what = f"PDF {image_id}, page {page or 1}" if image_id.startswith("pdf_") else f"Image {image_id}"
    if region is not None:
        what += (
            f" - zoomed crop of the region x {region[0]:g}..{region[2]:g}, y {region[1]:g}..{region[3]:g} "
            "(fractions of width/height); content at the crop edges may be cut off"
        )
    parts = [what + "."]
    if context.strip():
        parts.append(f"Context from the caller (may be wrong; the image wins):\n{context.strip()[:MAX_CONTEXT]}")
    parts.append(f"Question:\n{question.strip()[:MAX_QUESTION]}")
    return "\n\n".join(parts)


def clean_answer(text: str) -> str:
    """Drop any reasoning block (a reasoning model may emit one unless VISION_EXTRA_BODY turns thinking off)."""
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    return text.strip()


def request_body(cfg: Config, png: bytes, prompt: str) -> dict[str, Any]:
    """OpenAI chat/completions body; cfg.extra_body is merged on top but never replaces the reserved keys."""
    body: dict[str, Any] = {
        "model": cfg.vision_model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(png).decode()}},
                    {"type": "text", "text": prompt},
                ],
            },
        ],
        "temperature": 0.2,
        "max_tokens": 700,
    }
    body.update({k: v for k, v in cfg.extra_body.items() if k not in RESERVED_BODY_KEYS})
    return body


async def ask_vision(cfg: Config, png: bytes, prompt: str) -> str:
    body = request_body(cfg, png, prompt)
    headers = {"Authorization": f"Bearer {cfg.vision_api_key}"} if cfg.vision_api_key else {}
    url = chat_url(cfg.vision_base_url)
    try:
        async with httpx2.AsyncClient(timeout=httpx2.Timeout(cfg.vision_timeout, connect=5.0), trust_env=False) as client:
            resp = await client.post(url, json=body, headers=headers)
    except httpx2.TimeoutException as e:
        raise AskError(f"the vision model did not answer within {cfg.vision_timeout:g}s ({type(e).__name__}); "
                       "try again or ask a narrower question") from e
    except httpx2.HTTPError as e:
        raise AskError(f"the vision model is unreachable at {url} ({type(e).__name__}); "
                       "it may be offline, try again later") from e
    if resp.status_code != 200:
        raise AskError(f"the vision model returned HTTP {resp.status_code}: {resp.text[:300]}")
    try:
        choice = resp.json()["choices"][0]
        content = choice["message"]["content"]
        if isinstance(content, list):  # some servers return content parts
            content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
        answer = clean_answer(content or "")
    except (ValueError, KeyError, IndexError, TypeError) as e:
        raise AskError(f"the vision model returned an unexpected response: {resp.text[:300]}") from e
    if not answer:
        raise AskError("the vision model returned an empty answer; try rephrasing the question")
    if choice.get("finish_reason") == "length":
        answer += "\n[answer cut off at the length limit; ask a narrower follow-up question for the rest]"
    return answer


# ---- MCP server ---------------------------------------------------------------------------------------------------


def build_server(cfg: Config) -> MCPServer:
    mcp = MCPServer(
        "vision-mcp",
        version="1.0.0",
        instructions="Images and PDFs in this conversation are replaced by [image img_...] / [pdf pdf_...] references. "
        "Use ask_image to look at them.",
    )
    lim: dict[str, Any] = {}  # anyio primitives need a running event loop, so they are created on the first call

    @mcp.tool(
        description=TOOL_DESCRIPTION,
        annotations=ToolAnnotations(title="Ask about an image", read_only_hint=True, open_world_hint=False),
        structured_output=False,
    )
    async def ask_image(
        image_id: ImageIdArg, question: QuestionArg, context: ContextArg = "", page: PageArg = None, region: RegionArg = None
    ) -> str:
        t0 = time.monotonic()
        try:
            if not valid_id(image_id):
                if isinstance(image_id, str) and URL_ID_RE.fullmatch(image_id):
                    raise AskError(f"{image_id} is a remote image URL that was never downloaded, so it cannot be inspected")
                raise AskError(f'invalid image_id {str(image_id)[:80]!r}: expected "img_" or "pdf_" followed by 16 lowercase hex digits')
            if not question or not question.strip():
                raise AskError("question is empty; ask a specific question about the image")
            if page is not None and page < 1:
                raise AskError(f"page must be >= 1; got {page}")
            box = parse_region(region) if region and region.strip() else None
            if not lim:
                lim["ask"] = anyio.Semaphore(max(1, cfg.concurrency))
                lim["decode"] = anyio.CapacityLimiter(max(1, cfg.decode_concurrency))
            async with lim["ask"]:
                png = await anyio.to_thread.run_sync(prepare, cfg, image_id, page, box, limiter=lim["decode"])
                answer = await ask_vision(cfg, png, user_prompt(image_id, question, context or "", page, box))
            log.info("ask_image %s page=%s region=%r q=%dch -> %dch in %.1fs",
                     image_id, page, region, len(question), len(answer), time.monotonic() - t0)
            return answer
        except AskError as e:
            log.warning("ask_image %.80r failed in %.1fs: %s", image_id, time.monotonic() - t0, e)
            return f"ask_image error: {e}"
        except Exception as e:  # never surface a traceback to the model
            log.exception("ask_image %.80r crashed", image_id)
            return f"ask_image error: internal error ({type(e).__name__}); try again"

    @mcp.custom_route("/health", methods=["GET"])
    async def health(_: Request) -> PlainTextResponse:
        ok = cfg.store.is_dir() and os.access(cfg.store, os.R_OK | os.X_OK)
        return PlainTextResponse("ok\n" if ok else f"store {cfg.store} not readable\n", status_code=200 if ok else 503)

    return mcp


def build_app(cfg: Config):
    """Starlette app: stateless streamable HTTP at /mcp (JSON responses) + /health."""
    security = (
        TransportSecuritySettings(
            enable_dns_rebinding_protection=True,
            allowed_hosts=cfg.allowed_hosts,
            allowed_origins=[f"http://{h}" for h in cfg.allowed_hosts],
        )
        if cfg.allowed_hosts
        else TransportSecuritySettings(enable_dns_rebinding_protection=False)
    )
    return build_server(cfg).streamable_http_app(
        streamable_http_path="/mcp", stateless_http=True, json_response=True, transport_security=security
    )


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    for noisy in ("httpx2", "httpcore2", "mcp.server.streamable_http", "mcp.server.streamable_http_manager"):
        logging.getLogger(noisy).setLevel(logging.WARNING)  # per-request chatter


def describe(cfg: Config) -> str:
    """Startup log line: where requests go and how; says whether an API key is set, never the key."""
    return (f"store={cfg.store} vision={chat_url(cfg.vision_base_url)} model={cfg.vision_model} "
            f"api_key={'set' if cfg.vision_api_key else 'unset'} extra_body_keys={','.join(sorted(cfg.extra_body)) or '-'} "
            f"allowed_hosts={','.join(cfg.allowed_hosts) or 'any'}")


if __name__ == "__main__":
    setup_logging(os.environ.get("LOG_LEVEL", "INFO"))
    try:
        cfg = Config.from_env()
    except ConfigError as e:
        log.critical("configuration error: %s", e)
        sys.exit(2)
    log.info("%s", describe(cfg))
    uvicorn.run(build_app(cfg), host=os.environ.get("HOST", "0.0.0.0"), port=int(os.environ.get("PORT", "8102")))
