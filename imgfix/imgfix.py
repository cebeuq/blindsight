"""imgfix: mitmproxy addon that keeps image/document blocks away from a text-only model (blindsight).

HAProxy sends here every POST whose body holds image/document blocks, or that it could not see whole (> its 32 MiB
buffer, or still arriving after 300 s). Every POST body with a "messages" list is scanned (any path, any model name)
and every request goes on to the text model (TEXT_UPSTREAM). Every image/document block is replaced by a
deterministic text reference, its raw bytes go to the shared /store volume, and the text model later calls the
vision-mcp ask_image tool, which asks the vision model about the stored file. Fails CLOSED: no image/document block
is ever forwarded to the text model (a block that cannot be handled becomes "[image unavailable: could not be
decoded]"; a body that cannot be handled at all gets HTTP 400 and is not forwarded).

Prefix-cache rule: Claude Code re-sends the whole history every turn, so the text for a given image must be
byte-identical forever. It is therefore built only from write-once store files (or, on first sight, from the
exact values then written to them); nothing time-, load- or process-dependent goes into it. A meta file that
exists but cannot be read gives the fixed "unavailable" text (never a fresh caption).

Store (shared contract with vision-mcp):
  /store/img/<id>.<png|jpg|gif|webp>   id = "img_" + sha256(raw)[:16]   raw bytes as received
  /store/pdf/<id>.pdf                   id = "pdf_" + sha256(raw)[:16]
  /store/meta/<id>.json                 write-once (temp file + os.link): id kind media_type bytes
                                        width height (image) | pages (pdf) caption created
                                        (a PDF that failed on first sight: kind pdf, failed true, no pages)
  /store/meta/<pdf id>.txt              write-once: the PDF's full replacement text (poppler runs once per PDF)
Captions: one short vision-model sentence on first sight only (<= 4 per request, 2 at a time, 8 s each); any failure
or over-cap stores caption "" (shown as "no preview") so the text never changes later.
Limits: images up to MAX_PIXELS (50 MP, the same value as vision-mcp's max_pixels) and MAX_IMAGE_BYTES; new raw
files are refused (block -> unavailable) while the store's filesystem has less than IMGFIX_MIN_FREE_GB free.

Config (env): TEXT_UPSTREAM (text model, plain HTTP), VISION_BASE_URL (OpenAI-compatible, with or without /v1),
VISION_MODEL (required), VISION_API_KEY (optional Bearer token, never logged), VISION_EXTRA_BODY (optional JSON
object merged into every caption request; it may not set model/messages/stream), IMGFIX_STORE, IMGFIX_MIN_FREE_GB.
A bad VISION_MODEL / VISION_EXTRA_BODY stops mitmdump at startup with a clear error (exit 2).

/v1/messages/count_tokens is read-only: it writes no store files or meta and requests no captions. Known images
use their meta; unknown ones are shown with "no preview", so a count on an image's first turn can differ slightly
from the real request (which may get a caption).

Layout: prepare() / finish() are pure sync functions (JSON + disk, run via asyncio.to_thread); rewrite() wires
them to the async captioner; ImgFix is the thin mitmproxy hook layer. Logs carry counts and ids, never bodies.
"""
import asyncio
import base64
import hashlib
import io
import json
import logging
import os
import re
import select
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from collections import Counter, OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

from PIL import Image, ImageOps

try:  # only the hook layer needs mitmproxy / httpx; the rewrite core is importable without them
    from mitmproxy import http
except ImportError:  # pragma: no cover
    http = None
try:
    import httpx
except ImportError:  # pragma: no cover
    httpx = None

log = logging.getLogger("imgfix")

STORE_DIR = os.environ.get("IMGFIX_STORE", "/store")
TEXT_UPSTREAM = os.environ.get("TEXT_UPSTREAM", "").strip() or "http://host.docker.internal:8000"
VISION_BASE_URL = os.environ.get("VISION_BASE_URL", "").strip() or "http://host.docker.internal:8001/v1"
VISION_MODEL = os.environ.get("VISION_MODEL", "").strip()
VISION_API_KEY = os.environ.get("VISION_API_KEY", "").strip()
RESERVED_BODY_KEYS = ("model", "messages", "stream")  # VISION_EXTRA_BODY may not set these


def parse_extra_body(s):
    """VISION_EXTRA_BODY -> dict merged into every vision request ("" -> {}). Raises ValueError with an operator-facing
    message: not JSON, not an object, or a reserved key (the request's model/messages/non-streaming are fixed)."""
    s = (s or "").strip()
    if not s:
        return {}
    try:
        d = json.loads(s)
    except ValueError as e:
        raise ValueError(f"VISION_EXTRA_BODY is not valid JSON ({e}); expected an object like "
                         '{"chat_template_kwargs":{"enable_thinking":false}}') from None
    if not isinstance(d, dict):
        raise ValueError(f"VISION_EXTRA_BODY must be a JSON object, got {type(d).__name__}")
    bad = sorted(k for k in d if k in RESERVED_BODY_KEYS)
    if bad:
        raise ValueError(f"VISION_EXTRA_BODY may not set {', '.join(bad)} (fixed by blindsight)")
    return d


def _config():
    """(extra_body, error message or None). Errors are reported by ImgFix.load (mitmdump startup), not at import."""
    try:
        extra = parse_extra_body(os.environ.get("VISION_EXTRA_BODY", ""))
    except ValueError as e:
        return {}, str(e)
    if not VISION_MODEL:
        return extra, "VISION_MODEL is not set (the model name your vision server serves)"
    return extra, None


VISION_EXTRA_BODY, CONFIG_ERROR = _config()

CAPTION_TIMEOUT = 8.0          # s per caption (downscale + vision model call)
CAPTION_MAX_NEW = 4            # new captions per request; the rest get "" for good
CAPTION_CONCURRENCY = 2
CAPTION_PROMPT = "Describe this image in one short sentence of at most 20 words. Reply with the sentence only."
VISION_LONG_EDGE = 1600        # px, images are downscaled to this before going to the vision model
MAX_IMAGE_BYTES = 32 << 20     # decoded; larger => unavailable
MAX_PIXELS = 50_000_000        # larger => unavailable. MUST equal vision-mcp Config.max_pixels (shared contract)
MAX_PDF_BYTES = 64 << 20       # (mitmdump body_size_limit=64m caps whole requests, so ~47 MiB in practice)
PDF_TEXT_LIMIT = 60000         # chars of pdftotext output kept
PDF_TIMEOUT = 120              # s per poppler call
PDF_STDOUT_CAP = 1 << 20       # bytes of pdftotext output read (>> PDF_TEXT_LIMIT chars); more => killed, truncated
POPPLER_MEM = 768 << 20        # address-space cap per poppler process (prlimit --as)
DECODE_SLOTS = threading.BoundedSemaphore(2)  # full Pillow decodes at once, across all requests (memory bound)
MIN_FREE = int(float(os.environ.get("IMGFIX_MIN_FREE_GB", "5")) * (1 << 30))  # free bytes needed to store new files

UNAVAILABLE = "[image unavailable: could not be decoded]"
EXT = {"image/png": "png", "image/jpeg": "jpg", "image/jpg": "jpg", "image/gif": "gif", "image/webp": "webp"}
MEDIA = {"png": "image/png", "jpg": "image/jpeg", "gif": "image/gif", "webp": "image/webp"}
PIL_EXT = {"PNG": "png", "JPEG": "jpg", "GIF": "gif", "WEBP": "webp"}
PIL_FORMATS = tuple(PIL_EXT)   # the only decoders Pillow may try (no TIFF/PSD/BMP/... on pasted bytes)
MEDIA_TYPES = ("image", "document")
OPAQUE_TYPES = ("tool_use", "server_tool_use", "mcp_tool_use")  # .input is model-written JSON, not content

Image.MAX_IMAGE_PIXELS = MAX_PIXELS


# ---------------------------------------------------------------- replacement texts (the byte-stable part)

def image_text(meta):
    i, cap = meta["id"], meta.get("caption") or "no preview"
    return (f'[image {i} | {meta["width"]}x{meta["height"]} {EXT[meta["media_type"]]} | {cap}] The model cannot '
            f'see images. To inspect this one, call the ask_image tool with image_id="{i}" and a specific question.')


def pdf_text(pid, pages, text, cut=False):
    """cut: pdftotext output was longer than PDF_STDOUT_CAP (always marked truncated)."""
    if cut or len(text) > PDF_TEXT_LIMIT:
        text = text[:PDF_TEXT_LIMIT] + "\n[... truncated]"
    return (f'[pdf {pid} | {pages} pages] Text extracted with pdftotext (page images: ask_image with '
            f'image_id="{pid}" and page=<n>):\n{text}')


def url_text(kind, url):
    uid = "url_" + hashlib.sha256(url.encode("utf-8", "surrogatepass")).hexdigest()[:16]
    return f"[{kind} {uid} | remote {kind} not fetched: {url}]"


def clean_caption(s):
    """One line, no placeholder delimiters, bounded length (the caption lands inside [... | ... | caption])."""
    s = re.sub(r"<think>.*?</think>", "", s or "", flags=re.S)
    s = " ".join(s.split()).translate({ord("["): "(", ord("]"): ")", ord("|"): "/"}).strip(" \"'")
    words = s.split(" ")
    if len(words) > 30:
        s = " ".join(words[:30]) + "..."
    return s[:300]


def image_meta(iid, ext, raw, w, h, caption):
    return {"id": iid, "kind": "image", "media_type": MEDIA[ext], "bytes": len(raw), "width": w, "height": h,
            "caption": caption, "created": _now()}


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _image_meta_ok(m):
    return (isinstance(m, dict) and m.get("kind") == "image" and m.get("media_type") in EXT
            and isinstance(m.get("width"), int) and isinstance(m.get("height"), int))


def _pdf_meta_ok(m):
    return isinstance(m, dict) and m.get("kind") == "pdf" and isinstance(m.get("pages"), int)


# ---------------------------------------------------------------- store

class Store:
    """Write-once files on the shared volume. Readable by other uids (vision-mcp runs as 10001)."""

    def __init__(self, root):
        self.root = Path(root)

    def img_path(self, iid, ext):
        return self.root / "img" / f"{iid}.{ext}"

    def pdf_path(self, pid):
        return self.root / "pdf" / f"{pid}.pdf"

    def text_path(self, pid):
        return self.root / "meta" / f"{pid}.txt"

    def writable(self):
        """For /imgfix-health: img/, pdf/, meta/ exist (created if missing) and are writable by this uid."""
        try:
            for d in ("img", "pdf", "meta"):
                p = self.root / d
                p.mkdir(mode=0o755, parents=True, exist_ok=True)
                if not os.access(p, os.W_OK | os.X_OK):
                    return False
            return True
        except OSError:
            return False

    def write_once(self, path, data):
        """Atomically create path with data; never overwrite. True if this call created it."""
        if path.exists():
            return False
        path.parent.mkdir(mode=0o755, parents=True, exist_ok=True)
        if MIN_FREE and shutil.disk_usage(path.parent).free < MIN_FREE + len(data):
            raise OSError("store filesystem low on free space (IMGFIX_MIN_FREE_GB)")
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp-")
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
            os.chmod(tmp, 0o644)
            try:
                os.link(tmp, path)  # atomic create-if-absent: readers see all or nothing
            except FileExistsError:
                return False
            return True
        finally:
            os.unlink(tmp)

    def read_meta(self, mid):
        """The meta dict, or None if there is none. Raises ValueError if it exists but is unreadable/invalid:
        the caller must then NOT treat the id as new (a fresh caption would change its text every turn)."""
        p = self.root / "meta" / f"{mid}.json"
        try:
            m = json.loads(p.read_bytes())
        except FileNotFoundError:
            return None
        except (OSError, ValueError) as e:
            log.warning("imgfix: unreadable meta %s (%s)", mid, type(e).__name__)
            raise ValueError(f"unreadable meta {mid}") from e
        if not (isinstance(m, dict) and m.get("id") == mid):
            log.warning("imgfix: invalid meta %s", mid)
            raise ValueError(f"invalid meta {mid}")
        return m

    def write_meta(self, meta):
        """Write-once; returns the meta that is on disk (ours, or an earlier writer's if we lost a race).
        Raises if we lost and the winner's file is unreadable (never fall back to our in-memory meta)."""
        data = json.dumps(meta, ensure_ascii=True, separators=(",", ":")).encode()
        if self.write_once(self.root / "meta" / f"{meta['id']}.json", data):
            return meta
        disk = self.read_meta(meta["id"])
        if disk is None:
            raise ValueError(f"meta {meta['id']} vanished")
        return disk

    def read_text(self, pid):
        """Stored PDF replacement text, or None."""
        try:
            return self.text_path(pid).read_bytes().decode("utf-8", "surrogatepass")
        except FileNotFoundError:
            return None

    def write_text(self, pid, text):
        """Write-once PDF replacement text; returns the text on disk (an earlier writer's wins a race)."""
        if self.write_once(self.text_path(pid), text.encode("utf-8", "surrogatepass")):
            return text
        disk = self.read_text(pid)
        if disk is None:
            raise ValueError(f"text {pid} vanished")
        return disk

    def heal(self, path, raw):
        """Best effort: put back a missing raw file for a known id (its text never depends on this)."""
        try:
            self.write_once(path, raw)
        except Exception as e:  # noqa: BLE001 - e.g. disk low: the known id still gets its stored text
            log.warning("imgfix: could not restore %s (%s)", path.name, type(e).__name__)


class _LRU:
    def __init__(self, n):
        self.n, self.d, self.lock = n, OrderedDict(), threading.Lock()

    def get(self, k):
        with self.lock:
            v = self.d.get(k)
            if v is not None:
                self.d.move_to_end(k)
            return v

    def put(self, k, v):
        with self.lock:
            self.d[k] = v
            self.d.move_to_end(k)
            while len(self.d) > self.n:
                self.d.popitem(last=False)

    def clear(self):
        with self.lock:
            self.d.clear()


# pdf id -> replacement text as stored in meta/<id>.txt. Pure read cache (only ever holds stored texts).
PDF_CACHE = _LRU(64)


# ---------------------------------------------------------------- decoding helpers (sync, run in a worker thread)

def _b64(data):
    if not isinstance(data, str):
        raise ValueError("no base64 data")
    return base64.b64decode(data.encode("ascii").translate(None, b" \t\r\n"), validate=True)


def _probe(raw, media_type):
    """Fully decode once (first sight only) -> (w, h, ext), w x h as displayed (EXIF orientation applied, like
    vision-mcp and the caption path, so ask_image region fractions match). Raises if undecodable/oversized."""
    with DECODE_SLOTS, Image.open(io.BytesIO(raw), formats=PIL_FORMATS) as im:
        w, h = im.size
        if w * h > MAX_PIXELS:
            raise ValueError("too many pixels")
        fmt = PIL_EXT.get(im.format)
        if im.getexif().get(0x0112) in (5, 6, 7, 8):  # transposed/rotated 90 or 270 degrees on display
            w, h = h, w
        im.load()
    ext = EXT.get(str(media_type).lower()) or fmt  # ext follows the declared media type, else the real format
    if not ext:
        raise ValueError("unsupported image format")
    return w, h, ext


def vision_png(raw):
    """Image bytes -> PNG with long edge <= VISION_LONG_EDGE (first frame, EXIF orientation applied)."""
    with DECODE_SLOTS, Image.open(io.BytesIO(raw), formats=PIL_FORMATS) as im:
        if im.format == "JPEG":  # big photos: decode at 1/2..1/8 scale in the DCT domain (still >= the target)
            im.draft(im.mode, (VISION_LONG_EDGE, VISION_LONG_EDGE))
        im = ImageOps.exif_transpose(im)
        if im.mode not in ("RGB", "RGBA", "L"):
            alpha = im.mode in ("RGBA", "LA", "PA") or "transparency" in im.info
            im = im.convert("RGBA" if alpha else "RGB")
        im.thumbnail((VISION_LONG_EDGE, VISION_LONG_EDGE), Image.Resampling.LANCZOS, reducing_gap=2.0)
        buf = io.BytesIO()
        im.save(buf, "PNG")
        return buf.getvalue()


def _run(args, cap=None):
    """stdout of a poppler call, run under an address-space cap (prlimit) and PDF_TIMEOUT; stderr dropped.
    cap: read at most cap bytes, then kill the process -> (data, True). Raises CalledProcessError / TimeoutExpired."""
    p = subprocess.Popen(["prlimit", f"--as={POPPLER_MEM}", "--", *args],
                         stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    try:
        deadline, buf, fd = time.monotonic() + PDF_TIMEOUT, bytearray(), p.stdout.fileno()
        while True:
            left = deadline - time.monotonic()
            if left <= 0:
                raise subprocess.TimeoutExpired(args, PDF_TIMEOUT)
            if not select.select([fd], [], [], left)[0]:
                continue
            chunk = os.read(fd, 1 << 16)
            if not chunk:
                break
            buf += chunk
            if cap is not None and len(buf) > cap:
                return bytes(buf[:cap]), True
        try:
            rc = p.wait(max(0.0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            raise subprocess.TimeoutExpired(args, PDF_TIMEOUT) from None
        if rc != 0:
            raise subprocess.CalledProcessError(rc, args)
        return bytes(buf), False
    finally:
        if p.poll() is None:
            p.kill()
        p.wait()
        p.stdout.close()


def _pdf_extract(path=None, raw=None):
    """-> (pages, text, cut) via poppler. Unstored PDFs go through a scratch copy in /tmp."""
    if path is None:
        with tempfile.NamedTemporaryFile(suffix=".pdf") as f:
            f.write(raw)
            f.flush()
            return _pdf_extract(f.name)
    info = _run(["pdfinfo", str(path)])[0].decode("utf-8", "replace")
    m = re.search(r"^Pages:\s+(\d+)\s*$", info, re.M)
    if not m:
        raise ValueError("pdfinfo: no page count")
    out, cut = _run(["pdftotext", "-q", "-enc", "UTF-8", str(path), "-"], cap=PDF_STDOUT_CAP)
    return int(m.group(1)), out.decode("utf-8", "replace").rstrip(), cut


def _pdf_failure_is_permanent(e):
    """Failures caused by the PDF itself (a retry would fail the same way, or take PDF_TIMEOUT again every turn),
    as opposed to the environment (spawn failure, OOM-killer SIGKILL): only the former is stored for good."""
    if isinstance(e, subprocess.CalledProcessError):
        return e.returncode != -signal.SIGKILL
    return isinstance(e, (ValueError, subprocess.TimeoutExpired))


# ---------------------------------------------------------------- the rewrite

@dataclass
class _Slot:
    """A replaced block. json.dumps(default=_slot_json) turns it into a text block."""
    text: str | None
    new_id: str | None = None   # first-sight image: finish() fills text once its meta is written
    extra: dict | None = None   # cache_control carried over from the original block


@dataclass
class _NewImage:
    id: str
    ext: str
    raw: bytes
    width: int
    height: int


@dataclass
class Prepared:
    obj: object = None                          # parsed body, media blocks already swapped for _Slot
    changed: bool = False
    new: dict = field(default_factory=dict)     # id -> _NewImage, in order of appearance
    slots: list = field(default_factory=list)   # slots waiting for a new image's meta
    stats: Counter = field(default_factory=Counter)
    ids: list = field(default_factory=list)


def _slot_json(o):
    if isinstance(o, _Slot):
        d = {"type": "text", "text": o.text}
        if o.extra:
            d.update(o.extra)
        return d
    raise TypeError(f"not JSON serializable: {type(o).__name__}")


def _is_media(x):
    return isinstance(x, dict) and x.get("type") in MEDIA_TYPES


class _Rewriter:
    def __init__(self, store, count_only, prep):
        self.store, self.count_only, self.p = store, count_only, prep

    def walk(self, node):
        """Replace media blocks in every list/dict below node (messages, tool_result.content, ... any depth)."""
        if isinstance(node, list):
            out = []
            for item in node:
                out.extend(self.replace(item) if _is_media(item) else (self.walk(item),))
            return out
        if isinstance(node, dict) and node.get("type") not in OPAQUE_TYPES:
            for k, v in node.items():
                if _is_media(v):  # not valid API shape, but fail closed anyway
                    r = self.replace(v)
                    node[k] = r[0] if len(r) == 1 else r
                elif isinstance(v, (list, dict)):
                    node[k] = self.walk(v)
        return node

    def replace(self, block):
        kind = block.get("type")
        try:
            out = self.image(block) if kind == "image" else self.document(block)
        except Exception as e:  # noqa: BLE001 - any failure in a block => fail closed for that block
            self.p.stats["failed"] += 1
            log.warning("imgfix: %s block unavailable (%s)", kind, type(e).__name__)
            out = [_Slot(UNAVAILABLE)]
        self.p.changed = True
        if "cache_control" in block and out:
            last = out[-1]
            if isinstance(last, _Slot):
                last.extra = {"cache_control": block["cache_control"]}
            elif isinstance(last, dict):
                last.setdefault("cache_control", block["cache_control"])
        return out

    def _id(self, prefix, raw):
        mid = prefix + hashlib.sha256(raw).hexdigest()[:16]
        if mid not in self.p.ids:
            self.p.ids.append(mid)
        return mid

    def image(self, block):
        src = block.get("source")
        st = src.get("type") if isinstance(src, dict) else None
        if st == "url":
            self.p.stats["url"] += 1
            return [_Slot(url_text("image", str(src.get("url", ""))))]
        if st != "base64":
            raise ValueError(f"image source {st!r}")
        raw = _b64(src.get("data"))
        if len(raw) > MAX_IMAGE_BYTES:
            raise ValueError("image too large")
        iid = self._id("img_", raw)
        self.p.stats["images"] += 1
        meta = self.store.read_meta(iid)  # raises on a broken meta file -> unavailable (fixed text, no re-caption)
        if meta is not None:  # seen before: meta alone drives the text, no decode needed
            if not _image_meta_ok(meta):
                raise ValueError(f"invalid image meta {iid}")
            if not self.count_only:  # self-heal a missing raw file
                self.store.heal(self.store.img_path(iid, EXT[meta["media_type"]]), raw)
            return [_Slot(image_text(meta))]
        if iid in self.p.new:  # same new image twice in one request
            return [self._pending(iid)]
        w, h, ext = _probe(raw, src.get("media_type"))
        if self.count_only:
            return [_Slot(image_text(image_meta(iid, ext, raw, w, h, "")))]
        self.store.write_once(self.store.img_path(iid, ext), raw)  # raw before meta: meta implies the file
        self.p.new[iid] = _NewImage(iid, ext, raw, w, h)
        return [self._pending(iid)]

    def _pending(self, iid):
        s = _Slot(None, new_id=iid)
        self.p.slots.append(s)
        return s

    def document(self, block):
        src = block.get("source")
        st = src.get("type") if isinstance(src, dict) else None
        if st == "text":  # plain-text document -> plain text block
            if not isinstance(src.get("data"), str):
                raise ValueError("text document without data")
            self.p.stats["text_docs"] += 1
            return [{"type": "text", "text": src["data"]}]
        if st == "content":  # custom-content document -> its own blocks (images inside are replaced too)
            c = src.get("content")
            if isinstance(c, str):
                return [{"type": "text", "text": c}]
            if not isinstance(c, list):
                raise ValueError("content document without content")
            self.p.stats["content_docs"] += 1
            return self.walk(c)
        if st == "url":
            self.p.stats["url"] += 1
            return [_Slot(url_text("document", str(src.get("url", ""))))]
        if st != "base64" or src.get("media_type", "application/pdf") != "application/pdf":
            raise ValueError(f"document source {st!r}")
        raw = _b64(src.get("data"))
        if len(raw) > MAX_PDF_BYTES:
            raise ValueError("pdf too large")
        pid = self._id("pdf_", raw)
        self.p.stats["pdfs"] += 1
        return [_Slot(self.pdf(pid, raw))]

    def pdf(self, pid, raw):
        """Replacement text for a PDF. Poppler runs on first sight only; the text is then stored write-once in
        meta/<id>.txt and read back on every later turn, so it never depends on load, timing or poppler version."""
        path, meta = self.store.pdf_path(pid), self.store.read_meta(pid)  # raises on a broken meta file
        if meta is not None:
            if meta.get("failed") is True:  # failed on first sight: stays unavailable for good (stable text)
                raise ValueError(f"pdf {pid} failed on first sight")
            if not _pdf_meta_ok(meta):
                raise ValueError(f"invalid pdf meta {pid}")
            if not self.count_only:
                self.store.heal(path, raw)
            text = PDF_CACHE.get(pid) or self.store.read_text(pid)
            if text is None:  # meta without text (a text write failed earlier): rebuild once with the stored pages
                _, body, cut = _pdf_extract(path) if path.exists() else _pdf_extract(raw=raw)
                text = pdf_text(pid, meta["pages"], body, cut)
                if self.count_only:
                    return text
                text = self.store.write_text(pid, text)
            PDF_CACHE.put(pid, text)
            return text
        try:  # first sight; raises on a broken PDF before the raw file is stored
            pages, body, cut = _pdf_extract(path) if path.exists() else _pdf_extract(raw=raw)
        except Exception as e:
            if not self.count_only and _pdf_failure_is_permanent(e):
                self.store.write_meta({"id": pid, "kind": "pdf", "media_type": "application/pdf", "bytes": len(raw),
                                       "caption": "", "failed": True, "error": type(e).__name__, "created": _now()})
            raise
        text = pdf_text(pid, pages, body, cut)
        if self.count_only:
            return text
        self.store.write_once(path, raw)        # raw, then text, then meta: meta implies both
        text = self.store.write_text(pid, text)  # an earlier writer's text wins a race
        meta = self.store.write_meta({"id": pid, "kind": "pdf", "media_type": "application/pdf",
                                      "bytes": len(raw), "pages": pages, "caption": "", "created": _now()})
        if meta.get("failed") is True:  # lost a first-sight race to a writer that failed
            raise ValueError(f"pdf {pid} failed on first sight")
        PDF_CACHE.put(pid, text)
        return text


def prepare(raw, store, count_only=False):
    """Parse + swap media blocks (sync: JSON, hashing, Pillow, poppler, store writes). None if body is not JSON."""
    try:
        obj = json.loads(raw)
    except ValueError:  # includes UnicodeDecodeError / JSONDecodeError
        return None
    p = Prepared(obj=obj)
    if not isinstance(obj, dict) or not isinstance(obj.get("messages"), list):
        return p  # the model name does not matter: every messages list is scanned
    obj["messages"] = _Rewriter(store, count_only, p).walk(obj["messages"])
    return p


def finish(p, store, captions):
    """Write-once meta for first-sight images, fill their texts, serialize (sync)."""
    texts = {}
    for n in p.new.values():
        try:
            meta = store.write_meta(image_meta(n.id, n.ext, n.raw, n.width, n.height, captions.get(n.id, "")))
            texts[n.id] = image_text(meta)
        except Exception as e:  # noqa: BLE001 - e.g. store full: fail closed for this image
            p.stats["failed"] += 1
            log.warning("imgfix: meta write failed for %s (%s)", n.id, type(e).__name__)
            texts[n.id] = UNAVAILABLE
    for s in p.slots:
        s.text = texts[s.new_id]
    return json.dumps(p.obj, ensure_ascii=True, separators=(",", ":"), default=_slot_json).encode("ascii")


async def caption_new(new, caption):
    """Caption the newest CAPTION_MAX_NEW first-sight images, CAPTION_CONCURRENCY at a time. id -> caption|""."""
    sem = asyncio.Semaphore(CAPTION_CONCURRENCY)

    async def one(iid):
        async with sem:
            try:
                async with asyncio.timeout(CAPTION_TIMEOUT):
                    return iid, clean_caption(await caption(new[iid].raw))
            except Exception as e:  # noqa: BLE001 - caption is best effort; "" is stored for good
                log.info("imgfix: caption %s failed (%s)", iid, type(e).__name__)
                return iid, ""

    # The newest images sit at the end of the conversation (the one the user just pasted).
    return dict(await asyncio.gather(*(one(i) for i in list(new)[-CAPTION_MAX_NEW:])))


@dataclass
class Result:
    body: bytes | None   # None => forward the original bytes unchanged
    stats: Counter
    ids: list


async def rewrite(raw, store, caption=None, count_only=False):
    """Body bytes -> Result. caption: async (image bytes) -> str, or None for no captions."""
    p = await asyncio.to_thread(prepare, raw, store, count_only)
    if p is None:
        return Result(None, Counter(not_json=1), [])
    if not p.changed:
        return Result(None, p.stats, p.ids)
    caps = await caption_new(p.new, caption) if (p.new and caption) else {}
    p.stats["new"] = len(p.new)
    p.stats["captioned"] = sum(1 for c in caps.values() if c)
    body = await asyncio.to_thread(finish, p, store, caps)
    return Result(body, p.stats, p.ids)


# ---------------------------------------------------------------- vision-model captioner

def _caption_request(raw, model, extra=None):
    """OpenAI chat/completions body. extra (VISION_EXTRA_BODY) is merged on top but never replaces the reserved keys
    (e.g. {"chat_template_kwargs":{"enable_thinking":false}} keeps a reasoning model inside CAPTION_TIMEOUT)."""
    png = vision_png(raw)
    body = {
        "model": model,
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(png).decode()}},
            {"type": "text", "text": CAPTION_PROMPT}]}],
        "temperature": 0, "max_tokens": 60,
    }
    body.update({k: v for k, v in (extra or {}).items() if k not in RESERVED_BODY_KEYS})
    return json.dumps(body).encode()


class VisionCaptioner:
    """async (image bytes) -> caption via the vision model's OpenAI-compatible chat API. Never blocks the event loop."""

    def __init__(self, base_url=VISION_BASE_URL, model=VISION_MODEL, api_key=VISION_API_KEY, extra=None):
        base = base_url.rstrip("/")
        self.url = base + ("/chat/completions" if base.endswith("/v1") else "/v1/chat/completions")
        self.model, self.extra, self.client = model, VISION_EXTRA_BODY if extra is None else extra, None
        self.headers = {"content-type": "application/json"}
        if api_key:  # never logged
            self.headers["authorization"] = f"Bearer {api_key}"

    async def __call__(self, raw):
        if self.client is None:
            # trust_env=False: a Docker-injected HTTP(S)_PROXY must not divert caption calls (each failure is a ""
            # for good)
            self.client = httpx.AsyncClient(timeout=CAPTION_TIMEOUT, limits=httpx.Limits(max_connections=4),
                                            trust_env=False)
        body = await asyncio.to_thread(_caption_request, raw, self.model, self.extra)  # resize + base64 off the loop
        r = await self.client.post(self.url, content=body, headers=self.headers)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"] or ""

    async def aclose(self):
        if self.client is not None:
            await self.client.aclose()
            self.client = None


# ---------------------------------------------------------------- mitmproxy hooks

def _addr(url):
    u = urlsplit(url)
    return u.scheme, u.hostname, u.port or (443 if u.scheme == "https" else 80)


def _error(msg):
    body = json.dumps({"type": "error", "error": {"type": "invalid_request_error", "message": msg}}).encode()
    return http.Response.make(400, body, {"content-type": "application/json"})


class ImgFix:
    def __init__(self):
        self.store = Store(STORE_DIR)
        self.captioner = VisionCaptioner()
        self.text = _addr(TEXT_UPSTREAM)

    def load(self, loader):
        # Runs once when mitmdump loads this script. A config error must stop the container with a clear message:
        # mitmproxy would only log an exception here and keep serving without the addon. stderr, not the logger:
        # mitmproxy's log handler is asynchronous and would lose the line on os._exit.
        if CONFIG_ERROR:
            print(f"imgfix: configuration error: {CONFIG_ERROR}", file=sys.stderr, flush=True)
            os._exit(2)
        log.info("imgfix: text=%s vision=%s model=%s api_key=%s extra_body_keys=%s", TEXT_UPSTREAM,
                 self.captioner.url, VISION_MODEL, "set" if VISION_API_KEY else "unset",
                 ",".join(sorted(VISION_EXTRA_BODY)) or "-")

    async def done(self):
        await self.captioner.aclose()

    async def request(self, flow):
        # Fail closed: mitmproxy only logs an exception escaping a hook and then forwards the flow, so nothing may
        # escape. (The CMD's reverse target is also a dead address, never the text model.)
        try:
            await self._request(flow)
        except Exception as e:  # noqa: BLE001
            log.error("imgfix: request hook failed (%s); request rejected", type(e).__name__)
            flow.response = _error(f"imgfix failed on this request ({type(e).__name__}); it was not forwarded.")

    async def _request(self, flow):
        req = flow.request
        if req.path == "/imgfix-health":  # HAProxy's check: unhealthy => it answers image requests with a 400
            ok = await asyncio.to_thread(self.store.writable)
            flow.response = http.Response.make(200 if ok else 503, b"ok\n" if ok else b"store not writable\n",
                                               {"content-type": "text/plain"})
            return
        t0, path = time.monotonic(), req.path.split("?", 1)[0]
        info = {"t0": t0, "path": path, "up": "text"}
        flow.metadata["imgfix"] = info
        if req.method == "POST" and req.raw_content:
            try:  # every path: any JSON body with a "messages" list is scanned (count_tokens read-only)
                raw = req.get_content(strict=False)
                ct = path.endswith("/count_tokens")
                res = await rewrite(raw, self.store, None if ct else self.captioner, count_only=ct)
            except Exception as e:  # noqa: BLE001 - never forward a body we could not clean
                log.error("imgfix: %s rewrite failed (%s); request rejected", path, type(e).__name__)
                flow.response = _error(f"imgfix could not rewrite the image/document blocks in this request "
                                       f"({type(e).__name__}); it was not forwarded.")
                return
            if res.body is not None:
                if "transfer-encoding" in req.headers:
                    del req.headers["transfer-encoding"]
                req.content = res.body
                req.headers["content-length"] = str(len(req.raw_content))
            info.update(stats=res.stats, ids=res.ids, rewrite_ms=int((time.monotonic() - t0) * 1000))
        req.scheme, req.host, req.port = self.text

    def responseheaders(self, flow):
        flow.response.stream = True  # SSE passes straight through

    def response(self, flow):
        i = flow.metadata.get("imgfix")
        if i is None:
            return
        st = " ".join(f"{k}={v}" for k, v in sorted(i.get("stats", {}).items()))
        ids = ",".join(i.get("ids", [])[:16]) or "-"
        log.info("imgfix: %s %s -> %s %s %s ids=%s rewrite=%sms total=%.1fs", flow.request.method, i["path"],
                 i["up"], flow.response.status_code, st or "unchanged", ids, i.get("rewrite_ms", 0),
                 time.monotonic() - i["t0"])

    def error(self, flow):
        i = flow.metadata.get("imgfix") or {}
        log.warning("imgfix: %s %s -> %s error: %s", flow.request.method, i.get("path", "?"), i.get("up", "?"),
                    flow.error.msg if flow.error else "?")


addons = [ImgFix()]
