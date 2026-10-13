"""imgfix: mitmproxy addon that keeps image/document/file parts away from a text-only model (blindsight).

The router (HAProxy) sends here every POST whose body holds media parts or that it could not buffer whole (over
~30 MiB; a body still arriving after HAProxy's 300 s gets a 408 from HAProxy and never comes here). imgfix checks every
request body it gets (any path, any method, any model name except PASSTHROUGH_MODELS) before it goes on to the text
model (TEXT_URL). Media parts are recognised by SHAPE below the request's content keys, in five request formats; every
format gets the same REF strings, only the wrapper differs:
  Anthropic Messages  {"type":"image"|"document","source":...}                  -> {"type":"text","text":REF}
  OpenAI Chat         {"type":"image_url","image_url":{"url":U}|U}              -> {"type":"text","text":REF}
                      {"type":"file","file":{"file_data"|"file_id"}}            -> {"type":"text","text":REF}
                      {"image_url":...} without a "type" (vLLM extension)       -> {"type":"text","text":REF}
  OpenAI Responses    {"type":"input_image","image_url"|"file_id"}              -> {"type":"input_text","text":REF}
                      {"type":"input_file","file_data"|"file_url"|"file_id"}    -> {"type":"input_text","text":REF}
                      {"type":"computer_screenshot","image_url"|"file_id"}      -> {"type":"input_text","text":REF}
  Ollama native       "images":[base64,...] on a message (/api/chat) or the body (/api/generate): removed, and
                      "\n\n" + the REFs (one per line) appended to its "content" / "prompt"
  Gemini native       {"inline_data"|"inlineData":{"mime_type"|"mimeType","data"}}             -> {"text":REF}
                      {"file_data"|"fileData":{"mime_type"|"mimeType","file_uri"|"fileUri"}}   -> {"text":REF}
Content keys walked (top level, any letter case or snake/camel spelling): messages, input, prompt (an object: the
Responses prompt.variables), images, contents, systemInstruction, and generateContentRequest (Gemini countTokens),
whose own content keys are walked the same way. U is a data URL ("data:<mime>[;k=v]*;base64,<data>", parsed
strictly, never echoed into a text) or an http(s) URL. Gemini file_data: an http(s) URI with an image (or no) MIME type
is an image URL (fetched only with FETCH_REMOTE_IMAGES); other http(s) URIs are remote documents (never fetched); Gemini
Files API URIs (generativelanguage.googleapis.com) and other schemes (gs://...) are provider-side uploads (file_id
text). Model-written JSON is left alone: tool_use/server_tool_use/mcp_tool_use and function_call items, and any
"arguments", "function_call" or "functionCall" value (tool calls). Each image/PDF's raw bytes go to the shared /store
volume and the text model later asks vision-mcp about it (ask_image tool or its /ask endpoint, see ASK_HINT).

Fails CLOSED: no media part is ever forwarded to the text model. A part that cannot be handled becomes "[image
unavailable: could not be decoded]". The request gets HTTP 400 (JSON error.type/status + error.message, read by
Anthropic, OpenAI and Gemini clients) and is not forwarded when:
  - its Content-Encoding is not identity/gzip/deflate/br/zstd, is corrupt or has trailing data, or decodes to more
    than 64 MiB (a decoded body is checked and forwarded decoded, so the upstream reads exactly what was checked);
  - its body is not strict JSON (UTF-8, no comments, trailing commas or raw control characters, nesting < ~1000).
    The one exception, forwarded unchanged: a body that is clearly not a JSON API request, i.e. a multipart/*,
    application/octet-stream, audio/* or video/* Content-Type AND a body not starting with "{" or "[" AND none of
    the media markers image, document, input_file, file_data, file_id, base64, data:, screenshot, inline (any case)
    anywhere in it;
  - a JSON object repeats a key, or holds two spellings (letter case, "_") of one routing key: model, messages, input,
    images, prompt, content, type, contents, parts, systemInstruction, generateContentRequest, inlineData, fileData,
    mimeType, fileUri (an upstream might read the other one: Go's encoding/json matches field names in any case,
    protobuf JSON accepts snake and camel case);
  - after the rewrite, the structural backstop still finds a media-shaped node anywhere in the body outside
    model-written JSON and top-level request configuration (tools, functions, tool_choice, response_format, text,
    format, generationConfig, toolConfig): a "type" (key and value in any case) image, document, image_url,
    input_image, input_file, file, computer_screenshot, image_embeds or image_pil; an untyped part with an image_url;
    a non-empty "images" list; an Anthropic base64/url/file "source"; any inline_data/inlineData/file_data/fileData.
    Strings are never inspected as media. PASSTHROUGH_MODELS requests are exempt (after the duplicate-key check).

Prefix-cache rule: clients re-send the whole history every turn, so the text for a given part must be byte-identical
forever (per configuration). It is therefore built only from write-once store files (or, on first sight, from the
exact values then written to them) and the configuration; nothing time-, load- or process-dependent goes into it. A
meta file that exists but cannot be read gives the fixed "unavailable" text (never a fresh caption or fetch).

Store (shared contract with vision-mcp):
  /store/img/<id>.<png|jpg|gif|webp>   id = "img_" + sha256(raw)[:16]   raw bytes as received
  /store/pdf/<id>.pdf                   id = "pdf_" + sha256(raw)[:16]
  /store/meta/<id>.json                 write-once (temp file + os.link): id kind media_type bytes
                                        width height (image) | pages (pdf) caption created
                                        (a PDF that failed on first sight: kind pdf, failed true, no pages)
  /store/meta/<pdf id>.txt              write-once: the PDF's full replacement text with the mcp hint (poppler runs
                                        once per PDF; render_pdf swaps in the configured hint)
  /store/meta/url_<16hex>.json          write-once fetch decision for a remote image URL (FETCH_REMOTE_IMAGES):
                                        id kind=url url target=img_<id> | failed=true error; created
                                        url id = "url_" + sha256(url)[:16]
Captions (CAPTION_MODE): short = one vision-model sentence | detailed = a description quoting all visible text |
off. First sight only (<= 4 per request, 2 at a time, CAPTION_TIMEOUT each); any failure or over-cap stores caption ""
(shown as "no preview") so the text never changes later.
Limits: images up to MAX_PIXELS (50 MP, the same value as vision-mcp's max_pixels) and MAX_IMAGE_BYTES; new raw
files are refused (part -> unavailable) while the store's filesystem has less than IMGFIX_MIN_FREE_GB free. A URL or
file id echoed into a reference is cut at 200 characters.

Config (env): TEXT_URL (text model base URL: http(s), optional port and path prefix, checked exactly like
router/entrypoint.sh; https verified against the default CAs or TEXT_CA_FILE, which requires an https TEXT_URL),
PASSTHROUGH_MODELS (forwarded untouched when every model name the request carries matches: the body's "model" in any
letter case, Gemini's generateContentRequest.model and path models/<model>:..., a "models/" prefix ignored),
CAPTION_MODE, CAPTION_TIMEOUT, ASK_HINT, ASK_URL, FETCH_REMOTE_IMAGES, FETCH_ALLOW_PRIVATE, IMGFIX_STORE,
IMGFIX_MIN_FREE_GB, and the vision model's VISION_* (see vision_api.py). Any bad value stops mitmdump at startup with
a clear error (exit 2).

Count-only paths (ending /count_tokens, /input_tokens or :countTokens) are read-only: no store files or meta are
written, no captions requested, no URLs fetched. Known parts use their meta; unknown images are shown with "no
preview", so a count on an image's first turn can differ slightly from the real request (which may get a caption).

Layout: prepare() / settle() / finish() are pure sync functions (JSON + disk, run via asyncio.to_thread); rewrite()
wires them to the async captioner and the remote-image fetcher (a thread pool); ImgFix is the thin mitmproxy hook
layer. Logs carry counts and ids, never bodies, URLs or keys.
"""
import asyncio
import base64
import concurrent.futures
import fnmatch
import hashlib
import io
import ipaddress
import json
import logging
import math
import os
import re
import select
import shutil
import signal
import socket
import ssl
import subprocess
import sys
import tempfile
import threading
import time
import zlib
from collections import Counter, OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import unquote, urljoin, urlsplit

from PIL import Image, ImageOps

import vision_api  # shared/vision_api.py, copied next to this file by the Dockerfile

try:  # only the hook layer, captioner and fetcher need these; the rewrite core is importable without them
    from mitmproxy import ctx, http
except ImportError:  # pragma: no cover
    ctx = http = None
try:
    import certifi
    import httpcore
    import httpx
except ImportError:  # pragma: no cover
    certifi = httpcore = httpx = None
try:  # request Content-Encoding br / zstd (mitmproxy's own dependencies)
    import brotli
    import zstandard
except ImportError:  # pragma: no cover
    brotli = zstandard = None

log = logging.getLogger("imgfix")


# ---------------------------------------------------------------- configuration (env, parsed once at import)

DEFAULT_TEXT_URL = "http://host.docker.internal:8000"
DEFAULT_ASK_URL = "http://127.0.0.1:8102/ask"
CAPTION_MODES = ("short", "detailed", "off")   # first = default
ASK_HINTS = ("mcp", "http", "none")


@dataclass(frozen=True)
class TextUpstream:
    url: str         # normalized scheme://host:port/prefix (no userinfo or query: safe to log)
    scheme: str      # http | https
    host: str        # IPv6 literals without brackets
    port: int
    prefix: str      # "" or "/a/b" (no trailing slash), prepended to the client's path
    authority: str   # Host header: host (IPv6 in brackets), plus :port only if non-default


_HOST_BAD = re.compile(r"[^a-z0-9._-]")
_PATH_BAD = re.compile(r"[^A-Za-z0-9._~!$&'()*+,;=:@/%-]")                 # RFC 3986 path characters
_PCT_BAD = re.compile(r"%([^0-9A-Fa-f]|[0-9A-Fa-f][^0-9A-Fa-f]|[0-9A-Fa-f]?$)")  # % not followed by two hex digits


def _is_ipv4(host):
    """Dotted-quad IPv4: four parts 0-255 without leading zeros (no 127.1 or 010.0.0.1 shorthands)."""
    parts = host.split(".")
    return len(parts) == 4 and all(p.isdigit() and len(p) <= 3 and int(p) <= 255 and (p == "0" or p[0] != "0")
                                   for p in parts)


def parse_text_url(s):
    """TEXT_URL -> TextUpstream. Accepts and rejects exactly what router/entrypoint.sh does (the same checks in the
    same order), so the router and imgfix never disagree about the text model. Raises ValueError; like the router's,
    the messages never echo the URL (it may hold a secret in the wrong place)."""
    url = (s or "").strip(" \t\n\r\v\f") or DEFAULT_TEXT_URL
    if any(c.isspace() or c < " " or c == "\x7f" for c in url):
        raise ValueError("TEXT_URL contains whitespace or a control character")
    if "://" not in url:
        raise ValueError("TEXT_URL must look like http(s)://host[:port][/path], e.g. https://api.example.com/anthropic")
    scheme, rest = url.split("://", 1)
    scheme = scheme.lower()
    if scheme not in ("http", "https"):
        raise ValueError("TEXT_URL must start with http:// or https://")
    default = 443 if scheme == "https" else 80
    if "#" in rest:
        raise ValueError("TEXT_URL may not have a #fragment")
    if "?" in rest:
        raise ValueError("TEXT_URL may not have a ?query (the client's own path and query are appended to it)")
    authority = rest.split("/", 1)[0]
    path = rest[len(authority):]
    if "@" in authority:
        raise ValueError("TEXT_URL may not contain user:password@ (the client's own API key headers are passed "
                         "through)")
    if not authority:
        raise ValueError("TEXT_URL has no host")
    v6 = authority.startswith("[")
    if v6:
        inner = authority[1:]
        if "]" not in inner:
            raise ValueError("TEXT_URL: unclosed [ around the IPv6 address")
        host, after = inner.split("]", 1)
        if after and not after.startswith(":"):
            raise ValueError("TEXT_URL: unexpected text after the ] of the IPv6 address")
        port = after[1:]
        if "%" in host:
            raise ValueError("TEXT_URL: IPv6 zone ids (%...) are not supported")
        if ":" not in host or re.search(r"[^0-9A-Fa-f:.]", host):
            raise ValueError(f"TEXT_URL: [{host}] is not an IPv6 address")
    else:
        host, _, port = authority.partition(":")
        if ":" in port:
            raise ValueError("TEXT_URL: put an IPv6 address in brackets, e.g. http://[fd00::1]:8000")
    host = host.lower()
    if not host:
        raise ValueError("TEXT_URL has no host")
    ip = v6
    if not v6:
        if _HOST_BAD.search(host) or host.startswith((".", "-")) or host.endswith(".") or ".." in host:
            raise ValueError(f"TEXT_URL: '{host}' is not a valid host name or IP address")
        if not re.search(r"[^0-9.]", host):
            if not _is_ipv4(host):
                raise ValueError(f"TEXT_URL: '{host}' is not a valid IPv4 address")
            ip = True
    if port:
        if not (port.isascii() and port.isdigit()):
            raise ValueError("TEXT_URL: the port must be a number")
        port = port.lstrip("0") or "0"
        if not (len(port) <= 5 and 1 <= int(port) <= 65535):
            raise ValueError(f"TEXT_URL: port {port} is not in 1-65535")
        port = int(port)
    else:
        port = default
    if _PATH_BAD.search(path):
        raise ValueError("TEXT_URL: the path has a character that must be percent-encoded")
    if _PCT_BAD.search(path):
        raise ValueError("TEXT_URL: the path has a % that is not followed by two hex digits")
    path = path.rstrip("/")
    if scheme == "https" and ip:
        raise ValueError("TEXT_URL: https needs a host name, not an IP address (the certificate is checked against DNS "
                         "names). Name the address with extra_hosts in a compose override and use a certificate for "
                         "that name")
    shown = f"[{host}]" if v6 else host
    return TextUpstream(f"{scheme}://{shown}:{port}{path}", scheme, host, port, path,
                        shown + ("" if port == default else f":{port}"))


_ASK_URL_RE = re.compile(r"https?://[A-Za-z0-9._~%:\[\]-]+(/[A-Za-z0-9._~%/:@-]*)?")


def parse_ask_url(s):
    """ASK_URL (only used with ASK_HINT=http). It is shown to the model inside a curl command line, so it may not hold
    shell-special characters, spaces or a query (curl -G adds the query)."""
    s = (s or "").strip() or DEFAULT_ASK_URL
    ok = _ASK_URL_RE.fullmatch(s)
    try:
        ok = ok and urlsplit(s).hostname and urlsplit(s).port != 0
    except ValueError:
        ok = False
    if not ok:
        raise ValueError(f"ASK_URL must be an http(s) URL without query, quotes or spaces, like {DEFAULT_ASK_URL} "
                         f"(the address the client's harness uses to reach vision-mcp's /ask); got {s!r}")
    return s


def parse_choice(name, s, choices):
    v = (s or "").strip().lower() or choices[0]
    if v not in choices:
        raise ValueError(f"{name} must be one of {', '.join(choices)}; got {s!r}")
    return v


def parse_bool(name, s):
    v = (s or "").strip().lower()
    if v in ("", "0", "false", "no", "off"):
        return False
    if v in ("1", "true", "yes", "on"):
        return True
    raise ValueError(f"{name} must be true or false; got {s!r}")


def parse_number(name, s, default, positive=False, maximum=None):
    v = (s or "").strip()
    if not v:
        return default
    try:
        f = float(v)
    except ValueError:
        f = math.nan
    if not (math.isfinite(f) and (f > 0 if positive else f >= 0) and (maximum is None or f <= maximum)):
        raise ValueError(f"{name} must be a number {'>' if positive else '>='} 0"
                         f"{'' if maximum is None else f' and <= {maximum:g}'}; got {v[:40]!r}")
    return f


def check_ca_file(s, text):
    """TEXT_CA_FILE: a PEM bundle that replaces the default CAs for an https TEXT_URL (text: the parsed TEXT_URL, or
    None if that was invalid). None if unset."""
    path = (s or "").strip()
    if not path:
        return None
    if text is not None and text.scheme != "https":
        raise ValueError("TEXT_CA_FILE is set, but TEXT_URL is not https:// (the CA file only verifies an https text "
                         "model); unset TEXT_CA_FILE or use an https TEXT_URL")
    try:
        ssl.create_default_context(cafile=path)
    except OSError as e:  # includes ssl.SSLError (no certificate in the file)
        raise ValueError(f"TEXT_CA_FILE {path!r} is not a readable PEM file with CA certificates ({e})") from None
    return path


def parse_patterns(s):
    """PASSTHROUGH_MODELS: comma-separated model names / fnmatch globs."""
    return tuple(p.strip() for p in (s or "").split(",") if p.strip())


_errors = []


def _setting(parse, *args, default=None):
    """parse(*args), or default after recording the operator-facing error (ImgFix.load reports it: exit 2)."""
    try:
        return parse(*args)
    except ValueError as e:  # vision_api.VisionConfigError is a ValueError
        _errors.append(str(e))
        return default


_env = os.environ.get
STORE_DIR = _env("IMGFIX_STORE", "/store")
TEXT = _setting(parse_text_url, _env("TEXT_URL"))   # None only together with a config error
TEXT_CA_FILE = _setting(check_ca_file, _env("TEXT_CA_FILE"), TEXT)
VISION = _setting(vision_api.from_env)   # None only together with a config error
PASSTHROUGH_MODELS = parse_patterns(_env("PASSTHROUGH_MODELS"))
CAPTION_MODE = _setting(parse_choice, "CAPTION_MODE", _env("CAPTION_MODE"), CAPTION_MODES, default="short")
CAPTION_TIMEOUT = _setting(parse_number, "CAPTION_TIMEOUT", _env("CAPTION_TIMEOUT"),
                           45.0 if CAPTION_MODE == "detailed" else 8.0, True, 600, default=8.0)  # s per caption
ASK_HINT = _setting(parse_choice, "ASK_HINT", _env("ASK_HINT"), ASK_HINTS, default="mcp")
ASK_URL = _setting(parse_ask_url, _env("ASK_URL"), default=DEFAULT_ASK_URL) if ASK_HINT == "http" else DEFAULT_ASK_URL
FETCH_REMOTE_IMAGES = _setting(parse_bool, "FETCH_REMOTE_IMAGES", _env("FETCH_REMOTE_IMAGES"), default=False)
FETCH_ALLOW_PRIVATE = _setting(parse_bool, "FETCH_ALLOW_PRIVATE", _env("FETCH_ALLOW_PRIVATE"), default=False)
MIN_FREE = int(_setting(parse_number, "IMGFIX_MIN_FREE_GB", _env("IMGFIX_MIN_FREE_GB"), 5.0, False, 1e6,
                        default=5.0) * (1 << 30))   # free bytes needed to store new files
CONFIG_ERRORS = tuple(_errors)

CAPTION_MAX_NEW = 4            # new captions per request; the rest get "" for good
CAPTION_CONCURRENCY = 2
CAPTION_PROMPT = "Describe this image in one short sentence of at most 20 words. Reply with the sentence only."
CAPTION_DETAILED_PROMPT = (
    "Describe this image for someone who cannot see it, in at most 200 words: what it shows, its layout and the "
    "important details. Quote all visible text exactly as written, in quotes. Reply with the description only.")
CAPTION_SPEC = {"short": (CAPTION_PROMPT, 60), "detailed": (CAPTION_DETAILED_PROMPT, 600)}  # prompt, max_tokens
DETAILED_CAPTION_CHARS = 3000
VISION_LONG_EDGE = 1600        # px, images are downscaled to this before going to the vision model
MAX_IMAGE_BYTES = 32 << 20     # decoded; larger => unavailable
MAX_PIXELS = 50_000_000        # larger => unavailable. MUST equal vision-mcp Config.max_pixels (shared contract)
MAX_PDF_BYTES = 64 << 20       # (mitmdump body_size_limit=64m caps whole requests, so ~47 MiB in practice)
PDF_TEXT_LIMIT = 60000         # chars of pdftotext output kept
PDF_TIMEOUT = 120              # s per poppler call
PDF_STDOUT_CAP = 1 << 20       # bytes of pdftotext output read (>> PDF_TEXT_LIMIT chars); more => killed, truncated
POPPLER_MEM = 768 << 20        # address-space cap per poppler process (prlimit --as)
DECODE_SLOTS = threading.BoundedSemaphore(2)  # full Pillow decodes at once, across all requests (memory bound)
FETCH_TIMEOUT = 10.0           # s per remote image URL, everything included (DNS, connects, redirects, body)
FETCH_MAX_REDIRECTS = 3
FETCH_MAX_PER_REQUEST = 8      # new URLs fetched per request; the rest are recorded as failed ("limit") for good
FETCH_POOL = concurrent.futures.ThreadPoolExecutor(4, thread_name_prefix="imgfix-fetch")  # fetches never on the loop
DNS_POOL = concurrent.futures.ThreadPoolExecutor(4, thread_name_prefix="imgfix-dns")      # lookups bounded by deadline
FETCH_HEADERS = [(b"accept", b"image/png,image/jpeg,image/gif,image/webp,image/*;q=0.8"),
                 (b"accept-encoding", b"identity"), (b"user-agent", b"blindsight-imgfix")]
COUNT_ONLY_SUFFIXES = ("/count_tokens", "/input_tokens", ":countTokens")
_PATH_MODEL = re.compile(r"/models/([^/:]+):[A-Za-z]+$")   # Gemini: /v1beta/models/<model>:generateContent

UNAVAILABLE = "[image unavailable: could not be decoded]"
EXT = {"image/png": "png", "image/jpeg": "jpg", "image/jpg": "jpg", "image/gif": "gif", "image/webp": "webp"}
MEDIA = {"png": "image/png", "jpg": "image/jpeg", "gif": "image/gif", "webp": "image/webp"}
PIL_EXT = {"PNG": "png", "JPEG": "jpg", "GIF": "gif", "WEBP": "webp"}
PIL_FORMATS = tuple(PIL_EXT)   # the only decoders Pillow may try (no TIFF/PSD/BMP/... on pasted bytes)
MEDIA_TYPES = ("image", "document", "image_url", "file", "input_image", "input_file", "computer_screenshot")
INPUT_TYPES = ("input_image", "input_file", "computer_screenshot")   # OpenAI Responses parts: replaced by input_text
GEMINI_KEYS = ("inline_data", "inlineData", "file_data", "fileData")  # Gemini media parts (protobuf JSON spellings)
OPAQUE_TYPES = ("tool_use", "server_tool_use", "mcp_tool_use", "function_call")  # model-written JSON, not content
OPAQUE_KEYS = ("arguments", "function_call", "functionCall")  # tool calls / their arguments: model-written, not walked
IMG_ID_RE = re.compile(r"img_[0-9a-f]{16}")
# Fail-closed checks on the parsed body (module doc). Key names are compared normalized: lower case, without "_".
ROUTING_KEYS = frozenset(("model", "messages", "input", "images", "prompt", "content", "type", "contents", "parts",
                          "systeminstruction", "generatecontentrequest", "inlinedata", "filedata", "mimetype",
                          "fileuri"))
RESIDUAL_TYPES = frozenset(("image", "document", "image_url", "input_image", "input_file", "file",
                            "computer_screenshot", "image_embeds", "image_pil"))
CONFIG_KEYS = frozenset(("tools", "functions", "toolchoice", "responseformat", "text", "format", "generationconfig",
                         "toolconfig"))   # top-level request configuration (schemas), never content
MAX_BODY = 64 << 20            # decoded request body (= mitmdump body_size_limit for the raw one)
NON_JSON_TYPES = ("multipart/", "application/octet-stream", "audio/", "video/")
MEDIA_MARKERS = re.compile(rb"image|document|input_file|file_data|file_id|base64|data:|screenshot|inline", re.I)
URL_SHOWN = 200                # characters of a URL / file id echoed into a reference text

Image.MAX_IMAGE_PIXELS = MAX_PIXELS


# ---------------------------------------------------------------- replacement texts (the byte-stable part)

_MCP_PDF_HINT = ' (page images: ask_image with image_id="{pid}" and page=<n>)'


def ask_hint(i):
    """How the model can look at image i (ASK_HINT). mcp is byte-identical to blindsight v1."""
    if ASK_HINT == "http":
        return (f"The model cannot see images. To inspect this one, run: curl -s -G {ASK_URL} --data-urlencode "
                f'image_id={i} --data-urlencode "question=<your question>"')
    if ASK_HINT == "none":
        return "The model cannot see images."
    return (f'The model cannot see images. To inspect this one, call the ask_image tool with image_id="{i}" and a '
            f'specific question.')


def pdf_hint(pid):
    if ASK_HINT == "http":
        return (f" (page images: curl -s -G {ASK_URL} --data-urlencode image_id={pid} --data-urlencode page=<n> "
                f'--data-urlencode "question=...")')
    if ASK_HINT == "none":
        return ""
    return _MCP_PDF_HINT.format(pid=pid)


def image_text(meta):
    i, cap = meta["id"], meta.get("caption") or "no preview"
    return f'[image {i} | {meta["width"]}x{meta["height"]} {EXT[meta["media_type"]]} | {cap}] ' + ask_hint(i)


def pdf_text(pid, pages, text, cut=False):
    """The STORED form of a PDF's replacement text (meta/<id>.txt): always with the mcp hint, so the store never
    depends on ASK_HINT (render_pdf swaps in the configured one). cut: pdftotext output exceeded PDF_STDOUT_CAP
    (always marked truncated)."""
    if cut or len(text) > PDF_TEXT_LIMIT:
        text = text[:PDF_TEXT_LIMIT] + "\n[... truncated]"
    return f"[pdf {pid} | {pages} pages] Text extracted with pdftotext{_MCP_PDF_HINT.format(pid=pid)}:\n{text}"


_PDF_HEAD = re.compile(r"\[pdf (pdf_[0-9a-f]{16}) \| \d+ pages\] Text extracted with pdftotext")


def render_pdf(stored):
    """Stored PDF text -> the text the model sees (the mcp hint replaced by the ASK_HINT one; anything else as is)."""
    if ASK_HINT == "mcp":
        return stored
    m = _PDF_HEAD.match(stored)
    if m:
        canon = m.group(0) + _MCP_PDF_HINT.format(pid=m.group(1)) + ":\n"
        if stored.startswith(canon):
            return m.group(0) + pdf_hint(m.group(1)) + ":\n" + stored[len(canon):]
    return stored


def url_id(url):
    return "url_" + hashlib.sha256(url.encode("utf-8", "surrogatepass")).hexdigest()[:16]


def _shown(s):
    """A URL or file id as echoed into a reference: never a data: URL (its bytes), anything else cut at URL_SHOWN."""
    if s[:5].lower() == "data:":
        return "(data URL)"
    return s if len(s) <= URL_SHOWN else s[:URL_SHOWN] + "..."


def url_text(kind, url):
    return f"[{kind} {url_id(url)} | remote {kind} not fetched: {_shown(url)}]"


def url_failed_text(url, error):
    return f"[image {url_id(url)} | remote image not fetched ({error}): {_shown(url)}]"


def file_id_text(kind, fid):
    """A provider-side upload (file_id only): its bytes are not in the request, so it cannot be shown."""
    if not isinstance(fid, str) or not fid:
        raise ValueError("no file_id")
    return f"[{kind} unavailable: file_id {_shown(fid)} not readable]"


_DELIMS = {ord("["): "(", ord("]"): ")", ord("|"): "/"}


def clean_caption(s, detailed=False):
    """One line, no placeholder delimiters, bounded length (the caption lands inside [... | ... | caption]).
    A reasoning model may emit its thinking first: keep only what follows the last </think>, and drop an unclosed
    <think> block (cut-off reasoning is never a caption; captions are stored for good). detailed: no word cap,
    <= DETAILED_CAPTION_CHARS."""
    s = s or ""
    if "</think>" in s:
        s = s.rsplit("</think>", 1)[1]
    s = s.split("<think>", 1)[0]
    s = " ".join(s.split()).translate(_DELIMS)
    if detailed:
        n = DETAILED_CAPTION_CHARS
        return s if len(s) <= n else s[:n - 3].rstrip() + "..."
    s = s.strip(" \"'")
    words = s.split(" ")
    if len(words) > 30:
        s = " ".join(words[:30]) + "..."
    return s[:300]


def image_meta(iid, ext, raw, w, h, caption):
    return {"id": iid, "kind": "image", "media_type": MEDIA[ext], "bytes": len(raw), "width": w, "height": h,
            "caption": caption, "created": _now()}


def url_record(uid, url, target=None, error=None):
    rec = {"id": uid, "kind": "url", "url": url}
    rec.update({"target": target} if target else {"failed": True, "error": error})
    rec["created"] = _now()
    return rec


def url_record_text(store, rec, url):
    """A URL's text from its write-once record (a failure, or the target image's own text). Raises if unusable."""
    if rec.get("kind") != "url":
        raise ValueError(f"invalid url record {rec.get('id')}")
    if rec.get("failed") is True:
        return url_failed_text(url, str(rec.get("error") or "error"))
    target = rec.get("target")
    meta = store.read_meta(target) if isinstance(target, str) and IMG_ID_RE.fullmatch(target) else None
    if not _image_meta_ok(meta):
        raise ValueError(f"url record {rec.get('id')} without a usable target")
    return image_text(meta)


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

class Undecodable(ValueError):
    """The bytes are not a usable image: a property of the bytes (a retry would fail the same way)."""


def _b64(data):
    if not isinstance(data, str):
        raise ValueError("no base64 data")
    return base64.b64decode(data.encode("ascii").translate(None, b" \t\r\n"), validate=True)


# data:<type>/<subtype>[;attr=value]*;base64,<data>  (RFC 2397; base64 required, nothing percent-encoded)
_DATA_URL = re.compile(r"data:([a-z0-9!#$&^_.+-]+/[a-z0-9!#$&^_.+-]+)((?:;[a-z0-9!#$&^_.+-]+=[^;,]*)*);base64,(.*)",
                       re.I | re.S)


def _data_url(s):
    """Strict data URL -> (media type, bytes). Raises ValueError."""
    m = _DATA_URL.fullmatch(s)
    if not m:
        raise ValueError("not a base64 data URL")
    return m.group(1).lower(), _b64(m.group(3))


def _is_http(url):
    try:
        u = urlsplit(url)
        return u.scheme in ("http", "https") and bool(u.hostname)
    except ValueError:
        return False


def _images(v):
    """An "images" list to replace (Ollama's base64 strings; any other non-empty list fails closed item by item)."""
    return isinstance(v, list) and len(v) > 0


def _gemini_key(d):
    return next((k for k in GEMINI_KEYS if k in d), None)


def _b64_proto(data):
    """Protobuf-JSON bytes (Gemini inline data): standard or URL-safe base64, padding optional. Raises ValueError."""
    if not isinstance(data, str):
        raise ValueError("no base64 data")
    s = data.encode("ascii").translate(None, b" \t\r\n").translate(bytes.maketrans(b"-_", b"+/"))
    return base64.b64decode(s + b"=" * (-len(s) % 4), validate=True)


def _probe(raw, media_type):
    """Fully decode once (first sight only) -> (w, h, ext), w x h as displayed (EXIF orientation applied, like
    vision-mcp and the caption path, so ask_image region fractions match). Raises Undecodable."""
    try:
        with DECODE_SLOTS, Image.open(io.BytesIO(raw), formats=PIL_FORMATS) as im:
            w, h = im.size
            if w * h > MAX_PIXELS:
                raise ValueError("too many pixels")
            fmt = PIL_EXT.get(im.format)
            if im.getexif().get(0x0112) in (5, 6, 7, 8):  # transposed/rotated 90 or 270 degrees on display
                w, h = h, w
            im.load()
    except MemoryError:
        raise
    except Exception as e:
        raise Undecodable(type(e).__name__) from e
    ext = EXT.get(str(media_type).lower()) or fmt  # ext follows the declared media type, else the real format
    if not ext:
        raise Undecodable("unsupported image format")
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


# ---------------------------------------------------------------- request body checks (fail closed, module doc)

class Reject(Exception):
    """The request must not be forwarded: HTTP 400 with str(e) as error.message. Not a ValueError, so no parser or
    per-part handler swallows it."""


def _norm(k):
    return k.lower().replace("_", "")


def _keys(d):
    """Normalized key -> the object's own key (unique for ROUTING_KEYS, see _unique_keys)."""
    return {_norm(k): k for k in d}


def _unique_keys(pairs):
    """json object_pairs_hook: refuse a repeated key, and two spellings (letter case, "_") of one routing key. JSON
    parsers differ on which duplicate wins, and lenient upstreams match field names loosely."""
    d = dict(pairs)
    if len(d) != len(pairs):
        dup = Counter(k for k, _ in pairs).most_common(1)[0][0]
        raise Reject(f"The request repeats the JSON key {dup[:40]!r} in one object, which an upstream could read "
                     "either way; it was not forwarded.")
    if len(d) > 1:
        seen = set()
        for k in d:
            n = _norm(k)
            if n in ROUTING_KEYS:
                if n in seen:
                    raise Reject(f"The request spells the JSON key {n!r} twice in one object (differing in letter case "
                                 "or '_'), which an upstream could read either way; it was not forwarded.")
                seen.add(n)
    return d


def decode_body(raw, encoding):
    """The request body per its Content-Encoding: identity, gzip/x-gzip, deflate, br or zstd (one coding), at most
    MAX_BODY bytes decoded, all input consumed (no trailing data an upstream decoder might read differently).
    Raises Reject."""
    enc = (encoding or "").strip().lower()
    if enc in ("", "identity"):
        return raw
    try:
        if enc in ("gzip", "x-gzip"):
            return _inflate(raw, 47)  # gzip (or zlib) header
        if enc == "deflate":  # RFC 9110: zlib format; some clients send raw deflate
            try:
                return _inflate(raw, 15)
            except zlib.error:
                return _inflate(raw, -15)
        if enc == "br" and brotli is not None:
            return _unbrotli(raw)
        if enc == "zstd" and zstandard is not None:
            return _unzstd(raw)
    except Exception as e:  # noqa: BLE001 - zlib.error, brotli.error, zstandard.ZstdError, ValueError (ours)
        raise Reject(f"imgfix could not decode the request body (Content-Encoding {enc[:30]!r}: {e}); it was not "
                     "forwarded.") from None
    raise Reject(f"imgfix cannot decode Content-Encoding {enc[:30]!r} (supported: gzip, deflate, br, zstd), so it "
                 "cannot check the body for images; it was not forwarded.")


def _inflate(raw, wbits):
    d = zlib.decompressobj(wbits)
    out = d.decompress(raw, MAX_BODY + 1)
    if len(out) > MAX_BODY or d.unconsumed_tail:
        raise ValueError("more than 64 MiB decoded")
    if not d.eof or d.unused_data:
        raise ValueError("truncated, or data after the end of the stream")
    return out


def _unbrotli(raw):
    d = brotli.Decompressor()
    out = bytearray(d.process(raw, output_buffer_limit=1 << 20))
    while not d.is_finished():
        if len(out) > MAX_BODY:
            raise ValueError("more than 64 MiB decoded")
        if d.can_accept_more_data():  # all input used, stream not finished
            raise ValueError("truncated stream")
        out += d.process(b"", output_buffer_limit=1 << 20)
    if len(out) > MAX_BODY:
        raise ValueError("more than 64 MiB decoded")
    return bytes(out)


def _unzstd(raw):
    out = bytearray()
    with zstandard.ZstdDecompressor().stream_reader(io.BytesIO(raw), read_across_frames=True) as r:
        while len(out) <= MAX_BODY:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            out += chunk
    if len(out) > MAX_BODY:
        raise ValueError("more than 64 MiB decoded")
    return bytes(out)


def non_json_forwardable(content_type, body):
    """May a body that is not strict JSON go on unchanged? Only if it is clearly not a JSON API request: an explicit
    non-JSON Content-Type (multipart/*, application/octet-stream, audio/*, video/*), not starting with { or [, and no
    media marker anywhere in it. Anything else that does not parse could be read as JSON by a lenient upstream."""
    mt = (content_type or "").split(";", 1)[0].strip().lower()
    return (mt.startswith(NON_JSON_TYPES) and body.lstrip(b" \t\r\n\xef\xbb\xbf")[:1] not in (b"{", b"[")
            and not MEDIA_MARKERS.search(body))


# ---------------------------------------------------------------- the rewrite

@dataclass
class _Slot:
    """A replaced part. json.dumps(default=_slot_json) turns it into a text part."""
    text: str | None
    new_id: str | None = None   # first-sight image: finish() fills text once its meta is written
    extra: dict | None = None   # cache_control carried over from the original part
    wrap: str = "text"          # "text" | "input_text" (OpenAI Responses) | "gemini" ({"text": ...}, no type)
    url_id: str | None = None   # fetched remote image: finish() fills text once its URL record is written


@dataclass
class _Joined:
    """Ollama "content"/"prompt": the original string + the REFs of its removed images, one per line."""
    head: str
    slots: list


@dataclass
class _NewImage:
    id: str
    ext: str
    raw: bytes
    width: int
    height: int


@dataclass
class Prepared:
    obj: object = None                          # parsed body, media parts already swapped for _Slot
    changed: bool = False
    new: dict = field(default_factory=dict)     # id -> _NewImage, in order of appearance
    slots: list = field(default_factory=list)   # slots waiting for a new image's meta (or a URL record)
    fetch: dict = field(default_factory=dict)   # first-sight remote image URL -> its slots (FETCH_REMOTE_IMAGES)
    url_new: dict = field(default_factory=dict)  # url id -> (url, new image id): record written by finish()
    stats: Counter = field(default_factory=Counter)
    ids: list = field(default_factory=list)


def _slot_json(o):
    if isinstance(o, _Slot):
        if o.wrap == "gemini":
            return {"text": o.text}
        d = {"type": o.wrap, "text": o.text}
        if o.extra:
            d.update(o.extra)
        return d
    if isinstance(o, _Joined):
        return o.head + "\n".join(s.text for s in o.slots)
    raise TypeError(f"not JSON serializable: {type(o).__name__}")


def _is_media(x):
    """A media part: a known "type", a Gemini inline/file data part, or an untyped vLLM {"image_url": ...} part."""
    return isinstance(x, dict) and (x.get("type") in MEDIA_TYPES or _gemini_key(x) is not None
                                    or ("type" not in x and "image_url" in x))


def _as_slot(o):
    """A replaced part's output as one text slot (a nested text part keeps its text; anything else is unavailable)."""
    if isinstance(o, _Slot):
        return o
    return _Slot(o["text"] if isinstance(o, dict) and isinstance(o.get("text"), str) else UNAVAILABLE)


def _img_id(raw):
    return "img_" + hashlib.sha256(raw).hexdigest()[:16]


class _Rewriter:
    def __init__(self, store, count_only, prep):
        self.store, self.count_only, self.p = store, count_only, prep
        self.untyped_wrap = "text"   # for untyped {"image_url": ...} parts: input_text under Responses keys

    def request(self, d):
        """Rewrite one request object through its content keys (any letter case or snake/camel spelling)."""
        keys = _keys(d)
        k = keys.get("images")
        if k is not None and _images(d[k]):  # Ollama /api/generate
            self.images(d, k, "prompt")
        k = keys.get("messages")
        if k is not None and isinstance(d[k], list):
            for m in d[k]:  # Ollama /api/chat messages, with or without "role"
                ik = _keys(m).get("images") if isinstance(m, dict) else None
                if ik is not None and _images(m[ik]):
                    self.images(m, ik, "content")
            d[k] = self.walk(d[k])
        for name, wrap in (("input", "input_text"), ("prompt", "input_text"), ("contents", "text"),
                           ("systeminstruction", "text")):
            k = keys.get(name)
            if k is not None and isinstance(d[k], (list, dict)):  # (Ollama's prompt is a string: nothing to walk)
                self.untyped_wrap = wrap
                d[k] = self.walk(d[k])
                self.untyped_wrap = "text"
        k = keys.get("generatecontentrequest")
        if k is not None and isinstance(d[k], dict):  # Gemini countTokens: a whole request inside
            self.request(d[k])

    def walk(self, node):
        """Replace media parts in every list/dict below node (messages, tool_result.content, ... any depth)."""
        if isinstance(node, list):
            out = []
            for item in node:
                out.extend(self.replace(item) if _is_media(item) else (self.walk(item),))
            return out
        if isinstance(node, dict) and node.get("type") not in OPAQUE_TYPES:
            for k, v in node.items():
                if k in OPAQUE_KEYS:
                    continue
                if _is_media(v):  # e.g. Responses computer_call_output.output, Responses prompt.variables
                    r = self.replace(v)
                    node[k] = r[0] if len(r) == 1 else r
                elif isinstance(v, (list, dict)):
                    node[k] = self.walk(v)
        return node

    def replace(self, part):
        kind = part.get("type")
        if kind in MEDIA_TYPES:
            handler, wrap = getattr(self, "part_" + kind), "input_text" if kind in INPUT_TYPES else "text"
        elif (kind := _gemini_key(part)) is not None:
            handler, wrap = self.part_gemini, "gemini"
        else:  # untyped {"image_url": ...} (vLLM)
            handler, wrap, kind = self.part_image_url, self.untyped_wrap, "untyped image_url"
        try:
            out = handler(part)
        except Exception as e:  # noqa: BLE001 - any failure in a part => fail closed for that part
            self.p.stats["failed"] += 1
            log.warning("imgfix: %s part unavailable (%s)", kind, type(e).__name__)
            out = [_Slot(UNAVAILABLE)]
        self.p.changed = True
        for s in out:
            if isinstance(s, _Slot):
                s.wrap = wrap
        if "cache_control" in part and out:
            last = out[-1]
            if isinstance(last, _Slot):
                last.extra = {"cache_control": part["cache_control"]}
            elif isinstance(last, dict):
                last.setdefault("cache_control", part["cache_control"])
        return out

    def _id(self, prefix, raw):
        mid = prefix + hashlib.sha256(raw).hexdigest()[:16]
        if mid not in self.p.ids:
            self.p.ids.append(mid)
        return mid

    # -------- Anthropic Messages

    def part_image(self, part):
        src = part.get("source")
        st = src.get("type") if isinstance(src, dict) else None
        if st == "url":
            return [self.url_source(src.get("url"), "image")]
        if st == "file":
            return [_Slot(file_id_text("image", src.get("file_id")))]
        if st != "base64":
            raise ValueError(f"image source {st!r}")
        return [self.image_raw(_b64(src.get("data")), src.get("media_type"))]

    def part_document(self, part):
        src = part.get("source")
        st = src.get("type") if isinstance(src, dict) else None
        if st == "text":  # plain-text document -> plain text part
            if not isinstance(src.get("data"), str):
                raise ValueError("text document without data")
            self.p.stats["text_docs"] += 1
            return [{"type": "text", "text": src["data"]}]
        if st == "content":  # custom-content document -> its own parts (images inside are replaced too)
            c = src.get("content")
            if isinstance(c, str):
                return [{"type": "text", "text": c}]
            if not isinstance(c, list):
                raise ValueError("content document without content")
            self.p.stats["content_docs"] += 1
            return self.walk(c)
        if st == "url":
            return [self.url_source(src.get("url"), "document")]
        if st == "file":
            return [_Slot(file_id_text("file", src.get("file_id")))]
        if st != "base64":
            raise ValueError(f"document source {st!r}")
        return [self.media(_b64(src.get("data")), str(src.get("media_type") or "application/pdf").lower())]

    def url_source(self, url, kind):
        """An Anthropic url source: a data: URL is decoded like base64 data (never echoed); anything else is a remote
        image (remote_image) or a remote document (never fetched)."""
        if not isinstance(url, str):
            raise ValueError("url is not a string")
        if url[:5].lower() == "data:":
            return self.data_part(url)
        if kind == "image":
            return self.remote_image(url)
        self.p.stats["url"] += 1
        return _Slot(url_text("document", url))

    # -------- OpenAI Chat Completions (and vLLM's untyped {"image_url": ...})

    def part_image_url(self, part):
        iu = part.get("image_url")
        return [self.url_part(iu.get("url") if isinstance(iu, dict) else iu, "image")]

    def part_file(self, part):
        f = part.get("file")
        if not isinstance(f, dict):
            raise ValueError("file part without file")
        return [self.file_part(f.get("file_data"), None, f.get("file_id"))]

    # -------- OpenAI Responses

    def part_input_image(self, part):
        iu = part.get("image_url")
        if iu is not None:  # a string; a Chat-style {"url": ...} object is tolerated
            return [self.url_part(iu.get("url") if isinstance(iu, dict) else iu, "image")]
        return [_Slot(file_id_text("image", part.get("file_id")))]

    part_computer_screenshot = part_input_image  # computer_call_output.output: same fields

    def part_input_file(self, part):
        return [self.file_part(part.get("file_data"), part.get("file_url"), part.get("file_id"))]

    # -------- Ollama native

    def images(self, d, key, target):
        """d[key], an "images" list (base64 strings or data URLs; a media part item gets its REF, any other item the
        unavailable text) -> removed; "\\n\\n" + one REF per line appended to d's target key ("content" / "prompt",
        in d's own spelling; created if missing)."""
        slots = []
        for item in d.pop(key):
            try:
                if isinstance(item, str):
                    out = [self.data_part(item) if item[:5].lower() == "data:" else self.image_raw(_b64(item), None)]
                elif _is_media(item):
                    out = [_as_slot(o) for o in self.replace(item)]
                else:
                    raise ValueError("not an image")
            except Exception as e:  # noqa: BLE001 - fail closed for that image
                self.p.stats["failed"] += 1
                log.warning("imgfix: ollama image unavailable (%s)", type(e).__name__)
                out = [_Slot(UNAVAILABLE)]
            slots.extend(out)
        self.p.changed = True
        tk = _keys(d).get(target, target)
        cur = d.get(tk)
        if isinstance(cur, list):  # not Ollama's shape (a string), but keep the REFs anyway
            for s in slots:
                s.wrap = "text"
            cur.extend(slots)
        else:
            d[tk] = _Joined((cur if isinstance(cur, str) else "") + "\n\n", slots)

    # -------- Gemini native

    def part_gemini(self, part):
        """{"inline_data"|"inlineData": {mime type, base64 data}} or {"file_data"|"fileData": {mime type, URI}}."""
        k = _gemini_key(part)
        blob = part[k]
        if not isinstance(blob, dict):
            raise ValueError(f"{k} is not an object")
        keys = _keys(blob)
        mime = blob.get(keys.get("mimetype"))
        mime = mime.strip().lower() if isinstance(mime, str) else ""
        if k in ("inline_data", "inlineData"):
            return [self.media(_b64_proto(blob.get("data")), mime)]
        uri = blob.get(keys.get("fileuri"))
        if not isinstance(uri, str) or not uri:
            raise ValueError("file_data without file_uri")
        if uri[:5].lower() == "data:":
            return [self.data_part(uri)]
        if _is_http(uri) and urlsplit(uri).hostname != "generativelanguage.googleapis.com":
            if not mime or mime.startswith("image/"):
                return [self.remote_image(uri)]
            self.p.stats["url"] += 1
            return [_Slot(url_text("document", uri))]
        return [_Slot(file_id_text("file", uri))]  # Gemini Files API / gs:// upload: not readable here

    # -------- shared by the formats

    def file_part(self, data, url, fid):
        if data is not None:
            return self.data_part(data)
        if url is not None:
            return self.url_part(url, "document")
        return _Slot(file_id_text("file", fid))

    def url_part(self, url, kind):
        """An OpenAI url field: a data URL or an http(s) URL (anything else fails closed: never echoed as text)."""
        if not isinstance(url, str):
            raise ValueError("url is not a string")
        if url[:5].lower() == "data:":
            return self.data_part(url)
        if not _is_http(url):
            raise ValueError("unsupported url")
        if kind == "image":
            return self.remote_image(url)
        self.p.stats["url"] += 1
        return _Slot(url_text("document", url))

    def data_part(self, s):
        mt, raw = _data_url(s)
        return self.media(raw, mt)

    def media(self, raw, media_type):
        """Decoded bytes by media type: images, PDFs, plain text; anything else fails closed."""
        if media_type.startswith("image/"):
            return self.image_raw(raw, media_type)
        if media_type == "application/pdf":
            return self.pdf_raw(raw)
        if media_type == "text/plain":
            self.p.stats["text_docs"] += 1
            return _Slot(raw.decode("utf-8"))
        raise ValueError(f"unsupported media type {media_type[:40]!r}")

    def image_raw(self, raw, media_type):
        if len(raw) > MAX_IMAGE_BYTES:
            raise Undecodable("image too large")
        iid = self._id("img_", raw)
        self.p.stats["images"] += 1
        meta = self.store.read_meta(iid)  # raises on a broken meta file -> unavailable (fixed text, no re-caption)
        if meta is not None:  # seen before: meta alone drives the text, no decode needed
            if not _image_meta_ok(meta):
                raise ValueError(f"invalid image meta {iid}")
            if not self.count_only:  # self-heal a missing raw file
                self.store.heal(self.store.img_path(iid, EXT[meta["media_type"]]), raw)
            return _Slot(image_text(meta))
        if iid in self.p.new:  # same new image twice in one request
            return self._pending(iid)
        w, h, ext = _probe(raw, media_type)
        if self.count_only:
            return _Slot(image_text(image_meta(iid, ext, raw, w, h, "")))
        self.store.write_once(self.store.img_path(iid, ext), raw)  # raw before meta: meta implies the file
        self.p.new[iid] = _NewImage(iid, ext, raw, w, h)
        return self._pending(iid)

    def _pending(self, iid):
        s = _Slot(None, new_id=iid)
        self.p.slots.append(s)
        return s

    def remote_image(self, url):
        """An image URL. FETCH_REMOTE_IMAGES off (or not http(s)): the fixed not-fetched text. On: the URL's write-once
        record decides (first sight: queued for fetch_all, then settle(); count-only paths never fetch)."""
        self.p.stats["url"] += 1
        if not (FETCH_REMOTE_IMAGES and _is_http(url)):
            return _Slot(url_text("image", url))
        uid = url_id(url)
        if uid not in self.p.ids:
            self.p.ids.append(uid)
        rec = self.store.read_meta(uid)  # raises on a broken record -> unavailable (fixed text, never a refetch)
        if rec is not None:
            return _Slot(url_record_text(self.store, rec, url))
        if self.count_only:
            return _Slot(url_text("image", url))
        slots = self.p.fetch.get(url)
        if slots is None:
            if len(self.p.fetch) >= FETCH_MAX_PER_REQUEST:
                self.p.stats["fetch_failed"] += 1
                rec = self.store.write_meta(url_record(uid, url, error="limit"))
                return _Slot(url_record_text(self.store, rec, url))
            slots = self.p.fetch[url] = []
        s = _Slot(None)
        slots.append(s)
        return s

    def pdf_raw(self, raw):
        if len(raw) > MAX_PDF_BYTES:
            raise ValueError("pdf too large")
        pid = self._id("pdf_", raw)
        self.p.stats["pdfs"] += 1
        return _Slot(render_pdf(self.pdf(pid, raw)))

    def pdf(self, pid, raw):
        """Stored replacement text for a PDF. Poppler runs on first sight only; the text is then stored write-once in
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
        self.p.stats["new_pdfs"] += 1
        return text


def _passthrough(obj, path):
    """True if every model name the request carries matches PASSTHROUGH_MODELS: the body's "model" (any spelling),
    Gemini's generateContentRequest.model and path models/<model>:method ("models/" prefix ignored). A request that
    names no model, or any non-matching one, is rewritten (a second model field cannot smuggle images through)."""
    names = []
    if isinstance(obj, dict):
        for d in (obj, obj.get(_keys(obj).get("generatecontentrequest"))):
            if isinstance(d, dict) and (k := _keys(d).get("model")) is not None:
                names.append(d[k])
    m = _PATH_MODEL.search(path or "")
    if m:
        names.append(unquote(m.group(1)))
    return bool(names) and all(isinstance(n, str) and any(fnmatch.fnmatchcase(n.removeprefix("models/"), pat)
                                                          for pat in PASSTHROUGH_MODELS) for n in names)


def _residual(node, top=False):
    """The structural backstop: a description of the first media-shaped node left in the outgoing body, or None.
    Skips model-written JSON like walk() and, at the top level, request configuration (CONFIG_KEYS). Keys ("_"
    ignored) and "type" values are compared in any letter case; strings are never inspected as media."""
    if isinstance(node, list):
        for x in node:
            if found := _residual(x):
                return found
        return None
    if not isinstance(node, dict) or node.get("type") in OPAQUE_TYPES:
        return None
    typed = False
    for k, v in node.items():
        n = _norm(k)
        if n == "type" and isinstance(v, str):
            typed = True
            if v.lower() in RESIDUAL_TYPES:
                return f'a part of type "{v.lower()}"'
        elif n in ("inlinedata", "filedata"):
            return f"a Gemini {k[:40]!r} part"
        elif n == "images" and isinstance(v, list) and v:
            return 'a non-empty "images" list'
        elif (n == "source" and isinstance(v, dict)
              and str(v.get(_keys(v).get("type"), "")).lower() in ("base64", "url", "file")):
            return 'an Anthropic base64/url/file "source"'
    if not typed and any(_norm(k) == "imageurl" and isinstance(v, (str, dict)) for k, v in node.items()):
        return 'an untyped "image_url" part'
    for k, v in node.items():
        if k in OPAQUE_KEYS or (top and _norm(k) in CONFIG_KEYS) or not isinstance(v, (list, dict)):
            continue
        if found := _residual(v, top and _norm(k) == "generatecontentrequest"):
            return found
    return None


def prepare(raw, store, count_only=False, path=""):
    """Parse + swap media parts (sync: JSON, hashing, Pillow, poppler, store writes), then the structural backstop.
    None if the body is not strict JSON. Raises Reject (ambiguous keys; media left after the rewrite)."""
    try:
        obj = json.loads(raw, object_pairs_hook=_unique_keys)
    except (ValueError, RecursionError):  # not JSON: syntax, invalid UTF-8, nesting deeper than the parser allows
        return None
    p = Prepared(obj=obj)
    if PASSTHROUGH_MODELS and _passthrough(obj, path):
        p.stats["passthrough"] = 1  # a model that can see for itself: forwarded unchanged, images intact
        return p
    try:
        if isinstance(obj, dict):  # any other model name: every content key is scanned
            _Rewriter(store, count_only, p).request(obj)
        found = _residual(obj, top=True)
    except RecursionError:  # parsed (the C parser goes deeper than Python recursion) but too deep to check
        raise Reject("The request's JSON is nested too deeply for imgfix to check it for images; it was not "
                     "forwarded.") from None
    if found:
        raise Reject(f"This request still holds {found} after imgfix's rewrite (a shape imgfix cannot replace), so it "
                     "was not forwarded to the text-only model.")
    return p


def settle(p, store, fetched):
    """Decide each first-sight URL from its fetch result (sync). fetched: url -> (bytes, None) | (None, reason).
    A known image gets its URL record now; a new one is stored like a pasted image and its record is written by
    finish() after its meta. Every failure caused by the URL is recorded for good; a store failure (disk low, broken
    meta) gives the unavailable text and no record."""
    rw = _Rewriter(store, False, p)
    for url, slots in p.fetch.items():
        uid, (raw, err) = url_id(url), fetched.get(url, (None, "timeout"))
        try:
            if raw is not None:
                try:
                    s = rw.image_raw(raw, None)  # the real format decides the extension
                except Undecodable:
                    err = "undecodable"
                else:
                    if s.text is None:
                        p.url_new[uid] = (url, s.new_id)
                        for x in slots:
                            x.url_id = uid
                            p.slots.append(x)
                        p.stats["fetched"] += 1
                        continue
                    rec = store.write_meta(url_record(uid, url, target=_img_id(raw)))
            if err:
                rec = store.write_meta(url_record(uid, url, error=err))
            text = url_record_text(store, rec, url)
            p.stats["fetch_failed" if err else "fetched"] += 1
        except Exception as e:  # noqa: BLE001 - fail closed for this URL
            p.stats["failed"] += 1
            log.warning("imgfix: remote image %s unavailable (%s)", uid, type(e).__name__)
            text = UNAVAILABLE
        for x in slots:
            x.text = text


def finish(p, store, captions):
    """Write-once meta for first-sight images (then their URL records), fill their texts, serialize (sync)."""
    texts, failed = {}, set()
    for n in p.new.values():
        try:
            meta = store.write_meta(image_meta(n.id, n.ext, n.raw, n.width, n.height, captions.get(n.id, "")))
            texts[n.id] = image_text(meta)
        except Exception as e:  # noqa: BLE001 - e.g. store full: fail closed for this image
            p.stats["failed"] += 1
            log.warning("imgfix: meta write failed for %s (%s)", n.id, type(e).__name__)
            texts[n.id] = UNAVAILABLE
            failed.add(n.id)
    urls = {}
    for uid, (url, iid) in p.url_new.items():
        try:
            if iid in failed:  # no record: the next turn may try again
                raise ValueError(f"target {iid} not stored")
            urls[uid] = url_record_text(store, store.write_meta(url_record(uid, url, target=iid)), url)
        except Exception as e:  # noqa: BLE001
            p.stats["failed"] += 1
            log.warning("imgfix: url record failed for %s (%s)", uid, type(e).__name__)
            urls[uid] = UNAVAILABLE
    for s in p.slots:
        s.text = urls[s.url_id] if s.url_id else texts[s.new_id]
    return json.dumps(p.obj, ensure_ascii=True, separators=(",", ":"), default=_slot_json).encode("ascii")


async def caption_new(new, caption):
    """Caption the newest CAPTION_MAX_NEW first-sight images, CAPTION_CONCURRENCY at a time. id -> caption|""."""
    sem, detailed = asyncio.Semaphore(CAPTION_CONCURRENCY), CAPTION_MODE == "detailed"

    async def one(iid):
        async with sem:
            try:
                async with asyncio.timeout(CAPTION_TIMEOUT):
                    return iid, clean_caption(await caption(new[iid].raw), detailed)
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


async def rewrite(raw, store, caption=None, count_only=False, path=""):
    """Body bytes -> Result. caption: async (image bytes) -> str, or None for no captions. path: the request path
    (Gemini names the model there). Raises Reject (the request must not be forwarded)."""
    p = await asyncio.to_thread(prepare, raw, store, count_only, path)
    if p is None:
        return Result(None, Counter(not_json=1), [])
    if not p.changed:
        return Result(None, p.stats, p.ids)
    if p.fetch:  # first-sight remote images (never on count-only paths: prepare queues none there)
        await asyncio.to_thread(settle, p, store, await fetch_all(list(p.fetch)))
    caps = await caption_new(p.new, caption) if (p.new and caption) else {}
    p.stats["new"] = len(p.new)
    p.stats["captioned"] = sum(1 for c in caps.values() if c)
    body = await asyncio.to_thread(finish, p, store, caps)
    return Result(body, p.stats, p.ids)


# ---------------------------------------------------------------- remote image fetcher (FETCH_REMOTE_IMAGES)

class FetchError(Exception):
    """A failed fetch; str(e) is the short, stable reason stored in the URL record (and shown to the model)."""


def address_ok(ip):
    """May a fetch connect to ip? Only public addresses, unless FETCH_ALLOW_PRIVATE (SSRF guard)."""
    if FETCH_ALLOW_PRIVATE:
        return True
    a = ipaddress.ip_address(ip.split("%", 1)[0])
    if a.version == 6 and (a.ipv4_mapped or a.sixtofour):  # judge the embedded IPv4 address
        a = a.ipv4_mapped or a.sixtofour
    return a.is_global and not (a.is_private or a.is_loopback or a.is_link_local or a.is_reserved or a.is_multicast
                                or a.is_unspecified)


class _Deadline:
    def __init__(self, seconds):
        self.at = time.monotonic() + seconds

    def left(self, timeout=None):
        left = self.at - time.monotonic()
        if left <= 0:
            raise FetchError("timeout")
        return left if timeout is None else min(timeout, left)


class _GuardedStream:
    """httpcore network stream whose every read/write/handshake ends by the fetch deadline."""

    def __init__(self, inner, deadline):
        self.inner, self.deadline = inner, deadline

    def read(self, max_bytes, timeout=None):
        return self.inner.read(max_bytes, self.deadline.left(timeout))

    def write(self, buffer, timeout=None):
        self.inner.write(buffer, self.deadline.left(timeout))

    def close(self):
        self.inner.close()

    def start_tls(self, ssl_context, server_hostname=None, timeout=None):
        return _GuardedStream(self.inner.start_tls(ssl_context, server_hostname, self.deadline.left(timeout)),
                              self.deadline)

    def get_extra_info(self, info):
        return self.inner.get_extra_info(info)


class _GuardedBackend:
    """httpcore network backend for one fetch: resolves the host itself, refuses the connection unless EVERY resolved
    address passes address_ok, then connects to one of those checked addresses (never re-resolving, so DNS rebinding
    cannot swap in another one). TLS still uses the host name (SNI + certificate check). Applies to every redirect hop
    (each new host is a new connection)."""

    def __init__(self, deadline):
        self.deadline, self.inner = deadline, httpcore.SyncBackend()

    def connect_tcp(self, host, port, timeout=None, local_address=None, socket_options=None):
        lookup = DNS_POOL.submit(socket.getaddrinfo, host, port, type=socket.SOCK_STREAM)
        try:
            infos = lookup.result(self.deadline.left())
        except TimeoutError:  # concurrent.futures.TimeoutError is TimeoutError (an OSError): catch it first
            raise FetchError("timeout") from None
        except OSError:
            raise FetchError("dns failed") from None
        addrs = list(dict.fromkeys(str(i[4][0]) for i in infos))
        if not addrs or not all(address_ok(a) for a in addrs):
            raise FetchError("blocked address")
        err = None
        for a in addrs:
            try:
                stream = self.inner.connect_tcp(a, port, self.deadline.left(timeout), local_address, socket_options)
                return _GuardedStream(stream, self.deadline)
            except httpcore.ConnectError as e:
                err = e
        raise err

    def connect_unix_socket(self, path, timeout=None, socket_options=None):  # pragma: no cover - never used
        raise FetchError("bad url")

    def sleep(self, seconds):  # pragma: no cover - retries=0
        time.sleep(seconds)


def fetch_image(url):
    """GET one remote image (sync, in FETCH_POOL) -> (bytes, None) | (None, reason). http(s) only, at most
    FETCH_MAX_REDIRECTS redirects (each re-checked), body <= MAX_IMAGE_BYTES (streamed), FETCH_TIMEOUT in total.
    Never raises."""
    deadline = _Deadline(FETCH_TIMEOUT)
    try:
        ctx_ = ssl.create_default_context(cafile=certifi.where())
        with httpcore.ConnectionPool(ssl_context=ctx_, network_backend=_GuardedBackend(deadline), max_connections=1,
                                     retries=0) as pool:
            for _ in range(FETCH_MAX_REDIRECTS + 1):
                if not (url.isascii() and _is_http(url)):
                    raise FetchError("bad url")
                t = deadline.left()
                with pool.stream("GET", url, headers=FETCH_HEADERS,
                                 extensions={"timeout": dict.fromkeys(("connect", "read", "write", "pool"), t)}) as r:
                    hdrs = {k.lower(): v for k, v in r.headers}
                    if r.status in (301, 302, 303, 307, 308) and hdrs.get(b"location"):
                        url = urljoin(url, hdrs[b"location"].decode("latin-1"))
                        continue
                    if r.status != 200:
                        raise FetchError(f"http {r.status}")
                    if hdrs.get(b"content-encoding", b"identity").strip().lower() != b"identity":
                        raise FetchError("encoded body")  # no decompression bombs
                    size = hdrs.get(b"content-length", b"").strip()
                    if size.isdigit() and int(size) > MAX_IMAGE_BYTES:
                        raise FetchError("too large")
                    buf = bytearray()
                    for chunk in r.iter_stream():
                        buf += chunk
                        if len(buf) > MAX_IMAGE_BYTES:
                            raise FetchError("too large")
                    return bytes(buf), None
            raise FetchError("too many redirects")
    except FetchError as e:
        return None, str(e)
    except httpcore.TimeoutException:
        return None, "timeout"
    except httpcore.ConnectError:  # incl. TLS certificate errors
        return None, "connect failed"
    except Exception as e:  # noqa: BLE001 - protocol errors etc.
        return None, f"network error {type(e).__name__}"


async def fetch_all(urls):
    """Fetch first-sight URLs in FETCH_POOL threads (never on the event loop).
    -> url -> (bytes, None) | (None, reason)."""
    loop = asyncio.get_running_loop()
    got = await asyncio.gather(*(loop.run_in_executor(FETCH_POOL, fetch_image, u) for u in urls))
    for u, (_, err) in zip(urls, got):
        log.info("imgfix: fetch %s %s", url_id(u), f"failed ({err})" if err else "ok")
    return dict(zip(urls, got))


# ---------------------------------------------------------------- vision-model captioner

def _caption_request(raw, api, mode):
    """Caption request body (vision_api wire format) for one image: downscaled PNG + the CAPTION_MODE prompt."""
    prompt, max_tokens = CAPTION_SPEC[mode]
    return vision_api.build_body(api, vision_png(raw), prompt, max_tokens=max_tokens, temperature=0)


class VisionCaptioner:
    """async (image bytes) -> caption via the vision model (VISION_API openai | anthropic, see vision_api.py).
    Never blocks the event loop."""

    def __init__(self, api):
        self.api, self.client = api, None

    async def __call__(self, raw):
        if self.client is None:
            # trust_env=False: a Docker-injected HTTP(S)_PROXY must not divert caption calls (each failure is a ""
            # for good)
            self.client = httpx.AsyncClient(timeout=CAPTION_TIMEOUT, limits=httpx.Limits(max_connections=4),
                                            trust_env=False)
        body = await asyncio.to_thread(_caption_request, raw, self.api, CAPTION_MODE)  # resize + base64 off the loop
        r = await self.client.post(self.api.url, content=body, headers=self.api.headers)
        r.raise_for_status()
        text, truncated = vision_api.parse_reply(self.api, r.json())
        if truncated:
            # The caption fits easily in max_tokens, so a cut-off reply is reasoning or rambling (e.g. a thinking
            # model without VISION_EXTRA_BODY turning thinking off). No caption beats a wrong one for good.
            log.info("imgfix: caption cut off at max_tokens; stored without caption "
                     "(thinking model? see VISION_EXTRA_BODY)")
            return ""
        return text

    async def aclose(self):
        if self.client is not None:
            await self.client.aclose()
            self.client = None


# ---------------------------------------------------------------- mitmproxy hooks

def _error(msg):
    """400 with a body Anthropic, OpenAI and Gemini clients all read (error.type / error.status + error.message); they
    do not retry a 400."""
    body = json.dumps({"type": "error", "error": {"type": "invalid_request_error", "status": "INVALID_ARGUMENT",
                                                  "message": msg}}).encode()
    return http.Response.make(400, body, {"content-type": "application/json"})


class ImgFix:
    def __init__(self):
        self.store = Store(STORE_DIR)
        self.captioner = VisionCaptioner(VISION) if VISION is not None and CAPTION_MODE != "off" else None
        self.text = TEXT

    def load(self, loader):
        # Runs once when mitmdump loads this script. A config error must stop the container with a clear message:
        # mitmproxy would only log an exception here and keep serving without the addon. stderr, not the logger:
        # mitmproxy's log handler is asynchronous and would lose the line on os._exit.
        if CONFIG_ERRORS:
            for e in CONFIG_ERRORS:
                print(f"imgfix: configuration error: {e}", file=sys.stderr, flush=True)
            os._exit(2)
        if TEXT_CA_FILE:  # https TEXT_URL: verified against this bundle instead of the default CAs (never insecure)
            ctx.options.update(ssl_verify_upstream_trusted_ca=TEXT_CA_FILE)
        log.info("imgfix: text=%s ca=%s vision: %s caption=%s/%gs hint=%s%s fetch_remote_images=%s%s passthrough=%s",
                 TEXT.url, TEXT_CA_FILE or "default", VISION.describe(), CAPTION_MODE, CAPTION_TIMEOUT, ASK_HINT,
                 f" ask_url={ASK_URL}" if ASK_HINT == "http" else "", "on" if FETCH_REMOTE_IMAGES else "off",
                 " (private allowed)" if FETCH_REMOTE_IMAGES and FETCH_ALLOW_PRIVATE else "",
                 ",".join(PASSTHROUGH_MODELS) or "-")

    async def done(self):
        if self.captioner is not None:
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
        if req.raw_content:  # any method with a body (the router sends POSTs only; checked anyway)
            try:  # every path: any JSON body is scanned (count-only paths read-only)
                raw = decode_body(req.raw_content, req.headers.get("content-encoding"))
                ct = path.endswith(COUNT_ONLY_SUFFIXES)
                res = await rewrite(raw, self.store, None if ct else self.captioner, count_only=ct, path=path)
                if res.stats.get("not_json") and not non_json_forwardable(req.headers.get("content-type"), raw):
                    raise Reject("The request body is not valid JSON (strict UTF-8 JSON is required), so imgfix cannot "
                                 "check it for images; it was not forwarded.")
            except Reject as e:
                log.warning("imgfix: %s %s rejected: %s", req.method, path, e)
                flow.response = _error(str(e))
                return
            except Exception as e:  # noqa: BLE001 - never forward a body we could not clean
                log.error("imgfix: %s rewrite failed (%s); request rejected", path, type(e).__name__)
                flow.response = _error(f"imgfix could not rewrite the image/document parts in this request "
                                       f"({type(e).__name__}); it was not forwarded.")
                return
            body = res.body if res.body is not None else (raw if raw is not req.raw_content else None)
            if body is not None:  # rewritten, or decoded: the upstream gets exactly the bytes that were checked
                for h in ("transfer-encoding", "content-encoding"):
                    if h in req.headers:
                        del req.headers[h]
                req.raw_content = body
                req.headers["content-length"] = str(len(body))
            info.update(stats=res.stats, ids=res.ids, rewrite_ms=int((time.monotonic() - t0) * 1000))
        self._route(req)

    def _route(self, req):
        """To the text model: TEXT_URL's scheme/host/port, its path prefix + the client's path and query, and its
        authority as Host (hosted APIs route on it; the TLS SNI is the host). Other headers (the client's own API
        key) pass through untouched."""
        t = self.text
        req.scheme, req.host, req.port = t.scheme, t.host, t.port
        if t.prefix and req.path.startswith("/"):
            req.path = t.prefix + req.path
        req.headers["host"] = t.authority

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
