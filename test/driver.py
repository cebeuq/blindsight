"""Stack test driver (runs inside the test network; see run.sh). Uses Pillow from the vision-mcp image.

The configuration under test comes from the environment (compose.test.yaml passes run.sh's values), so the same
checks follow each phase's TEXT_URL (prefix, Host, TLS), VISION_API format, CAPTION_MODE and ASK_HINT/ASK_URL.

phase1 (config A: https TEXT_URL with a path prefix + test CA, VISION_API=openai with extra headers and
        max_completion_tokens, short captions, mcp hint, PASSTHROUGH_MODELS, no remote fetch)
        (a) text-only request reaches the text model byte-identical, directly from HAProxy, at TEXT_URL's prefix with
            TEXT_URL's Host over TLS; client headers untouched; a GET keeps its query
        (b) request with a tool_result image -> the text model gets the exact placeholder and no media part; one
            caption call to the vision model (key, extra headers, VISION_EXTRA_BODY merged, token parameter)
        (c) same request again -> byte-identical text-model body, no new caption call
        (d) request naming the vision model as its model: not special -> image replaced, via imgfix
        (e) MCP ask_image on the stored image -> the vision model receives that image and the answer comes back
        (h) ~34 MiB body, image after 33 MiB of text (past HAProxy's 32 MiB view) -> still via imgfix, image replaced
        (i) image request on another path (/v1/messages/) -> rewritten too
        (j) OpenAI Chat: image_url (object and bare string), file (PDF) and a remote URL part, each replaced
        (k) OpenAI Responses: input_image, file_id-only input_image, input_file (PDF); function_call untouched
        (l) Ollama /api/chat: message "images" -> appended to content    (m) /api/generate: top-level "images"
        (n) PASSTHROUGH_MODELS match -> forwarded unchanged (images intact), no caption
        (o) count-only paths (/count_tokens, /input_tokens): rewritten with "no preview", nothing stored, no caption
        (y) Gemini: inline_data image + inlineData PDF replaced, streamGenerateContent SSE passes through,
            countTokens read-only
        (q) bodies only the broad router rules catch (file / input_file / computer_screenshot / escaped / case variant /
            gzip / UTF-16 / images [null, ...] / untyped image_url): never straight to the text model, media gone
        (w) chunked bodies, WebSocket upgrade refused (400), a ?key= query passes through (run.sh: never logged)
        (p) vision-mcp GET/POST /ask (query, JSON, form; PDF page), 400 / 421 / 403 cases
badtls: routers with a certificate they must refuse (wrong name, untrusted CA, system CAs) -> 503, nothing forwarded
phase2: imgfix stopped -> media requests (Anthropic, OpenAI, Ollama, Gemini) get the 400 error JSON and nothing
        reaches the text model (fail closed); a text-only request still goes straight to the text model
phase3: imgfix started again -> routed through it again; (b) body still byte-identical (BODY_SHA), no caption call
phase4 (config B: plain-http TEXT_URL without prefix, VISION_API=anthropic, VISION_EXTRA_BODY deleting temperature,
        detailed captions, http hint with ASK_URL, remote fetch on)
        (r) text-only direct  (s) new image: Anthropic-format detailed caption + http hint  (t) the (b) image keeps
        its stored short caption, hint follows the config  (u) remote image URL fetched once, recorded, byte-stable
        (v) GET /ask in Anthropic format  (x) PDF with the http page hint
"""
import base64
import copy
import gzip
import hashlib
from http.client import HTTPConnection
import io
import json
import os
import socket
import sys
import time
import urllib.error
import urllib.request
from urllib.parse import urlencode, urlsplit

from PIL import Image, ImageDraw

ROUTER, STUB = "http://router:8100", "http://stub:8000"
MCP, ASK = "http://vision-mcp:8102/mcp", "http://vision-mcp:8102/ask"
ENV = os.environ
VISION_MODEL, API_KEY = ENV["VISION_MODEL"], ENV["VISION_API_KEY"]
CLIENT_KEY = ENV["CLIENT_API_KEY"]        # the client's own key (headers, a ?key= query): must never be logged
EXTRA = json.loads(ENV["VISION_EXTRA_BODY"])
VISION_HEADERS = {k.lower(): v for k, v in json.loads(ENV.get("VISION_HEADERS") or "{}").items()}
VISION_API = ENV["VISION_API"]
TOKEN_PARAM = (ENV.get("VISION_TOKEN_PARAM") or "max_tokens") if VISION_API == "openai" else "max_tokens"
VISION_PATH = "/v1/chat/completions" if VISION_API == "openai" else "/v1/messages"
CAPTION_MODE, ASK_HINT, ASK_URL = ENV["CAPTION_MODE"], ENV["ASK_HINT"], ENV["ASK_URL"]
_T = urlsplit(ENV["TEXT_URL"])
TEXT_PREFIX, TEXT_TLS = _T.path.rstrip("/"), _T.scheme == "https"
TEXT_HOST = _T.hostname + ("" if (_T.port or (443 if TEXT_TLS else 80)) == (443 if TEXT_TLS else 80) else f":{_T.port}")

CAPTION = "A red rectangle on a white background."                        # stub.py: short caption (60 tokens)
DETAILED = " ".join(("A large red rectangle fills the centre of a plain white canvas.\n\nIt has sharp corners, no border "
                     "and no shadow.  There is no visible text, logo, person or other object anywhere in the image, "
                     "and the background is a uniform bright white on every side of the shape.").split())
ANSWER = "The rectangle is red. [stub answer]"
GEMINI_SSE = (b'data: {"candidates": [{"content": {"role": "model", "parts": [{"text": "o"}]}}]}\r\n\r\n'   # stub.py
              b'data: {"candidates": [{"content": {"role": "model", "parts": [{"text": "k"}]}, "finishReason": "STOP"}]}'
              b'\r\n\r\n')
NEW_CAPTION = {"short": CAPTION, "detailed": DETAILED}[CAPTION_MODE]   # what a first-seen image gets now
CAPTION_TOKENS = {"short": 60, "detailed": 600}[CAPTION_MODE]
HDRS = {"content-type": "application/json", "anthropic-version": "2023-06-01", "x-api-key": CLIENT_KEY}
OAI_HDRS = {"content-type": "application/json", "authorization": f"Bearer {CLIENT_KEY}"}
GEM_HDRS = {"content-type": "application/json", "x-goog-api-key": CLIENT_KEY}
MEDIA_TYPES = ("image", "document", "image_url", "input_image", "input_file", "file")
OPAQUE = ("tool_use", "server_tool_use", "mcp_tool_use", "function_call")
IPS, FAILS = {}, []


def check(cond, label):
    print(("PASS " if cond else "FAIL ") + label, flush=True)
    if not cond:
        FAILS.append(label)


def http(method, url, body=None, headers=None, timeout=30):
    req = urllib.request.Request(url, data=body, method=method, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, {k.lower(): v for k, v in r.headers.items()}, r.read()
    except urllib.error.HTTPError as e:
        return e.code, {k.lower(): v for k, v in e.headers.items()}, e.read()


def reset():
    http("POST", STUB + "/_reset", b"{}")


def records(upstream=None, bodies=True, method="POST"):
    recs = json.loads(http("GET", STUB + ("/_records" if bodies else "/_records?bodies=0"))[2])
    for r in recs:
        r["headers"] = {k.lower(): v for k, v in r["headers"].items()}
        if bodies:
            r["body"] = base64.b64decode(r.pop("body_b64"))
    return [r for r in recs if (upstream is None or r["upstream"] == upstream) and (method is None or r["method"] == method)]


def post(body, path="/v1/messages", headers=HDRS):
    return http("POST", ROUTER + path, body, headers, timeout=120)


def where(rec):
    return f"[{rec['method']} {rec['path']} Host={rec['headers'].get('host')} tls={rec['tls']} peer={rec['peer']}]"


def text_ok(rec, path, via):
    """A text-model request at TEXT_URL's prefix + path, with TEXT_URL's Host and scheme, sent by router or imgfix."""
    return (rec["path"] == TEXT_PREFIX + path and rec["headers"].get("host") == TEXT_HOST and rec["tls"] == TEXT_TLS
            and rec["peer"] == IPS[via])


def sha16(b):
    return hashlib.sha256(b).hexdigest()[:16]


def png(w, h, color, rect=None):
    im = Image.new("RGB", (w, h), "white")
    ImageDraw.Draw(im).rectangle(rect or (w // 4, h // 4, 3 * w // 4, 3 * h // 4), fill=color)
    buf = io.BytesIO()
    im.save(buf, "PNG")
    return buf.getvalue()


def pdf_doc(text):
    """A one-page PDF whose text layer is `text` (built by hand: no PDF library in the image)."""
    stream = f"BT /F1 18 Tf 20 100 Td ({text}) Tj ET".encode()
    objs = [b"<< /Type /Catalog /Pages 2 0 R >>", b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 400 200] /Contents 4 0 R "
            b"/Resources << /Font << /F1 5 0 R >> >> >>",
            b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream",
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
    out, offsets = b"%PDF-1.4\n", []
    for i, o in enumerate(objs, 1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % i + o + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objs) + 1) + b"".join(b"%010d 00000 n \n" % o for o in offsets)
    return out + b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objs) + 1, xref)


def b64(raw):
    return base64.b64encode(raw).decode()


def durl(raw, mime="image/png"):
    return f"data:{mime};base64,{b64(raw)}"


def image_request(raw, model="text-model"):
    """Claude Code-shaped history: the Read tool returned a screenshot inside a tool_result."""
    obj = {"model": model, "max_tokens": 64, "system": [{"type": "text", "text": "You are a coding agent."}],
           "messages": [
               {"role": "user", "content": [{"type": "text", "text": "Look at shot.png"}]},
               {"role": "assistant", "content": [{"type": "tool_use", "id": "toolu_01", "name": "Read",
                                                  "input": {"file_path": "/w/shot.png"}}]},
               {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_01", "content": [
                   {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64(raw)}}]}]}]}
    return obj, json.dumps(obj, separators=(",", ":")).encode()


def with_text(obj, text):
    """image_request() object with its image block replaced by a text block."""
    obj = copy.deepcopy(obj)
    obj["messages"][2]["content"][0]["content"][0] = {"type": "text", "text": text}
    return obj


def hint(iid):
    return {"mcp": f'The model cannot see images. To inspect this one, call the ask_image tool with image_id="{iid}" '
                   f'and a specific question.',
            "http": f'The model cannot see images. To inspect this one, run: curl -s -G {ASK_URL} --data-urlencode '
                    f'image_id={iid} --data-urlencode "question=<your question>"',
            "none": "The model cannot see images."}[ASK_HINT]


def placeholder(iid, w, h, caption, fmt="png"):
    return f"[image {iid} | {w}x{h} {fmt} | {caption}] {hint(iid)}"


def pdf_head(pid, pages=1):
    page_hint = {"mcp": f' (page images: ask_image with image_id="{pid}" and page=<n>)',
                 "http": f' (page images: curl -s -G {ASK_URL} --data-urlencode image_id={pid} --data-urlencode '
                         f'page=<n> --data-urlencode "question=...")',
                 "none": ""}[ASK_HINT]
    return f"[pdf {pid} | {pages} pages] Text extracted with pdftotext{page_hint}:\n"


def url_ref(url):
    return f"[image url_{sha16(url.encode())} | remote image not fetched: {url}]"


def media_blocks(node):
    """Media parts of any client format left in a body (model-written tool inputs / call arguments excepted)."""
    if isinstance(node, dict):
        n = int(node.get("type") in MEDIA_TYPES) + int(isinstance(node.get("images"), list) and bool(node["images"]))
        skip = ("input", "arguments") if node.get("type") in OPAQUE else ()
        return n + sum(media_blocks(v) for k, v in node.items() if k not in skip)
    if isinstance(node, list):
        return sum(media_blocks(v) for v in node)
    return 0


def vision_image(req):
    """The PNG a vision request carries (either VISION_API format) as a PIL image."""
    parts = [p for m in req["messages"] if isinstance(m.get("content"), list) for p in m["content"]]
    if VISION_API == "anthropic":
        src = [p for p in parts if p.get("type") == "image"][0]["source"]
        assert src["type"] == "base64" and src["media_type"] == "image/png"
        data = src["data"]
    else:
        url = [p for p in parts if p.get("type") == "image_url"][0]["image_url"]["url"]
        assert url.startswith("data:image/png;base64,")
        data = url.split(",", 1)[1]
    return Image.open(io.BytesIO(base64.b64decode(data))).convert("RGB")


def vision_text(req):
    return "".join(p.get("text", "") for p in req["messages"][-1]["content"] if p.get("type") == "text")


def vision_headers_ok(rec):
    h = rec["headers"]
    if VISION_API == "anthropic":
        auth = h.get("x-api-key") == API_KEY and h.get("anthropic-version") == "2023-06-01" and "authorization" not in h
    else:
        auth = h.get("authorization") == f"Bearer {API_KEY}" and "x-api-key" not in h
    return auth and all(h.get(k) == v for k, v in VISION_HEADERS.items())


def vision_params_ok(req, tokens, temperature):
    """model, token limit under the configured parameter name, sampling and VISION_EXTRA_BODY (null deletes)."""
    other = "max_completion_tokens" if TOKEN_PARAM == "max_tokens" else "max_tokens"
    temp_ok = ("temperature" not in req) if EXTRA.get("temperature", 0) is None else req.get("temperature") == temperature
    extra_ok = all((k not in req) if v is None else req.get(k) == v for k, v in EXTRA.items())
    return (req.get("model") == VISION_MODEL and req.get(TOKEN_PARAM) == tokens and other not in req and temp_ok
            and extra_ok)


def has_system(req):
    if VISION_API == "anthropic":
        return bool(req.get("system"))
    return req["messages"][0].get("role") == "system"


def check_caption_call(rec, label, size=None):
    """One caption request: path, auth/extra headers, parameters; optionally the (downscaled) image size."""
    req = json.loads(rec["body"])
    check(rec["path"] == VISION_PATH and vision_headers_ok(rec),
          f"{label} caption request: {VISION_API} format at {rec['path']}, key + VISION_HEADERS")
    check(vision_params_ok(req, CAPTION_TOKENS, 0),
          f"{label} caption request parameters ({TOKEN_PARAM}={CAPTION_TOKENS}, sampling, VISION_EXTRA_BODY)")
    if size:
        im = vision_image(req)
        check(im.size == size, f"{label} caption image size {im.size} == {size}")
    return req


def wait_imgfix_routed(timeout=60):
    """Probe until HAProxy sends image requests to imgfix (its health check must see it up first)."""
    _, probe = image_request(png(16, 16, "blue"))
    end = time.time() + timeout
    while time.time() < end:
        reset()
        st = post(probe)[0]
        text = records("text")
        if st == 200 and text and media_blocks(json.loads(text[-1]["body"])) == 0:
            reset()
            return True
        time.sleep(1)
    return False


def mcp_call(method, params, rid, extra=None):
    hdrs = {"content-type": "application/json", "accept": "application/json, text/event-stream", **(extra or {})}
    msg = {"jsonrpc": "2.0", "method": method, "params": params}
    if rid is not None:
        msg["id"] = rid
    st, h, body = http("POST", MCP, json.dumps(msg).encode(), hdrs, timeout=90)
    if rid is None:
        return st, None
    if "text/event-stream" in h.get("content-type", ""):
        body = [ln[5:].strip() for ln in body.decode().splitlines() if ln.startswith("data:")][-1].encode()
    return st, json.loads(body)


def ips():
    for name in ("router", "imgfix"):
        try:
            IPS[name] = socket.gethostbyname(name)
        except OSError:  # a stopped container has no DNS name (phase2: imgfix)
            IPS[name] = None


def one_text(label, path, via="imgfix"):
    """The single text-model request of the last step, checked for path/Host/TLS/sender."""
    text = records("text")
    ok = len(text) == 1 and text_ok(text[0], path, via)
    check(ok, f"{label} one text-model request at TEXT_URL{path} via {via} {where(text[0]) if text else '(none)'}")
    return json.loads(text[0]["body"]) if text else {}


def phase1():
    ips()
    check(http("GET", ROUTER + "/router-health")[0] == 200, "router health")
    check(wait_imgfix_routed(), "HAProxy routes image requests to imgfix (probe)")

    # (a) text-only, odd formatting + an escaped '"type": "image"' inside a string: must not be touched or rerouted
    a = ('{ "model" : "text-model",  "max_tokens": 16, "stream": false,\n "messages": [ {"role": "user", "content": '
         '"caf\\u00e9 \\"type\\": \\"image\\" is just text"} ] }').encode()
    st, _, _ = post(a)
    text, vision = records("text"), records("vision")
    check(st == 200 and len(text) == 1 and not vision, "(a) text-only: one text-model request, none to the vision model")
    check(text and text[0]["body"] == a, "(a) text-only: text-model body byte-identical")
    check(text and text_ok(text[0], "/v1/messages", "router"),
          f"(a) text-only: sent directly by HAProxy to TEXT_URL (prefix, Host, scheme) {where(text[0]) if text else ''}")
    check(text and text[0]["headers"].get("x-api-key") == CLIENT_KEY and text[0]["headers"].get("anthropic-version")
          == "2023-06-01", "(a) client headers (its own API key) reach the text model untouched")
    reset()
    st = http("GET", ROUTER + "/v1/models?limit=2", headers={"x-api-key": CLIENT_KEY})[0]
    g = records("text", method="GET")
    check(st == 200 and len(g) == 1 and text_ok(g[0], "/v1/models?limit=2", "router"),
          f"(a) GET keeps its query under the TEXT_URL prefix {where(g[0]) if g else ''}")

    # (b) tool_result image, first sight
    raw = png(2000, 1000, "red")
    iid = "img_" + sha16(raw)
    obj, b = image_request(raw)
    reset()
    st, _, resp = post(b)
    vision = records("vision")
    check(st == 200 and json.loads(resp).get("id") == "msg_stub_text", "(b) client got the text model's response")
    got = one_text("(b)", "/v1/messages")
    text = records("text")
    body_b = text[0]["body"] if text else b""
    check(media_blocks(got) == 0 and b'"image"' not in body_b and b64(raw)[:64].encode() not in body_b,
          "(b) text-model body has no image/document block and no image bytes")
    check(got == with_text(obj, placeholder(iid, 2000, 1000, CAPTION)),
          "(b) text-model body == original with the image block replaced by the exact placeholder")
    print("     placeholder:", placeholder(iid, 2000, 1000, CAPTION))
    check(int(text[0]["headers"].get("content-length", -1)) == len(body_b) if text else False,
          "(b) Content-Length matches the rewritten body")
    check(len(vision) == 1, f"(b) exactly one caption call to the vision model (got {len(vision)})")
    if vision:
        creq = check_caption_call(vision[0], "(b)", (1600, 800))
        im = vision_image(creq)
        check(im.getpixel((800, 400))[0] > 200 and im.getpixel((800, 400))[1] < 50, "(b) caption image is the red rectangle")

    # (c) same request again (agents re-send the history)
    reset()
    st = post(b)[0]
    text, vision = records("text"), records("vision")
    check(st == 200 and len(text) == 1 and text[0]["body"] == body_b, "(c) re-sent request: text-model body byte-identical")
    check(not vision, "(c) no new caption call")

    # (d) a request naming the vision model: no special route, it is cleaned and goes to the text model like any other
    _, d = image_request(png(300, 200, "green"), model=VISION_MODEL)
    reset()
    st = post(d)[0]
    got = one_text("(d)", "/v1/messages")
    check(st == 200 and media_blocks(got) == 0 and got.get("model") == VISION_MODEL,
          "(d) vision-model name: image replaced, sent to the text model via imgfix")
    check([r["path"] for r in records("vision")] == [VISION_PATH], "(d) vision model got only its caption call")

    # (e) MCP ask_image on the image stored in (b)
    reset()
    st, init = mcp_call("initialize", {"protocolVersion": "2025-06-18", "capabilities": {},
                                       "clientInfo": {"name": "stack-test", "version": "1"}}, 1)
    check(st == 200 and "result" in init, f"(e) MCP initialize (server {init.get('result', {}).get('serverInfo')})")
    pv = {"mcp-protocol-version": init.get("result", {}).get("protocolVersion", "2025-06-18")}
    mcp_call("notifications/initialized", {}, None, pv)
    st, tl = mcp_call("tools/list", {}, 2, pv)
    check(st == 200 and [t["name"] for t in tl["result"]["tools"]] == ["ask_image"], "(e) tools/list -> ask_image")
    st, res = mcp_call("tools/call", {"name": "ask_image", "arguments": {
        "image_id": iid, "question": "What color is the rectangle?"}}, 3, pv)
    answer = "".join(c.get("text", "") for c in res.get("result", {}).get("content", []))
    check(st == 200 and answer == ANSWER and not res["result"].get("isError"), f"(e) ask_image answer: {answer!r}")
    check_ask_call("(e)", iid, "What color is the rectangle?", (1600, 800))

    # (h) body bigger than HAProxy's 32 MiB view, the only image after 33 MiB of text: imgfix must still get it
    raw_h = png(48, 48, "orange")
    obj_h, _ = image_request(raw_h)
    obj_h["messages"][0]["content"][0]["text"] = "x" * (33 << 20)
    big = json.dumps(obj_h, separators=(",", ":")).encode()
    want_h = json.dumps(with_text(obj_h, placeholder("img_" + sha16(raw_h), 48, 48, CAPTION)), ensure_ascii=True,
                        separators=(",", ":")).encode()
    del obj_h
    reset()
    st = post(big)[0]
    text = records("text", bodies=False)
    reset()
    check(st == 200 and len(text) == 1 and text_ok(text[0], "/v1/messages", "imgfix"),
          f"(h) {len(big) >> 20} MiB body, image past 32 MiB: routed via imgfix (status {st})")
    check(text and text[0]["sha256"] == hashlib.sha256(want_h).hexdigest() and text[0]["len"] == len(want_h),
          "(h) text-model body == original with the late image replaced by its placeholder")
    del big, want_h

    # (i) another path: imgfix scans every JSON body
    _, b_i = image_request(png(24, 24, "blue"))
    reset()
    post(b_i, "/v1/messages/")
    got = one_text("(i)", "/v1/messages/")
    check(media_blocks(got) == 0, "(i) /v1/messages/ path: image replaced too")

    pid_j = phase1_formats()
    phase1_passthrough_and_counts()
    phase1_gemini()
    phase1_triggers()
    phase1_transport()
    phase_ask(iid, pid_j)
    print(f"BODY_SHA={hashlib.sha256(body_b).hexdigest()}")


def phase1_formats():
    # (j) OpenAI Chat Completions: image_url (object + bare string), file (PDF data URL), remote URL (not fetched)
    r1, r2, pdf = png(320, 200, "blue"), png(200, 320, "green"), pdf_doc("Hello blindsight chat")
    i1, i2, pid = "img_" + sha16(r1), "img_" + sha16(r2), "pdf_" + sha16(pdf)
    url = "https://example.com/images/cat.png"
    obj = {"model": "text-model", "max_tokens": 32, "messages": [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": [
            {"type": "text", "text": "Compare these."},
            {"type": "image_url", "image_url": {"url": durl(r1), "detail": "high"}},
            {"type": "image_url", "image_url": durl(r2)},
            {"type": "file", "file": {"file_data": durl(pdf, "application/pdf"), "filename": "note.pdf"}},
            {"type": "image_url", "image_url": {"url": url}}]}]}
    reset()
    st, _, resp = post(json.dumps(obj).encode(), "/v1/chat/completions", OAI_HDRS)
    check(st == 200 and json.loads(resp).get("object") == "chat.completion", f"(j) OpenAI Chat client got the reply ({st})")
    got = one_text("(j)", "/v1/chat/completions")
    text = records("text")
    check(text and text[0]["headers"].get("authorization") == f"Bearer {CLIENT_KEY}",
          "(j) the client's own Authorization reaches the text model")
    parts = got.get("messages", [{}, {}])[1].get("content", [])
    pdf_ref = parts[3].get("text", "") if len(parts) == 5 else ""
    check(len(parts) == 5 and parts[3].get("type") == "text" and pdf_ref.startswith(pdf_head(pid))
          and "Hello blindsight chat" in pdf_ref, f"(j) file part (PDF) -> text part with its text: {pdf_ref[:160]!r}")
    want = copy.deepcopy(obj)
    want["messages"][1]["content"][1:] = [{"type": "text", "text": placeholder(i1, 320, 200, NEW_CAPTION)},
                                          {"type": "text", "text": placeholder(i2, 200, 320, NEW_CAPTION)},
                                          {"type": "text", "text": pdf_ref},
                                          {"type": "text", "text": url_ref(url)}]
    check(got == want and media_blocks(got) == 0, "(j) body == original with each media part replaced by its text")
    vision = records("vision")
    check(len(vision) == 2, f"(j) one caption call per new image (2, got {len(vision)})")
    for rec in vision:
        check_caption_call(rec, "(j)")

    # (k) OpenAI Responses: input_image (data URL), input_image file_id only, input_file (PDF); model-written
    # function_call arguments are opaque and stay as they are
    rk, pdf_k = png(256, 128, "purple"), pdf_doc("Hello blindsight responses")
    ik, pid_k = "img_" + sha16(rk), "pdf_" + sha16(pdf_k)
    obj = {"model": "text-model", "input": [
        {"type": "function_call", "call_id": "call_1", "name": "show", "arguments": '{"type":"image","path":"/a.png"}'},
        {"type": "function_call_output", "call_id": "call_1", "output": "done"},
        {"role": "user", "content": [
            {"type": "input_text", "text": "What is this?"},
            {"type": "input_image", "image_url": durl(rk)},
            {"type": "input_image", "file_id": "file-abc123"},
            {"type": "input_file", "filename": "r.pdf", "file_data": durl(pdf_k, "application/pdf")}]}]}
    reset()
    st, _, resp = post(json.dumps(obj).encode(), "/v1/responses", OAI_HDRS)
    check(st == 200 and json.loads(resp).get("object") == "response", f"(k) Responses client got the reply ({st})")
    got = one_text("(k)", "/v1/responses")
    parts = got.get("input", [{}, {}, {}])[2].get("content", [])
    pdf_ref = parts[3].get("text", "") if len(parts) == 4 else ""
    check(len(parts) == 4 and parts[3].get("type") == "input_text" and pdf_ref.startswith(pdf_head(pid_k))
          and "Hello blindsight responses" in pdf_ref, f"(k) input_file (PDF) -> input_text: {pdf_ref[:120]!r}")
    want = copy.deepcopy(obj)
    want["input"][2]["content"][1:] = [{"type": "input_text", "text": placeholder(ik, 256, 128, NEW_CAPTION)},
                                       {"type": "input_text", "text": "[image unavailable: file_id file-abc123 not readable]"},
                                       {"type": "input_text", "text": pdf_ref}]
    check(got == want and media_blocks(got) == 0,
          "(k) body == original with input_image / file_id / input_file replaced, function_call untouched")
    check(len(records("vision")) == 1, "(k) one caption call (the readable image)")

    # (l) Ollama /api/chat: images on a message -> refs appended to its content
    rl = png(64, 48, "orange")
    obj = {"model": "llava-text", "stream": False, "messages": [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "What is in this picture?", "images": [b64(rl)]}]}
    reset()
    st, _, resp = post(json.dumps(obj).encode(), "/api/chat", {"content-type": "application/json"})
    check(st == 200 and json.loads(resp).get("done") is True, f"(l) Ollama chat client got the reply ({st})")
    got = one_text("(l)", "/api/chat")
    want = copy.deepcopy(obj)
    want["messages"][1] = {"role": "user", "content": "What is in this picture?\n\n"
                           + placeholder("img_" + sha16(rl), 64, 48, NEW_CAPTION)}
    check(got == want, "(l) /api/chat: images removed, '\\n\\n' + ref appended to the message content")

    # (m) Ollama /api/generate: top-level images -> refs appended to the prompt
    m1, m2 = png(50, 50, "cyan"), png(60, 40, "magenta")
    obj = {"model": "llava-text", "stream": False, "prompt": "Describe both.", "images": [b64(m1), b64(m2)]}
    reset()
    st, _, resp = post(json.dumps(obj).encode(), "/api/generate", {"content-type": "application/json"})
    check(st == 200 and json.loads(resp).get("done") is True, f"(m) Ollama generate client got the reply ({st})")
    got = one_text("(m)", "/api/generate")
    want = {"model": "llava-text", "stream": False, "prompt": "Describe both.\n\n"
            + placeholder("img_" + sha16(m1), 50, 50, NEW_CAPTION) + "\n"
            + placeholder("img_" + sha16(m2), 60, 40, NEW_CAPTION)}
    check(got == want, "(m) /api/generate: images removed, refs appended to the prompt (one per line)")
    return pid


def phase1_passthrough_and_counts():
    # (n) PASSTHROUGH_MODELS (run.sh: "passthru-*,..."): forwarded unchanged, images intact, nothing captioned
    _, b_n = image_request(png(30, 30, "yellow"), model="passthru-vl")
    reset()
    st = post(b_n)[0]
    text, vision = records("text"), records("vision")
    check(st == 200 and len(text) == 1 and text[0]["body"] == b_n and text_ok(text[0], "/v1/messages", "imgfix")
          and not vision, f"(n) PASSTHROUGH_MODELS match: body unchanged (image intact) via imgfix, no caption "
                          f"{where(text[0]) if text else ''}")

    # (o) count-only paths: rewritten for counting, but read-only (no caption, nothing stored)
    ro = png(70, 35, "navy")
    io_ = "img_" + sha16(ro)
    obj_o, b_o = image_request(ro)
    reset()
    st = post(b_o, "/v1/messages/count_tokens")[0]
    got = one_text("(o)", "/v1/messages/count_tokens")
    check(st == 200 and got == with_text(obj_o, placeholder(io_, 70, 35, "no preview")) and not records("vision"),
          "(o) count_tokens: image shown as 'no preview', no caption call")
    ro2 = png(71, 35, "navy")
    io2 = "img_" + sha16(ro2)
    obj = {"model": "text-model", "input": [{"role": "user", "content": [{"type": "input_image", "image_url": durl(ro2)}]}]}
    reset()
    st = post(json.dumps(obj).encode(), "/v1/responses/input_tokens", OAI_HDRS)[0]
    got = one_text("(o)", "/v1/responses/input_tokens")
    want = copy.deepcopy(obj)
    want["input"][0]["content"][0] = {"type": "input_text", "text": placeholder(io2, 71, 35, "no preview")}
    check(st == 200 and got == want and not records("vision"), "(o) Responses input_tokens: 'no preview', no caption")
    stored = [p for i in (io_, io2) for p in (f"/store/img/{i}.png", f"/store/meta/{i}.json") if os.path.exists(p)]
    check(not stored, f"(o) count-only requests wrote nothing to the store {stored}")


def leaked(body, secrets):
    """Any secret (a media part's base64 prefix) in a body, read as is, gunzipped, or as UTF-16."""
    views = [body]
    if body[:2] == b"\x1f\x8b":
        try:
            views.append(gzip.decompress(body))
        except OSError:
            pass
    if b"\x00" in body:
        for enc in ("utf-16", "utf-16-le", "utf-16-be"):
            try:
                views.append(body.decode(enc).encode())
            except UnicodeDecodeError:
                pass
    return any(sec in v for v in views for sec in secrets)


def not_direct(label, path, body, headers, secrets):
    """A request the router must hand to imgfix: it never goes straight to the text model, and if imgfix forwards
    it, the media bytes are gone. imgfix may also refuse it with its own 4xx (what it cannot rewrite safely)."""
    reset()
    st, _, resp = http("POST", ROUTER + path, body, headers, timeout=120)
    text = records("text")
    direct = [where(r) for r in text if r["peer"] != IPS["imgfix"]]
    leak = [where(r) for r in text if leaked(r["body"], secrets)]
    rewritten = st == 200 and len(text) == 1
    refused = 400 <= st < 500 and not text and b"blindsight_imgfix_unavailable" not in resp
    how = "rewritten by imgfix" if rewritten else f"refused by imgfix ({st} {resp[:100]!r})" if refused else \
        f"status {st}, {len(text)} text-model requests {resp[:100]!r}"
    check(not direct and not leak and (rewritten or refused),
          f"(q) {label}: {how}; never direct {direct}, no media bytes at the text model {leak}")
    return json.loads(text[0]["body"]) if rewritten else None


def phase1_triggers():
    """(q) bodies that only the extended router rules catch (each alone), end to end."""
    def img(color):
        raw = png(36, 24, color)
        return raw, [b64(raw)[:40].encode()]
    pdf = pdf_doc("Hello blindsight trigger")
    pdf_secret = [b64(pdf)[:40].encode()]
    chat = lambda part: json.dumps({"model": "text-model", "messages": [{"role": "user", "content": [
        {"type": "text", "text": "look"}, part]}]}).encode()
    resp = lambda part: json.dumps({"model": "text-model", "input": [part]}).encode()
    not_direct("OpenAI file part (PDF)", "/v1/chat/completions",
               chat({"type": "file", "file": {"file_data": durl(pdf, "application/pdf"), "filename": "t.pdf"}}),
               OAI_HDRS, pdf_secret)
    not_direct("Responses input_file alone", "/v1/responses", resp({"role": "user", "content": [
        {"type": "input_file", "filename": "t.pdf", "file_data": durl(pdf, "application/pdf")}]}), OAI_HDRS, pdf_secret)
    raw, sec = img("teal")
    not_direct("Responses computer_screenshot", "/v1/responses", resp(
        {"type": "computer_call_output", "call_id": "call_9", "output": {"type": "computer_screenshot",
                                                                         "image_url": durl(raw)}}), OAI_HDRS, sec)
    raw, sec = img("olive")
    _, b = image_request(raw)
    not_direct("escaped type value (\\u0069mage)", "/v1/messages",
               b.replace(b'"type":"image"', b'"type":"\\u0069mage"'), HDRS, sec)
    raw, sec = img("maroon")
    not_direct('Ollama "Images" (case variant)', "/api/generate",
               json.dumps({"model": "llava", "prompt": "p", "Images": [b64(raw)]}).encode(), HDRS, sec)
    raw, sec = img("lime")
    not_direct("Ollama images [null, <b64>]", "/api/generate",
               json.dumps({"model": "llava", "prompt": "p", "images": [None, b64(raw)]}).encode(), HDRS, sec)
    raw, sec = img("silver")
    _, b = image_request(raw)
    not_direct("Content-Encoding: gzip body", "/v1/messages", gzip.compress(b), {**HDRS, "content-encoding": "gzip"}, sec)
    raw, sec = img("gold")
    _, b = image_request(raw)
    not_direct("UTF-16 body", "/v1/messages", b.decode().encode("utf-16-le"),
               {**HDRS, "content-type": "application/json; charset=utf-16le"}, sec)
    raw, sec = img("pink")
    not_direct("vLLM untyped image_url part", "/v1/chat/completions", chat({"image_url": {"url": durl(raw)}}),
               OAI_HDRS, sec)


def post_chunked(body, path, headers):
    c = HTTPConnection("router", 8100, timeout=120)
    c.request("POST", path, body=iter([body[:10], body[10:]]), headers=headers, encode_chunked=True)
    r = c.getresponse()
    data = r.read()
    c.close()
    return r.status, data


def phase1_transport():
    """(w) chunked request bodies, WebSocket upgrades, a key in the query string."""
    a = b'{"model":"text-model","max_tokens":8,"messages":[{"role":"user","content":"chunked text"}]}'
    reset()
    st, _ = post_chunked(a, "/v1/messages", HDRS)
    text = records("text")
    check(st == 200 and len(text) == 1 and text[0]["body"] == a and text_ok(text[0], "/v1/messages", "router"),
          f"(w) chunked text-only body: straight to the text model, byte-identical ({st})")
    raw = png(33, 22, "olive")
    obj, b = image_request(raw)
    reset()
    st, _ = post_chunked(b, "/v1/messages", HDRS)
    got = one_text("(w)", "/v1/messages")
    check(st == 200 and got == with_text(obj, placeholder("img_" + sha16(raw), 33, 22, NEW_CAPTION)),
          "(w) chunked image body: rewritten by imgfix")
    reset()
    c = HTTPConnection("router", 8100, timeout=30)
    c.request("GET", "/v1/responses", headers={"connection": "Upgrade", "upgrade": "websocket", **OAI_HDRS,
                                               "sec-websocket-version": "13",
                                               "sec-websocket-key": "dGhlIHNhbXBsZSBub25jZQ=="})
    r = c.getresponse()
    err = json.loads(r.read() or b"{}")
    c.close()
    check(r.status == 400 and err.get("type") == "error" and err.get("error", {}).get("code")
          == "blindsight_upgrade_unsupported" and not records(method=None),
          f"(w) WebSocket upgrade -> 400 JSON error, nothing forwarded ({r.status})")
    reset()
    st = http("GET", f"{ROUTER}/v1/models?key={CLIENT_KEY}")[0]
    g = records("text", method="GET")
    check(st == 200 and g and text_ok(g[0], f"/v1/models?key={CLIENT_KEY}", "router"),
          "(w) a key in the query string reaches the text model (run.sh checks it is never logged)")


def phase1_gemini():
    """(y) Gemini native format: inline_data / inlineData parts, SSE streaming passthrough, read-only countTokens."""
    raw, pdf = png(120, 80, "navy"), pdf_doc("Hello blindsight gemini")
    iid, pid = "img_" + sha16(raw), "pdf_" + sha16(pdf)
    obj = {"contents": [{"role": "user", "parts": [
        {"text": "What is in these?"},
        {"inline_data": {"mime_type": "image/png", "data": b64(raw)}},
        {"inlineData": {"mimeType": "application/pdf", "data": b64(pdf)}}]}],
        "generationConfig": {"maxOutputTokens": 16}}
    path = f"/v1beta/models/text-model:generateContent?key={CLIENT_KEY}"
    reset()
    st, _, resp = post(json.dumps(obj).encode(), path, GEM_HDRS)
    check(st == 200 and "candidates" in json.loads(resp or b"{}"), f"(y) Gemini client got the reply ({st})")
    got = one_text("(y)", path)
    parts = got.get("contents", [{}])[0].get("parts", [])
    pdf_ref = parts[2].get("text", "") if len(parts) == 3 else ""
    check(pdf_ref.startswith(pdf_head(pid)) and "Hello blindsight gemini" in pdf_ref,
          f"(y) inlineData PDF part -> text part with its text: {pdf_ref[:100]!r}")
    want = copy.deepcopy(obj)
    want["contents"][0]["parts"][1:] = [{"text": placeholder(iid, 120, 80, NEW_CAPTION)}, {"text": pdf_ref}]
    check(got == want and not leaked(json.dumps(got).encode(), [b64(raw)[:40].encode()]),
          "(y) generateContent: inline_data image and inlineData PDF replaced by text parts")
    check(len(records("vision")) == 1, "(y) one caption call (the image)")
    reset()
    st, h, resp = post(json.dumps(obj).encode(), "/v1beta/models/text-model:streamGenerateContent?alt=sse", GEM_HDRS)
    got = one_text("(y)", "/v1beta/models/text-model:streamGenerateContent?alt=sse")
    check(st == 200 and h.get("content-type", "").startswith("text/event-stream") and resp == GEMINI_SSE
          and got == want, f"(y) streamGenerateContent?alt=sse: same rewrite, the SSE stream passes through ({st})")
    raw_c = png(121, 80, "navy")
    ic = "img_" + sha16(raw_c)
    cnt = {"contents": [{"role": "user", "parts": [{"inline_data": {"mime_type": "image/png", "data": b64(raw_c)}}]}]}
    reset()
    st = post(json.dumps(cnt).encode(), "/v1beta/models/text-model:countTokens", GEM_HDRS)[0]
    got = one_text("(y)", "/v1beta/models/text-model:countTokens")
    check(st == 200 and got.get("contents", [{}])[0].get("parts") == [{"text": placeholder(ic, 121, 80, "no preview")}]
          and not records("vision") and not os.path.exists(f"/store/meta/{ic}.json")
          and not os.path.exists(f"/store/img/{ic}.png"), "(y) countTokens: 'no preview', no caption, nothing stored")


def check_ask_call(label, iid, question, size=None):
    vision = records("vision")
    check(len(vision) == 1 and vision[0]["path"] == VISION_PATH and vision_headers_ok(vision[0]),
          f"{label} one {VISION_API} call to the vision model (key + VISION_HEADERS)")
    if not vision:
        return
    req = json.loads(vision[0]["body"])
    user = vision_text(req)
    check(vision_params_ok(req, 700, 0.2) and has_system(req) and iid in user and question in user,
          f"{label} ask request: model/{TOKEN_PARAM}=700/sampling/extra body/system prompt/id/question")
    if size:
        im = vision_image(req)
        px, corner = im.getpixel((size[0] // 2, size[1] // 2)), im.getpixel((5, 5))
        check(im.size == size and px[0] > 200 and px[1] < 50 and min(corner) > 240,
              f"{label} the vision model received the stored image (size {im.size}, center {px}, corner {corner})")


def phase_ask(iid, pid):
    """(p) vision-mcp's plain-HTTP /ask (what ASK_HINT=http tells the model to curl)."""
    q = "What color is the rectangle?"
    reset()
    st, h, body = http("GET", ASK + "?" + urlencode({"image_id": iid, "question": q}), timeout=90)
    check(st == 200 and h.get("content-type", "").startswith("text/plain") and body.decode().strip() == ANSWER,
          f"(p) GET /ask -> 200 text/plain answer ({st} {body[:80]!r})")
    check_ask_call("(p) GET /ask:", iid, q, (1600, 800))
    reset()
    st, _, body = http("POST", ASK, json.dumps({"image_id": iid, "question": q + " (json)"}).encode(),
                       {"content-type": "application/json"}, timeout=90)
    check(st == 200 and body.decode().strip() == ANSWER and len(records("vision")) == 1,
          f"(p) POST /ask JSON -> 200 answer ({st} {body[:80]!r})")
    reset()
    st, _, body = http("POST", ASK, urlencode({"image_id": iid, "question": q + " (form)", "region": "0,0,0.5,0.5"}).encode(),
                       {"content-type": "application/x-www-form-urlencoded"}, timeout=90)
    check(st == 200 and body.decode().strip() == ANSWER and len(records("vision")) == 1,
          f"(p) POST /ask form (with region) -> 200 answer ({st} {body[:80]!r})")
    reset()
    st, _, body = http("GET", ASK + "?" + urlencode({"image_id": pid, "page": 1, "question": "What does it say?"}),
                       timeout=90)
    check(st == 200 and body.decode().strip() == ANSWER and len(records("vision")) == 1,
          f"(p) GET /ask on a PDF page -> 200 answer ({st} {body[:80]!r})")
    reset()
    st, h, body = http("GET", ASK + "?" + urlencode({"image_id": "img_0000000000000000", "question": "x"}))
    check(st == 400 and body.decode().startswith("ask_image error:") and not records("vision"),
          f"(p) unknown image_id -> 400 'ask_image error: ...' ({st} {body[:80]!r})")
    st = http("GET", ASK + "?" + urlencode({"image_id": iid, "question": "x"}), headers={"host": "evil.example:8102"})[0]
    check(st == 421, f"(p) Host header not allowed -> 421 ({st})")
    st = http("GET", ASK + "?" + urlencode({"image_id": iid, "question": "x"}), headers={"origin": "http://evil.example"})[0]
    check(st == 403 and not records("vision"), f"(p) foreign Origin -> 403, no vision call ({st})")


def media_request(kind):
    """A request with one new image in each client format (for the fail-closed checks)."""
    raw = png(64, 32, "purple")
    if kind == "openai":
        return "/v1/chat/completions", OAI_HDRS, json.dumps({"model": "text-model", "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": durl(raw)}}]}]}).encode()
    if kind == "ollama":
        return "/api/generate", {"content-type": "application/json"}, json.dumps(
            {"model": "m", "prompt": "hi", "images": [b64(raw)]}).encode()
    if kind == "gemini":
        return "/v1beta/models/m:generateContent", GEM_HDRS, json.dumps({"contents": [{"parts": [
            {"inlineData": {"mimeType": "image/png", "data": b64(raw)}}]}]}).encode()
    return "/v1/messages", HDRS, image_request(raw)[1]


def phase2():
    ips()
    path, hdrs, b = media_request("anthropic")
    reset()
    end, seen = time.time() + 45, []
    while time.time() < end:  # until HAProxy marks imgfix DOWN (inter 5s, fall 2): 503s (retryable), never the text model
        st, _, resp = post(b, path, hdrs)
        seen.append(st)
        if st == 400:
            break
        time.sleep(1)
    err = json.loads(resp) if seen and seen[-1] == 400 else {}
    e = err.get("error", {})
    msg = e.get("message", "")
    check(seen and seen[-1] == 400 and err.get("type") == "error" and e.get("type") == "invalid_request_error"
          and "param" in e and e.get("code") and "imgfix" in msg and "docker compose up -d imgfix" in msg,
          f"(f) imgfix down: image request -> 400 error JSON for Anthropic and OpenAI clients (statuses {seen})")
    for kind in ("openai", "ollama", "gemini"):
        path, hdrs, b = media_request(kind)
        st = post(b, path, hdrs)[0]
        check(st == 400, f"(f) imgfix down: {kind} media request -> 400 ({st})")
    check(not records("text"), "(f) imgfix down: nothing reached the text model")
    a = b'{"model":"text-model","max_tokens":8,"messages":[{"role":"user","content":"still text-only"}]}'
    st = post(a)[0]
    text = records("text")
    check(st == 200 and len(text) == 1 and text[0]["body"] == a and text_ok(text[0], "/v1/messages", "router"),
          "(f) imgfix down: text-only request still goes straight to the text model")


def phase3():
    ips()
    check(wait_imgfix_routed(), "(g) imgfix back: HAProxy routes through it again (DNS re-resolved)")
    _, b = image_request(png(2000, 1000, "red"))
    reset()
    st = post(b)[0]
    text, vision = records("text"), records("vision")
    sha = hashlib.sha256(text[0]["body"]).hexdigest() if text else ""
    check(st == 200 and sha == os.environ.get("BODY_SHA"), "(g) after imgfix restart: text-model body byte-identical to (b)")
    check(not vision, "(g) no caption call after restart (meta on the volume)")


def badtls():
    """Routers started by run.sh with a certificate they must refuse (wrong name, untrusted CA, system CAs): 503 and
    nothing reaches the text model."""
    for name in ENV["BADTLS_ROUTERS"].split():
        url, end = f"http://{name}:8100", time.time() + 30
        while time.time() < end:
            try:
                if http("GET", url + "/router-health", timeout=3)[0] == 200:
                    break
            except OSError:
                pass
            time.sleep(0.5)
        reset()
        a = b'{"model":"text-model","max_tokens":8,"messages":[{"role":"user","content":"tls"}]}'
        st1, st2 = http("POST", url + "/v1/messages", a, HDRS)[0], http("GET", url + "/v1/models", headers=HDRS)[0]
        check(st1 == 503 and st2 == 503 and not records(method=None),
              f"(tls) {name}: certificate refused -> 503, nothing reached the text model ({st1}, {st2})")


def phase4():
    ips()
    check(wait_imgfix_routed(), "(4) HAProxy routes image requests to the recreated imgfix")

    # (r) text-only straight to a plain-http TEXT_URL without prefix
    a = b'{"model":"text-model","max_tokens":8,"messages":[{"role":"user","content":"phase 4 text"}]}'
    reset()
    st = post(a)[0]
    text = records("text")
    check(st == 200 and len(text) == 1 and text[0]["body"] == a and text_ok(text[0], "/v1/messages", "router"),
          f"(r) text-only: directly to TEXT_URL {where(text[0]) if text else ''}")

    # (s) a new image: caption through the configured VISION_API (detailed mode), placeholder with the configured hint
    raw = png(900, 600, "red")
    iid = "img_" + sha16(raw)
    obj, b = image_request(raw)
    reset()
    st = post(b)[0]
    got = one_text("(s)", "/v1/messages")
    check(st == 200 and got == with_text(obj, placeholder(iid, 900, 600, NEW_CAPTION)),
          f"(s) new image: {CAPTION_MODE} caption + {ASK_HINT} hint: {placeholder(iid, 900, 600, NEW_CAPTION)[:90]}...")
    vision = records("vision")
    check(len(vision) == 1, f"(s) one caption call ({len(vision)})")
    if vision:
        check_caption_call(vision[0], "(s)", (900, 600))

    # (t) the image from (b): its stored caption is kept for good, the hint follows the configuration
    obj_b, b_b = image_request(png(2000, 1000, "red"))
    reset()
    post(b_b)
    got = one_text("(t)", "/v1/messages")
    iid_b = "img_" + sha16(png(2000, 1000, "red"))
    check(got == with_text(obj_b, placeholder(iid_b, 2000, 1000, CAPTION)) and not records("vision"),
          "(t) phase-1 image: stored short caption unchanged, new hint, no caption call")

    # (u) remote image URL with FETCH_REMOTE_IMAGES=true (FETCH_ALLOW_PRIVATE=true: the stub is a private address)
    url = "http://stub:8000/img/blue.png"
    raw_u = http("GET", url)[2]
    iu = "img_" + sha16(raw_u)
    obj = {"model": "text-model", "messages": [{"role": "user", "content": [
        {"type": "text", "text": "See"}, {"type": "image_url", "image_url": {"url": url}}]}]}
    body = json.dumps(obj).encode()
    reset()
    st = post(body, "/v1/chat/completions", OAI_HDRS)[0]
    got = one_text("(u)", "/v1/chat/completions")
    fetches = [r for r in records("text", method="GET") if r["path"] == "/img/blue.png"]
    want = copy.deepcopy(obj)
    want["messages"][0]["content"][1] = {"type": "text", "text": placeholder(iu, 40, 30, NEW_CAPTION)}
    check(st == 200 and got == want, "(u) remote URL fetched on first sight and shown like an inline image")
    check(len(fetches) == 1 and fetches[0]["peer"] == IPS["imgfix"], f"(u) exactly one fetch, by imgfix ({len(fetches)})")
    try:
        with open(f"/store/meta/url_{sha16(url.encode())}.json") as f:
            meta = json.load(f)
    except OSError as e:
        meta = {"error": str(e)}
    check(meta.get("url") == url and meta.get("target") == iu and meta.get("kind") == "url",
          f"(u) decision recorded write-once in meta/url_<id>.json: {meta}")
    first = records("text")[0]["body"] if records("text") else b""
    reset()
    post(body, "/v1/chat/completions", OAI_HDRS)
    text = records("text")
    check(text and text[0]["body"] == first and not records("text", method="GET") and not records("vision"),
          "(u) same URL again: byte-identical text, no new fetch, no caption call")

    # (v) /ask in the configured VISION_API format (system prompt, VISION_EXTRA_BODY)
    reset()
    st, _, body = http("GET", ASK + "?" + urlencode({"image_id": iid, "question": "Which colour?"}), timeout=90)
    check(st == 200 and body.decode().strip() == ANSWER, f"(v) GET /ask -> 200 answer ({st} {body[:80]!r})")
    check_ask_call("(v)", iid, "Which colour?", (900, 600))

    # (x) PDF (Anthropic document block): page hint follows ASK_HINT
    pdf = pdf_doc("Hello blindsight phase four")
    pid = "pdf_" + sha16(pdf)
    obj = {"model": "text-model", "max_tokens": 16, "messages": [{"role": "user", "content": [
        {"type": "document", "source": {"type": "base64", "media_type": "application/pdf", "data": b64(pdf)}},
        {"type": "text", "text": "Summarise."}]}]}
    reset()
    st = post(json.dumps(obj).encode())[0]
    got = one_text("(x)", "/v1/messages")
    part = got.get("messages", [{}])[0].get("content", [{}])[0]
    check(st == 200 and part.get("type") == "text" and part.get("text", "").startswith(pdf_head(pid))
          and "Hello blindsight phase four" in part.get("text", "") and media_blocks(got) == 0,
          f"(x) PDF document -> text with the {ASK_HINT} page hint: {part.get('text', '')[:160]!r}")


if __name__ == "__main__":
    {"phase1": phase1, "phase2": phase2, "phase3": phase3, "phase4": phase4, "badtls": badtls}[sys.argv[1]]()
    print("RESULT:", "OK" if not FAILS else f"{len(FAILS)} FAILED: {FAILS}", flush=True)
    sys.exit(1 if FAILS else 0)
