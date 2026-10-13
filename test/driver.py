"""Stack test driver (runs inside the test network; see run.sh). Uses Pillow from the vision-mcp image.

phase1: (a) text-only request reaches the text model byte-identical, directly from HAProxy
        (b) request with a tool_result image -> the text model gets the exact placeholder and no image/document
            block; one caption call to the vision model (Bearer key, VISION_EXTRA_BODY merged)
        (c) same request again -> byte-identical text-model body, no new caption call
        (d) request naming the vision model as its model: not special (no direct route) -> image replaced, sent to
            the text model via imgfix, nothing but a caption call reaches the vision model
        (e) MCP ask_image on the stored image -> the vision model receives that image (downscaled PNG) and the answer
            comes back
        (h) ~34 MiB body, image after 33 MiB of text (past HAProxy's 32 MiB view) -> still via imgfix, image replaced
        (i) image request on another path (/v1/messages/) -> rewritten too
phase2: imgfix stopped -> image request gets a 400 "imgfix offline" and nothing reaches the text model (fail
        closed); a text-only request still goes straight to the text model
phase3: imgfix started again -> routed through it again; (b) body still byte-identical (BODY_SHA), no caption call
"""
import base64
import hashlib
import io
import json
import os
import socket
import sys
import time
import urllib.error
import urllib.request

from PIL import Image, ImageDraw

ROUTER, STUB, MCP = "http://router:8100", "http://stub:8000", "http://vision-mcp:8102/mcp"
VISION_MODEL = os.environ["VISION_MODEL"]
API_KEY = os.environ["VISION_API_KEY"]
EXTRA = json.loads(os.environ["VISION_EXTRA_BODY"])
CAPTION = "A red rectangle on a white background."
ANSWER = "The rectangle is red. [stub answer]"
HDRS = {"content-type": "application/json", "anthropic-version": "2023-06-01", "x-api-key": "test"}
FAILS = []


def check(cond, label):
    print(("PASS " if cond else "FAIL ") + label, flush=True)
    if not cond:
        FAILS.append(label)


def http(method, url, body=None, headers=None, timeout=30):
    req = urllib.request.Request(url, data=body, method=method, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, dict(r.headers.items()), r.read()
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers.items()), e.read()


def reset():
    http("POST", STUB + "/_reset", b"{}")


def records(upstream=None, bodies=True):
    recs = json.loads(http("GET", STUB + ("/_records" if bodies else "/_records?bodies=0"))[2])
    for r in recs:
        r["headers"] = {k.lower(): v for k, v in r["headers"].items()}
        if bodies:
            r["body"] = base64.b64decode(r.pop("body_b64"))
    return [r for r in recs if upstream is None or r["upstream"] == upstream]


def post(body, path="/v1/messages"):
    return http("POST", ROUTER + path, body, HDRS, timeout=120)


def png(w, h, color, rect=None):
    im = Image.new("RGB", (w, h), "white")
    ImageDraw.Draw(im).rectangle(rect or (w // 4, h // 4, 3 * w // 4, 3 * h // 4), fill=color)
    buf = io.BytesIO()
    im.save(buf, "PNG")
    return buf.getvalue()


def image_request(raw, model="text-model"):
    """Claude Code-shaped history: the Read tool returned a screenshot inside a tool_result."""
    b64 = base64.b64encode(raw).decode()
    obj = {"model": model, "max_tokens": 64, "system": [{"type": "text", "text": "You are a coding agent."}],
           "messages": [
               {"role": "user", "content": [{"type": "text", "text": "Look at shot.png"}]},
               {"role": "assistant", "content": [{"type": "tool_use", "id": "toolu_01", "name": "Read",
                                                  "input": {"file_path": "/w/shot.png"}}]},
               {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_01", "content": [
                   {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64}}]}]}]}
    return obj, json.dumps(obj, separators=(",", ":")).encode()


def placeholder(iid, w, h, caption):
    return (f'[image {iid} | {w}x{h} png | {caption}] The model cannot see images. To inspect this one, call the '
            f'ask_image tool with image_id="{iid}" and a specific question.')


def media_blocks(node):
    if isinstance(node, dict):
        n = int(node.get("type") in ("image", "document"))
        return n + sum(media_blocks(v) for k, v in node.items() if k != "input")
    if isinstance(node, list):
        return sum(media_blocks(v) for v in node)
    return 0


def data_url_image(req_body):
    msg = json.loads(req_body)["messages"]
    parts = [p for m in msg if isinstance(m.get("content"), list) for p in m["content"] if p.get("type") == "image_url"]
    url = parts[0]["image_url"]["url"]
    assert url.startswith("data:image/png;base64,")
    return Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1]))).convert("RGB"), json.loads(req_body)


def extra_merged(req):
    return all(req.get(k) == v for k, v in EXTRA.items())


def wait_imgfix_routed(timeout=60):
    """Probe until HAProxy sends image requests to imgfix (its health check must see it up first)."""
    probe_obj, probe = image_request(png(16, 16, "blue"))
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
    if "text/event-stream" in h.get("content-type", h.get("Content-Type", "")):
        body = [ln[5:].strip() for ln in body.decode().splitlines() if ln.startswith("data:")][-1].encode()
    return st, json.loads(body)


def phase1():
    router_ip, imgfix_ip = socket.gethostbyname("router"), socket.gethostbyname("imgfix")
    check(http("GET", ROUTER + "/router-health")[0] == 200, "router health")
    check(wait_imgfix_routed(), "HAProxy routes image requests to imgfix (probe)")

    # (a) text-only, odd formatting + an escaped '"type": "image"' inside a string: must not be touched or rerouted
    a = ('{ "model" : "text-model",  "max_tokens": 16, "stream": false,\n "messages": [ {"role": "user", "content": '
         '"caf\\u00e9 \\"type\\": \\"image\\" is just text"} ] }').encode()
    st, _, _ = post(a)
    text, vision = records("text"), records("vision")
    check(st == 200 and len(text) == 1 and not vision, "(a) text-only: one text-model request, none to the vision model")
    check(text and text[0]["body"] == a, "(a) text-only: text-model body byte-identical")
    check(text and text[0]["peer"] == router_ip, "(a) text-only: sent directly by HAProxy (not via imgfix)")
    check(text and text[0]["path"] == "/v1/messages", "(a) path unchanged")

    # (b) tool_result image, first sight
    raw = png(2000, 1000, "red")
    iid = "img_" + hashlib.sha256(raw).hexdigest()[:16]
    obj, b = image_request(raw)
    reset()
    st, _, resp = post(b)
    text, vision = records("text"), records("vision")
    check(st == 200 and json.loads(resp).get("id") == "msg_stub_text", "(b) client got the text model's response")
    check(len(text) == 1 and text[0]["peer"] == imgfix_ip, "(b) text-model request came via imgfix")
    body_b = text[0]["body"] if text else b""
    got = json.loads(body_b) if body_b else {}
    want = json.loads(b)
    want["messages"][2]["content"][0]["content"][0] = {"type": "text", "text": placeholder(iid, 2000, 1000, CAPTION)}
    check(media_blocks(got) == 0 and b'"image"' not in body_b and base64.b64encode(raw)[:64] not in body_b,
          "(b) text-model body has no image/document block and no image bytes")
    check(got == want, "(b) text-model body == original with the image block replaced by the exact placeholder")
    print("     placeholder:", placeholder(iid, 2000, 1000, CAPTION))
    check(int(text[0]["headers"].get("content-length", -1)) == len(body_b) if text else False,
          "(b) Content-Length matches the rewritten body")
    caps = [r for r in vision if r["path"] == "/v1/chat/completions"]
    check(len(vision) == 1 and len(caps) == 1, "(b) exactly one caption call to the vision model")
    if caps:
        im, creq = data_url_image(caps[0]["body"])
        check(creq["model"] == VISION_MODEL and creq["temperature"] == 0 and creq["max_tokens"] == 60
              and extra_merged(creq), "(b) caption request parameters (model, sampling, VISION_EXTRA_BODY merged)")
        check(caps[0]["headers"].get("authorization") == f"Bearer {API_KEY}", "(b) caption request carries the Bearer key")
        check(im.size == (1600, 800) and im.getpixel((800, 400))[0] > 200 and im.getpixel((800, 400))[1] < 50,
              f"(b) caption image downscaled to 1600x800 and is the red rectangle (got {im.size})")

    # (c) same request again (Claude Code re-sends history)
    reset()
    st = post(b)[0]
    text, vision = records("text"), records("vision")
    check(st == 200 and len(text) == 1 and text[0]["body"] == body_b, "(c) re-sent request: text-model body byte-identical")
    check(not vision, "(c) no new caption call")

    # (d) a request naming the vision model: no special route, it is cleaned and goes to the text model like any other
    raw_d = png(300, 200, "green")
    _, d = image_request(raw_d, model=VISION_MODEL)
    reset()
    st = post(d)[0]
    text, vision = records("text"), records("vision")
    check(st == 200 and len(text) == 1 and text[0]["peer"] == imgfix_ip and media_blocks(json.loads(text[0]["body"])) == 0
          and json.loads(text[0]["body"])["model"] == VISION_MODEL,
          "(d) vision-model name: image replaced, sent to the text model via imgfix")
    check([r["path"] for r in vision] == ["/v1/chat/completions"], "(d) vision model got only its caption call")

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
    vision = records("vision")
    check(len(vision) == 1 and vision[0]["path"] == "/v1/chat/completions", "(e) one chat/completions call to the vision model")
    if vision:
        im, areq = data_url_image(vision[0]["body"])
        user_text = "".join(p.get("text", "") for p in areq["messages"][-1]["content"] if p.get("type") == "text")
        check(areq["model"] == VISION_MODEL and extra_merged(areq) and iid in user_text
              and "What color is the rectangle?" in user_text, "(e) ask request: model/extra body/id/question")
        check(vision[0]["headers"].get("authorization") == f"Bearer {API_KEY}", "(e) ask request carries the Bearer key")
        px, corner = im.getpixel((800, 400)), im.getpixel((5, 5))
        check(im.size == (1600, 800) and px[0] > 200 and px[1] < 50 and min(corner) > 240,
              f"(e) the vision model received the stored image (size {im.size}, center {px}, corner {corner})")

    # (h) body bigger than HAProxy's 32 MiB view, the only image after 33 MiB of text: imgfix must still get it
    raw_h = png(48, 48, "orange")
    iid_h = "img_" + hashlib.sha256(raw_h).hexdigest()[:16]
    obj_h, _ = image_request(raw_h)
    obj_h["messages"][0]["content"][0]["text"] = "x" * (33 << 20)
    big = json.dumps(obj_h, separators=(",", ":")).encode()
    obj_h["messages"][2]["content"][0]["content"][0] = {"type": "text", "text": placeholder(iid_h, 48, 48, CAPTION)}
    want_h = json.dumps(obj_h, ensure_ascii=True, separators=(",", ":")).encode()
    del obj_h
    reset()
    st = post(big)[0]
    text = records("text", bodies=False)
    reset()
    check(st == 200 and len(text) == 1 and text[0]["peer"] == imgfix_ip,
          f"(h) {len(big) >> 20} MiB body, image past 32 MiB: routed via imgfix (status {st})")
    check(text and text[0]["sha256"] == hashlib.sha256(want_h).hexdigest() and text[0]["len"] == len(want_h),
          "(h) text-model body == original with the late image replaced by its placeholder")

    # (i) another path: imgfix scans every JSON body with a messages list
    _, b_i = image_request(png(24, 24, "blue"))
    reset()
    post(b_i, "/v1/messages/")
    text = records("text")
    check(len(text) == 1 and text[0]["path"] == "/v1/messages/" and text[0]["peer"] == imgfix_ip
          and media_blocks(json.loads(text[0]["body"])) == 0, "(i) /v1/messages/ path: image replaced too")
    print(f"BODY_SHA={hashlib.sha256(body_b).hexdigest()}")


def phase2():
    router_ip = socket.gethostbyname("router")
    raw = png(64, 32, "purple")
    _, b = image_request(raw)
    reset()
    end, seen = time.time() + 45, []
    while time.time() < end:  # until HAProxy marks imgfix DOWN (inter 5s, fall 2): 503s (retryable), never the text model
        st, _, resp = post(b)
        seen.append(st)
        if st == 400:
            break
        time.sleep(1)
    err = json.loads(resp) if seen and seen[-1] == 400 else {}
    msg = err.get("error", {}).get("message", "")
    check(seen and seen[-1] == 400 and err.get("error", {}).get("type") == "invalid_request_error"
          and "imgfix" in msg and "docker compose up -d imgfix" in msg,
          f"(f) imgfix down: image request -> 400 'imgfix offline' (statuses {seen})")
    check(not records("text"), "(f) imgfix down: nothing reached the text model")
    a = b'{"model":"text-model","max_tokens":8,"messages":[{"role":"user","content":"still text-only"}]}'
    st = post(a)[0]
    text = records("text")
    check(st == 200 and len(text) == 1 and text[0]["body"] == a and text[0]["peer"] == router_ip,
          "(f) imgfix down: text-only request still goes straight to the text model")


def phase3():
    check(wait_imgfix_routed(), "(g) imgfix back: HAProxy routes through it again (DNS re-resolved)")
    raw = png(2000, 1000, "red")
    _, b = image_request(raw)
    reset()
    st = post(b)[0]
    text, vision = records("text"), records("vision")
    sha = hashlib.sha256(text[0]["body"]).hexdigest() if text else ""
    check(st == 200 and sha == os.environ.get("BODY_SHA"), "(g) after imgfix restart: text-model body byte-identical to (b)")
    check(not vision, "(g) no caption call after restart (meta on the volume)")


if __name__ == "__main__":
    {"phase1": phase1, "phase2": phase2, "phase3": phase3}[sys.argv[1]]()
    print("RESULT:", "OK" if not FAILS else f"{len(FAILS)} FAILED: {FAILS}", flush=True)
    sys.exit(1 if FAILS else 0)
