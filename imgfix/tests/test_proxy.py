"""End-to-end: real mitmdump + imgfix.py between a client and local stub text/vision model servers (loopback only)."""
import base64
import gzip
import hashlib
import http.client
import io
import json
import os
import shutil
import socket
import ssl
import subprocess
import tempfile
import threading
import time
import unittest
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from PIL import Image

from . import helpers as h

ADDON = Path(__file__).resolve().parents[1] / "imgfix.py"
VISION_MODEL = "test-vision-model"
API_KEY = "sk-imgfix-test-key-0123456789"
EXTRA_BODY = '{"chat_template_kwargs":{"enable_thinking":false},"top_k":20}'
REMOTE_PNG = h.png(21, 13, (200, 100, 0))  # GET /img.png on the vision stub (remote image fetch)


class Stub(ThreadingHTTPServer):
    """A stub model server: "text" (records requests; SSE on stream=true) or "vision" (OpenAI and Anthropic caption
    replies). tls=(cert, key) makes it HTTPS."""
    daemon_threads = True

    def __init__(self, name, tls=None):
        super().__init__(("127.0.0.1", 0), StubHandler)
        self.name, self.seen, self.gate, self.gate_ok = name, [], threading.Event(), None
        self.scheme = "https" if tls else "http"
        if tls:
            c = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            c.load_cert_chain(*tls)
            self.socket = c.wrap_socket(self.socket, server_side=True)
        threading.Thread(target=self.serve_forever, daemon=True).start()

    @property
    def port(self):
        return self.server_address[1]

    @property
    def url(self):
        return f"{self.scheme}://127.0.0.1:{self.port}"

    def handle_error(self, request, client_address):
        pass  # failed TLS handshakes (untrusted-CA test) are expected


class StubHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def do_GET(self):
        self.server.seen.append((self.path, self.headers, b""))
        if self.path == "/img.png":
            return self._send(REMOTE_PNG, "image/png")
        self._send(b"", "text/plain", 404)

    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("content-length", 0)))
        self.server.seen.append((self.path, self.headers, body))
        if self.server.name == "vision":  # caption calls: OpenAI chat/completions or Anthropic messages
            if self.path.endswith("/chat/completions"):
                return self._json({"choices": [{"message": {"content": " A tiny\ntest image. "},
                                                "finish_reason": "stop"}]})
            return self._json({"content": [{"type": "text", "text": 'A detailed\ndescription that quotes "OK" | [x].'}],
                               "stop_reason": "end_turn"})
        try:
            obj = json.loads(body)
        except ValueError:
            obj = None
        if isinstance(obj, dict) and obj.get("stream"):  # SSE: 2nd event only after the client saw the 1st
            self.send_response(200)
            self.send_header("content-type", "text/event-stream")
            self.send_header("transfer-encoding", "chunked")
            self.end_headers()
            self._chunk(b"event: message_start\ndata: {}\n\n")
            self.server.gate_ok = self.server.gate.wait(10)
            self._chunk(b"event: message_stop\ndata: {}\n\n")
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
            return
        self._json({"upstream": self.server.name})

    def _chunk(self, b):
        self.wfile.write(b"%x\r\n%s\r\n" % (len(b), b))
        self.wfile.flush()

    def _json(self, obj):
        self._send(json.dumps(obj).encode(), "application/json")

    def _send(self, b, ctype, status=200):
        self.send_response(status)
        self.send_header("content-type", ctype)
        self.send_header("content-length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def mitmdump(port, confdir, env, out):
    """Same flags as the Dockerfile CMD except listen address/port and confdir."""
    return subprocess.Popen(
        ["mitmdump", "--mode", "reverse:http://127.0.0.1:9", "--listen-host", "127.0.0.1",
         "--listen-port", str(port), "--set", f"confdir={confdir}", "--set",
         "connection_strategy=lazy", "--set", "tcp_timeout=3600", "--set", "body_size_limit=64m",
         "--set", "flow_detail=0", "-s", str(ADDON)],
        env=env, stdout=out, stderr=subprocess.STDOUT)


def base_env(tmp, text, vision, **extra):
    return dict(os.environ, IMGFIX_STORE=os.path.join(tmp, "store"), TEXT_URL=text, VISION_BASE_URL=vision,
                VISION_MODEL=VISION_MODEL, VISION_API_KEY=API_KEY, VISION_EXTRA_BODY=EXTRA_BODY, HOME=tmp,
                IMGFIX_MIN_FREE_GB="0", **extra)


class Mitm:
    """One mitmdump + imgfix.py with its own env, store, confdir and log (started by start(), healthy by wait())."""

    def __init__(self, tmp, name, env):
        self.port, self.store = free_port(), env["IMGFIX_STORE"]
        self.logf = open(os.path.join(tmp, f"{name}.log"), "w+")
        self.proc = mitmdump(self.port, os.path.join(tmp, f"{name}-conf"), env, self.logf)

    def wait(self, timeout=30):
        deadline = time.monotonic() + timeout
        while True:
            try:
                if self.call("GET", "/imgfix-health")[0] == 200:
                    return self
            except OSError:
                pass
            if time.monotonic() > deadline or self.proc.poll() is not None:
                self.logf.seek(0)
                raise RuntimeError("mitmdump did not start:\n" + self.logf.read())
            time.sleep(0.2)

    def call(self, method, path, body=None, headers=None):
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=30)
        try:
            c.request(method, path, body, headers or {"content-type": "application/json"})
            r = c.getresponse()
            return r.status, r.headers, r.read()
        finally:
            c.close()

    def log_text(self, needle, timeout=5):
        deadline = time.monotonic() + timeout
        while True:
            with open(self.logf.name) as f:
                t = f.read()
            if needle in t or time.monotonic() > deadline:
                return t
            time.sleep(0.1)

    def stop(self):
        self.proc.terminate()
        try:
            self.proc.wait(10)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        self.logf.close()


def last(stub, path_prefix="/v1/messages"):
    return [s for s in stub.seen if s[0].startswith(path_prefix)][-1]


class ProxyTest(unittest.TestCase):
    """Default configuration: plain-HTTP text model without path prefix, OpenAI-format vision model, short captions,
    mcp hint, no remote fetches."""

    @classmethod
    def setUpClass(cls):
        cls.text, cls.vision = Stub("text"), Stub("vision")
        cls.tmp = tempfile.mkdtemp(prefix="imgfix-proxy-")
        cls.env = base_env(cls.tmp, cls.text.url, cls.vision.url)
        cls.mitm = Mitm(cls.tmp, "main", cls.env).wait()
        cls.port, cls.store, cls.logf = cls.mitm.port, cls.mitm.store, cls.mitm.logf

    @classmethod
    def tearDownClass(cls):
        cls.mitm.stop()
        for s in (cls.text, cls.vision):
            s.shutdown()
            s.server_close()
        shutil.rmtree(cls.tmp, ignore_errors=True)

    @classmethod
    def call(cls, method, path, body=None, headers=None):
        return cls.mitm.call(method, path, body, headers)

    def last(self, stub, path_prefix="/v1/messages"):
        return last(stub, path_prefix)

    def log_text(self, needle, timeout=5):
        return self.mitm.log_text(needle, timeout)

    # ------------------------------------------------------------

    def test_health(self):
        self.assertEqual(self.call("GET", "/imgfix-health")[0], 200)

    def test_image_rewritten_captioned_and_streamed(self):
        big = h.png(2000, 500, (10, 200, 30))
        req = h.dumps(h.body(h.user(h.image_block(big), h.text("SECRET-BODY-TEXT")), stream=True))
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=30)
        c.request("POST", "/v1/messages?beta=true", req, {"content-type": "application/json"})
        r = c.getresponse()
        self.assertEqual(r.status, 200)
        first = r.read1(65536)
        self.assertIn(b"message_start", first)  # arrived while upstream is still holding the stream open
        self.text.gate.set()
        rest = r.read()
        c.close()
        self.assertIn(b"message_stop", first + rest)
        self.assertTrue(self.text.gate_ok, "response was buffered, not streamed")
        path, hdrs, sent = self.last(self.text)
        self.assertEqual(path, "/v1/messages?beta=true")
        self.assertEqual(int(hdrs["content-length"]), len(sent))
        self.assertIsNone(hdrs.get("transfer-encoding"))
        out = json.loads(sent)
        self.assertEqual(h.media_blocks(out["messages"]), [])
        self.assertTrue(sent.isascii())
        t = h.texts(out)
        self.assertRegex(t[0], r"^\[image img_[0-9a-f]{16} \| 2000x500 png \| A tiny test image\.\] The model")
        self.assertEqual(t[1], "SECRET-BODY-TEXT")
        # Caption call: OpenAI chat API, downscaled PNG data URL, deterministic sampling, Bearer key, extra body merged.
        _, cap_hdrs, cap = self.last(self.vision, "/v1/chat/completions")
        self.assertEqual(cap_hdrs["authorization"], f"Bearer {API_KEY}")
        cap = json.loads(cap)
        self.assertEqual((cap["model"], cap["temperature"], cap["max_tokens"], cap["chat_template_kwargs"], cap["top_k"]),
                         (VISION_MODEL, 0, 60, {"enable_thinking": False}, 20))
        url = cap["messages"][0]["content"][0]["image_url"]["url"]
        self.assertTrue(url.startswith("data:image/png;base64,"))
        with Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1]))) as im:
            self.assertEqual(im.size, (1600, 400))
        iid = t[0].split()[1]
        self.assertTrue(os.path.exists(os.path.join(self.store, "meta", iid + ".json")))
        # Logs: one summary line with counts + ids, never body content or the API key.
        logs = self.log_text(iid)
        self.assertIn(f"ids={iid}", logs)
        self.assertIn("auth=set", logs)  # startup line says whether a key is configured, not which
        self.assertNotIn("SECRET-BODY-TEXT", logs)
        self.assertNotIn(h.b64(big)[:40], logs)
        self.assertNotIn(API_KEY, logs)
        # Same history again (next turn): byte-identical body to the text model, no new caption call.
        n_cap = len([s for s in self.vision.seen if s[0] == "/v1/chat/completions"])
        self.call("POST", "/v1/messages", h.dumps(h.body(h.user(h.image_block(big), h.text("SECRET-BODY-TEXT")),
                                                       stream=False)))
        out2 = json.loads(self.last(self.text)[2])
        self.assertEqual(out2["messages"], out["messages"])
        self.assertEqual(len([s for s in self.vision.seen if s[0] == "/v1/chat/completions"]), n_cap)

    def test_count_tokens_read_only(self):
        raw = h.png(33, 44, (1, 2, 3))
        st, _, _ = self.call("POST", "/v1/messages/count_tokens?beta=true", h.dumps(h.body(h.user(h.image_block(raw)))))
        self.assertEqual(st, 200)
        out = json.loads(self.last(self.text, "/v1/messages/count_tokens")[2])
        t = h.texts(out)[0]
        self.assertRegex(t, r"^\[image img_[0-9a-f]{16} \| 33x44 png \| no preview\]")
        self.assertFalse(os.path.exists(os.path.join(self.store, "meta", t.split()[1] + ".json")))

    def test_vision_model_name_is_not_special(self):
        # No direct route to the vision model: a body naming it is rewritten and goes to the text model like any other.
        n = len(self.vision.seen)
        req = h.dumps(h.body(h.user(h.image_block(h.png(7, 7))), model=VISION_MODEL))
        st, _, resp = self.call("POST", "/v1/messages", req)
        self.assertEqual((st, json.loads(resp)), (200, {"upstream": "text"}))
        out = json.loads(self.last(self.text)[2])
        self.assertEqual((out["model"], h.media_blocks(out)), (VISION_MODEL, []))
        self.assertEqual([s[0] for s in self.vision.seen[n:]], ["/v1/chat/completions"])  # its caption, nothing else

    def test_non_json_rule(self):
        n = len(self.text.seen)
        for body, ctype in ((b'{"model": "text-model", "messages": [ {"type": "image" ... not json', "application/json"),
                            (b'{"model":"m","messages":[],}', "text/plain"),             # trailing comma
                            (b'{"model":"m\xff","messages":[]}', "application/json"),     # invalid UTF-8
                            (b'{"messages": [}', "multipart/form-data; boundary=x"),      # JSON-looking, any type
                            (b"--x\r\nContent-Type: image/png\r\n\r\n\x89PNG\r\n--x--", "multipart/form-data; boundary=x")):
            st, _, resp = self.call("POST", "/v1/messages", body, {"content-type": ctype})
            self.assertEqual(st, 400, body)
            self.assertEqual(json.loads(resp)["error"]["type"], "invalid_request_error")
        self.assertEqual(len(self.text.seen), n)  # nothing forwarded
        audio = (b"--x\r\nContent-Disposition: form-data; name=\"file\"; filename=\"a.mp3\"\r\n"
                 b"Content-Type: audio/mpeg\r\n\r\nID3\x03\x00\r\n--x--\r\n")
        st, _, resp = self.call("POST", "/v1/audio/transcriptions", audio,
                                {"content-type": "multipart/form-data; boundary=x"})
        self.assertEqual((st, json.loads(resp)), (200, {"upstream": "text"}))
        self.assertEqual(self.last(self.text, "/v1/audio")[2], audio)  # clearly not JSON: forwarded unchanged

    def test_compressed_body(self):
        raw = h.png(13, 7)
        req = h.dumps(h.body(h.user(h.image_block(raw), h.text("zipped"))))
        st, _, _ = self.call("POST", "/v1/messages", gzip.compress(req),
                             {"content-type": "application/json", "content-encoding": "gzip"})
        self.assertEqual(st, 200)
        _, hdrs, sent = self.last(self.text)
        self.assertIsNone(hdrs.get("content-encoding"))  # checked decoded, forwarded decoded
        self.assertEqual(int(hdrs["content-length"]), len(sent))
        out = json.loads(sent)
        self.assertEqual(h.media_blocks(out), [])
        self.assertRegex(h.texts(out)[0], r"^\[image img_[0-9a-f]{16} \| 13x7 png \| ")
        plain = h.dumps(h.body(h.user(h.text("no media")), stream=False))
        self.call("POST", "/v1/messages", zlib.compress(plain), {"content-type": "application/json",
                                                                  "content-encoding": "deflate"})
        _, hdrs, sent = self.last(self.text)
        self.assertEqual((sent, hdrs.get("content-encoding")), (plain, None))
        n = len(self.text.seen)
        for enc, body in (("gzip", b"not gzip"), ("gzip", gzip.compress(req) + b"trailing"), ("compress", req)):
            st, _, resp = self.call("POST", "/v1/messages", body, {"content-type": "application/json",
                                                                   "content-encoding": enc})
            self.assertEqual(st, 400, enc)
            self.assertIn("Content-Encoding", json.loads(resp)["error"]["message"])
        self.assertEqual(len(self.text.seen), n)

    def test_residual_shape_rejected(self):
        n = len(self.text.seen)
        for body in ({"model": "m", "messages": [h.user({"type": "IMAGE", "source": h.image_block(h.png())["source"]})]},
                     {"model": "m", "Messages": [h.user(h.text("x"))], "messages": [h.user(h.text("y"))]},
                     {"model": "m", "system": [h.image_block(h.png())], "messages": [h.user(h.text("x"))]}):
            st, _, resp = self.call("POST", "/v1/messages", h.dumps(body))
            self.assertEqual(st, 400)
            self.assertIn("it was not forwarded", json.loads(resp)["error"]["message"])
        self.assertEqual(len(self.text.seen), n)

    def test_gemini_native(self):
        raw = h.png(19, 11)
        part = {"inlineData": {"mimeType": "image/png", "data": h.b64(raw)}}
        body = h.dumps({"contents": [{"role": "user", "parts": [{"text": "what?"}, part]}]})
        path = "/v1beta/models/gemini-text:streamGenerateContent?alt=sse"
        self.assertEqual(self.call("POST", path, body, {"content-type": "application/json",
                                                        "x-goog-api-key": "client-key"})[0], 200)
        p, hdrs, sent = self.last(self.text, "/v1beta/")
        self.assertEqual((p, hdrs["x-goog-api-key"]), (path, "client-key"))
        parts = json.loads(sent)["contents"][0]["parts"]
        self.assertEqual(parts[0], {"text": "what?"})
        self.assertRegex(parts[1]["text"], r"^\[image img_[0-9a-f]{16} \| 19x11 png \| A tiny test image\.\]")
        other = h.png(12, 21, (9, 9, 9))
        count = h.dumps({"generateContentRequest": {"model": "models/gemini-text", "contents": [
            {"role": "user", "parts": [{"inline_data": {"mime_type": "image/png", "data": h.b64(other)}}]}]}})
        self.assertEqual(self.call("POST", "/v1beta/models/gemini-text:countTokens", count)[0], 200)
        t = json.loads(self.last(self.text, "/v1beta/models/gemini-text:countTokens")[2])
        ref = t["generateContentRequest"]["contents"][0]["parts"][0]["text"]
        self.assertRegex(ref, r"^\[image img_[0-9a-f]{16} \| 12x21 png \| no preview\]")
        self.assertFalse(os.path.exists(os.path.join(self.store, "meta", ref.split()[1] + ".json")))

    def test_health_store_not_writable(self):
        self.assertEqual(self.call("GET", "/imgfix-health")[0], 200)
        if os.geteuid() == 0:
            self.skipTest("root ignores file modes")
        img = os.path.join(self.store, "img")
        os.chmod(img, 0o555)
        try:
            st, _, body = self.call("GET", "/imgfix-health")
            self.assertEqual((st, body), (503, b"store not writable\n"))
        finally:
            os.chmod(img, 0o755)
        self.assertEqual(self.call("GET", "/imgfix-health")[0], 200)

    def test_any_path_is_scanned(self):
        st, _, _ = self.call("POST", "/v1/messages/", h.dumps(h.body(h.user(h.image_block(h.png(9, 9))))))
        self.assertEqual(st, 200)
        path, _, sent = self.last(self.text)
        self.assertEqual(path, "/v1/messages/")
        self.assertEqual(h.media_blocks(json.loads(sent)), [])

    def test_chunked_request(self):
        req = h.dumps(h.body(h.user(h.image_block(h.png(11, 12)), h.text("chunked"))))
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=30)
        try:
            c.request("POST", "/v1/messages", iter([req[:100], req[100:]]),
                      {"content-type": "application/json", "transfer-encoding": "chunked"}, encode_chunked=True)
            self.assertEqual(c.getresponse().status, 200)
        finally:
            c.close()
        _, hdrs, sent = self.last(self.text)
        self.assertEqual(int(hdrs["content-length"]), len(sent))
        self.assertIsNone(hdrs.get("transfer-encoding"))
        out = json.loads(sent)
        self.assertEqual(h.media_blocks(out), [])
        self.assertRegex(h.texts(out)[0], r"^\[image img_[0-9a-f]{16} \| 11x12 png \| ")

    def test_unparseable_media_body_rejected(self):
        n = len(self.text.seen)
        deep = b'{"model":"text-model","messages":[' + b"[" * 100000 + b'{"type":"image"}' + b"]" * 100000 + b"]}"
        st, _, resp = self.call("POST", "/v1/messages", deep)
        self.assertEqual(st, 400)
        self.assertEqual(json.loads(resp)["error"]["type"], "invalid_request_error")
        self.assertEqual(len(self.text.seen), n)  # nothing forwarded

    def test_body_size_limit(self):
        n = len(self.text.seen)
        with socket.create_connection(("127.0.0.1", self.port), timeout=30) as s:
            s.sendall(b"POST /v1/messages HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\n"
                      b"Content-Length: %d\r\n\r\n{" % (65 << 20))
            status = s.recv(4096).split(b"\r\n", 1)[0]
        self.assertIn(b" 413 ", status)
        self.assertEqual(len(self.text.seen), n)

    def test_bad_config_stops_mitmdump_at_startup(self):
        cases = (({"VISION_EXTRA_BODY": "{not json"}, "VISION_EXTRA_BODY is not valid JSON"),
                 ({"VISION_EXTRA_BODY": '{"model":"x"}'}, "VISION_EXTRA_BODY may not set model"),
                 ({"VISION_MODEL": ""}, "VISION_MODEL is not set"),
                 ({"TEXT_URL": "ftp://example.com"}, "TEXT_URL must start with http:// or https://"),
                 ({"TEXT_URL": "https://api.example.com/v1?key=1"}, "TEXT_URL may not have a ?query"),
                 ({"TEXT_URL": "https://10.0.0.5"}, "TEXT_URL: https needs a host name"),
                 ({"TEXT_CA_FILE": "/nonexistent/ca.pem"}, "TEXT_CA_FILE is set, but TEXT_URL is not https://"),
                 ({"TEXT_URL": "https://api.example.com", "TEXT_CA_FILE": "/nonexistent/ca.pem"},
                  "TEXT_CA_FILE '/nonexistent/ca.pem' is not a readable"),
                 ({"IMGFIX_MIN_FREE_GB": "1e300"}, "IMGFIX_MIN_FREE_GB must be a number >= 0 and <= 1e+06"),
                 ({"CAPTION_MODE": "loud"}, "CAPTION_MODE must be one of short, detailed, off"),
                 ({"ASK_HINT": "smoke"}, "ASK_HINT must be one of mcp, http, none"),
                 ({"FETCH_REMOTE_IMAGES": "maybe"}, "FETCH_REMOTE_IMAGES must be true or false"),
                 ({"VISION_API": "gemini"}, "VISION_API must be one of openai, anthropic"))
        procs = []
        for k, (env, _) in enumerate(cases):  # all at once: each exits as soon as the addon loads
            out = tempfile.TemporaryFile("w+")
            procs.append((out, mitmdump(free_port(), f"{self.tmp}/bad-{k}", dict(self.env, **env), out)))
        for (out, p), (_, needle) in zip(procs, cases):
            try:
                rc = p.wait(60)
            finally:
                if p.poll() is None:
                    p.kill()
                    p.wait()
            out.seek(0)
            text = out.read()
            out.close()
            self.assertEqual(rc, 2, text)
            self.assertIn(f"imgfix: configuration error: {needle}", text)

    # ------------------------------------------------------------ other request formats

    def test_openai_chat_and_host_header(self):
        raw = h.png(14, 15)
        req = {"model": "gpt-oss", "messages": [{"role": "user", "content": [
            {"type": "text", "text": "what?"}, {"type": "image_url", "image_url": {"url": h.data_url(raw)}}]}]}
        st, _, resp = self.call("POST", "/v1/chat/completions", h.dumps(req),
                                {"content-type": "application/json", "authorization": "Bearer client-key"})
        self.assertEqual((st, json.loads(resp)), (200, {"upstream": "text"}))
        path, hdrs, sent = self.last(self.text, "/v1/chat/completions")
        self.assertEqual(path, "/v1/chat/completions")
        self.assertEqual(hdrs["host"], f"127.0.0.1:{self.text.port}")  # TEXT_URL authority
        self.assertEqual(hdrs["authorization"], "Bearer client-key")  # the client's own key passes through
        out = json.loads(sent)
        self.assertEqual(h.media_blocks(out), [])
        self.assertRegex(out["messages"][0]["content"][1]["text"], r"^\[image img_[0-9a-f]{16} \| 14x15 png \| A tiny")

    def test_responses_and_input_tokens(self):
        raw = h.png(16, 17)
        req = {"model": "gpt-oss", "input": [{"role": "user", "content": [
            {"type": "input_image", "image_url": h.data_url(raw)}]}]}
        self.assertEqual(self.call("POST", "/v1/responses/input_tokens", h.dumps(req))[0], 200)
        out = json.loads(self.last(self.text, "/v1/responses/input_tokens")[2])
        t = out["input"][0]["content"][0]
        self.assertEqual(t["type"], "input_text")
        self.assertRegex(t["text"], r"^\[image img_[0-9a-f]{16} \| 16x17 png \| no preview\]")
        self.assertFalse(os.path.exists(os.path.join(self.store, "meta", t["text"].split()[1] + ".json")))
        self.assertEqual(self.call("POST", "/v1/responses", h.dumps(req))[0], 200)
        out = json.loads(self.last(self.text, "/v1/responses")[2])
        self.assertRegex(out["input"][0]["content"][0]["text"], r"^\[image img_[0-9a-f]{16} \| 16x17 png \| A tiny")

    def test_ollama_native(self):
        raw = h.png(18, 19)
        self.call("POST", "/api/chat", h.dumps({"model": "llava", "stream": False, "messages": [
            {"role": "user", "content": "describe", "images": [h.b64(raw)]}]}))
        msg = json.loads(self.last(self.text, "/api/chat")[2])["messages"][0]
        self.assertEqual(sorted(msg), ["content", "role"])
        self.assertRegex(msg["content"], r"^describe\n\n\[image img_[0-9a-f]{16} \| 18x19 png \| A tiny")
        self.call("POST", "/api/generate", h.dumps({"model": "llava", "prompt": "p", "images": [h.b64(raw)]}))
        out = json.loads(self.last(self.text, "/api/generate")[2])
        self.assertEqual(out["prompt"], "p\n\n" + msg["content"].split("\n\n", 1)[1])
        self.assertNotIn("images", out)


class AltProxyTest(unittest.TestCase):
    """Every non-default option at once: https text model (test CA, path prefix), Anthropic-format vision model,
    detailed captions, http hint, model passthrough, remote image fetch. Plus an https upstream whose certificate is
    not trusted (no TEXT_CA_FILE): nothing may be forwarded."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="imgfix-proxy-alt-")
        ca, cert, key = h.make_ca(cls.tmp)
        cls.text, cls.vision = Stub("text", tls=(cert, key)), Stub("vision")
        cls.ask = "http://127.0.0.1:8102/ask"
        env = base_env(cls.tmp, f"https://localhost:{cls.text.port}/anthropic/", cls.vision.url, TEXT_CA_FILE=ca,
                       VISION_API="anthropic", CAPTION_MODE="detailed", ASK_HINT="http", ASK_URL=cls.ask,
                       PASSTHROUGH_MODELS="claude-*, gpt-4o, gemini-2.5-*", FETCH_REMOTE_IMAGES="true", FETCH_ALLOW_PRIVATE="true")
        untrusted = base_env(os.path.join(cls.tmp, "u"), f"https://localhost:{cls.text.port}", cls.vision.url)
        os.makedirs(os.path.join(cls.tmp, "u"))
        cls.mitm, cls.untrusted = Mitm(cls.tmp, "alt", env), Mitm(cls.tmp, "untrusted", untrusted)
        cls.mitm.wait()
        cls.untrusted.wait()

    @classmethod
    def tearDownClass(cls):
        cls.mitm.stop()
        cls.untrusted.stop()
        for s in (cls.text, cls.vision):
            s.shutdown()
            s.server_close()
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_https_prefix_host_and_options(self):
        raw = h.png(30, 20, (5, 6, 7))
        req = h.dumps(h.body(h.user(h.image_block(raw), h.text("hi"))))
        st, _, resp = self.mitm.call("POST", "/v1/messages?beta=true", req,
                                     {"content-type": "application/json", "x-api-key": "client-key",
                                      "anthropic-version": "2023-06-01"})
        self.assertEqual((st, json.loads(resp)), (200, {"upstream": "text"}))
        path, hdrs, sent = last(self.text, "/anthropic/")
        self.assertEqual(path, "/anthropic/v1/messages?beta=true")  # prefix + the client's path and query
        self.assertEqual((hdrs["host"], hdrs["x-api-key"]), (f"localhost:{self.text.port}", "client-key"))
        t = h.texts(json.loads(sent))[0]
        i = "img_" + hashlib.sha256(raw).hexdigest()[:16]
        self.assertEqual(t, f'[image {i} | 30x20 png | A detailed description that quotes "OK" / (x).] The model '
                            f'cannot see images. To inspect this one, run: curl -s -G {self.ask} --data-urlencode '
                            f'image_id={i} --data-urlencode "question=<your question>"')
        # Caption call in the Anthropic Messages format, detailed mode.
        _, vh, vb = last(self.vision, "/v1/messages")
        vb = json.loads(vb)
        self.assertEqual((vh["x-api-key"], vh["anthropic-version"], vb["model"], vb["max_tokens"]),
                         (API_KEY, "2023-06-01", VISION_MODEL, 600))
        self.assertEqual(vb["messages"][0]["content"][0]["source"]["media_type"], "image/png")
        logs = self.mitm.log_text("ids=" + i)
        self.assertIn("caption=detailed/45s hint=http", logs)
        self.assertNotIn(API_KEY, logs)

    def test_passthrough_model(self):
        req = h.dumps(h.body(h.user(h.image_block(h.png(8, 9))), model="claude-sonnet-4-5"))
        self.assertEqual(self.mitm.call("POST", "/v1/messages", req)[0], 200)
        path, _, sent = last(self.text, "/anthropic/")
        self.assertEqual((path, sent), ("/anthropic/v1/messages", req))  # byte for byte, images intact

    def test_passthrough_gemini_path_model(self):
        part = {"inline_data": {"mime_type": "image/png", "data": h.b64(h.png(6, 9))}}
        req = h.dumps({"contents": [{"role": "user", "parts": [part]}]})
        path = "/v1beta/models/gemini-2.5-pro:generateContent"
        self.assertEqual(self.mitm.call("POST", path, req)[0], 200)
        got, _, sent = last(self.text, "/anthropic/")
        self.assertEqual((got, sent), ("/anthropic" + path, req))  # untouched: images intact
        # the body names a second model that does not match: rewritten
        self.mitm.call("POST", path, h.dumps({"model": "text-model", "contents": [{"role": "user", "parts": [part]}]}))
        self.assertEqual(h.media_blocks(json.loads(last(self.text, "/anthropic/")[2])), [])

    def test_remote_image_fetched_once(self):
        url = self.vision.url + "/img.png"
        req = h.dumps(h.body(h.user({"type": "image", "source": {"type": "url", "url": url}})))
        n = len([s for s in self.vision.seen if s[0] == "/img.png"])
        bodies = []
        for _ in range(2):
            self.assertEqual(self.mitm.call("POST", "/v1/messages", req)[0], 200)
            bodies.append(last(self.text, "/anthropic/")[2])
        self.assertEqual(bodies[0], bodies[1])
        self.assertRegex(h.texts(json.loads(bodies[0]))[0], r"^\[image img_[0-9a-f]{16} \| 21x13 png \| A detailed")
        self.assertEqual(len([s for s in self.vision.seen if s[0] == "/img.png"]), n + 1)

    def test_untrusted_https_upstream_not_forwarded(self):
        n = len(self.text.seen)
        st, _, _ = self.untrusted.call("POST", "/v1/messages", h.dumps(h.body(h.user(h.text("hello")))))
        self.assertEqual(st, 502)
        self.assertEqual(len(self.text.seen), n)


if __name__ == "__main__":
    unittest.main()
