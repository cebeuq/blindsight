"""End-to-end: real mitmdump + imgfix.py between a client and local stub text/vision model servers (loopback only)."""
import base64
import http.client
import io
import json
import os
import shutil
import socket
import subprocess
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from PIL import Image

from . import helpers as h

ADDON = Path(__file__).resolve().parents[1] / "imgfix.py"
VISION_MODEL = "test-vision-model"
API_KEY = "sk-imgfix-test-key-0123456789"
EXTRA_BODY = '{"chat_template_kwargs":{"enable_thinking":false},"top_k":20}'


class Stub(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, name):
        super().__init__(("127.0.0.1", 0), StubHandler)
        self.name, self.seen, self.gate, self.gate_ok = name, [], threading.Event(), None
        threading.Thread(target=self.serve_forever, daemon=True).start()

    @property
    def url(self):
        return f"http://127.0.0.1:{self.server_address[1]}"


class StubHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("content-length", 0)))
        self.server.seen.append((self.path, self.headers, body))
        if self.path == "/v1/chat/completions":  # vision model caption call
            return self._json({"choices": [{"message": {"content": " A tiny\ntest image. "}}]})
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
        b = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
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


class ProxyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text, cls.vision = Stub("text"), Stub("vision")
        cls.tmp = tempfile.mkdtemp(prefix="imgfix-proxy-")
        cls.store = os.path.join(cls.tmp, "store")
        cls.port = free_port()
        cls.logf = open(os.path.join(cls.tmp, "mitmdump.log"), "w+")
        cls.env = dict(os.environ, IMGFIX_STORE=cls.store, TEXT_UPSTREAM=cls.text.url, VISION_BASE_URL=cls.vision.url,
                       VISION_MODEL=VISION_MODEL, VISION_API_KEY=API_KEY, VISION_EXTRA_BODY=EXTRA_BODY,
                       HOME=cls.tmp, IMGFIX_MIN_FREE_GB="0")
        cls.proc = mitmdump(cls.port, f"{cls.tmp}/mitmproxy", cls.env, cls.logf)
        deadline = time.monotonic() + 30
        while True:
            try:
                st, _, _ = cls.call("GET", "/imgfix-health")
                if st == 200:
                    break
            except OSError:
                pass
            if time.monotonic() > deadline or cls.proc.poll() is not None:
                cls.logf.seek(0)
                raise RuntimeError("mitmdump did not start:\n" + cls.logf.read())
            time.sleep(0.2)

    @classmethod
    def tearDownClass(cls):
        cls.proc.terminate()
        try:
            cls.proc.wait(10)
        except subprocess.TimeoutExpired:
            cls.proc.kill()
        cls.logf.close()
        for s in (cls.text, cls.vision):
            s.shutdown()
            s.server_close()
        shutil.rmtree(cls.tmp, ignore_errors=True)

    @classmethod
    def call(cls, method, path, body=None, headers=None):
        c = http.client.HTTPConnection("127.0.0.1", cls.port, timeout=30)
        try:
            c.request(method, path, body, headers or {"content-type": "application/json"})
            r = c.getresponse()
            return r.status, r.headers, r.read()
        finally:
            c.close()

    def last(self, stub, path_prefix="/v1/messages"):
        return [s for s in stub.seen if s[0].startswith(path_prefix)][-1]

    def log_text(self, needle, timeout=5):
        deadline = time.monotonic() + timeout
        while True:
            with open(self.logf.name) as f:
                t = f.read()
            if needle in t or time.monotonic() > deadline:
                return t
            time.sleep(0.1)

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
        self.assertIn("api_key=set", logs)  # startup line says whether a key is configured, not which
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

    def test_not_json_passthrough(self):
        req = b'{"model": "text-model", "messages": [ {"type": "image" ... not json'
        st, _, resp = self.call("POST", "/v1/messages", req)
        self.assertEqual((st, json.loads(resp)), (200, {"upstream": "text"}))
        self.assertEqual(self.last(self.text)[2], req)

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
        for env, needle in (({"VISION_EXTRA_BODY": "{not json"}, "VISION_EXTRA_BODY is not valid JSON"),
                            ({"VISION_EXTRA_BODY": '{"model":"x"}'}, "VISION_EXTRA_BODY may not set model"),
                            ({"VISION_MODEL": ""}, "VISION_MODEL is not set")):
            with tempfile.TemporaryFile("w+") as out:
                p = mitmdump(free_port(), f"{self.tmp}/mitmproxy-bad", dict(self.env, **env), out)
                try:
                    rc = p.wait(30)
                finally:
                    if p.poll() is None:
                        p.kill()
                        p.wait()
                out.seek(0)
                text = out.read()
            self.assertEqual(rc, 2, text)
            self.assertIn(f"imgfix: configuration error: {needle}", text)


if __name__ == "__main__":
    unittest.main()
