"""Stub text model (:8000) and vision model (:8001) for the stack test (test/run.sh). Records every request
byte-for-byte, headers included.

Control (either port): GET /_records -> JSON list of recorded requests (body base64, sha256, len, peer IP;
GET /_records?bodies=0 omits the bodies); POST /_reset clears.
GET /health -> 200 (container health check). POST /v1/messages -> minimal Anthropic message.
POST /v1/chat/completions (vision only) -> fixed caption (caption request: max_tokens 60) or fixed answer (ask_image).
"""
import base64
import hashlib
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

CAPTION = "A red rectangle on a white background."
ANSWER = "The rectangle is red. [stub answer]"
RECORDS, LOCK = [], threading.Lock()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    upstream = "?"

    def log_message(self, fmt, *args):  # quiet: no bodies in logs
        pass

    def _send(self, code, obj, ctype="application/json"):
        data = obj if isinstance(obj, bytes) else json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("content-type", ctype)
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _body(self):
        if "chunked" in self.headers.get("transfer-encoding", "").lower():
            out = b""
            while True:
                n = int(self.rfile.readline().split(b";")[0], 16)
                if n == 0:
                    self.rfile.readline()
                    return out
                out += self.rfile.read(n)
                self.rfile.readline()
        return self.rfile.read(int(self.headers.get("content-length") or 0))

    def do_GET(self):
        if self.path == "/health":
            return self._send(200, b"ok\n", "text/plain")
        if self.path in ("/_records", "/_records?bodies=0"):
            with LOCK:
                recs = list(RECORDS)
            if self.path.endswith("bodies=0"):
                recs = [{k: v for k, v in r.items() if k != "body_b64"} for r in recs]
            return self._send(200, recs)
        self._send(404, {"error": "not found"})

    def do_POST(self):
        body = self._body()
        if self.path == "/_reset":
            with LOCK:
                RECORDS.clear()
            return self._send(200, {"ok": True})
        with LOCK:
            RECORDS.append({"upstream": self.upstream, "method": "POST", "path": self.path,
                            "peer": self.client_address[0], "headers": dict(self.headers.items()),
                            "sha256": hashlib.sha256(body).hexdigest(), "len": len(body),
                            "body_b64": base64.b64encode(body).decode()})
        if self.path.endswith("/v1/messages"):
            return self._send(200, {"id": f"msg_stub_{self.upstream}", "type": "message", "role": "assistant",
                                    "model": self.upstream, "content": [{"type": "text", "text": "ok"}],
                                    "stop_reason": "end_turn", "stop_sequence": None,
                                    "usage": {"input_tokens": 1, "output_tokens": 1}})
        if self.upstream == "vision" and self.path.endswith("/v1/chat/completions"):
            try:
                req = json.loads(body)
                caption = req.get("max_tokens") == 60
            except ValueError:
                caption = False
            return self._send(200, {"id": "chatcmpl-stub", "object": "chat.completion", "model": "stub",
                                    "choices": [{"index": 0, "finish_reason": "stop",
                                                 "message": {"role": "assistant",
                                                             "content": CAPTION if caption else ANSWER}}]})
        self._send(404, {"error": "not found"})


def serve(port, name):
    h = type(f"{name}Handler", (Handler,), {"upstream": name})
    ThreadingHTTPServer(("0.0.0.0", port), h).serve_forever()


if __name__ == "__main__":
    threading.Thread(target=serve, args=(8000, "text"), daemon=True).start()
    serve(8001, "vision")
