"""Stub text model and vision model for the stack test (test/run.sh). Records every request byte for byte, headers
included. Stdlib only.

Ports: 8000 text model (http), 8443 text model (https; /tls/server.pem + server.key from test/certgen.py), 8001
vision model (http). The text model answers whatever path suffix the client format uses, below any TEXT_URL prefix:
  .../v1/messages (Anthropic), .../count_tokens and .../input_tokens (counts), .../chat/completions (OpenAI Chat),
  .../responses (OpenAI Responses), .../api/chat and .../api/generate (Ollama), .../models/<m>:generateContent,
  :streamGenerateContent (SSE: GEMINI_SSE, which the client must receive byte for byte) and :countTokens (Gemini).
  GET .../img/<color>.png serves a small PNG (remote image fetches).
The vision model speaks both VISION_API formats: POST .../chat/completions (openai) and .../messages (anthropic).
Caption vs ask_image is told apart by the token limit: 60 -> CAPTION (short), 600 -> DETAILED (detailed caption),
anything else -> ANSWER. Requests come from the services only (no client keys), so any key seen is the test's.

Control (any port, never recorded): GET /health -> 200; GET /_records -> JSON list of recorded requests (body base64,
sha256, len, peer IP, tls; GET /_records?bodies=0 omits the bodies); POST /_reset clears.
"""
import base64
import hashlib
import json
import ssl
import struct
import threading
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

CAPTION = "A red rectangle on a white background."
# Detailed caption: > 30 words and > 300 characters (beyond the short-mode limits), with line breaks and double spaces
# that imgfix collapses; no [ ] | delimiters, so the cleaned text is predictable: " ".join(DETAILED.split()).
DETAILED = ("A large red rectangle fills the centre of a plain white canvas.\n\nIt has sharp corners, no border and no "
            "shadow.  There is no visible text, logo, person or other object anywhere in the image, and the "
            "background is a uniform bright white on every side of the shape.")
ANSWER = "The rectangle is red. [stub answer]"
COLORS = {"blue": (0, 0, 255), "green": (0, 160, 0), "red": (255, 0, 0)}
GEMINI_SSE = (b'data: {"candidates": [{"content": {"role": "model", "parts": [{"text": "o"}]}}]}\r\n\r\n'
              b'data: {"candidates": [{"content": {"role": "model", "parts": [{"text": "k"}]}, "finishReason": "STOP"}]}'
              b'\r\n\r\n')
RECORDS, LOCK = [], threading.Lock()


def png(w, h, rgb):
    """A minimal RGB PNG, stdlib only."""
    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
    raw = b"".join(b"\x00" + bytes(rgb) * w for _ in range(h))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


def limit(req):
    return req.get("max_tokens", req.get("max_completion_tokens"))


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

    def _record(self, body):
        with LOCK:
            RECORDS.append({"upstream": self.upstream, "method": self.command, "path": self.path,
                            "peer": self.client_address[0], "tls": isinstance(self.connection, ssl.SSLSocket),
                            "headers": dict(self.headers.items()), "sha256": hashlib.sha256(body).hexdigest(),
                            "len": len(body), "body_b64": base64.b64encode(body).decode()})

    def do_GET(self):
        if self.path == "/health":
            return self._send(200, b"ok\n", "text/plain")
        if self.path in ("/_records", "/_records?bodies=0"):
            with LOCK:
                recs = list(RECORDS)
            if self.path.endswith("bodies=0"):
                recs = [{k: v for k, v in r.items() if k != "body_b64"} for r in recs]
            return self._send(200, recs)
        self._record(b"")
        name = self.path.rsplit("/", 1)[-1]
        if self.upstream == "text" and "/img/" in self.path and name.endswith(".png") and name[:-4] in COLORS:
            return self._send(200, png(40, 30, COLORS[name[:-4]]), "image/png")
        if self.upstream == "text" and self.path.split("?")[0].endswith("/v1/models"):
            return self._send(200, {"object": "list", "data": [{"id": "text-model", "object": "model"}]})
        self._send(404, {"error": "not found"})

    def do_POST(self):
        body = self._body()
        if self.path == "/_reset":
            with LOCK:
                RECORDS.clear()
            return self._send(200, {"ok": True})
        self._record(body)
        path = self.path.split("?")[0].rstrip("/")
        try:
            req = json.loads(body)
        except ValueError:
            req = {}
        if self.upstream == "vision":
            n = limit(req) if isinstance(req, dict) else None
            text = CAPTION if n == 60 else DETAILED if n == 600 else ANSWER
            if path.endswith("/chat/completions"):
                return self._send(200, {"id": "chatcmpl-stub", "object": "chat.completion", "model": "stub",
                                        "choices": [{"index": 0, "finish_reason": "stop",
                                                     "message": {"role": "assistant", "content": text}}]})
            if path.endswith("/messages"):
                return self._send(200, {"id": "msg_stub_vision", "type": "message", "role": "assistant",
                                        "model": "stub", "content": [{"type": "text", "text": text}],
                                        "stop_reason": "end_turn", "stop_sequence": None,
                                        "usage": {"input_tokens": 1, "output_tokens": 1}})
            return self._send(404, {"error": "not found"})
        if path.endswith("/count_tokens") or path.endswith("/input_tokens"):
            return self._send(200, {"input_tokens": 1})
        if path.endswith(":countTokens"):
            return self._send(200, {"totalTokens": 1})
        if path.endswith(":generateContent"):
            return self._send(200, {"candidates": [{"content": {"role": "model", "parts": [{"text": "ok"}]},
                                                    "finishReason": "STOP"}]})
        if path.endswith(":streamGenerateContent"):
            return self._send(200, GEMINI_SSE, "text/event-stream")
        if path.endswith("/v1/messages"):
            return self._send(200, {"id": "msg_stub_text", "type": "message", "role": "assistant",
                                    "model": "text", "content": [{"type": "text", "text": "ok"}],
                                    "stop_reason": "end_turn", "stop_sequence": None,
                                    "usage": {"input_tokens": 1, "output_tokens": 1}})
        if path.endswith("/chat/completions"):
            return self._send(200, {"id": "chatcmpl-stub-text", "object": "chat.completion", "model": "text",
                                    "choices": [{"index": 0, "finish_reason": "stop",
                                                 "message": {"role": "assistant", "content": "ok"}}]})
        if path.endswith("/responses"):
            return self._send(200, {"id": "resp_stub_text", "object": "response", "status": "completed",
                                    "model": "text", "output": [{"type": "message", "role": "assistant", "content": [
                                        {"type": "output_text", "text": "ok", "annotations": []}]}]})
        if path.endswith("/api/chat"):
            return self._send(200, {"model": "text", "message": {"role": "assistant", "content": "ok"},
                                    "done": True, "done_reason": "stop"})
        if path.endswith("/api/generate"):
            return self._send(200, {"model": "text", "response": "ok", "done": True, "done_reason": "stop"})
        self._send(404, {"error": "not found"})


def serve(port, name, tls=False):
    srv = ThreadingHTTPServer(("0.0.0.0", port), type(f"{name}Handler", (Handler,), {"upstream": name}))
    if tls:
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain("/tls/server.pem", "/tls/server.key")
        srv.socket = ctx.wrap_socket(srv.socket, server_side=True)
    srv.serve_forever()


if __name__ == "__main__":
    for args in ((8000, "text"), (8443, "text", True)):
        threading.Thread(target=serve, args=args, daemon=True).start()
    serve(8001, "vision")
