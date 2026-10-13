"""Shared test helpers: a temp /store, test images/PDFs, a recording vision-model stub (OpenAI or Anthropic format),
and the server on a free port."""

import io
import json
import socket
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from PIL import Image

import server
import vision_api

HEX = "0123456789abcdef"
IMG_ID = "img_" + HEX
PDF_ID = "pdf_fedcba9876543210"
QUAD = {"tl": (255, 0, 0), "tr": (0, 255, 0), "bl": (0, 0, 255), "br": (255, 255, 0)}


def quad_image(w: int, h: int) -> Image.Image:
    """RGB image with four solid quadrants (colors in QUAD) so crops are easy to verify."""
    im = Image.new("RGB", (w, h))
    for key, box in (("tl", (0, 0, w // 2, h // 2)), ("tr", (w // 2, 0, w, h // 2)),
                     ("bl", (0, h // 2, w // 2, h)), ("br", (w // 2, h // 2, w, h))):
        im.paste(QUAD[key], box)
    return im


def two_page_pdf(path: Path) -> None:
    """Page 1: 720x360 pt quadrant image; page 2: 360x720 pt solid blue (Pillow PDF at 72 dpi: 1 px = 1 pt)."""
    quad_image(720, 360).save(path, save_all=True, append_images=[Image.new("RGB", (360, 720), (0, 0, 255))], resolution=72)


class Store:
    """Temporary store with imgfix's layout (img/, pdf/, meta/)."""

    def __init__(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        for d in ("img", "pdf", "meta"):
            (self.root / d).mkdir()

    def add_image(self, image_id: str, im: Image.Image, ext: str = "png", **save) -> Path:
        p = self.root / "img" / f"{image_id}.{ext}"
        im.save(p, format={"png": "PNG", "jpg": "JPEG", "gif": "GIF", "webp": "WEBP"}[ext], **save)
        return p

    def add_pdf(self, pdf_id: str) -> Path:
        p = self.root / "pdf" / f"{pdf_id}.pdf"
        two_page_pdf(p)
        return p

    def close(self):
        self._tmp.cleanup()


def decode_png_data_url(url: str) -> Image.Image:
    import base64
    prefix = "data:image/png;base64,"
    assert url.startswith(prefix), url[:40]
    im = Image.open(io.BytesIO(base64.b64decode(url[len(prefix):])))
    assert im.format == "PNG"
    im.load()
    return im


def vision(url: str, **env: str) -> "vision_api.VisionAPI":
    """VisionAPI for a stub at url, built the way the server builds it (from VISION_* settings)."""
    return vision_api.from_env({"VISION_BASE_URL": url, "VISION_MODEL": "test-vision-model", **env})


class VisionStub:
    """Minimal vision endpoint (OpenAI chat/completions or Anthropic messages reply shape, per api; "raw": the answer as
    a non-JSON body) that records every request. stop: the finish/stop reason sent; delay: seconds to wait before answering; inflight / max_inflight:
    concurrent requests now / at most."""

    def __init__(self, answer: str = "The dialog says \"Disk full\".", status: int = 200, api: str = "openai"):
        self.answer, self.status, self.api, self.requests = answer, status, api, []
        self.stop, self.delay, self.error_text = None, 0.0, "stub failure"
        self.inflight = self.max_inflight = 0
        self._lock = threading.Lock()
        stub = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                with stub._lock:
                    stub.requests.append({"path": self.path, "headers": dict(self.headers), "body": body})
                    stub.inflight += 1
                    stub.max_inflight = max(stub.max_inflight, stub.inflight)
                try:
                    time.sleep(stub.delay)
                finally:
                    with stub._lock:
                        stub.inflight -= 1
                if stub.api == "raw":  # answer as the raw (non-JSON) body
                    out = None
                elif stub.status != 200:
                    out = {"error": {"message": stub.error_text, "code": stub.status}}
                elif stub.api == "anthropic":
                    out = {"id": "msg_x", "type": "message", "role": "assistant", "model": body.get("model"),
                           "content": [{"type": "thinking", "thinking": "hmm", "signature": "s"},
                                       {"type": "text", "text": stub.answer}],
                           "stop_reason": stub.stop or "end_turn", "usage": {"input_tokens": 1, "output_tokens": 1}}
                else:
                    out = {"id": "x", "object": "chat.completion", "model": body.get("model"),
                           "choices": [{"index": 0, "message": {"role": "assistant", "content": stub.answer},
                                        "finish_reason": stub.stop or "stop"}]}
                data = stub.answer.encode() if out is None else json.dumps(out).encode()
                self.send_response(stub.status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *a):
                pass

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def reset(self, answer: str, api: str = "openai"):
        self.requests.clear()
        self.answer, self.api, self.status, self.stop, self.delay, self.error_text = answer, api, 200, None, 0.0, "stub failure"
        self.max_inflight = 0

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


def closed_port_url() -> str:
    """URL of a localhost port with nothing listening (connection refused)."""
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return f"http://127.0.0.1:{port}"


class RunningServer:
    """server.build_app(cfg) under uvicorn (set up like production: server.uvicorn_server) in a background thread, on a
    pre-bound free port."""

    def __init__(self, cfg: "server.Config"):
        self.sock = socket.socket()
        self.sock.bind(("127.0.0.1", 0))
        self.port = self.sock.getsockname()[1]
        self.url = f"http://127.0.0.1:{self.port}"
        self.uv = server.uvicorn_server(cfg, log_level="warning", lifespan="on")
        self.thread = threading.Thread(target=self.uv.run, kwargs={"sockets": [self.sock]}, daemon=True)
        self.thread.start()
        deadline = time.monotonic() + 10
        while not self.uv.started:
            if time.monotonic() > deadline:
                raise RuntimeError("uvicorn did not start")
            time.sleep(0.02)

    def close(self):
        self.uv.should_exit = True
        self.thread.join(10)
        self.sock.close()
