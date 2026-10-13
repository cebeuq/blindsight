"""End-to-end: the official MCP SDK client talks streamable HTTP to the real server (uvicorn), which calls a
recording vision-model stub. Checks what the text model would see (tool listing, answer text, error strings) and what
the vision model receives."""

import unittest

import anyio
import httpx2
from mcp import Client

import server
from tests.util import IMG_ID, PDF_ID, QUAD, RunningServer, Store, VisionStub, closed_port_url, decode_png_data_url, quad_image

ANSWER = 'The dialog title is "Disk full" and the button says "OK".'
MODEL = "test-vision-model"
API_KEY = "sk-vision-mcp-test-key"
EXTRA = {"chat_template_kwargs": {"enable_thinking": False}, "top_k": 20}


def call(url: str, args: dict, mode: str = "legacy") -> tuple[str, bool]:
    """One ask_image call over HTTP; returns (text, is_error). mode "legacy" = initialize handshake, like Claude Code."""
    async def go():
        async with Client(url + "/mcp", mode=mode) as c:
            r = await c.call_tool("ask_image", args)
            return "".join(b.text for b in r.content if b.type == "text"), r.is_error
    return anyio.run(go)


class TestEndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.store = Store()
        cls.store.add_image(IMG_ID, quad_image(2400, 1200))
        cls.store.add_pdf(PDF_ID)
        cls.vision = VisionStub(ANSWER)
        cls.cfg = server.Config(store=cls.store.root, vision_base_url=cls.vision.url, vision_model=MODEL,
                                vision_api_key=API_KEY, extra_body=EXTRA, allowed_hosts=["127.0.0.1:*", "localhost:*"])
        cls.srv = RunningServer(cls.cfg)

    @classmethod
    def tearDownClass(cls):
        cls.srv.close()
        cls.vision.close()
        cls.store.close()

    def setUp(self):
        self.vision.requests.clear()
        self.vision.status, self.vision.answer = 200, ANSWER

    def test_list_tools(self):
        async def go():
            async with Client(self.srv.url + "/mcp", mode="legacy") as c:
                return (await c.list_tools()).tools
        tools = anyio.run(go)
        self.assertEqual([t.name for t in tools], ["ask_image"])
        t = tools[0]
        self.assertEqual(t.description, server.TOOL_DESCRIPTION)
        for phrase in ("[image img_", "neutral", "Follow-up", "extracted text"):
            self.assertIn(phrase, t.description)
        self.assertEqual(list(t.input_schema["properties"]), ["image_id", "question", "context", "page", "region"])
        self.assertEqual(t.input_schema["required"], ["image_id", "question"])
        self.assertTrue(t.annotations.read_only_hint)

    def test_ask_image_roundtrip(self):
        text, is_err = call(self.srv.url, {"image_id": IMG_ID, "question": "What does the dialog title say?",
                                           "context": "User is debugging a backup script."})
        self.assertFalse(is_err)
        self.assertEqual(text, ANSWER)
        self.assertEqual(len(self.vision.requests), 1)
        req = self.vision.requests[0]
        self.assertEqual(req["path"], "/v1/chat/completions")
        self.assertEqual({k.lower(): v for k, v in req["headers"].items()}["authorization"], f"Bearer {API_KEY}")
        body = req["body"]
        self.assertEqual(body["model"], MODEL)
        self.assertEqual((body["chat_template_kwargs"], body["top_k"]), (EXTRA["chat_template_kwargs"], 20))  # merged
        self.assertEqual((body["temperature"], body["max_tokens"]), (0.2, 700))
        system, user = body["messages"]
        self.assertEqual(system["role"], "system")
        for phrase in ("[not visible]", "[illegible]", "exactly", "may be wrong"):
            self.assertIn(phrase, system["content"])
        img_part = next(p for p in user["content"] if p["type"] == "image_url")
        txt_part = next(p for p in user["content"] if p["type"] == "text")
        im = decode_png_data_url(img_part["image_url"]["url"])
        self.assertEqual(im.size, (1600, 800))  # 2400x1200 downscaled to long edge 1600
        self.assertEqual(im.getpixel((1200, 200)), QUAD["tr"])
        self.assertIn("What does the dialog title say?", txt_part["text"])
        self.assertIn("User is debugging a backup script.", txt_part["text"])
        self.assertIn(IMG_ID, txt_part["text"])

    def test_pdf_page_region_modern_protocol(self):
        text, is_err = call(self.srv.url, {"image_id": PDF_ID, "question": "What color is this page?", "page": 2,
                                           "region": "0,0,0.5,0.5"}, mode="auto")
        self.assertFalse(is_err)
        self.assertEqual(text, ANSWER)
        user = self.vision.requests[0]["body"]["messages"][1]
        im = decode_png_data_url(next(p for p in user["content"] if p["type"] == "image_url")["image_url"]["url"])
        self.assertEqual(im.size, (550, 1100))  # page 2 (360x720 pt) at 220 dpi = 1100x2200, top-left quarter
        r, g, b = im.getpixel((275, 550))
        self.assertTrue(b > 240 and r < 15 and g < 15, (r, g, b))
        txt = next(p for p in user["content"] if p["type"] == "text")["text"]
        self.assertIn("page 2", txt)
        self.assertIn("zoomed crop", txt)
        self.assertNotIn("Context", txt)  # empty context is omitted

    def test_error_strings(self):
        cases = [
            ({"image_id": "img_" + "9" * 16, "question": "q"}, "ask_image error: unknown image_id"),
            ({"image_id": "../store/meta/x", "question": "q"}, "ask_image error: invalid image_id"),
            ({"image_id": "url_" + "a" * 16, "question": "q"}, "remote image URL that was never downloaded"),
            ({"image_id": IMG_ID, "question": "  "}, "question is empty"),
            ({"image_id": IMG_ID, "question": "q", "region": "1,1,0,0"}, "region needs x0 < x1"),
            ({"image_id": IMG_ID, "question": "q", "page": 0}, "page must be >= 1"),
            ({"image_id": PDF_ID, "question": "q", "page": 7}, "out of range: this PDF has 2 pages"),
        ]
        for args, expect in cases:
            text, is_err = call(self.srv.url, args)
            self.assertFalse(is_err, args)
            self.assertIn(expect, text, args)
        self.assertEqual(self.vision.requests, [])  # none of these reached the vision model

    def test_vision_http_error(self):
        self.vision.status = 500
        text, _ = call(self.srv.url, {"image_id": IMG_ID, "question": "q"})
        self.assertTrue(text.startswith("ask_image error: the vision model returned HTTP 500"), text)

    def test_no_api_key_no_auth_header(self):
        srv = RunningServer(server.Config(store=self.store.root, vision_base_url=self.vision.url + "/v1/",
                                          vision_model=MODEL, allowed_hosts=["127.0.0.1:*"]))
        try:
            text, _ = call(srv.url, {"image_id": IMG_ID, "question": "q"})
        finally:
            srv.close()
        self.assertEqual(text, ANSWER)
        req = self.vision.requests[0]
        self.assertEqual(req["path"], "/v1/chat/completions")  # base URL with /v1/ works too
        self.assertNotIn("authorization", {k.lower() for k in req["headers"]})
        self.assertNotIn("chat_template_kwargs", req["body"])  # no extra body configured -> nothing extra sent

    def test_vision_unreachable(self):
        cfg = server.Config(store=self.store.root, vision_base_url=closed_port_url())
        srv = RunningServer(cfg)
        try:
            text, is_err = call(srv.url, {"image_id": IMG_ID, "question": "q"})
        finally:
            srv.close()
        self.assertFalse(is_err)
        self.assertTrue(text.startswith("ask_image error: the vision model is unreachable"), text)

    def test_health_and_host_guard(self):
        self.assertEqual(httpx2.get(self.srv.url + "/health").status_code, 200)
        init = {"jsonrpc": "2.0", "id": 1, "method": "initialize",
                "params": {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "t", "version": "0"}}}
        hdrs = {"Accept": "application/json, text/event-stream"}
        ok = httpx2.post(self.srv.url + "/mcp", json=init, headers=hdrs)
        self.assertEqual(ok.status_code, 200)
        self.assertEqual(ok.headers["content-type"], "application/json")  # stateless JSON responses
        bad = httpx2.post(self.srv.url + "/mcp", json=init, headers={**hdrs, "Host": "evil.example:8102"})
        self.assertGreaterEqual(bad.status_code, 400)  # DNS-rebinding guard rejects unknown Host headers

    def test_health_503_without_store(self):
        srv = RunningServer(server.Config(store=self.store.root / "missing"))
        try:
            self.assertEqual(httpx2.get(srv.url + "/health").status_code, 503)
        finally:
            srv.close()


if __name__ == "__main__":
    unittest.main()
