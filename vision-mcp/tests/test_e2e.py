"""End-to-end: the official MCP SDK client talks streamable HTTP to the real server (uvicorn), which calls a
recording vision-model stub (OpenAI or Anthropic format). Checks what the text model would see (tool listing, answer
text, error strings), what the vision model receives, and the /ask HTTP endpoint (same code path, plain HTTP)."""

import base64
import hashlib
import io
import json
import threading
import time
import unittest

import anyio
import httpx2
from mcp import Client
from PIL import Image

import server
from tests.util import (IMG_ID, PDF_ID, QUAD, RunningServer, Store, VisionStub, closed_port_url, decode_png_data_url,
                        quad_image)
from tests.util import vision as vision_cfg

ANSWER = 'The dialog title is "Disk full" and the button says "OK".'
MODEL = "test-vision-model"
API_KEY = "sk-vision-mcp-test-key"
EXTRA = {"chat_template_kwargs": {"enable_thinking": False}, "top_k": 20}
CUT = "\n[answer cut off at the length limit; ask a narrower follow-up question for the rest]"
# sha256 of the compact JSON of tools/list's "tools" (key order as sent). The tool description and parameter schemas
# are part of the text model's prompt: any change invalidates its prefix cache in every running session, so this pins
# them byte for byte (recorded from the pre-/ask server with the locked SDK; an SDK upgrade may need a new value).
TOOLS_SHA256 = "673ed9ffbeeb68fc4a9a14202a28b85d7c4a08031dc9084750115af0072021f8"


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
        cls.cfg = server.Config(store=cls.store.root, allowed_hosts=["127.0.0.1:*", "localhost:*"],
                                vision=vision_cfg(cls.vision.url, VISION_API_KEY=API_KEY, VISION_EXTRA_BODY=json.dumps(EXTRA)))
        cls.srv = RunningServer(cls.cfg)

    @classmethod
    def tearDownClass(cls):
        cls.srv.close()
        cls.vision.close()
        cls.store.close()

    def setUp(self):
        self.vision.reset(ANSWER)

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

    def test_tool_listing_byte_stable(self):
        hdrs = {"Accept": "application/json, text/event-stream"}
        r = httpx2.post(self.srv.url + "/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}},
                        headers=hdrs)
        tools = json.dumps(r.json()["result"]["tools"], separators=(",", ":"), ensure_ascii=False)
        self.assertEqual(hashlib.sha256(tools.encode()).hexdigest(), TOOLS_SHA256, tools)
        init = {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
            "protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "t", "version": "0"}}}
        result = httpx2.post(self.srv.url + "/mcp", json=init, headers=hdrs).json()["result"]
        self.assertEqual(result["instructions"], "Images and PDFs in this conversation are replaced by [image img_...] / "
                                                 "[pdf pdf_...] references. Use ask_image to look at them.")
        self.assertEqual(result["serverInfo"], {"name": "vision-mcp", "version": "2.0.0"})  # not part of TOOLS_SHA256

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
        self.assertEqual(system, {"role": "system", "content": server.SYSTEM_PROMPT})
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

    def test_vision_error_echoing_the_key_is_redacted(self):
        self.vision.status, self.vision.error_text = 401, f"Incorrect API key provided: {API_KEY}."
        with self.assertLogs("vision-mcp", "WARNING") as logs:
            text, _ = call(self.srv.url, {"image_id": IMG_ID, "question": "q"})
        self.assertIn("returned HTTP 401", text)
        self.assertIn("Incorrect API key provided: [redacted].", text)
        self.assertNotIn(API_KEY, text + "\n".join(logs.output))
        self.assertNotIn("Incorrect API key", "\n".join(logs.output))  # reply text is never logged

    def test_truncated_answer_noted(self):
        self.vision.stop = "length"
        text, _ = call(self.srv.url, {"image_id": IMG_ID, "question": "q"})
        self.assertEqual(text, ANSWER + CUT)

    def test_unexpected_reply_shape(self):
        self.vision.api = "anthropic"  # an Anthropic-shaped reply (holding the answer) while VISION_API is openai
        with self.assertLogs("vision-mcp", "WARNING") as logs:
            text, _ = call(self.srv.url, {"image_id": IMG_ID, "question": "q"})
        self.assertTrue(text.startswith("ask_image error: the vision model returned an unexpected response: {"), text)
        self.assertIn("Disk full", text)  # the caller sees the reply excerpt ...
        out = "\n".join(logs.output)
        self.assertNotIn("Disk full", out)  # ... the log only its shape
        self.assertIn("unexpected response (unexpected openai reply shape (KeyError); ", out)

    def test_reply_text_never_logged(self):
        question = "SECRET-QUESTION: what does the dialog say?"
        self.vision.status, self.vision.error_text = 400, f"invalid request: messages[1].content[1].text={question}"
        with self.assertLogs("vision-mcp", "WARNING") as logs:
            text, _ = call(self.srv.url, {"image_id": IMG_ID, "question": question})
        self.assertIn(question, text)  # a server echoing the input in its 400: shown to the model
        out = "\n".join(logs.output)
        self.assertNotIn("SECRET-QUESTION", out)  # but not logged
        self.assertRegex(out, r"returned HTTP 400 \(\d+ bytes\)")
        self.vision.reset("not json at all")
        self.vision.api = "raw"  # the stub sends its answer as the raw body
        with self.assertLogs("vision-mcp", "WARNING") as logs:
            text, _ = call(self.srv.url, {"image_id": IMG_ID, "question": "q"})
        self.assertIn("unexpected response: not json at all", text)
        self.assertNotIn("not json at all", "\n".join(logs.output))
        self.assertIn("unexpected response (not JSON; 15 bytes)", "\n".join(logs.output))

    def test_no_api_key_no_auth_header(self):
        srv = RunningServer(server.Config(store=self.store.root, vision=vision_cfg(self.vision.url + "/v1/"),
                                          allowed_hosts=["127.0.0.1:*"]))
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
        base = closed_port_url().replace("http://", "http://user:pw-secret@") + "/v1?api-version=1"
        srv = RunningServer(server.Config(store=self.store.root, vision=vision_cfg(base)))
        try:
            text, is_err = call(srv.url, {"image_id": IMG_ID, "question": "q"})
        finally:
            srv.close()
        self.assertFalse(is_err)
        self.assertTrue(text.startswith("ask_image error: the vision model is unreachable at http://127.0.0.1:"), text)
        self.assertNotIn("pw-secret", text)  # no user:password@ or query string in the message
        self.assertNotIn("api-version", text)

    def test_headers_token_param_null_deletion(self):
        """Azure-style: full URL with ?api-version, key in a custom header, max_completion_tokens, temperature removed."""
        path = "/openai/deployments/d1/chat/completions?api-version=2024-10-21"
        srv = RunningServer(server.Config(store=self.store.root, allowed_hosts=["127.0.0.1:*"], vision=vision_cfg(
            self.vision.url + path, VISION_HEADERS='{"api-key": "az-key-0123456789", "X-Title": "blindsight"}',
            VISION_TOKEN_PARAM="max_completion_tokens", VISION_EXTRA_BODY='{"temperature": null, "top_k": 5}')))
        try:
            text, _ = call(srv.url, {"image_id": IMG_ID, "question": "q"})
        finally:
            srv.close()
        self.assertEqual(text, ANSWER)
        req = self.vision.requests[0]
        self.assertEqual(req["path"], path)
        hdrs = {k.lower(): v for k, v in req["headers"].items()}
        self.assertEqual((hdrs["api-key"], hdrs["x-title"]), ("az-key-0123456789", "blindsight"))
        self.assertNotIn("authorization", hdrs)
        body = req["body"]
        self.assertEqual((body["max_completion_tokens"], body["top_k"]), (700, 5))
        self.assertNotIn("max_tokens", body)
        self.assertNotIn("temperature", body)  # null in VISION_EXTRA_BODY deletes the key

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



class TestAnthropic(unittest.TestCase):
    """VISION_API=anthropic: Messages API request (x-api-key, anthropic-version, top-level system, base64 image block)
    and reply (text blocks only, stop_reason max_tokens = cut off)."""

    @classmethod
    def setUpClass(cls):
        cls.store = Store()
        cls.store.add_image(IMG_ID, quad_image(2400, 1200))
        cls.vision = VisionStub(ANSWER, api="anthropic")
        # an Anthropic-style base URL (like ANTHROPIC_BASE_URL): /v1/messages is appended
        cls.srv = RunningServer(server.Config(store=cls.store.root, allowed_hosts=["127.0.0.1:*"], vision=vision_cfg(
            cls.vision.url + "/anthropic", VISION_API="anthropic", VISION_API_KEY=API_KEY, VISION_MODEL="claude-test")))

    @classmethod
    def tearDownClass(cls):
        cls.srv.close()
        cls.vision.close()
        cls.store.close()

    def setUp(self):
        self.vision.reset(ANSWER, api="anthropic")

    def test_request_and_answer(self):
        text, is_err = call(self.srv.url, {"image_id": IMG_ID, "question": "What does the title say?",
                                           "context": "debugging"})
        self.assertFalse(is_err)
        self.assertEqual(text, ANSWER)  # the stub's thinking block is not part of the answer
        req = self.vision.requests[0]
        self.assertEqual(req["path"], "/anthropic/v1/messages")
        hdrs = {k.lower(): v for k, v in req["headers"].items()}
        self.assertEqual((hdrs["x-api-key"], hdrs["anthropic-version"], hdrs["content-type"]),
                         (API_KEY, "2023-06-01", "application/json"))
        self.assertNotIn("authorization", hdrs)
        body = req["body"]
        self.assertEqual(sorted(body), ["max_tokens", "messages", "model", "system", "temperature"])
        self.assertEqual((body["model"], body["max_tokens"], body["temperature"], body["system"]),
                         ("claude-test", 700, 0.2, server.SYSTEM_PROMPT))
        (msg,) = body["messages"]
        self.assertEqual(msg["role"], "user")
        img, txt = msg["content"]
        self.assertEqual((img["type"], img["source"]["type"], img["source"]["media_type"]), ("image", "base64", "image/png"))
        im = Image.open(io.BytesIO(base64.b64decode(img["source"]["data"])))
        self.assertEqual((im.format, im.size), ("PNG", (1600, 800)))
        self.assertEqual(im.convert("RGB").getpixel((1200, 200)), QUAD["tr"])
        self.assertEqual(txt["type"], "text")
        self.assertIn("What does the title say?", txt["text"])
        self.assertIn("debugging", txt["text"])

    def test_max_tokens_is_cut_off(self):
        self.vision.stop = "max_tokens"
        text, _ = call(self.srv.url, {"image_id": IMG_ID, "question": "q"})
        self.assertEqual(text, ANSWER + CUT)

    def test_empty_and_error(self):
        self.vision.answer = "   "
        text, _ = call(self.srv.url, {"image_id": IMG_ID, "question": "q"})
        self.assertEqual(text, "ask_image error: the vision model returned an empty answer; try rephrasing the question")
        self.vision.status = 529
        text, _ = call(self.srv.url, {"image_id": IMG_ID, "question": "q"})
        self.assertTrue(text.startswith("ask_image error: the vision model returned HTTP 529"), text)


def multipart(fields: dict, file_field: str | None = None) -> tuple[bytes, dict]:
    b = "bs-test-boundary"
    out = "".join(f'--{b}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n' for k, v in fields.items())
    if file_field:
        out += (f'--{b}\r\nContent-Disposition: form-data; name="{file_field}"; filename="x.png"\r\n'
                "Content-Type: image/png\r\n\r\nPNGDATA\r\n")
    return (out + f"--{b}--\r\n").encode(), {"Content-Type": f"multipart/form-data; boundary={b}"}


class TestAskHttp(unittest.TestCase):
    """GET/POST /ask: same answers, errors, limits and guard as the MCP tool, as plain text."""

    @classmethod
    def setUpClass(cls):
        cls.store = Store()
        cls.store.add_image(IMG_ID, quad_image(2400, 1200))
        cls.store.add_pdf(PDF_ID)
        cls.vision = VisionStub(ANSWER)
        cls.srv = RunningServer(server.Config(store=cls.store.root, allowed_hosts=["127.0.0.1:*", "localhost:*"],
                                              vision=vision_cfg(cls.vision.url, VISION_API_KEY=API_KEY)))
        cls.ask = cls.srv.url + "/ask"

    @classmethod
    def tearDownClass(cls):
        cls.srv.close()
        cls.vision.close()
        cls.store.close()

    def setUp(self):
        self.vision.reset(ANSWER)

    def ok(self, r: httpx2.Response) -> None:
        self.assertEqual((r.status_code, r.text), (200, ANSWER + "\n"))
        self.assertTrue(r.headers["content-type"].startswith("text/plain"), r.headers["content-type"])

    def prompt(self, n: int = 0) -> str:
        return self.vision.requests[n]["body"]["messages"][1]["content"][1]["text"]

    def test_get(self):
        self.ok(httpx2.get(self.ask, params={"image_id": IMG_ID, "question": "What is the title?", "context": "ctx 1"}))
        self.assertEqual(len(self.vision.requests), 1)
        self.assertIn("What is the title?", self.prompt())
        self.assertIn("ctx 1", self.prompt())
        im = decode_png_data_url(self.vision.requests[0]["body"]["messages"][1]["content"][0]["image_url"]["url"])
        self.assertEqual(im.size, (1600, 800))

    def test_get_pdf_page_region(self):
        self.ok(httpx2.get(self.ask, params={"image_id": PDF_ID, "question": "q", "page": "2", "region": "0,0,0.5,0.5",
                                            "context": ""}))
        self.assertIn(f"PDF {PDF_ID}, page 2 - zoomed crop", self.prompt())
        self.assertNotIn("Context", self.prompt())

    def test_post_json(self):
        self.ok(httpx2.post(self.ask, json={"image_id": PDF_ID, "question": "Which color?", "page": 2, "region": None,
                                            "context": None}))
        self.assertIn("page 2.", self.prompt())
        self.assertIn("Which color?", self.prompt())

    def test_post_form(self):
        self.ok(httpx2.post(self.ask, data={"image_id": IMG_ID, "question": "Form question?", "page": ""}))
        self.assertIn("Form question?", self.prompt())
        body, hdrs = multipart({"image_id": IMG_ID, "question": "Multipart question?", "page": "1"})
        self.ok(httpx2.post(self.ask, content=body, headers=hdrs))
        self.assertIn("Multipart question?", self.prompt(1))

    def test_validation_errors(self):
        big = {"image_id": IMG_ID, "question": "x" * (server.ASK_MAX_BODY + 1)}
        mp_file = multipart({"question": "q"}, file_field="image_id")
        cases = [
            ("GET", {"params": {"question": "q"}}, "image_id is required"),
            ("GET", {"params": {"image_id": "img_123", "question": "q"}}, "invalid image_id 'img_123'"),
            ("GET", {"params": {"image_id": "url_" + "a" * 16, "question": "q"}}, "never downloaded"),
            ("GET", {"params": {"image_id": "img_" + "9" * 16, "question": "q"}}, "unknown image_id"),
            ("GET", {"params": {"image_id": IMG_ID}}, "question is empty"),
            ("GET", {"params": {"image_id": IMG_ID, "question": " "}}, "question is empty"),
            ("GET", {"params": {"image_id": IMG_ID, "question": "q", "page": "two"}}, "page must be a whole number"),
            ("GET", {"params": {"image_id": IMG_ID, "question": "q", "page": "1.5"}}, "page must be a whole number"),
            ("GET", {"params": {"image_id": PDF_ID, "question": "q", "page": "0"}}, "page must be >= 1"),
            ("GET", {"params": {"image_id": PDF_ID, "question": "q", "page": "9"}}, "out of range: this PDF has 2 pages"),
            ("GET", {"params": {"image_id": IMG_ID, "question": "q", "page": "2"}}, "page applies only to pdf_ ids"),
            ("GET", {"params": {"image_id": IMG_ID, "question": "q", "region": "1,1,0,0"}}, "region needs x0 < x1"),
            ("GET", {"params": {"image_id": IMG_ID, "q": "q"}}, "unknown parameter 'q'"),
            ("GET", {"params": [("image_id", IMG_ID), ("image_id", IMG_ID), ("question", "q")]}, "more than once"),
            ("POST", {"content": b"{bad", "headers": {"Content-Type": "application/json"}}, "not valid JSON"),
            ("POST", {"json": [IMG_ID, "q"]}, "the JSON body must be an object"),
            ("POST", {"json": {"image_id": IMG_ID, "question": 5}}, "question must be a string"),
            ("POST", {"json": {"image_id": 7, "question": "q"}}, "image_id must be a string"),
            ("POST", {"json": {"image_id": IMG_ID, "question": "q", "page": True}}, "page must be a whole number"),
            ("POST", {"json": {"image_id": IMG_ID, "question": "q", "page": 1.5}}, "page must be a whole number"),
            ("POST", {"json": {"image_id": IMG_ID, "question": "q", "extra": 1}}, "unknown parameter 'extra'"),
            ("POST", {"content": b"image_id=x", "headers": {"Content-Type": "text/plain"}}, "POST needs Content-Type"),
            ("POST", {"content": b"", "headers": {}}, "POST needs Content-Type"),
            ("POST", {"content": mp_file[0], "headers": mp_file[1]}, "could not read the form body"),
            ("POST", {"json": big}, f"larger than {server.ASK_MAX_BODY} bytes"),
            ("POST", {"content": iter([b'{"image_id": "', b"x" * server.ASK_MAX_BODY, b'"}']),  # chunked, no length
                      "headers": {"Content-Type": "application/json"}}, f"larger than {server.ASK_MAX_BODY} bytes"),
        ]
        for method, kw, expect in cases:
            r = httpx2.request(method, self.ask, **kw)
            self.assertEqual(r.status_code, 400, (method, kw.get("params") or kw.get("json"), r.text[:200]))
            self.assertTrue(r.text.startswith("ask_image error: "), r.text)
            self.assertIn(expect, r.text, (method, kw.get("params") or kw.get("json")))
            self.assertTrue(r.headers["content-type"].startswith("text/plain"))
        self.assertEqual(self.vision.requests, [])  # none of these reached the vision model

    def test_vision_failure_is_400(self):
        self.vision.status = 500
        r = httpx2.get(self.ask, params={"image_id": IMG_ID, "question": "q"})
        self.assertEqual(r.status_code, 400)
        self.assertTrue(r.text.startswith("ask_image error: the vision model returned HTTP 500"), r.text)
        self.vision.reset(ANSWER)
        self.vision.stop = "length"
        r = httpx2.get(self.ask, params={"image_id": IMG_ID, "question": "q"})
        self.assertEqual((r.status_code, r.text), (200, ANSWER + CUT + "\n"))

    def test_same_text_as_the_tool(self):
        for args in ({"image_id": IMG_ID, "question": "q", "region": "0,0,0.5,0.5"},
                     {"image_id": "img_" + "9" * 16, "question": "q"}, {"image_id": IMG_ID, "question": "q", "page": 3}):
            tool_text, _ = call(self.srv.url, args)
            r = httpx2.post(self.ask, json=args)
            self.assertEqual(r.text, tool_text + "\n", args)
            self.assertEqual(r.status_code, 400 if tool_text.startswith("ask_image error: ") else 200)

    def test_host_and_origin_guard(self):
        q = {"image_id": IMG_ID, "question": "q"}
        r = httpx2.get(self.ask, params=q, headers={"Host": "evil.example:8102"})  # DNS rebinding
        self.assertEqual((r.status_code, r.text), (421, "Invalid Host header\n"))
        r = httpx2.post(self.ask, json=q, headers={"Host": "evil.example"})
        self.assertEqual(r.status_code, 421)
        for origin in ("http://evil.example", "https://127.0.0.1:8102", "null", "http://127.0.0.1"):
            r = httpx2.post(self.ask, data=q, headers={"Origin": origin})  # a web page posting a form
            self.assertEqual((r.status_code, r.text), (403, "Invalid Origin header\n"), origin)
        # /mcp allows any allowed host:port as Origin, /ask only itself: a page on another local port is another site
        for origin in ("http://localhost:3000", f"http://localhost:{self.srv.port}", "http://127.0.0.1:1"):
            r = httpx2.get(self.ask, params=q, headers={"Origin": origin})
            self.assertEqual((r.status_code, r.text), (403, "Invalid Origin header\n"), origin)
        self.assertEqual(self.vision.requests, [])
        self.ok(httpx2.get(self.ask, params=q, headers={"Origin": f"http://127.0.0.1:{self.srv.port}"}))
        self.ok(httpx2.get(self.ask, params=q, headers={"Host": f"localhost:{self.srv.port}",
                                                         "Origin": f"http://localhost:{self.srv.port}"}))
        self.ok(httpx2.get(self.ask, params=q, headers={"Host": f"localhost:{self.srv.port}"}))

    def test_cross_site_browser_requests_refused(self):
        """A web page can make a browser GET /ask (e.g. <img src>) with no Origin; Sec-Fetch-Site gives it away."""
        q = {"image_id": IMG_ID, "question": "q"}
        for site in ("cross-site", "same-site", "Cross-Site", "bogus"):
            for r in (httpx2.get(self.ask, params=q, headers={"Sec-Fetch-Site": site}),
                      httpx2.post(self.ask, data=q, headers={"Sec-Fetch-Site": site}),
                      httpx2.post(self.ask, json=q, headers={"Sec-Fetch-Site": site,
                                                             "Origin": f"http://127.0.0.1:{self.srv.port}"})):
                self.assertEqual((r.status_code, r.text), (403, "Cross-site request refused\n"), site)
        self.assertEqual(self.vision.requests, [])
        for site in ("none", "same-origin"):  # typed into the address bar / the server's own page
            self.ok(httpx2.get(self.ask, params=q, headers={"Sec-Fetch-Site": site}))
        init = {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
            "protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "t", "version": "0"}}}
        r = httpx2.post(self.srv.url + "/mcp", json=init, headers={"Accept": "application/json, text/event-stream",
                                                                    "Sec-Fetch-Site": "cross-site"})
        self.assertEqual(r.status_code, 403)  # /mcp too (MCP clients send no Sec-Fetch-*: covered by every call())

    def test_guard_off_with_star(self):
        srv = RunningServer(server.Config(store=self.store.root, allowed_hosts=[], vision=vision_cfg(self.vision.url)))
        try:
            r = httpx2.get(srv.url + "/ask", params={"image_id": IMG_ID, "question": "q"},
                           headers={"Host": "anything.example", "Origin": "http://elsewhere.example"})
            cross = httpx2.get(srv.url + "/ask", params={"image_id": IMG_ID, "question": "q"},
                               headers={"Host": "anything.example", "Sec-Fetch-Site": "cross-site"})
        finally:
            srv.close()
        self.ok(r)
        self.assertEqual(cross.status_code, 403)  # "*" turns off the Host allowlist, not the cross-site refusal

    def test_methods(self):
        q = {"image_id": IMG_ID, "question": "q"}
        self.assertEqual(httpx2.head(self.ask, params=q).status_code, 405)  # never costs a vision call
        self.assertEqual(httpx2.put(self.ask, json=q).status_code, 405)
        self.assertEqual(self.vision.requests, [])

    def test_logs_never_contain_question_answer_or_key(self):
        question, context = "SECRET-QUESTION-TEXT?", "SECRET-CONTEXT-TEXT"
        with self.assertLogs("vision-mcp", "DEBUG") as logs:
            self.ok(httpx2.get(self.ask, params={"image_id": IMG_ID, "question": question, "context": context}))
            self.ok(httpx2.post(self.ask, json={"image_id": IMG_ID, "question": question}))
            call(self.srv.url, {"image_id": IMG_ID, "question": question, "context": context})
            httpx2.get(self.ask, params={"image_id": IMG_ID, "question": question, "page": "x"})
        out = "\n".join(logs.output)
        self.assertIn("via http", out)
        self.assertIn("via mcp", out)
        for secret in (question, context, ANSWER, "Disk full", API_KEY):
            self.assertNotIn(secret, out)

    def test_access_log_hides_query(self):
        with self.assertLogs("uvicorn.access", "INFO") as logs:  # (raises the logger to INFO for the duration)
            self.ok(httpx2.get(self.ask, params={"image_id": IMG_ID, "question": "SECRET-Q"}))
        out = "\n".join(logs.output)
        self.assertIn('"GET /ask?... HTTP/1.1" 200', out)
        self.assertNotIn("SECRET-Q", out)
        self.assertNotIn(IMG_ID, out)


class TestConcurrency(unittest.TestCase):
    """ASK_CONCURRENCY bounds MCP tool calls and /ask requests together (one semaphore, one code path)."""

    def setUp(self):
        self.store = Store()
        self.store.add_image(IMG_ID, quad_image(64, 64))
        self.vision = VisionStub(ANSWER)
        self.vision.delay = 0.25

    def tearDown(self):
        self.vision.close()
        self.store.close()

    def run_mixed(self, concurrency: int) -> tuple[list[str], float]:
        srv = RunningServer(server.Config(store=self.store.root, allowed_hosts=["127.0.0.1:*"], concurrency=concurrency,
                                          vision=vision_cfg(self.vision.url)))
        results: list[str] = []
        args = {"image_id": IMG_ID, "question": "q"}
        jobs = [lambda: call(srv.url, args)[0], lambda: call(srv.url, args)[0],
                lambda: httpx2.get(srv.url + "/ask", params=args, timeout=30).text.rstrip("\n"),
                lambda: httpx2.post(srv.url + "/ask", json=args, timeout=30).text.rstrip("\n")]
        threads = [threading.Thread(target=lambda j=j: results.append(j())) for j in jobs]
        t0 = time.monotonic()
        try:
            for t in threads:
                t.start()
            for t in threads:
                t.join(30)
        finally:
            srv.close()
        return results, time.monotonic() - t0

    def test_limit_shared_by_tool_and_http(self):
        results, elapsed = self.run_mixed(1)
        self.assertEqual(results, [ANSWER] * 4)
        self.assertEqual(self.vision.max_inflight, 1)  # one at a time, whichever way the call came in
        self.assertGreaterEqual(elapsed, 4 * self.vision.delay - 0.05)
        self.vision.reset(ANSWER)
        self.vision.delay = 0.25
        results, _ = self.run_mixed(2)
        self.assertEqual(results, [ANSWER] * 4)
        self.assertEqual(self.vision.max_inflight, 2)  # the limit, not serialisation elsewhere


if __name__ == "__main__":
    unittest.main()
