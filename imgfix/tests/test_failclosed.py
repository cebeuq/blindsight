"""Fail closed: strict JSON only, ambiguous keys refused, content keys in any letter case, the structural backstop,
Content-Encoding decoding, the non-JSON rule, echoed URLs (rewrite core; captions mocked; no network)."""
import gzip
import json
import zlib
from unittest import mock

import brotli
import zstandard

import imgfix
from . import helpers as h
from .test_rewrite import PLACEHOLDER, FakeCaptioner, StoreCase, iid


class FailClosedTest(StoreCase):
    def reject(self, obj, needle):
        with self.assertRaises(imgfix.Reject) as cm:
            self.run_rw(obj if isinstance(obj, bytes) else h.dumps(obj))
        self.assertIn(needle, str(cm.exception))
        self.assertIn("not forwarded", str(cm.exception))

    def out(self, obj, cap=None):
        res = self.run_rw(obj, cap)
        self.assertIsNotNone(res.body)
        out = json.loads(res.body)
        self.assertEqual(h.media_blocks(out), [], "media part left for the text model")
        return out

    def ref(self, raw):
        """The REF an Anthropic image block gets."""
        return h.texts(self.rewritten(h.body(h.user(h.image_block(raw))), FakeCaptioner())[1])[0]

    # ------------------------------------------------------------ parsing

    def test_not_strict_json(self):
        good = h.dumps(h.body(h.user(h.image_block(h.png(5, 5)))))
        for bad in (good.replace(b'"text-model"', b'"text-\xff-model"'),  # invalid UTF-8 (Go's encoding/json reads it)
                    good.replace(b'"text-model"', b'"text\tmodel"'),      # raw TAB inside a string
                    good[:-1] + b",}",                                    # trailing comma
                    b"// comment\n" + good):
            res = self.run_rw(bad)
            self.assertEqual((res.body, res.stats["not_json"]), (None, 1))  # the hook answers 400 (non-JSON rule)
        for depth in (5000, 200000):  # too deep to parse (not JSON -> 400), or to walk (Reject -> 400)
            try:
                res = self.run_rw(b'{"messages":' + b"[" * depth + b"]" * depth + b"}")
            except imgfix.Reject as e:
                self.assertIn("nested too deeply", str(e))
            else:
                self.assertEqual((res.body, res.stats["not_json"]), (None, 1))
        self.assertEqual(self.files(), [])

    def test_ambiguous_keys(self):
        img = h.image_block(h.png(5, 5))
        self.reject(b'{"model":"m","messages":[],"messages":[]}', "repeats the JSON key 'messages'")
        self.reject(b'{"model":"m","messages":[{"role":"user","content":[{"type":"text","text":"a","text":"b"}]}]}',
                    "repeats the JSON key 'text'")
        for obj, key in (({"model": "m", "Messages": [], "messages": [h.user(img)]}, "messages"),
                         ({"model": "m", "images": [], "IMAGES": [h.b64(h.png())]}, "images"),
                         ({"model": "m", "messages": [h.user({"type": "text", "Type": "image", "text": "x"})]}, "type"),
                         ({"model": "m", "messages": [{"role": "user", "content": "a", "Content": "b"}]}, "content"),
                         ({"model": "m", "prompt": "a", "PROMPT": "b"}, "prompt"),
                         ({"model": "m", "input": "a", "Input": []}, "input")):
            self.reject(obj, f"spells the JSON key '{key}' twice")
        # PASSTHROUGH_MODELS cannot be bypassed with a second spelling of "model" (checked before passthrough)
        with mock.patch.object(imgfix, "PASSTHROUGH_MODELS", ("passthru-*",)):
            self.reject({"model": "passthru-x", "Model": "text-model", "messages": [h.user(img)]}, "'model' twice")
        self.assertEqual(self.files(), [])

    # ------------------------------------------------------------ content keys in any case, Ollama variants

    def test_content_keys_any_case(self):
        raw = h.png(6, 7)
        ref = self.ref(raw)
        out = self.out({"model": "m", "Messages": [h.user(h.image_block(raw))]})
        self.assertEqual(out["Messages"][0]["content"], [{"type": "text", "text": ref}])
        out = self.out({"model": "m", "INPUT": [{"role": "user", "content": [{"type": "input_image",
                                                                              "image_url": h.data_url(raw)}]}]})
        self.assertEqual(out["INPUT"][0]["content"], [{"type": "input_text", "text": ref}])
        out = self.out({"model": "llava", "PROMPT": "p", "IMAGES": [h.b64(raw)]})
        self.assertEqual(out, {"model": "llava", "PROMPT": "p\n\n" + ref})
        out = self.out({"model": "llava", "Images": [h.b64(raw)]})
        self.assertEqual(out, {"model": "llava", "prompt": "\n\n" + ref})

    def test_ollama_variants(self):
        raw = h.png(8, 6)
        ref = self.ref(raw)
        msgs = [{"content": "no role", "images": [h.b64(raw)]},                        # Ollama message without role
                {"role": "user", "Content": "case", "Images": [None, h.b64(raw)]},       # null item, key cases
                {"role": "user", "images": [h.image_block(raw), {"x": 1}, {"image_url": h.data_url(raw)}]},
                {"role": "user", "content": [h.text("list")], "images": [h.data_url(raw)]}]
        out = self.out({"model": "llava", "messages": msgs})
        u = imgfix.UNAVAILABLE
        self.assertEqual(out["messages"], [{"content": f"no role\n\n{ref}"},
                                           {"role": "user", "Content": f"case\n\n{u}\n{ref}"},
                                           {"role": "user", "content": f"\n\n{ref}\n{u}\n{ref}"},
                                           {"role": "user", "content": [h.text("list"),
                                                                        {"type": "text", "text": ref}]}])

    def test_responses_prompt_variables_and_computer_screenshot(self):
        raw = h.png(9, 8)
        ref = self.ref(raw)
        body = {"model": "m", "prompt": {"id": "pmpt_1", "variables": {
                    "name": "x", "pic": {"type": "input_image", "image_url": h.data_url(raw)}}},
                "input": [{"type": "computer_call", "call_id": "c", "action": {"type": "screenshot"}},
                          {"type": "computer_call_output", "call_id": "c", "output": {
                              "type": "computer_screenshot", "image_url": h.data_url(raw)}},
                          {"role": "user", "content": [{"image_url": h.data_url(raw)}]}]}
        out = self.out(body)
        self.assertEqual(out["prompt"]["variables"], {"name": "x", "pic": {"type": "input_text", "text": ref}})
        self.assertEqual(out["input"][1]["output"], {"type": "input_text", "text": ref})
        self.assertEqual(out["input"][2]["content"], [{"type": "input_text", "text": ref}])
        self.assertEqual(self.run_rw(body).body, self.run_rw(body).body)

    def test_untyped_image_url_chat(self):
        raw = h.png(10, 4)
        ref = self.ref(raw)
        body = {"model": "m", "messages": [{"role": "user", "content": [
            {"image_url": {"url": h.data_url(raw)}}, {"image_url": h.data_url(raw)}, {"image_url": "ftp://x/a.png"}]}]}
        out = self.out(body)
        self.assertEqual(out["messages"][0]["content"], [{"type": "text", "text": ref}] * 2
                         + [{"type": "text", "text": imgfix.UNAVAILABLE}])
        self.assertEqual(self.run_rw(body).body, self.run_rw(body).body)

    # ------------------------------------------------------------ the structural backstop

    def test_backstop(self):
        png = h.png(5, 5)
        src = h.image_block(png)["source"]
        cases = [({"model": "m", "messages": [h.user({"type": "IMAGE", "source": src})]}, 'type "image"'),
                 ({"model": "m", "messages": [h.user({"type": "Image_URL", "image_url": h.data_url(png)})]},
                  'type "image_url"'),
                 ({"model": "m", "system": [{"type": "image", "source": src}]}, 'type "image"'),
                 ({"model": "m", "attachments": [{"Type": "input_image", "image_url": h.data_url(png)}]},
                  'type "input_image"'),
                 ({"model": "m", "messages": [h.user({"type": "text", "text": "x", "images": [h.b64(png)]})]},
                  '"images" list'),
                 ({"model": "m", "extra": {"source": {"type": "base64", "media_type": "image/png", "data": "x"}}},
                  '"source"'),
                 ({"model": "m", "extra": {"ImageUrl": {"url": h.data_url(png)}}}, 'untyped "image_url"'),
                 ({"model": "m", "extra": {"inlineData": {"mimeType": "image/png", "data": h.b64(png)}}}, "Gemini"),
                 ({"model": "m", "messages": [h.user({"type": "image_embeds", "image_embeds": "AAAA"})]},
                  'type "image_embeds"'),
                 ([{"type": "image", "source": src}], 'type "image"')]   # a JSON array body
        for obj, needle in cases:
            self.reject(obj, needle)
        self.assertEqual(self.files(), [])

    def test_backstop_ignores_configuration_and_model_written_json(self):
        schema = {"type": "object", "properties": {"image_url": {"type": "string"}, "source": {"type": "string"},
                                                   "images": {"type": "array"}},
                  "examples": [{"images": ["a.png"], "type": "image", "inlineData": {}}]}
        body = {"model": "m", "tools": [{"name": "look", "description": "d", "input_schema": schema}],
                "functions": [{"name": "f", "parameters": schema}], "tool_choice": {"type": "auto"},
                "response_format": {"type": "json_schema", "json_schema": {"schema": schema}},
                "text": {"format": {"type": "json_schema", "schema": schema}}, "format": schema,
                "messages": [
                    {"role": "assistant", "content": [{"type": "tool_use", "id": "t", "name": "look",
                                                       "input": {"type": "image", "images": ["x"], "inlineData": {}}}]},
                    {"role": "assistant", "content": None, "function_call": {"name": "f", "arguments": "{}"},
                     "tool_calls": [{"type": "function", "function": {"name": "f", "arguments": {
                         "image_url": "x", "images": ["y"], "source": {"type": "url", "url": "u"}}}}]},
                    h.user(h.text("hi"), {"type": "search_result", "source": "https://e.com", "title": "t",
                                          "content": [h.text("x")]})]}
        res = self.run_rw(body)
        self.assertIsNone(res.body)  # nothing to replace, nothing left: forwarded byte for byte

    def test_passthrough_exempt_from_backstop(self):
        with mock.patch.object(imgfix, "PASSTHROUGH_MODELS", ("claude-*",)):
            res = self.run_rw({"model": "claude-x", "system": [h.image_block(h.png())], "messages": []})
        self.assertEqual((res.body, res.stats["passthrough"]), (None, 1))

    # ------------------------------------------------------------ Content-Encoding

    def test_decode_body(self):
        body = h.dumps(h.body(h.user(h.text("hi"))))
        c = zlib.compressobj(wbits=-15)
        raw_deflate = c.compress(body) + c.flush()
        gz = gzip.compress(body)
        for enc, data in (("gzip", gz), ("x-gzip", gz), (" GZIP ", gz),
                          ("deflate", zlib.compress(body)), ("deflate", raw_deflate), ("br", brotli.compress(body)),
                          ("zstd", zstandard.ZstdCompressor().compress(body)), ("identity", body), ("", body),
                          (None, body)):
            self.assertEqual(imgfix.decode_body(data, enc), body, enc)
        bomb = b"\0" * (3 << 20)
        with mock.patch.object(imgfix, "MAX_BODY", 1 << 20):
            for enc, data in (("gzip", gzip.compress(bomb)), ("deflate", zlib.compress(bomb)),
                              ("br", brotli.compress(bomb)), ("zstd", zstandard.ZstdCompressor().compress(bomb))):
                with self.assertRaisesRegex(imgfix.Reject, "more than", msg=enc):
                    imgfix.decode_body(data, enc)
        for enc, data in (("gzip", b"not gzip"), ("gzip", gzip.compress(body)[:-12]),
                          ("gzip", gzip.compress(body) * 2), ("deflate", b"junk"), ("br", b"\xff\xff junk"),
                          ("br", brotli.compress(body * 50)[:-4]), ("zstd", b"junk"), ("compress", body),
                          ("gzip, br", body), ("utf-8", body)):
            with self.assertRaisesRegex(imgfix.Reject, "Content-Encoding", msg=enc):
                imgfix.decode_body(data, enc)

    def test_non_json_rule(self):
        f = imgfix.non_json_forwardable
        audio = (b"--b\r\nContent-Disposition: form-data; name=\"file\"; filename=\"a.mp3\"\r\n"
                 b"Content-Type: audio/mpeg\r\n\r\nID3\x03\r\n--b--\r\n")
        self.assertTrue(f("multipart/form-data; boundary=b", audio))
        self.assertTrue(f("Application/Octet-Stream", b"\x00\x01"))
        self.assertTrue(f("audio/wav", b"RIFF"))
        self.assertFalse(f("multipart/form-data; boundary=b", audio.replace(b"audio/mpeg", b"image/png")))
        for ctype in ("application/json", "text/plain", "application/x-www-form-urlencoded", None, "", "image/png",
                      "application/pdf"):
            self.assertFalse(f(ctype, b"x"), ctype)
        for body in (b' \n{"model":"m"', b"[1,", b"\xef\xbb\xbf{"):
            self.assertFalse(f("multipart/form-data", body), body)
        for marker in (b"Document", b"input_file", b"FILE_ID", b"file_data", b"data:", b"screenshot", b"inline_data",
                       b"IMAGE", b"Base64"):
            self.assertFalse(f("video/mp4", b"xx" + marker + b"xx"), marker)

    # ------------------------------------------------------------ echoed URLs and ids; stats

    def test_echoed_urls_and_data_urls(self):
        long = "https://example.com/" + "a" * 300
        self.assertEqual(h.texts(self.rewritten(h.body(h.user({"type": "image", "source": {"type": "url",
                                                                                           "url": long}})))[1]),
                         [f"[image {imgfix.url_id(long)} | remote image not fetched: {long[:200]}...]"])
        raw, pdf = h.png(5, 6), h.text_pdf(["data url pdf"])
        res, out = self.rewritten(h.body(h.user(
            {"type": "image", "source": {"type": "url", "url": h.data_url(raw)}},
            {"type": "image", "source": {"type": "url", "url": "data:image/png;base64,!!"}},
            {"type": "document", "source": {"type": "url", "url": h.data_url(pdf, "application/pdf")}},
            {"type": "image", "source": {"type": "url", "url": None}})))
        ts = h.texts(out)
        self.assertEqual(PLACEHOLDER.match(ts[0]).group(1), iid(raw))  # decoded like base64 data
        self.assertEqual(ts[1], imgfix.UNAVAILABLE)
        self.assertTrue(ts[2].startswith(f"[pdf {iid(pdf, 'pdf_')} | 1 pages]"))
        self.assertEqual(ts[3], imgfix.UNAVAILABLE)
        self.assertNotIn(b"data:", res.body)
        self.assertEqual(imgfix.file_id_text("file", "f" * 300),
                         f"[file unavailable: file_id {'f' * 200}... not readable]")
        d = "data:image/png;base64,AAAA"
        self.assertEqual(imgfix.url_text("image", d),
                         f"[image {imgfix.url_id(d)} | remote image not fetched: (data URL)]")

    def test_new_pdfs_stat(self):
        req = h.body(h.user(h.pdf_block(h.text_pdf(["first sight"]))))
        self.assertEqual([self.run_rw(req).stats["new_pdfs"] for _ in range(2)], [1, 0])
