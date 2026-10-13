"""Gemini native format (generateContent / streamGenerateContent / countTokens): inline_data / inlineData,
file_data / fileData, systemInstruction, generateContentRequest, passthrough by the path's model (rewrite core)."""
import asyncio
import base64
import json
from unittest import mock

import imgfix
from . import helpers as h
from .test_rewrite import FakeCaptioner, StoreCase, iid

GEN = "/v1beta/models/gemini-text:generateContent"
COUNT = "/v1beta/models/gemini-text:countTokens"


def contents(*parts):
    return [{"role": "user", "parts": list(parts)}]


def inline(raw, mime="image/png", snake=False, data=None):
    data = h.b64(raw) if data is None else data
    return {"inline_data": {"mime_type": mime, "data": data}} if snake else {"inlineData": {"mimeType": mime,
                                                                                           "data": data}}


def file_data(uri, mime=None, snake=False):
    d = {"file_uri" if snake else "fileUri": uri}
    if mime:
        d["mime_type" if snake else "mimeType"] = mime
    return {"file_data" if snake else "fileData": d}


class GeminiTest(StoreCase):
    def run_g(self, obj, cap=None, count_only=False, path=GEN):
        raw = obj if isinstance(obj, bytes) else h.dumps(obj)
        return asyncio.run(imgfix.rewrite(raw, self.store, cap, count_only=count_only, path=path))

    def out(self, obj, **kw):
        res = self.run_g(obj, **kw)
        self.assertIsNotNone(res.body)
        out = json.loads(res.body)
        self.assertEqual(h.media_blocks(out), [], "media part left for the text model")
        return res, out

    def ref(self, raw):
        return h.texts(self.rewritten(h.body(h.user(h.image_block(raw))), FakeCaptioner())[1])[0]

    def test_inline_data_both_spellings(self):
        raw, pdf = h.png(33, 22), h.text_pdf(["Gemini pdf"])
        for snake in (False, True):
            body = {"contents": contents({"text": "what?"}, inline(raw, snake=snake),
                                         inline(pdf, "application/pdf", snake=snake),
                                         inline(b"RIFF", "audio/wav", snake=snake),
                                         inline(b"plain text", "text/plain", snake=snake),
                                         inline(raw, snake=snake, data="!!not base64!!")),
                    "generationConfig": {"temperature": 0}}
            _, out = self.out(body, cap=FakeCaptioner())
            parts = out["contents"][0]["parts"]
            self.assertEqual(parts[0], {"text": "what?"})
            self.assertEqual(parts[1], {"text": self.ref(raw)})  # the same REF as every other format
            self.assertTrue(parts[2]["text"].startswith(f"[pdf {iid(pdf, 'pdf_')} | 1 pages] Text extracted"))
            self.assertEqual(parts[3:], [{"text": imgfix.UNAVAILABLE}, {"text": "plain text"},
                                         {"text": imgfix.UNAVAILABLE}])
            self.assertEqual(out["generationConfig"], {"temperature": 0})

    def test_protobuf_base64(self):
        raw = h.png(40, 41, (250, 251, 252))
        urlsafe = base64.urlsafe_b64encode(raw).decode().rstrip("=")
        _, out = self.out({"contents": contents(inline(raw, data=urlsafe))})
        self.assertEqual(out["contents"][0]["parts"][0]["text"].split()[1], iid(raw))

    def test_file_data(self):
        parts = [file_data("https://example.com/cat.png", "image/png"),
                 file_data("https://example.com/report.pdf", "application/pdf", snake=True),
                 file_data("https://generativelanguage.googleapis.com/v1beta/files/abc123", "image/png"),
                 file_data("gs://bucket/a.png", "image/png", snake=True),
                 file_data("https://example.com/no-type"),
                 file_data(h.data_url(h.png(3, 3))),
                 {"fileData": {"mimeType": "image/png"}}]
        _, out = self.out({"contents": contents(*parts)})
        u = imgfix.url_id
        self.assertEqual([p["text"] for p in out["contents"][0]["parts"]], [
            f"[image {u('https://example.com/cat.png')} | remote image not fetched: https://example.com/cat.png]",
            f"[document {u('https://example.com/report.pdf')} | remote document not fetched: "
            f"https://example.com/report.pdf]",
            "[file unavailable: file_id https://generativelanguage.googleapis.com/v1beta/files/abc123 not readable]",
            "[file unavailable: file_id gs://bucket/a.png not readable]",
            f"[image {u('https://example.com/no-type')} | remote image not fetched: https://example.com/no-type]",
            self.ref(h.png(3, 3)), imgfix.UNAVAILABLE])

    def test_system_instruction_and_function_parts(self):
        raw = h.png(12, 12)
        for key in ("systemInstruction", "system_instruction"):
            body = {key: {"parts": [{"text": "sys"}, inline(raw)]},
                    "contents": [{"role": "model", "parts": [{"functionCall": {"name": "shot", "args": {
                                     "inlineData": {"mimeType": "image/png", "data": "model-written"}}}}]},
                                 {"role": "user", "parts": [{"functionResponse": {"name": "shot", "response": {
                                     "inlineData": {"mimeType": "image/png", "data": h.b64(raw)}}}}]}]}
            _, out = self.out(body)
            ref = self.ref(raw)
            self.assertEqual(out[key]["parts"], [{"text": "sys"}, {"text": ref}])
            self.assertEqual(out["contents"][0], body["contents"][0])  # functionCall: model-written, untouched
            self.assertEqual(out["contents"][1]["parts"][0]["functionResponse"]["response"], {"text": ref})

    def test_byte_stable(self):
        raw = h.png(25, 15)
        body = {"contents": contents({"text": "q"}, inline(raw), inline(raw, snake=True))}
        first = self.run_g(body, cap=FakeCaptioner("Caption one.")).body
        self.store = imgfix.Store(self.dir)  # restart: no in-memory state, a vision model that says something else
        cap = FakeCaptioner("Caption two.")
        self.assertEqual(self.run_g(body, cap=cap).body, first)
        self.assertEqual(cap.calls, [])
        self.assertIn(b"| Caption one.]", first)

    def test_count_tokens_read_only(self):
        raw = h.png(17, 18)
        for body in ({"contents": contents(inline(raw))},
                     {"generateContentRequest": {"model": "models/gemini-text", "contents": contents(inline(raw)),
                                                 "systemInstruction": {"parts": [inline(raw, snake=True)]}}}):
            _, out = self.out(body, cap=FakeCaptioner(), count_only=True, path=COUNT)
            self.assertIn("| 17x18 png | no preview]", json.dumps(out))
        self.assertEqual(self.files(), [])

    def test_passthrough_by_path_model(self):
        raw = h.png(7, 7)
        body = {"contents": contents(inline(raw))}
        with mock.patch.object(imgfix, "PASSTHROUGH_MODELS", ("gemini-2.5-*",)):
            for b, path in ((body, "/v1beta/models/gemini-2.5-pro:streamGenerateContent"),
                            (dict(body, model="models/gemini-2.5-flash"), "/v1/whatever"),
                            ({"generateContentRequest": dict(body, model="models/gemini-2.5-pro")},
                             "/v1beta/models/gemini-2.5-pro:countTokens")):
                res = self.run_g(b, path=path)
                self.assertEqual((res.body, res.stats["passthrough"]), (None, 1), path)
            for b, path in ((body, GEN),                                                          # path model
                            (dict(body, model="text-model"), "/v1beta/models/gemini-2.5-pro:generateContent"),
                            ({"generateContentRequest": dict(body, model="models/text")},          # inner model
                             "/v1beta/models/gemini-2.5-pro:countTokens"),
                            (body, "/v1/chat/completions")):                                    # no model at all
                self.assertIsNotNone(self.run_g(b, path=path).body, path)

    def test_ambiguous_spellings_and_backstop(self):
        raw = h.png(5, 5)
        for bad, needle in (({"contents": [], "systemInstruction": {}, "system_instruction": {}}, "systeminstruction"),
                            ({"contents": contents({"inline_data": {}, "inlineData": {}})}, "inlinedata"),
                            ({"contents": contents({"inlineData": {"mime_type": "image/png", "mimeType": "x",
                                                                   "data": ""}})}, "mimetype"),
                            ({"contents": contents({"fileData": {"file_uri": "a", "fileUri": "b"}})}, "fileuri"),
                            ({"contents": [], "Contents": []}, "contents")):
            with self.assertRaisesRegex(imgfix.Reject, f"spells the JSON key '{needle}' twice"):
                self.run_g(bad)
        with self.assertRaisesRegex(imgfix.Reject, "a Gemini 'inlineData' part"):
            self.run_g({"contents": contents({"text": "x"}), "cachedParts": [inline(raw)]})
        self.assertEqual(self.files(), [])
