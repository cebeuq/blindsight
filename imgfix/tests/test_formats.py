"""OpenAI Chat / OpenAI Responses / Ollama request shapes, file_id parts, data URLs, model passthrough (rewrite core;
captions mocked; no network)."""
import json
from unittest import mock

import imgfix
from . import helpers as h
from .test_rewrite import PLACEHOLDER, FakeCaptioner, StoreCase, iid

URL = "https://example.com/cat.png"


def chat(*content, model="text-model", **extra):
    return {"model": model, "messages": [{"role": "user", "content": list(content)}], **extra}


def responses(*content, model="text-model", **extra):
    return {"model": model, "input": [{"type": "message", "role": "user", "content": list(content)}], **extra}


def image_url(url, **extra):
    return {"type": "image_url", "image_url": {"url": url, **extra}}


def input_image(**fields):
    return {"type": "input_image", "detail": "auto", **fields}


class FormatTest(StoreCase):
    def body_of(self, obj, cap=None, count_only=False):
        res = self.run_rw(obj, cap, count_only)
        self.assertIsNotNone(res.body)
        out = json.loads(res.body)
        self.assertEqual(h.media_blocks(out), [], "media part left for the text model")
        return res, out

    def ref(self, raw, cap="A red square."):
        """The REF an Anthropic image block gets (the reference text every format must reproduce)."""
        return h.texts(self.rewritten(h.body(h.user(h.image_block(raw))), FakeCaptioner(cap))[1])[0]

    # ------------------------------------------------------------ OpenAI Chat Completions

    def test_chat_image_url(self):
        raw = h.png(40, 30)
        _, out = self.body_of(chat(h.text("what is this?"), image_url(h.data_url(raw), detail="high")), FakeCaptioner())
        i = iid(raw)
        self.assertEqual(out["messages"][0]["content"], [
            h.text("what is this?"),
            {"type": "text", "text": f'[image {i} | 40x30 png | A red square.] The model cannot see images. To '
                                     f'inspect this one, call the ask_image tool with image_id="{i}" and a specific '
                                     f'question.'}])
        self.assertEqual(self.files(), [f"img/{i}.png", f"meta/{i}.json"])
        self.assertEqual(out["model"], "text-model")

    def test_chat_image_url_bare_string_and_remote(self):
        raw = h.png(9, 9)
        _, out = self.body_of(chat({"type": "image_url", "image_url": h.data_url(raw)}, image_url(URL)))
        ts = h.texts(out)
        self.assertEqual(PLACEHOLDER.match(ts[0]).group(1), iid(raw))
        self.assertEqual(ts[1], f"[image {imgfix.url_id(URL)} | remote image not fetched: {URL}]")

    def test_chat_file_parts(self):
        pdf = h.text_pdf(["Chat file text"])
        parts = [{"type": "file", "file": {"filename": "a.pdf", "file_data": h.data_url(pdf, "application/pdf")}},
                 {"type": "file", "file": {"file_id": "file-abc123"}},
                 {"type": "file", "file": {"filename": "n.txt",
                                           "file_data": h.data_url("plain é".encode(), "text/plain")}},
                 {"type": "file", "file": {"filename": "z.zip",
                                           "file_data": h.data_url(b"PK\x03\x04", "application/zip")}},
                 {"type": "file", "file": {"filename": "x.pdf"}},
                 {"type": "file", "file": "nope"}]
        res, out = self.body_of(chat(*parts))
        ts = h.texts(out)
        p = iid(pdf, "pdf_")
        self.assertTrue(ts[0].startswith(f'[pdf {p} | 1 pages] Text extracted with pdftotext (page images: ask_image '
                                         f'with image_id="{p}" and page=<n>):\nChat file text'), ts[0])
        self.assertEqual(ts[1:], ["[file unavailable: file_id file-abc123 not readable]", "plain é",
                                  imgfix.UNAVAILABLE, imgfix.UNAVAILABLE, imgfix.UNAVAILABLE])
        self.assertEqual(res.stats["failed"], 3)
        self.assertEqual(self.files(), [f"meta/{p}.json", f"meta/{p}.txt", f"pdf/{p}.pdf"])

    def test_chat_tool_message_and_cache_control(self):
        raw = h.png(12, 8)
        cc = {"type": "ephemeral"}
        msgs = [{"role": "assistant", "content": None, "tool_calls": [
                    {"id": "c1", "type": "function", "function": {"name": "shot", "arguments": "{}"}}]},
                {"role": "tool", "tool_call_id": "c1", "content": [image_url(h.data_url(raw))]},
                {"role": "user", "content": [dict(image_url(h.data_url(raw)), cache_control=cc)]}]
        _, out = self.body_of({"model": "m", "messages": msgs})
        self.assertEqual(out["messages"][0], msgs[0])
        self.assertEqual(PLACEHOLDER.match(out["messages"][1]["content"][0]["text"]).group(1), iid(raw))
        self.assertEqual(out["messages"][2]["content"][0]["cache_control"], cc)

    # ------------------------------------------------------------ OpenAI Responses

    def test_responses_parts(self):
        raw, pdf = h.png(20, 10), h.text_pdf(["Responses file"])
        content = [{"type": "input_text", "text": "look"},
                   input_image(image_url=h.data_url(raw)),
                   input_image(file_id="file-img1"),
                   {"type": "input_file", "filename": "r.pdf", "file_data": h.data_url(pdf, "application/pdf")},
                   {"type": "input_file", "file_url": "https://example.com/r.pdf"},
                   {"type": "input_file", "file_id": "file-pdf1"},
                   input_image(),
                   input_image(image_url={"url": h.data_url(raw)})]
        _, out = self.body_of(responses(*content), FakeCaptioner())
        parts = out["input"][0]["content"]
        self.assertEqual([x["type"] for x in parts], ["input_text"] * 8)
        self.assertEqual(sorted(parts[1]), ["text", "type"])  # the whole part is replaced ("detail" goes too)
        self.assertEqual(parts[1]["text"], self.ref(raw))
        self.assertEqual(parts[2]["text"], "[image unavailable: file_id file-img1 not readable]")
        self.assertTrue(parts[3]["text"].startswith(f"[pdf {iid(pdf, 'pdf_')} | 1 pages] "))
        u = imgfix.url_id("https://example.com/r.pdf")
        self.assertEqual(parts[4]["text"], f"[document {u} | remote document not fetched: https://example.com/r.pdf]")
        self.assertEqual(parts[5]["text"], "[file unavailable: file_id file-pdf1 not readable]")
        self.assertEqual(parts[6]["text"], imgfix.UNAVAILABLE)
        self.assertEqual(parts[7]["text"], parts[1]["text"])  # Chat-style {"url": ...} tolerated

    def test_responses_function_items(self):
        raw = h.png(6, 6)
        hidden = json.dumps({"type": "input_image", "image_url": h.data_url(raw)})
        items = [{"type": "function_call", "call_id": "c1", "name": "shot", "arguments": hidden},
                 {"type": "function_call", "call_id": "c2", "name": "x",
                  "arguments": {"type": "input_image", "image_url": "model-written"}},
                 {"type": "function_call_output", "call_id": "c1",
                  "output": [{"type": "input_text", "text": "screenshot:"}, input_image(image_url=h.data_url(raw))]}]
        res, out = self.body_of({"model": "m", "input": items})
        self.assertEqual(out["input"][:2], items[:2])  # model-written arguments never touched
        self.assertEqual(out["input"][2]["output"][0], {"type": "input_text", "text": "screenshot:"})
        self.assertEqual(PLACEHOLDER.match(out["input"][2]["output"][1]["text"]).group(1), iid(raw))
        self.assertEqual(res.ids, [iid(raw)])

    def test_responses_plain_input_unchanged(self):
        for body in ({"model": "m", "input": "just text"},
                     {"model": "m", "input": [{"role": "user", "content": "hi"}]}):
            self.assertIsNone(self.run_rw(body).body)
        self.assertEqual(self.files(), [])

    # ------------------------------------------------------------ Ollama native

    def test_ollama_chat(self):
        a, b = h.png(30, 20), h.png(20, 30, (0, 0, 255))
        msgs = [{"role": "system", "content": "be brief"},
                {"role": "user", "content": "compare these", "images": [h.b64(a), h.data_url(b)]},
                {"role": "user", "images": [h.b64(a), "!!not base64!!", 7]},
                {"role": "user", "content": "no images", "images": []}]
        res, out = self.body_of({"model": "llava", "messages": msgs, "stream": False}, FakeCaptioner())
        ra, rb = self.ref(a), self.ref(b)
        self.assertEqual(out["messages"][0], msgs[0])
        self.assertEqual(out["messages"][1], {"role": "user", "content": f"compare these\n\n{ra}\n{rb}"})
        self.assertEqual(out["messages"][2], {"role": "user", "content": f"\n\n{ra}\n{imgfix.UNAVAILABLE}\n"
                                                                         f"{imgfix.UNAVAILABLE}"})
        self.assertEqual(out["messages"][3], msgs[3])  # empty list: nothing to replace
        self.assertEqual(res.stats["failed"], 2)

    def test_ollama_generate(self):
        raw = h.png(16, 16)
        _, out = self.body_of({"model": "llava", "prompt": "describe", "images": [h.b64(raw)], "stream": False},
                              FakeCaptioner())
        self.assertEqual(out, {"model": "llava", "prompt": "describe\n\n" + self.ref(raw), "stream": False})
        _, out = self.body_of({"model": "llava", "images": [h.b64(raw)]})
        self.assertEqual(out["prompt"], "\n\n" + self.ref(raw))

    def test_ollama_tool_call_arguments_untouched(self):
        args = {"images": ["aGVsbG8="], "shot": {"type": "image", "source": {"type": "base64", "data": "x"}}}
        msgs = [{"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "f", "arguments": args}}]},
                {"role": "user", "content": "x", "images": [h.b64(h.png(5, 5))]}]
        _, out = self.body_of({"model": "llava", "messages": msgs})
        self.assertEqual(out["messages"][0], msgs[0])

    # ------------------------------------------------------------ every format: same REF, byte-stable

    def test_same_ref_in_every_format_and_stable(self):
        raw, pdf = h.png(50, 40, (3, 4, 5)), h.text_pdf(["same text"])
        bodies = [h.body(h.user(h.image_block(raw), h.pdf_block(pdf))),
                  chat(image_url(h.data_url(raw)),
                       {"type": "file", "file": {"file_data": h.data_url(pdf, "application/pdf")}}),
                  responses(input_image(image_url=h.data_url(raw)),
                            {"type": "input_file", "file_data": h.data_url(pdf, "application/pdf")}),
                  {"model": "m", "messages": [{"role": "user", "content": "q", "images": [h.b64(raw)]}]}]
        first = [self.run_rw(b, FakeCaptioner("Same caption.")).body for b in bodies]
        refs = [h.part_texts(json.loads(first[k])[key])[:2]
                for k, key in ((0, "messages"), (1, "messages"), (2, "input"))]
        self.assertEqual(refs[0], refs[1])
        self.assertEqual(refs[0], refs[2])
        self.assertEqual(json.loads(first[3])["messages"][0]["content"], "q\n\n" + refs[0][0])
        # Next turns, and after a restart with a vision model that would now say something else: identical bytes.
        imgfix.PDF_CACHE.clear()
        self.store = imgfix.Store(self.dir)
        cap = FakeCaptioner("Another caption.")
        self.assertEqual([self.run_rw(b, cap).body for b in bodies], first)
        self.assertEqual(cap.calls, [])

    # ------------------------------------------------------------ data URLs (strict)

    def test_data_urls(self):
        raw = h.png(7, 5)
        good = [h.data_url(raw), "DATA:IMAGE/PNG;BASE64," + h.b64(raw),
                f"data:image/png;name=a.png;base64,{h.b64(raw)}",
                f"data:image/png;base64,{h.b64(raw)[:20]}\n{h.b64(raw)[20:]}"]
        bad = [f"data:image/png,{h.b64(raw)}", f"data:;base64,{h.b64(raw)}", f"data:image/png;base64,{h.b64(raw)}%%",
               "data:image/png;base64", f"data:image;base64,{h.b64(raw)}", f"data: image/png;base64,{h.b64(raw)}",
               "ftp://example.com/a.png", "/local/a.png", h.b64(raw)]
        _, out = self.body_of(chat(*(image_url(u) for u in good + bad)))
        ts = h.texts(out)
        self.assertEqual([PLACEHOLDER.match(t).group(1) for t in ts[:len(good)]], [iid(raw)] * len(good))
        self.assertEqual(ts[len(good):], [imgfix.UNAVAILABLE] * len(bad))  # fails closed, never echoes the string

    # ------------------------------------------------------------ PASSTHROUGH_MODELS

    def test_passthrough_models(self):
        raw = h.png(8, 8)
        with mock.patch.object(imgfix, "PASSTHROUGH_MODELS", ("claude-*", "gpt-4o")):
            for model in ("claude-sonnet-4-5", "gpt-4o"):
                for b in (h.body(h.user(h.image_block(raw)), model=model),
                          chat(image_url(h.data_url(raw)), model=model),
                          {"model": model, "prompt": "p", "images": [h.b64(raw)]}):
                    res = self.run_rw(b, FakeCaptioner())
                    self.assertIsNone(res.body)  # forwarded unchanged, images intact
                    self.assertEqual(res.stats["passthrough"], 1)
            self.assertEqual(self.files(), [])
            for model in ("gpt-4o-mini", "text-model", "Claude-x", None, 7):
                _, out = self.body_of(chat(image_url(h.data_url(raw)), model=model))
                self.assertTrue(PLACEHOLDER.match(h.texts(out)[0]))

    def test_count_only_responses(self):
        raw = h.png(40, 30)
        _, out = self.body_of(responses(input_image(image_url=h.data_url(raw))), FakeCaptioner(), count_only=True)
        self.assertRegex(out["input"][0]["content"][0]["text"],
                         r"^\[image img_[0-9a-f]{16} \| 40x30 png \| no preview\]")
        self.assertEqual(self.files(), [])
