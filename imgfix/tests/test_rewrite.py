"""Rewrite core (prepare/finish/rewrite) without mitmproxy. Captions are mocked; no network."""
import asyncio
import hashlib
import io
import json
import os
import re
import shutil
import stat
import subprocess
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image

import imgfix
import vision_api
from . import helpers as h

PLACEHOLDER = re.compile(r'^\[image (img_[0-9a-f]{16}) \| (\d+)x(\d+) (png|jpg|gif|webp) \| ([^\n]*)\] The model cannot '
                         r'see images\. To inspect this one, call the ask_image tool with image_id="\1" and a specific '
                         r'question\.$')


def iid(raw, prefix="img_"):
    return prefix + hashlib.sha256(raw).hexdigest()[:16]


class FakeCaptioner:
    def __init__(self, reply="A red square.", fail=False, delay=0.0):
        self.reply, self.fail, self.delay = reply, fail, delay
        self.calls, self.active, self.max_active = [], 0, 0

    async def __call__(self, raw):
        self.calls.append(iid(raw))
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            await asyncio.sleep(self.delay)
            if self.fail:
                raise RuntimeError("vision model down")
            return self.reply
        finally:
            self.active -= 1


class StoreCase(unittest.TestCase):
    """Fresh store per test + helpers (no tests of its own; other modules subclass it)."""

    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="imgfix-test-")
        self.store = imgfix.Store(self.dir)
        imgfix.PDF_CACHE.clear()
        free = mock.patch.object(imgfix, "MIN_FREE", 0)  # tests run on a small tmpfs; test_low_disk_space sets it
        free.start()
        self.addCleanup(free.stop)

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def run_rw(self, obj, cap=None, count_only=False):
        raw = obj if isinstance(obj, bytes) else h.dumps(obj)
        return asyncio.run(imgfix.rewrite(raw, self.store, cap, count_only=count_only))

    def rewritten(self, obj, cap=None, count_only=False):
        res = self.run_rw(obj, cap, count_only)
        self.assertIsNotNone(res.body)
        out = json.loads(res.body)
        self.assertEqual(h.media_blocks(out["messages"]), [], "media block left for the text model")
        return res, out

    def files(self):
        return sorted(os.path.relpath(os.path.join(d, f), self.dir)
                      for d, _, fs in os.walk(self.dir) for f in fs)

    def meta(self, mid):
        with open(os.path.join(self.dir, "meta", mid + ".json")) as f:
            return json.load(f)


class RewriteTest(StoreCase):
    # ------------------------------------------------------------ images

    def test_user_image(self):
        raw = h.png(40, 30)
        res, out = self.rewritten(h.body(h.user(h.image_block(raw), h.text("what is this?"))), FakeCaptioner())
        i = iid(raw)
        self.assertEqual(out["messages"][0]["content"], [
            {"type": "text", "text": f'[image {i} | 40x30 png | A red square.] The model cannot see images. To inspect '
                                     f'this one, call the ask_image tool with image_id="{i}" and a specific question.'},
            {"type": "text", "text": "what is this?"}])
        self.assertEqual(self.files(), [f"img/{i}.png", f"meta/{i}.json"])
        with open(os.path.join(self.dir, "img", i + ".png"), "rb") as f:
            self.assertEqual(f.read(), raw)
        m = self.meta(i)
        self.assertEqual({k: m[k] for k in ("id", "kind", "media_type", "bytes", "width", "height", "caption")},
                         {"id": i, "kind": "image", "media_type": "image/png", "bytes": len(raw), "width": 40,
                          "height": 30, "caption": "A red square."})
        self.assertRegex(m["created"], r"^\d{4}-\d\d-\d\dT")
        for p in (f"img/{i}.png", f"meta/{i}.json"):  # vision-mcp reads as another uid
            self.assertEqual(stat.S_IMODE(os.stat(os.path.join(self.dir, p)).st_mode), 0o644)
        self.assertEqual((res.stats["images"], res.stats["new"], res.stats["captioned"]), (1, 1, 1))
        self.assertEqual(res.ids, [i])

    def test_tool_result_image(self):
        raw = h.png(10, 12, (0, 0, 255))
        msg = h.user({"type": "tool_result", "tool_use_id": "toolu_1",
                      "content": [h.text("screenshot:"), h.image_block(raw)]})
        _, out = self.rewritten(h.body(msg), FakeCaptioner("A blue box"))
        tr = out["messages"][0]["content"][0]
        self.assertEqual(tr["type"], "tool_result")
        self.assertEqual(tr["content"][0], h.text("screenshot:"))
        m = PLACEHOLDER.match(tr["content"][1]["text"])
        self.assertTrue(m)
        self.assertEqual(m.groups(), (iid(raw), "10", "12", "png", "A blue box"))

    def test_jpeg_gif_webp_ext(self):
        blocks = []
        for fmt, mt in (("JPEG", "image/jpeg"), ("GIF", "image/gif"), ("WEBP", "image/webp")):
            buf = io.BytesIO()
            Image.new("RGB", (8, 6), (1, 2, 3)).save(buf, fmt)
            blocks.append(h.image_block(buf.getvalue(), mt))
        _, out = self.rewritten(h.body(h.user(*blocks)))
        self.assertEqual([PLACEHOLDER.match(t).group(4) for t in h.texts(out)], ["jpg", "gif", "webp"])
        self.assertEqual(sorted(f.rsplit(".", 1)[1] for f in self.files() if f.startswith("img/")),
                         ["gif", "jpg", "webp"])

    def test_multiple_images_caption_cap_and_concurrency(self):
        raws = [h.png(20 + k, 20, (k * 30, 0, 0)) for k in range(6)]
        cap = FakeCaptioner(delay=0.05)
        content = [h.image_block(r) for r in raws] + [h.image_block(raws[5])]  # last one twice
        res, out = self.rewritten(h.body(h.user(*content)), cap)
        ts = h.texts(out)
        self.assertEqual(len(ts), 7)
        self.assertEqual(sorted(cap.calls), sorted(iid(r) for r in raws[2:]))  # newest 4 only, dup captioned once
        self.assertLessEqual(cap.max_active, imgfix.CAPTION_CONCURRENCY)
        caps = [PLACEHOLDER.match(t).group(5) for t in ts]
        self.assertEqual(caps, ["no preview"] * 2 + ["A red square."] * 5)
        self.assertEqual(ts[5], ts[6])
        for r in raws:  # over-cap images get caption "" written once, so their text never changes
            self.assertEqual(self.meta(iid(r))["caption"], "" if r in raws[:2] else "A red square.")
        self.assertEqual((res.stats["images"], res.stats["new"], res.stats["captioned"]), (7, 6, 4))

    def test_url_image(self):
        url = "https://example.com/cat.png?x=1"
        _, out = self.rewritten(h.body(h.user({"type": "image", "source": {"type": "url", "url": url}})))
        u = "url_" + hashlib.sha256(url.encode()).hexdigest()[:16]
        self.assertEqual(h.texts(out), [f"[image {u} | remote image not fetched: {url}]"])
        self.assertEqual(self.files(), [])

    def test_corrupt_and_undecodable(self):
        bad = [{"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "!!!not base64!!!"}},
               h.image_block(b"definitely not an image"),
               h.image_block(h.noise_png()[:2000]),  # truncated PNG
               {"type": "image"},
               {"type": "document", "source": {"type": "base64", "media_type": "application/pdf", "data": h.b64(b"%PDF-junk")}},
               {"type": "document", "source": {"type": "base64", "media_type": "application/zip", "data": h.b64(b"PK")}}]
        res, out = self.rewritten(h.body(h.user(*bad)), FakeCaptioner())
        self.assertEqual(h.texts(out), [imgfix.UNAVAILABLE] * len(bad))
        self.assertEqual(res.stats["failed"], len(bad))
        junk = iid(b"%PDF-junk", "pdf_")  # no raw bytes are stored; the broken PDF gets a write-once "failed" meta
        self.assertEqual(self.files(), [f"meta/{junk}.json"])
        self.assertEqual((self.meta(junk)["failed"], self.meta(junk)["kind"]), (True, "pdf"))

    def test_file_id_sources(self):
        # Provider-side uploads (Anthropic Files API): the bytes are not in the request, so a fixed text says so.
        _, out = self.rewritten(h.body(h.user({"type": "image", "source": {"type": "file", "file_id": "file_1"}},
                                              {"type": "document", "source": {"type": "file", "file_id": "file_2"}},
                                              {"type": "image", "source": {"type": "file"}})))
        self.assertEqual(h.texts(out), ["[image unavailable: file_id file_1 not readable]",
                                        "[file unavailable: file_id file_2 not readable]", imgfix.UNAVAILABLE])
        self.assertEqual(self.files(), [])

    def test_oversized(self):
        with mock.patch.object(imgfix, "MAX_IMAGE_BYTES", 10):
            _, out = self.rewritten(h.body(h.user(h.image_block(h.png()))))
        self.assertEqual(h.texts(out), [imgfix.UNAVAILABLE])

    def test_cache_control_carried(self):
        cc = {"type": "ephemeral"}
        _, out = self.rewritten(h.body(h.user(h.image_block(h.png(), cache_control=cc))))
        self.assertEqual(out["messages"][0]["content"][0]["cache_control"], cc)

    # ------------------------------------------------------------ documents

    def test_pdf_document(self):
        raw = h.text_pdf(["Hello from page one", "Second page here"])
        res, out = self.rewritten(h.body(h.user(h.pdf_block(raw), h.text("summarize"))))
        p = iid(raw, "pdf_")
        t = h.texts(out)[0]
        head = (f'[pdf {p} | 2 pages] Text extracted with pdftotext (page images: ask_image with image_id="{p}" '
                f'and page=<n>):\n')
        self.assertTrue(t.startswith(head), t[:200])
        self.assertIn("Hello from page one", t)
        self.assertIn("Second page here", t)
        self.assertEqual(self.files(), [f"meta/{p}.json", f"meta/{p}.txt", f"pdf/{p}.pdf"])
        self.assertEqual(Path(self.dir, "meta", p + ".txt").read_bytes().decode(), t)
        m = self.meta(p)
        self.assertEqual((m["kind"], m["media_type"], m["pages"], m["bytes"], m["caption"]),
                         ("pdf", "application/pdf", 2, len(raw), ""))
        self.assertEqual(res.stats["pdfs"], 1)

    def test_pillow_pdf_two_pages(self):
        raw = h.pillow_pdf(2)
        _, out = self.rewritten(h.body(h.user(h.pdf_block(raw))))
        self.assertTrue(h.texts(out)[0].startswith(f"[pdf {iid(raw, 'pdf_')} | 2 pages] "))

    def test_pdf_truncation(self):
        t = imgfix.pdf_text("pdf_0123456789abcdef", 3, "x" * 70000)
        self.assertTrue(t.endswith("page=<n>):\n" + "x" * 60000 + "\n[... truncated]"))
        self.assertFalse(imgfix.pdf_text("pdf_0123456789abcdef", 3, "x" * 60000).endswith("truncated]"))

    def test_text_and_content_documents(self):
        raw = h.png(5, 5)
        docs = [{"type": "document", "source": {"type": "text", "media_type": "text/plain", "data": "plain doc"},
                 "title": "t"},
                {"type": "document", "source": {"type": "content", "content": [h.text("c1"), h.image_block(raw)]}},
                {"type": "document", "source": {"type": "url", "url": "https://example.com/a.pdf"}}]
        _, out = self.rewritten(h.body(h.user(*docs)))
        ts = h.texts(out)
        self.assertEqual(ts[:2], ["plain doc", "c1"])
        self.assertEqual(PLACEHOLDER.match(ts[2]).group(1), iid(raw))
        self.assertTrue(ts[3].startswith("[document url_") and ts[3].endswith("remote document not fetched: "
                                                                               "https://example.com/a.pdf]"))

    # ------------------------------------------------------------ structure / passthrough

    def test_nested_content(self):
        a, b, c = h.png(3, 3), h.png(4, 4), h.png(5, 5)
        tool_use = {"type": "tool_use", "id": "toolu_1", "name": "x",
                    "input": {"type": "image", "note": "model-written JSON, left alone"}}
        msgs = [{"role": "assistant", "content": [tool_use]},
                h.user({"type": "tool_result", "tool_use_id": "toolu_1", "content": [
                    {"type": "search_result", "source": "s", "title": "t", "content": [h.image_block(a)]}]}),
                h.user({"type": "tool_result", "tool_use_id": "toolu_2", "content": h.image_block(b)}),
                {"role": "user", "content": [[h.image_block(c)]]}]
        _, out = self.rewritten(h.body(*msgs))
        self.assertEqual(out["messages"][0]["content"][0], tool_use)
        self.assertEqual([PLACEHOLDER.match(t).group(1) for t in h.texts(out)], [iid(a), iid(b), iid(c)])

    def test_no_images_unchanged(self):
        res = self.run_rw(h.body(h.user(h.text("hi")), system=[h.text("sys")]))
        self.assertIsNone(res.body)
        self.assertEqual(self.files(), [])

    def test_not_json(self):
        res = self.run_rw(b'{"model": "text-model", "messages": [')
        self.assertIsNone(res.body)
        self.assertEqual(res.stats["not_json"], 1)
        self.assertIsNone(self.run_rw(b"\xff\xfe garbage").body)

    def test_model_name_is_not_special(self):
        for model in ("some-vision-model", "", None, 7):  # no direct route: every body is scanned
            _, out = self.rewritten(h.body(h.user(h.image_block(h.png())), model=model))
            self.assertTrue(PLACEHOLDER.match(h.texts(out)[0]))
            self.assertEqual(out["model"], model)

    def test_lone_surrogate(self):
        raw = h.png()
        src = h.dumps(h.body(h.user(h.text("PLACEHOLDER"), h.image_block(raw))))
        src = src.replace(b'"PLACEHOLDER"', b'"bad \\ud800 surrogate \\u00e9"')
        res, out = self.rewritten(src)
        self.assertTrue(res.body.isascii())
        self.assertIn(b"\\ud800", res.body)
        self.assertEqual(out["messages"][0]["content"][0]["text"], "bad \ud800 surrogate é")
        self.assertTrue(PLACEHOLDER.match(h.texts(out)[1]))
        orig = json.loads(src)
        del orig["messages"][0]["content"][1], out["messages"][0]["content"][1]
        self.assertEqual(out, orig)  # everything but the image round-trips exactly

    def test_compact_serialization(self):
        res, _ = self.rewritten(h.body(h.user(h.image_block(h.png()))))
        self.assertEqual(res.body, json.dumps(json.loads(res.body), ensure_ascii=True, separators=(",", ":")).encode())

    # ------------------------------------------------------------ determinism / write-once

    def test_deterministic_across_turns_and_restart(self):
        raws = [h.png(30, 20), h.png(20, 30, (0, 255, 0))]
        pdf = h.text_pdf(["one", "two"])
        req = h.body(h.user(h.image_block(raws[0]), h.text("q1")),
                     {"role": "assistant", "content": [h.text("a1")]},
                     h.user({"type": "tool_result", "tool_use_id": "t", "content": [h.image_block(raws[1])]},
                            h.pdf_block(pdf)))
        first = self.run_rw(req, FakeCaptioner("First caption")).body
        again = self.run_rw(req, FakeCaptioner("First caption")).body
        self.assertEqual(first, again)
        # Simulated restart: no in-memory state, fresh Store object, a captioner that would now say something else.
        imgfix.PDF_CACHE.clear()
        self.store = imgfix.Store(self.dir)
        cap = FakeCaptioner("A totally different caption")
        self.assertEqual(self.run_rw(req, cap).body, first)
        self.assertEqual(cap.calls, [])  # meta on disk drives captions; vision model not asked again
        # Next turn appends to history: the old part of the prompt is unchanged byte for byte.
        req2 = json.loads(h.dumps(req))
        req2["messages"].append({"role": "assistant", "content": [h.text("a2")]})
        req2["messages"].append(h.user(h.text("q3")))
        later = json.loads(self.run_rw(req2, cap).body)
        self.assertEqual(later["messages"][:3], json.loads(first)["messages"])

    def test_caption_failure_is_permanent(self):
        raw = h.png()
        req = h.body(h.user(h.image_block(raw)))
        t1 = h.texts(self.rewritten(req, FakeCaptioner(fail=True))[1])
        self.assertIn("| no preview]", t1[0])
        cap = FakeCaptioner()
        t2 = h.texts(self.rewritten(req, cap)[1])
        self.assertEqual(t1, t2)
        self.assertEqual(cap.calls, [])
        self.assertEqual(self.meta(iid(raw))["caption"], "")

    def test_caption_timeout(self):
        with mock.patch.object(imgfix, "CAPTION_TIMEOUT", 0.05):
            _, out = self.rewritten(h.body(h.user(h.image_block(h.png()))), FakeCaptioner(delay=2))
        self.assertIn("| no preview]", h.texts(out)[0])

    def test_existing_meta_wins(self):
        raw = h.png(40, 30)
        i = iid(raw)
        pre = imgfix.image_meta(i, "png", raw, 40, 30, "Pre-existing caption.")
        self.store.write_meta(pre)
        mpath = Path(self.dir, "meta", i + ".json")
        before = mpath.read_bytes()
        cap = FakeCaptioner()
        _, out = self.rewritten(h.body(h.user(h.image_block(raw))), cap)
        self.assertIn("| Pre-existing caption.]", h.texts(out)[0])
        self.assertEqual(cap.calls, [])
        self.assertEqual(mpath.read_bytes(), before)
        self.assertTrue(os.path.exists(os.path.join(self.dir, "img", i + ".png")))  # raw file self-healed

    def test_write_meta_once(self):
        m1 = imgfix.image_meta("img_0000000000000000", "png", b"x", 1, 1, "first")
        m2 = dict(m1, caption="second")
        self.assertEqual(self.store.write_meta(m1)["caption"], "first")
        self.assertEqual(self.store.write_meta(m2)["caption"], "first")
        self.assertEqual(self.meta("img_0000000000000000")["caption"], "first")
        self.assertEqual([f for f in os.listdir(os.path.join(self.dir, "meta")) if f.startswith(".tmp")], [])

    def test_count_tokens_read_only(self):
        raw, pdf = h.png(40, 30), h.text_pdf(["count me"])
        req = h.body(h.user(h.image_block(raw), h.pdf_block(pdf)))
        cap = FakeCaptioner()
        ct1 = h.texts(self.rewritten(req, cap, count_only=True)[1])
        self.assertEqual(self.files(), [])  # no meta, no raw files
        self.assertEqual(cap.calls, [])
        self.assertIn("| no preview]", ct1[0])
        self.assertIn("count me", ct1[1])
        real = h.texts(self.rewritten(req, cap)[1])
        self.assertIn("| A red square.]", real[0])
        self.assertEqual(ct1[1], real[1])
        self.assertEqual(h.texts(self.rewritten(req, None, count_only=True)[1]), real)

    # ------------------------------------------------------------ review fixes: PDFs stored, fail closed, limits

    def test_pdf_text_stored_survives_poppler_failure(self):
        raw = h.text_pdf(["kept text"])
        req = h.body(h.user(h.pdf_block(raw)))
        first = self.run_rw(req).body
        imgfix.PDF_CACHE.clear()  # restart / LRU eviction
        boom = mock.Mock(side_effect=subprocess.TimeoutExpired("pdftotext", 120))
        with mock.patch.object(imgfix, "_run", boom):
            self.assertEqual(self.run_rw(req).body, first)  # read back from meta/<id>.txt, poppler not needed
        boom.assert_not_called()

    def test_pdf_first_sight_failure_is_permanent(self):
        raw = h.text_pdf(["slow"])
        req = h.body(h.user(h.pdf_block(raw)))
        with mock.patch.object(imgfix, "_run", side_effect=subprocess.TimeoutExpired("pdfinfo", 120)):
            self.assertEqual(h.texts(self.rewritten(req)[1]), [imgfix.UNAVAILABLE])
        p = iid(raw, "pdf_")
        self.assertEqual(self.files(), [f"meta/{p}.json"])
        self.assertTrue(self.meta(p)["failed"])
        spy = mock.Mock(wraps=imgfix._pdf_extract)
        with mock.patch.object(imgfix, "_pdf_extract", spy):  # later turns: same text, poppler not re-run
            self.assertEqual(h.texts(self.rewritten(req)[1]), [imgfix.UNAVAILABLE])
        spy.assert_not_called()

    def test_pdf_environment_failure_not_stored(self):
        raw = h.text_pdf(["retry me"])
        req = h.body(h.user(h.pdf_block(raw)))
        for err in (OSError("fork failed"), subprocess.CalledProcessError(-9, "pdftotext")):  # spawn / OOM kill
            with mock.patch.object(imgfix, "_run", side_effect=err):
                self.assertEqual(h.texts(self.rewritten(req)[1]), [imgfix.UNAVAILABLE])
            self.assertEqual(self.files(), [])
        self.assertIn("retry me", h.texts(self.rewritten(req)[1])[0])

    def test_pdf_stdout_cap(self):
        raw = h.text_pdf(["Hello from page one"])
        with mock.patch.object(imgfix, "PDF_STDOUT_CAP", 8):
            t = h.texts(self.rewritten(h.body(h.user(h.pdf_block(raw))))[1])[0]
        self.assertTrue(t.endswith("page=<n>):\nHello fr\n[... truncated]"), t)

    def test_run_limits(self):
        t0 = time.monotonic()
        out, cut = imgfix._run(["yes"], cap=100)  # endless output: read 100 bytes, then killed
        self.assertEqual((out, cut), (b"y\n" * 50, True))
        self.assertEqual(imgfix._run(["sh", "-c", "ulimit -v"]), (b"786432\n", False))  # prlimit --as=768 MiB
        with self.assertRaises(subprocess.CalledProcessError):
            imgfix._run(["false"])
        with mock.patch.object(imgfix, "PDF_TIMEOUT", 0.3), self.assertRaises(subprocess.TimeoutExpired):
            imgfix._run(["sleep", "30"])
        self.assertLess(time.monotonic() - t0, 10)

    def test_corrupt_meta_is_unavailable_not_recaptioned(self):
        raw = h.png(40, 30)
        i = iid(raw)
        Path(self.dir, "meta").mkdir()
        for bad in (b"{not json", json.dumps({"id": "img_ffffffffffffffff"}).encode(),
                    json.dumps({"id": i, "kind": "image"}).encode()):
            Path(self.dir, "meta", i + ".json").write_bytes(bad)
            cap = FakeCaptioner()
            for _ in range(2):  # stable: the same fixed text every turn, vision model never asked
                self.assertEqual(h.texts(self.rewritten(h.body(h.user(h.image_block(raw))), cap)[1]),
                                 [imgfix.UNAVAILABLE])
            self.assertEqual(cap.calls, [])
            self.assertEqual(Path(self.dir, "meta", i + ".json").read_bytes(), bad)

    def test_write_meta_link_race(self):
        m1 = imgfix.image_meta("img_0000000000000000", "png", b"x", 1, 1, "first")
        with mock.patch.object(Path, "exists", return_value=False):  # both writers get past the exists() check
            self.assertEqual(self.store.write_meta(m1)["caption"], "first")
            self.assertEqual(self.store.write_meta(dict(m1, caption="second"))["caption"], "first")  # loser reads disk
            Path(self.dir, "meta", "img_0000000000000000.json").write_bytes(b"garbage")
            with self.assertRaises(ValueError):  # winner's file unreadable: never fall back to our in-memory meta
                self.store.write_meta(dict(m1, caption="third"))
        self.assertEqual([f for f in os.listdir(os.path.join(self.dir, "meta")) if f.startswith(".tmp")], [])

    def test_concurrent_first_sight_same_image(self):
        raw = h.png(41, 31)
        req = h.dumps(h.body(h.user(h.image_block(raw))))

        async def both():
            return await asyncio.gather(imgfix.rewrite(req, self.store, FakeCaptioner("Caption A.", delay=0.05)),
                                        imgfix.rewrite(req, self.store, FakeCaptioner("Caption B.", delay=0.10)))
        a, b = asyncio.run(both())
        self.assertEqual(a.body, b.body)
        self.assertIn(b"| Caption A.]", a.body)  # the first meta written wins for both
        self.assertEqual(self.files(), [f"img/{iid(raw)}.png", f"meta/{iid(raw)}.json"])

    def test_exif_rotation_in_size(self):
        im = Image.new("RGB", (40, 20), (9, 9, 9))
        exif = im.getexif()
        exif[0x0112] = 6  # shown rotated 90 degrees: 20x40, as vision-mcp and the vision model see it
        buf = io.BytesIO()
        im.save(buf, "JPEG", exif=exif.tobytes())
        t = h.texts(self.rewritten(h.body(h.user(h.image_block(buf.getvalue(), "image/jpeg"))))[1])[0]
        self.assertEqual(PLACEHOLDER.match(t).group(2, 3, 4), ("20", "40", "jpg"))

    def test_only_png_jpeg_gif_webp_decoded(self):
        blocks = []
        for fmt in ("BMP", "TIFF", "ICO"):
            buf = io.BytesIO()
            Image.new("RGB", (8, 8)).save(buf, fmt)
            blocks.append(h.image_block(buf.getvalue(), "image/png"))  # declared png, really something else
        _, out = self.rewritten(h.body(h.user(*blocks)))
        self.assertEqual(h.texts(out), [imgfix.UNAVAILABLE] * 3)
        self.assertEqual(self.files(), [])

    def test_pixel_limit(self):
        self.assertEqual(imgfix.MAX_PIXELS, 50_000_000)  # == vision-mcp Config.max_pixels (shared contract)
        with mock.patch.object(imgfix, "MAX_PIXELS", 40 * 30 - 1):
            self.assertEqual(h.texts(self.rewritten(h.body(h.user(h.image_block(h.png(40, 30)))))[1]),
                             [imgfix.UNAVAILABLE])

    def test_low_disk_space(self):
        known, new = h.png(40, 30), h.png(30, 40)
        t_known = h.texts(self.rewritten(h.body(h.user(h.image_block(known))))[1])
        os.unlink(os.path.join(self.dir, "img", iid(known) + ".png"))  # raw pruned, meta kept
        with mock.patch.object(imgfix, "MIN_FREE", 1 << 60):
            _, out = self.rewritten(h.body(h.user(h.image_block(known), h.image_block(new))))
        self.assertEqual(h.texts(out), t_known + [imgfix.UNAVAILABLE])  # known id keeps its text; new one refused
        self.assertEqual(self.files(), [f"meta/{iid(known)}.json"])

    def test_store_writable(self):
        self.assertTrue(self.store.writable())
        self.assertEqual(sorted(os.listdir(self.dir)), ["img", "meta", "pdf"])
        if os.geteuid() != 0:
            os.chmod(os.path.join(self.dir, "meta"), 0o555)
            try:
                self.assertFalse(self.store.writable())
            finally:
                os.chmod(os.path.join(self.dir, "meta"), 0o755)

    # ------------------------------------------------------------ helpers

    def test_clean_caption(self):
        self.assertEqual(imgfix.clean_caption('  "A cat\n[sits] | here."  '), "A cat (sits) / here.")
        self.assertEqual(imgfix.clean_caption("<think>hmm</think>A dog."), "A dog.")
        self.assertTrue(imgfix.clean_caption(" ".join(["w"] * 50)).endswith("..."))
        self.assertEqual(imgfix.clean_caption(None), "")

    def test_clean_caption_reasoning(self):
        # Reasoning without an opening tag (chat template put <think> in the prompt): keep what follows </think>.
        self.assertEqual(imgfix.clean_caption("Hmm, describe it.\n</think>\n\nA red box on white."),
                         "A red box on white.")
        self.assertEqual(imgfix.clean_caption("<think>a</think>x<think>b</think> A dog."), "A dog.")
        # Unclosed (cut-off) reasoning block is never a caption.
        self.assertEqual(imgfix.clean_caption("<think>Okay, the user wants a one-sentence description"), "")
        self.assertEqual(imgfix.clean_caption("A cat. <think>more"), "A cat.")

    def test_captioner_length_cutoff(self):
        import httpx

        def run(choice):
            api = vision_api.from_env({"VISION_MODEL": "vision-x", "VISION_BASE_URL": "http://q:1"})
            cap = imgfix.VisionCaptioner(api)
            cap.client = httpx.AsyncClient(transport=httpx.MockTransport(
                lambda req: httpx.Response(200, json={"choices": [choice]})))

            async def go():
                try:
                    return await cap(h.png(20, 10))
                finally:
                    await cap.aclose()
            return asyncio.run(go())

        # Cut off at max_tokens (e.g. a thinking model mid-reasoning): no caption rather than a wrong one for good.
        self.assertEqual(run({"finish_reason": "length", "message": {
            "content": "Okay, the user wants a one-sentence description. Let me look at the image: it shows a red"}}),
            "")
        self.assertEqual(run({"finish_reason": "stop", "message": {"content": "A red box."}}), "A red box.")
        self.assertEqual(run({"message": {"content": None}}), "")

    def test_vision_png_downscale(self):
        for raw, size in ((h.png(4000, 1000), (1600, 400)), (h.png(300, 200), (300, 200)),
                          (h.jpeg_cmyk(3200, 1600), (1600, 800))):
            out = imgfix.vision_png(raw)
            with Image.open(io.BytesIO(out)) as im:
                self.assertEqual((im.format, im.size), ("PNG", size))

    def test_caption_request_shape(self):
        api = vision_api.from_env({"VISION_MODEL": "vision-x"})
        req = json.loads(imgfix._caption_request(h.png(2000, 100), api, "short"))
        self.assertEqual(sorted(req), ["max_tokens", "messages", "model", "temperature"])  # nothing extra by default
        self.assertEqual((req["model"], req["temperature"], req["max_tokens"]), ("vision-x", 0, 60))
        part = req["messages"][0]["content"][0]
        self.assertTrue(part["image_url"]["url"].startswith("data:image/png;base64,"))
        self.assertEqual(req["messages"][0]["content"][1]["text"], imgfix.CAPTION_PROMPT)

    def test_caption_request_extra_body(self):
        api = vision_api.from_env({"VISION_MODEL": "vision-x", "VISION_EXTRA_BODY":
                                   '{"chat_template_kwargs":{"enable_thinking":false},"top_k":20,"temperature":0.1}'})
        req = json.loads(imgfix._caption_request(h.png(20, 10), api, "short"))
        self.assertEqual((req["model"], req["chat_template_kwargs"], req["top_k"], req["temperature"]),
                         ("vision-x", {"enable_thinking": False}, 20, 0.1))
        self.assertEqual(len(req["messages"]), 1)
        self.assertNotIn("stream", req)

if __name__ == "__main__":
    unittest.main()
