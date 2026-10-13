"""Unit tests: id validation, store lookup, region/crop math, image preparation, PDF page rendering, prompt helpers,
configuration from the environment, the access-log query filter."""

import io
import logging
import os
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image

import server
from server import AskError
from tests.util import HEX, IMG_ID, PDF_ID, QUAD, Store, quad_image, vision


def png(data: bytes) -> Image.Image:
    im = Image.open(io.BytesIO(data))
    assert im.format == "PNG"
    im.load()
    return im


def near(a, b, tol=12) -> bool:
    """Pillow embeds PDF page images as JPEG, so rendered colors are only approximately exact."""
    return all(abs(x - y) <= tol for x, y in zip(a, b))


class TestIds(unittest.TestCase):
    def test_valid(self):
        for i in (IMG_ID, PDF_ID, "img_" + "0" * 16, "pdf_" + "f" * 16):
            self.assertTrue(server.valid_id(i), i)

    def test_invalid(self):
        bad = ["", "img_", "img_" + HEX[:15], "img_" + HEX + "0", "img_" + HEX.upper(), IMG_ID + "\n", " " + IMG_ID,
               IMG_ID + ".png", "img_../../etc/passwd", "../img/" + IMG_ID, "url_" + HEX, "doc_" + HEX, "IMG_" + HEX,
               "img_" + HEX[:15] + "g", None, 123]
        for i in bad:
            self.assertFalse(server.valid_id(i), repr(i))


class TestStore(unittest.TestCase):
    def setUp(self):
        self.s = Store()

    def tearDown(self):
        self.s.close()

    def test_lookup_each_ext(self):
        for n, ext in enumerate(server.IMG_EXTS):
            iid = f"img_{n:016x}"
            p = self.s.add_image(iid, Image.new("RGB", (4, 4)), ext)
            self.assertEqual(server.find_image(self.s.root, iid), p)

    def test_pdf_and_missing(self):
        p = self.s.add_pdf(PDF_ID)
        self.assertEqual(server.find_image(self.s.root, PDF_ID), p)
        self.assertIsNone(server.find_image(self.s.root, IMG_ID))
        self.assertIsNone(server.find_image(self.s.root, "pdf_" + "1" * 16))

    def test_directory_is_not_an_image(self):
        (self.s.root / "img" / f"{IMG_ID}.png").mkdir()
        self.assertIsNone(server.find_image(self.s.root, IMG_ID))

    def test_prepare_unknown_and_page_on_image(self):
        cfg = server.Config(store=self.s.root)
        with self.assertRaisesRegex(AskError, "unknown image_id"):
            server.prepare(cfg, IMG_ID, None, None)
        self.s.add_image(IMG_ID, Image.new("RGB", (8, 8)))
        with self.assertRaisesRegex(AskError, "only to pdf_"):
            server.prepare(cfg, IMG_ID, 2, None)
        self.assertEqual(png(server.prepare(cfg, IMG_ID, 1, None)).size, (8, 8))

    @unittest.skipIf(os.geteuid() == 0, "root ignores file modes")
    def test_unreadable_file(self):
        self.s.add_image(IMG_ID, Image.new("RGB", (8, 8))).chmod(0o000)
        with self.assertRaisesRegex(AskError, "not readable by vision-mcp"):
            server.prepare(server.Config(store=self.s.root), IMG_ID, None, None)


class TestRegion(unittest.TestCase):
    def test_parse(self):
        self.assertEqual(server.parse_region("0,0,1,1"), (0, 0, 1, 1))
        self.assertEqual(server.parse_region(" 0.25, 0.5 ,0.75,1 "), (0.25, 0.5, 0.75, 1.0))
        self.assertEqual(server.parse_region("[0.1, 0.2, 0.5, 0.6]"), (0.1, 0.2, 0.5, 0.6))
        self.assertEqual(server.parse_region("0.1 0.2 0.5 0.6"), (0.1, 0.2, 0.5, 0.6))

    def test_parse_rejects(self):
        for r in ("1,0,0,1", "0,0,0,1", "0,0,1.5,1", "-0.1,0,1,1", "a,b,c,d", "0,0,1", "0,0,1,1,1", "nan,0,1,1",
                  "0,0,inf,1", "10%,0,50%,1", ""):
            with self.assertRaises(AskError, msg=r):
                server.parse_region(r)

    def test_crop_box(self):
        self.assertEqual(server.crop_box(1000, 500, (0.1, 0.2, 0.3, 0.4)), (100, 100, 300, 200))
        self.assertEqual(server.crop_box(1000, 500, (0, 0, 1, 1)), (0, 0, 1000, 500))
        self.assertEqual(server.crop_box(10, 10, (0.15, 0.15, 0.25, 0.25)), (1, 1, 3, 3))  # rounds outward
        self.assertEqual(server.crop_box(1000, 1000, (0.3, 0.7, 0.6, 0.9)), (300, 700, 600, 900))  # no float creep
        self.assertEqual(server.crop_box(100, 100, (0.5, 0.5, 0.5001, 0.5001)), (50, 50, 51, 51))  # >= 1 px
        self.assertEqual(server.crop_box(100, 100, (0.999, 0.999, 1, 1)), (99, 99, 100, 100))


class TestPrepareImage(unittest.TestCase):
    def setUp(self):
        self.s = Store()

    def tearDown(self):
        self.s.close()

    def load(self, p, region=None, max_pixels=60_000_000):
        return png(server.load_image(p, region, max_pixels))

    def test_downscale_long_edge(self):
        out = self.load(self.s.add_image(IMG_ID, quad_image(4000, 1000)))
        self.assertEqual(out.size, (1600, 400))
        self.assertEqual(out.mode, "RGB")
        out = self.load(self.s.add_image(IMG_ID, quad_image(300, 200)))
        self.assertEqual(out.size, (300, 200))  # small images are not upscaled without a region

    def test_region_crop_and_zoom(self):
        p = self.s.add_image(IMG_ID, quad_image(1000, 1000))
        out = self.load(p, (0.5, 0.0, 1.0, 0.5))  # top-right quarter: 500x500 -> zoomed to 1024
        self.assertEqual(out.size, (1024, 1024))
        self.assertEqual(out.getpixel((512, 512)), QUAD["tr"])
        out = self.load(p, (0.0, 0.5, 0.01, 0.51))  # 10x10 px -> zoom capped at 4x
        self.assertEqual(out.size, (40, 40))
        self.assertEqual(out.getpixel((20, 20)), QUAD["bl"])
        big = self.load(self.s.add_image("img_" + "1" * 16, quad_image(8000, 4000)), (0.5, 0.5, 1, 1))
        self.assertEqual(big.size, (1600, 800))  # 4000x2000 crop -> downscaled
        self.assertEqual(big.getpixel((800, 400)), QUAD["br"])

    def test_transparency_flattened_on_white(self):
        im = Image.new("RGBA", (20, 10), (0, 0, 0, 0))
        im.paste((0, 0, 0, 255), (0, 0, 10, 10))  # left half opaque black, right half transparent black
        out = self.load(self.s.add_image(IMG_ID, im))
        self.assertEqual(out.mode, "RGB")
        self.assertEqual(out.getpixel((5, 5)), (0, 0, 0))
        self.assertEqual(out.getpixel((15, 5)), (255, 255, 255))

    def test_animated_gif_first_frame_and_formats(self):
        frames = [Image.new("RGB", (16, 16), c) for c in ((255, 0, 0), (0, 0, 255))]
        p = self.s.root / "img" / f"{IMG_ID}.gif"
        frames[0].save(p, save_all=True, append_images=frames[1:], duration=100, loop=0)
        self.assertEqual(self.load(p).getpixel((8, 8)), (255, 0, 0))
        for n, (ext, mode) in enumerate((("jpg", "CMYK"), ("webp", "RGB"), ("png", "L"), ("png", "P"))):
            q = self.s.add_image(f"img_{n + 10:016x}", Image.new(mode, (30, 20)), ext)
            self.assertEqual(self.load(q).size, (30, 20), (ext, mode))

    def test_exif_orientation_applied(self):
        im = quad_image(40, 20)
        exif = im.getexif()
        exif[0x0112] = 6  # rotate 90 CW on display
        out = self.load(self.s.add_image(IMG_ID, im, "jpg", exif=exif.tobytes(), quality=95))
        self.assertEqual(out.size, (20, 40))

    def test_big_jpeg_draft_scaled(self):
        p = self.s.add_image(IMG_ID, quad_image(6400, 4800), "jpg", quality=80)  # 30.7 Mpx
        out = self.load(p, max_pixels=10_000_000)  # decoded at 1/2 scale (7.7 Mpx) -> fits the limit
        self.assertEqual(out.size, (1600, 1200))
        with self.assertRaisesRegex(AskError, "too large"):  # a half-size crop needs full resolution -> 30.7 Mpx
            self.load(p, (0, 0, 0.5, 0.5), max_pixels=10_000_000)

    def test_corrupt_and_oversized(self):
        p = self.s.root / "img" / f"{IMG_ID}.png"
        p.write_bytes(b"\x89PNG\r\n\x1a\n" + b"garbage" * 10)
        with self.assertRaisesRegex(AskError, "could not decode"):
            self.load(p)
        q = self.s.add_image("img_" + "2" * 16, Image.new("RGB", (100, 100)))
        with self.assertRaisesRegex(AskError, "too large"):
            self.load(q, max_pixels=9_999)
        for fmt in ("BMP", "TIFF"):  # only PNG/JPEG/GIF/WEBP decoders are tried, whatever the file is called
            r = self.s.root / "img" / f"img_{'3' * 16}.png"
            Image.new("RGB", (8, 8)).save(r, format=fmt)
            with self.assertRaisesRegex(AskError, "could not decode", msg=fmt):
                self.load(r)

    def test_pixel_limit_matches_imgfix(self):
        self.assertEqual(server.Config().max_pixels, 50_000_000)  # == imgfix MAX_PIXELS (shared contract)


class TestPdf(unittest.TestCase):
    def setUp(self):
        self.s = Store()
        self.pdf = self.s.add_pdf(PDF_ID)
        self.cfg = server.Config(store=self.s.root)

    def tearDown(self):
        self.s.close()

    def test_info(self):
        self.assertEqual(server.pdf_info(self.pdf, 2), (2, 360.0, 720.0))
        with self.assertRaisesRegex(AskError, "out of range: this PDF has 2 pages"):
            server.pdf_info(self.pdf, 3)

    def test_render_pages(self):
        p1 = png(server.prepare(self.cfg, PDF_ID, None, None))  # default page 1 at 110 dpi: 720x360 pt -> 1100x550
        self.assertEqual(p1.size, (1100, 550))
        self.assertTrue(near(p1.getpixel((825, 137)), QUAD["tr"]), p1.getpixel((825, 137)))
        p2 = png(server.prepare(self.cfg, PDF_ID, 2, None))  # 360x720 pt -> 550x1100, solid blue
        self.assertEqual(p2.size, (550, 1100))
        self.assertTrue(near(p2.getpixel((275, 550)), (0, 0, 255)), p2.getpixel((275, 550)))
        with self.assertRaisesRegex(AskError, "out of range"):
            server.prepare(self.cfg, PDF_ID, 9, None)

    def test_region_renders_at_higher_dpi(self):
        self.assertEqual(server.pdf_dpi(720, 360, None, 40_000_000), 110)
        self.assertEqual(server.pdf_dpi(720, 360, (0, 0, 0.5, 0.5), 40_000_000), 220)
        self.assertEqual(server.pdf_dpi(720, 360, (0, 0, 0.01, 0.01), 40_000_000), 440)  # capped at 4x
        self.assertEqual(server.pdf_dpi(14400, 14400, None, 40_000_000), 31)  # 200-inch page: pixel budget wins
        out = png(server.prepare(self.cfg, PDF_ID, 1, (0.5, 0.5, 1, 1)))  # 220 dpi page = 2200x1100 -> crop 1100x550
        self.assertEqual(out.size, (1100, 550))
        self.assertTrue(near(out.getpixel((550, 275)), QUAD["br"]), out.getpixel((550, 275)))

    def test_corrupt_pdf(self):
        self.pdf.write_bytes(b"%PDF-1.4\nnot really a pdf")
        with self.assertRaisesRegex(AskError, "could not read the stored PDF"):
            server.prepare(self.cfg, PDF_ID, 1, None)


class TestHelpers(unittest.TestCase):
    def test_public_url(self):
        self.assertEqual(server.public_url("https://u:pw@v.example:8443/v1/chat/completions?api-version=1"),
                         "https://v.example:8443/v1/chat/completions")
        self.assertEqual(server.public_url("http://[::1]:8001/v1/messages"), "http://[::1]:8001/v1/messages")

    def test_redact(self):
        api = vision("http://v/v1", VISION_API_KEY="sk-secret-key-1", VISION_HEADERS='{"x-org": "org-secret-9", "x-n": "1"}')
        out = server.redact("bad key sk-secret-key-1 (Bearer sk-secret-key-1) org org-secret-9 n=1 application/json", api)
        self.assertEqual(out, "bad key [redacted] ([redacted]) org [redacted] n=1 application/json")
        anth = vision("http://v", VISION_API="anthropic", VISION_API_KEY="sk-ant-secret")
        self.assertEqual(server.redact("x-api-key sk-ant-secret is invalid", anth), "x-api-key [redacted] is invalid")

    def test_user_prompt(self):
        t = server.user_prompt(PDF_ID, " What is the title? ", "user is debugging", 3, (0, 0, 0.5, 0.25))
        self.assertIn("PDF " + PDF_ID + ", page 3", t)
        self.assertIn("x 0..0.5, y 0..0.25", t)
        self.assertIn("Context from the caller (may be wrong; the image wins):\nuser is debugging", t)
        self.assertTrue(t.endswith("Question:\nWhat is the title?"))
        self.assertNotIn("Context", server.user_prompt(IMG_ID, "q", "  ", None, None))

    def test_clean_answer(self):
        self.assertEqual(server.clean_answer("<think>hmm</think>\n\nRed."), "Red.")
        self.assertEqual(server.clean_answer("hmm</think>Red."), "Red.")
        self.assertEqual(server.clean_answer("  Red.  "), "Red.")

    def test_request_settings(self):
        self.assertEqual((server.TEMPERATURE, server.MAX_TOKENS), (0.2, 700))

    def test_access_log_hides_query(self):
        rec = logging.LogRecord("uvicorn.access", logging.INFO, "", 0, '%s - "%s %s HTTP/%s" %d',
                                ("127.0.0.1:5", "GET", "/ask?image_id=x&question=secret+q", "1.1", 200), None)
        self.assertTrue(server.HIDE_QUERY.filter(rec))
        self.assertEqual(rec.getMessage(), '127.0.0.1:5 - "GET /ask?... HTTP/1.1" 200')
        rec = logging.LogRecord("uvicorn.access", logging.INFO, "", 0, '%s - "%s %s HTTP/%s" %d',
                                ("127.0.0.1:5", "POST", "/mcp", "1.1", 200), None)
        self.assertTrue(server.HIDE_QUERY.filter(rec))
        self.assertEqual(rec.getMessage(), '127.0.0.1:5 - "POST /mcp HTTP/1.1" 200')


ENV_KEYS = ("VISION_API", "VISION_MODEL", "VISION_BASE_URL", "VISION_API_KEY", "VISION_HEADERS", "VISION_TOKEN_PARAM",
            "VISION_EXTRA_BODY", "VISION_TIMEOUT", "STORE_DIR", "ASK_CONCURRENCY", "DECODE_CONCURRENCY",
            "MCP_ALLOWED_HOSTS")


class TestConfigFromEnv(unittest.TestCase):
    def env(self, **kv):
        base = {k: v for k, v in os.environ.items() if k not in ENV_KEYS}
        return mock.patch.dict(os.environ, {**base, **kv}, clear=True)

    def test_defaults(self):
        with self.env(VISION_MODEL="m"):
            cfg = server.Config.from_env()
        self.assertEqual((cfg.store, cfg.vision.kind, cfg.vision.url, cfg.vision.model, cfg.vision_timeout),
                         (Path("/store"), "openai", "http://host.docker.internal:8001/v1/chat/completions", "m", 60.0))
        self.assertEqual(cfg.vision.headers, {"content-type": "application/json"})  # no key -> no auth header
        self.assertEqual((cfg.vision.extra, cfg.concurrency, cfg.decode_concurrency), ({}, 4, 2))
        self.assertEqual(cfg.allowed_hosts, ["127.0.0.1:*", "localhost:*", "vision-mcp:*"])

    def test_values(self):
        with self.env(VISION_MODEL=" m2 ", VISION_BASE_URL="https://v.example/v1", VISION_API_KEY=" sk-1 ",
                      VISION_EXTRA_BODY='{"chat_template_kwargs":{"enable_thinking":false}}', VISION_TIMEOUT="5",
                      ASK_CONCURRENCY="3", DECODE_CONCURRENCY=" 1 ", MCP_ALLOWED_HOSTS="a:*, b:1 ,"):
            cfg = server.Config.from_env()
        self.assertEqual((cfg.vision.model, cfg.vision.url, cfg.vision.headers["authorization"], cfg.vision_timeout),
                         ("m2", "https://v.example/v1/chat/completions", "Bearer sk-1", 5.0))
        self.assertEqual(cfg.vision.extra, {"chat_template_kwargs": {"enable_thinking": False}})
        self.assertEqual((cfg.concurrency, cfg.decode_concurrency), (3, 1))
        self.assertEqual(cfg.allowed_hosts, ["a:*", "b:1"])
        line = server.describe(cfg)
        self.assertEqual(line, "store=/store api=openai url=https://v.example/v1/chat/completions model=m2 auth=set "
                               "headers=authorization extra_body_keys=chat_template_kwargs allowed_hosts=a:*,b:1")
        self.assertNotIn("sk-1", line)  # the key is never logged

    def test_anthropic_and_headers(self):
        with self.env(VISION_MODEL="claude-x", VISION_API="anthropic", VISION_API_KEY="sk-ant-1",
                      VISION_HEADERS='{"anthropic-beta": "b1"}', VISION_TOKEN_PARAM="max_completion_tokens"):
            cfg = server.Config.from_env()
        self.assertEqual((cfg.vision.kind, cfg.vision.url, cfg.vision.token_param),
                         ("anthropic", "https://api.anthropic.com/v1/messages", "max_tokens"))
        self.assertEqual(cfg.vision.headers, {"content-type": "application/json", "anthropic-version": "2023-06-01",
                                              "x-api-key": "sk-ant-1", "anthropic-beta": "b1"})
        self.assertNotIn("sk-ant-1", server.describe(cfg))

    def test_allowed_hosts_off_only_with_star(self):
        for value, want in (("*", []), (" * ", []), ("", ["127.0.0.1:*", "localhost:*", "vision-mcp:*"]),
                            (" , ", ["127.0.0.1:*", "localhost:*", "vision-mcp:*"])):
            with self.env(VISION_MODEL="m", MCP_ALLOWED_HOSTS=value):
                self.assertEqual(server.Config.from_env().allowed_hosts, want, repr(value))

    def test_errors(self):
        for kv, msg in (({}, "VISION_MODEL is not set"), ({"VISION_MODEL": "  "}, "VISION_MODEL is not set"),
                        ({"VISION_EXTRA_BODY": "{bad"}, "VISION_EXTRA_BODY is not valid JSON"),
                        ({"VISION_EXTRA_BODY": "[]"}, "must be a JSON object"),
                        ({"VISION_EXTRA_BODY": '{"model":"x"}'}, "may not set model"),
                        ({"VISION_API": "gemini"}, "VISION_API must be one of"),
                        ({"VISION_BASE_URL": "vlm:8001"}, "VISION_BASE_URL must be an http"),
                        ({"VISION_HEADERS": '{"x": 1}'}, "VISION_HEADERS"),
                        ({"VISION_TOKEN_PARAM": "tokens"}, "VISION_TOKEN_PARAM must be one of"),
                        ({"VISION_TIMEOUT": "soon"}, "VISION_TIMEOUT must be a positive number"),
                        ({"VISION_TIMEOUT": "0"}, "VISION_TIMEOUT must be a positive number"),
                        ({"ASK_CONCURRENCY": "1.5"}, "ASK_CONCURRENCY must be a positive number"),
                        ({"DECODE_CONCURRENCY": "-1"}, "DECODE_CONCURRENCY must be a positive number")):
            kv = {"VISION_MODEL": "m", **kv} if kv else kv
            with self.env(**kv), self.assertRaisesRegex(server.ConfigError, msg, msg=kv):
                server.Config.from_env()


if __name__ == "__main__":
    unittest.main()
