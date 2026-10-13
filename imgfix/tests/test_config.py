"""Configuration parsing, ASK_HINT texts, CAPTION_MODE (with stub vision models of both wire formats). No network."""
import asyncio
import json
import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import imgfix
import vision_api
from . import helpers as h
from .test_rewrite import FakeCaptioner, StoreCase, iid

APP = str(Path(__file__).resolve().parents[1])
PROBE = ("import json, imgfix as m; print(json.dumps({'errors': m.CONFIG_ERRORS, 'mode': m.CAPTION_MODE, "
         "'timeout': m.CAPTION_TIMEOUT, 'hint': m.ASK_HINT, 'ask_url': m.ASK_URL, 'fetch': m.FETCH_REMOTE_IMAGES, "
         "'private': m.FETCH_ALLOW_PRIVATE, 'passthrough': m.PASSTHROUGH_MODELS, 'min_free': m.MIN_FREE, "
         "'text': m.TEXT and [m.TEXT.scheme, m.TEXT.host, m.TEXT.port, m.TEXT.prefix, m.TEXT.authority], "
         "'captioner': m.addons[0].captioner is not None}))")
BASE_ENV = {"VISION_MODEL": "vision-x", "PATH": os.environ.get("PATH", ""), "HOME": "/tmp"}


def load_config(**env):
    """imgfix's module-level configuration under env (a fresh interpreter: config is read once at import)."""
    r = subprocess.run([sys.executable, "-c", PROBE], cwd=APP, env=dict(BASE_ENV, **env), capture_output=True,
                       text=True, timeout=60, check=True)
    return json.loads(r.stdout)


class ParseTest(unittest.TestCase):
    def test_text_url(self):
        # The same accept/reject set as router/entrypoint.sh (checked against it on a shared corpus).
        def t(s):
            u = imgfix.parse_text_url(s)
            return u.scheme, u.host, u.port, u.prefix, u.authority
        self.assertEqual(t(""), ("http", "host.docker.internal", 8000, "", "host.docker.internal:8000"))
        self.assertEqual(t(" http://host.docker.internal:8000/ "), ("http", "host.docker.internal", 8000, "",
                                                                    "host.docker.internal:8000"))
        for url, want in (
                ("https://api.deepseek.com/anthropic",
                 ("https", "api.deepseek.com", 443, "/anthropic", "api.deepseek.com")),
                ("https://openrouter.ai/api/", ("https", "openrouter.ai", 443, "/api", "openrouter.ai")),
                ("HTTPS://API.Example.COM:443/a/b//", ("https", "api.example.com", 443, "/a/b", "api.example.com")),
                ("http://vllm:80", ("http", "vllm", 80, "", "vllm")),
                ("http://x:00080", ("http", "x", 80, "", "x")),
                ("http://x:", ("http", "x", 80, "", "x")),
                ("http://10.0.0.5:8000/v", ("http", "10.0.0.5", 8000, "/v", "10.0.0.5:8000")),
                ("http://[::1]:8000/p", ("http", "::1", 8000, "/p", "[::1]:8000")),
                ("http://[2001:DB8::1]", ("http", "2001:db8::1", 80, "", "[2001:db8::1]")),
                ("http://my_model.internal:8000", ("http", "my_model.internal", 8000, "", "my_model.internal:8000")),
                ("http://x/a%2Fb~!$&'()*+,;=:@c", ("http", "x", 80, "/a%2Fb~!$&'()*+,;=:@c", "x")),
                ("https://localhost:8443/anthropic", ("https", "localhost", 8443, "/anthropic", "localhost:8443"))):
            self.assertEqual(t(url), want, url)
        for bad, msg in (("ftp://x", "must start with http"), ("host:8000", "must look like"), ("http://", "no host"),
                         ("http:///p", "no host"), ("http://x/v1?key=1", "?query"), ("http://x/#f", "#fragment"),
                         ("http://x:99999", "not in 1-65535"), ("http://x:0", "not in 1-65535"),
                         ("http://x:port", "must be a number"), ("http://x:8000:9", "in brackets"),
                         ("http://bad host", "whitespace"), ("http://x\x00", "control"), ("http://-x", "valid host"),
                         ("http://a..b", "valid host"), ("http://a.", "valid host"), ("http://.a", "valid host"),
                         ("http://127.1", "IPv4"), ("http://010.0.0.1", "IPv4"), ("http://256.0.0.1", "IPv4"),
                         ("http://1234", "IPv4"), ("http://x/a b", "whitespace"), ('http://x/a"b', "percent-encoded"),
                         ("http://x/a%zz", "two hex digits"), ("http://x/a%2", "two hex digits"),
                         ("http://[zz::1]", "not an IPv6"), ("http://[::1", "unclosed"),
                         ("http://[::1]x", "after the ]"),
                         ("http://[fe80::1%25eth0]", "zone ids"), ("https://10.0.0.5/v", "https needs a host name"),
                         ("https://[2001:db8::1]/v", "https needs a host name"), ("http://é.example", "valid host")):
            with self.assertRaisesRegex(ValueError, f"^TEXT_URL.*{re.escape(msg)}", msg=bad):
                imgfix.parse_text_url(bad)
        # Never echoed: credentials or a query (the URL may hold a secret in the wrong place).
        for bad in ("https://user:SECRET1@api.example.com/x", "https://api.example.com/x?key=SECRET1",
                    "https://api.example.com/x#SECRET1", "http://x:SECRET1"):
            with self.assertRaises(ValueError) as cm:
                imgfix.parse_text_url(bad)
            self.assertNotIn("SECRET1", str(cm.exception))

    def test_ask_url(self):
        self.assertEqual(imgfix.parse_ask_url(""), "http://127.0.0.1:8102/ask")
        for ok in ("http://127.0.0.1:9000/ask", "https://vision.example.com/blindsight/ask", "http://[::1]:8102/ask"):
            self.assertEqual(imgfix.parse_ask_url(ok), ok)
        for bad in ("http://x/ask?a=1&b=2", "http://x/a b", "http://x/$(id)", "ftp://x/ask", "http://x/'", "x/ask",
                    "http://x:0/ask"):
            with self.assertRaisesRegex(ValueError, "^ASK_URL must be", msg=bad):
                imgfix.parse_ask_url(bad)

    def test_small_parsers(self):
        for v, want in (("", False), ("0", False), ("false", False), ("No", False), ("off", False), ("1", True),
                        (" TRUE ", True), ("yes", True), ("on", True)):
            self.assertIs(imgfix.parse_bool("X", v), want)
        with self.assertRaisesRegex(ValueError, "^X must be true or false; got 'maybe'"):
            imgfix.parse_bool("X", "maybe")
        self.assertEqual(imgfix.parse_choice("CAPTION_MODE", "", imgfix.CAPTION_MODES), "short")
        self.assertEqual(imgfix.parse_choice("CAPTION_MODE", " Detailed ", imgfix.CAPTION_MODES), "detailed")
        with self.assertRaisesRegex(ValueError, "^CAPTION_MODE must be one of short, detailed, off; got 'long'"):
            imgfix.parse_choice("CAPTION_MODE", "long", imgfix.CAPTION_MODES)
        self.assertEqual(imgfix.parse_number("T", "", 8.0, True), 8.0)
        self.assertEqual(imgfix.parse_number("T", "2.5", 8.0, True), 2.5)
        self.assertEqual(imgfix.parse_number("G", "0", 5.0), 0.0)
        self.assertEqual(imgfix.parse_number("G", "1e6", 5.0, maximum=1e6), 1e6)
        for bad, pos in (("0", True), ("-1", False), ("nan", False), ("inf", True), ("x", False), ("1e300", False)):
            with self.assertRaises(ValueError, msg=bad):
                imgfix.parse_number("T", bad, 1.0, pos, maximum=1e6)
        self.assertEqual(imgfix.parse_patterns(" claude-*, ,gpt-4o ,"), ("claude-*", "gpt-4o"))
        self.assertEqual(imgfix.parse_patterns(None), ())

    def test_ca_file(self):
        https, http = imgfix.parse_text_url("https://api.example.com"), imgfix.parse_text_url("http://x:8000")
        self.assertIsNone(imgfix.check_ca_file(" ", http))
        with tempfile.TemporaryDirectory() as d:
            ca, _, key = h.make_ca(d)
            self.assertEqual(imgfix.check_ca_file(ca, https), ca)
            self.assertEqual(imgfix.check_ca_file(ca, None), ca)  # TEXT_URL itself invalid: still checked
            with self.assertRaisesRegex(ValueError, "^TEXT_CA_FILE is set, but TEXT_URL is not https://"):
                imgfix.check_ca_file(ca, http)
            empty = os.path.join(d, "empty.pem")
            Path(empty).write_text("")
            for bad in (os.path.join(d, "missing.pem"), empty, key):
                with self.assertRaisesRegex(ValueError, "^TEXT_CA_FILE .* is not a readable PEM file", msg=bad):
                    imgfix.check_ca_file(bad, https)


class ModuleConfigTest(unittest.TestCase):
    """Env -> module configuration, as mitmdump loads it (errors make ImgFix.load exit 2: see test_proxy)."""

    def test_defaults(self):
        c = load_config()
        self.assertEqual(c["errors"], [])
        self.assertEqual((c["mode"], c["timeout"], c["hint"], c["ask_url"], c["fetch"], c["private"], c["passthrough"]),
                         ("short", 8.0, "mcp", "http://127.0.0.1:8102/ask", False, False, []))
        self.assertEqual(c["text"], ["http", "host.docker.internal", 8000, "", "host.docker.internal:8000"])
        self.assertEqual(c["min_free"], 5 << 30)
        self.assertTrue(c["captioner"])

    def test_values(self):
        c = load_config(CAPTION_MODE="detailed", ASK_HINT="http", ASK_URL="http://10.0.0.5:8102/ask",
                        FETCH_REMOTE_IMAGES="true", FETCH_ALLOW_PRIVATE="1", PASSTHROUGH_MODELS="claude-*,gpt-4o",
                        TEXT_URL="https://api.moonshot.ai/anthropic", IMGFIX_MIN_FREE_GB="0.5")
        self.assertEqual(c["errors"], [])
        self.assertEqual((c["mode"], c["timeout"], c["hint"], c["ask_url"], c["fetch"], c["private"], c["passthrough"]),
                         ("detailed", 45.0, "http", "http://10.0.0.5:8102/ask", True, True, ["claude-*", "gpt-4o"]))
        self.assertEqual(c["text"], ["https", "api.moonshot.ai", 443, "/anthropic", "api.moonshot.ai"])
        self.assertEqual(c["min_free"], 1 << 29)
        c = load_config(CAPTION_MODE="off", CAPTION_TIMEOUT="3")
        self.assertEqual((c["errors"], c["mode"], c["timeout"], c["captioner"]), ([], "off", 3.0, False))
        # ASK_URL is only checked when the http hint uses it
        self.assertEqual(load_config(ASK_URL="not a url")["errors"], [])

    def test_errors(self):
        c = load_config(TEXT_URL="ftp://x", CAPTION_MODE="loud", CAPTION_TIMEOUT="-1", ASK_HINT="smoke",
                        FETCH_REMOTE_IMAGES="maybe", FETCH_ALLOW_PRIVATE="2", TEXT_CA_FILE="/nonexistent.pem",
                        VISION_API="gemini", IMGFIX_MIN_FREE_GB="lots")
        self.assertEqual([e.split(" ", 1)[0] for e in c["errors"]],
                         ["TEXT_URL", "TEXT_CA_FILE", "VISION_API", "CAPTION_MODE", "CAPTION_TIMEOUT", "ASK_HINT",
                          "FETCH_REMOTE_IMAGES", "FETCH_ALLOW_PRIVATE", "IMGFIX_MIN_FREE_GB"])
        self.assertEqual(load_config(ASK_HINT="http", ASK_URL="http://x/ask?a=1&b=2")["errors"][0][:22],
                         "ASK_URL must be an htt")
        for env, needle in (({"IMGFIX_MIN_FREE_GB": "1e300"}, "IMGFIX_MIN_FREE_GB must be a number >= 0 and <= 1e+06"),
                            ({"IMGFIX_MIN_FREE_GB": "-inf"}, "IMGFIX_MIN_FREE_GB must be"),
                            ({"CAPTION_TIMEOUT": "1e9"}, "CAPTION_TIMEOUT must be a number > 0 and <= 600"),
                            ({"TEXT_URL": "http://x:8000", "TEXT_CA_FILE": "/any.pem"}, "TEXT_CA_FILE is set, but"),
                            ({"TEXT_URL": "https://user:SECRET2@x/?k=SECRET2"}, "TEXT_URL may not have a ?query")):
            errors = load_config(**env)["errors"]
            self.assertEqual(len(errors), 1, errors)
            self.assertTrue(errors[0].startswith(needle), errors)
            self.assertNotIn("SECRET2", errors[0])
        env = dict(BASE_ENV)
        del env["VISION_MODEL"]
        r = subprocess.run([sys.executable, "-c", PROBE], cwd=APP, env=env, capture_output=True, text=True, timeout=60)
        self.assertTrue(json.loads(r.stdout)["errors"][0].startswith("VISION_MODEL is not set"))


class HintTest(StoreCase):
    def texts_for(self, hint, raw, pdf, ask_url="http://127.0.0.1:8102/ask"):
        with mock.patch.object(imgfix, "ASK_HINT", hint), mock.patch.object(imgfix, "ASK_URL", ask_url):
            imgfix.PDF_CACHE.clear()
            return h.texts(self.rewritten(h.body(h.user(h.image_block(raw), h.pdf_block(pdf))), FakeCaptioner())[1])

    def test_hints_exact(self):
        raw, pdf = h.png(40, 30), h.text_pdf(["Page text"])
        i, p = iid(raw), iid(pdf, "pdf_")
        mcp = self.texts_for("mcp", raw, pdf)
        self.assertEqual(mcp[0], f'[image {i} | 40x30 png | A red square.] The model cannot see images. To inspect '
                                 f'this one, call the ask_image tool with image_id="{i}" and a specific question.')
        self.assertEqual(mcp[1], f'[pdf {p} | 1 pages] Text extracted with pdftotext (page images: ask_image with '
                                 f'image_id="{p}" and page=<n>):\nPage text')
        http = self.texts_for("http", raw, pdf, "http://127.0.0.1:9999/ask")
        self.assertEqual(http[0], f'[image {i} | 40x30 png | A red square.] The model cannot see images. To inspect '
                                  f'this one, run: curl -s -G http://127.0.0.1:9999/ask --data-urlencode image_id={i} '
                                  f'--data-urlencode "question=<your question>"')
        self.assertEqual(http[1], f'[pdf {p} | 1 pages] Text extracted with pdftotext (page images: curl -s -G '
                                  f'http://127.0.0.1:9999/ask --data-urlencode image_id={p} --data-urlencode page=<n> '
                                  f'--data-urlencode "question=..."):\nPage text')
        none = self.texts_for("none", raw, pdf)
        self.assertEqual(none, [f"[image {i} | 40x30 png | A red square.] The model cannot see images.",
                                f"[pdf {p} | 1 pages] Text extracted with pdftotext:\nPage text"])
        # The store keeps the canonical (mcp) PDF text whatever the hint: switching back gives the v1 text again.
        self.assertEqual(Path(self.dir, "meta", p + ".txt").read_text(), mcp[1])
        self.assertEqual(self.texts_for("mcp", raw, pdf), mcp)

    def test_render_pdf_leaves_unknown_text(self):
        with mock.patch.object(imgfix, "ASK_HINT", "none"):
            for s in ("[pdf pdf_0123456789abcdef | 2 pages] something else", "free text"):
                self.assertEqual(imgfix.render_pdf(s), s)
            canon = imgfix.pdf_text("pdf_0123456789abcdef", 2, "x" * 70000)
            self.assertEqual(imgfix.render_pdf(canon), "[pdf pdf_0123456789abcdef | 2 pages] Text extracted with "
                                                       "pdftotext:\n" + "x" * 60000 + "\n[... truncated]")


class CaptionModeTest(StoreCase):
    APIS = {"openai": {"VISION_MODEL": "vision-x", "VISION_BASE_URL": "http://vision:8001/v1", "VISION_API_KEY": "k1"},
            "anthropic": {"VISION_API": "anthropic", "VISION_MODEL": "claude-x", "VISION_API_KEY": "k2"}}

    def captioner(self, kind, reply, status=200):
        import httpx
        seen = []

        def handle(req):
            seen.append(req)
            return httpx.Response(status, json=reply)
        cap = imgfix.VisionCaptioner(vision_api.from_env(self.APIS[kind]))
        cap.client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
        return cap, seen

    def caption(self, cap, raw):
        async def go():
            try:
                return await cap(raw)
            finally:
                await cap.aclose()
        return asyncio.run(go())

    def test_request_per_mode_and_kind(self):
        for kind in ("openai", "anthropic"):
            api = vision_api.from_env(self.APIS[kind])
            for mode, prompt, tokens in (("short", imgfix.CAPTION_PROMPT, 60),
                                         ("detailed", imgfix.CAPTION_DETAILED_PROMPT, 600)):
                req = json.loads(imgfix._caption_request(h.png(30, 20), api, mode))
                self.assertEqual((req["max_tokens"], req["temperature"], req["messages"][-1]["content"][1]["text"]),
                                 (tokens, 0, prompt), (kind, mode))
        self.assertIn("Quote all visible text exactly", imgfix.CAPTION_DETAILED_PROMPT)

    def test_stub_vision_models(self):
        raw = h.png(30, 20)
        cap, seen = self.captioner("openai", {"choices": [{"message": {"content": "<think>x</think>A box."},
                                                           "finish_reason": "stop"}]})
        self.assertEqual(imgfix.clean_caption(self.caption(cap, raw)), "A box.")
        self.assertEqual((str(seen[0].url), seen[0].headers["authorization"]),
                         ("http://vision:8001/v1/chat/completions", "Bearer k1"))
        cap, seen = self.captioner("anthropic", {"content": [{"type": "text", "text": "A red box."}],
                                                 "stop_reason": "end_turn"})
        self.assertEqual(self.caption(cap, raw), "A red box.")
        r = seen[0]
        self.assertEqual((str(r.url), r.headers["x-api-key"], r.headers["anthropic-version"]),
                         ("https://api.anthropic.com/v1/messages", "k2", "2023-06-01"))
        body = json.loads(r.content)
        self.assertEqual((body["model"], body["max_tokens"], body["messages"][0]["content"][0]["source"]["media_type"]),
                         ("claude-x", 60, "image/png"))
        cap, _ = self.captioner("anthropic", {"content": [{"type": "text", "text": "A long"}],
                                              "stop_reason": "max_tokens"})
        self.assertEqual(self.caption(cap, raw), "")  # truncated: no caption for good
        cap, _ = self.captioner("anthropic", {"error": "x"}, status=500)
        with self.assertRaises(Exception):
            self.caption(cap, raw)

    def test_detailed_mode_rewrite(self):
        long = "Text \"Hello | world\" [x] " + "word " * 700
        with mock.patch.object(imgfix, "CAPTION_MODE", "detailed"):
            _, out = self.rewritten(h.body(h.user(h.image_block(h.png(30, 20)))), FakeCaptioner(long))
        cap = h.texts(out)[0].split(" | ", 2)[2].split("] The model", 1)[0]
        self.assertTrue(imgfix.DETAILED_CAPTION_CHARS - 5 < len(cap) <= imgfix.DETAILED_CAPTION_CHARS, len(cap))
        self.assertTrue(cap.startswith('Text "Hello / world" (x) word word') and cap.endswith("..."))
        self.assertEqual(imgfix.clean_caption("  A\n\nshort  one. ", detailed=True), "A short one.")
        self.assertEqual(imgfix.clean_caption("<think>a</think> 'Quoted' ", detailed=True), "'Quoted'")

    def test_off_mode_no_preview(self):
        # CAPTION_MODE=off: ImgFix has no captioner (ModuleConfigTest), so rewrite gets caption=None.
        _, out = self.rewritten(h.body(h.user(h.image_block(h.png(30, 20)))), None)
        self.assertIn("| no preview]", h.texts(out)[0])


if __name__ == "__main__":
    unittest.main()
