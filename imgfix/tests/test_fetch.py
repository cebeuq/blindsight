"""FETCH_REMOTE_IMAGES: first-sight fetch + write-once URL records, SSRF guard, limits (local stub server only)."""
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest import mock

import imgfix
from . import helpers as h
from .test_rewrite import PLACEHOLDER, FakeCaptioner, StoreCase, iid

PNG = h.png(24, 18, (0, 128, 255))


class ImageServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self):
        super().__init__(("127.0.0.1", 0), ImageHandler)
        self.hits, self.status = [], {}
        threading.Thread(target=self.serve_forever, daemon=True).start()

    def url(self, path):
        return f"http://127.0.0.1:{self.server_address[1]}{path}"

    def handle_error(self, request, client_address):
        pass  # the client gave up (timeout tests): broken pipes are expected


class ImageHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        self.server.hits.append(self.path)
        port = self.server.server_address[1]
        if path in self.server.status:  # per-test override, e.g. a URL that later starts working
            return self.send(self.server.status[path], b"")
        if path == "/img.png":
            return self.send(200, PNG, {"content-type": "image/png"})
        if path == "/redirect":
            return self.send(302, b"", {"location": "/img.png"})
        if path == "/redirect-private":  # a "public" first hop that redirects to a private address
            return self.send(302, b"", {"location": f"http://127.0.0.2:{port}/img.png"})
        if path == "/loop":
            return self.send(302, b"", {"location": "/loop"})
        if path == "/big":
            return self.send(200, b"x" * 5000, {"content-type": "image/png"})
        if path == "/big-chunked":  # no content-length: the cap applies while streaming
            self.send_response(200)
            self.send_header("transfer-encoding", "chunked")
            self.end_headers()
            for _ in range(10):
                self.wfile.write(b"200\r\n" + b"x" * 512 + b"\r\n")
            self.wfile.write(b"0\r\n\r\n")
            return
        if path == "/gzip":
            return self.send(200, b"\x1f\x8b", {"content-encoding": "gzip"})
        if path == "/text":
            return self.send(200, b"not an image", {"content-type": "text/plain"})
        if path == "/slow":
            time.sleep(2)
            return self.send(200, PNG)
        if path == "/drip":  # headers at once, then one byte every 0.2 s
            self.send_response(200)
            self.send_header("content-length", "100")
            self.end_headers()
            for _ in range(10):
                time.sleep(0.2)
                self.wfile.write(b"x")
                self.wfile.flush()
            return
        self.send(404, b"")

    def send(self, status, body, headers=None):
        self.send_response(status)
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class FetchTest(StoreCase):
    @classmethod
    def setUpClass(cls):
        cls.srv = ImageServer()

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()
        cls.srv.server_close()

    def setUp(self):
        super().setUp()
        self.srv.hits.clear()
        self.srv.status.clear()
        for name, value in (("FETCH_REMOTE_IMAGES", True), ("FETCH_ALLOW_PRIVATE", True)):
            p = mock.patch.object(imgfix, name, value)
            p.start()
            self.addCleanup(p.stop)

    def req(self, *urls):
        return h.body(h.user(*({"type": "image", "source": {"type": "url", "url": u}} for u in urls)))

    def texts_of(self, *urls, cap=None, count_only=False):
        return h.texts(self.rewritten(self.req(*urls), cap or FakeCaptioner("A blue box."), count_only)[1])

    def record(self, url):
        return self.meta(imgfix.url_id(url))

    def failed(self, url, reason):
        return f"[image {imgfix.url_id(url)} | remote image not fetched ({reason}): {url}]"

    # ------------------------------------------------------------

    def test_success_stored_once_and_stable(self):
        url = self.srv.url("/img.png?v=1")
        res = self.run_rw(self.req(url), FakeCaptioner("A blue box."))
        t = h.texts(json.loads(res.body))
        i, u = iid(PNG), imgfix.url_id(url)
        self.assertEqual(PLACEHOLDER.match(t[0]).groups(), (i, "24", "18", "png", "A blue box."))
        self.assertEqual(self.files(), [f"img/{i}.png", f"meta/{i}.json", f"meta/{u}.json"])
        rec = self.record(url)
        self.assertEqual({k: rec[k] for k in ("id", "kind", "url", "target")},
                         {"id": u, "kind": "url", "url": url, "target": i})
        self.assertNotIn("failed", rec)
        self.assertEqual((res.stats["fetched"], res.stats["new"], res.stats["captioned"]), (1, 1, 1))
        self.assertIn(u, res.ids)
        # Later turns (and a restart): the record decides; no refetch, no new caption, byte-identical text.
        self.store = imgfix.Store(self.dir)
        cap = FakeCaptioner("Something else.")
        self.assertEqual(self.run_rw(self.req(url), cap).body, res.body)
        self.assertEqual((len(self.srv.hits), cap.calls), (1, []))
        # The same bytes pasted inline are the same stored image.
        self.assertEqual(h.texts(self.rewritten(h.body(h.user(h.image_block(PNG))))[1]), t)

    def test_every_format_and_duplicates(self):
        url = self.srv.url("/img.png?v=2")
        body = {"model": "m", "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": url}}, {"type": "image_url", "image_url": url}]}],
            "input": [{"role": "user", "content": [{"type": "input_image", "image_url": url}]}]}
        res, out = self.rewritten(body, FakeCaptioner())
        ref = self.texts_of(url)[0]
        self.assertEqual(h.part_texts(out["messages"]), [ref, ref])
        self.assertEqual(out["input"][0]["content"], [{"type": "input_text", "text": ref}])
        self.assertEqual(len(self.srv.hits), 1)  # one fetch per URL per request

    def test_failure_recorded_and_stable(self):
        url = self.srv.url("/missing.png")
        self.assertEqual(self.texts_of(url), [self.failed(url, "http 404")])
        rec = self.record(url)
        self.assertEqual((rec["failed"], rec["error"], rec["url"]), (True, "http 404", url))
        self.srv.status["/missing.png"] = 200  # it works now, but the decision was recorded for good
        self.assertEqual(self.texts_of(url), [self.failed(url, "http 404")])
        self.assertEqual(len(self.srv.hits), 1)

    def test_failure_reasons(self):
        cases = {"/text": "undecodable", "/gzip": "encoded body", "/loop": "too many redirects", "/nope": "http 404"}
        urls = [self.srv.url(p) for p in cases]
        self.assertEqual(self.texts_of(*urls), [self.failed(u, r) for u, r in zip(urls, cases.values())])
        self.assertEqual(sum(1 for x in self.srv.hits if x == "/loop"), 1 + imgfix.FETCH_MAX_REDIRECTS)
        bad = "http://nonexistent.invalid/a.png"  # no DNS answer (and no network in the test container)
        self.assertEqual(self.texts_of(bad), [self.failed(bad, "dns failed")])
        self.assertEqual(self.files(), sorted(f"meta/{imgfix.url_id(u)}.json" for u in urls + [bad]))

    def test_redirect_followed(self):
        url = self.srv.url("/redirect")
        self.assertEqual(PLACEHOLDER.match(self.texts_of(url)[0]).group(1), iid(PNG))
        self.assertEqual(self.srv.hits, ["/redirect", "/img.png"])
        self.assertEqual(self.record(url)["url"], url)

    def test_private_address_refused_unless_allowed(self):
        url = self.srv.url("/img.png?v=3")
        with mock.patch.object(imgfix, "FETCH_ALLOW_PRIVATE", False):
            self.assertEqual(self.texts_of(url), [self.failed(url, "blocked address")])
            for bad in ("127.0.0.1", "10.1.2.3", "172.16.0.1", "192.168.1.1", "169.254.169.254", "100.64.0.1",
                        "0.0.0.0", "224.0.0.1", "240.0.0.1", "::1", "fe80::1", "fc00::1", "::ffff:127.0.0.1",
                        "2002:7f00:1::", "ff02::1", "::"):
                self.assertFalse(imgfix.address_ok(bad), bad)
            for good in ("1.1.1.1", "8.8.8.8", "2606:4700:4700::1111"):
                self.assertTrue(imgfix.address_ok(good), good)
        self.assertEqual(self.srv.hits, [])  # refused before connecting
        other = self.srv.url("/img.png?v=4")
        self.assertEqual(PLACEHOLDER.match(self.texts_of(other)[0]).group(1), iid(PNG))  # FETCH_ALLOW_PRIVATE=true

    def test_redirect_to_private_refused(self):
        url = self.srv.url("/redirect-private")
        public = mock.patch.object(imgfix, "address_ok", lambda ip: ip == "127.0.0.1")  # pretend 127.0.0.1 is public
        with public:
            self.assertEqual(self.texts_of(url), [self.failed(url, "blocked address")])
        self.assertEqual(self.srv.hits, ["/redirect-private"])  # the private hop was never contacted

    def test_size_cap(self):
        with mock.patch.object(imgfix, "MAX_IMAGE_BYTES", 1000):
            urls = [self.srv.url("/big"), self.srv.url("/big-chunked")]
            self.assertEqual(self.texts_of(*urls), [self.failed(u, "too large") for u in urls])

    def test_timeouts(self):
        urls = [self.srv.url("/slow"), self.srv.url("/drip")]
        t0 = time.monotonic()
        with mock.patch.object(imgfix, "FETCH_TIMEOUT", 0.5):
            self.assertEqual(self.texts_of(*urls), [self.failed(u, "timeout") for u in urls])
        self.assertLess(time.monotonic() - t0, 1.5)  # concurrent, each bounded by its total deadline

    def test_per_request_limit(self):
        urls = [self.srv.url(f"/img.png?n={k}") for k in range(3)]
        with mock.patch.object(imgfix, "FETCH_MAX_PER_REQUEST", 2):
            ts = self.texts_of(*urls, urls[2], urls[0])
        self.assertEqual([PLACEHOLDER.match(t).group(1) for t in ts[:2]], [iid(PNG)] * 2)
        self.assertEqual(ts[2:], [self.failed(urls[2], "limit"), self.failed(urls[2], "limit"), ts[0]])
        self.assertEqual(len(self.srv.hits), 2)
        self.assertEqual(self.texts_of(urls[2]), [self.failed(urls[2], "limit")])  # recorded for good

    def test_count_only_never_fetches(self):
        url = self.srv.url("/img.png?v=5")
        not_fetched = f"[image {imgfix.url_id(url)} | remote image not fetched: {url}]"
        self.assertEqual(self.texts_of(url, count_only=True), [not_fetched])
        self.assertEqual((self.srv.hits, self.files()), ([], []))
        real = self.texts_of(url)
        self.assertEqual(self.texts_of(url, count_only=True), real)  # an existing record is read (read-only)
        self.assertEqual(len(self.srv.hits), 1)

    def test_disabled_never_fetches(self):
        url = self.srv.url("/img.png?v=6")
        with mock.patch.object(imgfix, "FETCH_REMOTE_IMAGES", False):
            self.assertEqual(self.texts_of(url), [f"[image {imgfix.url_id(url)} | remote image not fetched: {url}]"])
        self.assertEqual((self.srv.hits, self.files()), ([], []))
        ftp = "ftp://example.com/a.png"  # never fetched (http/https only), no record
        self.assertEqual(self.texts_of(ftp), [f"[image {imgfix.url_id(ftp)} | remote image not fetched: {ftp}]"])
        self.assertEqual(self.files(), [])

    def test_broken_record_is_unavailable(self):
        url = self.srv.url("/img.png?v=7")
        (h_dir := self.store.root / "meta").mkdir(parents=True)
        (h_dir / f"{imgfix.url_id(url)}.json").write_bytes(b"{broken")
        for _ in range(2):
            self.assertEqual(self.texts_of(url), [imgfix.UNAVAILABLE])
        self.assertEqual(self.srv.hits, [])
