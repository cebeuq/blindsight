# imgfix: mitmproxy reverse proxy (:8080) that swaps image/document/file parts for text before the text-only model
# (see imgfix.py). Base: official multi-arch mitmproxy image pinned by tag+digest (linux/arm64 and linux/amd64).
# It lacks poppler/Pillow/httpx, hence this thin layer.
# Build contexts: this directory, plus a named context "shared" holding vision_api.py (the vision-model wire format,
# shared with vision-mcp): compose build.additional_contexts {shared: ../shared}, or
# docker build --build-context shared=../shared .
# Integration notes (the top-level compose.yaml owns the real service):
#   - runs as uid 1000 (mitmproxy); /store must be writable by it. A fresh named volume first mounted by imgfix
#     inherits the 1000:1000 /store below. Files are 0644 / dirs 0755 so vision-mcp (uid 10001) can read them.
#   - read_only: true works; needs tmpfs /tmp (mitmproxy confdir + count_tokens PDF scratch).
#   - env: TEXT_URL TEXT_CA_FILE PASSTHROUGH_MODELS CAPTION_MODE CAPTION_TIMEOUT ASK_HINT ASK_URL FETCH_REMOTE_IMAGES
#     FETCH_ALLOW_PRIVATE VISION_API VISION_BASE_URL VISION_MODEL VISION_API_KEY VISION_HEADERS VISION_TOKEN_PARAM
#     VISION_EXTRA_BODY IMGFIX_STORE IMGFIX_MIN_FREE_GB (defaults and meaning in imgfix.py / vision_api.py). Any bad
#     value makes mitmdump exit 2 at startup with "imgfix: configuration error: ..." on stderr.
FROM mitmproxy/mitmproxy:12.2.3@sha256:00b77b5d8804c8ad18cb6caefbf9d5849e895e8986c5ce011f4ae30f4385962f AS runtime

# poppler-utils: pdfinfo (page count) + pdftotext, run under prlimit (util-linux, in the base). Version follows
# the digest-pinned Debian trixie base.
RUN apt-get update \
 && apt-get install -y --no-install-recommends poppler-utils \
 && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_ROOT_USER_ACTION=ignore
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-deps --require-hashes --only-binary=:all: -r requirements.txt && pip check

COPY imgfix.py .
COPY --from=shared vision_api.py .
RUN install -d -o mitmproxy -g mitmproxy -m 0755 /store /store/img /store/pdf /store/meta
# Skip the base entrypoint: it runs usermod (needs root + writable /etc/passwd), incompatible with read_only.
USER mitmproxy
ENV HOME=/home/mitmproxy
ENTRYPOINT []
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD ["python3", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/imgfix-health', timeout=4)"]
# The reverse target is a dead placeholder (127.0.0.1:9): the addon routes every request itself (TEXT_URL), so
# a flow it did not route gets a 502, never an unrewritten body at the text model.
# lazy: no eager upstream connection; tcp_timeout 3600 matches HAProxy's 60m (long prefills).
# body_size_limit 64m: bigger requests get 413 instead of being buffered (memory bound; mem_limit 1g).
# Never set stream_large_bodies: a streamed request body would bypass the rewrite.
CMD ["mitmdump", "--mode", "reverse:http://127.0.0.1:9", "--listen-host", "0.0.0.0", "--listen-port", "8080", \
     "--set", "confdir=/tmp/mitmproxy", "--set", "connection_strategy=lazy", "--set", "tcp_timeout=3600", \
     "--set", "body_size_limit=64m", "--set", "flow_detail=0", "-s", "/app/imgfix.py"]

# Test image: runtime + stdlib unittest suite (text/vision models and upstreams are local stubs; needs no network).
FROM runtime AS test
COPY tests/ tests/
CMD ["python3", "-m", "unittest", "discover", "-s", "tests", "-t", ".", "-v"]
