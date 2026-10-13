# vision-mcp: "ask the image" MCP server. Streamable HTTP on :8102/mcp, the same tool as plain HTTP at /ask, GET /health.
# Reads the shared /store volume. Build with the named context "shared" (the repo's shared/ directory: vision_api.py and
# its tests), e.g. compose build.additional_contexts: {shared: ./shared} or docker build --build-context shared=../shared.
# Base is the multi-arch official python image pinned by tag+digest (linux/arm64 and linux/amd64).
FROM python:3.14.8-slim-trixie@sha256:a2b82f3c48559aa0a8446d9af49826b6e2b2016f4cd2afabfe6013ec53729170 AS runtime

# poppler-utils: pdfinfo (page count/size) + pdftoppm (page render). Versions follow the digest-pinned Debian base.
RUN apt-get update \
 && apt-get install -y --no-install-recommends poppler-utils \
 && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_ROOT_USER_ACTION=ignore
WORKDIR /app
# Fully pinned + hashed (see requirements.in / compose.dev.yaml "lock").
COPY requirements.txt .
RUN pip install --require-hashes --only-binary=:all: -r requirements.txt

COPY --from=shared vision_api.py .
COPY server.py .
# Unprivileged; only needs read access to /store and a writable /tmp (pdftoppm renders there).
RUN useradd --system --uid 10001 --no-create-home --home-dir /nonexistent vision
USER 10001
EXPOSE 8102
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8102/health', timeout=4)"]
# Exits 2 at startup with "configuration error: ..." if VISION_MODEL is missing or any VISION_* setting is invalid.
CMD ["python", "server.py"]

# Test image: runtime + stdlib unittest suite (the vision model is a local stub; needs no network), plus the shared
# vision_api tests (shared/tests/test_*.py, discovered next to ours).
FROM runtime AS test
COPY tests/ tests/
COPY --from=shared tests/test_*.py tests/
CMD ["python", "-m", "unittest", "discover", "-s", "tests", "-t", ".", "-v"]
