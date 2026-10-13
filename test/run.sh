#!/usr/bin/env bash
# Full-stack test with stub models (no GPU, no real models): the real router/imgfix/vision-mcp images and config from
# this repo plus a stub text/vision model, all on an internal Docker network; nothing is published on the host.
# Usage (from any directory): test/run.sh        KEEP=1 test/run.sh leaves the stack up for inspection.
# Uses its own compose project (TEST_PROJECT, default "blindsight-test") and removes only what it created: that
# project's containers and network, its volumes (<project>_blindsight-store, <project>_tls) and the
# <project>-imgfix/-vision-mcp image tags.
# Hermetic: ignores .env and any blindsight variables in your shell (every variable compose.yaml reads is set here).
# Two configurations (see test/driver.py): A for phases 1-3, B for phase 4 (the stack is recreated in between). Also:
# bad configurations exit 2, one-off routers refuse bad certificates, no key/question/answer in any log, and
# router/entrypoint.sh and imgfix parse every TEXT_URL in test/text_urls.txt alike.
set -euo pipefail
cd "$(dirname "$0")/.."
P=${TEST_PROJECT:-blindsight-test}
KEY=stack-test-key-0123456789abcdef           # dummy vision API key: must reach the stub, must never be logged
export CLIENT_API_KEY=client-key-5c7e19a2d4  # the driver's own API key (x-api-key, Bearer, ?key=): never logged either
unset COMPOSE_FILE COMPOSE_PROJECT_NAME COMPOSE_PROFILES COMPOSE_ENV_FILES

# A: text model over https with a path prefix and a private CA; OpenAI-format vision with extra headers and the newer
# token parameter; short captions, MCP hint, passthrough models, no remote fetches.
config_a() {
  export TEXT_URL=https://stub:8443/pfx TEXT_CA_FILE=/tls/ca.pem TEXT_HOST= TEXT_PORT= \
    PASSTHROUGH_MODELS='passthru-*,other-vl' \
    CAPTION_MODE=short CAPTION_TIMEOUT= ASK_HINT=mcp ASK_URL= FETCH_REMOTE_IMAGES=false FETCH_ALLOW_PRIVATE=false \
    VISION_API=openai VISION_BASE_URL=http://stub:8001/v1 VISION_MODEL=stub-vision-model VISION_API_KEY="$KEY" \
    VISION_HEADERS='{"X-Stub-Tenant":"blindsight-test"}' VISION_TOKEN_PARAM=max_completion_tokens \
    VISION_EXTRA_BODY='{"chat_template_kwargs":{"enable_thinking":false},"top_k":20}' \
    BIND_ADDR=127.0.0.1 ROUTER_PORT=8100 MCP_PORT=8102 MCP_ALLOWED_HOSTS='127.0.0.1:*,localhost:*,vision-mcp:*' \
    IMGFIX_MIN_FREE_GB=1
}
# B: plain-http text model without prefix; Anthropic-format vision (bare base URL, temperature deleted); detailed
# captions, http hint (ASK_URL derived from MCP_PORT), remote image fetches allowed (the stub is a private address).
config_b() {
  export TEXT_URL=http://stub:8000 TEXT_CA_FILE= PASSTHROUGH_MODELS= CAPTION_MODE=detailed ASK_HINT=http ASK_URL= \
    MCP_PORT=9102 FETCH_REMOTE_IMAGES=true FETCH_ALLOW_PRIVATE=true VISION_API=anthropic VISION_BASE_URL=http://stub:8001 \
    VISION_HEADERS= VISION_TOKEN_PARAM= VISION_EXTRA_BODY='{"temperature":null,"top_k":5}'
}
config_a
DC=(docker compose --env-file /dev/null -p "$P" -f compose.yaml -f test/compose.test.yaml)
BADTLS=("$P-badtls-wrongname" "$P-badtls-untrusted" "$P-badtls-systemca")   # one-off routers (see below)

cleanup() {
  [ "${KEEP:-0}" = 1 ] && { echo "KEEP=1: stack left up (${DC[*]} down; docker volume rm ${P}_blindsight-store ${P}_tls)"; return; }
  docker rm -f "${BADTLS[@]}" >/dev/null 2>&1 || true
  "${DC[@]}" --profile driver --profile tools down --remove-orphans --timeout 10 >/dev/null 2>&1 || true
  docker volume rm "${P}_blindsight-store" "${P}_tls" >/dev/null 2>&1 || true   # this project's test-only volumes
  docker image rm "${P}-imgfix:local" "${P}-vision-mcp:local" >/dev/null 2>&1 || true   # test-only tags
}
trap cleanup EXIT
fail() { echo "FAIL: $*"; exit 1; }
check_logs() {  # the current containers' logs (recreating a container drops its log, so this runs per configuration)
  local logs s
  logs=$("${DC[@]}" logs --no-color router imgfix vision-mcp 2>&1)
  for s in "$KEY" "$CLIENT_API_KEY" "What color is the rectangle" "What+color+is" "Which colour" "Which+colour" \
           "The rectangle is red"; do
    if grep -qF "$s" <<<"$logs"; then fail "logged: $s"; fi
  done
  echo "ok: no API key (vision or client), question or answer in the router/imgfix/vision-mcp logs"
}

echo "== compose config"
docker compose --env-file /dev/null -f compose.yaml config --quiet && "${DC[@]}" config --quiet && echo ok
out=$(env -u VISION_MODEL docker compose --env-file /dev/null -f compose.yaml config --quiet 2>&1) \
  && fail "config without VISION_MODEL must fail"
grep -q "VISION_MODEL is not set" <<<"$out" || fail "unclear error without VISION_MODEL: $out"
echo "ok: without VISION_MODEL -> $out"
for port in 8102 9102; do   # ASK_URL default follows MCP_PORT (nested interpolation)
  got=$(MCP_PORT=$port docker compose --env-file /dev/null -f compose.yaml config)
  grep -q "ASK_URL: http://127.0.0.1:$port/ask" <<<"$got" || fail "ASK_URL default with MCP_PORT=$port"
done
echo "ok: ASK_URL default follows MCP_PORT"

echo "== build (images for this machine's arch), then the test CA (certgen -> tls volume)"
"${DC[@]}" build
"${DC[@]}" --profile tools run --rm --no-deps -T certgen

echo "== up (wait for health)"
"${DC[@]}" up -d --wait --wait-timeout 120

echo "== haproxy -c inside the stack (through the entrypoint, real Docker DNS, TEXT_URL from the environment)"
"${DC[@]}" exec -T router sh /usr/local/etc/haproxy/entrypoint.sh haproxy -c -f /usr/local/etc/haproxy/haproxy.cfg

echo "== bad configuration: router / imgfix / vision-mcp refuse to start (exit 2, clear error)"
expect_refusal() {  # service VAR=value [VAR=value ...]: exit 2 with "configuration error" naming the first VAR
  local svc=$1 rc=0 out kv envs=()
  shift
  for kv in "$@"; do envs+=(-e "$kv"); done
  out=$("${DC[@]}" run --rm --no-deps -T "${envs[@]}" "$svc" 2>&1) || rc=$?
  if [ "$rc" = 2 ] && grep -q "configuration error" <<<"$out" && grep -q "${1%%=*}" <<<"$out"; then
    echo "ok: $svc with $* -> exit 2"
  else fail "$svc with $* -> exit $rc: $(tail -n 5 <<<"$out")"; fi
}
for bad in "TEXT_URL=ftp://stub" "TEXT_URL=https://user:pw@stub/x" "TEXT_URL=https://stub/v1?k=1" "TEXT_URL=http://[::1" \
           "TEXT_CA_FILE=/tls/missing.pem" "TEXT_HOST=gpu-box" "TEXT_PORT=8000"; do
  expect_refusal router "$bad"
done
for svc in router imgfix; do
  expect_refusal "$svc" "TEXT_CA_FILE=/tls/ca.pem" "TEXT_URL=http://stub:8000"   # a CA file with an http TEXT_URL
done
expect_refusal imgfix "TEXT_URL=ftp://stub"
for svc in imgfix vision-mcp; do
  for bad in "VISION_EXTRA_BODY={not json" "VISION_MODEL=" "VISION_API=bogus" "VISION_HEADERS=[1]"; do
    expect_refusal "$svc" "$bad"
  done
done

echo "== phase1 (config A): (a)-(e), (h)-(p)"
out=$("${DC[@]}" run --rm -T driver python -I /test/driver.py phase1) || { echo "$out"; exit 1; }
echo "$out"
sha=$(sed -n 's/^BODY_SHA=//p' <<<"$out")

echo "== routers that must refuse the text model's certificate: wrong name, untrusted CA, system CAs (one-off)"
for spec in "wrongname https://stub-wrongname:8443/pfx /tls/ca.pem" "untrusted https://stub:8443/pfx /tls/other-ca.pem" \
            "systemca https://stub:8443/pfx -"; do
  read -r name url ca <<<"$spec"
  [ "$ca" = - ] && ca=
  "${DC[@]}" run -d --rm --no-deps --name "$P-badtls-$name" -e TEXT_URL="$url" -e TEXT_CA_FILE="$ca" router >/dev/null
done
"${DC[@]}" run --rm -T -e BADTLS_ROUTERS="${BADTLS[*]}" driver python -I /test/driver.py badtls
docker rm -f "${BADTLS[@]}" >/dev/null

echo "== store contents as seen by vision-mcp (uid 10001, read-only)"
"${DC[@]}" exec -T vision-mcp sh -c 'id; ls -ln /store /store/img /store/pdf /store/meta | head -n 60'

echo "== store prune commands (nothing is older than 30 days here, so they delete nothing)"
"${DC[@]}" run --rm --no-deps -T --entrypoint find imgfix /store/img /store/pdf -type f -mtime +30 -print -delete
"${DC[@]}" run --rm --no-deps -T --entrypoint find imgfix /store -name '.tmp-*' -mmin +60 -print -delete
"${DC[@]}" run --rm --no-deps -T --entrypoint du imgfix -sh /store/img /store/pdf /store/meta

echo "== phase2: imgfix stopped -> fail closed (400), text-only still direct"
"${DC[@]}" stop imgfix
"${DC[@]}" run --rm -T driver python -I /test/driver.py phase2

echo "== phase3: imgfix started again"
"${DC[@]}" start imgfix
"${DC[@]}" run --rm -T -e BODY_SHA="$sha" driver python -I /test/driver.py phase3

check_logs

echo "== phase4 (config B): stack recreated with the new configuration"
config_b
"${DC[@]}" up -d --wait --wait-timeout 120
"${DC[@]}" run --rm -T driver python -I /test/driver.py phase4

check_logs

echo "== TEXT_URL parity: router/entrypoint.sh and imgfix accept/reject test/text_urls.txt alike, same parts"
router_view=$("${DC[@]}" run --rm --no-deps -T -v ./test/text_urls.txt:/t/urls:ro --entrypoint sh router -c '
  while IFS= read -r u; do
    case $u in "#"*) continue ;; esac
    if out=$(env -i PATH="$PATH" TEXT_URL="$u" sh /usr/local/etc/haproxy/entrypoint.sh env 2>/dev/null); then
      v() { printf "%s\n" "$out" | sed -n "s/^$1=//p"; }
      printf "%s|ok|%s|%s|%s|%s\n" "$u" "$(v TEXT_SCHEME)" "$(v TEXT_HOST)" "$(v TEXT_PORT)" "$(v TEXT_PREFIX)"
    else
      printf "%s|error\n" "$u"
    fi
  done < /t/urls')
imgfix_view=$("${DC[@]}" run --rm --no-deps -T -v ./test/text_urls.txt:/t/urls:ro \
  -v ./test/text_url_parity.py:/t/parity.py:ro --entrypoint python3 imgfix -I /t/parity.py /t/urls)
if ! diff <(echo "$router_view") <(echo "$imgfix_view"); then fail "TEXT_URL parity (< router, > imgfix)"; fi
echo "ok: $(grep -c '|ok|' <<<"$router_view") accepted, $(grep -c '|error$' <<<"$router_view") rejected, identically"

echo "== router log (routing decisions)"
"${DC[@]}" logs --no-color --no-log-prefix router | grep -E ' -> |blindsight router' | tail -n 30
echo "== imgfix log"
"${DC[@]}" logs --no-color --no-log-prefix imgfix | grep -v 'client \(dis\)\?connect' | tail -n 20
echo "== vision-mcp log"
"${DC[@]}" logs --no-color --no-log-prefix vision-mcp | tail -n 8
echo "== ps"
"${DC[@]}" ps --format 'table {{.Name}}\t{{.Status}}'
echo "STACK TEST PASSED"
