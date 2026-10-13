#!/usr/bin/env bash
# Full-stack test with stub models (no GPU, no real models): the real router/imgfix/vision-mcp images and config from
# this repo plus a stub text/vision model, all on an internal Docker network; nothing is published on the host.
# Usage (from any directory): test/run.sh        KEEP=1 test/run.sh leaves the stack up for inspection.
# Uses its own compose project "blindsight-test" and removes only what it created: that project's containers and
# network, its store volume (blindsight-test_blindsight-store) and the blindsight-test-* image tags.
# Hermetic: ignores .env and any blindsight variables in your shell (every variable compose.yaml reads is set here).
set -euo pipefail
cd "$(dirname "$0")/.."
P=blindsight-test
KEY=stack-test-key-0123456789abcdef           # dummy Bearer token: must reach the stub, must never be logged
unset COMPOSE_FILE COMPOSE_PROJECT_NAME COMPOSE_PROFILES COMPOSE_ENV_FILES
export TEXT_HOST=stub TEXT_PORT=8000 \
  VISION_BASE_URL=http://stub:8001/v1 VISION_MODEL=stub-vision-model VISION_API_KEY="$KEY" \
  VISION_EXTRA_BODY='{"chat_template_kwargs":{"enable_thinking":false},"top_k":20}' \
  BIND_ADDR=127.0.0.1 ROUTER_PORT=8100 MCP_PORT=8102 MCP_ALLOWED_HOSTS='127.0.0.1:*,localhost:*,vision-mcp:*' \
  IMGFIX_MIN_FREE_GB=1
DC=(docker compose --env-file /dev/null -p "$P" -f compose.yaml -f test/compose.test.yaml)

cleanup() {
  [ "${KEEP:-0}" = 1 ] && { echo "KEEP=1: stack left up (${DC[*]} down; docker volume rm ${P}_blindsight-store)"; return; }
  "${DC[@]}" --profile driver down --remove-orphans --timeout 10 >/dev/null 2>&1 || true
  docker volume rm "${P}_blindsight-store" >/dev/null 2>&1 || true   # this project's test-only store volume
  docker image rm "${P}-imgfix:local" "${P}-vision-mcp:local" >/dev/null 2>&1 || true   # test-only tags
}
trap cleanup EXIT

echo "== compose config"
docker compose --env-file /dev/null -f compose.yaml config --quiet && "${DC[@]}" config --quiet && echo ok
out=$(env -u VISION_MODEL docker compose --env-file /dev/null -f compose.yaml config --quiet 2>&1) \
  && { echo "FAIL: config without VISION_MODEL must fail"; exit 1; }
grep -q "VISION_MODEL is not set" <<<"$out" || { echo "FAIL: unclear error without VISION_MODEL: $out"; exit 1; }
echo "ok: without VISION_MODEL -> $out"

echo "== up (build images for this machine's arch, wait for health)"
"${DC[@]}" up -d --build --wait --wait-timeout 120

echo "== haproxy -c inside the stack (real Docker DNS, TEXT_HOST/TEXT_PORT from the environment)"
"${DC[@]}" exec -T router haproxy -c -f /usr/local/etc/haproxy/haproxy.cfg

echo "== bad VISION_EXTRA_BODY / missing VISION_MODEL: imgfix and vision-mcp refuse to start (exit 2, clear error)"
for svc in imgfix vision-mcp; do
  for bad in "VISION_EXTRA_BODY={not json" "VISION_MODEL="; do
    rc=0; out=$("${DC[@]}" run --rm --no-deps -T -e "$bad" "$svc" 2>&1) || rc=$?
    if [ "$rc" = 2 ] && grep -q "configuration error: ${bad%%=*}" <<<"$out"; then echo "ok: $svc with $bad -> exit 2"
    else echo "FAIL: $svc with $bad -> exit $rc: $out"; exit 1; fi
  done
done

echo "== phase1: (a)-(e), (h), (i)"
out=$("${DC[@]}" run --rm -T driver python -I /test/driver.py phase1) || { echo "$out"; exit 1; }
echo "$out"
sha=$(sed -n 's/^BODY_SHA=//p' <<<"$out")

echo "== store contents as seen by vision-mcp (uid 10001, read-only)"
"${DC[@]}" exec -T vision-mcp sh -c 'id; ls -ln /store /store/img /store/meta; cat /store/meta/*.json; echo'

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

echo "== the API key never appears in any service log"
if "${DC[@]}" logs --no-color router imgfix vision-mcp 2>&1 | grep -qF "$KEY"; then echo "FAIL: API key logged"; exit 1; fi
echo ok

echo "== router log (routing decisions)"
"${DC[@]}" logs --no-color --no-log-prefix router | grep -E ' -> ' | tail -n 30
echo "== imgfix log"
"${DC[@]}" logs --no-color --no-log-prefix imgfix | grep -v 'client \(dis\)\?connect' | tail -n 20
echo "== vision-mcp log"
"${DC[@]}" logs --no-color --no-log-prefix vision-mcp | tail -n 5
echo "== ps"
"${DC[@]}" ps --format 'table {{.Name}}\t{{.Status}}'
echo "STACK TEST PASSED"
