#!/usr/bin/env bash
# M1 service-layer integration tests (T9 segment: non-streaming OpenAI domain + error bodies + concurrency + 503 shape).
# Usage: run.sh <build dir> [port]
set -u
cd "$(dirname "$0")/../../.."                  # repo root

BUILD="${1:-build}"
PORT="${2:-18080}"
HOST=127.0.0.1
FIX="$BUILD/tests/qinfer-serve-fixture.gguf"
SERVER="$BUILD/qinfer-server"
LOG="$(mktemp)"
PID=""
PASS=0; FAIL=0

cleanup() { [ -n "$PID" ] && kill "$PID" 2>/dev/null; rm -f "$LOG"; }
trap cleanup EXIT

ok()   { PASS=$((PASS+1)); echo "  ok  $1"; }
bad()  { FAIL=$((FAIL+1)); echo "FAIL  $1"; }

# Start the server; wait for /health.
start() {   # start <extra args...>
  "$SERVER" --model "$FIX" --host "$HOST" --port "$PORT" --backend inhouse \
            --n-embd 256 --hc 2 --hc-lr 3 \
            --layers 2 --top-k 2 --gdn-head-dim 32 --qsa-head-dim 32 --n-rot 4 \
            --idx-dim 4 --idx-top-k 2048 --idx-block 4 --max-context 4096 "$@" >"$LOG" 2>&1 &
  PID=$!
  for _ in $(seq 1 80); do
    curl -sf "http://$HOST:$PORT/health" >/dev/null 2>&1 && return 0
    sleep 0.25
  done
  echo "server failed to start: $(tail -5 "$LOG")" >&2
  return 1
}

stop() { [ -n "$PID" ] && kill "$PID" 2>/dev/null; wait "$PID" 2>/dev/null; PID=""; }

# JSON getter: jget <json-file> <python path expression> → stdout
jget() { python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(eval(sys.argv[2]))' "$1" "$2" 2>/dev/null; }

# Generate the fixture (the parent dir build/tests may not exist, create it first — cli --out does not create parent dirs)
mkdir -p "$(dirname "$FIX")"
"$BUILD/serve_fixture_gen" --out "$FIX"

# ---- main instance with key ----
start --api-key testkey || exit 1

R="$(mktemp)"; H="Authorization: Bearer testkey"

# 1. chat completion: content non-empty, usage fields present, finish valid, id prefix correct.
# Prompt uses only letters inside the fixture vocabulary (a..e, avoiding the sole merge rule "a b"): the fixture
# vocabulary has just 8 tokens, and Chinese/letters x/h truly trigger out-of-vocabulary BPE → 500 tokenization
# failure, so real-language text cannot be used on this fixture.
code=$(curl -s -o "$R" -w '%{http_code}' -H "$H" -H 'Content-Type: application/json' \
  -X POST "http://$HOST:$PORT/v1/chat/completions" \
  -d '{"model":"x","messages":[{"role":"system","content":"c"},{"role":"user","content":"d"}],"max_tokens":16,"temperature":0}')
# Greedy ties pick the smaller index → always token 0 (<|im_start|>), never hits eos, necessarily hits max_tokens → finish is always length.
if [ "$code" = 200 ] \
   && [ -n "$(jget "$R" "d['choices'][0]['message']['content']")" ] \
   && [ "$(jget "$R" "d['usage']['prompt_tokens']")" -ge 1 ] \
   && [ "$(jget "$R" "d['choices'][0]['finish_reason']")" = length ] \
   && [ "$(jget "$R" "d['id'][:9]")" = chatcmpl- ]; then
  ok "chat completion (HTTP 200, content/usage/finish/id)"
else
  bad "chat completion: status $code, body $(head -c 200 "$R")"
fi

# 2. Text completion.
code=$(curl -s -o "$R" -w '%{http_code}' -H "$H" -H 'Content-Type: application/json' \
  -X POST "http://$HOST:$PORT/v1/completions" -d '{"prompt":"cd","max_tokens":8,"temperature":0}')
if [ "$code" = 200 ] && [ -n "$(jget "$R" "d['choices'][0]['text']")" ]; then
  ok "text completion"
else
  bad "text completion: status $code, body $(head -c 200 "$R")"
fi

# 3. Model list.
code=$(curl -s -o "$R" -w '%{http_code}' -H "$H" "http://$HOST:$PORT/v1/models")
if [ "$code" = 200 ] && [ "$(jget "$R" "d['data'][0]['id']")" = qinfer-serve-fixture ]; then
  ok "/v1/models single element, id correct"
else
  bad "/v1/models: status $code, body $(head -c 200 "$R")"
fi

# 4. Invalid parameters → 400 and error.type = invalid_request_error.
code=$(curl -s -o "$R" -w '%{http_code}' -H "$H" -H 'Content-Type: application/json' \
  -X POST "http://$HOST:$PORT/v1/chat/completions" \
  -d '{"messages":[{"role":"user","content":"x"}],"temperature":-1}')
if [ "$code" = 400 ] && [ "$(jget "$R" "d['error']['type']")" = invalid_request_error ]; then
  ok "400 error body (OpenAI shape)"
else
  bad "400: status $code, body $(head -c 200 "$R")"
fi

# 5. Missing key → 401 with correct error shape (OpenAI domain).
code=$(curl -s -o "$R" -w '%{http_code}' "http://$HOST:$PORT/v1/models")
if [ "$code" = 401 ] && [ "$(jget "$R" "d['error']['type']")" = authentication_error ]; then
  ok "401 error body (missing key)"
else
  bad "401: status $code, body $(head -c 200 "$R")"
fi

# 6. Concurrency 8: fire 8 requests at once, all 200 with valid JSON responses; the server stays alive afterwards.
# Note: wait only on this batch of curl PIDs — the server process is also a background child of this shell, and a
# bare wait would wait on it too (never returning).
allok=1
CURLPIDS=""
for i in $(seq 1 8); do
  curl -s -o "$R.$i" -w '%{http_code}' -H "$H" -H 'Content-Type: application/json' \
    -X POST "http://$HOST:$PORT/v1/chat/completions" \
    -d '{"messages":[{"role":"user","content":"c"}],"max_tokens":8,"temperature":0}' >"$LOG.c$i" &
  CURLPIDS="$CURLPIDS $!"
done
wait $CURLPIDS
for i in $(seq 1 8); do
  [ "$(cat "$LOG.c$i")" = 200 ] || allok=0
  python3 -c 'import json,sys; json.load(open(sys.argv[1]))' "$R.$i" 2>/dev/null || allok=0
  rm -f "$R.$i" "$LOG.c$i"
done
code=$(curl -s -o /dev/null -w '%{http_code}' -H "$H" "http://$HOST:$PORT/health")
[ "$code" = 200 ] || allok=0
if [ "$allok" = 1 ]; then
  ok "concurrency 8: all 200, responses valid JSON, server alive"
else
  bad "concurrency 8 had failures (server-alive check and the 8 responses verified one by one)"
fi

# 7. SSE: first fetch the reference text non-streaming, then stream the same request; the concatenated deltas
#    must equal the reference byte for byte.
# The prompt uses only in-vocabulary letter c (out-of-vocabulary characters trigger encode failure → 500);
# output is deterministic under greedy + seed 5.
# The reference side uses sys.stdout.write for exact content — print would add an extra newline, and cmp against
# the streamed text (no trailing newline) would always fail.
curl -s -H "$H" -H 'Content-Type: application/json' -X POST "http://$HOST:$PORT/v1/chat/completions" \
  -d '{"messages":[{"role":"user","content":"c"}],"max_tokens":24,"temperature":0,"seed":5}' \
  | python3 -c 'import json,sys; sys.stdout.write(json.load(sys.stdin)["choices"][0]["message"]["content"])' >/tmp/qinfer_ref.txt

HDRS="$R.hdrs"; BODY="$R.body"
curl -sN -D "$HDRS" -o "$BODY" -H "$H" -H 'Content-Type: application/json' \
  -X POST "http://$HOST:$PORT/v1/chat/completions" \
  -d '{"messages":[{"role":"user","content":"c"}],"max_tokens":24,"temperature":0,"seed":5,"stream":true}'
# Frame-sequence check: first frame carries role, middle frames carry content, last frame carries finish_reason,
# closing [DONE]; concatenate the text delta by delta.
python3 - "$BODY" >/tmp/qinfer_sse.txt <<'PY'
import json, sys
raw = open(sys.argv[1]).read()
frames = [l[6:] for l in raw.splitlines() if l.startswith("data: ") and l != "data: [DONE]"]
assert frames, "没有任何数据帧"
first = json.loads(frames[0])
assert first["choices"][0]["delta"].get("role") == "assistant", "首帧缺 role"
last = json.loads(frames[-1])
assert last["choices"][0]["finish_reason"] in ("stop", "length"), "末帧缺 finish_reason"
text = "".join(json.loads(f)["choices"][0]["delta"].get("content", "") for f in frames)
sys.stdout.write(text)
PY
if grep -q "data: \[DONE\]" "$BODY" && grep -qi "text/event-stream" "$HDRS" \
   && cmp -s /tmp/qinfer_ref.txt /tmp/qinfer_sse.txt; then
  ok "SSE: event stream, [DONE], streamed text byte-identical to non-streaming"
else
  bad "SSE: header $(grep -i event-stream "$HDRS" | head -1), body lines $(wc -l < "$BODY")"
fi
rm -f "$HDRS" "$BODY"

# 8. stream_options.include_usage: data frames carry usage, and completion_tokens ≥ 1.
curl -sN -H "$H" -H 'Content-Type: application/json' -X POST "http://$HOST:$PORT/v1/chat/completions" \
  -d '{"messages":[{"role":"user","content":"d"}],"max_tokens":8,"temperature":0,"stream":true,"stream_options":{"include_usage":true}}' \
  | python3 -c '
import json, sys
raw = sys.stdin.read()
frames = [l[6:] for l in raw.splitlines() if l.startswith("data: ") and l != "data: [DONE]"]
usage = [json.loads(f) for f in frames if json.loads(f).get("usage")]
assert usage, "usage 帧缺失"
assert usage[-1]["usage"]["completion_tokens"] >= 1
print("usage-ok")' | grep -q usage-ok && ok "include_usage trailing usage frame" || bad "include_usage"

# 9. Anthropic non-streaming: the same request must produce the same token sequence as the OpenAI chat domain (text byte-identical).
# The prompt uses in-vocabulary letter c (out-of-vocabulary characters → tokenization failure 500); under greedy, seed does not affect output.
# The reference side uses sys.stdout.write for exact content — print's trailing newline would break the byte-for-byte comparison.
curl -s -H "$H" -H 'Content-Type: application/json' -X POST "http://$HOST:$PORT/v1/chat/completions" \
  -d '{"messages":[{"role":"user","content":"c"}],"max_tokens":16,"temperature":0,"seed":3}' \
  | python3 -c 'import json,sys; sys.stdout.write(json.load(sys.stdin)["choices"][0]["message"]["content"])' >/tmp/qinfer_c.txt
curl -s -H "$H" -H 'Content-Type: application/json' -X POST "http://$HOST:$PORT/v1/messages" \
  -d '{"model":"x","messages":[{"role":"user","content":"c"}],"max_tokens":16,"temperature":0}' \
  > "$R"
STOP=$(jget "$R" "d['content'][0]['type']")
SR=$(jget "$R" "d['stop_reason']")
# Both stop_reason values are valid: greedy on the fixture always consumes max_tokens → max_tokens; a real model stops naturally → end_turn.
if [ "$(jget "$R" "d['type']")" = message ] && [ "$STOP" = text ] \
   && [ "$(jget "$R" "d['usage']['input_tokens']")" -ge 1 ] \
   && { [ "$SR" = end_turn ] || [ "$SR" = max_tokens ]; }; then
  ok "Anthropic message completion (type/content block/stop_reason/usage)"
else
  bad "Anthropic completion: body $(head -c 300 "$R")"
fi
if python3 - "$R" <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))
a = open("/tmp/qinfer_c.txt").read()
b = "".join(c["text"] for c in r["content"] if c["type"] == "text")
assert a == b, f"两个域产出不同：chat={a!r} anthropic={b!r}"
PY
then
  ok "both compatible domains, same request → identical token sequence (byte-for-byte)"
else
  bad "the two domains produced different output"
fi

# 10. count_tokens: cross-check id by id against the tokenizer (count once independently: encode the rendered
#     prompt). urllib must send the key.
if python3 - "$PORT" "$H" <<'PY'
import json, sys, urllib.request
h, _, v = sys.argv[2].partition(": ")
body = {"model": "x", "messages": [{"role": "user", "content": "d"}], "max_tokens": 8}
req = urllib.request.Request("http://127.0.0.1:%s/v1/messages/count_tokens" % sys.argv[1],
                             data=json.dumps(body).encode(),
                             headers={"Content-Type": "application/json", h: v})
n = json.load(urllib.request.urlopen(req))["input_tokens"]
assert isinstance(n, int) and n >= 1, n
PY
then
  ok "count_tokens returns a positive integer"
else
  bad "count_tokens"
fi

# 11. Anthropic streaming: event order message_start → content_block_start → content_block_delta… →
#     content_block_stop → message_delta → message_stop; the delta-concatenated text matches the non-streaming
#     text byte for byte.
# Parse and cross-check are wrapped in if: a parse failure (the event stream is empty when the endpoint is not
# implemented) must not be faked green by cmp of an empty file vs an empty file.
curl -sN -H "$H" -H 'Content-Type: application/json' -X POST "http://$HOST:$PORT/v1/messages" \
  -d '{"model":"x","messages":[{"role":"user","content":"c"}],"max_tokens":16,"temperature":0,"stream":true}' \
  > "$R.body"
if python3 - "$R.body" >/tmp/qinfer_a.txt <<'PY'
import json, sys
raw = open(sys.argv[1]).read()
evs = []
cur = None
for line in raw.splitlines():
    if line.startswith("event: "): cur = line[7:]
    elif line.startswith("data: "):
        assert cur, "data 前缺 event"
        evs.append((cur, json.loads(line[6:])))
names = [e[1]["type"] for e in evs]
assert names[0] == "message_start", names
assert "content_block_start" in names
assert "content_block_delta" in names, "缺 delta 事件"
assert names[-2:] == ["message_delta", "message_stop"], names[-3:]
text = "".join(e["delta"]["text"] for t, e in evs if t == "content_block_delta" and "text" in e["delta"])
sys.stdout.write(text)
PY
then
  # Cross-check: minimal text from non-streaming Anthropic (same body)
  curl -s -H "$H" -H 'Content-Type: application/json' -X POST "http://$HOST:$PORT/v1/messages" \
    -d '{"model":"x","messages":[{"role":"user","content":"c"}],"max_tokens":16,"temperature":0}' \
    | python3 -c 'import json,sys; sys.stdout.write("".join(c["text"] for c in json.load(sys.stdin)["content"]))' >/tmp/qinfer_an.txt
  if cmp -s /tmp/qinfer_a.txt /tmp/qinfer_an.txt; then
    ok "Anthropic SSE event order and concatenated text correct"
  else
    bad "Anthropic SSE: streamed text differs from non-streaming"
  fi
else
  bad "Anthropic SSE: event-order or concatenation parse failed"
fi

# 12. Status family: /v1/status with service self-description, /status probe, /props capabilities, /slots slot count.
curl -s -H "$H" "http://$HOST:$PORT/v1/status" > "$R"
[ "$(jget "$R" "d['service']")" = qinfer ] && [ "$(jget "$R" "d['loaded']")" = True ] \
  && [ "$(jget "$R" "d['context']['native']")" -ge 1 ] \
  && ok "/v1/status (service/loaded/context)" || bad "/v1/status: $(head -c 200 "$R")"

curl -s -H "$H" "http://$HOST:$PORT/status" > "$R"
[ "$(jget "$R" "d['busy']")" = False ] && [ "$(jget "$R" "d['queued']")" = 0 ] \
  && ok "/status probe (busy/queued)" || bad "/status: $(head -c 200 "$R")"

curl -s -H "$H" "http://$HOST:$PORT/props" > "$R"
[ "$(jget "$R" "d['total_slots']")" -ge 1 ] \
  && [ "$(jget "$R" "d['chat_template']")" ] \
  && ok "/props (total_slots/chat_template)" || bad "/props: $(head -c 200 "$R")"

curl -s -H "$H" "http://$HOST:$PORT/slots" > "$R"
[ "$(jget "$R" "len(d)")" -ge 1 ] && [ "$(jget "$R" "d[0]['is_processing']")" = False ] \
  && ok "/slots array and idle" || bad "/slots: $(head -c 200 "$R")"

# 12b. /v1/status machine: at is always present; on GPU-less CI gpu/ram are null, with a GPU they are
#      objects — pin the shape only.
# at is a JSON float (T6 %.9g serialization), so bash -gt is always false for non-integers — the assert goes
# through a python-side boolean.
curl -s -H "$H" "http://$HOST:$PORT/v1/status" > "$R"
[ "$(jget "$R" "d['machine']['at'] > 0")" = True ] \
  && ok "/v1/status machine.at" || bad "/v1/status machine: $(head -c 200 "$R")"

# 13. /metrics: all key types, 11 series (incl. igpu_freq_mhz), history has data, totals/conversation_cache/time shapes.
# conversation_cache reflects the real state once the token-prefix cache (--conversation-cache default 64) lands;
# before this case there are already sequential duplicate requests (case 129→136, same content "c"), so
# hits ≥ 1 and totals.reused ≥ 1 are deterministic.
curl -s -H "$H" "http://$HOST:$PORT/metrics" > "$R"
[ "$(jget "$R" "d['engine']['model']")" ] && [ "$(jget "$R" "d['live']['state']")" ] \
  && [ "$(jget "$R" "len(d['history'])")" = 11 ] \
  && [ "$(jget "$R" "len(d['history']['cpu']) >= 1")" = True ] \
  && [ "$(jget "$R" "d['requests_kept']")" = 100 ] \
  && [ "$(jget "$R" "d['totals']['reused'] >= 1")" = True ] \
  && [ "$(jget "$R" "d['conversation_cache']['enabled']")" = True ] \
  && [ "$(jget "$R" "d['conversation_cache']['hits'] >= 1")" = True ] \
  && [ "$(jget "$R" "d['conversation_cache']['capacity']")" = 64 ] \
  && [ "$(jget "$R" "d['time'] > 0")" = True ] \
  && ok "/metrics all keys" || bad "/metrics: $(head -c 300 "$R")"

# 14. /api/requests: list (retention=100, newest first) and ?id= detail; missing id → 404 error body.
curl -s -H "$H" "http://$HOST:$PORT/api/requests" > "$R"
[ "$(jget "$R" "d['retention']")" = 100 ] && [ "$(jget "$R" "len(d['requests']) >= 1")" = True ] \
  && [ "$(jget "$R" "d['requests'][0]['wallclock_s'] >= 0")" = True ] \
  && ok "/api/requests list" || bad "/api/requests: $(head -c 300 "$R")"
RID=$(jget "$R" "d['requests'][0]['id']" | tr -d '"')
curl -s -H "$H" "http://$HOST:$PORT/api/requests?id=$RID" > "$R"
[ "$(jget "$R" "d['id']")" = "$RID" ] && [ "$(jget "$R" "d['wallclock_s'] >= 0")" = True ] \
  && [ "$(jget "$R" "d['queue_s'] >= 0")" = True ] \
  && ok "/api/requests?id= detail" || bad "/api/requests?id=: $(head -c 300 "$R")"
curl -s -H "$H" "http://$HOST:$PORT/api/requests?id=deadbeef0000" > "$R"
[ "$(jget "$R" "d['error']['message']")" ] \
  && ok "/api/requests?id= 404 error body" || bad "/api/requests missing-id error body: $(head -c 300 "$R")"

# 15. Static pages: / is a three-tab shell containing the Chat panel, tokens.css carries theme tokens and the
#     dark theme, app.js carries the Chat logic.
curl -s "http://$HOST:$PORT/" > "$R"
grep -q 'id="panel-chat"' "$R" && grep -q '/web/tokens.css' "$R" && grep -q '/web/app.js' "$R" \
  && ok "/ three-tab shell (panel-chat/tokens.css/app.js)" || bad "/ static shell: $(head -c 200 "$R")"
curl -s "http://$HOST:$PORT/web/tokens.css" > "$R"
grep -q -- '--bg' "$R" && grep -q 'data-theme="dark"' "$R" \
  && ok "/web/tokens.css (theme tokens + dark theme)" || bad "/web/tokens.css: $(head -c 200 "$R")"
curl -s "http://$HOST:$PORT/web/app.js" > "$R"
grep -q 'pollMetrics' "$R" && grep -q 'panel-chat' "$R" \
  && ok "/web/app.js (Chat logic)" || bad "/web/app.js: $(head -c 200 "$R")"

# 16. Frontend part two: index carries the Monitor/About skeleton, app.js exposes QinferRequests and sparkline,
#     /api-monitor is shelled and monitor.js is reachable (reuses loadDetail).
curl -s "http://$HOST:$PORT/" > "$R"
grep -q 'id="state-badges"' "$R" && grep -q 'id="req-detail"' "$R" && grep -q 'id="about-engine"' "$R" \
  && ok "/ Monitor/About skeleton (state-badges/req-detail/about-engine)" || bad "/ skeleton: $(head -c 200 "$R")"
curl -s "http://$HOST:$PORT/web/app.js" > "$R"
grep -q 'QinferRequests' "$R" && grep -q 'initMonitor' "$R" && grep -q 'sparkline' "$R" \
  && ok "/web/app.js (Monitor/About/sparkline)" || bad "/web/app.js: $(head -c 200 "$R")"
curl -s "http://$HOST:$PORT/api-monitor" > "$R"
grep -q 'id="req-list"' "$R" && grep -q 'monitor.js' "$R" \
  && ok "/api-monitor shell (req-list/monitor.js)" || bad "/api-monitor: $(head -c 200 "$R")"
curl -s "http://$HOST:$PORT/web/monitor.js" > "$R"
grep -q 'QinferRequests.loadDetail' "$R" \
  && ok "/web/monitor.js (monitor page reuses the render layer)" || bad "/web/monitor.js: $(head -c 200 "$R")"

# 16a. chat encoding-failure bookkeeping: Chinese content necessarily triggers out-of-vocabulary tokenization
#      failure → 500, and the newest ring record is state=error, path=/v1/chat/completions (verifies the
#      encoding-failure path also completes its closing bookkeeping).
code=$(curl -s -o "$R" -w '%{http_code}' -H "$H" -H 'Content-Type: application/json' \
  -X POST "http://$HOST:$PORT/v1/chat/completions" \
  -d '{"messages":[{"role":"user","content":"中文"}],"max_tokens":8,"temperature":0}')
[ "$code" = 500 ] && [ "$(jget "$R" "d['error']['type']")" = server_error ] \
  || bad "16a chat encoding failure should be 500: status $code, body $(head -c 200 "$R")"
curl -s -H "$H" "http://$HOST:$PORT/api/requests" > "$R"
if [ "$(jget "$R" "d['requests'][0]['state']")" = error ] \
   && [ "$(jget "$R" "d['requests'][0]['path']")" = /v1/chat/completions ]; then
  ok "16a chat encoding-failure bookkeeping (newest record state=error, path=chat)"
else
  bad "16a chat encoding-failure bookkeeping: newest entry state=$(jget "$R" "d['requests'][0]['state']") path=$(jget "$R" "d['requests'][0]['path']")"
fi

# 16b. completions error bookkeeping: Chinese prompt → 500, and the newest entry is state=error, path=/v1/completions.
code=$(curl -s -o "$R" -w '%{http_code}' -H "$H" -H 'Content-Type: application/json' \
  -X POST "http://$HOST:$PORT/v1/completions" -d '{"prompt":"中文","max_tokens":8,"temperature":0}')
[ "$code" = 500 ] && [ "$(jget "$R" "d['error']['type']")" = server_error ] \
  || bad "16b completions encoding failure should be 500: status $code, body $(head -c 200 "$R")"
curl -s -H "$H" "http://$HOST:$PORT/api/requests" > "$R"
if [ "$(jget "$R" "d['requests'][0]['state']")" = error ] \
   && [ "$(jget "$R" "d['requests'][0]['path']")" = /v1/completions ]; then
  ok "16b completions error bookkeeping (newest record state=error, path=completions)"
else
  bad "16b completions error bookkeeping: newest entry state=$(jget "$R" "d['requests'][0]['state']") path=$(jget "$R" "d['requests'][0]['path']")"
fi

# 17. Disconnect release: start a streaming request, kill curl after 0.5 s; then all /slots must return to idle
#     (poll for ≤10 s).
curl -s -N -H "$H" -H 'Content-Type: application/json' -X POST "http://$HOST:$PORT/v1/chat/completions" \
  -d '{"messages":[{"role":"user","content":"c"}],"max_tokens":2048,"temperature":0,"stream":true}' >/dev/null &
CK=$!
sleep 0.5
kill -9 "$CK" 2>/dev/null
idle=0
for _ in $(seq 1 40); do
  curl -s -H "$H" "http://$HOST:$PORT/slots" > "$R"
  [ "$(jget "$R" "all(not x['is_processing'] for x in d)")" = True ] && { idle=1; break; }
  sleep 0.25
done
[ "$idle" = 1 ] && ok "slot released after disconnect" || bad "slot not released after disconnect: $(head -c 200 "$R")"

# 18. conversation_cache hit: send the same request twice in sequence, the second must hit (it is cached before
#     the first response completes); comparing /metrics before and after the requests, hits and totals.reused
#     each increase by at least 1, and the cache is non-empty.
curl -s -H "$H" "http://$HOST:$PORT/metrics" > "$R"
h0=$(jget "$R" "d['conversation_cache']['hits']")
r0=$(jget "$R" "d['totals']['reused']")
for _ in 1 2; do
  curl -s -o /dev/null -H "$H" -H 'Content-Type: application/json' -X POST \
    "http://$HOST:$PORT/v1/chat/completions" \
    -d '{"messages":[{"role":"user","content":"e"}],"max_tokens":4,"temperature":0}'
done
curl -s -H "$H" "http://$HOST:$PORT/metrics" > "$R"
h1=$(jget "$R" "d['conversation_cache']['hits']")
r1=$(jget "$R" "d['totals']['reused']")
if [ "$h1" -ge "$((h0 + 1))" ] && [ "$r1" -ge "$((r0 + 1))" ] \
   && [ "$(jget "$R" "d['conversation_cache']['entries'] >= 1")" = True ]; then
  ok "18 duplicate request hit cache (hits ${h0}→${h1}, reused ${r0}→${r1})"
else
  bad "18 duplicate request did not hit: hits ${h0}→${h1}, reused ${r0}→${r1}, $(head -c 200 "$R")"
fi

# 19. Prometheus text port: Accept: text/plain → text/plain content type, vLLM-family samples and histograms
#     present (this instance has already served many requests by now: request_success_total ≥ 1, e2e count ≥ 1,
#     cache hits ≥ 1 are all deterministic).
curl -s -D "$R.hdr" -o "$R.body" -H "$H" -H 'Accept: text/plain' "http://$HOST:$PORT/metrics"
if grep -qi '^Content-Type: text/plain' "$R.hdr" \
   && python3 - "$R.body" <<'PY'
import sys
text = open(sys.argv[1]).read()
assert text.startswith("# HELP vllm:num_requests_running"), text[:80]
def last_int(prefix):
    ls = [l for l in text.splitlines() if l.startswith(prefix) and "{" in l]
    assert ls, f"缺家族 {prefix}"
    return int(ls[0].rsplit(" ", 1)[1])
assert last_int("vllm:request_success_total") >= 1
assert last_int("vllm:e2e_request_latency_seconds_count") >= 1
assert last_int("vllm:time_to_first_token_seconds_count") >= 1
assert last_int("vllm:prefix_cache_hits_total") >= 1
assert "vllm:num_preemptions_total" in text
PY
then
  ok "19 /metrics Prometheus text (Accept negotiation + vLLM family + histograms)"
else
  bad "19 /metrics Prometheus text: header $(head -c 120 "$R.hdr") body $(head -c 200 "$R.body")"
fi

# 19b. ?format=prometheus yields text even without Accept; no Accept and no param keeps JSON (the defaults for
#      the frontend and curl are not broken).
curl -s -D "$R.hdr" -o "$R.body" -H "$H" "http://$HOST:$PORT/metrics?format=prometheus"
grep -qi '^Content-Type: text/plain' "$R.hdr" \
  && ok "19b ?format=prometheus → text" \
  || bad "19b ?format=prometheus: header $(head -c 120 "$R.hdr")"
curl -s -D "$R.hdr" -o "$R" -H "$H" "http://$HOST:$PORT/metrics"
if grep -qi '^Content-Type: application/json' "$R.hdr" \
   && [ "$(jget "$R" "d['totals']['requests'] >= 1")" = True ]; then
  ok "19b no Accept keeps JSON (totals.requests present)"
else
  bad "19b no Accept should be JSON: header $(head -c 120 "$R.hdr") body $(head -c 200 "$R")"
fi

# 20. qinfer: family (JSON-native facts, model label on each sample): same snapshot source as 19, model-name
#     label escaping correct.
curl -s -D "$R.hdr" -o "$R.body" -H "$H" -H 'Accept: text/plain' "http://$HOST:$PORT/metrics"
if python3 - "$R.body" <<'PY'
import sys
text = open(sys.argv[1]).read()
need = {"qinfer:live_state", "qinfer:totals_decode_seconds_total", "qinfer:conversation_cache_capacity",
        "qinfer:conversation_cache_hits_total", "qinfer:cpu", "qinfer:engine_max_context",
        "qinfer:gpu_util"}
assert need <= set(l.split("{", 1)[0] for l in text.splitlines() if not l.startswith("#")), "缺 qinfer 家族"
lab = 'model_name="qinfer-serve-fixture"'
assert ("qinfer:live_state{" + lab + ',state="idle"}') in text, "模型标签样本缺失"
PY
then
  ok "20 qinfer: family complete with model labels"
else
  bad "20 qinfer family: $(head -c 200 "$R.body")"
fi
rm -f "$R.hdr" "$R.body"

# 20b. iGPU telemetry (stage C): /metrics carries the availability flag, the
#      device name slot and the frequency pair (the fixture host may or may not
#      have an iGPU; both answers are valid, the keys must exist).
curl -s -H "$H" "http://$HOST:$PORT/metrics" > "$R"
ig=$(jget "$R" "d['hardware_static']['igpu_available']")
kf=$(jget "$R" "'igpu_freq_mhz' in d['hardware']")
if { [ "$ig" = "True" ] || [ "$ig" = "False" ]; } && [ "$kf" = "True" ]; then
  ok "20b iGPU telemetry present (igpu_available=$ig, igpu_freq key present)"
else
  bad "20b iGPU telemetry missing: igpu_available=$ig freq_key=$kf"
fi

# 21. Tools (OpenAI domain): 21a a well-formed tools wrapper yields 200 with non-empty content; 21b historical
#      assistant tool_calls (content null) + tool messages go through the render pipeline (the fixture template
#      only emits content — this case validates the pipeline: message filtering and tool-definition injection
#      are not dropped); 21c malformed tools shapes all yield 400 ("auto"/empty name/non-array).
code=$(curl -s -o "$R" -w '%{http_code}' -H "$H" -H 'Content-Type: application/json' \
  -X POST "http://$HOST:$PORT/v1/chat/completions" \
  -d '{"model":"x","messages":[{"role":"user","content":"c"}],"max_tokens":8,"temperature":0,
       "tools":[{"type":"function","function":{"name":"get_weather","description":"查天气","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}]}')
if [ "$code" = 200 ] && [ -n "$(jget "$R" "d['choices'][0]['message']['content']")" ]; then
  ok "21a OpenAI tools render (200 + non-empty content)"
else
  bad "21a tools: status $code, body $(head -c 200 "$R")"
fi
code=$(curl -s -o "$R" -w '%{http_code}' -H "$H" -H 'Content-Type: application/json' \
  -X POST "http://$HOST:$PORT/v1/chat/completions" \
  -d '{"model":"x",
       "messages":[{"role":"user","content":"c"},
                   {"role":"assistant","content":null,"tool_calls":[{"id":"call_1","type":"function","function":{"name":"get_weather","arguments":"{\"city\":\"c\"}"}}]},
                   {"role":"tool","tool_call_id":"call_1","content":"c"}],
       "max_tokens":8,"temperature":0}')
if [ "$code" = 200 ]; then
  ok "21b tool history (assistant tool_calls + tool messages)"
else
  bad "21b tool history: status $code, body $(head -c 200 "$R")"
fi
for bad_tools in '"auto"' '[{"type":"function","function":{"name":""}}]' '123'; do
  code=$(curl -s -o "$R" -w '%{http_code}' -H "$H" -H 'Content-Type: application/json' \
    -X POST "http://$HOST:$PORT/v1/chat/completions" \
    -d "{\"model\":\"x\",\"messages\":[{\"role\":\"user\",\"content\":\"c\"}],\"max_tokens\":4,\"temperature\":0,\"tools\":$bad_tools}")
  [ "$code" = 400 ] || break
done
if [ "$code" = 400 ]; then
  ok "21c malformed tools shapes → 400 (\"auto\"/empty name/non-array)"
else
  bad "21c malformed tools should be 400: got $code, body $(head -c 200 "$R")"
fi

# 22. Tools (Anthropic domain): tools' input_schema is renamed to parameters; tool_use/tool_result history
#      goes through the message assembly pipeline → 200; the generated block is text (the fixture does not
#      actually emit <tool_call>), input_tokens ≥ 1.
code=$(curl -s -o "$R" -w '%{http_code}' -H "$H" -H 'Content-Type: application/json' \
  -X POST "http://$HOST:$PORT/v1/messages" \
  -d '{"model":"x",
       "messages":[{"role":"assistant","content":[{"type":"text","text":"c"},
                                                  {"type":"tool_use","id":"toolu_1","name":"get_weather","input":{"city":"c"}}]},
                   {"role":"user","content":[{"type":"tool_result","tool_use_id":"toolu_1","content":"c"}]}],
       "tools":[{"name":"get_weather","description":"查天气","input_schema":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}],
       "max_tokens":8,"temperature":0}')
if [ "$code" = 200 ] && [ "$(jget "$R" "d['content'][0]['type']")" = text ] \
   && [ "$(jget "$R" "d['stop_reason']")" = max_tokens ] \
   && [ "$(jget "$R" "d['usage']['input_tokens']")" -ge 1 ]; then
  ok "22 Anthropic tools (input_schema/history/usage/stop_reason)"
else
  bad "22 Anthropic tools: status $code, body $(head -c 200 "$R")"
fi

# 23. count_tokens with tools: input_schema is normalized into rendering, count ≥ 1 (same prompt basis as
#     completion).
code=$(curl -s -o "$R" -w '%{http_code}' -H "$H" -H 'Content-Type: application/json' \
  -X POST "http://$HOST:$PORT/v1/messages/count_tokens" \
  -d '{"model":"x","messages":[{"role":"user","content":"c"}],
       "tools":[{"name":"get_weather","description":"查天气","input_schema":{"type":"object","properties":{"city":{"type":"string"}}}}]}')
if [ "$code" = 200 ] && [ "$(jget "$R" "d['input_tokens']")" -ge 1 ]; then
  ok "23 count_tokens with tools"
else
  bad "23 count_tokens+tools: status $code, body $(head -c 200 "$R")"
fi

# 24. OpenAI streaming with tools: SSE role first frame + non-empty concatenated text + [DONE] (the fixture
#     does not really emit <tool_call>; verifies tools shapes pass through the streaming pipeline without
#     breaking frames).
curl -sN -H "$H" -H 'Content-Type: application/json' -X POST "http://$HOST:$PORT/v1/chat/completions" \
  -d '{"model":"x","messages":[{"role":"user","content":"c"}],"max_tokens":8,"temperature":0,
       "stream":true,
       "tools":[{"type":"function","function":{"name":"get_weather","description":"查天气","parameters":{"type":"object","properties":{"city":{"type":"string"}}}}}]}' \
  | python3 -c '
import json, sys
raw = sys.stdin.read()
objs = [json.loads(l[6:]) for l in raw.splitlines() if l.startswith("data: ") and l != "data: [DONE]"]
assert objs, "无帧"
assert objs[0]["choices"][0]["delta"].get("role") == "assistant", "首帧缺 role"
text = "".join(o["choices"][0]["delta"].get("content", "") for o in objs)
assert text, "拼文空"
print("stream-tools-ok")' | grep -q stream-tools-ok \
  && ok "24 OpenAI streaming with tools (role first frame + non-empty text + [DONE])" \
  || bad "24 streaming with tools"

# 25. Frontend part three: Strata-style static assets reachable (components/sprite/fonts), / carries the new
#     shell markers (st-header/st-pill/theme button); the existing 15/16 compatibility ids are preserved.
for p in web/components.css web/app.css web/sprite.svg web/fonts/outfit-latin-wght.woff2; do
  code=$(curl -s -o /dev/null -w '%{http_code}' "http://$HOST:$PORT/$p")
  [ "$code" = 200 ] || bad "25 /$p should be 200, got $code"
done
curl -s "http://$HOST:$PORT/" > "$R"
grep -q 'st-header' "$R" && grep -q 'st-pill' "$R" && grep -q 'btn-theme' "$R" \
  && grep -q 'id="panel-chat"' "$R" && grep -q 'id="metrics"' "$R" \
  && ok "25 Strata assets + new shell (st-header/st-pill/btn-theme, compatibility points preserved)" \
  || bad "25 new shell markers: $(head -c 200 "$R")"

# 26. Frontend part four: i18n and settings consolidation — / carries the language button (btn-lang) and
#     data-i18n markers, About carries api-key (the key moved from the top bar into a settings card), and
#     app.js contains the qinfer.lang dictionary and the language-switching function.
curl -s "http://$HOST:$PORT/" > "$R"
grep -q 'id="btn-lang"' "$R" && grep -q 'data-i18n' "$R" \
  && grep -q 'id="api-key"' "$R" && grep -q 'data-i18n="about.apiKey"' "$R" \
  && ok "26 i18n markers + key consolidation (btn-lang/data-i18n/api-key in About)" \
  || bad "26 i18n/key consolidation: $(head -c 200 "$R")"
curl -s "http://$HOST:$PORT/web/app.js" > "$R"
grep -q 'qinfer.lang' "$R" && grep -q 'applyLang' "$R" && grep -q 'data-i18n' "$R" \
  && ok "26 app.js language layer (qinfer.lang/applyLang/data-i18n)" \
  || bad "26 app.js language layer: $(head -c 200 "$R")"

# 27. chunked terminator: the SSE stream must end with a canonical zero-length terminating chunk — without it,
#     browsers (strict chunked parsing) fail with ERR_INCOMPLETE_CHUNKED_ENCODING and the frontend never sees
#     the stream; python http.client is equally strict and raises IncompleteRead on an incomplete stream.
out27=$(python3 - "$HOST" "$PORT" <<'PYEOF'
import http.client, sys
host, port = sys.argv[1], int(sys.argv[2])
conn = http.client.HTTPConnection(host, port, timeout=60)
body = '{"messages":[{"role":"user","content":"d"}],"max_tokens":8,"temperature":0,"stream":true}'
conn.request("POST", "/v1/chat/completions", body,
             {"Content-Type": "application/json",
              "Authorization": "Bearer testkey"})
r = conn.getresponse()
assert r.status == 200, "HTTP %s" % r.status
assert r.getheader("Transfer-Encoding") == "chunked", "非 chunked"
data = r.read()   # 缺终止块 → 抛 IncompleteRead
assert data.endswith(b"data: [DONE]\n\n"), "流末缺 [DONE]"
assert data.count(b"[DONE]") == 1, "DONE 只应出现一次"
print("chunked-terminated-ok")
PYEOF
)
[ "$out27" = chunked-terminated-ok ] \
  && ok "27 chunked terminator (strict parsing reads all the way to [DONE])" \
  || bad "27 chunked terminator: missing zero-length terminating chunk → IncompleteRead/incomplete stream: $out27"

# 28. The inference thread pool is configurable and not saturated by default: --compute-threads 4 → /props
#     compute_threads=4; without the flag → default = half the physical concurrency, capped at 8 (users want
#     the default to not consume the whole CPU).
stop
start --api-key testkey --compute-threads 4 || exit 1
curl -s -H "$H" "http://$HOST:$PORT/props" > "$R"
ct=$(jget "$R" "d['compute_threads']")
[ "$ct" = 4 ] && ok "28 --compute-threads 4 → /props compute_threads=4" \
  || bad "28 compute_threads: expected 4, got $ct"
stop
start --api-key testkey || exit 1
curl -s -H "$H" "http://$HOST:$PORT/props" > "$R"
ct=$(jget "$R" "d['compute_threads']")
cores="$(getconf _NPROCESSORS_ONLN)"
half=$(( cores / 2 )); [ "$half" -gt 8 ] && half=8
[ -n "$ct" ] && [ "$ct" -ge 1 ] && [ "$ct" -le "$half" ] \
  && ok "28 default compute_threads=$ct (≤ half the physical cores $half, not saturated)" \
  || bad "28 default compute_threads: got $ct, expected 1..$half (cores=$cores)"
stop

# 29. --gpu semantics split by build: on a CUDA build --gpu 0 starts normally (dense device execution is
#     active; both *_cuda paths of layer 0 GDN + layer 1 QSA run), and one request verifies the device-inference
#     output; on a CUDA-less build startup must be rejected (an explicit error, no silent fallback). Without
#     --gpu the server starts as usual (pure-CPU path unchanged; the other cases in this script exercise it).
timeout 8 "$SERVER" --model "$FIX" --host "$HOST" --port "$PORT" --backend inhouse \
          --n-embd 256 --hc 2 --hc-lr 3 \
          --layers 2 --top-k 2 --gdn-head-dim 32 --qsa-head-dim 32 --n-rot 4 --idx-dim 4 \
          --idx-top-k 2048 --idx-block 4 --max-context 4096 --api-key testkey --kv-slots 8 \
          --kv-resident 4 --gpu 0 \
          >/tmp/gpu-reject.log 2>&1 &
BPID=$!
up=0
for _ in $(seq 1 30); do
  curl -sf "http://$HOST:$PORT/health" >/dev/null 2>&1 && { up=1; break; }
  sleep 0.25
done
if [ "$up" = 1 ]; then
  # Query /props first (right after health), then run the request: the timeout
# window is 8s and the request itself takes a few of them.
  curl -s -H "$H" "http://$HOST:$PORT/props" > "$R.p"
  kv_res=$(jget "$R.p" "d['kv_resident']")
  code=$(curl -s -o "$R" -w '%{http_code}' -H "$H" -H 'Content-Type: application/json' \
    -X POST "http://$HOST:$PORT/v1/chat/completions" \
    -d '{"model":"x","messages":[{"role":"system","content":"c"},{"role":"user","content":"d"}],"max_tokens":8,"temperature":0}')
  if [ "$code" = 200 ] && [ -n "$(jget "$R" "d['choices'][0]['message']['content']")" ] \
     && [ "$kv_res" = 4 ]; then
    ok "29 --gpu 0 device inference OK + device-KV eviction path (--kv-resident 4, /props kv_resident=4)"
  else
    bad "29 --gpu 0 device inference failed or kv_resident wrong: http=$code, kv_resident=$kv_res"
  fi
else
  wait "$BPID" 2>/dev/null; rc=$?
  if [ "$rc" -ne 0 ] && [ "$rc" -ne 124 ] && grep -q "CUDA-enabled build" /tmp/gpu-reject.log; then
    ok "29 --gpu refused to start on a CUDA-less build (exit=$rc, error body contains CUDA-enabled build)"
  else
    bad "29 --gpu refused startup: exit=$rc (124=silently hung), log $(tail -2 /tmp/gpu-reject.log)"
  fi
fi
# Kill only THIS case's server (B is the timeout wrapper; its child is the
# server). A blanket `pkill -x qinfer-server` also killed a developer's own
# long-running instance on the same machine.
kill -9 "$BPID" 2>/dev/null
pkill -9 -P "$BPID" 2>/dev/null
wait "$BPID" 2>/dev/null
sleep 1
rm -f /tmp/gpu-reject.log

stop

# ---- 503: --reject-after 1 → first request OK, then queue-full from then on ----
start --api-key testkey --slots 1 --reject-after 1 || exit 1
code=$(curl -s -o "$R" -w '%{http_code}' -H "$H" -H 'Content-Type: application/json' \
  -X POST "http://$HOST:$PORT/v1/chat/completions" \
  -d '{"messages":[{"role":"user","content":"a"}],"max_tokens":4,"temperature":0}')
[ "$code" = 200 ] || bad "reject-after first request should be 200, got $code"
code=$(curl -s -o "$R" -w '%{http_code}' -H "$H" -H 'Content-Type: application/json' \
  -X POST "http://$HOST:$PORT/v1/chat/completions" \
  -d '{"messages":[{"role":"user","content":"b"}],"max_tokens":4,"temperature":0}')
if [ "$code" = 503 ] && cmp -s "$R" <(printf '%s' '{"error":"queue full"}'); then
  ok "503 body exactly {\"error\":\"queue full\"}"
else
  bad "503: status $code, body $(head -c 200 "$R")"
fi
stop

# ---- P2: --backend inhouse is the explicit default; it behaves like passing no flag ----
start --backend inhouse || exit 1
code=$(curl -s -o "$R" -w '%{http_code}' -H "$H" -H 'Content-Type: application/json' \
  -X POST "http://$HOST:$PORT/v1/chat/completions" \
  -d '{"messages":[{"role":"user","content":"e"}],"max_tokens":4,"temperature":0}')
if [ "$code" = 200 ]; then
  ok "P2 --backend inhouse serves a chat completion"
else
  bad "P2 --backend inhouse: http=$code body=$(head -c 200 "$R")"
fi
stop

# ---- P2: --backend strata without --engine-exe is refused before spawning anything ----
timeout 10 "$SERVER" --model "$FIX" --backend strata --port "$PORT" >/tmp/backend-strata.log 2>&1
rc=$?
if [ "$rc" -ne 0 ] && [ "$rc" -ne 124 ] && grep -q -- "--engine-exe" /tmp/backend-strata.log; then
  ok "P2 --backend strata without --engine-exe is refused (exit=$rc)"
else
  bad "P2 --backend strata without --engine-exe: exit=$rc (124=hung), log $(tail -2 /tmp/backend-strata.log)"
fi
rm -f /tmp/backend-strata.log

# ---- P2: the default backend is strata, so a bare start must be refused with an actionable message
timeout 10 "$SERVER" --model "$FIX" --port "$PORT" >/tmp/backend-default.log 2>&1
rc=$?
if [ "$rc" -ne 0 ] && [ "$rc" -ne 124 ] && grep -q -- "--engine-exe" /tmp/backend-default.log \
   && grep -q -- "--backend inhouse" /tmp/backend-default.log; then
  ok "P2 default --backend is strata: a bare start is refused and names both remedies (exit=$rc)"
else
  bad "P2 default backend: exit=$rc (124=hung), log $(tail -2 /tmp/backend-default.log)"
fi
rm -f /tmp/backend-default.log

# ---- M5b: vision / image_url handling ----
# Start a fresh server for vision tests.
start --api-key testkey || exit 1

# 30. image_url parts are rejected with a clear message when vision is not configured
#     (the fixture has no --mmproj, so VisionProcess is not started).
code=$(curl -s -o "$R" -w '%{http_code}' -H "$H" -H 'Content-Type: application/json' \
  -X POST "http://$HOST:$PORT/v1/chat/completions" \
  -d '{"messages":[{"role":"user","content":[{"type":"text","text":"What is this?"},{"type":"image_url","image_url":{"url":"data:image/png;base64,iVBORw0KGgo="}}]}],"max_tokens":8,"temperature":0}')
if [ "$code" = 400 ] && grep -q "vision" "$R"; then
  ok "30 image_url rejected when vision not configured (400 + vision message)"
else
  bad "30 image_url without vision: status $code, body $(head -c 200 "$R")"
fi

stop

# Start a fresh server for capability tests.
start --api-key testkey || exit 1

# 31. /props reports vision capability (false when not configured, true when --mmproj is given).
#     Since the fixture has no vision, we check that the key exists and is false.
curl -s -H "$H" "http://$HOST:$PORT/props" > "$R"
if [ "$(jget "$R" "d['modalities']['vision']")" = "False" ]; then
  ok "31 /props modalities.vision=false when not configured"
else
  bad "31 /props vision capability: $(head -c 200 "$R")"
fi

# 32. /health reports images=false when vision not configured.
curl -s -H "$H" "http://$HOST:$PORT/health" > "$R"
if [ "$(jget "$R" "d['images']")" = "False" ]; then
  ok "32 /health images=false when not configured"
else
  bad "32 /health images: $(head -c 200 "$R")"
fi

# 33. /v1/models includes text in input_modalities (vision would add image when configured).
curl -s -H "$H" "http://$HOST:$PORT/v1/models" > "$R"
if [ "$(jget "$R" "d['data'][0]['architecture']['input_modalities']")" = "['text']" ]; then
  ok "33 /v1/models input_modalities=['text'] when vision not configured"
else
  bad "33 /v1/models input_modalities: $(head -c 200 "$R")"
fi

# 34. reasoning_effort: an unsupported value is rejected with invalid_request_error.
code=$(curl -s -o "$R" -w '%{http_code}' -H "$H" -H 'Content-Type: application/json' \
  -X POST "http://$HOST:$PORT/v1/chat/completions" \
  -d '{"messages":[{"role":"user","content":"x"}],"max_tokens":4,"reasoning_effort":"bogus"}')
if [ "$code" = "400" ] && [ "$(jget "$R" "d['error']['param']")" = "reasoning_effort" ]; then
  ok "34 reasoning_effort rejects an unknown value (400, param=reasoning_effort)"
else
  bad "34 reasoning_effort invalid value: HTTP $code $(head -c 200 "$R")"
fi

# 35. reasoning_effort: 'high' aliases xhigh, so validation accepts it. The tiny
#     fixture vocabulary cannot tokenize the reasoning system block, so assert on
#     acceptance (no reasoning_effort validation error) rather than a 200 body.
code=$(curl -s -o "$R" -w '%{http_code}' -H "$H" -H 'Content-Type: application/json' \
  -X POST "http://$HOST:$PORT/v1/chat/completions" \
  -d '{"messages":[{"role":"user","content":"x"}],"max_tokens":4,"reasoning_effort":"high"}')
if [ "$code" = "400" ] && [ "$(jget "$R" "d['error']['param']")" = "reasoning_effort" ]; then
  bad "35 reasoning_effort rejected 'high': $(head -c 200 "$R")"
else
  ok "35 reasoning_effort accepts 'high' (aliases xhigh)"
fi

# 36/37. /api/feedback: a valid rating is stored and served back; an unknown one is 400.
code=$(curl -s -o "$R" -w '%{http_code}' -H "$H" -H 'Content-Type: application/json' \
  -X POST "http://$HOST:$PORT/api/feedback" \
  -d '{"rating":"up","chat_id":"c1","excerpt":"hi"}')
if [ "$code" = "200" ]; then
  curl -s -H "$H" "http://$HOST:$PORT/api/feedback" > "$R"
  if [ "$(jget "$R" "d['data'][0]['rating']")" = "up" ] && [ "$(jget "$R" "d['data'][0]['chat_id']")" = "c1" ]; then
    ok "36 /api/feedback stores a rating and serves it back"
  else
    bad "36 /api/feedback list: $(head -c 200 "$R")"
  fi
else
  bad "36 /api/feedback post: HTTP $code $(head -c 200 "$R")"
fi

code=$(curl -s -o "$R" -w '%{http_code}' -H "$H" -H 'Content-Type: application/json' \
  -X POST "http://$HOST:$PORT/api/feedback" -d '{"rating":"sideways"}')
if [ "$code" = "400" ] && [ "$(jget "$R" "d['error']['param']")" = "rating" ]; then
  ok "37 /api/feedback rejects an unknown rating (400, param=rating)"
else
  bad "37 /api/feedback invalid rating: HTTP $code $(head -c 200 "$R")"
fi

echo "serve integration: passed $PASS, failed $FAIL"
[ "$FAIL" = 0 ]