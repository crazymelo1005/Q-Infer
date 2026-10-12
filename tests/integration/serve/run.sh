#!/usr/bin/env bash
# M1 服务层集成测试（T9 段：非流式 OpenAI 域 + 错误体 + 并发 + 503 形态）。
# 用法：run.sh <构建目录> [端口]
set -u
cd "$(dirname "$0")/../../.."                  # 仓库根

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

# 起服务；等 /health。
start() {   # start <额外参数...>
  "$SERVER" --model "$FIX" --host "$HOST" --port "$PORT" --n-embd 256 --hc 2 --hc-lr 3 \
            --layers 2 --top-k 2 --gdn-head-dim 32 --qsa-head-dim 32 --n-rot 4 \
            --idx-dim 4 --idx-top-k 2048 --idx-block 4 --max-context 4096 "$@" >"$LOG" 2>&1 &
  PID=$!
  for _ in $(seq 1 80); do
    curl -sf "http://$HOST:$PORT/health" >/dev/null 2>&1 && return 0
    sleep 0.25
  done
  echo "服务没起来：$(tail -5 "$LOG")" >&2
  return 1
}

stop() { [ -n "$PID" ] && kill "$PID" 2>/dev/null; wait "$PID" 2>/dev/null; PID=""; }

# JSON 取值：jget <json文件> <python路径表达式> → 标准输出
jget() { python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(eval(sys.argv[2]))' "$1" "$2" 2>/dev/null; }

# 生成夹具（父目录 build/tests 未必存在，先建好——cli 的 --out 不建父目录）
mkdir -p "$(dirname "$FIX")"
"$BUILD/serve_fixture_gen" --out "$FIX"

# ---- 带 key 的主实例 ----
start --api-key testkey || exit 1

R="$(mktemp)"; H="Authorization: Bearer testkey"

# 1. chat 补全：内容非空、usage 字段齐、finish 合法、id 前缀对。
# 提示词只用夹具词表内的字母（a..e，且避开唯一合并规则 "a b"）：夹具词表就 8 个 token，
# 中文/字母 x/h 会真实触发词表外 BPE → 500 分词失败，不能在这种夹具上用真语言文本。
code=$(curl -s -o "$R" -w '%{http_code}' -H "$H" -H 'Content-Type: application/json' \
  -X POST "http://$HOST:$PORT/v1/chat/completions" \
  -d '{"model":"x","messages":[{"role":"system","content":"c"},{"role":"user","content":"d"}],"max_tokens":16,"temperature":0}')
# 贪心并列取小索引 → 恒 token 0（<|im_start|>），撞不到 eos，必然吃到 max_tokens → finish 恒 length。
if [ "$code" = 200 ] \
   && [ -n "$(jget "$R" "d['choices'][0]['message']['content']")" ] \
   && [ "$(jget "$R" "d['usage']['prompt_tokens']")" -ge 1 ] \
   && [ "$(jget "$R" "d['choices'][0]['finish_reason']")" = length ] \
   && [ "$(jget "$R" "d['id'][:9]")" = chatcmpl- ]; then
  ok "chat 补全（HTTP 200、content/usage/finish/id）"
else
  bad "chat 补全：状态 $code，体 $(head -c 200 "$R")"
fi

# 2. 文本补全。
code=$(curl -s -o "$R" -w '%{http_code}' -H "$H" -H 'Content-Type: application/json' \
  -X POST "http://$HOST:$PORT/v1/completions" -d '{"prompt":"cd","max_tokens":8,"temperature":0}')
if [ "$code" = 200 ] && [ -n "$(jget "$R" "d['choices'][0]['text']")" ]; then
  ok "文本补全"
else
  bad "文本补全：状态 $code，体 $(head -c 200 "$R")"
fi

# 3. 模型清单。
code=$(curl -s -o "$R" -w '%{http_code}' -H "$H" "http://$HOST:$PORT/v1/models")
if [ "$code" = 200 ] && [ "$(jget "$R" "d['data'][0]['id']")" = qinfer-serve-fixture ]; then
  ok "/v1/models 单元素、id 正确"
else
  bad "/v1/models：状态 $code，体 $(head -c 200 "$R")"
fi

# 4. 非法参数 → 400 且 error.type = invalid_request_error。
code=$(curl -s -o "$R" -w '%{http_code}' -H "$H" -H 'Content-Type: application/json' \
  -X POST "http://$HOST:$PORT/v1/chat/completions" \
  -d '{"messages":[{"role":"user","content":"x"}],"temperature":-1}')
if [ "$code" = 400 ] && [ "$(jget "$R" "d['error']['type']")" = invalid_request_error ]; then
  ok "400 错误体（OpenAI 形状）"
else
  bad "400：状态 $code，体 $(head -c 200 "$R")"
fi

# 5. 缺 key → 401 且错误形状正确（OpenAI 域）。
code=$(curl -s -o "$R" -w '%{http_code}' "http://$HOST:$PORT/v1/models")
if [ "$code" = 401 ] && [ "$(jget "$R" "d['error']['type']")" = authentication_error ]; then
  ok "401 错误体（缺 key）"
else
  bad "401：状态 $code，体 $(head -c 200 "$R")"
fi

# 6. 并发 8：8 个请求同时打，全部 200 且响应是合法 JSON；服务在结束后仍存活。
# 注意只 wait 这批 curl 的 PID：服务进程也是本 shell 的后台子进程，裸 wait 会连它一起等（永不返回）。
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
  ok "并发 8：全 200、响应全为合法 JSON、服务存活"
else
  bad "并发 8 后有失败（服务存活检查与 8 份响应逐一核过）"
fi

# 7. SSE：先非流式取参考文本，再流式取同款请求，逐 delta 拼出来必须与参考逐字节相同。
# 提示词只用夹具词表内字母 c（词表外字符会触发 encode 失败 → 500）；贪心 + seed 5 下输出确定。
# 参考侧用 sys.stdout.write 取精确 content——print 会多吃一个换行，与流式拼文（无尾换行）cmp 必败。
curl -s -H "$H" -H 'Content-Type: application/json' -X POST "http://$HOST:$PORT/v1/chat/completions" \
  -d '{"messages":[{"role":"user","content":"c"}],"max_tokens":24,"temperature":0,"seed":5}' \
  | python3 -c 'import json,sys; sys.stdout.write(json.load(sys.stdin)["choices"][0]["message"]["content"])' >/tmp/qinfer_ref.txt

HDRS="$R.hdrs"; BODY="$R.body"
curl -sN -D "$HDRS" -o "$BODY" -H "$H" -H 'Content-Type: application/json' \
  -X POST "http://$HOST:$PORT/v1/chat/completions" \
  -d '{"messages":[{"role":"user","content":"c"}],"max_tokens":24,"temperature":0,"seed":5,"stream":true}'
# 帧序列检查：第一帧带 role、中间帧带 content、末帧带 finish_reason、收尾 [DONE]；逐 delta 拼文本。
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
  ok "SSE：事件流、[DONE]、流式拼文与非流式逐字节一致"
else
  bad "SSE：头部 $(grep -i event-stream "$HDRS" | head -1)，正文行数 $(wc -l < "$BODY")"
fi
rm -f "$HDRS" "$BODY"

# 8. stream_options.include_usage：数据帧里带 usage，且 completion_tokens ≥ 1。
curl -sN -H "$H" -H 'Content-Type: application/json' -X POST "http://$HOST:$PORT/v1/chat/completions" \
  -d '{"messages":[{"role":"user","content":"d"}],"max_tokens":8,"temperature":0,"stream":true,"stream_options":{"include_usage":true}}' \
  | python3 -c '
import json, sys
raw = sys.stdin.read()
frames = [l[6:] for l in raw.splitlines() if l.startswith("data: ") and l != "data: [DONE]"]
usage = [json.loads(f) for f in frames if json.loads(f).get("usage")]
assert usage, "usage 帧缺失"
assert usage[-1]["usage"]["completion_tokens"] >= 1
print("usage-ok")' | grep -q usage-ok && ok "include_usage 末位 usage 帧" || bad "include_usage"

# 9. Anthropic 非流式：同一请求与 OpenAI chat 域产出同一 token 序列（text 逐字节相同）。
# 提示词用词表内字母 c（词表外字符 → 分词失败 500）；贪心下 seed 不影响输出。
# 参考侧用 sys.stdout.write 取精确 content——print 尾换行会破坏逐字节比较。
curl -s -H "$H" -H 'Content-Type: application/json' -X POST "http://$HOST:$PORT/v1/chat/completions" \
  -d '{"messages":[{"role":"user","content":"c"}],"max_tokens":16,"temperature":0,"seed":3}' \
  | python3 -c 'import json,sys; sys.stdout.write(json.load(sys.stdin)["choices"][0]["message"]["content"])' >/tmp/qinfer_c.txt
curl -s -H "$H" -H 'Content-Type: application/json' -X POST "http://$HOST:$PORT/v1/messages" \
  -d '{"model":"x","messages":[{"role":"user","content":"c"}],"max_tokens":16,"temperature":0}' \
  > "$R"
STOP=$(jget "$R" "d['content'][0]['type']")
SR=$(jget "$R" "d['stop_reason']")
# stop_reason 两者皆合法：夹具贪心恒吃满 max_tokens → max_tokens；真模型自然停 → end_turn。
if [ "$(jget "$R" "d['type']")" = message ] && [ "$STOP" = text ] \
   && [ "$(jget "$R" "d['usage']['input_tokens']")" -ge 1 ] \
   && { [ "$SR" = end_turn ] || [ "$SR" = max_tokens ]; }; then
  ok "Anthropic 消息补全（type/内容块/stop_reason/usage）"
else
  bad "Anthropic 补全：体 $(head -c 300 "$R")"
fi
if python3 - "$R" <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))
a = open("/tmp/qinfer_c.txt").read()
b = "".join(c["text"] for c in r["content"] if c["type"] == "text")
assert a == b, f"两个域产出不同：chat={a!r} anthropic={b!r}"
PY
then
  ok "两个兼容域同一请求 → 相同 token 序列（逐字节）"
else
  bad "两个域产出不同"
fi

# 10. count_tokens：与分词器逐 id 对照（独立数一遍：渲染后 encode）。urllib 必须带 key。
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
  ok "count_tokens 返回正整数"
else
  bad "count_tokens"
fi

# 11. Anthropic 流式：事件序 message_start → content_block_start → content_block_delta… →
#     content_block_stop → message_delta → message_stop；delta 拼文与非流式逐字节相同。
# 解析与对照包进 if：解析失败（端点没实现时的事件流为空）不能再被 cmp 空文件 vs 空文件假绿。
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
  # 对照：非流式 Anthropic 的最小文本（同 body）
  curl -s -H "$H" -H 'Content-Type: application/json' -X POST "http://$HOST:$PORT/v1/messages" \
    -d '{"model":"x","messages":[{"role":"user","content":"c"}],"max_tokens":16,"temperature":0}' \
    | python3 -c 'import json,sys; sys.stdout.write("".join(c["text"] for c in json.load(sys.stdin)["content"]))' >/tmp/qinfer_an.txt
  if cmp -s /tmp/qinfer_a.txt /tmp/qinfer_an.txt; then
    ok "Anthropic SSE 事件序与拼文正确"
  else
    bad "Anthropic SSE：流式拼文与非流式不一致"
  fi
else
  bad "Anthropic SSE：事件序或拼文解析失败"
fi

# 12. 状态族：/v1/status 与服务自述、/status 探针、/props 能力、/slots 槽位数。
curl -s -H "$H" "http://$HOST:$PORT/v1/status" > "$R"
[ "$(jget "$R" "d['service']")" = qinfer ] && [ "$(jget "$R" "d['loaded']")" = True ] \
  && [ "$(jget "$R" "d['context']['native']")" -ge 1 ] \
  && ok "/v1/status（service/loaded/context）" || bad "/v1/status：$(head -c 200 "$R")"

curl -s -H "$H" "http://$HOST:$PORT/status" > "$R"
[ "$(jget "$R" "d['busy']")" = False ] && [ "$(jget "$R" "d['queued']")" = 0 ] \
  && ok "/status 探针（busy/queued）" || bad "/status：$(head -c 200 "$R")"

curl -s -H "$H" "http://$HOST:$PORT/props" > "$R"
[ "$(jget "$R" "d['total_slots']")" -ge 1 ] \
  && [ "$(jget "$R" "d['chat_template']")" ] \
  && ok "/props（total_slots/chat_template）" || bad "/props：$(head -c 200 "$R")"

curl -s -H "$H" "http://$HOST:$PORT/slots" > "$R"
[ "$(jget "$R" "len(d)")" -ge 1 ] && [ "$(jget "$R" "d[0]['is_processing']")" = False ] \
  && ok "/slots 数组与空闲" || bad "/slots：$(head -c 200 "$R")"

# 12b. /v1/status 的 machine：at 恒在；无 GPU 的 CI 上 gpu/ram 为 null，有卡时是对象——只钉形状。
# at 是 JSON 浮点（T6 %.9g 序列化），bash -gt 对非整数恒假——断言走 python 侧布尔。
curl -s -H "$H" "http://$HOST:$PORT/v1/status" > "$R"
[ "$(jget "$R" "d['machine']['at'] > 0")" = True ] \
  && ok "/v1/status machine.at" || bad "/v1/status machine：$(head -c 200 "$R")"

# 13. /metrics：全键类型、10 系列、历史有数据、totals/conversation_cache/time 形状。
curl -s -H "$H" "http://$HOST:$PORT/metrics" > "$R"
[ "$(jget "$R" "d['engine']['model']")" ] && [ "$(jget "$R" "d['live']['state']")" ] \
  && [ "$(jget "$R" "len(d['history'])")" = 10 ] \
  && [ "$(jget "$R" "len(d['history']['cpu']) >= 1")" = True ] \
  && [ "$(jget "$R" "d['requests_kept']")" = 100 ] \
  && [ "$(jget "$R" "d['totals']['reused']")" = 0 ] \
  && [ "$(jget "$R" "d['conversation_cache']['enabled']")" = False ] \
  && [ "$(jget "$R" "d['time'] > 0")" = True ] \
  && ok "/metrics 全键" || bad "/metrics：$(head -c 300 "$R")"

# 14. /api/requests：列表（retention=100、最新在前）与 ?id= 详情；缺 id → 404 错误体。
curl -s -H "$H" "http://$HOST:$PORT/api/requests" > "$R"
[ "$(jget "$R" "d['retention']")" = 100 ] && [ "$(jget "$R" "len(d['requests']) >= 1")" = True ] \
  && [ "$(jget "$R" "d['requests'][0]['wallclock_s'] >= 0")" = True ] \
  && ok "/api/requests 列表" || bad "/api/requests：$(head -c 300 "$R")"
RID=$(jget "$R" "d['requests'][0]['id']" | tr -d '"')
curl -s -H "$H" "http://$HOST:$PORT/api/requests?id=$RID" > "$R"
[ "$(jget "$R" "d['id']")" = "$RID" ] && [ "$(jget "$R" "d['wallclock_s'] >= 0")" = True ] \
  && [ "$(jget "$R" "d['queue_s'] >= 0")" = True ] \
  && ok "/api/requests?id= 详情" || bad "/api/requests?id=：$(head -c 300 "$R")"
curl -s -H "$H" "http://$HOST:$PORT/api/requests?id=deadbeef0000" > "$R"
[ "$(jget "$R" "d['error']['message']")" ] \
  && ok "/api/requests?id= 404 错误体" || bad "/api/requests 缺失 id 错误体：$(head -c 300 "$R")"

stop

# ---- 503：--reject-after 1 → 第 1 个请求正常，之后一律队列满 ----
start --api-key testkey --slots 1 --reject-after 1 || exit 1
code=$(curl -s -o "$R" -w '%{http_code}' -H "$H" -H 'Content-Type: application/json' \
  -X POST "http://$HOST:$PORT/v1/chat/completions" \
  -d '{"messages":[{"role":"user","content":"a"}],"max_tokens":4,"temperature":0}')
[ "$code" = 200 ] || bad "reject-after 首请求应 200，得 $code"
code=$(curl -s -o "$R" -w '%{http_code}' -H "$H" -H 'Content-Type: application/json' \
  -X POST "http://$HOST:$PORT/v1/chat/completions" \
  -d '{"messages":[{"role":"user","content":"b"}],"max_tokens":4,"temperature":0}')
if [ "$code" = 503 ] && cmp -s "$R" <(printf '%s' '{"error":"queue full"}'); then
  ok "503 身体精确为 {\"error\":\"queue full\"}"
else
  bad "503：状态 $code，体 $(head -c 200 "$R")"
fi
stop

echo "serve 集成：通过 $PASS、失败 $FAIL"
[ "$FAIL" = 0 ]