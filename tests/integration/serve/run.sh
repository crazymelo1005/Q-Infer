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