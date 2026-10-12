"use strict";
/* Q-Infer 前端公共层（app.js）。
   index.html   : 三 tab 入口 + Chat 面板（本文件）。
   monitor.html : 请求检查页——复用本文件定义的公共函数，首行前置
                  <script>window.__apiMonitorPage = true;</script> 跳过三 tab 入口。
   所有 API 调用统一走 api()：自动带 Authorization: Bearer <sessionStorage 密钥>。 */

function $(id) { return document.getElementById(id); }

/* ---- API 密钥（sessionStorage，随页面会话保留） ---- */
function apiKey() { return sessionStorage.getItem("qinfer_api_key") || ""; }
function setApiKey(k) { sessionStorage.setItem("qinfer_api_key", k); }

/* ---- 统一 fetch：带密钥头；有 body 时补 Content-Type ---- */
async function api(path, opts) {
  opts = opts || {};
  const h = new Headers(opts.headers || {});
  const k = apiKey();
  if (k) h.set("Authorization", "Bearer " + k);
  if (opts.body !== undefined && !h.has("Content-Type")) h.set("Content-Type", "application/json");
  return fetch(path, Object.assign({}, opts, { headers: h }));
}

function esc(s) {
  return String(s).replace(/[&<>"']/g, function (c) {
    return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c];
  });
}

/* ---- 页面级提示条 ---- */
function showBanner(msg, kind) {
  const b = $("banner");
  if (!b) return;
  b.textContent = msg;
  b.classList.toggle("err", kind === "err");
  b.classList.remove("hidden");
  window.clearTimeout(showBanner._t);
  showBanner._t = window.setTimeout(function () { b.classList.add("hidden"); }, 6000);
}

/* ---- 状态点 ---- */
function setStatusDot(ok) {
  const d = $("status-dot");
  if (!d) return;
  d.classList.toggle("ok", ok);
  d.classList.toggle("bad", !ok);
}

/* ---- tab 切换 ---- */
function switchTab(name) {
  document.querySelectorAll(".tab-btn").forEach(function (b) {
    b.classList.toggle("active", b.dataset.tab === name);
  });
  document.querySelectorAll(".panel").forEach(function (p) {
    p.classList.toggle("hidden", p.id !== "panel-" + name);
  });
}

/* ---- SSE 读取：把 data: 帧逐个喂给 onEvent；[DONE] 或流结束返回 ---- */
async function readSSE(res, onEvent) {
  const reader = res.body.getReader();
  const dec = new TextDecoder();
  let buf = "";
  // 吞一段解码文本，按 \n\n 切帧；遇 [DONE] 返回 true（消费完毕）。
  function framesFrom(data) {
    buf += data;
    while (true) {
      const i = buf.indexOf("\n\n");
      if (i < 0) return false;              // 剩半帧，等下一次拼接
      const frame = buf.slice(0, i);
      buf = buf.slice(i + 2);
      for (const line of frame.split("\n")) {
        if (!line.startsWith("data:")) continue;
        const d = line.slice(5).trim();
        if (d === "[DONE]") return true;
        if (!d) continue;
        try { onEvent(JSON.parse(d)); } catch (e) { /* 跳过坏帧 */ }
      }
    }
  }
  for (;;) {
    const r = await reader.read();
    const chunk = r.done ? dec.decode() : dec.decode(r.value, { stream: true });
    if (framesFrom(chunk)) return;
    if (r.done) break;
  }
  // 服务端在 [DONE] 前关闭：把残帧最后拼一次（多半无输出，半帧静默跳过）。
  framesFrom("");
}

/* ---- 采样器：默认取 /props，sessionStorage 有持久值时优先 ---- */
function saveSampler() {
  const s = {
    temperature: parseFloat($("s-temp").value),
    top_p: parseFloat($("s-top-p").value),
    seed: parseInt($("s-seed").value, 10),
  };
  sessionStorage.setItem("qinfer_sampler", JSON.stringify(s));
}
function loadSampler() {
  const raw = sessionStorage.getItem("qinfer_sampler");
  if (!raw) return false;
  try {
    const s = JSON.parse(raw);
    if (s.temperature !== undefined) $("s-temp").value = s.temperature;
    if (s.top_p !== undefined) $("s-top-p").value = s.top_p;
    if (s.seed !== undefined) $("s-seed").value = s.seed;
    return !!$("s-temp");
  } catch (e) { return false; }
}

/* ---- Chat ---- */
let modelId = "x";
let streaming = false;
let abortCtrl = null;

async function loadModel() {
  try {
    const res = await api("/v1/models");
    if (!res.ok) return;
    const d = await res.json();
    if (d.data && d.data.length) modelId = d.data[0].id;
  } catch (e) { /* 首屏不打断 */ }
}

async function loadProps() {
  try {
    const res = await api("/props");
    if (!res.ok) return;
    const d = await res.json();
    if (!loadSampler()) {
      const p = (d.default_generation_settings || {}).params || {};
      if (p.temperature !== undefined) $("s-temp").value = p.temperature;
      if (p.top_p !== undefined) $("s-top-p").value = p.top_p;
      if (p.seed !== undefined) $("s-seed").value = p.seed;
    }
    if (d.model_alias) $("brand-model").textContent = d.model_alias;
  } catch (e) { /* 默认值兜底 */ }
}

function addMsg(role, text) {
  const el = document.createElement("div");
  el.className = "bubble " + role;
  const label = document.createElement("div");
  label.className = "bubble-role";
  label.textContent = role === "user" ? "你" : role === "error" ? "错误" : "助手";
  const body = document.createElement("div");
  body.className = "bubble-body";
  body.textContent = text;
  el.appendChild(label);
  el.appendChild(body);
  $("msg-list").appendChild(el);
  $("msg-list").scrollTop = $("msg-list").scrollHeight;
  return body;
}

function collectMessages() {
  const messages = [];
  document.querySelectorAll("#msg-list .bubble").forEach(function (b) {
    const role = b.classList.contains("user") ? "user" : "assistant";
    const txt = b.querySelector(".bubble-body").textContent;
    if (txt || role === "user") messages.push({ role: role, content: txt });
  });
  return messages;
}

function setStreaming(on) {
  streaming = on;
  $("btn-send").disabled = on;
  $("btn-new").disabled = on;
  $("btn-stop").classList.toggle("hidden", !on);
}

async function sendChat() {
  const ta = $("chat-input");
  const text = ta.value.trim();
  if (!text || streaming) return;
  ta.value = "";
  addMsg("user", text);
  const body = {
    model: modelId,
    messages: collectMessages(),
    stream: true,
    max_tokens: 512,
    temperature: parseFloat($("s-temp").value),
    top_p: parseFloat($("s-top-p").value),
    seed: parseInt($("s-seed").value, 10),
  };
  const out = addMsg("assistant", "");
  setStreaming(true);
  abortCtrl = new AbortController();
  try {
    const res = await api("/v1/chat/completions", {
      method: "POST",
      body: JSON.stringify(body),
      signal: abortCtrl.signal,
      headers: { "Accept": "text/event-stream" },
    });
    if (!res.ok) {
      let msg = "HTTP " + res.status;
      try {
        const e = await res.json();
        if (e.error && e.error.message) msg = e.error.message;
      } catch (e) { /* 非 JSON 错误体 */ }
      out.textContent = msg;
      out.parentElement.className = "bubble error";
      if (res.status === 401) showBanner("API 密钥无效或未设置", "err");
      return;
    }
    await readSSE(res, function (ev) {
      if (ev.error) {
        out.textContent = ev.error.message || "生成失败";
        out.parentElement.className = "bubble error";
      } else if (ev.choices && ev.choices[0]) {
        const d = ev.choices[0].delta || {};
        if (typeof d.content === "string") out.textContent += d.content;
        $("msg-list").scrollTop = $("msg-list").scrollHeight;
      }
    });
  } catch (e) {
    if (e.name === "AbortError") {
      out.textContent += "（已停止）";
    } else {
      out.textContent = "请求失败：" + e.message;
      out.parentElement.className = "bubble error";
    }
  } finally {
    setStreaming(false);
    abortCtrl = null;
  }
}

function stopChat() {
  if (abortCtrl) abortCtrl.abort();
}

function newChat() {
  if (streaming) return;
  $("msg-list").innerHTML = "";
  const note = document.createElement("div");
  note.className = "note";
  note.textContent = "新会话已开始。";
  $("msg-list").appendChild(note);
}

/* ---- 每秒轮询 /metrics：状态行与状态点 ---- */
function fmtRate(v) {
  if (v === null || v === undefined) return "—";
  return v >= 1000 ? (v / 1000).toFixed(1) + "k" : v.toFixed(1);
}

async function pollMetrics() {
  try {
    const res = await api("/metrics");
    if (!res.ok) throw new Error("HTTP " + res.status);
    const d = await res.json();
    const l = d.live || {};
    const state = l.state || "?";
    const queued = l.queued === undefined ? 0 : l.queued;
    const tok = fmtRate(l.tok_s);
    $("chat-state").textContent = "状态 " + state + " · 排队 " + queued + " · " + tok + " tok/s";
    setStatusDot(true);
  } catch (e) {
    setStatusDot(false);
    $("chat-state").textContent = "服务不可达，正在重试…";
  }
}

/* ---- 公共初始化：Key 与 tab（index / monitor 两页通用） ---- */
function initCommon() {
  document.querySelectorAll(".tab-btn").forEach(function (b) {
    b.addEventListener("click", function () { switchTab(b.dataset.tab); });
  });
  const keyInput = $("api-key");
  if (keyInput) {
    keyInput.value = apiKey();
    keyInput.addEventListener("change", function () { setApiKey(keyInput.value.trim()); });
  }
  const kt = $("btn-key-toggle");
  if (kt) {
    kt.addEventListener("click", function () {
      const show = keyInput.type === "password";
      keyInput.type = show ? "text" : "password";
      kt.textContent = show ? "隐藏" : "显示";
    });
  }
}

/* ---- Chat 初始化 ---- */
function initChat() {
  ["s-temp", "s-top-p", "s-seed"].forEach(function (id) {
    $(id).addEventListener("change", saveSampler);
  });
  $("btn-send").addEventListener("click", sendChat);
  $("btn-stop").addEventListener("click", stopChat);
  $("btn-new").addEventListener("click", newChat);
  $("chat-input").addEventListener("keydown", function (e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendChat();
    }
  });
}

document.addEventListener("DOMContentLoaded", function () {
  initCommon();
  if (!$("panel-chat") || window.__apiMonitorPage) return; // api-monitor 页：只复用公共函数
  initChat();
  loadModel();
  loadProps();
  pollMetrics();
  window.setInterval(pollMetrics, 1000);
});
