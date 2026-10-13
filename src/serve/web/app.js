"use strict";
/* Q-Infer 前端公共层（app.js）。
   index.html   : 三 tab 入口 + Chat 面板（本文件）。
   monitor.html : 请求检查页——复用本文件定义的公共函数，首行前置
                  <script>window.__apiMonitorPage = true;</script> 跳过三 tab 入口。
   所有 API 调用统一走 api()：自动带 Authorization: Bearer <sessionStorage 密钥>。
   主题切换（浅/深）记 localStorage qinfer.theme，首帧锁在 index.html 内联脚本。 */

function $(id) { return document.getElementById(id); }

/* ---- i18n：zh/en 字典 + 切换（localStorage qinfer.lang，页面间共享） ---- */
const I18N = {
  zh: {
    "brand.tag": "本地推理",
    "lang.title": "切换语言（中 / EN）",
    "state.connecting": "连接中…",
    "chat.title": "向引擎提问吧",
    "chat.sub": "模型在本机运行，数据不出本机。密钥在 About 页设置。",
    "chat.sampler": "采样器",
    "chat.seedLabel": "seed（-1 = 随机）",
    "chat.inputPh": "输入消息，Enter 发送，Shift+Enter 换行",
    "chat.stop": "停止",
    "chat.you": "你",
    "chat.assistant": "助手",
    "chat.error": "错误",
    "chat.generating": "生成中…",
    "chat.prefilling": "正在阅读提示词…（真机预填约 2.5 分钟）",
    "chat.stopped": "已停止",
    "chat.stoppedSuffix": "（已停止）",
    "chat.failedPrefix": "请求失败：",
    "chat.idle": "空闲",
    "chat.newChat": "新会话已开始",
    "chat.exported": "已导出 qinfer-chat.md",
    "chat.noExport": "没有可导出的对话",
    "chat.exportTitle": "导出为 Markdown",
    "mon.state": "模型状态",
    "mon.retained": "请求保留",
    "mon.retainedSub": "条（/api/requests）",
    "mon.wall": "最近 wall-clock",
    "mon.decode": "最近解码速度",
    "mon.searchPh": "搜索路径 / 模型 / 状态",
    "mon.filterTitle": "状态过滤",
    "mon.all": "全部状态",
    "mon.noReq": "暂无请求记录",
    "mon.pickHint": "点左侧一条记录查看详情",
    "mon.stateLine": "状态",
    "mon.queued": "排队",
    "mon.unreachable": "服务不可达，正在重试…",
    "about.apiKey": "连接设置",
    "about.apiKeyLabel": "API 密钥",
    "about.apiKeyHint": "仅当服务端启动时带了 --api-key 才需要",
    "about.apiKeyPh": "不需要",
    "about.show": "显示",
    "about.apiKeyNote": "密钥只保存在本浏览器（localStorage，跨标签页共享，不上传）。",
    "about.engine": "模型与引擎",
    "about.props": "能力（/props）",
    "about.hw": "本机硬件",
    "about.svc": "服务与协议",
    "req.copy": "复制",
    "req.copyFail": "复制失败",
    "req.loading": "加载详情…",
    "req.detailFail": "请求详情不可用：HTTP ",
    "req.loadFail": "加载失败：",
  },
  en: {
    "brand.tag": "Local inference",
    "lang.title": "Switch language (CN / EN)",
    "state.connecting": "Connecting…",
    "chat.title": "Ask the engine anything",
    "chat.sub": "The model runs on this PC; nothing leaves it. Set the API key in the About tab.",
    "chat.sampler": "Sampler",
    "chat.seedLabel": "seed (-1 = random)",
    "chat.inputPh": "Type a message; Enter to send, Shift+Enter for a new line",
    "chat.stop": "Stop",
    "chat.you": "You",
    "chat.assistant": "Assistant",
    "chat.error": "Error",
    "chat.generating": "Generating…",
    "chat.prefilling": "Reading the prompt… (prefill can take minutes on real weights)",
    "chat.stopped": "Stopped",
    "chat.stoppedSuffix": " (stopped)",
    "chat.failedPrefix": "Request failed: ",
    "chat.idle": "Idle",
    "chat.newChat": "New chat started",
    "chat.exported": "Exported qinfer-chat.md",
    "chat.noExport": "Nothing to export",
    "chat.exportTitle": "Export as Markdown",
    "mon.state": "Model state",
    "mon.retained": "Requests kept",
    "mon.retainedSub": "entries (/api/requests)",
    "mon.wall": "Last wall-clock",
    "mon.decode": "Last decode speed",
    "mon.searchPh": "Search path / model / state",
    "mon.filterTitle": "Filter by state",
    "mon.all": "All states",
    "mon.noReq": "No requests yet",
    "mon.pickHint": "Pick a request on the left to inspect it",
    "mon.stateLine": "State",
    "mon.queued": "queued",
    "mon.unreachable": "Server unreachable, retrying…",
    "about.apiKey": "Connection settings",
    "about.apiKeyLabel": "API key",
    "about.apiKeyHint": "Only needed if the server was started with --api-key",
    "about.apiKeyPh": "Not needed",
    "about.show": "Show",
    "about.apiKeyNote": "The key stays in this browser only (localStorage, shared across tabs; never uploaded).",
    "about.engine": "Model and engine",
    "about.props": "Capabilities (/props)",
    "about.hw": "This PC",
    "about.svc": "Service and protocols",
    "req.copy": "Copy",
    "req.copyFail": "Copy failed",
    "req.loading": "Loading details…",
    "req.detailFail": "Request details unavailable: HTTP ",
    "req.loadFail": "Load failed: ",
  },
};

function tr(key) {
  const lang = localStorage.getItem("qinfer.lang") === "en" ? "en" : "zh";
  return (I18N[lang] && I18N[lang][key]) || (I18N.zh[key] || key);
}

// 应用语言：data-i18n 文本、data-i18n-ph 占位符、data-i18n-title/title 提示；语言钮标签随动。
function applyLang() {
  const lang = localStorage.getItem("qinfer.lang") === "en" ? "en" : "zh";
  document.documentElement.lang = lang === "en" ? "en" : "zh-CN";
  document.querySelectorAll("[data-i18n]").forEach(function (el) {
    el.textContent = tr(el.dataset.i18n);
  });
  document.querySelectorAll("[data-i18n-ph]").forEach(function (el) {
    el.setAttribute("placeholder", tr(el.dataset.i18nPh));
  });
  document.querySelectorAll("[data-i18n-title]").forEach(function (el) {
    const k = el.dataset.i18nTitle;
    el.setAttribute("title", tr(k));
    el.setAttribute("aria-label", tr(k));
  });
  const lb = $("btn-lang");
  if (lb) lb.textContent = lang === "en" ? "中" : "EN";
  const bd = document.querySelector(".st-header__brand small");
  if (bd) bd.textContent = tr("brand.tag");
  // 已渲染的语言气泡 role 标签随语言刷新（动态生成，无 data-i18n）。
  document.querySelectorAll("#msg-list .bubble").forEach(function (b) {
    const r = b.querySelector(".bubble-role");
    if (!r) return;
    r.textContent = b.classList.contains("user") ? tr("chat.you")
      : b.classList.contains("error") ? tr("chat.error") : tr("chat.assistant");
  });
}

function toggleLang() {
  const next = localStorage.getItem("qinfer.lang") === "en" ? "zh" : "en";
  try { localStorage.setItem("qinfer.lang", next); } catch (e) {}
  applyLang();
  showToast(next === "en" ? "English" : "中文", "info");
}

/* ---- API 密钥（localStorage，跨标签页共享；与 Strata 同款） ---- */
function apiKey() { return localStorage.getItem("qinfer.api_key") || ""; }
function setApiKey(k) {
  try { localStorage.setItem("qinfer.api_key", k); } catch (e) {}
}

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

/* ---- 顶栏状态 pill：data-state 驱动 dot 颜色（idle/generating/reading/queued/error） ---- */
function setPill(state, text) {
  const pill = $("pill");
  if (!pill) return;
  pill.dataset.state = state === "generating" || state === "loading" ? "generating"
    : state === "reading" ? "reading"
    : state === "queued" ? "queued"
    : state === "error" ? "error"
    : "idle";
  const t = $("pill-text");
  if (t && text) t.textContent = text;
}

/* ---- toast 通知 ---- */
function showToast(title, kind, msg) {
  const box = $("toasts");
  if (!box) return;
  const el = document.createElement("div");
  el.className = "st-toast st-toast--" + (kind || "info");
  const icon = kind === "success" ? "i-check" : kind === "error" ? "i-error" : kind === "warn" ? "i-warning" : "i-info";
  el.innerHTML =
    '<svg class="st-icon"><use href="/web/sprite.svg#' + icon + '"/></svg>' +
    '<div><div class="st-toast__title">' + esc(title) + "</div>" +
    (msg ? '<div class="small">' + esc(msg) + "</div>" : "") + "</div>";
  box.appendChild(el);
  window.setTimeout(function () { el.remove(); }, 5000);
}

/* ---- 主题切换：localStorage qinfer.theme + html[data-theme] ---- */
function currentTheme() { return document.documentElement.dataset.theme === "dark" ? "dark" : "light"; }
function setTheme(t) {
  document.documentElement.dataset.theme = t;
  try { localStorage.setItem("qinfer.theme", t); } catch (e) {}
  const ic = $("theme-icon");
  if (ic) ic.setAttribute("href", "/web/sprite.svg#" + (t === "dark" ? "i-sun" : "i-moon"));
  const tg = $("dark-toggle");
  if (tg) tg.setAttribute("aria-checked", String(t === "dark"));
}
function toggleTheme() {
  setTheme(currentTheme() === "dark" ? "light" : "dark");
  showToast(currentTheme() === "dark" ? "深色主题" : "浅色主题", "info");
}

/* ---- tab 切换 ---- */
function switchTab(name) {
  document.querySelectorAll(".tab-btn").forEach(function (b) {
    const on = b.dataset.tab === name;
    b.classList.toggle("active", on);
    b.setAttribute("aria-selected", String(on));
  });
  document.querySelectorAll(".panel").forEach(function (p) {
    p.classList.toggle("hidden", p.id !== "panel-" + name);
  });
  if (name === "about") refreshAbout();
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
  } catch (e) { /* 默认值兜底 */ }
}

function hideEmptyPrompt() {
  const e = $("chat-empty");
  if (e) e.style.display = "none";
}

function addMsg(role, text) {
  hideEmptyPrompt();
  const el = document.createElement("div");
  el.className = "bubble " + role;
  const label = document.createElement("div");
  label.className = "bubble-role";
  label.textContent = role === "user" ? tr("chat.you") : role === "error" ? tr("chat.error") : tr("chat.assistant");
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
  // prefill 期可见反馈：等待行（转圈 + 文案）→ 首个 delta 到达时移除，正文接管。
  const waitRow = document.createElement("div");
  waitRow.className = "msg-waiting";
  const spinner = document.createElement("span");
  spinner.className = "spinner";
  const waitTxt = document.createElement("span");
  waitTxt.textContent = tr("chat.prefilling");
  waitRow.appendChild(spinner);
  waitRow.appendChild(waitTxt);
  out.parentElement.appendChild(waitRow);
  setStreaming(true);
  setPill("reading", tr("chat.prefilling"));
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
      waitRow.remove();
      out.textContent = msg;
      out.parentElement.className = "bubble error";
      if (res.status === 401) showBanner("API 密钥无效或未设置", "err");
      setPill("error", tr("chat.failed"));
      return;
    }
    await readSSE(res, function (ev) {
      if (ev.error) {
        waitRow.remove();
        out.textContent = ev.error.message || tr("chat.failed");
        out.parentElement.className = "bubble error";
        setPill("error", tr("chat.failed"));
      } else if (ev.choices && ev.choices[0]) {
        const d = ev.choices[0].delta || {};
        if (d.content !== undefined) {
          if (waitRow.parentElement) waitRow.remove();
          out.textContent += d.content;
          setPill("generating", tr("chat.generating"));
        }
        $("msg-list").scrollTop = $("msg-list").scrollHeight;
      }
    });
    if (waitRow.parentElement) waitRow.remove();
  } catch (e) {
    waitRow.remove();
    if (e.name === "AbortError") {
      out.textContent += tr("chat.stoppedSuffix");
      setPill("idle", tr("chat.stopped"));
    } else {
      out.textContent = tr("chat.failedPrefix") + e.message;
      out.parentElement.className = "bubble error";
      setPill("error", tr("chat.failed"));
    }
  } finally {
    setStreaming(false);
    abortCtrl = null;
    setPill("idle", tr("chat.idle"));
  }
}

function stopChat() {
  if (abortCtrl) abortCtrl.abort();
}

function newChat() {
  if (streaming) return;
  $("msg-list").innerHTML = "";
  const e = $("chat-empty");
  if (e) e.style.display = "";
  setPill("idle", tr("chat.newChat"));
  showToast(tr("chat.newChat"), "info");
}

/* ---- 导出对话为 Markdown ---- */
function exportChat() {
  const rows = [];
  document.querySelectorAll("#msg-list .bubble").forEach(function (b) {
    const role = b.classList.contains("user") ? tr("chat.you") : tr("chat.assistant");
    const txt = b.querySelector(".bubble-body").textContent;
    rows.push("### " + role + "\n\n" + txt + "\n");
  });
  if (!rows.length) { showToast(tr("chat.noExport"), "warn"); return; }
  const blob = new Blob([rows.join("\n")], { type: "text/markdown;charset=utf-8" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "qinfer-chat.md";
  a.click();
  URL.revokeObjectURL(a.href);
  showToast(tr("chat.exported"), "success");
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
    window.lastMetrics = d;
    const l = d.live || {};
    const state = l.state || "?";
    const queued = l.queued === undefined ? 0 : l.queued;
    const tok = fmtRate(l.tok_s);
    $("chat-state").textContent = (tr("mon.stateLine") + " " + state + " · " + tr("mon.queued") + " " + queued + " · " + tok + " tok/s");
    setStatusDot(true);
    if (state !== "idle" && state !== "?") setPill(state, tr("mon.stateLine") + " " + state + " · " + tok + " tok/s");
  } catch (e) {
    setStatusDot(false);
    $("chat-state").textContent = tr("mon.unreachable");
    setPill("error", tr("mon.unreachable"));
  }
}

/* ---- 公共初始化：Key、语言、主题与 tab（index / monitor 两页通用） ---- */
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
      kt.textContent = show ? (tr("about.show") === "显示" ? "隐藏" : "Hide") : tr("about.show");
    });
  }
  const lg = $("btn-lang");
  if (lg) lg.addEventListener("click", toggleLang);
  applyLang();
  const th = $("btn-theme");
  if (th) {
    setTheme(currentTheme());   // 同步图标
    th.addEventListener("click", toggleTheme);
  }
}

/* ---- 请求记录渲染层（Monitor tab 与 /api-monitor 共用） ---- */
window.QinferRequests = {
  badge: function (state) {
    const cls = state === "completed" ? "ok" : state === "error" ? "err" : state === "queued" ? "busy" : "";
    return '<span class="badge ' + cls + '">' + esc(state) + "</span>";
  },
  fmtSec: function (v) {
    if (v === null || v === undefined) return "—";
    return v.toFixed(2) + "s";
  },
  timeOf: function (unix) {
    if (!unix) return "—";
    return new Date(unix * 1000).toLocaleString();
  },
  // records：/api/requests 列表视图（剥掉 input/output/reasoning）。
  renderList: function (records, listEl, onPick) {
    listEl.innerHTML = "";
    if (!records.length) {
      const n = document.createElement("div");
      n.className = "note";
      n.textContent = "暂无请求记录";
      listEl.appendChild(n);
      return;
    }
    records.forEach(function (rec) {
      const row = document.createElement("div");
      row.className = "req-row";
      row.dataset.id = rec.id;
      row.innerHTML =
        '<span class="mono">' + esc(rec.id) + "</span>" +
        QinferRequests.badge(rec.state) +
        "<span>" + esc(rec.path || "") + "</span>";
      row.title = (rec.model || "") + " · " + QinferRequests.fmtSec(rec.wallclock_s);
      row.addEventListener("click", function () {
        listEl.querySelectorAll(".req-row").forEach(function (r) {
          r.classList.toggle("active", r.dataset.id === rec.id);
        });
        onPick(rec);
      });
      listEl.appendChild(row);
    });
  },
  apiResponseHtml: function (rec) {
    const usage = {
      prompt_tokens: rec.prompt_tokens || 0,
      completion_tokens: rec.output_tokens || 0,
      total_tokens: (rec.prompt_tokens || 0) + (rec.output_tokens || 0),
    };
    const resp = {
      id: "msg_" + rec.id,
      object: "chat.completion",
      created: Math.floor(rec.created || 0),
      model: rec.model,
      choices: [{
        index: 0,
        message: { role: "assistant", content: rec.output || "" },
        finish_reason: rec.state === "error" ? "error" : "stop",
      }],
      usage: usage,
    };
    return JSON.stringify(resp, null, 2);
  },
  renderDetail: function (rec, el) {
    el.innerHTML = "";
    const dl = document.createElement("dl");
    dl.className = "kv";
    const kv = [
      ["id", rec.id],
      ["path", rec.path],
      ["model", rec.model],
      ["state", rec.state],
      ["created", QinferRequests.timeOf(rec.created)],
      ["stream", rec.stream ? "是" : "否"],
      ["response_format", rec.response_format || "—"],
      ["queue_s", QinferRequests.fmtSec(rec.queue_s)],
      ["first_token_s", QinferRequests.fmtSec(rec.first_token_s)],
      ["wallclock_s", QinferRequests.fmtSec(rec.wallclock_s)],
      ["decode_tok_s", QinferRequests.fmtSec(rec.decode_tok_s)],
      ["prompt_tokens", rec.prompt_tokens === undefined ? "—" : rec.prompt_tokens],
      ["output_tokens", rec.output_tokens === undefined ? "—" : rec.output_tokens],
    ];
    kv.forEach(function (p) {
      const dt = document.createElement("dt");
      dt.textContent = p[0];
      const dd = document.createElement("dd");
      dd.textContent = p[1];
      dl.appendChild(dt);
      dl.appendChild(dd);
    });
    el.appendChild(dl);
    const tabs = [
      ["input", "Input" + (rec.input_truncated ? "（已截断）" : ""), rec.input || "（空）"],
      ["output", "Output" + (rec.output_truncated ? "（已截断）" : ""), rec.output || "（空）"],
      ["reasoning", "Reasoning", rec.reasoning || "（无）"],
      ["response", "API response（按记录重建）", QinferRequests.apiResponseHtml(rec)],
    ];
    const set = document.createElement("div");
    set.className = "tabset";
    const box = document.createElement("div");
    box.className = "codebox";
    tabs.forEach(function (t, i) {
      const b = document.createElement("button");
      b.className = "tab-btn" + (i === 0 ? " active" : "");
      b.textContent = t[1];
      b.addEventListener("click", function () {
        set.querySelectorAll(".tab-btn").forEach(function (x) { x.classList.remove("active"); });
        b.classList.add("active");
        box.textContent = t[2];
        box.dataset.text = t[2];
      });
      set.appendChild(b);
    });
    const copy = document.createElement("button");
    copy.className = "btn ghost";
    copy.textContent = tr("req.copy");
    copy.addEventListener("click", function () {
      const txt = box.dataset.text !== undefined ? box.dataset.text : box.textContent;
      navigator.clipboard.writeText(txt).catch(function () { showBanner(tr("req.copyFail"), "err"); });
    });
    set.appendChild(copy);
    box.dataset.text = tabs[0][2];
    box.textContent = tabs[0][2];
    el.appendChild(set);
    el.appendChild(box);
  },
  loadDetail: function (id, el) {
    el.innerHTML = '<div class="note">' + tr("req.loading") + "</div>";
    return api("/api/requests?id=" + encodeURIComponent(id)).then(function (res) {
      if (!res.ok) {
        el.innerHTML = '<div class="note">' + esc(tr("req.detailFail") + " HTTP " + res.status) + "</div>";
        return;
      }
      return res.json().then(function (d) { QinferRequests.renderDetail(d, el); });
    }).catch(function (e) {
      el.innerHTML = '<div class="note">' + esc(tr("req.loadFail") + e.message) + "</div>";
    });
  },
  sparkline: function (canvas, points) {
    const w = canvas.width;
    const h = canvas.height;
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, w, h);
    let min = Infinity;
    let max = -Infinity;
    let any = 0;
    points.forEach(function (p) {
      if (p !== null) { any++; if (p < min) min = p; if (p > max) max = p; }
    });
    if (!any || max === min) return;
    const accent = getComputedStyle(document.documentElement).getPropertyValue("--accent").trim() || "#10b981";
    ctx.strokeStyle = accent;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    let pen = false;
    for (let i = 0; i < points.length; i++) {
      const p = points[i];
      if (p === null) { pen = false; continue; }
      const x = (i / (points.length - 1)) * w;
      const y = h - 3 - ((p - min) / (max - min)) * (h - 6);
      if (pen) ctx.lineTo(x, y); else ctx.moveTo(x, y);
      pen = true;
    }
    ctx.stroke();
  },
};

/* ---- Chat 初始化 ---- */
function initChat() {
  ["s-temp", "s-top-p", "s-seed"].forEach(function (id) {
    $(id).addEventListener("change", saveSampler);
  });
  $("btn-send").addEventListener("click", sendChat);
  $("btn-stop").addEventListener("click", stopChat);
  $("btn-new").addEventListener("click", newChat);
  $("btn-export").addEventListener("click", exportChat);
  const composer = $("composer");
  if (composer) composer.addEventListener("submit", function (e) { e.preventDefault(); sendChat(); });
  $("chat-input").addEventListener("keydown", function (e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendChat();
    }
  });
}

/* ---- Monitor：四卡片 + 请求列表/详情 + 10 系列 sparkline ---- */
function kvTable(tbodyId, rows) {
  const tb = $(tbodyId);
  tb.innerHTML = "";
  rows.forEach(function (r) {
    const tr = document.createElement("tr");
    const th = document.createElement("th");
    th.textContent = r[0];
    const td = document.createElement("td");
    td.textContent = r[1];
    tr.appendChild(th);
    tr.appendChild(td);
    tb.appendChild(tr);
  });
}

async function refreshAbout() {
  if (!$("panel-about")) return;
  try {
    const m = await api("/metrics").then(function (r) { return r.json(); });
    const p = await api("/props").then(function (r) { return r.json(); });
    const s = await api("/v1/status").then(function (r) { return r.json(); });
    const eng = m.engine || {};
    kvTable("about-engine", [
      ["model", eng.model || "—"],
      ["max_context", eng.max_context === undefined ? "—" : String(eng.max_context)],
      ["images", eng.images !== undefined ? String(eng.images) : "—"],
      ["engine", s.engine || "—"],
      ["service", s.service || "—"],
      ["started", QinferRequests.timeOf(s.started)],
      ["uptime_s", s.uptime_s === undefined ? "—" : s.uptime_s.toFixed(0) + "s"],
      ["slots", s.concurrency ? String(s.concurrency.serving) : "—"],
    ]);
    const dg = p.default_generation_settings || {};
    const prm = dg.params || {};
    const caps = p.chat_template_caps || {};
    const ct = p.chat_template || "";
    kvTable("about-props", [
      ["model_alias", p.model_alias || "—"],
      ["total_slots", p.total_slots === undefined ? "—" : String(p.total_slots)],
      ["chat_template", ct.length > 120 ? ct.slice(0, 120) + "…" : ct],
      ["templater", caps.templater || "—"],
      ["vision", p.modalities && p.modalities.vision !== undefined ? String(p.modalities.vision) : "—"],
      ["temperature 默认", prm.temperature === undefined ? "—" : String(prm.temperature)],
      ["top_p 默认", prm.top_p === undefined ? "—" : String(prm.top_p)],
      ["seed 默认", prm.seed === undefined ? "—" : String(prm.seed)],
      ["build_info", p.build_info || "—"],
    ]);
    const hs = m.hardware_static || {};
    kvTable("about-hardware", [
      ["gpu_name", hs.gpu_name || "—"],
      ["gpu_count", hs.gpu_count === undefined ? "—" : String(hs.gpu_count)],
      ["gpu_note", hs.gpu_note || "—"],
      ["cpu_name", hs.cpu_name || "—"],
      ["cores", hs.cores === undefined ? "—" : String(hs.cores)],
      ["threads", hs.threads === undefined ? "—" : String(hs.threads)],
    ]);
    $("about-ports").textContent = location.protocol + "//" + location.host;
  } catch (e) {
    showBanner("About 信息加载失败：" + e.message, "err");
  }
}

const SPARK_SERIES = [
  { key: "gpu_util", label: "GPU 利用率", fmt: function (v) { return v.toFixed(1) + "%"; } },
  { key: "gpu_mem_used", label: "GPU 显存", fmt: function (v) { return (v / 1073741824).toFixed(2) + " GB"; } },
  { key: "gpu_temp", label: "GPU 温度", fmt: function (v) { return v.toFixed(0) + " ℃"; } },
  { key: "gpu_power", label: "GPU 功率", fmt: function (v) { return v.toFixed(0) + " W"; } },
  { key: "gpu_pcie_rx_mb", label: "PCIe RX", fmt: function (v) { return v.toFixed(1) + " MB/s"; } },
  { key: "cpu", label: "CPU", fmt: function (v) { return v.toFixed(1) + "%"; } },
  { key: "ram_used", label: "内存", fmt: function (v) { return (v / 1073741824).toFixed(1) + " GB"; } },
  { key: "disk_read_mb", label: "磁盘读", fmt: function (v) { return v.toFixed(1) + " MB/s"; } },
  { key: "tok_s", label: "tok/s", fmt: function (v) { return v.toFixed(1); } },
  { key: "prefill_tok_s_mean", label: "prefill tok/s", fmt: function (v) { return v.toFixed(1); } },
];

function renderSparklines(h) {
  if (!$("spark-grid")) return;
  const grid = $("spark-grid");
  grid.innerHTML = "";
  const hw = h.hardware || {};
  const history = h.history || {};
  SPARK_SERIES.forEach(function (s) {
    const cell = document.createElement("div");
    cell.className = "spark-cell";
    const head = document.createElement("div");
    head.className = "spark-head";
    const name = document.createElement("span");
    name.className = "name";
    name.textContent = s.label;
    const val = document.createElement("span");
    val.className = "val";
    const cur = hw[s.key];
    val.textContent = cur === null || cur === undefined ? "—" : s.fmt(cur);
    head.appendChild(name);
    head.appendChild(val);
    const cv = document.createElement("canvas");
    cv.width = 120;
    cv.height = 30;
    QinferRequests.sparkline(cv, history[s.key] || []);
    cell.appendChild(head);
    cell.appendChild(cv);
    grid.appendChild(cell);
  });
}

let reqCache_ = [];
let activeReqId_ = null;

function renderFiltered() {
  const q = ($("req-search").value || "").toLowerCase();
  const f = $("req-filter").value;
  let list = reqCache_;
  if (q || f) {
    list = list.filter(function (r) {
      const hay = ((r.path || "") + " " + (r.model || "") + " " + (r.state || "") + " " + (r.id || "")).toLowerCase();
      return (!q || hay.indexOf(q) >= 0) && (!f || r.state === f);
    });
  }
  QinferRequests.renderList(list, $("req-list"), function (rec) {
    activeReqId_ = rec.id;
    QinferRequests.loadDetail(rec.id, $("req-detail"));
  });
}

async function refreshRequests() {
  try {
    const res = await api("/api/requests");
    if (!res.ok) throw new Error("HTTP " + res.status);
    const d = await res.json();
    $("card-kept").textContent = d.retention === undefined ? "—" : d.retention;
    reqCache_ = d.requests || [];
    renderFiltered();
  } catch (e) { /* 下一次轮询再试 */ }
}

async function refreshMonitor() {
  if (!$("panel-monitor")) return;
  try {
    const res = await api("/v1/status");
    if (!res.ok) throw new Error("HTTP " + res.status);
    const st = await res.json();
    const lt = st.last_timings;
    $("card-state").textContent = st.loaded ? "已加载" : "未加载";
    $("card-state-sub").textContent = (st.model || "") +
      (st.activity && st.activity.in_flight ? " · 进行中 " + st.activity.in_flight : "");
    $("card-wall").textContent = lt && lt.wallclock_s !== undefined ? lt.wallclock_s.toFixed(2) + "s" : "—";
    $("card-wall-sub").textContent = lt && lt.prompt_tokens !== undefined
      ? lt.prompt_tokens + " prompt → " + lt.output_tokens + " out" : "尚无完成请求";
    $("card-tok").textContent = lt && lt.decode_tok_s !== undefined ? lt.decode_tok_s.toFixed(1) + " tok/s" : "—";
    $("card-tok-sub").textContent = "上次请求解码速度";
    setStatusDot(true);
  } catch (e) {
    $("card-state").textContent = "不可达";
    $("card-state-sub").textContent = "服务无响应";
    setStatusDot(false);
  }
  if (window.lastMetrics) renderSparklines(window.lastMetrics);
  refreshRequests();
}

function initMonitor() {
  if (!$("panel-monitor")) return;
  refreshMonitor();
  window.setInterval(refreshMonitor, 2000);
  $("req-search").addEventListener("input", renderFiltered);
  $("req-filter").addEventListener("change", renderFiltered);
}

document.addEventListener("DOMContentLoaded", function () {
  initCommon();
  if (!$("panel-chat") || window.__apiMonitorPage) return; // api-monitor 页：只复用公共函数
  initChat();
  initMonitor();
  loadModel();
  loadProps();
  pollMetrics();
  window.setInterval(pollMetrics, 1000);
});