"use strict";
/* Q-Infer shared front-end layer (app.js).
   index.html   : three-tab shell + Chat panel (this file).
   monitor.html : request-monitor page — reuses the shared functions; the first
                  inline <script>window.__apiMonitorPage = true;</script> skips
                  the three-tab shell.
   All API calls go through api() — they carry Authorization: Bearer
   <sessionStorage key> automatically.
   Theme (light/dark) is stored in localStorage qinfer.theme; the first frame
   is locked by an inline script in the HTML files. */

function $(id) { return document.getElementById(id); }

/* ---- i18n: zh/en dictionaries + toggle (localStorage qinfer.lang, shared across pages) ---- */
const I18N = {
  zh: {
    "brand.tag": "本地推理",
    "lang.title": "切换语言（中 / EN）",
    "state.connecting": "连接中…",
    "chat.title": "本地推理引擎",
    "chat.sub": "模型在本机运行，数据不会离开您的设备。可在「关于」页设置 API 密钥。",
    "chat.sampler": "采样器",
    "chat.seedLabel": "seed（-1 = 随机）",
    "chat.inputPh": "输入消息，Enter 发送，Shift+Enter 换行",
    "chat.stop": "停止",
    "chat.you": "你",
    "chat.assistant": "助手",
    "chat.error": "错误",
    "chat.generating": "生成中…",
    "chat.prefilling": "正在处理提示词…",
    "chat.stopped": "已停止",
    "chat.stoppedSuffix": "（已停止）",
    "chat.failedPrefix": "请求失败：",
    "chat.idle": "空闲",
    "chat.newChat": "新会话已开始",
    "chat.exported": "已导出 qinfer-chat.md",
    "chat.noExport": "没有可导出的对话",
    "chat.exportTitle": "导出为 Markdown",
    "mon.state": "模型状态",
    "mon.retained": "已保留请求",
    "mon.retainedSub": "条请求记录",
    "mon.wall": "最近耗时",
    "mon.decode": "上次解码速度",
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
    "about.apiKeyNote": "密钥仅保存在浏览器本地存储中，跨标签页共享，不会上传到任何服务器。",
    "about.engine": "模型与引擎",
    "about.props": "能力（/props）",
    "about.hw": "硬件信息",
    "about.svc": "服务与协议",
    "req.copy": "复制",
    "req.copyFail": "复制失败",
    "req.loading": "加载详情…",
    "req.detailFail": "请求详情不可用：HTTP ",
    "req.loadFail": "加载失败：",
    "about.hide": "隐藏",
    "about.svcBody": "推理端点为 <span class=\"mono\" id=\"about-ports\"></span>。OpenAI 域走 <span class=\"mono\">/v1/chat/completions</span>（含 SSE 流式）、<span class=\"mono\">/v1/completions</span>、<span class=\"mono\">/v1/models</span>；Anthropic 域走 <span class=\"mono\">/v1/messages</span> 与 <span class=\"mono\">/v1/messages/count_tokens</span>；监控走 <span class=\"mono\">/metrics</span> 与 <span class=\"mono\">/api/requests</span>。",
    "theme.title": "浅色 / 深色",
    "theme.dark": "深色主题",
    "theme.light": "浅色主题",
    "hw.gpuUtil": "GPU 利用率",
    "hw.gpuMem": "GPU 显存",
    "hw.gpuTemp": "GPU 温度",
    "hw.gpuPower": "GPU 功率",
    "hw.pcieRx": "PCIe RX",
    "hw.cpu": "CPU",
    "hw.ram": "内存",
    "hw.diskRead": "磁盘读取",
    "hw.tokS": "tok/s",
    "hw.prefill": "prefill tok/s",
    "hw.igpuFreq": "核显频率",
    "state.loaded": "已加载",
    "state.unloaded": "未加载",
    "mon.lastDecode": "上次解码速度",
    "mon.noDone": "尚无完成请求",
    "err.keyInvalid": "API key 无效或未设置",
    "err.aboutLoad": "About 信息加载失败：",
    "req.yes": "是",
    "req.no": "否",
    "req.truncated": "（已截断）",
    "req.empty": "（空）",
    "req.none": "（无）",
    "req.apiResp": "API response（按记录重建）",
    "about.tempDefault": "temperature 默认",
    "about.topPDefault": "top_p 默认",
    "about.seedDefault": "seed 默认",
    "state.inFlight": " · 进行中 ",
    "state.unreachable": "不可达",
    "state.noResponse": "服务无响应",

    "tab.chat": "对话",
    "tab.monitor": "监控",
    "tab.about": "关于",
    "nav.views": "视图",
    "chat.newChatTitle": "新会话",
    "chat.exportTitleLong": "导出为 Markdown",
    "chat.send": "发送",
    "state.idle": "空闲",
    "state.reading": "阅读中",
    "state.generating": "生成中",
    "state.queuedBadge": "排队中",
    "state.errorBadge": "错误",
    "mon.waiting": "等待请求",
    "mon.ctxFill": "上下文填充",
    "mon.vram": "显存",
    "mon.sysRam": "系统内存",
    "mon.gpuTemp": "GPU 温度",
    "mon.recentReq": "最近请求",
    "mon.convCache": "对话缓存",
    "req.thTime": "时间",
    "req.thStatus": "状态",
    "req.thPrompt": "提示词",
    "req.thOutput": "输出",
    "req.thDecode": "解码 t/s",
    "req.thDuration": "耗时",
    "req.done": "完成",
    "req.maxTokens": "最大 token",
    "req.stoppedBadge": "已停止",
    "req.closed": "已关闭",
    "cc.cachedConv": "已缓存对话",
    "cc.theirMemory": "记忆占用",
    "cc.enabledNote": "继续已缓存的对话时，引擎会直接恢复状态，无需重新处理。",
    "cc.disabledNote": "引擎保留上次对话的状态，后续请求只需处理新增内容。",
    "cc.lastReq": "上次请求",
    "cc.reusedSince": "自启动以来已复用",
    "cc.parkedRestored": "已暂停 / 已恢复",
    "about.clientDesc": "任何兼容 OpenAI 或 Anthropic 协议的客户端均可使用以下地址。接口地址：",
    "about.endpointPaths": "端点路径",
    "about.openaiLabel": "OpenAI",
    "about.anthropicLabel": "Anthropic",
    "about.monitoringLabel": "监控",
    "about.model": "模型",
    "about.engineLabel": "引擎",
    "about.service": "服务",
    "about.started": "启动时间",
    "about.uptime": "运行时长",
    "about.maxCtx": "最大上下文",
    "about.imagesCap": "图像",
    "about.slotsCap": "槽位",
    "about.gpuCount": "GPU 数量",
    "about.igpu": "核显",
    "about.coresThreads": "核心 / 线程",
    "about.openaiBase": "OpenAI 接口地址",
    "about.anthropicBase": "Anthropic 接口地址",
    "about.modelName": "模型名称",
    "about.modelAlias": "模型别名",
    "about.totalSlots": "总槽位",
    "about.chatTemplate": "对话模板",
    "about.templater": "模板引擎",
    "about.vision": "视觉",
    "about.buildInfo": "构建信息",
    "mon.decodeNow": "实时解码",
    "mon.decodeLabel": "解码",
    "mon.notAvail": "不可用",
    "mon.limit": "上限",
    "mon.cores": "核心",
    "mon.threads": "线程",
    "conv.reusedOf": "复用",
    "conv.ofPrompt": "提示词 token 数",
    "conv.pctAll": "占全部提示词",
    "conv.of": "/",
    "conv.tokens": "token",
    "conv.ofAll": "占全部提示词",
    "toast.copied": "已复制",
    "toast.imgOver": "超过 20 MB",
    "toast.fileOver": "超过 512 KB",
    "toast.notText": "不是文本文件",
    "metric.speed": "速度",
    "metric.gpuLoad": "GPU 负载",
    "metric.vram": "显存",
    "metric.temp": "GPU 温度",
    "metric.power": "功率",
    "metric.pcie": "PCIe",
    "metric.cpu": "CPU",
    "metric.disk": "磁盘读取",
    "metric.igpu": "核显频率",

    "apiMon.title": "请求检查",
    "apiMon.keyLabel": "API 密钥",
    "apiMon.keyPh": "可选",
    "apiMon.back": "返回主面板",

    "chat.temperature": "温度",
    "chat.topP": "top_p",
    "filter.queued": "排队中",
    "filter.completed": "已完成",
    "filter.error": "错误",
    "req.tabInput": "输入",
    "req.tabOutput": "输出",
    "req.tabReasoning": "推理",
    "req.field.id": "ID",
    "req.field.path": "路径",
    "req.field.model": "模型",
    "req.field.state": "状态",
    "req.field.created": "创建时间",
    "req.field.stream": "流式",
    "req.field.responseFormat": "响应格式",
    "req.field.queueS": "排队耗时",
    "req.field.firstTokenS": "首 token",
    "req.field.wallclockS": "总耗时",
    "req.field.decodeTokS": "解码速度",
    "req.field.promptTokens": "提示词 token",
    "req.field.outputTokens": "输出 token",
    "aria.remove": "移除",
    "aria.copy": "复制",
    "misc.pastedImage": "粘贴的图片",
    "misc.gpu": "GPU",
    "misc.evicted": "已驱逐",
    "unit.tokS": "tok/s",
    "unit.tokens": "tokens",
    "unit.seconds": "s",
  },
  en: {
    "brand.tag": "Local inference",
    "lang.title": "Switch language (CN / EN)",
    "state.connecting": "Connecting…",
    "chat.title": "Local inference engine",
    "chat.sub": "The model runs locally — your data never leaves this device. Set the API key in the About tab.",
    "chat.sampler": "Sampler",
    "chat.seedLabel": "seed (-1 = random)",
    "chat.inputPh": "Type a message; Enter to send, Shift+Enter for a new line",
    "chat.stop": "Stop",
    "chat.you": "You",
    "chat.assistant": "Assistant",
    "chat.error": "Error",
    "chat.generating": "Generating…",
    "chat.prefilling": "Processing prompt…",
    "chat.stopped": "Stopped",
    "chat.stoppedSuffix": " (stopped)",
    "chat.failedPrefix": "Request failed: ",
    "chat.idle": "Idle",
    "chat.newChat": "New chat started",
    "chat.exported": "Exported qinfer-chat.md",
    "chat.noExport": "Nothing to export",
    "chat.exportTitle": "Export as Markdown",
    "mon.state": "Model state",
    "mon.retained": "Retained requests",
    "mon.retainedSub": "request records",
    "mon.wall": "Last duration",
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
    "about.apiKeyNote": "The key is stored only in this browser's local storage, shared across tabs, and never uploaded.",
    "about.engine": "Model and engine",
    "about.props": "Capabilities (/props)",
    "about.hw": "Hardware",
    "about.svc": "Service and protocols",
    "req.copy": "Copy",
    "req.copyFail": "Copy failed",
    "req.loading": "Loading details…",
    "req.detailFail": "Request details unavailable: HTTP ",
    "req.loadFail": "Load failed: ",
    "about.hide": "Hide",
    "about.svcBody": "Endpoints: <span class=\"mono\" id=\"about-ports\"></span>. OpenAI domain at <span class=\"mono\">/v1/chat/completions</span> (incl. SSE streaming), <span class=\"mono\">/v1/completions</span>, <span class=\"mono\">/v1/models</span>; Anthropic domain at <span class=\"mono\">/v1/messages</span> and <span class=\"mono\">/v1/messages/count_tokens</span>; monitoring at <span class=\"mono\">/metrics</span> and <span class=\"mono\">/api/requests</span>.",
    "theme.title": "Light / dark theme",
    "theme.dark": "Dark theme",
    "theme.light": "Light theme",
    "hw.gpuUtil": "GPU load",
    "hw.gpuMem": "VRAM",
    "hw.gpuTemp": "GPU temp",
    "hw.gpuPower": "GPU power",
    "hw.pcieRx": "PCIe RX",
    "hw.cpu": "CPU",
    "hw.ram": "RAM",
    "hw.diskRead": "Disk read",
    "hw.tokS": "tok/s",
    "hw.prefill": "prefill tok/s",
    "hw.igpuFreq": "iGPU freq",
    "state.loaded": "Loaded",
    "state.unloaded": "Not loaded",
    "mon.lastDecode": "Last decode speed",
    "mon.noDone": "No completed requests yet",
    "err.keyInvalid": "API key invalid or not set",
    "err.aboutLoad": "Failed to load About info: ",
    "req.yes": "Yes",
    "req.no": "No",
    "req.truncated": " (truncated)",
    "req.empty": " (empty)",
    "req.none": " (none)",
    "req.apiResp": "API response (rebuilt)",
    "about.tempDefault": "temperature default",
    "about.topPDefault": "top_p default",
    "about.seedDefault": "seed default",
    "state.inFlight": " · in flight ",
    "state.unreachable": "Unreachable",
    "state.noResponse": "Service not responding",

    "tab.chat": "Chat",
    "tab.monitor": "Monitor",
    "tab.about": "About",
    "nav.views": "Views",
    "chat.newChatTitle": "New chat",
    "chat.exportTitleLong": "Export chat as Markdown",
    "chat.send": "Send",
    "state.idle": "Idle",
    "state.reading": "Reading",
    "state.generating": "Generating",
    "state.queuedBadge": "Queued",
    "state.errorBadge": "Error",
    "mon.waiting": "Waiting for a request",
    "mon.ctxFill": "Context fill",
    "mon.vram": "VRAM",
    "mon.sysRam": "System RAM",
    "mon.gpuTemp": "GPU temperature",
    "mon.recentReq": "Recent requests",
    "mon.convCache": "Conversation cache",
    "req.thTime": "Time",
    "req.thStatus": "Status",
    "req.thPrompt": "Prompt",
    "req.thOutput": "Output",
    "req.thDecode": "Decode t/s",
    "req.thDuration": "Duration",
    "req.done": "Done",
    "req.maxTokens": "Max tokens",
    "req.stoppedBadge": "Stopped",
    "req.closed": "Closed",
    "cc.cachedConv": "Cached conversations",
    "cc.theirMemory": "Memory usage",
    "cc.enabledNote": "Continuing a cached conversation restores its state without reprocessing.",
    "cc.disabledNote": "The engine keeps the last conversation's state so follow-ups only process new content.",
    "cc.lastReq": "Last request",
    "cc.reusedSince": "Total reused",
    "cc.parkedRestored": "Parked / Restored",
    "about.clientDesc": "Any OpenAI- or Anthropic-compatible client can use these addresses. Base URL:",
    "about.endpointPaths": "Endpoint paths",
    "about.openaiLabel": "OpenAI",
    "about.anthropicLabel": "Anthropic",
    "about.monitoringLabel": "Monitoring",
    "about.model": "Model",
    "about.engineLabel": "Engine",
    "about.service": "Service",
    "about.started": "Started",
    "about.uptime": "Uptime",
    "about.maxCtx": "Max context",
    "about.imagesCap": "Images",
    "about.slotsCap": "Slots",
    "about.gpuCount": "GPU count",
    "about.igpu": "iGPU",
    "about.coresThreads": "Cores / threads",
    "about.openaiBase": "OpenAI base URL",
    "about.anthropicBase": "Anthropic base URL",
    "about.modelName": "Model name",
    "about.modelAlias": "Model alias",
    "about.totalSlots": "Total slots",
    "about.chatTemplate": "Chat template",
    "about.templater": "Templater",
    "about.vision": "Vision",
    "about.buildInfo": "Build info",
    "mon.decodeNow": "Current decode",
    "mon.decodeLabel": "Decode",
    "mon.notAvail": "not available",
    "mon.limit": "limit",
    "mon.cores": "cores",
    "mon.threads": "threads",
    "conv.reusedOf": "reused of",
    "conv.ofPrompt": "prompt tokens",
    "conv.pctAll": "of all prompt tokens",
    "conv.of": "of",
    "conv.tokens": "tokens",
    "conv.ofAll": "of all prompt tokens",
    "toast.copied": "Copied",
    "toast.imgOver": "is over 20 MB",
    "toast.fileOver": "is over 512 KB",
    "toast.notText": "is not a text file",
    "metric.speed": "Speed",
    "metric.gpuLoad": "GPU load",
    "metric.vram": "VRAM",
    "metric.temp": "GPU temp",
    "metric.power": "Power",
    "metric.pcie": "PCIe",
    "metric.cpu": "CPU",
    "metric.disk": "Disk read",
    "metric.igpu": "iGPU freq",

    "apiMon.title": "Request monitor",
    "apiMon.keyLabel": "API key",
    "apiMon.keyPh": "optional",
    "apiMon.back": "Back to main panel",

    "chat.temperature": "Temperature",
    "chat.topP": "top_p",
    "filter.queued": "Queued",
    "filter.completed": "Completed",
    "filter.error": "Error",
    "req.tabInput": "Input",
    "req.tabOutput": "Output",
    "req.tabReasoning": "Reasoning",
    "req.field.id": "ID",
    "req.field.path": "Path",
    "req.field.model": "Model",
    "req.field.state": "State",
    "req.field.created": "Created",
    "req.field.stream": "Stream",
    "req.field.responseFormat": "Response format",
    "req.field.queueS": "Queue time",
    "req.field.firstTokenS": "First token",
    "req.field.wallclockS": "Wall clock",
    "req.field.decodeTokS": "Decode speed",
    "req.field.promptTokens": "Prompt tokens",
    "req.field.outputTokens": "Output tokens",
    "aria.remove": "Remove",
    "aria.copy": "Copy",
    "misc.pastedImage": "pasted image",
    "misc.gpu": "GPU",
    "misc.evicted": "evicted",
    "unit.tokS": "tok/s",
    "unit.tokens": "tokens",
    "unit.seconds": "s",
  },
};

// Resolve the UI language: an explicit choice stored in localStorage wins;
// otherwise fall back to the browser language (zh prefix -> Chinese, anything
// else -> English). English is the default on this basis.
function uiLang() {
  const v = localStorage.getItem("qinfer.lang");
  if (v === "en" || v === "zh") return v;
  const nav = (navigator.language || "").toLowerCase();
  return nav.indexOf("zh") === 0 ? "zh" : "en";
}

function tr(key) {
  const lang = uiLang();
  return (I18N[lang] && I18N[lang][key]) || (I18N.en[key] || key);
}

// Translate raw server state strings into the current UI language
var STATE_TR = {
  idle: "state.idle",
  reading: "state.reading",
  generating: "state.generating",
  queued: "state.queuedBadge",
  error: "state.errorBadge",
};
function trState(s) {
  return STATE_TR[s] ? tr(STATE_TR[s]) : s;
}

// Apply the UI language: data-i18n text, data-i18n-ph placeholders,
// data-i18n-title/title tooltips; the toggle button label follows along.
function applyLang() {
  const lang = uiLang();
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
  document.querySelectorAll("[data-i18n-aria]").forEach(function (el) {
    el.setAttribute("aria-label", tr(el.dataset.i18nAria));
  });
  // Rich key with inline markup (e.g. the /props paragraph): safe here because
  // the dictionaries are static, no user input ever reaches them.
  document.querySelectorAll("[data-i18n-html]").forEach(function (el) {
    el.innerHTML = tr(el.dataset.i18nHtml);
  });
  const lb = $("btn-lang");
  if (lb) lb.textContent = lang === "en" ? "中" : "EN";
  const bd = document.querySelector(".st-header__brand small");
  // Only the shell pages' brand carries no data-i18n; the monitor page sets its
  // own title through the [data-i18n] pass above. Skip it to avoid overriding.
  if (bd && !bd.dataset.i18n) bd.textContent = tr("brand.tag");
  // Refresh role labels of already-rendered bubbles (dynamically built, no data-i18n).
  document.querySelectorAll("#msg-list .bubble").forEach(function (b) {
    const r = b.querySelector(".bubble-role");
    if (!r) return;
    r.textContent = b.classList.contains("user") ? tr("chat.you")
      : b.classList.contains("error") ? tr("chat.error") : tr("chat.assistant");
  });
  // Refresh dynamically-rendered panels that use tr() at build time
  // Re-render metric card labels (built once by initMetricCards)
  initMetricCards();
  if (window.lastMetrics) renderMonitorTab(window.lastMetrics);
  if (!$("panel-about").classList.contains("hidden")) refreshAbout();
}

function toggleLang() {
  const next = uiLang() === "en" ? "zh" : "en";
  try { localStorage.setItem("qinfer.lang", next); } catch (e) {}
  applyLang();
  showToast(next === "en" ? "English" : "中文", "info");
}

/* ---- API key (localStorage, shared across tabs; same approach as Strata) ---- */
function apiKey() { return localStorage.getItem("qinfer.api_key") || ""; }
function setApiKey(k) {
  try { localStorage.setItem("qinfer.api_key", k); } catch (e) {}
}

/* ---- Unified fetch: Authorization header; Content-Type when a body is present ---- */
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

/* ---- Page-level banner ---- */
function showBanner(msg, kind) {
  const b = $("banner");
  if (!b) return;
  b.textContent = msg;
  b.classList.toggle("err", kind === "err");
  b.classList.remove("hidden");
  window.clearTimeout(showBanner._t);
  showBanner._t = window.setTimeout(function () { b.classList.add("hidden"); }, 6000);
}

/* ---- Status dot ---- */
function setStatusDot(ok) {
  const d = $("status-dot");
  if (!d) return;
  d.classList.toggle("ok", ok);
  d.classList.toggle("bad", !ok);
}

/* ---- Header status pill: data-state drives the dot color (idle/generating/reading/queued/error) ---- */
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

/* ---- Toast notifications ---- */
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

/* ---- Theme toggle: localStorage qinfer.theme + html[data-theme] ---- */
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
  showToast(currentTheme() === "dark" ? tr("theme.dark") : tr("theme.light"), "info");
}

/* ---- Tab switching ---- */
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

/* ---- SSE reader: feeds data: frames to onEvent one by one; returns on [DONE] or stream end ---- */
async function readSSE(res, onEvent) {
  const reader = res.body.getReader();
  const dec = new TextDecoder();
  let buf = "";
  // Consume a decoded chunk, split into frames on \n\n; [DONE] returns true.
  function framesFrom(data) {
    buf += data;
    while (true) {
      const i = buf.indexOf("\n\n");
      if (i < 0) return false;              // half frame left, wait for the next chunk
      const frame = buf.slice(0, i);
      buf = buf.slice(i + 2);
      for (const line of frame.split("\n")) {
        if (!line.startsWith("data:")) continue;
        const d = line.slice(5).trim();
        if (d === "[DONE]") return true;
        if (!d) continue;
        try { onEvent(JSON.parse(d)); } catch (e) { /* skip malformed frame */ }
      }
    }
  }
  for (;;) {
    const r = await reader.read();
    const chunk = r.done ? dec.decode() : dec.decode(r.value, { stream: true });
    if (framesFrom(chunk)) return;
    if (r.done) break;
  }
  // Server closed before [DONE]: flush the remaining frame once (usually no output).
  framesFrom("");
}

/* ---- Sampler: /props defaults, sessionStorage value wins when present ---- */
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
  } catch (e) { /* never break first paint */ }
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
  } catch (e) { /* fall back to defaults */ }
}

function hideEmptyPrompt() {
  const e = $("chat-empty");
  if (e) e.style.display = "none";
}

function addMsg(role, text, attachList) {
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
  // Show attached images as thumbnails above the text
  if (attachList && attachList.length) {
    var imgs = document.createElement("div");
    imgs.className = "msg-images";
    attachList.forEach(function (a) {
      if (a.kind === "image" && a.url) {
        var img = document.createElement("img");
        img.src = a.url;
        img.alt = a.name;
        imgs.appendChild(img);
      }
    });
    if (imgs.childNodes.length) el.appendChild(imgs);
    // Store attachments on the element for collectMessages()
    el._attachments = attachList;
  }
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
    const attachList = b._attachments || [];
    // Build content: if there are image attachments, use OpenAI content array format
    const imageAttachments = attachList.filter(function (a) { return a.kind === "image" && a.url; });
    const fileAttachments = attachList.filter(function (a) { return a.kind === "file" && a.text; });
    if (imageAttachments.length) {
      var parts = [];
      // Include file text inline before the text content
      var fullText = txt;
      fileAttachments.forEach(function (f) {
        fullText = "File: " + f.name + "\n```\n" + f.text + "\n```\n\n" + fullText;
      });
      if (fullText) parts.push({ type: "text", text: fullText });
      imageAttachments.forEach(function (a) {
        parts.push({ type: "image_url", image_url: { url: a.url } });
      });
      messages.push({ role: role, content: parts });
    } else {
      // No images: plain text, possibly with file content prepended
      var content = txt;
      fileAttachments.forEach(function (f) {
        content = "File: " + f.name + "\n```\n" + f.text + "\n```\n\n" + content;
      });
      if (content || role === "user") messages.push({ role: role, content: content });
    }
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
  if ((!text && !attachments.length) || streaming) return;
  ta.value = "";
  // Capture and clear attachments before rendering the user bubble
  var currentAttachments = attachments.slice();
  attachments = [];
  renderAttachments();
  addMsg("user", text, currentAttachments);
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
  // Visible feedback during prefill: a waiting row (spinner + text), removed once
  // the first delta arrives and the real text takes over.
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
      } catch (e) { /* non-JSON error body */ }
      waitRow.remove();
      out.textContent = msg;
      out.parentElement.className = "bubble error";
      if (res.status === 401) showBanner(tr("err.keyInvalid"), "err");
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

/* ---- Export chat as Markdown ---- */
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

/* ---- Poll /metrics every second: status line and dot ---- */
function fmtRate(v) {
  if (v === null || v === undefined) return "\u2014";
  return v >= 1000 ? (v / 1000).toFixed(1) + "k" : v.toFixed(1);
}

async function pollMetrics() {
  try {
    const res = await api("/metrics");
    if (!res.ok) throw new Error("HTTP " + res.status);
    const d = await res.json();
    window.lastMetrics = d;
    const l = d.live || {};
    const hw = d.hardware || {};
    const state = l.state || "?";
    const queued = l.queued === undefined ? 0 : l.queued;
    const tok = fmtRate(l.tok_s);
    var chatState = $("chat-state");
    if (chatState) chatState.textContent = (tr("mon.stateLine") + " " + trState(state) + " \u00b7 " + tr("mon.queued") + " " + queued + " \u00b7 " + tok + " " + tr("unit.tokS"));
    setStatusDot(true);
    if (state !== "idle" && state !== "?") setPill(state, tr("mon.stateLine") + " " + trState(state) + " \u00b7 " + tok + " " + tr("unit.tokS"));
    // Push sparkline data even when monitor tab isn't visible, so switching to it shows history
    pushSpark("tok_s", l.tok_s);
    pushSpark("gpu_util", hw.gpu_util);
    pushSpark("gpu_mem_used", hw.gpu_mem_used);
    pushSpark("gpu_temp", hw.gpu_temp);
    pushSpark("gpu_power", hw.gpu_power);
    pushSpark("gpu_pcie_rx_mb", hw.gpu_pcie_rx_mb);
    pushSpark("cpu", hw.cpu);
    pushSpark("disk_read_mb", hw.disk_read_mb);
    pushSpark("igpu_freq_mhz", hw.igpu_freq_mhz);
  } catch (e) {
    setStatusDot(false);
    var chatStateEl = $("chat-state");
    if (chatStateEl) chatStateEl.textContent = tr("mon.unreachable");
    setPill("error", tr("mon.unreachable"));
  }
}

/* ---- Shared init: key, language, theme and tabs (index / monitor pages) ---- */
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
      kt.textContent = show ? tr("about.hide") : tr("about.show");
    });
  }
  const lg = $("btn-lang");
  if (lg) lg.addEventListener("click", toggleLang);
  applyLang();
  const th = $("btn-theme");
  if (th) {
    setTheme(currentTheme());   // sync the icon
    th.addEventListener("click", toggleTheme);
  }
}

/* ---- Request-record rendering layer (Monitor tab and /api-monitor share it) ---- */
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
  // records: /api/requests list view (drops input/output/reasoning).
  renderList: function (records, listEl, onPick) {
    listEl.innerHTML = "";
    if (!records.length) {
      const n = document.createElement("div");
      n.className = "note";
      n.textContent = tr("mon.noReq");
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
      ["stream", rec.stream ? tr("req.yes") : tr("req.no")],
      ["response_format", rec.response_format || "—"],
      ["queue_s", QinferRequests.fmtSec(rec.queue_s)],
      ["first_token_s", QinferRequests.fmtSec(rec.first_token_s)],
      ["wallclock_s", QinferRequests.fmtSec(rec.wallclock_s)],
      ["decode_tok_s", QinferRequests.fmtSec(rec.decode_tok_s)],
      ["prompt_tokens", rec.prompt_tokens === undefined ? "—" : rec.prompt_tokens],
      ["output_tokens", rec.output_tokens === undefined ? "—" : rec.output_tokens],
    ];
    const fieldLabels = {
      "id": tr("req.field.id"),
      "path": tr("req.field.path"),
      "model": tr("req.field.model"),
      "state": tr("req.field.state"),
      "created": tr("req.field.created"),
      "stream": tr("req.field.stream"),
      "response_format": tr("req.field.responseFormat"),
      "queue_s": tr("req.field.queueS"),
      "first_token_s": tr("req.field.firstTokenS"),
      "wallclock_s": tr("req.field.wallclockS"),
      "decode_tok_s": tr("req.field.decodeTokS"),
      "prompt_tokens": tr("req.field.promptTokens"),
      "output_tokens": tr("req.field.outputTokens"),
    };
    kv.forEach(function (p) {
      const dt = document.createElement("dt");
      dt.textContent = fieldLabels[p[0]] || p[0];
      const dd = document.createElement("dd");
      dd.textContent = p[1];
      dl.appendChild(dt);
      dl.appendChild(dd);
    });
    el.appendChild(dl);
    const tabs = [
      ["input", tr("req.tabInput") + (rec.input_truncated ? tr("req.truncated") : ""), rec.input || tr("req.empty")],
      ["output", tr("req.tabOutput") + (rec.output_truncated ? tr("req.truncated") : ""), rec.output || tr("req.empty")],
      ["reasoning", tr("req.tabReasoning"), rec.reasoning || tr("req.none")],
      ["response", tr("req.apiResp"), QinferRequests.apiResponseHtml(rec)],
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

/* ---- Attachments: paste/drop images and files, show as preview strip ---- */
var attachments = []; // {name, kind:"image"|"file", url?, text?}
var MAX_IMG_SIZE = 20 * 1024 * 1024; // 20 MB
var MAX_FILE_SIZE = 512 * 1024; // 512 KB

function renderAttachments() {
  var box = $("attachments");
  if (!box) return;
  box.innerHTML = "";
  if (!attachments.length) { box.style.display = "none"; return; }
  box.style.display = "";
  attachments.forEach(function (a, i) {
    var chip = document.createElement("span");
    chip.className = "st-attach-chip";
    if (a.kind === "image" && a.url) {
      var img = document.createElement("img");
      img.src = a.url;
      img.alt = a.name;
      chip.appendChild(img);
    } else {
      var ico = document.createElement("span");
      ico.innerHTML = '<svg class="st-icon st-icon--sm"><use href="/web/sprite.svg#i-attach"/></svg>';
      chip.appendChild(ico);
    }
    var name = document.createElement("span");
    name.className = "chip-name";
    name.textContent = a.name;
    chip.appendChild(name);
    var rm = document.createElement("button");
    rm.type = "button";
    rm.className = "st-btn st-btn--icon";
    rm.setAttribute("aria-label", tr("aria.remove"));
    rm.innerHTML = '<svg class="st-icon st-icon--sm"><use href="/web/sprite.svg#i-trash"/></svg>';
    rm.addEventListener("click", function () { attachments.splice(i, 1); renderAttachments(); });
    chip.appendChild(rm);
    box.appendChild(chip);
  });
}

function addImageFile(f) {
  if (f.size > MAX_IMG_SIZE) { showToast(tr("chat.error"), "warn", f.name + " " + tr("toast.imgOver")); return; }
  var r = new FileReader();
  r.onload = function () {
    attachments.push({ kind: "image", name: f.name || tr("misc.pastedImage"), url: r.result });
    renderAttachments();
  };
  r.readAsDataURL(f);
}

function addFiles(files) {
  for (var i = 0; i < files.length; i++) {
    var f = files[i];
    if (f.type.startsWith("image/")) { addImageFile(f); continue; }
    // Non-image files: show name preview, store as text if readable
    if (f.size > MAX_FILE_SIZE) { showToast(tr("chat.error"), "warn", f.name + " " + tr("toast.fileOver")); continue; }
    (function (file) {
      var r = new FileReader();
      r.onload = function () {
        var text = String(r.result);
        if (text.indexOf("\u0000") >= 0) { showToast(tr("chat.error"), "warn", file.name + " " + tr("toast.notText")); return; }
        attachments.push({ kind: "file", name: file.name, text: text });
        renderAttachments();
      };
      r.readAsText(file);
    })(f);
  }
}

function initAttachmentHandlers() {
  var composer = $("composer");
  if (!composer) return;
  // Paste handler: capture images from clipboard
  document.addEventListener("paste", function (e) {
    var items = (e.clipboardData || window.clipboardData);
    if (!items || !items.files || !items.files.length) return;
    var files = [];
    for (var i = 0; i < items.files.length; i++) {
      if (items.files[i].type.startsWith("image/")) files.push(items.files[i]);
    }
    if (files.length) { e.preventDefault(); addFiles(files); }
  });
  // Drag-and-drop on composer
  composer.addEventListener("dragover", function (e) {
    var types = e.dataTransfer ? e.dataTransfer.types : [];
    var hasFiles = false;
    for (var i = 0; i < types.length; i++) { if (types[i] === "Files") { hasFiles = true; break; } }
    if (!hasFiles) return;
    e.preventDefault();
    composer.classList.add("dragging");
  });
  composer.addEventListener("dragleave", function () { composer.classList.remove("dragging"); });
  composer.addEventListener("drop", function (e) {
    composer.classList.remove("dragging");
    if (!e.dataTransfer || !e.dataTransfer.files.length) return;
    e.preventDefault();
    addFiles(e.dataTransfer.files);
    $("chat-input").focus();
  });
}

/* ---- Chat setup ---- */
function initChat() {
  ["s-temp", "s-top-p", "s-seed"].forEach(function (id) {
    $(id).addEventListener("change", saveSampler);
  });
  initAttachmentHandlers();
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

/* ---- Monitor: state card + metric grid + context gauge + requests + conv cache ---- */
function kvTable(tbodyId, rows) {
  const tb = $(tbodyId);
  if (!tb) return;
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

// Metric card definitions for the 4-column grid
var MON_METRICS = [
  {key: "speed", labelKey: "metric.speed", icon: "i-gauge", unit: "t/s", series: "tok_s"},
  {key: "gpu", labelKey: "metric.gpuLoad", icon: "i-gpu", unit: "%", series: "gpu_util", max: 100},
  {key: "vram", labelKey: "metric.vram", icon: "i-layers", unit: "GB", series: "gpu_mem_used"},
  {key: "temp", labelKey: "metric.temp", icon: "i-thermometer", unit: "\u00b0C", series: "gpu_temp", tone: "warn"},
  {key: "power", labelKey: "metric.power", icon: "i-bolt", unit: "W", series: "gpu_power"},
  {key: "pcie", labelKey: "metric.pcie", icon: "i-link", unit: "", series: "gpu_pcie_rx_mb", tone: "info"},
  {key: "cpu", labelKey: "metric.cpu", icon: "i-cpu", unit: "%", series: "cpu", max: 100},
  {key: "disk", labelKey: "metric.disk", icon: "i-disk", unit: "MB/s", series: "disk_read_mb", tone: "info"},
  {key: "igpu", labelKey: "metric.igpu", icon: "i-gpu", unit: "MHz", series: "igpu_freq_mhz"},
];

function initMetricCards() {
  var el = $("metrics");
  if (!el) return;
  el.innerHTML = MON_METRICS.map(function (m) {
    var toneAttr = m.tone ? ' data-tone="' + m.tone + '"' : '';
    return '<div class="st-card metric-card"><div class="st-metric">' +
      '<span class="st-metric__label"><svg class="st-icon st-icon--sm"><use href="/web/sprite.svg#' + m.icon + '"/></svg>' + esc(tr(m.labelKey)) + '</span>' +
      '<span class="st-metric__value" id="mv-' + m.key + '">\u2013</span>' +
      '<span class="st-metric__sub" id="ms-' + m.key + '"></span>' +
      '<svg class="st-metric__spark" id="sp-' + m.key + '" viewBox="0 0 100 32" preserveAspectRatio="none"' + toneAttr + '>' +
      '<path class="area" fill="currentColor" opacity=".12"/>' +
      '<path class="line" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linejoin="round" stroke-linecap="round" vector-effect="non-scaling-stroke"/>' +
      '</svg></div></div>';
  }).join("");
}

// Render an SVG sparkline from an array of values (rolling window)
function sparkline(id, values, max) {
  var svg = $(id);
  if (!svg) return;
  var v = (values || []).map(function (x) { return x == null ? 0 : x; });
  if (v.length < 2) {
    svg.querySelector(".line").setAttribute("d", "");
    svg.querySelector(".area").setAttribute("d", "");
    return;
  }
  var top = Math.max(max || 0, Math.max.apply(null, v), 1);
  var pts = v.map(function (x, i) {
    return [(i / (v.length - 1)) * 100, 30 - (x / top) * 26];
  });
  var line = pts.map(function (p, i) { return (i ? "L" : "M") + p[0].toFixed(2) + "," + p[1].toFixed(2); }).join("");
  svg.querySelector(".line").setAttribute("d", line);
  svg.querySelector(".area").setAttribute("d", line + "L100,32L0,32Z");
}

function setMetricValue(key, value, unit, sub) {
  var el = $("mv-" + key);
  if (el) el.innerHTML = value == null ? "\u2013" : esc(value) + (unit ? '<small>' + esc(unit) + '</small>' : '');
  var s = $("ms-" + key);
  if (s) s.textContent = sub || "";
}

// Rolling sparkline history: 60 samples per series
var sparkHistory = {};
var SPARK_WINDOW = 60;

function pushSpark(series, value) {
  if (!sparkHistory[series]) sparkHistory[series] = [];
  sparkHistory[series].push(value);
  if (sparkHistory[series].length > SPARK_WINDOW) sparkHistory[series].shift();
}

async function refreshAbout() {
  if (!$("panel-about")) return;
  try {
    var m = await api("/metrics").then(function (r) { return r.json(); });
    var p = await api("/props").then(function (r) { return r.json(); });
    var s = await api("/v1/status").then(function (r) { return r.json(); });
    var eng = m.engine || {};
    var hs = m.hardware_static || {};
    var hw = m.hardware || {};
    renderAboutCards(eng, p, s, hs, hw);
  } catch (e) {
    showBanner(tr("err.aboutLoad") + e.message, "err");
  }
}

// ---- About page: stacked card layout (Strata-inspired) ----
function factsHtml(rows) {
  return rows.filter(function (r) { return r[1] != null && r[1] !== ""; })
    .map(function (r) {
      var copyBtn = r[2] ? '<button class="st-btn st-btn--icon" data-copy="' + esc(r[1]) + '" aria-label="' + esc(tr("aria.copy")) + '"><svg class="st-icon"><use href="/web/sprite.svg#i-copy"/></svg></button>' : "";
      return '<dt>' + esc(r[0]) + '</dt><dd>' + (r[2] ? '<code>' + esc(r[1]) + '</code>' + copyBtn : esc(r[1])) + '</dd>';
    }).join("");
}

function renderAboutCards(eng, props, status, hs, hw) {
  // Card 1: Model/engine facts
  var engEl = $("facts-engine");
  if (engEl) {
    engEl.innerHTML = factsHtml([
      [tr("about.model"), eng.model || "\u2014"],
      [tr("about.engineLabel"), status.engine || "\u2014"],
      [tr("about.service"), status.service || "\u2014"],
      [tr("about.started"), QinferRequests.timeOf(status.started)],
      [tr("about.uptime"), status.uptime_s !== undefined ? status.uptime_s.toFixed(0) + "s" : "\u2014"],
      [tr("about.maxCtx"), eng.max_context !== undefined ? String(eng.max_context) : "\u2014"],
      [tr("about.imagesCap"), eng.images !== undefined ? String(eng.images) : "\u2014"],
      [tr("about.slotsCap"), status.concurrency ? String(status.concurrency.serving) : "\u2014"],
    ]);
  }
  // Card 2: Hardware facts
  var hwEl = $("facts-hw");
  if (hwEl) {
    hwEl.innerHTML = factsHtml([
      [tr("misc.gpu"), hs.gpu_name || "\u2014"],
      [tr("about.gpuCount"), hs.gpu_count !== undefined ? String(hs.gpu_count) : "\u2014"],
      [tr("about.igpu"), hs.igpu_name || "\u2014"],
      [tr("hw.cpu"), hs.cpu_name || "\u2014"],
      [tr("about.coresThreads"), (hs.cores !== undefined ? hs.cores : "\u2014") + " / " + (hs.threads !== undefined ? hs.threads : "\u2014")],
      [tr("hw.ram"), hw.ram_total ? (hw.ram_total / 1073741824).toFixed(1) + " GB" : "\u2014"],
    ]);
  }
  // Card 3: API URLs with copy buttons
  var apiEl = $("facts-api");
  if (apiEl) {
    var base = location.origin;
    apiEl.innerHTML = factsHtml([
      [tr("about.openaiBase"), base + "/v1", true],
      [tr("about.anthropicBase"), base, true],
      [tr("about.modelName"), eng.model || "\u2014", true],
    ]);
  }
  // Card 4: Ports info
  var portsEl = $("about-ports");
  if (portsEl) portsEl.textContent = base;
  // Card 5: /props details
  var propsEl = $("facts-props");
  if (propsEl) {
    var dg = props.default_generation_settings || {};
    var prm = dg.params || {};
    var caps = props.chat_template_caps || {};
    var ct = props.chat_template || "";
    propsEl.innerHTML = factsHtml([
      [tr("about.modelAlias"), props.model_alias || "\u2014"],
      [tr("about.totalSlots"), props.total_slots !== undefined ? String(props.total_slots) : "\u2014"],
      [tr("about.chatTemplate"), ct.length > 120 ? ct.slice(0, 120) + "\u2026" : ct],
      [tr("about.templater"), caps.templater || "\u2014"],
      [tr("about.vision"), props.modalities && props.modalities.vision !== undefined ? String(props.modalities.vision) : "\u2014"],
      [tr("about.tempDefault"), prm.temperature !== undefined ? String(prm.temperature) : "\u2014"],
      [tr("about.topPDefault"), prm.top_p !== undefined ? String(prm.top_p) : "\u2014"],
      [tr("about.seedDefault"), prm.seed !== undefined ? String(prm.seed) : "\u2014"],
      [tr("about.buildInfo"), props.build_info || "\u2014"],
    ]);
  }
}

// Copy-to-clipboard delegation for facts with data-copy
document.addEventListener("click", function (e) {
  var btn = e.target.closest("[data-copy]");
  if (!btn) return;
  var text = btn.dataset.copy;
  navigator.clipboard.writeText(text).then(function () {
    showToast(tr("toast.copied"), "success");
  }).catch(function () {
    // Fallback for non-HTTPS contexts
    var ta = document.createElement("textarea");
    ta.value = text;
    document.body.appendChild(ta);
    ta.select();
    document.execCommand("copy");
    ta.remove();
    showToast(tr("toast.copied"), "success");
  });
});

let reqCache_ = [];
let activeReqId_ = null;

function renderFiltered() {
  var el = $("req-search");
  if (!el) return;
  const q = (el.value || "").toLowerCase();
  var fEl = $("req-filter");
  const f = fEl ? fEl.value : "";
  let list = reqCache_;
  if (q || f) {
    list = list.filter(function (r) {
      const hay = ((r.path || "") + " " + (r.model || "") + " " + (r.state || "") + " " + (r.id || "")).toLowerCase();
      return (!q || hay.indexOf(q) >= 0) && (!f || r.state === f);
    });
  }
  var reqList = $("req-list");
  if (reqList) QinferRequests.renderList(list, reqList, function (rec) {
    activeReqId_ = rec.id;
    QinferRequests.loadDetail(rec.id, $("req-detail"));
  });
}

async function refreshRequests() {
  try {
    const res = await api("/api/requests");
    if (!res.ok) throw new Error("HTTP " + res.status);
    const d = await res.json();
    reqCache_ = d.requests || [];
    renderFiltered();
  } catch (e) { /* retry on the next poll */ }
}

// ---- Full monitor render from /metrics data ----
function renderMonitorTab(m) {
  var live = m.live || {}, hw = m.hardware || {}, st = m.hardware_static || {}, eng = m.engine || {};
  var h = m.history || {};
  var requests = m.requests || [];

  // State card badges
  var on = live.queued > 0 ? "queued" : live.state;
  var badges = document.querySelectorAll("#state-badges .st-badge");
  for (var i = 0; i < badges.length; i++) {
    badges[i].classList.toggle("on", badges[i].dataset.s === on || badges[i].dataset.s === live.state);
  }

  // State card label/detail/progress
  var prog = $("state-progress");
  var label = tr("mon.waiting"), detail = "", pct = 0;
  if (live.state === "reading") {
    label = tr("state.reading");
    if (prog) prog.dataset.tone = "info";
    if (live.prompt_total) {
      pct = (100 * live.prompt_read) / live.prompt_total;
      detail = fmtNum(live.prompt_read) + " / " + fmtNum(live.prompt_total) + " " + tr("unit.tokens") + " \u00b7 " + fmtNum(pct) + "%";
    } else {
      detail = fmtNum(live.prompt_tokens) + " " + tr("unit.tokens");
    }
  } else if (live.state === "generating") {
    label = live.phase ? live.phase[0].toUpperCase() + live.phase.slice(1) : tr("state.generating");
    if (prog) delete prog.dataset.tone;
    pct = live.max_tokens ? Math.min(100, (100 * live.generated) / live.max_tokens) : 0;
    detail = fmtNum(live.generated) + " " + tr("unit.tokens") + " \u00b7 " + fmtNum(live.tok_s, 1) + " " + tr("unit.tokS");
  }
  var stateLabel = $("state-label");
  if (stateLabel) stateLabel.textContent = label;
  var stateDetail = $("state-detail");
  if (stateDetail) stateDetail.textContent = detail;
  var stateBar = $("state-bar");
  if (stateBar) stateBar.style.width = pct + "%";

  // Push current values into rolling sparkline history
  pushSpark("tok_s", live.tok_s);
  pushSpark("gpu_util", hw.gpu_util);
  pushSpark("gpu_mem_used", hw.gpu_mem_used);
  pushSpark("gpu_temp", hw.gpu_temp);
  pushSpark("gpu_power", hw.gpu_power);
  pushSpark("gpu_pcie_rx_mb", hw.gpu_pcie_rx_mb);
  pushSpark("cpu", hw.cpu);
  pushSpark("disk_read_mb", hw.disk_read_mb);
  pushSpark("igpu_freq_mhz", hw.igpu_freq_mhz);

  // Metric cards
  var speed = live.state === "generating" ? live.tok_s : null;
  setMetricValue("speed", speed == null ? null : fmtNum(speed, 1), "t/s",
    live.state === "generating" ? tr("mon.decodeNow") : tr("mon.decodeLabel"));
  setMetricValue("gpu", hw.gpu_util == null ? null : fmtNum(hw.gpu_util), "%",
    st.gpu_name || (st.gpu_note ? tr("mon.notAvail") : ""));
  var vramUsed = hw.gpu_mem_used != null ? (hw.gpu_mem_used / 1073741824).toFixed(2) : null;
  var vramTotal = hw.gpu_mem_total ? " / " + (hw.gpu_mem_total / 1073741824).toFixed(0) + " GB" : " GB";
  setMetricValue("vram", vramUsed, vramTotal, "");
  setMetricValue("temp", hw.gpu_temp == null ? null : fmtNum(hw.gpu_temp), "\u00b0C", "");
  setMetricValue("power", hw.gpu_power == null ? null : fmtNum(hw.gpu_power), "W",
    hw.gpu_power_limit ? tr("mon.limit") + " " + fmtNum(hw.gpu_power_limit) + " W" : "");
  setMetricValue("pcie", hw.gpu_pcie_rx_mb == null ? null : fmtNum(hw.gpu_pcie_rx_mb, 1), "MB/s", "");
  setMetricValue("cpu", hw.cpu == null ? null : fmtNum(hw.cpu), "%",
    st.threads ? (st.cores ? st.cores + " " + tr("mon.cores") + " \u00b7 " : "") + st.threads + " " + tr("mon.threads") : "");
  setMetricValue("disk", hw.disk_read_mb == null ? null : fmtNum(hw.disk_read_mb, 1), "MB/s", "");
  // iGPU frequency
  var igpuSub = "";
  if (hw.igpu_freq_max_mhz != null) igpuSub = tr("mon.limit") + " " + fmtNum(hw.igpu_freq_max_mhz) + " MHz";
  if (st.igpu_name) igpuSub += (igpuSub ? " · " : "") + st.igpu_name;
  setMetricValue("igpu", hw.igpu_freq_mhz == null ? null : fmtNum(hw.igpu_freq_mhz), "MHz", igpuSub);

  // Sparklines
  sparkline("sp-speed", sparkHistory.tok_s);
  sparkline("sp-gpu", sparkHistory.gpu_util, 100);
  sparkline("sp-vram", sparkHistory.gpu_mem_used, hw.gpu_mem_total);
  sparkline("sp-temp", sparkHistory.gpu_temp, 90);
  sparkline("sp-power", sparkHistory.gpu_power, hw.gpu_power_limit);
  sparkline("sp-pcie", sparkHistory.gpu_pcie_rx_mb);
  sparkline("sp-cpu", sparkHistory.cpu, 100);
  sparkline("sp-disk", sparkHistory.disk_read_mb);
  sparkline("sp-igpu", sparkHistory.igpu_freq_mhz, hw.igpu_freq_max_mhz);

  // Context fill gauge
  var ctx = eng.max_context || 0;
  var used = 0;
  if (live.state !== "idle") used = (live.prompt_tokens || 0) + (live.generated || 0);
  var frac = ctx ? Math.min(1, used / ctx) : 0;
  var ctxFill = $("ctx-fill");
  if (ctxFill) {
    ctxFill.setAttribute("stroke-dasharray", (235.6 * frac).toFixed(1) + " 314.2");
    ctxFill.style.opacity = 235.6 * frac >= 3 ? "1" : "0";
  }
  var ctxPct = $("ctx-pct");
  if (ctxPct) ctxPct.textContent = Math.round(frac * 100) + "%";
  var ctxSub = $("ctx-sub");
  if (ctxSub) ctxSub.textContent = ctx ? fmtNum(used) + " / " + fmtNum(ctx) : "\u2013";

  // VRAM/RAM/temp progress bars in context card
  var vramBar = $("vram-bar");
  if (vramBar && hw.gpu_mem_total) vramBar.style.width = Math.min(100, (100 * hw.gpu_mem_used) / hw.gpu_mem_total) + "%";
  var vramText = $("vram-text");
  if (vramText) vramText.textContent = hw.gpu_mem_used != null ? (hw.gpu_mem_used / 1073741824).toFixed(1) + " / " + (hw.gpu_mem_total / 1073741824).toFixed(0) + " GB" : "\u2013";
  var ramBar = $("ram-bar");
  if (ramBar && hw.ram_total) ramBar.style.width = Math.min(100, (100 * hw.ram_used) / hw.ram_total) + "%";
  var ramText = $("ram-text");
  if (ramText) ramText.textContent = hw.ram_total ? (hw.ram_used / 1073741824).toFixed(1) + " / " + (hw.ram_total / 1073741824).toFixed(0) + " GB" : "\u2013";
  var tempBar = $("temp-bar");
  if (tempBar) tempBar.style.width = hw.gpu_temp != null ? Math.min(100, hw.gpu_temp) + "%" : "0%";
  var tempText = $("temp-text");
  if (tempText) tempText.textContent = hw.gpu_temp != null ? fmtNum(hw.gpu_temp) + " \u00b0C" : "\u2013";
  // iGPU frequency bar
  var igpuBar = $("igpu-bar");
  if (igpuBar && hw.igpu_freq_max_mhz) igpuBar.style.width = Math.min(100, (100 * (hw.igpu_freq_mhz || 0)) / hw.igpu_freq_max_mhz) + "%";
  var igpuText = $("igpu-text");
  if (igpuText) {
    if (hw.igpu_freq_mhz != null) {
      igpuText.textContent = fmtNum(hw.igpu_freq_mhz) + " / " + fmtNum(hw.igpu_freq_max_mhz) + " MHz";
    } else {
      igpuText.textContent = "\u2013";
    }
  }

  // Recent requests table (last 10)
  var body = $("req-body");
  if (body) {
    if (!requests.length) {
      body.innerHTML = '<tr><td colspan="6" class="muted">' + esc(tr("mon.noReq")) + '</td></tr>';
    } else {
      body.innerHTML = requests.slice(0, 10).map(function (r) {
        var badge = {stop: ["", tr("req.done")], length: ["", tr("req.maxTokens")], cancel: ["st-badge--queued", tr("req.stoppedBadge")],
                     disconnect: ["st-badge--queued", tr("req.closed")], error: ["st-badge--error", tr("chat.error")]};
        var bInfo = badge[r.finish] || ["", r.finish || "\u2013"];
        var t = new Date(r.time * 1000).toLocaleTimeString([], {hour: "2-digit", minute: "2-digit", second: "2-digit"});
        return '<tr><td>' + esc(t) + '</td>' +
          '<td><span class="st-badge ' + bInfo[0] + '">' + esc(bInfo[1]) + '</span></td>' +
          '<td class="num">' + fmtNum(r.prompt_tokens) + '</td>' +
          '<td class="num">' + fmtNum(r.output_tokens) + '</td>' +
          '<td class="num">' + fmtNum(r.decode_tok_s, 1) + '</td>' +
          '<td class="num">' + fmtNum(r.duration_s, 1) + ' ' + tr("unit.seconds") + '</td></tr>';
      }).join("");
    }
  }

  // Conversation cache
  renderConvCache(m.conversation_cache);
}

function fmtNum(n, d) {
  if (n == null || Number.isNaN(n)) return "\u2013";
  return Number(n).toLocaleString(undefined, {maximumFractionDigits: d || 0, minimumFractionDigits: d || 0});
}

function renderConvCache(c) {
  var card = $("cc-card");
  if (!card) return;
  card.style.display = c ? "" : "none";
  if (!c) return;
  var bars = $("cc-bars");
  if (bars) bars.hidden = !c.enabled;
  if (c.enabled) {
    var slotsText = $("cc-slots-text");
    if (slotsText) slotsText.textContent = fmtNum(c.parked) + " / " + fmtNum(c.slots);
    var slotsBar = $("cc-slots-bar");
    if (slotsBar) slotsBar.style.width = (c.slots ? Math.min(100, (100 * c.parked) / c.slots) : 0) + "%";
    var budget = c.budget_mib * 1048576;
    var memText = $("cc-mem-text");
    if (memText) {
      var bytesGB = (c.bytes != null && !isNaN(c.bytes)) ? (c.bytes / 1073741824).toFixed(1) : "\u2013";
      var budgetGB = (budget && !isNaN(budget)) ? (budget / 1073741824).toFixed(0) : "\u2013";
      memText.textContent = bytesGB + " / " + budgetGB + " GB";
    }
    var memBar = $("cc-mem-bar");
    if (memBar) memBar.style.width = (budget && c.bytes != null ? Math.min(100, (100 * c.bytes) / budget) : 0) + "%";
  }
  var sum = $("cc-sum");
  if (sum) sum.textContent = c.requests ? fmtNum(c.requests_reused) + " " + tr("conv.reusedOf") + " " + fmtNum(c.requests) + " " + tr("conv.ofPrompt") : "";
  var factsEl = $("cc-facts");
  if (factsEl) {
    var share = c.prompt_tokens ? " (" + fmtNum((100 * c.reused_tokens) / c.prompt_tokens) + "% " + tr("conv.pctAll") + ")" : "";
    factsEl.innerHTML =
      (c.last_prompt != null ? '<dt>' + esc(tr("cc.lastReq")) + '</dt><dd>' + fmtNum(c.last_reused || 0) + ' ' + tr("conv.reusedOf") + ' ' + fmtNum(c.last_prompt) + ' ' + tr("conv.ofPrompt") + '</dd>' : '') +
      (c.requests ? '<dt>' + esc(tr("cc.reusedSince")) + '</dt><dd>' + fmtNum(c.reused_tokens) + ' ' + tr("conv.tokens") + share + '</dd>' : '') +
      (c.enabled ? '<dt>' + esc(tr("cc.parkedRestored")) + '</dt><dd>' + fmtNum(c.parks) + ' / ' + fmtNum(c.restores) + (c.evictions ? ' \u00b7 ' + fmtNum(c.evictions) + ' ' + tr("misc.evicted") : '') + '</dd>' : '');
  }
  var note = $("cc-note");
  if (note) {
    note.textContent = c.enabled
      ? tr("cc.enabledNote")
      : tr("cc.disabledNote");
  }
}

async function refreshMonitor() {
  if (!$("panel-monitor")) return;
  try {
    var res = await api("/metrics");
    if (!res.ok) throw new Error("HTTP " + res.status);
    var d = await res.json();
    window.lastMetrics = d;
    renderMonitorTab(d);
    setStatusDot(true);
    // Update the header pill from metrics
    var live = d.live || {};
    var tok = live.tok_s != null ? fmtNum(live.tok_s, 1) : "\u2014";
    if (live.state !== "idle" && live.state !== "?") setPill(live.state, tr("mon.stateLine") + " " + trState(live.state) + " \u00b7 " + tok + " " + tr("unit.tokS"));
  } catch (e) {
    setStatusDot(false);
    var stateLabel = $("state-label");
    if (stateLabel) stateLabel.textContent = tr("state.unreachable");
    var stateDetail = $("state-detail");
    if (stateDetail) stateDetail.textContent = tr("state.noResponse");
  }
  refreshRequests();
}

function initMonitor() {
  if (!$("panel-monitor")) return;
  initMetricCards();
  refreshMonitor();
  window.setInterval(refreshMonitor, 1000);
  var searchEl = $("req-search");
  if (searchEl) searchEl.addEventListener("input", renderFiltered);
  var filterEl = $("req-filter");
  if (filterEl) filterEl.addEventListener("change", renderFiltered);
}

document.addEventListener("DOMContentLoaded", function () {
  initCommon();
  if (!$("panel-chat") || window.__apiMonitorPage) return; // api-monitor page: shared helpers only
  initChat();
  initMonitor();
  loadModel();
  loadProps();
  pollMetrics();
  window.setInterval(pollMetrics, 1000);
});