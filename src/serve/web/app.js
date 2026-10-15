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

/* ---- Lightweight Markdown renderer (no external dependencies) ---- */
var QinferMD = {
  // Escape HTML to prevent XSS
  escapeHtml: function(text) {
    var map = { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#039;' };
    return text.replace(/[&<>"']/g, function(m) { return map[m]; });
  },

  // A table row carries at least one pipe and is not the |---|---| separator.
  isTableRow: function(line) {
    var t = (line || '').trim();
    return t.indexOf('|') >= 0 && !this.isTableSep(t);
  },
  isTableSep: function(line) {
    var t = (line || '').trim();
    if (t.indexOf('|') < 0) return false;
    return /^\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)*\|?$/.test(t);
  },
  // If lines[i] opens a table (header row + separator), return { html, next } where
  // next is the first line after the table; otherwise null.
  tableAt: function(lines, i, inlineCodes) {
    if (!this.isTableRow(lines[i])) return null;
    if (i + 1 >= lines.length || !this.isTableSep(lines[i + 1])) return null;
    var self = this;
    var cells = function(row) {
      return row.trim().replace(/^\|/, '').replace(/\|$/, '').split('|').map(function(c) { return c.trim(); });
    };
    var head = cells(lines[i]);
    var rows = [];
    var j = i + 2;
    while (j < lines.length && this.isTableRow(lines[j])) { rows.push(cells(lines[j])); j++; }
    var h = '<table class="md-table"><thead><tr>';
    head.forEach(function(c) { h += '<th>' + self.processInline(c, inlineCodes) + '</th>'; });
    h += '</tr></thead><tbody>';
    rows.forEach(function(r) {
      h += '<tr>';
      for (var k = 0; k < head.length; k++) h += '<td>' + self.processInline(r[k] || '', inlineCodes) + '</td>';
      h += '</tr>';
    });
    h += '</tbody></table>';
    return { html: h, next: j };
  },

  // Render Markdown to HTML
  render: function(text) {
    if (!text) return '';
    
    // Step 1: Escape HTML
    text = this.escapeHtml(text);
    
    // Step 2: Extract code blocks and replace with placeholders
    var codeBlocks = [];
    text = text.replace(/```(\w*)\n([\s\S]*?)```/g, function(match, lang, code) {
      var idx = codeBlocks.length;
      codeBlocks.push({ lang: lang, code: code });
      return '\x00CODEBLOCK' + idx + '\x00';
    });
    
    // Step 3: Extract inline code
    var inlineCodes = [];
    text = text.replace(/`([^`\n]+)`/g, function(match, code) {
      var idx = inlineCodes.length;
      inlineCodes.push(code);
      return '\x00INLINECODE' + idx + '\x00';
    });
    
    // Step 4: Process lines
    var lines = text.split('\n');
    var result = [];
    var inList = false;
    var listType = '';
    var inBlockquote = false;
    
    for (var i = 0; i < lines.length; i++) {
      var line = lines[i];
      
      // Check for code block placeholder
      var codeMatch = line.match(/^\x00CODEBLOCK(\d+)\x00$/);
      if (codeMatch) {
        if (inList) { result.push(listType === 'ul' ? '</ul>' : '</ol>'); inList = false; }
        if (inBlockquote) { result.push('</blockquote>'); inBlockquote = false; }
        var cb = codeBlocks[parseInt(codeMatch[1])];
        var langClass = cb.lang ? ' class="language-' + cb.lang + '"' : '';
        result.push('<pre class="md-code-block"><code' + langClass + '>' + cb.code + '</code><button class="md-copy-btn" data-code="' + encodeURIComponent(cb.code) + '">Copy</button></pre>');
        continue;
      }
      
      // Markdown table: a header row followed by a |---| separator row.
      var tbl = this.tableAt(lines, i, inlineCodes);
      if (tbl) {
        if (inList) { result.push(listType === 'ul' ? '</ul>' : '</ol>'); inList = false; }
        if (inBlockquote) { result.push('</blockquote>'); inBlockquote = false; }
        result.push(tbl.html);
        i = tbl.next - 1;
        continue;
      }

      // Headings
      var headingMatch = line.match(/^(#{1,6})\s+(.+)$/);
      if (headingMatch) {
        if (inList) { result.push(listType === 'ul' ? '</ul>' : '</ol>'); inList = false; }
        if (inBlockquote) { result.push('</blockquote>'); inBlockquote = false; }
        var level = headingMatch[1].length;
        result.push('<h' + level + '>' + this.processInline(headingMatch[2], inlineCodes) + '</h' + level + '>');
        continue;
      }
      
      // Horizontal rule
      if (/^(\*{3,}|-{3,}|_{3,})$/.test(line.trim())) {
        if (inList) { result.push(listType === 'ul' ? '</ul>' : '</ol>'); inList = false; }
        if (inBlockquote) { result.push('</blockquote>'); inBlockquote = false; }
        result.push('<hr>');
        continue;
      }
      
      // Blockquote (match both > and &gt; since HTML is already escaped)
      var quoteMatch = line.match(/^(?:&gt;|>)\s?(.*)$/);
      if (quoteMatch) {
        if (inList) { result.push(listType === 'ul' ? '</ul>' : '</ol>'); inList = false; }
        if (!inBlockquote) { result.push('<blockquote>'); inBlockquote = true; }
        result.push('<p>' + this.processInline(quoteMatch[1], inlineCodes) + '</p>');
        continue;
      } else if (inBlockquote) {
        result.push('</blockquote>');
        inBlockquote = false;
      }
      
      // Task list item: - [ ] / - [x] (checked before the plain ul, which would
      // otherwise swallow the "[ ]" into the text).
      var taskMatch = line.match(/^[\*\-]\s+\[([ xX])\]\s+(.*)$/);
      if (taskMatch) {
        if (!inList || listType !== 'ul') {
          if (inList) result.push(listType === 'ul' ? '</ul>' : '</ol>');
          result.push('<ul class="md-tasks">');
          inList = true;
          listType = 'ul';
        }
        var done = taskMatch[1].toLowerCase() === 'x';
        result.push('<li class="md-task' + (done ? ' done' : '') + '">' +
          '<input type="checkbox" disabled' + (done ? ' checked' : '') + '> ' +
          this.processInline(taskMatch[2], inlineCodes) + '</li>');
        continue;
      }

      // Unordered list
      var ulMatch = line.match(/^[\*\-]\s+(.+)$/);
      if (ulMatch) {
        if (!inList || listType !== 'ul') {
          if (inList) result.push(listType === 'ul' ? '</ul>' : '</ol>');
          result.push('<ul>');
          inList = true;
          listType = 'ul';
        }
        result.push('<li>' + this.processInline(ulMatch[1], inlineCodes) + '</li>');
        continue;
      }
      
      // Ordered list
      var olMatch = line.match(/^\d+\.\s+(.+)$/);
      if (olMatch) {
        if (!inList || listType !== 'ol') {
          if (inList) result.push(listType === 'ul' ? '</ul>' : '</ol>');
          result.push('<ol>');
          inList = true;
          listType = 'ol';
        }
        result.push('<li>' + this.processInline(olMatch[1], inlineCodes) + '</li>');
        continue;
      }
      
      // Close list if we're no longer in one
      if (inList) {
        result.push(listType === 'ul' ? '</ul>' : '</ol>');
        inList = false;
      }
      
      // Empty line
      if (line.trim() === '') {
        continue;
      }
      
      // Regular paragraph
      result.push('<p>' + this.processInline(line, inlineCodes) + '</p>');
    }
    
    // Close any open tags
    if (inList) result.push(listType === 'ul' ? '</ul>' : '</ol>');
    if (inBlockquote) result.push('</blockquote>');
    
    // Step 5: Join and restore code blocks (should already be restored, but just in case)
    text = result.join('\n');
    text = text.replace(/\x00CODEBLOCK(\d+)\x00/g, function(match, idx) {
      var cb = codeBlocks[parseInt(idx)];
      var langClass = cb.lang ? ' class="language-' + cb.lang + '"' : '';
      return '<pre class="md-code-block"><code' + langClass + '>' + cb.code + '</code><button class="md-copy-btn" aria-label="' + tr("chat.codeCopy") + '" data-code="' + encodeURIComponent(cb.code) + '">' + tr("chat.codeCopy") + '</button></pre>';
    });
    
    return text;
  },

  // Process inline Markdown (bold, italic, links, inline code)
  processInline: function(text, inlineCodes) {
    // Restore inline codes
    text = text.replace(/\x00INLINECODE(\d+)\x00/g, function(match, idx) {
      return '<code>' + inlineCodes[parseInt(idx)] + '</code>';
    });
    
    // Bold: **text** or __text__
    text = text.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
    text = text.replace(/__(.+?)__/g, '<strong>$1</strong>');
    
    // Italic: *text* or _text_
    text = text.replace(/\*(.+?)\*/g, '<em>$1</em>');
    text = text.replace(/_(.+?)_/g, '<em>$1</em>');
    
    // Images: ![alt](url). Only https is turned into an <img> (the HTML is already
    // escaped, so the URL cannot break out of the attribute); anything else stays
    // literal text. Must run before the link rule, which would match the [alt](url).
    text = text.replace(/!\[([^\]]*)\]\((https:\/\/[^)\s]+)\)/g,
      '<img class="md-img" src="$2" alt="$1" loading="lazy">');

    // Links: [text](url)
    text = text.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');
    
    return text;
  }
};

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
    "chat.assistant": "Q-Infer",
    "chat.copy": "复制",
    "chat.regenerate": "重新生成",
    "chat.good": "赞",
    "chat.bad": "踩",
    "chat.codeCopy": "复制",
    "chat.codeCopied": "已复制",
    "chat.usageMeta": "上下文 {p} · 生成 {c} · 合计 {t} tokens",
    "chat.edit": "编辑",
    "chat.editHint": "已撤回该轮，改好后发送",
    "chat.thinkMode": "思考模式",
    "chat.thinkOff": "关",
    "chat.contextLen": "上下文长度",
    "chat.ctxAuto": "自动",
    "chat.scrollBottom": "回到底部",
    "conv.closeSidebar": "收起侧栏",
    "conv.searchPh": "搜索对话",
    "conv.noMatch": "无匹配对话",
    "conv.timeNow": "刚刚",
    "conv.timeMin": "{n} 分钟前",
    "conv.timeHour": "{n} 小时前",
    "conv.timeDay": "{n} 天前",
    "conv.branch": "从这里分支",
    "conv.branched": "分支对话",
    "conv.branchedToast": "已从该轮分支出新对话",
    "chat.error": "错误",
    "chat.generating": "生成中…",
    "chat.prefilling": "正在处理提示词…",
    "chat.thinking": "思考中…",
    "chat.thinkingToggle": "思考模式",
    "chat.thinkingOn": "思考模式已开启",
    "chat.thinkingOff": "思考模式已关闭",
    "chat.topK": "top_k",
    "chat.minP": "min_p",
    "chat.presPen": "存在惩罚",
    "chat.freqPen": "频率惩罚",
    "chat.repPen": "重复惩罚",
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
    "chat.maxTokens": "最大 token 数",
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

    // Agent tools
    "tool.calling": "调用工具",
    "tool.executing": "执行中…",
    "tool.result": "工具结果",
    "tool.error": "工具执行失败",
    "tool.args": "参数",
    "tool.output": "输出",
    "tool.expand": "展开",
    "tool.collapse": "收起",
    "tool.renderHtml": "渲染页面",
    "tool.renderMermaid": "渲染图表",
    "tool.webFetch": "获取网页",
    "tool.jsExec": "执行代码",
    "tool.enabled": "工具已启用",
    "tool.disabled": "工具已禁用",
    "tool.settings": "工具设置",
    "tool.noTools": "未启用任何工具",

    // Conversation history
    "conv.history": "对话历史",
    "conv.new": "新对话",
    "conv.toggle": "显示对话列表",
    "conv.empty": "暂无对话记录",
    "conv.untitled": "未命名对话",
    "conv.delete": "删除",
    "conv.rename": "重命名",
    "conv.confirmDelete": "确定删除此对话？",
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
    "chat.assistant": "Q-Infer",
    "chat.copy": "Copy",
    "chat.regenerate": "Regenerate",
    "chat.good": "Good",
    "chat.bad": "Bad",
    "chat.codeCopy": "Copy",
    "chat.codeCopied": "Copied!",
    "chat.usageMeta": "prompt {p} · completion {c} · total {t} tokens",
    "chat.edit": "Edit",
    "chat.editHint": "Turn retracted — edit and send again",
    "chat.thinkMode": "Thinking mode",
    "chat.thinkOff": "Off",
    "chat.contextLen": "Context length",
    "chat.ctxAuto": "Auto",
    "chat.scrollBottom": "Scroll to bottom",
    "conv.closeSidebar": "Close sidebar",
    "conv.searchPh": "Search conversations",
    "conv.noMatch": "No matching conversations",
    "conv.timeNow": "just now",
    "conv.timeMin": "{n} min ago",
    "conv.timeHour": "{n} h ago",
    "conv.timeDay": "{n} d ago",
    "conv.branch": "Branch from here",
    "conv.branched": "Branched conversation",
    "conv.branchedToast": "Branched a new conversation from this turn",
    "chat.error": "Error",
    "chat.generating": "Generating…",
    "chat.prefilling": "Processing prompt…",
    "chat.thinking": "Thinking…",
    "chat.thinkingToggle": "Thinking mode",
    "chat.thinkingOn": "Thinking mode enabled",
    "chat.thinkingOff": "Thinking mode disabled",
    "chat.topK": "top_k",
    "chat.minP": "min_p",
    "chat.presPen": "presence_penalty",
    "chat.freqPen": "frequency_penalty",
    "chat.repPen": "repeat_penalty",
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
    "chat.maxTokens": "Max tokens",
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

    // Agent tools
    "tool.calling": "Calling tool",
    "tool.executing": "Executing…",
    "tool.result": "Tool result",
    "tool.error": "Tool execution failed",
    "tool.args": "Arguments",
    "tool.output": "Output",
    "tool.expand": "Expand",
    "tool.collapse": "Collapse",
    "tool.renderHtml": "Render page",
    "tool.renderMermaid": "Render diagram",
    "tool.webFetch": "Fetch page",
    "tool.jsExec": "Run code",
    "tool.enabled": "Tool enabled",
    "tool.disabled": "Tool disabled",
    "tool.settings": "Tool settings",
    "tool.noTools": "No tools enabled",

    // Conversation history
    "conv.history": "Conversations",
    "conv.new": "New conversation",
    "conv.toggle": "Show conversations",
    "conv.empty": "No conversations yet",
    "conv.untitled": "Untitled conversation",
    "conv.delete": "Delete",
    "conv.rename": "Rename",
    "conv.confirmDelete": "Delete this conversation?",
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
  // Refresh tool panel
  renderToolList();
}

function toggleLang() {
  const next = uiLang() === "en" ? "zh" : "en";
  try { localStorage.setItem("qinfer.lang", next); } catch (e) {}
  applyLang();
  showToast(next === "en" ? "English" : "中文", "info");
}

/* ---- Agent tool system: registry + built-in tools + execution engine ---- */
var QinferTools = {
  // Registry: name → { definition, handler, enabled }
  _registry: {},

  // Register a tool
  register: function(name, definition, handler) {
    this._registry[name] = {
      definition: definition, // OpenAI function schema: { name, description, parameters }
      handler: handler,       // async function(args) → string result
      enabled: true,
    };
    // Restore enabled state from localStorage
    try {
      var saved = JSON.parse(localStorage.getItem("qinfer.tools") || "{}");
      if (saved[name] === false) this._registry[name].enabled = false;
    } catch (e) {}
  },

  // Toggle tool on/off
  toggle: function(name, enabled) {
    if (!this._registry[name]) return;
    this._registry[name].enabled = enabled;
    // Persist
    var state = {};
    for (var k in this._registry) state[k] = this._registry[k].enabled;
    try { localStorage.setItem("qinfer.tools", JSON.stringify(state)); } catch (e) {}
  },

  // Get OpenAI-format tool definitions for enabled tools
  getDefinitions: function() {
    var defs = [];
    for (var name in this._registry) {
      if (!this._registry[name].enabled) continue;
      defs.push({
        type: "function",
        function: this._registry[name].definition,
      });
    }
    return defs;
  },

  // Execute a tool call, return string result
  execute: function(name, argsStr) {
    var tool = this._registry[name];
    if (!tool) return JSON.stringify({ error: "Unknown tool: " + name });
    try {
      var args = typeof argsStr === "string" ? JSON.parse(argsStr) : argsStr;
    } catch (e) {
      return JSON.stringify({ error: "Invalid arguments: " + e.message });
    }
    return tool.handler(args);
  },

  // List all registered tools with their state
  list: function() {
    var result = [];
    for (var name in this._registry) {
      var t = this._registry[name];
      result.push({
        name: name,
        description: t.definition.description,
        enabled: t.enabled,
      });
    }
    return result;
  },
};

// ---- Built-in tool: render_html ----
// Renders HTML content in a sandboxed iframe within the chat
QinferTools.register("render_html", {
  name: "render_html",
  description: "Render an HTML page or widget in a sandboxed iframe. Use this to create rich visual content like dashboards, charts, tables, interactive widgets, or any HTML/CSS/JS content. The HTML runs in a sandboxed iframe with no network access.",
  parameters: {
    type: "object",
    properties: {
      title: { type: "string", description: "Short title for the rendered content" },
      html: { type: "string", description: "Complete HTML document or fragment to render. Include all CSS inline or in <style> tags." },
      height: { type: "number", description: "Approximate height in pixels (default: 400)" },
    },
    required: ["html"],
  },
}, function(args) {
  // Return a special marker that the UI will render as an iframe
  return JSON.stringify({
    __tool_render: "html",
    title: args.title || "HTML",
    html: args.html,
    height: args.height || 400,
  });
});

// ---- Built-in tool: render_mermaid ----
// Renders a Mermaid diagram
QinferTools.register("render_mermaid", {
  name: "render_mermaid",
  description: "Render a diagram using Mermaid syntax. Supports flowchart, sequence, class, state, ER, gantt, pie, gitgraph, and other Mermaid diagram types. Use this for visualizing processes, architectures, timelines, relationships, etc.",
  parameters: {
    type: "object",
    properties: {
      title: { type: "string", description: "Short title for the diagram" },
      diagram: { type: "string", description: "Mermaid diagram definition (e.g. 'graph TD\\nA-->B\\nB-->C')" },
    },
    required: ["diagram"],
  },
}, function(args) {
  return JSON.stringify({
    __tool_render: "mermaid",
    title: args.title || "Diagram",
    diagram: args.diagram,
  });
});

// ---- Built-in tool: web_fetch ----
// Fetches content from a URL
QinferTools.register("web_fetch", {
  name: "web_fetch",
  description: "Fetch content from a URL. Returns the text content of the page (HTML is converted to text). Useful for reading documentation, articles, or any web resource.",
  parameters: {
    type: "object",
    properties: {
      url: { type: "string", description: "The URL to fetch (must be https)" },
      max_length: { type: "number", description: "Maximum characters to return (default: 8000)" },
    },
    required: ["url"],
  },
}, async function(args) {
  try {
    var url = args.url;
    if (!url.startsWith("https://")) return JSON.stringify({ error: "Only HTTPS URLs are allowed" });
    var res = await fetch(url, { headers: { "Accept": "text/html,text/plain,application/json" } });
    if (!res.ok) return JSON.stringify({ error: "HTTP " + res.status });
    var ct = res.headers.get("content-type") || "";
    var text = await res.text();
    // Simple HTML to text conversion
    if (ct.indexOf("html") >= 0) {
      var doc = new DOMParser().parseFromString(text, "text/html");
      // Remove script/style
      doc.querySelectorAll("script,style,noscript").forEach(function(el) { el.remove(); });
      text = doc.body ? doc.body.textContent : doc.documentElement.textContent;
      text = text.replace(/\s+/g, " ").trim();
    }
    var maxLen = args.max_length || 8000;
    if (text.length > maxLen) text = text.slice(0, maxLen) + "\n…(truncated)";
    return JSON.stringify({ url: url, content: text, length: text.length });
  } catch (e) {
    return JSON.stringify({ error: e.message });
  }
});

// ---- Built-in tool: js_exec ----
// Executes JavaScript in a sandboxed context
QinferTools.register("js_exec", {
  name: "js_exec",
  description: "Execute JavaScript code in a sandboxed environment. The code runs with access to Math, JSON, Date, and basic utilities. Use for calculations, data transformations, or generating structured output. Returns the value of the last expression or console.log output.",
  parameters: {
    type: "object",
    properties: {
      code: { type: "string", description: "JavaScript code to execute" },
    },
    required: ["code"],
  },
}, function(args) {
  try {
    var logs = [];
    var fakeConsole = { log: function() { logs.push(Array.from(arguments).map(function(a) { return typeof a === "object" ? JSON.stringify(a) : String(a); }).join(" ")); } };
    // Execute in a limited scope
    var fn = new Function("console", "return (function() { " + args.code + " })()");
    var result = fn(fakeConsole);
    var output = logs.length ? logs.join("\n") : "";
    if (result !== undefined) {
      output += (output ? "\n" : "") + (typeof result === "object" ? JSON.stringify(result, null, 2) : String(result));
    }
    return JSON.stringify({ result: output || "(no output)" });
  } catch (e) {
    return JSON.stringify({ error: e.message });
  }
});

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
    top_k: parseInt($("s-top-k").value, 10),
    min_p: parseFloat($("s-min-p").value),
    presence_penalty: parseFloat($("s-pres-pen").value),
    frequency_penalty: parseFloat($("s-freq-pen").value),
    repeat_penalty: parseFloat($("s-rep-pen").value),
    seed: parseInt($("s-seed").value, 10),
    max_tokens: parseInt($("s-max-tokens").value, 10),
  };
  if ($("s-think")) s.think = $("s-think").value;
  if ($("s-ctx")) s.ctx = $("s-ctx").value;
  sessionStorage.setItem("qinfer_sampler", JSON.stringify(s));
}
function loadSampler() {
  const raw = sessionStorage.getItem("qinfer_sampler");
  if (!raw) return false;
  try {
    const s = JSON.parse(raw);
    if (s.temperature !== undefined) $("s-temp").value = s.temperature;
    if (s.top_p !== undefined) $("s-top-p").value = s.top_p;
    if (s.top_k !== undefined) $("s-top-k").value = s.top_k;
    if (s.min_p !== undefined) $("s-min-p").value = s.min_p;
    if (s.presence_penalty !== undefined) $("s-pres-pen").value = s.presence_penalty;
    if (s.frequency_penalty !== undefined) $("s-freq-pen").value = s.frequency_penalty;
    if (s.repeat_penalty !== undefined) $("s-rep-pen").value = s.repeat_penalty;
    if (s.seed !== undefined) $("s-seed").value = s.seed;
    if (s.max_tokens !== undefined) $("s-max-tokens").value = s.max_tokens;
    if (s.think !== undefined && $("s-think")) $("s-think").value = s.think;
    if (s.ctx !== undefined && $("s-ctx")) $("s-ctx").value = s.ctx;
    return !!$("s-temp");
  } catch (e) { return false; }
}

// Reflect the Generation selector's thinking mode on the composer toggle.
function syncThinkBtn() {
  var b = $("btn-thinking"), s = $("s-think");
  if (b && s) b.classList.toggle("active", s.value !== "off");
}
// The sampler + generation fields as a plain object, for per-conversation overrides.
function samplerSnapshot() {
  var s = {};
  ["s-temp", "s-top-p", "s-top-k", "s-min-p", "s-pres-pen", "s-freq-pen", "s-rep-pen", "s-seed", "s-max-tokens", "s-think", "s-ctx"].forEach(function (id) {
    if ($(id)) s[id] = $(id).value;
  });
  return s;
}
// Apply a saved override set to the fields (no-op for a missing/empty set).
function samplerApply(s) {
  if (!s) return;
  Object.keys(s).forEach(function (id) { if ($(id)) $(id).value = s[id]; });
  syncThinkBtn();
}

// A short relative time for the conversation list (i18n'd).
function relTime(ts) {
  if (!ts) return "";
  var m = Math.floor((Date.now() - ts) / 60000);
  if (m < 1) return tr("conv.timeNow");
  if (m < 60) return tr("conv.timeMin").replace("{n}", m);
  var h = Math.floor(m / 60);
  if (h < 24) return tr("conv.timeHour").replace("{n}", h);
  return tr("conv.timeDay").replace("{n}", Math.floor(h / 24));
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
  // User messages: plain text. Assistant/error: will be rendered as Markdown when complete.
  body.textContent = text;
  el.appendChild(label);
  // Add thinking section for assistant messages (initially hidden)
  if (role === "assistant") {
    var thinkingSection = document.createElement("div");
    thinkingSection.className = "thinking-section";
    thinkingSection.style.display = "none";
    var thinkingHeader = document.createElement("div");
    thinkingHeader.className = "thinking-header";
    thinkingHeader.setAttribute("role", "button");
    thinkingHeader.setAttribute("tabindex", "0");
    thinkingHeader.setAttribute("aria-expanded", "false");
    thinkingHeader.setAttribute("aria-label", tr("chat.thinking"));
    thinkingHeader.innerHTML = '<svg class="st-icon st-icon--sm thinking-icon"><use href="/web/sprite.svg#i-thinking"/></svg><span>' + tr("chat.thinking") + '</span><svg class="st-icon st-icon--sm thinking-chevron"><use href="/web/sprite.svg#i-chevron"/></svg>';
    thinkingHeader.style.cursor = "pointer";
    var thinkingBody = document.createElement("div");
    thinkingBody.className = "thinking-body";
    thinkingSection.appendChild(thinkingHeader);
    thinkingSection.appendChild(thinkingBody);
    el.appendChild(thinkingSection);
    el._thinkingSection = thinkingSection;
    el._thinkingBody = thinkingBody;
    // Toggle the thinking section: click, or Enter/Space when focused (it is a
    // role=button so it must answer the keyboard).
    var toggleThinking = function() {
      var isHidden = thinkingBody.style.display === "none" || !thinkingBody.style.display;
      thinkingBody.style.display = isHidden ? "block" : "none";
      thinkingHeader.classList.toggle("expanded", isHidden);
      thinkingHeader.setAttribute("aria-expanded", isHidden ? "true" : "false");
    };
    thinkingHeader.addEventListener("click", toggleThinking);
    thinkingHeader.addEventListener("keydown", function (e) {
      if (e.key === "Enter" || e.key === " " || e.key === "Spacebar") {
        e.preventDefault();
        toggleThinking();
      }
    });
  }
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
    el._attachments = attachList;
  }
  el.appendChild(body);
  // Add message action bar. Assistant messages get copy/regenerate/feedback;
  // user messages get edit (which resends from that point).
  if (role === "assistant") {
    var actions = document.createElement("div");
    actions.className = "bubble-actions";
    actions.innerHTML = '<button class="bubble-action" data-action="copy" title="' + tr("chat.copy") + '" aria-label="' + tr("chat.copy") + '"><svg class="st-icon st-icon--sm"><use href="/web/sprite.svg#i-copy"/></svg></button>' +
      '<button class="bubble-action" data-action="regenerate" title="' + tr("chat.regenerate") + '" aria-label="' + tr("chat.regenerate") + '"><svg class="st-icon st-icon--sm"><use href="/web/sprite.svg#i-refresh"/></svg></button>' +
      '<button class="bubble-action" data-action="thumbup" title="' + tr("chat.good") + '" aria-label="' + tr("chat.good") + '"><svg class="st-icon st-icon--sm"><use href="/web/sprite.svg#i-check"/></svg></button>' +
      '<button class="bubble-action" data-action="thumbdown" title="' + tr("chat.bad") + '" aria-label="' + tr("chat.bad") + '"><svg class="st-icon st-icon--sm"><use href="/web/sprite.svg#i-error"/></svg></button>';
    el.appendChild(actions);
  } else if (role === "user") {
    var uactions = document.createElement("div");
    uactions.className = "bubble-actions";
    uactions.innerHTML = '<button class="bubble-action" data-action="copy" title="' + tr("chat.copy") + '" aria-label="' + tr("chat.copy") + '"><svg class="st-icon st-icon--sm"><use href="/web/sprite.svg#i-copy"/></svg></button>' +
      '<button class="bubble-action" data-action="edit" title="' + tr("chat.edit") + '" aria-label="' + tr("chat.edit") + '"><svg class="st-icon st-icon--sm"><use href="/web/sprite.svg#i-settings"/></svg></button>' +
      '<button class="bubble-action" data-action="branch" title="' + tr("conv.branch") + '" aria-label="' + tr("conv.branch") + '"><svg class="st-icon st-icon--sm"><use href="/web/sprite.svg#i-new-chat"/></svg></button>';
    el.appendChild(uactions);
  }
  $("msg-list").appendChild(el);
  $("msg-list").scrollTop = $("msg-list").scrollHeight;
  return body;
}

// Strip special tokens from answer text (safety net)
function filterThinkingContent(text) {
  if (!text) return text;
  // Remove <think>/</think> tags if any leaked through
  text = text.replace(/<\/?think>/g, "");
  // Remove special tokens like <think>, </think>, etc.
  text = text.replace(/<\|[^|>]+\|>/g, "");
  return text;
}

// Parse  tags from text, returning { answer, thinking }
function parseThinkingContent(text) {
  if (!text) return { answer: "", thinking: "" };
  var answer = "";
  var thinking = "";
  var buf = text;
  var inThink = false;
  while (true) {
    if (!inThink) {
      var otIdx = buf.indexOf("<think>");
      if (otIdx < 0) { answer += buf; break; }
      answer += buf.substring(0, otIdx);
      buf = buf.substring(otIdx + 7);
      inThink = true;
    } else {
      var ctIdx = buf.indexOf("</think>");
      if (ctIdx < 0) { thinking += buf; break; }
      thinking += buf.substring(0, ctIdx);
      buf = buf.substring(ctIdx + 8);
      inThink = false;
    }
  }
  return { answer: answer, thinking: thinking };
}
// Control tags that must never be painted as content.
var SPECIAL_TOKENS = [
  '<|im_end|>', '<|im_start|>', '<|im_sep|>', '<think>', '</think>'
];

var TAG_PREFIXES = SPECIAL_TOKENS.map(function (t) { return t.substring(0, t.length - 1); });

function stripSpecialTokens(text) {
  if (!text) return text;
  var t = text;
  for (var i = 0; i < SPECIAL_TOKENS.length; i++) {
    if (t.indexOf(SPECIAL_TOKENS[i]) >= 0) t = t.split(SPECIAL_TOKENS[i]).join("");
  }
  return t;
}

// The delimiters a reply can carry. Any of them can arrive split across streaming
// deltas, so text next to an unfinished one is held back until the token either
// completes or the stream ends.
var CONTROL_TOKENS = SPECIAL_TOKENS.slice()
  .sort(function (a, b) { return b.length - a.length; });

var CHATML_TOKENS = CONTROL_TOKENS.filter(function (t) {
  return t.indexOf("|") >= 0;
});

// The control token starting at index i of text, or null.
function tokenStartingAt(text, i) {
  for (var k = 0; k < CONTROL_TOKENS.length; k++) {
    if (text.startsWith(CONTROL_TOKENS[k], i)) return CONTROL_TOKENS[k];
  }
  return null;
}

// Length of the longest suffix of buf that could still grow into a ChatML control
// token. These only ever end a turn, so an unfinished one at the very end of a
// stream is control text, while an unfinished thinking delimiter may be a delimiter
// the next delta completes.
function partialChatToken(buf) {
  var max = 0;
  for (var i = 0; i < CHATML_TOKENS.length; i++) {
    var n = partialThinkTag(buf, CHATML_TOKENS[i]);
    if (n > max) max = n;
  }
  return max;
}

// Length of the longest suffix of buf that could still grow into a control token.
function partialControlToken(buf) {
  var max = 0;
  for (var i = 0; i < CONTROL_TOKENS.length; i++) {
    var n = partialThinkTag(buf, CONTROL_TOKENS[i]);
    if (n > max) max = n;
  }
  return max;
}

// Mid-stream the answer may hold an unterminated code fence; close it for display
// so raw triple backticks do not leak into the bubble as literal text.
function safePartialMarkdown(text) {
  if (!text) return text;
  var fences = (text.match(/```/g) || []).length;
  return (fences % 2 === 1) ? text + "\n```" : text;
}

// --- Streaming think-tag router ----------------------------------------------
// Delimiters of a thinking run. The reply stream carries these; ChatML markers are
// a different family and are only ever stripped, never used as structure.
var THINK_OPEN = "<" + "think>";
var THINK_CLOSE = "<" + "/think>";

// Length of the longest suffix of buf that is a proper prefix of tag. A delimiter
// can be split across deltas, and a delimiter painted as content is unrecoverable,
// so text next to such a tail waits in the buffer until the delimiter completes or
// the stream ends.
function partialThinkTag(buf, tag) {
  var max = Math.min(tag.length - 1, buf.length);
  for (var n = max; n > 0; n--) {
    if (tag.lastIndexOf(buf.substring(buf.length - n)) === 0) return n;
  }
  return 0;
}

// The engine streams a reply as text, and a thinking run is the text a closing
// delimiter ends. The opening delimiter may be present or swallowed by the chat
// template, and reply text may come before an explicit block, so attribution is a
// guess the stream corrects. The router holds delimiters back until they either
// complete or the stream ends, so a control token is never painted and a torn
// answer is never truncated:
//
//   - a leading run (state "pre") streams live into the thinking panel, since in
//     thinking mode the head of a reply is almost always the reasoning;
//   - a closing delimiter confirms that run as thinking; an opening delimiter
//     instead means the head was the reply, so the provisional run moves back to
//     the answer bubble;
//   - after the first delimiter the reply streams straight into the answer bubble,
//     and a later block introduced by an opening delimiter streams into the
//     thinking panel until its closing delimiter settles it;
//   - a block the model never closes is the reply after all: it is shown rather
//     than hidden, so content is never lost;
//   - ChatML markers and thinking delimiters are never painted, and a torn
//     delimiter at the very end is dropped rather than shown as text.
//
// `blockStart` is where the provisional run begins inside `thinking`, so only the
// still-undecided run is ever reclaimed - an already-confirmed block is kept.
function ensureThinkStream(plain) {
  return {
    buf: "",          // text not yet attributed to a destination
    thinking: "",     // visible thinking (scrubbed)
    answer: "",       // visible answer (scrubbed)
    rThinking: "",    // raw thinking, before control tokens are dropped
    rAnswer: "",      // raw answer
    state: plain ? "answer" : "pre",   // pre | think | answer
    plain: !!plain,   // nothinking mode: no thinking blocks, delimiters are stripped
    blockStart: 0,    // index in `rThinking` where the provisional run starts
    // Feed one delta. Returns true once anything is on screen for either panel.
    push: function (chunk) {
      this.buf += chunk;
      this.scan();
      // Control tokens can be formed by joining text across deltas, so scrub the
      // whole accumulated run rather than each fragment.
      this.answer = this.scrub(this.rAnswer);
      this.thinking = this.scrub(this.rThinking);
      return this.thinking.length > 0 || this.answer.length > 0;
    },
    // One delta can carry several delimiters, so step until the buffer holds
    // nothing that can be decided yet.
    scan: function () {
      for (var guard = 0; guard < 64 && this.step(); guard++) {}
    },
    step: function () {
      var o = this.plain ? -1 : this.buf.indexOf(THINK_OPEN);
      var c = this.plain ? -1 : this.buf.indexOf(THINK_CLOSE);
      if (c >= 0 && (o < 0 || c < o)) {
        var run = this.takeUpTo(c);
        this.buf = this.buf.substring(THINK_CLOSE.length);
        if (this.state === "answer") {
          // The delimiter closes nothing: the run before it is already reply text,
          // and the delimiter itself is control text.
          this.emit(run, "answer");
        } else {
          // A leading run ("pre") or an open block ("think"): the delimiter settles
          // it as thinking.
          this.emit(run, "thinking");
        }
        this.state = "answer";
        return true;
      }
      if (o >= 0) {
        var head = this.takeUpTo(o);
        this.buf = this.buf.substring(THINK_OPEN.length);
        if (this.state === "pre") {
          // Text before an explicit block was the reply, not reasoning: the whole
          // provisional run moves back to the answer bubble.
          this.emit(head, "thinking");
          this.reclaimToAnswer();
        } else {
          this.emit(head, this.state === "think" ? "thinking" : "answer");
        }
        this.state = "think";
        this.blockStart = this.rThinking.length;
        return true;
      }
      if (this.state === "pre" || this.state === "think") {
        // The provisional run streams live; only a torn delimiter is held back.
        var keepRun = partialControlToken(this.buf);
        if (keepRun >= this.buf.length) return false;
        this.emit(this.takeUpTo(this.buf.length - keepRun), "thinking");
        return true;
      }
      var keep = partialControlToken(this.buf);
      if (keep >= this.buf.length) return false;
      this.emit(this.takeUpTo(this.buf.length - keep), "answer");
      return true;
    },
    // Append text to a destination. Control tokens are dropped later, when the
    // whole run is scrubbed - a token may only become complete once the next
    // fragment joins it.
    emit: function (text, dest) {
      if (!text) return;
      if (dest === "thinking") this.rThinking += text; else this.rAnswer += text;
    },
    // Move the still-undecided provisional run out of thinking into the answer.
    reclaimToAnswer: function () {
      if (this.blockStart >= this.rThinking.length) return;
      this.rAnswer += this.rThinking.substring(this.blockStart);
      this.rThinking = this.rThinking.substring(0, this.blockStart);
    },
    // Drop control text: ChatML markers are never content, and neither are the
    // thinking delimiters themselves. Removing an inner token can join the text
    // around it into a new token, so keep sweeping until nothing is left to drop.
    // A fragment that is only the beginning of a token is left visible - it may be
    // text the model produced.
    scrub: function (text) {
      if (!text) return "";
      var cur = text;
      for (var sweep = 0; sweep < 32; sweep++) {
        var out = "";
        var changed = false;
        for (var i = 0; i < cur.length; i++) {
          var tok = tokenStartingAt(cur, i);
          if (tok) { i += tok.length - 1; changed = true; continue; }
          out += cur.charAt(i);
        }
        cur = out;
        if (!changed) break;
      }
      return cur;
    },
    takeUpTo: function (n) {
      var head = this.buf.substring(0, n);
      this.buf = this.buf.substring(n);
      return head;
    },
    // End of stream: attribute whatever is still held back.
    finish: function () {
      // A held tail is control text when it is nothing but the start of any control
      // token. While the leading run is still undecided ("pre") a torn *opening*
      // delimiter is kept instead: the model may have meant to write it as text.
      var anyControl = function (t) { return t.length > 0 && partialControlToken(t) === t.length; };
      var preControl = function (t) {
        return t.length > 0 &&
          (partialChatToken(t) === t.length || partialThinkTag(t, THINK_CLOSE) === t.length);
      };
      if (this.state === "pre") {
        // The leading run was never settled by a delimiter: it is the reply. The
        // provisional run precedes the buffer, so move it over first, then the tail.
        this.reclaimToAnswer();
        if (!preControl(this.buf)) this.emit(this.buf, "answer");
        this.buf = "";
      } else if (this.state === "think") {
        // A block the model never closed is the reply after all.
        if (!anyControl(this.buf)) this.emit(this.buf, "answer");
        this.buf = "";
        this.reclaimToAnswer();
      } else if (this.buf) {
        if (!anyControl(this.buf)) this.emit(this.buf, "answer");
        this.buf = "";
      }
      this.state = "answer";
      this.blockStart = 0;
      this.answer = this.scrub(this.rAnswer);
      this.thinking = this.scrub(this.rThinking);
      return true;
    },
  };
}
// Append the per-reply token usage under an assistant bubble (prompt/completion/total).
function renderUsageMeta(bubble, usage) {
  if (!bubble || !usage) return;
  var p = usage.prompt_tokens, c = usage.completion_tokens, t = usage.total_tokens;
  if (p == null && c == null && t == null) return;
  var el = document.createElement("div");
  el.className = "bubble-meta";
  el.textContent = tr("chat.usageMeta")
    .replace("{p}", p != null ? p : "-")
    .replace("{c}", c != null ? c : "-")
    .replace("{t}", t != null ? t : "-");
  bubble.appendChild(el);
}
// Render Markdown in an assistant message body (called after streaming completes)
function renderMsgMarkdown(bodyEl) {
  if (!bodyEl || !bodyEl.textContent) return;
  var text = bodyEl.textContent;
  bodyEl.innerHTML = QinferMD.render(text);
  // Store raw text for collectMessages
  bodyEl.parentElement._rawText = text;
  // Attach copy button handlers
  bodyEl.querySelectorAll(".md-copy-btn").forEach(function(btn) {
    btn.addEventListener("click", function() {
      var code = decodeURIComponent(btn.dataset.code);
      navigator.clipboard.writeText(code).then(function() {
        btn.textContent = tr("chat.codeCopied");
        setTimeout(function() { btn.textContent = tr("chat.codeCopy"); }, 1500);
      });
    });
  });
}

// Render a tool call bubble (shows tool name + arguments, collapsible)
function addToolCallMsg(toolCalls) {
  hideEmptyPrompt();
  var el = document.createElement("div");
  el.className = "bubble tool-call";
  el._toolCalls = toolCalls; // Store for collectMessages
  var label = document.createElement("div");
  label.className = "bubble-role";
  var names = toolCalls.map(function(tc) { return tc.function.name; }).join(", ");
  label.textContent = tr("tool.calling") + ": " + names;
  el.appendChild(label);
  toolCalls.forEach(function(tc, i) {
    var card = document.createElement("div");
    card.className = "tool-card";
    var head = document.createElement("div");
    head.className = "tool-card__head";
    var icon = document.createElement("span");
    icon.className = "tool-card__icon";
    icon.innerHTML = '<svg class="st-icon st-icon--sm"><use href="/web/sprite.svg#i-tool"/></svg>';
    var name = document.createElement("span");
    name.className = "tool-card__name";
    name.textContent = tc.function.name;
    var status = document.createElement("span");
    status.className = "tool-card__status";
    status.textContent = tr("tool.executing");
    head.appendChild(icon);
    head.appendChild(name);
    head.appendChild(status);
    var argsPre = document.createElement("pre");
    argsPre.className = "tool-card__args";
    try {
      argsPre.textContent = JSON.stringify(JSON.parse(tc.function.arguments), null, 2);
    } catch(e) {
      argsPre.textContent = tc.function.arguments;
    }
    card.appendChild(head);
    card.appendChild(argsPre);
    card.dataset.callId = tc.id;
    card.dataset.toolName = tc.function.name;
    el.appendChild(card);
  });
  $("msg-list").appendChild(el);
  $("msg-list").scrollTop = $("msg-list").scrollHeight;
  return el;
}

// Update a tool call card with its result
function updateToolCallResult(callId, result, isError) {
  var card = document.querySelector('.tool-card[data-call-id="' + callId + '"]');
  if (!card) return;
  var status = card.querySelector(".tool-card__status");
  if (status) {
    status.textContent = isError ? tr("tool.error") : tr("tool.result");
    status.className = "tool-card__status" + (isError ? " tool-card__status--error" : " tool-card__status--ok");
  }
  // Check if result is a special render instruction
  var resultDiv = document.createElement("div");
  resultDiv.className = "tool-card__result";
  try {
    var parsed = JSON.parse(result);
    if (parsed.__tool_render === "html") {
      resultDiv.className += " tool-card__result--html";
      var iframe = document.createElement("iframe");
      iframe.className = "tool-render-frame";
      iframe.sandbox = "allow-scripts"; // Sandboxed: no network, no same-origin
      iframe.style.height = (parsed.height || 400) + "px";
      card.parentElement.appendChild(resultDiv);
      resultDiv.appendChild(iframe);
      // Write HTML to iframe
      var doc = iframe.contentDocument || iframe.contentWindow.document;
      doc.open();
      doc.write(parsed.html);
      doc.close();
      // Auto-resize iframe
      setTimeout(function() {
        try { iframe.style.height = doc.body.scrollHeight + 20 + "px"; } catch(e) {}
      }, 500);
      return;
    } else if (parsed.__tool_render === "mermaid") {
      resultDiv.className += " tool-card__result--mermaid";
      // Render mermaid as a pre block (full mermaid.js integration would need the library)
      var pre = document.createElement("pre");
      pre.className = "tool-mermaid";
      pre.textContent = parsed.diagram;
      resultDiv.appendChild(pre);
      card.parentElement.appendChild(resultDiv);
      return;
    }
  } catch(e) {}
  // Default: show as text
  var pre = document.createElement("pre");
  pre.className = "tool-card__output";
  pre.textContent = result.length > 2000 ? result.slice(0, 2000) + "\n…(truncated)" : result;
  card.parentElement.appendChild(resultDiv);
  resultDiv.appendChild(pre);
}

// Approximate a message's size in characters (text parts only; images ignored).
function msgChars(m) {
  if (!m) return 0;
  if (typeof m.content === "string") return m.content.length;
  if (Array.isArray(m.content)) {
    var n = 0;
    m.content.forEach(function (p) { if (p && typeof p.text === "string") n += p.text.length; });
    return n;
  }
  return 0;
}
// Keep the newest messages within a budget (~4 chars/token). A leading system
// message is always kept. budgetTokens <= 0 means no truncation.
function truncateMessages(messages, budgetTokens) {
  if (!budgetTokens || budgetTokens <= 0 || !messages || messages.length === 0) return messages;
  var budgetChars = budgetTokens * 4;
  var head = null, rest = messages;
  if (messages[0] && messages[0].role === "system") { head = messages[0]; rest = messages.slice(1); }
  var kept = [], used = head ? msgChars(head) : 0;
  for (var i = rest.length - 1; i >= 0; i--) {
    var c = msgChars(rest[i]);
    if (used + c > budgetChars && kept.length) break;
    kept.unshift(rest[i]);
    used += c;
  }
  return head ? [head].concat(kept) : kept;
}
function collectMessages() {
  const messages = [];
  document.querySelectorAll("#msg-list .bubble").forEach(function (b) {
    // Tool call bubble: emit assistant message with tool_calls
    if (b.classList.contains("tool-call")) {
      var tcs = b._toolCalls || [];
      if (tcs.length) {
        messages.push({
          role: "assistant",
          content: b.querySelector(".bubble-body") ? b.querySelector(".bubble-body").textContent : null,
          tool_calls: tcs.map(function(tc) {
            return { id: tc.id, type: "function", function: { name: tc.function.name, arguments: tc.function.arguments } };
          }),
        });
      }
      return;
    }
    // Tool result bubble: emit tool role message
    if (b.classList.contains("tool-result")) {
      messages.push({
        role: "tool",
        content: b._toolResult || "",
        tool_call_id: b._toolCallId || "",
      });
      return;
    }
    // Normal user/assistant bubble
    const role = b.classList.contains("user") ? "user" : "assistant";
    // Use raw text (before Markdown rendering) if available, otherwise textContent
    const bodyEl = b.querySelector(".bubble-body");
    const txt = b._rawText || bodyEl.textContent;
    const attachList = b._attachments || [];
    const imageAttachments = attachList.filter(function (a) { return a.kind === "image" && a.url; });
    const fileAttachments = attachList.filter(function (a) { return a.kind === "file" && a.text; });
    if (imageAttachments.length) {
      var parts = [];
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
  $("btn-stop").classList.toggle("hidden", !on);
}

async function sendChat() {
  const ta = $("chat-input");
  const text = ta.value.trim();
  if ((!text && !attachments.length) || streaming) return;
  ta.value = "";
  var currentAttachments = attachments.slice();
  attachments = [];
  renderAttachments();
  addMsg("user", text, currentAttachments);
  setStreaming(true);
  abortCtrl = new AbortController();
  setPill("reading", tr("chat.prefilling"));
  // Agentic loop: keep going while model returns tool_calls (max 8 rounds)
  var MAX_ROUNDS = 8;
  for (var round = 0; round < MAX_ROUNDS; round++) {
    var result = await doChatRound();
    if (result.error) {
      setPill("error", tr("chat.error"));
      break;
    }
    if (result.aborted) {
      setPill("idle", tr("chat.stopped"));
      break;
    }
    if (!result.toolCalls || !result.toolCalls.length) {
      // Normal completion — no more tool calls
      setPill("idle", tr("chat.idle"));
      break;
    }
    // Execute tool calls and add results to the conversation
    setPill("generating", tr("tool.executing"));
    for (var i = 0; i < result.toolCalls.length; i++) {
      var tc = result.toolCalls[i];
      var toolResult = QinferTools.execute(tc.function.name, tc.function.arguments);
      // Handle async tool results
      if (toolResult && typeof toolResult.then === "function") {
        toolResult = await toolResult;
      }
      // Update the tool call card with the result
      updateToolCallResult(tc.id, toolResult, false);
      // Add a hidden tool-result bubble for collectMessages
      var toolBubble = document.createElement("div");
      toolBubble.className = "bubble tool-result";
      toolBubble.style.display = "none";
      toolBubble._toolResult = toolResult;
      toolBubble._toolCallId = tc.id;
      $("msg-list").appendChild(toolBubble);
    }
    $("msg-list").scrollTop = $("msg-list").scrollHeight;
    // Continue the loop — the next round will include tool results
  }
  if (streaming) {
    setStreaming(false);
    abortCtrl = null;
    setPill("idle", tr("chat.idle"));
    // Save conversation after completion
    QinferConv.saveCurrent();
    QinferConv.renderList();
  }
}

// Execute a single chat round: send request, stream response, return result
// Returns: { text, toolCalls, error, aborted }
async function doChatRound() {
  var tools = QinferTools.getDefinitions();
  var body = {
    model: modelId,
    messages: collectMessages(),
    stream: true,
    stream_options: { include_usage: true },   // ask for the final usage frame
    max_tokens: parseInt($("s-max-tokens").value, 10) || 2048,
    temperature: parseFloat($("s-temp").value),
    top_p: parseFloat($("s-top-p").value),
    top_k: parseInt($("s-top-k").value, 10),
    min_p: parseFloat($("s-min-p").value),
    presence_penalty: parseFloat($("s-pres-pen").value),
    frequency_penalty: parseFloat($("s-freq-pen").value),
    repeat_penalty: parseFloat($("s-rep-pen").value),
    seed: parseInt($("s-seed").value, 10),
  };
  // Thinking mode + reasoning effort come from the Generation selector (the
  // composer toggle is a shortcut that writes this selector).
  var thinkMode = $("s-think") ? $("s-think").value : "xhigh";
  body.enable_thinking = thinkMode !== "off";
  if (thinkMode !== "off") body.reasoning_effort = thinkMode;
  // Context tiers: truncate the history to the chosen budget (0 = auto/none).
  body.messages = truncateMessages(body.messages, parseInt($("s-ctx") ? $("s-ctx").value : "0", 10));
  if (tools.length) body.tools = tools;
  var out = addMsg("assistant", "");
  var waitRow = document.createElement("div");
  waitRow.className = "msg-waiting";
  var spinner = document.createElement("span");
  spinner.className = "spinner";
  var waitTxt = document.createElement("span");
  waitTxt.textContent = tr("chat.prefilling");
  waitRow.appendChild(spinner);
  waitRow.appendChild(waitTxt);
  out.parentElement.appendChild(waitRow);
  // Accumulate tool calls from streaming deltas
  var toolCallAccum = {}; // index → { id, function: { name, arguments } }
  var finishReason = null;
  var usageAcc = null;    // final usage frame (prompt/completion/total tokens)
  // Incremental think-tag router. In nothinking mode there is nothing to hold
  // back: the engine emits the opening tag itself and the template swallows it,
  // so the reply is plain text. Declared at function scope because the finalizer
  // below runs after the try/catch - a block-scoped declaration inside the try
  // would not be visible there.
  var think = ensureThinkStream(!body.enable_thinking);
  var thinkingBody = out.parentElement._thinkingBody;
  var thinkingSec = out.parentElement._thinkingSection;
  function paintThinking() {
    if (!thinkingBody) return;
    thinkingBody.textContent = think.thinking;
    if (thinkingSec && think.thinking) thinkingSec.style.display = "";
    out.parentElement._thinkingText = think.thinking;
  }
  try {
    var res = await api("/v1/chat/completions", {
      method: "POST",
      body: JSON.stringify(body),
      signal: abortCtrl.signal,
      headers: { "Accept": "text/event-stream" },
    });
    if (!res.ok) {
      var msg = "HTTP " + res.status;
      try {
        var e = await res.json();
        if (e.error && e.error.message) msg = e.error.message;
      } catch (e) {}
      waitRow.remove();
      out.textContent = msg;
      out.parentElement.className = "bubble error";
      if (res.status === 401) showBanner(tr("err.keyInvalid"), "err");
      return { error: msg };
    }
    // Incremental think-tag router and paint helper are declared above the try.
    await readSSE(res, function (ev) {
      // The usage frame arrives last with an empty choices array; capture it
      // before the choices guard drops the frame.
      if (ev.usage) usageAcc = ev.usage;
      if (ev.error) {
        waitRow.remove();
        out.textContent = ev.error.message || tr("chat.failed");
        out.parentElement.className = "bubble error";
        return;
      }
      if (!(ev.choices && ev.choices[0])) return;
      var d = ev.choices[0].delta || {};
      if (ev.choices[0].finish_reason) finishReason = ev.choices[0].finish_reason;

      // --- Text content: feed the think-tag state machine ---
      if (d.content !== undefined && d.content !== null) {
        if (think.push(d.content)) {
          if (waitRow.parentElement) waitRow.remove();
          paintThinking();
          out.textContent = safePartialMarkdown(think.answer);
        } else {
          // Still deciding where the leading run belongs: keep the spinner row
          // and label it as reasoning rather than a stalled prompt.
          waitTxt.textContent = tr("chat.thinking");
        }
      }
      // Tool calls (streaming: accumulated by index)
      if (d.tool_calls) {
        if (waitRow.parentElement) waitRow.remove();
        d.tool_calls.forEach(function (tc) {
          var idx = tc.index;
          if (!toolCallAccum[idx]) {
            toolCallAccum[idx] = {
              id: tc.id || ("call_" + idx),
              type: "function",
              function: { name: "", arguments: "" },
            };
          }
          if (tc.id) toolCallAccum[idx].id = tc.id;
          if (tc.function) {
            if (tc.function.name) toolCallAccum[idx].function.name += tc.function.name;
            if (tc.function.arguments) toolCallAccum[idx].function.arguments += tc.function.arguments;
          }
        });
        setPill("generating", tr("tool.calling"));
      }
      $("msg-list").scrollTop = $("msg-list").scrollHeight;
    });
    if (waitRow.parentElement) waitRow.remove();
  } catch (e) {
    waitRow.remove();
    if (e.name === "AbortError") {
      out.textContent += tr("chat.stoppedSuffix");
      return { aborted: true };
    } else {
      out.textContent = tr("chat.failedPrefix") + e.message;
      out.parentElement.className = "bubble error";
      return { error: e.message };
    }
  }
  // If there are tool calls, render them and return
  var toolCalls = [];
  for (var idx in toolCallAccum) toolCalls.push(toolCallAccum[idx]);
  if (toolCalls.length) {
    out.parentElement.remove();
    addToolCallMsg(toolCalls);
    return { toolCalls: toolCalls };
  }
  // Finalize: flush any held-back tail and paint the thinking panel
  think.finish();
  paintThinking();
  out.textContent = think.answer;
  // Render Markdown for completed assistant messages
  if (out.textContent) renderMsgMarkdown(out);
  if (usageAcc) renderUsageMeta(out.parentElement, usageAcc);
  return { text: out.textContent };
}

function stopChat() {
  if (abortCtrl) abortCtrl.abort();
}

function newChat() {
  if (streaming) return;
  // Create a new conversation via the conversation system
  var conv = QinferConv.create();
  QinferConv.switchTo(conv.id);
  setPill("idle", tr("chat.newChat"));
  showToast(tr("chat.newChat"), "info");
}

/* ---- Export chat as Markdown ---- */
function exportChat() {
  const rows = [];
  document.querySelectorAll("#msg-list .bubble").forEach(function (b) {
    // Skip hidden tool result bubbles
    if (b.classList.contains("tool-result") && b.style.display === "none") return;
    
    // Tool call bubble
    if (b.classList.contains("tool-call")) {
      var tcs = b._toolCalls || [];
      tcs.forEach(function(tc) {
        rows.push("### " + tr("tool.calling") + ": " + tc.function.name + "\n\n```json\n" + tc.function.arguments + "\n```\n");
      });
      // Include tool results
      b.querySelectorAll(".tool-card").forEach(function(card) {
        var name = card.dataset.toolName;
        var output = card.querySelector(".tool-card__output");
        if (output) {
          rows.push("### " + tr("tool.result") + ": " + name + "\n\n```\n" + output.textContent + "\n```\n");
        }
      });
      return;
    }
    
    // Normal user/assistant bubble
    const role = b.classList.contains("user") ? tr("chat.you") : tr("chat.assistant");
    const txt = b._rawText || b.querySelector(".bubble-body").textContent;
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
    if (state !== "idle" && state !== "?") {
      setPill(state, tr("mon.stateLine") + " " + trState(state) + " \u00b7 " + tok + " " + tr("unit.tokS"));
    } else {
      setPill("idle", tr("state.idle"));
    }
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
  ["s-temp", "s-top-p", "s-top-k", "s-min-p", "s-pres-pen", "s-freq-pen", "s-rep-pen", "s-seed", "s-max-tokens", "s-ctx"].forEach(function (id) {
    if ($(id)) $(id).addEventListener("change", saveSampler);
  });
  // Thinking mode: the composer toggle is a shortcut for the Generation selector
  // (Off <-> last non-off level).
  var thinkBtn = $("btn-thinking");
  var lastThink = "xhigh";
  if ($("s-think")) {
    $("s-think").addEventListener("change", function () {
      if ($("s-think").value !== "off") lastThink = $("s-think").value;
      syncThinkBtn(); saveSampler();
    });
  }
  if (thinkBtn) {
    thinkBtn.addEventListener("click", function() {
      var sel = $("s-think");
      if (!sel) return;
      if (sel.value === "off") { sel.value = lastThink; } else { lastThink = sel.value; sel.value = "off"; }
      syncThinkBtn();
      saveSampler();
      showToast(sel.value === "off" ? tr("chat.thinkingOff") : tr("chat.thinkingOn"));
    });
  }
  syncThinkBtn();
  initAttachmentHandlers();
  $("btn-send").addEventListener("click", sendChat);
  $("btn-stop").addEventListener("click", stopChat);
  $("btn-export").addEventListener("click", exportChat);
  const composer = $("composer");
  if (composer) composer.addEventListener("submit", function (e) { e.preventDefault(); sendChat(); });
  $("chat-input").addEventListener("keydown", function (e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendChat();
    }
  });  // Global keyboard shortcuts
  document.addEventListener("keydown", function (e) {
    // Escape to stop generation
    if (e.key === "Escape" && streaming && abortCtrl) {
      stopChat();
    }
    // Cmd/Ctrl+K focuses the composer.
    if ((e.metaKey || e.ctrlKey) && (e.key === "k" || e.key === "K")) {
      e.preventDefault();
      var ci = $("chat-input");
      if (ci) ci.focus();
    }
  });
  // Tool panel
  initToolPanel();
  // Conversation history search: filter the rendered list by title.
  var convSearch = $("conv-search");
  if (convSearch) {
    convSearch.addEventListener("input", function () {
      var q = convSearch.value.trim().toLowerCase();
      var list = $("conv-list");
      if (!list) return;
      var shown = 0;
      list.querySelectorAll(".conv-item").forEach(function (it) {
        var nameEl = it.querySelector(".conv-item__name");
        var t = (nameEl ? nameEl.textContent : "").toLowerCase();
        var hit = !q || t.indexOf(q) >= 0;
        it.classList.toggle("hidden", !hit);
        if (hit) shown++;
      });
      var note = list.querySelector(".conv-no-match");
      if (q && shown === 0 && list.querySelector(".conv-item")) {
        if (!note) {
          note = document.createElement("p");
          note.className = "muted small conv-no-match";
          note.textContent = tr("conv.noMatch");
          list.appendChild(note);
        }
      } else if (note) {
        note.remove();
      }
    });
  }
  // Message action handlers (event delegation)
  $("msg-list").addEventListener("click", function(e) {
    var btn = e.target.closest(".bubble-action");
    if (!btn) return;
    var bubble = btn.closest(".bubble");
    if (!bubble) return;
    var action = btn.dataset.action;
    var bodyEl = bubble.querySelector(".bubble-body");
    var text = bubble._rawText || bodyEl.textContent;
    
    if (action === "copy") {
      navigator.clipboard.writeText(text).then(function() {
        showToast(tr("toast.copied"), "success");
      });
    } else if (action === "regenerate") {
      // Remove this assistant message and resend the last user message
      var prevBubble = bubble.previousElementSibling;
      while (prevBubble && !prevBubble.classList.contains("user")) {
        prevBubble = prevBubble.previousElementSibling;
      }
      if (prevBubble && prevBubble.classList.contains("user")) {
        bubble.remove();
        // Get the user message text
        var userText = prevBubble._rawText || prevBubble.querySelector(".bubble-body").textContent;
        // Remove the user bubble too and resend
        prevBubble.remove();
        // Re-add user bubble and send
        addMsg("user", userText);
        sendChat();
      }
    } else if (action === "edit") {
      // Retract this user turn: drop it and everything after it, and put the
      // text back in the composer so it can be edited and sent again.
      if (streaming) return;
      var editBody = bubble.querySelector(".bubble-body");
      var editText = bubble._rawText || (editBody ? editBody.textContent : "");
      var sib = bubble.nextElementSibling;
      while (sib) { var nxt = sib.nextElementSibling; sib.remove(); sib = nxt; }
      bubble.remove();
      var editTa = $("chat-input");
      editTa.value = editText;
      editTa.focus();
      showToast(tr("chat.editHint"), "info");
    } else if (action === "branch") {
      QinferConv.branchFrom(bubble);
    } else if (action === "thumbup" || action === "thumbdown") {
      // Toggle active state
      var wasActive = btn.classList.contains("active");
      // Remove active from both thumb buttons
      bubble.querySelectorAll('.bubble-action[data-action^="thumb"]').forEach(function(b) {
        b.classList.remove("active");
      });
      if (!wasActive) {
        btn.classList.add("active");
        // Store feedback in localStorage
        var feedback = JSON.parse(localStorage.getItem("qinfer.feedback") || "{}");
        var msgId = bubble.dataset.msgId || ("msg_" + Date.now());
        bubble.dataset.msgId = msgId;
        feedback[msgId] = action === "thumbup" ? "up" : "down";
        localStorage.setItem("qinfer.feedback", JSON.stringify(feedback));
      } else {
        // Remove feedback
        var msgId = bubble.dataset.msgId;
        if (msgId) {
          var feedback = JSON.parse(localStorage.getItem("qinfer.feedback") || "{}");
          delete feedback[msgId];
          localStorage.setItem("qinfer.feedback", JSON.stringify(feedback));
        }
      }
    }
  });
  // Conversation history sidebar
  QinferConv.init();
  // Scroll to bottom button
  initScrollToBottom();
}

function initToolPanel() {
  var btn = $("btn-tools");
  var panel = $("tool-panel");
  if (!btn || !panel) return;
  btn.addEventListener("click", function () {
    panel.classList.toggle("hidden");
    renderToolList();
  });
  renderToolList();
}

function renderToolList() {
  var list = $("tool-list");
  if (!list) return;
  var tools = QinferTools.list();
  if (!tools.length) {
    list.innerHTML = '<p class="muted small">' + esc(tr("tool.noTools")) + '</p>';
    updateToolCount();
    return;
  }
  list.innerHTML = "";
  tools.forEach(function (t) {
    var item = document.createElement("div");
    item.className = "chat-tool-item";
    var info = document.createElement("div");
    info.className = "chat-tool-item__info";
    var name = document.createElement("div");
    name.className = "chat-tool-item__name";
    name.textContent = t.name;
    var desc = document.createElement("div");
    desc.className = "chat-tool-item__desc";
    desc.textContent = t.description;
    info.appendChild(name);
    info.appendChild(desc);
    var toggle = document.createElement("button");
    toggle.className = "st-toggle";
    toggle.setAttribute("role", "switch");
    toggle.setAttribute("aria-checked", String(t.enabled));
    toggle.addEventListener("click", function () {
      var next = !QinferTools._registry[t.name].enabled;
      QinferTools.toggle(t.name, next);
      toggle.setAttribute("aria-checked", String(next));
      updateToolCount();
      showToast(t.name + (next ? " " + tr("tool.enabled") : " " + tr("tool.disabled")), "info");
    });
    item.appendChild(info);
    item.appendChild(toggle);
    list.appendChild(item);
  });
  updateToolCount();
}

function updateToolCount() {
  var count = $("tool-count");
  if (!count) return;
  var n = QinferTools.getDefinitions().length;
  count.textContent = n || "";
}

function initScrollToBottom() {
  var scrollContainer = $("chat-scroll");
  var btn = $("btn-scroll-bottom");
  if (!scrollContainer || !btn) return;
  
  // Show/hide button based on scroll position
  scrollContainer.addEventListener("scroll", function() {
    var isAtBottom = scrollContainer.scrollHeight - scrollContainer.scrollTop - scrollContainer.clientHeight < 100;
    btn.classList.toggle("hidden", isAtBottom);
  });
  
  // Scroll to bottom on click
  btn.addEventListener("click", function() {
    scrollContainer.scrollTo({ top: scrollContainer.scrollHeight, behavior: "smooth" });
  });
}

/* ---- Conversation history management ---- */
var QinferConv = {
  STORAGE_KEY: "qinfer.conversations",
  currentId: null,

  // Load all conversations from localStorage
  loadAll: function() {
    try {
      return JSON.parse(localStorage.getItem(this.STORAGE_KEY) || "[]");
    } catch (e) { return []; }
  },

  // Save all conversations to localStorage
  saveAll: function(convs) {
    try { localStorage.setItem(this.STORAGE_KEY, JSON.stringify(convs)); } catch (e) {}
  },

  // Get current conversation
  getCurrent: function() {
    if (!this.currentId) return null;
    var convs = this.loadAll();
    return convs.find(function(c) { return c.id === QinferConv.currentId; }) || null;
  },

  // Create a new conversation (does NOT switch to it — caller must call switchTo)
  create: function() {
    var convs = this.loadAll();
    var id = "conv_" + Date.now();
    var conv = { id: id, title: tr("conv.untitled"), messages: [], created: Date.now(), updated: Date.now() };
    convs.unshift(conv);
    this.saveAll(convs);
    this.renderList();
    return conv;
  },

  // Branch: copy the messages up to and including a bubble into a new conversation.
  branchFrom: function(bubble) {
    var convs = this.loadAll();
    var cur = convs.find(function(c) { return c.id === QinferConv.currentId; });
    var msgs = [];
    var all = document.querySelectorAll("#msg-list .bubble");
    for (var i = 0; i < all.length; i++) {
      var b = all[i];
      if (b.classList.contains("tool-call") || b.classList.contains("tool-result")) continue;
      var role = b.classList.contains("user") ? "user" : "assistant";
      var bodyEl = b.querySelector(".bubble-body");
      var text = b._rawText || (bodyEl ? bodyEl.textContent : "");
      if (text) msgs.push({ role: role, content: text });
      if (b === bubble) break;
    }
    var nid = "conv_" + Date.now();
    convs.unshift({
      id: nid,
      title: tr("conv.branched"),
      messages: msgs,
      created: Date.now(),
      updated: Date.now(),
      params: cur ? cur.params : undefined,
    });
    this.saveAll(convs);
    this.switchTo(nid);
    this.renderList();
    showToast(tr("conv.branchedToast"), "success");
  },

  // Switch to a conversation
  switchTo: function(id) {
    // Save current conversation first
    this.saveCurrent();
    this.currentId = id;
    // Load messages into the UI
    var conv = this.getCurrent();
    if (!conv) return;
    // Restore this conversation's parameter overrides (absent = keep current).
    samplerApply(conv.params);
    $("msg-list").innerHTML = "";
    if (conv.messages.length === 0) {
      var empty = $("chat-empty");
      if (empty) empty.style.display = "";
    } else {
      var empty = $("chat-empty");
      if (empty) empty.style.display = "none";
      conv.messages.forEach(function(msg) {
        if (msg.role === "user") {
          addMsg("user", msg.content);
        } else if (msg.role === "assistant") {
          var body = addMsg("assistant", "");
          // Parse thinking content from saved data
          var parsed = parseThinkingContent(msg.content);
          var answerText = parsed.answer || msg.content;
          var thinkingText = parsed.thinking || msg.thinking || "";
          // Render answer as Markdown
          if (answerText) {
            body.innerHTML = QinferMD.render(answerText);
          }
          body.parentElement._rawText = answerText;
          // Populate thinking panel if present
          if (thinkingText) {
            var tSec = body.parentElement._thinkingSection;
            var tBody = body.parentElement._thinkingBody;
            if (tBody) tBody.textContent = thinkingText;
            if (tSec) tSec.style.display = "";
            body.parentElement._thinkingText = thinkingText;
          }
        }
      });
    }
    this.renderList();
  },

  // Save current conversation state
  saveCurrent: function() {
    if (!this.currentId) return;
    var convs = this.loadAll();
    var conv = convs.find(function(c) { return c.id === QinferConv.currentId; });
    if (!conv) return;
    // Collect messages from the UI
    var messages = [];
    document.querySelectorAll("#msg-list .bubble").forEach(function(b) {
      if (b.classList.contains("tool-call") || b.classList.contains("tool-result")) return;
      var role = b.classList.contains("user") ? "user" : "assistant";
      var text = b._rawText || b.querySelector(".bubble-body").textContent;
      if (text) {
        var msg = { role: role, content: text };
        // Save thinking content separately for assistant messages
        if (role === "assistant") {
          var thinkingText = b._thinkingText || "";
          if (thinkingText) msg.thinking = thinkingText;
        }
        messages.push(msg);
      }
    });
    conv.messages = messages;
    conv.params = samplerSnapshot();
    conv.updated = Date.now();
    // Auto-generate title from first user message
    if (conv.title === tr("conv.untitled") && messages.length > 0) {
      var firstUser = messages.find(function(m) { return m.role === "user"; });
      if (firstUser) {
        conv.title = firstUser.content.slice(0, 40) + (firstUser.content.length > 40 ? "…" : "");
      }
    }
    this.saveAll(convs);
  },

  // Delete a conversation
  delete: function(id) {
    if (!confirm(tr("conv.confirmDelete"))) return;
    var convs = this.loadAll();
    convs = convs.filter(function(c) { return c.id !== id; });
    this.saveAll(convs);
    if (this.currentId === id) {
      this.currentId = null;
      $("msg-list").innerHTML = "";
      var empty = $("chat-empty");
      if (empty) empty.style.display = "";
    }
    this.renderList();
  },

  // Rename a conversation
  rename: function(id) {
    var convs = this.loadAll();
    var conv = convs.find(function(c) { return c.id === id; });
    if (!conv) return;
    var newTitle = prompt(tr("conv.rename"), conv.title);
    if (newTitle !== null && newTitle.trim()) {
      conv.title = newTitle.trim();
      this.saveAll(convs);
      this.renderList();
    }
  },

  // Render the conversation list
  renderList: function() {
    var list = $("conv-list");
    if (!list) return;
    var convs = this.loadAll();
    if (convs.length === 0) {
      list.innerHTML = '<p class="muted small">' + esc(tr("conv.empty")) + '</p>';
      return;
    }
    list.innerHTML = "";
    convs.forEach(function(conv) {
      var item = document.createElement("div");
      item.className = "conv-item" + (conv.id === QinferConv.currentId ? " active" : "");
      item.dataset.id = conv.id;
      item.innerHTML = '<span class="conv-item__title"><span class="conv-item__name">' + esc(conv.title) +
        '</span><span class="conv-item__time">' + relTime(conv.updated) + '</span></span>' +
        '<span class="conv-item__actions">' +
        '<button class="conv-item__btn" data-action="rename" title="' + esc(tr("conv.rename")) + '"><svg class="st-icon st-icon--sm"><use href="/web/sprite.svg#i-edit"/></svg></button>' +
        '<button class="conv-item__btn" data-action="delete" title="' + esc(tr("conv.delete")) + '"><svg class="st-icon st-icon--sm"><use href="/web/sprite.svg#i-trash"/></svg></button>' +
        '</span>';
      item.addEventListener("click", function(e) {
        if (e.target.closest(".conv-item__btn")) return;
        QinferConv.switchTo(conv.id);
      });
      item.querySelector('[data-action="rename"]').addEventListener("click", function(e) {
        e.stopPropagation();
        QinferConv.rename(conv.id);
      });
      item.querySelector('[data-action="delete"]').addEventListener("click", function(e) {
        e.stopPropagation();
        QinferConv.delete(conv.id);
      });
      list.appendChild(item);
    });
  },

  // Initialize the sidebar
  init: function() {
    var sidebar = $("conv-sidebar");
    var btnToggle = $("btn-toggle-conv");
    var btnClose = $("btn-close-conv");
    var btnNew = $("btn-new-conv");
    if (btnToggle) {
      btnToggle.addEventListener("click", function() {
        sidebar.classList.toggle("hidden");
      });
    }
    if (btnClose) {
      btnClose.addEventListener("click", function() {
        sidebar.classList.add("hidden");
      });
    }
    if (btnNew) {
      btnNew.addEventListener("click", function() {
        var conv = QinferConv.create();
        QinferConv.switchTo(conv.id);
      });
    }
    // Load or create initial conversation
    var convs = this.loadAll();
    if (convs.length > 0) {
      this.switchTo(convs[0].id);
    } else {
      this.create();
    }
    this.renderList();
  }
};

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
      '<div class="spark-wrapper">' +
      '<svg class="st-metric__spark" id="sp-' + m.key + '" viewBox="0 0 100 32" preserveAspectRatio="none"' + toneAttr + '>' +
      '<path class="area" fill="currentColor" opacity=".12"/>' +
      '<path class="line" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linejoin="round" stroke-linecap="round" vector-effect="non-scaling-stroke"/>' +
      '</svg>' +
      '<div class="spark-axis"><span class="axis-max">0</span><span class="axis-min">0</span></div>' +
      '</div></div></div>';
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
  // vram stores bytes; convert to GB for display and shape calculation
  var isVram = (id === "sp-vram");
  var divisor = isVram ? 1073741824 : 1;
  var vDisplay = v.map(function(x) { return x / divisor; });
  var maxDisplay = max ? max / divisor : 0;
  
  var top = Math.max(maxDisplay, Math.max.apply(null, vDisplay), 0.01);
  var positiveValues = vDisplay.filter(function(x) { return x > 0; });
  var min = positiveValues.length ? Math.min.apply(null, positiveValues) : 0;
  
  var pts = vDisplay.map(function (x, i) {
    return [(i / (vDisplay.length - 1)) * 100, 30 - (x / top) * 26];
  });
  var line = pts.map(function (p, i) { return (i ? "L" : "M") + p[0].toFixed(2) + "," + p[1].toFixed(2); }).join("");
  svg.querySelector(".line").setAttribute("d", line);
  svg.querySelector(".area").setAttribute("d", line + "L100,32L0,32Z");
  
  // Update axis labels (now in HTML, not SVG)
  var wrapper = svg.parentElement;
  if (wrapper) {
    var maxLabel = wrapper.querySelector(".axis-max");
    var minLabel = wrapper.querySelector(".axis-min");
    if (maxLabel) maxLabel.textContent = fmtAxis(top);
    if (minLabel) minLabel.textContent = fmtAxis(min);
  }
}

function fmtAxis(v) {
  if (v === 0) return "0";
  if (v >= 1000) return (v / 1000).toFixed(1) + "k";
  if (v >= 100) return Math.round(v);
  if (v >= 10) return v.toFixed(1);
  return v.toFixed(2);
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
        var t = r.time ? new Date(r.time * 1000).toLocaleTimeString([], {hour: "2-digit", minute: "2-digit", second: "2-digit"}) : "\u2013";
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