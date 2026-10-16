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

  // Reverse of escapeHtml. render() escapes the whole document before it pulls the
  // fences out, so a fence's payload comes back escaped; this restores the raw source
  // the highlighter and the copy button both want. One pass, so "&amp;lt;" -> "&lt;".
  unescapeHtml: function(text) {
    var map = { '&lt;': '<', '&gt;': '>', '&quot;': '"', '&#039;': "'", '&amp;': '&' };
    return text.replace(/&(lt|gt|quot|#039|amp);/g, function(x) { return map[x]; });
  },

  // Fenced-code syntax highlighting: dependency-free and deliberately small. One
  // tokenizer pass over the RAW source wraps line/block comments, string literals,
  // numbers, keywords and call-like identifiers; every token and every gap is escaped
  // through escapeHtml, so nothing unescaped can reach the DOM (the payload is
  // untrusted model output).
  _kw: /\b(?:abstract|and|as|assert|async|await|begin|break|case|catch|class|const|continue|def|default|del|do|elif|else|end|enum|except|export|extends|false|final|finally|fn|for|from|func|function|global|goto|if|impl|import|in|instanceof|interface|lambda|let|local|match|module|mut|namespace|new|nil|none|null|of|or|package|pass|private|protected|public|raise|require|rescue|return|self|static|struct|super|switch|template|then|this|throw|true|try|type|typename|typeof|undefined|unless|until|use|using|var|virtual|void|when|where|while|with|yield)\b/,
  _num: /\b(?:0[xX][0-9a-fA-F]+|\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\b/,
  _hashLang: /^(?:py|python|rb|ruby|sh|bash|zsh|fish|yaml|yml|toml|ini|conf|cfg|dockerfile|make|pl|perl|r|jl|nim)$/i,
  _dashLang: /^(?:sql|lua|hs|haskell|elm|ada)$/i,
  highlight: function(code, lang) {
    if (code == null) return "";
    code = String(code);
    if (!code) return "";
    var l = (lang || "").toLowerCase();
    var line = this._hashLang.test(l) ? "#[^\\n]*" : (this._dashLang.test(l) ? "--[^\\n]*" : "\\/\\/[^\\n]*");
    var re = new RegExp("(" + line + ")"
      + "|(\\/\\*[\\s\\S]*?\\*\\/)"
      + "|(\"(?:\\\\.|[^\"\\\\\\n])*\"|'(?:\\\\.|[^'\\\\\\n])*'|`(?:\\\\.|[^`\\\\])*`)"
      + "|(" + this._num.source + ")"
      + "|(" + this._kw.source + ")"
      + "|([A-Za-z_$][\\w$]*)(?=\\s*\\()", "gi");
    var out = "", last = 0, m;
    while ((m = re.exec(code)) !== null) {
      if (m.index > last) out += this.escapeHtml(code.slice(last, m.index));
      var cls = (m[1] || m[2]) ? "tok-cmt" : m[3] ? "tok-str" : m[4] ? "tok-num" : m[5] ? "tok-kw" : "tok-fn";
      out += '<span class="' + cls + '">' + this.escapeHtml(m[0]) + '</span>';
      last = m.index + m[0].length;
      if (m[0].length === 0) re.lastIndex++;
    }
    out += this.escapeHtml(code.slice(last));
    return out;
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
      codeBlocks.push({ lang: lang, code: QinferMD.unescapeHtml(code) });
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
        result.push('<pre class="md-code-block"><code' + langClass + '>' + this.highlight(cb.code, cb.lang) + '</code><button class="md-copy-btn" data-code="' + encodeURIComponent(cb.code) + '">Copy</button></pre>');
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
      return '<pre class="md-code-block"><code' + langClass + '>' + QinferMD.highlight(cb.code, cb.lang) + '</code><button class="md-copy-btn" aria-label="' + tr("chat.codeCopy") + '" data-code="' + encodeURIComponent(cb.code) + '">' + tr("chat.codeCopy") + '</button></pre>';
    });
    
    return text;
  },

  // Process inline Markdown (bold, italic, links, inline code)
  processInline: function(text, inlineCodes) {
    // Restore inline codes
    text = text.replace(/\x00INLINECODE(\d+)\x00/g, function(match, idx) {
      return '<code>' + inlineCodes[parseInt(idx)] + '</code>';
    });
    
    // Bold: **text** or __text__. Asterisks emphasise anywhere; an underscore only
    // at a word boundary, because the underscore rules below are the same rules.
    text = text.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
    text = text.replace(/(^|[^\w])__(?=\S)([\s\S]*?\S)__(?!\w)/g, '$1<strong>$2</strong>');

    // Italic: *text* or _text_. An underscore never emphasises inside a word —
    // replies are full of snake_case identifiers (max_tokens, has_rings) and a
    // greedy _(.*?)_ swallows the underscores right out of the name being named.
    text = text.replace(/\*(.+?)\*/g, '<em>$1</em>');
    text = text.replace(/(^|[^\w])_(?=\S)([\s\S]*?\S)_(?!\w)/g, '$1<em>$2</em>');
    
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
    "brand.tag": "私有推理",
    "lang.title": "切换语言（中 / EN）",
    "state.connecting": "连接中…",
    "chat.title": "随便问，都在本机",
    "chat.sub": "模型完全跑在这台电脑上，你输入的内容不会离开它。",
    "chat.suggestions": "起步建议",
    "chat.sug1": "解释一段代码",
    "chat.sug2": "写一组单元测试",
    "chat.sug3": "总结一段文字",
    "chat.sug4": "翻译成中文",
    "chat.sampler": "采样器",
    "chat.seedLabel": "seed（-1 = 随机）",
    "chat.inputPh": "输入消息，Enter 发送，Shift+Enter 换行",
    "chat.stop": "停止",
    "chat.you": "你",
    "chat.assistant": "Q-Infer",
    "chat.copy": "复制",
    "chat.regenerate": "重新生成",
    "chat.good": "有帮助",
    "chat.bad": "没帮助",
    "chat.codeCopy": "复制",
    "chat.codeCopied": "已复制",
    "chat.usageMeta": "提示词 {p} · 生成 {c} · 合计 {t} tokens",
    "chat.edit": "编辑",
    "chat.editHint": "已撤回该轮，改好后发送",
    "chat.continueTitle": "继续生成",
    "chat.continueWord": "继续",
    "chat.thinkMode": "思考模式",
    "chat.thinkOff": "关",
    "chat.contextLen": "上下文长度",
    "chat.ctxAuto": "自动",
    "chat.scrollBottom": "回到最新",
    "chat.copyFailed": "复制失败：浏览器拒绝了剪贴板访问",
    "chat.outline": "对话大纲",
    "chat.outlineEmpty": "还没有内容",
    "chat.prevTurn": "上一轮",
    "chat.nextTurn": "下一轮",
    "queue.title": "消息队列",
    "queue.pause": "暂停",
    "queue.resume": "继续",
    "queue.paused": "已暂停，队列不再自动发送",
    "queue.waiting": "待发送",
    "queue.clear": "清空",
    "queue.remove": "移除",
    "queue.edit": "编辑",
    "queue.interject": "打断并发送",
    "queue.reorder": "拖动排序",
    "queue.full": "队列已满（上限 {n} 条）",
    "queue.queued": "已加入队列",
    "queue.cleared": "队列已清空",
    "queue.attachment": "（仅附件）",
    "conv.closeSidebar": "收起侧栏",
    "conv.searchPh": "搜索对话",
    "conv.noMatch": "无匹配对话",
    "conv.timeNow": "刚刚",
    "conv.timeMin": "{n} 分钟前",
    "conv.timeHour": "{n} 小时前",
    "conv.timeDay": "{n} 天前",
    "conv.branch": "从这里分支",
    "conv.branched": "分支对话",
    "conv.branchedToast": "已从这一轮新建对话",
    "chat.error": "错误",
    "chat.prefilling": "正在预填充…",
    "chat.thinking": "思考中…",
    "chat.topK": "top_k",
    "chat.minP": "min_p",
    "chat.presPen": "存在惩罚",
    "chat.freqPen": "频率惩罚",
    "chat.repPen": "重复惩罚",
    "chat.stopped": "已停止",
    "chat.stoppedSuffix": "（已停止）",
    "chat.failed": "请求失败",
    "chat.failedPrefix": "请求失败：",
    "chat.idle": "空闲",
    "chat.newChat": "新对话已开始",
    "chat.exported": "已导出 qinfer-chat.md",
    "chat.noExport": "没有可导出的对话",
    "metric.speed": "解码速度",
    "metric.prefill": "预填充速度",
    "metric.gpuLoad": "GPU 负载",
    "metric.vram": "显存",
    "metric.ram": "系统内存",
    "metric.temp": "GPU 温度",
    "metric.power": "功率",
    "metric.pcie": "PCIe",
    "metric.cpu": "CPU",
    "metric.disk": "磁盘读取",
    "metric.igpu": "核显频率",
    "sampler.preset": "预设",
    "sampler.precise": "精确",
    "sampler.balanced": "平衡",
    "sampler.creative": "创造",
    "sampler.custom": "自定义",
    "sampler.sampling": "采样",
    "sampler.generation": "生成",
    "sampler.reset": "恢复默认",
    "sampler.resetDone": "已恢复默认参数",
    "sampler.overrideTip": "本对话有自己的参数，切回时自动生效",
    "sampler.overrideSwitch": "仅本对话使用",
    "sampler.overrideKept": "本对话已保存自己的参数",
    "sampler.overrideCleared": "已改为跟随全局参数",
    "sampler.tipTemp": "0-2：越大越随机；0.8 平衡，0.2 稳定",
    "sampler.tipTopP": "0.05-1：只从累计概率前 p 的候选里采样",
    "sampler.tipTopK": "0-1000：只保留概率前 k 个词；0 = 关闭",
    "sampler.tipMinP": "0-1：丢弃概率低于最高词 p 倍的候选",
    "sampler.tipPresPen": "0-2：按是否已出现惩罚，鼓励展开新话题",
    "sampler.tipFreqPen": "0-2：按出现次数惩罚，抑制重复",
    "sampler.tipRepPen": "1-2：对已出现的词施加对数概率惩罚；1 = 关闭",
    "sampler.tipMaxTokens": "单次回复的最大 token 数",
    "sampler.tipSeed": "固定种子可复现采样；-1 每次随机",
    "chat.thinkLow": "低",
    "chat.thinkMedium": "中",
    "chat.thinkXhigh": "极高",
    "keys.title": "键盘快捷键",
    "keys.focusInput": "聚焦输入框",
    "keys.newConv": "新建对话",
    "keys.toggleSidebar": "显示或隐藏对话栏",
    "keys.help": "打开这份快捷键清单",
    "keys.escape": "关闭当前弹层；没有弹层时停止生成",
    "keys.send": "发送消息",
    "keys.newline": "输入框内换行",
    "chat.system": "系统提示词",
    "chat.systemPh": "只对本对话生效的系统指令（可留空）",
    "chat.systemHint": "作为本对话的首条 system 消息发送，不影响其他对话。",
    "misc.close": "关闭",
    "mon.state": "模型状态",
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
    "about.apiKeyNote": "密钥只存在浏览器本地（localStorage），跨标签页共享，不会上传。",
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
    "theme.title": "浅色 / 深色",
    "theme.dark": "深色主题",
    "theme.light": "浅色主题",
    "settings.appearance": "外观",
    "settings.theme": "主题",
    "settings.themeLight": "浅色",
    "settings.themeDark": "深色",
    "settings.themeSystem": "跟随系统",
    "settings.accentColor": "主题色",
    "settings.accentQwen": "千问紫",
    "settings.accentVibrant": "活力橙",
    "settings.accentBlue": "蓝",
    "settings.accentPurple": "紫",
    "settings.accentGreen": "绿",
    "settings.accentRose": "玫红",
    "settings.accentTeal": "青绿",
    "settings.accentAmber": "琥珀",
    "settings.accentIndigo": "靛蓝",
    "settings.accentCyan": "青蓝",
    "settings.accentCustom": "自定义",
    "settings.customColor": "选择自定义颜色",
    "settings.connection": "连接",
    "settings.defaultSystemPrompt": "默认系统提示词",
    "settings.systemPromptPh": "所有新对话的默认系统指令（可留空）",
    "settings.defaults": "默认生成参数",
    "settings.defaultsHint": "新建对话或点击重置时使用这些值。",
    "settings.profiles": "生成参数配置",
    "settings.profileDefault": "默认",
    "settings.profileName": "配置",
    "settings.profileCopySuffix": "副本",
    "settings.profileNew": "新建配置",
    "settings.profileDuplicate": "复制配置",
    "settings.profileRename": "重命名",
    "settings.profileDelete": "删除配置",
    "settings.profileRenamePrompt": "配置名称",
    "settings.profileDeleteConfirm": "删除配置「{name}」？",
    "settings.profileAdded": "已新建配置",
    "settings.profileDuplicated": "已复制配置",
    "settings.profileDeleted": "已删除配置",
    "settings.applyDefaults": "应用到当前对话",
    "settings.resetDefaults": "恢复出厂默认",
    "settings.saved": "设置已保存",
    "settings.defaultsApplied": "默认参数已应用到当前对话",
    "settings.defaultsReset": "已恢复出厂默认",
    "hw.cpu": "CPU",
    "hw.ram": "内存",
    "hw.igpuFreq": "核显频率",
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
    "state.unreachable": "不可达",
    "state.noResponse": "服务无响应",

    "tab.chat": "对话",
    "tab.monitor": "监控",
    "tab.requests": "请求",
    "tab.about": "关于",
    "tab.settings": "设置",
    "nav.views": "视图",
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
    "mon.session": "本次会话",
    "mon.reqHistory": "请求记录",
    "act.requests": "请求数",
    "act.prompt": "提示词 token",
    "act.completion": "输出 token",
    "act.reused": "复用 token",
    "act.busy": "计算耗时",
    "act.perReq": "平均每次 {n}",
    "act.ofPrompt": "占输入 {n}",
    "act.prefill": "预填充",
    "act.decode": "解码",
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
    "cc.reusedPrompt": "已复用提示词",
    "cc.enabledNote": "继续已缓存的对话时，引擎会直接恢复状态，无需重新处理。",
    "cc.disabledNote": "引擎保留上次对话的状态，后续请求只需处理新增内容。",
    "cc.hitRate": "命中率",
    "cc.hits": "命中",
    "cc.misses": "未命中",
    "cc.cachedTokens": "缓存 token",
    "cc.capacity": "缓存容量",
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
    "mon.decodeMean": "会话均值",
    "mon.decodeLast": "上次请求",
    "mon.decodeLabel": "解码",
    "mon.prefillLabel": "上轮预填充",
    "mon.notAvail": "不可用",
    "mon.limit": "上限",
    "mon.cores": "核心",
    "mon.threads": "线程",
    "toast.copied": "已复制",
    "toast.imgOver": "超过 20 MB",
    "toast.fileOver": "超过 512 KB",
    "toast.notText": "不是文本文件",
    "toast.storageFull": "浏览器存储空间已满，本次保存已丢弃图片附件",

    "chat.temperature": "温度",
    "chat.topP": "top_p",
    "chat.maxTokens": "最大 token 数",
    "filter.queued": "排队中",
    "filter.completed": "已完成",
    "filter.error": "错误",
    "req.tabInput": "输入",
    "req.tabOutput": "输出",
    "req.tabReasoning": "思考",
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
    "misc.cancel": "取消",
    "misc.confirm": "确认",
    "misc.pastedImage": "粘贴的图片",
    "misc.gpu": "GPU",
    "unit.tokS": "tok/s",
    "unit.tokens": "tokens",
    "unit.seconds": "s",

    // Agent tools
    "tool.calling": "调用工具",
    "tool.executing": "执行中…",
    "tool.result": "工具结果",
    "tool.error": "工具执行失败",
    "tool.enabled": "工具已启用",
    "tool.disabled": "工具已禁用",
    "tool.settings": "工具设置",
    "tool.noTools": "未启用任何工具",

    // Conversation history
    "conv.history": "对话",
    "conv.new": "新对话",
    "conv.toggle": "显示对话列表",
    "conv.empty": "暂无对话记录",
    "conv.untitled": "未命名对话",
    "conv.delete": "删除",
    "conv.rename": "重命名",
    "conv.confirmDelete": "确定删除此对话？",
    "conv.pin": "置顶",
    "conv.unpin": "取消置顶",
    "conv.groupPinned": "置顶",
    "conv.groupToday": "今天",
    "conv.groupYesterday": "昨天",
    "conv.groupWeek": "近 7 天",
    "conv.groupEarlier": "更早",
    "conv.exportAll": "导出全部对话",
    "conv.importAll": "导入对话",
    "conv.importBtn": "导入",
    "conv.exportBtn": "导出",
    "conv.exportedToast": "已导出 {n} 个对话",
    "conv.importedToast": "已导入 {n} 个对话",
    "conv.importBadFormat": "导入失败：不是 Q-Infer 导出文件",
  },
  en: {
    "brand.tag": "Private inference",
    "lang.title": "Switch language (CN / EN)",
    "state.connecting": "Connecting…",
    "chat.title": "Ask anything, locally.",
    "chat.sub": "It runs entirely on this machine — what you type never leaves it.",
    "chat.suggestions": "Starter suggestions",
    "chat.sug1": "Explain this code",
    "chat.sug2": "Write unit tests",
    "chat.sug3": "Summarize this text",
    "chat.sug4": "Translate to Chinese",
    "chat.sampler": "Sampler",
    "chat.seedLabel": "seed (-1 = random)",
    "chat.inputPh": "Type a message; Enter to send, Shift+Enter for a new line",
    "chat.stop": "Stop",
    "chat.you": "You",
    "chat.assistant": "Q-Infer",
    "chat.copy": "Copy",
    "chat.regenerate": "Regenerate",
    "chat.good": "Helpful",
    "chat.bad": "Not helpful",
    "chat.codeCopy": "Copy",
    "chat.codeCopied": "Copied",
    "chat.usageMeta": "prompt {p} · completion {c} · total {t} tokens",
    "chat.edit": "Edit",
    "chat.editHint": "Turn retracted — edit and send again",
    "chat.continueTitle": "Continue",
    "chat.continueWord": "continue",
    "chat.thinkMode": "Thinking mode",
    "chat.thinkOff": "Off",
    "chat.contextLen": "Context length",
    "chat.ctxAuto": "Auto",
    "chat.scrollBottom": "Jump to newest",
    "chat.copyFailed": "Copy failed: the browser refused clipboard access",
    "chat.outline": "Outline",
    "chat.outlineEmpty": "Nothing here yet",
    "chat.prevTurn": "Previous turn",
    "chat.nextTurn": "Next turn",
    "queue.title": "Message queue",
    "queue.pause": "Pause",
    "queue.resume": "Resume",
    "queue.paused": "paused — the queue is not sending",
    "queue.waiting": "waiting",
    "queue.clear": "Clear",
    "queue.remove": "Remove",
    "queue.edit": "Edit",
    "queue.interject": "Interject & send",
    "queue.reorder": "Drag to reorder",
    "queue.full": "The queue is full ({n} items max)",
    "queue.queued": "Added to the queue",
    "queue.cleared": "Queue cleared",
    "queue.attachment": "(attachments only)",
    "conv.closeSidebar": "Close sidebar",
    "conv.searchPh": "Search conversations",
    "conv.noMatch": "No matching conversations",
    "conv.timeNow": "just now",
    "conv.timeMin": "{n} min ago",
    "conv.timeHour": "{n} h ago",
    "conv.timeDay": "{n} d ago",
    "conv.branch": "Branch from here",
    "conv.branched": "Branched conversation",
    "conv.branchedToast": "Created a new conversation branching from this turn",
    "chat.error": "Error",
    "chat.prefilling": "Prefilling…",
    "chat.thinking": "Thinking…",
    "chat.topK": "top_k",
    "chat.minP": "min_p",
    "chat.presPen": "presence_penalty",
    "chat.freqPen": "frequency_penalty",
    "chat.repPen": "repeat_penalty",
    "chat.stopped": "Stopped",
    "chat.stoppedSuffix": " (stopped)",
    "chat.failed": "Request failed",
    "chat.failedPrefix": "Request failed: ",
    "chat.idle": "Idle",
    "chat.newChat": "New chat started",
    "chat.exported": "Exported qinfer-chat.md",
    "chat.noExport": "Nothing to export",
    "metric.speed": "Decode speed",
    "metric.prefill": "Prefill speed",
    "metric.ram": "System RAM",
    "metric.gpuLoad": "GPU load",
    "metric.vram": "VRAM",
    "metric.temp": "GPU temperature",
    "metric.power": "Power",
    "metric.pcie": "PCIe",
    "metric.cpu": "CPU",
    "metric.disk": "Disk read",
    "metric.igpu": "iGPU freq",
    "sampler.preset": "Preset",
    "sampler.precise": "Precise",
    "sampler.balanced": "Balanced",
    "sampler.creative": "Creative",
    "sampler.custom": "Custom",
    "sampler.sampling": "Sampling",
    "sampler.generation": "Generation",
    "sampler.reset": "Reset to defaults",
    "sampler.resetDone": "Sampler reset to defaults",
    "sampler.overrideTip": "This conversation has its own settings, applied whenever you switch back to it",
    "sampler.overrideSwitch": "This conversation only",
    "sampler.overrideKept": "This conversation keeps its own settings",
    "sampler.overrideCleared": "Now follows the global parameters",
    "sampler.tipTemp": "0-2: higher is more random; 0.8 balanced, 0.2 steady",
    "sampler.tipTopP": "0.05-1: sample only from the top-p probability mass",
    "sampler.tipTopK": "0-1000: keep only the k most likely tokens; 0 = off",
    "sampler.tipMinP": "0-1: drop candidates below p times the top token's probability",
    "sampler.tipPresPen": "0-2: penalize tokens that already appeared, encouraging new topics",
    "sampler.tipFreqPen": "0-2: penalize by how often a token appeared, curbing repetition",
    "sampler.tipRepPen": "1-2: log-probability penalty on tokens already used; 1 = off",
    "sampler.tipMaxTokens": "Maximum tokens in one reply",
    "sampler.tipSeed": "A fixed seed makes sampling reproducible; -1 draws a new one each time",
    "chat.thinkLow": "Low",
    "chat.thinkMedium": "Medium",
    "chat.thinkXhigh": "Extra high",
    "keys.title": "Keyboard shortcuts",
    "keys.focusInput": "Focus the message box",
    "keys.newConv": "Start a new conversation",
    "keys.toggleSidebar": "Show or hide the conversation list",
    "keys.help": "Open this shortcut list",
    "keys.escape": "Close the open panel; if none is open, stop generating",
    "keys.send": "Send the message",
    "keys.newline": "New line in the message box",
    "chat.system": "System prompt",
    "chat.systemPh": "Instruction just for this conversation (optional)",
    "chat.systemHint": "Sent as this conversation's leading system message; other conversations are untouched.",
    "misc.close": "Close",
    "mon.state": "Model state",
    "mon.searchPh": "Search path / model / state",
    "mon.filterTitle": "Filter by state",
    "mon.all": "All states",
    "mon.noReq": "No requests yet",
    "mon.pickHint": "Pick a request on the left to inspect it",
    "mon.stateLine": "State",
    "mon.queued": "Queued",
    "mon.unreachable": "Server unreachable, retrying…",
    "about.apiKey": "Connection settings",
    "about.apiKeyLabel": "API key",
    "about.apiKeyHint": "Only needed if the server was started with --api-key",
    "about.apiKeyPh": "Not needed",
    "about.show": "Show",
    "about.apiKeyNote": "The key lives only in this browser's local storage, shared across tabs; it is never uploaded.",
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
    "theme.title": "Light / dark theme",
    "theme.dark": "Dark theme",
    "theme.light": "Light theme",
    "settings.appearance": "Appearance",
    "settings.theme": "Theme",
    "settings.themeLight": "Light",
    "settings.themeDark": "Dark",
    "settings.themeSystem": "System",
    "settings.accentColor": "Accent color",
    "settings.accentQwen": "Qwen",
    "settings.accentVibrant": "Vibrant",
    "settings.accentBlue": "Blue",
    "settings.accentPurple": "Purple",
    "settings.accentGreen": "Green",
    "settings.accentRose": "Rose",
    "settings.accentTeal": "Teal",
    "settings.accentAmber": "Amber",
    "settings.accentIndigo": "Indigo",
    "settings.accentCyan": "Cyan",
    "settings.accentCustom": "Custom",
    "settings.customColor": "Pick a custom color",
    "settings.connection": "Connection",
    "settings.defaultSystemPrompt": "Default system prompt",
    "settings.systemPromptPh": "Default instruction for all new conversations (optional)",
    "settings.defaults": "Default generation parameters",
    "settings.defaultsHint": "These values are used when you start a new conversation or click Reset.",
    "settings.profiles": "Profiles",
    "settings.profileDefault": "Default",
    "settings.profileName": "Profile",
    "settings.profileCopySuffix": "copy",
    "settings.profileNew": "New profile",
    "settings.profileDuplicate": "Duplicate profile",
    "settings.profileRename": "Rename profile",
    "settings.profileDelete": "Delete profile",
    "settings.profileRenamePrompt": "Profile name",
    "settings.profileDeleteConfirm": "Delete profile \"{name}\"?",
    "settings.profileAdded": "Profile created",
    "settings.profileDuplicated": "Profile duplicated",
    "settings.profileDeleted": "Profile deleted",
    "settings.applyDefaults": "Apply to current conversation",
    "settings.resetDefaults": "Reset to factory defaults",
    "settings.saved": "Settings saved",
    "settings.defaultsApplied": "Defaults applied to current conversation",
    "settings.defaultsReset": "Factory defaults restored",
    "hw.cpu": "CPU",
    "hw.ram": "RAM",
    "hw.igpuFreq": "iGPU freq",
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
    "state.unreachable": "Unreachable",
    "state.noResponse": "Service not responding",

    "tab.chat": "Chat",
    "tab.monitor": "Monitor",
    "tab.requests": "Requests",
    "tab.about": "About",
    "tab.settings": "Settings",
    "nav.views": "Views",
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
    "mon.session": "This session",
    "mon.reqHistory": "Request log",
    "act.requests": "Requests",
    "act.prompt": "Prompt tokens",
    "act.completion": "Output tokens",
    "act.reused": "Reused tokens",
    "act.busy": "Compute time",
    "act.perReq": "{n} per request",
    "act.ofPrompt": "{n} of prompt",
    "act.prefill": "Prefill",
    "act.decode": "Decode",
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
    "cc.reusedPrompt": "Reused prompt tokens",
    "cc.enabledNote": "Continuing a cached conversation restores its state without reprocessing.",
    "cc.disabledNote": "The engine keeps the last conversation's state so follow-ups only process new content.",
    "cc.hitRate": "Hit rate",
    "cc.hits": "Hits",
    "cc.misses": "Misses",
    "cc.cachedTokens": "Cached tokens",
    "cc.capacity": "Cache capacity",
    "about.clientDesc": "Any OpenAI- or Anthropic-compatible client can use these addresses. Base URL:",
    "about.endpointPaths": "Endpoint paths",
    "about.openaiLabel": "OpenAI",
    "about.anthropicLabel": "Anthropic",
    "about.monitoringLabel": "Monitor",
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
    "mon.decodeMean": "session mean",
    "mon.decodeLast": "last request",
    "mon.decodeLabel": "Decode",
    "mon.prefillLabel": "last prefill",
    "mon.notAvail": "not available",
    "mon.limit": "limit",
    "mon.cores": "cores",
    "mon.threads": "threads",
    "toast.copied": "Copied",
    "toast.imgOver": "is over 20 MB",
    "toast.fileOver": "is over 512 KB",
    "toast.notText": "is not a text file",
    "toast.storageFull": "browser storage is full — image attachments were dropped from this save",

    "chat.temperature": "Temperature",
    "chat.topP": "top_p",
    "chat.maxTokens": "Max tokens",
    "filter.queued": "Queued",
    "filter.completed": "Completed",
    "filter.error": "Error",
    "req.tabInput": "Input",
    "req.tabOutput": "Output",
    "req.tabReasoning": "Thinking",
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
    "misc.cancel": "Cancel",
    "misc.confirm": "Confirm",
    "misc.pastedImage": "pasted image",
    "misc.gpu": "GPU",
    "unit.tokS": "tok/s",
    "unit.tokens": "tokens",
    "unit.seconds": "s",

    // Agent tools
    "tool.calling": "Calling tool",
    "tool.executing": "Executing…",
    "tool.result": "Tool result",
    "tool.error": "Tool execution failed",
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
    "conv.pin": "Pin",
    "conv.unpin": "Unpin",
    "conv.groupPinned": "Pinned",
    "conv.groupToday": "Today",
    "conv.groupYesterday": "Yesterday",
    "conv.groupWeek": "Last 7 days",
    "conv.groupEarlier": "Earlier",
    "conv.exportAll": "Export all conversations",
    "conv.importAll": "Import conversations",
    "conv.importBtn": "Import",
    "conv.exportBtn": "Export",
    "conv.exportedToast": "Exported {n} conversations",
    "conv.importedToast": "Imported {n} conversations",
    "conv.importBadFormat": "Import failed: not a Q-Infer export file",
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

// tr() with one value slot: dictionaries carry "{n}" so each language can put
// the number where its word order wants it.
function trp(key, value) {
  return tr(key).replace("{n}", value);
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

// The header pill reads as a badge, so it carries the state word and the live
// rate — the "State" label belongs to the chat footer summary, not the pill.
function pillText(state, tok) {
  return trState(state) + " · " + tok + " " + tr("unit.tokS");
}

// An untitled conversation stores an empty title, and the placeholder is resolved
// here at render time. Storing tr("conv.untitled") instead would freeze the title
// to whatever language the conversation happened to be created in.
function convTitle(conv) {
  return conv.title || tr("conv.untitled");
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
  // Two runtime surfaces carry their own labels: the composer's primary button
  // (Send or Stop, depending on whether a reply is running) and the queue rows.
  if (typeof syncSendButton === "function") syncSendButton();
  if (typeof queueRender === "function") queueRender();
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
  // Refresh dynamically-built bubbles: they carry no data-i18n, so re-label them
  // here. Only the inner .bubble-who span changes — overwriting .bubble-role would
  // destroy the timestamp beside it. Times are locale-formatted, so they refresh too.
  document.querySelectorAll("#msg-list .bubble").forEach(function (b) {
    if (b.classList.contains("tool-call")) {
      // Its label is plain text, so it is safe to rewrite in place.
      const r = b.querySelector(".bubble-role");
      const names = (b._toolCalls || []).map(function (tc) { return tc.function.name; }).join(", ");
      if (r) r.textContent = tr("tool.calling") + ": " + names;
      return;
    }
    const who = b.querySelector(".bubble-who");
    if (!who) return;
    who.textContent = b.classList.contains("user") ? tr("chat.you")
      : b.classList.contains("error") ? tr("chat.error") : tr("chat.assistant");
    const when = b.querySelector(".bubble-time");
    if (when && b._ts != null) {
      when.textContent = QinferTime.clock(b._ts);
      when.title = QinferTime.full(b._ts);
    }
  });
  // Sidebar group headings ("Today"/"今天") come from tr() at build time, so the
  // list needs a re-render; conversation titles are user content and stay as typed.
  QinferConv.renderList();
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

// No built-in tools ship with the client: the QinferTools registry is the generic
// tool_calls protocol (register/list/execute + tool cards). HTML/mermaid rendering,
// web fetch, and JS execution are agent concerns, not chat, and were removed
// (2026-10-15); register tools from devtools or future agent integrations.

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

/* ---- st-* confirm dialog (replaces native confirm(); prompt-style edits use
   inline inputs instead). Resolves true/false; Esc/overlay click = cancel. ---- */
function showConfirm(message, confirmLabel) {
  return new Promise(function (resolve) {
    var prev = document.querySelector(".st-modal");
    if (prev) prev.remove();
    var overlay = document.createElement("div");
    overlay.className = "st-modal";
    overlay.setAttribute("role", "dialog");
    overlay.setAttribute("aria-modal", "true");
    var card = document.createElement("div");
    card.className = "st-modal__card";
    var msg = document.createElement("p");
    msg.className = "st-modal__msg";
    msg.id = "st-modal-msg";
    overlay.setAttribute("aria-labelledby", msg.id);
    msg.textContent = message;
    var btns = document.createElement("div");
    btns.className = "st-modal__btns";
    var bCancel = document.createElement("button");
    bCancel.type = "button";
    bCancel.className = "st-btn st-btn--secondary";
    bCancel.textContent = tr("misc.cancel");
    var bOk = document.createElement("button");
    bOk.type = "button";
    bOk.className = "st-btn st-btn--primary";
    bOk.textContent = confirmLabel || tr("misc.confirm");
    btns.appendChild(bCancel);
    btns.appendChild(bOk);
    card.appendChild(msg);
    card.appendChild(btns);
    overlay.appendChild(card);
    document.body.appendChild(overlay);
    function close(v) {
      document.removeEventListener("keydown", onKey);
      overlay.remove();
      resolve(v);
    }
    function onKey(e) {
      if (e.key === "Escape") close(false);
      if (e.key === "Enter") close(true);
    }
    bCancel.addEventListener("click", function () { close(false); });
    bOk.addEventListener("click", function () { close(true); });
    overlay.addEventListener("click", function (e) { if (e.target === overlay) close(false); });
    document.addEventListener("keydown", onKey);
    bOk.focus();
  });
}

/* ---- Shortcut help dialog: renders the same table the handler binds, so the
   list can never drift from what the keys actually do. ---- */
function isMacKeys() {
  var nav = window.navigator || {};
  return /mac|iphone|ipad|ipod/i.test(nav.platform || nav.userAgent || "");
}
function showHelpDialog() {
  if (document.querySelector(".st-modal")) return;   // one modal at a time
  var overlay = document.createElement("div");
  overlay.className = "st-modal";
  overlay.setAttribute("role", "dialog");
  overlay.setAttribute("aria-modal", "true");
  var card = document.createElement("div");
  card.className = "st-modal__card keys-card";
  var title = document.createElement("div");
  title.className = "keys-card__title";
  title.innerHTML = '<svg class="st-icon st-icon--sm"><use href="/web/sprite.svg#i-keyboard"/></svg>';
  var h = document.createElement("span");
  h.id = "keys-dialog-title";
  h.textContent = tr("keys.title");
  title.appendChild(h);
  overlay.setAttribute("aria-labelledby", h.id);
  var dl = document.createElement("dl");
  dl.className = "keys-list";
  QinferKeys.rows(isMacKeys()).forEach(function (r) {
    var dt = document.createElement("dt");
    var kbd = document.createElement("kbd");
    kbd.className = "st-kbd";
    kbd.textContent = r.keys;
    dt.appendChild(kbd);
    var dd = document.createElement("dd");
    dd.textContent = tr(r.labelKey);
    dl.appendChild(dt);
    dl.appendChild(dd);
  });
  var btns = document.createElement("div");
  btns.className = "st-modal__btns";
  var bClose = document.createElement("button");
  bClose.type = "button";
  bClose.className = "st-btn st-btn--secondary";
  bClose.textContent = tr("misc.close");
  btns.appendChild(bClose);
  card.appendChild(title);
  card.appendChild(dl);
  card.appendChild(btns);
  overlay.appendChild(card);
  document.body.appendChild(overlay);
  function close() {
    document.removeEventListener("keydown", onKey);
    overlay.remove();
    bClose.blur();
  }
  function onKey(e) {
    if (e.key === "Escape") { e.preventDefault(); close(); }
  }
  bClose.addEventListener("click", close);
  overlay.addEventListener("click", function (e) { if (e.target === overlay) close(); });
  document.addEventListener("keydown", onKey);
  bClose.focus();
}

/* ---- Dismissible layers -----------------------------------------------------
   Escape walks these top-down before it reaches "stop generating". The thinking
   popover floats above the footer drawers, and within the drawers the one opened
   last is the one the user means. */
var drawerStack = [];
function trackDrawer(id) {
  const d = $(id);
  if (!d) return;
  d.addEventListener("toggle", function () {
    const i = drawerStack.indexOf(id);
    if (i >= 0) drawerStack.splice(i, 1);
    if (d.open) drawerStack.unshift(id);
  });
}
function setDrawer(id, open) {
  const d = $(id);
  if (d) d.open = open;
}
function overlayLayers() {
  const outline = $("outline-pop");
  // Topmost first: the outline popover is transient, then the drawers.
  const layers = [
    {name: "outline", open: !!outline && !outline.classList.contains("hidden")},
  ];
  drawerStack.forEach(function (id) {
    const d = $(id);
    layers.push({name: "drawer:" + id, open: !!d && d.open});
  });
  return layers;
}
function toggleConvSidebar() {
  const sb = $("conv-sidebar");
  if (sb) sb.classList.toggle("hidden");
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
  // Re-apply accent for the new theme (inline styles override CSS, so we must update them)
  if (typeof applyAccent === "function") {
    var s = loadSettings();
    applyAccent(s.accent);
  }
}
function toggleTheme() {
  setTheme(currentTheme() === "dark" ? "light" : "dark");
  showToast(currentTheme() === "dark" ? tr("theme.dark") : tr("theme.light"), "info");
}

/* ---- Settings: accent color presets and user preferences ---- */
var ACCENT_PRESETS = {
  qwen: {
    light: { accent:"#615CED", strong:"#4f45d6", hover:"#564ee2", ink:"#ffffff", tint:"#f1f0ff", tintHover:"#e4e2ff", text:"#4a3fd0", glow:"rgba(97,92,237,.30)" },
    dark:  { accent:"#8b88f7", strong:"#8b88f7", hover:"#a5a3fb", ink:"#16112e", tint:"rgba(139,135,245,.14)", tintHover:"rgba(139,135,245,.22)", text:"#b3b0fa", glow:"rgba(139,135,245,.30)" }
  },
  vibrant: {
    light: { accent:"#ff6b1a", strong:"#e05508", hover:"#f05e0e", ink:"#ffffff", tint:"#fff3eb", tintHover:"#ffe5d0", text:"#c2410c", glow:"rgba(255,107,26,.30)" },
    dark:  { accent:"#ff8c47", strong:"#ff8c47", hover:"#ffa668", ink:"#2b1200", tint:"rgba(255,140,71,.14)", tintHover:"rgba(255,140,71,.22)", text:"#ffbc8a", glow:"rgba(255,140,71,.30)" }
  },
  blue: {
    light: { accent:"#3b82f6", strong:"#2563eb", hover:"#2d72e8", ink:"#ffffff", tint:"#eff6ff", tintHover:"#dbeafe", text:"#1d4ed8", glow:"rgba(59,130,246,.30)" },
    dark:  { accent:"#60a5fa", strong:"#60a5fa", hover:"#93c5fd", ink:"#0c1a3d", tint:"rgba(96,165,250,.14)", tintHover:"rgba(96,165,250,.22)", text:"#93c5fd", glow:"rgba(96,165,250,.30)" }
  },
  purple: {
    light: { accent:"#8b5cf6", strong:"#7c3aed", hover:"#8350f0", ink:"#ffffff", tint:"#f5f3ff", tintHover:"#ede9fe", text:"#6d28d9", glow:"rgba(139,92,246,.30)" },
    dark:  { accent:"#a78bfa", strong:"#a78bfa", hover:"#c4b5fd", ink:"#1a0c3d", tint:"rgba(167,139,250,.14)", tintHover:"rgba(167,139,250,.22)", text:"#c4b5fd", glow:"rgba(167,139,250,.30)" }
  },
  green: {
    light: { accent:"#10b981", strong:"#059669", hover:"#0da876", ink:"#ffffff", tint:"#ecfdf5", tintHover:"#d1fae5", text:"#047857", glow:"rgba(16,185,129,.30)" },
    dark:  { accent:"#34d399", strong:"#34d399", hover:"#6ee7b7", ink:"#0c2b1a", tint:"rgba(52,211,153,.14)", tintHover:"rgba(52,211,153,.22)", text:"#6ee7b7", glow:"rgba(52,211,153,.30)" }
  },
  rose: {
    light: { accent:"#f43f5e", strong:"#e11d48", hover:"#ec2d54", ink:"#ffffff", tint:"#fff1f2", tintHover:"#ffe4e6", text:"#be123c", glow:"rgba(244,63,94,.30)" },
    dark:  { accent:"#fb7185", strong:"#fb7185", hover:"#fda4af", ink:"#3d0c15", tint:"rgba(251,113,133,.14)", tintHover:"rgba(251,113,133,.22)", text:"#fda4af", glow:"rgba(251,113,133,.30)" }
  },
  teal: {
    light: { accent:"#14b8a6", strong:"#0d9488", hover:"#11a899", ink:"#ffffff", tint:"#f0fdfa", tintHover:"#ccfbf1", text:"#0f766e", glow:"rgba(20,184,166,.30)" },
    dark:  { accent:"#2dd4bf", strong:"#2dd4bf", hover:"#5eead4", ink:"#0c2b26", tint:"rgba(45,212,191,.14)", tintHover:"rgba(45,212,191,.22)", text:"#5eead4", glow:"rgba(45,212,191,.30)" }
  },
  amber: {
    light: { accent:"#f59e0b", strong:"#d97706", hover:"#e8910a", ink:"#ffffff", tint:"#fffbeb", tintHover:"#fef3c7", text:"#b45309", glow:"rgba(245,158,11,.30)" },
    dark:  { accent:"#fbbf24", strong:"#fbbf24", hover:"#fcd34d", ink:"#2b1a0c", tint:"rgba(251,191,36,.14)", tintHover:"rgba(251,191,36,.22)", text:"#fcd34d", glow:"rgba(251,191,36,.30)" }
  },
  indigo: {
    light: { accent:"#6366f1", strong:"#4f46e5", hover:"#5958eb", ink:"#ffffff", tint:"#eef2ff", tintHover:"#e0e7ff", text:"#4338ca", glow:"rgba(99,102,241,.30)" },
    dark:  { accent:"#818cf8", strong:"#818cf8", hover:"#a5b4fc", ink:"#0c0c3d", tint:"rgba(129,140,248,.14)", tintHover:"rgba(129,140,248,.22)", text:"#a5b4fc", glow:"rgba(129,140,248,.30)" }
  },
  cyan: {
    light: { accent:"#06b6d4", strong:"#0891b2", hover:"#05a5c4", ink:"#ffffff", tint:"#ecfeff", tintHover:"#cffafe", text:"#0e7490", glow:"rgba(6,182,212,.30)" },
    dark:  { accent:"#22d3ee", strong:"#22d3ee", hover:"#67e8f9", ink:"#0c2b30", tint:"rgba(34,211,238,.14)", tintHover:"rgba(34,211,238,.22)", text:"#67e8f9", glow:"rgba(34,211,238,.30)" }
  }
};
var SETTINGS_DEFAULTS = {
  theme: "system",
  accent: "qwen",
  "sd-temp": "0.8", "sd-top-p": "0.95", "sd-top-k": "40", "sd-min-p": "0.05",
  "sd-pres-pen": "0", "sd-freq-pen": "0", "sd-rep-pen": "1",
  "sd-max-tokens": "2048", "sd-seed": "-1", "sd-ctx": "0",
  "sd-system-prompt": ""
};
var SETTINGS_KEYS = ["theme", "accent", "sd-temp", "sd-top-p", "sd-top-k", "sd-min-p",
  "sd-pres-pen", "sd-freq-pen", "sd-rep-pen", "sd-max-tokens", "sd-seed", "sd-ctx",
  "sd-system-prompt"];

/* ---- Generation-parameter profiles ----------------------------------------
   A profile is a named bundle of the numeric sampling parameters. The active
   profile is mirrored into the flat "sd-*" keys on every load, so every
   existing consumer (new-conversation defaults, apply-to-conversation, reset)
   keeps reading exactly the keys it always did. The profile list is the single
   source of truth; the flat keys are a derived view of the active profile. */
var PROFILE_PARAM_KEYS = ["sd-temp", "sd-top-p", "sd-top-k", "sd-min-p",
  "sd-pres-pen", "sd-freq-pen", "sd-rep-pen", "sd-max-tokens", "sd-seed", "sd-ctx"];

function profileDefaultName() {
  try { return tr("settings.profileDefault") || "Default"; } catch (e) { return "Default"; }
}
function newProfileId() {
  return "p_" + Date.now().toString(36) + Math.floor(Math.random() * 1296).toString(36);
}
function profileParamsFrom(s) {
  var p = {};
  PROFILE_PARAM_KEYS.forEach(function (k) { p[k] = (s[k] !== undefined) ? s[k] : SETTINGS_DEFAULTS[k]; });
  return p;
}
// Make sure a non-empty profile list exists, migrating the flat values on first run.
function ensureProfiles(s) {
  if (!Array.isArray(s.profiles) || s.profiles.length === 0) {
    s.profiles = [{ id: "p_default", name: profileDefaultName(), params: profileParamsFrom(s) }];
    s.activeProfile = s.profiles[0].id;
  }
  if (!s.profiles.some(function (p) { return p.id === s.activeProfile; })) {
    s.activeProfile = s.profiles[0].id;
  }
  return s.profiles;
}
function activeProfile(s) {
  ensureProfiles(s);
  return s.profiles.filter(function (p) { return p.id === s.activeProfile; })[0] || s.profiles[0];
}
// Copy the active profile's parameters onto the flat keys.
function mirrorActiveProfile(s) {
  var p = activeProfile(s);
  PROFILE_PARAM_KEYS.forEach(function (k) { if (p.params[k] !== undefined) s[k] = p.params[k]; });
}
// Write one parameter to the active profile and to the flat mirror, then persist.
function setProfileParam(s, key, value) {
  activeProfile(s).params[key] = value;
  s[key] = value;
  saveSettings(s);
}

function loadSettings() {
  var s = {};
  try { var raw = localStorage.getItem("qinfer.settings"); if (raw) s = JSON.parse(raw); } catch (e) {}
  SETTINGS_KEYS.forEach(function (k) { if (s[k] === undefined) s[k] = SETTINGS_DEFAULTS[k]; });
  // One-time migration: the old default accent ("vibrant" orange) is force-replaced
  // by the new default (qwen purple). The marker stops a later, deliberate re-pick
  // of orange from being clobbered on every load.
  if (s.accent === "vibrant" && !s.accentDefaultMigrated) {
    s.accent = "qwen";
    s.accentDefaultMigrated = true;
    try { localStorage.setItem("qinfer.settings", JSON.stringify(s)); } catch (e) {}
  }
  // Profiles: create the first one from the flat values on upgrade, then mirror
  // the active profile back onto the flat keys.
  var hadProfiles = Array.isArray(s.profiles) && s.profiles.length > 0;
  ensureProfiles(s);
  mirrorActiveProfile(s);
  if (!hadProfiles) { try { localStorage.setItem("qinfer.settings", JSON.stringify(s)); } catch (e) {} }
  return s;
}
function saveSettings(s) {
  try { localStorage.setItem("qinfer.settings", JSON.stringify(s)); } catch (e) {}
}
function applyAccent(name) {
  var theme = currentTheme();
  var c;
  if (name === "custom") {
    // Generate a full accent palette from a single hex color
    var hex = loadSettings()["accent-custom"] || "#615CED";
    c = generateCustomPalette(hex, theme);
  } else {
    var preset = ACCENT_PRESETS[name] || ACCENT_PRESETS.qwen;
    c = preset[theme] || preset.light;
  }
  var root = document.documentElement.style;
  root.setProperty("--st-accent", c.accent);
  root.setProperty("--st-accent-strong", c.strong);
  root.setProperty("--st-accent-hover", c.hover);
  root.setProperty("--st-accent-ink", c.ink);
  root.setProperty("--st-accent-tint", c.tint);
  root.setProperty("--st-accent-tint-hover", c.tintHover);
  root.setProperty("--st-accent-text", c.text);
  root.setProperty("--st-accent-glow", c.glow);
}
// Generate a full accent palette from a single hex color for light or dark theme
function generateCustomPalette(hex, theme) {
  var r = parseInt(hex.slice(1,3),16), g = parseInt(hex.slice(3,5),16), b = parseInt(hex.slice(5,7),16);
  if (theme === "dark") {
    // Lighten for dark bg
    var lr = Math.min(255, r + 60), lg = Math.min(255, g + 60), lb = Math.min(255, b + 60);
    var accent = rgbToHex(lr, lg, lb);
    return {
      accent: accent, strong: accent,
      hover: rgbToHex(Math.min(255, lr+30), Math.min(255, lg+30), Math.min(255, lb+30)),
      ink: "#1a1a1a",
      tint: "rgba("+r+","+g+","+b+",.14)",
      tintHover: "rgba("+r+","+g+","+b+",.22)",
      text: rgbToHex(Math.min(255, lr+40), Math.min(255, lg+40), Math.min(255, lb+40)),
      glow: "rgba("+r+","+g+","+b+",.30)"
    };
  }
  // Darken for light bg
  var dr = Math.max(0, r - 30), dg = Math.max(0, g - 30), db = Math.max(0, b - 30);
  return {
    accent: hex,
    strong: rgbToHex(dr, dg, db),
    hover: rgbToHex(Math.max(0, r-15), Math.max(0, g-15), Math.max(0, b-15)),
    ink: "#ffffff",
    tint: "rgba("+Math.min(255,r+100)+","+Math.min(255,g+100)+","+Math.min(255,b+100)+",.15)",
    tintHover: "rgba("+Math.min(255,r+80)+","+Math.min(255,g+80)+","+Math.min(255,b+80)+",.25)",
    text: rgbToHex(Math.max(0, r-60), Math.max(0, g-60), Math.max(0, b-60)),
    glow: "rgba("+r+","+g+","+b+",.30)"
  };
}
function rgbToHex(r, g, b) {
  return "#" + [r,g,b].map(function(x) { return Math.round(x).toString(16).padStart(2,"0"); }).join("");
}
function applyThemeFromSettings(setting) {
  var t = setting;
  if (setting === "system") {
    t = matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  }
  setTheme(t);
  // Re-apply accent for the new theme
  var s = loadSettings();
  applyAccent(s.accent);
}
function syncSettingsUI(s) {
  // Theme chips
  document.querySelectorAll("#setting-theme .st-chip").forEach(function (c) {
    c.classList.toggle("active", c.dataset.value === s.theme);
    c.setAttribute("aria-checked", String(c.dataset.value === s.theme));
  });
  // Accent chips (preset buttons only, not the custom label)
  document.querySelectorAll("#setting-accent .st-chip[data-value]").forEach(function (c) {
    c.classList.toggle("active", c.dataset.value === s.accent);
    c.setAttribute("aria-checked", String(c.dataset.value === s.accent));
  });
  // Custom accent swatch
  var customSwatch = $("custom-accent-swatch");
  var customPicker = $("custom-accent-picker");
  if (s.accent === "custom" && s["accent-custom"] && customSwatch && customPicker) {
    customSwatch.style.background = s["accent-custom"];
    customPicker.value = s["accent-custom"];
  }
  // Defaults sliders/inputs
  SETTINGS_KEYS.forEach(function (k) {
    if (k === "theme" || k === "accent" || k === "accent-custom") return;
    var el = $(k);
    if (el) el.value = s[k];
  });
  syncDefaultsGrid(s);
}
// Push a parameter value into both its slider and its number twin.
function setDefaultsField(k, v) {
  var el = $(k), num = $(k + "-num");
  if (el) el.value = v;
  if (num) num.value = v;
}
function syncDefaultsGrid(s) {
  PROFILE_PARAM_KEYS.forEach(function (k) {
    setDefaultsField(k, (s[k] !== undefined) ? s[k] : SETTINGS_DEFAULTS[k]);
  });
  syncSettingsOutputs();
}
function syncSettingsOutputs() {
  var SLIDER_IDS = ["sd-temp", "sd-top-p", "sd-min-p", "sd-pres-pen", "sd-freq-pen", "sd-rep-pen"];
  SLIDER_IDS.forEach(function (id) {
    var el = $(id), out = $(id + "-out");
    if (el && out) {
      var digits = (id === "sd-min-p" || id === "sd-pres-pen" || id === "sd-freq-pen") ? 2 : 2;
      out.textContent = Number(el.value).toFixed(digits);
    }
  });
}
function initSettings() {
  var s = loadSettings();
  // Apply theme
  applyThemeFromSettings(s.theme);
  // Apply accent
  applyAccent(s.accent);
  // Sync UI
  syncSettingsUI(s);

  // Theme chip clicks
  document.querySelectorAll("#setting-theme .st-chip").forEach(function (c) {
    c.addEventListener("click", function () {
      var s = loadSettings();
      s.theme = c.dataset.value;
      saveSettings(s);
      applyThemeFromSettings(s.theme);
      syncSettingsUI(s);
      showToast(tr("settings.saved"), "success");
    });
  });
  // Accent chip clicks (preset buttons only)
  document.querySelectorAll("#setting-accent .st-chip[data-value]").forEach(function (c) {
    c.addEventListener("click", function () {
      var s = loadSettings();
      s.accent = c.dataset.value;
      saveSettings(s);
      applyAccent(s.accent);
      syncSettingsUI(s);
      showToast(tr("settings.saved"), "success");
    });
  });
  // Custom color picker
  var customPicker = $("custom-accent-picker");
  if (customPicker) {
    customPicker.addEventListener("input", function () {
      var s = loadSettings();
      s.accent = "custom";
      s["accent-custom"] = customPicker.value;
      saveSettings(s);
      applyAccent("custom");
      var swatch = $("custom-accent-swatch");
      if (swatch) swatch.style.background = customPicker.value;
      syncSettingsUI(s);
    });
    customPicker.addEventListener("click", function (e) {
      e.stopPropagation();
    });
  }
  // Defaults: parameter edits land on the active profile (and the flat mirror).
  // seed and ctx have no slider twin but persist through the same path.
  PROFILE_PARAM_KEYS.forEach(function (k) {
    var el = $(k), num = $(k + "-num");
    if (el) el.addEventListener("input", function () {
      var s = loadSettings();
      setProfileParam(s, k, el.value);
      if (num) num.value = el.value;
    });
    if (num) num.addEventListener("input", function () {
      var v = clampToRange(el, num.value);
      if (v === null) return;
      var s = loadSettings();
      setProfileParam(s, k, String(v));
      if (el) el.value = v;
    });
  });
  // The default system prompt stays a single global value, not per profile.
  var sysDef = $("sd-system-prompt");
  if (sysDef) sysDef.addEventListener("input", function () {
    var s = loadSettings();
    s["sd-system-prompt"] = sysDef.value;
    saveSettings(s);
  });
  initProfiles();
  // Apply to current conversation
  var btnApply = $("btn-apply-defaults");
  if (btnApply) {
    btnApply.addEventListener("click", function () {
      var s = loadSettings();
      // Map settings keys to sampler field IDs
      var mapping = { "sd-temp":"s-temp", "sd-top-p":"s-top-p", "sd-top-k":"s-top-k", "sd-min-p":"s-min-p",
        "sd-pres-pen":"s-pres-pen", "sd-freq-pen":"s-freq-pen", "sd-rep-pen":"s-rep-pen",
        "sd-max-tokens":"s-max-tokens", "sd-seed":"s-seed", "sd-ctx":"s-ctx" };
      Object.keys(mapping).forEach(function (sk) {
        var target = $(mapping[sk]);
        if (target) target.value = s[sk];
      });
      // Also apply system prompt default
      if (s["sd-system-prompt"]) {
        var sysEl = $("sys-prompt");
        if (sysEl) sysEl.value = s["sd-system-prompt"];
      }
      syncSliderOutputs();
      saveSampler();
      showToast(tr("settings.defaultsApplied"), "success");
    });
  }
  // Reset the active profile back to factory parameters
  var btnReset = $("btn-reset-defaults");
  if (btnReset) {
    btnReset.addEventListener("click", function () {
      var s = loadSettings();
      activeProfile(s).params = profileParamsFrom(SETTINGS_DEFAULTS);
      mirrorActiveProfile(s);
      saveSettings(s);
      syncSettingsUI(s);
      showToast(tr("settings.defaultsReset"), "success");
    });
  }
  // Listen for system theme changes when "system" is selected
  try {
    matchMedia("(prefers-color-scheme: dark)").addEventListener("change", function () {
      var s = loadSettings();
      if (s.theme === "system") {
        applyThemeFromSettings("system");
      }
    });
  } catch (e) {}

  // --- Inline panel toggles (sampler & system prompt) ---
  initInlinePanels();
  // --- Slider ↔ number input sync ---
  initSliderNumberSync();
  // --- Empty-state starter chips fill the composer ---
  initEmptyChips();
}

function initInlinePanels() {
  var btnSampler = $("btn-toggle-sampler");
  var panelSampler = $("sampler-panel");
  var btnSys = $("btn-toggle-sys");
  var panelSys = $("sys-panel");
  var btnCloseSampler = $("btn-close-sampler");
  var btnCloseSys = $("btn-close-sys");

  function togglePanel(btn, panel) {
    if (!panel || !btn) return;
    var isHidden = panel.classList.contains("hidden");
    panel.classList.toggle("hidden", !isHidden);
    btn.classList.toggle("active", isHidden);
  }

  if (btnSampler) btnSampler.addEventListener("click", function () { togglePanel(btnSampler, panelSampler); });
  if (btnSys) btnSys.addEventListener("click", function () { togglePanel(btnSys, panelSys); });
  if (btnCloseSampler) btnCloseSampler.addEventListener("click", function () {
    if (panelSampler) panelSampler.classList.add("hidden");
    if (btnSampler) btnSampler.classList.remove("active");
  });
  if (btnCloseSys) btnCloseSys.addEventListener("click", function () {
    if (panelSys) panelSys.classList.add("hidden");
    if (btnSys) btnSys.classList.remove("active");
  });
}

// Clamp a raw input to the range of a paired slider; null when not numeric.
function clampToRange(rangeEl, raw) {
  var v = parseFloat(raw);
  if (isNaN(v)) return null;
  if (!rangeEl) return v;
  var mn = parseFloat(rangeEl.min), mx = parseFloat(rangeEl.max);
  if (!isNaN(mn) && v < mn) v = mn;
  if (!isNaN(mx) && v > mx) v = mx;
  return v;
}
function initSliderNumberSync() {
  // Pair each slider with its number twin: dragging fills the number, typing
  // clamps into the slider. Shared by the chat sampler and the settings profiles.
  var pairs = [
    ["s-temp", "s-temp-num"], ["s-top-p", "s-top-p-num"], ["s-min-p", "s-min-p-num"],
    ["s-pres-pen", "s-pres-pen-num"], ["s-freq-pen", "s-freq-pen-num"], ["s-rep-pen", "s-rep-pen-num"],
    ["sd-temp", "sd-temp-num"], ["sd-top-p", "sd-top-p-num"], ["sd-top-k", "sd-top-k-num"],
    ["sd-min-p", "sd-min-p-num"], ["sd-pres-pen", "sd-pres-pen-num"], ["sd-freq-pen", "sd-freq-pen-num"],
    ["sd-rep-pen", "sd-rep-pen-num"], ["sd-max-tokens", "sd-max-tokens-num"]
  ];
  pairs.forEach(function (pair) {
    var slider = $(pair[0]), num = $(pair[1]);
    if (!slider || !num) return;
    slider.addEventListener("input", function () { num.value = slider.value; });
    num.addEventListener("input", function () {
      var v = clampToRange(slider, num.value);
      if (v !== null) slider.value = v;
    });
  });
}

// Empty-state starter chips: clicking one drops its label into the composer,
// focuses the input and parks the caret at the end. The label is read at click
// time, so it follows whichever language is active.
function initEmptyChips() {
  var wrap = $("chat-empty-chips");
  var input = $("chat-input");
  if (!wrap || !input) return;
  wrap.addEventListener("click", function (e) {
    var btn = e.target.closest(".chat-empty__chip");
    if (!btn) return;
    var text = (btn.textContent || "").trim();
    if (!text) return;
    input.value = text;
    input.dispatchEvent(new Event("input", { bubbles: true }));
    input.focus();
    try { input.setSelectionRange(text.length, text.length); } catch (err) {}
  });
}

// ---- Generation-parameter profiles: the select + add/duplicate/rename/delete --
function renderProfileSelect(s) {
  var sel = $("profile-select");
  if (!sel) return;
  sel.innerHTML = "";
  s.profiles.forEach(function (p) {
    var o = document.createElement("option");
    o.value = p.id;
    o.textContent = p.name;
    if (p.id === s.activeProfile) o.selected = true;
    sel.appendChild(o);
  });
  var del = $("btn-profile-del");
  if (del) del.disabled = s.profiles.length <= 1;
}
function switchActiveProfile(id) {
  var s = loadSettings();
  s.activeProfile = id;
  mirrorActiveProfile(s);
  saveSettings(s);
  renderProfileSelect(s);
  syncSettingsUI(s);
}
function initProfiles() {
  var sel = $("profile-select");
  if (!sel) return;
  var s0 = loadSettings();
  ensureProfiles(s0);
  renderProfileSelect(s0);
  syncDefaultsGrid(s0);

  sel.addEventListener("change", function () { switchActiveProfile(sel.value); });

  var mk = $("btn-profile-new");
  if (mk) mk.addEventListener("click", function () {
    var s = loadSettings();
    s.profiles.push({ id: newProfileId(), name: tr("settings.profileName") + " " + (s.profiles.length + 1), params: profileParamsFrom(SETTINGS_DEFAULTS) });
    s.activeProfile = s.profiles[s.profiles.length - 1].id;
    mirrorActiveProfile(s);
    saveSettings(s);
    renderProfileSelect(s);
    syncSettingsUI(s);
    showToast(tr("settings.profileAdded"), "success");
  });
  var dup = $("btn-profile-dup");
  if (dup) dup.addEventListener("click", function () {
    var s = loadSettings();
    var cur = activeProfile(s);
    s.profiles.push({ id: newProfileId(), name: cur.name + " " + tr("settings.profileCopySuffix"), params: JSON.parse(JSON.stringify(cur.params)) });
    s.activeProfile = s.profiles[s.profiles.length - 1].id;
    mirrorActiveProfile(s);
    saveSettings(s);
    renderProfileSelect(s);
    syncSettingsUI(s);
    showToast(tr("settings.profileDuplicated"), "success");
  });
  var ren = $("btn-profile-rename");
  if (ren) ren.addEventListener("click", function () {
    var s = loadSettings();
    var cur = activeProfile(s);
    var name = window.prompt(tr("settings.profileRenamePrompt"), cur.name);
    if (name === null) return;
    name = name.trim();
    if (!name) return;
    cur.name = name;
    saveSettings(s);
    renderProfileSelect(s);
  });
  var del = $("btn-profile-del");
  if (del) del.addEventListener("click", function () {
    var s = loadSettings();
    if (s.profiles.length <= 1) return;
    var cur = activeProfile(s);
    if (!window.confirm(tr("settings.profileDeleteConfirm").replace("{name}", cur.name))) return;
    s.profiles = s.profiles.filter(function (p) { return p.id !== cur.id; });
    if (s.activeProfile === cur.id) s.activeProfile = s.profiles[0].id;
    mirrorActiveProfile(s);
    saveSettings(s);
    renderProfileSelect(s);
    syncSettingsUI(s);
    showToast(tr("settings.profileDeleted"), "success");
  });
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
  if (name === "settings") { var s = loadSettings(); syncSettingsUI(s); }
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

/* ---- Sampler: presets, persistence, and what the current conversation overrides ----
   Presets are the three starting points the drawer offers; "custom" is whatever the
   sliders say after an edit. Settings live in localStorage (shared across tabs, like
   theme/lang); the old per-tab sessionStorage key is migrated once on first load. */
var QinferSampler = {
  PRESETS: {
    precise: { temperature: 0.2, top_p: 0.9, top_k: 20, min_p: 0.1, presence_penalty: 0, frequency_penalty: 0, repeat_penalty: 1 },
    balanced: { temperature: 0.8, top_p: 0.95, top_k: 40, min_p: 0.05, presence_penalty: 0, frequency_penalty: 0, repeat_penalty: 1 },
    creative: { temperature: 1.2, top_p: 0.98, top_k: 80, min_p: 0.02, presence_penalty: 0, frequency_penalty: 0, repeat_penalty: 1 },
  },
  STORAGE_KEY: "qinfer.sampler",
  LEGACY_KEY: "qinfer_sampler",
  // Name of the preset these sampling fields equal, else "custom".
  matchPreset: function(fields) {
    for (var name in this.PRESETS) {
      var p = this.PRESETS[name], same = true;
      for (var k in p) { if (Number(fields[k]) !== Number(p[k])) { same = false; break; } }
      if (same) return name;
    }
    return "custom";
  },
  // Saved settings; a pre-existing per-tab sessionStorage value is migrated once.
  load: function() {
    try {
      var raw = localStorage.getItem(this.STORAGE_KEY);
      if (!raw) {
        raw = sessionStorage.getItem(this.LEGACY_KEY);
        if (raw) {
          localStorage.setItem(this.STORAGE_KEY, raw);
          sessionStorage.removeItem(this.LEGACY_KEY);
        }
      }
      return raw ? JSON.parse(raw) : null;
    } catch (e) { return null; }
  },
  save: function(obj) {
    try { localStorage.setItem(this.STORAGE_KEY, JSON.stringify(obj)); } catch (e) {}
  },
};

/* ---- Sampler: /props defaults, saved value wins when present ---- */
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
  s.think = thinkLevel;
  if ($("s-ctx")) s.ctx = $("s-ctx").value;
  QinferSampler.save(s);
  syncPresetChips();
  // A conversation on an override follows its own edits; an edit that brings the
  // drawer level with the global set drops the badge.
  var cur = QinferConv.currentId ? QinferConv.loadAll().find(function (c) { return c.id === QinferConv.currentId; }) : null;
  if (cur && cur.params) { cur.params = samplerSnapshot(); QinferConv.saveAll(QinferConv.loadAll().map(function (c) { return c.id === cur.id ? cur : c; })); }
  if (typeof syncOverrideSwitch === "function") syncOverrideSwitch();
}
function loadSampler() {
  const s = QinferSampler.load();
  if (!s) return false;
  if (s.temperature !== undefined) $("s-temp").value = s.temperature;
  if (s.top_p !== undefined) $("s-top-p").value = s.top_p;
  if (s.top_k !== undefined) $("s-top-k").value = s.top_k;
  if (s.min_p !== undefined) $("s-min-p").value = s.min_p;
  if (s.presence_penalty !== undefined) $("s-pres-pen").value = s.presence_penalty;
  if (s.frequency_penalty !== undefined) $("s-freq-pen").value = s.frequency_penalty;
  if (s.repeat_penalty !== undefined) $("s-rep-pen").value = s.repeat_penalty;
  if (s.seed !== undefined) $("s-seed").value = s.seed;
  if (s.max_tokens !== undefined) $("s-max-tokens").value = s.max_tokens;
  if (s.think !== undefined) setThinkLevel(s.think, { silent: true });
  if (s.ctx !== undefined && $("s-ctx")) $("s-ctx").value = s.ctx;
  return !!$("s-temp");
}

/* ---- Thinking level: the composer chip + popover is the only control ----
   The drawer's four-level select was removed (2026-10-15): one source of truth,
   and the chip shows the current level instead of a bare on/off highlight. */
var THINK_LEVELS = ["off", "low", "medium", "xhigh"];
var thinkLevel = "xhigh";
function setThinkLevel(level, opts) {
  if (THINK_LEVELS.indexOf(level) < 0) level = "xhigh";
  thinkLevel = level;
  // The segments are the control, so the state lives on them: aria-checked for
  // assistive tech, .active for the eye.
  var seg = $("think-seg");
  if (seg) {
    seg.querySelectorAll(".think-seg__opt").forEach(function (b) {
      var on = b.dataset.level === level;
      b.classList.toggle("active", on);
      b.setAttribute("aria-checked", on ? "true" : "false");
      b.setAttribute("tabindex", on ? "0" : "-1");
    });
    seg.classList.toggle("is-off", level === "off");
  }
  if (!opts || !opts.silent) saveSampler();
}
// The sampler + generation fields as a plain object, for per-conversation overrides.
function samplerSnapshot() {
  var s = {};
  SAMPLER_FIELDS.forEach(function (id) {
    if ($(id)) s[id] = $(id).value;
  });
  s["s-think"] = thinkLevel;
  return s;
}
// Apply a saved override set to the fields (no-op for a missing/empty set).
function samplerApply(s) {
  if (!s) return;
  Object.keys(s).forEach(function (id) {
    if (id === "s-think") { setThinkLevel(s[id], { silent: true }); return; }
    if ($(id)) $(id).value = s[id];
  });
  syncSliderOutputs();
  syncPresetChips();
}

/* ---- Sampler drawer helpers: field tables, slider readouts, presets ----
   Field ids are the same ones the snapshot/apply pair and the request body use,
   so a saved conversation keeps working across this redesign. */
var SAMPLER_FIELDS = ["s-temp", "s-top-p", "s-top-k", "s-min-p", "s-pres-pen", "s-freq-pen", "s-rep-pen", "s-seed", "s-max-tokens", "s-ctx"];
var SLIDER_FIELDS = ["s-temp", "s-top-p", "s-min-p", "s-pres-pen", "s-freq-pen", "s-rep-pen"];
// Field id -> param name for preset matching.
var SAMPLER_PARAM_NAMES = {
  "s-temp": "temperature", "s-top-p": "top_p", "s-top-k": "top_k", "s-min-p": "min_p",
  "s-pres-pen": "presence_penalty", "s-freq-pen": "frequency_penalty", "s-rep-pen": "repeat_penalty",
};
var samplerDefaults = null;

function captureSamplerDefaults() {
  var d = {};
  SAMPLER_FIELDS.forEach(function (id) { if ($(id)) d[id] = $(id).value; });
  d["s-think"] = THINK_LEVELS.indexOf(thinkLevel) >= 0 ? thinkLevel : "xhigh";
  // Override with user settings defaults if available
  if (typeof loadSettings === "function") {
    var s = loadSettings();
    var mapping = { "sd-temp":"s-temp", "sd-top-p":"s-top-p", "sd-top-k":"s-top-k", "sd-min-p":"s-min-p",
      "sd-pres-pen":"s-pres-pen", "sd-freq-pen":"s-freq-pen", "sd-rep-pen":"s-rep-pen",
      "sd-max-tokens":"s-max-tokens", "sd-seed":"s-seed", "sd-ctx":"s-ctx" };
    Object.keys(mapping).forEach(function (sk) {
      var target = mapping[sk];
      if (s[sk] !== undefined) d[target] = s[sk];
    });
  }
  samplerDefaults = d;
}
// Decimals for the little readout beside a slider.
function fmtSlider(id, v) {
  var digits = (id === "s-min-p" || id === "s-temp" || id === "s-top-p" || id === "s-rep-pen") ? 2 : 1;
  return Number(v).toFixed(digits);
}
function syncSliderOutputs() {
  SLIDER_FIELDS.forEach(function (id) {
    var el = $(id), out = $(id + "-out");
    if (el && out) out.textContent = fmtSlider(id, el.value);
  });
}
// Light the chip of the preset these fields equal, else the Custom chip.
function syncPresetChips() {
  var wrap = $("preset-chips");
  if (!wrap) return;
  var fields = {};
  Object.keys(SAMPLER_PARAM_NAMES).forEach(function (id) {
    if ($(id)) fields[SAMPLER_PARAM_NAMES[id]] = Number($(id).value);
  });
  var name = QinferSampler.matchPreset(fields);
  wrap.querySelectorAll("[data-preset]").forEach(function (b) {
    b.classList.toggle("active", b.dataset.preset === name);
  });
  var custom = $("preset-custom");
  if (custom) custom.classList.toggle("active", name === "custom");
}
function applyPreset(name) {
  var p = QinferSampler.PRESETS[name];
  if (!p) return;
  Object.keys(SAMPLER_PARAM_NAMES).forEach(function (id) {
    if ($(id) && p[SAMPLER_PARAM_NAMES[id]] !== undefined) $(id).value = p[SAMPLER_PARAM_NAMES[id]];
  });
  syncSliderOutputs();
  saveSampler();
}
function resetSampler() {
  if (!samplerDefaults) captureSamplerDefaults();
  Object.keys(samplerDefaults).forEach(function (id) {
    if (id === "s-think") { setThinkLevel(samplerDefaults[id], { silent: true }); return; }
    if ($(id)) $(id).value = samplerDefaults[id];
  });
  syncSliderOutputs();
  saveSampler();
  showToast(tr("sampler.resetDone"));
}
// An explicit switch replaces the old diff-based badge: "on" means this
// conversation carries its own parameter set, and that is a state the reader
// chooses — not something they have to infer from a difference.
function convHasOverride() {
  var conv = QinferConv.currentId
    ? QinferConv.loadAll().find(function (c) { return c.id === QinferConv.currentId; }) : null;
  return !!(conv && conv.params);
}
function syncOverrideSwitch() {
  var el = $("s-override");
  if (el) el.checked = convHasOverride();
}
// Turning the switch on snapshots what the fields hold now; turning it off drops
// the conversation's set and puts the global one back in the fields.
function setConvOverride(on) {
  if (!QinferConv.currentId) return;
  var convs = QinferConv.loadAll();
  var conv = convs.find(function (c) { return c.id === QinferConv.currentId; });
  if (!conv) return;
  if (on) {
    conv.params = samplerSnapshot();
  } else {
    delete conv.params;
  }
  QinferConv.saveAll(convs);
  if (!on) {
    loadSampler();
    syncSliderOutputs();
    syncPresetChips();
  }
  syncOverrideSwitch();
  if (on) showToast(tr("sampler.overrideKept"));
  else showToast(tr("sampler.overrideCleared"));
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
    const p = (d.default_generation_settings || {}).params || {};
    // The server's defaults are what "Reset to defaults" restores.
    if (samplerDefaults) {
      if (p.temperature !== undefined) samplerDefaults["s-temp"] = p.temperature;
      if (p.top_p !== undefined) samplerDefaults["s-top-p"] = p.top_p;
      if (p.seed !== undefined) samplerDefaults["s-seed"] = p.seed;
    }
    if (!loadSampler()) {
      if (p.temperature !== undefined) $("s-temp").value = p.temperature;
      if (p.top_p !== undefined) $("s-top-p").value = p.top_p;
      if (p.seed !== undefined) $("s-seed").value = p.seed;
      syncSliderOutputs();
      syncPresetChips();
    }
  } catch (e) { /* fall back to defaults */ }
}

function hideEmptyPrompt() {
  const e = $("chat-empty");
  if (e) e.style.display = "none";
}

function addMsg(role, text, attachList, ts) {
  hideEmptyPrompt();
  const el = document.createElement("div");
  el.className = "bubble " + role;
  el._ts = ts == null ? Date.now() : ts;
  const label = document.createElement("div");
  label.className = "bubble-role";
  const name = document.createElement("span");
  name.className = "bubble-who";
  name.textContent = role === "user" ? tr("chat.you") : role === "error" ? tr("chat.error") : tr("chat.assistant");
  const when = document.createElement("span");
  when.className = "bubble-time";
  when.textContent = QinferTime.clock(el._ts);
  when.title = QinferTime.full(el._ts);
  label.appendChild(name);
  label.appendChild(when);
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
      '<button class="bubble-action" data-action="thumbup" title="' + tr("chat.good") + '" aria-label="' + tr("chat.good") + '"><svg class="st-icon st-icon--sm"><use href="/web/sprite.svg#i-thumb-up"/></svg></button>' +
      '<button class="bubble-action" data-action="thumbdown" title="' + tr("chat.bad") + '" aria-label="' + tr("chat.bad") + '"><svg class="st-icon st-icon--sm"><use href="/web/sprite.svg#i-thumb-down"/></svg></button>';
    el.appendChild(actions);
  } else if (role === "user") {
    var uactions = document.createElement("div");
    uactions.className = "bubble-actions";
    uactions.innerHTML = '<button class="bubble-action" data-action="copy" title="' + tr("chat.copy") + '" aria-label="' + tr("chat.copy") + '"><svg class="st-icon st-icon--sm"><use href="/web/sprite.svg#i-copy"/></svg></button>' +
      '<button class="bubble-action" data-action="edit" title="' + tr("chat.edit") + '" aria-label="' + tr("chat.edit") + '"><svg class="st-icon st-icon--sm"><use href="/web/sprite.svg#i-edit"/></svg></button>' +
      '<button class="bubble-action" data-action="branch" title="' + tr("conv.branch") + '" aria-label="' + tr("conv.branch") + '"><svg class="st-icon st-icon--sm"><use href="/web/sprite.svg#i-branch"/></svg></button>';
    el.appendChild(uactions);
  }
  $("msg-list").appendChild(el);
  onMessageArrived();
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
  onMessageArrived();
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
  // Tool results are always shown as text; the old HTML-iframe and mermaid
  // render branches were agent features and were removed (2026-10-15).
  var resultDiv = document.createElement("div");
  resultDiv.className = "tool-card__result";
  var pre = document.createElement("pre");
  pre.className = "tool-card__output";
  pre.textContent = result.length > 2000 ? result.slice(0, 2000) + "\n…" + tr("req.truncated") : result;
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

// The transcript turn a rendered bubble stands for, or null when it stands for
// nothing the model should ever see. Two kinds of bubble are not turns: an error
// line, which reports a failed request rather than something the model said
// (saving it would put words in its mouth on the next send), and an empty shell
// left over from a reply that produced nothing.
// Both the request builder and the sidebar save read this one shape, so the
// history the UI shows and the history that gets sent cannot drift apart.
function msgFromBubble(b, fallbackTs) {
  if (b.classList.contains("error")) return null;
  var isUser = b.classList.contains("user");
  var bodyEl = b.querySelector(".bubble-body");
  // Raw text first: a rendered Markdown body loses the source the model must see.
  var text = b._rawText || (bodyEl ? bodyEl.textContent : "");
  var list = [];
  if (isUser) {
    (b._attachments || []).forEach(function (a) {
      if (a.kind === "image" && a.url) list.push({ kind: "image", name: a.name, url: a.url });
      else if (a.kind === "file" && a.text) list.push({ kind: "file", name: a.name, text: a.text });
    });
  }
  if (!text && !list.length) return null;
  var msg = { role: isUser ? "user" : "assistant", content: text, ts: b._ts || fallbackTs };
  if (list.length) msg.attachments = list;
  if (!isUser && b._thinkingText) msg.thinking = b._thinkingText;
  return msg;
}

// localStorage caps out around 5 MB and an image attachment is a base64 data URL,
// so saving a conversation can fail outright. This is what is kept in that case:
// the whole transcript, minus the image payloads.
function withoutImages(convs) {
  return convs.map(function (c) {
    var copy = Object.assign({}, c);
    copy.messages = (c.messages || []).map(function (m) {
      if (!m.attachments) return m;
      var keep = m.attachments.filter(function (a) { return a.kind !== "image"; });
      var t = Object.assign({}, m);
      if (keep.length) t.attachments = keep;
      else delete t.attachments;
      return t;
    });
    return copy;
  });
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
    const turn = msgFromBubble(b, null);
    if (!turn) return;
    const attachList = turn.attachments || [];
    const imageAttachments = attachList.filter(function (a) { return a.kind === "image"; });
    const fileAttachments = attachList.filter(function (a) { return a.kind === "file"; });
    if (imageAttachments.length) {
      var parts = [];
      var fullText = turn.content;
      fileAttachments.forEach(function (f) {
        fullText = "File: " + f.name + "\n```\n" + f.text + "\n```\n\n" + fullText;
      });
      if (fullText) parts.push({ type: "text", text: fullText });
      imageAttachments.forEach(function (a) {
        parts.push({ type: "image_url", image_url: { url: a.url } });
      });
      messages.push({ role: turn.role, content: parts });
    } else {
      var content = turn.content;
      fileAttachments.forEach(function (f) {
        content = "File: " + f.name + "\n```\n" + f.text + "\n```\n\n" + content;
      });
      messages.push({ role: turn.role, content: content });
    }
  });
  return messages;
}

// One button owns the composer's primary slot: Send when idle, Stop while a reply
// runs. Two separate buttons for one decision made the stop control easy to miss
// and easy to leave behind after the run ended.
function syncSendButton() {
  var send = $("btn-send");
  if (!send) return;
  send.disabled = false;              // while streaming it is the stop control
  send.classList.toggle("st-btn--stop", !!streaming);
  send.dataset.mode = streaming ? "stop" : "send";
  send.setAttribute("aria-label", streaming ? tr("chat.stop") : tr("chat.send"));
  var use = send.querySelector("use");
  if (use) use.setAttribute("href", "/web/sprite.svg#" + (streaming ? "i-stop" : "i-send"));
}

// Swap an action's glyph for a moment to confirm the click landed.
function flashAction(btn, icon) {
  if (!btn || btn._flashing) return;
  var use = btn.querySelector("use");
  var prev = use ? use.getAttribute("href") : null;
  btn._flashing = true;
  btn.classList.add("active");
  if (use) use.setAttribute("href", "/web/sprite.svg#" + icon);
  setTimeout(function () {
    btn.classList.remove("active");
    if (use && prev) use.setAttribute("href", prev);
    btn._flashing = false;
  }, 1200);
}

function setStreaming(on) {
  streaming = on;
  syncSendButton();
}

// After a stop, offer to continue the reply on the same bubble. The partial text
// stays in the conversation, so continuing sends a short "continue" turn and the
// model picks the run back up.
function addContinueAction(bubble) {
  if (!bubble) return;
  var actions = bubble.querySelector(".bubble-actions");
  if (!actions || actions.querySelector('[data-action="continue"]')) return;
  var btn = document.createElement("button");
  btn.className = "bubble-action";
  btn.dataset.action = "continue";
  btn.title = tr("chat.continueTitle");
  btn.setAttribute("aria-label", tr("chat.continueTitle"));
  btn.innerHTML = '<svg class="st-icon st-icon--sm"><use href="/web/sprite.svg#i-send"/></svg>';
  actions.appendChild(btn);
}
function continueChat() {
  if (streaming) return;
  var ta = $("chat-input");
  ta.value = tr("chat.continueWord");
  sendChat();
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
  scrollChatToEnd(true);   // sending is an explicit "show me the newest" move
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
    scrollChatToEnd(false);
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
  // Whatever ended the run — completion, an error, or a stop — the next queued
  // message goes now, unless the queue is paused. "Interject" relies on this
  // being the single exit path: it stops the run and lets this send the head.
  queueAdvance();
}

// Execute a single chat round: send request, stream response, return result
// Returns: { text, toolCalls, error, aborted }
async function doChatRound() {
  var tools = QinferTools.getDefinitions();
  var body = {
    model: modelId,
    messages: QinferSystem.apply(convSystemText(), collectMessages()),
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
  var thinkMode = thinkLevel;
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
      scrollChatToEnd(false);
    });
    if (waitRow.parentElement) waitRow.remove();
  } catch (e) {
    waitRow.remove();
    if (e.name === "AbortError") {
      out.textContent += tr("chat.stoppedSuffix");
      addContinueAction(out.parentElement);
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
  // A reply cut off by the token limit reads like a finished answer, but the model
  // stopped for budget, not because it was done — offer the same "continue"
  // affordance a manual stop gets.
  if (finishReason === "length") addContinueAction(out.parentElement);
  // The final Markdown render can change the height in one go (plain text turning
  // into a table, code blocks that grow). Re-settle the bottom if we are following.
  scrollChatToEnd(false);
  return { text: out.textContent };
}

function stopChat() {
  if (abortCtrl) abortCtrl.abort();
}

// Start a new conversation. The sidebar "New" button is the only caller, so the
// dedupe rule lives here: an untouched empty conversation already is "new".
function newChat() {
  if (streaming) return;
  var cur = QinferConv.getCurrent();
  if (cur && !cur.messages.length && !document.querySelector("#msg-list .bubble")) {
    QinferConv.switchTo(cur.id);
  } else {
    var conv = QinferConv.create();
    QinferConv.switchTo(conv.id);
  }
  setPill("idle", tr("chat.newChat"));
  showToast(tr("chat.newChat"), "info");
  var ta = $("chat-input");
  if (ta) ta.focus();
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
    const state = l.state || "?";
    const queued = l.queued === undefined ? 0 : l.queued;
    const tok = fmtRate(l.tok_s);
    var chatState = $("chat-state");
    if (chatState) chatState.textContent = (tr("mon.stateLine") + " " + trState(state) + " \u00b7 " + tr("mon.queued") + " " + queued + " \u00b7 " + tok + " " + tr("unit.tokS"));
    setStatusDot(true);
    if (state !== "idle" && state !== "?") {
      setPill(state, pillText(state, tok));
    } else {
      setPill("idle", tr("state.idle"));
    }
    // The Monitor tab renders sparklines from the server's own history window,
    // so the chat-side poll only needs the status line and the pill.
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
  initSettings();
}

/* ---- Request-record rendering layer (Monitor tab) ---- */
window.QinferRequests = {
  badge: function (state) {
    const cls = state === "completed" ? "ok" : state === "error" ? "err" : state === "queued" ? "busy" : "";
    return '<span class="badge ' + cls + '">' + esc(trState(state)) + "</span>";
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
      ["state", trState(rec.state)],
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
    const accent = getComputedStyle(document.documentElement).getPropertyValue("--accent").trim() || "#f2600c";
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
  // Defaults for "Reset" come from the page as shipped; /props may refine them (loadProps).
  captureSamplerDefaults();
  // Sliders show a live readout and persist on release; other fields on change.
  SAMPLER_FIELDS.forEach(function (id) {
    var el = $(id);
    if (!el) return;
    el.addEventListener("input", function () {
      if (SLIDER_FIELDS.indexOf(id) >= 0) { syncSliderOutputs(); syncPresetChips(); }
    });
    el.addEventListener("change", saveSampler);
  });
  // Thinking: a segmented radiogroup — every level is one click, and the current
  // one is visible without opening anything. Arrow keys move between segments,
  // which is what role=radio implies.
  var thinkSeg = $("think-seg");
  if (thinkSeg) {
    var segOpts = Array.prototype.slice.call(thinkSeg.querySelectorAll(".think-seg__opt"));
    segOpts.forEach(function (b, i) {
      b.addEventListener("click", function () { setThinkLevel(b.dataset.level); });
      b.addEventListener("keydown", function (e) {
        var d = e.key === "ArrowRight" || e.key === "ArrowDown" ? 1
              : e.key === "ArrowLeft" || e.key === "ArrowUp" ? -1 : 0;
        if (!d) return;
        e.preventDefault();
        var next = segOpts[(i + d + segOpts.length) % segOpts.length];
        setThinkLevel(next.dataset.level);
        next.focus();
      });
    });
  }
  // Presets: one click sets the seven sampling fields.
  var chips = $("preset-chips");
  if (chips) chips.querySelectorAll("[data-preset]").forEach(function (b) {
    b.addEventListener("click", function () { applyPreset(b.dataset.preset); });
  });
  var btnReset = $("btn-reset-sampler");
  if (btnReset) btnReset.addEventListener("click", function () { resetSampler(); });
  var btnClear = $("s-override");
  if (btnClear) btnClear.addEventListener("change", function () { setConvOverride(btnClear.checked); });
  setThinkLevel(thinkLevel, { silent: true });
  syncSliderOutputs();
  syncPresetChips();
  syncOverrideSwitch();
  initAttachmentHandlers();
  // The primary button sends, or stops what is running — never both at once.
  $("btn-send").addEventListener("click", function (e) {
    if (streaming) { e.preventDefault(); stopChat(); }
  });
  $("btn-export").addEventListener("click", exportChat);
  const composer = $("composer");
  if (composer) composer.addEventListener("submit", function (e) { e.preventDefault(); submitOrQueue(); });
  $("chat-input").addEventListener("keydown", function (e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      submitOrQueue();
    }
  });  // Global keyboard shortcuts: QinferKeys.match names the action, this runs it.
  document.addEventListener("keydown", function (e) {
    const action = QinferKeys.match(e);
    if (!action) return;
    if (action === "escape") {
      // A modal (confirm dialog, shortcut help) owns its own Escape handler.
      if (document.querySelector(".st-modal")) return;
      const top = QinferKeys.firstOpen(overlayLayers());
      if (top === "outline") closeOutline();
      else if (top && top.indexOf("drawer:") === 0) setDrawer(top.slice(7), false);
      else if (streaming && abortCtrl) stopChat();
      return;
    }
    e.preventDefault();
    if (action === "focusInput") {
      const ci = $("chat-input");
      if (ci) ci.focus();
    } else if (action === "newConv") {
      newChat();
    } else if (action === "toggleSidebar") {
      toggleConvSidebar();
    } else if (action === "help") {
      showHelpDialog();
    }
  });
  // Per-conversation system prompt: the drawer is the editor, the record is the store.
  const sysEl = $("sys-prompt");
  if (sysEl) {
    sysEl.addEventListener("input", function () {
      syncSysFlag();
      QinferConv.saveSystem(sysEl.value);
    });
  }
  const btnHelp = $("btn-help");
  if (btnHelp) btnHelp.addEventListener("click", showHelpDialog);
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
      // Hide group labels whose every item is filtered out.
      var labels = list.querySelectorAll(".conv-group-label");
      labels.forEach(function (lbl) {
        var any = false, el = lbl.nextElementSibling;
        while (el && !el.classList.contains("conv-group-label")) {
          if (el.classList.contains("conv-item") && !el.classList.contains("hidden")) { any = true; break; }
          el = el.nextElementSibling;
        }
        lbl.classList.toggle("hidden", !any);
      });
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
        // Confirm on the button itself: a toast is easy to miss when the eyes are
        // on the text that was just copied.
        flashAction(btn, "i-check");
      }).catch(function () {
        showToast(tr("chat.error"), "warn", tr("chat.copyFailed"));
      });
    } else if (action === "regenerate") {
      // Remove this assistant message and resend the last user message
      var prevBubble = bubble.previousElementSibling;
      while (prevBubble && !prevBubble.classList.contains("user")) {
        prevBubble = prevBubble.previousElementSibling;
      }
      if (prevBubble && prevBubble.classList.contains("user")) {
        // Get the user message text and the files that turn carried
        var userText = prevBubble._rawText || prevBubble.querySelector(".bubble-body").textContent;
        var userAttach = (prevBubble._attachments || []).slice();
        bubble.remove();
        prevBubble.remove();
        // sendChat() reads the composer and adds the user bubble itself, so the
        // prompt goes back there — adding a bubble by hand would leave the send
        // with an empty composer and nothing would be regenerated.
        attachments = userAttach;
        renderAttachments();
        $("chat-input").value = userText;
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
      // The retraction is a real edit to the conversation, so persist it now —
      // otherwise switching away and back would resurrect the removed turns.
      QinferConv.saveCurrent();
      showToast(tr("chat.editHint"), "info");
    } else if (action === "continue") {
      continueChat();
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
        // Also send it to the server (fire-and-forget: the local copy is the fallback).
        try {
          api("/api/feedback", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              rating: action === "thumbup" ? "up" : "down",
              chat_id: (typeof QinferConv !== "undefined" ? QinferConv.currentId : "") || "",
              excerpt: (text || "").slice(0, 200),
            }),
          }).catch(function () {});
        } catch (e) { /* never break the UI on a feedback post */ }
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
  // The newest-turn follower and its jump button
  initChatScroll();
  // Outline, turn paging, and the message queue
  initChatNav();
  initQueue();
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

/* ---- Message queue ----------------------------------------------------------
   A reply can take a while, and the next thing to say often arrives before it
   ends. Enter while generating now lines that text up instead of dropping it,
   with the contract qwenpaw uses — the one the reader already knows:
     · the queue advances only when nothing is running and it is not paused;
     · Pause gates the queue alone: the running reply keeps going;
     · Interject drops the running reply and sends the head immediately.
   Items persist to localStorage so a reload does not lose them. */
var QinferQueue = {
  STORAGE_KEY: "qinfer.queue",
  CAP: 50,
  // Ids come from a counter, not from the array length: two messages lined up in
  // the same millisecond used to get the same id, and removing one took both.
  _seq: 0,
  // Append. Returns the stored item, or null when the queue is full — a silently
  // dropped message would be worse than being told no.
  push: function (items, text, atts) {
    var t = String(text == null ? "" : text).trim();
    var a = (atts || []).slice();
    if (!t && !a.length) return null;
    if (items.length >= QinferQueue.CAP) return null;
    QinferQueue._seq += 1;
    var item = { id: "q" + Date.now() + "-" + QinferQueue._seq, text: t, attachments: a };
    items.push(item);
    return item;
  },
  removeId: function (items, id) {
    return items.filter(function (i) { return i.id !== id; });
  },
  move: function (items, from, to) {
    if (from === to || from < 0 || to < 0 || from >= items.length || to >= items.length) return items;
    var out = items.slice();
    var it = out.splice(from, 1)[0];
    out.splice(to, 0, it);
    return out;
  },
  // Pause is about the queue only: a paused queue never starts the next turn, and
  // pausing must never interrupt the turn that is already running.
  canAdvance: function (paused, streaming) {
    return !paused && !streaming;
  },
  read: function () {
    try {
      var v = JSON.parse(localStorage.getItem(QinferQueue.STORAGE_KEY) || "null");
      if (!v || !Array.isArray(v.items)) return { items: [], paused: false };
      var items = v.items.filter(function (i) { return i && typeof i.text === "string"; });
      return { items: items, paused: !!v.paused };
    } catch (e) { return { items: [], paused: false }; }
  },
  write: function (items, paused) {
    try {
      localStorage.setItem(QinferQueue.STORAGE_KEY, JSON.stringify({ items: items, paused: !!paused }));
    } catch (e) {}
  },
};

var queueItems = [];
var queuePaused = false;

// The per-row actions, in the order they render. "Interject & send" is the whole
// point of the queue: it cuts the running reply short so this message is next.
var QUEUE_ACTIONS = [
  {action: "interject", labelKey: "queue.interject", icon: "i-send"},
  {action: "edit", labelKey: "queue.edit", icon: "i-edit"},
  {action: "remove", labelKey: "queue.remove", icon: "i-trash"},
];

function queueFullMessage() {
  return tr("queue.full").replace("{n}", String(QinferQueue.CAP));
}

function queueSync() {
  QinferQueue.write(queueItems, queuePaused);
  queueRender();
}

// One queued message: its place in line, what it says, and what can be done with
// it. Rows are draggable — the order is the send order, so it has to be editable.
function queueRow(item, index) {
  var li = document.createElement("li");
  li.className = "queue-item";
  li.draggable = true;
  li.dataset.index = String(index);

  var grip = document.createElement("span");
  grip.className = "queue-item__grip";
  grip.title = tr("queue.reorder");
  grip.innerHTML = '<svg class="st-icon st-icon--sm"><use href="/web/sprite.svg#i-list"/></svg>';
  li.appendChild(grip);

  var idx = document.createElement("span");
  idx.className = "queue-item__idx";
  idx.textContent = String(index + 1) + ".";
  li.appendChild(idx);

  var text = document.createElement("span");
  text.className = "queue-item__text";
  text.textContent = item.text || tr("queue.attachment");
  text.title = item.text || "";
  li.appendChild(text);

  var actions = document.createElement("span");
  actions.className = "queue-item__actions";
  // The keys live in a literal table so the i18n lint can see them (it scans
  // labelKey: values); building them inline would hide them from that check.
  QUEUE_ACTIONS.forEach(function (a) {
    var b = document.createElement("button");
    b.type = "button";
    b.className = "st-btn st-btn--icon st-btn--sm";
    b.dataset.action = a.action;
    b.title = tr(a.labelKey);
    b.setAttribute("aria-label", tr(a.labelKey));
    b.innerHTML = '<svg class="st-icon st-icon--sm"><use href="/web/sprite.svg#' + a.icon + '"/></svg>';
    actions.appendChild(b);
  });
  li.appendChild(actions);

  li.addEventListener("dragstart", function (e) {
    li.classList.add("dragging");
    if (e.dataTransfer) {
      e.dataTransfer.effectAllowed = "move";
      e.dataTransfer.setData("text/plain", String(index));
    }
  });
  li.addEventListener("dragend", function () { li.classList.remove("dragging"); });
  li.addEventListener("dragover", function (e) {
    e.preventDefault();
    li.classList.add("queue-item--over");
  });
  li.addEventListener("dragleave", function () { li.classList.remove("queue-item--over"); });
  li.addEventListener("drop", function (e) {
    e.preventDefault();
    li.classList.remove("queue-item--over");
    var from = parseInt((e.dataTransfer || {}).getData ? e.dataTransfer.getData("text/plain") : "-1", 10);
    var to = index;
    if (isNaN(from) || from === to) return;
    queueItems = QinferQueue.move(queueItems, from, to);
    queueSync();
  });

  li.addEventListener("click", function (e) {
    var b = e.target.closest ? e.target.closest("[data-action]") : null;
    if (!b) return;
    var action = b.dataset.action;
    if (action === "remove") {
      queueItems = QinferQueue.removeId(queueItems, item.id);
      queueSync();
    } else if (action === "interject") {
      queueSendNow(item.id);
    } else if (action === "edit") {
      queueEditRow(li, item);
    }
  });
  return li;
}

// Swap a row for an input so the queued text can be corrected before it goes.
function queueEditRow(li, item) {
  var input = document.createElement("input");
  input.type = "text";
  input.className = "input queue-item__input";
  input.value = item.text;
  input.setAttribute("aria-label", tr("queue.edit"));
  li.innerHTML = "";
  li.appendChild(input);
  input.focus();
  input.select();
  var commit = function (keep) {
    if (keep) item.text = input.value.trim();
    if (!item.text) queueItems = QinferQueue.removeId(queueItems, item.id);
    queueSync();
  };
  input.addEventListener("keydown", function (e) {
    if (e.key === "Enter") { e.preventDefault(); commit(true); }
    else if (e.key === "Escape") { e.preventDefault(); commit(false); }
  });
  input.addEventListener("blur", function () { commit(true); });
}

function queueRender() {
  var panel = $("queue-panel"), list = $("queue-list");
  if (!panel || !list) return;
  panel.classList.toggle("hidden", queueItems.length === 0);
  var count = $("queue-count");
  if (count) count.textContent = String(queueItems.length);
  var state = $("queue-state");
  if (state) state.textContent = queuePaused ? tr("queue.paused") : tr("queue.waiting");
  var label = $("queue-pause-label"), icon = $("queue-pause-icon");
  if (label) label.textContent = queuePaused ? tr("queue.resume") : tr("queue.pause");
  if (icon) icon.querySelector("use").setAttribute("href", "/web/sprite.svg#" + (queuePaused ? "i-play" : "i-pause"));
  var btn = $("btn-queue-pause");
  if (btn) {
    btn.setAttribute("aria-label", queuePaused ? tr("queue.resume") : tr("queue.pause"));
    btn.classList.toggle("is-paused", queuePaused);
  }
  list.innerHTML = "";
  queueItems.forEach(function (item, i) { list.appendChild(queueRow(item, i)); });
}

function queueAdd(text, atts) {
  var item = QinferQueue.push(queueItems, text, atts);
  if (!item) { showToast(tr("chat.error"), "warn", queueFullMessage()); return false; }
  queueSync();
  showToast(tr("queue.queued"), "info");
  return true;
}

// Pull the head and run it. sendChat() reads the composer, so the item goes back
// through the composer rather than into the transcript directly.
function queueAdvance() {
  if (!QinferQueue.canAdvance(queuePaused, streaming)) return;
  var item = queueItems.shift();
  if (!item) { queueRender(); return; }
  QinferQueue.write(queueItems, queuePaused);
  queueRender();
  var ta = $("chat-input");
  if (ta) ta.value = item.text || "";
  attachments = (item.attachments || []).slice();
  renderAttachments();
  sendChat();
}

// "Interject": the reader wants this answered next, so the running reply is cut
// short — the queue head then goes out through the run's own exit path.
function queueInterject() {
  if (!queueItems.length) return;
  queuePaused = false;      // an explicit interject overrides a paused queue
  queueSync();
  if (streaming) stopChat();
  else queueAdvance();
}

// Send one specific queued message now, ahead of anything before it.
function queueSendNow(id) {
  var item = queueItems.filter(function (i) { return i.id === id; })[0];
  if (!item) return;
  queueItems = QinferQueue.removeId(queueItems, id);
  queueItems.unshift(item);
  queueSync();
  queueInterject();
}

function queueDropAll() {
  if (!queueItems.length) return;
  queueItems = [];
  queuePaused = false;
  queueSync();
  showToast(tr("queue.cleared"), "info");
}

function initQueue() {
  var st = QinferQueue.read();
  queueItems = st.items;
  queuePaused = st.paused;
  queueRender();
  var pause = $("btn-queue-pause"), clear = $("btn-queue-clear");
  if (pause) pause.addEventListener("click", function () { queuePaused = !queuePaused; queueSync(); if (!queuePaused) queueAdvance(); });
  if (clear) clear.addEventListener("click", queueDropAll);
}

// The composer submits through here: with nothing running it is an ordinary send,
// with a reply in flight the text lines up instead of being refused.
function submitOrQueue() {
  var ta = $("chat-input");
  if (!ta) return;
  var text = ta.value.trim();
  if (!text && !attachments.length) return;
  if (streaming) {
    if (!queueAdd(text, attachments.slice())) return;
    ta.value = "";
    attachments = [];
    renderAttachments();
    return;
  }
  sendChat();
}

/* ---- Chat navigation: outline and turn paging -------------------------------
   A turn is a user message; it anchors the reply under it, so paging between
   turns is paging between exchanges. The outline lists the same turns by their
   opening words, which is how a long conversation gets navigable again. */
function chatTurns() {
  return Array.prototype.slice.call(document.querySelectorAll("#msg-list .bubble.user"));
}

// The turn whose top is the last one at or above the viewport top.
function currentTurnIndex() {
  var el = chatScroller(), turns = chatTurns();
  if (!el || !turns.length) return -1;
  var top = el.scrollTop + 8;
  var found = 0;
  for (var i = 0; i < turns.length; i++) {
    if (turns[i].offsetTop <= top) found = i; else break;
  }
  return found;
}

function gotoTurn(i) {
  var el = chatScroller(), turns = chatTurns();
  if (!el || !turns.length) return;
  var idx = Math.max(0, Math.min(turns.length - 1, i));
  el.scrollTo({ top: Math.max(0, turns[idx].offsetTop - 8), behavior: "smooth" });
  turns[idx].classList.add("turn-flash");
  setTimeout(function () { turns[idx].classList.remove("turn-flash"); }, 900);
}

function stepTurn(delta) {
  var turns = chatTurns();
  if (!turns.length) return;
  var cur = currentTurnIndex();
  gotoTurn((cur < 0 ? 0 : cur) + delta);
}

function outlineLabel(bubble) {
  var raw = bubble._rawText || (bubble.querySelector(".bubble-body") || bubble).textContent || "";
  var one = raw.replace(/\s+/g, " ").trim();
  return one.length > 60 ? one.slice(0, 60) + "…" : one;
}

function renderOutline() {
  var list = $("outline-list");
  if (!list) return;
  var turns = chatTurns();
  list.innerHTML = "";
  if (!turns.length) {
    var p = document.createElement("p");
    p.className = "muted small outline-pop__empty";
    p.textContent = tr("chat.outlineEmpty");
    list.appendChild(p);
    return;
  }
  var active = currentTurnIndex();
  turns.forEach(function (b, i) {
    var item = document.createElement("button");
    item.type = "button";
    item.className = "outline-item" + (i === active ? " active" : "");
    item.setAttribute("role", "menuitem");
    var n = document.createElement("span");
    n.className = "outline-item__idx";
    n.textContent = String(i + 1);
    item.appendChild(n);
    item.appendChild(document.createTextNode(outlineLabel(b)));
    item.addEventListener("click", function () { gotoTurn(i); closeOutline(); });
    list.appendChild(item);
  });
}

function closeOutline() {
  var pop = $("outline-pop"), btn = $("btn-outline");
  if (pop) pop.classList.add("hidden");
  if (btn) btn.setAttribute("aria-expanded", "false");
}

function toggleOutline(force) {
  var pop = $("outline-pop"), btn = $("btn-outline");
  if (!pop) return;
  var open = force == null ? pop.classList.contains("hidden") : force;
  if (open) renderOutline();
  pop.classList.toggle("hidden", !open);
  if (btn) btn.setAttribute("aria-expanded", open ? "true" : "false");
}

function initChatNav() {
  var out = $("btn-outline"), prev = $("btn-prev-turn"), next = $("btn-next-turn");
  if (out) out.addEventListener("click", function (e) { e.stopPropagation(); toggleOutline(); });
  if (prev) prev.addEventListener("click", function () { stepTurn(-1); });
  if (next) next.addEventListener("click", function () { stepTurn(1); });
  document.addEventListener("click", function (e) {
    var pop = $("outline-pop");
    if (!pop || pop.classList.contains("hidden")) return;
    if (pop.contains(e.target) || e.target.closest("#btn-outline")) return;
    closeOutline();
  });
}

/* ---- Chat scrolling --------------------------------------------------------
   The scroller is #chat-scroll; the message list inside it has no overflow of
   its own, which is why the old `#msg-list.scrollTop = …` lines moved nothing
   and a reply could stream on below the fold.

   Following is a *decision*, not a per-call measurement. Measuring on every
   stream chunk looked equivalent but was not: one frame where the geometry
   lagged (a smooth scroll still animating, a layout that had not settled) read
   as "the reader scrolled away", the follower gave up for the rest of the run,
   and every chunk after that counted as an unread turn. So the reader's intent
   is held in chatFollow, only the reader's own scroll changes it, and the unread
   count rises once per message that lands — not once per chunk. */
function chatScroller() {
  return $("chat-scroll");
}

// Pure: is the viewport within `slack` px of the end? A missing scroller counts
// as at-the-end, so callers stay one line long.
function nearBottom(m, slack) {
  if (!m) return true;
  var s = slack == null ? 80 : slack;
  return m.scrollHeight - m.scrollTop - m.clientHeight <= s;
}

var chatFollow = true;             // is the reader watching the newest turn?
var chatUnread = 0;                // messages that landed while they were not
var chatWrittenTop = -1;           // the scrollTop our own write produced

function updateJumpButton() {
  var btn = $("btn-scroll-bottom");
  if (!btn) return;
  btn.classList.toggle("hidden", chatFollow);
  var n = $("scroll-count");
  if (n) {
    n.textContent = chatUnread > 99 ? "99+" : String(chatUnread);
    n.classList.toggle("hidden", chatUnread === 0);
  }
}

// Put the newest turn in view and remember the position the browser accepted, so
// the scroll event this raises can be told apart from the reader scrolling.
var followPending = false;
function followNewest() {
  var el = chatScroller();
  chatUnread = 0;
  if (el) {
    el.scrollTop = el.scrollHeight;
    chatWrittenTop = el.scrollTop;
    // Text appended immediately before this call may not be laid out yet, in
    // which case scrollTop stops short of the real end and the gap only grows as
    // the reply streams. Settle it again once the frame has been laid out.
    if (!followPending) {
      followPending = true;
      requestAnimationFrame(function () {
        followPending = false;
        if (!chatFollow) return;
        var e2 = chatScroller();
        if (e2) {
          e2.scrollTop = e2.scrollHeight;
          chatWrittenTop = e2.scrollTop;
        }
        updateJumpButton();
      });
    }
  }
  updateJumpButton();
}

// A message landed. Follow it if the reader is watching; otherwise count it.
function onMessageArrived() {
  if (chatFollow) followNewest();
  else { chatUnread++; updateJumpButton(); }
}

// Follow the end when the reader is there, or when the caller knows the move is
// wanted (they just sent something, switched conversation, opened the chat).
function scrollChatToEnd(force) {
  if (force) { chatFollow = true; followNewest(); return; }
  if (chatFollow) followNewest();
}

// Jump to the end from the button: always shows the newest turn.
function updateChatNavState() {
  updateJumpButton();
}

function jumpToEnd() {
  var el = chatScroller();
  chatFollow = true;
  chatUnread = 0;
  if (el) {
    // The smooth scroll passes through positions far from the bottom, and every
    // one of them would look like the reader scrolling; ignore that window.
    el.scrollTo({ top: el.scrollHeight, behavior: "smooth" });
    chatWrittenTop = el.scrollHeight;
  }
  updateJumpButton();
}

function initChatScroll() {
  var el = chatScroller(), btn = $("btn-scroll-bottom");
  if (!el) return;
  // The reader's intent is an input, not a scroll position. A scroll event cannot
  // be attributed — our own writes raise them too, and in a tight stream race the
  // write lands first — but a wheel, a touch, or a paging key is unambiguously
  // someone at the machine. Registering it here means the reader wins the moment
  // they reach for the view, before the next chunk can pull it back.
  var readerTookOver = function () {
    if (!chatFollow) return;
    chatFollow = false;
    updateJumpButton();
  };
  ["wheel", "touchmove"].forEach(function (ev) {
    el.addEventListener(ev, readerTookOver, { passive: true });
  });
  el.addEventListener("keydown", function (e) {
    if (["PageUp", "PageDown", "ArrowUp", "ArrowDown", "Home", "End", " "].indexOf(e.key) >= 0) readerTookOver();
  });
  el.addEventListener("scroll", function () {
    // Our own write lands exactly on chatWrittenTop (bar layout rounding), so a
    // scroll anywhere else — a dragged scrollbar, a programmatic jump — is what
    // decides: back at the bottom resumes following, away from it stops.
    if (Math.abs(el.scrollTop - chatWrittenTop) <= 4) return;
    if (nearBottom(el)) {
      chatFollow = true;
      chatUnread = 0;
    } else if (chatFollow) {
      chatFollow = false;
    }
    updateChatNavState();
  });
  if (btn) btn.addEventListener("click", jumpToEnd);
  updateJumpButton();
}

/* ---- Per-conversation system prompt --------------------------------------
   The textarea is the live editor; the conversation record is the store. Keep
   the two in sync through these three helpers so a drawer left collapsed still
   shows a dot when the conversation carries an instruction. */
function convSystemText() {
  const el = $("sys-prompt");
  return el ? el.value : "";
}
function syncSysFlag() {
  const flag = $("sys-flag");
  if (flag) flag.classList.toggle("hidden", !convSystemText().trim());
}
function sysApplyToConv(conv) {
  const el = $("sys-prompt");
  if (el) el.value = conv && conv.system ? conv.system : "";
  syncSysFlag();
}

/* ---- Conversation history management ---- */
var QinferConv = {
  STORAGE_KEY: "qinfer.conversations",
  currentId: null,

  // Load all conversations from localStorage
  loadAll: function() {
    try {
      var convs = JSON.parse(localStorage.getItem(this.STORAGE_KEY) || "[]");
      // Older builds stored the translated placeholder as the title. Migrate it
      // back to the empty "untitled" marker so the label follows the language.
      var untitled = [I18N.zh["conv.untitled"], I18N.en["conv.untitled"]];
      var migrated = false;
      convs.forEach(function (c) {
        if (c && untitled.indexOf(c.title) >= 0) { c.title = ""; migrated = true; }
      });
      if (migrated) this.saveAll(convs);
      return convs;
    } catch (e) { return []; }
  },

  // Save all conversations to localStorage
  saveAll: function(convs) {
    try { localStorage.setItem(this.STORAGE_KEY, JSON.stringify(convs)); return; } catch (e) {}
    // An image attachment is a base64 data URL and localStorage caps out around
    // 5 MB. Rather than lose the whole write, keep every conversation's text and
    // drop only the image payloads.
    try { localStorage.setItem(this.STORAGE_KEY, JSON.stringify(withoutImages(convs))); } catch (e2) {}
    showToast(tr("chat.error"), "warn", tr("toast.storageFull"));
  },

  // Persist only the instruction: typing must not rewrite the message list.
  saveSystem: function(text) {
    var convs = this.loadAll();
    var conv = convs.find(function(c) { return c.id === QinferConv.currentId; });
    if (!conv || conv.system === text) return;
    conv.system = text;
    this.saveAll(convs);
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
    // Apply default system prompt from settings if available
    var defaultSys = "";
    if (typeof loadSettings === "function") {
      var s = loadSettings();
      defaultSys = s["sd-system-prompt"] || "";
    }
    var conv = { id: id, title: "", messages: [], system: defaultSys, created: Date.now(), updated: Date.now() };
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
      var msg = msgFromBubble(b, null);
      if (msg) msgs.push(msg);
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
      system: cur ? cur.system || "" : "",
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
    if (typeof syncOverrideSwitch === "function") syncOverrideSwitch();
    // Restore the conversation's own system prompt.
    sysApplyToConv(conv);
    $("msg-list").innerHTML = "";
    if (conv.messages.length === 0) {
      var empty = $("chat-empty");
      if (empty) empty.style.display = "";
    } else {
      var empty = $("chat-empty");
      if (empty) empty.style.display = "none";
      conv.messages.forEach(function(msg) {
        if (msg.role === "user") {
          addMsg("user", msg.content, msg.attachments || null, msg.ts);
        } else if (msg.role === "assistant") {
          var body = addMsg("assistant", "", null, msg.ts);
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
      // Opening a conversation always shows its newest turn, not wherever the
      // previous conversation happened to be scrolled.
      scrollChatToEnd(true);
    }
    this.renderList();
  },

  // Save current conversation state
  saveCurrent: function() {
    var convs = this.loadAll();
    var conv = this.currentId
      ? convs.find(function(c) { return c.id === QinferConv.currentId; })
      : null;
    if (!conv) {
      // A draft shown on an unbound screen becomes a conversation only once it
      // holds a message — otherwise merely switching away would store empty rows.
      if (!document.querySelector("#msg-list .bubble")) return;
      var draft = this.create();
      this.currentId = draft.id;
      convs = this.loadAll();
      conv = convs.find(function(c) { return c.id === QinferConv.currentId; });
      if (!conv) return;
      this.renderList();  // the adopted conversation is the active one
    }
    // Collect messages from the UI
    var messages = [];
    document.querySelectorAll("#msg-list .bubble").forEach(function(b) {
      if (b.classList.contains("tool-call") || b.classList.contains("tool-result")) return;
      var msg = msgFromBubble(b, conv.updated);
      if (msg) messages.push(msg);
    });
    conv.messages = messages;
    conv.system = convSystemText();
    if (conv.params) conv.params = samplerSnapshot();
    if (typeof syncOverrideSwitch === "function") syncOverrideSwitch();
    conv.updated = Date.now();
    // Auto-generate title from first user message (empty title == untitled)
    if (!conv.title && messages.length > 0) {
      var firstUser = messages.find(function(m) { return m.role === "user"; });
      if (firstUser) {
        conv.title = firstUser.content.slice(0, 40) + (firstUser.content.length > 40 ? "…" : "");
      }
    }
    this.saveAll(convs);
  },

  // Delete a conversation
  delete: function(id) {
    // st-* confirm dialog (native confirm() clashes with the design system and
    // is disabled in some embedded WebViews).
    showConfirm(tr("conv.confirmDelete"), tr("conv.delete")).then(function (yes) {
      if (!yes) return;
      var convs = QinferConv.loadAll();
      convs = convs.filter(function(c) { return c.id !== id; });
      QinferConv.saveAll(convs);
      if (QinferConv.currentId === id) {
        QinferConv.currentId = null;
        $("msg-list").innerHTML = "";
        var empty = $("chat-empty");
        if (empty) empty.style.display = "";
      }
      QinferConv.renderList();
    });
  },

  // Rename a conversation: inline edit in the sidebar (replaces native prompt()).
  rename: function(id) {
    var item = document.querySelector('.conv-item[data-id="' + id + '"]');
    if (!item) return;
    var nameEl = item.querySelector(".conv-item__name");
    if (!nameEl || item.querySelector(".conv-item__edit")) return;
    var convs = this.loadAll();
    var conv = convs.find(function(c) { return c.id === id; });
    if (!conv) return;
    var input = document.createElement("input");
    input.type = "text";
    input.className = "input conv-item__edit";
    input.value = conv.title;
    // The placeholder hints what the empty entry means, but never becomes the
    // stored title — that keeps "untitled" language-independent.
    input.placeholder = convTitle(conv);
    input.setAttribute("aria-label", tr("conv.rename"));
    nameEl.style.display = "none";
    nameEl.parentNode.insertBefore(input, nameEl);
    input.focus();
    input.select();
    function commit() {
      var v = input.value.trim();
      input.remove();
      nameEl.style.display = "";
      if (v && v !== conv.title) {
        conv.title = v;
        QinferConv.saveAll(convs);
      }
      QinferConv.renderList();
    }
    input.addEventListener("keydown", function (e) {
      if (e.key === "Enter") { e.preventDefault(); commit(); }
      if (e.key === "Escape") { input.remove(); nameEl.style.display = ""; }
    });
    input.addEventListener("blur", commit);
    input.addEventListener("click", function (e) { e.stopPropagation(); });
  },

  // Pin/unpin a conversation (pinned ones sort first, above date groups).
  togglePin: function(id) {
    var convs = this.loadAll();
    var conv = convs.find(function(c) { return c.id === id; });
    if (!conv) return;
    conv.pinned = !conv.pinned;
    this.saveAll(convs);
    this.renderList();
  },

  // Export every conversation as one JSON file (schema: qinfer.convExport.v1).
  exportAll: function() {
    var convs = this.loadAll();
    if (!convs.length) return;
    var payload = { format: "qinfer.convExport.v1", exported: new Date().toISOString(), conversations: convs };
    var blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
    var a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = "qinfer-conversations-" + new Date().toISOString().slice(0, 10) + ".json";
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(function () { URL.revokeObjectURL(a.href); }, 5000);
    showToast(tr("conv.exportedToast").replace("{n}", convs.length));
  },

  // Import conversations from a qinfer.convExport.v1 file. Existing ids get a
  // fresh id (conflicts are merged under a new name, never overwritten).
  importAll: function(file) {
    var reader = new FileReader();
    reader.onload = function () {
      var data;
      try { data = JSON.parse(reader.result); } catch (e) { data = null; }
      if (!data || data.format !== "qinfer.convExport.v1" || !Array.isArray(data.conversations)) {
        showToast(tr("conv.importBadFormat"));
        return;
      }
      var convs = QinferConv.loadAll();
      var existing = {};
      convs.forEach(function (c) { existing[c.id] = true; });
      var n = 0;
      data.conversations.forEach(function (c) {
        if (!c || !Array.isArray(c.messages)) return;
        var id = c.id && !existing[c.id] ? c.id : "c" + Date.now().toString(36) + Math.random().toString(36).slice(2, 7);
        existing[id] = true;
        convs.push({
          id: id,
          title: String(c.title || "").slice(0, 80),
          messages: c.messages,
          params: c.params || undefined,
          system: typeof c.system === "string" ? c.system : "",
          pinned: !!c.pinned,
          created: c.created || Date.now(),
          updated: c.updated || Date.now(),
        });
        n++;
      });
      if (!n) { showToast(tr("conv.importBadFormat")); return; }
      QinferConv.saveAll(convs);
      QinferConv.renderList();
      showToast(tr("conv.importedToast").replace("{n}", n));
    };
    reader.readAsText(file);
  },

  // Date bucket for the sidebar grouping: 0 today, 1 yesterday, 2 last 7 days,
  // 3 earlier. Local calendar days (a 23:59 conversation still counts as today).
  _bucket: function(ts) {
    var d = new Date(ts), now = new Date();
    var day = function (x) { return new Date(x.getFullYear(), x.getMonth(), x.getDate()).getTime(); };
    var diff = (day(now) - day(d)) / 86400000;
    if (diff <= 0) return 0;
    if (diff === 1) return 1;
    if (diff < 7) return 2;
    return 3;
  },

  // Render the conversation list, grouped: Pinned, then Today / Yesterday /
  // Last 7 days / Earlier (newest first inside each group).
  renderList: function() {
    var list = $("conv-list");
    if (!list) return;
    var convs = this.loadAll();
    if (convs.length === 0) {
      list.innerHTML = '<p class="muted small">' + esc(tr("conv.empty")) + '</p>';
      return;
    }
    convs.sort(function (a, b) { return (b.updated || 0) - (a.updated || 0); });
    var groups = [
      { labelKey: "conv.groupPinned", items: convs.filter(function (c) { return c.pinned; }) },
      { labelKey: "conv.groupToday", items: convs.filter(function (c) { return !c.pinned && QinferConv._bucket(c.updated) === 0; }) },
      { labelKey: "conv.groupYesterday", items: convs.filter(function (c) { return !c.pinned && QinferConv._bucket(c.updated) === 1; }) },
      { labelKey: "conv.groupWeek", items: convs.filter(function (c) { return !c.pinned && QinferConv._bucket(c.updated) === 2; }) },
      { labelKey: "conv.groupEarlier", items: convs.filter(function (c) { return !c.pinned && QinferConv._bucket(c.updated) === 3; }) },
    ];
    list.innerHTML = "";
    groups.forEach(function (g) {
      if (!g.items.length) return;
      var head = document.createElement("div");
      head.className = "conv-group-label";
      head.textContent = tr(g.labelKey);
      list.appendChild(head);
      g.items.forEach(function(conv) {
        var item = document.createElement("div");
        item.className = "conv-item" + (conv.id === QinferConv.currentId ? " active" : "");
        item.dataset.id = conv.id;
        item.innerHTML = '<span class="conv-item__title"><span class="conv-item__name">' + esc(convTitle(conv)) +
          '</span><span class="conv-item__time">' + relTime(conv.updated) + '</span></span>' +
          '<span class="conv-item__actions">' +
          '<button class="conv-item__btn" data-action="pin" title="' + esc(conv.pinned ? tr("conv.unpin") : tr("conv.pin")) + '"><svg class="st-icon st-icon--sm"><use href="/web/sprite.svg#i-pin"/></svg></button>' +
          '<button class="conv-item__btn" data-action="rename" title="' + esc(tr("conv.rename")) + '"><svg class="st-icon st-icon--sm"><use href="/web/sprite.svg#i-edit"/></svg></button>' +
          '<button class="conv-item__btn" data-action="delete" title="' + esc(tr("conv.delete")) + '"><svg class="st-icon st-icon--sm"><use href="/web/sprite.svg#i-trash"/></svg></button>' +
          '</span>';
        item.addEventListener("click", function(e) {
          if (e.target.closest(".conv-item__btn")) return;
          QinferConv.switchTo(conv.id);
        });
        item.querySelector('[data-action="pin"]').addEventListener("click", function(e) {
          e.stopPropagation();
          QinferConv.togglePin(conv.id);
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
    });
  },

  // Initialize the sidebar
  init: function() {
    var sidebar = $("conv-sidebar");
    var btnToggle = $("btn-toggle-conv");
    var btnClose = $("btn-close-conv");
    var btnNew = $("btn-new-conv");
    if (btnToggle) {
      btnToggle.addEventListener("click", toggleConvSidebar);
    }
    if (btnClose) {
      btnClose.addEventListener("click", function() {
        sidebar.classList.add("hidden");
      });
    }
    if (btnNew) {
      btnNew.addEventListener("click", newChat);
    }
    // Sidebar footer: export all conversations as JSON, import back in.
    var btnExport = $("btn-export-conv");
    if (btnExport) btnExport.addEventListener("click", function () { QinferConv.exportAll(); });
    var btnImport = $("btn-import-conv");
    var importFile = $("conv-import-file");
    if (btnImport && importFile) {
      btnImport.addEventListener("click", function () { importFile.value = ""; importFile.click(); });
      importFile.addEventListener("change", function () {
        if (importFile.files && importFile.files[0]) QinferConv.importAll(importFile.files[0]);
      });
    }
    // Load the most recent conversation. An empty screen is deliberately NOT a
    // conversation yet — saveCurrent() adopts one on the first message, so the
    // sidebar can show its empty state instead of a phantom "Untitled" entry.
    var convs = this.loadAll();
    if (convs.length > 0) this.switchTo(convs[0].id);
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

// ---- Message timestamps ------------------------------------------------------
// Bubbles carry epoch milliseconds; the label shows the clock for today's
// messages and grows a date for anything older. Formatting goes through Intl so
// the stamp follows the UI language rather than a hand-rolled pattern.
var QinferTime = {
  locale: function () { return uiLang() === "en" ? "en-GB" : "zh-CN"; },
  clock: function (ms, now, locale) {
    if (ms == null) return "";
    var a = new Date(ms), b = new Date(now == null ? Date.now() : now);
    var sameDay = a.getFullYear() === b.getFullYear() && a.getMonth() === b.getMonth() && a.getDate() === b.getDate();
    return new Intl.DateTimeFormat(locale || QinferTime.locale(), sameDay
      ? {hour: "2-digit", minute: "2-digit"}
      : {month: "short", day: "numeric", hour: "2-digit", minute: "2-digit"}).format(a);
  },
  full: function (ms, locale) {
    if (ms == null) return "";
    return new Intl.DateTimeFormat(locale || QinferTime.locale(),
      {dateStyle: "medium", timeStyle: "medium"}).format(new Date(ms));
  },
};

// ---- Keyboard shortcuts ------------------------------------------------------
// match() only names the action; the handler decides what it does, so Escape can
// close an overlay or stop a run depending on what is open. Typing in a field
// must never trigger the bare "?" command.
var QinferKeys = {
  isTyping: function (el) {
    if (!el || !el.tagName) return false;
    var t = el.tagName.toUpperCase();
    return t === "INPUT" || t === "TEXTAREA" || t === "SELECT" || el.isContentEditable === true;
  },
  match: function (e) {
    var mod = !!(e.metaKey || e.ctrlKey);
    var k = e.key;
    if (k === "Escape") return "escape";
    if (mod && e.shiftKey && !e.altKey && (k === "O" || k === "o")) return "newConv";
    if (mod && !e.shiftKey && !e.altKey && (k === "K" || k === "k")) return "focusInput";
    if (mod && !e.shiftKey && !e.altKey && (k === "B" || k === "b")) return "toggleSidebar";
    if (!mod && !e.altKey && k === "?" && !QinferKeys.isTyping(e.target)) return "help";
    return null;
  },
  // rows(): one table feeds both the "?" dialog and the reader, so the help
  // panel can never list a binding the handler does not implement.
  rows: function (mac) {
    var mod = mac ? "⌘" : "Ctrl";
    return [
      {keys: mod + " + K", labelKey: "keys.focusInput"},
      {keys: mod + " + Shift + O", labelKey: "keys.newConv"},
      {keys: mod + " + B", labelKey: "keys.toggleSidebar"},
      {keys: "?", labelKey: "keys.help"},
      {keys: "Esc", labelKey: "keys.escape"},
      {keys: "Enter", labelKey: "keys.send"},
      {keys: "Shift + Enter", labelKey: "keys.newline"},
    ];
  },
  // firstOpen(): Escape walks the dismissible layers in z-order and returns the
  // topmost one that is up; null means nothing is up, so Escape stops a run.
  firstOpen: function (layers) {
    var list = layers || [];
    for (var i = 0; i < list.length; i++) {
      if (list[i] && list[i].open) return list[i].name;
    }
    return null;
  },
};

// ---- Per-conversation system prompt ------------------------------------------
// A conversation may carry its own instruction. It goes in front of the history
// as the system message, replacing any system message already there so repeated
// sends never stack instructions.
var QinferSystem = {
  apply: function (system, messages) {
    var text = (system == null ? "" : String(system)).trim();
    var out = messages || [];
    if (!text) return out;
    var head = { role: "system", content: text };
    if (out.length && out[0].role === "system") return [head].concat(out.slice(1));
    return [head].concat(out);
  },
};

// ---- Monitor speed: the decode rate is a per-request number ------------------
// The server's live.tok_s is the *last request's* rate, so it goes stale the
// moment a run ends and leaves the card blank while idle. The card reads the
// session mean instead (live.tok_s_mean), and the line under it plots the
// server's history of that same mean — one 1 Hz sample window, exactly like the
// prefill card, so the two rate charts cannot drift apart.
var QinferSpeed = {
  card: function (live) {
    var l = live || {};
    if (l.state === "generating" && l.tok_s != null) return { value: l.tok_s, labelKey: "mon.decodeNow" };
    if (l.tok_s_mean != null) return { value: l.tok_s_mean, labelKey: "mon.decodeMean" };
    if (l.tok_s != null) return { value: l.tok_s, labelKey: "mon.decodeLast" };
    return { value: null, labelKey: "mon.decodeLabel" };
  },
};

// Session totals for the Requests page. /metrics carries raw counters only, so
// every value shown here is either passed through or a ratio the card itself
// computes (per-request average, reuse percentage, prefill+decode wall time).
// Absent data stays null and renders as an em dash rather than a zero.
var QinferActivity = {
  reuseShare: function (t) {
    var x = t || {};
    return x.prompt_tokens ? (100 * (x.reused || 0)) / x.prompt_tokens : null;
  },
  rows: function (t) {
    var x = t || {};
    var num = function (v) { return v == null ? null : v; };
    var reqs = x.requests || 0;
    var perReq = function (v) { return reqs && v != null ? v / reqs : null; };
    var busy = (x.prompt_ms || 0) + (x.decode_ms || 0);
    return [
      {labelKey: "act.requests", value: num(x.requests), kind: "count"},
      {labelKey: "act.prompt", value: num(x.prompt_tokens), kind: "tokens", avg: perReq(x.prompt_tokens)},
      {labelKey: "act.completion", value: num(x.completion_tokens), kind: "tokens", avg: perReq(x.completion_tokens)},
      {labelKey: "act.reused", value: num(x.reused), kind: "tokens", share: this.reuseShare(x)},
      {labelKey: "act.busy", value: busy ? busy / 1000 : null, kind: "seconds",
       parts: busy ? [(x.prompt_ms || 0) / 1000, (x.decode_ms || 0) / 1000] : null},
    ];
  },
};

// Metric card definitions for the 4-column grid. `series` is the key of the
// server-side 60-sample history block in /metrics; every card plots one of those
// windows, including speed, whose series is the session decode mean (tok_s_mean).
var MON_METRICS = [
  {key: "speed", labelKey: "metric.speed", icon: "i-gauge", unit: "t/s", series: "tok_s_mean"},
  {key: "prefill", labelKey: "metric.prefill", icon: "i-bolt", unit: "t/s", series: "prefill_tok_s_mean"},
  {key: "gpu", labelKey: "metric.gpuLoad", icon: "i-gpu", unit: "%", series: "gpu_util", max: 100},
  {key: "vram", labelKey: "metric.vram", icon: "i-layers", unit: "GB", series: "gpu_mem_used", bytes: true, maxFrom: "gpu_mem_total"},
  {key: "ram", labelKey: "metric.ram", icon: "i-layers", unit: "GB", series: "ram_used", bytes: true, maxFrom: "ram_total"},
  {key: "temp", labelKey: "metric.temp", icon: "i-thermometer", unit: "\u00b0C", series: "gpu_temp", tone: "warn", max: 90},
  {key: "power", labelKey: "metric.power", icon: "i-bolt", unit: "W", series: "gpu_power", maxFrom: "gpu_power_limit"},
  {key: "pcie", labelKey: "metric.pcie", icon: "i-link", unit: "", series: "gpu_pcie_rx_mb", tone: "info"},
  {key: "cpu", labelKey: "metric.cpu", icon: "i-cpu", unit: "%", series: "cpu", max: 100},
  {key: "disk", labelKey: "metric.disk", icon: "i-disk", unit: "MB/s", series: "disk_read_mb", tone: "info"},
  {key: "igpu", labelKey: "metric.igpu", icon: "i-gpu", unit: "MHz", series: "igpu_freq_mhz", maxFrom: "igpu_freq_max_mhz"},
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

// Render an SVG sparkline from an array of values (the server's 60-sample
// history window, or a per-request series). `divisor` scales raw bytes to GB.
function sparkline(id, values, max, divisor) {
  var svg = $(id);
  if (!svg) return;
  var d = divisor || 1;
  var v = (values || []).map(function (x) { return x == null ? 0 : x / d; });
  if (v.length < 2) {
    svg.querySelector(".line").setAttribute("d", "");
    svg.querySelector(".area").setAttribute("d", "");
    return;
  }
  var maxDisplay = (max || 0) / d;
  var top = Math.max(maxDisplay, Math.max.apply(null, v), 0.01);
  var positiveValues = v.filter(function (x) { return x > 0; });
  var min = positiveValues.length ? Math.min.apply(null, positiveValues) : 0;

  var pts = v.map(function (x, i) {
    return [(i / (v.length - 1)) * 100, 30 - (x / top) * 26];
  });
  var line = pts.map(function (p, i) { return (i ? "L" : "M") + p[0].toFixed(2) + "," + p[1].toFixed(2); }).join("");
  svg.querySelector(".line").setAttribute("d", line);
  svg.querySelector(".area").setAttribute("d", line + "L100,32L0,32Z");

  // Axis labels live in the wrapper, not the SVG.
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
    label = live.phase === "prefill" ? tr("act.prefill") : live.phase === "decode" ? tr("act.decode") : tr("state.generating");
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

  // Metric cards. Every sparkline reads the server's 60-sample history window
  // (m.history), so the charts cover the minute before the page was opened and
  // never depend on how long this tab has been polling.
  var GB = 1073741824;
  var speedCard = QinferSpeed.card(live);
  setMetricValue("speed", speedCard.value == null ? null : fmtNum(speedCard.value, 1), "t/s", tr(speedCard.labelKey));
  setMetricValue("prefill", live.prefill_tok_s_mean == null ? null : fmtNum(live.prefill_tok_s_mean, 1), "t/s",
    tr("mon.prefillLabel"));
  setMetricValue("gpu", hw.gpu_util == null ? null : fmtNum(hw.gpu_util), "%",
    st.gpu_name || (st.gpu_note ? tr("mon.notAvail") : ""));
  var vramUsed = hw.gpu_mem_used != null ? (hw.gpu_mem_used / GB).toFixed(2) : null;
  var vramTotal = hw.gpu_mem_total ? " / " + (hw.gpu_mem_total / GB).toFixed(0) + " GB" : " GB";
  setMetricValue("vram", vramUsed, vramTotal, "");
  var ramUsed = hw.ram_used != null ? (hw.ram_used / GB).toFixed(1) : null;
  setMetricValue("ram", ramUsed, hw.ram_total ? " / " + (hw.ram_total / GB).toFixed(0) + " GB" : " GB", "");
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

  MON_METRICS.forEach(function (m) {
    var series = h[m.series] || [];
    var mx = m.max != null ? m.max : (m.maxFrom ? hw[m.maxFrom] : null);
    sparkline("sp-" + m.key, series, mx, m.bytes ? GB : 1);
  });

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
}

// ---- Requests page: what this session spent and what the cache saved --------
// Fed from the same /metrics poll as the Monitor tab: both panels live in one
// document, so the requests page is warm the moment it is opened.
function renderRequestsTab(m) {
  var requests = m.requests || [];
  renderActivity(m.totals);

  // Recent requests table (last 10)
  var body = $("req-body");
  if (body) {
    if (!requests.length) {
      body.innerHTML = '<tr><td colspan="6" class="muted">' + esc(tr("mon.noReq")) + '</td></tr>';
    } else {
      body.innerHTML = requests.slice(0, 10).map(function (r) {
        var badge = {stop: ["", tr("req.done")], length: ["", tr("req.maxTokens")], cancel: ["st-badge--queued", tr("req.stoppedBadge")],
                     disconnect: ["st-badge--queued", tr("req.closed")], error: ["st-badge--error", tr("chat.error")]};
        var bInfo = badge[r.finish] || ["", r.finish || tr("req.none")];
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

  renderConvCache(m.conversation_cache, m.totals);
}

// Session totals card: five tiles, each with the one secondary reading that
// gives the raw counter meaning (per-request average, reuse share of the prompt,
// prefill vs decode split). Everything comes from /metrics totals.
function renderActivity(totals) {
  var el = $("act-facts");
  if (!el) return;
  el.innerHTML = QinferActivity.rows(totals || {}).map(function (r) {
    var value = "\u2013";
    if (r.value != null) {
      if (r.kind === "count") value = fmtNum(r.value);
      else if (r.kind === "seconds") value = fmtNum(r.value, 1) + '<small>' + esc(tr("unit.seconds")) + '</small>';
      else value = fmtNum(r.value) + '<small>' + esc(tr("unit.tokens")) + '</small>';
    }
    var sub = "";
    if (r.avg != null) sub = trp("act.perReq", fmtNum(r.avg));
    if (r.share != null) sub = trp("act.ofPrompt", fmtNum(r.share, 1) + "%");
    if (r.parts) sub = tr("act.prefill") + " " + fmtNum(r.parts[0], 1) + tr("unit.seconds") +
      " \u00b7 " + tr("act.decode") + " " + fmtNum(r.parts[1], 1) + tr("unit.seconds");
    return '<div class="act-tile">' +
      '<span class="st-metric__label">' + esc(tr(r.labelKey)) + '</span>' +
      '<span class="st-metric__value act-tile__value">' + value + '</span>' +
      '<span class="st-metric__sub">' + esc(sub) + '</span></div>';
  }).join("");
}

function fmtNum(n, d) {
  if (n == null || Number.isNaN(n)) return "\u2013";
  return Number(n).toLocaleString(undefined, {maximumFractionDigits: d || 0, minimumFractionDigits: d || 0});
}

// Conversation cache card. /metrics reports the server's real counters:
// {enabled, capacity, hits, misses, entries, tokens}; the reuse share comes from
// the session totals (reused / prompt_tokens), which is what the cache saves.
function renderConvCache(c, totals) {
  var card = $("cc-card");
  if (!card) return;
  card.style.display = c ? "" : "none";
  if (!c) return;
  var t = totals || {};
  var bars = $("cc-bars");
  if (bars) bars.hidden = !c.enabled;
  if (c.enabled) {
    var slotsText = $("cc-slots-text");
    if (slotsText) slotsText.textContent = fmtNum(c.entries) + " / " + fmtNum(c.capacity);
    var slotsBar = $("cc-slots-bar");
    if (slotsBar) slotsBar.style.width = (c.capacity ? Math.min(100, (100 * (c.entries || 0)) / c.capacity) : 0) + "%";
    var share = QinferActivity.reuseShare(t);
    var memText = $("cc-mem-text");
    if (memText) {
      memText.textContent = fmtNum(t.reused) + " / " + fmtNum(t.prompt_tokens) + " " + tr("unit.tokens") +
        (share != null ? " (" + fmtNum(share) + "%)" : "");
    }
    var memBar = $("cc-mem-bar");
    if (memBar) memBar.style.width = (share == null ? 0 : Math.min(100, share)) + "%";
  }
  var attempts = (c.hits || 0) + (c.misses || 0);
  var sum = $("cc-sum");
  if (sum) sum.textContent = attempts ? tr("cc.hitRate") + " " + fmtNum((100 * (c.hits || 0)) / attempts) + "%" : "";
  var factsEl = $("cc-facts");
  if (factsEl) {
    factsEl.innerHTML = factsHtml([
      [tr("cc.hits"), fmtNum(c.hits)],
      [tr("cc.misses"), fmtNum(c.misses)],
      [tr("cc.cachedTokens"), fmtNum(c.tokens) + " " + tr("unit.tokens")],
      [tr("cc.capacity"), fmtNum(c.capacity)],
    ]);
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
    renderRequestsTab(d);
    setStatusDot(true);
    // Update the header pill from metrics
    var live = d.live || {};
    var tok = live.tok_s != null ? fmtNum(live.tok_s, 1) : "\u2014";
    if (live.state !== "idle" && live.state !== "?") setPill(live.state, pillText(live.state, tok));
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
  if (!$("panel-chat")) return;
  initChat();
  initMonitor();
  loadModel();
  loadProps();
  pollMetrics();
  window.setInterval(pollMetrics, 1000);
});