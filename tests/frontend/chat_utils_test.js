// Unit tests for the chat-side helpers in src/serve/web/app.js: the context-tier
// history truncation and the zh/en i18n key parity. The helpers are plain browser
// JavaScript, so they are sliced out of the page script and exercised here without
// a browser. Run: node tests/frontend/chat_utils_test.js
const fs = require("fs");
const path = require("path");
const src = fs.readFileSync(path.join(__dirname, "..", "..", "src", "serve", "web", "app.js"), "utf8");

// --- the truncation helpers --------------------------------------------------
const s1 = src.indexOf("// Approximate a message's size in characters");
const e1 = src.indexOf("function collectMessages()", s1);
if (s1 < 0 || e1 < s1) throw new Error("missing truncation sources");
eval(src.slice(s1, e1) + "\n");

let failed = 0;
function report(why, detail) {
  failed++;
  console.log("FAIL " + why);
  console.log("  " + JSON.stringify(detail));
}
const m = (role, content) => ({ role: role, content: content });
const long = "x".repeat(400); // 400 chars -> 100 tokens at the ~4 chars/token estimate

function eq(name, got, want) {
  const g = JSON.stringify(got), w = JSON.stringify(want);
  if (g !== w) report(name, { got: got, want: want });
  else console.log("ok   " + name);
}

// msgChars: string content, block content, and non-text blocks.
eq("msgChars counts a string", msgChars(m("user", "abcd")), 4);
eq("msgChars sums text blocks",
   msgChars({ role: "user", content: [{ type: "text", text: "abcd" }, { type: "image_url", image_url: {} }] }), 4);
eq("msgChars ignores a missing message", msgChars(null), 0);

// Truncation: budget 0 or absent keeps everything.
eq("budget 0 = unchanged", truncateMessages([m("user", "a"), m("assistant", "b")], 0).length, 2);
eq("empty stays empty", truncateMessages([], 100), []);
eq("all fit", truncateMessages([m("user", "a"), m("assistant", "b")], 10000).length, 2);

// The oldest messages are dropped first; the newest always survives.
eq("drops oldest when over budget",
   truncateMessages([m("user", long), m("assistant", "short")], 50).map(x => x.role), ["assistant"]);
eq("keeps the tail run",
   truncateMessages([m("user", "a"), m("assistant", long), m("user", "b"), m("assistant", "c")], 50)
     .map(x => x.role), ["user", "assistant"]);
// A single message over budget is still kept (never send an empty conversation).
eq("single over budget kept", truncateMessages([m("user", long)], 10).length, 1);
// A leading system message is always kept, even alone.
eq("system head kept",
   truncateMessages([m("system", long), m("user", long)], 10).map(x => x.role), ["system", "user"]);

// --- QinferMD: Markdown tables ----------------------------------------------
function tr(k) { return k; }   // stub: the renderer uses tr() for the copy label
const s3 = src.indexOf("var QinferMD = {");
const e3 = src.indexOf("const I18N = {", s3);
if (s3 < 0 || e3 < s3) throw new Error("missing QinferMD source");
eval(src.slice(s3, e3) + "\n");
if (typeof QinferMD === "undefined") {
  report("QinferMD not defined", {});
} else {
  const t1 = QinferMD.render("| a | b |\n| --- | --- |\n| 1 | 2 |\n| 3 | 4 |");
  if (t1.indexOf('<table class="md-table">') < 0 || t1.indexOf("<th>a</th>") < 0 ||
      t1.indexOf("<td>1</td>") < 0 || t1.indexOf("<td>4</td>") < 0 ||
      t1.indexOf("<tbody>") < 0 || t1.indexOf("</tbody></table>") < 0) {
    report("table renders cells", { got: t1 });
  } else console.log("ok   markdown table renders head and body cells");

  // A pipe row with no separator is not a table.
  const t2 = QinferMD.render("a | b | c");
  if (t2.indexOf("<table") >= 0) report("pipe row without a separator is not a table", { got: t2 });
  else console.log("ok   a pipe row with no separator is not a table");

  // Cells keep inline formatting.
  const t3 = QinferMD.render("| h |\n| --- |\n| **bold** |");
  if (t3.indexOf("<strong>bold</strong>") < 0) report("table cell keeps inline formatting", { got: t3 });
  else console.log("ok   table cell keeps inline formatting");
}

// --- QinferMD: images and links ---------------------------------------------
if (typeof QinferMD !== "undefined") {
  const i1 = QinferMD.render("![cat](https://example.com/c.png)");
  if (i1.indexOf('<img class="md-img"') < 0 || i1.indexOf('src="https://example.com/c.png"') < 0 ||
      i1.indexOf('alt="cat"') < 0) {
    report("https image becomes an img", { got: i1 });
  } else console.log("ok   https image becomes an img");

  const i2 = QinferMD.render("![cat](http://example.com/c.png)");
  if (i2.indexOf("<img") >= 0) report("non-https image stays text", { got: i2 });
  else console.log("ok   a non-https image stays literal text");

  const l1 = QinferMD.render("[docs](https://example.com/x)");
  if (l1.indexOf('<a href="https://example.com/x"') < 0 || l1.indexOf("target=\"_blank\"") < 0) {
    report("plain link still renders", { got: l1 });
  } else console.log("ok   a plain link still renders with target=_blank");
}

// --- QinferMD: task lists ----------------------------------------------------
if (typeof QinferMD !== "undefined") {
  const tl = QinferMD.render("- [ ] todo\n- [x] done");
  if (tl.indexOf('<ul class="md-tasks">') < 0 || tl.indexOf('<li class="md-task">') < 0 ||
      tl.indexOf('<li class="md-task done">') < 0 || tl.indexOf("checked") < 0 ||
      tl.indexOf("> todo") < 0) {
    report("task list renders checked and unchecked items", { got: tl });
  } else console.log("ok   task list renders checked and unchecked items");

  const plainUl = QinferMD.render("- just a bullet");
  if (plainUl.indexOf('md-tasks') >= 0 || plainUl.indexOf('<li>just a bullet</li>') < 0) {
    report("a plain bullet is not a task", { got: plainUl });
  } else console.log("ok   a plain bullet stays a normal list item");
}

// --- QinferMD: intraword underscores -----------------------------------------
// Model output is full of snake_case identifiers (max_tokens, has_rings). A rule
// that lets "_" emphasise inside a word turns those into italics and eats the
// underscore, so the parameter name the model is talking about stops matching the
// code. CommonMark forbids intraword "_" emphasis; "_" next to a word boundary
// still emphasises, and "*" keeps its intraword behaviour.
if (typeof QinferMD !== "undefined") {
  ["columns planet, has_rings and nothing else",
   "a_snake_case_name stays literal",
   "max_tokens: 512 and temperature: 0.7",
   "| planet | has_rings |",
   "a__b__c stays literal"].forEach(function (line) {
    const out = QinferMD.render(line);
    if (out.indexOf("<em>") >= 0 || out.indexOf("<strong>") >= 0) {
      report("no intraword emphasis for: " + line, { got: out });
    } else console.log("ok   no intraword emphasis in: " + line);
  });
  // The positive half of the rule: a real italic still renders.
  const ital = QinferMD.render("this is _italic text_ here");
  if (ital.indexOf("<em>italic text</em>") < 0) report("boundary underscore still emphasises", { got: ital });
  else console.log("ok   boundary underscore still emphasises");
  // Two adjacent italics must both render (the rule consumes the leading space).
  const pair = QinferMD.render("a _b_ and _c_ end");
  if ((pair.match(/<em>/g) || []).length !== 2) report("adjacent italics both render", { got: pair });
  else console.log("ok   adjacent italics both render");
  // Underscore emphasis that opens a line is still emphasis (nothing precedes it).
  const lead = QinferMD.render("_starts the line_");
  if (lead.indexOf("<em>starts the line</em>") < 0) report("leading underscore emphasises", { got: lead });
  else console.log("ok   leading underscore emphasises");
  // Asterisks keep intraword behaviour (CommonMark allows a*b*c).
  const star = QinferMD.render("a*b*c");
  if (star.indexOf("<em>b</em>") < 0) report("star emphasis still applies", { got: star });
  else console.log("ok   star emphasis still applies");
}

// --- QinferMD: fenced-code syntax highlighting ------------------------------
// The highlighter runs on RAW code and escapes as it wraps, so a payload inside a
// code fence can never come back as live HTML.
const hlJs = QinferMD.highlight('const total = 42;', 'js');
if (hlJs.indexOf('tok-kw') < 0) report("highlight marks a keyword", { got: hlJs });
else if (hlJs.indexOf('tok-num') < 0) report("highlight marks a number", { got: hlJs });
else console.log("ok   highlight marks keywords and numbers");

const hlEsc = QinferMD.highlight('var s = "<img src=x onerror=1>";', 'js');
if (hlEsc.indexOf('<img') >= 0 || hlEsc.indexOf('&lt;img') < 0) report("highlight escapes payloads", { got: hlEsc });
else console.log("ok   highlight escapes HTML payloads");

const hlCmt = QinferMD.highlight('x = 1  # note', 'py');
if (hlCmt.indexOf('tok-cmt') < 0) report("python # is a comment", { got: hlCmt });
else console.log("ok   python # line comment highlighted");

const hlPlain = QinferMD.highlight('plain <text> & more', '');
if (hlPlain.indexOf('&lt;text&gt;') < 0 || hlPlain.indexOf('&amp;') < 0) report("plain code still escaped", { got: hlPlain });
else console.log("ok   plain code is escaped verbatim");

// The copy button must still carry the RAW code, and the block must be highlighted.
const mdBlock = QinferMD.render('```js\nconst a = 1;\n```');
if (mdBlock.indexOf('tok-kw') < 0) report("rendered fenced block is highlighted", { got: mdBlock });
else if (mdBlock.indexOf('data-code="' + encodeURIComponent('const a = 1;\n') + '"') < 0) report("copy payload is raw", { got: mdBlock });
else console.log("ok   rendered fenced block highlights and keeps raw copy payload");

// --- i18n zh/en key parity ---------------------------------------------------
const s2 = src.indexOf("const I18N = {");
const e2 = src.indexOf("function tr(", s2);
if (s2 < 0 || e2 < s2) throw new Error("missing i18n source");
// eval returns the completion value, so the object survives even though `const`
// inside eval is block-scoped to that eval.
const I18N = eval(src.slice(s2, e2) + "\nI18N;");
if (typeof I18N === "undefined" || !I18N.zh || !I18N.en) {
  report("I18N not defined with zh/en", {});
} else {
  const zh = Object.keys(I18N.zh).sort(), en = Object.keys(I18N.en).sort();
  const onlyZh = zh.filter(k => !(k in I18N.en));
  const onlyEn = en.filter(k => !(k in I18N.zh));
  if (onlyZh.length || onlyEn.length) report("i18n key parity", { onlyZh: onlyZh, onlyEn: onlyEn });
  else console.log("ok   i18n zh/en define the same " + zh.length + " keys");
  // The assistant label is the product name in both languages.
  if (I18N.zh["chat.assistant"] !== "Q-Infer" || I18N.en["chat.assistant"] !== "Q-Infer") {
    report("assistant name is Q-Infer", { zh: I18N.zh["chat.assistant"], en: I18N.en["chat.assistant"] });
  } else console.log("ok   assistant label is Q-Infer in both languages");
  // trp() fills a "{n}" slot, so a key that offers the slot in one language must
  // offer it in the other too — a dictionary without the slot silently drops the
  // number it is supposed to carry.
  const slotMismatch = Object.keys(I18N.zh).filter(k =>
    I18N.zh[k].includes("{n}") !== (I18N.en[k] || "").includes("{n}"));
  if (slotMismatch.length) report("i18n {n} slot parity", { keys: slotMismatch });
  else console.log("ok   every {n} placeholder is present in both languages");
  // The placeholder helper itself: value lands where the dictionary says.
  {
    const at = src.indexOf("function trp(");
    if (at < 0) throw new Error("missing trp source");
    const end = src.indexOf("\n}", at);
    const realTr = tr;
    try {
      tr = (k) => "«" + (I18N.en[k] || k) + "»";
      eq("trp fills the slot", eval(src.slice(at, end + 2) + "\ntrp")(
        "act.perReq", "571"), "«" + I18N.en["act.perReq"].replace("{n}", "571") + "»");
    } finally { tr = realTr; }
  }
}

// --- the header pill ---------------------------------------------------------
// The pill is a badge: it names the state and the live rate. The "State" label
// belongs to the chat footer summary, so a pill that repeats it reads wrong.
{
  const a = src.indexOf("var STATE_TR = {");
  const p = src.indexOf("function pillText(", a);
  if (a < 0 || p < a) throw new Error("missing pill helpers");
  const end = src.indexOf("\n}", p);
  const realTr = tr;
  try {
    tr = (k) => I18N.en[k] || k;
    const pill = eval(src.slice(a, end + 2) + "\npillText;");
    eq("pill names the state and the rate",
       pill("generating", "82.4"), "Generating · 82.4 " + I18N.en["unit.tokS"]);
    eq("unknown state passes through", pill("queued", "0"),
       "Queued · 0 " + I18N.en["unit.tokS"]);
    if (pill("generating", "82.4").indexOf(I18N.en["mon.stateLine"]) === 0) {
      report("pill has no redundant state label", { got: pill("generating", "82.4") });
    } else console.log("ok   pill has no redundant state label");
  } finally { tr = realTr; }
}

// --- the sidebar title -------------------------------------------------------
// An untitled conversation stores an empty title, so changing the language
// re-labels it instead of freezing the words of whatever language created it.
{
  const p = src.indexOf("function convTitle(");
  if (p < 0) throw new Error("missing convTitle helper");
  const end = src.indexOf("\n}", p);
  const realTr = tr;
  try {
    tr = (k) => I18N.en[k] || k;
    const titleOf = eval(src.slice(p, end + 2) + "\nconvTitle;");
    eq("untitled shows the localized placeholder", titleOf({ title: "" }), I18N.en["conv.untitled"]);
    eq("a real title shows as typed", titleOf({ title: "Paris weather" }), "Paris weather");
  } finally { tr = realTr; }
  // create() storing a translated title is exactly the bug the sentinel removes.
  if (/title:\s*tr\("conv\.untitled"\)/.test(src)) {
    report("empty title is the untitled marker", { got: "create() stores a translated title" });
  } else console.log("ok   empty title is the untitled marker");
}

// --- the transcript turn -------------------------------------------------------
// A saved/replayed turn comes from one rendered bubble. Two things the UI shows
// must never become model context: an error line (the request failed, the model
// never said that) and the empty shell of a bubble that holds nothing.
{
  const p = src.indexOf("function msgFromBubble(");
  if (p < 0) throw new Error("missing msgFromBubble helper");
  const end = src.indexOf("\n}", p);
  const turn = eval(src.slice(p, end + 2) + "\nmsgFromBubble;");

  // A stand-in for a rendered bubble: classList + a body holding textContent.
  const bubble = (cls, text, extra) => Object.assign({
    classList: { contains: (c) => cls.split(" ").indexOf(c) >= 0 },
    querySelector: () => ({ textContent: text }),
  }, extra || {});

  eq("an error bubble is not a turn", turn(bubble("bubble error", "boom"), 1), null);
  eq("an empty assistant bubble is not a turn", turn(bubble("bubble assistant", ""), 1), null);
  eq("a user turn keeps its raw text",
     turn(bubble("bubble user", "**hi**", { _rawText: "**hi**" }), 7),
     { role: "user", content: "**hi**", ts: 7 });
  eq("a turn falls back to the given time",
     turn(bubble("bubble user", "hi"), 42).ts, 42);
  eq("a turn prefers its own time",
     turn(bubble("bubble user", "hi", { _ts: 9 }), 42).ts, 9);
  eq("assistant thinking rides along",
     turn(bubble("bubble assistant", "ans", { _thinkingText: "why" }), 1).thinking, "why");

  // Attachments are what makes a resumed conversation still hold its image.
  const img = { kind: "image", name: "dot.png", url: "data:image/png;base64,AA" };
  const file = { kind: "file", name: "a.txt", text: "hello" };
  eq("a text-free image turn is still a turn",
     turn(bubble("bubble user", "", { _attachments: [img] }), 1),
     { role: "user", content: "", ts: 1, attachments: [img] });
  eq("attachments keep both kinds",
     turn(bubble("bubble user", "q", { _attachments: [img, file] }), 1).attachments.length, 2);
  eq("an assistant bubble stores no attachments",
     turn(bubble("bubble assistant", "a", { _attachments: [img] }), 1).attachments, undefined);

  // The same shape is what the sidebar save and the request builder read, so a
  // divergence between them is the bug this helper exists to prevent.
  ["function collectMessages()", "saveCurrent: function"].forEach(function (fn) {
    const at = src.indexOf(fn);
    if (at < 0) { report("missing " + fn, {}); return; }
    const stop = src.indexOf("\n}", at);
    const body = src.slice(at, stop);
    if (body.indexOf("msgFromBubble(") < 0) {
      report(fn.split("(")[0].replace("saveCurrent: function", "saveCurrent") + " uses the shared turn helper",
             { got: "no msgFromBubble call" });
    } else console.log("ok   " + fn + " uses the shared turn helper");
  });
}

// --- the storage-quota fallback ------------------------------------------------
// Images are data URLs and localStorage caps out around 5 MB, so a save can fail.
// Dropping the image payloads keeps the text of the conversation.
{
  const p = src.indexOf("function withoutImages(");
  if (p < 0) throw new Error("missing withoutImages helper");
  const end = src.indexOf("\n}", p);
  const strip = eval(src.slice(p, end + 2) + "\nwithoutImages;");
  const conv = {
    id: "c", title: "t",
    messages: [
      { role: "user", content: "q", attachments: [{ kind: "image", name: "a", url: "data:..." },
                                                  { kind: "file", name: "b", text: "x" }] },
      { role: "assistant", content: "a" },
    ],
  };
  const out = strip([conv]);
  eq("images are dropped", out[0].messages[0].attachments.length, 1);
  eq("text attachments survive", out[0].messages[0].attachments[0].kind, "file");
  eq("other turns are untouched", out[0].messages[1].content, "a");
  if (conv.messages[0].attachments.length !== 2) report("the original is not mutated", {});
  else console.log("ok   the original is not mutated");
}

// --- sampler presets + persistence -------------------------------------------
// The block is pure apart from localStorage/sessionStorage, which are stubbed
// with in-memory objects before the eval.
const sp = src.indexOf("var QinferSampler = {");
const spEnd = src.indexOf("\n};", sp);
if (sp < 0 || spEnd < sp) {
  report("QinferSampler block present", { at: sp });
} else {
  const lsData = {}, ssData = {};
  globalThis.localStorage = {
    getItem: k => (k in lsData ? lsData[k] : null),
    setItem: (k, v) => { lsData[k] = String(v); },
    removeItem: k => { delete lsData[k]; },
  };
  globalThis.sessionStorage = {
    getItem: k => (k in ssData ? ssData[k] : null),
    setItem: (k, v) => { ssData[k] = String(v); },
    removeItem: k => { delete ssData[k]; },
  };
  const S = eval(src.slice(sp, spEnd + 3) + "\nQinferSampler;");
  if (!S || !S.PRESETS || !S.matchPreset || !S.load || !S.save) {
    report("QinferSampler shape", { keys: S && Object.keys(S) });
  } else {
    // The balanced preset is the shipped default, so it must match the drawer
    // defaults as written in index.html.
    const idxHtml = require("fs").readFileSync(require("path").join(__dirname, "../../src/serve/web/index.html"), "utf8");
    const defs = {};
    for (const m of idxHtml.matchAll(/id="s-([a-z-]+)"[^>]*value="([^"]+)"/g)) defs[m[1]] = m[2];
    const balanced = S.PRESETS.balanced;
    const defsAgree = Number(defs.temp) === balanced.temperature && Number(defs["top-p"]) === balanced.top_p
      && Number(defs["top-k"]) === balanced.top_k && Number(defs["min-p"]) === balanced.min_p;
    if (!defsAgree) report("balanced preset equals the drawer defaults", { drawer: defs, balanced: balanced });
    else console.log("ok   balanced preset equals the drawer defaults");

    if (S.matchPreset(S.PRESETS.balanced) !== "balanced") report("matchPreset finds balanced", {});
    else if (S.matchPreset(Object.assign({}, S.PRESETS.precise)) !== "precise") report("matchPreset finds precise", {});
    else {
      const tweaked = Object.assign({}, S.PRESETS.balanced, { temperature: 0.95 });
      if (S.matchPreset(tweaked) !== "custom") report("matchPreset says custom after a tweak", { got: S.matchPreset(tweaked) });
      else console.log("ok   matchPreset names the built-ins and calls a tweak custom");
    }

    // Legacy per-tab value is migrated into localStorage exactly once.
    ssData[src.match(/LEGACY_KEY: "([^"]+)"/)[1]] = JSON.stringify({ temperature: 0.5, think: "low" });
    const migrated = S.load();
    const legacyKey = src.match(/LEGACY_KEY: "([^"]+)"/)[1], newKey = src.match(/STORAGE_KEY: "([^"]+)"/)[1];
    if (!migrated || migrated.temperature !== 0.5 || migrated.think !== "low") report("legacy sampler migrates", { got: migrated });
    else if (!lsData[newKey]) report("migration writes the new key", { ls: lsData });
    else if (legacyKey in ssData) report("migration removes the legacy key", { ss: ssData });
    else console.log("ok   legacy sessionStorage sampler migrates to localStorage once");

    // Round trip through save/load.
    S.save({ temperature: 1.1, think: "off" });
    const back = S.load();
    if (!back || back.temperature !== 1.1 || back.think !== "off") report("sampler save/load round trip", { got: back });
    else console.log("ok   sampler save/load round trips");
  }
}

// --- QinferSpeed: the monitor speed card value, and where its line comes from ---
{
  const q = src.indexOf("var QinferSpeed = {");
  if (q < 0) throw new Error("missing QinferSpeed source");
  const qEnd = src.indexOf("\n};", q);
  const QS = eval(src.slice(q, qEnd + 3) + "\nQinferSpeed;");
  // Card value: the live rate while generating, the session mean once idle — the
  // card must never read empty while the server has a rate to show.
  const c1 = QS.card({ state: "generating", tok_s: 100, tok_s_mean: 87 });
  if (c1.value !== 100 || c1.labelKey !== "mon.decodeNow") report("card shows the live rate while generating", { got: c1 });
  else console.log("ok   card shows the live rate while generating");
  const c2 = QS.card({ state: "idle", tok_s: 87.2, tok_s_mean: 87.2 });
  if (c2.value !== 87.2 || c2.labelKey !== "mon.decodeMean") report("card falls back to the session mean when idle", { got: c2 });
  else console.log("ok   card falls back to the session mean when idle");
  const c3 = QS.card({ state: "idle", tok_s: 60, tok_s_mean: null });
  if (c3.value !== 60 || c3.labelKey !== "mon.decodeLast") report("card names the last request when there is no mean", { got: c3 });
  else console.log("ok   card names the last request when there is no mean");
  const c4 = QS.card({ state: "idle", tok_s: null, tok_s_mean: null });
  if (c4.value !== null) report("card stays blank with no data at all", { got: c4 });
  else console.log("ok   card stays blank with no data at all");

  // The line under the card. It used to be assembled in the browser from the
  // request ring, which draws one step per request and nothing at all between
  // them; every other card reads the server's 1 Hz history window. The decode
  // mean now does the same, so the two rate charts share one source of truth.
  const mmStart = src.indexOf("var MON_METRICS = [");
  const mmEnd = src.indexOf("\n];", mmStart);
  if (mmStart < 0 || mmEnd < mmStart) throw new Error("missing MON_METRICS");
  const MON = eval(src.slice(mmStart, mmEnd + 3) + "\nMON_METRICS;");
  const byKey = (k) => MON.filter((m) => m.key === k)[0];
  eq("the decode-mean line reads the server history", byKey("speed").series, "tok_s_mean");
  eq("the prefill line reads the server history", byKey("prefill").series, "prefill_tok_s_mean");
  eq("QinferSpeed no longer assembles its own series", typeof QS.series, "undefined");
  if (src.indexOf('m.key === "speed"') >= 0) {
    report("the render loop has no per-card special case", { got: 'still branches on m.key === "speed"' });
  } else console.log("ok   the render loop has no per-card special case");
}

// --- chat scrolling: one writer, and it knows when to stop following ----------
// Following the newest turn is what makes a long reply readable, but yanking the
// viewport while someone is reading history is worse than not following at all.
// nearBottom() is the whole decision, so it is asserted here.
{
  const p = src.indexOf("function nearBottom(");
  if (p < 0) throw new Error("missing nearBottom helper");
  const end = src.indexOf("\n}", p);
  const nb = eval(src.slice(p, end + 2) + "\nnearBottom;");
  const at = (scrollTop) => ({ scrollTop: scrollTop, scrollHeight: 1000, clientHeight: 100 });
  eq("sitting at the end is near", nb(at(900), 80), true);
  eq("just inside the slack is near", nb(at(830), 80), true);
  eq("just outside the slack is not", nb(at(800), 80), false);
  eq("scrolled up into history is not", nb(at(120), 80), false);
  eq("a missing scroller counts as near", nb(null, 80), true);
  // The scroller is #chat-scroll; #msg-list has no overflow of its own, so every
  // write to its scrollTop was a no-op and the newest turn could land below the
  // fold mid-stream.
  if (src.indexOf('$("msg-list").scrollTop') >= 0) {
    report("only the real scroller is scrolled", { got: 'still writes #msg-list.scrollTop' });
  } else console.log("ok   only the real scroller is scrolled");
}

// --- QinferQueue: the line-up behind the running reply ------------------------
// The queue is the one place a message can sit without being sent, so the rules
// that matter are the ones about *when* it moves: never while a reply is running,
// never while paused, and never by silently dropping what it cannot hold.
{
  const q = src.indexOf("var QinferQueue = {");
  const qEnd = src.indexOf("\n};", q);
  if (q < 0 || qEnd < q) throw new Error("missing QinferQueue source");
  const realLS = global.localStorage;
  const store = {};
  global.localStorage = {
    getItem: (k) => (k in store ? store[k] : null),
    setItem: (k, v) => { store[k] = String(v); },
    removeItem: (k) => { delete store[k]; },
  };
  try {
    const Q = eval(src.slice(q, qEnd + 3) + "\nQinferQueue;");
    let items = [];
    const first = Q.push(items, "  first  ", []);
    eq("push trims the text", first.text, "first");
    eq("push appends", items.length, 1);
    eq("push refuses an empty message", Q.push(items, "   ", []), null);
    eq("push keeps an attachment-only message", Q.push(items, "", [{ kind: "image", url: "data:x" }]) !== null, true);
    // The cap exists so a full queue is announced rather than quietly eaten.
    let many = [];
    for (let i = 0; i < Q.CAP; i++) Q.push(many, "m" + i, []);
    eq("a full queue refuses more", Q.push(many, "one too many", []), null);
    eq("and holds exactly its cap", many.length, Q.CAP);

    const a = Q.push([], "a", []), b = Q.push([], "b", []), c = Q.push([], "c", []);
    const list = [a, b, c];
    eq("remove drops one by id", Q.removeId(list, b.id).map((i) => i.text), ["a", "c"]);
    eq("move reorders forwards", Q.move(list, 0, 2).map((i) => i.text), ["b", "c", "a"]);
    eq("move reorders backwards", Q.move(list, 2, 0).map((i) => i.text), ["c", "a", "b"]);
    eq("an out-of-range move is a no-op", Q.move(list, 5, 0).map((i) => i.text), ["a", "b", "c"]);
    if (list.map((i) => i.text).join() !== "a,b,c") report("move leaves the input alone", { got: list });
    else console.log("ok   move leaves the input alone");

    // Pause gates the queue and nothing else: it must not fire the next turn while
    // a reply runs, and a paused queue stays put even with the engine idle.
    eq("advances when idle and unpaused", Q.canAdvance(false, false), true);
    eq("a running reply blocks the queue", Q.canAdvance(false, true), false);
    eq("a paused queue stays put", Q.canAdvance(true, false), false);
    eq("pause never resumes mid-reply", Q.canAdvance(true, true), false);

    Q.write([a], true);
    const back = Q.read();
    eq("the items survive a reload", back.items.map((i) => i.text), ["a"]);
    eq("so does the paused flag", back.paused, true);
    store["qinfer.queue"] = "{not json";
    eq("a corrupt store reads as empty", Q.read().items.length, 0);
  } finally { global.localStorage = realLS; }
}

// --- QinferTime: the clock a message bubble carries ---------------------------
// Bubbles store epoch milliseconds; the label has to say "today at 20:17" for
// this session's traffic and fall back to a date for anything older. The locale
// is passed in so the assertions do not depend on the machine running them.
{
  const t = src.indexOf("var QinferTime = {");
  if (t < 0) throw new Error("missing QinferTime source");
  const tEnd = src.indexOf("\n};", t);
  const QT = eval(src.slice(t, tEnd + 3) + "\nQinferTime;");
  const now = new Date(2026, 9, 15, 20, 17, 30).getTime();   // 2026-10-15 20:17:30 local
  eq("same day shows the clock only", QT.clock(now - 60000, now, "en-GB"), "20:16");
  eq("yesterday shows the date", QT.clock(now - 26 * 3600 * 1000, now, "en-GB"), "14 Oct, 18:17");
  eq("missing stamp shows nothing", QT.clock(null, now, "en-GB"), "");
  eq("full() carries date and time", QT.full(now, "en-GB"), "15 Oct 2026, 20:17:30");
}

// --- QinferKeys: one place decides what a shortcut means ----------------------
// The handler only executes the action name; matching stays pure so the
// modifier combinations and the "do not steal typing" rule are testable.
{
  const k = src.indexOf("var QinferKeys = {");
  if (k < 0) throw new Error("missing QinferKeys source");
  const kEnd = src.indexOf("\n};", k);
  const QK = eval(src.slice(k, kEnd + 3) + "\nQinferKeys;");
  const ev = (key, mod, target) => Object.assign({key: key, target: target || {tagName: "BODY"}}, mod || {});
  eq("Cmd/Ctrl+Shift+O opens a conversation", QK.match(ev("O", {metaKey: true, shiftKey: true})), "newConv");
  eq("Cmd/Ctrl+O alone is left to the browser", QK.match(ev("O", {metaKey: true})), null);
  eq("Cmd/Ctrl+K focuses the composer", QK.match(ev("k", {ctrlKey: true})), "focusInput");
  eq("Cmd/Ctrl+B toggles the sidebar", QK.match(ev("b", {ctrlKey: true})), "toggleSidebar");
  eq("bare ? asks for help", QK.match(ev("?")), "help");
  eq("? while typing is not a shortcut",
     QK.match(ev("?", null, {tagName: "TEXTAREA"})), null);
  eq("Escape always reports itself", QK.match(ev("Escape", null, {tagName: "INPUT"})), "escape");
  eq("a plain letter is no shortcut", QK.match(ev("k")), null);
  eq("unknown keys are no shortcut", QK.match(ev("F9", {altKey: true})), null);
  eq("typing is detected in editable hosts",
     [QK.isTyping({tagName: "INPUT"}), QK.isTyping({tagName: "DIV", isContentEditable: true}),
      QK.isTyping({tagName: "DIV"}), QK.isTyping(null)], [true, true, false, false]);
  // The ? dialog and the dispatch share one table, so the help panel can never
  // advertise a shortcut the handler does not implement.
  const mac = QK.rows(true), win = QK.rows(false);
  eq("help lists the same actions on both platforms", mac.map(r => r.labelKey), win.map(r => r.labelKey));
  eq("mac shows the command glyph", mac[0].keys.indexOf("⌘"), 0);
  eq("windows shows Ctrl", win[0].keys.indexOf("Ctrl"), 0);
  const acts = ["keys.focusInput", "keys.newConv", "keys.toggleSidebar", "keys.help",
                "keys.escape", "keys.send", "keys.newline"];
  eq("help covers every bound action", mac.map(r => r.labelKey), acts);
  eq("every help row names its keys", mac.every(r => typeof r.keys === "string" && r.keys.length > 0), true);
  // Escape walks the open layers top-down and only stops a run when nothing is open.
  eq("firstOpen picks the topmost layer",
     QK.firstOpen([{name: "confirm", open: false}, {name: "help", open: true}, {name: "think", open: true}]),
     "help");
  eq("firstOpen is null when nothing is open",
     QK.firstOpen([{name: "help", open: false}, {name: "think", open: false}]), null);
  eq("firstOpen survives an empty layer list", QK.firstOpen([]), null);
}

// --- QinferSystem: the per-conversation instruction ---------------------------
// A conversation may carry its own system prompt. It becomes the leading system
// message of the payload — and replaces one already there rather than stacking.
{
  const y = src.indexOf("var QinferSystem = {");
  if (y < 0) throw new Error("missing QinferSystem source");
  const yEnd = src.indexOf("\n};", y);
  const QS = eval(src.slice(y, yEnd + 3) + "\nQinferSystem;");
  const hist = [{role: "user", content: "hi"}, {role: "assistant", content: "yo"}];
  eq("no instruction leaves the payload alone", QS.apply("", hist), hist);
  eq("blank instruction leaves the payload alone", QS.apply("   \n ", hist), hist);
  eq("missing instruction leaves the payload alone", QS.apply(undefined, hist), hist);
  eq("instruction becomes the head", QS.apply("Be terse.", hist)[0], {role: "system", content: "Be terse."});
  eq("instruction keeps the history after it", QS.apply("Be terse.", hist).length, 3);
  eq("instruction is trimmed", QS.apply("  Be terse.  ", hist)[0].content, "Be terse.");
  eq("a second apply replaces the head", QS.apply("Be warm.", QS.apply("Be terse.", hist))[0].content, "Be warm.");
  eq("replacement keeps one system message",
     QS.apply("Be warm.", QS.apply("Be terse.", hist)).filter(m => m.role === "system").length, 1);
  eq("empty history still carries the instruction", QS.apply("Be terse.", []).length, 1);
}

// --- QinferActivity: the requests page session totals -------------------------
// /metrics reports totals as raw counters; the card turns them into labelled rows
// and a reuse share. Nothing here may invent a number the server did not send.
{
  const a = src.indexOf("var QinferActivity = {");
  if (a < 0) throw new Error("missing QinferActivity source");
  const aEnd = src.indexOf("\n};", a);
  const QA = eval(src.slice(a, aEnd + 3) + "\nQinferActivity;");
  const totals = { requests: 7, prompt_tokens: 4000, completion_tokens: 900, reused: 2500, prompt_ms: 1500, decode_ms: 3000 };
  const labels = QA.rows(totals).map((r) => r.labelKey);
  eq("rows name every session total", labels,
    ["act.requests", "act.prompt", "act.completion", "act.reused", "act.busy"]);
  eq("busy sums prefill and decode into seconds", QA.rows(totals)[4].value, 4.5);
  eq("reuse share comes from reused over prompt", QA.reuseShare(totals), 62.5);
  eq("reuse share is null before the first prompt", QA.reuseShare({ reused: 0, prompt_tokens: 0 }), null);
  eq("busy is null before the first request", QA.rows({})[4].value, null);
  eq("rows survive missing totals", QA.rows(undefined).map((r) => r.value), [null, null, null, null, null]);
  // Each row also carries the one secondary reading that gives the number meaning.
  const rows = QA.rows(totals);
  eq("prompt row carries the per-request average", rows[1].avg, 4000 / 7);
  eq("completion row carries the per-request average", rows[2].avg, 900 / 7);
  eq("reused row carries the share of prompt", rows[3].share, 62.5);
  eq("busy row splits prefill from decode", rows[4].parts, [1.5, 3]);
  eq("no requests means no averages", QA.rows({prompt_tokens: 10}).slice(1, 3).map((r) => r.avg), [null, null]);
}

// --- i18n three-way lint: used keys must exist, defined keys must be used ------
// Scans every tr("...")/trp("...") in app.js and every data-i18n* attribute in
// index.html. Keys built dynamically at runtime go on the whitelist with a
// justification.
const html = fs.readFileSync(path.join(__dirname, "../../src/serve/web/index.html"), "utf8");
const used = new Set();
for (const m of src.matchAll(/\btr\("([a-zA-Z0-9_.]+)"/g)) used.add(m[1]);
for (const m of src.matchAll(/\btrp\("([a-zA-Z0-9_.]+)"/g)) used.add(m[1]);
for (const m of html.matchAll(/data-i18n(?:-ph|-title|-aria)?="([a-zA-Z0-9_.]+)"/g)) used.add(m[1]);
// trState maps runtime states to fixed keys; the map is literal so its values
// count as used (STATE_TR is a plain object literal in app.js).
const st = src.indexOf("const STATE_TR");
if (st >= 0) {
  const stEnd = src.indexOf("};", st);
  for (const m of src.slice(st, stEnd).matchAll(/"([a-zA-Z0-9_.]+)"/g)) used.add(m[1]);
}
// MON_METRICS-style tables consume keys through a labelKey property (tr(m.labelKey));
// every labelKey value in a table literal counts as used.
for (const m of src.matchAll(/labelKey:\s*"([a-zA-Z0-9_.]+)"/g)) used.add(m[1]);
const defined = new Set(Object.keys(I18N.zh));
const I18N_WHITELIST = []; // runtime-constructed keys only; keep empty
const missing = [...used].filter(k => !defined.has(k)).sort();
const dead = [...defined].filter(k => !used.has(k) && !I18N_WHITELIST.includes(k)).sort();
if (missing.length) report("i18n used-but-undefined keys", { missing: missing });
else console.log("ok   every used i18n key is defined (" + used.size + " used)");
if (dead.length) report("i18n dead dictionary keys", { dead: dead });
else console.log("ok   no dead i18n keys");

if (failed) { console.log(failed + " failures"); process.exit(1); }
console.log("chat utils: all cases pass");
