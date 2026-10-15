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

// --- i18n three-way lint: used keys must exist, defined keys must be used ------
// Scans every tr("...") in app.js and every data-i18n* attribute in index.html.
// Keys built dynamically at runtime go on the whitelist with a justification.
const html = fs.readFileSync(path.join(__dirname, "../../src/serve/web/index.html"), "utf8");
const used = new Set();
for (const m of src.matchAll(/tr\("([a-zA-Z0-9_.]+)"/g)) used.add(m[1]);
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
