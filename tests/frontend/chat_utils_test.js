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

if (failed) { console.log(failed + " failures"); process.exit(1); }
console.log("chat utils: all cases pass");
