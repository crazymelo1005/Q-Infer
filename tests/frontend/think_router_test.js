// Unit and fuzz tests for the streaming think-tag router in src/serve/web/app.js.
// The router is plain browser JavaScript with no DOM dependency, so it is sliced
// out of the page script and exercised here - that pins the streaming attribution
// down without a browser. Run: node tests/frontend/think_router_test.js
const fs = require("fs");
const path = require("path");
const src = fs.readFileSync(path.join(__dirname, "..", "..", "src", "serve", "web", "app.js"), "utf8");

// Everything from the control-token table to the Markdown renderer: the router and
// the helpers it depends on, in source order.
const start = src.indexOf("var SPECIAL_TOKENS = [");
const end = src.indexOf("// Render Markdown in an assistant message body", start);
if (start < 0 || end < start) throw new Error("missing router sources");
eval(src.slice(start, end) + "\n");

// Tags are written by concatenation: the tool channel that carries this file to
// disk strips literal tag strings.
const O = "<" + "think>", C = "<" + "/think>", END = "<|im_" + "end|>";
const CONTROL = [O, C, END, "<" + "|im_start|>", "<|im_sep|>"];

function feed(chunks, plain) {
  const t = ensureThinkStream(!!plain);
  let firstVisible = -1;
  for (let i = 0; i < chunks.length; i++) {
    const v = t.push(chunks[i]);            // never skip a delta
    if (v && firstVisible < 0) firstVisible = i;
  }
  t.finish();
  return {answer: t.answer, thinking: t.thinking, firstVisible};
}

// `firstVisible` is the index of the first delta after which either panel holds
// text: the router streams a provisional run live, so it is 0 as soon as the
// first content delta lands, and only a content-free delta (a lone delimiter)
// pushes it later.
const cases = [
  // What the live Strata engine sends in thinking mode: reasoning first, then a
  // bare closing tag, then the answer, then the end-of-turn marker. The leading
  // run streams into the thinking panel and the closing tag confirms it.
  ["closing tag only (observed protocol)",
    ["The user wants", " 108", "\n", C, "\n\n1", "08", END], false,
    {answer: "\n\n108", thinking: "The user wants 108\n", firstVisible: 0}],
  ["opening tag then closing tag",
    ["Sure. ", O, "thinking here", C, "final"], false,
    {answer: "Sure. final", thinking: "thinking here", firstVisible: 0}],
  ["reply with no tags at all",
    ["plain", " answer"], false,
    {answer: "plain answer", thinking: "", firstVisible: 0}],
  ["tags split across deltas",
    ["x", O + "thin", "ker", C + "ans"], false,
    {answer: "xans", thinking: "thinker", firstVisible: 0}],
  ["closing tag split across deltas",
    ["reaso", "ning<", "/thi", "nk>a"], false,
    {answer: "a", thinking: "reasoning", firstVisible: 0}],
  ["thinking the model never closes falls back to the answer",
    [O + "only reasoning"], false,
    {answer: "only reasoning", thinking: "", firstVisible: 0}],
  ["bare closing tag with empty thinking",
    [C, "tail"], false,
    {answer: "tail", thinking: "", firstVisible: 1}],
  ["leading text before a bare closing tag is thinking",
    ["weird ", C, " tail"], false,
    {answer: " tail", thinking: "weird ", firstVisible: 0}],
  ["answer text, then a fake opening tag, then more answer",
    ["11 is prime", O, " really not", "x", "y"], false,
    {answer: "11 is prime really notxy", thinking: "", firstVisible: 0}],
  ["control tokens never reach the bubble",
    ["answer", END], false,
    {answer: "answer", thinking: "", firstVisible: 0}],
  ["a control token split across deltas still never reaches the bubble",
    ["answer", "<|im_", "end", "|>"], false,
    {answer: "answer", thinking: "", firstVisible: 0}],
  // The other reading of the protocol: answer text first, then a thinking block
  // introduced by its opening tag. The head stays in the reply because a closing
  // delimiter settles it.
  ["thinking block after answer text",
    ["Answer head ", O, "deliberate", C, " tail"], false,
    {answer: "Answer head  tail", thinking: "deliberate", firstVisible: 0}],
  // An unterminated block that starts after reply text is shown as the answer: the
  // router would rather show a stray run than hide content the model produced.
  ["opening tag mid-reply never followed by a close",
    ["head ", O, "still answer"], false,
    {answer: "head still answer", thinking: "", firstVisible: 0}],
  ["orphan closing delimiter after the thinking run",
    [O, "t", C, "a", C, "b"], false,
    {answer: "ab", thinking: "t", firstVisible: 1}],
  // A leading run whose thinking never settled stays visible: the router would
  // rather show a stray fragment than an empty bubble.
  ["a delimiter fragment at the end of an undecided reply stays visible",
    ["answer", "<thi"], false,
    {answer: "answer<thi", thinking: "", firstVisible: 0}],
  // Once the reply is settled the same fragment reads as control text and hides.
  ["a delimiter fragment after the thinking run stays hidden",
    [O, "t", C, "answer", "<thi"], false,
    {answer: "answer", thinking: "t", firstVisible: 1}],
  ["a closing delimiter fragment after the thinking run stays hidden",
    [O, "t", C, "answer", "</thi"], false,
    {answer: "answer", thinking: "t", firstVisible: 1}],
  ["an unfinished close at the end leaves the guess alone",
    ["reasoning</th"], false,
    {answer: "reasoning", thinking: "", firstVisible: 0}],
  ["several delimiters inside one delta",
    ["a" + C + "b" + C + "c"], false,
    {answer: "bc", thinking: "a", firstVisible: 0}],
  // In nothinking mode the engine still ends the turn with a ChatML marker. Held
  // back as a possible split token, it is dropped when the stream ends rather than
  // painted into the bubble.
  ["nothinking mode paints every delta",
    ["1", "2", END], true,
    {answer: "12", thinking: "", firstVisible: 0}],
  ["nothinking mode hides a ChatML token split across deltas",
    ["1", "2", "<|im_", "end", "|>"], true,
    {answer: "12", thinking: "", firstVisible: 0}],
  ["nothinking mode never holds back",
    ["abc", "def", "ghi"], true,
    {answer: "abcdefghi", thinking: "", firstVisible: 0}],
];

let failed = 0;
function report(why, detail) {
  failed++;
  console.log("FAIL " + why);
  console.log("  " + JSON.stringify(detail));
}

for (const [name, chunks, plain, want] of cases) {
  const got = feed(chunks, plain);
  if (got.answer !== want.answer || got.thinking !== want.thinking ||
      got.firstVisible !== want.firstVisible) {
    report(name, {got: got, want: want, chunks: chunks});
  } else {
    console.log("ok   " + name);
  }
}

// --- Invariants over generated streams ---------------------------------------
// A router can be wrong about an ambiguous fragment, but it must never paint a
// control token, never alter text that contains no angle bracket, and always
// settle. A deterministic PRNG keeps a failure reproducible by seed.
const PIECES = [
  "alpha", " beta", "\n", "108", " x=1", " ", "<", "<<", ">>>", "</", "<thi",
  "nk>", O, C, END, "think", "</thi", "<|im_", "end|>", " gamma", " ", "0",
  "<|im_sep", "|>", "<|im_start|>",
];

function makeRnd(seed) {
  let s = seed >>> 0;
  return function () {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 4294967296;
  };
}

function randomText(rnd) {
  const n = 1 + Math.floor(rnd() * 14);
  const parts = [];
  for (let i = 0; i < n; i++) parts.push(PIECES[Math.floor(rnd() * PIECES.length)]);
  return parts.join("");
}

function splitDeltas(text, rnd) {
  const out = [];
  let i = 0;
  while (i < text.length) {
    const len = 1 + Math.floor(rnd() * 6);
    out.push(text.substring(i, i + len));
    i += len;
  }
  return out;
}

function checkInvariants(streams, plain, limit) {
  var shown = 0;
  for (var seed = 1; seed <= streams; seed++) {
    var rnd = makeRnd(plain ? seed + 4242 : seed);
    var full = randomText(rnd);
    var deltas = splitDeltas(full, rnd);
    if (deltas.join("") !== full) {
      report("fuzz harness lost text for seed " + seed, {deltas: deltas, full: full});
      continue;
    }
    var t = ensureThinkStream(!!plain);
    for (const d of deltas) t.push(d);
    t.finish();
    var painted = t.answer + "|" + t.thinking;
    var problems = [];
    // The reply panel must never show a control token: the engine ends a turn with
    // a ChatML marker, and the thinking delimiters are structure, not content.
    for (const c of CONTROL) {
      if (t.answer.indexOf(c) >= 0) problems.push("control token in the reply");
    }
    if (t.thinking.indexOf(O) >= 0 || t.thinking.indexOf(C) >= 0) {
      problems.push("thinking delimiter in the thinking panel");
    }
    // Text without an angle bracket can never be mistaken for a delimiter, so it
    // is preserved exactly.
    if (full.indexOf("<") < 0 && painted !== full + "|") {
      problems.push("bracket-free stream altered");
    }
    if (t.state !== "answer") problems.push("state not settled");
    if (t.buf !== "") problems.push("buffer not drained");
    if (problems.length && shown++ < limit) {
      report("fuzz " + (plain ? "nothinking" : "thinking") + " seed " + seed,
             {problems: problems, deltas: deltas, painted: painted});
    } else if (problems.length) {
      failed++;
    }
  }
}

checkInvariants(20000, false, 8);
checkInvariants(5000, true, 8);
console.log("fuzz swept: 25000 generated streams");

// Display helpers.
if (!safePartialMarkdown("a ```js\ncode").endsWith("```")) {
  report("unterminated code fence is not closed for display", {});
} else console.log("ok   unterminated code fence closed for display");
if (safePartialMarkdown("a ```js\ncode\n```") !== "a ```js\ncode\n```") {
  report("completed code fence was altered", {});
} else console.log("ok   completed code fence untouched");

if (failed) {
  console.log(failed + " failures");
  process.exit(1);
}
console.log("think router: all " + cases.length + " cases and 25000 fuzz streams clean");
