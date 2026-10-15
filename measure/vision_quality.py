#!/usr/bin/env python3
"""Vision-arm quality A/B at ADR-009's distribution level (default-config tier).

The byte oracle (vision_oracle.py) is the gate for a *fixed-switch* build; the iGPU
(ggml-vulkan) arm diverges at fp16 accumulation by design, so its quality claim has to
be argued the way --kv int8 was: same images and prompts through the live service on
both arms, greedy decoding, long generations, then first-divergence position and
agreement rate per image, plus 100% self-repeatability inside each arm.

Images are ffmpeg lavfi renders - deterministic on purpose, so a rerun of either arm
reproduces its own bytes exactly and any answer drift is the encoder's, not the
harness's.

  run     vision_quality.py run  --label cpu --repeats 2 [--max-tokens 220] [--port 8080]
  compare vision_quality.py compare <cpu.json> <gpu.json>
"""

import argparse
import base64
import datetime
import json
import pathlib
import subprocess
import time
import urllib.request

REPO = pathlib.Path(__file__).resolve().parent.parent
WORK = REPO / "build-vision" / "quality"
RESULTS = REPO / "measure" / "results"

# (name, ffmpeg lavfi source, prompt).  Every prompt asks for a description long
# enough that a real divergence shows up past the opening words.
CASES = [
    ("grid", "color=red:size=512x512,drawgrid=w=64:h=64:color=white@0.9:t=2",
     "Describe this image in detail: colors, patterns, and structure."),
    ("bars", "smptebars=size=512x512",
     "List the colors you see from top to bottom and describe how the image is organized."),
    ("src", "testsrc2=size=512x512",
     "What is depicted in this image? Describe its layout and the elements you recognize."),
    ("text", "color=black:size=512x512,drawtext=text='VISION QUALITY CHECK 42':fontcolor=white:fontsize=48:x=64:y=232",
     "Transcribe any text you can read in this image, then describe the rest of the picture."),
    ("grad", "gradients=size=512x512",
     "Describe the color transitions and the overall visual pattern of this image."),
]


def make_case_images() -> dict:
    WORK.mkdir(parents=True, exist_ok=True)
    out = {}
    for name, src, _ in CASES:
        path = WORK / f"{name}.png"
        if not path.exists():
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", src,
                            "-frames:v", "1", str(path)], check=True)
        out[name] = path
    return out


def ask(port: int, image: pathlib.Path, prompt: str, max_tokens: int) -> dict:
    data = base64.b64encode(image.read_bytes()).decode()
    body = json.dumps({
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{data}"}}]}],
        "max_tokens": max_tokens, "temperature": 0.0, "top_p": 1.0, "stream": False,
    }).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=600) as r:
        payload = json.load(r)
    ms = (time.monotonic() - t0) * 1000.0
    text = payload["choices"][0]["message"].get("content", "")
    return {"text": text, "ms": round(ms, 1),
            "completion_tokens": payload.get("usage", {}).get("completion_tokens")}


def run_arm(label: str, port: int, repeats: int, max_tokens: int) -> dict:
    images = make_case_images()
    runs = []
    for name, _, prompt in CASES:
        for i in range(repeats):
            r = ask(port, images[name], prompt, max_tokens)
            runs.append({"image": name, "repeat": i, **r})
            print(f"{label}/{name}#{i}: {r['completion_tokens']} tokens in {r['ms']:.0f} ms")
    record = {"gate": "vision-quality", "date": datetime.date.today().isoformat(),
              "label": label, "max_tokens": max_tokens, "repeats": repeats, "runs": runs}
    out = RESULTS / f"{datetime.datetime.now():%Y%m%d}-vision-quality-{label}-环境2.json"
    out.write_text(json.dumps(record, indent=1) + "\n")
    print(f"wrote {out}")
    return record


def compare_texts(a: str, b: str) -> dict:
    wa, wb = a.split(), b.split()
    common = 0
    for x, y in zip(wa, wb):
        if x != y:
            break
        common += 1
    shorter = min(len(wa), len(wb)) or 1
    return {"words_a": len(wa), "words_b": len(wb), "common_prefix_words": common,
            "agreement": round(common / shorter, 4),
            "identical": a == b,
            "first_divergence_at": None if common == min(len(wa), len(wb)) else common}


def compare(cpu: dict, gpu: dict) -> int:
    by_key = lambda rec: {(r["image"], r["repeat"]): r["text"] for r in rec["runs"]}
    ca, ga = by_key(cpu), by_key(gpu)
    bad = 0
    print(f"{'image':<6} {'self-cpu':<9} {'self-gpu':<9} {'agree':<7} {'first-div':<9} identical")
    for name, _, _ in CASES:
        self_cpu = ca.get((name, 0)) == ca.get((name, 1))
        self_gpu = ga.get((name, 0)) == ga.get((name, 1))
        m = compare_texts(ca.get((name, 0), ""), ga.get((name, 0), ""))
        print(f"{name:<6} {str(self_cpu):<9} {str(self_gpu):<9} {m['agreement']:<7} "
              f"{str(m['first_divergence_at']):<9} {m['identical']}")
        if not (self_cpu and self_gpu):
            bad += 1
    print("\n-- answers (cpu #0 / gpu #0), truncated to 400 chars --")
    for name, _, _ in CASES:
        print(f"[{name}] cpu: {ca.get((name, 0), '')[:400]!r}")
        print(f"[{name}] gpu: {ga.get((name, 0), '')[:400]!r}\n")
    return 1 if bad else 0


def main() -> int:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--label", required=True)
    r.add_argument("--port", type=int, default=8080)
    r.add_argument("--repeats", type=int, default=2)
    r.add_argument("--max-tokens", type=int, default=220)
    c = sub.add_parser("compare")
    c.add_argument("cpu")
    c.add_argument("gpu")
    args = p.parse_args()

    if args.cmd == "run":
        run_arm(args.label, args.port, args.repeats, args.max_tokens)
        return 0
    return compare(json.loads(pathlib.Path(args.cpu).read_text()),
                   json.loads(pathlib.Path(args.gpu).read_text()))


if __name__ == "__main__":
    raise SystemExit(main())
