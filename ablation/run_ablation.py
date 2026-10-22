#!/usr/bin/env python3
"""Prompt ablation for the flash result: which part of the flash prompt caps the design?

Source prompt: the exact flash request c2hls sent in the fir-site skills run
(logs/shim_20261007T023853/call_00005: system.txt + prompt.txt, 134 skills of the
_plus_gemm_flatten_v2 pack). Each variant cuts one component (exact substrings, asserted),
or uses the clean "best performance for U280" prompt with/without the same skill library.
Every variant uses the same flash system prompt, model and plain (tool-less) shim, and is
judged by the same plain-Vitis csim + csynth evaluator as the clean baselines.

  run_ablation.py generate [--reps 2]   # LLM calls via the plain shim -> <variant>_r<n>/kernel.cpp + eval dirs
  run_ablation.py submit                # sbatch csim + csynth for every generated kernel
  run_ablation.py table                 # collect results -> results.md
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SRC = ROOT / "logs/shim_20261007T023853/call_00005"
BENCH = Path("/scratch/asa582/workspaces/code_translation_c2hls/artifacts/pc2/autosa_mm_ijk_benches/n1024/autosa_mm")
EVAL_SRC = Path("/scratch/asa582/clean_u280/eval")
FENCE_RE = re.compile(r"```(?:cpp|c\+\+|c|hls)?[^\n]*\n(.*?)```", re.S)

GUIDANCE_CUTS = [
    # Closing paragraph of the flash guidance list.
    "Prefer a compact, synthesizable implementation over a heroic rewrite. For\n"
    "small PolyBench/HLSFactory-style kernels, simple pipelining, modest unroll,\n"
    "and a few local buffers often beat complex tiling/dataflow.\n",
    # Item 6: narrow-ABI / no wide-bus types.
    "and local staging while preserving the narrow host/testbench ABI by\n"
    "   default. Do NOT introduce `ap_uint<512>`, `MARS_WIDE_BUS_TYPE`, or\n"
    "   `memcpy_wide_bus_*` helpers unless the benchmark context explicitly says\n"
    "   the active harness supports a wide-bus ABI.\n",
    # Output format: anchor to the existing structure.
    "- Prefer surgical edits (pragmas, bundles, II, modest unroll, local buffers)\n"
    "  on the existing structure over inventing a shorter incomplete rewrite.\n",
]
GUIDANCE_REPLACEMENTS = ["", "and local staging.\n", ""]


def load_source() -> tuple[str, str]:
    return (SRC / "system.txt").read_text(), (SRC / "prompt.txt").read_text()


def cut_between(text: str, start: str, end: str) -> str:
    i, j = text.index(start), text.index(end)
    assert i < j, (start, end)
    return text[:i] + text[j:]


def skills_block(prompt: str) -> str:
    return prompt[prompt.index("GLOBAL SKILL LIBRARY"):]


def variants() -> dict[str, str]:
    _, full = load_source()
    assert full.count("RESOURCE CONSTRAINTS") == 1 and full.count("GLOBAL SKILL LIBRARY") == 1
    no_res = cut_between(full, "RESOURCE CONSTRAINTS", "Strategy guidance")
    no_guid = full
    for cut, rep in zip(GUIDANCE_CUTS, GUIDANCE_REPLACEMENTS):
        assert no_guid.count(cut) == 1, cut[:60]
        no_guid = no_guid.replace(cut, rep)
    no_guid = cut_between(no_guid, "Strategy guidance", "GLOBAL SKILL LIBRARY")
    both = cut_between(no_guid, "RESOURCE CONSTRAINTS", "GLOBAL SKILL LIBRARY")
    no_skills = full[: full.index("GLOBAL SKILL LIBRARY")]
    clean = ("Generate the best performance code for U280. The input code is kernel.cpp (with kernel.h), shown below. "
             "Keep the autosa_mm signature so existing host code still calls it. Reply with the complete kernel.cpp "
             "in one ```cpp fence.\n\nkernel.h:\n```cpp\n" + (BENCH / "kernel.h").read_text()
             + "```\n\nkernel.cpp:\n```cpp\n" + (BENCH / "plain.cpp").read_text() + "```\n")
    return {
        "F0_full": full,
        "F1_no_resource_caps": no_res,
        "F2_no_conservative_guidance": no_guid,
        "F3_no_caps_no_guidance": both,
        "F4_no_skills": no_skills,
        "C0_clean": clean,
        "C1_clean_plus_skills": clean + "\n\n" + skills_block(full),
    }


def kernel_from(reply: str) -> str:
    blocks = [b for b in FENCE_RE.findall(reply) if re.search(r"\bautosa_mm\s*\([^;]*\)\s*\{", b)]
    return blocks[-1] if blocks else ""


def call_shim(url: str, system: str, prompt: str) -> dict:
    body = json.dumps({"model": "claude-opus-5-5", "max_tokens": 65536,
                       "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request(url.rstrip("/") + "/chat/completions", body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=7200) as r:
        return json.loads(r.read())


def make_eval(run: Path) -> None:
    e = run / "eval"
    e.mkdir(exist_ok=True)
    (e / "kernel.cpp").write_text((run / "kernel.cpp").read_text())
    for f in ("kernel.h", "testbench.cpp", "run_eval.py"):
        (e / f).write_text((EVAL_SRC / f).read_text())
    for tag in ("csim", "csynth"):
        s = (EVAL_SRC / f"sbatch_{tag}.sh").read_text().replace(str(EVAL_SRC), str(e))
        s = s.replace(f"--job-name=clean_u280_{tag}", f"--job-name=abl_{run.name}_{tag}")
        (e / f"sbatch_{tag}.sh").write_text(s)
        (e / f"sbatch_{tag}.sh").chmod(0o750)


def generate(reps: int) -> None:
    url = (ROOT / "shim_url.txt").read_text().strip()
    system, _ = load_source()
    vs = variants()
    (HERE / "prompts").mkdir(exist_ok=True)
    for name, prompt in vs.items():
        (HERE / "prompts" / f"{name}.txt").write_text(prompt)
    print({k: len(v) for k, v in vs.items()})
    only = set(sys.argv[sys.argv.index("--only") + 1].split(",")) if "--only" in sys.argv else set(vs)
    jobs = [(name, r) for r in range(1, reps + 1) for name in vs if name in only]
    from concurrent.futures import ThreadPoolExecutor

    def one(job):
        name, r = job
        run = HERE / f"{name}_r{r}"
        if (run / "kernel.cpp").is_file():
            return f"{run.name}: exists"
        run.mkdir(exist_ok=True)
        resp = call_shim(url, system, vs[name])
        reply = resp["choices"][0]["message"]["content"]
        (run / "reply.txt").write_text(reply)
        (run / "usage.json").write_text(json.dumps(resp.get("usage"), indent=1))
        k = kernel_from(reply)
        if not k:
            return f"{run.name}: NO KERNEL in reply"
        (run / "kernel.cpp").write_text(k)
        make_eval(run)
        return f"{run.name}: kernel {len(k)} chars"

    with ThreadPoolExecutor(2) as ex:
        for msg in ex.map(one, jobs):
            print(msg, flush=True)


def submit() -> None:
    for run in sorted(HERE.glob("*_r*")):
        e = run / "eval"
        if not (e / "kernel.cpp").is_file() or (e / "jobs.txt").is_file():
            continue
        ids = [subprocess.run(["sbatch", "--parsable", str(e / f"sbatch_{t}.sh")], capture_output=True, text=True)
               .stdout.strip() for t in ("csim", "csynth")]
        (e / "jobs.txt").write_text(" ".join(ids) + "\n")
        print(run.name, ids)


def table() -> None:
    rows = ["| Variant | Rep | csim | Cycles | DSP | BRAM_18K | LUT | Est. clock | Time (ms) |",
            "|---|---:|---|---:|---:|---:|---:|---:|---:|"]
    for run in sorted(HERE.glob("*_r*")):
        e = run / "eval"
        name, rep = run.name.rsplit("_r", 1)
        csim = (e / "csim_summary.txt").read_text() if (e / "csim_summary.txt").is_file() else ""
        ok = "Passed!" if "correct=True" in csim else ("FAILED" if csim else "pending")
        try:
            rep_ = json.loads((e / "csynth_summary.json").read_text())["report"]
            cells = [f"{rep_['latency_cycles']:,}", f"{rep_['dsp']:,}", f"{rep_['bram']:,}", f"{rep_['lut']:,}",
                     str(rep_.get("estimated_clock_period_ns")), f"{rep_['latency_ns'] / 1e6:.2f}"]
        except (OSError, KeyError, TypeError, ValueError):
            cells = ["pending" if not (e / "csynth_summary.json").is_file() else "csynth failed"] + [""] * 5
        rows.append(f"| {name} | {rep} | {ok} | " + " | ".join(cells) + " |")
    (HERE / "results.md").write_text("\n".join(rows) + "\n")
    print("\n".join(rows))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "table"
    if cmd == "generate":
        generate(int(sys.argv[sys.argv.index("--reps") + 1]) if "--reps" in sys.argv else 2)
    elif cmd == "submit":
        submit()
    else:
        table()
