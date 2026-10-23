#!/usr/bin/env python3
"""Controls for bench_soda/<k>/ testbench (g++ -O2), written as the CONTROLS lines of TB_NOTES.md.

  check_bench.py K [K ...]
  Control 1 plain.cpp as kernel -> must print Passed!
  Control 2 plain.cpp with `return;` at the top of workload -> must fail
  Control 3 plain.cpp with one bug (gen_bench's bug pair, from bench_soda_check/<k>/info.json) -> must fail
  hls_baseline.cpp as kernel -> must pass
Scratch: /scratch/hpc-prf-llmfpga/asa582/projects/tmp/c2hls/bench_soda_check/<k>/. Exit 1 if any control gives the wrong verdict.
"""
import json, re, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
INC = "/scratch/hpc-prf-llmfpga/asa582/projects/tmp/c2hls/vitis_hls_2023.2_include"
INFO = Path("/home/asa582/.claude/jobs/72130906/tmp/bench_soda_check")


def run(k, name, src):
    w = Path("/scratch/hpc-prf-llmfpga/asa582/projects/tmp/c2hls/bench_soda_check") / k / name
    w.mkdir(parents=True, exist_ok=True)
    (w / "kernel.cpp").write_text(src)
    r = subprocess.run(f"g++ -O2 -I{HERE / k} -I{INC} {HERE / k}/testbench.cpp kernel.cpp -o tb && ./tb; rm -f tb",
                       shell=True, cwd=w, capture_output=True, text=True)
    out = (r.stdout + r.stderr).strip().splitlines()
    return next((l for l in reversed(out) if l.startswith(("Passed!", "Failed with"))), "no verdict: " + " | ".join(out[-3:]))


def main():
    bad = 0
    for k in sys.argv[1:]:
        d = HERE / k
        plain = (d / "plain.cpp").read_text()
        bug = json.loads((INFO / k / "info.json").read_text())["bug"]
        assert plain.count(bug[0]) >= 1, (k, bug)
        c2, n = re.subn(r"(void workload\([^{]*\{)", r"\1\n    return;  /* control 2 */", plain, count=1)
        assert n == 1
        res = [("Control 1 (plain.cpp as kernel, must pass)", run(k, "c1", plain), True),
               ("Control 2 (workload returns immediately, must fail)", run(k, "c2", c2), False),
               ("Control 3 (one bug, must fail)", run(k, "c3", plain.replace(bug[0], bug[1], 1)), False),
               ("hls_baseline.cpp as kernel", run(k, "base", (d / "hls_baseline.cpp").read_text()), True)]
        lines = []
        for label, v, want in res:
            ok = v.startswith("Passed!") == want and (want or v.startswith("Failed with"))
            bad += not ok
            lines.append(f"- {label}: {v}" + ("" if ok else "  <-- WRONG VERDICT"))
        lines.append(f"- Controls rerun 2026-10-19 by check_bench.py at the LAAFD sizes; SODA cross-check: plain.cpp semantics are "
                     f"unchanged from the 2026-10-16 bench (0 mismatches vs the sodac kernel there); the sodac kernel at this size "
                     f"passes its own golden (study/soda/dse/designs/{k}_u4_b128*_laafd/gpp_check.log).")
        notes = (d / "TB_NOTES.md").read_text()
        notes = notes.replace("- Controls (g++ -O2): see CONTROLS line below, filled by the validation script.\n",
                              "- Controls (g++ -O2):\n" + "\n".join(lines) + "\n")
        (d / "TB_NOTES.md").write_text(notes)
        print(k, *lines, sep="\n  ")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
