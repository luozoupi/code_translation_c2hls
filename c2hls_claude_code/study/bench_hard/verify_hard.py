#!/usr/bin/env python3
"""Re-check matrix-multiply designs on the hardened bench (bench_hard/n1024/autosa_mm), g++ -O2 with the Vitis headers.

  verify_hard.py [--jobs N] [--csv OUT]
Kernels checked:
  * every flow run's final kernel (<bench>_final.cpp) under study/{agentic_ablation,dsp5065,s90,exp_hlsmith}/runs and
    runs_controlled/ (the latter target the old C += bench and read C, so they are expected to fail the poisoned C;
    verify_zeroC/ holds their C-not-read edits, checked too);
  * study/manual_df/v*/kernel.cpp;
  * every free-agent experiment (study/agentfw/runs/*/exp/*) whose csynth succeeded, and each workspace's final kernel.cpp.
Output: one row per kernel: source, verdict (Passed / Failed N / build error / timeout), and its csynth worst latency,
DSP and LUT where a summary sits next to it. This is a C-level check (the same semantics csim checks).
"""
import argparse, csv, glob, json, os, re, subprocess, sys, tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

CC = Path("/scratch/asa582/workspaces/c2hls_claude_code")
BENCH = CC / "study/bench_hard/n1024/autosa_mm"
INC = "/scratch/asa582/tmp/c2hls/vitis_hls_2023.2_include"
TMP = Path(os.environ.get("SLURM_TMPDIR") or "/scratch/asa582/tmp/c2hls") / "verify_hard"


def kernels():
    out = []
    for root in ("study/agentic_ablation/runs", "study/dsp5065/runs", "study/s90/runs", "study/exp_hlsmith/runs", "runs_controlled"):
        for f in sorted(glob.glob(str(CC / root / "*/autosa_mm/autosa_mm_final.cpp"))):
            if "/_" not in f:
                d = Path(f).parent
                out.append((f, d / "autosa_mm_multistep_results.json"))
    for f in sorted(glob.glob(str(CC / "verify_zeroC/*/kernel.cpp"))):
        out.append((f, None))
    for f in sorted(glob.glob(str(CC / "study/manual_df/v*/kernel.cpp"))):
        out.append((f, Path(f).parent / "csynth_summary.json"))
    for w in sorted(glob.glob(str(CC / "study/agentfw/runs/mm1024_*"))):
        if "/_" in w:
            continue
        out.append((str(Path(w) / "kernel.cpp"), None))
        for e in sorted(glob.glob(w + "/exp/*")):
            s = Path(e) / "csynth_summary.json"
            if s.exists() and (Path(e) / "kernel.cpp").exists():
                out.append((str(Path(e) / "kernel.cpp"), s))
    return [(k, s) for k, s in out if Path(k).exists()]


def qor(s):
    try:
        d = json.loads(Path(s).read_text())
    except Exception:
        return {}
    r = d.get("final_report") or d.get("report") or {}
    return {"lat": r.get("latency_cycles_worst") or r.get("latency_cycles"), "dsp": r.get("dsp"), "lut": r.get("lut")}


def check(item):
    k, s = item
    w = Path(tempfile.mkdtemp(dir=TMP))
    b = subprocess.run(["g++", "-O2", "-std=c++17", "-w", f"-I{BENCH}", f"-I{INC}", str(BENCH / "testbench.cpp"), k, "-o",
                        str(w / "t")], capture_output=True, text=True)
    if b.returncode:
        v = "build error: " + (b.stderr.strip().splitlines() or [""])[-1][:160]
    else:
        try:
            r = subprocess.run([str(w / "t")], capture_output=True, text=True, timeout=3600)
            lines = (r.stdout + r.stderr).strip().splitlines()
            v = next((l for l in reversed(lines) if l.startswith(("Passed!", "Failed with"))), "no verdict (rc %d)" % r.returncode)
        except subprocess.TimeoutExpired:
            v = "timeout"
    subprocess.run(["rm", "-rf", str(w)])
    return {"kernel": str(Path(k).relative_to(CC)), "verdict": v, **(qor(s) if s else {})}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    ap.add_argument("--csv", default=str(CC / "study/bench_hard/verify_hard.csv"))
    a = ap.parse_args()
    TMP.mkdir(parents=True, exist_ok=True)
    ks = kernels()
    print(f"{len(ks)} kernels", flush=True)
    with ThreadPoolExecutor(a.jobs) as ex:
        rows = list(ex.map(check, ks))
    with open(a.csv, "w") as fp:
        w = csv.DictWriter(fp, ["kernel", "verdict", "lat", "dsp", "lut"])
        w.writeheader()
        w.writerows(rows)
    print(sum(r["verdict"] == "Passed!" for r in rows), "passed of", len(rows))


if __name__ == "__main__":
    main()
