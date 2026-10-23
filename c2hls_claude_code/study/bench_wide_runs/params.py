#!/usr/bin/env python3
"""--export string for one bench of the 58-kernel Refined + Res.-Req study (floors_final.json, approved 2026-10-15).

  floor > 0 : refined v2 + DSP window floor..8121 (dsp_window.py); balance gate on unless the kernel is dependency-bound
  floor = 0 : refined v2, no DSP floor (NO_DSP_FLOOR), 9024 cap, no balance gate
Balance gate per kernel: FLASH_REFINED_TOTAL_MACS = DSP-weighted fp work, FLASH_REFINED_DSP_PER_MAC = 1, so the
gate's compute floor is work / DSP (fp32 mul 3 / add 2, fp64 mul 11 / add 3 DSP).
  params.py BENCH MODEL(cc|ds) TAG
"""
import json, sys
from pathlib import Path

CC = Path("/scratch/hpc-prf-llmfpga/asa582/projects/workspaces/c2hls_claude_code")
BW = CC / "study/bench_wide"
OUT = CC / "study/bench_wide_runs/runs"
INC = "/scratch/hpc-prf-llmfpga/asa582/projects/tmp/c2hls/vitis_hls_2023.2_include"


def export(bench, model, tag):
    f = {r["bench"]: r for r in json.loads((BW / "floors_final.json").read_text())}[bench]
    meta = json.loads((BW / bench / "metadata.json").read_text())
    seed = BW / bench / (meta.get("gold_hls_baseline_file") or "hls_baseline.cpp")
    kv = ["ALL", "CONTROLLED=1", "C2HLS_THINKING=enabled", "C2HLS_FLASH_MAX_TOKENS=262144", "CTRL_MIN_MAX_TOKENS=262144",
          f"OUT_ROOT={OUT}", f"C2HLS_COMPILE_INCLUDE_PATHS={INC}", f"BENCH_ROOT={BW}", f"BENCH={bench}",
          f"SEED_CODE={seed}", "FLAVOR=skills", "REFINED=1"]
    if f["floor"]:
        kv.append(f"DSP_WINDOW={f['floor']}:8121")
    else:
        kv.append("NO_DSP_FLOOR=1")
    if f.get("balance_work"):
        kv += [f"FLASH_REFINED_TOTAL_MACS={int(f['balance_work'])}", "FLASH_REFINED_DSP_PER_MAC=1",
               f"FLASH_REFINED_MEM_CYCLES={int(f['mem_cycles'])}"]
    if model == "cc":
        kv += ["MODEL=claude-opus-5-5", f"CLAUDE_CODE_SHIM_URL={(CC / 'shim_url.txt').read_text().strip()}"]
    else:
        kv += ["MODEL=deepseek-v4-flash", f"CLAUDE_CODE_SHIM_URL={(CC / 'controlled/ds_proxy_url.txt').read_text().strip()}"]
    kv.append(f"TAG={tag}")
    return ",".join(kv)


if __name__ == "__main__":
    print(export(*sys.argv[1:4]))
