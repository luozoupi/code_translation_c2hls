#!/usr/bin/env python3
"""Serial Opus queue for the 11 SODA kernels (Refined + Res.-Req), run after the 58-bench Opus queue.

Reuses agentic_ablation/serial_driver.py (usage-limit probe and wait, invalid-run rerun, timeout escalation,
192G OOM rerun) with its own queue, state and log. Waits until bench_wide_runs/driver.log says "driver done".
Tags: sd_<kernel>_cc; start at the 3 h tier.
"""
import importlib.util, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ABL = HERE.parent / "agentic_ablation"
spec = importlib.util.spec_from_file_location("serial_driver", ABL / "serial_driver.py")
sd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sd)
sys.path.insert(0, str(HERE))
import params

KERNELS = ["blur", "contrast", "denoise2d", "denoise3d", "erosion", "heat3d", "jacobi2d", "jacobi3d", "seidel2d",
           "sobel2d", "xcorr"]
sd.QUEUE = [f"sd_{k}_cc" for k in KERNELS]
sd.RUNS = HERE / "runs"
sd.STATE = HERE / "driver_state.json"
sd.LOG = HERE / "driver.log"
sd.base_argv = lambda tag: [f"--job-name={tag}", "--mem=128G", f"--export={params.export(tag[3:-3], 'cc', tag)}"]

if __name__ == "__main__":
    prev = HERE.parent / "bench_wide_runs/driver.log"
    while "driver done" not in (prev.read_text() if prev.exists() else ""):
        time.sleep(600)
    sd.main()
