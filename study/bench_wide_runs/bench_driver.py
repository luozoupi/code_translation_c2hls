#!/usr/bin/env python3
"""Serial Opus queue for the 58-kernel Refined + Res.-Req study, run after the agentic ablation.

Reuses agentic_ablation/serial_driver.py (usage-limit probe and wait, invalid-run rerun, 1 h -> 3 h -> 6 h timeout
escalation, 192G OOM rerun) with its own queue, state and log. Waits until the ablation driver logs "driver done".
Tags: bw_<bench>_cc; start at the 3 h tier (a rerun costs another Claude call).
"""
import importlib.util, json, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ABL = HERE.parent / "agentic_ablation"
spec = importlib.util.spec_from_file_location("serial_driver", ABL / "serial_driver.py")
sd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sd)
sys.path.insert(0, str(HERE))
import params

BENCHES = sorted(r["bench"] for r in json.loads((HERE.parent / "bench_wide/floors_final.json").read_text()))
sd.QUEUE = [f"bw_{b}_cc" for b in BENCHES]
sd.RUNS = HERE / "runs"
sd.STATE = HERE / "driver_state.json"
sd.LOG = HERE / "driver.log"
sd.base_argv = lambda tag: [f"--job-name={tag}", "--mem=128G",
                            f"--export={params.export(tag[3:-3], 'cc', tag)}"]

if __name__ == "__main__":
    abl_log = ABL / "driver.log"
    while "driver done" not in (abl_log.read_text() if abl_log.exists() else ""):
        time.sleep(600)
    sd.main()
