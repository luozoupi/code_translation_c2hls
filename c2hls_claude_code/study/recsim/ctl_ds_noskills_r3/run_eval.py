#!/usr/bin/env python3
"""Judge kernel.cpp with plain Vitis HLS: csim on the n1024 bench, or csynth (xcu280, 3.33 ns).

No DSE trial spec, no v3 gate, no harness TCL: hls_eval defaults only.
Usage: run_eval.py csim|csynth   (writes <mode>_summary.{txt,json})
"""
import json, re, sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, "/scratch/asa582/workspaces/code_translation_c2hls")
import hls_eval  # noqa: E402

PART, CLOCK, TOP = "xcu280-fsvh2892-2L-e", 3.33, "autosa_mm"


def main(mode):
    src = (HERE / "kernel.cpp").read_text()
    hdr = (HERE / "kernel.h").read_text()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    work = HERE / f"{mode}_work" / stamp
    work.mkdir(parents=True, exist_ok=True)
    if mode == "csim":
        r = hls_eval.run_csim(src, (HERE / "testbench.cpp").read_text(), hdr, header_name="kernel.h",
                              top_function=TOP, part=PART, clock_ns=CLOCK, work_dir=str(work))
        log = str(r.get("log") or "")
        failed = re.search(r"Failed with ([1-9]\d*) errors!", log)
        ok = re.search(r"^Passed!$", log, re.M) is not None and failed is None
        s = {"correct": ok, "bench_line": "Passed!" if ok else (failed.group(0) if failed else ""),
             "success": bool(r.get("success")), "error": r.get("error") or "", "work_dir": str(work)}
        text = "\n".join(f"{k}={v}" for k, v in s.items() if k != "work_dir") + "\n"
        (HERE / "csim_log.txt").write_text(log)
    else:
        r = hls_eval.run_hls_synthesis(src, hdr, header_name="kernel.h", top_function=TOP, part=PART,
                                       clock_ns=CLOCK, work_dir=str(work), allow_compile_jobs=False)
        rep = r.get("report") or {}
        s = {"success": bool(r.get("success")), "error": r.get("error") or "", "report": rep,
             "work_dir": str(r.get("work_dir") or work)}
        text = f"success={s['success']}\n" + hls_eval.format_report_summary(rep) + "\n" + (f"\nerror:\n{s['error']}\n" if s["error"] else "")
    (HERE / f"{mode}_summary.json").write_text(json.dumps(s, indent=2, default=str) + "\n")
    (HERE / f"{mode}_summary.txt").write_text(text)
    print(text, end="")
    return 0 if s["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
