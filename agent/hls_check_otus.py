#!/usr/bin/env python3
"""Agent-side check of the workspace kernel, run on a compute node by hls_check.sh.

  hls_check.py compile|csim|csynth   (cwd = the agent workspace: kernel.cpp, kernel.h, testbench.cpp)

Uses the same c2hls functions the flash flow judges with: compile_check_cpp (g++ with the Vitis
include paths), hls_eval.run_csim on the bench testbench, hls_eval.run_hls_synthesis
(part / clock / top from AGENT_PART, AGENT_CLOCK_NS, AGENT_TOP). Prints a short summary and
writes the full one to checks/<n>_<mode>.txt.
"""
import os
import re
import subprocess
import sys
from pathlib import Path

CC = Path(__file__).resolve().parents[1]           # c2hls_claude_code (this file is agent/hls_check.py)
sys.path.insert(0, os.environ.get("C2HLS_ROOT", str(CC.parent / "code_translation_c2hls")))
import hls_eval  # noqa: E402

PART = os.environ.get("AGENT_PART", "xcu280-fsvh2892-2L-e")
CLOCK = float(os.environ.get("AGENT_CLOCK_NS", "3.33"))
TOP = os.environ.get("AGENT_TOP", "autosa_mm")
HDR = os.environ.get("AGENT_HEADER", "kernel.h")   # the bench header the kernel includes


def main(mode: str) -> int:
    ws = Path.cwd()
    src, hdr = (ws / "kernel.cpp").read_text(), (ws / HDR).read_text()
    work = ws / "checks" / f"{os.environ.get('AGENT_CHECK_N', '0')}_{mode}_work"
    work.mkdir(parents=True, exist_ok=True)
    # file-count quota on /scratch: run Vitis on node-local storage; copy back only reports/sources/json/logs
    local = os.environ.get("HLS_LOCAL_ROOT") or os.environ.get("SLURM_TMPDIR")
    own = None
    if not local and os.environ.get("SLURM_JOB_ID"):   # Otus: node-local fallback when $SLURM_TMPDIR is unset
        local = own = os.path.join(os.environ.get("OTUS_LOCAL_TMP", "/tmp"), os.environ.get("USER", "u"), os.environ["SLURM_JOB_ID"])
    run = Path(local) / "agent_check" / work.name if local else work
    run.mkdir(parents=True, exist_ok=True)
    if local:
        os.environ["C2HLS_TMP_ROOT"] = str(Path(local) / "c2hls")
    if mode == "compile":
        from c2hls import compile_check_cpp
        ok, err = compile_check_cpp(src, hdr, HDR, work_dir=str(run))
        text = "compile: OK\n" if ok else f"compile: FAILED\n{(err or '')[-6000:]}\n"
    elif mode == "csim":
        r = hls_eval.run_csim(src, (ws / "testbench.cpp").read_text(), hdr, header_name=HDR,
                              top_function=TOP, part=PART, clock_ns=CLOCK, work_dir=str(run))
        log = str(r.get("log") or "")
        failed = re.search(r"Failed with ([1-9]\d*) errors!", log)
        ok = re.search(r"^Passed!$", log, re.M) is not None and failed is None
        text = (f"csim: {'PASSED (testbench printed Passed!)' if ok else 'FAILED'}\n"
                + ("" if ok else f"{failed.group(0) if failed else ''}\n{(r.get('error') or '')[-3000:]}\nlog tail:\n{log[-4000:]}\n"))
    elif mode == "csynth":
        r = hls_eval.run_hls_synthesis(src, hdr, header_name=HDR, top_function=TOP, part=PART,
                                       clock_ns=CLOCK, work_dir=str(run), allow_compile_jobs=False)
        text = f"csynth: {'OK' if r.get('success') else 'FAILED'}\n"
        text += hls_eval.format_report_summary(r.get("report") or {}) + "\n"
        if r.get("error"):
            text += f"error:\n{str(r['error'])[-6000:]}\n"
    else:
        print("usage: hls_check.sh compile|csim|csynth")
        return 2
    if run != work:
        subprocess.run([str(CC / "study/jobs/keep_copy.sh"), str(run), str(work)], check=False)
        if own:
            import shutil
            shutil.rmtree(own, ignore_errors=True)
    (ws / "checks" / f"{os.environ.get('AGENT_CHECK_N', '0')}_{mode}.txt").write_text(text)
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else ""))
