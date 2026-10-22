#!/usr/bin/env python3
"""Run inside the Slurm job: csim or csynth of <expdir>/kernel.cpp for the task in task.json (plain Vitis HLS via
code_translation_c2hls/hls_eval.py, read only). Writes <expdir>/<mode>_summary.{txt,json}.
Otus copy of hls_eval_job.py: the judge and keep_copy.sh are found from this file's location (C2HLS_ROOT overrides the
judge); node-local work falls back to OTUS_LOCAL_TMP or /tmp/$USER/$SLURM_JOB_ID when $SLURM_TMPDIR is unset."""
import json, os, re, shutil, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path
# study/: the first parent with jobs/keep_copy.sh (this file runs from agentfw/tools/ or a workspace's bin/)
STUDY = next(p for p in Path(__file__).resolve().parents if (p / "jobs" / "keep_copy.sh").is_file())
sys.path.insert(0, os.environ.get("C2HLS_ROOT") or str(STUDY.parents[1] / "code_translation_c2hls"))
import hls_eval  # noqa: E402

mode, d, task = sys.argv[1], Path(sys.argv[2]), json.loads(Path(sys.argv[3]).read_text())
ws = Path(sys.argv[3]).parent
src = (d / "kernel.cpp").read_text()
hdr = (ws / task["header"]).read_text()
work = d / f"{mode}_work" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
work.mkdir(parents=True, exist_ok=True)
# file-count quota: run Vitis on node-local storage, then copy back only reports/sources/json/logs (keep_copy.sh)
_local = os.environ.get("HLS_LOCAL_ROOT") or os.environ.get("SLURM_TMPDIR")
_own = None
if not _local and os.environ.get("SLURM_JOB_ID"):
    _local = _own = os.path.join(os.environ.get("OTUS_LOCAL_TMP", "/tmp"), os.environ.get("USER", "u"), os.environ["SLURM_JOB_ID"])
run = Path(_local) / "hls_work" / f"{mode}_{work.name}" if _local else work
run.mkdir(parents=True, exist_ok=True)
if _local:
    os.environ["C2HLS_TMP_ROOT"] = str(Path(_local) / "c2hls")
if mode == "csim":
    r = hls_eval.run_csim(src, (ws / task["testbench"]).read_text(), hdr, header_name=task["header"],
                          top_function=task["top"], part=task["part"], clock_ns=task["clock_ns"], work_dir=str(run))
    log = str(r.get("log") or "")
    failed = re.search(task.get("fail_regex", r"Failed with ([1-9]\d*) errors!"), log)
    ok = re.search(task.get("pass_regex", r"^Passed!$"), log, re.M) is not None and failed is None
    s = {"correct": ok, "bench_line": "Passed!" if ok else (failed.group(0) if failed else ""),
         "success": bool(r.get("success")), "error": (r.get("error") or "")[:2000], "work_dir": str(work)}
    (d / "csim_log.txt").write_text(log[-200000:])
    text = "\n".join(f"{k}={v}" for k, v in s.items() if k != "work_dir") + "\n"
else:
    r = hls_eval.run_hls_synthesis(src, hdr, header_name=task["header"], top_function=task["top"], part=task["part"],
                                   clock_ns=task["clock_ns"], work_dir=str(run), allow_compile_jobs=False)
    rep = r.get("report") or {}
    s = {"success": bool(r.get("success")), "error": (r.get("error") or "")[:4000],
         "report": {k: v for k, v in rep.items() if k != "feedback"}, "work_dir": str(r.get("work_dir") or run).replace(str(run), str(work))}
    text = f"success={s['success']}\n" + hls_eval.format_report_summary(rep) + "\n" + (f"\nerror:\n{s['error']}\n" if s["error"] else "")
if run != work:
    subprocess.run([str(STUDY / "jobs/keep_copy.sh"), str(run), str(work)], check=False)
    if _own:
        shutil.rmtree(_own, ignore_errors=True)
(d / f"{mode}_summary.json").write_text(json.dumps(s, indent=1, default=str))
(d / f"{mode}_summary.txt").write_text(text)
print(text)
