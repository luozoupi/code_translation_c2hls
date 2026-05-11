"""Phase 7a offline smoke: static-report harvest (burst.xml,
fe/be_messages.xml, csynth_design_size.rpt). Pure offline.

Uses the on-disk sample at /tmp/hls_synth_qu63pmih (left over from a
recent Vitis HLS run) when present. Falls back to inline xml fixtures
otherwise so the test still runs in a clean environment.

    cd /home/luo00466/code_translation-c2hls
    python tests/test_phase7_smoke.py [--out artifacts/phase7_smoke_<ts>.md]
"""

from __future__ import annotations

import argparse
import datetime as _dt
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import hls_feedback as hf  # noqa: E402


# A minimal burst.xml fixture that exercises all three classification
# branches (passed, widened, failed). Mimics the real Vitis schema
# including the unbound `VitisHLS:` namespace prefix.
_BURST_FIXTURE = """<VitisHLS:BurstInfo>
    <burst group="BURST_VERBOSE_PASSED" msg_severity="INFO" src_info="kernel.cpp:40" msg_body="Sequential read of length 1024" BundleName="gmem" VarName="A" Direction="read" Length="1024" LoopName="L1" ParentFunc="workload"/>
    <burst group="BURST_VERBOSE_WIDEN_PASSED" msg_severity="INFO" src_info="kernel.cpp:42" msg_body="widened by 16: 64 x 512bit" BundleName="gmem" VarName="A" Direction="read" Length="64" LoopName="L1" ParentFunc="workload"/>
    <burst group="BURST_VERBOSE_FAILED" msg_severity="WARNING" src_info="kernel.cpp:55" msg_body="Could not infer burst" BundleName="gmem" VarName="B" Direction="write" Length="0" LoopName="L2" ParentFunc="workload"/>
    <burst group="BURST_SUMMARY" msg_severity="INFO" msg_body="Multiple burst reads on gmem" BundleName="gmem" Direction="read" Length="65472" Width="512"/>
</VitisHLS:BurstInfo>
"""

_FE_MSG_FIXTURE = """<xilinx:hls_fe_msgs>
    <msg msg_groups="" msg_id="207-100" msg_severity="INFO" msg_loc="kernel.cpp:10" msg_body="generic info"/>
    <msg msg_groups="PRAGMA_INVALID" msg_id="207-6973" msg_severity="WARNING" msg_loc="support/wide_bus.h:26" msg_body="the 'self/all' option to 'Inline' pragma is not supported and will be ignored"/>
    <msg msg_groups="PRAGMA_INVALID DEPENDENCE_PRAGMA" msg_id="207-7000" msg_severity="WARNING" msg_loc="kernel.cpp:33" msg_body="dependence pragma cannot be applied"/>
</xilinx:hls_fe_msgs>
"""

_DESIGN_SIZE_FIXTURE = """================================================================
== Design Size Report
================================================================

* Total Instructions per Compilation Phase
+---------------+----------------------------+--------------+----------------------------------------------------------------------------------------+
| Phase         | Step                       | Instructions | Description                                                                            |
+---------------+----------------------------+--------------+----------------------------------------------------------------------------------------+
| Compile/Link  |                            |  92          | After all functions are compiled and linked into a single design                       |
|               |                            |              |                                                                                        |
| Unroll/Inline |                            |              | After user unroll and inline pragmas are applied                                       |
|               | (1) unroll                 |  93          | user unroll pragmas are applied                                                        |
|               | (2) simplification         |  77          | simplification of applied user unroll pragmas                                          |
|               |                            |              |                                                                                        |
| Performance   |                            |              | After transformations are applied to meet performance pragma targets                   |
|               | (1) loop simplification    | 109          | loop and instruction simplification                                                    |
|               | (2) parallelization        | 188          | loops are unrolled or pipelined to meet performance targets                            |
+---------------+----------------------------+--------------+----------------------------------------------------------------------------------------+

* Instructions per Function for each Compilation Phase
"""


class Tee:
    def __init__(self):
        self.buf: List[str] = []

    def __call__(self, *a):
        line = " ".join(str(x) for x in a)
        print(line); self.buf.append(line)

    def section(self, t):
        print(); print(f"=== {t} ===")
        self.buf.append(""); self.buf.append(f"## {t}"); self.buf.append("")

    def md(self):
        return "\n".join(self.buf) + "\n"


def _check(label, ok, detail, results, tee):
    results.append({"check": label, "ok": bool(ok), "detail": detail})
    icon = "OK  " if ok else "FAIL"
    tee(f"[{icon}] {label}: {detail}")
    return ok


def _make_fixture_workdir() -> Path:
    """Build a temp dir laid out like a real Vitis sol1 work_dir,
    pre-populated with our minimal fixtures."""
    work = Path(tempfile.mkdtemp(prefix="phase7_smoke_"))
    db = work / "hls_proj" / "sol1" / ".autopilot" / "db"
    syn = work / "hls_proj" / "sol1" / "syn" / "report"
    db.mkdir(parents=True)
    syn.mkdir(parents=True)
    (db / "burst.xml").write_text(_BURST_FIXTURE, encoding="utf-8")
    (db / "fe_messages.xml").write_text(_FE_MSG_FIXTURE, encoding="utf-8")
    (syn / "csynth_design_size.rpt").write_text(_DESIGN_SIZE_FIXTURE, encoding="utf-8")
    return work


def test_burst_parser_fixture(results, tee):
    tee.section("Phase 7a: burst.xml parser (synthetic fixture)")
    work = _make_fixture_workdir()
    try:
        b = hf.parse_burst_info(str(work))
        _check("counts-passed-1", b["counts"]["passed"] == 2,
               f"got {b['counts']}", results, tee)
        _check("counts-widened-1", b["counts"]["widened"] == 1,
               f"widened={b['counts']['widened']}", results, tee)
        _check("counts-failed-1", b["counts"]["failed"] == 1,
               f"failed={b['counts']['failed']}", results, tee)
        _check("counts-summary-1", b["counts"]["summary"] == 1,
               f"summary={b['counts']['summary']}", results, tee)

        widened_record = b["widened"][0] if b["widened"] else {}
        _check("widened-record-shape",
               widened_record.get("var") == "A"
               and widened_record.get("direction") == "read"
               and "widened" in (widened_record.get("msg") or "").lower(),
               f"got {widened_record}", results, tee)

        failed_record = b["failed"][0] if b["failed"] else {}
        _check("failed-record-shape",
               failed_record.get("var") == "B"
               and failed_record.get("direction") == "write",
               f"got {failed_record}", results, tee)
    finally:
        shutil.rmtree(work, ignore_errors=True)


def test_diagnostic_messages_parser_fixture(results, tee):
    tee.section("Phase 7a: fe/be_messages.xml parser (synthetic fixture)")
    work = _make_fixture_workdir()
    try:
        d = hf.parse_diagnostic_messages(str(work))
        _check("warnings-counted", d["warnings"] == 2,
               f"warnings={d['warnings']}", results, tee)
        _check("info-counted", d["info"] == 1,
               f"info={d['info']}", results, tee)
        _check("rejected-pragmas-found", len(d["rejected_pragmas"]) == 2,
               f"got {len(d['rejected_pragmas'])} pragma rejections", results, tee)
        if d["rejected_pragmas"]:
            sample = d["rejected_pragmas"][0]
            _check("rejected-pragma-shape",
                   "PRAGMA_INVALID" in (sample.get("groups") or [])
                   and sample.get("id") == "207-6973",
                   f"got {sample}", results, tee)
    finally:
        shutil.rmtree(work, ignore_errors=True)


def test_design_size_parser_fixture(results, tee):
    tee.section("Phase 7a: csynth_design_size.rpt parser (synthetic fixture)")
    work = _make_fixture_workdir()
    try:
        ds = hf.parse_design_size_report(str(work))
        _check("phases-detected",
               set(ds["phases"].keys()) >= {"Compile/Link", "Unroll/Inline", "Performance"},
               f"got phases {list(ds['phases'].keys())}", results, tee)
        _check("compile-link-total-92",
               ds["phases"].get("Compile/Link", {}).get("_total") == 92,
               f"got {ds['phases'].get('Compile/Link')}", results, tee)
        _check("unroll-step-counts-correct",
               ds["phases"].get("Unroll/Inline", {}).get("(1) unroll") == 93,
               f"unroll: {ds['phases'].get('Unroll/Inline')}",
               results, tee)
        _check("compile-to-hw-growth-positive",
               (ds.get("compile_to_hw_growth") or 0) > 1.0,
               f"growth={ds.get('compile_to_hw_growth')}", results, tee)
    finally:
        shutil.rmtree(work, ignore_errors=True)


def test_attach_feedback_with_workdir(results, tee):
    tee.section("Phase 7a: attach_feedback wires static_extras when work_dir given")
    work = _make_fixture_workdir()
    try:
        report = {"latency_ns": 1000.0, "fmax_mhz": 250.0,
                  "requested_clock_period_ns": 4.0}
        hf.attach_feedback(report, work_dir=str(work))
        feedback = report.get("feedback") or {}
        _check("feedback-attached", "feedback" in report and "scopes" in feedback,
               f"keys={list(feedback.keys())}", results, tee)
        extras = feedback.get("static_extras") or {}
        _check("static-extras-attached",
               "bursts" in extras and "diagnostic" in extras and "design_size" in extras,
               f"static_extras keys: {list(extras.keys())}", results, tee)
        _check("extras-summary-rolled-up",
               "summary" in extras
               and extras["summary"].get("rejected_pragmas") == 2
               and extras["summary"].get("bursts_widened") == 1,
               f"summary={extras.get('summary')}",
               results, tee)
    finally:
        shutil.rmtree(work, ignore_errors=True)


def test_attach_feedback_no_workdir_no_extras(results, tee):
    tee.section("Phase 7a: no work_dir → no static_extras (legacy behaviour)")
    report = {"latency_ns": 1000.0, "fmax_mhz": 250.0,
              "requested_clock_period_ns": 4.0}
    hf.attach_feedback(report)  # no work_dir
    extras = (report.get("feedback") or {}).get("static_extras")
    _check("static-extras-absent-when-workdir-missing",
           extras is None,
           f"got extras={extras}", results, tee)


def test_render_block_useful(results, tee):
    tee.section("Phase 7a: render_static_extras_for_prompt produces an actionable block")
    work = _make_fixture_workdir()
    try:
        extras = {
            "bursts": hf.parse_burst_info(str(work)),
            "diagnostic": hf.parse_diagnostic_messages(str(work)),
            "design_size": hf.parse_design_size_report(str(work)),
        }
        block = hf.render_static_extras_for_prompt(extras)
        _check("block-mentions-burst-failure",
               "FAILED" in block,
               f"block first 200 chars:\n{block[:200]}", results, tee)
        _check("block-mentions-rejected-pragma",
               "rejected" in block.lower() and "PRAGMA_INVALID" not in block,
               # PRAGMA_INVALID is a tag, not user-facing — we expect a
               # human-readable summary instead.
               f"block first 300 chars:\n{block[:300]}", results, tee)
        _check("block-mentions-design-size",
               "growth" in block.lower(),
               "design-size growth line present", results, tee)
    finally:
        shutil.rmtree(work, ignore_errors=True)


def test_real_sample_optional(results, tee):
    """If a recent Vitis HLS work_dir is on disk, parse it and confirm
    we extract real data. Skipped (with PASS) when no sample exists."""
    tee.section("Phase 7a: real on-disk sample (optional)")
    sample_glob = "/tmp/hls_synth_*"
    import glob
    candidates = [p for p in glob.glob(sample_glob)
                  if os.path.isdir(os.path.join(p, "hls_proj"))]
    if not candidates:
        _check("real-sample-skipped",
               True, "no /tmp/hls_synth_* directory present (skipped)",
               results, tee)
        return
    sample = candidates[0]
    extras = {
        "bursts": hf.parse_burst_info(sample),
        "diagnostic": hf.parse_diagnostic_messages(sample),
        "design_size": hf.parse_design_size_report(sample),
    }
    bursts_total = sum(extras["bursts"]["counts"].values())
    _check("real-sample-bursts-parsed",
           bursts_total > 0,
           f"sample={sample}, total burst records: {bursts_total}",
           results, tee)
    _check("real-sample-design-size-parsed",
           bool(extras["design_size"]["phases"]),
           f"phase count: {len(extras['design_size']['phases'])}",
           results, tee)


def main() -> int:
    parser = argparse.ArgumentParser()
    timestamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    parser.add_argument("--out", type=Path,
                        default=REPO_ROOT / "artifacts" / f"phase7_smoke_{timestamp}.md")
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    tee = Tee()
    results: List[Dict[str, Any]] = []

    tee(f"# Phase 7 Smoke Test ({timestamp})")
    tee(f"REPO_ROOT={REPO_ROOT}")

    test_burst_parser_fixture(results, tee)
    test_diagnostic_messages_parser_fixture(results, tee)
    test_design_size_parser_fixture(results, tee)
    test_attach_feedback_with_workdir(results, tee)
    test_attach_feedback_no_workdir_no_extras(results, tee)
    test_render_block_useful(results, tee)
    test_real_sample_optional(results, tee)

    tee.section("Summary")
    total = len(results)
    failed = [r for r in results if not r["ok"]]
    tee(f"checks_total: {total}")
    tee(f"checks_passed: {total - len(failed)}")
    tee(f"checks_failed: {len(failed)}")
    if failed:
        tee("failed_checks:")
        for r in failed:
            tee(f"  - {r['check']}: {r['detail']}")

    args.out.write_text(tee.md(), encoding="utf-8")
    print()
    print(f"report written: {args.out}")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
