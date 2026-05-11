"""Phase 5 offline smoke test:
- 5a: LLM-aided third retry path is wired in run_optimization_step.
- 5b: GT pre-population at run_multistep entry, skill-template injection
  in _optimization_step_attempt.

Pure offline — no Vitis or LLM calls.

    cd /home/luo00466/code_translation-c2hls
    python tests/test_phase5_smoke.py [--out artifacts/phase5_smoke_<ts>.md]
"""

from __future__ import annotations

import argparse
import datetime as _dt
import inspect
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import c2hls  # noqa: E402


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


def test_5a_phase5_llm_retry_wiring(results, tee):
    tee.section("Phase 5a: LLM-aided third retry path")
    src = inspect.getsource(c2hls.C2HLSOrchestrator.run_optimization_step)
    _check("phase5-env-flag-read",
           "C2HLS_PHASE5_LLM_RETRY" in src,
           "C2HLS_PHASE5_LLM_RETRY env var read", results, tee)
    _check("max-outer-turns-extends-to-3",
           "max_outer_turns = 3 if phase5_llm_retry else 2" in src,
           "loop range expands to 3 when flag is on", results, tee)
    _check("compose-with-llm-called",
           "compose_with_llm(" in src and "kernel_diff=" in src,
           "compose_with_llm called with kernel_diff payload",
           results, tee)
    _check("captures-failing-edit",
           "last_llm_edit_code" in src,
           "failing edit captured for compose_with_llm to read",
           results, tee)
    _check("llm-aided-history-line-emitted",
           "LLM-aided composition (Phase 5a)" in src,
           "history line records the LLM-aided retry",
           results, tee)


def test_5b_gt_prepop_wiring(results, tee):
    tee.section("Phase 5b: GT pre-population at run_multistep entry")
    src = inspect.getsource(c2hls.C2HLSOrchestrator.run_multistep)
    _check("gt-prepop-env-flag",
           "C2HLS_PHASE5_GT_PREPOP" in src,
           "C2HLS_PHASE5_GT_PREPOP env var read", results, tee)
    _check("iterates-gt-variants",
           "for gt_step_name, gt_code in gt_variants.items()" in src,
           "loops over all GT variants", results, tee)
    _check("populates-gt-cache",
           "self._gt_step_reports[gt_step_name]" in src,
           "writes GT report into cache", results, tee)
    _check("logs-prepop-event",
           "Phase 5b" in src and "pre-populated GT cache" in src,
           "log line on each populated entry", results, tee)


def test_5b_skill_injection_wiring(results, tee):
    tee.section("Phase 5b: skill-template injection in optimization prompt")
    src = inspect.getsource(c2hls.C2HLSOrchestrator._optimization_step_attempt)
    _check("renders-skill-set",
           "render_skill_set_for_prompt" in src,
           "render_skill_set_for_prompt imported and called",
           results, tee)
    _check("queries-by-top-bottleneck",
           "top_bottleneck_kind" in src and "skill_library.query(" in src,
           "queries skills by feedback's top bottleneck kind",
           results, tee)
    _check("guarded-by-skill-library-presence",
           "self.skill_library is not None" in src,
           "no-op when skill_library is None (static-order path)",
           results, tee)
    _check("guarded-by-best-effort",
           "Phase 5b skill-template injection failed" in src,
           "wrapped in try/except so any failure can't break the step",
           results, tee)


def test_phase5_default_off(results, tee):
    tee.section("Phase 5: default behaviors are off when env flags unset")
    os.environ.pop("C2HLS_PHASE5_LLM_RETRY", None)
    os.environ.pop("C2HLS_PHASE5_GT_PREPOP", None)
    src = inspect.getsource(c2hls.C2HLSOrchestrator.run_optimization_step)

    # Ensure max_outer_turns gates correctly: 3 with phase5 on, 2 otherwise.
    has_3_branch = "if phase5_llm_retry else 2" in src
    _check("legacy-2-turn-when-flag-off",
           has_3_branch,
           "max_outer_turns = 2 when C2HLS_PHASE5_LLM_RETRY unset",
           results, tee)


def test_per_step_regression_thresholds(results, tee):
    tee.section("Phase 5 follow-up: per-step regression thresholds")
    # Clear any explicit env override so per-step default fires.
    os.environ.pop("C2HLS_STEP_REGRESSION_THRESHOLD", None)
    os.environ.pop("C2HLS_STEP_REGRESSION_THRESHOLDS_JSON", None)

    prev = {"latency_ns": 100, "lut": 100, "ff": 100, "bram": 100, "dsp": 100}

    # unroll legitimately grows DSP 5x and FF 4x; per-step threshold tolerates.
    unroll_ok = {"latency_ns": 95, "lut": 200, "ff": 400, "bram": 100, "dsp": 500}
    r = c2hls._step_regression_reasons(unroll_ok, prev, 1.10, step_name="unroll")
    _check("unroll-resource-growth-tolerated", not r,
           f"got {r}", results, tee)

    # unroll latency regression should still trip.
    unroll_bad_lat = {"latency_ns": 150, "lut": 110, "ff": 110,
                       "bram": 110, "dsp": 110}
    r = c2hls._step_regression_reasons(unroll_bad_lat, prev, 1.10, step_name="unroll")
    _check("unroll-latency-regression-still-flagged",
           any("latency" in s for s in r), f"got {r}", results, tee)

    # doublebuffer BRAM 2x — within per-step tolerance.
    db_ok = {"latency_ns": 50, "lut": 130, "ff": 130, "bram": 200, "dsp": 100}
    r = c2hls._step_regression_reasons(db_ok, prev, 1.10, step_name="doublebuffer")
    _check("doublebuffer-2x-BRAM-tolerated", not r,
           f"got {r}", results, tee)

    # coalescing knn-style: DSP 8x, latency 0.15x — within per-step tolerance.
    coa_ok = {"latency_ns": 15, "lut": 100, "ff": 130, "bram": 100, "dsp": 800}
    r = c2hls._step_regression_reasons(coa_ok, prev, 1.10, step_name="coalescing")
    _check("coalescing-8x-DSP-tolerated", not r,
           f"got {r}", results, tee)

    # Unknown step name falls through to tight _default.
    unknown = {"latency_ns": 95, "lut": 200, "ff": 200, "bram": 200, "dsp": 200}
    r = c2hls._step_regression_reasons(unknown, prev, 1.10, step_name="unknown_step")
    _check("unknown-step-uses-tight-default",
           any("resource_growth" in s for s in r), f"got {r}", results, tee)

    # Legacy single-threshold env override: explicit 1.10x means the per-step
    # tuning is skipped, all resources held to 1.10x.
    os.environ["C2HLS_STEP_REGRESSION_THRESHOLD"] = "1.10"
    try:
        r = c2hls._step_regression_reasons(unroll_ok, prev, 1.10, step_name="unroll")
        _check("legacy-env-override-tightens-back-to-110",
               any("resource_growth" in s for s in r),
               f"unroll's resource growth flagged again under explicit 1.10x: {r}",
               results, tee)
    finally:
        del os.environ["C2HLS_STEP_REGRESSION_THRESHOLD"]


def test_phase5_legacy_signal_paths_intact(results, tee):
    tee.section("Phase 5 regression: legacy paths still wired")
    src_step = inspect.getsource(c2hls.C2HLSOrchestrator.run_optimization_step)
    # Phase 1 regression detector still runs
    _check("phase1-regression-still-checked",
           "_step_regression_reasons(" in src_step,
           "Phase 1 regression detection still in place",
           results, tee)
    # Phase 9 no-op detector still runs
    _check("phase9-no-op-still-checked",
           "_step_no_op_reasons(" in src_step,
           "Phase 9 no-op detection still in place",
           results, tee)
    # Phase 3 alignment check still runs
    _check("phase3-alignment-still-checked",
           "is_consistent_with_gt_trajectory(" in src_step,
           "Phase 3 alignment still in place",
           results, tee)
    # Phase 4 FeedbackAgent dispatch still in place
    _check("phase4-feedback-agent-still-routed",
           "self.feedback.render(" in src_step
           and "self.feedback.compose_with_llm(" in src_step,
           "render + compose_with_llm both used",
           results, tee)


def main() -> int:
    parser = argparse.ArgumentParser()
    timestamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    parser.add_argument("--out", type=Path,
                        default=REPO_ROOT / "artifacts" / f"phase5_smoke_{timestamp}.md")
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    tee = Tee()
    results: List[Dict[str, Any]] = []

    tee(f"# Phase 5 Smoke Test ({timestamp})")
    tee(f"REPO_ROOT={REPO_ROOT}")

    test_5a_phase5_llm_retry_wiring(results, tee)
    test_5b_gt_prepop_wiring(results, tee)
    test_5b_skill_injection_wiring(results, tee)
    test_phase5_default_off(results, tee)
    test_per_step_regression_thresholds(results, tee)
    test_phase5_legacy_signal_paths_intact(results, tee)

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
