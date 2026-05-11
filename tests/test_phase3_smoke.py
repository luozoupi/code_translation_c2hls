"""Phase 3 offline smoke test:

- trajectory_alignment.is_consistent_with_gt_trajectory cases
- combo-step prompt presence
- C2HLSOrchestrator strategy + gt_aware_revert wiring
- _previous_gt_report_for_step helper

Pure offline: no Vitis required.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import inspect
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import c2hls          # noqa: E402
import prompt_c2hls   # noqa: E402
from trajectory_alignment import (  # noqa: E402
    classify_step_effect_with_alignment,
    is_consistent_with_gt_trajectory,
    render_alignment_for_history,
    render_alignment_for_prompt,
)


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


def test_alignment_tiling_enabling_regression(results, tee):
    tee.section("Phase 3 alignment: tiling enabling regression (knn-shaped)")
    # Gen step roughly matches GT step shape → consistent.
    d = is_consistent_with_gt_trajectory(
        gen_report={"latency_cycles": 4_080_000},
        parent_gen_report={"latency_cycles": 1_050_000},
        gt_report={"latency_cycles": 4_276_372},
        parent_gt_report={"latency_cycles": 1_048_818},
    )
    _check("tiling-enabling-keeps", d.consistent_with_gt,
           f"reason={d.reason[:80]} | gen_ratio={d.gen_latency_ratio:.3f} "
           f"gt_ratio={d.gt_latency_ratio:.3f}", results, tee)


def test_alignment_genuine_bad_step(results, tee):
    tee.section("Phase 3 alignment: genuine bad step (gen regresses, GT improves)")
    d = is_consistent_with_gt_trajectory(
        gen_report={"latency_cycles": 30_000_000},
        parent_gen_report={"latency_cycles": 4_276_372},
        gt_report={"latency_cycles": 4_044_880},     # GT slightly improved
        parent_gt_report={"latency_cycles": 4_276_372},
    )
    _check("genuine-bad-revertable", not d.consistent_with_gt,
           f"reason={d.reason[:80]}", results, tee)


def test_alignment_no_gt_data(results, tee):
    tee.section("Phase 3 alignment: missing GT trajectory")
    d = is_consistent_with_gt_trajectory(
        gen_report={"latency_cycles": 4_000_000},
        parent_gen_report={"latency_cycles": 1_000_000},
        gt_report=None,
        parent_gt_report=None,
    )
    _check("no-gt-data-not-consistent", not d.consistent_with_gt,
           f"reason={d.reason}", results, tee)


def test_alignment_step_effect_relabel(results, tee):
    tee.section("Phase 3 alignment: relabel regressed → enabling_regress")
    from trajectory_alignment import is_consistent_with_gt_trajectory
    d = is_consistent_with_gt_trajectory(
        gen_report={"latency_cycles": 4_080_000},
        parent_gen_report={"latency_cycles": 1_050_000},
        gt_report={"latency_cycles": 4_276_372},
        parent_gt_report={"latency_cycles": 1_048_818},
    )
    relabeled = classify_step_effect_with_alignment("regressed", d)
    _check("regressed-becomes-enabling_regress", relabeled == "enabling_regress",
           f"got '{relabeled}'", results, tee)

    relabeled2 = classify_step_effect_with_alignment("regressed", None)
    _check("no-alignment-keeps-regressed", relabeled2 == "regressed",
           f"got '{relabeled2}'", results, tee)

    relabeled3 = classify_step_effect_with_alignment("improved", None,
                                                      realized_after_enabling=True)
    _check("improved-after-enabling-becomes-realized",
           relabeled3 == "improvement_realized",
           f"got '{relabeled3}'", results, tee)


def test_combo_prompts_present(results, tee):
    tee.section("Phase 3 combo prompts present")
    keys = set(prompt_c2hls.OPTIMIZATION_PROMPTS.keys())
    needed = {"combo_full", "combo_structural", "combo_parallel"}
    _check("combo-keys-in-OPTIMIZATION_PROMPTS",
           needed.issubset(keys),
           f"missing={needed - keys}, all={sorted(keys)}", results, tee)

    _check("COMBO_FULL_STEPS-defined",
           prompt_c2hls.COMBO_FULL_STEPS == ["combo_full"],
           f"got {prompt_c2hls.COMBO_FULL_STEPS}", results, tee)
    _check("COMBO_PROGRESSIVE_STEPS-defined",
           prompt_c2hls.COMBO_PROGRESSIVE_STEPS == ["combo_structural", "combo_parallel"],
           f"got {prompt_c2hls.COMBO_PROGRESSIVE_STEPS}", results, tee)


def test_orchestrator_phase3_fields(results, tee):
    tee.section("Phase 3 orchestrator fields")
    src = inspect.getsource(c2hls.C2HLSOrchestrator.__init__)
    for field in ("self.strategy",
                  "self.gt_aware_revert",
                  "self._gt_step_reports",
                  "self._gt_baseline_report"):
        _check(f"init-has::{field}",
               field in src, "present in __init__", results, tee)

    src_ms = inspect.getsource(c2hls.C2HLSOrchestrator.run_multistep)
    _check("run_multistep-knows-combo-strategy",
           "COMBO_FULL_STEPS" in src_ms or "combo_full" in src_ms,
           "combo dispatch present", results, tee)

    src_step = inspect.getsource(c2hls.C2HLSOrchestrator.run_optimization_step)
    _check("revert-consults-trajectory-alignment",
           "is_consistent_with_gt_trajectory" in src_step,
           "alignment check wired into revert path", results, tee)
    _check("alignment-history-rendered",
           "render_alignment_for_history" in src_step,
           "alignment history record line written", results, tee)


def test_previous_gt_report_for_step(results, tee):
    tee.section("Phase 3 _previous_gt_report_for_step helper")
    orch = c2hls.C2HLSOrchestrator.__new__(c2hls.C2HLSOrchestrator)
    # Bypass __init__; set the minimum fields the helper touches.
    orch._gt_baseline_report = {"latency_cycles": 1_048_818}
    orch._gt_step_reports = {
        "tiling": {"latency_cycles": 4_276_372},
        "pipeline": {"latency_cycles": 4_276_372},
    }
    # parent of pipeline must be tiling
    parent = orch._previous_gt_report_for_step("pipeline")
    _check("parent-of-pipeline-is-tiling",
           parent == orch._gt_step_reports["tiling"],
           f"got {parent}", results, tee)
    # parent of tiling must be baseline
    parent = orch._previous_gt_report_for_step("tiling")
    _check("parent-of-tiling-is-baseline",
           parent == orch._gt_baseline_report,
           f"got {parent}", results, tee)
    # parent of unroll: GT cache only has tiling/pipeline → returns pipeline
    parent = orch._previous_gt_report_for_step("unroll")
    _check("parent-of-unroll-is-pipeline-from-cache",
           parent == orch._gt_step_reports["pipeline"],
           f"got {parent}", results, tee)


def test_render_alignment_outputs(results, tee):
    tee.section("Phase 3 alignment renderers")
    d = is_consistent_with_gt_trajectory(
        gen_report={"latency_cycles": 4_080_000},
        parent_gen_report={"latency_cycles": 1_050_000},
        gt_report={"latency_cycles": 4_276_372},
        parent_gt_report={"latency_cycles": 1_048_818},
    )
    h = render_alignment_for_history(d, "tiling")
    _check("history-line-mentions-step", "[tiling]" in h,
           f"history='{h}'", results, tee)
    p = render_alignment_for_prompt(d, "tiling")
    _check("prompt-block-non-empty", bool(p) and "tiling" in p,
           f"prompt-len={len(p)}", results, tee)
    # When NOT consistent, prompt block should be empty.
    d2 = is_consistent_with_gt_trajectory(
        gen_report={"latency_cycles": 30_000_000},
        parent_gen_report={"latency_cycles": 4_276_372},
        gt_report={"latency_cycles": 4_044_880},
        parent_gt_report={"latency_cycles": 4_276_372},
    )
    p2 = render_alignment_for_prompt(d2, "unroll")
    _check("prompt-block-empty-on-genuine-bad", p2 == "",
           f"prompt='{p2}'", results, tee)


def main() -> int:
    parser = argparse.ArgumentParser()
    timestamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    parser.add_argument("--out", type=Path,
                        default=REPO_ROOT / "artifacts" / f"phase3_smoke_{timestamp}.md")
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    tee = Tee()
    results: List[Dict[str, Any]] = []

    tee(f"# Phase 3 Smoke Test ({timestamp})")
    tee(f"REPO_ROOT={REPO_ROOT}")

    test_alignment_tiling_enabling_regression(results, tee)
    test_alignment_genuine_bad_step(results, tee)
    test_alignment_no_gt_data(results, tee)
    test_alignment_step_effect_relabel(results, tee)
    test_combo_prompts_present(results, tee)
    test_orchestrator_phase3_fields(results, tee)
    test_previous_gt_report_for_step(results, tee)
    test_render_alignment_outputs(results, tee)

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
