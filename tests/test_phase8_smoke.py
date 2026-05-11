"""Phase 8 offline smoke: baseline-alignment gap computer, metric-only
guidance renderer, TranslatorAgent.retranslate_with_guidance,
Orchestrator._baseline_alignment_loop wiring. Pure offline — no Vitis,
no LLM calls.

    cd /home/luo00466/code_translation-c2hls
    python tests/test_phase8_smoke.py [--out artifacts/phase8_smoke_<ts>.md]
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


# Use philip's pathfinder reference baseline as the comparison fixture.
REF_PATHFINDER_BASELINE = {
    "latency_cycles": 2_113_742,
    "latency_ns": 7_045_000.0,
    "interval": 2_113_743,
    "bram": 30, "dsp": 0, "ff": 11_602, "lut": 23_216,
    "fmax_mhz": 392.9,
}


def test_baseline_gap_within_tolerance(results, tee):
    tee.section("Phase 8: gap detection — matched baseline")
    # Sonnet's pathfinder baseline matched ref to within 0.1% in cycles
    # and was within 2x on each resource. Should report within_tolerance.
    ours = {"latency_cycles": 2_110_674, "bram": 35, "dsp": 0,
            "ff": 8_125, "lut": 5_355, "fmax_mhz": 187.0,
            "latency_ns": 11_285_000}
    gap = c2hls._compute_baseline_gap(ours, REF_PATHFINDER_BASELINE)
    _check("within-tolerance-true", gap["within_tolerance"],
           f"got {gap.get('within_tolerance')}, "
           f"lat_ratio={gap.get('latency_ratio'):.3f}", results, tee)
    _check("latency-ratio-near-1",
           abs((gap.get("latency_ratio") or 0) - 1.0) < 0.05,
           f"latency_ratio={gap.get('latency_ratio')}", results, tee)


def test_baseline_gap_70x_worse(results, tee):
    tee.section("Phase 8: gap detection — 70x worse baseline (knn-static-style)")
    # The original knn-static run had a 72M-cycle baseline vs philip's
    # 1.05M reference baseline = 70x worse.
    ours = {"latency_cycles": 72_351_885, "bram": 32, "dsp": 5,
            "ff": 7_071, "lut": 4_961, "fmax_mhz": 22.6,
            "latency_ns": 3_206_000_000}
    ref_knn = {"latency_cycles": 1_048_818, "bram": 30, "dsp": 14,
               "ff": 8_012, "lut": 5_802, "fmax_mhz": 411.0}
    gap = c2hls._compute_baseline_gap(ours, ref_knn)
    _check("within-tolerance-false-on-70x",
           not gap["within_tolerance"],
           f"got {gap.get('within_tolerance')}", results, tee)
    _check("latency-ratio-correct-70x",
           60 < (gap.get("latency_ratio") or 0) < 75,
           f"latency_ratio={gap.get('latency_ratio')}", results, tee)
    _check("latency-over-flag-set",
           gap.get("latency_over") is True,
           f"latency_over={gap.get('latency_over')}", results, tee)


def test_baseline_gap_resource_blowup(results, tee):
    tee.section("Phase 8: gap detection — resource over-budget but latency OK")
    ours = {"latency_cycles": 2_200_000, "bram": 200, "dsp": 0,  # BRAM 6.7x over
            "ff": 9_000, "lut": 5_500}
    gap = c2hls._compute_baseline_gap(ours, REF_PATHFINDER_BASELINE)
    _check("within-tolerance-false-on-resource",
           not gap["within_tolerance"],
           f"got {gap.get('within_tolerance')}", results, tee)
    over_resources = gap.get("over_resources") or []
    _check("bram-flagged-as-over",
           any(k == "bram" for k, *_ in over_resources),
           f"over_resources={over_resources}", results, tee)


def test_guidance_renderer(results, tee):
    tee.section("Phase 8: guidance renderer (metric-only, no GT-leak)")
    ours = {"latency_cycles": 72_351_885, "bram": 32, "dsp": 5,
            "ff": 7_071, "lut": 4_961, "fmax_mhz": 22.6}
    ref_knn = {"latency_cycles": 1_048_818, "bram": 30, "dsp": 14,
               "ff": 8_012, "lut": 5_802, "fmax_mhz": 411.0}
    gap = c2hls._compute_baseline_gap(ours, ref_knn)
    g = c2hls._render_baseline_alignment_guidance(gap)
    _check("guidance-non-empty", bool(g),
           f"guidance length {len(g)}", results, tee)
    _check("guidance-mentions-latency-multiplier",
           "× slower" in g,
           f"first 200 chars: {g[:200]}", results, tee)
    _check("guidance-no-gt-code-leak",
           "void workload" not in g and "#pragma HLS pipeline" not in g,
           "guidance does not include reference HLS source",
           results, tee)
    _check("guidance-no-pragma-instructions",
           "Do NOT add optimization pragmas" in g,
           "explicitly tells LLM to keep the retranslation conservative",
           results, tee)
    _check("guidance-empty-when-aligned",
           c2hls._render_baseline_alignment_guidance(
               {"within_tolerance": True}) == "",
           "no guidance emitted when already aligned", results, tee)


def test_translator_retranslate_method(results, tee):
    tee.section("Phase 8: TranslatorAgent.retranslate_with_guidance signature")
    src = inspect.getsource(c2hls.TranslatorAgent.retranslate_with_guidance)
    _check("uses-translation-template",
           "q_translate_c_to_hls.format(" in src,
           "reuses the canonical translate prompt as base", results, tee)
    _check("appends-alignment-feedback",
           "BASELINE ALIGNMENT FEEDBACK" in src,
           "appends a clearly-marked guidance section", results, tee)
    _check("returns-extracted-code",
           "extract_cpp_code(reply)" in src,
           "extracts cpp from LLM reply", results, tee)
    _check("logs-as-phase-8",
           "Phase 8" in src,
           "log lines tagged with Phase 8 for grepping", results, tee)


def test_orchestrator_wiring(results, tee):
    tee.section("Phase 8: orchestrator wiring")
    src_loop = inspect.getsource(c2hls.C2HLSOrchestrator._baseline_alignment_loop)
    _check("env-flag-gated", "C2HLS_PHASE8_BASELINE_ALIGN" in src_loop,
           "loop is opt-in via env flag", results, tee)
    _check("returns-when-disabled",
           "outcome[\"enabled\"] = False" in src_loop or 'enabled": False' in src_loop,
           "noop when not enabled", results, tee)
    _check("respects-attempt-cap",
           "C2HLS_PHASE8_MAX_ATTEMPTS" in src_loop,
           "max attempts is tunable", results, tee)
    _check("respects-tolerance-envs",
           "C2HLS_PHASE8_BASELINE_LATENCY_TOL" in src_loop
           and "C2HLS_PHASE8_BASELINE_RESOURCE_TOL" in src_loop,
           "both tolerance envs are read", results, tee)
    _check("calls-retranslate",
           "retranslate_with_guidance(" in src_loop,
           "delegates to TranslatorAgent.retranslate_with_guidance",
           results, tee)
    _check("synth-after-retranslate",
           "synthesize_with_repair()" in src_loop,
           "re-runs synth after each retranslation", results, tee)

    src_ms = inspect.getsource(c2hls.C2HLSOrchestrator.run_multistep)
    _check("multistep-invokes-alignment-loop",
           "_baseline_alignment_loop(" in src_ms,
           "run_multistep calls the alignment loop", results, tee)
    _check("multistep-runs-alignment-before-phase-c",
           src_ms.find("_baseline_alignment_loop(") < src_ms.find("self.run_phase_c("),
           "alignment loop runs before Phase C", results, tee)
    _check("results-include-baseline-alignment",
           '"baseline_alignment"' in src_ms,
           "outcome surfaced in returned results dict", results, tee)


def test_disabled_by_default(results, tee):
    tee.section("Phase 8: disabled by default (no env flag set)")
    os.environ.pop("C2HLS_PHASE8_BASELINE_ALIGN", None)
    orch = c2hls.C2HLSOrchestrator.__new__(c2hls.C2HLSOrchestrator)
    orch.synth_report = {"latency_cycles": 100}
    orch.history = []
    outcome = orch._baseline_alignment_loop(REF_PATHFINDER_BASELINE)
    _check("disabled-when-flag-unset",
           outcome.get("enabled") is False,
           f"got {outcome}", results, tee)
    _check("attempts-zero-when-disabled",
           outcome.get("attempts") == 0,
           f"attempts={outcome.get('attempts')}", results, tee)


def main() -> int:
    parser = argparse.ArgumentParser()
    timestamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    parser.add_argument("--out", type=Path,
                        default=REPO_ROOT / "artifacts" / f"phase8_smoke_{timestamp}.md")
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    tee = Tee()
    results: List[Dict[str, Any]] = []

    tee(f"# Phase 8 Smoke Test ({timestamp})")
    tee(f"REPO_ROOT={REPO_ROOT}")

    test_baseline_gap_within_tolerance(results, tee)
    test_baseline_gap_70x_worse(results, tee)
    test_baseline_gap_resource_blowup(results, tee)
    test_guidance_renderer(results, tee)
    test_translator_retranslate_method(results, tee)
    test_orchestrator_wiring(results, tee)
    test_disabled_by_default(results, tee)

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
