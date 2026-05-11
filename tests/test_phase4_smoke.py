"""Phase 4 offline smoke: FeedbackAgent dispatch + deterministic-mode
parity with the pre-Phase-4 free-function renderers + LLM-aided fallback
behavior.

Pure offline; no Vitis or LLM calls.

    cd /home/luo00466/code_translation-c2hls
    python tests/test_phase4_smoke.py [--out artifacts/phase4_smoke_<ts>.md]
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

import c2hls                  # noqa: E402
import robustness as _robustness  # noqa: E402
from trajectory_alignment import is_consistent_with_gt_trajectory  # noqa: E402


class _StubOrch:
    """Minimum orchestrator surface FeedbackAgent touches in deterministic
    mode. Avoids spinning up an Anthropic client just to test rendering."""
    part = "xc7a100t-csg324-1"
    clock_ns = 4.0
    gpt_model = "claude-haiku-4-5-20251001"
    benchmark_name = "knn"


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


def test_feedback_agent_class_exists(results, tee):
    tee.section("Phase 4: FeedbackAgent class wiring")
    _check("FeedbackAgent-defined", hasattr(c2hls, "FeedbackAgent"),
           "class present in c2hls", results, tee)
    _check("FeedbackAgent-AGENT_NAME", c2hls.FeedbackAgent.AGENT_NAME == "feedback",
           f"got '{c2hls.FeedbackAgent.AGENT_NAME}'", results, tee)
    _check("FeedbackAgent-MODEL_ENV",
           c2hls.FeedbackAgent.MODEL_ENV == "C2HLS_FEEDBACK_MODEL",
           f"got '{c2hls.FeedbackAgent.MODEL_ENV}'", results, tee)
    init_src = inspect.getsource(c2hls.C2HLSOrchestrator.__init__)
    _check("orch-init-creates-feedback", "self.feedback = FeedbackAgent(self)" in init_src,
           "FeedbackAgent attached to orchestrator", results, tee)


def test_dispatch_kind_table(results, tee):
    tee.section("Phase 4: render() dispatch by kind")
    fa = c2hls.FeedbackAgent(_StubOrch())

    # Each tuple: (kind, kwargs, expected_min_len, expected_substr_or_None)
    # ``expected_substr=None`` only checks that a string came back. The
    # compile_error / synth_error renderers produce generic guidance prose
    # rather than echoing the specific error literal, so we check for the
    # generic phrasing.
    cases = [
        ("regression", {"step_name": "unroll",
                         "reasons": ["latency_ns regressed 6.88x"]},
         100, "your previous attempt at the `unroll` step was rejected"),
        ("no_op", {"step_name": "pipeline",
                    "reasons": ["identical_synth_tuple", "all unchanged"]},
         100, "your previous attempt at the `pipeline` step did not change"),
        ("compile_error", {"err": "fatal error: ap_int.h: No such file or directory"},
         50, "smallest change"),
        ("synth_error", {"err": "ERROR: [HLS 200-1023]"}, 50, "smallest change"),
        ("quality_gap", {
            "bench_name": "knn",
            "report": {"slack_ns": -0.5, "fmax_mhz": 250.0},
            "ground_truth_report": {"fmax_mhz": 300.0},
            "comparison": {"latency_ns": {"ratio": 2.5}}}, 0, None),
        ("unknown_kind", {}, 0, None),
    ]
    for kind, kwargs, min_len, expect_substr in cases:
        out = fa.render(kind, **kwargs)
        ok_len = isinstance(out, str) and len(out) >= min_len
        if expect_substr is None:
            _check(f"dispatch::{kind}-returns-str", ok_len,
                   f"kind={kind} len={len(out)} >= {min_len}", results, tee)
        else:
            ok = ok_len and expect_substr.lower() in out.lower()
            _check(f"dispatch::{kind}-emits", ok,
                   f"kind={kind}, len={len(out)}, expect-substr='{expect_substr[:30]}', "
                   f"first-100='{out[:100].replace(chr(10), ' ')}'", results, tee)


def test_alignment_dispatch(results, tee):
    tee.section("Phase 4: alignment dispatch (Phase 3 piece)")
    fa = c2hls.FeedbackAgent(_StubOrch())

    consistent = is_consistent_with_gt_trajectory(
        gen_report={"latency_cycles": 4_080_000},
        parent_gen_report={"latency_cycles": 1_050_000},
        gt_report={"latency_cycles": 4_276_372},
        parent_gt_report={"latency_cycles": 1_048_818},
    )
    out = fa.render("alignment", step_name="tiling", decision=consistent)
    _check("alignment-consistent-emits-block",
           "structural enabler" in out.lower() and "tiling" in out,
           f"len={len(out)}", results, tee)

    inconsistent = is_consistent_with_gt_trajectory(
        gen_report={"latency_cycles": 30_000_000},
        parent_gen_report={"latency_cycles": 4_276_372},
        gt_report={"latency_cycles": 4_044_880},
        parent_gt_report={"latency_cycles": 4_276_372},
    )
    out = fa.render("alignment", step_name="unroll", decision=inconsistent)
    _check("alignment-inconsistent-empty", out == "",
           f"genuine bad step → empty fragment, got len={len(out)}", results, tee)


def test_throughput_dispatch(results, tee):
    tee.section("Phase 4: throughput-regression dispatch")
    fa = c2hls.FeedbackAgent(_StubOrch())

    flagged = _robustness.throughput_regression_check(
        {"latency_cycles": 342676, "interval": 680597},
        {"latency_cycles": 1056310, "interval": 1056311},
    )
    out = fa.render("throughput_regression", step_name="doublebuffer", check=flagged)
    _check("throughput-flagged-emits-block",
           "throughput" in out.lower() and "doublebuffer" in out,
           f"len={len(out)}", results, tee)

    not_flagged = _robustness.throughput_regression_check(
        {"latency_cycles": 524000, "interval": 524001},
        {"latency_cycles": 1048816, "interval": 1048817},
    )
    out = fa.render("throughput_regression", step_name="ok", check=not_flagged)
    _check("throughput-not-flagged-empty", out == "",
           f"clean step → empty, got len={len(out)}", results, tee)


def test_compose_with_llm_default_off(results, tee):
    tee.section("Phase 4: compose_with_llm() default-off behavior")
    fa = c2hls.FeedbackAgent(_StubOrch())

    # Default off: returns prior_template verbatim, no LLM call.
    os.environ.pop("C2HLS_FEEDBACK_LLM", None)
    out = fa.compose_with_llm(
        "regression", prior_template="deterministic-test-string",
        kernel_diff="// some diff",
    )
    _check("default-off-returns-template", out == "deterministic-test-string",
           f"got '{out[:60]}...'", results, tee)

    # Falls back to render() when no prior_template provided
    out2 = fa.compose_with_llm(
        "regression", step_name="unroll",
        reasons=["latency_ns regressed 2.0x"],
    )
    _check("default-off-falls-through-to-render",
           "unroll" in out2 and "rejected" in out2.lower(),
           f"len={len(out2)}, first-80='{out2[:80]}'", results, tee)


def test_renderers_match_legacy_free_functions(results, tee):
    tee.section("Phase 4: deterministic-mode parity with legacy renderers")
    fa = c2hls.FeedbackAgent(_StubOrch())

    legacy_no_op = c2hls._render_no_op_guidance(
        "pipeline", ["identical_synth_tuple", "all unchanged"],
    )
    new_no_op = fa.render("no_op", step_name="pipeline",
                            reasons=["identical_synth_tuple", "all unchanged"])
    _check("no_op-parity-with-legacy-fn", new_no_op == legacy_no_op,
           "FeedbackAgent.render('no_op', ...) == _render_no_op_guidance(...)",
           results, tee)

    legacy_reg = c2hls._render_regression_guidance(
        "unroll", ["latency_ns regressed 6.88x"],
    )
    new_reg = fa.render("regression", step_name="unroll",
                          reasons=["latency_ns regressed 6.88x"])
    _check("regression-parity-with-legacy-fn", new_reg == legacy_reg,
           "FeedbackAgent.render('regression', ...) == _render_regression_guidance(...)",
           results, tee)

    legacy_compile = c2hls._build_repair_guidance(
        "fatal error: ap_int.h: No such file"
    )
    new_compile = fa.render("compile_error", err="fatal error: ap_int.h: No such file")
    _check("compile_error-parity-with-legacy-fn",
           new_compile == legacy_compile,
           "FeedbackAgent.render('compile_error', ...) == _build_repair_guidance(...)",
           results, tee)


def test_call_sites_use_feedback_agent(results, tee):
    tee.section("Phase 4: orchestrator call-sites delegate to feedback agent")
    # The call-sites use multi-line formatting, so we look for the
    # "self.feedback.render(" or "orch.feedback.render(" substring plus
    # the kind literal anywhere in the source (allowing line breaks
    # between).
    src_step = inspect.getsource(c2hls.C2HLSOrchestrator.run_optimization_step)
    _check("multistep-revert-uses-feedback",
           ("self.feedback.render(" in src_step
            and '"regression"' in src_step
            and '"no_op"' in src_step),
           "regression + no_op paths route through feedback agent",
           results, tee)

    src_qr = inspect.getsource(c2hls.QualityRepairAgent.run)
    _check("quality-repair-uses-feedback",
           "orch.feedback.render(" in src_qr and '"quality_gap"' in src_qr,
           "quality_gap path routes through feedback agent",
           results, tee)


def main() -> int:
    parser = argparse.ArgumentParser()
    timestamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    parser.add_argument("--out", type=Path,
                        default=REPO_ROOT / "artifacts" / f"phase4_smoke_{timestamp}.md")
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    tee = Tee()
    results: List[Dict[str, Any]] = []

    tee(f"# Phase 4 Smoke Test ({timestamp})")
    tee(f"REPO_ROOT={REPO_ROOT}")

    test_feedback_agent_class_exists(results, tee)
    test_dispatch_kind_table(results, tee)
    test_alignment_dispatch(results, tee)
    test_throughput_dispatch(results, tee)
    test_compose_with_llm_default_off(results, tee)
    test_renderers_match_legacy_free_functions(results, tee)
    test_call_sites_use_feedback_agent(results, tee)

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
