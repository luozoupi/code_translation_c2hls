"""Phase 6 offline smoke: best-so-far tracking (6a) + forward_eval
strategy (6b). Pure offline.

    cd /home/luo00466/code_translation-c2hls
    python tests/test_phase6_smoke.py [--out artifacts/phase6_smoke_<ts>.md]
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


def test_score_function(results, tee):
    tee.section("Phase 6a: best-so-far score function")

    f = c2hls.C2HLSOrchestrator._best_so_far_score

    # Lower latency wins
    a = f({"latency_ns": 1000, "bram": 0, "dsp": 0, "ff": 0, "lut": 0})
    b = f({"latency_ns": 500,  "bram": 1000, "dsp": 1000, "ff": 5000, "lut": 5000})
    _check("lower-latency-wins-over-larger-resources",
           b < a,
           f"a(lat=1000)={a}, b(lat=500,large)={b} → b<a? {b<a}",
           results, tee)

    # Tie on latency, smaller resources wins
    c = f({"latency_ns": 1000, "bram": 100, "dsp": 0, "ff": 1000, "lut": 1000})
    d = f({"latency_ns": 1000, "bram": 50,  "dsp": 0, "ff": 500,  "lut": 500})
    _check("tie-on-latency-smaller-resources-wins",
           d < c,
           f"c(big_res)={c}, d(small_res)={d} → d<c? {d<c}",
           results, tee)

    # None / empty report = inf
    none_score = f({})
    _check("empty-report-is-inf",
           none_score == float("inf"),
           f"got {none_score}",
           results, tee)


def test_record_and_promote_best(results, tee):
    tee.section("Phase 6a: _record_best_so_far + _promote_best_so_far")

    orch = c2hls.C2HLSOrchestrator.__new__(c2hls.C2HLSOrchestrator)
    orch.hls_code = "// step5 code"
    orch.synth_report = {"latency_ns": 8000, "bram": 100, "dsp": 0, "ff": 1000, "lut": 1000}
    orch.history = []  # _append_history needs this
    history = []

    # Record baseline
    orch.synth_report = {"latency_ns": 11000, "bram": 50, "dsp": 0, "ff": 500, "lut": 500}
    orch._record_best_so_far(history, step_index=-1, step_name="baseline", source="baseline")

    # Record after step1 = better
    orch.synth_report = {"latency_ns": 6000, "bram": 80, "dsp": 0, "ff": 800, "lut": 800}
    orch.hls_code = "// step1 code"
    orch._record_best_so_far(history, step_index=0, step_name="coalescing", source="step")

    # Record after step2 = worse than step1 (drift!)
    orch.synth_report = {"latency_ns": 14000, "bram": 90, "dsp": 0, "ff": 900, "lut": 900}
    orch.hls_code = "// step2 code"
    orch._record_best_so_far(history, step_index=1, step_name="tiling", source="step")

    # Record after step3 = even worse
    orch.synth_report = {"latency_ns": 14000, "bram": 100, "dsp": 0, "ff": 1000, "lut": 1000}
    orch.hls_code = "// step3 code"
    orch._record_best_so_far(history, step_index=2, step_name="pipeline", source="step")

    # Record after final step = partial recovery
    orch.synth_report = {"latency_ns": 6500, "bram": 110, "dsp": 0, "ff": 1100, "lut": 1100}
    orch.hls_code = "// final code"
    orch._record_best_so_far(history, step_index=3, step_name="unroll", source="step")

    _check("history-records-all-snapshots",
           len(history) == 5,
           f"got {len(history)} records (expect 5)", results, tee)

    # Now promote best-so-far. The trajectory final is 6500; the best
    # mid-state is step1 (6000).
    promoted = orch._promote_best_so_far(history)
    _check("promotion-happened",
           promoted is not None,
           f"promotion result: {promoted is not None}", results, tee)
    if promoted:
        _check("promoted-the-coalescing-mid-step",
               promoted.get("step_name") == "coalescing",
               f"step_name={promoted.get('step_name')} (expect coalescing)",
               results, tee)
        _check("orchestrator-state-snapped-back",
               orch.synth_report.get("latency_ns") == 6000
               and orch.hls_code == "// step1 code",
               f"lat={orch.synth_report.get('latency_ns')}, "
               f"code-prefix='{orch.hls_code[:20]}'",
               results, tee)


def test_promote_no_op_when_final_is_best(results, tee):
    tee.section("Phase 6a: promote is a no-op when final IS the best")

    orch = c2hls.C2HLSOrchestrator.__new__(c2hls.C2HLSOrchestrator)
    orch.history = []
    history = []

    # Monotone improvement trajectory
    orch.synth_report = {"latency_ns": 11000, "bram": 50, "dsp": 0, "ff": 500, "lut": 500}
    orch.hls_code = "// baseline"
    orch._record_best_so_far(history, step_index=-1, step_name="baseline", source="baseline")

    orch.synth_report = {"latency_ns": 8000, "bram": 60, "dsp": 0, "ff": 600, "lut": 600}
    orch.hls_code = "// step1"
    orch._record_best_so_far(history, step_index=0, step_name="coalescing", source="step")

    orch.synth_report = {"latency_ns": 4000, "bram": 70, "dsp": 0, "ff": 700, "lut": 700}
    orch.hls_code = "// step2"
    orch._record_best_so_far(history, step_index=1, step_name="doublebuffer", source="step")

    promoted = orch._promote_best_so_far(history)
    _check("monotone-improvement-no-promotion",
           promoted is None,
           f"got {promoted}; expected None (final state already best)",
           results, tee)
    _check("orchestrator-state-untouched",
           orch.synth_report.get("latency_ns") == 4000
           and orch.hls_code == "// step2",
           "final state preserved", results, tee)


def test_forward_eval_helper_exists(results, tee):
    tee.section("Phase 6b: run_optimization_step_forward helper")

    src = inspect.getsource(c2hls.C2HLSOrchestrator.run_optimization_step_forward)
    _check("forward-helper-defined",
           "forward_eval" in src.lower(),
           "function exists with forward_eval semantics",
           results, tee)
    _check("forward-helper-no-revert-on-regression",
           "regression_warnings" in src and "forward_eval_committed" in src,
           "logs regression but commits unconditionally",
           results, tee)
    _check("forward-helper-uses-attempt-helper",
           "_optimization_step_attempt(" in src,
           "delegates to existing per-step attempt (which has correctness gates)",
           results, tee)


def test_run_multistep_forward_eval_dispatch(results, tee):
    tee.section("Phase 6b: run_multistep dispatches forward_eval correctly")
    src = inspect.getsource(c2hls.C2HLSOrchestrator.run_multistep)
    _check("dispatch-checks-forward-eval-static-loop",
           'if self.strategy == "forward_eval":' in src
           and "run_optimization_step_forward(" in src,
           "static loop branches on strategy=forward_eval",
           results, tee)
    # Same dispatch in the dynamic-routing branch
    _check("dispatch-also-in-dynamic-branch",
           src.count('if self.strategy == "forward_eval":') >= 2
           and src.count("run_optimization_step_forward(") >= 2,
           "both static and dynamic branches dispatch to forward helper",
           results, tee)


def test_run_multistep_emits_best_so_far(results, tee):
    tee.section("Phase 6a: run_multistep emits best-so-far telemetry")
    src = inspect.getsource(c2hls.C2HLSOrchestrator.run_multistep)
    _check("best-so-far-history-tracked",
           "best_so_far_history" in src,
           "history list is created and populated",
           results, tee)
    _check("promotion-runs-at-end",
           "_promote_best_so_far(best_so_far_history)" in src,
           "promotion is invoked before returning results",
           results, tee)
    _check("results-include-best-so-far-keys",
           '"best_so_far_history"' in src and '"best_so_far_promotion"' in src,
           "results dict surfaces the new fields",
           results, tee)


def test_cli_strategy_flag(results, tee):
    tee.section("Phase 6b: --strategy=forward_eval CLI flag accepted")
    # Find the argparse setup
    src = inspect.getsource(c2hls)
    _check("forward-eval-in-cli-choices",
           '"forward_eval"' in src and 'choices=' in src,
           "added to argparse choices for --strategy",
           results, tee)


def main() -> int:
    parser = argparse.ArgumentParser()
    timestamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    parser.add_argument("--out", type=Path,
                        default=REPO_ROOT / "artifacts" / f"phase6_smoke_{timestamp}.md")
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    tee = Tee()
    results: List[Dict[str, Any]] = []

    tee(f"# Phase 6 Smoke Test ({timestamp})")
    tee(f"REPO_ROOT={REPO_ROOT}")

    test_score_function(results, tee)
    test_record_and_promote_best(results, tee)
    test_promote_no_op_when_final_is_best(results, tee)
    test_forward_eval_helper_exists(results, tee)
    test_run_multistep_forward_eval_dispatch(results, tee)
    test_run_multistep_emits_best_so_far(results, tee)
    test_cli_strategy_flag(results, tee)

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
