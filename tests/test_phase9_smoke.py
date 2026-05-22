"""Phase 9 offline smoke: csim/cosim correctness-repair loop in
_optimization_step_attempt. Pure offline — no Vitis, no LLM calls.

Verifies:
  * hls_correctness_repair_fix prompt is importable and well-formed
  * _optimization_step_attempt source contains the csim/cosim gating
    block and continues the loop on correctness failure
  * default-on (no env flag needed); disable via
    C2HLS_DISABLE_CORRECTNESS_REPAIR=1
  * loop-exhaustion error message distinguishes correctness-budget from
    synth-budget exhaustion

    cd /home/luo00466/code_translation-c2hls
    python tests/test_phase9_smoke.py [--out artifacts/phase9_smoke_<ts>.md]
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
import prompt_c2hls  # noqa: E402


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


def test_correctness_prompt_shape(results, tee):
    tee.section("Phase 9: correctness-repair prompt shape")
    p = getattr(prompt_c2hls, "hls_correctness_repair_fix", None)
    _check("prompt-defined", p is not None,
           "hls_correctness_repair_fix exists in prompt_c2hls", results, tee)
    if not p:
        return
    rendered = p.format(
        step_name="coalescing",
        gate_name="csim",
        gate_error="Testbench mismatch at index 17: expected 42, got 0",
        hls_code="// (broken kernel)",
        header_code="// kernel.h",
        benchmark_context="(none)",
        attempt_history="",
    )
    _check("mentions-gate-name",
           "csim" in rendered,
           "rendered prompt includes the failing gate name", results, tee)
    _check("mentions-step-name",
           "coalescing" in rendered,
           "rendered prompt includes step name", results, tee)
    _check("warns-no-revert-whole-step",
           "do NOT revert the whole step" in rendered,
           "instructs LLM to keep optimization intent",
           results, tee)
    _check("mentions-defect-categories",
           all(k in rendered for k in (
               "Loop bounds", "Buffer indices", "Reduction order",
               "AXI burst", "Pipelined inner loop",
           )),
           "lists the common HLS-correctness defect classes",
           results, tee)
    _check("provides-error-context",
           "Testbench mismatch at index 17" in rendered,
           "the actual testbench error makes it into the prompt",
           results, tee)


def test_step_attempt_wires_correctness_repair(results, tee):
    tee.section("Phase 9: _optimization_step_attempt wires correctness repair")
    src = inspect.getsource(c2hls.C2HLSOrchestrator._optimization_step_attempt_single)
    _check("references-prompt",
           "hls_correctness_repair_fix" in src,
           "_optimization_step_attempt builds the correctness-repair prompt",
           results, tee)
    _check("checks-csim-failed",
           "csim_summary" in src and "csim_failed" in src,
           "checks csim outcome before returning success", results, tee)
    _check("checks-cosim-failed",
           "cosim_summary" in src and "cosim_failed" in src,
           "checks cosim outcome before returning success", results, tee)
    _check("env-flag-default-on",
           "C2HLS_DISABLE_CORRECTNESS_REPAIR" in src,
           "default-on; disable via env flag", results, tee)
    _check("continue-on-correctness-fail",
           "continue" in src.split("hls_correctness_repair_fix.format")[1],
           "after building the repair prompt, the loop continues to retry",
           results, tee)
    _check("preserves-csim-cosim-on-success",
           ('step_result["csim"] = outcome["csim"]' in src
            and 'step_result["cosim"] = outcome["cosim"]' in src),
           "csim/cosim outcomes still attached to successful step result",
           results, tee)
    _check("budget-exhaustion-distinguishes-modes",
           "Correctness repair exhausted" in src,
           "loop-exhausted error message distinguishes correctness vs synth",
           results, tee)


def test_summary_failure_log_extraction(results, tee):
    tee.section("Phase 9: failure log excerpt extraction shape")
    # Verify _summarize_test_result populates error/log_excerpt on failure
    # — that's what the new repair prompt feeds to the LLM. Pure shape
    # check, no Vitis run.
    fake_failure = {
        "success": True,  # tool ran successfully
        "passed": False,  # but testbench mismatch
        "error": "Testbench did not pass",
        "log": (
            "[INFO] Running C-simulation\n"
            "TEST FAILED: Output mismatch at index 17: expected 42, got 0\n"
            "[ERROR] csim_design failed\n"
        ),
    }
    summary = c2hls._summarize_test_result(fake_failure, supported=True)
    _check("summary-records-failure",
           summary.get("ran") and not summary.get("passed"),
           f"ran={summary.get('ran')} passed={summary.get('passed')}",
           results, tee)
    _check("summary-includes-error",
           bool(summary.get("error")),
           f"error={summary.get('error')!r}", results, tee)
    _check("summary-may-include-log-excerpt",
           "log_excerpt" in summary or summary.get("error"),
           "either log_excerpt or error carries diagnostic info",
           results, tee)


def test_disable_env_flag(results, tee):
    tee.section("Phase 9: disable env flag honored")
    src = inspect.getsource(c2hls.C2HLSOrchestrator._optimization_step_attempt_single)
    # Env is read into a local flag, then that flag gates the repair
    # branch. Both halves must be present for runtime disable to work.
    _check("env-read-into-local-flag",
           "correctness_disabled = bool(int(" in src
           and "C2HLS_DISABLE_CORRECTNESS_REPAIR" in src,
           "env read into a local 'correctness_disabled' flag",
           results, tee)
    _check("flag-blocks-repair",
           "and not correctness_disabled" in src,
           "repair branch only fires when not disabled",
           results, tee)


def main() -> int:
    parser = argparse.ArgumentParser()
    timestamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    parser.add_argument("--out", type=Path,
                        default=REPO_ROOT / "artifacts" / f"phase9_smoke_{timestamp}.md")
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    tee = Tee()
    results: List[Dict[str, Any]] = []

    tee(f"# Phase 9 Smoke Test ({timestamp})")
    tee(f"REPO_ROOT={REPO_ROOT}")

    test_correctness_prompt_shape(results, tee)
    test_step_attempt_wires_correctness_repair(results, tee)
    test_summary_failure_log_extraction(results, tee)
    test_disable_env_flag(results, tee)

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
