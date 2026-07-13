"""Phase 12 offline smoke: functional Phase B + persisted skill routing.

No Vitis or LLM calls. This verifies the performance-plan plumbing:
conservative multistep baseline, AXI burst-widening skill selection,
static burst bottlenecks, and bounded candidate configuration.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import inspect
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import bottleneck_router  # noqa: E402
import c2hls  # noqa: E402
import hls_feedback as hf  # noqa: E402
import prompt_c2hls  # noqa: E402
from skill_library import make_default_library, render_skill_for_prompt  # noqa: E402


class Tee:
    def __init__(self):
        self.buf: List[str] = []

    def __call__(self, *a):
        line = " ".join(str(x) for x in a)
        print(line)
        self.buf.append(line)

    def section(self, title):
        print()
        print(f"=== {title} ===")
        self.buf.append("")
        self.buf.append(f"## {title}")
        self.buf.append("")

    def md(self):
        return "\n".join(self.buf) + "\n"


def _check(label, ok, detail, results, tee):
    results.append({"check": label, "ok": bool(ok), "detail": detail})
    icon = "OK  " if ok else "FAIL"
    tee(f"[{icon}] {label}: {detail}")


def test_phaseb_functional_mode(results, tee):
    tee.section("Phase B functional baseline mode")
    os.environ.pop(c2hls.PHASEB_MODE_ENV, None)
    _check(
        "multistep-default-functional",
        c2hls._normalize_phaseb_mode("", multistep=True) == "functional",
        "empty env resolves to functional for multistep",
        results,
        tee,
    )
    _check(
        "single-default-optimized",
        c2hls._normalize_phaseb_mode("", multistep=False) == "optimized",
        "empty env resolves to optimized for single-shot",
        results,
        tee,
    )
    prompt = prompt_c2hls.q_translate_c_to_hls_functional
    _check(
        "functional-prompt-forbids-performance-pragmas",
        "Do NOT add performance optimizations" in prompt
        and "no `#pragma HLS PIPELINE`" in prompt
        and "no `#pragma HLS UNROLL`" in prompt
        and "no `ap_uint<512>`" in prompt,
        "functional prompt explicitly keeps optimizations out of Phase B",
        results,
        tee,
    )
    src = inspect.getsource(c2hls.TranslatorAgent.translate_initial)
    _check(
        "translator-selects-functional-template",
        "q_translate_c_to_hls_functional" in src and "phaseb_mode" in src,
        "TranslatorAgent chooses prompt template by phaseb_mode",
        results,
        tee,
    )


def test_flash_mode_wiring(results, tee):
    tee.section("Flash mode wiring")
    _check(
        "flash-prompt-registered",
        "flash" in prompt_c2hls.OPTIMIZATION_PROMPTS
        and prompt_c2hls.FLASH_STEPS == ["flash"],
        "flash is a one-step optimization prompt",
        results,
        tee,
    )
    run_src = inspect.getsource(c2hls.C2HLSOrchestrator.run_multistep)
    _check(
        "flash-short-circuits-steps",
        "FLASH_STEPS" in run_src and 'self.strategy == "flash"' in run_src,
        "run_multistep maps strategy=flash to one step",
        results,
        tee,
    )
    sweep_src = Path(REPO_ROOT / "run_agentic_sweep.py").read_text()
    _check(
        "sweep-strategy-env",
        "C2HLS_SWEEP_STRATEGY" in sweep_src and "sweep_strategy" in sweep_src,
        "sweep runner can select flash without code edits",
        results,
        tee,
    )
    prompt = prompt_c2hls.q_optimize_flash
    _check(
        "flash-axi-legal-pragmas",
        "powers of two" in prompt and "adapter parameters" in prompt,
        "flash prompt constrains Vitis m_axi pragma values",
        results,
        tee,
    )
    clean = {
        "latency_ns": 1000.0,
        "slack_ns": 0.1,
        "estimated_clock_period_ns": 3.2,
        "requested_clock_period_ns": 3.33,
        "bram": 1,
        "dsp": 1,
        "ff": 100,
        "lut": 100,
        "uram": 0,
    }
    timing_bad = {
        "latency_ns": 10.0,
        "slack_ns": -0.5,
        "estimated_clock_period_ns": 3.83,
        "requested_clock_period_ns": 3.33,
        "bram": 1,
        "dsp": 1,
        "ff": 100,
        "lut": 100,
        "uram": 0,
    }
    slow_timing_bad = {
        "latency_ns": 1000.0,
        "slack_ns": -0.5,
        "estimated_clock_period_ns": 3.83,
        "requested_clock_period_ns": 3.33,
        "bram": 1,
        "dsp": 1,
        "ff": 100,
        "lut": 100,
        "uram": 0,
    }
    _check(
        "flash-ranker-allows-large-latency-win",
        c2hls.C2HLSOrchestrator._best_so_far_score(timing_bad)
        < c2hls.C2HLSOrchestrator._best_so_far_score(clean),
        "large latency wins are not erased solely by estimated negative slack",
        results,
        tee,
    )
    _check(
        "paper-feasibility-rejects-slow-timing-bad",
        c2hls._paper_candidate_feasibility(
            clean,
            csim={"ran": True, "passed": True},
            part="xcu280-fsvh2892-2L-e",
            clock_ns=3.33,
        )["feasible"]
        and not c2hls._paper_candidate_feasibility(
            slow_timing_bad,
            csim={"ran": True, "passed": True},
            part="xcu280-fsvh2892-2L-e",
            clock_ns=3.33,
        )["feasible"],
        "paper selection gates target timing before latency ranking",
        results,
        tee,
    )
    reasons = c2hls._step_regression_reasons(
        slow_timing_bad,
        clean,
        step_name="flash",
        part="xcu280-fsvh2892-2L-e",
    )
    _check(
        "flash-regression-rejects-negative-slack-without-win",
        any("timing_not_clean" in r for r in reasons),
        "negative slack remains explicit when there is no large latency win",
        results,
        tee,
    )
    fast_reasons = c2hls._step_regression_reasons(
        timing_bad,
        clean,
        step_name="flash",
        part="xcu280-fsvh2892-2L-e",
    )
    _check(
        "flash-regression-allows-fast-device-fitting-slack",
        not any("timing_not_clean" in r for r in fast_reasons),
        "negative slack is recorded in the report but not an automatic revert for a 2x+ latency win that fits device",
        results,
        tee,
    )


def test_skill_library_and_router(results, tee):
    tee.section("Skill library and router")
    with tempfile.TemporaryDirectory() as td:
        lib = make_default_library(Path(td) / "skills.json", persist=False)
        ids = {sk.id for sk in lib.all()}
        _check(
            "schema11-skill-package-present",
            "axi-burst-widening-512" in ids
            and "hls-coalescing-512-compound-transform" in ids
            and len(ids) >= 55,
            f"skill ids include schema-1.1 coalescing package ({len(ids)} total)",
            results,
            tee,
        )
        coalescing_skill = lib.get("hls-coalescing-512-compound-transform")
        rendered = (
            render_skill_for_prompt(coalescing_skill)
            if coalescing_skill is not None else ""
        )
        _check(
            "schema11-skill-renders-guards",
            "required steps:" in rendered
            and "guards:" in rendered
            and "interface pragmas alone" in rendered,
            "renderer exposes package checklists and coalescing guardrails",
            results,
            tee,
        )
        old_prompt_mode = os.environ.get("C2HLS_SKILL_PROMPT_MODE")
        os.environ["C2HLS_SKILL_PROMPT_MODE"] = "action_only"
        try:
            action_rendered = (
                render_skill_for_prompt(coalescing_skill)
                if coalescing_skill is not None else ""
            )
        finally:
            if old_prompt_mode is None:
                os.environ.pop("C2HLS_SKILL_PROMPT_MODE", None)
            else:
                os.environ["C2HLS_SKILL_PROMPT_MODE"] = old_prompt_mode
        _check(
            "schema11-skill-action-only-trims-guards",
            "required steps:" in action_rendered
            and "guards:" not in action_rendered
            and "interface pragmas alone" not in action_rendered,
            "action_only runtime prompt mode omits guard bullets",
            results,
            tee,
        )
        src_single = inspect.getsource(c2hls.C2HLSOrchestrator._optimization_step_attempt_single)
        _check(
            "action-only-suppresses-avoid-skills",
            "avoid_skills_suppressed" in src_single
            and "prompt_mode=render_skill_prompt_mode" in src_single,
            "orchestrator records action_only mode and renders without avoid-tier additions",
            results,
            tee,
        )
        decision = bottleneck_router.select_next_step(
            feedback={
                "bottlenecks": [
                    {
                        "kind": "memory_bandwidth",
                        "severity": "high",
                        "evidence": "no widened AXI bursts",
                    }
                ]
            },
            library=lib,
            completed_steps=[],
            available_steps=["tiling", "pipeline", "unroll", "doublebuffer", "coalescing"],
            vitis_version="2023.2",
            fpga="xcu280-fsvh2892-2L-e",
        )
        _check(
            "memory-bandwidth-routes-coalescing",
            decision.step_name == "coalescing"
            and decision.skill_id in {
                "axi-burst-coalescing-narrow-safe",
                "axi-burst-widening-512",
            }
            and not decision.fallback,
            f"decision={decision}",
            results,
            tee,
        )
        decision2 = bottleneck_router.select_next_step(
            feedback={
                "bottlenecks": [
                    {
                        "kind": "ii_target_miss",
                        "severity": "high",
                        "evidence": "II=64 on AXI gmem_addr_read",
                    }
                ]
            },
            library=lib,
            completed_steps=[],
            available_steps=["tiling", "pipeline", "unroll", "doublebuffer", "coalescing"],
            vitis_version="2023.2",
            fpga="xcu280-fsvh2892-2L-e",
        )
        _check(
            "generic-ii-routes-local-staging",
            decision2.step_name == "tiling"
            and decision2.skill_id == "local-axi-staging-for-ii",
            f"decision={decision2}",
            results,
            tee,
        )


def test_static_bottleneck_and_candidate_config(results, tee):
    tee.section("Static bottlenecks and bounded candidates")
    extras = {
        "bursts": {
            "counts": {"passed": 1, "widened": 0, "failed": 0, "summary": 1},
            "failed": [],
        }
    }
    bns = hf.derive_static_bottleneck_records(extras)
    _check(
        "no-widened-burst-produces-memory-bandwidth",
        any(b.get("kind") == "memory_bandwidth" for b in bns),
        f"bottlenecks={bns}",
        results,
        tee,
    )
    os.environ[c2hls.STEP_CANDIDATES_ENV] = json.dumps({"coalescing": 3, "default": 1})
    try:
        _check(
            "candidate-json-env-per-step",
            c2hls._step_candidate_count("coalescing") == 3
            and c2hls._step_candidate_count("tiling") == 1,
            f"coalescing={c2hls._step_candidate_count('coalescing')}",
            results,
            tee,
        )
    finally:
        os.environ.pop(c2hls.STEP_CANDIDATES_ENV, None)
    src = inspect.getsource(c2hls.C2HLSOrchestrator._optimization_step_attempt)
    _check(
        "candidate-wrapper-present",
        "candidate_count" in src and "_optimization_step_attempt_single" in src,
        "optimization attempt wraps multiple candidates",
        results,
        tee,
    )
    src_single = inspect.getsource(c2hls.C2HLSOrchestrator._optimization_step_attempt_single)
    _check(
        "selected-skill-id-injected",
        "selected_skill = self.skill_library.get(skill_id)" in src_single,
        "selected routing skill is injected directly",
        results,
        tee,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    timestamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "artifacts" / f"phase12_smoke_{timestamp}.md",
    )
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    tee = Tee()
    results: List[Dict[str, Any]] = []

    tee(f"# Phase 12 Smoke Test ({timestamp})")
    tee(f"REPO_ROOT={REPO_ROOT}")
    test_phaseb_functional_mode(results, tee)
    test_flash_mode_wiring(results, tee)
    test_skill_library_and_router(results, tee)
    test_static_bottleneck_and_candidate_config(results, tee)

    tee.section("Summary")
    failed = [r for r in results if not r["ok"]]
    tee(f"checks_total: {len(results)}")
    tee(f"checks_passed: {len(results) - len(failed)}")
    tee(f"checks_failed: {len(failed)}")
    for item in failed:
        tee(f"  - {item['check']}: {item['detail']}")

    args.out.write_text(tee.md(), encoding="utf-8")
    print()
    print(f"report written: {args.out}")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
