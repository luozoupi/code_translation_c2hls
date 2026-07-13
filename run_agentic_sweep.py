#!/usr/bin/env python3
"""Run a filtered agentic multistep sweep and emit canonical JSONL.

Defaults are conservative for a corpus sweep:
  - excludes StreamCluster
  - uses Haiku only
  - leaves final hw_emu off unless C2HLS_SWEEP_HW_EMU=1
  - applies the hpca2027_reference_blind profile; set
    C2HLS_SWEEP_PROFILE=legacy only for explicitly labelled oracle runs

Use environment filters:
  C2HLS_SWEEP_BENCHES=pathfinder,lud,nw
  C2HLS_SWEEP_EXCLUDE=StreamCluster,srad
  C2HLS_SWEEP_MODELS=haiku,sonnet
  C2HLS_SWEEP_MAX_BENCHES=4
  C2HLS_SWEEP_HW_EMU=1
  C2HLS_SWEEP_SYNTH_TIMEOUT=420
  C2HLS_SWEEP_CANDIDATES_PER_STEP=5
  C2HLS_SWEEP_ATTEMPTS_PER_CANDIDATE=5
  C2HLS_SWEEP_EXHAUSTIVE_CANDIDATE_ATTEMPTS=1
  C2HLS_SWEEP_GT_PREPOP=1          # legacy/oracle profile only
  C2HLS_SWEEP_BASELINE_ALIGN=1     # legacy/oracle profile only
  C2HLS_SWEEP_STEPS=tiling,pipeline
  C2HLS_SWEEP_STRATEGY=flash
  C2HLS_SWEEP_SKILL_MODES=off,on
  C2HLS_SWEEP_COSIM_REQUIRED=0
  C2HLS_SWEEP_TMP_ROOT=/mnt/data/luo00466/tmp
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

from c2hls_temp import configure_temp_env
from evaluation_repro import (
    apply_evaluation_profile,
    attach_run_provenance,
    build_run_fingerprint,
    fingerprint_completeness,
    fingerprint_matches,
    skill_snapshot_manifest,
)
from reference_isolation import audit_history_file

STAMP = os.getenv("C2HLS_SWEEP_STAMP") or datetime.now().strftime("%Y%m%d_%H%M%S")
BENCHMARKS_DIR = Path(os.getenv("C2HLS_SWEEP_BENCHMARKS_DIR", str(REPO / "benchmarks")))
OUT_ROOT = REPO / "results_sweeps" / f"agentic_no_streamcluster_{STAMP}"
OUT_JSONL = REPO / "artifacts" / f"agentic_no_streamcluster_{STAMP}.jsonl"
SUMMARY_JSON = REPO / "artifacts" / f"agentic_no_streamcluster_{STAMP}.summary.json"
SUMMARY_MD = REPO / "artifacts" / f"agentic_no_streamcluster_{STAMP}.md"

MODELS = {
    "haiku": "claude-haiku-4-5-20251001",
    "sonnet": "claude-sonnet-4-6",
}


def _split_csv(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _env_enabled(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _recorded_fingerprint(result: dict[str, Any]) -> dict[str, Any] | None:
    candidate = result.get("run_fingerprint")
    if not isinstance(candidate, dict):
        run = result.get("run")
        candidate = run.get("run_fingerprint") if isinstance(run, dict) else None
    return candidate if isinstance(candidate, dict) else None


def _load_resumable_result(
    path: Path,
    benchmark: str,
    expected_fingerprint: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Load a result only when its entire run fingerprint is identical.

    Historical artifacts did not carry a complete fingerprint and are
    intentionally non-resumable.  Keeping the optional parameter preserves
    call compatibility while making an omitted expected identity fail closed.
    """

    if not path.is_file():
        return None
    try:
        result = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(result, dict):
        return None
    recorded_benchmark = result.get("benchmark")
    if recorded_benchmark not in (None, "", benchmark):
        return None
    if expected_fingerprint is None:
        return None
    if not fingerprint_matches(_recorded_fingerprint(result), expected_fingerprint):
        return None
    return result


def _set_default_env() -> None:
    os.environ.setdefault("C2HLS_TMP_ROOT", os.getenv("C2HLS_SWEEP_TMP_ROOT", "/mnt/data/luo00466/tmp"))
    configure_temp_env(create=True)
    os.environ.setdefault("C2HLS_VITIS_SETTINGS", "/mnt/data/luo00466/Xilinx/Vitis/2023.2/settings64.sh")
    os.environ.setdefault("C2HLS_VITIS_VERSION", "2023.2")
    os.environ.setdefault("C2HLS_PART", "xcu280-fsvh2892-2L-e")
    os.environ.setdefault("C2HLS_CLOCK_NS", "3.33")
    os.environ.setdefault("C2HLS_FLOW_TARGET", "vitis")
    os.environ.setdefault("C2HLS_EMU_ENV_SCRIPT", str(REPO / "scripts" / "setup_emu_env.sh"))
    os.environ.setdefault("C2HLS_DEVICE_PLATFORM", "xilinx_u280_gen3x16_xdma_1_202211_1")
    os.environ.setdefault("C2HLS_CLAUDE_KEY_FILE", "/home/luo00466/claude-api-key.txt")
    sweep_strategy = os.getenv("C2HLS_SWEEP_STRATEGY", "").strip()
    if sweep_strategy:
        os.environ["C2HLS_STRATEGY"] = sweep_strategy
    else:
        os.environ.setdefault("C2HLS_STRATEGY", "dynamic")
    if os.getenv("C2HLS_STRATEGY", "").strip().lower() == "dynamic":
        os.environ.setdefault("C2HLS_DYNAMIC_ROUTING", "1")
    else:
        os.environ.setdefault("C2HLS_DYNAMIC_ROUTING", "0")
    os.environ.setdefault("C2HLS_PHASE8_BASELINE_ALIGN", os.getenv("C2HLS_SWEEP_BASELINE_ALIGN", "0"))
    os.environ.setdefault("C2HLS_PHASE5_GT_PREPOP", os.getenv("C2HLS_SWEEP_GT_PREPOP", "0"))
    os.environ.setdefault("C2HLS_PHASE7A", "1")
    os.environ.setdefault("C2HLS_PHASEB_MODE", "functional")
    os.environ.setdefault("C2HLS_CANDIDATES_PER_STEP", os.getenv("C2HLS_SWEEP_CANDIDATES_PER_STEP", "5"))
    os.environ.setdefault("C2HLS_ATTEMPTS_PER_CANDIDATE", os.getenv("C2HLS_SWEEP_ATTEMPTS_PER_CANDIDATE", "5"))
    os.environ.setdefault("C2HLS_EXHAUSTIVE_CANDIDATE_ATTEMPTS", os.getenv("C2HLS_SWEEP_EXHAUSTIVE_CANDIDATE_ATTEMPTS", "1"))
    os.environ.setdefault("C2HLS_REFERENCE_VALIDATE_MODE", os.getenv("C2HLS_SWEEP_REFERENCE_VALIDATE_MODE", "trusted_external"))
    os.environ.setdefault(
        "C2HLS_REFERENCE_CACHE_DIR",
        os.getenv(
            "C2HLS_SWEEP_REFERENCE_CACHE_DIR",
            str(REPO / "artifacts" / "reference_validation_cache"),
        ),
    )
    os.environ.setdefault(
        "C2HLS_REFERENCE_CACHE_REQUIRE_COSIM",
        os.getenv("C2HLS_SWEEP_REFERENCE_CACHE_REQUIRE_COSIM", "0"),
    )
    os.environ.setdefault("C2HLS_HW_EMU_DISABLE_DEBUG_SYMBOLS", "1")
    os.environ.setdefault("C2HLS_SYNTH_TIMEOUT", os.getenv("C2HLS_SWEEP_SYNTH_TIMEOUT", "420"))
    os.environ.setdefault("C2HLS_CSIM_TIMEOUT", os.getenv("C2HLS_SWEEP_CSIM_TIMEOUT", "180"))
    os.environ.setdefault("C2HLS_COSIM_TIMEOUT", os.getenv("C2HLS_SWEEP_COSIM_TIMEOUT", "1200"))
    os.environ.setdefault("C2HLS_COSIM_REQUIRED", os.getenv("C2HLS_SWEEP_COSIM_REQUIRED", "1"))
    if os.getenv("C2HLS_COSIM_REQUIRED", "1").strip().lower() in {"0", "false", "no", "off"}:
        os.environ.setdefault("C2HLS_REFERENCE_COSIM", "0")
    os.environ.setdefault("C2HLS_COSIM_TRACE_LEVEL", os.getenv("C2HLS_SWEEP_COSIM_TRACE_LEVEL", "none"))
    os.environ.setdefault("C2HLS_HW_EMU_TIMEOUT", "7200")
    os.environ.setdefault("C2HLS_LLM_TIMEOUT", "900")
    if os.getenv("C2HLS_SWEEP_PROFILE", "").strip().lower() != "legacy":
        # The paper budget counts candidate-generating responses. Optional
        # prose-only feedback-agent calls are outside that contract and are
        # therefore disabled for reference-isolated sweeps.
        os.environ["C2HLS_FEEDBACK_LLM"] = "0"
    os.environ["C2HLS_HW_EMU_FINAL"] = os.getenv("C2HLS_SWEEP_HW_EMU", os.getenv("C2HLS_HW_EMU_FINAL", "0"))


def _discover_benches() -> list[tuple[str, Path]]:
    available: dict[str, Path] = {}
    disabled: dict[str, str] = {}
    for meta_path in sorted(BENCHMARKS_DIR.glob("*/metadata.json")):
        try:
            meta = json.loads(meta_path.read_text())
        except json.JSONDecodeError:
            continue
        name = meta.get("benchmark") or meta_path.parent.name
        if meta.get("status") == "disabled":
            disabled[name] = meta.get("disabled_reason", "metadata status=disabled")
            continue
        available[name] = meta_path.parent

    requested = _split_csv(os.getenv("C2HLS_SWEEP_BENCHES", ""))
    if requested:
        missing = [name for name in requested if name not in available]
        if missing:
            disabled_requested = {name: disabled[name] for name in missing if name in disabled}
            if disabled_requested:
                raise ValueError(
                    "disabled benchmark(s) in C2HLS_SWEEP_BENCHES: "
                    f"{disabled_requested}"
                )
            raise ValueError(f"unknown benchmark(s) in C2HLS_SWEEP_BENCHES: {missing}")
        names = requested
    else:
        names = sorted(available)

    excluded = set(_split_csv(os.getenv("C2HLS_SWEEP_EXCLUDE", "StreamCluster")))
    names = [name for name in names if name not in excluded]
    max_benches = int(os.getenv("C2HLS_SWEEP_MAX_BENCHES", "0") or "0")
    if max_benches > 0:
        names = names[:max_benches]
    return [(name, available[name]) for name in names]


def _selected_models() -> list[tuple[str, str]]:
    raw = os.getenv("C2HLS_SWEEP_MODELS", "haiku")
    selected = []
    for item in _split_csv(raw):
        key = item.lower()
        if key in MODELS:
            selected.append((key, MODELS[key]))
        elif item in MODELS.values():
            label = next(label for label, model in MODELS.items() if model == item)
            selected.append((label, item))
        else:
            label = (
                item.rsplit("/", 1)[-1]
                .replace(".", "_")
                .replace("-", "_")
                .replace(":", "_")
            )
            selected.append((label, item))
    return selected or [("haiku", MODELS["haiku"])]


def _selected_skill_modes() -> list[tuple[str, bool | None]]:
    raw = os.getenv("C2HLS_SWEEP_SKILL_MODES", "").strip()
    if not raw:
        raw = os.getenv("C2HLS_SWEEP_SKILLS", "").strip()
    if not raw:
        return [("default", None)]
    items = ["off", "on"] if raw.lower() == "both" else _split_csv(raw)
    out: list[tuple[str, bool | None]] = []
    for item in items:
        low = item.lower()
        if low in {"on", "skill_on", "skills_on", "1", "true", "yes"}:
            out.append(("skill_on", True))
        elif low in {"off", "skill_off", "skills_off", "0", "false", "no"}:
            out.append(("skill_off", False))
        elif low in {"default", "auto"}:
            out.append(("default", None))
        else:
            raise ValueError(f"unknown C2HLS_SWEEP_SKILL_MODES entry: {item!r}")
    return out


def _selected_steps() -> list[str] | None:
    raw = os.getenv("C2HLS_SWEEP_STEPS", "").strip()
    return _split_csv(raw) if raw else None


def _cycles(report: dict[str, Any] | None) -> int | None:
    if not isinstance(report, dict):
        return None
    value = report.get("latency_cycles") or report.get("latency_cycle")
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return None


def _best_step(data: dict[str, Any]) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    if data.get("baseline_report"):
        candidates.append({"step": "baseline", "report": data["baseline_report"]})
    for step in data.get("steps") or []:
        if step.get("success") and step.get("report"):
            candidates.append({"step": step.get("step_name"), "report": step.get("report")})
    if not candidates:
        return {}
    best = min(candidates, key=lambda item: _cycles(item.get("report")) or 10**30)
    return {"step": best["step"], "cycles": _cycles(best.get("report"))}


def _compact_llm_usage(data: dict[str, Any]) -> dict[str, Any]:
    usage = data.get("llm_usage") or {}
    fields = [
        "calls",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "cache_creation_input_tokens",
        "cache_read_input_tokens",
        "cached_tokens",
        "reasoning_tokens",
        "usage_missing_calls",
    ]
    return {field: int(usage.get(field) or 0) for field in fields}


def _candidate_telemetry_contract(result: dict[str, Any]) -> dict[str, Any]:
    """Audit, without inventing counters, the producer fields used at freeze."""
    usage = result.get("llm_usage") if isinstance(result.get("llm_usage"), dict) else {}
    synthesis = (
        result.get("synthesis_evaluations")
        if isinstance(result.get("synthesis_evaluations"), dict)
        else {}
    )
    llm_events = usage.get("events") if isinstance(usage.get("events"), list) else []
    candidate_events = (
        synthesis.get("events") if isinstance(synthesis.get("events"), list) else []
    )
    required_fields = {
        "candidate_evaluation_index",
        "cumulative_tokens",
        "cumulative_llm_calls",
        "cumulative_synthesis_evaluations",
        "cumulative_elapsed_seconds",
        "correctness_status",
        "synthesis_status",
        "resource_fit",
        "timing_met",
        "synthesized_latency_cycles",
        "latency_source",
        "failure_class",
        "selected_for_executed_cosim",
        "code_sha256",
        "report_sha256",
    }
    joins_complete = (
        len(llm_events) == len(candidate_events)
        and all(
            isinstance(event, dict)
            and event.get("candidate_evaluation_index") == index
            for index, event in enumerate(llm_events)
        )
        and all(
            isinstance(event, dict)
            and event.get("candidate_evaluation_index") == index
            and required_fields.issubset(event)
            for index, event in enumerate(candidate_events)
        )
    )
    selection_count = result.get("selected_winner_cosim_count")
    implementation_count = result.get("post_route_implementation_count")
    synthesis_count = synthesis.get("count")
    total_synthesis_calls = result.get("total_synthesis_calls")
    total_tool_calls = result.get("total_tool_calls")
    expected_total = (
        synthesis_count + selection_count + implementation_count
        if all(
            isinstance(value, int) and not isinstance(value, bool)
            for value in (synthesis_count, selection_count, implementation_count)
        )
        else None
    )
    synthesis_attribution_complete = (
        isinstance(selection_count, int)
        and not isinstance(selection_count, bool)
        and selection_count in {0, 1}
        and isinstance(implementation_count, int)
        and not isinstance(implementation_count, bool)
        and implementation_count in {0, 1}
        and isinstance(synthesis_count, int)
        and not isinstance(synthesis_count, bool)
        and synthesis_count >= 0
        and isinstance(total_synthesis_calls, int)
        and not isinstance(total_synthesis_calls, bool)
        and isinstance(total_tool_calls, int)
        and not isinstance(total_tool_calls, bool)
        and total_synthesis_calls == expected_total
        and total_tool_calls == expected_total
    )
    complete = bool(
        synthesis.get("complete_candidate_event_stream") is True
        and joins_complete
        and usage.get("candidate_requests") == len(candidate_events)
        and usage.get("calls") == len(llm_events)
        and synthesis_attribution_complete
        and (
            result.get("selected_winner_cosim_count") == 0
            or (
                result.get("selected_code_sha256")
                == result.get("cosim_target_code_sha256")
                and isinstance(result.get("selected_code_sha256"), str)
            )
        )
    )
    return {
        "schema_version": "c2hls.agentic-candidate-telemetry.v1",
        "complete": complete,
        "candidate_event_count": len(candidate_events),
        "llm_event_count": len(llm_events),
        "joins_complete": joins_complete,
        "synthesis_attribution_complete": synthesis_attribution_complete,
    }


def _result_path(bench: str, label: str) -> Path:
    return OUT_ROOT / f"{bench}_{label}" / f"{bench}_multistep_results.json"


def _test_status(summary: Any) -> str | None:
    if not isinstance(summary, dict):
        return None
    status = str(summary.get("status") or "").strip().lower()
    if status == "passed":
        return "pass"
    if status == "failed":
        return "timeout" if "timed out" in str(summary.get("error") or "").lower() else "fail"
    return status or None


def _summarize(data: dict[str, Any]) -> dict[str, Any]:
    steps = data.get("steps") or []
    hw = data.get("hw_emu") or {}
    return {
        "success": bool(data.get("success")),
        "phase": data.get("phase"),
        "error": data.get("error"),
        "phase_b_mode": data.get("phase_b_mode"),
        "llm_usage": _compact_llm_usage(data),
        "skill_library_provenance": (
            data.get("skill_library_provenance")
            or (data.get("run") or {}).get("skill_library_provenance")
        ),
        "baseline_cycles": _cycles(data.get("baseline_report")),
        "baseline_csim": (
            (data.get("baseline_csim") or data.get("csim") or {}).get("passed")
            if isinstance(data.get("baseline_csim") or data.get("csim"), dict)
            else None
        ),
        "baseline_cosim_status": _test_status(
            data.get("baseline_cosim") or data.get("cosim")
        ),
        "best": _best_step(data),
        "steps_attempted": len(steps),
        "steps_success": sum(1 for step in steps if step.get("success")),
        "tool_calls": {
            "selection_synthesis_evaluations": data.get(
                "synthesis_evaluation_count"
            ),
            "selected_winner_cosim": data.get("selected_winner_cosim_count"),
            "post_route_implementation": data.get(
                "post_route_implementation_count"
            ),
            "total_synthesis_calls": data.get("total_synthesis_calls"),
            "total_tool_calls": data.get("total_tool_calls"),
        },
        "step_cycles": [
            {
                "step": step.get("step_name"),
                "success": bool(step.get("success")),
                "cycles": _cycles(step.get("report")),
                "csim": (step.get("csim") or {}).get("passed") if isinstance(step.get("csim"), dict) else None,
                "cosim": (step.get("cosim") or {}).get("passed") if isinstance(step.get("cosim"), dict) else None,
                "cosim_status": _test_status(step.get("cosim")),
                "cosim_cycles": (step.get("cosim") or {}).get("kernel_runtime_cycles") if isinstance(step.get("cosim"), dict) else None,
                "cosim_policy": (step.get("cosim") or {}).get("cosim_policy") if isinstance(step.get("cosim"), dict) else None,
                "candidate_attempts": len(step.get("candidate_attempts") or []),
                "candidate_search": step.get("candidate_search"),
                "attempt_stats": step.get("attempt_stats"),
                "skill_id": ((step.get("routing_decision") or {}).get("skill_id")),
                "skill_prompt": step.get("skill_prompt"),
            }
            for step in steps
        ],
        "hw_emu": {
            "ran": hw.get("ran"),
            "success": hw.get("success"),
            "passed": hw.get("passed"),
            "cycles": hw.get("kernel_runtime_cycles"),
            "variant": hw.get("variant_name"),
            "implementation_call_count": hw.get("implementation_call_count"),
            "error": hw.get("error") or hw.get("skip_reason"),
        },
    }


def _export_jsonl(completed: list[tuple[str, str, Path]]) -> int:
    import export_schema_jsonl as ex

    OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with OUT_JSONL.open("w") as handle:
        for bench, label, bench_dir in completed:
            path = _result_path(bench, label)
            if not path.exists():
                continue
            for record in ex._records_from_multistep(
                bench_dir,
                path,
                default_part=os.getenv("C2HLS_PART", "xcu280-fsvh2892-2L-e"),
                default_clock_ns=float(os.getenv("C2HLS_CLOCK_NS", "3.33")),
            ):
                handle.write(ex._strict_json_dumps(record) + "\n")
                count += 1
    validation = ex.validate_jsonl(OUT_JSONL)
    if validation.get("invalid"):
        raise RuntimeError(f"invalid JSONL records={validation['invalid']} path={OUT_JSONL}")
    return count


def _write_reports(
    rows: list[dict[str, Any]],
    jsonl_count: int,
    *,
    profile: dict[str, Any] | None = None,
) -> None:
    SUMMARY_JSON.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "stamp": STAMP,
        "benchmarks_dir": str(BENCHMARKS_DIR),
        "out_root": str(OUT_ROOT),
        "jsonl": str(OUT_JSONL),
        "jsonl_records": jsonl_count,
        "evaluation_profile": profile or {},
        "env": {
            key: os.getenv(key)
            for key in [
                "C2HLS_SWEEP_BENCHES",
                "C2HLS_SWEEP_PROFILE",
                "C2HLS_SWEEP_EXCLUDE",
                "C2HLS_SWEEP_MODELS",
                "C2HLS_SWEEP_MAX_BENCHES",
                "C2HLS_SWEEP_HW_EMU",
                "C2HLS_SWEEP_SYNTH_TIMEOUT",
                "C2HLS_SWEEP_CSIM_TIMEOUT",
                "C2HLS_SWEEP_COSIM_TIMEOUT",
                "C2HLS_SWEEP_COSIM_REQUIRED",
                "C2HLS_SWEEP_COSIM_TRACE_LEVEL",
                "C2HLS_SWEEP_CANDIDATES_PER_STEP",
                "C2HLS_SWEEP_ATTEMPTS_PER_CANDIDATE",
                "C2HLS_SWEEP_EXHAUSTIVE_CANDIDATE_ATTEMPTS",
                "C2HLS_SWEEP_GT_PREPOP",
                "C2HLS_SWEEP_BASELINE_ALIGN",
                "C2HLS_SWEEP_REFERENCE_VALIDATE_MODE",
                "C2HLS_SWEEP_STEPS",
                "C2HLS_SWEEP_STRATEGY",
                "C2HLS_SWEEP_SKILL_MODES",
                "C2HLS_REFERENCE_VALIDATE_MODE",
                "C2HLS_STRATEGY",
                "C2HLS_DYNAMIC_ROUTING",
                "C2HLS_FORCE_SKILL_PROMPTS",
                "C2HLS_SKILL_MODE",
                "C2HLS_SKILL_PROMPT_MODE",
                "C2HLS_SKILL_PROMPT_SCOPE",
                "C2HLS_SKILL_LIBRARY_PERSIST",
                "C2HLS_SKILL_LIBRARY_FROZEN",
                "C2HLS_SKILL_UPDATE_STATS",
                "C2HLS_SYNTH_TIMEOUT",
                "C2HLS_CSIM_TIMEOUT",
                "C2HLS_COSIM_TIMEOUT",
                "C2HLS_COSIM_REQUIRED",
                "C2HLS_COSIM_TRACE_LEVEL",
                "C2HLS_COSIM_SKIP_SLOWER_THAN_GOLD",
                "C2HLS_COSIM_SKIP_GOLD_RATIO",
                "C2HLS_REFERENCE_BLIND",
                "C2HLS_ORACLE_MODE",
                "C2HLS_GT_AWARE_REVERT",
                "C2HLS_GT_COMPARISON_IN_CONTROL",
                "C2HLS_REFERENCE_METRICS_IN_PROMPTS",
                "C2HLS_REFERENCE_CODE_IN_PROMPTS",
                "C2HLS_TRANSCRIPT_AUDIT",
                "C2HLS_LLM_TEMPERATURE",
                "C2HLS_LLM_TOP_P",
                "C2HLS_LLM_SEED",
                "C2HLS_MODEL_REVISION",
                "C2HLS_SYNTHESIS_EVAL_BUDGET",
                "C2HLS_REFERENCE_CACHE_DIR",
                "C2HLS_REFERENCE_CACHE_REQUIRE_COSIM",
                "C2HLS_SWEEP_RESUME",
                "C2HLS_HW_EMU_FINAL",
                "C2HLS_HW_EMU_DISABLE_DEBUG_SYMBOLS",
                "C2HLS_PHASE5_GT_PREPOP",
                "C2HLS_PHASE8_BASELINE_ALIGN",
                "C2HLS_CANDIDATES_PER_STEP",
                "C2HLS_ATTEMPTS_PER_CANDIDATE",
                "C2HLS_EXHAUSTIVE_CANDIDATE_ATTEMPTS",
                "C2HLS_HW_EMU_TIMEOUT",
                "C2HLS_PHASEB_MODE",
            ]
        },
        "rows": rows,
    }
    SUMMARY_JSON.write_text(json.dumps(payload, indent=2) + "\n")

    lines = [
        "# Agentic Sweep",
        "",
        f"stamp: `{STAMP}`",
        f"benchmarks_dir: `{BENCHMARKS_DIR}`",
        f"results root: `{OUT_ROOT}`",
        f"jsonl: `{OUT_JSONL}`",
        f"jsonl records: `{jsonl_count}`",
        "",
        "| bench | model | skill | status | steps | best step | best cycles | baseline cycles | LLM calls | input tok | output tok | total tok | cosim | cosim cycles | hw_emu | hw cycles | note |",
        "|---|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---|---:|---|---:|---|",
    ]
    for row in rows:
        cur = row.get("current") or {}
        best = cur.get("best") or {}
        hw = cur.get("hw_emu") or {}
        usage = cur.get("llm_usage") or {}
        final_step = (cur.get("step_cycles") or [{}])[-1] if cur.get("step_cycles") else {}
        cosim_status = final_step.get("cosim_status") or cur.get("baseline_cosim_status") or "skip"
        lines.append(
            f"| {row.get('bench')} | {row.get('model')} | {row.get('skill_mode', 'default')} | {'pass' if cur.get('success') else 'fail'} | "
            f"{cur.get('steps_success')}/{cur.get('steps_attempted')} | {best.get('step') or '-'} | "
            f"{best.get('cycles') if best.get('cycles') is not None else '-'} | "
            f"{cur.get('baseline_cycles') if cur.get('baseline_cycles') is not None else '-'} | "
            f"{usage.get('calls', 0)} | "
            f"{usage.get('input_tokens', 0)} | "
            f"{usage.get('output_tokens', 0)} | "
            f"{usage.get('total_tokens', 0)} | "
            f"{cosim_status} | "
            f"{final_step.get('cosim_cycles') if final_step.get('cosim_cycles') is not None else '-'} | "
            f"{'pass' if hw.get('success') else ('fail' if hw.get('ran') else 'skip')} | "
            f"{hw.get('cycles') if hw.get('cycles') is not None else '-'} | "
            f"{hw.get('error') or '-'} |"
        )
    SUMMARY_MD.write_text("\n".join(lines) + "\n")


def main() -> int:
    _set_default_env()
    profile = apply_evaluation_profile()
    from c2hls import run_benchmark_multistep

    benches = _discover_benches()
    models = _selected_models()
    skill_modes = _selected_skill_modes()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    resume = _env_enabled("C2HLS_SWEEP_RESUME")
    rows: list[dict[str, Any]] = []
    completed: list[tuple[str, str, Path]] = []
    print(
        f"SELECTED benches={','.join(name for name, _ in benches)} "
        f"models={','.join(label for label, _ in models)} "
        f"skill_modes={','.join(label for label, _ in skill_modes)} "
        f"hw_emu={os.getenv('C2HLS_HW_EMU_FINAL')}",
        flush=True,
    )

    for label, model_id in models:
        for skill_label, skill_enabled in skill_modes:
            if skill_enabled is not None:
                os.environ["C2HLS_SKILL_MODE"] = skill_label
                os.environ["C2HLS_FORCE_SKILL_PROMPTS"] = "1" if skill_enabled else "0"
            else:
                os.environ.setdefault("C2HLS_SKILL_MODE", "default")
            run_label = label if skill_label == "default" else f"{label}_{skill_label}"
            for bench, bench_dir in benches:
                out_dir = OUT_ROOT / f"{bench}_{run_label}"
                result_json = _result_path(bench, run_label)
                selected_steps = _selected_steps()
                run_fingerprint = build_run_fingerprint(
                    repo=REPO,
                    benchmark_dir=bench_dir,
                    benchmark=bench,
                    model_id=model_id,
                    model_label=label,
                    skill_mode=skill_label,
                    steps=selected_steps,
                    profile=profile,
                )
                preflight = fingerprint_completeness(run_fingerprint)
                if profile.get("reference_blind") and not preflight.get("complete"):
                    raise RuntimeError(
                        "paper evaluation fingerprint is incomplete; refusing "
                        "to start: " + ", ".join(preflight.get("issues") or [])
                    )
                t0 = time.time()
                result = (
                    _load_resumable_result(result_json, bench, run_fingerprint)
                    if resume
                    else None
                )
                resumed = result is not None
                if resumed:
                    print(
                        f"RESUME bench={bench} model={label} skill={skill_label} "
                        f"result={result_json}",
                        flush=True,
                    )
                else:
                    if resume and result_json.is_file():
                        print(
                            f"RESUME_REJECT bench={bench} model={label} "
                            f"skill={skill_label} reason=fingerprint_mismatch_or_missing",
                            flush=True,
                        )
                    print(f"START bench={bench} model={label} skill={skill_label} out={out_dir}", flush=True)
                    try:
                        result = run_benchmark_multistep(
                            str(bench_dir),
                            output_dir=str(out_dir),
                            gpt_model=model_id,
                            turns_limitation=int(os.getenv("C2HLS_TURNS", "4")),
                            steps=selected_steps,
                        )
                    except Exception as exc:
                        out_dir.mkdir(parents=True, exist_ok=True)
                        result = {
                            "benchmark": bench,
                            "success": False,
                            "phase": "exception",
                            "error": str(exc),
                            "steps": [],
                            "hw_emu": {
                                "ran": False,
                                "skip_reason": f"agentic exception: {exc}",
                                "profile_required": True,
                            },
                            "run": {
                                "model": model_id,
                                "skill_mode": skill_label,
                                "skill_prompts": bool(skill_enabled),
                            },
                        }
                        result_json.write_text(json.dumps(result, indent=2) + "\n")
                        print(f"ERROR bench={bench} model={label} skill={skill_label}: {exc}", flush=True)

                    result["candidate_telemetry_contract"] = (
                        _candidate_telemetry_contract(result)
                    )
                    history_path = out_dir / f"{bench}_history.json"
                    if _env_enabled("C2HLS_TRANSCRIPT_AUDIT"):
                        reference_audit = audit_history_file(
                            history_path,
                            benchmark_dir=bench_dir,
                            reference_data=(result or {}).get("reference_validation"),
                        )
                    else:
                        reference_audit = {
                            "schema_version": "c2hls.reference-isolation-audit.v1",
                            "passed": False,
                            "finding_count": 0,
                            "findings": [],
                            "error": "transcript audit disabled",
                        }
                    total_elapsed_seconds = time.time() - t0
                    # Method-cost comparisons exclude common CPU-golden and
                    # expert-frontier preflight.  Dynamic C2HLS records its
                    # search interval inside run_benchmark_multistep, matching
                    # the baseline engine's timer around engine.run().
                    search_elapsed_seconds = float(
                        (result or {}).get(
                            "search_elapsed_seconds", total_elapsed_seconds
                        )
                    )
                    attach_run_provenance(
                        result,
                        fingerprint=run_fingerprint,
                        profile=profile,
                        elapsed_seconds=search_elapsed_seconds,
                        history_path=history_path,
                        reference_audit=reference_audit,
                    )
                    result.setdefault("run", {}).update({
                        "search_elapsed_seconds": search_elapsed_seconds,
                        "preflight_elapsed_seconds": float(
                            (result or {}).get("preflight_elapsed_seconds", 0.0)
                        ),
                        "post_route_elapsed_seconds": float(
                            (result or {}).get("post_route_elapsed_seconds", 0.0)
                        ),
                        "total_elapsed_seconds": total_elapsed_seconds,
                        "paper_method_wall_time_field": "search_elapsed_seconds",
                    })
                    pre_skill_sha256 = (
                        run_fingerprint.get("payload", {})
                        .get("skills", {})
                        .get("sha256")
                    )
                    post_skill_sha256 = skill_snapshot_manifest(REPO).get("sha256")
                    skill_snapshot_integrity = {
                        "pre_run_sha256": pre_skill_sha256,
                        "post_run_sha256": post_skill_sha256,
                        "unchanged": pre_skill_sha256 == post_skill_sha256,
                    }
                    result["skill_snapshot_integrity"] = skill_snapshot_integrity
                    if not skill_snapshot_integrity["unchanged"]:
                        result["controller_success_before_skill_integrity"] = bool(
                            result.get("success")
                        )
                        result["success"] = False
                        result["phase"] = "skill_integrity"
                        result["error"] = (
                            "skill snapshot changed during evaluation; run rejected"
                        )
                    reference_audit_path = (
                        out_dir / f"{bench}_reference_isolation_audit.json"
                    )
                    out_dir.mkdir(parents=True, exist_ok=True)
                    reference_audit_path.write_text(
                        json.dumps(reference_audit, indent=2) + "\n"
                    )
                    result.setdefault("run", {})[
                        "reference_isolation_audit_path"
                    ] = reference_audit_path.name
                    if (
                        profile.get("reference_blind")
                        and not reference_audit.get("passed")
                        and _env_enabled("C2HLS_REFERENCE_BLIND_FAIL_ON_LEAK", "1")
                    ):
                        result["controller_success_before_isolation_audit"] = bool(
                            result.get("success")
                        )
                        result["success"] = False
                        result["phase"] = "reference_isolation"
                        result["error"] = (
                            "reference-isolation audit failed; see hashed findings"
                        )
                    result_json.write_text(json.dumps(result, indent=2) + "\n")

                if not result_json.exists():
                    out_dir.mkdir(parents=True, exist_ok=True)
                    result_json.write_text(json.dumps(result, indent=2) + "\n")

                current = _summarize(result)
                current["elapsed_sec"] = 0.0 if resumed else round(time.time() - t0, 3)
                current["resumed"] = resumed
                current["json"] = str(result_json)
                current["evaluation_status"] = result.get("evaluation_status")
                current["reference_isolation_audit"] = result.get(
                    "reference_isolation_audit"
                )
                current["run_fingerprint_sha256"] = run_fingerprint.get("sha256")
                rows.append({
                    "bench": bench,
                    "bench_dir": str(bench_dir),
                    "model": label,
                    "model_id": model_id,
                    "skill_mode": skill_label,
                    "current": current,
                })
                completed.append((bench, run_label, bench_dir))
                jsonl_count = _export_jsonl(completed)
                _write_reports(rows, jsonl_count, profile=profile)
                best = current.get("best") or {}
                print(
                    f"DONE bench={bench} model={label} skill={skill_label} success={current.get('success')} "
                    f"steps={current.get('steps_success')}/{current.get('steps_attempted')} "
                    f"best={best.get('step')} cycles={best.get('cycles')} "
                    f"elapsed={current['elapsed_sec']}s",
                    flush=True,
                )

    print(f"SUMMARY {SUMMARY_MD}", flush=True)
    print(f"JSONL {OUT_JSONL}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
