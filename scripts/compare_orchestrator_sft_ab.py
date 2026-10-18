#!/usr/bin/env python3
"""Compare held-out direct and agentic orchestrator-SFT experiments."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _as_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _ratio(numerator: int | None, denominator: int | None) -> float | None:
    if numerator is None or denominator in (None, 0):
        return None
    return numerator / denominator


def _index_rows(summary: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(row.get("benchmark") or row.get("bench")): row
        for row in summary.get("rows") or []
        if row.get("benchmark") or row.get("bench")
    }


def _agentic_row(row: dict[str, Any]) -> dict[str, Any]:
    current = row.get("current") or {}
    steps = current.get("step_cycles") or []
    best = current.get("best") or {}
    best_step = best.get("step")
    best_result = next(
        (step for step in steps if step.get("step") == best_step),
        None,
    )
    skill_prompts = [
        step.get("skill_prompt")
        for step in steps
        if isinstance(step.get("skill_prompt"), dict)
    ]
    all_positive_verified = bool(skill_prompts) and all(
        prompt.get("enabled")
        and prompt.get("prompt_mode") == "action_only"
        and prompt.get("prompt_scope") == "all_positive"
        and prompt.get("avoid_skills_suppressed") is True
        and not (prompt.get("avoid_skill_ids") or [])
        and len(prompt.get("injected_skill_ids") or [])
        == int(prompt.get("matched_skill_count") or 0)
        and int(prompt.get("matched_skill_count") or 0) > 0
        for prompt in skill_prompts
    )
    injected_counts = [
        len(prompt.get("injected_skill_ids") or []) for prompt in skill_prompts
    ]

    routing = {}
    result_path_value = str(current.get("json") or "")
    result_path = Path(result_path_value) if result_path_value else None
    if result_path is not None and result_path.is_file():
        routing = (_load(result_path).get("run") or {})

    return {
        "success": bool(current.get("success")),
        "phase": current.get("phase"),
        "baseline_cycles": _as_int(current.get("baseline_cycles")),
        "baseline_csim": current.get("baseline_csim"),
        "baseline_cosim_status": current.get("baseline_cosim_status"),
        "best_step": best_step,
        "best_cycles": _as_int(best.get("cycles")),
        "best_csim": (best_result or {}).get("csim"),
        "best_cosim_status": (best_result or {}).get("cosim_status"),
        "best_cosim_cycles": _as_int((best_result or {}).get("cosim_cycles")),
        "steps_attempted": current.get("steps_attempted"),
        "steps_success": current.get("steps_success"),
        "skill_injected_counts": injected_counts,
        "all_positive_injection_verified": all_positive_verified,
        "routing": {
            key: routing.get(key)
            for key in (
                "model",
                "model_translator",
                "model_synthesis",
                "model_quality_repair",
                "model_feedback",
            )
        },
        "result_json": str(result_path) if result_path is not None else None,
    }


def _pair_counts(rows: list[dict[str, Any]], ratio_key: str) -> dict[str, Any]:
    ratios = [row[ratio_key] for row in rows if row.get(ratio_key) is not None]
    return {
        "benchmarks": len(rows),
        "comparable": len(ratios),
        "treatment_faster": sum(value < 1.0 for value in ratios),
        "control_faster": sum(value > 1.0 for value in ratios),
        "ties": sum(value == 1.0 for value in ratios),
        "geomean_treatment_over_control": (
            math.exp(sum(math.log(value) for value in ratios) / len(ratios))
            if ratios and all(value > 0 for value in ratios)
            else None
        ),
    }


def _server_context_audit(manifest: dict[str, Any] | None) -> dict[str, Any] | None:
    if not manifest:
        return None
    audit = manifest.get("context_audit") or {}
    requests = int(manifest.get("requests_completed") or 0)
    truncated = int(audit.get("requests_with_truncation") or 0)
    dropped = int(audit.get("prompt_tokens_dropped_total") or 0)
    return {
        "requests_completed": requests,
        "max_input_tokens": manifest.get("max_input_tokens"),
        "requests_with_truncation": truncated,
        "prompt_tokens_dropped_total": dropped,
        "max_prompt_tokens_seen": audit.get("max_prompt_tokens_seen"),
        "max_prompt_tokens_dropped": audit.get("max_prompt_tokens_dropped"),
        "truncation_free": requests > 0 and truncated == 0 and dropped == 0,
    }


def _aggregate_server_context_audit(
    manifests: list[dict[str, Any]],
) -> dict[str, Any] | None:
    audits = [
        audit
        for manifest in manifests
        if (audit := _server_context_audit(manifest)) is not None
    ]
    if not audits:
        return None
    requests = sum(int(audit["requests_completed"] or 0) for audit in audits)
    truncated = sum(int(audit["requests_with_truncation"] or 0) for audit in audits)
    dropped = sum(int(audit["prompt_tokens_dropped_total"] or 0) for audit in audits)
    max_input_tokens = [
        int(audit["max_input_tokens"])
        for audit in audits
        if audit.get("max_input_tokens") is not None
    ]
    return {
        "manifests": len(audits),
        "requests_completed": requests,
        "max_input_tokens": min(max_input_tokens) if max_input_tokens else None,
        "requests_with_truncation": truncated,
        "prompt_tokens_dropped_total": dropped,
        "max_prompt_tokens_seen": max(
            int(audit["max_prompt_tokens_seen"] or 0) for audit in audits
        ),
        "max_prompt_tokens_dropped": max(
            int(audit["max_prompt_tokens_dropped"] or 0) for audit in audits
        ),
        "truncation_free": requests > 0 and truncated == 0 and dropped == 0,
    }


def compare_direct(
    control_vitis: dict[str, Any],
    treatment_vitis: dict[str, Any],
    control_cosim: dict[str, Any],
    treatment_cosim: dict[str, Any],
) -> dict[str, Any]:
    control_vitis_rows = _index_rows(control_vitis)
    treatment_vitis_rows = _index_rows(treatment_vitis)
    control_cosim_rows = _index_rows(control_cosim)
    treatment_cosim_rows = _index_rows(treatment_cosim)
    benchmarks = sorted(set(control_vitis_rows) | set(treatment_vitis_rows))
    rows = []
    for benchmark in benchmarks:
        control = control_vitis_rows.get(benchmark, {})
        treatment = treatment_vitis_rows.get(benchmark, {})
        control_synth = control.get("synth") or {}
        treatment_synth = treatment.get("synth") or {}
        control_rtl = control_cosim_rows.get(benchmark, {})
        treatment_rtl = treatment_cosim_rows.get(benchmark, {})
        control_cycles = _as_int(control_synth.get("latency_cycles"))
        treatment_cycles = _as_int(treatment_synth.get("latency_cycles"))
        control_cosim_cycles = _as_int(control_rtl.get("vllm_cosim_cycles"))
        treatment_cosim_cycles = _as_int(treatment_rtl.get("vllm_cosim_cycles"))
        rows.append(
            {
                "benchmark": benchmark,
                "control_synth_status": control_synth.get("status", "not_run"),
                "treatment_synth_status": treatment_synth.get("status", "not_run"),
                "control_csim_status": (control.get("csim") or {}).get("status", "not_run"),
                "treatment_csim_status": (treatment.get("csim") or {}).get("status", "not_run"),
                "control_csynth_cycles": control_cycles,
                "treatment_csynth_cycles": treatment_cycles,
                "treatment_over_control_csynth": _ratio(treatment_cycles, control_cycles),
                "control_cosim_status": control_rtl.get("vllm_cosim_status", "not_run"),
                "treatment_cosim_status": treatment_rtl.get("vllm_cosim_status", "not_run"),
                "control_cosim_cycles": control_cosim_cycles,
                "treatment_cosim_cycles": treatment_cosim_cycles,
                "treatment_over_control_cosim": _ratio(
                    treatment_cosim_cycles,
                    control_cosim_cycles,
                ),
            }
        )
    return {
        "rows": rows,
        "csynth_counts": _pair_counts(rows, "treatment_over_control_csynth"),
        "cosim_counts": _pair_counts(rows, "treatment_over_control_cosim"),
    }


def compare_agentic(
    label: str,
    control_summary: dict[str, Any],
    treatment_summary: dict[str, Any],
) -> dict[str, Any]:
    controls = _index_rows(control_summary)
    treatments = _index_rows(treatment_summary)
    rows = []
    for benchmark in sorted(treatments):
        control = _agentic_row(controls.get(benchmark, {}))
        treatment = _agentic_row(treatments.get(benchmark, {}))
        rows.append(
            {
                "benchmark": benchmark,
                "control": control,
                "treatment": treatment,
                "treatment_over_control_best_cycles": _ratio(
                    treatment["best_cycles"],
                    control["best_cycles"],
                ),
            }
        )
    return {
        "label": label,
        "rows": rows,
        "counts": {
            **_pair_counts(rows, "treatment_over_control_best_cycles"),
            "control_success": sum(row["control"]["success"] for row in rows),
            "treatment_success": sum(row["treatment"]["success"] for row in rows),
            "all_positive_verified": sum(
                row["treatment"]["all_positive_injection_verified"] for row in rows
            ),
        },
    }


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def _write_markdown(summary: dict[str, Any], path: Path) -> None:
    server_audit = summary.get("server_context_audit") or {}
    direct_audit = server_audit.get("direct") or {}
    agentic_audit = server_audit.get("agentic") or {}
    lines = [
        "# Qwen Orchestrator SFT A/B",
        "",
        f"- Evaluation splits: `{','.join((summary.get('evaluation_scope') or {}).get('splits') or [])}`",
        f"- Evaluation benchmarks: `{len((summary.get('evaluation_scope') or {}).get('benchmarks') or [])}`",
        "",
        "## Delivery audit",
        "",
        f"- Direct server truncation-free: `{direct_audit.get('truncation_free')}`",
        f"- Agentic server truncation-free: `{agentic_audit.get('truncation_free')}`",
        f"- All-positive skills delivered without truncation: `{(summary.get('delivery_verification') or {}).get('all_positive_skills_delivered')}`",
        "",
        "## Direct held-out optimization prompts",
        "",
        "| benchmark | base synth/csim/cosim | SFT synth/csim/cosim | base cycles | SFT cycles | SFT/base |",
        "|---|---|---|---:|---:|---:|",
    ]
    for row in summary["direct"]["rows"]:
        lines.append(
            "| {benchmark} | {cs}/{cc}/{cr} | {ts}/{tc}/{tr} | {cb} | {ct} | {ratio} |".format(
                benchmark=row["benchmark"],
                cs=row["control_synth_status"],
                cc=row["control_csim_status"],
                cr=row["control_cosim_status"],
                ts=row["treatment_synth_status"],
                tc=row["treatment_csim_status"],
                tr=row["treatment_cosim_status"],
                cb=_fmt(row["control_csynth_cycles"]),
                ct=_fmt(row["treatment_csynth_cycles"]),
                ratio=_fmt(row["treatment_over_control_csynth"]),
            )
        )

    for pair in summary["agentic"]:
        lines.extend(
            [
                "",
                f"## Agentic {pair['label']}",
                "",
                "| benchmark | base success/best | SFT success/best | SFT/base cycles | all-positive verified |",
                "|---|---|---|---:|---:|",
            ]
        )
        for row in pair["rows"]:
            control = row["control"]
            treatment = row["treatment"]
            lines.append(
                "| {benchmark} | {cs}/{cb} | {ts}/{tb} | {ratio} | {skills} |".format(
                    benchmark=row["benchmark"],
                    cs=control["success"],
                    cb=_fmt(control["best_cycles"]),
                    ts=treatment["success"],
                    tb=_fmt(treatment["best_cycles"]),
                    ratio=_fmt(row["treatment_over_control_best_cycles"]),
                    skills=treatment["all_positive_injection_verified"],
                )
            )
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direct-control-vitis", type=Path, required=True)
    parser.add_argument("--direct-treatment-vitis", type=Path, required=True)
    parser.add_argument("--direct-control-cosim", type=Path, required=True)
    parser.add_argument("--direct-treatment-cosim", type=Path, required=True)
    parser.add_argument("--direct-server-manifest", type=Path)
    parser.add_argument("--agentic-server-manifest", type=Path, action="append")
    parser.add_argument("--dataset-manifest", type=Path)
    parser.add_argument("--evaluation-split", action="append", default=[])
    parser.add_argument(
        "--agentic-pair",
        nargs=3,
        action="append",
        metavar=("LABEL", "CONTROL_SUMMARY", "TREATMENT_SUMMARY"),
        default=[],
    )
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()

    agentic = [
        compare_agentic(label, _load(Path(control)), _load(Path(treatment)))
        for label, control, treatment in args.agentic_pair
    ]
    server_context_audit = {
        "direct": _server_context_audit(
            _load(args.direct_server_manifest) if args.direct_server_manifest else None
        ),
        "agentic": _aggregate_server_context_audit(
            [_load(path) for path in (args.agentic_server_manifest or [])]
        ),
    }
    all_positive_pairs = [
        pair for pair in agentic if "all_positive" in pair["label"]
    ]
    framework_all_positive_verified = bool(all_positive_pairs) and all(
        row["treatment"]["all_positive_injection_verified"]
        for pair in all_positive_pairs
        for row in pair["rows"]
    )
    agentic_truncation_free = bool(
        (server_context_audit.get("agentic") or {}).get("truncation_free")
    )
    dataset_manifest = (
        _load(args.dataset_manifest) if args.dataset_manifest else {}
    )
    evaluation_benchmarks = sorted(
        {
            benchmark
            for split in args.evaluation_split
            for benchmark in (dataset_manifest.get("benchmarks") or {}).get(split, [])
        }
    )

    summary = {
        "schema_version": "c2hls.orchestrator_sft_ab.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "direct": compare_direct(
            _load(args.direct_control_vitis),
            _load(args.direct_treatment_vitis),
            _load(args.direct_control_cosim),
            _load(args.direct_treatment_cosim),
        ),
        "agentic": agentic,
        "server_context_audit": server_context_audit,
        "evaluation_scope": {
            "dataset_manifest": (
                str(args.dataset_manifest) if args.dataset_manifest else None
            ),
            "splits": args.evaluation_split,
            "benchmarks": evaluation_benchmarks,
        },
        "delivery_verification": {
            "framework_all_positive_injection_verified": framework_all_positive_verified,
            "agentic_server_truncation_free": agentic_truncation_free,
            "all_positive_skills_delivered": (
                framework_all_positive_verified and agentic_truncation_free
            ),
        },
        "sources": {
            "direct_control_vitis": str(args.direct_control_vitis),
            "direct_treatment_vitis": str(args.direct_treatment_vitis),
            "direct_control_cosim": str(args.direct_control_cosim),
            "direct_treatment_cosim": str(args.direct_treatment_cosim),
            "direct_server_manifest": (
                str(args.direct_server_manifest) if args.direct_server_manifest else None
            ),
            "agentic_server_manifest": (
                [str(path) for path in args.agentic_server_manifest]
                if args.agentic_server_manifest
                else []
            ),
            "agentic_pairs": args.agentic_pair,
            "dataset_manifest": (
                str(args.dataset_manifest) if args.dataset_manifest else None
            ),
        },
    }

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = args.output_prefix.with_suffix(".json")
    md_path = args.output_prefix.with_suffix(".md")
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    _write_markdown(summary, md_path)
    print(f"json={json_path}")
    print(f"markdown={md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
