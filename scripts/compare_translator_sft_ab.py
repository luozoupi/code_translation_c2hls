#!/usr/bin/env python3
"""Compare matched agentic sweep summaries for a translator-only SFT A/B."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


TRANSLATOR_ENV_KEYS = {
    "C2HLS_TRANSLATOR_MODEL",
    "C2HLS_TRANSLATOR_BASE_URL",
    "C2HLS_TRANSLATOR_REVISION",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--treatment", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    return parser.parse_args()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict) or not isinstance(value.get("rows"), list):
        raise ValueError(f"invalid sweep summary: {path}")
    return value


def _row_by_benchmark(summary: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for row in summary["rows"]:
        benchmark = str(row.get("bench") or "")
        if not benchmark:
            raise ValueError("sweep row is missing bench")
        if benchmark in rows:
            raise ValueError(f"duplicate sweep row for {benchmark}")
        rows[benchmark] = row
    return rows


def _ratio(numerator: Any, denominator: Any) -> float | None:
    try:
        numerator_value = float(numerator)
        denominator_value = float(denominator)
    except (TypeError, ValueError):
        return None
    if denominator_value == 0:
        return None
    return numerator_value / denominator_value


def _condition_metrics(row: dict[str, Any]) -> dict[str, Any]:
    current = row.get("current") or {}
    steps = current.get("step_cycles") or []
    final_step = steps[-1] if steps else {}
    best = current.get("best") or {}
    return {
        "success": bool(current.get("success")),
        "phase": current.get("phase"),
        "error": current.get("error"),
        "baseline_cycles": current.get("baseline_cycles"),
        "baseline_csim": current.get("baseline_csim"),
        "baseline_cosim_status": current.get("baseline_cosim_status"),
        "best_step": best.get("step"),
        "best_cycles": best.get("cycles"),
        "final_step": final_step.get("step"),
        "final_cycles": final_step.get("cycles"),
        "final_csim": final_step.get("csim"),
        "final_cosim_status": final_step.get("cosim_status"),
        "final_cosim_cycles": final_step.get("cosim_cycles"),
        "llm_usage": current.get("llm_usage") or {},
        "result_json": current.get("json"),
        "run_fingerprint_sha256": row.get("run_fingerprint_sha256"),
    }


def compare_summaries(
    control: dict[str, Any], treatment: dict[str, Any]
) -> dict[str, Any]:
    control_rows = _row_by_benchmark(control)
    treatment_rows = _row_by_benchmark(treatment)
    if set(control_rows) != set(treatment_rows):
        raise ValueError(
            "benchmark sets differ: "
            f"control={sorted(control_rows)}, treatment={sorted(treatment_rows)}"
        )

    control_env = control.get("env") or {}
    treatment_env = treatment.get("env") or {}
    compared_keys = sorted((set(control_env) | set(treatment_env)) - TRANSLATOR_ENV_KEYS)
    configuration_mismatches = {
        key: {"control": control_env.get(key), "treatment": treatment_env.get(key)}
        for key in compared_keys
        if control_env.get(key) != treatment_env.get(key)
    }

    rows = []
    for benchmark in sorted(control_rows):
        control_metrics = _condition_metrics(control_rows[benchmark])
        treatment_metrics = _condition_metrics(treatment_rows[benchmark])
        rows.append(
            {
                "benchmark": benchmark,
                "control": control_metrics,
                "treatment": treatment_metrics,
                "ratios": {
                    "baseline_cycles_treatment_over_control": _ratio(
                        treatment_metrics["baseline_cycles"],
                        control_metrics["baseline_cycles"],
                    ),
                    "best_cycles_treatment_over_control": _ratio(
                        treatment_metrics["best_cycles"],
                        control_metrics["best_cycles"],
                    ),
                    "final_cycles_treatment_over_control": _ratio(
                        treatment_metrics["final_cycles"],
                        control_metrics["final_cycles"],
                    ),
                },
            }
        )

    return {
        "schema_version": "c2hls.translator_sft_ab.v1",
        "control_stamp": control.get("stamp"),
        "treatment_stamp": treatment.get("stamp"),
        "control_translator": {
            key: control_env.get(key) for key in sorted(TRANSLATOR_ENV_KEYS)
        },
        "treatment_translator": {
            key: treatment_env.get(key) for key in sorted(TRANSLATOR_ENV_KEYS)
        },
        "non_translator_configuration_match": not configuration_mismatches,
        "configuration_mismatches": configuration_mismatches,
        "rows": rows,
    }


def _markdown(comparison: dict[str, Any]) -> str:
    lines = [
        "# Translator SFT A/B",
        "",
        f"Control: `{comparison.get('control_stamp')}`  ",
        f"Treatment: `{comparison.get('treatment_stamp')}`  ",
        "Non-translator configuration match: "
        f"`{comparison.get('non_translator_configuration_match')}`",
        "",
        "| benchmark | base baseline csim/cosim | SFT baseline csim/cosim | base final csim/cosim | SFT final csim/cosim | base baseline cycles | SFT baseline cycles | baseline ratio | base best cycles | SFT best cycles | best ratio |",
        "|---|---|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in comparison["rows"]:
        control = row["control"]
        treatment = row["treatment"]
        ratios = row["ratios"]

        def show(value: Any) -> str:
            if value is None:
                return "-"
            if isinstance(value, float):
                return f"{value:.4f}"
            return str(value)

        lines.append(
            f"| {row['benchmark']} | {show(control['baseline_csim'])}/{show(control['baseline_cosim_status'])} "
            f"| {show(treatment['baseline_csim'])}/{show(treatment['baseline_cosim_status'])} "
            f"| {show(control['final_csim'])}/{show(control['final_cosim_status'])} "
            f"| {show(treatment['final_csim'])}/{show(treatment['final_cosim_status'])} "
            f"| {show(control['baseline_cycles'])} | {show(treatment['baseline_cycles'])} "
            f"| {show(ratios['baseline_cycles_treatment_over_control'])} "
            f"| {show(control['best_cycles'])} | {show(treatment['best_cycles'])} "
            f"| {show(ratios['best_cycles_treatment_over_control'])} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = _parse_args()
    comparison = compare_summaries(_load(args.control), _load(args.treatment))
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = args.output_prefix.with_suffix(".json")
    md_path = args.output_prefix.with_suffix(".md")
    json_path.write_text(json.dumps(comparison, indent=2, sort_keys=True) + "\n")
    md_path.write_text(_markdown(comparison))
    print(json_path)
    print(md_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
