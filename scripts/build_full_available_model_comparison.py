#!/usr/bin/env python3
"""Build metric-aware HLSFactory tables from all currently available models."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, OrderedDict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable


REPO = Path(__file__).resolve().parent.parent
NA = "N/A"
BENCHMARKS = (
    "hlsfactory_2mm",
    "hlsfactory_3mm",
    "hlsfactory_atax",
    "hlsfactory_bicg",
    "hlsfactory_cholesky",
    "hlsfactory_correlation",
    "hlsfactory_covariance",
    "hlsfactory_doitgen",
    "hlsfactory_durbin",
    "hlsfactory_fdtd-2d",
    "hlsfactory_floyd-warshall",
    "hlsfactory_gemm",
    "hlsfactory_gemver",
    "hlsfactory_gesummv",
    "hlsfactory_gramschmidt",
    "hlsfactory_heat-3d",
    "hlsfactory_jacobi-1d",
    "hlsfactory_jacobi-2d",
    "hlsfactory_lu",
    "hlsfactory_ludcmp",
    "hlsfactory_mvt",
    "hlsfactory_nussinov",
    "hlsfactory_seidel-2d",
    "hlsfactory_symm",
    "hlsfactory_syr2k",
    "hlsfactory_syrk",
    "hlsfactory_trisolv",
    "hlsfactory_trmm",
)


@dataclass
class Arm:
    arm_id: str
    label: str
    model: str
    training: str
    campaign: str
    strategy: str
    skill_mode: str
    metric_type: str
    source_paths: list[str] = field(default_factory=list)
    note: str = ""


@dataclass
class Record:
    benchmark: str
    arm_id: str
    observed: bool
    valid: bool
    status: str
    raw_cycles: float | None
    source_reported_speedup: float | None
    source_reported_reference_kind: str | None
    source_path: str
    note: str = ""


class Builder:
    def __init__(self) -> None:
        self.arms: OrderedDict[str, Arm] = OrderedDict()
        self.records: dict[tuple[str, str], Record] = {}
        self.references: dict[str, dict[str, float]] = {
            "csynth": {},
            "cosim": {},
        }
        self.reference_sources: dict[str, dict[str, str]] = {
            "csynth": {},
            "cosim": {},
        }
        self.reference_conflicts: list[dict[str, Any]] = []
        self.sources: OrderedDict[str, dict[str, Any]] = OrderedDict()

    def source(self, path: Path) -> str:
        resolved = path.resolve()
        key = str(resolved)
        if key not in self.sources:
            raw = resolved.read_bytes()
            self.sources[key] = {
                "path": key,
                "bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        return key

    def arm(self, arm: Arm) -> None:
        existing = self.arms.get(arm.arm_id)
        if existing is None:
            self.arms[arm.arm_id] = arm
            return
        if asdict(existing) != asdict(arm):
            raise ValueError(f"conflicting arm metadata: {arm.arm_id}")

    def reference(
        self, kind: str, benchmark: str, value: float | None, source_path: str
    ) -> None:
        if value is None or benchmark not in BENCHMARKS:
            return
        previous = self.references[kind].get(benchmark)
        if previous is not None and not math.isclose(previous, value, rel_tol=0, abs_tol=0.5):
            self.reference_conflicts.append({
                "kind": kind,
                "benchmark": benchmark,
                "existing": previous,
                "new": value,
                "existing_source": self.reference_sources[kind][benchmark],
                "new_source": source_path,
            })
            return
        self.references[kind][benchmark] = value
        self.reference_sources[kind][benchmark] = source_path

    def record(self, record: Record) -> None:
        key = (record.arm_id, record.benchmark)
        if key in self.records:
            raise ValueError(f"duplicate model/setup/benchmark cell: {key}")
        if record.arm_id not in self.arms:
            raise ValueError(f"record references unknown arm: {record.arm_id}")
        if record.benchmark not in BENCHMARKS:
            raise ValueError(f"unknown benchmark: {record.benchmark}")
        self.records[key] = record


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def number(value: Any) -> float | None:
    text = str(value or "").strip()
    if not text or text.upper() in {"N/A", "NA", "FAIL", "NONE", "NULL"}:
        return None
    try:
        parsed = float(text)
    except ValueError:
        return None
    return parsed if math.isfinite(parsed) else None


def boolean(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "pass", "passed"}


def benchmark(value: str) -> str | None:
    text = str(value or "").strip()
    suffix = text.removeprefix("hlsfactory_").replace("_", "-")
    candidate = f"hlsfactory_{suffix}"
    return candidate if candidate in BENCHMARKS else None


def cycle_cell(value: float | None) -> str:
    if value is None:
        return NA
    return str(int(round(value))) if float(value).is_integer() else f"{value:.6f}"


def ratio_cell(value: float | None) -> str:
    return NA if value is None else f"{value:.6f}"


def metric_reference_kind(metric_type: str) -> str:
    return "cosim" if metric_type == "cosim" else "csynth"


def add_arm(
    builder: Builder,
    *,
    arm_id: str,
    label: str,
    model: str,
    training: str,
    campaign: str,
    strategy: str,
    skill_mode: str,
    metric_type: str,
    source_paths: Iterable[str],
    note: str = "",
) -> None:
    builder.arm(Arm(
        arm_id=arm_id,
        label=label,
        model=model,
        training=training,
        campaign=campaign,
        strategy=strategy,
        skill_mode=skill_mode,
        metric_type=metric_type,
        source_paths=list(dict.fromkeys(source_paths)),
        note=note,
    ))


def add_observation(
    builder: Builder,
    *,
    benchmark_name: str,
    arm_id: str,
    raw_cycles: float | None,
    valid: bool,
    status: str,
    source_path: str,
    source_reported_speedup: float | None = None,
    source_reported_reference_kind: str | None = None,
    note: str = "",
) -> None:
    builder.record(Record(
        benchmark=benchmark_name,
        arm_id=arm_id,
        observed=True,
        valid=bool(valid and raw_cycles is not None),
        status=status,
        raw_cycles=raw_cycles,
        source_reported_speedup=source_reported_speedup,
        source_reported_reference_kind=source_reported_reference_kind,
        source_path=source_path,
        note=note,
    ))


def ingest_devstral(builder: Builder, aav_path: Path, skillless_path: Path) -> None:
    aav_source = builder.source(aav_path)
    skillless_source = builder.source(skillless_path)
    aav_id = "devstral2__flash__aav_n__cosim"
    skillless_id = "devstral2__flash__skillless__csynth_avg"
    add_arm(
        builder,
        arm_id=aav_id,
        label="Devstral-2 flash / AAV_N (COSIM)",
        model="Devstral-2",
        training="base",
        campaign="devstral2_fixed_cosim",
        strategy="flash",
        skill_mode="aav_n",
        metric_type="cosim",
        source_paths=[aav_source],
    )
    add_arm(
        builder,
        arm_id=skillless_id,
        label="Devstral-2 flash / skillless (CSynth average)",
        model="Devstral-2",
        training="base",
        campaign="devstral2_fixed_cosim",
        strategy="flash",
        skill_mode="skillless",
        metric_type="csynth_avg",
        source_paths=[skillless_source],
        note="Source-reported speedup divides gold COSIM cycles by candidate CSynth average cycles; matching-reference speedup in this export uses CSynth/CSynth.",
    )
    for row in rows(aav_path):
        bench = benchmark(row.get("bench", ""))
        if not bench:
            continue
        gold = number(row.get("gold_cosim_kernel_runtime_cycles"))
        builder.reference("cosim", bench, gold, aav_source)
        raw = number(row.get("flash_cosim_kernel_runtime_cycles"))
        passed = boolean(row.get("flash_pass"))
        add_observation(
            builder,
            benchmark_name=bench,
            arm_id=aav_id,
            raw_cycles=raw,
            valid=passed,
            status="pass" if passed and raw is not None else "fail",
            source_path=aav_source,
            source_reported_speedup=number(row.get("speedup_gold_over_flash")),
            source_reported_reference_kind="cosim",
        )
    for row in rows(skillless_path):
        bench = benchmark(row.get("bench", ""))
        if not bench:
            continue
        gold = number(row.get("gold_cosim_kernel_runtime_cycles"))
        builder.reference("cosim", bench, gold, skillless_source)
        raw = number(row.get("flash_csynth_avg_latency_cycles"))
        passed = boolean(row.get("flash_csynth_ok"))
        add_observation(
            builder,
            benchmark_name=bench,
            arm_id=skillless_id,
            raw_cycles=raw,
            valid=passed,
            status="pass" if passed and raw is not None else "fail",
            source_path=skillless_source,
            source_reported_speedup=number(
                row.get("speedup_gold_cosim_over_flash_csynth_avg")
            ),
            source_reported_reference_kind="cosim_mixed_with_csynth_candidate",
        )
    # Explicit counterpart placeholders make the missing like-for-like cells
    # visible in the core tables instead of silently dropping the comparison.
    add_arm(
        builder,
        arm_id="devstral2__flash__aav_n__csynth",
        label="Devstral-2 flash / AAV_N (CSynth unavailable)",
        model="Devstral-2",
        training="base",
        campaign="devstral2_fixed_cosim",
        strategy="flash",
        skill_mode="aav_n",
        metric_type="csynth_latency",
        source_paths=[aav_source],
        note="Placeholder: the supplied AAV_N file contains COSIM, not CSynth cycles.",
    )
    add_arm(
        builder,
        arm_id="devstral2__flash__skillless__cosim",
        label="Devstral-2 flash / skillless (COSIM unavailable)",
        model="Devstral-2",
        training="base",
        campaign="devstral2_fixed_cosim",
        strategy="flash",
        skill_mode="skillless",
        metric_type="cosim",
        source_paths=[skillless_source],
        note="Placeholder: the supplied skillless file contains CSynth average, not COSIM cycles.",
    )


def ingest_fixed_pair(
    builder: Builder,
    *,
    model_slug: str,
    model_label: str,
    campaign: str,
    csynth_path: Path,
    cosim_path: Path,
) -> None:
    csynth_source = builder.source(csynth_path)
    cosim_source = builder.source(cosim_path)
    ids = {
        ("aav_n", "csynth_worst"): f"{model_slug}__{campaign}__aav_n__csynth_worst",
        ("skillless", "csynth_worst"): f"{model_slug}__{campaign}__skillless__csynth_worst",
        ("aav_n", "cosim"): f"{model_slug}__{campaign}__aav_n__cosim",
        ("skillless", "cosim"): f"{model_slug}__{campaign}__skillless__cosim",
    }
    for (mode, metric), arm_id in ids.items():
        add_arm(
            builder,
            arm_id=arm_id,
            label=f"{model_label} flash / {mode} ({metric})",
            model=model_label,
            training="base",
            campaign=campaign,
            strategy="flash",
            skill_mode=mode,
            metric_type=metric,
            source_paths=[csynth_source if metric != "cosim" else cosim_source],
        )
    for row in rows(csynth_path):
        bench = benchmark(row.get("bench", ""))
        if not bench:
            continue
        for mode, prefix in (("aav_n", "skills"), ("skillless", "noskills")):
            raw = number(row.get(f"{prefix}_latency_worst"))
            add_observation(
                builder,
                benchmark_name=bench,
                arm_id=ids[(mode, "csynth_worst")],
                raw_cycles=raw,
                valid=raw is not None,
                status="pass" if raw is not None else "fail",
                source_path=csynth_source,
            )
    for row in rows(cosim_path):
        bench = benchmark(row.get("bench", ""))
        if not bench:
            continue
        for mode, prefix in (("aav_n", "skills"), ("skillless", "noskills")):
            raw = number(row.get(f"{prefix}_cycles"))
            passed = boolean(row.get(f"{prefix}_pass"))
            add_observation(
                builder,
                benchmark_name=bench,
                arm_id=ids[(mode, "cosim")],
                raw_cycles=raw,
                valid=passed,
                status="pass" if passed and raw is not None else "fail",
                source_path=cosim_source,
            )


def ingest_deepseek_speedup_references(
    builder: Builder, aav_path: Path, skillless_path: Path
) -> None:
    for path in (aav_path, skillless_path):
        source = builder.source(path)
        for row in rows(path):
            bench = benchmark(row.get("bench", ""))
            if bench:
                builder.reference(
                    "cosim",
                    bench,
                    number(row.get("gold_cosim_kernel_runtime_cycles")),
                    source,
                )


def ingest_simple_csynth_wide(
    builder: Builder,
    *,
    path: Path,
    model: str,
    model_slug: str,
    campaign: str,
    gold_column: str,
    ignored_columns: set[str],
    metric_type: str,
    column_parser: Any,
) -> None:
    source = builder.source(path)
    source_rows = rows(path)
    if not source_rows:
        return
    columns = [
        column
        for column in source_rows[0]
        if column not in ignored_columns and column != gold_column
    ]
    metadata: dict[str, tuple[str, str, str, str]] = {}
    for column in columns:
        strategy, skill_mode, label_suffix = column_parser(column)
        arm_id = f"{model_slug}__{campaign}__{strategy}__{skill_mode}__{metric_type}"
        metadata[column] = (arm_id, strategy, skill_mode, label_suffix)
        add_arm(
            builder,
            arm_id=arm_id,
            label=f"{model} {label_suffix} ({metric_type})",
            model=model,
            training="base",
            campaign=campaign,
            strategy=strategy,
            skill_mode=skill_mode,
            metric_type=metric_type,
            source_paths=[source],
        )
    for row in source_rows:
        bench = benchmark(row.get("benchmark") or row.get("bench") or "")
        if not bench:
            continue
        builder.reference("csynth", bench, number(row.get(gold_column)), source)
        for column, (arm_id, _strategy, _mode, _label) in metadata.items():
            raw = number(row.get(column))
            observed = str(row.get(column) or "").strip() != ""
            if not observed:
                continue
            add_observation(
                builder,
                benchmark_name=bench,
                arm_id=arm_id,
                raw_cycles=raw,
                valid=raw is not None,
                status="pass" if raw is not None else "fail",
                source_path=source,
            )


def ingest_skill_v3_qwen_gemma(builder: Builder, path: Path) -> None:
    source = builder.source(path)
    source_rows = rows(path)
    model_names = {
        "qwen3_6_27b": "Qwen3.6-27B",
        "gemma4_31b": "Gemma-4-31B",
    }
    columns = [
        key for key in source_rows[0]
        if key not in {"benchmark", "reference_worst_case_cycles", "reference_source_kind"}
    ]
    metadata: dict[str, str] = {}
    for column in columns:
        model_slug, strategy, skill_mode = column.split("__", 2)
        arm_id = f"{model_slug}__skill_v3_20260731__{strategy}__{skill_mode}__csynth_worst"
        metadata[column] = arm_id
        add_arm(
            builder,
            arm_id=arm_id,
            label=f"{model_names[model_slug]} skill-v3 {strategy} / {skill_mode} (CSynth worst)",
            model=model_names[model_slug],
            training="base",
            campaign="skill_v3_20260731",
            strategy=strategy,
            skill_mode=skill_mode,
            metric_type="csynth_worst",
            source_paths=[source],
        )
    for row in source_rows:
        bench = benchmark(row.get("benchmark", ""))
        if not bench:
            continue
        builder.reference(
            "csynth", bench, number(row.get("reference_worst_case_cycles")), source
        )
        for column, arm_id in metadata.items():
            raw = number(row.get(column))
            if raw is None:
                continue
            add_observation(
                builder,
                benchmark_name=bench,
                arm_id=arm_id,
                raw_cycles=raw,
                valid=True,
                status="pass",
                source_path=source,
            )


def ingest_agentic_summary(
    builder: Builder,
    path: Path,
    *,
    model: str,
    model_slug: str,
    campaign: str,
    strategy: str,
    require_reference_audit: bool,
) -> None:
    source = builder.source(path)
    payload = json.loads(path.read_text())
    for row in payload.get("rows") or []:
        bench = benchmark(row.get("bench", ""))
        current = row.get("current") or {}
        skill_mode = str(row.get("skill_mode") or "unknown")
        arm_id = (
            f"{model_slug}__{campaign}__{strategy}__{skill_mode}__csynth_latency"
        )
        audit_note = (
            "Strict reference-isolation audit required."
            if require_reference_audit
            else "CSim/CSynth-valid measurements retained even when the post-hoc "
            "reference-isolation audit failed; consult per-cell status."
        )
        add_arm(
            builder,
            arm_id=arm_id,
            label=f"{model} {campaign} {strategy} / {skill_mode} (CSynth)",
            model=model,
            training="base",
            campaign=campaign,
            strategy=strategy,
            skill_mode=skill_mode,
            metric_type="csynth_latency",
            source_paths=[source],
            note=f"{audit_note} CSim and CSynth; no COSIM.",
        )
        if not bench:
            continue
        raw = number((current.get("best") or {}).get("cycles"))
        evaluation = current.get("evaluation_status") or {}
        audit = current.get("reference_isolation_audit") or {}
        synthesis_valid = bool(
            current.get("success")
            and evaluation.get("correctness_status") == "passed"
            and evaluation.get("synthesis_status") == "passed"
            and raw is not None
        )
        audit_passed = audit.get("passed") is True
        valid = synthesis_valid and (audit_passed or not require_reference_audit)
        if not synthesis_valid:
            status = "fail"
        elif audit_passed:
            status = "pass"
        elif require_reference_audit:
            status = "reference_audit_fail"
        else:
            status = "pass_reference_audit_failed"
        finding_count = number(audit.get("finding_count"))
        add_observation(
            builder,
            benchmark_name=bench,
            arm_id=arm_id,
            raw_cycles=raw,
            valid=valid,
            status=status,
            source_path=source,
            note=(
                f"reference_isolation_audit_passed={str(audit_passed).lower()}; "
                f"finding_count={cycle_cell(finding_count)}"
            ),
        )


def ingest_gpt_or_opus(
    builder: Builder, path: Path, *, model: str, model_slug: str
) -> None:
    source = builder.source(path)
    column_map = {
        "oneshot": ("one_shot", "none"),
        "flash_noskills": ("flash", "skillless"),
        "flash_curated": ("flash", "curated"),
        "flash_allpos": ("flash", "all_positive"),
        "ms_noskills": ("multistep", "skillless"),
        "ms_curated": ("multistep", "curated"),
        "ms_allpos": ("multistep", "all_positive"),
    }
    arm_ids: dict[str, str] = {}
    for column, (strategy, mode) in column_map.items():
        arm_id = f"{model_slug}__newskills_20260730__{strategy}__{mode}__csynth_latency"
        arm_ids[column] = arm_id
        add_arm(
            builder,
            arm_id=arm_id,
            label=f"{model} new-skills {strategy} / {mode} (CSynth)",
            model=model,
            training="base_api",
            campaign="newskills_20260730",
            strategy=strategy,
            skill_mode=mode,
            metric_type="csynth_latency",
            source_paths=[source],
            note="Reference comparison is indicative because the source report notes a Vivado/Vitis flow mismatch.",
        )
    for row in rows(path):
        bench = benchmark(row.get("bench", ""))
        if not bench:
            continue
        builder.reference("csynth", bench, number(row.get("gold")), source)
        for column, arm_id in arm_ids.items():
            text = str(row.get(column) or "").strip()
            raw = number(text)
            add_observation(
                builder,
                benchmark_name=bench,
                arm_id=arm_id,
                raw_cycles=raw,
                valid=raw is not None,
                status="pass" if raw is not None else "fail",
                source_path=source,
            )


def ingest_legacy_top5(builder: Builder, path: Path) -> None:
    source = builder.source(path)
    model_map = {
        "Sonnet": ("sonnet46", "Claude Sonnet 4.6"),
        "Luna": ("luna", "Luna"),
        "Grok": ("grok45", "Grok-4.5"),
        "Haiku": ("haiku", "Claude Haiku"),
        "DeepSeek": ("deepseek_v4_flash", "DeepSeek-v4-Flash"),
    }
    for row in rows(path):
        model_key, model_label = model_map[row["Model"]]
        mode = "aav_n" if row["Arm"] == "skills" else "skillless"
        arm_id = f"{model_key}__legacy_top5__flash__{mode}__cosim"
        add_arm(
            builder,
            arm_id=arm_id,
            label=f"{model_label} legacy top-5 flash / {mode} (COSIM)",
            model=model_label,
            training="base",
            campaign="legacy_top5",
            strategy="flash",
            skill_mode=mode,
            metric_type="cosim",
            source_paths=[source],
            note="Only five benchmarks were present in the legacy consolidated table.",
        )
        for problem, value in row.items():
            bench = benchmark(problem)
            raw = number(value)
            if not bench or raw is None:
                continue
            add_observation(
                builder,
                benchmark_name=bench,
                arm_id=arm_id,
                raw_cycles=raw,
                valid=True,
                status="pass",
                source_path=source,
            )


def ingest_qwen_gemma_history(
    builder: Builder,
    cycles_path: Path,
    status_path: Path,
    setup_summary_path: Path,
) -> None:
    cycles_source = builder.source(cycles_path)
    status_source = builder.source(status_path)
    summary_source = builder.source(setup_summary_path)
    setup_meta = {row["setup_id"]: row for row in rows(setup_summary_path)}
    cycle_rows = rows(cycles_path)
    status_rows = {row["benchmark"]: row for row in rows(status_path)}
    ignored = {"benchmark", "split", "reference_cycles_worst"}
    columns = [key for key in cycle_rows[0] if key not in ignored]
    arm_ids: dict[str, str] = {}
    for column in columns:
        meta = setup_meta.get(column, {})
        model_family = meta.get("model_family") or (
            "gemma4_31b" if column.startswith("gemma") else "qwen3_6_27b"
        )
        model = "Gemma-4-31B" if model_family == "gemma4_31b" else "Qwen3.6-27B"
        arm_id = f"{model_family}__historical_and_sft__{column}__csynth_worst"
        arm_ids[column] = arm_id
        add_arm(
            builder,
            arm_id=arm_id,
            label=meta.get("setup_label") or column,
            model=model,
            training=meta.get("training_state") or "unknown",
            campaign=meta.get("cohort") or "historical_and_sft_20260729",
            strategy=meta.get("strategy") or meta.get("workflow") or "unknown",
            skill_mode=meta.get("skill_mode") or "none",
            metric_type="csynth_worst",
            source_paths=[cycles_source, status_source, summary_source],
            note="Historical and SFT matrix; many setups cover only smoke or val/test subsets.",
        )
    for row in cycle_rows:
        bench = benchmark(row.get("benchmark", ""))
        if not bench:
            continue
        builder.reference(
            "csynth", bench, number(row.get("reference_cycles_worst")), cycles_source
        )
        status_row = status_rows.get(row["benchmark"], {})
        for column, arm_id in arm_ids.items():
            text = str(row.get(column) or "").strip()
            if not text:
                continue
            raw = number(text)
            status = str(status_row.get(column) or "unknown").strip().lower()
            valid = status == "pass" and raw is not None
            add_observation(
                builder,
                benchmark_name=bench,
                arm_id=arm_id,
                raw_cycles=raw,
                valid=valid,
                status=status if status else "unknown",
                source_path=cycles_source,
            )


def matching_speedup(builder: Builder, arm: Arm, record: Record | None) -> float | None:
    if record is None or not record.valid or record.raw_cycles in (None, 0):
        return None
    kind = metric_reference_kind(arm.metric_type)
    reference = builder.references[kind].get(record.benchmark)
    if reference is None:
        return None
    return reference / record.raw_cycles


def write_csv(path: Path, fieldnames: list[str], output_rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(output_rows)


def wide_rows(builder: Builder, arm_ids: list[str], value_kind: str) -> list[dict[str, str]]:
    output = []
    for bench in BENCHMARKS:
        row = {
            "benchmark": bench,
            "reference_csynth_cycles": cycle_cell(builder.references["csynth"].get(bench)),
            "reference_cosim_cycles": cycle_cell(builder.references["cosim"].get(bench)),
        }
        for arm_id in arm_ids:
            arm = builder.arms[arm_id]
            record = builder.records.get((arm_id, bench))
            if value_kind == "cycles":
                value = record.raw_cycles if record and record.valid else None
                row[arm_id] = cycle_cell(value)
            elif value_kind == "speedup":
                row[arm_id] = ratio_cell(matching_speedup(builder, arm, record))
            elif value_kind == "source_speedup":
                value = record.source_reported_speedup if record else None
                row[arm_id] = ratio_cell(value)
            elif value_kind == "status":
                row[arm_id] = record.status if record else NA
            else:
                raise ValueError(value_kind)
        output.append(row)
    return output


def markdown_table(
    builder: Builder, arm_ids: list[str], *, metric: str, title: str
) -> str:
    headers = ["benchmark", f"reference_{metric}"] + [
        builder.arms[arm_id].label for arm_id in arm_ids
    ]
    lines = [f"# {title}", "", "| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    for bench in BENCHMARKS:
        values = [bench, cycle_cell(builder.references[metric].get(bench))]
        for arm_id in arm_ids:
            record = builder.records.get((arm_id, bench))
            values.append(cycle_cell(record.raw_cycles if record and record.valid else None))
        lines.append("| " + " | ".join(values) + " |")
    lines.extend([
        "",
        "`N/A` means the arm/benchmark/metric was absent or did not pass its recorded validation gate.",
    ])
    return "\n".join(lines) + "\n"


def export_model_subset(
    builder: Builder,
    *,
    output_dir: Path,
    model: str,
    all_long_rows: list[dict[str, Any]],
    long_fields: list[str],
    all_manifest_rows: list[dict[str, Any]],
    manifest_fields: list[str],
    latest_campaign: str,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    arm_ids = [
        arm_id for arm_id, arm in builder.arms.items() if arm.model == model
    ]
    if not arm_ids:
        raise ValueError(f"no arms available for model subset: {model}")
    latest_ids = [
        arm_id for arm_id in arm_ids
        if builder.arms[arm_id].campaign == latest_campaign
    ]
    if not latest_ids:
        raise ValueError(
            f"no arms available for {model} campaign {latest_campaign}"
        )

    for prefix, selected_ids in (
        ("all_campaigns", arm_ids),
        ("latest_full_matrix", latest_ids),
    ):
        fields = [
            "benchmark", "reference_csynth_cycles", "reference_cosim_cycles",
            *selected_ids,
        ]
        for kind, suffix in (
            ("cycles", "cycles_wide.csv"),
            ("speedup", "speedup_wide.csv"),
            ("status", "status_wide.csv"),
        ):
            write_csv(
                output_dir / f"{prefix}_{suffix}",
                fields,
                wide_rows(builder, selected_ids, kind),
            )

    selected = set(arm_ids)
    subset_long_rows = [
        row for row in all_long_rows if row["arm_id"] in selected
    ]
    subset_manifest_rows = [
        row for row in all_manifest_rows if row["arm_id"] in selected
    ]
    write_csv(
        output_dir / "all_campaigns_records_long.csv",
        long_fields,
        subset_long_rows,
    )
    write_csv(
        output_dir / "arm_manifest.csv",
        manifest_fields,
        subset_manifest_rows,
    )
    (output_dir / "latest_full_matrix_cycles.md").write_text(
        markdown_table(
            builder,
            latest_ids,
            metric="csynth",
            title=f"{model} Latest Full Matrix CSynth Cycles",
        )
    )

    observed_records = [
        builder.records[(arm_id, bench)]
        for arm_id in arm_ids
        for bench in BENCHMARKS
        if (arm_id, bench) in builder.records
    ]
    latest_records = [
        builder.records[(arm_id, bench)]
        for arm_id in latest_ids
        for bench in BENCHMARKS
        if (arm_id, bench) in builder.records
    ]
    source_paths = {
        path for arm_id in arm_ids for path in builder.arms[arm_id].source_paths
    }
    manifest = {
        "schema_version": "c2hls.model-comparison-subset.v1",
        "model": model,
        "benchmark_count": len(BENCHMARKS),
        "arm_count": len(arm_ids),
        "grid_cell_count": len(BENCHMARKS) * len(arm_ids),
        "observed_cell_count": len(observed_records),
        "valid_cell_count": sum(record.valid for record in observed_records),
        "campaigns": sorted({builder.arms[arm_id].campaign for arm_id in arm_ids}),
        "status_counts": dict(sorted(Counter(
            record.status for record in observed_records
        ).items())),
        "latest_full_matrix": {
            "campaign": latest_campaign,
            "arm_count": len(latest_ids),
            "grid_cell_count": len(BENCHMARKS) * len(latest_ids),
            "observed_cell_count": len(latest_records),
            "valid_cell_count": sum(record.valid for record in latest_records),
            "status_counts": dict(sorted(Counter(
                record.status for record in latest_records
            ).items())),
            "arm_ids": latest_ids,
        },
        "sources": [
            builder.sources[path] for path in sorted(source_paths)
        ],
        "missing_cell_literal": NA,
    }
    manifest["sources_sha256"] = hashlib.sha256(
        json.dumps(manifest["sources"], sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    readme = f"""# {model} Comparison Tables

This directory extracts every completed {model} arm indexed by the parent full-model comparison.

- Benchmarks: **{manifest['benchmark_count']}** (27 current kernels plus historical `doitgen`)
- All campaigns: **{manifest['arm_count']} arms**, **{manifest['observed_cell_count']} observed cells**, **{manifest['valid_cell_count']} valid cells**
- Latest full matrix: **{manifest['latest_full_matrix']['arm_count']} arms**, **{manifest['latest_full_matrix']['observed_cell_count']} observed cells**, **{manifest['latest_full_matrix']['valid_cell_count']} CSim/CSynth-valid cells**
- Missing or invalid cells are literal **`N/A`**.

## Campaigns

- `skill_v3_nonthinking16k_20260803`: 27 kernels crossed with flash/dynamic and five skill modes. Its CSim/CSynth measurements passed, but the later strict reference-isolation audit failed; cells are marked `pass_reference_audit_failed` and must not support leakage-controlled claims.
- `fixed_cosim_20260730`: flash `skillless` and `aav_n` CSynth/COSIM measurements.
- `legacy_top5`: five-kernel consolidated COSIM measurements retained as a separate historical campaign.

## Files

- `latest_full_matrix_cycles_wide.csv`, `latest_full_matrix_speedup_wide.csv`, and `latest_full_matrix_status_wide.csv`: the focused 10-arm completed matrix.
- `all_campaigns_cycles_wide.csv`, `all_campaigns_speedup_wide.csv`, and `all_campaigns_status_wide.csv`: all indexed {model} campaigns.
- `all_campaigns_records_long.csv`: normalized grid with metric type, provenance, validation, and audit status.
- `arm_manifest.csv`: arm definitions, coverage, and matching-reference geomean speedups.
- `latest_full_matrix_cycles.md`: readable CSynth table for the latest full matrix.
- `manifest.json`: counts, source hashes, and export invariants.
"""
    (output_dir / "README.md").write_text(readme)
    return manifest


def export(builder: Builder, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    arm_ids = list(builder.arms)
    long_fields = [
        "benchmark", "arm_id", "label", "model", "training", "campaign",
        "strategy", "skill_mode", "metric_type", "observed", "valid", "status",
        "cycles", "raw_cycles", "reference_kind", "reference_cycles",
        "speedup_vs_matching_reference", "source_reported_speedup",
        "source_reported_reference_kind", "source_path", "note",
    ]
    long_rows: list[dict[str, Any]] = []
    for bench in BENCHMARKS:
        for arm_id, arm in builder.arms.items():
            record = builder.records.get((arm_id, bench))
            reference_kind = metric_reference_kind(arm.metric_type)
            reference = builder.references[reference_kind].get(bench)
            speedup = matching_speedup(builder, arm, record)
            long_rows.append({
                "benchmark": bench,
                "arm_id": arm_id,
                "label": arm.label,
                "model": arm.model,
                "training": arm.training,
                "campaign": arm.campaign,
                "strategy": arm.strategy,
                "skill_mode": arm.skill_mode,
                "metric_type": arm.metric_type,
                "observed": str(bool(record)).lower(),
                "valid": str(bool(record and record.valid)).lower(),
                "status": record.status if record else NA,
                "cycles": cycle_cell(record.raw_cycles if record and record.valid else None),
                "raw_cycles": cycle_cell(record.raw_cycles if record else None),
                "reference_kind": reference_kind,
                "reference_cycles": cycle_cell(reference),
                "speedup_vs_matching_reference": ratio_cell(speedup),
                "source_reported_speedup": ratio_cell(
                    record.source_reported_speedup if record else None
                ),
                "source_reported_reference_kind": (
                    record.source_reported_reference_kind
                    if record and record.source_reported_reference_kind else NA
                ),
                "source_path": record.source_path if record else NA,
                "note": "; ".join(filter(None, [arm.note, record.note if record else ""])),
            })
    write_csv(output_dir / "all_available_model_setups_long.csv", long_fields, long_rows)

    wide_fields = ["benchmark", "reference_csynth_cycles", "reference_cosim_cycles", *arm_ids]
    for kind, filename in (
        ("cycles", "all_available_model_setups_cycles_wide.csv"),
        ("speedup", "all_available_model_setups_speedup_wide.csv"),
        ("source_speedup", "all_available_model_setups_source_reported_speedup_wide.csv"),
        ("status", "all_available_model_setups_status_wide.csv"),
    ):
        write_csv(output_dir / filename, wide_fields, wide_rows(builder, arm_ids, kind))

    manifest_rows: list[dict[str, Any]] = []
    for arm_id, arm in builder.arms.items():
        arm_records = [
            builder.records.get((arm_id, bench)) for bench in BENCHMARKS
        ]
        observed = sum(record is not None for record in arm_records)
        valid = sum(bool(record and record.valid) for record in arm_records)
        ratios = [
            matching_speedup(builder, arm, record)
            for record in arm_records
        ]
        ratios = [value for value in ratios if value is not None and value > 0]
        geomean = math.exp(sum(math.log(value) for value in ratios) / len(ratios)) if ratios else None
        manifest_rows.append({
            **asdict(arm),
            "source_paths": " | ".join(arm.source_paths),
            "observed_benchmarks": observed,
            "valid_benchmarks": valid,
            "matching_reference_benchmarks": len(ratios),
            "geomean_speedup_vs_matching_reference": ratio_cell(geomean),
        })
    manifest_fields = [
        "arm_id", "label", "model", "training", "campaign", "strategy",
        "skill_mode", "metric_type", "observed_benchmarks", "valid_benchmarks",
        "matching_reference_benchmarks", "geomean_speedup_vs_matching_reference",
        "source_paths", "note",
    ]
    write_csv(output_dir / "arm_manifest.csv", manifest_fields, manifest_rows)

    deepseek_flash_subset = export_model_subset(
        builder,
        output_dir=output_dir / "deepseek_v4_flash",
        model="DeepSeek-v4-Flash",
        all_long_rows=long_rows,
        long_fields=long_fields,
        all_manifest_rows=manifest_rows,
        manifest_fields=manifest_fields,
        latest_campaign="skill_v3_nonthinking16k_20260803",
    )

    core_csynth = [
        "devstral2__flash__skillless__csynth_avg",
        "devstral2__flash__aav_n__csynth",
        "deepseek_v4_flash__skill_v3_nonthinking16k_20260803__flash__skillless__csynth_latency",
        "deepseek_v4_flash__skill_v3_nonthinking16k_20260803__flash__all_positive__csynth_latency",
        "deepseek_v4_flash__fixed_cosim_20260730__skillless__csynth_worst",
        "deepseek_v4_flash__fixed_cosim_20260730__aav_n__csynth_worst",
        "grok45__skill_v3_audited_20260812__flash__skillless__csynth_latency",
        "grok45__skill_v3_audited_20260812__flash__all_positive__csynth_latency",
        "sonnet46__skill_v2_20260726__flash__skillless__csynth_latency",
        "sonnet46__skill_v2_20260726__flash__all_positive__csynth_latency",
        "gemma4_31b__skill_v3_20260731__flash__skillless__csynth_worst",
        "gemma4_31b__skill_v3_20260731__flash__all_positive__csynth_worst",
        "qwen3_6_27b__skill_v3_20260731__flash__skillless__csynth_worst",
        "qwen3_6_27b__skill_v3_20260731__flash__all_positive__csynth_worst",
        "gpt55__newskills_20260730__flash__skillless__csynth_latency",
        "gpt55__newskills_20260730__flash__all_positive__csynth_latency",
        "opus__newskills_20260730__flash__skillless__csynth_latency",
        "opus__newskills_20260730__flash__all_positive__csynth_latency",
    ]
    core_cosim = [
        "devstral2__flash__skillless__cosim",
        "devstral2__flash__aav_n__cosim",
        "deepseek_v4_flash__fixed_cosim_20260730__skillless__cosim",
        "deepseek_v4_flash__fixed_cosim_20260730__aav_n__cosim",
        "grok45__fixed_cosim_20260730__skillless__cosim",
        "grok45__fixed_cosim_20260730__aav_n__cosim",
        "sonnet46__legacy_top5__flash__skillless__cosim",
        "sonnet46__legacy_top5__flash__aav_n__cosim",
        "luna__legacy_top5__flash__skillless__cosim",
        "luna__legacy_top5__flash__aav_n__cosim",
        "haiku__legacy_top5__flash__skillless__cosim",
        "haiku__legacy_top5__flash__aav_n__cosim",
    ]
    for arm_id in [*core_csynth, *core_cosim]:
        if arm_id not in builder.arms:
            raise ValueError(f"core arm unavailable: {arm_id}")
    core_ids = [*core_csynth, *core_cosim]
    core_fields = ["benchmark", "reference_csynth_cycles", "reference_cosim_cycles", *core_ids]
    write_csv(
        output_dir / "core_flash_model_comparison_cycles_wide.csv",
        core_fields,
        wide_rows(builder, core_ids, "cycles"),
    )
    write_csv(
        output_dir / "core_flash_model_comparison_speedup_wide.csv",
        core_fields,
        wide_rows(builder, core_ids, "speedup"),
    )
    (output_dir / "core_flash_csynth_cycles.md").write_text(
        markdown_table(
            builder,
            core_csynth,
            metric="csynth",
            title="Core Flash CSynth Cycle Comparison",
        )
    )
    (output_dir / "core_flash_cosim_cycles.md").write_text(
        markdown_table(
            builder,
            core_cosim,
            metric="cosim",
            title="Core Flash COSIM Cycle Comparison",
        )
    )

    manifest = {
        "schema_version": "c2hls.full-available-model-comparison.v1",
        "benchmark_count": len(BENCHMARKS),
        "arm_count": len(builder.arms),
        "grid_cell_count": len(BENCHMARKS) * len(builder.arms),
        "observed_cell_count": len(builder.records),
        "valid_cell_count": sum(record.valid for record in builder.records.values()),
        "reference_coverage": {
            kind: len(values) for kind, values in builder.references.items()
        },
        "missing_reference_benchmarks": {
            kind: sorted(set(BENCHMARKS) - set(values))
            for kind, values in builder.references.items()
        },
        "reference_conflicts": builder.reference_conflicts,
        "sources": list(builder.sources.values()),
        "core_csynth_arms": core_csynth,
        "core_cosim_arms": core_cosim,
        "model_subsets": {
            "deepseek_v4_flash": {
                "path": "deepseek_v4_flash",
                "arm_count": deepseek_flash_subset["arm_count"],
                "observed_cell_count": deepseek_flash_subset[
                    "observed_cell_count"
                ],
                "valid_cell_count": deepseek_flash_subset["valid_cell_count"],
                "manifest": "deepseek_v4_flash/manifest.json",
            }
        },
        "missing_cell_literal": NA,
    }
    manifest["sources_sha256"] = hashlib.sha256(
        json.dumps(manifest["sources"], sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    readme = f"""# Full Available Model Comparison

This export joins the two supplied Devstral-2 tables with all currently indexed HLSFactory model/setup tables in this repository.

- Benchmarks: **{manifest['benchmark_count']}** (the 27 current kernels plus `doitgen` for historical coverage)
- Setup/metric arms: **{manifest['arm_count']}**
- Grid cells: **{manifest['grid_cell_count']}**
- Observed cells: **{manifest['observed_cell_count']}**
- Valid comparison cells: **{manifest['valid_cell_count']}**
- Missing or invalid cells are written literally as **`N/A`** in cycle/speedup tables.

## Metric Policy

- `csynth_*` cycles are compared only with the common CSynth reference.
- `cosim` cycles are compared only with the fixed-COSIM reference.
- Candidate CSynth statistics are not rewritten: each arm remains explicitly `csynth_avg`, `csynth_worst`, or selected `csynth_latency` in `arm_manifest.csv` and the long table.
- Devstral-2 skillless originally reports `gold COSIM / candidate CSynth average`; that mixed source ratio is retained in `source_reported_speedup`, but the normalized speedup table recomputes CSynth/CSynth.
- Fixed gold COSIM cycles are absent for `fdtd-2d`, `heat-3d`, `seidel-2d`, and `syr2k`; their matching-reference COSIM speedups are therefore `N/A`.
- A failed validation remains visible in the status and raw-cycle columns, but its comparison cycle is `N/A`.
- The latest DeepSeek full matrix passes CSim/CSynth but failed the later strict reference-isolation audit. Its cycles are retained for inventory completeness and marked `pass_reference_audit_failed`; do not use that campaign for leakage-controlled claims.
- These are deterministic samples from different campaign generations and skill catalogs; the table is an inventory, not a claim that every arm is a controlled apples-to-apples ablation.

## Main Files

- `core_flash_model_comparison_cycles_wide.csv`: compact cross-model flash comparison.
- `core_flash_model_comparison_speedup_wide.csv`: like-for-like speedups for the compact comparison.
- `core_flash_csynth_cycles.md` and `core_flash_cosim_cycles.md`: readable metric-separated tables.
- `all_available_model_setups_cycles_wide.csv`: exhaustive benchmark-by-arm cycle matrix.
- `all_available_model_setups_speedup_wide.csv`: exhaustive matching-reference speedup matrix.
- `all_available_model_setups_source_reported_speedup_wide.csv`: ratios exactly reported by source tables where available.
- `all_available_model_setups_status_wide.csv`: pass/fail/`N/A` status matrix.
- `all_available_model_setups_long.csv`: normalized full grid with provenance.
- `arm_manifest.csv`: setup definitions, coverage, and geomean matching-reference speedup.
- `deepseek_v4_flash/`: dedicated all-campaign and latest-full-matrix V4-Flash tables.
- `manifest.json`: source hashes and export invariants.
"""
    (output_dir / "README.md").write_text(readme)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO / "results_sweeps" / "full_model_comparison_20260816",
    )
    parser.add_argument(
        "--devstral-aav",
        type=Path,
        default=Path("/home/luo00466/devstral2_aav_n_flash_cosim_speedup_vs_gold.csv"),
    )
    parser.add_argument(
        "--devstral-skillless",
        type=Path,
        default=Path("/home/luo00466/devstral2_noskills_flash_csynth_avg_speedup_vs_gold_cosim.csv"),
    )
    args = parser.parse_args()

    builder = Builder()
    ingest_devstral(builder, args.devstral_aav, args.devstral_skillless)

    result_dir = REPO / "results_sweeps" / "results-20260730"
    ingest_deepseek_speedup_references(
        builder,
        REPO / "deepseek_v4_flash_aav_n_flash_cosim_speedup_vs_gold.csv",
        REPO / "deepseek_v4_flash_noskills_flash_cosim_speedup_vs_gold.csv",
    )
    ingest_fixed_pair(
        builder,
        model_slug="deepseek_v4_flash",
        model_label="DeepSeek-v4-Flash",
        campaign="fixed_cosim_20260730",
        csynth_path=result_dir / "deepseek_v4_flash_flash_csynth_skills_vs_noskills.csv",
        cosim_path=result_dir / "deepseek_v4_flash_flash_cosim_skills_vs_noskills.csv",
    )
    ingest_fixed_pair(
        builder,
        model_slug="grok45",
        model_label="Grok-4.5",
        campaign="fixed_cosim_20260730",
        csynth_path=result_dir / "grok45_flash_csynth_skills_vs_noskills.csv",
        cosim_path=result_dir / "grok45_flash_cosim_skills_vs_noskills.csv",
    )

    sonnet_path = (
        REPO / "results_sweeps" / "sonnet46_skillv2_latency_matrix_20260726"
        / "sonnet46_skillv2_latency_cycles_wide.csv"
    )
    ingest_simple_csynth_wide(
        builder,
        path=sonnet_path,
        model="Claude Sonnet 4.6",
        model_slug="sonnet46",
        campaign="skill_v2_20260726",
        gold_column="gold_reference_cycles",
        ignored_columns={"benchmark", "problem"},
        metric_type="csynth_latency",
        column_parser=lambda column: (
            column.split("__", 1)[0],
            column.split("__", 1)[1],
            column.replace("__", " / "),
        ),
    )

    qg_v3 = (
        REPO / "results_sweeps" / "skill_v3_qwen_gemma_matrix_20260731"
        / "worst_cycles_wide.csv"
    )
    ingest_skill_v3_qwen_gemma(builder, qg_v3)

    grok_prefix = (
        "agentic_no_streamcluster_skillv3_no_rmw_fc27133_grok_4_5_"
    )
    for strategy in ("flash", "dynamic"):
        ingest_agentic_summary(
            builder,
            REPO / "artifacts" /
            f"{grok_prefix}{strategy}_setups5_audited_v2_nocosim_"
            "grok45_lowreasoning16k_full_audited_20260812.summary.json",
            model="Grok-4.5",
            model_slug="grok45",
            campaign="skill_v3_audited_20260812",
            strategy=strategy,
            require_reference_audit=True,
        )

    deepseek_prefix = (
        "agentic_no_streamcluster_skillv3_no_rmw_fc27133_"
        "deepseek_v4_flash_"
    )
    for strategy in ("flash", "dynamic"):
        ingest_agentic_summary(
            builder,
            REPO / "artifacts" /
            f"{deepseek_prefix}{strategy}_skills5_nocosim_"
            "deepseek_nonthinking16k_complete10_20260803.summary.json",
            model="DeepSeek-v4-Flash",
            model_slug="deepseek_v4_flash",
            campaign="skill_v3_nonthinking16k_20260803",
            strategy=strategy,
            require_reference_audit=False,
        )

    ingest_gpt_or_opus(
        builder,
        result_dir / "newskills_gpt55_csynth_table.csv",
        model="GPT-5.5",
        model_slug="gpt55",
    )
    ingest_gpt_or_opus(
        builder,
        result_dir / "newskills_opus_csynth_table.csv",
        model="Claude Opus",
        model_slug="opus",
    )
    ingest_legacy_top5(builder, REPO / "model_arm_benchmark_cycles_with_deepseek.csv")

    qg_history = REPO / "results_sweeps" / "qwen_gemma_oneshot_agentic_sft_20260729"
    ingest_qwen_gemma_history(
        builder,
        qg_history / "comparison_cycles_wide.csv",
        qg_history / "comparison_status_wide.csv",
        qg_history / "setup_summary.csv",
    )

    manifest = export(builder, args.output_dir)
    if manifest["reference_conflicts"]:
        raise RuntimeError(
            f"reference conflicts detected: {len(manifest['reference_conflicts'])}"
        )
    if manifest["reference_coverage"]["csynth"] != len(BENCHMARKS):
        raise RuntimeError("incomplete CSynth reference coverage")
    missing_cosim = set(BENCHMARKS) - set(builder.references["cosim"])
    expected_missing_cosim = {
        "hlsfactory_fdtd-2d",
        "hlsfactory_heat-3d",
        "hlsfactory_seidel-2d",
        "hlsfactory_syr2k",
    }
    if missing_cosim != expected_missing_cosim:
        raise RuntimeError(
            "unexpected COSIM reference coverage: "
            f"missing={sorted(missing_cosim)}"
        )
    print(json.dumps({
        key: manifest[key]
        for key in (
            "benchmark_count", "arm_count", "grid_cell_count",
            "observed_cell_count", "valid_cell_count", "reference_coverage",
        )
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
