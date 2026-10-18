#!/usr/bin/env python3
"""Generate the checksum-bound July-30 evaluation views.

The source bundle contains three different evidence levels:

* compact DeepSeek/Grok CSVs with reported RTL co-simulation cycles;
* a full Claude Opus 4.8 C-simulation/HLS-synthesis schema; and
* a compact HLSmith/ChatHLS RTL co-simulation comparison.

The generator deliberately keeps these levels separate.  In particular, it
does not interpret a successful Opus C simulation as output equivalence, and
it excludes Grok rows with asymmetric post-hoc selection from the primary
aggregate.  Every paper-facing value is recomputed from the pinned inputs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


EXPECTED_HASHES = {
    "deepseek_v4_flash_flash_cosim_skills_vs_noskills.csv":
        "8f5c5ef4fe85565f43b551f73a7427c44496a48f8d5983715edd35ebffad61a5",
    "deepseek_v4_flash_flash_csynth_skills_vs_noskills.csv":
        "63883c68258340376b46406a341b4abd981357c2a80a826291f1d55c8630efec",
    "grok45_flash_cosim_skills_vs_noskills.csv":
        "6f84e3cb8df659752a432932d3ae5d2bd142f198a7aff71e9e2b1a6f5b9e83fa",
    "grok45_flash_csynth_skills_vs_noskills.csv":
        "975ff6c8e4d8b1968399d1909fa264b6f456b06b773713466c4fd9dc953b149c",
    "newskills_opus_csynth_table.csv":
        "4e923a9e09f4a8385d1f14ee52553a40ec209fb09794794d5f7329236869811b",
    "newskills_opus_csynth_table.md":
        "8ec9bb591d6765ad81d293bbbaa2e11c8ff1d5b094f54faeea8ac48a0dd34d11",
    "schema_records_newskills_opus.jsonl":
        "c83745a19bf1b94d8b24408f69d00eb3dbf273528b212e4340cddb030cd590ff",
    "u280_hlsmith_vs_chathls_cosim_cycles_20260722.csv":
        "465d1cc3acc86053f244ff34fd7fddac1c09c60ac0cb22b2b28fdeda051bd9d7",
}

DEEPSEEK_FILE = "deepseek_v4_flash_flash_cosim_skills_vs_noskills.csv"
DEEPSEEK_CSYNTH_FILE = "deepseek_v4_flash_flash_csynth_skills_vs_noskills.csv"
GROK_FILE = "grok45_flash_cosim_skills_vs_noskills.csv"
GROK_CSYNTH_FILE = "grok45_flash_csynth_skills_vs_noskills.csv"
OPUS_SCHEMA_FILE = "schema_records_newskills_opus.jsonl"
OPUS_CSV_FILE = "newskills_opus_csynth_table.csv"
OPUS_MD_FILE = "newskills_opus_csynth_table.md"
CHATHLS_FILE = "u280_hlsmith_vs_chathls_cosim_cycles_20260722.csv"

COSIM_FIELDS = (
    "bench",
    "set",
    "skills_cycles",
    "skills_pass",
    "skills_source",
    "noskills_cycles",
    "noskills_pass",
    "noskills_source",
    "speedup_noskills_over_skills",
)

OPUS_ORIGINS = {
    "enh_newskills__opus__oneshot__no_skills": "oneshot",
    "enh_newskills__opus__flash__no_skills": "flash_noskills",
    "enh_newskills__opus__flash__curated": "flash_curated",
    "enh_newskills__opus__flash__all_positive": "flash_allpos",
    "enh_newskills__opus__multistep__no_skills": "ms_noskills",
    "enh_newskills__opus__multistep__curated": "ms_curated",
    "enh_newskills__opus__multistep__all_positive": "ms_allpos",
}

OPUS_ARM_ORDER = (
    "oneshot",
    "flash_noskills",
    "flash_curated",
    "flash_allpos",
    "ms_noskills",
    "ms_curated",
    "ms_allpos",
)

OPUS_AGENTIC_ARMS = OPUS_ARM_ORDER[1:]
OPUS_OFF_ARMS = ("flash_noskills", "ms_noskills")

OPUS_ARM_LABELS = {
    "oneshot": ("One-shot", "Expertise disabled"),
    "flash_noskills": ("Single-step", "Expertise disabled"),
    "flash_curated": ("Single-step", "Selected expertise subset"),
    "flash_allpos": ("Single-step", "All applicable positive recipes"),
    "ms_noskills": ("Multi-step", "Expertise disabled"),
    "ms_curated": ("Multi-step", "Selected expertise subset"),
    "ms_allpos": ("Multi-step", "All applicable positive recipes"),
}

OPUS_COLORS = {
    "flash_noskills": "#777777",
    "flash_curated": "#7B61A8",
    "flash_allpos": "#2A9D8F",
    "ms_noskills": "#777777",
    "ms_curated": "#6A4C93",
    "ms_allpos": "#277DA1",
}

OPUS_HATCHES = {
    "flash_noskills": "//",
    "flash_curated": "..",
    "flash_allpos": "\\\\",
    "ms_noskills": "xx",
    "ms_curated": "oo",
    "ms_allpos": "++",
}

CANONICAL_28 = (
    "atax",
    "mvt",
    "2mm",
    "3mm",
    "heat-3d",
    "gemm",
    "floyd-warshall",
    "bicg",
    "doitgen",
    "fdtd-2d",
    "cholesky",
    "correlation",
    "covariance",
    "durbin",
    "gemver",
    "gesummv",
    "gramschmidt",
    "jacobi-1d",
    "jacobi-2d",
    "lu",
    "ludcmp",
    "nussinov",
    "seidel-2d",
    "symm",
    "syr2k",
    "syrk",
    "trisolv",
    "trmm",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_hash(path: Path) -> str:
    expected = EXPECTED_HASHES[path.name]
    observed = sha256(path)
    if observed != expected:
        raise ValueError(
            f"{path.name}: SHA-256 mismatch; expected {expected}, observed {observed}"
        )
    return observed


def geometric_mean(values: Iterable[float]) -> float:
    values = list(values)
    if not values or any(value <= 0 or not math.isfinite(value) for value in values):
        raise ValueError(f"geometric mean requires finite positive values: {values}")
    return math.exp(math.fsum(math.log(value) for value in values) / len(values))


def parse_bool(value: str) -> bool:
    if value == "True":
        return True
    if value == "False":
        return False
    raise ValueError(f"expected True/False, observed {value!r}")


def parse_optional_positive_int(value: str) -> int | None:
    if value == "":
        return None
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"cycle count must be positive: {parsed}")
    return parsed


def latex_escape(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in value)


def format_cycles(value: float) -> str:
    return f"{value:,.0f}"


def ratio_tick(exponent: int) -> str:
    if exponent == 0:
        return r"$1\times$"
    if exponent > 0:
        return rf"${2 ** exponent:g}\times$"
    return rf"$1/{2 ** (-exponent):g}\times$"


@dataclass(frozen=True)
class CosimRow:
    bench: str
    group: str
    enabled_cycles: int | None
    enabled_pass: bool
    enabled_source: str
    disabled_cycles: int | None
    disabled_pass: bool
    disabled_source: str
    ratio: float | None


def read_cosim_csv(path: Path) -> list[CosimRow]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != COSIM_FIELDS:
            raise ValueError(f"{path.name}: unexpected columns {reader.fieldnames}")
        raw_rows = list(reader)
    if len(raw_rows) != 28:
        raise ValueError(f"{path.name}: expected 28 rows, observed {len(raw_rows)}")
    if [row["bench"] for row in raw_rows] != list(CANONICAL_28):
        # The Grok file uses the same cohorts but a different within-cohort
        # ordering.  Require the same complete benchmark set.
        if set(row["bench"] for row in raw_rows) != set(CANONICAL_28):
            raise ValueError(f"{path.name}: unexpected benchmark set")
    if len({row["bench"] for row in raw_rows}) != 28:
        raise ValueError(f"{path.name}: duplicate benchmark row")

    result: list[CosimRow] = []
    for raw in raw_rows:
        enabled_pass = parse_bool(raw["skills_pass"])
        disabled_pass = parse_bool(raw["noskills_pass"])
        enabled_cycles = parse_optional_positive_int(raw["skills_cycles"])
        disabled_cycles = parse_optional_positive_int(raw["noskills_cycles"])
        # A failed row may retain a cycle field from a non-passing replay.
        # Never use that value in a paired latency comparison.  A passing row,
        # however, must provide a positive cycle count.
        if enabled_pass and enabled_cycles is None:
            raise ValueError(f"{path.name}:{raw['bench']}: enabled pass lacks cycles")
        if disabled_pass and disabled_cycles is None:
            raise ValueError(f"{path.name}:{raw['bench']}: disabled pass lacks cycles")
        ratio = None
        if enabled_pass and disabled_pass:
            ratio = disabled_cycles / enabled_cycles
            supplied = float(raw["speedup_noskills_over_skills"])
            if not math.isclose(ratio, supplied, rel_tol=2e-6, abs_tol=2e-6):
                raise ValueError(
                    f"{path.name}:{raw['bench']}: supplied ratio {supplied} != {ratio}"
                )
        elif raw["speedup_noskills_over_skills"]:
            raise ValueError(f"{path.name}:{raw['bench']}: ratio supplied for failed pair")
        result.append(
            CosimRow(
                bench=raw["bench"],
                group=raw["set"],
                enabled_cycles=enabled_cycles,
                enabled_pass=enabled_pass,
                enabled_source=raw["skills_source"],
                disabled_cycles=disabled_cycles,
                disabled_pass=disabled_pass,
                disabled_source=raw["noskills_source"],
                ratio=ratio,
            )
        )
    return result


def cross_check_csynth_cosim(path: Path, rows: list[CosimRow]) -> None:
    """Verify that companion synthesis exports carry the same RTL outcomes."""
    def undecorate(source: str) -> str:
        return source[5:-1] if source.startswith("best(") and source.endswith(")") else source

    required = {
        "bench",
        "set",
        "skills_source",
        "noskills_source",
        "skills_flash_cosim_cycles",
        "noskills_flash_cosim_cycles",
        "skills_flash_pass",
        "noskills_flash_pass",
    }
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError(f"{path.name}: missing companion co-simulation columns")
        raw_rows = {raw["bench"]: raw for raw in reader}
    if set(raw_rows) != {row.bench for row in rows} or len(raw_rows) != 28:
        raise ValueError(f"{path.name}: companion benchmark coverage mismatch")
    for row in rows:
        raw = raw_rows[row.bench]
        if raw["set"] != row.group:
            raise ValueError(f"{path.name}:{row.bench}: cohort mismatch")
        if parse_bool(raw["skills_flash_pass"]) != row.enabled_pass:
            raise ValueError(f"{path.name}:{row.bench}: enabled status mismatch")
        if parse_bool(raw["noskills_flash_pass"]) != row.disabled_pass:
            raise ValueError(f"{path.name}:{row.bench}: disabled status mismatch")
        if parse_optional_positive_int(
            raw["skills_flash_cosim_cycles"]
        ) != row.enabled_cycles:
            raise ValueError(f"{path.name}:{row.bench}: enabled cycles mismatch")
        if parse_optional_positive_int(
            raw["noskills_flash_cosim_cycles"]
        ) != row.disabled_cycles:
            raise ValueError(f"{path.name}:{row.bench}: disabled cycles mismatch")
        if raw["skills_source"] != undecorate(row.enabled_source):
            raise ValueError(f"{path.name}:{row.bench}: enabled source mismatch")
        if raw["noskills_source"] != undecorate(row.disabled_source):
            raise ValueError(f"{path.name}:{row.bench}: disabled source mismatch")


def cosim_summary(rows: list[CosimRow], include: set[str] | None = None) -> dict[str, Any]:
    selected = [row for row in rows if include is None or row.bench in include]
    paired = [row for row in selected if row.ratio is not None]
    ratios = [row.ratio for row in paired]
    enabled_cycles = [row.enabled_cycles for row in paired]
    disabled_cycles = [row.disabled_cycles for row in paired]
    assert all(value is not None for value in enabled_cycles + disabled_cycles)
    wins = sum(ratio > 1.0 for ratio in ratios)
    ties = sum(math.isclose(ratio, 1.0, rel_tol=1e-12) for ratio in ratios)
    losses = len(ratios) - wins - ties
    return {
        "enabled_pass": sum(row.enabled_pass for row in selected),
        "disabled_pass": sum(row.disabled_pass for row in selected),
        "denominator": len(selected),
        "paired": len(paired),
        "enabled_gmean_cycles": geometric_mean(enabled_cycles),
        "disabled_gmean_cycles": geometric_mean(disabled_cycles),
        "paired_reduction": geometric_mean(ratios),
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "median_reduction": sorted(ratios)[len(ratios) // 2],
    }


def baseline_cosim_summary(
    rows: list[CosimRow],
    baseline_cycles: dict[str, int],
    include: set[str] | None = None,
) -> dict[str, Any]:
    """Summarize rows with an executed baseline and two passing RTL arms."""
    selected = [
        row for row in rows
        if row.bench in baseline_cycles
        and row.ratio is not None
        and (include is None or row.bench in include)
    ]
    baseline_ratios = [
        baseline_cycles[row.bench] / row.enabled_cycles for row in selected
    ]
    expertise_ratios = [row.ratio for row in selected]
    assert all(row.enabled_cycles is not None for row in selected)
    return {
        "coverage": len(selected),
        "baseline_reduction": geometric_mean(baseline_ratios),
        "expertise_reduction": geometric_mean(expertise_ratios),
        "wins": sum(ratio > 1.0 for ratio in expertise_ratios),
        "losses": sum(ratio < 1.0 for ratio in expertise_ratios),
    }


def enabled_cross_model_summary(
    deepseek: list[CosimRow],
    grok: list[CosimRow],
    baseline_cycles: dict[str, int],
) -> dict[str, Any]:
    """Compare enabled RTL arms on their baseline-available primary cohort."""
    deep_by_bench = {row.bench: row for row in deepseek}
    grok_by_bench = {row.bench: row for row in grok}
    benches = [
        bench for bench in CANONICAL_28
        if bench in baseline_cycles
        and deep_by_bench[bench].enabled_pass
        and grok_by_bench[bench].enabled_pass
        and grok_by_bench[bench].group != "next5_overlap"
    ]
    ratios = [
        deep_by_bench[bench].enabled_cycles / grok_by_bench[bench].enabled_cycles
        for bench in benches
    ]
    assert all(value is not None for value in ratios)
    summary = {
        "coverage": len(benches),
        "grok_over_deepseek": geometric_mean(ratios),
        "benches": benches,
    }
    if summary["coverage"] != 18 or not math.isclose(
        summary["grok_over_deepseek"], 1.6569, rel_tol=2e-4
    ):
        raise ValueError(f"unexpected Grok/DeepSeek common-cohort result: {summary}")
    return summary


@dataclass(frozen=True)
class OpusCell:
    bench: str
    arm: str
    cycles: int
    estimated_period: float
    target_period: float
    resource_fit: bool
    csim_pass: bool
    synth_pass: bool

    @property
    def timing_fit(self) -> bool:
        return self.estimated_period <= self.target_period + 1e-12

    @property
    def feasible(self) -> bool:
        return self.csim_pass and self.synth_pass and self.timing_fit and self.resource_fit


def read_opus_schema(path: Path, source_csv: Path) -> dict[str, Any]:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if len(records) != 442:
        raise ValueError(f"{path.name}: expected 442 records, observed {len(records)}")
    report_counts = Counter(record["report_type"] for record in records)
    if report_counts != Counter({"hls_synth": 208, "sw_run": 208, "rtl_sim": 26}):
        raise ValueError(f"{path.name}: unexpected report counts {report_counts}")

    synth_records: dict[tuple[str, str], dict[str, Any]] = {}
    csim_records: dict[tuple[str, str], dict[str, Any]] = {}
    rtl_candidate_count = 0
    baseline_rtl_records: dict[str, dict[str, Any]] = {}
    model_labels: set[str] = set()
    for record in records:
        bench = record["problem"]["group_path"][0]
        implementation = record["implementation"]
        origin = implementation.get("origin_version") or ""
        report_type = record["report_type"]
        if origin:
            if origin not in OPUS_ORIGINS:
                raise ValueError(f"{path.name}: unknown candidate origin {origin}")
            arm = OPUS_ORIGINS[origin]
            model = implementation.get("origin_meta", {}).get("model")
            if model:
                model_labels.add(model)
        else:
            arm = "baseline"
        key = (bench, arm)
        if report_type == "hls_synth":
            if key in synth_records:
                raise ValueError(f"{path.name}: duplicate synthesis cell {key}")
            synth_records[key] = record
        elif report_type == "sw_run":
            if key in csim_records:
                raise ValueError(f"{path.name}: duplicate C-simulation cell {key}")
            csim_records[key] = record
        elif report_type == "rtl_sim":
            if arm != "baseline":
                rtl_candidate_count += 1
            else:
                if bench in baseline_rtl_records:
                    raise ValueError(f"{path.name}: duplicate baseline RTL row {bench}")
                baseline_rtl_records[bench] = record

    if model_labels != {"claude-opus-4-8"}:
        raise ValueError(f"{path.name}: unexpected model labels {model_labels}")
    if rtl_candidate_count != 0:
        raise ValueError(f"{path.name}: generated candidates unexpectedly have RTL records")

    benches = sorted(
        bench for bench, arm in synth_records if arm == "baseline"
    )
    if len(benches) != 26 or set(benches) != set(CANONICAL_28) - {"heat-3d", "seidel-2d"}:
        raise ValueError(f"{path.name}: unexpected 26-kernel set")
    if set(baseline_rtl_records) != set(benches):
        raise ValueError(f"{path.name}: baseline RTL coverage differs from synthesis")

    baseline_cycles: dict[str, int] = {}
    baseline_rtl_cycles: dict[str, int] = {}
    cells: dict[tuple[str, str], OpusCell] = {}
    flow_targets: set[str] = set()
    parts: set[str] = set()
    versions: set[str] = set()
    target_periods: set[float] = set()
    for bench in benches:
        baseline = synth_records[(bench, "baseline")]
        baseline_hls = baseline["hls_synth"]
        baseline_cycles[bench] = int(
            baseline_hls["PerformanceEstimates"]["SummaryOfOverallLatency"][
                "Worst-caseLatency"
            ]
        )
        baseline_rtl = baseline_rtl_records[bench]["rtl_sim"]
        if baseline_rtl["status"] != "pass":
            raise ValueError(f"{path.name}:{bench}: baseline RTL did not pass")
        baseline_rtl_cycles[bench] = int(baseline_rtl["kernel_runtime_cycles"])
        if baseline_rtl_cycles[bench] <= 0:
            raise ValueError(f"{path.name}:{bench}: invalid baseline RTL cycles")
        for arm in OPUS_ARM_ORDER:
            key = (bench, arm)
            if key not in synth_records or key not in csim_records:
                raise ValueError(f"{path.name}: missing candidate cell {key}")
            synth = synth_records[key]
            csim = csim_records[key]
            hls = synth["hls_synth"]
            assignments = hls["UserAssignments"]
            flow_targets.add(assignments["FlowTarget"])
            parts.add(assignments["Part"])
            versions.add(synth["run"]["vitis_version"])
            target_period = float(assignments["TargetClockPeriod"])
            target_periods.add(target_period)
            timing = hls["PerformanceEstimates"]["SummaryOfTimingAnalysis"]
            latency = hls["PerformanceEstimates"]["SummaryOfOverallLatency"]
            resources = hls["AreaEstimates"]["Resources"]
            available = hls["AreaEstimates"]["AvailableResources"]
            resource_fit = all(
                int(resources[name]) <= int(available[name])
                for name in ("BRAM_18K", "DSP", "FF", "LUT", "URAM")
            )
            cells[key] = OpusCell(
                bench=bench,
                arm=arm,
                cycles=int(latency["Worst-caseLatency"]),
                estimated_period=float(timing["EstimatedClockPeriod"]),
                target_period=target_period,
                resource_fit=resource_fit,
                csim_pass=csim["sw_run"]["status"] == "pass",
                synth_pass=hls["status"] == "pass",
            )

    if flow_targets != {"vitis"}:
        raise ValueError(f"{path.name}: expected Vitis flow, observed {flow_targets}")
    if parts != {"xcu280-fsvh2892-2L-e"}:
        raise ValueError(f"{path.name}: unexpected parts {parts}")
    if versions != {"2023.2"} or target_periods != {3.33}:
        raise ValueError(
            f"{path.name}: unexpected tool/clock {versions}, {target_periods}"
        )

    # Cross-check every cycle in the compact table.  The raw schema remains
    # authoritative for timing and resource qualification.
    with source_csv.open(newline="", encoding="utf-8") as stream:
        compact_rows = list(csv.DictReader(stream))
    expected_fields = (
        "bench",
        "gold",
        "oneshot",
        "flash_noskills",
        "flash_curated",
        "flash_allpos",
        "ms_noskills",
        "ms_curated",
        "ms_allpos",
    )
    if tuple(compact_rows[0].keys()) != expected_fields or len(compact_rows) != 26:
        raise ValueError(f"{source_csv.name}: unexpected compact table shape")
    for raw in compact_rows:
        bench = raw["bench"]
        if int(raw["gold"]) != baseline_cycles[bench]:
            raise ValueError(f"{source_csv.name}:{bench}: baseline mismatch")
        for arm in OPUS_ARM_ORDER:
            if int(raw[arm]) != cells[(bench, arm)].cycles:
                raise ValueError(f"{source_csv.name}:{bench}:{arm}: cycle mismatch")

    arm_summary: dict[str, dict[str, Any]] = {}
    baseline_gmean = geometric_mean(baseline_cycles.values())
    for arm in OPUS_ARM_ORDER:
        arm_cells = [cells[(bench, arm)] for bench in benches]
        gmean_cycles = geometric_mean(cell.cycles for cell in arm_cells)
        arm_summary[arm] = {
            "tool_complete": sum(cell.csim_pass and cell.synth_pass for cell in arm_cells),
            "resource_fit": sum(cell.resource_fit for cell in arm_cells),
            "timing_resource_feasible": sum(cell.feasible for cell in arm_cells),
            "gmean_cycles": gmean_cycles,
            "baseline_reduction": baseline_gmean / gmean_cycles,
        }

    selections: list[dict[str, Any]] = []
    for bench in sorted(benches, key=lambda value: (not value[0].isdigit(), value)):
        eligible = [
            cells[(bench, arm)] for arm in OPUS_AGENTIC_ARMS
            if cells[(bench, arm)].feasible
        ]
        eligible_off = [
            cells[(bench, arm)] for arm in OPUS_OFF_ARMS
            if cells[(bench, arm)].feasible
        ]
        if not eligible or not eligible_off:
            raise ValueError(f"{path.name}:{bench}: no feasible agentic/off candidate")
        best = min(eligible, key=lambda cell: (cell.cycles, OPUS_AGENTIC_ARMS.index(cell.arm)))
        best_off = min(
            eligible_off, key=lambda cell: (cell.cycles, OPUS_OFF_ARMS.index(cell.arm))
        )
        base = baseline_cycles[bench]
        selections.append(
            {
                "bench": bench,
                "baseline_cycles": base,
                "best_arm": best.arm,
                "best_cycles": best.cycles,
                "best_baseline_reduction": base / best.cycles,
                "best_off_arm": best_off.arm,
                "best_off_cycles": best_off.cycles,
                "best_off_baseline_reduction": base / best_off.cycles,
                "best_over_off": best_off.cycles / best.cycles,
            }
        )

    return {
        "benches": benches,
        "baseline_cycles": baseline_cycles,
        "baseline_rtl_cycles": baseline_rtl_cycles,
        "baseline_gmean_cycles": baseline_gmean,
        "cells": cells,
        "arm_summary": arm_summary,
        "selections": selections,
        "best_arm_counts": Counter(row["best_arm"] for row in selections),
        "best_gmean_reduction": geometric_mean(
            row["best_baseline_reduction"] for row in selections
        ),
        "best_off_gmean_reduction": geometric_mean(
            row["best_off_baseline_reduction"] for row in selections
        ),
        "best_over_off_gmean": geometric_mean(
            row["best_over_off"] for row in selections
        ),
        "best_guided_count": sum(
            row["best_arm"] not in OPUS_OFF_ARMS for row in selections
        ),
        "best_off_count": sum(
            row["best_arm"] in OPUS_OFF_ARMS for row in selections
        ),
    }


def read_chathls_csv(path: Path) -> dict[str, Any]:
    raw_lines = path.read_text(encoding="utf-8").splitlines()
    if len(raw_lines) != 17:
        raise ValueError(f"{path.name}: expected header, 14 rows, and two trailer lines")
    if raw_lines[-2:] != ["===============", "hlsmith vs chathls cosim cycles deepseek."]:
        raise ValueError(f"{path.name}: unexpected malformed trailer")
    reader = csv.DictReader(raw_lines[:-2])
    expected = (
        "bench",
        "hlsmith_cosim_cycles",
        "chathls_cosim_cycles",
        "chathls_over_hlsmith",
        "hlsmith_status",
        "chathls_status",
        "paired_pass",
    )
    if tuple(reader.fieldnames or ()) != expected:
        raise ValueError(f"{path.name}: unexpected columns")
    rows: list[dict[str, Any]] = []
    for raw in reader:
        hlsmith_pass = raw["hlsmith_status"] == "PASS"
        chathls_pass = raw["chathls_status"] == "PASS"
        paired = raw["paired_pass"] == "yes"
        if paired != (hlsmith_pass and chathls_pass):
            raise ValueError(f"{path.name}:{raw['bench']}: paired status mismatch")
        hlsmith_cycles = int(raw["hlsmith_cosim_cycles"])
        chathls_cycles = int(raw["chathls_cosim_cycles"]) if raw["chathls_cosim_cycles"] else None
        ratio = chathls_cycles / hlsmith_cycles if paired else None
        if paired and not math.isclose(
            ratio, float(raw["chathls_over_hlsmith"]), rel_tol=2e-6
        ):
            raise ValueError(f"{path.name}:{raw['bench']}: ratio mismatch")
        rows.append(
            {
                "bench": raw["bench"],
                "hlsmith_cycles": hlsmith_cycles,
                "chathls_cycles": chathls_cycles,
                "hlsmith_pass": hlsmith_pass,
                "chathls_pass": chathls_pass,
                "ratio": ratio,
            }
        )
    if len(rows) != 14 or len({row["bench"] for row in rows}) != 14:
        raise ValueError(f"{path.name}: expected 14 unique benchmark rows")
    paired_ratios = [row["ratio"] for row in rows if row["ratio"] is not None]
    without_syr2k = [
        row["ratio"] for row in rows
        if row["ratio"] is not None and row["bench"] != "kernel_syr2k"
    ]
    return {
        "rows": rows,
        "hlsmith_pass": sum(row["hlsmith_pass"] for row in rows),
        "chathls_pass": sum(row["chathls_pass"] for row in rows),
        "hlsmith_only_pass": sum(
            row["hlsmith_pass"] and not row["chathls_pass"] for row in rows
        ),
        "chathls_failure_count": sum(not row["chathls_pass"] for row in rows),
        "paired": len(paired_ratios),
        "paired_reduction": geometric_mean(paired_ratios),
        "without_syr2k_count": len(without_syr2k),
        "without_syr2k_reduction": geometric_mean(without_syr2k),
        "malformed_trailer_ignored": raw_lines[-2:],
    }


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 6.8,
            "axes.titlesize": 7.4,
            "axes.labelsize": 7.0,
            "xtick.labelsize": 5.2,
            "ytick.labelsize": 6.0,
            "legend.fontsize": 5.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.55,
            "grid.linewidth": 0.35,
        }
    )


def render_cosim_figure(
    deepseek: list[CosimRow], grok: list[CosimRow], output: Path
) -> None:
    configure_matplotlib()
    order = list(CANONICAL_28)
    row_maps = {
        "DeepSeek V4 Flash (bundle label)": {row.bench: row for row in deepseek},
        "Grok 4.5 (bundle label)": {row.bench: row for row in grok},
    }
    fig, axes = plt.subplots(
        2, 1, figsize=(7.08, 3.65), sharex=True,
        gridspec_kw={"hspace": 0.16, "top": 0.86, "bottom": 0.23, "left": 0.07, "right": 0.995}
    )
    positive = "#238B45"
    negative = "#D95F02"
    x_positions = list(range(len(order)))
    for axis, (title, row_map) in zip(axes, row_maps.items()):
        for index, bench in enumerate(order):
            row = row_map[bench]
            if row.ratio is not None:
                height = math.log2(row.ratio)
                asymmetric = title.startswith("Grok") and row.group == "next5_overlap"
                axis.bar(
                    index,
                    height,
                    width=0.74,
                    color=positive if height >= 0 else negative,
                    edgecolor="0.22",
                    linewidth=0.35,
                    hatch="////" if asymmetric else "",
                    alpha=0.45 if asymmetric else 0.92,
                    zorder=3,
                )
            else:
                if row.enabled_pass:
                    label = "disabled arm fails"
                    color = positive
                else:
                    label = "enabled arm fails"
                    color = negative
                axis.scatter(
                    index, 9.65, marker="x", s=16, linewidths=0.8,
                    color=color, zorder=5, clip_on=False
                )
                axis.text(
                    index, 9.1, "D" if row.enabled_pass else "E",
                    ha="center", va="top", fontsize=4.5, color=color,
                )
        axis.axhline(0, color="0.15", linewidth=0.7, zorder=4)
        axis.axvline(4.5, color="0.55", linestyle="--", linewidth=0.45)
        axis.axvline(9.5, color="0.55", linestyle="--", linewidth=0.45)
        axis.grid(axis="y", color="0.87", zorder=0)
        axis.set_ylim(-7.0, 10.0)
        ticks = (-6, -3, 0, 3, 6, 9)
        axis.set_yticks(ticks, [ratio_tick(value) for value in ticks])
        axis.set_ylabel("Disabled / enabled\nRTL cycles")
        axis.set_title(title, loc="left", pad=2.0, fontweight="bold")
        axis.text(
            2.0, 9.55, "preselected top five", ha="center", va="top",
            fontsize=5.0, color="0.35"
        )
        axis.text(
            7.0, 9.55, "next five", ha="center", va="top",
            fontsize=5.0, color="0.35"
        )
        axis.text(
            18.5, 9.55, "remaining kernels", ha="center", va="top",
            fontsize=5.0, color="0.35"
        )

    axes[-1].set_xticks(x_positions)
    axes[-1].set_xticklabels(
        [bench.upper() if bench in {"atax", "mvt"} else bench for bench in order],
        rotation=58,
        ha="right",
        rotation_mode="anchor",
    )
    axes[-1].set_xlabel("HLSFactory / PolyBench configuration")
    legend_handles = [
        Patch(facecolor=positive, edgecolor="0.22", label="Architectural expertise reduces cycles"),
        Patch(facecolor=negative, edgecolor="0.22", label="Architectural expertise increases cycles"),
        Patch(
            facecolor="0.78", edgecolor="0.22", hatch="////", alpha=0.55,
            label="Asymmetric Grok repeat selection (excluded from primary aggregate)",
        ),
        Line2D(
            [], [], marker="x", linestyle="None", color="0.35",
            label="One arm fails; E/D identifies enabled/disabled arm",
        ),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.52, 0.995),
        columnspacing=1.2,
        handlelength=1.7,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output,
        bbox_inches="tight",
        pad_inches=0.015,
        metadata={
            "Title": "Architectural-expertise sensitivity with reported RTL co-simulation",
            "Author": "Anonymous",
            "Creator": "generate_july30_results.py",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    plt.close(fig)


def render_opus_figure(opus: dict[str, Any], output: Path) -> None:
    configure_matplotlib()
    rows = opus["selections"]
    fig, axis = plt.subplots(figsize=(7.08, 2.34))
    fig.subplots_adjust(left=0.075, right=0.995, top=0.67, bottom=0.29)
    positions = list(range(len(rows)))
    for index, row in enumerate(rows):
        arm = row["best_arm"]
        value = row["best_baseline_reduction"]
        axis.bar(
            index,
            value,
            width=0.73,
            color=OPUS_COLORS[arm],
            hatch=OPUS_HATCHES[arm],
            edgecolor="0.20",
            linewidth=0.35,
            zorder=3,
        )
        axis.text(
            index,
            min(value * 1.09, 430),
            f"{value:.1f}$\\times$",
            rotation=64,
            ha="left",
            va="bottom",
            fontsize=4.6,
            color="0.20",
            clip_on=True,
        )
    axis.scatter(
        positions,
        [row["best_off_baseline_reduction"] for row in rows],
        marker="D",
        facecolors="white",
        edgecolors="black",
        linewidths=0.6,
        s=13,
        zorder=5,
        label="Best expertise-disabled setup",
    )
    axis.axhline(1.0, color="0.15", linewidth=0.7, zorder=4)
    axis.set_yscale("log", base=2)
    axis.set_ylim(0.5, 512)
    ticks = (0.5, 1, 2, 8, 32, 128, 512)
    axis.set_yticks(ticks, [f"{tick:g}$\\times$" for tick in ticks])
    axis.grid(axis="y", color="0.87", linewidth=0.35, zorder=0)
    axis.set_ylabel("Stored-baseline /\nselected HLS cycles")
    axis.set_xticks(positions)
    axis.set_xticklabels(
        [row["bench"] for row in rows],
        rotation=58,
        ha="right",
        rotation_mode="anchor",
    )
    fig.suptitle(
        "HLSmith with Claude Opus 4.8 and exhaustive six-setup selection",
        y=0.755,
        fontsize=7.6,
        fontweight="bold",
    )
    selected_arms = [arm for arm in OPUS_AGENTIC_ARMS if opus["best_arm_counts"][arm]]
    legend_handles = [
        Patch(
            facecolor=OPUS_COLORS[arm],
            edgecolor="0.20",
            hatch=OPUS_HATCHES[arm],
            label=" / ".join(OPUS_ARM_LABELS[arm]),
        )
        for arm in selected_arms
    ]
    legend_handles.append(
        Line2D(
            [], [], marker="D", markerfacecolor="white", markeredgecolor="black",
            linestyle="None", markersize=4.2, label="Best expertise-disabled setup",
        )
    )
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=3,
        frameon=False,
        columnspacing=0.9,
        handlelength=1.55,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output,
        bbox_inches="tight",
        pad_inches=0.015,
        metadata={
            "Title": "Claude Opus 4.8 exhaustive setup-selection synthesis snapshot",
            "Author": "Anonymous",
            "Creator": "generate_july30_results.py",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    plt.close(fig)


def render_chathls_figure(chathls: dict[str, Any], output: Path) -> None:
    configure_matplotlib()
    rows = chathls["rows"]

    def compact_cycles(value: int) -> str:
        if value >= 1_000_000:
            return f"{value / 1_000_000:.3g}M"
        if value >= 1_000:
            return f"{value / 1_000:.3g}K"
        return str(value)

    positions = list(range(len(rows)))
    width = 0.34
    fig, axis = plt.subplots(figsize=(3.45, 1.17))
    fig.subplots_adjust(left=0.17, right=0.995, top=0.80, bottom=0.34)

    axis.bar(
        [position - width / 2 for position in positions],
        [row["hlsmith_cycles"] for row in rows],
        width=width,
        color="#277DA1",
        edgecolor="0.18",
        linewidth=0.35,
        label="HLSmith",
        zorder=3,
    )
    for position, row in zip(positions, rows):
        chat_position = position + width / 2
        if row["chathls_pass"]:
            axis.bar(
                chat_position,
                row["chathls_cycles"],
                width=width,
                color="#F4A261",
                edgecolor="0.18",
                linewidth=0.35,
                zorder=3,
            )
            axis.text(
                position,
                max(row["hlsmith_cycles"], row["chathls_cycles"]) * 1.15,
                f"{row['ratio']:.3g}$\\times$",
                ha="center",
                va="bottom",
                fontsize=3.7,
                color="0.15",
                clip_on=True,
                zorder=5,
            )
        else:
            axis.text(
                position - width / 2,
                row["hlsmith_cycles"] * 1.25,
                compact_cycles(row["hlsmith_cycles"]),
                ha="center",
                va="bottom",
                fontsize=3.6,
                color="#18546D",
                clip_on=True,
                zorder=5,
            )
            axis.scatter(
                chat_position,
                7.5e6,
                marker="x",
                s=12,
                linewidths=0.8,
                color="#A23E00",
                zorder=5,
            )
            axis.text(
                chat_position,
                9.2e6,
                "fail",
                ha="center",
                va="bottom",
                fontsize=3.7,
                color="#A23E00",
                clip_on=True,
            )

    axis.set_yscale("log")
    # Leave a full log-scale band above the tallest bar/failure annotation.
    axis.set_ylim(5.0e2, 3.0e7)
    ticks = (1.0e3, 1.0e4, 1.0e5, 1.0e6, 1.0e7)
    axis.set_yticks(ticks, ("1K", "10K", "100K", "1M", "10M"))
    axis.set_xticks(
        positions,
        [row["bench"].replace("kernel_", "").replace("_", "-") for row in rows],
        rotation=52,
        ha="right",
        rotation_mode="anchor",
    )
    axis.tick_params(axis="x", labelsize=4.2, pad=1.0)
    axis.grid(axis="y", which="major", color="0.87", linewidth=0.35, zorder=0)
    axis.set_ylabel("Executed RTL cycles\n(lower is better)", fontsize=5.2, labelpad=2.0)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    fig.legend(
        handles=(
            Patch(facecolor="#277DA1", edgecolor="0.18", label="HLSmith"),
            Patch(facecolor="#F4A261", edgecolor="0.18", label="ChatHLS"),
            Line2D(
                [], [], marker="x", color="#A23E00", linestyle="None",
                markersize=4.2, label="ChatHLS failure",
            ),
        ),
        loc="upper center",
        bbox_to_anchor=(0.57, 0.995),
        ncol=3,
        frameon=False,
        columnspacing=0.65,
        handlelength=1.15,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output,
        bbox_inches="tight",
        pad_inches=0.025,
        metadata={
            "Title": "HLSmith and ChatHLS executed RTL outcomes",
            "Author": "Anonymous",
            "Creator": "generate_july30_results.py",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    plt.close(fig)


def write_cosim_csv(
    path: Path,
    deepseek: list[CosimRow],
    grok: list[CosimRow],
    baseline_rtl_cycles: dict[str, int],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        fieldnames = (
            "model_label",
            "bench",
            "cohort",
            "benchmark_baseline_rtl_cycles",
            "expertise_enabled_cycles",
            "expertise_enabled_pass",
            "benchmark_baseline_over_enabled",
            "expertise_disabled_cycles",
            "expertise_disabled_pass",
            "benchmark_baseline_over_disabled",
            "disabled_over_enabled",
            "primary_aggregate",
            "selection_note",
        )
        writer = csv.DictWriter(
            stream, fieldnames=fieldnames, lineterminator="\n"
        )
        writer.writeheader()
        for model, rows in (("DeepSeek V4 Flash", deepseek), ("Grok 4.5", grok)):
            for row in rows:
                asymmetric = model == "Grok 4.5" and row.group == "next5_overlap"
                baseline = baseline_rtl_cycles.get(row.bench)
                writer.writerow(
                    {
                        "model_label": model,
                        "bench": row.bench,
                        "cohort": row.group,
                        "benchmark_baseline_rtl_cycles": baseline or "",
                        "expertise_enabled_cycles": row.enabled_cycles or "",
                        "expertise_enabled_pass": row.enabled_pass,
                        "benchmark_baseline_over_enabled": (
                            f"{baseline / row.enabled_cycles:.12g}"
                            if baseline is not None and row.enabled_pass else ""
                        ),
                        "expertise_disabled_cycles": row.disabled_cycles or "",
                        "expertise_disabled_pass": row.disabled_pass,
                        "benchmark_baseline_over_disabled": (
                            f"{baseline / row.disabled_cycles:.12g}"
                            if baseline is not None and row.disabled_pass else ""
                        ),
                        "disabled_over_enabled": (
                            f"{row.ratio:.12g}" if row.ratio is not None else ""
                        ),
                        "primary_aggregate": not asymmetric,
                        "selection_note": (
                            "excluded: expertise-enabled cell uses asymmetric post-hoc best"
                            if asymmetric else ""
                        ),
                    }
                )


def write_opus_csv(path: Path, opus: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        fieldnames = (
            "bench",
            "stored_baseline_cycles",
            "selected_setup",
            "selected_cycles",
            "stored_baseline_over_selected",
            "best_expertise_disabled_setup",
            "best_expertise_disabled_cycles",
            "stored_baseline_over_best_expertise_disabled",
            "best_expertise_disabled_over_selected",
        )
        writer = csv.DictWriter(
            stream, fieldnames=fieldnames, lineterminator="\n"
        )
        writer.writeheader()
        for row in opus["selections"]:
            writer.writerow(
                {
                    "bench": row["bench"],
                    "stored_baseline_cycles": row["baseline_cycles"],
                    "selected_setup": row["best_arm"],
                    "selected_cycles": row["best_cycles"],
                    "stored_baseline_over_selected": f"{row['best_baseline_reduction']:.12g}",
                    "best_expertise_disabled_setup": row["best_off_arm"],
                    "best_expertise_disabled_cycles": row["best_off_cycles"],
                    "stored_baseline_over_best_expertise_disabled":
                        f"{row['best_off_baseline_reduction']:.12g}",
                    "best_expertise_disabled_over_selected":
                        f"{row['best_over_off']:.12g}",
                }
            )


def write_chathls_csv(path: Path, chathls: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        fieldnames = (
            "bench",
            "hlsmith_rtl_cycles",
            "hlsmith_pass",
            "chathls_rtl_cycles",
            "chathls_pass",
            "chathls_over_hlsmith",
        )
        writer = csv.DictWriter(
            stream, fieldnames=fieldnames, lineterminator="\n"
        )
        writer.writeheader()
        for row in chathls["rows"]:
            writer.writerow(
                {
                    "bench": row["bench"],
                    "hlsmith_rtl_cycles": row["hlsmith_cycles"],
                    "hlsmith_pass": row["hlsmith_pass"],
                    "chathls_rtl_cycles": row["chathls_cycles"] or "",
                    "chathls_pass": row["chathls_pass"],
                    "chathls_over_hlsmith": (
                        f"{row['ratio']:.12g}" if row["ratio"] is not None else ""
                    ),
                }
            )


def write_tables(
    path: Path,
    deep_summary: dict[str, Any],
    grok_summary: dict[str, Any],
    grok_full: dict[str, Any],
    opus: dict[str, Any],
) -> None:
    lines = [
        "% AUTO-GENERATED by scripts/generate_july30_results.py.",
        r"\begin{table*}[!t]",
        r"  \centering",
        r"  \caption{July-30 model studies at their recorded evidence levels.  "
        r"(a) reports supplied RTL co-simulation status and paired cycles; it is "
        r"not an independent output-value audit.  Grok excludes five rows whose "
        r"expertise-enabled side uses an asymmetric post-hoc best-of-repeat selection.  "
        r"(b) reports Claude Opus~4.8 C-simulation and HLS-synthesis completion; "
        r"completion does not establish output equivalence, and generated-design "
        r"RTL simulation is absent.  Estimated cycles use all 26 tool-complete rows, "
        r"including rows that miss the 3.33-ns target.}",
        r"  \label{tab:july30-model-results}",
        r"  \scriptsize",
        r"  \setlength{\tabcolsep}{2.1pt}",
        r"  \renewcommand{\arraystretch}{0.92}",
        r"  \begin{minipage}[t]{0.455\textwidth}",
        r"  \centering",
        r"  \textbf{(a) Executed RTL co-simulation}\\[-2pt]",
        r"  \begin{tabular}{@{}lrrrrr@{}}",
        r"    \hline",
        r"    \textbf{Model label} & \shortstack{\textbf{Enabled}\\\textbf{pass}} & "
        r"\shortstack{\textbf{Disabled}\\\textbf{pass}} & \textbf{Paired} & "
        r"\shortstack{\textbf{Cycle}\\\textbf{reduction}} & "
        r"\shortstack{\textbf{Enabled}\\\textbf{win/loss}} \\",
        r"    \hline",
        (
            f"    DeepSeek V4 Flash & {deep_summary['enabled_pass']}/28 & "
            f"{deep_summary['disabled_pass']}/28 & {deep_summary['paired']} & "
            f"{deep_summary['paired_reduction']:.2f}$\\times$ & "
            f"{deep_summary['wins']}/{deep_summary['losses']} \\\\"
        ),
        (
            f"    Grok 4.5 & {grok_summary['enabled_pass']}/23 & "
            f"{grok_summary['disabled_pass']}/23 & {grok_summary['paired']} & "
            f"{grok_summary['paired_reduction']:.2f}$\\times$ & "
            f"{grok_summary['wins']}/{grok_summary['losses']} \\\\"
        ),
        r"    \hline",
        r"  \end{tabular}",
        (
            r"  \par\raggedright\fontsize{6.1}{6.6}\selectfont "
            f"Full Grok bundle: 28/28 passes per arm and "
            f"{grok_full['paired_reduction']:.2f}$\\times$; five asymmetric rows "
            r"remain visible in Fig.~\ref{fig:july30-cosim}."
        ),
        r"  \end{minipage}\hfill",
        r"  \begin{minipage}[t]{0.525\textwidth}",
        r"  \centering",
        r"  \textbf{(b) Claude Opus 4.8 synthesis-only sweep}\\[-2pt]",
        r"  \begin{tabular}{@{}llrrr@{}}",
        r"    \hline",
        r"    \textbf{Workflow} & \textbf{Expertise policy} & "
        r"\shortstack{\textbf{Tool}\\\textbf{completion}} & "
        r"\shortstack{\textbf{Timing and}\\\textbf{resource feasible}} & "
        r"\shortstack{\textbf{Geometric-mean}\\\textbf{estimated cycles}} \\",
        r"    \hline",
    ]
    for arm in OPUS_ARM_ORDER:
        workflow, expertise = OPUS_ARM_LABELS[arm]
        summary = opus["arm_summary"][arm]
        lines.append(
            f"    {workflow} & {expertise} & "
            f"{summary['tool_complete']}/26 & "
            f"{summary['timing_resource_feasible']}/26 & "
            f"{format_cycles(summary['gmean_cycles'])} \\\\"
        )
    lines.extend(
        [
            r"    \hline",
            r"  \end{tabular}",
            r"  \end{minipage}",
            r"\end{table*}",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_macros(
    path: Path,
    bundle_hash: str,
    deep_summary: dict[str, Any],
    deep_groups: dict[str, dict[str, Any]],
    grok_summary: dict[str, Any],
    grok_full: dict[str, Any],
    deep_strict: dict[str, Any],
    grok_strict: dict[str, Any],
    cross_model: dict[str, Any],
    opus: dict[str, Any],
    chathls: dict[str, Any],
) -> None:
    values = {
        "JulyThirtyEvidenceHash": bundle_hash[:12],
        "JulyThirtyDeepSeekPairedCount": deep_summary["paired"],
        "JulyThirtyDeepSeekEnabledPass": deep_summary["enabled_pass"],
        "JulyThirtyDeepSeekDisabledPass": deep_summary["disabled_pass"],
        "JulyThirtyDeepSeekGain": f"{deep_summary['paired_reduction']:.2f}",
        "JulyThirtyDeepSeekMedianGain": f"{deep_summary['median_reduction']:.2f}",
        "JulyThirtyDeepSeekWins": deep_summary["wins"],
        "JulyThirtyDeepSeekLosses": deep_summary["losses"],
        "JulyThirtyDeepSeekTopFiveGain": f"{deep_groups['top5']['paired_reduction']:.2f}",
        "JulyThirtyDeepSeekNextFiveGain": f"{deep_groups['next5']['paired_reduction']:.2f}",
        "JulyThirtyDeepSeekRemainderGain":
            f"{deep_groups['rest23_excl']['paired_reduction']:.2f}",
        "JulyThirtyGrokPrimaryCount": grok_summary["paired"],
        "JulyThirtyGrokPrimaryGain": f"{grok_summary['paired_reduction']:.2f}",
        "JulyThirtyGrokPrimaryWins": grok_summary["wins"],
        "JulyThirtyGrokPrimaryLosses": grok_summary["losses"],
        "JulyThirtyGrokFullGain": f"{grok_full['paired_reduction']:.2f}",
        "JulyThirtyDeepSeekStrictCount": deep_strict["coverage"],
        "JulyThirtyDeepSeekBaselineGain":
            f"{deep_strict['baseline_reduction']:.2f}",
        "JulyThirtyDeepSeekStrictExpertiseGain":
            f"{deep_strict['expertise_reduction']:.2f}",
        "JulyThirtyGrokStrictCount": grok_strict["coverage"],
        "JulyThirtyGrokBaselineGain":
            f"{grok_strict['baseline_reduction']:.2f}",
        "JulyThirtyGrokStrictExpertiseGain":
            f"{grok_strict['expertise_reduction']:.2f}",
        "JulyThirtyGrokDeepSeekCommonCount": cross_model["coverage"],
        "JulyThirtyGrokOverDeepSeekGain":
            f"{cross_model['grok_over_deepseek']:.2f}",
        "JulyThirtyOpusKernelCount": len(opus["benches"]),
        "JulyThirtyOpusBestBaselineGain": f"{opus['best_gmean_reduction']:.2f}",
        "JulyThirtyOpusBestOffBaselineGain":
            f"{opus['best_off_gmean_reduction']:.2f}",
        "JulyThirtyOpusBestOverOffGain": f"{opus['best_over_off_gmean']:.2f}",
        "JulyThirtyOpusGuidedSelections": opus["best_guided_count"],
        "JulyThirtyOpusOffSelections": opus["best_off_count"],
        "JulyThirtyOpusFlashSelectedOverOff": (
            f"{opus['arm_summary']['flash_noskills']['gmean_cycles'] / opus['arm_summary']['flash_curated']['gmean_cycles']:.2f}"
        ),
        "JulyThirtyOpusFlashAllOverOff": (
            f"{opus['arm_summary']['flash_noskills']['gmean_cycles'] / opus['arm_summary']['flash_allpos']['gmean_cycles']:.2f}"
        ),
        "JulyThirtyOpusMultiSelectedOverOff": (
            f"{opus['arm_summary']['ms_noskills']['gmean_cycles'] / opus['arm_summary']['ms_curated']['gmean_cycles']:.2f}"
        ),
        "JulyThirtyOpusMultiAllOverOff": (
            f"{opus['arm_summary']['ms_noskills']['gmean_cycles'] / opus['arm_summary']['ms_allpos']['gmean_cycles']:.2f}"
        ),
        "JulyThirtyHLSmithPass": chathls["hlsmith_pass"],
        "JulyThirtyChatHLSPass": chathls["chathls_pass"],
        "JulyThirtyChatHLSDenominator": len(chathls["rows"]),
        "JulyThirtyChatHLSPaired": chathls["paired"],
        "JulyThirtyHLSmithOnlyPass": chathls["hlsmith_only_pass"],
        "JulyThirtyChatHLSFailureCount": chathls["chathls_failure_count"],
        "JulyThirtyChatHLSGain": f"{chathls['paired_reduction']:.2f}",
        "JulyThirtyChatHLSWithoutSyrTwoKCount": chathls["without_syr2k_count"],
        "JulyThirtyChatHLSWithoutSyrTwoKGain":
            f"{chathls['without_syr2k_reduction']:.2f}",
    }
    lines = [
        "% AUTO-GENERATED by scripts/generate_july30_results.py.",
        *[
            f"\\providecommand{{\\{name}}}{{{value}}}"
            for name, value in values.items()
        ],
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def combined_bundle_hash(input_hashes: dict[str, str]) -> str:
    payload = json.dumps(input_hashes, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def write_manifest(
    path: Path,
    input_hashes: dict[str, str],
    outputs: list[Path],
    metrics: dict[str, Any],
) -> None:
    manifest = {
        "schema": "hlsmith.paper.july30-results.v4",
        "bundle_sha256": combined_bundle_hash(input_hashes),
        "inputs": [
            {"source_id": name, "sha256": digest}
            for name, digest in sorted(input_hashes.items())
        ],
        "outputs": [
            {"path": str(output), "sha256": sha256(output)}
            for output in outputs
        ],
        "metrics": metrics,
        "evidence_boundaries": [
            "DeepSeek and Grok pass fields and cycles are reproduced from compact "
            "CSV files; raw generated source, tool reports, exact model revisions, "
            "and output-comparison records are absent.",
            "The companion synthesis tables reproduce every reported co-simulation "
            "cycle, pass flag, cohort, and source identifier.",
            "Five Grok next-five rows use asymmetric post-hoc best selection on "
            "the expertise-enabled arm and are excluded from the primary aggregate.",
            "The Opus schema supplies passing benchmark-baseline RTL cycles for "
            "26 kernels; Heat-3D and Seidel-2D have no baseline RTL record.",
            "Claude Opus candidate records contain C-simulation and HLS-synthesis "
            "completion but no independent output comparison or generated RTL simulation.",
            "The Opus Markdown note claims a Vivado baseline, while the raw schema "
            "records both baseline and candidates as Vitis 2023.2; the schema is used.",
            "The ChatHLS compact CSV has two malformed trailer lines; the generator "
            "requires and ignores exactly those lines and validates the preceding 14 rows.",
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--chathls-csv", type=Path, required=True)
    parser.add_argument(
        "--cosim-figure",
        type=Path,
        default=Path("figures/july30_cosim_expertise.pdf"),
    )
    parser.add_argument(
        "--opus-figure",
        type=Path,
        default=Path("figures/opus_exhaustive_setup.pdf"),
    )
    parser.add_argument(
        "--chathls-figure",
        type=Path,
        default=Path("figures/chathls_cosim_comparison.pdf"),
    )
    parser.add_argument(
        "--tables",
        type=Path,
        default=Path("generated/july30_results_tables.tex"),
    )
    parser.add_argument(
        "--macros",
        type=Path,
        default=Path("generated/july30_results_macros.tex"),
    )
    parser.add_argument(
        "--cosim-data",
        type=Path,
        default=Path("generated/july30_cosim_expertise.csv"),
    )
    parser.add_argument(
        "--opus-data",
        type=Path,
        default=Path("generated/july30_opus_selection.csv"),
    )
    parser.add_argument(
        "--chathls-data",
        type=Path,
        default=Path("generated/july30_chathls_comparison.csv"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("generated/july30_results_manifest.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_paths = {
        name: args.results_dir / name
        for name in (
            DEEPSEEK_FILE,
            DEEPSEEK_CSYNTH_FILE,
            GROK_FILE,
            GROK_CSYNTH_FILE,
            OPUS_SCHEMA_FILE,
            OPUS_CSV_FILE,
            OPUS_MD_FILE,
        )
    }
    input_paths[CHATHLS_FILE] = args.chathls_csv
    for path in input_paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    input_hashes = {name: require_hash(path) for name, path in input_paths.items()}

    deepseek = read_cosim_csv(input_paths[DEEPSEEK_FILE])
    grok = read_cosim_csv(input_paths[GROK_FILE])
    cross_check_csynth_cosim(input_paths[DEEPSEEK_CSYNTH_FILE], deepseek)
    cross_check_csynth_cosim(input_paths[GROK_CSYNTH_FILE], grok)
    opus = read_opus_schema(input_paths[OPUS_SCHEMA_FILE], input_paths[OPUS_CSV_FILE])
    chathls = read_chathls_csv(input_paths[CHATHLS_FILE])

    deep_summary = cosim_summary(deepseek)
    deep_groups = {
        group: cosim_summary(deepseek, {row.bench for row in deepseek if row.group == group})
        for group in ("top5", "next5", "rest23_excl")
    }
    grok_primary_benches = {
        row.bench for row in grok if row.group != "next5_overlap"
    }
    grok_summary = cosim_summary(grok, grok_primary_benches)
    grok_full = cosim_summary(grok)
    baseline_rtl_cycles = opus["baseline_rtl_cycles"]
    deep_strict = baseline_cosim_summary(deepseek, baseline_rtl_cycles)
    grok_strict = baseline_cosim_summary(
        grok, baseline_rtl_cycles, grok_primary_benches
    )
    cross_model = enabled_cross_model_summary(
        deepseek, grok, baseline_rtl_cycles
    )

    render_cosim_figure(deepseek, grok, args.cosim_figure)
    render_opus_figure(opus, args.opus_figure)
    render_chathls_figure(chathls, args.chathls_figure)
    write_cosim_csv(
        args.cosim_data, deepseek, grok, opus["baseline_rtl_cycles"]
    )
    write_opus_csv(args.opus_data, opus)
    write_chathls_csv(args.chathls_data, chathls)
    write_tables(args.tables, deep_summary, grok_summary, grok_full, opus)
    bundle_hash = combined_bundle_hash(input_hashes)
    write_macros(
        args.macros,
        bundle_hash,
        deep_summary,
        deep_groups,
        grok_summary,
        grok_full,
        deep_strict,
        grok_strict,
        cross_model,
        opus,
        chathls,
    )
    outputs = [
        args.cosim_figure,
        args.opus_figure,
        args.chathls_figure,
        args.tables,
        args.macros,
        args.cosim_data,
        args.opus_data,
        args.chathls_data,
    ]
    write_manifest(
        args.manifest,
        input_hashes,
        outputs,
        {
            "deepseek_primary": deep_summary,
            "deepseek_by_supplied_cohort": deep_groups,
            "grok_primary_same_source": grok_summary,
            "grok_full_sensitivity": grok_full,
            "deepseek_strict_baseline_cohort": deep_strict,
            "grok_strict_baseline_cohort": grok_strict,
            "enabled_cross_model_common_cohort": cross_model,
            "opus": {
                "baseline_gmean_cycles": opus["baseline_gmean_cycles"],
                "baseline_rtl_kernel_count": len(opus["baseline_rtl_cycles"]),
                "baseline_rtl_gmean_cycles": geometric_mean(
                    opus["baseline_rtl_cycles"].values()
                ),
                "arm_summary": opus["arm_summary"],
                "best_arm_counts": dict(opus["best_arm_counts"]),
                "best_gmean_reduction": opus["best_gmean_reduction"],
                "best_off_gmean_reduction": opus["best_off_gmean_reduction"],
                "best_over_off_gmean": opus["best_over_off_gmean"],
                "best_guided_count": opus["best_guided_count"],
                "best_off_count": opus["best_off_count"],
            },
            "chathls": {
                key: value for key, value in chathls.items() if key != "rows"
            },
        },
    )
    print(
        json.dumps(
            {
                "bundle_sha256": bundle_hash,
                "deepseek_paired_reduction": deep_summary["paired_reduction"],
                "grok_primary_paired_reduction": grok_summary["paired_reduction"],
                "grok_over_deepseek_common_cohort":
                    cross_model["grok_over_deepseek"],
                "grok_full_paired_reduction": grok_full["paired_reduction"],
                "opus_best_over_off": opus["best_over_off_gmean"],
                "chathls_paired_reduction": chathls["paired_reduction"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
