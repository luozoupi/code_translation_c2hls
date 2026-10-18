#!/usr/bin/env python3
"""Generate the aligned Sonnet/Opus/GPT-5.5/DeepSeek/Grok expertise view.

The five rows share only a kernel order and an improvement direction:

* Sonnet, Opus, and GPT-5.5 report stored-baseline cycles divided by the post-hoc
  best feasible HLS-synthesis estimate across six agentic setups.  A hollow
  diamond marks the better expertise-disabled setup.
* DeepSeek and Grok report executed benchmark-baseline RTL cycles divided by
  expertise-enabled RTL cycles.  Hollow diamonds use the same baseline and
  mark expertise-disabled RTL cycles.  Executed baselines cover 26 kernels.

The script deliberately preserves these different denominators and evidence
levels rather than pooling them into a five-model aggregate.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


EXPECTED_HASHES = {
    "newskills_sonnet_snapshot.csv":
        "5f803841dcd162867283d7c397dd5818784802edd161d8a2007e92bfb5a5c226",
    "july30_opus_selection.csv":
        "8647630e726fa1aa864f1c138e8eace521279c4c72e03366a5c5e70b8b904687",
    "newskills_gpt55_snapshot.csv":
        "3a4899c8fda4d5a2329d5ad087aec6ea6cf6e1a4c74a3c79671780279d8c97aa",
    "july30_cosim_expertise.csv":
        "c2c100b808b956d74c51f16e4a572cf5d63c623e0e7a81733a278b43194e67ca",
}

CANONICAL_28 = (
    "2mm",
    "3mm",
    "atax",
    "bicg",
    "cholesky",
    "correlation",
    "covariance",
    "doitgen",
    "durbin",
    "fdtd-2d",
    "floyd-warshall",
    "gemm",
    "gemver",
    "gesummv",
    "gramschmidt",
    "heat-3d",
    "jacobi-1d",
    "jacobi-2d",
    "lu",
    "ludcmp",
    "mvt",
    "nussinov",
    "seidel-2d",
    "symm",
    "syr2k",
    "syrk",
    "trisolv",
    "trmm",
)

OFF_ARMS = {"flash_noskills", "ms_noskills"}
ARM_LABELS = {
    "flash_noskills": "Single-step / expertise disabled",
    "flash_curated": "Single-step / selected expertise",
    "flash_allpos": "Single-step / all skills",
    "ms_noskills": "Multi-step / expertise disabled",
    "ms_curated": "Multi-step / selected expertise",
    "ms_allpos": "Multi-step / all skills",
}
ARM_POLICY = {
    "flash_noskills": "no_skills",
    "flash_curated": "selected",
    "flash_allpos": "all_skills",
    "ms_noskills": "no_skills",
    "ms_curated": "selected",
    "ms_allpos": "all_skills",
}
ARM_DEPTH = {
    "flash_noskills": "single_step",
    "flash_curated": "single_step",
    "flash_allpos": "single_step",
    "ms_noskills": "multi_step",
    "ms_curated": "multi_step",
    "ms_allpos": "multi_step",
}
POLICY_COLORS = {
    "no_skills": "#8A8A8A",
    "selected": "#CC79A7",
    "all_skills": "#009E73",
}
DEPTH_HATCHES = {
    "single_step": "",
    "multi_step": "///",
}

SONNET_FIELDS = (
    "benchmark",
    "gold_cycles",
    "oneshot_cycles",
    "oneshot_timing_closed",
    "oneshot_resource_fit",
    "flash_noskills_cycles",
    "flash_noskills_timing_closed",
    "flash_noskills_resource_fit",
    "flash_curated_cycles",
    "flash_curated_timing_closed",
    "flash_curated_resource_fit",
    "flash_allpos_cycles",
    "flash_allpos_timing_closed",
    "flash_allpos_resource_fit",
    "ms_noskills_cycles",
    "ms_noskills_timing_closed",
    "ms_noskills_resource_fit",
    "ms_curated_cycles",
    "ms_curated_timing_closed",
    "ms_curated_resource_fit",
    "ms_allpos_cycles",
    "ms_allpos_timing_closed",
    "ms_allpos_resource_fit",
    "best_arm",
    "best_cycles",
    "best_speedup_vs_gold",
    "best_noskill_arm",
    "best_noskill_cycles",
    "noskill_speedup_vs_gold",
    "incremental_gain_over_noskill",
)

OPUS_FIELDS = (
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

GPT55_FIELDS = (
    "benchmark",
    "stored_baseline_cycles",
    "selected_setup",
    "selected_cycles",
    "stored_baseline_over_selected",
    "best_no_skills_setup",
    "best_no_skills_cycles",
    "stored_baseline_over_best_no_skills",
    "best_no_skills_over_selected",
    "partial_setup_grid",
    "missing_agentic_setups",
)

COSIM_FIELDS = (
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
    sequence = list(values)
    if not sequence or any(
        value <= 0 or not math.isfinite(value) for value in sequence
    ):
        raise ValueError(f"invalid geometric-mean input: {sequence}")
    return math.exp(
        math.fsum(math.log(value) for value in sequence) / len(sequence)
    )


def parse_bool(value: str) -> bool:
    if value == "True":
        return True
    if value == "False":
        return False
    raise ValueError(f"expected True/False, observed {value!r}")


def parse_optional_positive_int(value: str) -> int | None:
    if not value:
        return None
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"expected a positive integer, observed {value!r}")
    return parsed


def read_rows(path: Path, expected_fields: tuple[str, ...]) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != expected_fields:
            raise ValueError(f"{path.name}: unexpected columns {reader.fieldnames}")
        return list(reader)


def read_sonnet(path: Path) -> dict[str, Any]:
    raw_rows = read_rows(path, SONNET_FIELDS)
    expected = set(CANONICAL_28) - {"heat-3d", "seidel-2d"}
    if len(raw_rows) != 26 or {row["benchmark"] for row in raw_rows} != expected:
        raise ValueError(f"{path.name}: unexpected benchmark coverage")
    rows: dict[str, dict[str, Any]] = {}
    for raw in raw_rows:
        bench = raw["benchmark"]
        selected = raw["best_arm"]
        off = raw["best_noskill_arm"]
        if selected not in ARM_LABELS or off not in OFF_ARMS:
            raise ValueError(f"{path.name}:{bench}: unexpected selected setup")
        base = float(raw["gold_cycles"])
        cycles = float(raw["best_cycles"])
        off_cycles = float(raw["best_noskill_cycles"])
        ratio = float(raw["best_speedup_vs_gold"])
        off_ratio = float(raw["noskill_speedup_vs_gold"])
        incremental = float(raw["incremental_gain_over_noskill"])
        if not math.isclose(ratio, base / cycles, rel_tol=2e-10):
            raise ValueError(f"{path.name}:{bench}: selected ratio mismatch")
        if not math.isclose(off_ratio, base / off_cycles, rel_tol=2e-10):
            raise ValueError(f"{path.name}:{bench}: expertise-off ratio mismatch")
        if not math.isclose(incremental, off_cycles / cycles, rel_tol=2e-10):
            raise ValueError(f"{path.name}:{bench}: incremental ratio mismatch")
        rows[bench] = {
            "bench": bench,
            "baseline_cycles": base,
            "selected_cycles": cycles,
            "off_cycles": off_cycles,
            "ratio": ratio,
            "off_ratio": off_ratio,
            "selected_setup": selected,
            "partial": bench == "doitgen",
            "missing_setups": (
                "ms_noskills;ms_curated" if bench == "doitgen" else ""
            ),
        }
    common = [row for bench, row in rows.items() if bench != "doitgen"]
    selected_counts = Counter(row["selected_setup"] for row in common)
    if len(common) != 25 or sum(selected_counts.values()) != 25:
        raise ValueError(f"{path.name}: unexpected common set")
    summary = {
        "displayed": 26,
        "common": 25,
        "baseline_reduction": geometric_mean(row["ratio"] for row in common),
        "off_baseline_reduction": geometric_mean(
            row["off_ratio"] for row in common
        ),
        "expertise_reduction": geometric_mean(
            row["ratio"] / row["off_ratio"] for row in common
        ),
        "expertise_selections": sum(
            count
            for setup, count in selected_counts.items()
            if setup not in OFF_ARMS
        ),
        "off_selections": sum(
            count for setup, count in selected_counts.items() if setup in OFF_ARMS
        ),
    }
    if not math.isclose(summary["baseline_reduction"], 17.4563513810, rel_tol=2e-8):
        raise ValueError(f"{path.name}: Sonnet baseline aggregate drift")
    if not math.isclose(summary["expertise_reduction"], 2.8776076258, rel_tol=2e-8):
        raise ValueError(f"{path.name}: Sonnet expertise aggregate drift")
    return {"rows": rows, "summary": summary}


def read_opus(path: Path) -> dict[str, Any]:
    raw_rows = read_rows(path, OPUS_FIELDS)
    expected = set(CANONICAL_28) - {"heat-3d", "seidel-2d"}
    if len(raw_rows) != 26 or {row["bench"] for row in raw_rows} != expected:
        raise ValueError(f"{path.name}: unexpected benchmark coverage")
    rows: dict[str, dict[str, Any]] = {}
    for raw in raw_rows:
        bench = raw["bench"]
        selected = raw["selected_setup"]
        off = raw["best_expertise_disabled_setup"]
        if selected not in ARM_LABELS or off not in OFF_ARMS:
            raise ValueError(f"{path.name}:{bench}: unexpected selected setup")
        base = float(raw["stored_baseline_cycles"])
        cycles = float(raw["selected_cycles"])
        off_cycles = float(raw["best_expertise_disabled_cycles"])
        ratio = float(raw["stored_baseline_over_selected"])
        off_ratio = float(raw["stored_baseline_over_best_expertise_disabled"])
        incremental = float(raw["best_expertise_disabled_over_selected"])
        if not math.isclose(ratio, base / cycles, rel_tol=2e-10):
            raise ValueError(f"{path.name}:{bench}: selected ratio mismatch")
        if not math.isclose(off_ratio, base / off_cycles, rel_tol=2e-10):
            raise ValueError(f"{path.name}:{bench}: expertise-off ratio mismatch")
        if not math.isclose(incremental, off_cycles / cycles, rel_tol=2e-10):
            raise ValueError(f"{path.name}:{bench}: incremental ratio mismatch")
        rows[bench] = {
            "bench": bench,
            "baseline_cycles": base,
            "selected_cycles": cycles,
            "off_cycles": off_cycles,
            "ratio": ratio,
            "off_ratio": off_ratio,
            "selected_setup": selected,
            "partial": False,
            "missing_setups": "",
        }
    selected_counts = Counter(row["selected_setup"] for row in rows.values())
    summary = {
        "displayed": 26,
        "common": 26,
        "baseline_reduction": geometric_mean(
            row["ratio"] for row in rows.values()
        ),
        "off_baseline_reduction": geometric_mean(
            row["off_ratio"] for row in rows.values()
        ),
        "expertise_reduction": geometric_mean(
            row["ratio"] / row["off_ratio"] for row in rows.values()
        ),
        "expertise_selections": sum(
            count
            for setup, count in selected_counts.items()
            if setup not in OFF_ARMS
        ),
        "off_selections": sum(
            count for setup, count in selected_counts.items() if setup in OFF_ARMS
        ),
    }
    if not math.isclose(summary["baseline_reduction"], 14.5057003882, rel_tol=2e-8):
        raise ValueError(f"{path.name}: Opus baseline aggregate drift")
    if not math.isclose(summary["expertise_reduction"], 2.6334948973, rel_tol=2e-8):
        raise ValueError(f"{path.name}: Opus expertise aggregate drift")
    return {"rows": rows, "summary": summary}


def read_gpt55(path: Path) -> dict[str, Any]:
    raw_rows = read_rows(path, GPT55_FIELDS)
    expected = set(CANONICAL_28) - {"heat-3d", "seidel-2d"}
    if len(raw_rows) != 26 or {row["benchmark"] for row in raw_rows} != expected:
        raise ValueError(f"{path.name}: unexpected benchmark coverage")
    rows: dict[str, dict[str, Any]] = {}
    for raw in raw_rows:
        bench = raw["benchmark"]
        selected = raw["selected_setup"]
        off = raw["best_no_skills_setup"]
        if selected not in ARM_LABELS or off not in OFF_ARMS:
            raise ValueError(f"{path.name}:{bench}: unexpected selected setup")
        base = float(raw["stored_baseline_cycles"])
        cycles = float(raw["selected_cycles"])
        off_cycles = float(raw["best_no_skills_cycles"])
        ratio = float(raw["stored_baseline_over_selected"])
        off_ratio = float(raw["stored_baseline_over_best_no_skills"])
        incremental = float(raw["best_no_skills_over_selected"])
        partial = parse_bool(raw["partial_setup_grid"])
        missing_setups = raw["missing_agentic_setups"]
        if partial != bool(missing_setups):
            raise ValueError(f"{path.name}:{bench}: partial-grid metadata mismatch")
        if not math.isclose(ratio, base / cycles, rel_tol=2e-10):
            raise ValueError(f"{path.name}:{bench}: selected ratio mismatch")
        if not math.isclose(off_ratio, base / off_cycles, rel_tol=2e-10):
            raise ValueError(f"{path.name}:{bench}: no-skills ratio mismatch")
        if not math.isclose(incremental, off_cycles / cycles, rel_tol=2e-10):
            raise ValueError(f"{path.name}:{bench}: incremental ratio mismatch")
        rows[bench] = {
            "bench": bench,
            "baseline_cycles": base,
            "selected_cycles": cycles,
            "off_cycles": off_cycles,
            "ratio": ratio,
            "off_ratio": off_ratio,
            "selected_setup": selected,
            "partial": partial,
            "missing_setups": missing_setups,
        }
    common = [row for row in rows.values() if not row["partial"]]
    if len(common) != 23:
        raise ValueError(f"{path.name}: expected 23 complete setup grids")
    selected_counts = Counter(row["selected_setup"] for row in common)
    summary = {
        "displayed": 26,
        "common": 23,
        "partial": 3,
        "baseline_reduction": geometric_mean(
            row["ratio"] for row in common
        ),
        "off_baseline_reduction": geometric_mean(
            row["off_ratio"] for row in common
        ),
        "expertise_reduction": geometric_mean(
            row["ratio"] / row["off_ratio"] for row in common
        ),
        "expertise_selections": sum(
            count
            for setup, count in selected_counts.items()
            if setup not in OFF_ARMS
        ),
        "off_selections": sum(
            count for setup, count in selected_counts.items() if setup in OFF_ARMS
        ),
    }
    if not math.isclose(summary["baseline_reduction"], 11.3865402469, rel_tol=2e-8):
        raise ValueError(f"{path.name}: GPT-5.5 baseline aggregate drift")
    if not math.isclose(summary["expertise_reduction"], 2.3020681837, rel_tol=2e-8):
        raise ValueError(f"{path.name}: GPT-5.5 expertise aggregate drift")
    return {"rows": rows, "summary": summary}


def read_cosim(path: Path) -> dict[str, Any]:
    raw_rows = read_rows(path, COSIM_FIELDS)
    if len(raw_rows) != 56:
        raise ValueError(f"{path.name}: expected 56 rows")
    models: dict[str, dict[str, dict[str, Any]]] = {
        "DeepSeek V4 Flash": {},
        "Grok 4.5": {},
    }
    for raw in raw_rows:
        model = raw["model_label"]
        if model not in models:
            raise ValueError(f"{path.name}: unexpected model label {model}")
        bench = raw["bench"]
        enabled_pass = parse_bool(raw["expertise_enabled_pass"])
        disabled_pass = parse_bool(raw["expertise_disabled_pass"])
        baseline_cycles = parse_optional_positive_int(
            raw["benchmark_baseline_rtl_cycles"]
        )
        enabled_cycles = parse_optional_positive_int(
            raw["expertise_enabled_cycles"]
        )
        disabled_cycles = parse_optional_positive_int(
            raw["expertise_disabled_cycles"]
        )
        if enabled_pass and enabled_cycles is None:
            raise ValueError(f"{path.name}:{model}:{bench}: enabled pass lacks cycles")
        if disabled_pass and disabled_cycles is None:
            raise ValueError(f"{path.name}:{model}:{bench}: disabled pass lacks cycles")
        baseline_ratio = (
            float(raw["benchmark_baseline_over_enabled"])
            if raw["benchmark_baseline_over_enabled"] else None
        )
        off_baseline_ratio = (
            float(raw["benchmark_baseline_over_disabled"])
            if raw["benchmark_baseline_over_disabled"] else None
        )
        expected_baseline_ratio = (
            baseline_cycles / enabled_cycles
            if baseline_cycles is not None and enabled_pass else None
        )
        expected_off_baseline_ratio = (
            baseline_cycles / disabled_cycles
            if baseline_cycles is not None and disabled_pass else None
        )
        if (baseline_ratio is None) != (expected_baseline_ratio is None) or (
            baseline_ratio is not None and not math.isclose(
                baseline_ratio, expected_baseline_ratio, rel_tol=2e-10
            )
        ):
            raise ValueError(f"{path.name}:{model}:{bench}: baseline/enabled mismatch")
        if (off_baseline_ratio is None) != (expected_off_baseline_ratio is None) or (
            off_baseline_ratio is not None and not math.isclose(
                off_baseline_ratio, expected_off_baseline_ratio, rel_tol=2e-10
            )
        ):
            raise ValueError(f"{path.name}:{model}:{bench}: baseline/disabled mismatch")
        ratio = (
            float(raw["disabled_over_enabled"])
            if raw["disabled_over_enabled"] else None
        )
        if ratio is not None and not (enabled_pass and disabled_pass):
            raise ValueError(f"{path.name}:{model}:{bench}: ratio on failed pair")
        if ratio is not None and not math.isclose(
            ratio, disabled_cycles / enabled_cycles, rel_tol=2e-10
        ):
            raise ValueError(f"{path.name}:{model}:{bench}: disabled/enabled mismatch")
        models[model][bench] = {
            "bench": bench,
            "ratio": ratio,
            "baseline_cycles": baseline_cycles,
            "enabled_cycles": enabled_cycles,
            "disabled_cycles": disabled_cycles,
            "baseline_ratio": baseline_ratio,
            "off_baseline_ratio": off_baseline_ratio,
            "enabled_pass": enabled_pass,
            "disabled_pass": disabled_pass,
            "primary": parse_bool(raw["primary_aggregate"]),
            "asymmetric": bool(raw["selection_note"]),
        }
    summaries: dict[str, dict[str, Any]] = {}
    for model, rows in models.items():
        if set(rows) != set(CANONICAL_28):
            raise ValueError(f"{path.name}:{model}: unexpected benchmark set")
        primary = [
            row for row in rows.values()
            if row["primary"] and row["ratio"] is not None
        ]
        common = [
            row for row in primary if row["baseline_ratio"] is not None
        ]
        summaries[model] = {
            "baseline_available": sum(
                row["baseline_cycles"] is not None for row in rows.values()
            ),
            "enabled_pass": sum(row["enabled_pass"] for row in rows.values()),
            "disabled_pass": sum(row["disabled_pass"] for row in rows.values()),
            "paired": len(primary),
            "common": len(common),
            "baseline_reduction": geometric_mean(
                row["baseline_ratio"] for row in common
            ),
            "common_expertise_reduction": geometric_mean(
                row["ratio"] for row in common
            ),
            "common_wins": sum(row["ratio"] > 1.0 for row in common),
            "common_losses": sum(row["ratio"] < 1.0 for row in common),
            "expertise_reduction": geometric_mean(
                row["ratio"] for row in primary
            ),
            "wins": sum(row["ratio"] > 1.0 for row in primary),
            "losses": sum(row["ratio"] < 1.0 for row in primary),
        }
    deep = summaries["DeepSeek V4 Flash"]
    grok = summaries["Grok 4.5"]
    if deep["paired"] != 23 or not math.isclose(
        deep["expertise_reduction"], 3.2215760975, rel_tol=2e-8
    ):
        raise ValueError(f"{path.name}: DeepSeek aggregate drift")
    if deep["common"] != 22 or not math.isclose(
        deep["baseline_reduction"], 2.5790737749, rel_tol=2e-8
    ) or not math.isclose(
        deep["common_expertise_reduction"], 2.5535618973, rel_tol=2e-8
    ):
        raise ValueError(f"{path.name}: DeepSeek common-cohort drift")
    if grok["paired"] != 23 or not math.isclose(
        grok["expertise_reduction"], 1.5473608953, rel_tol=2e-8
    ):
        raise ValueError(f"{path.name}: Grok aggregate drift")
    if grok["common"] != 21 or not math.isclose(
        grok["baseline_reduction"], 4.4721452859, rel_tol=2e-8
    ) or not math.isclose(
        grok["common_expertise_reduction"], 1.8726536980, rel_tol=2e-8
    ):
        raise ValueError(f"{path.name}: Grok common-cohort drift")
    return {"models": models, "summaries": summaries}


def ratio_tick(exponent: int) -> str:
    if exponent == 0:
        return r"$1\times$"
    if exponent > 0:
        return rf"${2 ** exponent:g}\times$"
    return rf"$1/{2 ** (-exponent):g}\times$"


def bar_ratio_label(ratio: float) -> str:
    """Keep horizontal labels compact while preserving near-unity changes."""
    if ratio < 1.1:
        return f"{ratio:.2f}"
    if ratio >= 100:
        return f"{ratio:.0f}"
    return f"{ratio:.1f}"


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 6.5,
            "axes.titlesize": 7.0,
            "axes.labelsize": 6.3,
            "xtick.labelsize": 5.0,
            "ytick.labelsize": 5.6,
            "legend.fontsize": 5.4,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.5,
            "grid.linewidth": 0.32,
            "hatch.linewidth": 0.42,
        }
    )


def draw_teacher_row(
    axis: plt.Axes,
    rows: dict[str, dict[str, Any]],
    title: str,
) -> None:
    for index, bench in enumerate(CANONICAL_28):
        if bench not in rows:
            axis.text(
                index, 0.25, "N/R", ha="center", va="bottom",
                fontsize=4.2, color="0.50", rotation=90,
            )
            continue
        row = rows[bench]
        height = math.log2(row["ratio"])
        arm = row["selected_setup"]
        bar = axis.bar(
            index,
            height,
            width=0.72,
            color=POLICY_COLORS[ARM_POLICY[arm]],
            hatch=DEPTH_HATCHES[ARM_DEPTH[arm]],
            edgecolor="0.18",
            linewidth=0.35,
            zorder=3,
        )[0]
        label = bar_ratio_label(row["ratio"])
        if row["partial"]:
            label += r"$^\dagger$"
        axis.annotate(
            label,
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 1.0),
            textcoords="offset points",
            ha="center",
            va="bottom",
            rotation=0,
            fontsize=3.8,
            color="0.17",
            clip_on=True,
            zorder=5,
        )
        axis.scatter(
            index,
            math.log2(row["off_ratio"]),
            marker="D",
            facecolors="white",
            edgecolors="black",
            linewidths=0.45,
            s=9,
            zorder=6,
        )
    axis.set_title(title, loc="left", pad=1.8, fontweight="bold")
    axis.set_facecolor("#F7F8FA")


def draw_rtl_row(
    axis: plt.Axes,
    rows: dict[str, dict[str, Any]],
    title: str,
    policy: str,
) -> None:
    for index, bench in enumerate(CANONICAL_28):
        row = rows[bench]
        if row["baseline_cycles"] is None:
            axis.text(
                index, 0.25, "N/R", ha="center", va="bottom",
                fontsize=4.2, color="0.50", rotation=90,
            )
            continue
        if row["off_baseline_ratio"] is not None:
            axis.scatter(
                index,
                math.log2(row["off_baseline_ratio"]),
                marker="D",
                facecolors="white",
                edgecolors="black",
                linewidths=0.45,
                s=9,
                zorder=6,
            )
        failed_arms = []
        if not row["enabled_pass"]:
            failed_arms.append("E")
        if not row["disabled_pass"]:
            failed_arms.append("O")
        if failed_arms:
            axis.scatter(
                index,
                9.25,
                marker="x",
                s=15,
                linewidths=0.8,
                color="0.25",
                zorder=6,
                clip_on=False,
            )
            axis.text(
                index,
                8.72,
                "/".join(failed_arms),
                ha="center",
                va="top",
                fontsize=4.2,
                color="0.25",
            )
        if row["baseline_ratio"] is None:
            continue
        height = math.log2(row["baseline_ratio"])
        axis.bar(
            index,
            height,
            width=0.72,
            color=POLICY_COLORS[policy],
            hatch="xx" if row["asymmetric"] else "",
            alpha=0.52 if row["asymmetric"] else 0.92,
            edgecolor="0.18",
            linewidth=0.35,
            zorder=3,
        )
        if row["baseline_ratio"] > 1.0:
            axis.annotate(
                bar_ratio_label(row["baseline_ratio"]),
                xy=(index, height),
                xytext=(0, 1.0),
                textcoords="offset points",
                ha="center",
                va="bottom",
                rotation=0,
                fontsize=3.6,
                color="0.17",
                clip_on=True,
                zorder=5,
            )
    axis.set_title(title, loc="left", pad=1.8, fontweight="bold")
    axis.set_facecolor("#F7F8FA")


def render_figure(
    sonnet: dict[str, Any],
    opus: dict[str, Any],
    gpt55: dict[str, Any],
    cosim: dict[str, Any],
    output: Path,
) -> None:
    configure_matplotlib()
    figure, axes = plt.subplots(
        5,
        1,
        figsize=(7.08, 4.80),
        sharex=True,
        gridspec_kw={
            "left": 0.073,
            "right": 0.995,
            "top": 0.89,
            "bottom": 0.17,
            "hspace": 0.16,
        },
    )
    draw_teacher_row(
        axes[0],
        sonnet["rows"],
        "(a) Claude Sonnet 4.6",
    )
    draw_teacher_row(
        axes[1],
        opus["rows"],
        "(b) Claude Opus 4.8",
    )
    draw_teacher_row(
        axes[2],
        gpt55["rows"],
        "(c) GPT-5.5",
    )
    draw_rtl_row(
        axes[3],
        cosim["models"]["DeepSeek V4 Flash"],
        "(d) DeepSeek V4 Flash",
        "all_skills",
    )
    draw_rtl_row(
        axes[4],
        cosim["models"]["Grok 4.5"],
        "(e) Grok 4.5",
        "all_skills",
    )

    ticks = (-6, -3, 0, 3, 6, 9)
    for axis in axes:
        axis.axhline(0, color="0.12", linewidth=0.65, zorder=4)
        axis.grid(axis="y", color="0.86", zorder=0)
        axis.set_ylim(-7.0, 10.8)
        axis.set_yticks(ticks, [ratio_tick(value) for value in ticks])
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    for axis in axes[:-1]:
        axis.tick_params(axis="x", which="both", bottom=False)

    figure.supylabel(
        "Speedup over benchmark baseline",
        x=0.012,
        fontsize=6.3,
    )
    axes[-1].set_xticks(range(len(CANONICAL_28)))
    axes[-1].set_xticklabels(
        CANONICAL_28,
        rotation=56,
        ha="right",
        rotation_mode="anchor",
    )
    axes[-1].set_xlabel("HLSFactory kernel (numeric/alphabetic order)")

    no_skills_handle = (
        Patch(
            facecolor=POLICY_COLORS["no_skills"],
            edgecolor="0.18",
        ),
        Line2D(
            [],
            [],
            marker="D",
            markerfacecolor="white",
            markeredgecolor="black",
            linestyle="None",
            markersize=3.8,
        ),
    )
    policy_handles = [
        no_skills_handle,
        Patch(
            facecolor=POLICY_COLORS["selected"],
            edgecolor="0.18",
        ),
        Patch(
            facecolor=POLICY_COLORS["all_skills"],
            edgecolor="0.18",
        ),
    ]
    figure.legend(
        handles=policy_handles,
        labels=(
            "Expertise off (bar/diamond)",
            "Selected expertise",
            "All expertise",
        ),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=3,
        frameon=False,
        columnspacing=0.7,
        handlelength=1.35,
        handler_map={tuple: HandlerTuple(ndivide=None, pad=0.15)},
    )
    figure.text(
        0.5,
        0.925,
        "Solid/hatched: single-/multi-step; "
        r"$\times$E/$\times$O: expertise-on/off failure; N/R: unavailable comparison; "
        r"cross-hatch: excluded Grok repeat; $^\dagger$: partial grid",
        ha="center",
        va="center",
        fontsize=5.0,
        color="0.35",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        output,
        bbox_inches="tight",
        pad_inches=0.015,
        metadata={
            "Title": "Aligned architectural-expertise results across five model labels",
            "Author": "Anonymous",
            "Creator": "generate_aligned_model_expertise.py",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    plt.close(figure)


def write_table(
    path: Path,
    sonnet: dict[str, Any],
    opus: dict[str, Any],
    gpt55: dict[str, Any],
    cosim: dict[str, Any],
) -> None:
    sonnet_summary = sonnet["summary"]
    opus_summary = opus["summary"]
    gpt55_summary = gpt55["summary"]
    deep = cosim["summaries"]["DeepSeek V4 Flash"]
    grok = cosim["summaries"]["Grok 4.5"]
    lines = [
        "% AUTO-GENERATED by scripts/generate_aligned_model_expertise.py.",
        "\\begin{table*}[!t]",
        "  \\centering",
        "  \\caption{Common-cohort summary of Fig.~\\ref{fig:aligned-model-expertise}. "
        "Coverage is eligible kernels/26; teacher rows require all six setups, "
        "whereas RTL rows require an executed baseline and both flash arms. "
        "Five unmatched Grok reruns are excluded.}",
        "  \\label{tab:aligned-model-expertise}",
        "  \\scriptsize",
        "  \\setlength{\\tabcolsep}{3.5pt}",
        "  \\renewcommand{\\arraystretch}{1.00}",
        "  \\begin{tabular}{@{}lllcrrr@{}}",
        "    \\hline",
        "    \\textbf{Model} & \\textbf{Evidence} & "
        "\\textbf{Compared setups} & \\textbf{Coverage} & "
        "\\shortstack{\\textbf{Benchmark-baseline}\\\\\\textbf{speedup}} & "
        "\\shortstack{\\textbf{Expertise-off}\\\\\\textbf{speedup}} & "
        "\\shortstack{\\textbf{Expertise}\\\\\\textbf{wins/losses}} \\\\",
        "    \\hline",
        "    Claude Sonnet 4.6 & HLS synthesis & Exhaustive setup selection & "
        f"{sonnet_summary['common']}/{sonnet_summary['displayed']} & "
        f"{sonnet_summary['baseline_reduction']:.2f}$\\times$ & "
        f"{sonnet_summary['expertise_reduction']:.2f}$\\times$ & "
        f"{sonnet_summary['expertise_selections']}/"
        f"{sonnet_summary['off_selections']} \\\\",
        "    Claude Opus 4.8 & HLS synthesis & Exhaustive setup selection & "
        f"{opus_summary['common']}/{opus_summary['displayed']} & "
        f"{opus_summary['baseline_reduction']:.2f}$\\times$ & "
        f"{opus_summary['expertise_reduction']:.2f}$\\times$ & "
        f"{opus_summary['expertise_selections']}/"
        f"{opus_summary['off_selections']} \\\\",
        "    GPT-5.5 & HLS synthesis & Exhaustive setup selection & "
        f"{gpt55_summary['common']}/{gpt55_summary['displayed']} & "
        f"{gpt55_summary['baseline_reduction']:.2f}$\\times$ & "
        f"{gpt55_summary['expertise_reduction']:.2f}$\\times$ & "
        f"{gpt55_summary['expertise_selections']}/"
        f"{gpt55_summary['off_selections']} \\\\",
        "    DeepSeek V4 Flash & RTL co-simulation & Flash mode & "
        f"{deep['common']}/{deep['baseline_available']} & "
        f"{deep['baseline_reduction']:.2f}$\\times$ & "
        f"{deep['common_expertise_reduction']:.2f}$\\times$ & "
        f"{deep['common_wins']}/{deep['common_losses']} \\\\",
        "    Grok 4.5 & RTL co-simulation & Flash mode & "
        f"{grok['common']}/{grok['baseline_available']} & "
        f"{grok['baseline_reduction']:.2f}$\\times$ & "
        f"{grok['common_expertise_reduction']:.2f}$\\times$ & "
        f"{grok['common_wins']}/{grok['common_losses']} \\\\",
        "    \\hline",
        "  \\end{tabular}",
        "\\end{table*}",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_data(
    path: Path,
    sonnet: dict[str, Any],
    opus: dict[str, Any],
    gpt55: dict[str, Any],
    cosim: dict[str, Any],
) -> None:
    fields = (
        "model_label",
        "evidence",
        "benchmark",
        "benchmark_baseline_cycles",
        "selected_cycles",
        "expertise_off_cycles",
        "plotted_ratio",
        "expertise_disabled_marker_ratio",
        "selected_setup",
        "enabled_pass",
        "disabled_pass",
        "primary_aggregate",
        "common_table_aggregate",
        "note",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=fields, lineterminator="\n"
        )
        writer.writeheader()
        for model, study in (
            ("Claude Sonnet 4.6", sonnet),
            ("Claude Opus 4.8", opus),
            ("GPT-5.5", gpt55),
        ):
            for bench in CANONICAL_28:
                row = study["rows"].get(bench)
                writer.writerow(
                    {
                        "model_label": model,
                        "evidence": "hls_synthesis",
                        "benchmark": bench,
                        "benchmark_baseline_cycles": (
                            "" if row is None else f"{row['baseline_cycles']:.12g}"
                        ),
                        "selected_cycles": (
                            "" if row is None else f"{row['selected_cycles']:.12g}"
                        ),
                        "expertise_off_cycles": (
                            "" if row is None else f"{row['off_cycles']:.12g}"
                        ),
                        "plotted_ratio": "" if row is None else f"{row['ratio']:.12g}",
                        "expertise_disabled_marker_ratio": (
                            "" if row is None else f"{row['off_ratio']:.12g}"
                        ),
                        "selected_setup": "" if row is None else row["selected_setup"],
                        "enabled_pass": "",
                        "disabled_pass": "",
                        "primary_aggregate": (
                            False if row is None or row["partial"] else True
                        ),
                        "common_table_aggregate": (
                            False if row is None or row["partial"] else True
                        ),
                        "note": (
                            "not run" if row is None else
                            f"partial: missing {row['missing_setups']}"
                            if row["partial"] else ""
                        ),
                    }
                )
        for model in ("DeepSeek V4 Flash", "Grok 4.5"):
            for bench in CANONICAL_28:
                row = cosim["models"][model][bench]
                writer.writerow(
                    {
                        "model_label": model,
                        "evidence": "reported_rtl_cosimulation",
                        "benchmark": bench,
                        "benchmark_baseline_cycles": (
                            row["baseline_cycles"] or ""
                        ),
                        "selected_cycles": (
                            row["enabled_cycles"] if row["enabled_pass"] else ""
                        ),
                        "expertise_off_cycles": (
                            row["disabled_cycles"] if row["disabled_pass"] else ""
                        ),
                        "plotted_ratio": (
                            "" if row["baseline_ratio"] is None
                            else f"{row['baseline_ratio']:.12g}"
                        ),
                        "expertise_disabled_marker_ratio": (
                            "" if row["off_baseline_ratio"] is None
                            else f"{row['off_baseline_ratio']:.12g}"
                        ),
                        "selected_setup": "fixed_flash_all_skills",
                        "enabled_pass": row["enabled_pass"],
                        "disabled_pass": row["disabled_pass"],
                        "primary_aggregate": row["primary"],
                        "common_table_aggregate": (
                            row["primary"]
                            and row["baseline_ratio"] is not None
                            and row["ratio"] is not None
                        ),
                        "note": (
                            "baseline RTL unavailable"
                            if row["baseline_cycles"] is None else
                            "excluded: asymmetric enabled-arm repeat selection"
                            if row["asymmetric"] else ""
                        ),
                    }
                )


def write_manifest(
    path: Path,
    input_hashes: dict[str, str],
    outputs: list[Path],
    sonnet: dict[str, Any],
    opus: dict[str, Any],
    gpt55: dict[str, Any],
    cosim: dict[str, Any],
) -> None:
    bundle_payload = json.dumps(
        input_hashes, sort_keys=True, separators=(",", ":")
    ).encode()
    manifest = {
        "schema": "hlsmith.paper.aligned-model-expertise.v3",
        "bundle_sha256": hashlib.sha256(bundle_payload).hexdigest(),
        "inputs": [
            {"source_id": name, "sha256": digest}
            for name, digest in sorted(input_hashes.items())
        ],
        "outputs": [
            {"path": str(output), "sha256": sha256(output)}
            for output in outputs
        ],
        "metrics": {
            "sonnet": sonnet["summary"],
            "opus": opus["summary"],
            "gpt55": gpt55["summary"],
            "deepseek": cosim["summaries"]["DeepSeek V4 Flash"],
            "grok": cosim["summaries"]["Grok 4.5"],
        },
        "evidence_boundaries": [
            "Sonnet, Opus, and GPT-5.5 rows use post-hoc best-of-six "
            "HLS-synthesis "
            "selection and stored HLSFactory baseline cycles.",
            "Sonnet aggregates use the 25-kernel all-arm common set; the "
            "displayed Doitgen row has only four of six agentic setups.",
            "The teacher-model candidates lack independent output comparison "
            "and generated-design RTL simulation.",
            "The GPT-5.5 source records use only the model key gpt55 and do not "
            "record an exact checkpoint revision.",
            "GPT-5.5 aggregates use 23 kernels with all six agentic setups "
            "attempted; three displayed kernels have one unexecuted setup.",
            "DeepSeek and Grok bars divide executed benchmark-baseline RTL "
            "cycles by fixed-flash reported RTL cycles; executed baselines cover "
            "26 kernels and are unavailable for Heat-3D and Seidel-2D.",
            "The common-cohort table intersects the executed baseline, passing "
            "expertise-on/off arms, and the primary-selection cohort.",
            "Experiment metadata identifies both enabled RTL arms as the "
            "all-skills policy. The compact DeepSeek campaign key is aav_n "
            "(all positive plus avoid rules), while the Grok input labels the "
            "arm skills.",
            "Five Grok rows with asymmetric enabled-arm repeat selection are "
            "shown but excluded from the strict aggregate.",
            "The five rows share kernel positions and improvement direction, "
            "not a common evidence level, search budget, or expertise snapshot.",
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sonnet-data",
        type=Path,
        default=Path("generated/newskills_sonnet_snapshot.csv"),
    )
    parser.add_argument(
        "--opus-data",
        type=Path,
        default=Path("generated/july30_opus_selection.csv"),
    )
    parser.add_argument(
        "--gpt55-data",
        type=Path,
        default=Path("generated/newskills_gpt55_snapshot.csv"),
    )
    parser.add_argument(
        "--cosim-data",
        type=Path,
        default=Path("generated/july30_cosim_expertise.csv"),
    )
    parser.add_argument(
        "--figure",
        type=Path,
        default=Path("figures/model_expertise_aligned.pdf"),
    )
    parser.add_argument(
        "--table",
        type=Path,
        default=Path("generated/model_expertise_summary_table.tex"),
    )
    parser.add_argument(
        "--output-data",
        type=Path,
        default=Path("generated/model_expertise_aligned.csv"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("generated/model_expertise_aligned_manifest.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inputs = {
        path.name: path
        for path in (
            args.sonnet_data,
            args.opus_data,
            args.gpt55_data,
            args.cosim_data,
        )
    }
    for path in inputs.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    input_hashes = {
        name: require_hash(path) for name, path in inputs.items()
    }
    sonnet = read_sonnet(args.sonnet_data)
    opus = read_opus(args.opus_data)
    gpt55 = read_gpt55(args.gpt55_data)
    cosim = read_cosim(args.cosim_data)
    render_figure(sonnet, opus, gpt55, cosim, args.figure)
    write_table(args.table, sonnet, opus, gpt55, cosim)
    write_data(args.output_data, sonnet, opus, gpt55, cosim)
    outputs = (args.figure, args.table, args.output_data)
    write_manifest(
        args.manifest,
        input_hashes,
        list(outputs),
        sonnet,
        opus,
        gpt55,
        cosim,
    )
    print(
        json.dumps(
            {
                "figure": str(args.figure),
                "sonnet_common_expertise_reduction":
                    sonnet["summary"]["expertise_reduction"],
                "opus_expertise_reduction":
                    opus["summary"]["expertise_reduction"],
                "gpt55_expertise_reduction":
                    gpt55["summary"]["expertise_reduction"],
                "deepseek_expertise_reduction":
                    cosim["summaries"]["DeepSeek V4 Flash"][
                        "expertise_reduction"
                    ],
                "grok_strict_expertise_reduction":
                    cosim["summaries"]["Grok 4.5"]["expertise_reduction"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
