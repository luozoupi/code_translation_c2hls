#!/usr/bin/env python3
"""Export paired agentic/QoR outcomes, including pending and invalid cells."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from qor_design_space import extract_qor_metrics, selection_rank  # noqa: E402


def code_hash(code: str) -> str:
    return hashlib.sha256(code.encode()).hexdigest()


def rejection_cause(candidate: dict) -> str:
    error = str(candidate.get("error") or "")
    if "same adapter parameters" in error:
        return "axi_bundle_parameter_mismatch"
    if "Invalid value for m_axi max_read/write_burst_length" in error:
        return "illegal_axi_burst_length"
    return str(candidate.get("status") or "unknown_rejection")


def characterize_result(result: dict) -> dict:
    design = result.get("qor_design_sweep") or {}
    parent = design.get("parent") or {}
    parent_metrics = parent.get("metrics") or {}
    parent_cycles = parent_metrics.get("latency_cycles_worst")
    if not parent.get("feasible") or not parent_cycles:
        return {}
    candidates = design.get("candidates") or []
    outcomes = Counter()
    rejected = Counter()
    resource_costs = Counter()
    comparisons = []
    for candidate in candidates:
        metrics = candidate.get("metrics") or {}
        cycles = metrics.get("latency_cycles_worst")
        valid = candidate.get("feasible") is True and cycles is not None
        effect = ("faster" if cycles < parent_cycles else "slower" if cycles > parent_cycles else "unchanged") if valid else "invalid"
        outcomes[effect] += 1
        if not valid:
            rejected[rejection_cause(candidate)] += 1
        deltas = {
            key: metrics[key] - parent_metrics[key]
            for key in ("dsp", "bram", "lut", "ff", "uram", "estimated_clock_period_ns", "achieved_pipeline_ii_max", "interval")
            if valid and metrics.get(key) is not None and parent_metrics.get(key) is not None
        }
        for key in ("dsp", "bram", "lut", "ff", "uram"):
            if deltas.get(key, 0) > 0:
                resource_costs[key] += 1
        comparisons.append({
            "candidate_id": candidate["candidate_id"], "effect": effect,
            "changed_knobs": candidate.get("changed_knobs") or [],
            "latency_change_percent": 100 * (cycles / parent_cycles - 1) if valid else None,
            "metric_deltas": deltas, "rejection_cause": None if valid else rejection_cause(candidate),
        })
    winner_id = design.get("winner_candidate_id")
    winner = next((c for c in [parent, *candidates] if c.get("candidate_id") == winner_id), parent)
    winner_metrics = winner.get("metrics") or {}
    final_cycles = winner_metrics.get("latency_cycles_worst")
    controls = winner.get("changed_knobs") or []
    control_text = ", ".join(f"{c['name']}: {c.get('from')} -> {c['to']}" for c in controls)
    if not control_text:
        control_text = "retained frozen parent" if winner is parent else "winner control metadata unavailable"
    deltas = {
        key: winner_metrics[key] - parent_metrics[key]
        for key in ("dsp", "bram", "lut", "ff", "uram", "estimated_clock_period_ns", "achieved_pipeline_ii_max", "interval")
        if winner_metrics.get(key) is not None and parent_metrics.get(key) is not None
    }
    effect = "improved" if final_cycles < parent_cycles else "regressed" if final_cycles > parent_cycles else "unchanged"
    why = [f"{control_text}; selected latency {parent_cycles:g} -> {final_cycles:g} cycles ({effect})."]
    if effect == "unchanged":
        why.append(f"No lower-cycle feasible candidate was found in the tested budget ({outcomes['invalid']} rejected, {outcomes['slower']} slower).")
    changed_metrics = ", ".join(f"{key} {value:+g}" for key, value in deltas.items() if value)
    if changed_metrics:
        why.append("Measured accompanying changes: " + changed_metrics + ".")
    overrides = sum(bool((c.get("compiler_realization") or {}).get("any_overridden")) for c in candidates)
    if overrides:
        why.append(f"Compiler-reported overrides occurred in {overrides} candidates.")
    why.append("These are measured associations; top-level II/resource totals alone do not establish the critical-path mechanism.")
    return {
        "outcomes": {key: outcomes[key] for key in ("faster", "unchanged", "slower", "invalid")},
        "rejection_causes": dict(rejected), "resource_increase_candidates": dict(resource_costs),
        "selected_effect": effect, "selected_controls": controls, "selected_metric_deltas": deltas,
        "explanation": " ".join(why), "candidate_comparisons": comparisons,
        "knob_trends": design.get("knob_trends") or [],
    }


def analyze_cell(cell: dict, config: dict) -> dict:
    row = {key: cell[key] for key in ("cell_id", "benchmark", "family", "strategy", "skill_mode")}
    row.update(schema_version="c2hls.qor-paired-measurement.v1", pair_valid=False,
               status=cell["status"], parent_cycles=None, final_cycles=None,
               speedup=None, result_path=cell["result_path"], issues=[])
    path = Path(cell["result_path"])
    if not path.is_file():
        return row
    try:
        result = json.loads(path.read_text())
        parent = json.loads(Path(cell["parent_path"]).read_text())
    except (OSError, json.JSONDecodeError):
        row["issues"] = ["paired artifacts unavailable or incomplete"]
        return row
    design = result.get("qor_design_sweep") or {}
    before = design.get("parent") or {}
    reviews = result.get("qor_profile_reviews") or []
    row["profile_reviews"] = len(reviews)
    row["skill_reviews"] = sum(len(r.get("skill_reviews") or []) for r in reviews)
    row["unsupported_skill_declarations"] = sum(len(r.get("unsupported_declarations") or []) for r in reviews)
    before_metrics = before.get("metrics") or extract_qor_metrics(parent.get("final_report"))
    after_metrics = extract_qor_metrics(result.get("final_report"))
    parent_code_hash = code_hash(parent.get("hls_code") or "")
    final_code_hash = code_hash(result.get("hls_code") or "")
    issues = []
    if config.get("review_version"):
        if not reviews:
            issues.append("registered profile reviewer produced no evidence records")
        elif any(r.get("review_version") != config["review_version"] for r in reviews):
            issues.append("profile reviewer version differs from registered campaign")
    if config.get("method_version") and design.get("method_version") != config["method_version"]:
        issues.append("QoR method version differs from registered campaign")
    if before_metrics != extract_qor_metrics(parent.get("final_report")):
        issues.append("paired parent report differs from frozen QoR metrics")
    if result.get("success") is not True:
        issues.append("agentic result unsuccessful")
    if (result.get("reference_isolation_audit") or {}).get("passed") is not True:
        issues.append("reference audit missing or failed")
    if (result.get("skill_snapshot_integrity") or {}).get("unchanged") is not True:
        issues.append("skill snapshot integrity missing or failed")
    if before.get("feasible") is not True or (result.get("candidate_feasibility") or {}).get("feasible") is not True:
        issues.append("parent or final candidate infeasible")
    if design.get("parent_integrity_passed") is not True:
        issues.append("frozen-parent integrity missing or failed")
    if parent_code_hash != parent.get("code_sha256") or parent_code_hash != design.get("parent_code_sha256"):
        issues.append("paired parent code hash mismatch")
    if final_code_hash != design.get("winner_code_sha256"):
        issues.append("final code differs from declared QoR winner")
    if (parent.get("csim") or {}).get("passed") is not True or (result.get("csim") or {}).get("passed") is not True:
        issues.append("parent or final CSim did not pass")
    usage = result.get("llm_usage") or {}
    parent_usage = parent.get("llm_usage") or {}
    models = sorted({event.get("model") for event in usage.get("events", []) if event.get("model")})
    if models != [config["model"]]:
        issues.append("LLM events do not exclusively identify the registered model")
    if usage.get("calls") != parent_usage.get("calls"):
        issues.append("extra model calls occurred after the paired control")
    if usage.get("calls", 0) > 5:
        issues.append("five-call model budget exceeded")
    if result.get("selected_winner_cosim_count", 0) != 0:
        issues.append("unexpected selected-winner COSIM")
    pre = before_metrics.get("latency_cycles_worst")
    post = after_metrics.get("latency_cycles_worst")
    if pre is None or post is None or pre <= 0 or post <= 0 or not math.isfinite(pre) or not math.isfinite(post):
        issues.append("missing positive finite cycles")
    elif post > pre:
        issues.append("QoR selection regressed cycles")
    if design.get("winner_metrics") != after_metrics:
        issues.append("final report differs from QoR winner metrics")
    candidates = design.get("candidates") or []
    row.update(
        pair_valid=not issues, status="valid_pair" if not issues else "invalid_pair",
        issues=issues, model=config["model"], observed_models=models,
        parent_cycles=pre, final_cycles=post,
        speedup=pre / post if not issues else None,
        parent_code_sha256=parent_code_hash, final_code_sha256=final_code_hash,
        parent_origin=(design.get("parent_origin") or {}).get("step_name"),
        winner_candidate_id=design.get("winner_candidate_id"),
        winner_explanation=design.get("winner_explanation"),
        discovered_knobs=len(design.get("discovered_knobs") or []),
        knob_kinds=sorted({knob["kind"] for knob in design.get("discovered_knobs") or []}),
        candidates=len(candidates), feasible_candidates=sum(candidate.get("feasible") is True for candidate in candidates),
        skipped_candidates=design.get("skipped_candidate_count", 0),
        compiler_overrides=design.get("compiler_override_candidate_count", 0),
        active_knobs=sum(trend.get("observable_synthesis_response") is True for trend in design.get("knob_trends") or []),
        llm_calls=usage.get("calls"), input_tokens=usage.get("input_tokens"),
        output_tokens=usage.get("output_tokens"),
        llm_synthesis_calls=result.get("llm_candidate_synthesis_evaluation_count"),
        qor_synthesis_calls=result.get("qor_synthesis_evaluation_count"),
        total_synthesis_calls=result.get("total_synthesis_calls"),
        parent_csim=True if (parent.get("csim") or {}).get("passed") else False,
        final_csim=True if (result.get("csim") or {}).get("passed") else False,
        fingerprint_sha256=(result.get("run_fingerprint") or {}).get("sha256"),
        method_version=config.get("method_version", "legacy_source_local_v1"),
        characterization=characterize_result(result) if not issues else {},
    )
    for metric in ("dsp", "bram", "lut", "ff", "uram", "estimated_clock_period_ns", "achieved_pipeline_ii_max"):
        row[f"parent_{metric}"] = before_metrics.get(metric)
        row[f"final_{metric}"] = after_metrics.get(metric)
    return row


def budget_rows(cell: dict, pair: dict) -> list[dict]:
    if not pair["pair_valid"]:
        return []
    design = json.loads(Path(cell["result_path"]).read_text())["qor_design_sweep"]
    candidates = design.get("candidates") or []
    rows = []
    for budget in (0, 4, 8, 16, 18):
        evaluated = candidates[:budget]
        eligible = [record for record in [design["parent"], *evaluated] if record.get("feasible") is True]
        winner = min(eligible, key=selection_rank)
        cycles = winner["metrics"]["latency_cycles_worst"]
        rows.append({
            "cell_id": cell["cell_id"], "benchmark": cell["benchmark"],
            "strategy": cell["strategy"], "candidate_budget": budget,
            "evaluated_candidates": len(evaluated), "best_cycles": cycles,
            "speedup": pair["parent_cycles"] / cycles,
            "winner_candidate_id": winner["candidate_id"],
        })
    return rows


def export_rows(path: Path, rows: list[dict]) -> None:
    path.with_suffix(".json").write_text(json.dumps(rows, indent=2, allow_nan=False) + "\n")
    path.with_suffix(".jsonl").write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in rows))
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value) if isinstance(value, (list, dict)) else value
                             for key, value in row.items()})


def statistics(rows: list[dict]) -> dict:
    import numpy as np
    valid = [row for row in rows if row["pair_valid"]]
    clusters = defaultdict(list)
    for row in valid:
        clusters[row["benchmark"]].append(math.log(row["speedup"]))
    cluster_means = np.array([sum(values) / len(values) for values in clusters.values()])
    interval = None
    if len(cluster_means) >= 2:
        samples = np.random.default_rng(42).choice(cluster_means, (5000, len(cluster_means)), replace=True)
        interval = np.exp(np.quantile(samples.mean(axis=1), [0.025, 0.975])).tolist()
    return {
        "planned_pairs": len(rows), "valid_pairs": len(valid),
        "benchmark_clusters": len(clusters),
        "improved_pairs": sum(row["speedup"] > 1.0 for row in valid),
        "improved_over_5_percent": sum(row["speedup"] >= 1.05 for row in valid),
        "unchanged_pairs": sum(row["speedup"] == 1.0 for row in valid),
        "geomean_speedup_benchmark_weighted": float(np.exp(cluster_means.mean())) if len(cluster_means) else None,
        "bootstrap_95_percent_interval": interval,
        "observed_qor_candidates": sum(row.get("candidates", 0) for row in rows),
        "observed_feasible_qor_candidates": sum(row.get("feasible_candidates", 0) for row in rows),
    }


def plot(rows: list[dict], budgets: list[dict], output: Path, model: str = "deepseek-v4-pro") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    benches = list(dict.fromkeys(row["benchmark"] for row in rows))
    fig, axes = plt.subplots(2, 1, figsize=(max(15, len(benches) * 0.7), 9), sharex=True)
    for axis, strategy in zip(axes, ("flash", "dynamic")):
        values = {row["benchmark"]: row for row in rows if row["strategy"] == strategy}
        x = np.arange(len(benches))
        for shift, field, label, color in ((-0.18, "parent_cycles", "Agentic pre-QoR", "#0072B2"),
                                           (0.18, "final_cycles", "Agentic + QoR", "#D55E00")):
            y = [values[bench][field] if values[bench]["pair_valid"] else np.nan for bench in benches]
            axis.bar(x + shift, y, 0.36, label=label, color=color)
        for i, bench in enumerate(benches):
            if not values[bench]["pair_valid"]:
                axis.text(i, 0.08, "pending/invalid", transform=axis.get_xaxis_transform(),
                          rotation=90, ha="center", fontsize=8)
        if any(row["pair_valid"] for row in values.values()):
            axis.set_yscale("log")
        axis.set_ylabel("CSynth cycles")
        axis.set_title("Flash" if strategy == "flash" else "Multistep (dynamic)")
        axis.grid(axis="y", alpha=0.2)
        axis.legend()
    axes[-1].set_xticks(range(len(benches)), [bench.removeprefix("hlsfactory_") for bench in benches], rotation=35, ha="right")
    fig.suptitle(f"{model}: exact paired pre/post-QoR comparison")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"paired_cycles.{suffix}", dpi=180, bbox_inches="tight")
    plt.close(fig)
    if budgets:
        fig, axis = plt.subplots(figsize=(9, 5))
        for strategy in ("flash", "dynamic"):
            grouped = defaultdict(list)
            for row in budgets:
                if row["strategy"] == strategy:
                    grouped[row["candidate_budget"]].append(math.log(row["speedup"]))
            x = sorted(grouped)
            axis.plot(x, [math.exp(sum(grouped[n]) / len(grouped[n])) for n in x], marker="o", label=strategy)
        axis.set_xlabel("Additional candidate budget (observed prefixes)")
        axis.set_ylabel("Geomean cycle speedup over the exact parent")
        axis.grid(alpha=0.2)
        axis.legend()
        fig.tight_layout()
        for suffix in ("png", "pdf"):
            fig.savefig(output / f"candidate_budget_response.{suffix}", dpi=180)
        plt.close(fig)


def write_characterization(rows: list[dict], output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    valid = [row for row in rows if row["pair_valid"]]
    outcomes = []
    trends = []
    lines = ["# Parameter Search Effects", "",
             "Latency is Vitis CSynth worst-case estimated cycles; CSim is the correctness gate, not a hardware-cycle measurement. No COSIM was executed.", "",
             "`interval` is the reported top-function initiation interval in cycles between invocations, not the target II of an individual loop. `achieved_pipeline_ii_max` is the maximum reported loop II, a diagnostic proxy rather than proof of the critical bottleneck.", "",
             "Raw candidates may be slower or use more resources. Retaining the feasible parent prevents final latency regression; it does not prove that parameter increases improve performance monotonically. Invalid candidates have no imputed cycle change. A faster winner may cost more resources while still fitting U280.", "",
             "The reports compare each new trajectory with its exact frozen pre-QoR parent. Comparisons with historical fresh trajectories are not isolated tests of the method change. The separate saved-parent AXI regression checks isolate legality fixes.", "",
             "| Benchmark | Mode | Faster | Same | Slower | Invalid | Winner speedup | Parent origin |",
             "|---|---|---:|---:|---:|---:|---:|---|"]
    for row in valid:
        data = row["characterization"]
        base = {key: row[key] for key in ("cell_id", "benchmark", "family", "strategy", "method_version")}
        outcomes.extend({**base, **c} for c in data.get("candidate_comparisons", []))
        trends.extend({**base, **trend} for trend in data.get("knob_trends", []))
        counts = data["outcomes"]
        lines.append(f"| {row['benchmark']} | {row['strategy']} | {counts['faster']} | {counts['unchanged']} | {counts['slower']} | {counts['invalid']} | {row['speedup']:.4f}x | {row['parent_origin']} |")
    lines.extend(["", "## Per-Kernel Evidence", ""])
    for row in valid:
        data = row["characterization"]
        lines.extend([f"### {row['benchmark']} / {row['strategy']}", "", data["explanation"], ""])
        if data["rejection_causes"]:
            lines.extend(["Rejected: " + json.dumps(data["rejection_causes"], sort_keys=True), ""])
        for trend in data["knob_trends"]:
            lines.append(f"- {trend['name']}: tested values {trend['tested_values']}; cycles {trend['worst_cycles']}; response {trend['realization_status']}; direction violations {trend['monotonicity_violations']}.")
        lines.append("")
    (output / "parameter_effects.md").write_text("\n".join(lines) + "\n")
    export_rows(output / "candidate_effects", outcomes)
    export_rows(output / "parameter_trends", trends)
    if not valid:
        return
    fig, axis = plt.subplots(figsize=(13, max(6, len(valid) * 0.3)))
    y = np.arange(len(valid))
    left = np.zeros(len(valid))
    for category, color in (("faster", "#009E73"), ("unchanged", "#999999"), ("slower", "#E69F00"), ("invalid", "#D55E00")):
        counts = np.array([row["characterization"]["outcomes"][category] for row in valid])
        axis.barh(y, counts, left=left, label=category, color=color)
        left += counts
    axis.set_yticks(y, [row["benchmark"].removeprefix("hlsfactory_") + " / " + row["strategy"] for row in valid])
    axis.invert_yaxis()
    axis.set_xlabel("Evaluated QoR candidates relative to the frozen parent")
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, 1.05), ncol=4)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"candidate_outcomes.{suffix}", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    root = Path(config["report_root"])
    state = json.loads((root / "campaign_state.json").read_text())
    rows = [analyze_cell(cell, config) for cell in state["cells"]]
    budgets = [point for cell, pair in zip(state["cells"], rows) for point in budget_rows(cell, pair)]
    candidates = []
    profile_reviews = []
    schema_lines = []
    for cell in state["cells"]:
        if cell["status"] not in {"complete", "failed"}:
            continue
        result_path = Path(cell["result_path"])
        if not result_path.is_file():
            continue
        try:
            result = json.loads(result_path.read_text())
        except json.JSONDecodeError:
            continue
        design = result.get("qor_design_sweep") or {}
        for index, review in enumerate(result.get("qor_profile_reviews") or []):
            profile_reviews.append({"cell_id": cell["cell_id"], "review_index": index, **review})
        for candidate in [design.get("parent") or {}, *(design.get("candidates") or [])]:
            if candidate:
                candidates.append({
                    "cell_id": cell["cell_id"], "benchmark": cell["benchmark"],
                    "strategy": cell["strategy"], "skill_mode": cell["skill_mode"],
                    **candidate,
                })
        schema = Path(cell["schema_jsonl"])
        if cell["status"] == "complete" and schema.is_file():
            schema_lines.extend(schema.read_text().splitlines())
    export_rows(root / "paired_measurements", rows)
    export_rows(root / "candidate_budget_measurements", budgets)
    export_rows(root / "all_candidate_measurements", candidates)
    export_rows(root / "skill_profile_reviews", profile_reviews)
    (root / "schema_records.jsonl").write_text("".join(line + "\n" for line in schema_lines if line))
    stats = statistics(rows)
    (root / "summary.json").write_text(json.dumps(stats, indent=2) + "\n")
    lines = [
        f"# {config['model']} Additive QoR Campaign", "",
        f"State: **{state['status']}**. Valid paired results: **{stats['valid_pairs']}/{len(rows)}**.", "",
        "Each pair shares one fresh model trajectory and its exact selected pre-QoR kernel. "
        "The added search tests source-local controls with CSim, CSynth, timing, and resource-fit gates. "
        f"All active model roles use {config['model']}, matched skills, and the five-call model budget. "
        "The additional budget is 16 OFAT candidates plus up to two interactions, on Vitis 2023.2, "
        "U280 xcu280-fsvh2892-2L-e, 3.33 ns. COSIM is disabled.", "",
        "The extra synthesis cost is reported explicitly. This measures whether added parameter "
        "exploration improves the same model's output; it does not establish superiority at equal tool budget. "
        "The retained best can stay unchanged. Invalid and missing outcomes are never assigned cycles. "
        "Each cell has one trajectory; this is an in-domain comparison rather than an external or multi-seed study.", "",
        f"Benchmark-weighted geomean speedup: {stats['geomean_speedup_benchmark_weighted']}. "
        f"Benchmark-clustered bootstrap 95% interval: {stats['bootstrap_95_percent_interval']}.", "",
        "| Benchmark | Family | Mode | Skills | Status | Parent cycles | Final cycles | Speedup | Valid QoR candidates | Extra CSynth |",
        "|---|---|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        valid = row["pair_valid"]
        lines.append(
            f"| {row['benchmark'].removeprefix('hlsfactory_')} | {row['family']} | "
            f"{row['strategy']} | {row['skill_mode']} | {row['status']} | "
            f"{int(row['parent_cycles']) if valid else 'N/A'} | "
            f"{int(row['final_cycles']) if valid else 'N/A'} | "
            f"{format(row['speedup'], '.4f') + 'x' if valid else 'N/A'} | "
            f"{row.get('feasible_candidates', 0)}/{row.get('candidates', 0)} | "
            f"{row.get('qor_synthesis_calls', 'N/A')} |"
        )
    lines.extend(["", "## Candidate evidence", ""])
    lines.extend(["See [parameter effects](parameter_effects.md) for slower candidates, resource tradeoffs, rejection causes, and measured control-response tables.", ""])
    for row in rows:
        if row["pair_valid"]:
            lines.append(f"- {row['cell_id']}: {row['winner_explanation']}")
        elif row["issues"]:
            lines.append(f"- {row['cell_id']}: {'; '.join(row['issues'])}")
    (root / "report.md").write_text("\n".join(lines) + "\n")
    plot(rows, budgets, root, config["model"])
    write_characterization(rows, root)
    products = [
        path for path in root.iterdir()
        if path.is_file() and (path.stem in {
            "paired_measurements", "candidate_budget_measurements", "all_candidate_measurements",
            "schema_records", "summary", "report", "paired_cycles", "candidate_budget_response",
            "parameter_effects", "candidate_effects", "parameter_trends", "candidate_outcomes",
            "skill_profile_reviews",
        })
    ]
    (root / "artifact_manifest.json").write_text(json.dumps({
        "schema_version": "c2hls.qor-paired-report-manifest.v1",
        "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "outputs": [{"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                    for path in sorted(products)],
    }, indent=2) + "\n")
    print(root / "report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
