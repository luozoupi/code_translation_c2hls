#!/usr/bin/env python3
"""Render frozen, identity-checked RTL evidence for the original Figure 8 designs.

Import is an explicit local audit of archived source/measurement files. Rendering
needs only the resulting portable bundle; no running experiment state is read.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Patch

from generate_skill_v3_open_models import (
    CANONICAL_28, MODEL_ORDER, MODEL_SPECS, SKILL_MODES,
)

BENCHMARKS = tuple("hlsfactory_" + b.replace("-", "_") for b in CANONICAL_28 if b != "doitgen")
GROUPS = tuple((MODEL_SPECS[m]["label"], mode) for m in MODEL_ORDER for mode in ("flash", "multistep"))
STATUS_CODE = {"pass": "P", "timeout": "T", "fail": "F", "blocked_identity": "U"}
STATUS_COLOR = {"pass": "#007A5E", "timeout": "#A85800", "fail": "#B22222", "blocked_identity": "#5C5685"}
METRICS = (("synthesis_ratio", "CSynth", "#648DB1", -.16),
           ("rtl_ratio_1h", "COSim", "#BC6840", .16))
BAR_WIDTH = .28
FILES = ("figure8_source_snapshot.csv", "figure8_identity_join.json", "cosim_evidence_snapshot.json", "inventory.json", "source_audit.json", "measurements.json")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def fail(condition, message):
    if not condition:
        raise ValueError(message)


def dump(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def import_bundle(evidence: Path, historical_manifest: Path, destination: Path):
    """Copy evidence only after checking original result/kernel/measurement bytes."""
    fail(not destination.exists(), "Refusing to replace an existing evidence bundle")
    join = json.loads((evidence / "figure8_identity_join.json").read_text())
    report = json.loads((evidence / "cosim_evidence_snapshot.json").read_text())
    inventory = json.loads(historical_manifest.read_text())
    wanted = {r["source_result"] for r in join["rows"] if r.get("validation_id")}
    by_path = {r["resolved_result"]: r for r in inventory["rows"] if r["resolved_result"] in wanted}
    fail(set(by_path) == wanted, "Source inventory is incomplete")
    source_audit = {}
    for path, row in by_path.items():
        raw = Path(path).read_bytes()
        kernel = json.loads(raw)["hls_code"].encode()
        fail(sha(raw) == row["source_result_sha256"], f"Source result changed: {path}")
        fail(sha(kernel) == row["kernel_sha256"], f"Kernel changed: {path}")
        source_audit[path] = {"source_result_sha256": sha(raw), "kernel_sha256": sha(kernel)}
    ids = {r["validation_id"] for r in join["rows"] if r.get("validation_id")}
    measurements = {}
    for v in report["validations"]:
        if (v["cohort"] == "historical_1h" and v["validation_id"] in ids) or (
            v["cohort"] == "replay_3h" and v["original_validation_id"] in ids and v["status"] == "pass"
        ):
            raw = Path(v["measurement_path"]).read_bytes()
            fail(sha(raw) == v["measurement_sha256"], f"Measurement changed: {v['validation_id']}")
            measurements[v["validation_id"]] = {"sha256": sha(raw), "raw_text": raw.decode()}
    destination.mkdir(parents=True)
    for name in FILES[:3]:
        (destination / name).write_bytes((evidence / name).read_bytes())
    dump(destination / "inventory.json", {"source_manifest_sha256": sha(historical_manifest.read_bytes()), "rows": list(by_path.values())})
    dump(destination / "source_audit.json", source_audit)
    dump(destination / "measurements.json", measurements)
    dump(destination / "manifest.json", {
        "schema": "c2hls.figure8-rtl-evidence.v1", "captured_at": join["captured_at"],
        "files": {name: sha((destination / name).read_bytes()) for name in FILES},
        "source_result_bytes_reverified": len(source_audit), "measurement_bytes_reverified": len(measurements),
        "historical_manifest_sha256": sha(historical_manifest.read_bytes()),
    })
    load_bundle(destination)


def load_bundle(root: Path):
    manifest = json.loads((root / "manifest.json").read_text())
    fail(manifest.get("schema") == "c2hls.figure8-rtl-evidence.v1", "Unsupported evidence schema")
    fail(set(manifest["files"]) == set(FILES), "Evidence files missing or unexpected")
    for name, expected in manifest["files"].items():
        fail(sha((root / name).read_bytes()) == expected, f"Evidence hash mismatch: {name}")
    bundle = {name.removesuffix(".json"): json.loads((root / name).read_text()) for name in FILES if name.endswith(".json")}
    bundle["source_rows"] = list(csv.DictReader(io.StringIO((root / FILES[0]).read_text())))
    bundle["source_sha256"] = sha((root / FILES[0]).read_bytes())
    bundle["manifest"] = manifest
    return bundle


def key(row):
    return row["model"], row["workflow"], row["benchmark"]


def positive_cycles(value, label):
    fail(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0 and int(value) == value, f"Invalid RTL cycles: {label}")
    return int(value)


def validate_measurement(v, measurements):
    entry = measurements.get(v["validation_id"])
    fail(entry is not None, f"Missing original measurement: {v['validation_id']}")
    raw = entry["raw_text"].encode()
    fail(sha(raw) == entry["sha256"] == v["measurement_sha256"], "Measurement digest disagreement")
    m = json.loads(raw)
    fail(m["validation_id"] == v["validation_id"] and m["kernel_sha256"] == v["kernel_sha256"], "Measurement identity disagreement")
    fail(m["toolchain"] == v["toolchain"], "Measurement toolchain disagreement")
    c = m["cosim"]
    fail(c["status"] == v["status"], "Measurement status disagreement")
    if v["status"] == "pass":
        fail(c.get("ran") is True and c.get("passed") is True and c.get("success") is True, "Incomplete RTL pass")
        fail(c.get("correctness", {}).get("passed") is True and c["correctness"].get("rtl_postcheck_dump_count", 0) > 0, "No independent RTL output check")
        fail(m["csim"].get("passed") is True and m["csim"].get("correctness", {}).get("passed") is True, "No independent C pass")
        fail(c.get("target_code_sha256") == v["kernel_sha256"], "RTL source hash disagreement")
        fail(positive_cycles(c["kernel_runtime_cycles"], v["validation_id"]) == positive_cycles(v["rtl_cycles"], v["validation_id"]), "RTL cycle disagreement")
    return m


def build_rows(bundle):
    source = bundle["source_rows"]
    join = bundle["figure8_identity_join"]
    fail(join["source_figure_sha256"] == bundle["source_sha256"], "Selection CSV differs from audited selection")
    expected = {(m, mode, bench) for m, mode in GROUPS for bench in BENCHMARKS}
    source_map = {key(r): r for r in source}
    fail(len(source) == len(source_map) == len(expected) and set(source_map) == expected, "Incomplete or duplicate 108-design selection")
    arms = {(key(r), r["arm"]): r for r in join["rows"]}
    fail(len(arms) == len(join["rows"]) == 216 and set(arms) == {(k, arm) for k in expected for arm in ("selected", "off")}, "Missing or duplicate arm identity")
    inventory = {r["resolved_result"]: r for r in bundle["inventory"]["rows"]}
    report = bundle["cosim_evidence_snapshot"]
    limits = {c["id"]: c["timeout_seconds"] for c in report["cohorts"]}
    fail(limits.get("historical_1h") == 3600 and limits.get("replay_3h") == 10800, "Validation limits differ from figure policy")
    one = {v["validation_id"]: v for v in report["validations"] if v["cohort"] == "historical_1h"}
    replay = {v["original_validation_id"]: v for v in report["validations"] if v["cohort"] == "replay_3h"}
    fail(len(one) == sum(v["cohort"] == "historical_1h" for v in report["validations"]), "Duplicate original validation")
    fail(len(replay) == sum(v["cohort"] == "replay_3h" for v in report["validations"]), "Duplicate replay validation")
    output = []
    for model, mode in GROUPS:
        for benchmark in BENCHMARKS:
            k = model, mode, benchmark
            original = source_map[k]
            fail(original["selected_skill_mode"] in SKILL_MODES, "Unknown selected policy")
            row = {"model": model, "workflow": mode, "benchmark": benchmark,
                   "selected_policy": original["selected_skill_mode"], "raw_recorded_model_revision": original["raw_recorded_model_revision"],
                   "selected_synthesis_cycles": positive_cycles(float(original["selected_worst_case_cycles"]), str(k)),
                   "off_synthesis_cycles": positive_cycles(float(original["no_skills_worst_case_cycles"]), str(k))}
            # Both estimates come from the same historical selection record, even
            # where the original kernel needed for later RTL validation is absent.
            row["synthesis_ratio"] = row["off_synthesis_cycles"] / row["selected_synthesis_cycles"]
            fail(math.isclose(row["synthesis_ratio"], float(original["no_skills_over_selected"]), rel_tol=1e-12),
                 "Historical synthesis ratio disagrees with the frozen selection")
            valid = {}
            for arm, path_key in (("selected", "selected_source_result_path"), ("off", "no_skills_source_result_path")):
                a = arms[k, arm]
                fail(a["source_result"] == original[path_key], "Plotted design was reselected or relinked")
                status = a["status_1h"]
                fail(status in STATUS_CODE, f"Unknown validation state: {status}")
                row.update({f"{arm}_status_1h": status, f"{arm}_validation_id": a.get("validation_id", ""),
                            f"{arm}_source_result": a["source_result"], f"{arm}_identity_reason": a.get("reason", ""),
                            f"{arm}_cycles_1h": None, f"{arm}_cycles_with_replay": None,
                            f"{arm}_status_with_replay": status, f"{arm}_reference_audit_passed": None})
                if status == "blocked_identity":
                    fail(not a.get("validation_id") and a.get("reason"), "Unresolved identity must stay unmeasured")
                    continue
                r = inventory.get(a["source_result"])
                fail(r is not None, "Missing source inventory record")
                audit = bundle["source_audit"].get(a["source_result"], {})
                fail(audit.get("source_result_sha256") == r["source_result_sha256"] and audit.get("kernel_sha256") == r["kernel_sha256"], "Missing source byte audit")
                fail(all(a.get(field) is True for field in ("input_identity_matched", "toolchain_matched", "result_sha256_match", "source_sha256_match")), "Identity audit is incomplete")
                v = one.get(a["validation_id"])
                fail(v is not None and v["status"] == status, "One-hour measurement missing or inconsistent")
                fail(a["validation_id"] == r["validation_id"] and a["kernel_sha256"] == r["kernel_sha256"] == v["kernel_sha256"], "Kernel/validation join disagrees")
                fail(r["input_hashes"] == v["input_hashes"] and r["historical_toolchain"] == v["toolchain"], "Validation input or toolchain differs")
                fail(float(r["csynth_cycles"]) == row[f"{arm}_synthesis_cycles"], "Historical selected estimate changed")
                validate_measurement(v, bundle["measurements"])
                row[f"{arm}_reference_audit_passed"] = r["historical_reference_audit_passed"]
                valid[arm] = v
                if status == "pass":
                    row[f"{arm}_cycles_1h"] = row[f"{arm}_cycles_with_replay"] = positive_cycles(v["rtl_cycles"], str(k))
                z = replay.get(a["validation_id"])
                if z and z["status"] == "pass":
                    fail(status == "timeout", "Replay must originate from a timed-out design")
                    fail(z["kernel_sha256"] == v["kernel_sha256"] and z["input_hashes"] == v["input_hashes"] and z["toolchain"] == v["toolchain"], "Replay identity differs")
                    validate_measurement(z, bundle["measurements"])
                    row[f"{arm}_cycles_with_replay"] = positive_cycles(z["rtl_cycles"], str(k))
                    row[f"{arm}_status_with_replay"] = "pass_replay"
                fail(row[f"{arm}_status_with_replay"] == a["status_with_replay"], "Sensitivity status differs from audit")
            if len(valid) == 2:
                fail(valid["selected"]["input_hashes"] == valid["off"]["input_hashes"] and valid["selected"]["toolchain"] == valid["off"]["toolchain"], "Numerator and denominator are incomparable")
            row["synthesis_identity_linked"] = len(valid) == 2
            for scope in ("1h", "with_replay"):
                numerator, denominator = row[f"off_cycles_{scope}"], row[f"selected_cycles_{scope}"]
                row[f"rtl_ratio_{scope}"] = numerator / denominator if numerator is not None and denominator is not None else None
            output.append(row)
    return output


def summarize(rows):
    groups = []
    for model, mode in GROUPS:
        rr = [r for r in rows if r["model"] == model and r["workflow"] == mode]
        g = {"model": model, "workflow": mode, "total": len(rr),
             "recorded_synthesis_pairs": sum(r["synthesis_ratio"] is not None for r in rr),
             "identity_linked_synthesis_pairs": sum(r["synthesis_identity_linked"] for r in rr)}
        for arm in ("selected", "off"):
            g[f"{arm}_statuses_1h"] = dict(Counter(r[f"{arm}_status_1h"] for r in rr))
            g[f"{arm}_pass_with_replay"] = sum(r[f"{arm}_cycles_with_replay"] is not None for r in rr)
        for scope in ("1h", "with_replay"):
            ratios = [r[f"rtl_ratio_{scope}"] for r in rr if r[f"rtl_ratio_{scope}"] is not None]
            g[f"paired_{scope}"] = len(ratios)
            g[f"conditional_geomean_{scope}"] = math.exp(sum(math.log(r) for r in ratios) / len(ratios)) if ratios else None
            g[f"same_design_pairs_{scope}"] = sum(r[f"rtl_ratio_{scope}"] is not None and r["selected_validation_id"] == r["off_validation_id"] for r in rr)
        groups.append(g)
    return {"groups": groups,
            "recorded_synthesis_pairs": sum(r["synthesis_ratio"] is not None for r in rows),
            "identity_linked_synthesis_pairs": sum(r["synthesis_identity_linked"] for r in rows),
            "selected_pass_1h": sum(r["selected_status_1h"] == "pass" for r in rows),
            "off_pass_1h": sum(r["off_status_1h"] == "pass" for r in rows),
            "strict_wins_1h": sum(r["rtl_ratio_1h"] is not None and r["rtl_ratio_1h"] > 1 for r in rows),
            "ties_1h": sum(r["rtl_ratio_1h"] == 1 for r in rows),
            "selected_pass_with_replay": sum(r["selected_cycles_with_replay"] is not None for r in rows),
            "paired_1h": sum(r["rtl_ratio_1h"] is not None for r in rows),
            "paired_with_replay": sum(r["rtl_ratio_with_replay"] is not None for r in rows),
            "selected_total": len(rows), "identity_unresolved": sum(r["off_status_1h"] == "blocked_identity" for r in rows)}


def ratio_bar(ax, x, ratio, color):
    """A missing measurement has no artist; a measured tie has a hollow marker."""
    if ratio is None:
        return
    fail(math.isfinite(ratio) and ratio > 0, "Invalid cycle ratio")
    height = math.log2(ratio)
    if ratio == 1:
        ax.scatter(x, height, facecolors="white", edgecolors=color,
                   marker="o", linewidths=.65, s=8, zorder=5, clip_on=False)
    else:
        ax.bar(x, height, width=BAR_WIDTH, color=color, edgecolor="0.2", linewidth=.25, zorder=3)


def plot(rows, summary, path):
    font_manager.findfont("Times New Roman", fallback_to_default=False)
    plt.rcParams.update({"font.family": "Times New Roman", "font.size": 6.2, "axes.labelsize": 6.2,
                         "pdf.fonttype": 42, "ps.fonttype": 42, "svg.hashsalt": "c2hls-figure8-rtl-v1"})
    figure = plt.figure(figsize=(6.4, 4.5))
    grid = figure.add_gridspec(8, 1, height_ratios=[3.7, 1.0] * 4, hspace=0.14,
                              left=0.11, right=0.994, bottom=0.125, top=0.904)
    for panel, ((model, mode), group) in enumerate(zip(GROUPS, summary["groups"])):
        rr = [r for r in rows if r["model"] == model and r["workflow"] == mode]
        ax = figure.add_subplot(grid[2 * panel])
        status_ax = figure.add_subplot(grid[2 * panel + 1], sharex=ax)
        for x, row in enumerate(rr):
            for field, _, color, offset in METRICS:
                ratio_bar(ax, x + offset, row[field], color)
            for y, arm in ((1, "selected"), (0, "off")):
                status = row[f"{arm}_status_1h"]
                status_ax.text(x, y, STATUS_CODE[status], ha="center", va="center", fontsize=5.5,
                               color=STATUS_COLOR[status], fontweight="bold" if status != "pass" else "normal")
        ax.set_xlim(-.7, 26.7)
        logs = [math.log2(r[field]) for r in rows for field, _, _, _ in METRICS if r[field] is not None]
        lower = min(0, 2 * math.floor(min(logs) / 2))
        upper = max(10, 2 * math.ceil(max(logs) / 2))
        ticks = tuple(range(lower, upper + 1, 2))
        # Keep panel headings above the tallest historical synthesis bars.
        ax.set_ylim(lower - .32, upper + 2)
        ax.set_yticks(ticks, [f"{2 ** tick:g}" for tick in ticks])
        ax.tick_params(axis="x", bottom=False, labelbottom=False)
        ax.tick_params(axis="y", labelsize=5.4, length=2, pad=2)
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="0.86", linewidth=.35)
        ax.axhline(0, color="0.35", linewidth=.6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_facecolor("#FAFAFA")
        ax.text(.006, .97, f"{chr(97 + panel)}) {model} · {'Flash' if mode == 'flash' else 'Multistep'}", transform=ax.transAxes,
                fontsize=6.3, weight="bold", va="top", bbox={"facecolor": "#FAFAFA", "edgecolor": "none", "pad": .2})
        ax.text(.99, .97, f"{group['paired_1h']}/27 RTL pairs", transform=ax.transAxes,
                fontsize=5.8, va="top", ha="right", bbox={"facecolor": "#FAFAFA", "edgecolor": "none", "pad": .2})
        status_ax.set_ylim(-.6, 1.65)
        status_ax.set_yticks([1, 0], ["Selected", "No recipes"], fontsize=5.5)
        status_ax.tick_params(axis="both", length=0, pad=2)
        status_ax.spines[:].set_visible(False)
        status_ax.set_facecolor("#F4F4F4")
        status_ax.set_xticks(range(27))
        if panel == 3:
            status_ax.set_xticklabels([b.removeprefix("hlsfactory_").replace("_", "-") for b in BENCHMARKS], rotation=56, ha="right", rotation_mode="anchor", fontsize=6.0)
        else:
            status_ax.tick_params(labelbottom=False)
    figure.supylabel("No-recipes / selected cycles (log scale)", x=.013, fontsize=7)
    figure.legend(handles=[Patch(facecolor=color, edgecolor="0.3", linewidth=.3, label=label)
                           for _, label, color, _ in METRICS],
                  loc="upper center", bbox_to_anchor=(.52, .994), ncol=2, frameon=False, fontsize=6.2, columnspacing=1.4, handlelength=1.15)
    figure.text(.54, .940, "Fixed 1 h limit · P: pass   T: timeout   F: validation failure   U: unresolved identity", ha="center", fontsize=6.1)
    path.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "svg", "png"):
        target = path.with_suffix("." + suffix)
        metadata = {"Creator": "generate_skill_v3_rtl.py"}
        if suffix == "pdf":
            metadata.update({"Title": "CSynth and COSim cycle ratios for fixed synthesis-selected designs", "Author": "Anonymous", "CreationDate": None, "ModDate": None})
        elif suffix == "svg":
            metadata["Date"] = None
        figure.savefig(target, dpi=220, metadata=metadata)
    plt.close(figure)


def generate(evidence_dir: Path, root: Path):
    bundle = load_bundle(evidence_dir)
    rows = build_rows(bundle)
    summary = summarize(rows)
    generated = root / "generated/sigmetrics"
    generated.mkdir(parents=True, exist_ok=True)
    snapshot = generated / "skill_v3_rtl_snapshot.csv"
    with snapshot.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary_path = generated / "skill_v3_rtl_summary.json"
    dump(summary_path, summary)
    macros = generated / "skill_v3_rtl_macros.tex"
    macros.write_text("% Frozen RTL evidence; generator: scripts/generate_skill_v3_rtl.py\n" + "\n".join(
        f"\\newcommand{{\\SkillVThreeRTL{name}}}{{{value}}}" for name, value in (
            ("SelectedPass", summary["selected_pass_1h"]), ("SelectedTotal", summary["selected_total"]),
            ("Slots", summary["selected_total"]), ("OffPass", summary["off_pass_1h"]),
            ("Wins", summary["strict_wins_1h"]), ("Ties", summary["ties_1h"]),
            ("PairedPass", summary["paired_1h"]), ("ReplaySelectedPass", summary["selected_pass_with_replay"]),
            ("ReplayPairedPass", summary["paired_with_replay"]), ("Unresolved", summary["identity_unresolved"]))) + "\n")
    table = generated / "skill_v3_rtl_coverage_table.tex"
    lines = [r"\begin{tabular}{@{}llrrrr@{}}", r"\toprule", r"& & \multicolumn{2}{c}{Fixed 1 h} & \multicolumn{2}{c}{With 3 h retries} \\",
             r"Model & Mode & Selected & Pairs & Selected & Pairs \\", r"\midrule"]
    for group in summary["groups"]:
        lines.append(f"{group['model']} & {'Flash' if group['workflow'] == 'flash' else 'Multistep'} & {group['selected_statuses_1h'].get('pass', 0)}/27 & {group['paired_1h']}/27 & {group['selected_pass_with_replay']}/27 & {group['paired_with_replay']}/27 " + r"\\")
    lines.extend([r"\midrule", f"Total & & {summary['selected_pass_1h']}/108 & {summary['paired_1h']}/108 & {summary['selected_pass_with_replay']}/108 & {summary['paired_with_replay']}/108 " + r"\\", r"\bottomrule", r"\end{tabular}"])
    table.write_text("\n".join(lines) + "\n")
    figure = root / "figures/sigmetrics/skill_v3_rtl.pdf"
    plot(rows, summary, figure)
    outputs = [snapshot, summary_path, macros, table] + [figure.with_suffix("." + s) for s in ("pdf", "svg", "png")]
    dump(generated / "skill_v3_rtl_manifest.json", {
        "schema": "c2hls.figure8-rtl.v1", "captured_at": bundle["manifest"]["captured_at"],
        "evidence_manifest_sha256": sha((evidence_dir / "manifest.json").read_bytes()),
        "evidence_bundle": str(evidence_dir.relative_to(root)) if evidence_dir.is_relative_to(root) else str(evidence_dir),
        "generator_sha256": sha(Path(__file__).read_bytes()),
        "wrapper_sha256": sha((root / "scripts/generate_skill_v3_open_models.py").read_bytes()),
        "metric": "same-model same-mode no-recipes cycles / fixed synthesis-selected cycles, computed separately within CSynth and COSim",
        "metrics": {"CSynth": "historical synthesis estimate ratio", "COSim": "independently output-checked measured RTL cycle ratio"},
        "font": {"family": "Times New Roman", "path": font_manager.findfont("Times New Roman", fallback_to_default=False)},
        "figure_inches": [6.4, 4.5],
        "bar_width": BAR_WIDTH, "pair_outer_width": .32 + BAR_WIDTH,
        "primary_limit_seconds": 3600, "sensitivity_limit_seconds": 10800,
        "timeout_scope": "entire historical Vitis invocation, including fresh synthesis and co-simulation; not a cosim_design-only limit",
        "summary": summary, "outputs": {str(p.relative_to(root)): sha(p.read_bytes()) for p in outputs},
        "evidence_boundaries": [
            "108 designs were selected by the existing post-hoc best-of-five synthesis rule, not reselected using RTL results.",
            "CSynth ratios use the 108 frozen historical synthesis-record pairs; 103 pairs are identity-linked to backfill artifacts and five retain unresolved no-recipes identities, marked U in the status strip.",
            "COSim ratios are plotted only when both independently checked designs pass under the same one-hour limit. Missing ratios have separate selected/comparator statuses and are never plotted as zero.",
            "Three-hour retry passes appear only in the coverage sensitivity table/CSV, never in primary plotted bars.",
            "The upstream HLS reference synthesis estimates are not used as RTL denominators.",
            "The 27-kernel cohort excludes Doitgen, which was not run in the source sweep.",
            "All linked historical designs retain failed advisory reference-isolation audits. Co-simulation validates tested outputs, not isolation or general functional equivalence.",
            "Distinct Gemma serving revisions and the four workflows remain separate; these are not matched-budget causal recipe comparisons.",
            "Conditional geomeans describe surviving pairs only and are not aggregate speedups across all 27 kernels.",
            "Five no-recipes identities are unresolved and remain unmeasured; their failures are not replaced by new candidates.",
        ],
    })
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    imp = sub.add_parser("import")
    imp.add_argument("--evidence", type=Path, required=True)
    imp.add_argument("--historical-manifest", type=Path, required=True)
    imp.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    import_bundle(args.evidence, args.historical_manifest, args.destination)
