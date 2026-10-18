#!/usr/bin/env python3
"""Audit frozen-parent reviewer A/B outcomes without imputing missing cycles."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from evaluation_repro import fingerprint_matches
from phase_b_manifest import canonical_json_sha256, file_sha256, text_sha256
from qor_design_space import extract_qor_metrics
from analyze_qor_paired_model_campaign import export_rows


def analyze_cell(cell, config, seeds):
    row = {k: cell[k] for k in ("cell_id", "benchmark", "strategy", "arm", "repeat", "pair_id", "status")}
    row.update(audit_ok=False, feasible=False, cycles=None, issues=[], result_path=cell["result_path"])
    path = Path(cell["result_path"])
    if not path.is_file():
        return row
    result = json.loads(path.read_text())
    seed = seeds["entries"][cell["benchmark"]]
    provenance = result.get("frozen_phase_b") or {}
    fingerprint = result.get("run_fingerprint") or {}
    payload = fingerprint.get("payload") or {}
    search = payload.get("search") or {}
    budgets = payload.get("budgets") or {}
    usage = result.get("llm_usage") or {}
    synthesis = result.get("synthesis_evaluations") or {}
    history_path = path.with_name(cell["benchmark"] + "_history.json")
    history = json.loads(history_path.read_text()) if history_path.is_file() else {}
    events = history.get("event_history") or []
    prompts = "\n".join(e.get("content", "") for e in events if e.get("role") == "user")
    reviews = result.get("qor_profile_reviews") or []
    on = cell["arm"] == "review_on"
    issues = []
    if not fingerprint_matches(fingerprint, fingerprint):
        issues.append("fingerprint digest invalid")
    if search.get("qor_profile_review") is not on or search.get("sample_id") != cell["cell_id"]:
        issues.append("review arm or independent sample identity mismatch")
    if search.get("qor_design_sweep") or result.get("qor_synthesis_evaluation_count", 0):
        issues.append("parameter search unexpectedly executed")
    design = result.get("qor_design_sweep") or {}
    if design.get("enabled") or design.get("attempted") or design.get("candidates"):
        issues.append("parameter-search artifact unexpectedly present")
    for field in ("code_sha256", "csynth_report_sha256", "csim_sha256", "input_c_sha256", "header_sha256"):
        if provenance.get(field) != seed[field]:
            issues.append("frozen baseline identity mismatch: " + field)
    if provenance.get("manifest_sha256") != file_sha256(Path(config["phase_b_manifest"])):
        issues.append("frozen seed manifest mismatch")
    if canonical_json_sha256(result.get("baseline_report")) != seed["csynth_report_sha256"]:
        issues.append("starting report changed")
    if canonical_json_sha256(result.get("baseline_csim")) != seed["csim_sha256"]:
        issues.append("starting CSim changed")
    if (result.get("reference_isolation_audit") or {}).get("passed") is not True:
        issues.append("reference isolation missing or failed")
    if (result.get("skill_snapshot_integrity") or {}).get("unchanged") is not True:
        issues.append("skill snapshot changed")
    observed_models = {e.get("model") for e in usage.get("events", []) if e.get("model")}
    if observed_models != {config["model"]}:
        issues.append("observed models do not match registration")
    if usage.get("calls", 0) > 5 or synthesis.get("count", 0) > 5:
        issues.append("five-call or five-synthesis budget exceeded")
    if str(budgets.get("llm_candidate_budget")) != "5" or str(budgets.get("candidate_budget")) != "5":
        issues.append("registered budget mismatch")
    if any(e.get("label") == "[Phase B]" for e in synthesis.get("events", [])):
        issues.append("baseline was regenerated instead of replayed")
    if result.get("selected_winner_cosim_count", 0) or (result.get("cosim") or {}).get("ran"):
        issues.append("COSIM unexpectedly executed")
    guidance_present = "HARDWARE/PERFORMANCE DESIGN REVIEW:" in prompts
    feedback_present = "QoR REVIEW (profile_skill_tradeoff_v1)" in prompts
    if guidance_present is not on:
        issues.append("resource-tradeoff prompt does not match arm")
    if not on and (reviews or feedback_present):
        issues.append("reviewer evidence leaked into control arm")
    if on and any(r.get("review_version") != config["review_version"] for r in reviews):
        issues.append("reviewer version mismatch")
    metrics = extract_qor_metrics(result.get("final_report"))
    cycles = metrics.get("latency_cycles_worst")
    feasible = bool(result.get("success") and (result.get("candidate_feasibility") or {}).get("feasible")
                    and (result.get("csim") or {}).get("passed") is True
                    and isinstance(cycles, (float, int)) and math.isfinite(cycles) and cycles > 0)
    row.update(audit_ok=not issues, feasible=feasible and not issues, issues=issues,
               cycles=cycles if feasible and not issues else None,
               phase_b_cycles=seed["csynth_report"].get("latency_cycles_worst", seed["csynth_report"]["latency_cycles"]),
               seed_code_sha256=seed["code_sha256"], selected_code_sha256=text_sha256(result.get("hls_code") or ""),
               baseline_retained=text_sha256(result.get("hls_code") or "") == seed["code_sha256"],
               llm_calls=usage.get("calls"), input_tokens=usage.get("input_tokens"), output_tokens=usage.get("output_tokens"),
               synthesis_calls=synthesis.get("count"), review_count=len(reviews),
               feedback_in_prompt=feedback_present, resource_guidance_in_prompt=guidance_present,
               explicit_review_retries=sum("Retrying with measured skill/QoR evidence." in e.get("content", "") for e in events),
               feasible_generated_candidates=sum(e.get("success") is True and e.get("resource_fit") is True and e.get("timing_met") is True for e in synthesis.get("events", [])),
               unsupported_skill_declarations=sum(len(r.get("unsupported_declarations") or []) for r in reviews),
               fingerprint_sha256=fingerprint.get("sha256"), result_success=result.get("success"))
    row.update({k: metrics.get(k) for k in ("dsp", "bram", "lut", "ff", "uram", "estimated_clock_period_ns")})
    return row


def pair_rows(rows):
    grouped = defaultdict(dict)
    for row in rows:
        grouped[row["pair_id"]][row["arm"]] = row
    pairs = []
    for pair_id, arms in grouped.items():
        off, on = arms.get("review_off", {}), arms.get("review_on", {})
        template = off or on
        valid = bool(off.get("feasible") and on.get("feasible")
                     and off.get("seed_code_sha256") == on.get("seed_code_sha256"))
        ratio = off["cycles"] / on["cycles"] if valid else None
        pairs.append({"pair_id": pair_id, "benchmark": template["benchmark"], "strategy": template["strategy"],
                      "repeat": template["repeat"], "valid_pair": valid,
                      "off_cycles": off.get("cycles"), "on_cycles": on.get("cycles"),
                      "speedup_on_over_off": ratio,
                      "effect": "improved" if valid and ratio > 1 + 1e-12 else "regressed" if valid and ratio < 1 - 1e-12 else "unchanged" if valid else "unpaired_or_invalid"})
    return pairs


def summary(rows, pairs):
    import numpy as np
    groups = defaultdict(list)
    for pair in pairs:
        if pair["valid_pair"]:
            groups[pair["benchmark"]].append(math.log(pair["speedup_on_over_off"]))
    cluster_means = [sum(v) / len(v) for v in groups.values()]
    interval = None
    if len(cluster_means) >= 2:
        rng = np.random.default_rng(42)
        draws = rng.choice(cluster_means, size=(10000, len(cluster_means)), replace=True).mean(axis=1)
        interval = np.exp(np.quantile(draws, [0.025, 0.975])).tolist()
    return {"planned_runs": len(rows), "audit_valid_runs": sum(r["audit_ok"] for r in rows),
            "feasible_runs": sum(r["feasible"] for r in rows), "planned_pairs": len(pairs),
            "valid_pairs": sum(p["valid_pair"] for p in pairs), "benchmark_clusters": len(groups),
            "effects": dict(Counter(p["effect"] for p in pairs)),
            "geomean_speedup_on_over_off": math.exp(sum(cluster_means) / len(cluster_means)) if cluster_means else None,
            "benchmark_cluster_bootstrap_95_interval": interval,
            "arm_counts": {arm: {"audited": sum(r["audit_ok"] for r in rows if r["arm"] == arm),
                                 "feasible": sum(r["feasible"] for r in rows if r["arm"] == arm)}
                           for arm in ("review_off", "review_on")}}


def write_report(config):
    root = Path(config["report_root"])
    state = json.loads((root / "campaign_state.json").read_text())
    rows = [{**c["measurement"], "status": c["status"]} if c.get("measurement") else
            {**{k:c[k] for k in ("cell_id", "benchmark", "strategy", "arm", "repeat", "pair_id", "status")},
             "audit_ok": False, "feasible": False, "cycles": None, "issues": []}
            for c in state["cells"]]
    pairs = pair_rows(rows)
    stats = summary(rows, pairs)
    export_rows(root / "arm_measurements", rows)
    export_rows(root / "paired_measurements", pairs)
    (root / "summary.json").write_text(json.dumps(stats, indent=2, allow_nan=False) + "\n")
    lines = ["# Frozen-Parent QoR Reviewer A/B", "", f"State: **{state['status']}**; audited runs **{stats['audit_valid_runs']}/{len(rows)}**; valid pairs **{stats['valid_pairs']}/{len(pairs)}**.", "",
             "Both arms use deepseek-flash, identical measured Phase-B parents, matched skills, five-call/five-synthesis caps, Vitis 2023.2, U280, 3.33 ns. No parameter search or COSIM. Flash has two attempts in both arms so review feedback can reach a follow-up. Three independent API samples are not seed-controlled replicas; the API alias is not immutable.", "",
             "A=review_off (existing generic feedback); B=review_on (skill/profile review plus hardware-tradeoff guidance). This tests the bundled change, not either component in isolation. Dynamic routes may diverge after code changes; library, initial parent and routing policy are fixed. Invalid or missing outcomes have no imputed latency. Frozen-parent fallback is identified separately.", "",
             f"Benchmark-weighted geomean B speedup over A: **{stats['geomean_speedup_on_over_off']}**; clustered 95% interval: {stats['benchmark_cluster_bootstrap_95_interval']}. Partial-run statistics remain provisional.", "",
             "| Benchmark | Mode | Sample | A cycles | B cycles | B speedup | Effect |",
             "|---|---|---:|---:|---:|---:|---|"]
    for p in pairs:
        lines.append(f"| {p['benchmark']} | {p['strategy']} | {p['repeat']} | {p['off_cycles'] or 'N/A'} | {p['on_cycles'] or 'N/A'} | {format(p['speedup_on_over_off'], '.4f') if p['valid_pair'] else 'N/A'} | {p['effect']} |")
    (root / "report.md").write_text("\n".join(lines) + "\n")
    schema_lines = []
    for c in state["cells"]:
        if c.get("measurement", {}).get("audit_ok") and Path(c["schema_jsonl"]).is_file():
            schema_lines.extend(Path(c["schema_jsonl"]).read_text().splitlines())
    (root / "schema_records.jsonl").write_text("".join(l + "\n" for l in schema_lines if l))
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(write_report(json.loads(args.config.read_text()))))
