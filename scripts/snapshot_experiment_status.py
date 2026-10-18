#!/usr/bin/env python3
"""Read saved experiment evidence; write a separate, timestamped status snapshot.

No framework imports, API calls, HLS launches, signals, or campaign-file writes.
Live files are sampled, not transactionally locked. Each snapshot records its
read window and the controller's own timestamps; measurements win over a stale
active list. Historical rows and distinct validation identities stay separate.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_CAMPAIGN = REPO / "results_sweeps/july_cosim_backfill_20260928"
CAMPAIGNS = (
    "deepseek_flash_reviewer_ab27_20260925",
    "deepseek_flash_qor_profile27_20260923",
    "deepseek_v4_pro_qor_paired27_axi_v2_20260908",
)


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    if not rows:
        path.write_text("")
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def command(args):
    try:
        proc = subprocess.run(args, cwd=REPO, capture_output=True, text=True, timeout=15)
        return {"returncode": proc.returncode, "stdout": proc.stdout.strip(),
                "stderr": proc.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"error": str(exc)}


def registry_match(row, registry):
    entry = registry.get(row["benchmark"])
    if not entry:
        return "missing_benchmark"
    recorded = row.get("input_hashes", {}).get("testbench.cpp")
    if not recorded:
        return "missing_recorded_testbench_hash"
    return "match" if entry.get("testbench_sha256") == recorded else "testbench_hash_mismatch"


def failure_kind(measurement):
    cosim = measurement.get("cosim", {})
    if cosim.get("status") == "pass":
        return ""
    reason = cosim.get("skip_reason", "")
    if reason == "invalid_native_golden":
        return "oracle_or_registry_error"
    if reason == "candidate_csim_failed":
        correctness = measurement.get("csim", {}).get("correctness", {})
        return "candidate_csim:" + str(correctness.get("reason") or "unclassified")
    return cosim.get("status", "unknown") + ":" + str(reason or cosim.get("error") or "unclassified")


def snapshot(output, campaign):
    started = now()
    manifest_path = campaign / "manifest.json"
    manifest = read(manifest_path)
    state = read(campaign / "state.json")
    reported = read(campaign / "summary.json")
    raw = Path(manifest["raw_root"])
    measurements = {p.stem: read(p) for p in sorted((raw / "measurements").glob("*.json"))}
    rows = manifest["rows"]
    queued = {r["validation_id"] for r in rows if r["inventory_status"] == "queued"}
    known = {r["validation_id"] for r in rows if r.get("validation_id")}
    finished = queued & measurements.keys()
    active = (set(state.get("active", [])) & queued) - finished
    pending = queued - finished - active
    defaults = read(REPO / "configs/hlsfactory_output_shapes.json")["benchmarks"]
    imported = read(REPO / "configs/hlsfactory_output_shapes_fc27133.json")["benchmarks"]
    integrity = {name: {"expected": value, "observed": sha(REPO / name)}
                 for name, value in manifest["code_hashes"].items()}
    for value in integrity.values():
        value["matches"] = value["expected"] == value["observed"]

    records, by_id = [], defaultdict(list)
    for row in rows:
        key = row.get("validation_id")
        measurement = measurements.get(key, {})
        status = row["inventory_status"]
        if row.get("existing_cosim"):
            status = "existing_cosim"
        elif measurement:
            status = measurement["cosim"]["status"]
        elif key in active:
            status = "active"
        elif status == "queued":
            status = "pending"
        cosim = row.get("existing_cosim") or measurement.get("cosim", {})
        record = {k: row.get(k) for k in (
            "record_id", "campaign", "benchmark", "model", "strategy", "skill_mode",
            "validation_id", "kernel_sha256", "inventory_status", "reason", "csynth_cycles",
            "historical_reference_audit_passed", "source_result", "source_result_sha256")}
        record.update(status=status, failure_kind=failure_kind(measurement) if measurement else "",
                      error=cosim.get("error"),
                      measured_cosim_cycles=cosim.get("kernel_runtime_cycles") if cosim.get("passed") else None,
                      default_registry=registry_match(row, defaults),
                      imported_registry=registry_match(row, imported),
                      measurement_path=str(raw / "measurements" / f"{key}.json") if measurement else "")
        records.append(record)
        if key in queued:
            by_id[key].append(record)

    validations = []
    for key, group in sorted(by_id.items()):
        measurement = measurements.get(key, {})
        first = group[0]
        validations.append({
            "validation_id": key, "historical_rows": len(group),
            "benchmarks": ";".join(sorted({r["benchmark"] for r in group})),
            "models": ";".join(sorted({r["model"] or "unknown" for r in group})),
            "status": first["status"], "failure_kind": first["failure_kind"],
            "error": first["error"], "default_registry": first["default_registry"],
            "imported_registry": first["imported_registry"],
            "started_at": measurement.get("started_at"), "finished_at": measurement.get("finished_at"),
            "runtime_seconds": measurement.get("cosim", {}).get("runtime_seconds"),
            "measured_cosim_cycles": first["measured_cosim_cycles"],
            "measurement_path": first["measurement_path"],
        })
    counts = Counter(r["status"] for r in validations)
    failures = [r for r in validations if r["status"] not in {"pass", "active", "pending"}]
    affected = [r for r in records if r["inventory_status"] == "queued" and r["default_registry"] != "match"]
    coverage = []
    for name in sorted({r["campaign"] for r in records}):
        group = [r for r in records if r["campaign"] == name]
        coverage.append({"campaign": name, "historical_rows": len(group),
                         "distinct_queued_validations": len({r["validation_id"] for r in group if r["inventory_status"] == "queued"}),
                         **{status: sum(r["status"] == status for r in group) for status in (
                             "pass", "existing_cosim", "active", "pending", "not_run", "timeout", "fail", "blocked")}})

    completed, costs = [], []
    for name in CAMPAIGNS:
        root = REPO / "results_sweeps" / name
        table = root / ("arm_measurements.csv" if "reviewer_ab" in name else "paired_measurements.csv")
        summary_path = root / "summary.json"
        completed.append({"campaign": name, "root": str(root), "summary": read(summary_path),
                          "state": {k: v for k, v in read(root / "campaign_state.json").items()
                                    if k in {"status", "updated_at", "finished_at"}},
                          "summary_sha256": sha(summary_path), "measurements_sha256": sha(table),
                          "performance_measure": "CSYNTH estimated cycles; not RTL runtime"})
        for row in read_csv(table):
            entry = {"campaign": name, **{k: row.get(k) for k in (
                "cell_id", "benchmark", "strategy", "arm", "repeat", "status", "audit_ok",
                "pair_valid", "baseline_retained", "feasible_generated_candidates", "llm_calls",
                "input_tokens", "output_tokens", "synthesis_calls", "llm_synthesis_calls",
                "qor_synthesis_calls", "total_synthesis_calls", "result_path")}}
            # Select only measurement fields; do not export prompts or generated code.
            data = read(Path(row["result_path"]))
            cosim = data.get("cosim") or {}
            entry.update(search_elapsed_seconds=data.get("search_elapsed_seconds"),
                         cached_tokens=data.get("llm_usage", {}).get("cached_tokens"),
                         usage_missing_calls=data.get("llm_usage", {}).get("usage_missing_calls"),
                         independent_golden_status=data.get("independent_golden", {}).get("status"),
                         csim_passed=data.get("csim", {}).get("passed"),
                         selected_cosim_measurement_valid=data.get("selected_cosim_measurement_valid"),
                         selected_winner_cosim_count=data.get("selected_winner_cosim_count"),
                         cosim_ran=cosim.get("ran", False),
                         cosim_passed=cosim.get("passed", False),
                         rtl_executed_pass=bool(cosim.get("ran") and cosim.get("passed")
                                                and data.get("selected_cosim_measurement_valid")
                                                and (data.get("selected_winner_cosim_count") or 0) > 0),
                         cost_scope="saved search; frozen-parent construction not amortized; dollars not inferred")
            costs.append(entry)

    tmux = command(["tmux", "list-panes", "-a", "-F",
                    "#{session_name} #{pane_pid} #{pane_current_command} #{pane_dead}"])
    controller = command(["ps", "-p", str(int(state["pid"])), "-o", "pid=,ppid=,etimes=,stat=,comm="])
    pass_times = [float(r["runtime_seconds"]) for r in validations
                  if r["status"] == "pass" and r["runtime_seconds"] is not None]
    result = {
        "schema_version": "c2hls.experiment-status.v1", "read_started_at": started, "read_finished_at": now(),
        "repo_head": command(["git", "rev-parse", "HEAD"]),
        "campaign": str(campaign), "manifest_sha256": sha(manifest_path), "frozen_code_integrity": integrity,
        "tracker_sha256": sha(Path(__file__)),
        "registry_sha256": {name: sha(REPO / "configs" / name) for name in (
            "hlsfactory_output_shapes.json", "hlsfactory_output_shapes_fc27133.json")},
        "controller_state": state, "controller_process": controller, "tmux_panes": tmux,
        "controller_summary": reported,
        "snapshot_semantics": "Non-transactional read of live files; completed measurement overrides active state. Pending means unresolved and not in sampled active state.",
        "historical_rows": len(rows), "historical_campaigns": len(coverage), "row_status_counts": dict(Counter(r["status"] for r in records)),
        "distinct_validations": len(queued), "finished_validations": len(finished),
        "validation_status_counts": dict(counts), "failure_counts": dict(Counter(r["failure_kind"] for r in failures)),
        "excluded_measurements_outside_manifest": sorted(measurements.keys() - known),
        "registry_issue": {"affected_rows": len(affected), "affected_distinct_validations": len({r["validation_id"] for r in affected}),
                           "default_reasons": dict(Counter(r["default_registry"] for r in affected)),
                           "matching_imported_registry_rows": sum(r["imported_registry"] == "match" for r in affected),
                           "unresolved_distinct_validations": len({r["validation_id"] for r in affected} - finished)},
        "historical_audit_counts": dict(Counter(str(r.get("historical_reference_audit_passed")) for r in rows)),
        "pass_runtime_seconds": {"count": len(pass_times), "median": statistics.median(pass_times) if pass_times else None,
                                 "max": max(pass_times) if pass_times else None,
                                 "eta": None, "eta_reason": "Fastest-CSYNTH-first queue, early registry failures, and hour-long timeouts bias observed throughput."},
        "completed_campaigns": completed,
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / "snapshot.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    for filename, data in (("historical_rows.csv", records), ("validations.csv", validations),
                           ("failures.csv", failures), ("registry_affected_rows.csv", affected),
                           ("campaign_coverage.csv", coverage), ("completed_search_costs.csv", costs)):
        write_csv(output / filename, data)
    lines = ["# Experiment status snapshot", "", f"Read window: {started} through {result['read_finished_at']} (UTC).", "",
             f"Historical backfill: {len(finished)}/{len(queued)} distinct validations completed; "
             f"{len(active)} sampled active; {len(pending)} unresolved outside sampled active list.", "",
             "| Validation status | Distinct identities |", "|---|---:|"]
    lines += [f"| {status} | {count} |" for status, count in sorted(counts.items())]
    lines += ["", f"{len(rows)} historical rows across {len(coverage)} campaigns; row counts are in `historical_rows.csv`.", "",
              f"Frozen evaluator hashes: {'all match' if all(v['matches'] for v in integrity.values()) else 'MISMATCH; inspect snapshot.json'}.",
              f"Registry conflict: {len(affected)} rows / {result['registry_issue']['affected_distinct_validations']} distinct validations; "
              f"{result['registry_issue']['matching_imported_registry_rows']} rows match the imported registry.", "",
              "| Failure category | Distinct identities |", "|---|---:|"]
    lines += [f"| {kind} | {count} |" for kind, count in sorted(result["failure_counts"].items())]
    lines += ["", "Two preflight records may remain in the shared measurement directory; only identities in the frozen manifest count here.",
              "New passes use the saved harness. They do not establish correctness on additional inputs or resolve historical reference-isolation flags.",
              "No completion ETA: queue order and failures bias the completed subset.", "",
              "## Completed campaigns", "", "| Campaign | Valid pairs | Estimated-cycle GM improvement | 95% interval |", "|---|---:|---:|---|"]
    for item in completed:
        summary = item["summary"]
        gm = summary.get("geomean_speedup_on_over_off", summary.get("geomean_speedup_benchmark_weighted"))
        ci = summary.get("benchmark_cluster_bootstrap_95_interval", summary.get("bootstrap_95_percent_interval"))
        lines.append(f"| {item['campaign']} | {summary['valid_pairs']}/{summary['planned_pairs']} | {gm:.4f} | [{ci[0]:.4f}, {ci[1]:.4f}] |")
    lines += ["", "These studies measure different effects; their gains must not be multiplied or presented as RTL speedups.",
              "`completed_search_costs.csv` records per-cell calls, tokens, synthesis, cached tokens, search time, and validation status.",
              "Cost totals include failed/excluded cells; performance effects retain each campaign's original audit filter.", "",
              "`selected_cosim_measurement_valid=True` alone is not an executed RTL pass. Check `rtl_executed_pass`, the run count, and saved COSIM record together.", "",
              "## Files", "", "- `snapshot.json`: provenance, live state, hash checks, failure totals and saved summaries.",
              "- `historical_rows.csv`: all historical rows, including blocked/existing/pending cases.",
              "- `validations.csv`: one row per distinct queued validation identity.",
              "- `failures.csv`: completed non-pass validations with reasons and evidence paths.",
              "- `registry_affected_rows.csv`: cohort-specific registry conflicts.",
              "- `campaign_coverage.csv`: row-level coverage for every historical campaign (deduplicated jobs can span campaigns).",
              "- `completed_search_costs.csv`: cost and validation ledger for the three completed September studies.", "",
              "See `docs/experiment_status_20260928.md` for interpretation and the prioritized revision plan."]
    (output / "report.md").write_text("\n".join(lines) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, default=DEFAULT_CAMPAIGN)
    parser.add_argument("--output-dir", type=Path,
                        default=REPO / "artifacts/experiment_status" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    args = parser.parse_args()
    # Restrict writes to a dedicated analysis tree, including explicit output paths.
    output = args.output_dir.resolve()
    output.relative_to((REPO / "artifacts/experiment_status").resolve())
    if output.exists():
        parser.error("output directory already exists; use a fresh timestamp")
    result = snapshot(output, args.campaign.resolve())
    print(json.dumps({"output_dir": str(output), "captured_at": result["read_finished_at"],
                      "validation_status_counts": result["validation_status_counts"]}, indent=2))


if __name__ == "__main__":
    main()
