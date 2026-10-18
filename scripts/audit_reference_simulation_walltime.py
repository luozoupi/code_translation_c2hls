#!/usr/bin/env python3
"""Read-only audit of reference host co-simulation timing and exact-input reuse.

No Vitis or provider calls. Hardware latency fields are never host-time inputs.
The inventory distinguishes cosim_design, XSIM run, and whole run_cosim scopes.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HOME = ROOT.parent
PHASE = re.compile(r"Finished Command (\w+).*?Elapsed time:\s*([\d.]+) seconds")
XSIM = re.compile(r"^run: Time \(s\):.*?elapsed\s*=\s*(\d+):(\d+):(\d+)", re.M)
TIMEOUT = re.compile(r"timed out after\s+(\d+(?:\.\d+)?)s", re.I)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest() if Path(path).is_file() else None


def canon(name):
    name = name.replace("-", "_")
    return "hlsfactory_gramschmidt" if name == "hlsfactory_gramschmidt_m60_n40" else name


def parse_log(log):
    phases = collections.defaultdict(list)
    for command, seconds in PHASE.findall(log or ""):
        phases[command].append(float(seconds))
    return {
        "cosim_design_wall_seconds": phases.get("cosim_design", []),
        "csynth_design_wall_seconds": phases.get("csynth_design", []),
        "xsim_run_wall_seconds": [int(h)*3600+int(m)*60+int(s) for h, m, s in XSIM.findall(log or "")],
        "scopes_overlap_not_additive": True,
    }


def status(cosim):
    error = str(cosim.get("error") or "")
    if TIMEOUT.search(error):
        return "timeout"
    value = cosim.get("status")
    if value in ("not_run", "timeout", "skipped"):
        return value
    if cosim.get("passed") is True or value in ("passed", "pass"):
        return "pass"
    if value in ("failed", "fail") or cosim.get("success") is False:
        return "fail"
    return "unknown"


def target_inventory(root):
    result = {}
    for directory in sorted(root.glob("hlsfactory_*")):
        meta_path = directory / "metadata.json"
        if not meta_path.is_file():
            continue
        meta = read(meta_path)
        names = {"plain_c": meta.get("plain_c_file", "plain.cpp"),
                 "public_testbench": meta.get("testbench_file", "testbench.cpp"),
                 "header": meta.get("header_file", "kernel.h"),
                 "reference_hls": meta.get("gold_hls_baseline_file", "hls_baseline.cpp")}
        hashes = {k+"_sha256": sha(directory / v) for k, v in names.items()}
        result[canon(directory.name)] = {
            "benchmark": directory.name, "directory": str(directory), **hashes,
            "metadata_sha256": sha(meta_path), "header_name": names["header"],
            "cosim_depths": meta.get("cosim_depths", {}),
            "alternate_cosim_testbench_sha256": sha(directory / meta.get("cosim_testbench_file", "testbench_cosim.cpp")),
            "historical_cosim_size_overrides": meta.get("cosim_size_overrides", {}),
            "historical_cosim_support_files": meta.get("cosim_support_files", []),
            "public_only_staging_requirement": "Do not copy reference HLS or expert cosim support; disable alternate cosim testbench and size overrides for the public workload.",
        }
    return result


def base_record(path, benchmark, kind, pointer):
    return {
        "record_id": hashlib.sha256((str(path)+":"+pointer).encode()).hexdigest(),
        "benchmark": canon(benchmark), "original_benchmark": benchmark,
        "reference_kind": kind, "source_path": str(path.resolve()),
        "source_sha256": sha(path), "json_pointer": pointer,
        "status": "unknown", "case_id": None, "source_code_sha256": None,
        "header_sha256": None, "testbench_sha256": None,
        "toolchain": {}, "harness_sha256": None, "golden_output_sha256": None,
        "whole_cosim_flow_wall_seconds": None, "cosim_design_wall_seconds": None,
        "xsim_run_wall_seconds": None, "timeout_cap_seconds": None,
        "completion_time_right_censored": False, "censoring_scope": None,
        "reusable_cosim_design_wall_seconds": None,
    }


def cache_records(cache):
    for path in sorted(cache.glob("*.json")):
        data = read(path)
        if not isinstance(data.get("configuration"), dict) or not isinstance(data.get("reference_validation"), dict):
            continue
        config, ref = data["configuration"], data["reference_validation"]
        cosim = ref.get("cosim") or {}
        record = base_record(path, config["benchmark"], "historical_reference_hls_cache", "/reference_validation/cosim")
        candidates = config.get("candidates") or []
        selected = ref.get("selected_variant_file")
        matching = [c for c in candidates if c.get("file") == selected] if selected else candidates
        candidate = matching[0] if len(matching) == 1 else {}
        match = TIMEOUT.search(str(cosim.get("error") or ""))
        record.update(status=status(cosim), source_code_sha256=candidate.get("code_sha256"),
                      header_sha256=candidate.get("header_sha256"), testbench_sha256=config.get("testbench_sha256"),
                      toolchain={k:config.get(k) for k in ("vitis_version", "part", "clock_ns", "flow_target", "toolchain_identity")},
                      reference_cache_fingerprint=data.get("fingerprint"), reference_cache_schema=data.get("schema_version"),
                      cosim_depths=config.get("cosim_depths"), extra_files=config.get("extra_files"),
                      golden_output_sha256=config.get("independent_golden_output_sha256"),
                      historical_source_result_json=data.get("source_result_json"),
                      historical_source_result_available=Path(data["source_result_json"]).is_file() if data.get("source_result_json") else None,
                      timeout_cap_seconds=float(match.group(1)) if match else None,
                      hardware_latency_cycles=cosim.get("kernel_runtime_cycles"),
                      hardware_latency_us=cosim.get("kernel_runtime_us"))
        record["completion_time_right_censored"] = record["status"] == "timeout"
        record["censoring_scope"] = "whole_cosim_flow_including_resynthesis" if record["status"] == "timeout" else None
        yield record


def parent_records(root):
    manifest_path = root / "results_sweeps/sigmetrics_recipe_main_20260930/manifest.json"
    manifest = read(manifest_path)
    entries = {x["benchmark"]: x for x in manifest["benchmark_entries"]}
    paths = sorted((root / "results_sweeps/sigmetrics_recipe_main_20260930/parents").glob("*/measurement.json"))
    paths += [root / "artifacts/sigmetrics2027/parent_pilot_20260928/atax/measurement.json"]
    paths += sorted((root / "results_sweeps/sigmetrics_timeout_replay_20261001/parents").glob("*/measurement.json"))
    harness = Path(manifest["source"]) / "hls_eval.py"
    for path in paths:
        if not path.is_file():
            continue
        data = read(path)
        benchmark = data["benchmark"]
        entry = entries.get(benchmark, {})
        cosim = data.get("cosim") or {}
        cases = (cosim.get("validation_suite") or {}).get("cases") or [cosim]
        events = [e for e in data.get("tool_flows", {}).get("events", []) if e.get("kind") == "rtl_cosim"]
        for index, case in enumerate(cases):
            record = base_record(path, benchmark, "unchanged_public_c_parent", f"/cosim/validation_suite/cases/{index}")
            event = events[index] if index < len(events) else {}
            # Aggregator retains the representative case's complete log only.
            log = case.get("log")
            if log is None and case.get("work_dir") == cosim.get("work_dir"):
                log = cosim.get("log", "")
            timings = parse_log(log)
            match = TIMEOUT.search(str(case.get("error") or ""))
            record.update(status=status(case), case_id=case.get("case_id", index),
                          source_code_sha256=event.get("code_sha256"), public_plain_sha256=entry.get("input_c_sha256"),
                          header_sha256=entry.get("header_sha256"), testbench_sha256=case.get("testbench_sha256"),
                          golden_output_sha256=case.get("golden_output_sha256"),
                          whole_cosim_flow_wall_seconds=event.get("elapsed_seconds"),
                          cosim_design_wall_seconds=timings["cosim_design_wall_seconds"][0] if len(timings["cosim_design_wall_seconds"]) == 1 else None,
                          xsim_run_wall_seconds=timings["xsim_run_wall_seconds"][0] if len(timings["xsim_run_wall_seconds"]) == 1 else None,
                          phase_log_evidence=timings, harness_sha256=sha(harness),
                          timeout_cap_seconds=float(match.group(1)) if match else None,
                          toolchain={"vitis_version":"2023.2", "part":"xcu280-fsvh2892-2L-e", "clock_ns":3.33, "flow_target":"vitis"},
                          validation_manifest_sha256=data.get("validation_manifest_sha256"),
                          input_manifest_path=str(manifest_path), input_manifest_sha256=sha(manifest_path),
                          hardware_latency_cycles=case.get("kernel_runtime_cycles"), hardware_latency_us=case.get("kernel_runtime_us"),
                          dimensions_amendment=entry.get("domain_amendment"))
            record["completion_time_right_censored"] = record["status"] == "timeout"
            record["censoring_scope"] = "whole_cosim_flow_including_resynthesis" if record["status"] == "timeout" else None
            yield record


def historical_direct_records(original):
    names = ("hlsfactory_direct_reference_merged_cosim10800_20260601.summary.json",
             "hlsfactory_direct_reference_hlsfactory_direct_cosim10800_remaining_20260601.summary.json")
    for name in names:
        path = original / "artifacts" / name
        if not path.is_file():
            continue
        data = read(path)
        for index, row in enumerate(data.get("rows", [])):
            record = base_record(path, row["bench"], "historical_direct_public_baseline_hls", f"/rows/{index}")
            error = row.get("cosim_error") or ""
            match = TIMEOUT.search(error)
            record.update(status=row.get("cosim_status", "unknown"),
                          whole_cosim_flow_wall_seconds=row.get("cosim_runtime_seconds"),
                          toolchain=data.get("vitis", {}),
                          timeout_cap_seconds=float(match.group(1)) if match else None,
                          hardware_latency_cycles=row.get("cosim_cycles"),
                          historical_variant=row.get("variant"),
                          record_note="Merged tables retain an earlier timeout's main runtime field; later 10800-second supplement is a separate record. No historical source/input hash is inferred from current files.")
            record["completion_time_right_censored"] = record["status"] == "timeout"
            record["censoring_scope"] = "whole_cosim_flow_including_resynthesis" if record["status"] == "timeout" else None
            yield record


def assess(record, target):
    """Conservative scope and provenance checks; missing identity is not equality."""
    comparisons = {}
    for actual, wanted in (("header_sha256", "header_sha256"), ("testbench_sha256", "public_testbench_sha256")):
        a, b = record.get(actual), target.get(wanted)
        comparisons[actual] = None if not a or not b else a == b
    expected_source = target.get("plain_c_sha256") if record["reference_kind"] == "unchanged_public_c_parent" else target.get("reference_hls_sha256")
    actual_source = record.get("public_plain_sha256") if record["reference_kind"] == "unchanged_public_c_parent" else record.get("source_code_sha256")
    comparisons["source_input_sha256"] = None if not actual_source or not expected_source else actual_source == expected_source
    record["target_file_matches"] = comparisons
    reasons = []
    if record["status"] != "pass":
        reasons.append("reference_not_a_completed_pass")
    if record.get("cosim_design_wall_seconds") is None:
        reasons.append("no_separate_completed_cosim_design_wall_measurement")
    for key, value in comparisons.items():
        if value is not True:
            reasons.append(("unknown_" if value is None else "mismatch_") + key)
    # Current candidate validation has a new phase-watch wrapper and public-only
    # contract. Matching benchmark strings never prove full harness equivalence.
    reasons.append("new_validation_harness_and_toolchain_equivalence_not_established")
    if record["reference_kind"] != "unchanged_public_c_parent":
        reasons.append("historical_reference_hls_is_not_the_new_public_parent")
    record["reuse_blockers"] = reasons
    record["exactly_reusable_for_new_public_study"] = False
    return record


def build_report(root, target_root, hpca, original):
    targets = target_inventory(target_root)
    records = list(cache_records(hpca / "artifacts/reference_validation_cache"))
    records += list(parent_records(root))
    records += list(historical_direct_records(original))
    for record in records:
        assess(record, targets.get(record["benchmark"], {}))
    by_benchmark = []
    for benchmark, target in targets.items():
        evidence = [r for r in records if r["benchmark"] == benchmark]
        by_benchmark.append({"benchmark": benchmark, "target": target,
                             "record_count":len(evidence), "status_counts":dict(collections.Counter(r["status"] for r in evidence)),
                             "records_with_cosim_design_wall":sum(r["cosim_design_wall_seconds"] is not None for r in evidence),
                             "records_with_xsim_wall":sum(r["xsim_run_wall_seconds"] is not None for r in evidence),
                             "records_with_whole_cosim_flow_wall":sum(r["whole_cosim_flow_wall_seconds"] is not None for r in evidence),
                             "reusable_cosim_design_wall_seconds":None,
                             "recommended_phase_cap_seconds":3600,
                             "phase_cap_reason":"No exact matched completed cosim_design reference; predeclared fixed fallback, not predicted runtime."})
    return {"schema_version":"c2hls.reference-walltime-audit.v1", "created_at":datetime.now(timezone.utc).isoformat(),
            "target_root":str(target_root), "target_output_registry_path":str(root / "configs/hlsfactory_output_shapes_fc27133.json"),
            "target_output_registry_sha256":sha(root / "configs/hlsfactory_output_shapes_fc27133.json"),
            "scope_definitions":{"whole_cosim_flow":"run_cosim call: candidate resynthesis + cosim_design + wrapper overhead",
                                 "cosim_design":"Vitis Finished Command cosim_design wall elapsed: setup/elaboration, RTL simulation, and co-sim postcheck; excludes preceding csynth_design",
                                 "xsim_run":"XSIM run command elapsed only; excludes compiler/setup and enclosing co-sim work",
                                 "kernel_runtime_us":"Simulated hardware kernel latency; NEVER host wall time"},
            "counts":{"packaged_benchmarks":len(targets), "reference_records":len(records),
                      "by_reference_kind":dict(collections.Counter(r["reference_kind"] for r in records)),
                      "with_cosim_design_wall":sum(r["cosim_design_wall_seconds"] is not None for r in records),
                      "with_xsim_wall":sum(r["xsim_run_wall_seconds"] is not None for r in records),
                      "exact_reusable_cosim_design_references":0},
            "policy":{"scope":"new queued pass@K study only; historical/frozen studies unchanged",
                      "matched_reference_rule":"min(10800, max(120, 2 * matching completed cosim_design wall seconds))",
                      "unmatched_reference_fallback_seconds":3600, "timeout_is_censored":True,
                      "do_not_use":"cycles, kernel_runtime_us, XSIM-only wall, whole flow wall, or timeout exposure as completed cosim_design wall",
                      "reference_comparability_requirements":["same input domain/header/testbench hashes", "recorded public reference kernel hash", "same tool version/build and target part/clock", "same validation harness/trace/logging settings", "reference phase completes and independently passes", "record concurrency/host load"],
                      "interpretation":"Resource-control heuristic and measured capped success@K; an early stop is a censored attempt, not evidence of functional incorrectness."},
            "benchmarks":by_benchmark, "records":records}


def write_report(report, output):
    output.mkdir(parents=True, exist_ok=False)
    (output / "report.json").write_text(json.dumps(report, indent=2)+"\n")
    fields = ["benchmark", "reference_kind", "status", "case_id", "whole_cosim_flow_wall_seconds", "cosim_design_wall_seconds", "xsim_run_wall_seconds", "timeout_cap_seconds", "completion_time_right_censored", "censoring_scope", "source_code_sha256", "header_sha256", "testbench_sha256", "reusable_cosim_design_wall_seconds", "source_path", "source_sha256", "json_pointer", "reuse_blockers"]
    with (output / "reference_records.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        for record in report["records"]:
            writer.writerow({k:json.dumps(record[k]) if isinstance(record.get(k), (list, dict)) else record.get(k) for k in fields})
    lines = ["# Reference co-simulation host-time audit", "", f"Generated {report['created_at']}. Read-only evidence audit; no experiments launched.", "",
             "The target contains **28 packaged PolyBench kernels, including Doitgen**. Historical reference records exist, but **none supplies an exact matched completed `cosim_design` host-time baseline for the new public-only validation**. Use the declared 3,600-second phase fallback until an equivalent reference is measured.", "",
             "`cosim_design` includes simulator preparation, elaboration, RTL execution, and postcheck, but excludes preceding synthesis. XSIM `run` wall time and the whole `run_cosim` invocation have different scopes. Hardware cycles and `kernel_runtime_us` are not host time. Do not sum nested timings.", "",
             "The July HPCA cache has 81 entries: 17 co-sim passes, one actual 10,800-second timeout recorded as failed, and 63 not-run records. It records no host phase time. The May/June direct baseline table covers 28 kernels: 19 pass, eight timeout, one not-run (Doitgen); its latest timeout supplement records 10,800-second exposure separately from earlier merged fields.", "",
             "September/October public-parent evidence covers 12 benchmark configurations. Eight passing configurations retain one complete phase log each; other input cases retain whole-flow times. GramSchmidt uses amended M=60,N=40. These domains and harnesses must not be silently substituted for July inputs.", "",
             "| Benchmark | Recorded statuses (all histories/cases) | cosim_design timings | XSIM timings | New cap (s) |", "|---|---|---:|---:|---:|"]
    for row in report["benchmarks"]:
        statuses = ", ".join(f"{k}: {v}" for k, v in sorted(row["status_counts"].items())) or "no evidence"
        lines.append(f"| {row['benchmark']} | {statuses} | {row['records_with_cosim_design_wall']} | {row['records_with_xsim_wall']} | 3600 |")
    lines += ["", "Counts are evidence records, not independent designs or a pass-rate denominator. Cache entries, multiple input cases, merged tables, and retries overlap.", "", "The enhanced benchmark package also contains alternate reduced co-sim testbenches and expert support kernels. Public-only staging must disable these metadata overrides and preserve the requested public workload. Both public and alternate testbench hashes are recorded in the JSON inventory.", "", "A future reference-relative phase cap is `min(10800, max(120, 2 * reference_cosim_design_seconds))` only after all recorded provenance checks pass. Unknown or censored references use the fixed fallback. Early stops consume an attempt and remain separate from functional failures; report the cap and retry budget with success@K.", "", "- [All evidence and fingerprints](report.json)", "- [Per-reference CSV](reference_records.csv)", ""]
    (output / "report.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--target-root", type=Path, default=HOME / "code_translation-c2hls-enhanced-fc27133/benchmarks")
    parser.add_argument("--hpca-root", type=Path, default=HOME / "code_translation-c2hls-hpca2027")
    parser.add_argument("--original-root", type=Path, default=HOME / "code_translation-c2hls")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build_report(args.root, args.target_root, args.hpca_root, args.original_root)
    write_report(report, args.output)
    print(json.dumps(report["counts"], indent=2))


if __name__ == "__main__":
    main()
