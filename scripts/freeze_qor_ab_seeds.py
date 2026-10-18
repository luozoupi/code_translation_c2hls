#!/usr/bin/env python3
"""Recover measured Phase-B code by hash, never by first-reply position."""

from __future__ import annotations

import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import c2hls
from phase_b_manifest import (SCHEMA_VERSION, canonical_json_sha256,
                              file_sha256, text_sha256, toolchain_fingerprint)


def recover_baseline(result: dict, history: dict) -> tuple[str, dict]:
    report = result.get("baseline_report") or {}
    csim = result.get("baseline_csim") or {}
    if csim.get("passed") is not True or csim.get("ran") is not True:
        raise ValueError("baseline lacks an executed passing CSim")
    events = (result.get("synthesis_evaluations") or {}).get("events") or []
    matches = [e for e in events if e.get("label") == "[Phase B]"
               and e.get("report_sha256") == canonical_json_sha256(report)
               and e.get("synthesis_status") == "passed"
               and e.get("correctness_status") == "passed"
               and e.get("timing_met") is True and e.get("resource_fit") is True]
    hashes = {e.get("code_sha256") for e in matches}
    if len(hashes) != 1 or not next(iter(hashes), None):
        raise ValueError("baseline report is not uniquely linked to feasible synthesized code")
    expected = next(iter(hashes))
    for event in history.get("event_history") or []:
        if event.get("role") != "assistant":
            continue
        code = c2hls.extract_cpp_code(event.get("content") or "")
        if not code:
            continue
        normalized, _ = c2hls._normalize_vitis_s_axilite_bundles(code)
        for candidate in (code, normalized):
            if text_sha256(candidate) == expected:
                return candidate, matches[-1]
    raise ValueError("no saved reply reproduces the measured baseline code hash")


def freeze(config: dict) -> dict:
    source_root = Path(config["source_campaign"])
    state_path = source_root / "campaign_state.json"
    state = json.loads(state_path.read_text())
    source_config = json.loads((source_root / "manifest.json").read_text())["config"]
    output = Path(config["phase_b_manifest"])
    if output.exists():
        raise ValueError("refusing to overwrite frozen seeds")
    tc = toolchain_fingerprint(vitis_version="2023.2", part="xcu280-fsvh2892-2L-e", clock_ns=3.33)
    entries = {}
    for bench in config["benchmarks"]:
        name = bench["name"]
        # Registered source strategy, not whichever historical outcome was fastest.
        cell = next(c for c in state["cells"] if c["benchmark"] == name and c["strategy"] == "dynamic")
        if cell["status"] != "complete":
            raise ValueError("historical source cell is incomplete: " + name)
        result_path = Path(cell["result_path"])
        history_path = result_path.with_name(name + "_history.json")
        result, history = json.loads(result_path.read_text()), json.loads(history_path.read_text())
        if (result.get("reference_isolation_audit") or {}).get("passed") is not True:
            raise ValueError("historical source lacks reference isolation: " + name)
        run = result["run"]
        observed = toolchain_fingerprint(vitis_version=run["vitis_version"], part=run["part"],
                                         clock_ns=run["clock_ns"], flow_target=run.get("flow_target", "vitis"))
        if observed != tc or source_config["benchmarks_root"] != config["benchmarks_root"]:
            raise ValueError("historical source toolchain or benchmark root mismatch")
        code, event = recover_baseline(result, history)
        inputs = c2hls._load_benchmark_inputs(str(Path(config["benchmarks_root"]) / name))
        relative_code = Path("code") / (name + ".cpp")
        code_path = output.parent / relative_code
        code_path.parent.mkdir(parents=True, exist_ok=True)
        code_path.write_text(code)
        report, csim = result["baseline_report"], result["baseline_csim"]
        entries[name] = {
            "benchmark": name, "code_path": str(relative_code), "code_sha256": text_sha256(code),
            "input_c_sha256": text_sha256(inputs["c_code"]), "header_sha256": text_sha256(inputs["header_code"]),
            "csim": csim, "csim_sha256": canonical_json_sha256(csim),
            "csynth_report": report, "csynth_report_sha256": canonical_json_sha256(report),
            "source_artifacts": {"result_path": str(result_path), "result_sha256": file_sha256(result_path),
                                 "history_path": str(history_path), "history_sha256": file_sha256(history_path),
                                 "selection_policy": "registered_dynamic_phase_b_not_best_outcome",
                                 "synthesis_event": event},
        }
    payload = {"schema_version": SCHEMA_VERSION, "purpose": "qor_reviewer_ab_frozen_parent",
               "toolchain": tc, "entries": entries}
    output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    return payload
