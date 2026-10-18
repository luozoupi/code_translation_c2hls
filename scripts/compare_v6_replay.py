#!/usr/bin/env python3
"""Compare frozen v6 and a cleaned checkout using saved responses and reports.

Each runtime is imported in a separate subprocess. Model and tool boundaries
are replaced with checksum-bound archived records; network connections and
unexpected subprocess execution are rejected. No model clients are created.
"""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import inspect
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIXTURE = ROOT / "tests/fixtures/v6_replay"
CONTRACT_HELPERS = (
    "_align_generated_top_signature", "_canonical_function_signature",
    "_expected_top_signature", "_extract_defined_function_names",
    "_extract_function_signature", "_extract_interface_ports",
    "_extract_prototype_names", "_extract_struct_names",
    "_header_contract_definitions", "_normalize_contract_tokens",
    "_normalize_signature_text", "_normalize_vitis_s_axilite_bundles",
    "_public_header_contract_audit", "_render_function_signature",
    "_sha256_text", "_strip_contract_comments_and_literals",
    "_top_signature_mismatch_reason", "extract_cpp_code",
)
REPORT_FIELDS = (
    "latency_cycles", "latency_cycles_worst", "latency_ns", "latency_ns_worst",
    "bram", "dsp", "ff", "lut", "uram", "fmax_mhz", "slack_ns",
    "estimated_clock_period_ns", "requested_clock_period_ns",
)


def digest(value):
    if not isinstance(value, str):
        value = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(value.encode()).hexdigest()


def load_fixture(root):
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("schema") != "hlsmith.replay-fixture-manifest.v1":
        raise ValueError("Unsupported replay fixture manifest")
    for name, expected in manifest["files"].items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Archived replay fixture changed: {name}")
    fixture = json.loads((root / "snapshot.json").read_text())
    for item in fixture["cases"] + fixture["selection_candidates"]:
        if digest(item["reply"]) != item["provenance"]["reply_sha256"]:
            raise ValueError(f"Archived reply changed: {item['id']}")
        if hashlib.sha256(item["kernel"].encode()).hexdigest() != item["provenance"]["kernel_sha256"]:
            raise ValueError(f"Archived source changed: {item['id']}")
        if digest(item["report"]) != item["provenance"]["report_sha256"]:
            raise ValueError(f"Archived synthesis report changed: {item['id']}")
    return fixture


def report_values(report):
    return {key: report.get(key) for key in REPORT_FIELDS}


def make_orchestrator(runtime, item):
    orch = runtime.C2HLSOrchestrator.__new__(runtime.C2HLSOrchestrator)
    orch.gpt_model = "archived-replay"
    orch.c_code = item["plain"]
    orch.header_code = item["header"]
    orch.header_name = item["header_name"]
    orch.testbench_code = item["testbench"]
    orch.translated_hls_top = item["metadata"].get("translated_hls_top", "workload")
    orch.benchmark_name = item["benchmark"]
    orch.benchmark_context = runtime._build_benchmark_context(
        item["metadata"], item["header_name"], item["header"], item["plain"], item["testbench"]
    )
    orch.part = "xcu280-fsvh2892-2L-e"
    orch.clock_ns = 3.33
    orch.phaseb_mode = "functional"
    orch._phaseb_multistep_context = True
    orch.turns_limitation = 2
    orch.extra_files = []
    orch.preflight_patches = []
    orch.history = []
    orch.messages = []
    orch.turn_results = []
    orch.generated_csim = None
    orch.generated_cosim = None
    orch.hls_code = item["kernel"]
    orch.synth_report = item["report"]
    return orch


def agent_replay(runtime, item):
    """Run real translation/repair control with archived boundary responses."""
    orch = make_orchestrator(runtime, item)
    requests, finalized, tool_calls = [], [], []

    def response(messages, **kwargs):
        requests.append({"messages": copy.deepcopy(messages), **kwargs})
        return item["reply"]

    orch._call_llm_with_model = response
    orch._finalize_candidate_evaluation = lambda *args, **kwargs: finalized.append(kwargs)
    evidence = item["independent_measurement"]

    def tool(name, result):
        def call(code, *args, **kwargs):
            tool_calls.append({"tool": name, "source_sha256": digest(code)})
            return copy.deepcopy(result)
        return call

    def evaluate(code, **kwargs):
        return runtime._run_synth_csim_cosim(
            code, header_code=orch.header_code, header_name=orch.header_name,
            top_function=orch.translated_hls_top, part=orch.part,
            clock_ns=orch.clock_ns, extra_files=[], testbench_code=orch.testbench_code,
            run_csim_check=True, run_cosim_check=True, cosim_requires_csim_pass=True,
            correctness_first_override=True,
        )

    orch._synth_and_test = evaluate
    extracted = runtime.TranslatorAgent(orch).translate_initial()
    if digest(extracted) != digest(item["kernel"].strip()):
        raise ValueError(f"Saved response no longer extracts its linked source: {item['id']}")
    orch.hls_code = extracted
    with (
        mock.patch.object(runtime, "compile_check_cpp", return_value=(True, "")),
        mock.patch.object(runtime, "run_csim", side_effect=tool("csim", evidence["csim"])),
        mock.patch.object(runtime, "run_cosim", side_effect=tool("cosim", evidence["cosim"])),
        mock.patch.object(runtime, "run_hls_synthesis", side_effect=tool("synthesis", {
            "success": True, "error": "", "report": item["report"],
        })),
    ):
        accepted = runtime.SynthesisAgent(orch).synthesize_with_repair()
    csim = runtime._summarize_test_result(evidence["csim"], True)
    cosim = (runtime._summarize_test_result(evidence["cosim"], True)
             if evidence["cosim"].get("ran", True) else None)
    # Keep wrapper/control preflight and a derived split-bundle contract probe
    # separate from saved experimental outcomes.
    probe, note = runtime._normalize_vitis_s_axilite_bundles(
        extracted.replace("bundle=control", "bundle=control_r")
    )
    return {
        "id": item["id"],
        "context_sha256": digest(orch.benchmark_context),
        "requests_sha256": digest(requests),
        "requests": len(requests),
        "history_sha256": digest(orch.history),
        "extracted_source_sha256": digest(extracted),
        "preflight_source_sha256": digest(orch.hls_code),
        "phase_b_accepted": accepted,
        "tool_calls": tool_calls,
        "finalized_failures": finalized,
        "turn_results_sha256": digest(orch.turn_results),
        "report": report_values(orch.synth_report),
        "feasibility": runtime._paper_candidate_feasibility(
            item["report"], csim=csim, part=orch.part, clock_ns=orch.clock_ns,
        ),
        "csim_status": runtime.C2HLSOrchestrator._candidate_test_status(csim),
        "cosim_status": runtime.C2HLSOrchestrator._candidate_test_status(cosim),
        "cosim_measurement_eligible": runtime._selected_cosim_measurement_ok(cosim),
        "derived_bundle_probe": {"sha256": digest(probe), "note": note},
        "public_contract": runtime._public_header_contract_audit(
            item["header"], item["header"], plain_code=item["plain"], testbench_code=item["testbench"],
        ),
        "signature_mismatch": runtime._top_signature_mismatch_reason(
            orch.hls_code, item["header"], item["testbench"], orch.translated_hls_top,
        ),
    }


def selection_replay(runtime, candidates):
    """Restore the best feasible source among real seven-arm GEMM outputs."""
    orch = make_orchestrator(runtime, candidates[0])
    history = []
    # End at a slower observed candidate so best-state restoration is exercised.
    ordered = sorted(candidates, key=lambda item: runtime.C2HLSOrchestrator._best_so_far_score(item["report"]))
    for index, item in enumerate(ordered):
        orch.hls_code = item["kernel"]
        orch.synth_report = item["report"]
        orch.generated_csim = item["historical_csim"]
        orch.generated_cosim = None
        orch._record_best_so_far(history, step_index=index, step_name=item["id"], source="step")
    promoted = orch._promote_best_so_far(history)
    return {
        "registered_candidates": len(candidates), "eligible_candidates": len(history),
        "winner_source_sha256": digest(orch.hls_code), "winner_report": report_values(orch.synth_report),
        "promoted_step": promoted.get("step_name") if promoted else None,
        "candidate_decisions": [{"step_name": item["step_name"], "score": item["score"],
                                 "feasibility": item["feasibility"]} for item in history],
    }


def _blocked(*args, **kwargs):
    raise RuntimeError("Deterministic replay forbids network and unmocked tool execution")


def worker(repo, fixture_root):
    sys.path.insert(0, str(repo.resolve()))
    with mock.patch.object(socket.socket, "connect", side_effect=_blocked), mock.patch.object(socket, "create_connection", side_effect=_blocked):
        import c2hls as runtime
        fixture = load_fixture(fixture_root)
        helper_hashes = {name: digest(ast.dump(ast.parse(inspect.getsource(getattr(runtime, name)))))
                         for name in CONTRACT_HELPERS}
        with (
            mock.patch.object(subprocess, "Popen", side_effect=_blocked),
            mock.patch.object(runtime, "run_native_testbench", side_effect=_blocked),
        ):
            cases = [agent_replay(runtime, item) for item in fixture["cases"]]
            selection = selection_replay(runtime, fixture["selection_candidates"])
    return {"schema": "hlsmith.v6-deterministic-results.v1", "source_contract_helpers": helper_hashes,
            "cases": cases, "selection": selection}


def run_worker(repo, fixture_root=DEFAULT_FIXTURE):
    environment = {key: value for key, value in os.environ.items()
                   if not key.startswith(("C2HLS_", "HLSMITH_"))}
    environment.update({
        "PYTHONDONTWRITEBYTECODE": "1", "PYTHON_DOTENV_DISABLED": "1",
        "C2HLS_FEASIBILITY_SELECTION": "1", "C2HLS_REFERENCE_BLIND": "1",
        "C2HLS_CORRECTNESS_BEFORE_SYNTH": "1", "C2HLS_COSIM_REQUIRED": "1",
        "C2HLS_SYNTH_REVERT_THRESHOLD": "0",
    })
    process = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--worker", "--repo", str(repo.resolve()),
         "--fixture", str(fixture_root.resolve())], cwd=repo, env=environment,
        capture_output=True, text=True, timeout=60,
    )
    if process.returncode:
        raise RuntimeError(f"Replay failed in {repo}:\n{process.stderr}")
    return json.loads(process.stdout)


def compare_repositories(frozen, cleaned=ROOT, fixture_root=DEFAULT_FIXTURE):
    old = run_worker(frozen, fixture_root)
    new = run_worker(cleaned, fixture_root)
    if old != new:
        changed = [name for name in old if old[name] != new.get(name)]
        raise ValueError(f"Frozen v6 / cleaned replay differs in: {', '.join(changed)}")
    return {"status": "passed", "schema": "hlsmith.v6-replay-comparison.v1",
            "frozen_runtime": str(frozen.resolve()), "cleaned_runtime": str(cleaned.resolve()),
            "result_sha256": digest(new), "fixture_cases": len(new["cases"]),
            "contract_helpers": len(new["source_contract_helpers"]),
            "selection_candidates": new["selection"]["registered_candidates"],
            "cases": [{key: item[key] for key in ("id", "csim_status", "cosim_status", "phase_b_accepted")}
                      for item in new["cases"]],
            "scope": "Archived-boundary deterministic control replay; no compilation, Vitis, model generation, or full-corpus rerun."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen", type=Path, help="Checkout containing the frozen v6 implementation")
    parser.add_argument("--cleaned", type=Path, default=ROOT)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--repo", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        result = worker(args.repo, args.fixture)
    else:
        if args.frozen is None:
            parser.error("--frozen is required for an old/new comparison")
        result = compare_repositories(args.frozen, args.cleaned, args.fixture)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
