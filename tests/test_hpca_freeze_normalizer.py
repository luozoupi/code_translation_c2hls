import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from evaluation_repro import REFERENCE_BLIND_OVERRIDES
from scripts.normalize_hpca_freeze_index import (
    FreezeNormalizationError,
    normalize_freeze_index,
    normalize_to_file,
)


TARGET = {
    "vitis_version": "2023.2",
    "part": "xcu280-fsvh2892-2L-e",
    "clock_ns": "3.33",
}
MODEL_ID = "Qwen/Qwen3-Coder-30B-A3B-Instruct"
MODEL_REVISION = "0123456789abcdef"
REPORT_METRICS = {
    "bram": 32,
    "dsp": 16,
    "ff": 8000,
    "lut": 4000,
    "uram": 2,
    "fmax_mhz": 312.5,
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _report_sha256(report: dict) -> str:
    encoded = json.dumps(
        report,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _rehash_fingerprint(fingerprint: dict) -> None:
    encoded = json.dumps(
        fingerprint["payload"],
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    fingerprint["sha256"] = hashlib.sha256(encoded).hexdigest()


def _content_manifest(path: str) -> dict:
    files = [{"path": path, "bytes": 1, "sha256": "a" * 64}]
    return {
        "files": files,
        "file_count": 1,
        "sha256": hashlib.sha256(
            json.dumps(
                files,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest(),
    }


def _llm_call_fields(seed: int | str) -> dict:
    return {
        "provider": "local",
        "model": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "max_tokens": 8192,
        "prompt_sha256": "e" * 64,
        "decoding": {
            "temperature": 0.2,
            "top_p": 0.95,
            "seed": seed,
            "seed_supported": True,
        },
    }


def _fingerprint(
    kernel: str,
    seed: int,
    *,
    strategy: str = "best_of_five",
    skill_mode: str = "skill_off",
    frozen: bool = False,
    baseline_method: str | None = None,
) -> dict:
    payload = {
        "schema_version": "c2hls.run-fingerprint.v1",
        "profile": "hpca2027_reference_blind",
        "benchmark": {"name": kernel, "inputs": _content_manifest("plain.cpp")},
        "implementation": {
            "git_head": "f" * 40,
            "sources": _content_manifest("c2hls.py"),
        },
        "prompts": _content_manifest("prompt_c2hls.py"),
        "model": {
            "id": MODEL_ID,
            "endpoint": {
                "valid": True,
                "unsafe_components": [],
                "endpoint_sha256": "e" * 64,
            },
            "agents": {
                "translator": MODEL_ID,
                "synthesis": MODEL_ID,
                "quality_repair": MODEL_ID,
                "feedback": MODEL_ID,
            },
            "revision": {
                "value": MODEL_REVISION,
                "source": "C2HLS_MODEL_REVISION",
                "resolved": True,
            },
        },
        "decoding": {
            "seed": seed,
            "temperature": "0.2",
            "top_p": "0.95",
            "max_completion_tokens": "8192",
        },
        "toolchain": {
            **TARGET,
            "flow_target": "vitis",
            "device_platform": "xilinx_u280_gen3x16_xdma_1_202211_1",
            "vitis_settings_sha256": "c" * 64,
            "vitis_version_probe": {
                "version": "2023.2",
                "resolved": True,
                "ran": True,
                "returncode": 0,
                "executable": "/opt/Xilinx/Vitis/2023.2/bin/vitis-run",
                "executable_sha256": "d" * 64,
                "error": None,
            },
            "vitis_user_home": {
                "configured_absolute": True,
                "path": {
                    "scope": "external",
                    "absolute": True,
                    "path_sha256": "a" * 64,
                },
                "state": "home_absent",
                "files": [],
                "file_count": 0,
                "errors": [],
                "sha256": "b" * 64,
            },
        },
        "reference_cache": {
            "enabled": False,
            "path": None,
            "state": "disabled",
            "files": [],
            "file_count": 0,
            "errors": [],
            "sha256": "c" * 64,
        },
        "post_route": {
            "hw_emu_final": {"configured": "0", "effective": False},
            "allow_wide_abi": {"configured": "0", "effective": False},
            "disable_debug_symbols": {"configured": "1", "effective": True},
            "clock_mhz": None,
            "clock_ns": "3.33",
            "emu_environment_script": {
                "state": "present",
                "configured_absolute": True,
                "path": {
                    "scope": "repository",
                    "absolute": True,
                    "path": "scripts/setup_emu_env.sh",
                },
                "bytes": 1,
                "sha256": "d" * 64,
            },
        },
        "search": {"strategy": strategy},
        "skills": {
            "mode": skill_mode,
            "prompt_injection": skill_mode == "skill_on",
            "frozen": True,
            "persistence": False,
            "online_statistics": False,
            "source_mode": (
                "explicit_frozen_snapshot" if skill_mode == "skill_on" else "disabled"
            ),
            "file_count": 1 if skill_mode == "skill_on" else 0,
            "explicit_path_configured": skill_mode == "skill_on",
            "expected_sha256": "b" * 64 if skill_mode == "skill_on" else None,
            "matches_expected": True if skill_mode == "skill_on" else None,
        },
        "budgets": {
            "candidate_budget": "5",
            "llm_candidate_budget": "5",
        },
        "reference_isolation": dict(REFERENCE_BLIND_OVERRIDES),
    }
    if baseline_method is not None:
        baseline_seed = int(seed)
        candidate_seed_schedule = [
            {
                "candidate_index": index,
                "requested_seed": baseline_seed + index,
                "effective_seed": baseline_seed + index,
                "seed_supported": True,
            }
            for index in range(5)
        ]
        payload["paper_baseline"] = {
            "schema_version": "c2hls.paper-baseline.v1",
            "method": baseline_method,
            "max_llm_candidates": 5,
            "max_synthesis_evaluations": 5,
            "correctness_order": "csim_golden_before_synthesis",
            "cosim_policy": "selected_winner_only",
            "base_seed": baseline_seed,
            "seed_policy": "base_plus_candidate_index",
            "candidate_seed_schedule": candidate_seed_schedule,
        }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return {
        "schema_version": "c2hls.run-fingerprint.v1",
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "payload": payload,
    }


def _frontier_entry(cycles: int, *, role: str) -> dict:
    return {
        "variant_name": role,
        "file": f"{role}.cpp",
        "step_name": "baseline" if role == "baseline" else "optimized",
        "selected": role == "expert",
        "benchmark_ready": True,
        "synthesis": {"status": "passed", "ran": True, "success": True},
        "csim": {"status": "passed", "ran": True, "passed": True},
        "cosim": {
            "status": "passed",
            "ran": True,
            "passed": True,
            "kernel_runtime_cycles": cycles,
        },
        "report": {"latency_cycles_worst": cycles, **REPORT_METRICS},
        "feasibility": {
            "feasible": True,
            "resource_fit": True,
            "timing_met": True,
        },
        "reference_contract_status": "passed",
        "public_contract_audit": {"passed": True, "differences": []},
    }


def _common_root(
    kernel: str,
    seed: int,
    *,
    strategy: str,
    skill_mode: str,
    frozen: bool = False,
    baseline_method: str | None = None,
) -> dict:
    fingerprint = _fingerprint(
        kernel,
        seed,
        strategy=strategy,
        skill_mode=skill_mode,
        frozen=frozen,
        baseline_method=baseline_method,
    )
    root = {
        "benchmark": kernel,
        "run_fingerprint": fingerprint,
        "run": {
            "run_fingerprint": fingerprint,
            "reference_blind": True,
            "reproducibility": {"complete": True, "issues": []},
            "elapsed_seconds": 75.0,
            "search_elapsed_seconds": 70.0,
            "preflight_elapsed_seconds": 4.0,
            "post_route_elapsed_seconds": 1.0,
            "total_elapsed_seconds": 75.0,
            "paper_method_wall_time_field": "search_elapsed_seconds",
        },
        "reference_isolation_audit": {"passed": True, "finding_count": 0},
        "reference_validation": {
            "benchmark_ready": True,
            "frontier_synthesis_csim_valid": True,
            "rtl_measurement_pair_valid": True,
            "validation_scope": "all",
            "reference_source": "local_vitis",
            "skipped_candidates": [],
            "selected_variant_name": "expert",
            "selected_variant_file": "expert.cpp",
            "selected_variant_step": "optimized",
            "selected_reference_cosim_measurement_valid": True,
            "baseline_reference_cosim_measurement_valid": True,
            "workflow": [
                _frontier_entry(1000, role="baseline"),
                _frontier_entry(500, role="expert"),
            ],
        },
    }
    root["reference_validation"]["baseline_reference"] = {
        key: copy.deepcopy(root["reference_validation"]["workflow"][0][key])
        for key in (
            "variant_name",
            "file",
            "step_name",
            "report",
            "synthesis",
            "csim",
            "cosim",
        )
    }
    return root


def _baseline_root(kernel: str = "k0", seed: int = 0) -> dict:
    root = _common_root(
        kernel,
        seed,
        strategy="one_shot_best_of_five",
        skill_mode="skill_off",
        baseline_method="one_shot_best_of_five",
    )
    candidates = []
    llm_events = []
    synthesis_events = []
    for index in range(5):
        latency = 1000 - index * 100
        response_hash = hashlib.sha256(f"response-{index}".encode()).hexdigest()
        code_hash = hashlib.sha256(f"code-{index}".encode()).hexdigest()
        candidates.append(
            {
                "index": index,
                "kind": "independent_full_translation",
                "response_sha256": response_hash,
                "code_extracted": True,
                "code_sha256": code_hash,
                "cumulative_elapsed_seconds": float((index + 1) * 10),
                "csim": {"status": "passed", "ran": True, "passed": True},
                "synthesis": {
                    "status": "passed",
                    "ran": True,
                    "success": True,
                },
                "report": {"latency_cycles_worst": latency, **REPORT_METRICS},
                "feasibility": {
                    "feasible": True,
                    "resource_fit": True,
                    "timing_met": True,
                },
            }
        )
        effective_seed = int(seed) + index
        llm_events.append(
            {
                **_llm_call_fields(effective_seed),
                "candidate_index": index,
                "requested_seed": effective_seed,
                "effective_seed": effective_seed,
                "seed_supported": True,
                "usage_available": True,
                "total_tokens": 100,
                "response_sha256": response_hash,
            }
        )
        synthesis_events.append(
            {
                "index": index,
                "candidate_index": index,
                "code_sha256": code_hash,
                "success": True,
                "elapsed_seconds": 2.0,
            }
        )
    root.update(
        {
            "schema_version": "c2hls.paper-baseline.v1",
            "method": "one_shot_best_of_five",
            "success": True,
            "correctness_status": "pass",
            "evaluation_status": {
                "schema_version": "c2hls.evaluation-status.v1",
                "correctness_status": "pass",
                "synthesis_status": "pass",
                "cosim_execution_status": "pass",
                "cosim_ran": True,
                "cosim_predicted_skip": False,
                "timeout": False,
                "tool_failure": False,
                "provider_failure": False,
            },
            "executed_cosim_status": "pass",
            "predicted_cosim_skip": False,
            "timeout_status": "none",
            "tool_failure_status": "none",
            "candidates": candidates,
            "candidate_count": 5,
            "selected_candidate_index": 4,
            "selected_code_sha256": candidates[4]["code_sha256"],
            "cosim_target_code_sha256": candidates[4]["code_sha256"],
            "final_report": dict(candidates[4]["report"]),
            "candidate_feasibility": dict(candidates[4]["feasibility"]),
            "cosim": {
                "status": "passed",
                "ran": True,
                "passed": True,
                "kernel_runtime_cycles": 650,
                "target_code_sha256": candidates[4]["code_sha256"],
            },
            "executed_cosim_cycles": 650,
            "llm_usage": {
                "calls": 5,
                "total_tokens": 500,
                "events": llm_events,
            },
            "synthesis_evaluations": {
                "count": 5,
                "events": synthesis_events,
            },
            # Five selection syntheses plus the selected-winner cosim flow.
            "synthesis_evaluation_count": 5,
            "total_synthesis_calls": 6,
            "total_tool_calls": 6,
            "selected_winner_cosim_count": 1,
            "post_route_implementation_count": 0,
        }
    )
    return root


def _agentic_root(kernel: str = "k0", seed: int = 0) -> dict:
    root = _common_root(
        kernel,
        seed,
        strategy="dynamic",
        skill_mode="skill_off",
    )
    final_report = {"latency_cycles_worst": 600, **REPORT_METRICS}
    first_code_hash = hashlib.sha256(b"agentic-code-0").hexdigest()
    selected_code_hash = hashlib.sha256(b"agentic-code-1").hexdigest()
    root.update(
        {
            "success": True,
            "correctness_status": "passed",
            "evaluation_status": {
                "schema_version": "c2hls.evaluation-status.v1",
                "correctness_status": "pass",
                "synthesis_status": "pass",
                "cosim_execution_status": "pass",
                "cosim_ran": True,
                "cosim_predicted_skip": False,
                "timeout": False,
                "tool_failure": False,
                "provider_failure": False,
            },
            "executed_cosim_status": "pass",
            "predicted_cosim_skip": False,
            "timeout_status": "none",
            "tool_failure_status": "none",
            "final_report": final_report,
            "selected_code_sha256": selected_code_hash,
            "cosim_target_code_sha256": selected_code_hash,
            "candidate_feasibility": {
                "feasible": True,
                "resource_fit": True,
                "timing_met": True,
            },
            "cosim": {
                "status": "passed",
                "ran": True,
                "passed": True,
                "kernel_runtime_cycles": 650,
            },
            "llm_usage": {
                "calls": 2,
                "candidate_requests": 2,
                "total_tokens": 200,
                "usage_missing_calls": 0,
                "events": [
                    {
                        **_llm_call_fields(seed),
                        "candidate_evaluation_index": 0,
                        "usage_available": True,
                        "total_tokens": 100,
                    },
                    {
                        **_llm_call_fields(seed),
                        "candidate_evaluation_index": 1,
                        "usage_available": True,
                        "total_tokens": 100,
                    },
                ],
            },
            "synthesis_evaluation_count": 1,
            "total_synthesis_calls": 2,
            "total_tool_calls": 2,
            "selected_winner_cosim_count": 1,
            "post_route_implementation_count": 0,
            "synthesis_evaluations": {
                "complete_candidate_event_stream": True,
                "count": 1,
                "events": [
                    {
                        "candidate_evaluation_index": 0,
                        "code_sha256": first_code_hash,
                        "report_sha256": None,
                        "cumulative_tokens": 100,
                        "cumulative_llm_calls": 1,
                        "cumulative_synthesis_evaluations": 0,
                        "cumulative_elapsed_seconds": 10.0,
                        "correctness_status": "failed",
                        "synthesis_status": "not_run",
                        "resource_fit": None,
                        "timing_met": None,
                        "synthesized_latency_cycles": None,
                        "latency_source": "none",
                        "failure_class": "wrong_output",
                        "selected_for_executed_cosim": False,
                    },
                    {
                        "candidate_evaluation_index": 1,
                        "code_sha256": selected_code_hash,
                        "report_sha256": _report_sha256(final_report),
                        "cumulative_tokens": 200,
                        "cumulative_llm_calls": 2,
                        "cumulative_synthesis_evaluations": 1,
                        "cumulative_elapsed_seconds": 40.0,
                        "correctness_status": "passed",
                        "synthesis_status": "passed",
                        "resource_fit": True,
                        "timing_met": True,
                        "synthesized_latency_cycles": 600,
                        "latency_source": "vitis_csynth_report",
                        "failure_class": None,
                        "selected_for_executed_cosim": True,
                    },
                ],
            },
        }
    )
    return root


def _agentic_terminal_failure(
    source: dict, correctness: str, *, provider_failure: bool = False
) -> None:
    is_timeout = correctness == "timeout"
    is_tool_failure = correctness == "tool_failure" or provider_failure
    source.update(
        {
            "success": False,
            "correctness_status": correctness,
            "executed_cosim_status": "not_run",
            "predicted_cosim_skip": False,
            "timeout_status": "timeout" if is_timeout else "none",
            "tool_failure_status": "tool_failure" if is_tool_failure else "none",
            "candidate_feasibility": {},
            "cosim": {
                "status": "not_run",
                "ran": False,
                "passed": False,
            },
            "synthesis_evaluation_count": 0,
            "total_synthesis_calls": 0,
            "total_tool_calls": 0,
            "selected_winner_cosim_count": 0,
            "post_route_implementation_count": 0,
            "selected_code_sha256": None,
            "cosim_target_code_sha256": None,
        }
    )
    source.pop("final_report", None)
    source["evaluation_status"].update(
        {
            "correctness_status": correctness,
            "synthesis_status": "not_run",
            "cosim_execution_status": "not_run",
            "cosim_ran": False,
            "timeout": is_timeout,
            "tool_failure": is_tool_failure,
            "provider_failure": provider_failure,
        }
    )
    llm_event = {
        **_llm_call_fields(
            source["run_fingerprint"]["payload"]["decoding"]["seed"]
        ),
        "candidate_evaluation_index": 0,
        "usage_available": True,
        "total_tokens": 100,
    }
    if provider_failure:
        llm_event["error"] = "provider unavailable"
        source["provider_failure"] = True
    source["llm_usage"] = {
        "calls": 1,
        "candidate_requests": 1,
        "total_tokens": 100,
        "usage_missing_calls": 0,
        "events": [llm_event],
    }
    source["synthesis_evaluations"] = {
        "complete_candidate_event_stream": True,
        "count": 0,
        "events": [
            {
                "candidate_evaluation_index": 0,
                "code_sha256": hashlib.sha256(b"failed-agentic-code").hexdigest(),
                "report_sha256": None,
                "cumulative_tokens": 100,
                "cumulative_llm_calls": 1,
                "cumulative_synthesis_evaluations": 0,
                "cumulative_elapsed_seconds": 10.0,
                "correctness_status": correctness,
                "synthesis_status": "not_run",
                "resource_fit": None,
                "timing_met": None,
                "synthesized_latency_cycles": None,
                "latency_source": "none",
                "failure_class": (
                    "tool_failure"
                    if correctness in {"timeout", "tool_failure"}
                    else "compile_or_interface_failure"
                ),
                "selected_for_executed_cosim": False,
            }
        ],
    }


class FreezeFixture:
    def __init__(self, root: Path, source: dict, *, method: str, runner: str):
        self.root = root
        self.source_path = root / "runner-result.json"
        self.transcript_path = root / "transcript.json"
        self.audit_path = root / "reference-isolation-audit.json"
        self.index_path = root / "freeze-index.json"
        self.method = method
        self.runner = runner
        self.source = source
        self.transcript_path.write_text(
            json.dumps({"messages": []}, indent=2) + "\n", encoding="utf-8"
        )
        self.source["reference_isolation_audit"] = {
            "schema_version": "c2hls.reference-isolation-audit.v1",
            "passed": True,
            "finding_count": 0,
            "finding_counts": {},
            "findings": [],
            "transcript_sha256": _sha256(self.transcript_path),
        }
        payload = self.source["run_fingerprint"]["payload"]
        json_hash = lambda value: hashlib.sha256(
            json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
        self.index = {
            "schema_version": "c2hls.hpca-freeze-index.v1",
            "target": dict(TARGET),
            "cohort": {
                "implementation_sha256": json_hash(payload["implementation"]),
                "prompts_sha256": json_hash(payload["prompts"]),
                "reference_isolation_sha256": json_hash(payload["reference_isolation"]),
                "decoding": {
                    key: payload["decoding"][key]
                    for key in ("temperature", "top_p", "max_completion_tokens")
                },
                "budgets": {
                    key: payload["budgets"][key]
                    for key in ("candidate_budget", "llm_candidate_budget")
                },
                "toolchain": {
                    "flow_target": "vitis",
                    "device_platform": "xilinx_u280_gen3x16_xdma_1_202211_1",
                },
            },
            "methods": [
                {
                    "id": method,
                    "display_name": "Test method",
                    "runner": runner,
                    "runner_method": method,
                    "model": {"id": MODEL_ID, "revision": MODEL_REVISION},
                }
            ],
            "expected_kernels": ["k0"],
            "expected_cells": [
                {"kernel": "k0", "seed": 0, "method": method}
            ],
            "generated_rows": [
                {
                    "kernel": "k0",
                    "seed": 0,
                    "method": method,
                    "runner": runner,
                    "run_id": "generated-k0-s0",
                    "artifact": {
                        "path": self.source_path.name,
                        "sha256": "pending",
                    },
                    "json_pointer": "",
                    "transcript": {
                        "artifact": {
                            "path": self.transcript_path.name,
                            "sha256": "pending",
                        },
                        "json_pointer": "",
                    },
                    "reference_isolation_audit": {
                        "artifact": {
                            "path": self.audit_path.name,
                            "sha256": "pending",
                        },
                        "json_pointer": "",
                    },
                }
            ],
            "frontiers": [
                {
                    "kernel": "k0",
                    "baseline": {
                        "source_kind": "reference_workflow_entry",
                        "run_id": "baseline-k0",
                        "artifact": {
                            "path": self.source_path.name,
                            "sha256": "pending",
                        },
                        "json_pointer": "/reference_validation/workflow/0",
                    },
                    "expert": {
                        "source_kind": "reference_workflow_entry",
                        "run_id": "expert-k0",
                        "artifact": {
                            "path": self.source_path.name,
                            "sha256": "pending",
                        },
                        "json_pointer": "/reference_validation/workflow/1",
                    },
                }
            ],
        }
        self.write()

    def write(self) -> None:
        self.audit_path.write_text(
            json.dumps(self.source["reference_isolation_audit"], indent=2) + "\n",
            encoding="utf-8",
        )
        self.source_path.write_text(
            json.dumps(self.source, indent=2) + "\n", encoding="utf-8"
        )
        digest = _sha256(self.source_path)
        self.index["generated_rows"][0]["artifact"]["sha256"] = digest
        self.index["generated_rows"][0]["transcript"]["artifact"]["sha256"] = _sha256(
            self.transcript_path
        )
        self.index["generated_rows"][0]["reference_isolation_audit"]["artifact"][
            "sha256"
        ] = _sha256(self.audit_path)
        for role in ("baseline", "expert"):
            self.index["frontiers"][0][role]["artifact"]["sha256"] = digest
        self.index_path.write_text(
            json.dumps(self.index, indent=2) + "\n", encoding="utf-8"
        )


class FreezeNormalizerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def test_normalizes_extended_paper_baseline_without_discovery(self):
        fixture = FreezeFixture(
            self.root,
            _baseline_root(),
            method="one_shot_best_of_five",
            runner="run_paper_baseline.py",
        )
        normalized = normalize_freeze_index(fixture.index_path)
        self.assertEqual(normalized["schema_version"], 2)
        record = normalized["evaluation_units"][0]["results"]["one_shot_best_of_five"]
        self.assertEqual(record["executed_cosim_cycles"], 650)
        self.assertEqual(record["selection_synthesis_evaluations"], 5)
        self.assertEqual(record["synthesis_calls"], 6)
        self.assertEqual(record["total_tool_calls"], 6)
        self.assertEqual(record["selected_winner_cosim_count"], 1)
        self.assertEqual(record["post_route_implementation_count"], 0)
        self.assertEqual(record["candidate_events"][-1]["cumulative_tokens"], 500)
        self.assertTrue(record["candidate_events"][-1]["selected_for_executed_cosim"])
        self.assertEqual(
            normalized["baseline_expert"][0]["expert"]["executed_cosim_cycles"], 500
        )
        provenance_text = json.dumps(normalized["normalization_provenance"])
        self.assertNotIn(self.source_path_name(fixture), provenance_text)
        self.assertIn("source_sha256", provenance_text)

    def test_post_route_implementation_is_attributed_outside_selection_budget(self):
        source = _agentic_root()
        source["post_route_implementation_count"] = 1
        source["total_synthesis_calls"] = 3
        source["total_tool_calls"] = 3
        fixture = FreezeFixture(
            self.root,
            source,
            method="dynamic_no_skills",
            runner="run_agentic_sweep.py",
        )
        normalized = normalize_freeze_index(fixture.index_path)
        record = normalized["evaluation_units"][0]["results"]["dynamic_no_skills"]
        self.assertEqual(1, record["selection_synthesis_evaluations"])
        self.assertEqual(1, record["selected_winner_cosim_count"])
        self.assertEqual(1, record["post_route_implementation_count"])
        self.assertEqual(3, record["synthesis_calls"])
        self.assertEqual(3, record["total_tool_calls"])

    def test_post_route_and_total_tool_attribution_tamper_are_rejected(self):
        tampered_sources = []
        missing_post_total = _agentic_root()
        missing_post_total["post_route_implementation_count"] = 1
        tampered_sources.append(missing_post_total)
        mismatched_total_tool = _agentic_root()
        mismatched_total_tool["total_tool_calls"] = 1
        tampered_sources.append(mismatched_total_tool)

        for index, source in enumerate(tampered_sources):
            with self.subTest(index=index):
                case_root = self.root / f"tamper-{index}"
                case_root.mkdir()
                fixture = FreezeFixture(
                    case_root,
                    source,
                    method="dynamic_no_skills",
                    runner="run_agentic_sweep.py",
                )
                with self.assertRaises(FreezeNormalizationError) as raised:
                    normalize_freeze_index(fixture.index_path)
                self.assertEqual("synthesis_attribution_mismatch", raised.exception.code)

    def test_raw_csynth_resources_and_fmax_are_normalized_with_u280_capacity(self):
        fixture = FreezeFixture(
            self.root,
            _baseline_root(),
            method="one_shot_best_of_five",
            runner="run_paper_baseline.py",
        )
        normalized = normalize_freeze_index(fixture.index_path)
        generated = normalized["evaluation_units"][0]["results"][
            "one_shot_best_of_five"
        ]
        metrics = generated["synthesis_metrics"]
        self.assertEqual("vitis_csynth_report", metrics["source"])
        self.assertEqual(312.5, metrics["fmax_mhz"])
        self.assertEqual(32, metrics["resources"]["bram"]["used"])
        self.assertEqual(4032, metrics["resources"]["bram"]["capacity"])
        self.assertAlmostEqual(
            32 / 4032, metrics["resources"]["bram"]["utilization"]
        )
        frontier = normalized["baseline_expert"][0]["expert"][
            "synthesis_metrics"
        ]
        self.assertEqual(960, frontier["resources"]["uram"]["capacity"])
        self.assertEqual(
            "xcu280_part_table",
            normalized["normalization_provenance"]["resource_capacity_source"],
        )

    def test_missing_resource_or_fmax_in_passing_synthesis_fails_closed(self):
        for field in (*REPORT_METRICS.keys(),):
            with self.subTest(field=field):
                source = _baseline_root()
                source["final_report"].pop(field)
                case_root = self.root / field
                case_root.mkdir()
                fixture = FreezeFixture(
                    case_root,
                    source,
                    method="one_shot_best_of_five",
                    runner="run_paper_baseline.py",
                )
                with self.assertRaises(FreezeNormalizationError) as raised:
                    normalize_freeze_index(fixture.index_path)
                self.assertIn(
                    raised.exception.code,
                    {"measured_resource_missing", "measured_fmax_missing"},
                )

    def test_frontier_passing_synthesis_requires_resource_fmax_evidence(self):
        source = _baseline_root()
        source["reference_validation"]["workflow"][1]["report"].pop("uram")
        fixture = FreezeFixture(
            self.root,
            source,
            method="one_shot_best_of_five",
            runner="run_paper_baseline.py",
        )
        with self.assertRaises(FreezeNormalizationError) as raised:
            normalize_freeze_index(fixture.index_path)
        self.assertEqual("measured_resource_missing", raised.exception.code)

    def test_explicit_u280_capacity_tampering_is_rejected(self):
        fixture = FreezeFixture(
            self.root,
            _baseline_root(),
            method="one_shot_best_of_five",
            runner="run_paper_baseline.py",
        )
        fixture.index["target"]["resource_capacities"] = {
            "bram": 1,
            "dsp": 9024,
            "ff": 2_607_360,
            "lut": 1_303_680,
            "uram": 960,
        }
        fixture.index_path.write_text(json.dumps(fixture.index) + "\n")
        with self.assertRaises(FreezeNormalizationError) as raised:
            normalize_freeze_index(fixture.index_path)
        self.assertEqual("resource_capacities_target_mismatch", raised.exception.code)

    def test_expected_cells_do_not_imply_a_cartesian_method_product(self):
        fixture = FreezeFixture(
            self.root,
            _baseline_root(),
            method="one_shot_best_of_five",
            runner="run_paper_baseline.py",
        )
        unused = copy.deepcopy(fixture.index["methods"][0])
        unused["id"] = "unused_method_cell"
        fixture.index["methods"].append(unused)
        fixture.index_path.write_text(json.dumps(fixture.index) + "\n")
        normalized = normalize_freeze_index(fixture.index_path)
        self.assertEqual(
            ["one_shot_best_of_five"],
            list(normalized["evaluation_units"][0]["results"]),
        )
        self.assertEqual(1, len(normalized["expected_cells"]))

    def test_provider_failure_is_authenticated_from_llm_events(self):
        source = _agentic_root()
        _agentic_terminal_failure(source, "not_run", provider_failure=True)
        fixture = FreezeFixture(
            self.root,
            source,
            method="dynamic_no_skills",
            runner="run_agentic_sweep.py",
        )
        normalized = normalize_freeze_index(fixture.index_path)
        record = normalized["evaluation_units"][0]["results"]["dynamic_no_skills"]
        self.assertEqual("tool_failure", record["failure_class"])
        self.assertTrue(record["provider_failure"])
        source = _agentic_root()
        _agentic_terminal_failure(source, "not_run", provider_failure=True)
        source["evaluation_status"]["provider_failure"] = False
        mismatch_root = self.root / "mismatch"
        mismatch_root.mkdir()
        mismatch = FreezeFixture(
            mismatch_root,
            source,
            method="dynamic_no_skills",
            runner="run_agentic_sweep.py",
        )
        with self.assertRaises(FreezeNormalizationError) as raised:
            normalize_freeze_index(mismatch.index_path)
        self.assertEqual("typed_status_mismatch", raised.exception.code)

    def test_csim_tool_failure_contributes_to_tool_failure_status(self):
        source = _agentic_root()
        _agentic_terminal_failure(source, "tool_failure")
        fixture = FreezeFixture(
            self.root,
            source,
            method="dynamic_no_skills",
            runner="run_agentic_sweep.py",
        )
        record = normalize_freeze_index(fixture.index_path)["evaluation_units"][0][
            "results"
        ]["dynamic_no_skills"]
        self.assertEqual("tool_failure", record["correctness_status"])
        self.assertEqual("tool_failure", record["failure_class"])

    def test_csim_timeout_contributes_to_timeout_status(self):
        source = _agentic_root()
        _agentic_terminal_failure(source, "timeout")
        fixture = FreezeFixture(
            self.root,
            source,
            method="dynamic_no_skills",
            runner="run_agentic_sweep.py",
        )
        record = normalize_freeze_index(fixture.index_path)["evaluation_units"][0][
            "results"
        ]["dynamic_no_skills"]
        self.assertEqual("timeout", record["correctness_status"])
        self.assertEqual("not_run", record["synthesis_status"])

    @staticmethod
    def source_path_name(fixture: FreezeFixture) -> str:
        return fixture.source_path.name

    def test_normalizes_enriched_agentic_candidate_stream(self):
        fixture = FreezeFixture(
            self.root,
            _agentic_root(),
            method="dynamic_no_skills",
            runner="run_agentic_sweep.py",
        )
        normalized = normalize_freeze_index(fixture.index_path)
        record = normalized["evaluation_units"][0]["results"]["dynamic_no_skills"]
        self.assertEqual(record["candidates_evaluated"], 2)
        self.assertEqual(record["selection_synthesis_evaluations"], 1)
        self.assertEqual(record["candidate_events"][0]["failure_class"], "wrong_output")

    def test_selected_winner_report_and_cosim_hashes_are_authenticated(self):
        cases = {
            "report": (
                lambda source: source["final_report"].update(lut=4001),
                "selected_winner_report_hash_mismatch",
            ),
            "cosim_target": (
                lambda source: source.update(cosim_target_code_sha256="f" * 64),
                "selected_winner_code_hash_mismatch",
            ),
        }
        for name, (mutate, expected_code) in cases.items():
            with self.subTest(name=name):
                source = _agentic_root()
                mutate(source)
                case_root = self.root / f"winner-{name}"
                case_root.mkdir()
                fixture = FreezeFixture(
                    case_root,
                    source,
                    method="dynamic_no_skills",
                    runner="run_agentic_sweep.py",
                )
                with self.assertRaises(FreezeNormalizationError) as raised:
                    normalize_freeze_index(fixture.index_path)
                self.assertEqual(expected_code, raised.exception.code)

    def test_actual_llm_call_telemetry_is_recomputed_not_trusted(self):
        def set_decoding(event: dict, **values) -> None:
            event["decoding"].update(values)

        cases = {
            "model": lambda event: event.update(model="wrong/model"),
            "revision": lambda event: event.update(model_revision="wrong-revision"),
            "prompt": lambda event: event.update(prompt_sha256="not-a-digest"),
            "token_cap": lambda event: event.update(max_tokens=4096),
            "temperature": lambda event: set_decoding(event, temperature=0.3),
            "top_p": lambda event: set_decoding(event, top_p=0.8),
            "seed": lambda event: set_decoding(event, seed=99),
            "seed_support": lambda event: (
                event.update(provider="local"),
                set_decoding(event, seed_supported=False, seed=None),
            ),
        }
        for name, mutate in cases.items():
            with self.subTest(name=name):
                source = _agentic_root()
                mutate(source["llm_usage"]["events"][0])
                case_root = self.root / name
                case_root.mkdir()
                fixture = FreezeFixture(
                    case_root,
                    source,
                    method="dynamic_no_skills",
                    runner="run_agentic_sweep.py",
                )
                with self.assertRaises(FreezeNormalizationError) as raised:
                    normalize_freeze_index(fixture.index_path)
                self.assertEqual(
                    "effective_llm_call_mismatch", raised.exception.code
                )

    def test_failed_frozen_skill_integrity_is_an_explicit_unmeasured_failure(self):
        source = _agentic_root()
        fingerprint = source["run_fingerprint"]
        fingerprint["payload"]["skills"].update(
            {
                "mode": "skill_on",
                "prompt_injection": True,
                "source_mode": "explicit_frozen_snapshot",
                "file_count": 1,
                "explicit_path_configured": True,
                "expected_sha256": "b" * 64,
                "matches_expected": True,
            }
        )
        _rehash_fingerprint(fingerprint)
        source["skill_snapshot_integrity"] = {"unchanged": False}
        source["success"] = False
        fixture = FreezeFixture(
            self.root,
            source,
            method="dynamic_frozen_skills",
            runner="run_agentic_sweep.py",
        )
        normalized = normalize_freeze_index(fixture.index_path)
        record = normalized["evaluation_units"][0]["results"]["dynamic_frozen_skills"]
        self.assertEqual(record["terminal_status"], "failure")
        self.assertEqual(record["failure_class"], "other")
        self.assertEqual(record["failure_detail"], "skill_snapshot_integrity_failure")
        self.assertIsNone(record["executed_cosim_cycles"])

    def test_string_seed_identity_is_preserved_without_numeric_coercion(self):
        fixture = FreezeFixture(
            self.root,
            _baseline_root(seed="01"),
            method="one_shot_best_of_five",
            runner="run_paper_baseline.py",
        )
        fixture.index["expected_cells"][0]["seed"] = "01"
        fixture.index["generated_rows"][0]["seed"] = "01"
        fixture.index_path.write_text(
            json.dumps(fixture.index, indent=2) + "\n", encoding="utf-8"
        )
        normalized = normalize_freeze_index(fixture.index_path)
        self.assertEqual(normalized["evaluation_units"][0]["seed"], "01")

    def test_current_baseline_schema_fails_with_exact_producer_gaps(self):
        source = _baseline_root()
        source.pop("total_synthesis_calls")
        for candidate in source["candidates"]:
            candidate.pop("cumulative_elapsed_seconds")
        fixture = FreezeFixture(
            self.root,
            source,
            method="one_shot_best_of_five",
            runner="run_paper_baseline.py",
        )
        with self.assertRaises(FreezeNormalizationError) as raised:
            normalize_freeze_index(fixture.index_path)
        error = raised.exception
        self.assertEqual(error.code, "paper_baseline_candidate_telemetry_incomplete")
        self.assertIn("total_synthesis_calls", error.missing_fields)
        self.assertIn("candidates[0].cumulative_elapsed_seconds", error.missing_fields)
        self.assertTrue(
            any(
                "PaperBaselineEngine._evaluate" in item
                for item in error.producer_functions
            )
        )

    def test_current_agentic_schema_fails_with_exact_producer_gaps(self):
        source = _agentic_root()
        source.pop("total_synthesis_calls")
        summary = source["synthesis_evaluations"]
        summary.pop("complete_candidate_event_stream")
        summary["events"] = [
            {
                "candidate_evaluation_index": 0,
                "code_sha256": "0" * 64,
                "synthesis_ran": True,
                "correctness_gate_passed": True,
                "success": True,
            }
        ]
        fixture = FreezeFixture(
            self.root,
            source,
            method="dynamic_no_skills",
            runner="run_agentic_sweep.py",
        )
        with self.assertRaises(FreezeNormalizationError) as raised:
            normalize_freeze_index(fixture.index_path)
        error = raised.exception
        self.assertEqual(error.code, "agentic_candidate_telemetry_incomplete")
        self.assertIn(
            "synthesis_evaluations.complete_candidate_event_stream",
            error.missing_fields,
        )
        self.assertIn(
            "synthesis_evaluations.events[*].cumulative_tokens", error.missing_fields
        )
        self.assertTrue(
            any("_synth_and_test" in item for item in error.producer_functions)
        )

    def test_hash_mismatch_is_rejected_before_parsing(self):
        fixture = FreezeFixture(
            self.root,
            _baseline_root(),
            method="one_shot_best_of_five",
            runner="run_paper_baseline.py",
        )
        fixture.index["generated_rows"][0]["artifact"]["sha256"] = "0" * 64
        fixture.index_path.write_text(
            json.dumps(fixture.index) + "\n", encoding="utf-8"
        )
        with self.assertRaises(FreezeNormalizationError) as raised:
            normalize_freeze_index(fixture.index_path)
        self.assertEqual(raised.exception.code, "artifact_hash_mismatch")

    def test_missing_row_is_not_converted_to_failure(self):
        fixture = FreezeFixture(
            self.root,
            _baseline_root(),
            method="one_shot_best_of_five",
            runner="run_paper_baseline.py",
        )
        fixture.index["generated_rows"] = []
        fixture.index_path.write_text(
            json.dumps(fixture.index) + "\n", encoding="utf-8"
        )
        with self.assertRaises(FreezeNormalizationError) as raised:
            normalize_freeze_index(fixture.index_path)
        self.assertEqual(raised.exception.code, "row_coverage_mismatch")

    def test_predicted_or_skipped_cosim_cycles_are_never_normalized(self):
        source = _baseline_root()
        source["cosim"] = {
            "status": "not_run",
            "ran": False,
            "passed": False,
            "skip_reason": "predicted_longer_than_gold",
            "cosim_policy": {"classification": "predicted_timeout"},
        }
        source["evaluation_status"].update(
            {
                "cosim_execution_status": "not_run",
                "cosim_ran": False,
                "cosim_predicted_skip": True,
            }
        )
        source["executed_cosim_status"] = "not_run"
        source["predicted_cosim_skip"] = True
        fixture = FreezeFixture(
            self.root,
            source,
            method="one_shot_best_of_five",
            runner="run_paper_baseline.py",
        )
        with self.assertRaises(FreezeNormalizationError) as raised:
            normalize_freeze_index(fixture.index_path)
        self.assertEqual(raised.exception.code, "predicted_cosim_forbidden")

    def test_fingerprint_digest_must_attest_payload(self):
        source = _baseline_root()
        source["run_fingerprint"]["payload"]["decoding"]["seed"] = 7
        fixture = FreezeFixture(
            self.root,
            source,
            method="one_shot_best_of_five",
            runner="run_paper_baseline.py",
        )
        with self.assertRaises(FreezeNormalizationError) as raised:
            normalize_freeze_index(fixture.index_path)
        self.assertEqual(raised.exception.code, "fingerprint_digest_mismatch")

    def test_output_is_atomic_deterministic_and_immutable(self):
        fixture = FreezeFixture(
            self.root,
            _baseline_root(),
            method="one_shot_best_of_five",
            runner="run_paper_baseline.py",
        )
        output = self.root / "normalized.json"
        normalize_to_file(fixture.index_path, output)
        first = output.read_bytes()
        normalize_to_file(fixture.index_path, output)
        self.assertEqual(first, output.read_bytes())

        second_index = copy.deepcopy(fixture.index)
        second_index["methods"][0]["display_name"] = "Changed display name"
        second_path = self.root / "second-freeze-index.json"
        second_path.write_text(json.dumps(second_index) + "\n", encoding="utf-8")
        with self.assertRaises(FreezeNormalizationError) as raised:
            normalize_to_file(second_path, output)
        self.assertEqual(raised.exception.code, "output_exists")
        self.assertEqual(first, output.read_bytes())


if __name__ == "__main__":
    unittest.main()
