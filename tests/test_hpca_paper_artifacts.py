import copy
import csv
import hashlib
import json
import math
import tempfile
import unittest
from pathlib import Path

from scripts.generate_hpca_paper_artifacts import (
    ManifestError,
    compute_analysis,
    expert_recovery,
    generate_bundle,
    load_and_validate,
    paired_bootstrap,
    verify_bundle,
)


METHODS = [
    ("one_shot", "Best-of-five one-shot"),
    ("pragma_only", "Pragma-only"),
    ("dynamic_no_skill", "Dynamic, no skill"),
    ("dynamic_skill", "Dynamic, frozen skill"),
]
CAPACITIES = {
    "bram": 4032,
    "dsp": 9024,
    "ff": 2_607_360,
    "lut": 1_303_680,
    "uram": 960,
}


def _synthesis_metrics(scale: int = 1, *, report_sha256: str = "a" * 64) -> dict:
    used = {
        "bram": 10 * scale,
        "dsp": 8 * scale,
        "ff": 1000 * scale,
        "lut": 500 * scale,
        "uram": scale,
    }
    return {
        "source": "vitis_csynth_report",
        "report_sha256": report_sha256,
        "fmax_mhz": 300.0 + scale,
        "resources": {
            key: {
                "used": used[key],
                "capacity": CAPACITIES[key],
                "utilization": used[key] / CAPACITIES[key],
            }
            for key in CAPACITIES
        },
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _successful_record(run_id: str, cycles: int, *, generated: bool) -> dict:
    selected_report_sha256 = hashlib.sha256(
        f"{run_id}:selected-report".encode()
    ).hexdigest()
    selected_code_sha256 = hashlib.sha256(
        f"{run_id}:selected-code".encode()
    ).hexdigest()
    record = {
        "run_id": run_id,
        "terminal_status": "success",
        "correctness_status": "passed",
        "synthesis_status": "passed",
        "resource_fit": True,
        "timing_met": True,
        "cosim_status": "passed",
        "cycle_source": "executed_rtl_cosim",
        "executed_cosim_cycles": cycles,
        "failure_class": None,
        "synthesis_metrics": _synthesis_metrics(
            report_sha256=selected_report_sha256
        ),
    }
    if generated:
        events = []
        for candidate_index in range(1, 6):
            code_sha256 = (
                selected_code_sha256
                if candidate_index == 5
                else hashlib.sha256(
                    f"{run_id}:code:{candidate_index}".encode()
                ).hexdigest()
            )
            events.append(
                {
                    "event_id": f"{run_id}.c{candidate_index}",
                    "candidate_index": candidate_index,
                    "code_sha256": code_sha256,
                    "report_sha256": (
                        selected_report_sha256
                        if candidate_index == 5
                        else hashlib.sha256(
                            f"{run_id}:report:{candidate_index}".encode()
                        ).hexdigest()
                    ),
                    "cumulative_tokens": candidate_index * 200,
                    "cumulative_llm_calls": candidate_index,
                    "cumulative_synthesis_evaluations": candidate_index,
                    "cumulative_elapsed_seconds": candidate_index * 20.0,
                    "correctness_status": "passed",
                    "synthesis_status": "passed",
                    "resource_fit": True,
                    "timing_met": True,
                    "synthesized_latency_cycles": cycles + (5 - candidate_index) * 50,
                    "latency_source": "vitis_csynth_report",
                    "failure_class": None,
                    "selected_for_executed_cosim": candidate_index == 5,
                }
            )
        record.update(
            {
                "reference_isolation_status": "passed",
                "selected_code_sha256": selected_code_sha256,
                "cosim_target_code_sha256": selected_code_sha256,
                "provider_failure": False,
                "tokens": 1000,
                "llm_calls": 5,
                "synthesis_calls": 5,
                "selection_synthesis_evaluations": 5,
                "wall_time_seconds": 120.0,
                "candidates_evaluated": 5,
                "candidate_events": events,
            }
        )
    return record


def _failed_record(run_id: str, failure_class: str = "cosim_failure") -> dict:
    record = _successful_record(run_id, 1, generated=True)
    for candidate_index, event in enumerate(record["candidate_events"], start=1):
        if candidate_index == 5:
            event["cumulative_synthesis_evaluations"] = 1
            continue
        event.update(
            {
                "cumulative_synthesis_evaluations": 0,
                "correctness_status": "failed",
                "synthesis_status": "not_run",
                "resource_fit": None,
                "timing_met": None,
                "synthesized_latency_cycles": None,
                "latency_source": "none",
                "report_sha256": None,
                "failure_class": "wrong_output",
            }
        )
    record.update(
        {
            "terminal_status": "failure",
            "cosim_status": "failed",
            "cycle_source": "none",
            "executed_cosim_cycles": None,
            "failure_class": failure_class,
            "synthesis_calls": 2,
            "selection_synthesis_evaluations": 1,
        }
    )
    return record


class Fixture:
    def __init__(self, root: Path):
        self.root = root
        self.result_path = root / "results.json"
        self.evidence_path = root / "evidence.json"
        self.leak_path = root / "leakage-audit.json"
        self.skill_path = root / "skill-snapshot.json"
        self.candidate_audit_path = root / "candidate-audit.json"
        self.fingerprint_audit_path = root / "fingerprint-audit.json"
        self.leak_path.write_text('{"status":"passed"}\n', encoding="utf-8")
        self.skill_path.write_text('{"frozen":true,"mutations":0}\n', encoding="utf-8")
        self.candidate_audit_path.write_text(
            '{"every_candidate_csim":true,"only_selected_winner_cosim":true}\n',
            encoding="utf-8",
        )
        self.fingerprint_audit_path.write_text(
            '{"all_fingerprints_complete":true,"resume_mismatches":0}\n',
            encoding="utf-8",
        )
        kernels = [f"k{i}" for i in range(8)]
        self.results = {
            "schema_version": 2,
            "methods": [{"id": method_id, "display_name": name} for method_id, name in METHODS],
            "expected_cells": [],
            "normalization_provenance": {
                "schema_version": "c2hls.hpca-freeze-normalizer.v1",
                "target": {
                    "vitis_version": "2023.2",
                    "part": "xcu280-fsvh2892-2L-e",
                    "clock_ns": "3.33",
                },
                "device_resource_capacities": dict(CAPACITIES),
                "resource_capacity_source": "xcu280_part_table",
            },
            "baseline_expert": [],
            "evaluation_units": [],
        }
        for index, kernel in enumerate(kernels):
            baseline = 1000 + index * 10
            expert = 500 + index * 5
            self.results["baseline_expert"].append(
                {
                    "kernel": kernel,
                    "baseline": _successful_record(f"baseline-{kernel}", baseline, generated=False),
                    "expert": _successful_record(f"expert-{kernel}", expert, generated=False),
                }
            )
            records = {
                "one_shot": _successful_record(f"one-{kernel}", 900 + index * 9, generated=True),
                "pragma_only": _successful_record(f"pragma-{kernel}", 850 + index * 8, generated=True),
                "dynamic_no_skill": _successful_record(f"noskill-{kernel}", 700 + index * 7, generated=True),
                "dynamic_skill": _successful_record(f"skill-{kernel}", 600 + index * 6, generated=True),
            }
            if index == 7:
                records["one_shot"] = _failed_record(f"one-{kernel}")
            self.results["evaluation_units"].append(
                {"kernel": kernel, "seed": 0, "results": records}
            )
            self.results["expected_cells"].extend(
                {"kernel": kernel, "seed": 0, "method": method_id}
                for method_id, _ in METHODS
            )
        self.evidence = {
            "schema_version": 2,
            "frozen": True,
            "evidence_freeze_timestamp": "2026-07-13T00:00:00Z",
            "run_set": {"path": self.result_path.name, "sha256": "pending"},
            "expected_kernels": kernels,
            "expected_methods": [method_id for method_id, _ in METHODS],
            "expected_cells": [
                {"kernel": kernel, "seed": 0, "method": method_id}
                for kernel in kernels
                for method_id, _ in METHODS
            ],
            "headline_units": [{"kernel": kernel, "seed": 0} for kernel in kernels],
            "profile_units": [{"kernel": kernel, "seed": 0} for kernel in kernels],
            "bootstrap_units": [{"kernel": kernel, "seed": 0} for kernel in kernels],
            "claim_methods": {
                "primary": "dynamic_skill",
                "one_shot": "one_shot",
                "dynamic_no_skill": "dynamic_no_skill",
                "dynamic_frozen_skill": "dynamic_skill",
            },
            "profile_taus": [1.0, 1.25, 2.0, 4.0, 10.0],
            "profile_tau_max": 10.0,
            "budget_synthesis_checkpoints": [1, 2, 3, 4, 5],
            "budget_token_checkpoints": [200, 400, 600, 800, 1000],
            "bootstrap": {"confidence": 0.95, "replicates": 10000, "seed": 2027},
            "policy": {"minimum_valid_baseline_expert_pairs": 8},
            "artifacts": [
                {"id": "leakage", "path": self.leak_path.name, "sha256": _sha256(self.leak_path)},
                {"id": "skills", "path": self.skill_path.name, "sha256": _sha256(self.skill_path)},
                {
                    "id": "candidate-audit",
                    "path": self.candidate_audit_path.name,
                    "sha256": _sha256(self.candidate_audit_path),
                },
                {
                    "id": "fingerprint-audit",
                    "path": self.fingerprint_audit_path.name,
                    "sha256": _sha256(self.fingerprint_audit_path),
                },
            ],
            "gate_evidence": {
                "transcript_leakage_audit": {"status": "passed", "artifact_id": "leakage"},
                "matched_budget": {"status": "passed", "candidate_limit": 5, "synthesis_limit": 5},
                "candidate_validation_audit": {
                    "status": "passed",
                    "artifact_id": "candidate-audit",
                },
                "complete_candidate_event_stream": {
                    "status": "passed",
                    "artifact_id": "candidate-audit",
                },
                "fingerprint_consistency_audit": {
                    "status": "passed",
                    "artifact_id": "fingerprint-audit",
                },
                "frozen_skill_snapshot": {
                    "status": "passed",
                    "artifact_id": "skills",
                    "frozen_before_evaluation": True,
                    "no_evaluation_persistence": True,
                },
            },
        }
        self.write()

    def write(self) -> None:
        self.result_path.write_text(json.dumps(self.results, indent=2) + "\n", encoding="utf-8")
        self.evidence["run_set"]["sha256"] = _sha256(self.result_path)
        self.evidence_path.write_text(json.dumps(self.evidence, indent=2) + "\n", encoding="utf-8")


class ArtifactGeneratorTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.fixture = Fixture(self.root)

    def tearDown(self):
        self.temporary.cleanup()

    def test_expert_recovery_formula(self):
        self.assertAlmostEqual(expert_recovery(1000, 750, 500), math.log(4 / 3) / math.log(2))
        self.assertAlmostEqual(expert_recovery(1000, 400, 500), math.log(2.5) / math.log(2))
        with self.assertRaises(ValueError):
            expert_recovery(1000, 900, 1000)

    def test_complete_bundle_has_claims_provenance_and_failure_denominator(self):
        destination = generate_bundle(self.fixture.evidence_path, self.root / "bundles")
        self.assertEqual(destination.name, _sha256(self.fixture.result_path))
        expected_files = {
            "ablation_table.tex",
            "artifact_manifest.json",
            "budget.pdf",
            "budget_curves.csv",
            "cell_provenance.json",
            "claim_decisions.json",
            "claim_to_artifact_manifest.json",
            "cost_summary.csv",
            "failure_accounting.csv",
            "paired_bootstrap.csv",
            "per_kernel_recovery.csv",
            "performance_profile.svg",
            "performance_profiles.csv",
            "recovery.svg",
            "recovery.pdf",
            "recovery_table.tex",
            "resource_table.tex",
            "resource_utilization_fmax.csv",
            "result_macros.tex",
            "render_provenance.json",
        }
        self.assertEqual({path.name for path in destination.iterdir()}, expected_files)

        with (destination / "per_kernel_recovery.csv").open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 8)
        self.assertAlmostEqual(float(rows[0]["recovery"]), expert_recovery(1000, 600, 500))
        with (destination / "performance_profiles.csv").open(encoding="utf-8") as handle:
            profiles = list(csv.DictReader(handle))
        # The one-shot failure remains infinity and therefore never enters a finite-tau profile.
        self.assertLessEqual(float(profiles[-1]["one_shot"]), 7 / 8)

        claims = json.loads((destination / "claim_decisions.json").read_text())
        self.assertEqual(claims["claims"]["headline_reference_blind_recovery"]["status"], "passed")
        self.assertEqual(claims["claims"]["frozen_skill_transfer"]["status"], "passed")
        self.assertEqual(claims["claims"]["compact_model_enablement"]["status"], "passed")
        self.assertEqual(claims["claims"]["post_route_and_board_validation"]["status"], "blocked")

        provenance = json.loads((destination / "cell_provenance.json").read_text())
        self.assertEqual(
            provenance["cells"]["recovery.k0.recovery"],
            ["baseline-k0", "skill-k0", "expert-k0"],
        )
        self.assertIn(
            "budget.synthesis_evaluations.5.dynamic_skill.qor_profile_auc",
            provenance["cells"],
        )
        claims_manifest = json.loads((destination / "claim_to_artifact_manifest.json").read_text())
        self.assertIn(
            "skills",
            claims_manifest["claims"]["frozen_skill_transfer"]["source_artifact_ids"],
        )
        manifest = json.loads((destination / "artifact_manifest.json").read_text())
        for filename, digest in manifest["outputs"].items():
            self.assertEqual(_sha256(destination / filename), digest)
        self.assertIn("Failures remain at infinity", (destination / "performance_profile.svg").read_text())
        self.assertIn("Paired profile-AUC deltas", (destination / "ablation_table.tex").read_text())
        self.assertTrue((destination / "recovery.pdf").read_bytes().startswith(b"%PDF-"))
        self.assertTrue((destination / "budget.pdf").read_bytes().startswith(b"%PDF-"))
        with (destination / "budget_curves.csv").open(encoding="utf-8") as handle:
            budget_rows = list(csv.DictReader(handle))
        self.assertTrue(any(row["failure_count"] != "0" for row in budget_rows))
        with (destination / "resource_utilization_fmax.csv").open(
            encoding="utf-8"
        ) as handle:
            resource_rows = list(csv.DictReader(handle))
        self.assertEqual(48, len(resource_rows))
        first_resource = resource_rows[0]
        self.assertEqual("301.0", first_resource["fmax_mhz"])
        self.assertEqual("4032", first_resource["bram_capacity"])
        self.assertAlmostEqual(
            10 / 4032, float(first_resource["bram_utilization"])
        )
        failed_row = next(
            row for row in resource_rows if row["run_id"] == "one-k7"
        )
        self.assertEqual("failure", failed_row["terminal_status"])
        self.assertEqual("cosim_failure", failed_row["failure_class"])
        self.assertEqual("available", failed_row["measurement_status"])
        with (destination / "cost_summary.csv").open(encoding="utf-8") as handle:
            cost = next(csv.DictReader(handle))
        self.assertEqual("301.0", cost["mean_fmax_mhz"])
        self.assertEqual("8", cost["qor_records"])
        self.assertTrue(verify_bundle(destination))

    def test_predicted_cycles_cannot_be_published(self):
        record = self.fixture.results["evaluation_units"][0]["results"]["dynamic_skill"]
        record["cycle_source"] = "predicted"
        self.fixture.write()
        with self.assertRaisesRegex(ManifestError, "without a passing executed RTL"):
            load_and_validate(self.fixture.evidence_path)

    def test_preregistered_statistics_cannot_drift(self):
        self.fixture.evidence["bootstrap"]["replicates"] = 9999
        self.fixture.write()
        with self.assertRaisesRegex(ManifestError, "preregistered paper contract"):
            load_and_validate(self.fixture.evidence_path)

    def test_reference_isolation_failure_is_retained_as_unmeasured_failure(self):
        record = self.fixture.results["evaluation_units"][0]["results"]["dynamic_skill"]
        record.update(
            {
                "terminal_status": "failure",
                "cosim_status": "not_run",
                "cycle_source": "none",
                "executed_cosim_cycles": None,
                "failure_class": "reference_isolation_failure",
                "reference_isolation_status": "failed",
            }
        )
        self.fixture.write()
        validated = load_and_validate(self.fixture.evidence_path)
        normalized = validated.evaluations[("k0", "0")]["dynamic_skill"]
        self.assertEqual(normalized["failure_class"], "reference_isolation_failure")
        self.assertIsNone(normalized["executed_cosim_cycles"])

    def test_predicted_candidate_latency_is_rejected(self):
        event = self.fixture.results["evaluation_units"][0]["results"]["dynamic_skill"]["candidate_events"][0]
        event["latency_source"] = "predicted"
        self.fixture.write()
        with self.assertRaisesRegex(ManifestError, "predicted data is forbidden"):
            load_and_validate(self.fixture.evidence_path)

    def test_incomplete_candidate_trace_is_rejected(self):
        self.fixture.results["evaluation_units"][0]["results"]["dynamic_skill"]["candidate_events"].pop()
        self.fixture.write()
        with self.assertRaisesRegex(ManifestError, "exactly one event"):
            load_and_validate(self.fixture.evidence_path)

    def test_selected_candidate_must_be_best_feasible_synthesized_state(self):
        record = self.fixture.results["evaluation_units"][0]["results"]["dynamic_skill"]
        events = record["candidate_events"]
        events[-1]["selected_for_executed_cosim"] = False
        events[0]["selected_for_executed_cosim"] = True
        record["selected_code_sha256"] = events[0]["code_sha256"]
        record["cosim_target_code_sha256"] = events[0]["code_sha256"]
        record["synthesis_metrics"]["report_sha256"] = events[0]["report_sha256"]
        self.fixture.write()
        with self.assertRaisesRegex(ManifestError, "not the minimum-latency"):
            load_and_validate(self.fixture.evidence_path)

    def test_selected_report_hash_must_match_final_synthesis_metrics(self):
        record = self.fixture.results["evaluation_units"][0]["results"]["dynamic_skill"]
        record["synthesis_metrics"]["report_sha256"] = "f" * 64
        self.fixture.write()
        with self.assertRaisesRegex(ManifestError, "report hash disagrees"):
            load_and_validate(self.fixture.evidence_path)

    def test_candidate_cannot_bypass_correctness_gate_before_synthesis(self):
        record = self.fixture.results["evaluation_units"][0]["results"]["dynamic_skill"]
        event = record["candidate_events"][0]
        event["correctness_status"] = "failed"
        event["failure_class"] = "wrong_output"
        self.fixture.write()
        with self.assertRaisesRegex(ManifestError, "bypasses the correctness gate"):
            load_and_validate(self.fixture.evidence_path)

    def test_missing_method_is_error_not_implicit_failure(self):
        del self.fixture.results["evaluation_units"][0]["results"]["pragma_only"]
        self.fixture.write()
        with self.assertRaisesRegex(ManifestError, "explicitly expected methods"):
            load_and_validate(self.fixture.evidence_path)

    def test_invalid_expert_frontier_is_visible_and_blocks_pair_gate(self):
        self.fixture.results["baseline_expert"][0]["expert"]["executed_cosim_cycles"] = 1100
        self.fixture.write()
        destination = generate_bundle(self.fixture.evidence_path, self.root / "bundles")
        with (destination / "per_kernel_recovery.csv").open(encoding="utf-8") as handle:
            first = next(csv.DictReader(handle))
        self.assertEqual(first["pair_valid"], "False")
        self.assertEqual(first["status"], "invalid_expert_frontier")
        claims = json.loads((destination / "claim_decisions.json").read_text())
        headline = claims["claims"]["headline_reference_blind_recovery"]
        self.assertEqual(headline["status"], "blocked")
        self.assertFalse(headline["gates"]["at_least_minimum_valid_baseline_expert_pairs"])

    def test_hash_mismatch_refuses_generation_before_writing(self):
        self.fixture.result_path.write_text("{}\n", encoding="utf-8")
        output_root = self.root / "bundles"
        with self.assertRaisesRegex(ManifestError, "hash mismatch"):
            generate_bundle(self.fixture.evidence_path, output_root)
        self.assertFalse(output_root.exists())

    def test_skill_claim_blocks_when_paired_interval_is_not_positive(self):
        for unit in self.fixture.results["evaluation_units"]:
            kernel_index = int(unit["kernel"][1:])
            unit["results"]["dynamic_skill"]["executed_cosim_cycles"] = 800 + kernel_index * 8
        self.fixture.write()
        analysis = compute_analysis(load_and_validate(self.fixture.evidence_path))
        self.assertLess(analysis["skill_bootstrap"]["ci_high"], 0)
        self.assertEqual(analysis["claims"]["frozen_skill_transfer"]["status"], "blocked")

    def test_replication_units_do_not_silently_reweight_primary_profile(self):
        extra = json.loads(json.dumps(self.fixture.results["evaluation_units"][0]))
        extra["seed"] = 1
        for method_id, record in extra["results"].items():
            record["run_id"] = f"{record['run_id']}-seed1"
        # Make the replicate favor one-shot; it must affect paired statistics
        # only because the evidence manifest excludes it from profile_units.
        # Pragma-only is deliberately absent from this sparse replication cell.
        del extra["results"]["pragma_only"]
        extra["results"]["one_shot"]["executed_cosim_cycles"] = 400
        self.fixture.results["evaluation_units"].append(extra)
        extra_cells = [
            {"kernel": "k0", "seed": 1, "method": method_id}
            for method_id in ("one_shot", "dynamic_no_skill", "dynamic_skill")
        ]
        self.fixture.results["expected_cells"].extend(extra_cells)
        self.fixture.evidence["expected_cells"].extend(copy.deepcopy(extra_cells))
        self.fixture.evidence["bootstrap_units"].append({"kernel": "k0", "seed": 1})
        self.fixture.write()
        analysis = compute_analysis(load_and_validate(self.fixture.evidence_path))
        self.assertEqual(len(analysis["ratios"]["one_shot"]), 8)
        self.assertEqual(analysis["skill_bootstrap"]["n"], 9)
        self.assertEqual(analysis["claims"]["compact_model_enablement"]["status"], "passed")

    def test_paired_bootstrap_is_deterministic_and_paired(self):
        result_a = paired_bootstrap([2, 3, 4], [1, 2, 3], replicates=500, confidence=0.95, seed=7)
        result_b = paired_bootstrap([2, 3, 4], [1, 2, 3], replicates=500, confidence=0.95, seed=7)
        self.assertEqual(result_a, result_b)
        self.assertEqual(result_a["estimate"], 1.0)
        self.assertEqual(result_a["ci_low"], 1.0)

    def test_immutable_bundle_refuses_overwrite(self):
        output_root = self.root / "bundles"
        generate_bundle(self.fixture.evidence_path, output_root)
        with self.assertRaisesRegex(ManifestError, "already exists"):
            generate_bundle(self.fixture.evidence_path, output_root)

    def test_resource_metric_tampering_is_rejected(self):
        record = self.fixture.results["evaluation_units"][0]["results"]["dynamic_skill"]
        record["synthesis_metrics"]["resources"]["bram"]["utilization"] = 0.5
        self.fixture.write()
        with self.assertRaisesRegex(ManifestError, "must equal used/capacity"):
            load_and_validate(self.fixture.evidence_path)

    def test_target_capacity_tampering_is_rejected(self):
        self.fixture.results["normalization_provenance"][
            "device_resource_capacities"
        ]["bram"] = 1
        self.fixture.write()
        with self.assertRaisesRegex(ManifestError, "XCU280 part table"):
            load_and_validate(self.fixture.evidence_path)

    def test_bundle_resource_csv_tampering_is_detected(self):
        destination = generate_bundle(self.fixture.evidence_path, self.root / "bundles")
        csv_path = destination / "resource_utilization_fmax.csv"
        csv_path.write_bytes(csv_path.read_bytes() + b"tampered\n")
        with self.assertRaisesRegex(ManifestError, "output hash mismatch"):
            verify_bundle(destination)

    def test_pdf_bytes_are_deterministic_for_same_frozen_evidence(self):
        first = generate_bundle(self.fixture.evidence_path, self.root / "bundles-a")
        second = generate_bundle(self.fixture.evidence_path, self.root / "bundles-b")
        for filename in ("recovery.pdf", "budget.pdf"):
            self.assertEqual(_sha256(first / filename), _sha256(second / filename))


if __name__ == "__main__":
    unittest.main()
