from __future__ import annotations

import copy
import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path

from scripts.freeze_hpca_skill_snapshot import (
    INPUT_SCHEMA,
    SnapshotValidationError,
    freeze_snapshot,
    verify_snapshot,
)


def _bytes(payload: object) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _skill(skill_id: str, *, confidence: str = "high") -> dict:
    return {
        "id": skill_id,
        "pattern": f"pattern for {skill_id}",
        "strategy": f"strategy for {skill_id}",
        "template": "#pragma HLS PIPELINE II=1",
        "confidence": confidence,
        "kind": "transformation",
        "bottleneck_kinds": ["ii_target_miss"],
        "applicable_versions": ["2023.2"],
        "applicable_fpgas": ["xcu280-fsvh2892-2L-e"],
        "tags": ["test"],
        "guards": ["preserve behavior"],
        "required_steps": ["run correctness"],
        # These mutable source statistics must not flow into the frozen
        # snapshot.  The freezer derives fresh statistics only from accepted
        # trajectory evidence.
        "occurrences": 999,
        "sec_pass": 1,
        "mean_advantage": 99.0,
        "last_used_at": "2026-01-01T00:00:00Z",
        "origin": "manual",
    }


def _valid_trajectory(kernel: str = "adi") -> dict:
    golden_hash = "a" * 64
    passing_csim = {
        "status": "passed",
        "ran": True,
        "passed": True,
        "success": True,
        "golden_output_sha256": golden_hash,
        "correctness": {
            "passed": True,
            "correctness_status": "passed",
            "reason": "match",
        },
    }
    return {
        "benchmark": f"hlsfactory_{kernel}",
        "source_repo": "HLSFactory",
        "correctness_status": "passed",
        "independent_golden": {
            "schema_version": "c2hls.independent-golden.v1",
            "required": True,
            "status": "passed",
            "source": "pragma_stripped_plain_c_and_public_testbench",
            "output_sha256": golden_hash,
            "specs_sha256": "b" * 64,
            "output_count": 2,
            "value_count": 64,
        },
        "csim": copy.deepcopy(passing_csim),
        "synthesis_evaluations": {
            "schema_version": "c2hls.synthesis-evaluations.v1",
            "count": 1,
            "events": [
                {"synthesis_ran": True, "success": True, "synthesis_index": 0}
            ],
        },
        "candidate_feasibility": {"feasible": True, "reasons": []},
        "baseline_report": {
            "latency_cycles_worst": 1600,
            "bram": 1,
            "dsp": 2,
            "ff": 3,
            "lut": 4,
            "uram": 0,
        },
        "baseline_csim": copy.deepcopy(passing_csim),
        "steps": [
            {
                "step_name": "pipeline",
                "success": True,
                "routing_decision": {"skill_id": "alpha", "fallback": False},
                "skill_prompt": {
                    "requested_skill_id": "alpha",
                    "injected": True,
                    "injected_skill_ids": ["alpha"],
                },
                "csim": copy.deepcopy(passing_csim),
                "feasibility": {"feasible": True, "reasons": []},
                "report": {
                    "latency_cycles_worst": 1200,
                    "bram": 1,
                    "dsp": 2,
                    "ff": 3,
                    "lut": 4,
                    "uram": 0,
                },
            }
        ],
        "final_report": {
            "latency_cycles_worst": 1200,
            "bram": 1,
            "dsp": 2,
            "ff": 3,
            "lut": 4,
            "uram": 0,
        },
    }


class FrozenSkillSnapshotTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.skill_source = self.root / "skills.json"
        self.skill_source.write_bytes(
            _bytes({"schema": "1.1", "skills": [_skill("alpha"), _skill("beta")]})
        )
        self.trajectory = self.root / "adi-result.json"
        self.trajectory.write_bytes(_bytes(_valid_trajectory()))
        self.manifest = self.root / "review.json"
        self.output_root = self.root / "snapshots"
        self._write_manifest()

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _write_manifest(
        self,
        *,
        kernel: str = "adi",
        trajectory_hash: str | None = None,
        trajectories: list[dict] | None = None,
        skill_id: str = "alpha",
    ) -> dict:
        if trajectories is None:
            trajectories = [
                {
                    "path": self.trajectory.name,
                    "sha256": trajectory_hash or _sha256(self.trajectory),
                    "kernel": kernel,
                    "source_suite": "HLSFactory",
                    "benchmark_role": "development",
                    "validated_skills": [
                        {"id": skill_id, "relative_advantage": 0.25}
                    ],
                }
            ]
        payload = {
            "schema_version": INPUT_SCHEMA,
            "source_suite": "HLSFactory",
            "benchmark_role": "development",
            "skill_source": {
                "path": self.skill_source.name,
                "sha256": _sha256(self.skill_source),
            },
            "evaluation_kernels": ["private_eval_case"],
            "trajectories": trajectories,
        }
        self.manifest.write_bytes(_bytes(payload))
        return payload

    def test_freezes_only_reviewed_skill_and_rebuilds_statistics(self) -> None:
        source_before = self.skill_source.read_bytes()
        trajectory_hash = _sha256(self.trajectory)

        result = freeze_snapshot(self.manifest, self.output_root)

        self.assertTrue(result["created"])
        snapshot_path = Path(result["snapshot_path"])
        self.assertTrue(snapshot_path.name.startswith("sha256-"))
        self.assertEqual(source_before, self.skill_source.read_bytes())
        frozen = json.loads((snapshot_path / "skills.json").read_text())
        self.assertEqual(["alpha"], [entry["id"] for entry in frozen["skills"]])
        alpha = frozen["skills"][0]
        self.assertEqual(1, alpha["occurrences"])
        self.assertEqual(1, alpha["sec_pass"])
        self.assertEqual(0.25, alpha["mean_advantage"])
        self.assertIsNone(alpha["last_used_at"])
        self.assertEqual("medium", alpha["confidence"])

        descriptor = result["content_descriptor"]
        self.assertEqual(
            trajectory_hash,
            descriptor["trajectory_evidence"][0]["trajectory_sha256"],
        )
        self.assertEqual(
            trajectory_hash,
            descriptor["skill_evidence"][0]["observations"][0][
                "trajectory_sha256"
            ],
        )
        observation = descriptor["skill_evidence"][0]["observations"][0]
        self.assertEqual(0, observation["step_evidence"]["step_index"])
        self.assertEqual("latency_cycles_worst", observation["step_evidence"]["previous_latency_field"])
        self.assertEqual("latency_cycles_worst", observation["step_evidence"]["current_latency_field"])
        self.assertEqual(0.25, observation["relative_advantage"])
        self.assertEqual(result["content_id"], verify_snapshot(snapshot_path)["content_id"])

        repeated = freeze_snapshot(self.manifest, self.output_root)
        self.assertFalse(repeated["created"])
        self.assertEqual(snapshot_path, Path(repeated["snapshot_path"]))

    def test_rejects_rodinia_and_manifest_declared_evaluation_kernels(self) -> None:
        for kernel in ("pathfinder", "private_eval_case"):
            with self.subTest(kernel=kernel):
                trajectory = _valid_trajectory(kernel)
                self.trajectory.write_bytes(_bytes(trajectory))
                self._write_manifest(kernel=kernel)
                with self.assertRaisesRegex(
                    SnapshotValidationError, "primary evaluation kernel"
                ):
                    freeze_snapshot(self.manifest, self.output_root)

    def test_requires_independent_golden_candidate_correctness(self) -> None:
        mutations = {
            "oracle_not_required": lambda value: value["independent_golden"].update(
                {"required": False}
            ),
            "oracle_failed": lambda value: value["independent_golden"].update(
                {"status": "invalid"}
            ),
            "candidate_failed": lambda value: value["csim"]["correctness"].update(
                {"passed": False, "correctness_status": "failed"}
            ),
            "wrong_oracle_hash": lambda value: value["csim"].update(
                {"golden_output_sha256": "c" * 64}
            ),
            "empty_output": lambda value: value["independent_golden"].update(
                {"value_count": 0}
            ),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                payload = _valid_trajectory()
                mutate(payload)
                self.trajectory.write_bytes(_bytes(payload))
                self._write_manifest()
                with self.assertRaises(SnapshotValidationError):
                    freeze_snapshot(self.manifest, self.output_root)

    def test_requires_executed_successful_synthesis_and_feasibility(self) -> None:
        mutations = {
            "zero_syntheses": lambda value: value["synthesis_evaluations"].update(
                {"count": 0}
            ),
            "no_success_event": lambda value: value["synthesis_evaluations"].update(
                {"events": [{"synthesis_ran": True, "success": False}]}
            ),
            "not_executed": lambda value: value["synthesis_evaluations"].update(
                {"events": [{"synthesis_ran": False, "success": True}]}
            ),
            "infeasible": lambda value: value["candidate_feasibility"].update(
                {"feasible": False}
            ),
            "no_latency": lambda value: value["final_report"].pop(
                "latency_cycles_worst"
            ),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                payload = _valid_trajectory()
                mutate(payload)
                self.trajectory.write_bytes(_bytes(payload))
                self._write_manifest()
                with self.assertRaises(SnapshotValidationError):
                    freeze_snapshot(self.manifest, self.output_root)

    def test_rejects_unused_and_routed_but_not_injected_skills(self) -> None:
        payload = _valid_trajectory()
        payload["steps"][0]["routing_decision"]["skill_id"] = "beta"
        payload["steps"][0]["skill_prompt"].update(
            {"requested_skill_id": "beta", "injected_skill_ids": ["beta"]}
        )
        self.trajectory.write_bytes(_bytes(payload))
        self._write_manifest()
        with self.assertRaisesRegex(SnapshotValidationError, "no successful step"):
            freeze_snapshot(self.manifest, self.output_root)

        payload = _valid_trajectory()
        payload["steps"][0]["skill_prompt"].update(
            {"injected": False, "injected_skill_ids": []}
        )
        self.trajectory.write_bytes(_bytes(payload))
        self._write_manifest()
        with self.assertRaisesRegex(SnapshotValidationError, "routes skill.*never injects"):
            freeze_snapshot(self.manifest, self.output_root)

    def test_rejects_injected_failed_step_and_final_pass_masking(self) -> None:
        mutations = {
            "failed_step": lambda step: step.update(
                {"success": False, "error": "synthesis failed"}
            ),
            "step_csim_failed": lambda step: step["csim"].update(
                {"passed": False, "success": False}
            ),
            "step_golden_missing": lambda step: step["csim"].pop(
                "golden_output_sha256"
            ),
            "step_infeasible": lambda step: step["feasibility"].update(
                {"feasible": False, "reasons": ["timing"]}
            ),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                payload = _valid_trajectory()
                # Top-level/final evidence deliberately remains passing.
                mutate(payload["steps"][0])
                self.trajectory.write_bytes(_bytes(payload))
                self._write_manifest()
                with self.assertRaises(SnapshotValidationError):
                    freeze_snapshot(self.manifest, self.output_root)

    def test_derives_advantage_and_rejects_forged_declaration(self) -> None:
        payload = self._write_manifest()
        payload["trajectories"][0]["validated_skills"][0][
            "relative_advantage"
        ] = 0.99
        self.manifest.write_bytes(_bytes(payload))
        with self.assertRaisesRegex(SnapshotValidationError, "derive"):
            freeze_snapshot(self.manifest, self.output_root)

        payload["trajectories"][0]["validated_skills"][0].pop(
            "relative_advantage"
        )
        self.manifest.write_bytes(_bytes(payload))
        result = freeze_snapshot(self.manifest, self.output_root)
        observation = result["content_descriptor"]["skill_evidence"][0][
            "observations"
        ][0]
        self.assertEqual(0.25, observation["relative_advantage"])

    def test_advantage_uses_previous_accepted_report_and_ns_fallback(self) -> None:
        payload = _valid_trajectory()
        payload["baseline_report"].pop("latency_cycles_worst")
        payload["baseline_report"]["latency_ns"] = 1000

        first = copy.deepcopy(payload["steps"][0])
        first["step_name"] = "tiling"
        first["routing_decision"]["skill_id"] = "beta"
        first["skill_prompt"].update(
            {"requested_skill_id": "beta", "injected_skill_ids": ["beta"]}
        )
        first["report"].pop("latency_cycles_worst")
        first["report"]["latency_ns"] = 800

        second = copy.deepcopy(payload["steps"][0])
        second["report"].pop("latency_cycles_worst")
        second["report"]["latency_ns"] = 600
        payload["steps"] = [first, second]
        self.trajectory.write_bytes(_bytes(payload))
        self._write_manifest()

        result = freeze_snapshot(self.manifest, self.output_root)
        observation = result["content_descriptor"]["skill_evidence"][0][
            "observations"
        ][0]
        proof = observation["step_evidence"]
        self.assertEqual(1, proof["step_index"])
        self.assertEqual("latency_ns", proof["previous_latency_field"])
        self.assertEqual(800, proof["previous_latency"])
        self.assertEqual(600, proof["current_latency"])
        self.assertEqual(0.25, observation["relative_advantage"])

    def test_verifies_and_fingerprints_sibling_source_snapshot_bundle(self) -> None:
        source_result = freeze_snapshot(self.manifest, self.root / "source-snapshots")
        source_snapshot = Path(source_result["snapshot_path"])
        payload = json.loads(self.manifest.read_text())
        source_path = source_snapshot / "skills.json"
        payload["skill_source"] = {
            "path": source_path.relative_to(self.root).as_posix(),
            "sha256": _sha256(source_path),
        }
        self.manifest.write_bytes(_bytes(payload))

        result = freeze_snapshot(self.manifest, self.root / "derived-snapshots")
        bundle = result["content_descriptor"]["source_skill_bundle"]
        self.assertTrue(bundle["present"])
        self.assertEqual(source_result["content_id"], bundle["content_id"])
        self.assertEqual(
            _sha256(source_snapshot / "snapshot_manifest.json"),
            bundle["snapshot_manifest_sha256"],
        )

        manifest_path = source_snapshot / "snapshot_manifest.json"
        original = manifest_path.read_bytes()
        manifest_path.chmod(0o644)
        manifest_path.write_bytes(original + b" ")
        with self.assertRaises(SnapshotValidationError):
            freeze_snapshot(self.manifest, self.root / "tampered-derived")

    def test_rejects_hash_mismatch_duplicate_trajectory_and_unknown_skill(self) -> None:
        with self.assertRaisesRegex(SnapshotValidationError, "hash mismatch"):
            self._write_manifest(trajectory_hash="0" * 64)
            freeze_snapshot(self.manifest, self.output_root)

        one = self._write_manifest()["trajectories"][0]
        with self.assertRaisesRegex(SnapshotValidationError, "duplicates a trajectory"):
            self._write_manifest(trajectories=[copy.deepcopy(one), copy.deepcopy(one)])
            freeze_snapshot(self.manifest, self.output_root)

        self._write_manifest(skill_id="does-not-exist")
        with self.assertRaisesRegex(SnapshotValidationError, "unknown skill id"):
            freeze_snapshot(self.manifest, self.output_root)

    def test_existing_snapshot_is_never_overwritten_and_tampering_is_rejected(self) -> None:
        result = freeze_snapshot(self.manifest, self.output_root)
        snapshot_path = Path(result["snapshot_path"])
        skills_path = snapshot_path / "skills.json"
        original = skills_path.read_bytes()
        skills_path.chmod(0o644)
        skills_path.write_bytes(original + b" ")

        with self.assertRaisesRegex(SnapshotValidationError, "hash-mismatched"):
            freeze_snapshot(self.manifest, self.output_root)
        with self.assertRaisesRegex(SnapshotValidationError, "hash"):
            verify_snapshot(snapshot_path)
        self.assertEqual(original + b" ", skills_path.read_bytes())


if __name__ == "__main__":
    unittest.main()
