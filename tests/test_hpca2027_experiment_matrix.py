from __future__ import annotations

import copy
import importlib.util
import os
import sys
import unittest
from pathlib import Path
from unittest import mock


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "validate_hpca2027_matrix.py"
SPEC = importlib.util.spec_from_file_location("validate_hpca2027_matrix", SCRIPT)
assert SPEC and SPEC.loader
matrix_tool = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = matrix_tool
SPEC.loader.exec_module(matrix_tool)


class HPCA2027ExperimentMatrixTests(unittest.TestCase):
    def setUp(self) -> None:
        self.matrix = matrix_tool.load_matrix()

    def test_checked_in_matrix_and_golden_testbenches_validate(self) -> None:
        issues = matrix_tool.validate_matrix(self.matrix, repo=REPO)
        self.assertEqual([], [issue for issue in issues if issue.severity == "error"])
        cosim_warnings = [
            issue for issue in issues if issue.code == "selected_cosim_not_enabled"
        ]
        self.assertEqual(0, len(cosim_warnings))
        self.assertEqual(
            "1",
            self.matrix["common_runner_env"]["C2HLS_FORCE_SELECTED_COSIM"],
        )
        self.assertEqual(
            "1",
            self.matrix["common_runner_env"]["C2HLS_CORRECTNESS_BEFORE_SYNTH"],
        )
        self.assertEqual("qwen3.6-27b", self.matrix["models"]["qwen_27b"]["model_id"])

    def test_expected_campaign_expansion_and_mapping_counts(self) -> None:
        rows = matrix_tool.expand_matrix(self.matrix)
        self.assertEqual(132, len(rows))
        self.assertEqual(90, sum(row["campaign"] == "primary_seed0" for row in rows))
        self.assertEqual(
            36,
            sum(row["campaign"] == "representative_extra_seeds" for row in rows),
        )
        self.assertEqual(
            6,
            sum(row["campaign"] == "oracle_upper_bound_seed0" for row in rows),
        )
        self.assertEqual(
            132,
            sum(row["execution"]["mapping_status"] == "supported" for row in rows),
        )
        self.assertEqual(
            0,
            sum(row["execution"]["mapping_status"] == "unsupported" for row in rows),
        )
        self.assertEqual(len(rows), len({row["row_id"] for row in rows}))

    def test_primary_cross_product_is_complete_at_seed_zero(self) -> None:
        rows = [
            row
            for row in matrix_tool.expand_matrix(self.matrix)
            if row["campaign"] == "primary_seed0"
        ]
        observed = {
            (row["benchmark"], row["model_key"], row["method"], row["seed"])
            for row in rows
        }
        expected = {
            (benchmark, model, method, 0)
            for benchmark in matrix_tool.PRIMARY_BENCHMARKS
            for model in matrix_tool.MODEL_KEYS
            for method in matrix_tool.METHOD_KEYS
        }
        self.assertEqual(expected, observed)

    def test_replication_is_only_three_methods_three_strata_two_extra_seeds(self) -> None:
        rows = [
            row
            for row in matrix_tool.expand_matrix(self.matrix)
            if row["campaign"] == "representative_extra_seeds"
        ]
        self.assertEqual({1, 2}, {row["seed"] for row in rows})
        self.assertEqual(
            set(matrix_tool.REPRESENTATIVE_STRATA),
            {row["benchmark"] for row in rows},
        )
        self.assertEqual(
            set(matrix_tool.REPLICATION_METHODS), {row["method"] for row in rows}
        )
        self.assertEqual(set(matrix_tool.MODEL_KEYS), {row["model_key"] for row in rows})
        qwen_rows = [row for row in rows if row["model_key"] == "qwen_27b"]
        claude_rows = [
            row for row in rows if row["model_key"] == "claude_sonnet_4_6"
        ]
        self.assertTrue(all(row["seed_control"] == "provider_enforced" for row in qwen_rows))
        self.assertTrue(
            all(
                row["seed_control"]
                == "unsupported_by_provider_record_as_repeated_trial"
                for row in claude_rows
            )
        )

    def test_explicit_method_model_cells_match_all_132_planned_rows_without_cartesian_fill(self) -> None:
        rows = matrix_tool.expand_matrix(self.matrix)
        cells = {
            (
                row["campaign"],
                row["benchmark"],
                row["seed"],
                f"{row['model_key']}::{row['method']}",
            )
            for row in rows
        }
        self.assertEqual(132, len(cells))
        replication = [
            row
            for row in rows
            if row["campaign"] == "representative_extra_seeds"
        ]
        self.assertFalse(
            {"pragma_only", "flash_c2hls"}
            & {row["method"] for row in replication}
        )
        full_replication_product = (
            len(matrix_tool.REPRESENTATIVE_STRATA)
            * 2
            * len(matrix_tool.MODEL_KEYS)
            * len(matrix_tool.METHOD_KEYS)
        )
        self.assertEqual(24, full_replication_product - len(replication))

    def test_oracle_is_separate_guided_three_kernel_ablation(self) -> None:
        rows = [
            row
            for row in matrix_tool.expand_matrix(self.matrix)
            if row["campaign"] == "oracle_upper_bound_seed0"
        ]
        self.assertEqual(set(matrix_tool.REPRESENTATIVE_STRATA), {r["benchmark"] for r in rows})
        self.assertTrue(all(row["oracle_upper_bound"] for row in rows))
        self.assertTrue(all(not row["reference_blind"] for row in rows))
        self.assertEqual({"dynamic_frozen_skills"}, {row["method"] for row in rows})
        self.assertEqual({0}, {row["seed"] for row in rows})
        for row in rows:
            env = row["execution"]["environment"]
            self.assertEqual("legacy", env["C2HLS_SWEEP_PROFILE"])
            self.assertEqual("1", env["C2HLS_ORACLE_MODE"])
            self.assertEqual("1", env["C2HLS_GT_COMPARISON_IN_CONTROL"])

    def test_every_row_has_the_same_global_candidate_and_synthesis_contract(self) -> None:
        for row in matrix_tool.expand_matrix(self.matrix):
            self.assertEqual(5, row["budget"]["max_llm_candidates"])
            self.assertEqual(5, row["budget"]["max_synthesis_evaluations"])
            self.assertEqual("selected_winner_only", row["budget"]["cosim_policy"])
            if row["execution"]["mapping_status"] == "supported":
                env = row["execution"]["environment"]
                self.assertEqual("5", env["C2HLS_CANDIDATES_PER_STEP"])
                self.assertEqual("5", env["C2HLS_LLM_CANDIDATE_BUDGET"])
                self.assertEqual("5", env["C2HLS_SYNTHESIS_EVAL_BUDGET"])
                self.assertEqual("1", env["C2HLS_COSIM_SELECTED_ONLY"])
                self.assertEqual("1", env["C2HLS_CORRECTNESS_BEFORE_SYNTH"])
                self.assertEqual("8192", env["C2HLS_MAX_COMPLETION_TOKENS"])
                for role in (
                    "C2HLS_TRANSLATOR_MODEL",
                    "C2HLS_SYNTHESIS_MODEL",
                    "C2HLS_QUALITY_REPAIR_MODEL",
                    "C2HLS_FEEDBACK_MODEL",
                ):
                    self.assertEqual(row["model_id"], env[role])

    def test_baselines_use_dedicated_fingerprinted_runner(self) -> None:
        rows = matrix_tool.expand_matrix(self.matrix)
        for method in ("one_shot_best_of_five", "pragma_only"):
            method_rows = [row for row in rows if row["method"] == method]
            self.assertTrue(method_rows)
            for row in method_rows:
                execution = row["execution"]
                self.assertEqual("supported", execution["mapping_status"])
                self.assertEqual(
                    ["python", "run_paper_baseline.py", "--method", method],
                    execution["command"],
                )
                self.assertEqual(method, execution["environment"]["C2HLS_BASELINE_METHOD"])
                self.assertEqual(method, execution["environment"]["C2HLS_STRATEGY"])
                self.assertEqual("", execution["required_implementation"])

    def test_primary_supported_mapping_disables_all_reference_control(self) -> None:
        row = next(
            row
            for row in matrix_tool.expand_matrix(self.matrix)
            if row["campaign"] == "primary_seed0"
            and row["method"] == "dynamic_no_skills"
        )
        env = row["execution"]["environment"]
        self.assertEqual("1", env["C2HLS_REFERENCE_BLIND"])
        self.assertEqual("0", env["C2HLS_GT_COMPARISON_IN_CONTROL"])
        self.assertEqual("0", env["C2HLS_PHASE8_BASELINE_ALIGN"])
        self.assertEqual("0", env["C2HLS_PHASE5_GT_PREPOP"])
        self.assertEqual("0", env["C2HLS_REFERENCE_CODE_IN_PROMPTS"])
        self.assertEqual("0", env["C2HLS_REFERENCE_METRICS_IN_PROMPTS"])

    def test_resolving_identity_env_never_serializes_credentials(self) -> None:
        fake_env = {
            "OPENAI_BASE_URL": "http://local.example/v1",
            "C2HLS_QWEN27B_REVISION": "weights-sha256-abc",
            "ANTHROPIC_API_KEY": "secret-must-not-appear",
            "C2HLS_SONNET46_REVISION": "provider-version-123",
            "C2HLS_SKILL_SNAPSHOT_SHA256": "skill-sha256-def",
        }
        with mock.patch.dict(os.environ, fake_env, clear=True):
            rows = matrix_tool.expand_matrix(self.matrix, resolve_env=True)
        serialized = repr(rows)
        self.assertNotIn("secret-must-not-appear", serialized)
        qwen_row = next(
            row
            for row in rows
            if row["model_key"] == "qwen_27b"
            and row["method"] == "dynamic_frozen_skills"
        )
        env = qwen_row["execution"]["environment"]
        self.assertEqual("weights-sha256-abc", env["C2HLS_MODEL_REVISION"])
        self.assertEqual("skill-sha256-def", env["C2HLS_SKILL_SNAPSHOT_SHA256"])

    def test_validator_rejects_budget_drift(self) -> None:
        matrix = copy.deepcopy(self.matrix)
        matrix["budget"]["max_synthesis_evaluations"] = 6
        issues = matrix_tool.validate_matrix(matrix)
        self.assertIn("unmatched_budget", {issue.code for issue in issues})

    def test_validator_rejects_wrong_runner_for_baseline_method(self) -> None:
        matrix = copy.deepcopy(self.matrix)
        matrix["methods"]["pragma_only"]["runner"] = "run_agentic_sweep.py"
        issues = matrix_tool.validate_matrix(matrix)
        self.assertIn("unknown_runner_mapping", {issue.code for issue in issues})

    def test_all_rows_are_executable_when_external_identities_are_supplied(self) -> None:
        identities = {
            "OPENAI_BASE_URL": "http://127.0.0.1:8000/v1",
            "C2HLS_QWEN27B_REVISION": "weights-sha256-qwen",
            "ANTHROPIC_API_KEY": "not-serialized",
            "C2HLS_SONNET46_REVISION": "provider-version-sonnet",
            "C2HLS_SKILL_LIBRARY_PATH": "/frozen/skills.json",
            "C2HLS_SKILL_SNAPSHOT_SHA256": "skills-sha256-frozen",
            "C2HLS_VITIS_SETTINGS": "/opt/Xilinx/Vitis/2023.2/settings64.sh",
        }
        with mock.patch.dict(os.environ, identities, clear=True):
            rows = matrix_tool.expand_matrix(self.matrix, resolve_env=True)
        self.assertEqual(132, len(rows))
        self.assertTrue(
            all(row["execution"]["mapping_status"] == "supported" for row in rows)
        )
        self.assertTrue(
            all(not row["execution"]["unresolved_required_env"] for row in rows)
        )
        self.assertNotIn("not-serialized", repr(rows))


if __name__ == "__main__":
    unittest.main()
