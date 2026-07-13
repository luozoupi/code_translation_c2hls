from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
VALIDATOR_PATH = REPO / "scripts" / "validate_hpca2027_matrix.py"
SPEC = importlib.util.spec_from_file_location(
    "validate_hpca2027_matrix_finite_test", VALIDATOR_PATH
)
assert SPEC and SPEC.loader
validator = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = validator
SPEC.loader.exec_module(validator)

TOLERANCE_TESTBENCH_CALLS = {
    "StreamCluster": 2,
    "hotspot": 1,
    "knn": 1,
    "lavaMD": 1,
    "lud": 1,
    "srad": 1,
}


class RodiniaFiniteToleranceGuardTests(unittest.TestCase):
    def test_every_float_tolerance_comparison_uses_finite_safe_helper(self) -> None:
        for benchmark, expected_calls in TOLERANCE_TESTBENCH_CALLS.items():
            with self.subTest(benchmark=benchmark):
                path = REPO / "benchmarks" / benchmark / "testbench.cpp"
                text = path.read_text()
                self.assertIn("#include <cmath>", text)
                self.assertEqual("", validator._finite_tolerance_policy_error(text))
                # Include one occurrence for the helper definition itself.
                self.assertEqual(
                    expected_calls + 1,
                    text.count("tolerance_mismatch("),
                    f"unexpected tolerance-comparison count in {path}",
                )
                self.assertEqual(1, text.count("!std::isfinite(reference)"))
                self.assertEqual(1, text.count("!std::isfinite(actual)"))

    def test_policy_rejects_missing_operand_guard_and_raw_bypass(self) -> None:
        original = (REPO / "benchmarks" / "hotspot" / "testbench.cpp").read_text()
        missing_guard = original.replace(
            "!std::isfinite(actual)", "std::isfinite(actual)", 1
        )
        self.assertIn(
            "reject non-finite",
            validator._finite_tolerance_policy_error(missing_guard),
        )

        raw_bypass = original.replace(
            "tolerance_mismatch(result_ref[i], result_dut[i], tol)",
            "fabsf(result_ref[i] - result_dut[i]) > "
            "tol * (fabsf(result_ref[i]) + 1.0f)",
            1,
        )
        self.assertIn(
            "bypasses finite-value checks",
            validator._finite_tolerance_policy_error(raw_bypass),
        )

    def test_repo_validator_emits_typed_error_for_unsafe_tolerance(self) -> None:
        original = (REPO / "benchmarks" / "knn" / "testbench.cpp").read_text()
        unsafe = original.replace(
            "!std::isfinite(reference)", "std::isfinite(reference)", 1
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            repo = Path(temp_dir)
            bench_dir = repo / "benchmarks" / "knn"
            bench_dir.mkdir(parents=True)
            (bench_dir / "plain.cpp").write_text("void workload() {}\n")
            (bench_dir / "testbench.cpp").write_text(unsafe)
            (bench_dir / "metadata.json").write_text(
                json.dumps(
                    {
                        "source_repo": "Rodinia-HLS",
                        "supports_csim": True,
                        "supports_cosim": True,
                        "testbench_file": "testbench.cpp",
                    }
                )
            )
            issues = []
            validator._validate_repo_benchmark(
                issues,
                repo,
                {
                    "id": "knn",
                    "path": "benchmarks/knn",
                    "testbench": "benchmarks/knn/testbench.cpp",
                    "golden_check_required": True,
                },
                0,
                force_selected_cosim=True,
            )
        self.assertIn(
            "nonfinite_tolerance_comparison", {issue.code for issue in issues}
        )


if __name__ == "__main__":
    unittest.main()
