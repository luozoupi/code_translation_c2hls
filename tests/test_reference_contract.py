from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch


REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import c2hls  # noqa: E402


PRIMARY = (
    "StreamCluster",
    "hotspot",
    "kmeans",
    "knn",
    "lavaMD",
    "lud",
    "nw",
    "pathfinder",
    "srad",
)

EXPECTED_EXCLUDED = {
    "streamcluster_1_tiling",
    "streamcluster_4_coalescing",
    "hotspot_5_coalescing",
    "hotspot_6_multiddr",
    "kmeans_5_coalescing",
    "kmeans_6_multiddr",
    "knn_5_coalescing",
    "lavaMD_5_coalescing",
    "lud_2_coalescing",
    "lud_3_unrolling",
    "nw_5_coalescing",
    "pathfinder_5_coalescing",
    "srad_5_coalescing",
}


def _contract_reasons(candidate: dict, inputs: dict) -> list[str]:
    reasons = []
    if not candidate["public_contract_audit"]["passed"]:
        reasons.append("public_workload_contract_mismatch")
    expected = c2hls._expected_top_signature(
        inputs["header_code"],
        inputs["testbench_code"],
        inputs["meta"].get("hls_top", "workload"),
    )
    current = c2hls._extract_function_signature(
        candidate["code"],
        inputs["meta"].get("hls_top", "workload"),
        definitions_only=True,
    )
    if expected is None or current is None:
        reasons.append("unparseable_workload_signature")
    elif c2hls._top_signature_mismatch_reason(
        candidate["code"],
        inputs["header_code"],
        inputs["testbench_code"],
        inputs["meta"].get("hls_top", "workload"),
    ):
        reasons.append("workload_abi_mismatch")
    return reasons


class ReferenceContractTests(unittest.TestCase):
    def test_all_rodinia_variants_match_expected_contract_partition(self):
        accepted = set()
        excluded = set()
        for benchmark in PRIMARY:
            inputs = c2hls._load_benchmark_inputs(str(REPO / "benchmarks" / benchmark))
            for candidate in c2hls._ground_truth_candidates(inputs):
                name = candidate["variant_name"]
                if _contract_reasons(candidate, inputs):
                    excluded.add(name)
                else:
                    accepted.add(name)

        self.assertEqual(EXPECTED_EXCLUDED, excluded)
        self.assertEqual(38, len(accepted))
        self.assertEqual(51, len(accepted | excluded))

    def test_public_macro_change_and_transitive_change_are_rejected(self):
        direct = c2hls._public_header_contract_audit(
            "#define SIZE 128\n",
            "#define SIZE 1024\n",
            plain_code="void workload(int x[SIZE]) {}",
            testbench_code="int out[SIZE];",
        )
        self.assertFalse(direct["passed"])
        self.assertEqual("SIZE", direct["differences"][0]["identifier"])

        transitive = c2hls._public_header_contract_audit(
            "#define SIZE BASE\n#define BASE 128\n",
            "#define SIZE BASE\n#define BASE 1024\n",
            plain_code="void workload(int x[SIZE]) {}",
            testbench_code="",
        )
        self.assertFalse(transitive["passed"])
        self.assertEqual({"BASE"}, {
            item["identifier"] for item in transitive["differences"]
        })

    def test_comments_include_paths_and_private_macros_do_not_change_contract(self):
        audit = c2hls._public_header_contract_audit(
            '#include "support/common/mc.h"\n#define SIZE (16 * 8)\n',
            '#include "../../../common/mc.h"\n#define SIZE (16*8) // same\n#define TILE_SIZE 16\n',
            plain_code="void workload(int x[SIZE]) {}",
            testbench_code="int x[SIZE];",
        )
        self.assertTrue(audit["passed"])

    def test_contract_and_unparseable_abi_exclusions_use_no_tools(self):
        inputs = c2hls._load_benchmark_inputs(
            str(REPO / "benchmarks" / "StreamCluster")
        )
        tiling = next(
            candidate
            for candidate in c2hls._ground_truth_candidates(inputs)
            if candidate["variant_name"] == "streamcluster_1_tiling"
        )
        with patch.object(c2hls, "_run_synth_csim_cosim") as tools:
            result = c2hls._validate_ground_truth_candidate(
                tiling, inputs, True, True
            )
        tools.assert_not_called()
        self.assertFalse(result["benchmark_ready"])
        self.assertEqual("excluded", result["reference_contract_status"])
        self.assertEqual("not_run", result["synthesis"]["status"])

        malformed = dict(tiling)
        malformed["header_code"] = inputs["header_code"]
        malformed["public_contract_audit"] = {
            "passed": True,
            "differences": [],
        }
        malformed["code"] = "int not_the_workload = 0;"
        with patch.object(c2hls, "_run_synth_csim_cosim") as tools:
            result = c2hls._validate_ground_truth_candidate(
                malformed, inputs, True, True
            )
        tools.assert_not_called()
        self.assertFalse(result["benchmark_ready"])
        self.assertIn("missing or unparseable", result["invalid_reason"])


if __name__ == "__main__":
    unittest.main()
