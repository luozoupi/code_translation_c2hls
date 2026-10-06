"""Functional csim must prefer gold-check (cosim) TBs over dump TBs."""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

GRAMSCHMIDT = REPO / "benchmarks_cosim" / "hlsfactory_gramschmidt"


class CsimGoldTbGateTests(unittest.TestCase):
    def setUp(self) -> None:
        self._prev = os.environ.get("C2HLS_CSIM_USE_COSIM_TB")

    def tearDown(self) -> None:
        if self._prev is None:
            os.environ.pop("C2HLS_CSIM_USE_COSIM_TB", None)
        else:
            os.environ["C2HLS_CSIM_USE_COSIM_TB"] = self._prev

    def test_auto_prefers_cosim_tb_when_present(self) -> None:
        os.environ.pop("C2HLS_CSIM_USE_COSIM_TB", None)
        from c2hls import _load_benchmark_inputs

        inputs = _load_benchmark_inputs(str(GRAMSCHMIDT))
        self.assertEqual(inputs["testbench_mode"], "cosim_gold")
        self.assertIn("kernel_gramschmidt_gold", inputs["testbench_code"])
        self.assertIn("diff A[", inputs["testbench_code"])
        self.assertIn("FAIL:", inputs["testbench_code"])
        # Dump TB kept for LLM signature context only.
        self.assertIn("print_array", inputs["dump_testbench_code"])
        self.assertNotIn("kernel_gramschmidt_gold", inputs["dump_testbench_code"])
        self.assertNotIn("gold", inputs["benchmark_context"].lower())

        paths = {item["path"] for item in inputs["extra_files"]}
        self.assertIn("gold_kernel_for_cosim.cpp", paths)
        self.assertIn("gold_hls_source.cpp", paths)
        gold_tb = next(
            item for item in inputs["extra_files"] if item["path"] == "gold_kernel_for_cosim.cpp"
        )
        self.assertTrue(gold_tb.get("tb"))

    def test_opt_out_uses_dump_tb(self) -> None:
        os.environ["C2HLS_CSIM_USE_COSIM_TB"] = "0"
        from c2hls import _load_benchmark_inputs

        inputs = _load_benchmark_inputs(str(GRAMSCHMIDT))
        self.assertEqual(inputs["testbench_mode"], "dump")
        self.assertIn("print_array", inputs["testbench_code"])
        self.assertNotIn("kernel_gramschmidt_gold", inputs["testbench_code"])
        paths = {item["path"] for item in inputs["extra_files"]}
        self.assertNotIn("gold_kernel_for_cosim.cpp", paths)

    def test_missing_store_a_would_fail_gold_tb_not_dump(self) -> None:
        """Regression: dump TB hides missing A writeback; gold TB diffs A."""
        os.environ.pop("C2HLS_CSIM_USE_COSIM_TB", None)
        from c2hls import _load_benchmark_inputs

        seed = (
            REPO
            / "artifacts/pc2/batch_parallel_hlsfactory_ds_v4f_skills_lat_opt_20260726_085609_lat_opt"
            / "variants/aav_n/hlsfactory_gramschmidt"
            / "deepseek-v4-flash__flash__fixed_cosim__aav_n"
            / "hlsfactory_gramschmidt_flash_seed.cpp"
        )
        if not seed.is_file():
            self.skipTest(f"missing artifact seed: {seed}")

        code = seed.read_text(encoding="utf-8")
        self.assertNotIn("store_A", code)
        # Reads A into local_A and updates locals / Q / R, but never writes A[] back.
        self.assertRegex(code, r"local_A\[i\]\[j\]\s*=\s*A\[i\]\[j\]")
        writebacks = [
            ln
            for ln in code.splitlines()
            if ln.lstrip().startswith("A[") and "=" in ln and not ln.lstrip().startswith("//")
        ]
        self.assertEqual(writebacks, [], msg="expected no A[] writeback statements")

        inputs = _load_benchmark_inputs(str(GRAMSCHMIDT))
        self.assertEqual(inputs["testbench_mode"], "cosim_gold")
        tb = inputs["testbench_code"]
        # Gold-check compares A (among other arrays) — this is the gate dump TB lacks.
        self.assertIn("diff A[", tb)
        self.assertIn("return 1", tb)
        dump = inputs["dump_testbench_code"]
        self.assertIn("return 0", dump)
        self.assertNotIn("kernel_gramschmidt_gold", dump)


if __name__ == "__main__":
    unittest.main()
