"""Tests for benchmarks/ cosim baseline loader and cosim speedup metrics."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "explorer"))

from benchmark_cosim_baseline import (  # noqa: E402
    bench_short_from_group_path,
    is_benchmarks_dir_cosim_record,
    load_benchmark_cosim_baseline,
)
from metrics import (  # noqa: E402
    bench_cosim_metrics_from_multistep_doc,
    bench_cosim_speedup,
    geomean_cosim_speedup_from_benches,
)


class BenchmarkCosimBaselineTests(unittest.TestCase):
    def test_bench_short_from_group_path(self) -> None:
        self.assertEqual(bench_short_from_group_path(["jacobi_1d"]), "jacobi-1d")
        self.assertEqual(bench_short_from_group_path(["fdtd_2d"]), "fdtd-2d")

    def test_rejects_fixed_cosim_corpus(self) -> None:
        record = {
            "report_type": "rtl_sim",
            "implementation": {"origin_meta": {"cosim_export_suffix": "fixed_cosim"}},
            "rtl_sim": {"status": "pass", "kernel_runtime_cycles": 1},
        }
        self.assertFalse(is_benchmarks_dir_cosim_record(record))

    def test_accepts_benchmarks_dir_record(self) -> None:
        record = {
            "report_type": "rtl_sim",
            "problem": {"group_path": ["2mm"]},
            "implementation": {
                "origin_meta": {
                    "cosim_export_suffix": "naive_cosim",
                    "benchmark_dir": "/repo/benchmarks/hlsfactory_2mm",
                }
            },
            "rtl_sim": {"status": "pass", "kernel_runtime_cycles": 1_200_412},
        }
        self.assertTrue(is_benchmarks_dir_cosim_record(record))

    def test_load_naive_baseline_jsonl(self) -> None:
        baseline = load_benchmark_cosim_baseline(force_reload=True)
        self.assertIn("2mm", baseline)
        self.assertGreater(baseline["2mm"], 0)
        self.assertNotIn("", baseline)

    def test_cosim_speedup(self) -> None:
        self.assertEqual(bench_cosim_speedup(1000, 500), 2.0)
        self.assertIsNone(bench_cosim_speedup(None, 500))

    def test_bench_cosim_metrics_from_doc(self) -> None:
        baseline = {"jacobi-1d": 10_000}
        doc = {
            "success": True,
            "cosim": {"kernel_runtime_cycles": 5_000, "passed": True},
        }
        metrics = bench_cosim_metrics_from_multistep_doc(
            doc,
            baseline_map=baseline,
            bench_short_name="jacobi-1d",
        )
        assert metrics is not None
        self.assertEqual(metrics["baseline"], 10_000)
        self.assertEqual(metrics["generated"], 5_000)
        self.assertEqual(metrics["speedup"], 2.0)

    def test_geomean_cosim_speedup_from_benches(self) -> None:
        benches = {
            "a": {"status": "ok", "cosim": {"speedup": 2.0}},
            "b": {"status": "ok", "cosim": {"speedup": 8.0}},
            "c": {"status": "failed", "cosim": {"speedup": 100.0}},
        }
        gm = geomean_cosim_speedup_from_benches(benches)
        self.assertEqual(gm["n"], 2)
        self.assertAlmostEqual(gm["geomean"], 4.0)


if __name__ == "__main__":
    unittest.main()
