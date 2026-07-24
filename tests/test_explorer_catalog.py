"""Tests for experiment explorer catalog and metrics."""

from __future__ import annotations

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "explorer"))

from catalog import (  # noqa: E402
    bench_corpus,
    bench_short,
    build_experiment_record,
    compare_experiments,
    merge_registry,
    parse_cosim,
    parse_skill_variant,
    parse_skills_mode,
    parse_stamp,
    parse_workflow,
    scan_all,
)
from metrics import (  # noqa: E402
    bench_cosim_latency_from_multistep_doc,
    bench_csynth_latency_from_multistep_doc,
    bench_speedup_from_multistep_doc,
    geomean,
    geomean_from_bench_speedups,
    mean_latency_from_benches,
)


class TestParsing(unittest.TestCase):
    def test_bench_short(self) -> None:
        self.assertEqual(bench_short("hlsfactory_2mm"), "2mm")

    def test_bench_corpus(self) -> None:
        self.assertEqual(bench_corpus("hlsfactory_gemm"), "hlsfactory")
        self.assertEqual(bench_corpus("forgebench_foo"), "tier_a")

    def test_parse_stamp(self) -> None:
        self.assertEqual(parse_stamp("flash_smoke_20260705_204107"), "20260705_204107")
        self.assertIsNone(parse_stamp("batch_parallel_full_hlsfactory_20260706"))

    def test_parse_workflow(self) -> None:
        self.assertEqual(parse_workflow("flash_all_skills_avoids_global_20260705"), "flash")
        self.assertEqual(parse_workflow("multistep_fixed_cosim_nav_n_20260630"), "multistep")
        self.assertEqual(parse_workflow("hls_baseline_smoke_20260705"), "baseline")
        self.assertEqual(parse_workflow("foo", mode="multistep"), "multistep")

    def test_parse_cosim(self) -> None:
        self.assertEqual(parse_cosim("flash_fixed_cosim_aav_n_20260628"), "on")
        self.assertEqual(parse_cosim("flash_smoke_20260705"), "off")
        self.assertEqual(
            parse_cosim("batch_parallel_x", campaign={"config": {"pilot": {"run_cosim": True}}}),
            "on",
        )

    def test_parse_skills_mode(self) -> None:
        self.assertEqual(
            parse_skills_mode("flash_all_skills_avoids_global_x", setup="flash__all_skills_avoids_global"),
            "all_skills_avoids_global",
        )
        self.assertEqual(parse_skills_mode("flash_noskills_x"), "noskills")
        self.assertEqual(
            parse_skills_mode(
                "flash_zero_shot_cosim_phaseb_20260706",
                setup="flash__zero_shot_cosim__phaseb",
            ),
            "zero_shot",
        )
        self.assertEqual(
            parse_skills_mode(
                "flash_zero_shot_cosim_direct_20260706",
                manifest={"flash_opt_prompt_mode": "zero_shot"},
            ),
            "zero_shot",
        )

    def test_parse_skill_variant(self) -> None:
        self.assertEqual(parse_skill_variant("flash_fixed_cosim_aav_n_20260628"), "aav_n")


class TestMetrics(unittest.TestCase):
    def test_geomean(self) -> None:
        self.assertAlmostEqual(geomean([2.0, 8.0]), 4.0)
        self.assertIsNone(geomean([]))

    def test_bench_speedup_from_doc(self) -> None:
        doc = {
            "success": True,
            "ground_truth_report": {"latency_cycles": 100, "latency_cycles_worst": 200},
            "final_report": {"latency_cycles": 50, "latency_cycles_worst": 100},
        }
        sp = bench_speedup_from_multistep_doc(doc)
        assert sp is not None
        self.assertEqual(sp["avg"], 2.0)
        self.assertEqual(sp["worst"], 2.0)

    def test_bench_speedup_prefers_ground_truth_over_phase_b(self) -> None:
        doc = {
            "success": True,
            "baseline_report": {"latency_cycles": 40},
            "ground_truth_report": {"latency_cycles": 100},
            "final_report": {"latency_cycles": 50},
        }
        sp = bench_speedup_from_multistep_doc(doc)
        assert sp is not None
        self.assertEqual(sp["avg"], 2.0)

    def test_bench_cosim_latency_from_doc(self) -> None:
        doc = {
            "success": True,
            "final_report": {"latency_cycles": 100},
            "steps": [
                {
                    "success": True,
                    "step_name": "flash",
                    "report": {"latency_cycles": 100},
                    "cosim": {
                        "ran": True,
                        "passed": True,
                        "kernel_runtime_cycles": 1_234_567,
                    },
                }
            ],
        }
        self.assertEqual(bench_cosim_latency_from_multistep_doc(doc), 1_234_567)
        self.assertIsNone(
            bench_cosim_latency_from_multistep_doc({"success": True, "steps": [{"cosim": None}]})
        )

    def test_bench_cosim_latency_ignores_failed_flash_when_flash_wins(self) -> None:
        doc = {
            "success": True,
            "baseline_report": {"latency_cycles": 47_837_251},
            "final_report": {"latency_cycles": 90_175},
            "generated_step_history": [
                {
                    "step_name": "baseline",
                    "report": {"latency_cycles": 47_837_251},
                    "cosim": {
                        "ran": True,
                        "passed": True,
                        "kernel_runtime_cycles": 47_837_251,
                    },
                }
            ],
            "steps": [
                {
                    "success": True,
                    "step_name": "flash",
                    "report": {"latency_cycles": 90_175},
                    "cosim": {
                        "ran": True,
                        "passed": True,
                        "kernel_runtime_cycles": 90_175,
                    },
                }
            ],
        }
        self.assertEqual(bench_cosim_latency_from_multistep_doc(doc), 90_175)

    def test_bench_csynth_latency_from_doc(self) -> None:
        doc = {
            "success": True,
            "final_report": {
                "latency_cycles": 100,
                "latency_cycles_best": 80,
                "latency_cycles_worst": 120,
            },
        }
        lat = bench_csynth_latency_from_multistep_doc(doc)
        assert lat is not None
        self.assertEqual(lat["best"], 80)
        self.assertEqual(lat["avg"], 100)
        self.assertEqual(lat["worst"], 120)

    def test_mean_latency_from_benches(self) -> None:
        benches = {
            "a": {"status": "ok", "latency": {"best": 100, "avg": 200, "worst": 300}},
            "b": {"status": "ok", "latency": {"best": 200, "avg": 400, "worst": 600}},
            "c": {"status": "failed", "latency": {"avg": 999}},
        }
        mean = mean_latency_from_benches(benches)
        self.assertEqual(mean["n"], 2)
        self.assertEqual(mean["best"], 150)
        self.assertEqual(mean["avg"], 300)
        self.assertEqual(mean["worst"], 450)

    def test_geomean_from_benches(self) -> None:
        benches = {
            "a": {"status": "ok", "speedup": {"best": 2.0, "avg": 2.0, "worst": 2.0}},
            "b": {"status": "ok", "speedup": {"best": 8.0, "avg": 8.0, "worst": 8.0}},
            "c": {"status": "failed", "speedup": {"avg": 100.0}},
        }
        gm = geomean_from_bench_speedups(benches)
        self.assertEqual(gm["n"], 2)
        self.assertAlmostEqual(gm["avg"], 4.0)

    def test_geomean_excludes_infra_run_issues(self) -> None:
        benches = {
            "ok": {"status": "ok", "speedup": {"avg": 4.0}},
            "llm": {
                "status": "ok",
                "run_issues": ["llm_connection_error"],
                "speedup": {"avg": 1.0},
            },
            "revert": {
                "status": "ok",
                "run_issues": ["flash_reverted"],
                "speedup": {"avg": 2.0},
            },
        }
        gm = geomean_from_bench_speedups(benches)
        self.assertEqual(gm["n"], 2)
        self.assertAlmostEqual(gm["avg"], math.sqrt(8.0))


class TestCatalogIntegration(unittest.TestCase):
    def test_scan_fir_campaign_present(self) -> None:
        exps = scan_all(REPO)
        ids = {e["id"] for e in exps}
        self.assertIn("fir/batch_parallel_full_hlsfactory_20260706", ids)

    def test_build_experiment_record_has_geomean(self) -> None:
        root = REPO / "artifacts/fir/batch_parallel_full_hlsfactory_20260706"
        if not root.is_dir():
            self.skipTest("campaign not on disk")
        record = build_experiment_record(root, site="fir", dirname=root.name)
        assert record is not None
        self.assertGreater(record["counts"]["ok"], 0)
        self.assertGreater(record["geomean"]["n"], 0)
        if record["latency_mean"]["n"] > 0:
            self.assertIsNotNone(record["latency_mean"]["avg"])

    def test_registry_merge_planned(self) -> None:
        scanned = [{"id": "fir/foo", "label": "foo"}]
        registry = [
            {
                "id": "pc2/planned_test",
                "label": "Planned",
                "site": "pc2",
                "path": None,
                "workflow": "flash",
            }
        ]
        merged = merge_registry(scanned, registry, repo_root=REPO)
        planned = [e for e in merged if e.get("planned")]
        self.assertEqual(len(planned), 1)
        self.assertEqual(planned[0]["id"], "pc2/planned_test")

    def test_compare_experiments(self) -> None:
        exps = [
            {
                "id": "a",
                "label": "A",
                "planned": False,
                "benches": {
                    "2mm": {"status": "ok", "speedup": {"avg": 2.0, "best": 2.0, "worst": 2.0}},
                    "3mm": {"status": "ok", "speedup": {"avg": 8.0, "best": 8.0, "worst": 8.0}},
                },
            },
            {
                "id": "b",
                "label": "B",
                "planned": False,
                "benches": {
                    "2mm": {"status": "ok", "speedup": {"avg": 4.0, "best": 4.0, "worst": 4.0}},
                },
            },
        ]
        result = compare_experiments(exps, ids=["a", "b"], bench_filter=["2mm"])
        self.assertEqual(len(result["summaries"]), 2)
        self.assertEqual(result["summaries"][0]["geomean"]["n"], 1)
        self.assertEqual(result["summaries"][1]["geomean"]["n"], 1)

    def test_synthetic_campaign(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "flash_smoke_test"
            root.mkdir()
            bench = "hlsfactory_2mm"
            cell = root / bench / "devstral2__flash__all_skills_avoids_global"
            cell.mkdir(parents=True)
            manifest = {
                "setup": "flash__all_skills_avoids_global",
                "skill_prompt_mode": "all_skills_avoids_global",
                "benches": [bench],
                "model": "test/model",
            }
            (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            results = {
                "success": True,
                "phase": "flash",
                "baseline_report": {"latency_cycles": 1000},
                "final_report": {"latency_cycles": 100},
            }
            (cell / f"{bench}_multistep_results.json").write_text(
                json.dumps(results), encoding="utf-8"
            )
            record = build_experiment_record(root, site="fir", dirname=root.name)
            assert record is not None
            self.assertEqual(record["workflow"], "flash")
            self.assertEqual(record["cosim"], "off")
            self.assertAlmostEqual(record["geomean"]["avg"], 10.0)


if __name__ == "__main__":
    unittest.main()
