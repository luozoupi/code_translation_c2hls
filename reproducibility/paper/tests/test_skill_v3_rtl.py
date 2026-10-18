"""Scientific-integrity regressions for the portable Figure 8 RTL evidence."""
import copy
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import generate_skill_v3_rtl as rtl


class FigureEightEvidence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.evidence = ROOT / "generated/sigmetrics/skill_v3_rtl_evidence"
        cls.original = rtl.load_bundle(cls.evidence)

    def setUp(self):
        self.bundle = copy.deepcopy(self.original)

    def test_reconstructs_all_plotted_denominators_and_numerators(self):
        rows = rtl.build_rows(self.bundle)
        stats = rtl.summarize(rows)
        self.assertEqual((stats["selected_pass_1h"], stats["off_pass_1h"], stats["paired_1h"]), (84, 57, 52))
        self.assertEqual((stats["strict_wins_1h"], stats["ties_1h"]), (35, 17))
        self.assertEqual(len(rows), 108)
        for row in rows:
            if row["rtl_ratio_1h"] is not None:
                self.assertEqual(row["rtl_ratio_1h"], row["off_cycles_1h"] / row["selected_cycles_1h"])
            else:
                self.assertTrue(row["selected_cycles_1h"] is None or row["off_cycles_1h"] is None)

    def test_longer_timeout_recovery_never_enters_primary_ratios(self):
        rows = rtl.build_rows(self.bundle)
        recovered = [r for r in rows if r["rtl_ratio_1h"] is None and r["rtl_ratio_with_replay"] is not None]
        self.assertEqual(len(recovered), 13)
        for row in recovered:
            self.assertIn("timeout", (row["selected_status_1h"], row["off_status_1h"]))

    def test_unknown_identity_is_not_an_rtl_failure_or_zero_cycle_value(self):
        unknown = [r for r in rtl.build_rows(self.bundle) if r["off_status_1h"] == "blocked_identity"]
        self.assertEqual(len(unknown), 5)
        self.assertTrue(all(r["off_cycles_1h"] is None and r["rtl_ratio_with_replay"] is None for r in unknown))

    def test_synthesis_bars_preserve_frozen_records_and_disclose_unresolved_identity(self):
        rows = rtl.build_rows(self.bundle)
        source = {rtl.key(r): r for r in self.bundle["source_rows"]}
        stats = rtl.summarize(rows)
        self.assertEqual((stats["recorded_synthesis_pairs"], stats["identity_linked_synthesis_pairs"]), (108, 103))
        for row in rows:
            historical = source[rtl.key(row)]
            self.assertEqual(row["selected_policy"], historical["selected_skill_mode"])
            self.assertEqual(row["synthesis_ratio"], float(historical["no_skills_worst_case_cycles"]) / float(historical["selected_worst_case_cycles"]))
            self.assertAlmostEqual(row["synthesis_ratio"], float(historical["no_skills_over_selected"]))
            if not row["synthesis_identity_linked"]:
                self.assertEqual(row["off_status_1h"], "blocked_identity")
                self.assertIsNone(row["rtl_ratio_1h"])

    def test_missing_cosim_is_blank_and_ties_use_hollow_markers(self):
        figure, ax = rtl.plt.subplots()
        try:
            # A CSynth estimate does not create a COSim bar when RTL is missing.
            rtl.ratio_bar(ax, -.16, 8, rtl.METRICS[0][2])
            rtl.ratio_bar(ax, .16, None, rtl.METRICS[1][2])
            self.assertEqual(len(ax.patches), 1)
            self.assertEqual(len(ax.collections), 0)
            self.assertEqual(ax.patches[0].get_height(), 3)
            # A measured 1x result is distinguishable from a missing result.
            rtl.ratio_bar(ax, .16, 1, rtl.METRICS[1][2])
            self.assertEqual(len(ax.patches), 1)
            self.assertEqual(len(ax.collections), 1)
            self.assertEqual(tuple(ax.collections[0].get_offsets()[0]), (.16, 0))
            self.assertEqual(tuple(ax.collections[0].get_facecolors()[0]), (1, 1, 1, 1))
            # Both metric slots fit within the old single-bar footprint.
            outer_width = rtl.METRICS[1][3] - rtl.METRICS[0][3] + rtl.BAR_WIDTH
            self.assertLessEqual(outer_width, .64)
        finally:
            rtl.plt.close(figure)

    def test_preserves_both_statuses_when_pair_has_different_failures(self):
        rows = rtl.build_rows(self.bundle)
        self.assertTrue(any(r["selected_status_1h"] == "fail" and r["off_status_1h"] == "timeout" for r in rows))
        self.assertTrue(any(r["selected_status_1h"] == "timeout" and r["off_status_1h"] == "pass" for r in rows))

    def test_refuses_missing_benchmark(self):
        self.bundle["source_rows"].pop()
        with self.assertRaisesRegex(ValueError, "108-design selection"):
            rtl.build_rows(self.bundle)

    def test_refuses_duplicate_identity_even_with_unchanged_row_count(self):
        self.bundle["figure8_identity_join"]["rows"][0] = self.bundle["figure8_identity_join"]["rows"][1]
        with self.assertRaisesRegex(ValueError, "duplicate arm identity"):
            rtl.build_rows(self.bundle)

    def test_refuses_reselection_after_validation(self):
        self.bundle["source_rows"][0]["selected_source_result_path"] = "different_candidate.json"
        with self.assertRaisesRegex(ValueError, "reselected"):
            rtl.build_rows(self.bundle)

    def test_refuses_missing_source_hash_audit(self):
        self.bundle["source_audit"].clear()
        with self.assertRaisesRegex(ValueError, "source byte audit"):
            rtl.build_rows(self.bundle)

    def test_refuses_incomparable_input_hashes(self):
        first = self.bundle["figure8_identity_join"]["rows"][0]
        record = next(r for r in self.bundle["inventory"]["rows"] if r["resolved_result"] == first["source_result"])
        record["input_hashes"]["testbench.cpp"] = "different dataset"
        with self.assertRaisesRegex(ValueError, "input or toolchain differs"):
            rtl.build_rows(self.bundle)

    def test_refuses_incomplete_independent_output_check(self):
        v = next(v for v in self.bundle["cosim_evidence_snapshot"]["validations"] if v["cohort"] == "historical_1h" and v["status"] == "pass" and v["validation_id"] in self.bundle["measurements"])
        entry = self.bundle["measurements"][v["validation_id"]]
        raw = json.loads(entry["raw_text"])
        raw["cosim"]["correctness"]["passed"] = False
        entry["raw_text"] = json.dumps(raw)
        entry["sha256"] = v["measurement_sha256"] = rtl.sha(entry["raw_text"].encode())
        with self.assertRaisesRegex(ValueError, "independent RTL output check"):
            rtl.validate_measurement(v, self.bundle["measurements"])

    def test_refuses_negative_zero_nan_or_boolean_cycles(self):
        for value in (-1, 0, float("nan"), float("inf"), True, "100", 1.5):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "Invalid RTL cycles"):
                rtl.positive_cycles(value, "fixture")

    def test_refuses_csv_tampering_during_portable_rebuild(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "evidence"
            shutil.copytree(self.evidence, dest)
            with (dest / "figure8_source_snapshot.csv").open("a") as stream:
                stream.write("\n")
            with self.assertRaisesRegex(ValueError, "Evidence hash mismatch"):
                rtl.load_bundle(dest)

    def test_refuses_missing_measurements(self):
        self.bundle["measurements"].clear()
        with self.assertRaisesRegex(ValueError, "Missing original measurement"):
            rtl.build_rows(self.bundle)

    def test_refuses_silent_timeout_policy_change(self):
        self.bundle["cosim_evidence_snapshot"]["cohorts"][0]["timeout_seconds"] = 10800
        with self.assertRaisesRegex(ValueError, "Validation limits differ"):
            rtl.build_rows(self.bundle)

    def test_validated_old_selection_file_has_not_changed(self):
        old_source = ROOT / "generated/skill_v3_open_models_snapshot.csv"
        self.assertEqual(rtl.sha(old_source.read_bytes()), self.bundle["source_sha256"])


if __name__ == "__main__":
    unittest.main()
