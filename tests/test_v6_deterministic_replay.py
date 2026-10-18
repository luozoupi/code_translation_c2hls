"""Independent archived-input regressions captured from the frozen v6 runtime."""
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from scripts import compare_v6_replay as replay


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/v6_replay"


class DeterministicV6Replay(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = replay.load_fixture(FIXTURE)
        cls.expected = json.loads((FIXTURE / "expected-v6.json").read_text())
        cls.current = replay.run_worker(ROOT, FIXTURE)

    def test_archived_agent_paths_match_independent_frozen_v6_capture(self):
        self.assertEqual(self.current, self.expected)

    def test_all_extracted_contract_helpers_preserve_original_function_ast(self):
        self.assertEqual(len(self.current["source_contract_helpers"]), 18)
        self.assertEqual(self.current["source_contract_helpers"], self.expected["source_contract_helpers"])

    def test_independent_csim_failures_skip_synthesis_and_remain_failed(self):
        failed = [item for item in self.current["cases"] if item["csim_status"] == "failed"]
        self.assertEqual(len(failed), 2)
        for item in failed:
            with self.subTest(candidate=item["id"]):
                self.assertFalse(item["phase_b_accepted"])
                self.assertFalse(item["feasibility"]["feasible"])
                self.assertIn("golden_correctness_not_passed", item["feasibility"]["reasons"])
                self.assertEqual(item["cosim_status"], "not_run")
                self.assertFalse(any(call["tool"] == "synthesis" for call in item["tool_calls"]))
                self.assertEqual(item["requests"], 2)

    def test_timeout_is_inconclusive_and_failed_rtl_is_repaired(self):
        timeout = next(item for item in self.current["cases"] if item["cosim_status"] == "timeout")
        self.assertFalse(timeout["cosim_measurement_eligible"])
        self.assertEqual(timeout["requests"], 1)
        failed = next(item for item in self.current["cases"] if item["cosim_status"] == "failed")
        self.assertFalse(failed["phase_b_accepted"])
        self.assertFalse(failed["cosim_measurement_eligible"])
        self.assertEqual(failed["requests"], 2)

    def test_seven_arm_selection_restores_saved_feasible_winner(self):
        selection = self.current["selection"]
        self.assertEqual(selection["registered_candidates"], 7)
        self.assertGreater(selection["eligible_candidates"], 0)
        self.assertIsNotNone(selection["promoted_step"])
        self.assertEqual(selection, self.expected["selection"])
        self.assertIn(selection["winner_source_sha256"], {
            replay.digest(item["kernel"]) for item in self.fixture["selection_candidates"]
        })

    def test_replay_rejects_changed_archived_inputs_before_runtime_use(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory)
            for name in ("manifest.json", "snapshot.json", "expected-v6.json"):
                shutil.copyfile(FIXTURE / name, destination / name)
            with (destination / "snapshot.json").open("a") as stream:
                stream.write("\n")
            with self.assertRaisesRegex(ValueError, "Archived replay fixture changed"):
                replay.load_fixture(destination)


if __name__ == "__main__":
    unittest.main()
