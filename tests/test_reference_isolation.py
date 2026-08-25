from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import reference_isolation as isolation  # noqa: E402


class ReferenceIsolationTests(unittest.TestCase):
    def _benchmark(self, root: Path) -> Path:
        bench = root / "tiny"
        bench.mkdir()
        (bench / "metadata.json").write_text(json.dumps({
            "benchmark": "tiny",
            "plain_c_file": "plain.cpp",
            "gold_hls_source_file": "gold_hls_source.cpp",
            "gold_hls_source_path": "/private/expert/tiny/gold_hls_source.cpp",
            "variants": [{
                "name": "tiny_3_secretpipeline",
                "file": "hls_tiny_3_secretpipeline.cpp",
                "source_path": "/private/expert/tiny/secret.cpp",
            }],
        }))
        (bench / "plain.cpp").write_text(
            "int workload(int x) {\n  int output = x;\n  return output;\n}\n"
        )
        (bench / "gold_hls_source.cpp").write_text(
            "int workload(int x) {\n"
            "  int expertAccumulatorName = x * 17;\n"
            "  expertAccumulatorName += x * 31;\n"
            "  return expertAccumulatorName + 123456;\n"
            "}\n"
        )
        (bench / "hls_tiny_3_secretpipeline.cpp").write_text(
            (bench / "gold_hls_source.cpp").read_text()
        )
        return bench

    def test_safe_plain_input_and_generic_optimization_prompt_pass(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            bench = self._benchmark(Path(tmpdir))
            audit = isolation.audit_messages(
                [{
                    "role": "user",
                    "content": "Optimize the supplied plain C using pipelining and unrolling.",
                }],
                benchmark_dir=bench,
                reference_data={"report": {"latency_cycles": 987654}},
            )
        self.assertTrue(audit["passed"])
        self.assertEqual(audit["finding_count"], 0)

    def test_paths_identifiers_code_blocks_and_metrics_are_detected(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            bench = self._benchmark(Path(tmpdir))
            copied_block = (
                "int workload(int x) { int expertAccumulatorName = x * 17; "
                "expertAccumulatorName += x * 31; return expertAccumulatorName + 123456; }"
            )
            messages = [
                {"role": "system", "content": copied_block},
                {
                    "role": "user",
                    "content": (
                        "Read /private/expert/tiny/gold_hls_source.cpp and target "
                        "expert cycles: 987654 using tiny_3_secretpipeline."
                    ),
                },
            ]
            audit = isolation.audit_messages(
                messages,
                benchmark_dir=bench,
                reference_data={"report": {"latency_cycles": 987654}},
            )
        self.assertFalse(audit["passed"])
        rules = set(audit["finding_counts"])
        self.assertIn("expert_path", rules)
        self.assertIn("expert_identifier", rules)
        self.assertIn("expert_code_signature", rules)
        self.assertIn("absolute_reference_metric", rules)
        serialized = json.dumps(audit)
        self.assertNotIn("expertAccumulatorName", serialized)
        self.assertNotIn("/private/expert", serialized)
        self.assertNotIn("987654", serialized)

    def test_assistant_recovery_does_not_trigger_code_signature_rule(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            bench = self._benchmark(Path(tmpdir))
            audit = isolation.audit_messages(
                [{
                    "role": "assistant",
                    "content": (
                        "int workload(int x) { int expertAccumulatorName = x * 17; "
                        "expertAccumulatorName += x * 31; return expertAccumulatorName + 123456; }"
                    ),
                }],
                benchmark_dir=bench,
            )
        self.assertNotIn("expert_code_signature", audit["finding_counts"])
        self.assertNotIn("expert_identifier", audit["finding_counts"])

    def test_alternate_variant_metric_is_collected_recursively_without_label(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            bench = self._benchmark(Path(tmpdir))
            reference_data = {
                "report": {"latency_cycles_worst": 765431},
                "workflow": [
                    {
                        "variant_name": "baseline",
                        "report": {"latency_cycles_worst": 456781},
                    },
                    {
                        "variant_name": "unselected_frontier_variant",
                        "report": {
                            "latency_cycles_worst": 876543,
                            "fmax_mhz": 392.91,
                        },
                    },
                ],
            }
            audit = isolation.audit_messages(
                [{
                    "role": "tool",
                    "content": "The optimization target is exactly 876543 cycles.",
                }],
                benchmark_dir=bench,
                reference_data=reference_data,
            )
        self.assertFalse(audit["passed"])
        self.assertIn("unlabeled_reference_metric", audit["finding_counts"])

    def test_unlabeled_metric_filter_ignores_tiny_round_and_common_numbers(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            bench = self._benchmark(Path(tmpdir))
            reference_data = {
                "workflow": [{
                    "report": {
                        "dsp": 5,
                        "bram": 16,
                        "latency_cycles": 1024,
                        "ff": 2023,
                        "lut": 10000,
                        "fmax_mhz": 300.0,
                    },
                }],
            }
            audit = isolation.audit_messages(
                [{
                    "role": "user",
                    "content": (
                        "Try 5 candidates, unroll by 16, use a 1024-element "
                        "buffer, Vitis 2023, a 10000-cycle cap, and 300 MHz."
                    ),
                }],
                benchmark_dir=bench,
                reference_data=reference_data,
            )
        self.assertTrue(audit["passed"])
        self.assertNotIn("unlabeled_reference_metric", audit["finding_counts"])

    def test_explicit_expert_field_label_still_detects_small_metric(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            bench = self._benchmark(Path(tmpdir))
            audit = isolation.audit_messages(
                [{
                    "role": "tool",
                    "content": "The expert report has latency_cycles = 16.",
                }],
                benchmark_dir=bench,
                reference_data={"workflow": [{"report": {"latency_cycles": 16}}]},
            )
        self.assertFalse(audit["passed"])
        self.assertIn("absolute_reference_metric", audit["finding_counts"])
        self.assertNotIn("unlabeled_reference_metric", audit["finding_counts"])

    def test_generated_metric_equal_to_reference_is_audited_collision(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            bench = self._benchmark(Path(tmpdir))
            audit = isolation.audit_messages(
                [{
                    "role": "user",
                    "content": "Current synthesis report: latency_cycles: 987654.",
                }],
                benchmark_dir=bench,
                reference_data={"report": {"latency_cycles": 987654}},
                controller_data={
                    "baseline_report": {"latency_cycles": 987654},
                },
            )
        self.assertTrue(audit["passed"])
        self.assertEqual(audit["finding_count"], 0)
        self.assertEqual(audit["allowed_controller_metric_match_count"], 1)
        self.assertEqual(
            audit["allowed_controller_metric_match_counts"],
            {"generated_controller_metric_collision": 1},
        )
        self.assertNotIn("987654", json.dumps(audit))

    def test_labeled_reference_metric_remains_fatal_on_generated_collision(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            bench = self._benchmark(Path(tmpdir))
            audit = isolation.audit_messages(
                [{
                    "role": "user",
                    "content": "The hidden reference latency_cycles is 987654.",
                }],
                benchmark_dir=bench,
                reference_data={"report": {"latency_cycles": 987654}},
                controller_data={
                    "baseline_report": {"latency_cycles": 987654},
                },
            )
        self.assertFalse(audit["passed"])
        self.assertIn("absolute_reference_metric", audit["finding_counts"])

    def test_reference_subtree_cannot_authorize_metric_collision(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            bench = self._benchmark(Path(tmpdir))
            audit = isolation.audit_messages(
                [{
                    "role": "user",
                    "content": "Optimize toward exactly 987654 cycles.",
                }],
                benchmark_dir=bench,
                reference_data={"report": {"latency_cycles": 987654}},
                controller_data={
                    "reference_validation": {
                        "report": {"latency_cycles": 987654},
                    },
                },
            )
        self.assertFalse(audit["passed"])
        self.assertIn("unlabeled_reference_metric", audit["finding_counts"])

    def test_public_plain_c_literal_is_not_an_unlabeled_metric_signature(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            bench = self._benchmark(Path(tmpdir))
            with (bench / "plain.cpp").open("a") as handle:
                handle.write("int public_tripcount = 123457;\n")
            audit = isolation.audit_messages(
                [{
                    "role": "user",
                    "content": "Preserve the public tripcount 123457.",
                }],
                benchmark_dir=bench,
                reference_data={"report": {"latency_cycles": 123457}},
            )
        self.assertTrue(audit["passed"])

    def test_reformatted_partial_expert_code_is_detected_by_token_kgram(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            bench = self._benchmark(Path(tmpdir))
            audit = isolation.audit_messages(
                [{
                    "role": "system",
                    "content": (
                        "int\nexpertAccumulatorName /* copied fragment */\n= x*17;\n"
                        "expertAccumulatorName"
                    ),
                }],
                benchmark_dir=bench,
            )
        self.assertFalse(audit["passed"])
        self.assertIn("expert_code_signature", audit["finding_counts"])

    def test_short_relative_expert_filename_path_and_identifier_are_detected(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            bench = self._benchmark(Path(tmpdir))
            metadata = json.loads((bench / "metadata.json").read_text())
            metadata["variants"].append({
                "name": "v3",
                "file": "gt.c",
                "source_path": "ref/x.c",
            })
            (bench / "metadata.json").write_text(json.dumps(metadata))
            (bench / "gt.c").write_text("int hidden_short_variant(void) { return 7; }\n")
            audit = isolation.audit_messages(
                [{
                    "role": "user",
                    "content": "Open gt.c (ref/x.c), then reproduce variant v3.",
                }],
                benchmark_dir=bench,
            )
        self.assertFalse(audit["passed"])
        self.assertIn("expert_path", audit["finding_counts"])
        self.assertIn("expert_identifier", audit["finding_counts"])

    def test_short_filename_requires_path_token_boundaries(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            bench = self._benchmark(Path(tmpdir))
            metadata = json.loads((bench / "metadata.json").read_text())
            metadata["variants"].append({"name": "v3", "file": "gt.c"})
            (bench / "metadata.json").write_text(json.dumps(metadata))
            (bench / "gt.c").write_text("int hidden_short_variant(void) { return 7; }\n")
            audit = isolation.audit_messages(
                [{
                    "role": "user",
                    "content": "Compile target.c and keep the public API unchanged.",
                }],
                benchmark_dir=bench,
            )
        self.assertTrue(audit["passed"])

    def test_history_audit_hashes_exact_persisted_bytes_before_parsing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            bench = self._benchmark(root)
            history = root / "history.json"
            raw = (
                b'{\n  "messages": [{"role": "user", '
                b'"content": "Optimize the public kernel."}]\n}\n'
            )
            history.write_bytes(raw)
            audit = isolation.audit_history_file(history, benchmark_dir=bench)
        self.assertTrue(audit["passed"])
        self.assertEqual(hashlib.sha256(raw).hexdigest(), audit["transcript_sha256"])
        self.assertEqual(len(raw), audit["transcript_bytes"])

    def test_malformed_persisted_transcript_is_bound_and_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            bench = self._benchmark(root)
            history = root / "history.json"
            raw = b'{"messages": [}\n'
            history.write_bytes(raw)
            audit = isolation.audit_history_file(history, benchmark_dir=bench)
        self.assertFalse(audit["passed"])
        self.assertEqual(hashlib.sha256(raw).hexdigest(), audit["transcript_sha256"])
        self.assertIn("transcript unavailable", audit["error"])


if __name__ == "__main__":
    unittest.main()
