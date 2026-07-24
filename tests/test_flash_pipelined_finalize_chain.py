"""Flash pipelined finalize should chain pragma_opt + latency_opt like c2hls.py."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

REPO = Path(__file__).resolve().parents[1]
SCRIPTS_PC2 = REPO / "scripts" / "pc2"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(SCRIPTS_PC2))

from flash_pipelined_bench import FlashPipelinedBenchSession  # noqa: E402


class FakeOrch:
    def __init__(self) -> None:
        self.synth_report = {"latency_cycles": 100, "lut": 10, "dsp": 1, "ff": 20, "bram": 0}
        self.hls_code = "void kernel() {}"
        self.generated_csim = {"passed": True}
        self.generated_cosim = None
        self.phaseb_mode = "functional"
        self.preflight_patches = []
        self._flow_phase_b_report = {"latency_cycles": 200}
        self._pipelined_ctx = {
            "flash_step_result": {
                "success": True,
                "step_name": "flash",
                "report": self.synth_report,
                "code": self.hls_code,
            }
        }
        self.saved = []

    def _llm_usage_summary(self):
        return {}

    def save_multistep_results(self, output_dir, bench_name, results):
        self.saved.append((output_dir, bench_name, results))


class FlashFinalizeChainTest(unittest.TestCase):
    def test_finalize_success_chains_pragma_then_latency(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cell = Path(tmp) / "cell"
            cell.mkdir()
            bench_dir = Path(tmp) / "bench"
            bench_dir.mkdir()
            (bench_dir / "metadata.json").write_text(
                json.dumps({
                    "hls_top": "kernel",
                    "translated_hls_top": "kernel",
                    "part": "xcu280-fsvh2892-2L-e",
                    "clock_ns": 3.33,
                    "supports_cosim": False,
                }),
                encoding="utf-8",
            )
            (bench_dir / "kernel.cpp").write_text("void kernel() {}\n", encoding="utf-8")
            (bench_dir / "kernel_tb.cpp").write_text("int main(){return 0;}\n", encoding="utf-8")

            session = FlashPipelinedBenchSession.__new__(FlashPipelinedBenchSession)
            session.variant_key = "test"
            session.bench = "chathls_kernel_2mm"
            session.bench_dir = bench_dir
            session.cell_dir = cell
            session.model_id = "fake"
            session.turns = 1
            session.pipelined_dir = cell / "pipelined"
            session.pipelined_dir.mkdir()
            session.inputs = {
                "meta": {
                    "hls_top": "kernel",
                    "translated_hls_top": "kernel",
                    "part": "xcu280-fsvh2892-2L-e",
                    "clock_ns": 3.33,
                    "supports_cosim": False,
                }
            }
            session.reference_validation = {
                "synthesis": {"status": "ok"},
                "workflow": [],
            }
            orch = FakeOrch()
            session.orchestrator = orch

            pragma_calls = []
            latency_calls = []

            def fake_pragma(**kwargs):
                pragma_calls.append(kwargs)
                return MagicMock(success=True)

            def fake_latency(**kwargs):
                latency_calls.append(kwargs)
                return MagicMock(success=True)

            with patch(
                "flash_pipelined_bench._sanitize_saved_result_record",
                side_effect=lambda results, _rv: results,
            ), patch(
                "flash_pipelined_bench._build_coverage",
                return_value={},
            ), patch(
                "flash_pipelined_bench._build_run_attribution",
                return_value={},
            ), patch(
                "post_flash_pragma_opt.maybe_chain_pragma_opt",
                side_effect=fake_pragma,
            ), patch(
                "post_flash_latency_opt.maybe_chain_latency_opt",
                side_effect=fake_latency,
            ):
                session._finalize_success()

            self.assertEqual(len(pragma_calls), 1)
            self.assertEqual(len(latency_calls), 1)
            self.assertEqual(pragma_calls[0]["source_role"], "flash_final")
            self.assertEqual(latency_calls[0]["source_role"], "flash_final")
            self.assertEqual(pragma_calls[0]["bench"], "chathls_kernel_2mm")
            self.assertEqual(latency_calls[0]["cell_dir"], cell)
            self.assertTrue((cell / "chathls_kernel_2mm_multistep_results.json").is_file())


if __name__ == "__main__":
    unittest.main()
