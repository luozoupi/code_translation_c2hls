"""Unit tests for csim TCL staging of bench-local support files."""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import hls_eval


class CsimTclExtraFilesTests(unittest.TestCase):
    def test_includes_support_txt_and_headers(self) -> None:
        work_dir = "/tmp/hls_csim_test"
        extra_files = [
            {"path": "support/DRAM_attn_input.txt", "content": "0.0\n", "tb": True},
            {"path": "top.h", "content": "// hdr\n", "tb": True},
            {"path": "helper.cpp", "content": "// cpp\n", "tb": True},
        ]
        lines = hls_eval._tcl_tb_extra_add_lines(work_dir, extra_files, relative=True)
        self.assertEqual(
            lines,
            [
                "add_files -tb support/DRAM_attn_input.txt",
                "add_files -tb top.h",
                "add_files -tb helper.cpp",
            ],
        )

    def test_skips_paths_already_added(self) -> None:
        work_dir = "/tmp/hls_csim_test"
        extra_files = [{"path": "support/a.txt", "content": "1", "tb": True}]
        skip = {os.path.normpath(os.path.abspath(f"{work_dir}/support/a.txt"))}
        lines = hls_eval._tcl_tb_extra_add_lines(work_dir, extra_files, skip_abs_paths=skip)
        self.assertEqual(lines, [])

    def test_honors_tb_false(self) -> None:
        work_dir = "/tmp/hls_csim_test"
        extra_files = [{"path": "support/a.txt", "content": "1", "tb": False}]
        self.assertEqual(hls_eval._tcl_tb_extra_add_lines(work_dir, extra_files), [])

    def test_forgebench_attention_extra_files_from_inputs(self) -> None:
        from c2hls import _load_benchmark_inputs

        bench_dir = (
            REPO
            / "related_work/benchmarks/HLSFactory_benchmarks/tier_A_ready/forgebench_attention_op_p1"
        )
        inputs = _load_benchmark_inputs(str(bench_dir))
        work_dir = "/tmp/forgebench_attn_csim"
        lines = hls_eval._tcl_tb_extra_add_lines(work_dir, inputs.get("extra_files") or [], relative=True)
        joined = "\n".join(lines)
        for rel in (
            "support/DRAM_attn_input.txt",
            "support/DRAM_weights_q.txt",
            "support/DRAM_weights_k.txt",
            "support/DRAM_weights_v.txt",
            "support/DRAM_output.txt",
        ):
            self.assertIn(f"add_files -tb {rel}", joined)

    def test_forgebench_tiled_attn_extra_files_and_bare_tb_paths(self) -> None:
        from c2hls import _load_benchmark_inputs

        bench_dir = (
            REPO
            / "related_work/benchmarks/HLSFactory_benchmarks/tier_A_ready/forgebench_tiled_attn_p1"
        )
        tb = (bench_dir / "testbench.cpp").read_text(encoding="utf-8")
        self.assertIn('"DRAM_attn_input.txt"', tb)
        self.assertNotIn("support/DRAM_attn_input.txt", tb)

        inputs = _load_benchmark_inputs(str(bench_dir))
        lines = hls_eval._tcl_tb_extra_add_lines(
            "/tmp/forgebench_tiled_attn_csim",
            inputs.get("extra_files") or [],
            relative=True,
        )
        joined = "\n".join(lines)
        self.assertIn("add_files -tb support/DRAM_attn_input.txt", joined)


class PrepareTierAReadyNormalsTests(unittest.TestCase):
    def test_normals_vmap_is_spatially_varying(self) -> None:
        sys.path.insert(0, str(REPO / "scripts"))
        from prepare_tier_a_ready import _patch_normals_testbench_vmap

        upstream = (
            '    for (int i = 0; i < rows * cols * 3; i++)\n'
            '        vmap[i] = 1;\n'
        )
        patched = _patch_normals_testbench_vmap(upstream)
        self.assertIn("vmap[i * 3 + 0] = r + 1", patched)
        self.assertNotIn("vmap[i] = 1", patched)

        tier_a_ready_tb = (
            REPO
            / "related_work/benchmarks/HLSFactory_benchmarks/tier_A_ready/spector_hls_normals/testbench.cpp"
        ).read_text(encoding="utf-8")
        self.assertIn("vmap[i * 3 + 0] = r + 1", tier_a_ready_tb)
        self.assertNotIn("vmap[i] = 1", tier_a_ready_tb)


class PrepareTierAReadyAttentionScaleTests(unittest.TestCase):
    def test_forgebench_sqrt_patch_in_gold_only(self) -> None:
        sys.path.insert(0, str(REPO / "scripts"))
        from prepare_tier_a_ready import _patch_forgebench_ap_fixed_sqrt, _patch_normals_gold_normalized

        scale_line = "    const data_t scale = (data_t)1.0 / hls::sqrt((data_t)head_dim);\n"
        self.assertIn("std::sqrt((double)head_dim)", _patch_forgebench_ap_fixed_sqrt(scale_line))

        rms_line = "        data_t rms = hls::sqrt(sum_sq / (data_t)32 + (data_t)0.01);\n"
        self.assertIn("std::sqrt((double)(sum_sq", _patch_forgebench_ap_fixed_sqrt(rms_line))

        norm_old = (
            "void normalized(int v[3]) {\n"
            "    float t = sqrt(1 / int((v[0] * v[0]) + (v[1] * v[1]) + (v[2] * v[2])));\n"
        )
        self.assertIn("if (mag_sq == 0)", _patch_normals_gold_normalized(norm_old))

        gold = (
            REPO
            / "related_work/benchmarks/HLSFactory_benchmarks/tier_A_ready/forgebench_attention_op_p1/hls_baseline.cpp"
        ).read_text(encoding="utf-8")
        plain = (
            REPO
            / "related_work/benchmarks/HLSFactory_benchmarks/tier_A_ready/forgebench_attention_op_p1/plain.cpp"
        ).read_text(encoding="utf-8")
        self.assertIn("std::sqrt((double)head_dim)", gold)
        self.assertIn("hls::sqrt((data_t)head_dim)", plain)


class SynthTimeoutTests(unittest.TestCase):
    def test_reads_env_at_call_time(self) -> None:
        prev = os.environ.get("C2HLS_SYNTH_TIMEOUT")
        try:
            os.environ["C2HLS_SYNTH_TIMEOUT"] = "3600"
            self.assertEqual(hls_eval._synth_timeout(), 3600)
            os.environ["C2HLS_SYNTH_TIMEOUT"] = "900"
            self.assertEqual(hls_eval._synth_timeout(), 900)
        finally:
            if prev is None:
                os.environ.pop("C2HLS_SYNTH_TIMEOUT", None)
            else:
                os.environ["C2HLS_SYNTH_TIMEOUT"] = prev


if __name__ == "__main__":
    unittest.main()
