#!/usr/bin/env python3
"""autosa_ready TB/kernel must share C linkage so Phase B csim can link."""

from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.prepare_autosa_ready import (  # noqa: E402
    _insert_forward_decl,
    prepare_one,
)
from c2hls import (  # noqa: E402
    _expected_top_signature,
    _top_signature_mismatch_reason,
)


TB_PREFIX = '#include "kernel.h"\nint leftover;\n'
PHASE_B_KERNEL = '''\
#include "kernel.h"
extern "C" {
void autosa_mm(data_t A[I][K], data_t B[J][K], data_t C[I][J]) {
  C[0][0] = A[0][0] * B[0][0];
}
}
'''


class AutosaReadyCsimAbiTest(unittest.TestCase):
    def test_tb_forward_decl_is_extern_c(self) -> None:
        tb = _insert_forward_decl(TB_PREFIX, "autosa_mm", "data_t A[I][K], data_t B[J][K], data_t C[I][J]")
        self.assertIn('extern "C" void autosa_mm(data_t A[I][K], data_t B[J][K], data_t C[I][J]);', tb)
        sig = _expected_top_signature("", tb, "autosa_mm")
        self.assertIsNotNone(sig)
        self.assertTrue(sig["extern_c"])

    def test_phase_b_extern_c_kernel_matches_tb(self) -> None:
        tb = _insert_forward_decl(TB_PREFIX, "autosa_mm", "data_t A[I][K], data_t B[J][K], data_t C[I][J]")
        reason = _top_signature_mismatch_reason(
            PHASE_B_KERNEL, "", tb, "autosa_mm"
        )
        self.assertEqual(reason, "")

    def test_prepare_mm_gold_and_tb_share_c_linkage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            result = prepare_one("mm", "autosa_tests/mm", out)
            self.assertTrue(result.get("ok"))
            bench = out / "autosa_mm"
            tb = (bench / "testbench.cpp").read_text(encoding="utf-8")
            gold = (bench / "hls_baseline.cpp").read_text(encoding="utf-8")
            header = (bench / "kernel.h").read_text(encoding="utf-8")
            self.assertIn('extern "C" void autosa_mm(', tb)
            self.assertIn('extern "C"', gold)
            sig = _expected_top_signature(header, tb, "autosa_mm")
            self.assertTrue(sig and sig["extern_c"])
            self.assertEqual(
                _top_signature_mismatch_reason(gold, header, tb, "autosa_mm"),
                "",
            )
            self.assertEqual(
                _top_signature_mismatch_reason(PHASE_B_KERNEL, header, tb, "autosa_mm"),
                "",
            )

    def test_gxx_links_extern_c_kernel_against_tb(self) -> None:
        gxx = subprocess.run(["g++", "--version"], capture_output=True, text=True)
        if gxx.returncode != 0:
            self.skipTest("g++ not available")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "kernel.h").write_text(
                "typedef float data_t;\n#define I 2\n#define J 2\n#define K 2\n",
                encoding="utf-8",
            )
            (root / "kernel.cpp").write_text(
                '#include "kernel.h"\n'
                'extern "C" void autosa_mm(data_t A[I][K], data_t B[J][K], data_t C[I][J]) {\n'
                "  C[0][0] = A[0][0] * B[0][0];\n"
                "}\n",
                encoding="utf-8",
            )
            tb = _insert_forward_decl(
                '#include "kernel.h"\n#include <stdio.h>\n',
                "autosa_mm",
                "data_t A[I][K], data_t B[J][K], data_t C[I][J]",
            )
            tb += (
                "int main() {\n"
                "  data_t A[I][K] = {}; data_t B[J][K] = {}; data_t C[I][J] = {};\n"
                "  autosa_mm(A, B, C);\n"
                "  return 0;\n"
                "}\n"
            )
            (root / "testbench.cpp").write_text(tb, encoding="utf-8")
            linked = subprocess.run(
                ["g++", "-o", str(root / "csim.exe"), str(root / "kernel.cpp"), str(root / "testbench.cpp")],
                capture_output=True,
                text=True,
            )
            self.assertEqual(linked.returncode, 0, linked.stderr)


if __name__ == "__main__":
    unittest.main()
