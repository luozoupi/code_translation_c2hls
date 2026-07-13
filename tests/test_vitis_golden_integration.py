"""Opt-in real Vitis integration for the paper correctness gate.

Run only on a host with the paper toolchain sourced:

  C2HLS_RUN_VITIS_INTEGRATION=1 \
  C2HLS_VITIS_SETTINGS=/path/to/Vitis/2023.2/settings64.sh \
  python -m pytest -q tests/test_vitis_golden_integration.py
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import hls_eval  # noqa: E402


HEADER = r'''
#ifndef TINY_H
#define TINY_H
extern "C" void workload(const int input[4], int output[4]);
#endif
'''

TESTBENCH = r'''
#include "tiny.h"
#include <cstdio>
int main() {
  int input[4] = {1, 3, 5, 7};
  int output[4] = {0, 0, 0, 0};
  workload(input, output);
  for (int i = 0; i < 4; ++i) {
    if (output[i] != input[i] + 1) {
      std::fprintf(stderr, "FAIL index=%d expected=%d actual=%d\n",
                   i, input[i] + 1, output[i]);
      return 1;
    }
  }
  std::puts("PASS tiny golden testbench");
  return 0;
}
'''


def kernel(delta: int) -> str:
    return f'''
#include "tiny.h"
extern "C" void workload(const int input[4], int output[4]) {{
#pragma HLS INTERFACE m_axi port=input offset=slave bundle=gmem0 depth=4
#pragma HLS INTERFACE m_axi port=output offset=slave bundle=gmem1 depth=4
#pragma HLS INTERFACE s_axilite port=input bundle=control
#pragma HLS INTERFACE s_axilite port=output bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control
  for (int i = 0; i < 4; ++i) {{
#pragma HLS PIPELINE II=1
    output[i] = input[i] + {delta};
  }}
}}
'''


@unittest.skipUnless(
    os.getenv("C2HLS_RUN_VITIS_INTEGRATION", "0").lower()
    in {"1", "true", "yes", "on"},
    "set C2HLS_RUN_VITIS_INTEGRATION=1 for real Vitis CSim/cosim",
)
class RealVitisGoldenIntegrationTests(unittest.TestCase):
    def test_passing_and_corrupted_candidates_run_through_csim_and_cosim(self):
        settings = Path(os.getenv("C2HLS_VITIS_SETTINGS", ""))
        self.assertTrue(settings.is_file(), "C2HLS_VITIS_SETTINGS is required")
        common = {
            "testbench_code": TESTBENCH,
            "header_code": HEADER,
            "header_name": "tiny.h",
            "top_function": "workload",
            "part": "xcu280-fsvh2892-2L-e",
            "clock_ns": 3.33,
            "extra_files": [],
        }
        for delta, expected in ((1, True), (2, False)):
            with self.subTest(delta=delta, stage="csim"):
                csim = hls_eval.run_csim(kernel(delta), **common)
                self.assertEqual(expected, bool(csim.get("passed")), csim.get("error"))
            with self.subTest(delta=delta, stage="cosim"):
                cosim = hls_eval.run_cosim(
                    kernel(delta),
                    interface_depths={"input": 4, "output": 4},
                    **common,
                )
                self.assertEqual(expected, bool(cosim.get("passed")), cosim.get("error"))
                if expected:
                    self.assertGreater(int(cosim.get("kernel_runtime_cycles") or 0), 0)


if __name__ == "__main__":
    unittest.main()
