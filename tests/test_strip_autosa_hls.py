#!/usr/bin/env python3
"""AutoSA plain packaging: no modules+kernel duplication; keep streams."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.strip_autosa_hls import (  # noqa: E402
    aggregate_autosa_sources,
    dedupe_top_level_void_functions,
    strip_autosa_hls,
)


MODULES = """\
#include "kernel_kernel.h"

void A_IO_L1_in(hls::stream<ap_uint<128> > &in, hls::stream<ap_uint<128> > &out) {
  out.write(in.read());
}

void A_IO_L1_in_wrapper(hls::stream<ap_uint<128> > &in, hls::stream<ap_uint<128> > &out) {
  A_IO_L1_in(in, out);
}
"""

KERNEL = """\
#include "kernel_kernel.h"

void A_IO_L1_in(hls::stream<ap_uint<128> > &in, hls::stream<ap_uint<128> > &out) {
  out.write(in.read());
}

void A_IO_L1_in_wrapper(hls::stream<ap_uint<128> > &in, hls::stream<ap_uint<128> > &out) {
  A_IO_L1_in(in, out);
}

void kernel0(A_t16 *A) {
#pragma HLS DATAFLOW
  hls::stream<ap_uint<128> > fifo_A;
#pragma HLS STREAM variable=fifo_A depth=2
  A_IO_L1_in_wrapper(fifo_A, fifo_A);
}
"""


class StripAutosaHlsTest(unittest.TestCase):
    def test_aggregate_uses_kernel_alone_when_modules_embedded(self) -> None:
        merged = aggregate_autosa_sources(MODULES, KERNEL)
        self.assertEqual(merged.count("void A_IO_L1_in("), 1)
        self.assertIn("void kernel0(", merged)
        self.assertIn("hls::stream<ap_uint<128> > fifo_A;", merged)

    def test_dedupe_keeps_first_body(self) -> None:
        dup = MODULES + "\n" + MODULES
        cleaned, report = dedupe_top_level_void_functions(dup)
        self.assertEqual(report["dropped"], 2)
        self.assertEqual(cleaned.count("void A_IO_L1_in("), 1)
        self.assertEqual(cleaned.count("void A_IO_L1_in_wrapper("), 1)

    def test_strip_keeps_stream_decls_drops_perf_pragmas(self) -> None:
        plain, report = strip_autosa_hls(KERNEL)
        self.assertIn("hls::stream<ap_uint<128> > fifo_A;", plain)
        self.assertNotIn("#pragma HLS DATAFLOW", plain)
        self.assertNotIn("#pragma HLS STREAM", plain)
        self.assertTrue(report.get("kept_hls_stream_declarations"))
        self.assertEqual(report.get("removed_hls_stream_declarations"), 0)


class PhaseBModuleIntegrityTest(unittest.TestCase):
    def test_repair_preserves_seed_modules(self) -> None:
        from c2hls import _repair_preserves_seed_modules

        seed = {"A_IO_L1_in", "A_IO_L1_in_wrapper", "PE", "kernel0"}
        wrappers_only = """
void A_IO_L1_in_wrapper() {}
void kernel0() { A_IO_L1_in(); }
"""
        ok, missing = _repair_preserves_seed_modules(seed, wrappers_only)
        self.assertFalse(ok)
        self.assertIn("A_IO_L1_in", missing)
        self.assertIn("PE", missing)

        full = MODULES + "\nvoid PE() {}\nvoid kernel0() {}\n"
        ok2, missing2 = _repair_preserves_seed_modules(seed, full)
        self.assertTrue(ok2)
        self.assertEqual(missing2, [])


if __name__ == "__main__":
    unittest.main()
