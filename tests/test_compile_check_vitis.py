"""Tests for Phase A g++ compile checks with Vitis HLS includes."""

from __future__ import annotations

import os
import unittest
from unittest import mock

from c2hls import _vitis_hls_compile_include_paths, compile_check_cpp


class CompileCheckVitisTests(unittest.TestCase):
    def test_manual_include_paths(self):
        with mock.patch.dict(os.environ, {"C2HLS_COMPILE_INCLUDE_PATHS": "/tmp"}, clear=False):
            self.assertEqual(_vitis_hls_compile_include_paths(), ["/tmp"])

    def test_derives_from_vitis_settings(self):
        env = {
            "C2HLS_COMPILE_INCLUDE_PATHS": "",
            "XILINX_HLS": "",
            "C2HLS_VITIS_SETTINGS": "/opt/software/FPGA/Xilinx/Vitis/2023.2/settings64.sh",
            "C2HLS_VITIS_VERSION": "2023.2",
        }
        with mock.patch.dict(os.environ, env, clear=False):
            with mock.patch("os.path.isdir", side_effect=lambda p: p.endswith("Vitis_HLS/2023.2/include")):
                paths = _vitis_hls_compile_include_paths()
        self.assertEqual(paths, ["/opt/software/FPGA/Xilinx/Vitis_HLS/2023.2/include"])

    def test_forgebench_plain_compiles_with_vitis_includes(self):
        inc = "/opt/software/FPGA/Xilinx/Vitis_HLS/2023.2/include"
        if not os.path.isfile(os.path.join(inc, "ap_fixed.h")):
            self.skipTest("Vitis HLS headers not installed on this host")
        plain = (
            "#include <ap_fixed.h>\n"
            "typedef ap_fixed<16, 5> data_t;\n"
            "void top(data_t x[4]) { x[0] = (data_t)1; }\n"
        )
        with mock.patch("c2hls._vitis_hls_compile_include_paths", return_value=[inc]):
            ok, err = compile_check_cpp(plain)
        self.assertTrue(ok, err)


if __name__ == "__main__":
    unittest.main()
