"""Cosim cycle parsing: prefer lat.rpt, fall back to *.result.lat.rb (MT_OFF)."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from hls_eval import _parse_lat_rpt_cycles, _parse_total_execute_time_cycles  # noqa: E402


class ParseLatRptCyclesTests(unittest.TestCase):
    def test_parse_total_execute_time(self) -> None:
        text = '$TOTAL_EXECUTE_TIME = "37741"\n$AVER_LATENCY = "37741"\n'
        self.assertEqual(_parse_total_execute_time_cycles(text), 37741)

    def test_prefers_lat_rpt_over_result_lat_rb(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "hls_proj" / "sol1" / "sim"
            report = root / "report" / "verilog"
            verilog = root / "verilog"
            report.mkdir(parents=True)
            verilog.mkdir(parents=True)
            (report / "lat.rpt").write_text(
                '$TOTAL_EXECUTE_TIME = "100"\n', encoding="utf-8"
            )
            (verilog / "fir_hls.result.lat.rb").write_text(
                '$TOTAL_EXECUTE_TIME = "37741"\n', encoding="utf-8"
            )
            self.assertEqual(_parse_lat_rpt_cycles(td, "hls_proj"), 100)

    def test_falls_back_to_result_lat_rb(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            verilog = Path(td) / "hls_proj" / "sol1" / "sim" / "verilog"
            verilog.mkdir(parents=True)
            (verilog / "fir_hls.result.lat.rb").write_text(
                "$MAX_LATENCY = \"37741\"\n"
                "$MIN_LATENCY = \"37741\"\n"
                "$AVER_LATENCY = \"37741\"\n"
                "$TOTAL_EXECUTE_TIME = \"37741\"\n",
                encoding="utf-8",
            )
            self.assertEqual(_parse_lat_rpt_cycles(td, "hls_proj"), 37741)

    def test_missing_reports_return_none(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            (Path(td) / "hls_proj").mkdir()
            self.assertIsNone(_parse_lat_rpt_cycles(td, "hls_proj"))


if __name__ == "__main__":
    unittest.main()
