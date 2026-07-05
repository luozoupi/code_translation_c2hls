#!/usr/bin/env python3
"""Smoke test for post-flash csynth bundle export."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(REPO))

from scripts.pc2.export_post_flash_dataflow_csynth_bundle import export_csynth_bundle


class ExportCsynthBundleTest(unittest.TestCase):
    def test_export_creates_manifest_and_dirs(self) -> None:
        matrix = REPO / "artifacts/pc2/flash_all_new_skills_avoids_global_20260623_024548"
        flash_bundle = REPO / "artifacts/pc2/flash_selected_bundle" / matrix.name
        if not matrix.is_dir() or not flash_bundle.is_dir():
            self.skipTest("matrix artifacts not present on this host")

        with tempfile.TemporaryDirectory() as tmp:
            bundle = Path(tmp) / "kernel_bundle"
            manifest = export_csynth_bundle(
                matrix_root=matrix,
                flash_bundle_root=flash_bundle,
                kernel_bundle=bundle,
                force=True,
                benches=["hlsfactory_gemver"],
            )
            self.assertEqual(manifest["flash"]["exported"], 1)
            self.assertEqual(manifest["dataflow"]["exported"], 1)
            gemver_flash = bundle / "hlsfactory_gemver" / "flash_csynth"
            gemver_df = bundle / "hlsfactory_gemver" / "dataflow_csynth"
            self.assertTrue((gemver_flash / "csynth.rpt").is_file())
            self.assertTrue((gemver_df / "csynth.rpt").is_file())
            # Per-loop pipeline report with achieved II
            pipeline = list(gemver_df.glob("*_Pipeline_*_csynth.rpt"))
            self.assertTrue(pipeline, "expected per-pipeline csynth reports")
            self.assertTrue((gemver_df / "vitis_hls.log").is_file())
            log_text = (gemver_df / "vitis_hls.log").read_text(encoding="utf-8")
            self.assertIn("Vitis HLS", log_text)
            manifest_path = bundle / "csynth_bundle_manifest.json"
            self.assertTrue(manifest_path.is_file())
            loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertIn("flash", loaded)
            self.assertIn("dataflow", loaded)


if __name__ == "__main__":
    unittest.main()
