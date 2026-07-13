from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import run_agentic_sweep as sweep  # noqa: E402


class RunAgenticSweepResumeTests(unittest.TestCase):
    def _fingerprint(self, marker: str = "same") -> dict:
        payload = {
            "schema_version": sweep.FINGERPRINT_SCHEMA
            if hasattr(sweep, "FINGERPRINT_SCHEMA")
            else "c2hls.run-fingerprint.v1",
            "marker": marker,
        }
        from evaluation_repro import sha256_json

        return {
            "schema_version": "c2hls.run-fingerprint.v1",
            "sha256": sha256_json(payload),
            "payload": payload,
        }

    def test_loads_only_matching_valid_fingerprinted_result(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "result.json"
            fingerprint = self._fingerprint()
            path.write_text(json.dumps({
                "benchmark": "hlsfactory_2mm",
                "success": True,
                "steps": [],
                "run_fingerprint": fingerprint,
            }))
            result = sweep._load_resumable_result(
                path, "hlsfactory_2mm", fingerprint
            )
            self.assertIsNotNone(result)
            self.assertTrue(result["success"])
            self.assertIsNone(
                sweep._load_resumable_result(path, "hlsfactory_3mm", fingerprint)
            )
            self.assertIsNone(
                sweep._load_resumable_result(
                    path, "hlsfactory_2mm", self._fingerprint("changed")
                )
            )

            path.write_text("{broken json")
            self.assertIsNone(
                sweep._load_resumable_result(path, "hlsfactory_2mm", fingerprint)
            )

    def test_legacy_result_without_complete_fingerprint_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "result.json"
            path.write_text(json.dumps({
                "benchmark": "hlsfactory_2mm",
                "success": True,
                "steps": [],
            }))
            self.assertIsNone(
                sweep._load_resumable_result(
                    path, "hlsfactory_2mm", self._fingerprint()
                )
            )
            self.assertIsNone(
                sweep._load_resumable_result(path, "hlsfactory_2mm")
            )


if __name__ == "__main__":
    unittest.main()
