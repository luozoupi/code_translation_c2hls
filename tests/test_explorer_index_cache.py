"""Tests for explorer disk index cache."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "explorer"))

from index_cache import (  # noqa: E402
    build_and_cache_index,
    compute_fingerprint,
    get_index,
    load_disk_cache,
)


class ExplorerIndexCacheTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.registry = self.root / "experiments_registry.json"
        self.registry.write_text(json.dumps({"experiments": []}) + "\n", encoding="utf-8")
        self._seed_campaign("fir", "flash_smoke_20260705_204107")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _seed_campaign(self, site: str, name: str) -> Path:
        camp = self.root / "artifacts" / site / name
        camp.mkdir(parents=True)
        bench = camp / "hlsfactory_2mm" / "devstral2__flash__all_skills_avoids_global"
        bench.mkdir(parents=True)
        result = {
            "success": True,
            "steps": [
                {
                    "step_name": "flash",
                    "success": True,
                    "report": {"latency_cycles": 100, "target_cycles": 200},
                }
            ],
        }
        (bench / "hlsfactory_2mm_multistep_results.json").write_text(
            json.dumps(result) + "\n",
            encoding="utf-8",
        )
        (camp / "matrix.json").write_text(
            json.dumps([
                {
                    "bench": "hlsfactory_2mm",
                    "model": "test/model",
                    "mode": "flash",
                    "status": "ok",
                    "result_path": str(bench / "hlsfactory_2mm_multistep_results.json"),
                }
            ])
            + "\n",
            encoding="utf-8",
        )
        return camp

    def test_disk_cache_hit_after_build(self) -> None:
        index1, meta1 = build_and_cache_index(self.root, self.registry)
        self.assertEqual(meta1["source"], "rebuild")
        self.assertEqual(len(index1["experiments"]), 1)

        disk_index, disk_meta = load_disk_cache(self.root)
        self.assertIsNotNone(disk_index)
        self.assertEqual(disk_meta.get("experiment_count"), 1)

        index2, meta2 = get_index(self.root, self.registry, cache_sec=300.0)
        self.assertEqual(meta2["source"], "disk")
        self.assertEqual(len(index2["experiments"]), 1)

    def test_fingerprint_changes_when_matrix_updates(self) -> None:
        fp1 = compute_fingerprint(self.root, self.registry)
        camp = self.root / "artifacts" / "fir" / "flash_smoke_20260705_204107"
        matrix = json.loads((camp / "matrix.json").read_text(encoding="utf-8"))
        matrix[0]["status"] = "fail"
        (camp / "matrix.json").write_text(json.dumps(matrix) + "\n", encoding="utf-8")
        fp2 = compute_fingerprint(self.root, self.registry)
        self.assertNotEqual(fp1, fp2)


if __name__ == "__main__":
    unittest.main()
