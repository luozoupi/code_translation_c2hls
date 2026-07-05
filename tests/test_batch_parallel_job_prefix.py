"""Tests for batch_parallel Slurm job-name prefixes."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "pc2"))

from batch_parallel_config import (
    BatchParallelConfig,
    campaign_job_prefix,
    init_campaign_json,
    load_config,
)


class BatchParallelJobPrefixTests(unittest.TestCase):
    def test_full_flash_config_uses_bpfcosim(self) -> None:
        import os

        config = REPO / "scripts/pc2/batch_parallel_full_aav_n_park.json"
        prev = os.environ.get("BATCH_PARALLEL_CONFIG")
        os.environ["BATCH_PARALLEL_CONFIG"] = str(config)
        try:
            cfg = load_config()
            self.assertEqual(cfg.job_prefix, "bpfcosim")
        finally:
            if prev is None:
                os.environ.pop("BATCH_PARALLEL_CONFIG", None)
            else:
                os.environ["BATCH_PARALLEL_CONFIG"] = prev

    def test_campaign_job_prefix_top_level(self) -> None:
        doc = {"job_prefix": "bpfcosim", "config": {"job_prefix": "bpcplx"}}
        self.assertEqual(campaign_job_prefix(doc), "bpfcosim")

    def test_init_campaign_json_stores_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cfg = BatchParallelConfig(job_prefix="bpfcosim")
            doc = init_campaign_json(root, cfg, stamp="test_stamp")
            self.assertEqual(doc["job_prefix"], "bpfcosim")
            saved = json.loads((root / "campaign.json").read_text())
            self.assertEqual(saved["job_prefix"], "bpfcosim")
            self.assertEqual(saved["config"]["job_prefix"], "bpfcosim")


if __name__ == "__main__":
    unittest.main()
