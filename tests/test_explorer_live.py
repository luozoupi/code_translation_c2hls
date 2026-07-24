"""Tests for experiment explorer live campaign discovery."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "explorer"))

from live import discover_live_campaigns, resolve_campaign_path  # noqa: E402


class TestLiveDiscovery(unittest.TestCase):
    def test_discover_fir_campaign(self) -> None:
        campaigns = discover_live_campaigns(REPO)
        ids = {c["id"] for c in campaigns}
        self.assertIn("fir/batch_parallel_full_hlsfactory_20260706", ids)

    def test_resolve_campaign_path(self) -> None:
        path = resolve_campaign_path(
            REPO,
            "fir/batch_parallel_full_hlsfactory_20260706",
        )
        if path is None:
            self.skipTest("campaign not on disk")
        self.assertTrue((path / "queue.db").is_file())

    def test_synthetic_campaign_discovery(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            site = root / "artifacts" / "fir" / "batch_parallel_test"
            site.mkdir(parents=True)
            (site / "campaign.json").write_text(
                json.dumps({"campaign_status": "running"}),
                encoding="utf-8",
            )
            import sqlite3

            conn = sqlite3.connect(site / "queue.db")
            conn.execute(
                "CREATE TABLE jobs (id INTEGER PRIMARY KEY, bench TEXT, status TEXT)"
            )
            conn.execute("INSERT INTO jobs (bench, status) VALUES ('hlsfactory_2mm', 'claimed')")
            conn.commit()
            conn.close()
            found = discover_live_campaigns(root)
            self.assertEqual(len(found), 1)
            self.assertTrue(found[0]["active"])
            self.assertEqual(found[0]["summary"]["claimed"], 1)

    def test_aborted_campaign_not_active(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            site = root / "artifacts" / "fir" / "batch_parallel_aborted"
            site.mkdir(parents=True)
            (site / "campaign.json").write_text(
                json.dumps({"campaign_status": "aborted"}),
                encoding="utf-8",
            )
            import sqlite3

            conn = sqlite3.connect(site / "queue.db")
            conn.execute(
                "CREATE TABLE jobs (id INTEGER PRIMARY KEY, bench TEXT, status TEXT)"
            )
            conn.execute("INSERT INTO jobs (bench, status) VALUES ('hlsfactory_2mm', 'claimed')")
            conn.commit()
            conn.close()
            found = discover_live_campaigns(root)
            self.assertEqual(len(found), 1)
            self.assertFalse(found[0]["active"])
            self.assertEqual(found[0]["campaign_status"], "aborted")

    def test_pending_only_not_active(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            site = root / "artifacts" / "fir" / "batch_parallel_pending"
            site.mkdir(parents=True)
            (site / "campaign.json").write_text(
                json.dumps({"campaign_status": "running"}),
                encoding="utf-8",
            )
            import sqlite3

            conn = sqlite3.connect(site / "queue.db")
            conn.execute(
                "CREATE TABLE jobs (id INTEGER PRIMARY KEY, bench TEXT, status TEXT)"
            )
            conn.execute("INSERT INTO jobs (bench, status) VALUES ('hlsfactory_2mm', 'pending')")
            conn.commit()
            conn.close()
            found = discover_live_campaigns(root)
            self.assertEqual(len(found), 1)
            self.assertFalse(found[0]["active"])


if __name__ == "__main__":
    unittest.main()
