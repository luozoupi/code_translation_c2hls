"""Helpers must discharge when a batch_parallel campaign is terminal."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "pc2"))

from batch_parallel_lifecycle import (
    campaign_is_terminal,
    campaign_root_is_terminal,
    campaign_status_is_terminal,
    discharge_helper_jobs,
    helper_job_ids_to_discharge,
)


class HelperDischargeTests(unittest.TestCase):
    def test_terminal_statuses(self) -> None:
        for status in ("complete", "completed", "failed", "aborted", "COMPLETE"):
            self.assertTrue(campaign_status_is_terminal(status), status)
        for status in ("running", "completing", "", None):
            self.assertFalse(campaign_status_is_terminal(status), status)

    def test_campaign_dict(self) -> None:
        self.assertTrue(campaign_is_terminal({"campaign_status": "complete"}))
        self.assertFalse(campaign_is_terminal({"campaign_status": "running"}))
        self.assertFalse(campaign_is_terminal({}))

    def test_skips_self_and_empty_helper_ids(self) -> None:
        campaign = {
            "helper_jobs": {
                "watch": "2469973",
                "drain": "2469974",
                "coord": "2469975",
                "missing": None,
            }
        }
        self.assertEqual(
            helper_job_ids_to_discharge(campaign, self_job_id="2469975"),
            ["2469973", "2469974"],
        )
        self.assertEqual(
            helper_job_ids_to_discharge(campaign, self_job_id=None),
            ["2469973", "2469974", "2469975"],
        )

    def test_complete_marker_file(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.assertFalse(campaign_root_is_terminal(root))
            (root / "campaign.json").write_text('{"campaign_status": "running"}\n')
            self.assertFalse(campaign_root_is_terminal(root))
            (root / "CAMPAIGN_COMPLETE").write_text("done\n")
            self.assertTrue(campaign_root_is_terminal(root))
            (root / "campaign.json").write_text('{"campaign_status": "complete"}\n')
            (root / "CAMPAIGN_COMPLETE").unlink()
            self.assertTrue(campaign_root_is_terminal(root))

    def test_discharge_calls_scancel_except_self(self) -> None:
        cancelled: list[str] = []
        campaign = {
            "campaign_status": "complete",
            "helper_jobs": {"watch": "10", "drain": "11", "coord": "12"},
        }
        discharged = discharge_helper_jobs(
            campaign,
            self_job_id="12",
            scancel=cancelled.append,
        )
    def test_coordinator_discharge_skips_own_slurm_job(self) -> None:
        import os
        import batch_parallel_coordinator as coord

        cancelled: list[str] = []
        saved = os.environ.get("SLURM_JOB_ID")
        os.environ["SLURM_JOB_ID"] = "99"
        try:
            orig = coord._scancel
            coord._scancel = lambda jid, **_kwargs: cancelled.append(str(jid))
            coord._discharge_helpers({
                "helper_jobs": {"watch": "1", "drain": "2", "coord": "99"},
            })
            self.assertEqual(cancelled, ["1", "2"])
        finally:
            coord._scancel = orig
            if saved is None:
                os.environ.pop("SLURM_JOB_ID", None)
            else:
                os.environ["SLURM_JOB_ID"] = saved


if __name__ == "__main__":
    unittest.main()
