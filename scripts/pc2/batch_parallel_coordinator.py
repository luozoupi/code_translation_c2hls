#!/usr/bin/env python3
"""Login-node coordinator: batch_park GPU session + campaign completion."""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCRIPT_DIR = REPO / "scripts" / "pc2"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(SCRIPT_DIR))

from batch_parallel_config import campaign_paths, gpu_parking_enabled, gpu_policy_from_campaign, load_campaign, load_config, save_campaign
from batch_parallel_flow import BatchParallelFlow
from batch_parallel_gpu_state import gpu_must_stay_up, snapshot_gpu_busy
from batch_parallel_park import can_hard_park, evaluate_park_request, should_unpark
from batch_parallel_queue import BatchParallelQueue


def _job_active(job_id: str | None) -> bool:
    if not job_id or job_id in ("None", "null"):
        return False
    try:
        out = subprocess.check_output(["squeue", "-h", "-j", str(job_id)], stderr=subprocess.DEVNULL, text=True)
        return bool(out.strip())
    except subprocess.CalledProcessError:
        return False


def _endpoint_healthy(endpoint_file: Path) -> bool:
    if not endpoint_file.is_file():
        return False
    try:
        payload = json.loads(endpoint_file.read_text(encoding="utf-8"))
        url = payload.get("url", "").rstrip("/")
        if not url:
            return False
        import urllib.request

        for suffix in ("/models", "/health"):
            try:
                with urllib.request.urlopen(f"{url}{suffix}", timeout=5) as resp:
                    if resp.status == 200:
                        return True
            except Exception:
                continue
    except Exception:
        return False
    return False


def _gpu_borrowed(campaign: dict, endpoint_file: Path) -> bool:
    if campaign.get("gpu_borrowed"):
        return True
    if not endpoint_file.is_file():
        return False
    try:
        payload = json.loads(endpoint_file.read_text(encoding="utf-8"))
        return bool(payload.get("borrowed"))
    except Exception:
        return False


def _scancel(job_id: str | None, *, campaign: dict | None = None, endpoint_file: Path | None = None) -> None:
    if not job_id or job_id in ("None", "null"):
        return
    if campaign is not None and endpoint_file is not None and _gpu_borrowed(campaign, endpoint_file):
        logging.info("skip scancel gpu job %s (borrowed endpoint)", job_id)
        return
    subprocess.run(["scancel", str(job_id)], check=False)


def _submit_gpu(campaign_root: Path) -> str:
    env = os.environ.copy()
    env["BATCH_PARALLEL_CAMPAIGN_ROOT"] = str(campaign_root)
    out = subprocess.check_output(
        [str(SCRIPT_DIR / "batch_parallel_submit_gpu.sh")],
        env=env,
        text=True,
    ).strip()
    return out.split(";")[-1].strip()


def vitis_pipeline_busy(queue: BatchParallelQueue) -> bool:
    return queue.pending_or_claimed_count(kinds=("synth", "cosim")) > 0


def _clear_park_pending(campaign: dict) -> None:
    campaign.pop("park_pending_at", None)
    campaign.pop("park_pending_reason", None)


def write_summary(campaign_root: Path, queue: BatchParallelQueue, campaign: dict) -> None:
    summary = {
        "campaign_status": campaign.get("campaign_status"),
        "completed_at": campaign.get("completed_at"),
        "gpu_mode": campaign.get("gpu_mode"),
        "pending_codegen_final": queue.pending_codegen(),
    }
    (campaign_root / "campaign_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="batch_parallel coordinator")
    parser.add_argument("--campaign-root", required=True)
    args = parser.parse_args()

    campaign_root = Path(args.campaign_root)
    paths = campaign_paths(campaign_root)
    cfg = load_config()
    queue = BatchParallelQueue(paths["queue_db"])
    flow = BatchParallelFlow(campaign_root)
    paths["coordinator_pid"].write_text(str(os.getpid()) + "\n", encoding="utf-8")

    while True:
        campaign = load_campaign(campaign_root)
        active_variants = campaign.get("active_variants") or [cfg.pilot_variant]
        gpu_mode = str(campaign.get("gpu_mode") or "up")
        external_llm = bool(campaign.get("external_llm"))

        flow.write_status({
            "gpu_mode": gpu_mode,
            "gpu_policy": gpu_policy_from_campaign(campaign, cfg),
            **snapshot_gpu_busy(queue, campaign_root),
            "codegen_demand": queue.codegen_demand_count(),
            "pending_synth": queue.pending_or_claimed_count(kinds=("synth",)),
            "pending_cosim": queue.pending_or_claimed_count(kinds=("cosim",)),
            "claimed_cosim": len(queue.claimed_cosim_jobs()),
            "benches_non_terminal": queue.benches_non_terminal(),
            "park_pending_reason": campaign.get("park_pending_reason"),
        })
        flow.write_node_map(queue.snapshot_node_map())

        stale_s = float(
            (campaign.get("config") or {}).get("stale_claim_s")
            or cfg.stale_claim_s
            or 1800.0
        )
        stale = queue.requeue_stale_claimed(max_age_s=stale_s)
        if stale:
            logging.info(
                "stale-claim sweeper requeued %d job(s) (max_age_s=%.0f): %s",
                len(stale),
                stale_s,
                stale,
            )
            flow.emit(
                "stale_claims_requeued",
                scope="campaign",
                count=len(stale),
                job_ids=stale,
                max_age_s=stale_s,
            )

        if queue.campaign_complete(active_variants):
            campaign["campaign_status"] = "completing"
            save_campaign(campaign_root, campaign)
            if queue.pending_codegen() > 0 or queue.claimed_codegen() > 0:
                if external_llm:
                    flow.emit("codegen_tail_wait", scope="gpu", reason="external_llm_always_on")
                    while queue.pending_codegen() > 0 or queue.claimed_codegen() > 0:
                        time.sleep(cfg.poll_sec)
                else:
                    reason = "tail_flush"
                    campaign["gpu_mode"] = "pending_unpark"
                    _clear_park_pending(campaign)
                    save_campaign(campaign_root, campaign)
                    job_id = _submit_gpu(campaign_root)
                    campaign["gpu_job_id"] = job_id
                    save_campaign(campaign_root, campaign)
                    flow.emit("gpu_unpark_request", scope="gpu", reason=reason)
                    while queue.pending_codegen() > 0 or queue.claimed_codegen() > 0:
                        time.sleep(cfg.poll_sec)
            if not external_llm and _job_active(campaign.get("gpu_job_id")):
                _scancel(campaign.get("gpu_job_id"), campaign=campaign, endpoint_file=paths["endpoint"])
            campaign = load_campaign(campaign_root)
            campaign["campaign_status"] = "complete"
            campaign["gpu_mode"] = "up" if external_llm else "parked"
            campaign["completed_at"] = datetime.now(timezone.utc).isoformat()
            _clear_park_pending(campaign)
            save_campaign(campaign_root, campaign)
            flow.emit("campaign_complete", scope="campaign")
            write_summary(campaign_root, queue, campaign)
            flow._render_reports(force=True)
            paths["complete_marker"].write_text(campaign["completed_at"] + "\n", encoding="utf-8")
            return 0

        unpark_reason = None if external_llm else should_unpark(queue, cfg, campaign)
        if unpark_reason:
            flow.emit("batch_unpark_trigger", scope="gpu", reason=unpark_reason, pending=queue.pending_codegen())
            campaign["gpu_mode"] = "pending_unpark"
            _clear_park_pending(campaign)
            save_campaign(campaign_root, campaign)
            job_id = _submit_gpu(campaign_root)
            campaign["gpu_job_id"] = job_id
            save_campaign(campaign_root, campaign)
            for _ in range(600):
                if _endpoint_healthy(paths["endpoint"]):
                    campaign = load_campaign(campaign_root)
                    campaign["gpu_mode"] = "up"
                    campaign["parked_codegen_since"] = None
                    _clear_park_pending(campaign)
                    save_campaign(campaign_root, campaign)
                    flow.emit("gpu_up", scope="gpu", reason=unpark_reason)
                    break
                time.sleep(5)
            else:
                logging.warning("GPU endpoint not healthy after unpark")

        elif str(campaign.get("gpu_mode") or "up") == "pending_unpark":
            if _endpoint_healthy(paths["endpoint"]):
                campaign = load_campaign(campaign_root)
                campaign["gpu_mode"] = "up"
                campaign["parked_codegen_since"] = None
                _clear_park_pending(campaign)
                save_campaign(campaign_root, campaign)
                flow.emit("gpu_up", scope="gpu", reason="pending_unpark_recovery")

        if not external_llm and gpu_parking_enabled(campaign, cfg):
            if gpu_must_stay_up(queue, campaign_root, campaign) and campaign.get("park_pending_at"):
                _clear_park_pending(campaign)
                save_campaign(campaign_root, campaign)

            park_reason = evaluate_park_request(queue, campaign, cfg, campaign_root)
            if park_reason and _job_active(campaign.get("gpu_job_id")):
                if not campaign.get("park_pending_at"):
                    campaign["park_pending_at"] = time.time()
                    campaign["park_pending_reason"] = park_reason
                    save_campaign(campaign_root, campaign)
                    flow.emit(
                        "gpu_park_pending",
                        scope="gpu",
                        reason=park_reason,
                        grace_s=cfg.park_grace_s,
                    )
                else:
                    ready, hard_reason = can_hard_park(queue, campaign, cfg, campaign_root)
                    if ready and hard_reason:
                        job_id = campaign.get("gpu_job_id")
                        flow.emit("gpu_parked", scope="gpu", reason=hard_reason)
                        campaign["gpu_mode"] = "parked"
                        if not campaign.get("parked_codegen_since") and queue.pending_codegen() > 0:
                            campaign["parked_codegen_since"] = time.time()
                        _clear_park_pending(campaign)
                        save_campaign(campaign_root, campaign)
                        _scancel(job_id, campaign=campaign, endpoint_file=paths["endpoint"])
            elif campaign.get("park_pending_at"):
                _clear_park_pending(campaign)
                save_campaign(campaign_root, campaign)

            if gpu_mode == "parked" and queue.pending_codegen() > 0 and not campaign.get("parked_codegen_since"):
                campaign["parked_codegen_since"] = time.time()
                save_campaign(campaign_root, campaign)

        flow._render_reports()
        time.sleep(cfg.coordinator_poll_sec)


if __name__ == "__main__":
    raise SystemExit(main())
