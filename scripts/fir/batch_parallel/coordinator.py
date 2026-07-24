#!/usr/bin/env python3
"""Login-node coordinator for Fir batch_parallel campaigns."""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
SCRIPT_DIR = REPO / "scripts" / "fir"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(SCRIPT_DIR))

from batch_parallel.config import (
    campaign_paths,
    gpu_parking_enabled,
    gpu_policy_from_campaign,
    load_campaign,
    load_config,
    save_campaign,
)
from batch_parallel.flow import FirBatchParallelFlow
from batch_parallel.gpu_state import gpu_must_stay_up, snapshot_gpu_busy
from batch_parallel.park import can_hard_park, evaluate_park_request, should_unpark
from batch_parallel.queue import FirBatchParallelQueue


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


def _clear_park_pending(campaign: dict) -> None:
    campaign.pop("park_pending_at", None)
    campaign.pop("park_pending_reason", None)


def write_matrix(campaign_root: Path, queue: FirBatchParallelQueue, campaign: dict) -> None:
    pilot = (campaign.get("config") or {}).get("pilot") or {}
    model_id = pilot.get("model", "")
    rows = []
    for job in queue.all_jobs():
        row = {
            "bench": job["bench"],
            "model": model_id,
            "mode": "flash",
            "status": "ok" if job["status"] == "done" else "fail",
            "error": job.get("error"),
            "result_path": job.get("result_path"),
            "worker_id": job.get("worker_id"),
            "node_index": job.get("node_index"),
            "worker_slot": job.get("worker_slot"),
        }
        result_path = job.get("result_path")
        if result_path and Path(result_path).is_file():
            try:
                result = json.loads(Path(result_path).read_text())
                from batch_parallel.dispatch import compact_summary

                row["summary"] = compact_summary(result)
            except Exception:
                pass
        rows.append(row)
    paths = campaign_paths(campaign_root)
    paths["matrix"].write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Fir batch_parallel coordinator")
    parser.add_argument("--campaign-root", required=True)
    args = parser.parse_args()

    campaign_root = Path(args.campaign_root)
    paths = campaign_paths(campaign_root)
    cfg = load_config()
    queue = FirBatchParallelQueue(paths["queue_db"])
    flow = FirBatchParallelFlow(campaign_root)
    paths["coordinator_pid"].write_text(str(os.getpid()) + "\n", encoding="utf-8")

    while True:
        campaign = load_campaign(campaign_root)
        gpu_mode = str(campaign.get("gpu_mode") or "up")

        flow.write_status({
            "campaign_status": campaign.get("campaign_status"),
            "compute_state": campaign.get("compute_state"),
            "gpu_mode": gpu_mode,
            "gpu_policy": gpu_policy_from_campaign(campaign, cfg),
            **snapshot_gpu_busy(queue, campaign_root, campaign),
            "pending_flash": queue.pending_flash_count(),
            "claimed_flash": queue.claimed_flash_count(),
            "park_pending_reason": campaign.get("park_pending_reason"),
        })
        flow.write_node_map(queue.snapshot_node_map())

        endpoint_ok = _endpoint_healthy(paths["endpoint"])
        if endpoint_ok and queue.retryable_failed_count() > 0:
            requeued = queue.requeue_retryable_failures()
            if requeued:
                flow.emit("flash_requeued", count=requeued, reason="endpoint_recovery")
                logging.info("requeued %d retryable failed benches", requeued)

        if queue.campaign_complete():
            if not endpoint_ok and (queue.pending_flash_count() > 0 or queue.claimed_flash_count() > 0):
                logging.info(
                    "defer campaign complete: endpoint down with %d pending / %d claimed",
                    queue.pending_flash_count(),
                    queue.claimed_flash_count(),
                )
                time.sleep(cfg.coordinator_poll_sec)
                continue
            campaign["campaign_status"] = "completing"
            save_campaign(campaign_root, campaign)

            if queue.pending_flash_count() > 0 and not _gpu_borrowed(campaign, paths["endpoint"]):
                campaign["gpu_mode"] = "pending_unpark"
                _clear_park_pending(campaign)
                save_campaign(campaign_root, campaign)
                job_id = _submit_gpu(campaign_root)
                campaign["gpu_job_id"] = job_id
                save_campaign(campaign_root, campaign)
                flow.emit("gpu_unpark_request", reason="tail_flush")
                while queue.pending_flash_count() > 0:
                    time.sleep(cfg.poll_sec)

            if _job_active(campaign.get("gpu_job_id")):
                _scancel(campaign.get("gpu_job_id"), campaign=campaign, endpoint_file=paths["endpoint"])

            campaign = load_campaign(campaign_root)
            campaign["campaign_status"] = "complete"
            campaign["gpu_mode"] = "stopped"
            campaign["completed_at"] = datetime.now(timezone.utc).isoformat()
            _clear_park_pending(campaign)
            save_campaign(campaign_root, campaign)
            write_matrix(campaign_root, queue, campaign)
            flow.emit("campaign_complete")
            logging.info("campaign complete: %s", campaign_root)
            return 0

        unpark_reason = should_unpark(queue, cfg, campaign)
        if unpark_reason and not _gpu_borrowed(campaign, paths["endpoint"]):
            flow.emit("batch_unpark_trigger", reason=unpark_reason, pending=queue.pending_flash_count())
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
                    campaign["parked_flash_since"] = None
                    _clear_park_pending(campaign)
                    save_campaign(campaign_root, campaign)
                    flow.emit("gpu_up", reason=unpark_reason)
                    break
                time.sleep(5)
            else:
                logging.warning("GPU endpoint not healthy after unpark")

        elif gpu_mode == "pending_unpark":
            if _endpoint_healthy(paths["endpoint"]):
                campaign = load_campaign(campaign_root)
                campaign["gpu_mode"] = "up"
                campaign["parked_flash_since"] = None
                _clear_park_pending(campaign)
                save_campaign(campaign_root, campaign)
                flow.emit("gpu_up", reason="pending_unpark_recovery")

        if gpu_parking_enabled(campaign, cfg) and not _gpu_borrowed(campaign, paths["endpoint"]):
            if gpu_must_stay_up(queue, campaign_root, campaign) and campaign.get("park_pending_at"):
                _clear_park_pending(campaign)
                save_campaign(campaign_root, campaign)

            park_reason = evaluate_park_request(queue, campaign, cfg, campaign_root)
            if park_reason and _job_active(campaign.get("gpu_job_id")):
                if not campaign.get("park_pending_at"):
                    campaign["park_pending_at"] = time.time()
                    campaign["park_pending_reason"] = park_reason
                    save_campaign(campaign_root, campaign)
                    flow.emit("gpu_park_pending", reason=park_reason, grace_s=cfg.park_grace_s)
                else:
                    ready, hard_reason = can_hard_park(queue, campaign, cfg, campaign_root)
                    if ready and hard_reason:
                        job_id = campaign.get("gpu_job_id")
                        flow.emit("gpu_parked", reason=hard_reason)
                        campaign["gpu_mode"] = "parked"
                        if not campaign.get("parked_flash_since") and queue.pending_flash_count() > 0:
                            campaign["parked_flash_since"] = time.time()
                        _clear_park_pending(campaign)
                        save_campaign(campaign_root, campaign)
                        _scancel(job_id, campaign=campaign, endpoint_file=paths["endpoint"])
            elif campaign.get("park_pending_at"):
                _clear_park_pending(campaign)
                save_campaign(campaign_root, campaign)

            if gpu_mode == "parked" and queue.pending_flash_count() > 0 and not campaign.get("parked_flash_since"):
                campaign["parked_flash_since"] = time.time()
                save_campaign(campaign_root, campaign)

        time.sleep(cfg.coordinator_poll_sec)


if __name__ == "__main__":
    raise SystemExit(main())
