#!/usr/bin/env python3
"""GPU-node codegen drain for batch_parallel campaigns."""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "pc2"))

from batch_parallel_bench import execute_job
from batch_parallel_config import campaign_paths, campaign_benches, load_campaign, load_config
from batch_parallel_flow import BatchParallelFlow
from batch_parallel_gpu_state import (
    begin_llm_request,
    end_llm_request,
    is_retriable_llm_error,
)
from batch_parallel_queue import BatchParallelQueue
from c2hls_paths import configure_site
from flash_fixed_cosim_lib import VARIANTS, configure_fixed_cosim_flash_env, resolve_cosim_benches
from run_flash_fixed_cosim_batch import _cell_dir, model_cell_tag


def _load_endpoint_env(endpoint_file: Path) -> None:
    import json

    if not endpoint_file.is_file():
        return
    try:
        payload = json.loads(endpoint_file.read_text(encoding="utf-8"))
        url = payload.get("url")
        if url:
            os.environ["OPENAI_BASE_URL"] = str(url)
    except Exception:
        pass


def _endpoint_healthy(endpoint_file: Path) -> bool:
    import json
    import urllib.request

    if not endpoint_file.is_file():
        return False
    try:
        payload = json.loads(endpoint_file.read_text(encoding="utf-8"))
        url = str(payload.get("url", "")).rstrip("/")
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


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="batch_parallel GPU codegen drain")
    parser.add_argument("--campaign-root", required=True)
    args = parser.parse_args()

    configure_site("pc2")
    campaign_root = Path(args.campaign_root)
    paths = campaign_paths(campaign_root)
    _load_endpoint_env(paths["endpoint"])
    cfg = load_config()
    campaign = load_campaign(campaign_root)
    active_variants = campaign.get("active_variants") or [cfg.pilot_variant]
    benches_order = campaign_benches(campaign, cfg)
    model_id = cfg.model
    turns = cfg.turns
    model_tag = model_cell_tag(model_id)
    cell_root = campaign_root / "variants"
    queue = BatchParallelQueue(paths["queue_db"])
    flow = BatchParallelFlow(campaign_root)
    bench_cache: dict[str, Path] = {}
    worker = f"gpu-drain-{os.getpid()}"

    def bench_dir_for(name: str) -> Path | None:
        if name in bench_cache:
            return bench_cache[name]
        try:
            resolved = dict(resolve_cosim_benches([name]))
            bench_cache[name] = resolved[name]
            return bench_cache[name]
        except ValueError:
            return None

    while True:
        if queue.campaign_complete(active_variants):
            return 0
        campaign = load_campaign(campaign_root)
        gpu_mode = str(campaign.get("gpu_mode") or "up")
        if gpu_mode in ("parked", "pending_unpark"):
            time.sleep(cfg.poll_sec)
            continue
        _load_endpoint_env(paths["endpoint"])
        if not _endpoint_healthy(paths["endpoint"]):
            time.sleep(cfg.poll_sec)
            continue

        job = queue.claim(kind="codegen", worker_id=worker)
        if job is None:
            time.sleep(cfg.poll_sec)
            continue

        variant = VARIANTS.get(job.variant)
        if variant is None:
            queue.complete(job.id, error=f"unknown variant {job.variant}")
            continue
        configure_fixed_cosim_flash_env(variant)
        bench_dir = bench_dir_for(job.bench)
        if not bench_dir:
            queue.complete(job.id, error=f"unknown bench {job.bench}")
            continue

        begin_llm_request(
            campaign_root,
            job_id=job.id,
            variant=job.variant,
            bench=job.bench,
            phase=job.phase,
            worker=worker,
        )
        flow.emit(
            "codegen_start",
            scope="gpu",
            variant=job.variant,
            bench=job.bench,
            phase=job.phase,
            pending_codegen=queue.pending_codegen(),
            job_id=job.id,
        )
        cell = _cell_dir(cell_root / job.variant, job.bench, model_tag, variant)
        cell.mkdir(parents=True, exist_ok=True)
        try:
            execute_job(
                job=job,
                queue=queue,
                bench_dir=bench_dir,
                cell_dir=cell,
                variant_key=job.variant,
                model_id=model_id,
                turns=turns,
            )
            queue.complete(job.id)
            flow.emit(
                "codegen_done",
                scope="gpu",
                variant=job.variant,
                bench=job.bench,
                phase=job.phase,
                pending_codegen=queue.pending_codegen(),
                job_id=job.id,
            )
        except Exception as exc:
            if is_retriable_llm_error(exc) and queue.requeue(job.id):
                queue.set_bench_status(job.variant, job.bench, "active")
                flow.emit(
                    "codegen_retry",
                    scope="gpu",
                    variant=job.variant,
                    bench=job.bench,
                    phase=job.phase,
                    job_id=job.id,
                    error=str(exc),
                )
                logging.warning(
                    "retriable GPU/codegen error for %s/%s; requeued job %s: %s",
                    job.variant,
                    job.bench,
                    job.id,
                    exc,
                )
            else:
                queue.complete(job.id, error=str(exc))
                flow.emit(
                    "codegen_failed",
                    scope="gpu",
                    variant=job.variant,
                    bench=job.bench,
                    error=str(exc),
                    job_id=job.id,
                )
        finally:
            end_llm_request(campaign_root, job_id=job.id)


if __name__ == "__main__":
    raise SystemExit(main())
