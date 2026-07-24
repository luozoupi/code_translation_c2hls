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

from batch_parallel_config import campaign_paths, load_campaign, load_config
from batch_parallel_dispatch import (
    campaign_benches_resolved,
    cell_dir_for_job,
    configure_campaign_env,
    resolve_bench_map,
    run_batch_parallel_job,
    seed_kwargs_for_campaign,
    validate_variant,
)
from batch_parallel_flow import BatchParallelFlow
from batch_parallel_gpu_state import (
    begin_llm_request,
    end_llm_request,
    is_retriable_llm_error,
)
from batch_parallel_queue import BatchParallelQueue
from c2hls_paths import configure_site
from deepseek_peak import is_beijing_peak, sleep_hint_sec

PEAK_PAUSE_EVENT_MIN_INTERVAL_SEC = 300.0


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




def _endpoint_model(endpoint_file: Path) -> str:
    import json

    if not endpoint_file.is_file():
        return ""
    try:
        payload = json.loads(endpoint_file.read_text(encoding="utf-8"))
        model = payload.get("model")
        return str(model).strip() if model else ""
    except Exception:
        return ""

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


def _peak_pause_active(campaign: dict) -> bool:
    """Return True if codegen claiming should pause for the Beijing DeepSeek peak window.

    Bypass (either is enough):
      - env C2HLS_DEEPSEEK_SKIP_PEAK=1
      - campaign.json field skip_peak_pause=true
    """
    if os.getenv("C2HLS_DEEPSEEK_SKIP_PEAK", "0") == "1":
        return False
    if campaign.get("skip_peak_pause"):
        return False
    gated = bool(campaign.get("external_llm")) or os.getenv(
        "C2HLS_DEEPSEEK_PEAK_PAUSE", "0"
    ) == "1"
    return gated and is_beijing_peak()


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
    benches_order = campaign_benches_resolved(campaign, cfg)
    # Model selection:
    # - external_llm A/B: BATCH_PARALLEL_EXTERNAL_MODEL / campaign endpoint model
    #   is authoritative (base vs dpo on the same vLLM).
    # - borrowed/local GPU: prefer live endpoint served name so a stale
    #   deepseek-chat env override cannot 404 against Devstral.
    model_id = cfg.model
    ep_model = _endpoint_model(paths["endpoint"])
    ext_model = (
        os.getenv("BATCH_PARALLEL_EXTERNAL_MODEL", "").strip()
        or os.getenv("C2HLS_MODEL", "").strip()
    )
    if campaign.get("external_llm") and ext_model:
        model_id = ext_model
    elif ep_model:
        model_id = ep_model
    turns = cfg.turns
    queue = BatchParallelQueue(paths["queue_db"])
    flow = BatchParallelFlow(campaign_root)
    bench_cache = resolve_bench_map(campaign, cfg, benches_order)
    worker = f"gpu-drain-{os.getpid()}"
    last_peak_pause_emit = 0.0
    logging.info("gpu_drain model_id=%s external_llm=%s", model_id, bool(campaign.get("external_llm")))

    while True:
        if queue.campaign_complete(active_variants):
            return 0
        campaign = load_campaign(campaign_root)
        # Re-resolve model each loop so A/B arms stay on the intended adapter.
        ep_model = _endpoint_model(paths["endpoint"])
        ext_model = (
            os.getenv("BATCH_PARALLEL_EXTERNAL_MODEL", "").strip()
            or os.getenv("C2HLS_MODEL", "").strip()
        )
        if campaign.get("external_llm") and ext_model:
            model_id = ext_model
        elif ep_model:
            model_id = ep_model
        gpu_mode = str(campaign.get("gpu_mode") or "up")
        if gpu_mode in ("parked", "pending_unpark"):
            time.sleep(cfg.poll_sec)
            continue
        _load_endpoint_env(paths["endpoint"])
        if not _endpoint_healthy(paths["endpoint"]):
            time.sleep(cfg.poll_sec)
            continue

        if _peak_pause_active(campaign):
            now = time.time()
            if now - last_peak_pause_emit >= PEAK_PAUSE_EVENT_MIN_INTERVAL_SEC:
                flow.emit(
                    "codegen_peak_pause",
                    scope="gpu",
                    pending_codegen=queue.pending_codegen(),
                )
                last_peak_pause_emit = now
            time.sleep(sleep_hint_sec(max_sleep=max(cfg.poll_sec * 30, 60)) or cfg.poll_sec)
            continue

        job = queue.claim(kind="codegen", worker_id=worker)
        if job is None:
            time.sleep(cfg.poll_sec)
            continue

        if not validate_variant(campaign, job.variant):
            queue.complete(job.id, error=f"unknown variant {job.variant}")
            continue
        configure_campaign_env(campaign, job.variant)
        bench_dir = bench_cache.get(job.bench)
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
        cell = cell_dir_for_job(campaign_root, campaign, job, model_id)
        cell.mkdir(parents=True, exist_ok=True)
        try:
            run_batch_parallel_job(
                job=job,
                queue=queue,
                campaign=campaign,
                bench_dir=bench_dir,
                cell_dir=cell,
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
                queue.set_bench_status(job.variant, job.bench, "failed")
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
