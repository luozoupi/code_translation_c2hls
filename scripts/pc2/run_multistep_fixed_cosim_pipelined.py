#!/usr/bin/env python3
"""PC2 pipelined multistep on ``benchmarks_cosim/`` — overlaps LLM codegen and Vitis csynth."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "pc2"))

from c2hls_paths import configure_site
from multistep_fixed_cosim_lib import (
    PILOT_BENCHES,
    STAMP_ENV,
    VARIANTS,
    VARIANT_ORDER,
    MultistepFixedCosimVariant,
    configure_fixed_cosim_multistep_env,
    list_cosim_benches,
    resolve_cosim_benches,
    variant_env_snapshot,
    verify_variant_skills,
)
from multistep_pipelined_bench import execute_job
from multistep_pipelined_queue import MultistepPipelinedQueue
from run_multistep_fixed_cosim_batch import (
    _cell_dir,
    _compact_summary,
    _load_existing_result,
    _matrix_row,
    model_cell_tag,
)


def _split_csv(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _resolve_synth_workers(explicit: int | None = None) -> int:
    if explicit is not None:
        raw = explicit
    else:
        raw = int(os.getenv("C2HLS_PIPELINED_SYNTH_WORKERS", "4"))
    return max(1, min(int(raw), 8))


def _worker_loop(
    *,
    kind: str,
    variant: MultistepFixedCosimVariant,
    queue: MultistepPipelinedQueue,
    bench_map: dict[str, Path],
    cell_root: Path,
    model_id: str,
    turns: int,
    stop_event: threading.Event,
    poll_sec: float,
    worker_id: str,
) -> None:
    model_tag = model_cell_tag(model_id)
    while not stop_event.is_set():
        if queue.all_benches_terminal(variant.key) and queue.pending_count(variant=variant.key) == 0:
            return
        job = queue.claim(kind=kind, variant=variant.key, worker_id=worker_id)
        if job is None:
            time.sleep(poll_sec)
            continue
        cell = _cell_dir(cell_root, job.bench, model_tag, variant)
        cell.mkdir(parents=True, exist_ok=True)
        try:
            execute_job(
                job=job,
                queue=queue,
                bench_dir=bench_map[job.bench],
                cell_dir=cell,
                variant_key=variant.key,
                model_id=model_id,
                turns=turns,
            )
            queue.complete(job.id)
        except Exception as exc:
            queue.complete(job.id, error=str(exc))
        if queue.all_benches_terminal(variant.key) and queue.pending_count(variant=variant.key) == 0:
            return


def run_variant_pipelined(
    variant: MultistepFixedCosimVariant,
    *,
    stamp: str,
    out_root: Path | None,
    model_id: str,
    benches: list[tuple[str, Path]],
    dry_run: bool,
    poll_sec: float,
    synth_workers: int,
) -> int:
    check = verify_variant_skills(variant)
    if not check.get("ok"):
        raise SystemExit(f"variant preflight failed: {check['errors']}")

    out = out_root or Path(
        os.getenv(variant.out_env) or REPO / "artifacts" / "pc2" / f"{variant.artifact_prefix}_{stamp}"
    )
    model_tag = model_cell_tag(model_id)
    snap = variant_env_snapshot(variant)
    queue_path = out / "pipelined" / "queue.db"

    plan = {
        "matrix_family": snap["matrix_family"],
        "corpus": snap["corpus"],
        "runner": "pipelined",
        "strategy": "static",
        "record_flow": True,
        "variant": variant.key,
        "stamp": stamp,
        "out_root": str(out),
        "model": model_id,
        "benches": [name for name, _ in benches],
        "queue_db": str(queue_path),
        "synth_workers": synth_workers,
        "origin_meta": snap.get("origin_meta"),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "manifest.json").write_text(json.dumps(plan, indent=2) + "\n")

    print(f"variant={variant.key} multistep pipelined benches={len(benches)} synth_workers={synth_workers}")
    print(f"queue={queue_path} out_root={out}")
    for bench, _ in benches:
        print(f"  - {bench} -> {_cell_dir(out, bench, model_tag, variant)}")

    if dry_run:
        print("dry-run ok")
        return 0

    configure_fixed_cosim_multistep_env(variant)
    turns = int(os.getenv("C2HLS_TURNS", "4"))
    queue = MultistepPipelinedQueue(queue_path)
    bench_map = {name: path for name, path in benches}

    rows: list[dict[str, Any]] = []
    if (out / "matrix.json").exists():
        rows = json.loads((out / "matrix.json").read_text(encoding="utf-8"))

    for bench, _ in benches:
        cell = _cell_dir(out, bench, model_tag, variant)
        result_json = cell / f"{bench}_multistep_results.json"
        existing = _load_existing_result(result_json, bench)
        if existing is not None:
            print(f"SKIP {bench} (existing)", flush=True)
            queue.set_bench_status(variant.key, bench, "done")
            rows.append(
                _matrix_row(
                    bench=bench,
                    model_id=model_id,
                    variant=variant,
                    result=existing,
                    status="ok",
                    elapsed=0.0,
                    cell=cell,
                    error="",
                )
            )
            continue
        queue.seed_bench(variant.key, bench)

    stop_event = threading.Event()
    threads = [
        threading.Thread(
            target=_worker_loop,
            name=f"codegen-{variant.key}",
            kwargs={
                "kind": "codegen",
                "variant": variant,
                "queue": queue,
                "bench_map": bench_map,
                "cell_root": out,
                "model_id": model_id,
                "turns": turns,
                "stop_event": stop_event,
                "poll_sec": poll_sec,
                "worker_id": f"codegen-{variant.key}",
            },
            daemon=True,
        ),
    ]
    for idx in range(synth_workers):
        threads.append(
            threading.Thread(
                target=_worker_loop,
                name=f"synth-{variant.key}-{idx}",
                kwargs={
                    "kind": "synth",
                    "variant": variant,
                    "queue": queue,
                    "bench_map": bench_map,
                    "cell_root": out,
                    "model_id": model_id,
                    "turns": turns,
                    "stop_event": stop_event,
                    "poll_sec": poll_sec,
                    "worker_id": f"synth-{variant.key}-{idx}",
                },
                daemon=True,
            )
        )
    t0 = time.time()
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    stop_event.set()

    for bench, _ in benches:
        cell = _cell_dir(out, bench, model_tag, variant)
        result_json = cell / f"{bench}_multistep_results.json"
        if not result_json.is_file():
            continue
        if any(r.get("bench") == bench for r in rows):
            continue
        try:
            result = json.loads(result_json.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        rows.append(
            _matrix_row(
                bench=bench,
                model_id=model_id,
                variant=variant,
                result=result,
                status="ok" if result.get("success") else "fail",
                elapsed=round(time.time() - t0, 1),
                cell=cell,
                error=result.get("error") or "",
            )
        )
        print(f"DONE {bench} success={result.get('success')}", flush=True)

    (out / "matrix.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="PC2 pipelined multistep fixed cosim (aav_n)")
    parser.add_argument("--pc2", action="store_true", required=True)
    parser.add_argument("--variant", type=str, default="aav_n")
    parser.add_argument("--list-variants", action="store_true")
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--benches", type=str, default="")
    parser.add_argument("--out-root", type=str, default="")
    parser.add_argument("--stamp", type=str, default="")
    parser.add_argument("--model", type=str, default="")
    parser.add_argument("--poll-sec", type=float, default=float(os.getenv("C2HLS_PIPELINED_POLL_SEC", "2")))
    parser.add_argument("--synth-workers", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.list_variants:
        for key in VARIANT_ORDER:
            print(f"{key:10s} {VARIANTS[key].label}")
        return 0

    if args.variant not in VARIANTS:
        parser.error(f"--variant required; choose from: {', '.join(VARIANT_ORDER)}")

    os.environ["C2HLS_SITE"] = "pc2"
    configure_site()

    variant = VARIANTS[args.variant]
    stamp = args.stamp or os.getenv(STAMP_ENV) or datetime.now().strftime("%Y%m%d_%H%M%S")
    if not stamp.endswith("_pipelined") and os.getenv("C2HLS_PIPELINED_STAMP_SUFFIX", "1") == "1":
        stamp = f"{stamp}_pipelined"
    model_id = args.model or os.getenv("C2HLS_MODEL", "mistralai/Devstral-2-123B-Instruct-2512")
    if args.pilot:
        requested = list(PILOT_BENCHES)
    elif args.benches:
        requested = _split_csv(args.benches)
    else:
        requested = list_cosim_benches()
    benches = resolve_cosim_benches(requested)
    out_root = Path(args.out_root) if args.out_root else None
    synth_workers = _resolve_synth_workers(args.synth_workers)

    return run_variant_pipelined(
        variant,
        stamp=stamp,
        out_root=out_root,
        model_id=model_id,
        benches=benches,
        dry_run=args.dry_run,
        poll_sec=args.poll_sec,
        synth_workers=synth_workers,
    )


if __name__ == "__main__":
    raise SystemExit(main())
