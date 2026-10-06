#!/usr/bin/env python3
"""Rank autosa_mm stage-box replicates and cosim until first pass.

Walks the lowest-latency complete replicate of a (family, skill, floor, dse)
box, then the next-best on fail. Writes only to a sibling tree — never into
autosa_mm_variant_sweep_20260918.

Usage:
  python scripts/pc2/autosa_mm_stage_ranked_cosim.py \\
    --sweep-root artifacts/pc2/autosa_mm_variant_sweep_20260918 \\
    --out artifacts/pc2/autosa_mm_stage_ranked_cosim_20260921 --list
  python scripts/pc2/autosa_mm_stage_ranked_cosim.py \\
    --sweep-root ... --out ... --bucket flash_ns_f0
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from autosa_mm_variant_sweep import (  # noqa: E402
    find_variant_cell,
    load_cells,
)
from harvest_autosa_mm_variant_sweep import harvest_sweep  # noqa: E402
from scripts.pc2.flash_cosim_lib import (  # noqa: E402
    CosimCell,
    make_cell_id,
    run_cell_cosim,
)

FROZEN_WRITE_MARKER = "autosa_mm_variant_sweep_20260918"
COSIM_FAMILIES = frozenset({"flash", "oneshot", "dse_v1", "dse_v2", "stream", "enf"})
FAMILY_STAGE = {
    "flash": "flash",
    "oneshot": "oneshot",
    "dse_v1": "dse",
    "dse_v2": "dse",
    "stream": "stream",
    "enf": "enf",
}
KERNEL_CANDIDATES = {
    "flash": ("autosa_mm_flash_opt.cpp", "autosa_mm_selected.cpp"),
    "oneshot": ("autosa_mm_flash_opt.cpp", "autosa_mm_selected.cpp"),
    "dse_v1": ("autosa_mm_dse.cpp", "autosa_mm_selected.cpp"),
    "dse_v2": ("autosa_mm_selected.cpp",),
    "stream": ("autosa_mm_stream.cpp", "autosa_mm_selected.cpp"),
    "enf": ("autosa_mm_selected.cpp",),
}
DEFAULT_SOURCE_BENCH = REPO / "related_work" / "benchmarks" / "autosa_ready" / "autosa_mm"
BENCH = "autosa_mm"


def refuse_frozen_write(path: Path | str) -> None:
    text = str(path)
    if FROZEN_WRITE_MARKER in text:
        raise ValueError(
            f"refusing to write frozen tree matching {FROZEN_WRITE_MARKER!r}: {text}"
        )


def stage_for_family(family: str) -> str:
    try:
        return FAMILY_STAGE[family]
    except KeyError as exc:
        raise ValueError(f"unknown family for stage ranked cosim: {family}") from exc


def bucket_key(cell: dict[str, Any]) -> tuple[Any, ...]:
    return (
        cell.get("family") or "",
        cell.get("skill") or "",
        int(cell.get("floor") or 0),
        cell.get("dse") or "",
    )


def bucket_id(cell: dict[str, Any]) -> str:
    cid = str(cell.get("cell_id") or cell.get("bucket_id") or "")
    return re.sub(r"_r\d+$", "", cid)


def resolve_kernel_cpp(campaign_root: Path, family: str) -> Path | None:
    root = Path(campaign_root)
    names = KERNEL_CANDIDATES.get(family) or ("autosa_mm_selected.cpp",)
    for name in names:
        for pattern in (
            name,
            f"variants/*/{BENCH}/*/{name}",
            f"variants/*/*/{name}",
        ):
            hits = sorted(p for p in root.glob(pattern) if p.is_file())
            if hits:
                return hits[0]
    cell = find_variant_cell(root)
    search = cell if cell is not None else root
    for name in names:
        path = search / name
        if path.is_file():
            return path
        hits = sorted(p for p in search.glob(name) if p.is_file())
        if hits:
            return hits[0]
    return None


def _as_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _rank_tuple(cand: dict[str, Any]) -> tuple[Any, ...]:
    lat = cand["latency_cycles"]
    dsp = cand.get("dsp")
    dsp_key = dsp if isinstance(dsp, (int, float)) else 10**18
    return (lat, dsp_key, str(cand.get("cell_id") or ""))


def build_buckets(
    cells: list[dict[str, Any]],
    harvest: list[dict[str, Any]],
    *,
    require_kernel: bool = True,
) -> list[dict[str, Any]]:
    harvest_by_root_stage: dict[tuple[str, str], dict[str, Any]] = {}
    harvest_by_flags: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in harvest:
        family = row.get("family") or ""
        stage = row.get("stage") or ""
        if family not in COSIM_FAMILIES:
            continue
        if stage != stage_for_family(family):
            continue
        lat = _as_int(row.get("latency_cycles"))
        if lat is None:
            continue
        root = str(row.get("campaign_root") or "")
        if root:
            harvest_by_root_stage[(root, stage)] = row
        harvest_by_flags[
            (
                family,
                row.get("skill") or "",
                int(row.get("floor") or 0),
                row.get("dse") or "",
                int(row.get("stream") or 0),
                int(row.get("enf") or 0),
                int(row.get("rep") or 0),
            )
        ] = row

    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    meta: dict[tuple[Any, ...], dict[str, Any]] = {}
    for cell in cells:
        family = cell.get("family") or ""
        if family not in COSIM_FAMILIES:
            continue
        stage = stage_for_family(family)
        root = str(cell.get("campaign_root") or "")
        row = harvest_by_root_stage.get((root, stage))
        if row is None:
            row = harvest_by_flags.get(
                (
                    family,
                    cell.get("skill") or "",
                    int(cell.get("floor") or 0),
                    cell.get("dse") or "",
                    int(cell.get("stream") or 0),
                    int(cell.get("enf") or 0),
                    int(cell.get("rep") or 0),
                )
            )
        if row is None:
            continue
        lat = _as_int(row.get("latency_cycles"))
        if lat is None:
            continue
        code_path = None
        if require_kernel:
            if not root:
                continue
            kernel = resolve_kernel_cpp(Path(root), family)
            if kernel is None:
                continue
            code_path = str(kernel)
        cand = {
            "cell_id": cell.get("cell_id"),
            "family": family,
            "skill": cell.get("skill") or "",
            "floor": int(cell.get("floor") or 0),
            "dse": cell.get("dse") or "",
            "rep": int(cell.get("rep") or 0),
            "campaign_root": root,
            "latency_cycles": lat,
            "dsp": _as_int(row.get("dsp")),
            "stage": stage,
            "code_path": code_path,
        }
        key = bucket_key(cell)
        grouped.setdefault(key, []).append(cand)
        meta.setdefault(
            key,
            {
                "bucket_id": bucket_id(cell),
                "family": family,
                "skill": cell.get("skill") or "",
                "floor": int(cell.get("floor") or 0),
                "dse": cell.get("dse") or "",
            },
        )

    buckets: list[dict[str, Any]] = []
    for key, cands in grouped.items():
        ranked = sorted(cands, key=_rank_tuple)
        for idx, cand in enumerate(ranked, start=1):
            cand["rank"] = idx
        info = dict(meta[key])
        info["candidates"] = ranked
        buckets.append(info)
    buckets.sort(key=lambda b: b["bucket_id"])
    return buckets


def load_harvest_csv(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def load_sweep_buckets(
    sweep_root: Path,
    *,
    require_kernel: bool = True,
) -> list[dict[str, Any]]:
    root = Path(sweep_root)
    cells = load_cells(root / "cells.jsonl")
    harvest = load_harvest_csv(root / "sweep_qor.csv")
    if not harvest:
        try:
            harvest = harvest_sweep(root)
        except OSError:
            harvest = []
    return build_buckets(cells, harvest, require_kernel=require_kernel)


def stage_autosa_mm_bench(
    out_root: Path,
    source: Path | None = None,
) -> Path:
    refuse_frozen_write(out_root)
    src = Path(source) if source is not None else DEFAULT_SOURCE_BENCH
    dest = Path(out_root) / "bench" / BENCH
    dest.mkdir(parents=True, exist_ok=True)
    for name in ("kernel.h", "testbench.cpp", "plain.cpp"):
        src_file = src / name
        if src_file.is_file():
            shutil.copy2(src_file, dest / name)
    meta_path = src / "metadata.json"
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    else:
        meta = {"benchmark": BENCH, "hls_top": "autosa_mm"}
    meta["supports_cosim"] = True
    meta["supports_csim"] = True
    meta["hls_top"] = meta.get("hls_top") or "autosa_mm"
    meta["translated_hls_top"] = meta.get("translated_hls_top") or "autosa_mm"
    meta["kernel_top"] = meta.get("kernel_top") or "autosa_mm"
    meta["header_file"] = meta.get("header_file") or "kernel.h"
    meta["testbench_file"] = "testbench.cpp"
    meta["cosim_testbench_file"] = "testbench.cpp"
    meta["target_part"] = meta.get("target_part") or meta.get("part") or "xcu280-fsvh2892-2L-e"
    meta["part"] = meta.get("part") or meta["target_part"]
    meta["clock_ns"] = float(meta.get("clock_ns") or meta.get("target_clock_ns") or 3.33)
    dest.joinpath("metadata.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    return dest


def write_bucket_ranking(out_root: Path, bucket: dict[str, Any]) -> Path:
    refuse_frozen_write(out_root)
    bucket_dir = Path(out_root) / "buckets" / bucket["bucket_id"]
    bucket_dir.mkdir(parents=True, exist_ok=True)
    path = bucket_dir / "ranking.json"
    payload = {
        "schema": "autosa_mm_stage_ranked_ranking_v1",
        "bucket_id": bucket["bucket_id"],
        "family": bucket.get("family"),
        "skill": bucket.get("skill"),
        "floor": bucket.get("floor"),
        "dse": bucket.get("dse"),
        "n_candidates": len(bucket.get("candidates") or []),
        "candidates": bucket.get("candidates") or [],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def _bucket_paths(out_root: Path, bucket_id_value: str) -> dict[str, Path]:
    bucket_dir = Path(out_root) / "buckets" / bucket_id_value
    return {
        "dir": bucket_dir,
        "ranking": bucket_dir / "ranking.json",
        "result": bucket_dir / "result.json",
        "attempts": bucket_dir / "attempts.jsonl",
        "run_root": bucket_dir / "cosim_runs",
    }


def _build_cosim_cell(
    *,
    out_root: Path,
    bucket: dict[str, Any],
    candidate: dict[str, Any],
    index: int,
) -> CosimCell:
    setup_tag = f"{bucket['bucket_id']}__{candidate.get('cell_id') or index}"
    art_base = Path(out_root).name
    return CosimCell(
        index=index,
        cell_id=make_cell_id(art_base, BENCH, setup_tag),
        artifact_dir=str(out_root),
        artifact_basename=art_base,
        artifact_stamp="",
        matrix_family="autosa_mm_stage_ranked",
        bench=BENCH,
        setup_tag=setup_tag,
        variant=str(bucket.get("family") or ""),
        mode="stage_ranked",
        model=str(candidate.get("cell_id") or ""),
        curation_focus="",
        skills_json="",
        cell_dir=str(candidate.get("campaign_root") or ""),
        final_cpp=str(Path(str(candidate["code_path"])).resolve()),
        kernel_source="ranked",
        source_matrix_status="ranked",
        supports_cosim=True,
    )


def run_bucket_cosim(
    *,
    bucket: dict[str, Any],
    out_root: Path,
    force: bool = False,
    dry_run: bool = False,
    cosim_fn: Optional[Callable[..., dict[str, Any]]] = None,
) -> dict[str, Any]:
    out_root = Path(out_root)
    refuse_frozen_write(out_root)
    bid = str(bucket["bucket_id"])
    paths = _bucket_paths(out_root, bid)
    if paths["result"].is_file() and not force and not dry_run:
        try:
            existing = json.loads(paths["result"].read_text(encoding="utf-8"))
            if isinstance(existing, dict) and existing.get("status") in {
                "pass",
                "fail",
                "skipped",
            }:
                return existing
        except (OSError, json.JSONDecodeError):
            pass

    paths["dir"].mkdir(parents=True, exist_ok=True)
    write_bucket_ranking(out_root, bucket)
    ranked = list(bucket.get("candidates") or [])
    attempts: list[dict[str, Any]] = []
    stamp = datetime.now(timezone.utc).isoformat()
    if not ranked:
        result = {
            "schema": "autosa_mm_stage_ranked_cosim_result_v1",
            "bucket_id": bid,
            "status": "fail",
            "passed": False,
            "reason": "no_ranked_candidates",
            "attempts": attempts,
            "finished_at": stamp,
        }
        paths["result"].write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        return result

    os.environ.setdefault("C2HLS_COSIM_XELAB_MT_OFF", "1")
    os.environ.setdefault("C2HLS_FLASH_COSIM_FULL_SIZE", "1")
    os.environ.setdefault(
        "C2HLS_COSIM_BENCHMARKS_ROOT", str(out_root / "bench")
    )

    runner = cosim_fn or run_cell_cosim
    run_root = paths["run_root"]
    run_root.mkdir(parents=True, exist_ok=True)
    winner: Optional[dict[str, Any]] = None
    paths["attempts"].write_text("", encoding="utf-8")
    for idx, cand in enumerate(ranked):
        code_path = Path(str(cand.get("code_path") or ""))
        if not code_path.is_file() and cand.get("campaign_root"):
            resolved = resolve_kernel_cpp(
                Path(str(cand["campaign_root"])), str(bucket.get("family") or "")
            )
            if resolved is not None:
                code_path = resolved
                cand = dict(cand)
                cand["code_path"] = str(code_path)
        attempt: dict[str, Any] = {
            "rank": int(cand.get("rank") or idx + 1),
            "id": cand.get("cell_id"),
            "latency_cycles": cand.get("latency_cycles"),
            "dsp": cand.get("dsp"),
            "code_path": str(code_path),
            "campaign_root": cand.get("campaign_root"),
        }
        if not code_path.is_file():
            attempt["status"] = "fail"
            attempt["error"] = "missing_code_path"
            attempts.append(attempt)
            with paths["attempts"].open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(attempt) + "\n")
            continue
        cell = _build_cosim_cell(
            out_root=out_root,
            bucket=bucket,
            candidate=cand,
            index=idx,
        )
        cosim = runner(cell, run_root, force=True, dry_run=dry_run)
        attempt["status"] = cosim.get("status")
        attempt["passed"] = bool(cosim.get("passed"))
        attempt["error"] = cosim.get("error") or ""
        attempt["kernel_runtime_cycles"] = cosim.get("kernel_runtime_cycles")
        attempt["cosim_result_cell_id"] = cell.cell_id
        attempts.append(attempt)
        with paths["attempts"].open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(attempt) + "\n")
        if dry_run:
            winner = {"id": cand.get("cell_id"), "attempt": attempt}
            break
        if cosim.get("status") == "pass" or cosim.get("passed") is True:
            winner = {"id": cand.get("cell_id"), "attempt": attempt}
            break

    finished = datetime.now(timezone.utc).isoformat()
    if winner is not None:
        result = {
            "schema": "autosa_mm_stage_ranked_cosim_result_v1",
            "bucket_id": bid,
            "family": bucket.get("family"),
            "status": "pass" if not dry_run else "dry_run",
            "passed": not dry_run,
            "winner_id": winner["id"],
            "winner": winner["attempt"],
            "attempts": attempts,
            "run_root": str(run_root),
            "finished_at": finished,
        }
    else:
        result = {
            "schema": "autosa_mm_stage_ranked_cosim_result_v1",
            "bucket_id": bid,
            "family": bucket.get("family"),
            "status": "fail",
            "passed": False,
            "reason": "all_candidates_failed",
            "attempts": attempts,
            "run_root": str(run_root),
            "finished_at": finished,
        }
    paths["result"].write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def candidate_result_path(out_root: Path, bucket_id: str, cell_id: str) -> Path:
    return Path(out_root) / "buckets" / bucket_id / "candidates" / cell_id / "result.json"


def _attempted_cell_ids(out_root: Path, bucket: dict[str, Any]) -> set[str]:
    ids: set[str] = set()
    bid = str(bucket.get("bucket_id") or "")
    result_path = _bucket_paths(out_root, bid)["result"]
    if result_path.is_file():
        try:
            doc = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            doc = {}
        for attempt in doc.get("attempts") or []:
            cid = attempt.get("id")
            status = attempt.get("status")
            if cid and status not in {None, "", "pending"}:
                ids.add(str(cid))
    cand_root = Path(out_root) / "buckets" / bid / "candidates"
    if cand_root.is_dir():
        for path in cand_root.glob("*/result.json"):
            ids.add(path.parent.name)
    return ids


def remaining_candidates(
    buckets: list[dict[str, Any]],
    out_root: Path,
) -> list[dict[str, Any]]:
    left: list[dict[str, Any]] = []
    out_root = Path(out_root)
    for bucket in buckets:
        done = _attempted_cell_ids(out_root, bucket)
        for cand in bucket.get("candidates") or []:
            cid = cand.get("cell_id")
            if not cid or cid in done:
                continue
            row = dict(cand)
            row["bucket_id"] = bucket["bucket_id"]
            row["family"] = bucket.get("family")
            left.append(row)
    return left


def _candidate_attempt_map(
    bucket: dict[str, Any],
    out_root: Path,
) -> dict[str, dict[str, Any]]:
    bid = str(bucket["bucket_id"])
    by_id: dict[str, dict[str, Any]] = {}
    result_path = _bucket_paths(out_root, bid)["result"]
    if result_path.is_file():
        try:
            doc = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            doc = {}
        for attempt in doc.get("attempts") or []:
            cid = attempt.get("id")
            if cid:
                by_id[str(cid)] = attempt
    cand_root = Path(out_root) / "buckets" / bid / "candidates"
    if cand_root.is_dir():
        for path in cand_root.glob("*/result.json"):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            attempt = payload.get("attempt") if isinstance(payload, dict) else None
            if isinstance(attempt, dict) and attempt.get("id"):
                by_id[str(attempt["id"])] = attempt
            elif payload.get("id"):
                by_id[str(payload["id"])] = payload
    return by_id


def reduce_bucket_result(
    *,
    bucket: dict[str, Any],
    out_root: Path,
) -> dict[str, Any]:
    out_root = Path(out_root)
    refuse_frozen_write(out_root)
    bid = str(bucket["bucket_id"])
    ranked = list(bucket.get("candidates") or [])
    by_id = _candidate_attempt_map(bucket, out_root)
    attempts: list[dict[str, Any]] = []
    pending = False
    winner: Optional[dict[str, Any]] = None
    for idx, cand in enumerate(ranked):
        cid = str(cand.get("cell_id") or "")
        attempt = by_id.get(cid)
        if attempt is None:
            pending = True
            continue
        attempts.append(attempt)
        passed = bool(attempt.get("passed")) or attempt.get("status") in {"pass", "ok"}
        if winner is None and passed:
            winner = {"id": cid, "attempt": attempt}
    finished = datetime.now(timezone.utc).isoformat()
    run_root = str(_bucket_paths(out_root, bid)["run_root"])
    if winner is not None:
        result = {
            "schema": "autosa_mm_stage_ranked_cosim_result_v1",
            "bucket_id": bid,
            "family": bucket.get("family"),
            "status": "pass",
            "passed": True,
            "winner_id": winner["id"],
            "winner": winner["attempt"],
            "attempts": attempts,
            "pending": pending,
            "run_root": run_root,
            "finished_at": finished,
        }
    elif pending:
        result = {
            "schema": "autosa_mm_stage_ranked_cosim_result_v1",
            "bucket_id": bid,
            "family": bucket.get("family"),
            "status": "running",
            "passed": False,
            "reason": "candidates_pending",
            "attempts": attempts,
            "pending": True,
            "run_root": run_root,
            "finished_at": finished,
        }
    else:
        result = {
            "schema": "autosa_mm_stage_ranked_cosim_result_v1",
            "bucket_id": bid,
            "family": bucket.get("family"),
            "status": "fail",
            "passed": False,
            "reason": "all_candidates_failed",
            "attempts": attempts,
            "pending": False,
            "run_root": run_root,
            "finished_at": finished,
        }
    paths = _bucket_paths(out_root, bid)
    paths["dir"].mkdir(parents=True, exist_ok=True)
    paths["result"].write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def run_candidate_cosim(
    *,
    bucket: dict[str, Any],
    cell_id: str,
    out_root: Path,
    force: bool = False,
    dry_run: bool = False,
    cosim_fn: Optional[Callable[..., dict[str, Any]]] = None,
) -> dict[str, Any]:
    out_root = Path(out_root)
    refuse_frozen_write(out_root)
    bid = str(bucket["bucket_id"])
    cand = next(
        (c for c in (bucket.get("candidates") or []) if c.get("cell_id") == cell_id),
        None,
    )
    if cand is None:
        raise KeyError(f"cell_id {cell_id!r} not in bucket {bid}")
    out_path = candidate_result_path(out_root, bid, cell_id)
    if out_path.is_file() and not force and not dry_run:
        try:
            existing = json.loads(out_path.read_text(encoding="utf-8"))
            if isinstance(existing, dict) and existing.get("attempt"):
                return reduce_bucket_result(bucket=bucket, out_root=out_root)
        except (OSError, json.JSONDecodeError):
            pass

    write_bucket_ranking(out_root, bucket)
    os.environ.setdefault("C2HLS_COSIM_XELAB_MT_OFF", "1")
    os.environ.setdefault("C2HLS_FLASH_COSIM_FULL_SIZE", "1")
    os.environ.setdefault("C2HLS_COSIM_BENCHMARKS_ROOT", str(out_root / "bench"))

    code_path = Path(str(cand.get("code_path") or ""))
    if not code_path.is_file() and cand.get("campaign_root"):
        resolved = resolve_kernel_cpp(
            Path(str(cand["campaign_root"])), str(bucket.get("family") or "")
        )
        if resolved is not None:
            code_path = resolved
            cand = dict(cand)
            cand["code_path"] = str(code_path)
    rank = int(cand.get("rank") or 0)
    if rank <= 0:
        for idx, row in enumerate(bucket.get("candidates") or [], start=1):
            if row.get("cell_id") == cell_id:
                rank = idx
                break
    attempt: dict[str, Any] = {
        "rank": rank,
        "id": cell_id,
        "latency_cycles": cand.get("latency_cycles"),
        "dsp": cand.get("dsp"),
        "code_path": str(code_path),
        "campaign_root": cand.get("campaign_root"),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not code_path.is_file():
        attempt["status"] = "fail"
        attempt["passed"] = False
        attempt["error"] = "missing_code_path"
    else:
        runner = cosim_fn or run_cell_cosim
        run_root = out_path.parent / "cosim_runs"
        run_root.mkdir(parents=True, exist_ok=True)
        idx = int(cand.get("rank") or 1) - 1
        cell = _build_cosim_cell(
            out_root=out_root,
            bucket=bucket,
            candidate=cand,
            index=max(idx, 0),
        )
        cosim = runner(cell, run_root, force=True, dry_run=dry_run)
        attempt["status"] = cosim.get("status")
        attempt["passed"] = bool(cosim.get("passed"))
        attempt["error"] = cosim.get("error") or ""
        attempt["kernel_runtime_cycles"] = cosim.get("kernel_runtime_cycles")
        attempt["cosim_result_cell_id"] = cell.cell_id
    payload = {
        "schema": "autosa_mm_stage_candidate_cosim_result_v1",
        "bucket_id": bid,
        "cell_id": cell_id,
        "status": attempt.get("status"),
        "passed": bool(attempt.get("passed")),
        "attempt": attempt,
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return reduce_bucket_result(bucket=bucket, out_root=out_root)


def write_all_rankings(out_root: Path, buckets: list[dict[str, Any]]) -> Path:
    refuse_frozen_write(out_root)
    summary = []
    for bucket in buckets:
        write_bucket_ranking(out_root, bucket)
        cands = bucket.get("candidates") or []
        summary.append(
            {
                "bucket_id": bucket["bucket_id"],
                "family": bucket.get("family"),
                "skill": bucket.get("skill"),
                "floor": bucket.get("floor"),
                "dse": bucket.get("dse"),
                "n_candidates": len(cands),
                "rank1_id": cands[0]["cell_id"] if cands else None,
                "rank1_latency": cands[0]["latency_cycles"] if cands else None,
            }
        )
    path = Path(out_root) / "rank_summary.json"
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--bucket", default="", help="Run one bucket id")
    parser.add_argument("--candidate", default="", help="Run one ranked replicate cell_id")
    parser.add_argument("--list", action="store_true", help="Print bucket ids and exit")
    parser.add_argument(
        "--list-candidates",
        action="store_true",
        help="Print remaining uncosimed ranked cell ids",
    )
    parser.add_argument(
        "--write-rankings",
        action="store_true",
        help="Write ranking.json for every bucket under --out",
    )
    parser.add_argument("--stage-bench", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--no-kernel",
        action="store_true",
        help="Rank without requiring kernel cpp (debug)",
    )
    args = parser.parse_args()
    refuse_frozen_write(args.out)
    buckets = load_sweep_buckets(
        args.sweep_root, require_kernel=not args.no_kernel
    )
    if args.list:
        for bucket in buckets:
            n = len(bucket.get("candidates") or [])
            rank1 = (bucket["candidates"][0]["cell_id"] if n else "-")
            print(f"{bucket['bucket_id']}\t{n}\t{rank1}")
        print(f"buckets={len(buckets)}")
        return 0
    if args.list_candidates:
        left = remaining_candidates(buckets, args.out)
        for cand in left:
            print(f"{cand['bucket_id']}\t{cand['cell_id']}\t{cand.get('rank') or ''}")
        print(f"remaining={len(left)}")
        return 0
    if args.stage_bench:
        dest = stage_autosa_mm_bench(args.out)
        print(f"staged_bench={dest}")
    if args.write_rankings:
        path = write_all_rankings(args.out, buckets)
        print(f"rank_summary={path} buckets={len(buckets)}")
        if not args.bucket and not args.candidate:
            return 0
    if args.candidate:
        match = next(
            (
                b
                for b in buckets
                if any(c.get("cell_id") == args.candidate for c in (b.get("candidates") or []))
            ),
            None,
        )
        if match is None:
            raise SystemExit(f"unknown candidate {args.candidate!r}")
        result = run_candidate_cosim(
            bucket=match,
            cell_id=args.candidate,
            out_root=args.out,
            force=args.force,
            dry_run=args.dry_run,
        )
        print(json.dumps(result, indent=2))
        status = result.get("status")
        if status in {"pass", "skipped", "dry_run", "running"}:
            return 0
        return 1
    if not args.bucket:
        parser.error("pass --bucket ID, --candidate ID, or --list / --write-rankings")
    match = next((b for b in buckets if b["bucket_id"] == args.bucket), None)
    if match is None:
        raise SystemExit(f"unknown bucket {args.bucket!r}")
    result = run_bucket_cosim(
        bucket=match,
        out_root=args.out,
        force=args.force,
        dry_run=args.dry_run,
    )
    print(json.dumps(result, indent=2))
    status = result.get("status")
    if status in {"pass", "skipped", "dry_run"}:
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
