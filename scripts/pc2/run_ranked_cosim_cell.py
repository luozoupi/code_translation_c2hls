#!/usr/bin/env python3
"""Walk flash/dataflow candidate ranking and cosim until first pass or exhausted.

Usage:
  .venv/bin/python scripts/pc2/run_ranked_cosim_cell.py \\
    --cell-dir CELL --bench BENCH --side flash|dataflow \\
    --campaign-root ROOT [--force] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from flash_df_candidate_rank import (  # noqa: E402
    collect_dataflow_side_candidates,
    collect_flash_side_candidates,
    promote_rank1_kernel,
    rank_and_promote,
    rank_candidates,
    write_ranking,
)
from scripts.pc2.flash_cosim_lib import (  # noqa: E402
    CosimCell,
    cosim_benchmarks_root,
    make_cell_id,
    run_cell_cosim,
)


def _load_ranking(cell_dir: Path, bench: str, side: str) -> list[dict[str, Any]]:
    path = cell_dir / f"{bench}_{side}_candidate_ranking.json"
    if path.is_file():
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            doc = {}
        cands = doc.get("candidates") if isinstance(doc, dict) else None
        if isinstance(cands, list) and cands:
            return [c for c in cands if isinstance(c, dict) and c.get("code_path")]
    # Rebuild from artifacts.
    if side == "flash":
        ranked = rank_candidates(collect_flash_side_candidates(cell_dir, bench))
    else:
        ranked = rank_candidates(collect_dataflow_side_candidates(cell_dir, bench))
    write_ranking(cell_dir, bench, side, ranked)
    return ranked


def _supports_cosim(bench: str) -> bool:
    meta_path = cosim_benchmarks_root() / bench / "metadata.json"
    if not meta_path.is_file():
        return False
    try:
        return bool(json.loads(meta_path.read_text(encoding="utf-8")).get("supports_cosim"))
    except (OSError, json.JSONDecodeError):
        return False


def _result_path(cell_dir: Path, bench: str, side: str) -> Path:
    return cell_dir / f"{bench}_{side}_cosim_opt_result.json"


def _build_cell(
    *,
    campaign_root: Path,
    cell_dir: Path,
    bench: str,
    side: str,
    candidate: dict[str, Any],
    index: int,
) -> CosimCell:
    setup_tag = cell_dir.name
    cand_id = str(candidate.get("id") or f"{side}:rank{index}")
    safe_tag = f"{setup_tag}__{cand_id.replace(':', '_')}"
    art_base = campaign_root.name
    return CosimCell(
        index=index,
        cell_id=make_cell_id(art_base, bench, safe_tag),
        artifact_dir=str(campaign_root),
        artifact_basename=art_base,
        artifact_stamp="",
        matrix_family=f"hlsfactory_{side}_ranked",
        bench=bench,
        setup_tag=safe_tag,
        variant=str(candidate.get("variant") or ""),
        mode=side,
        model=setup_tag,
        curation_focus="",
        skills_json="",
        cell_dir=str(cell_dir),
        final_cpp=str(Path(candidate["code_path"]).resolve()),
        kernel_source="ranked",
        source_matrix_status="ranked",
        supports_cosim=_supports_cosim(bench),
    )


def run_ranked_cosim(
    *,
    cell_dir: Path,
    bench: str,
    side: str,
    campaign_root: Path,
    force: bool = False,
    dry_run: bool = False,
    rebuild_rank: bool = False,
) -> dict[str, Any]:
    cell_dir = Path(cell_dir)
    campaign_root = Path(campaign_root)
    out_path = _result_path(cell_dir, bench, side)
    if out_path.is_file() and not force and not dry_run:
        try:
            existing = json.loads(out_path.read_text(encoding="utf-8"))
            if isinstance(existing, dict) and existing.get("status") in {"pass", "fail", "skipped"}:
                return existing
        except (OSError, json.JSONDecodeError):
            pass

    if rebuild_rank:
        ranked = rank_and_promote(cell_dir, bench, side=side)
    else:
        ranked = _load_ranking(cell_dir, bench, side)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_root = (
        campaign_root
        / f"{side}_ranked_cosim"
        / f"{campaign_root.name}_{bench}_{side}_{stamp}"
    )
    run_root.mkdir(parents=True, exist_ok=True)

    attempts: list[dict[str, Any]] = []
    if not ranked:
        result = {
            "schema": "flash_df_ranked_cosim_result_v1",
            "benchmark": bench,
            "side": side,
            "status": "fail",
            "passed": False,
            "reason": "no_ranked_candidates",
            "attempts": attempts,
            "finished_at": datetime.now(timezone.utc).isoformat(),
        }
        out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        return result

    if not _supports_cosim(bench):
        result = {
            "schema": "flash_df_ranked_cosim_result_v1",
            "benchmark": bench,
            "side": side,
            "status": "skipped",
            "passed": False,
            "reason": "benchmark does not support cosim",
            "rank1_id": ranked[0].get("id"),
            "attempts": attempts,
            "finished_at": datetime.now(timezone.utc).isoformat(),
        }
        out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        return result

    os.environ.setdefault("C2HLS_COSIM_XELAB_MT_OFF", "1")
    os.environ.setdefault("C2HLS_FLASH_COSIM_FULL_SIZE", "1")

    winner: Optional[dict[str, Any]] = None
    for idx, cand in enumerate(ranked):
        code_path = Path(str(cand.get("code_path") or ""))
        attempt: dict[str, Any] = {
            "rank": int(cand.get("rank") or idx + 1),
            "id": cand.get("id"),
            "variant": cand.get("variant"),
            "latency_cycles": cand.get("latency_cycles"),
            "code_path": str(code_path),
        }
        if not code_path.is_file():
            attempt["status"] = "fail"
            attempt["error"] = "missing_code_path"
            attempts.append(attempt)
            continue
        cell = _build_cell(
            campaign_root=campaign_root,
            cell_dir=cell_dir,
            bench=bench,
            side=side,
            candidate=cand,
            index=idx,
        )
        cosim = run_cell_cosim(cell, run_root, force=True, dry_run=dry_run)
        attempt["status"] = cosim.get("status")
        attempt["passed"] = bool(cosim.get("passed"))
        attempt["error"] = cosim.get("error") or ""
        attempt["kernel_runtime_cycles"] = cosim.get("kernel_runtime_cycles")
        attempt["cosim_result_cell_id"] = cell.cell_id
        attempts.append(attempt)
        if dry_run:
            # Dry-run: treat first candidate as staged success for wiring checks.
            winner = {"id": cand.get("id"), "attempt": attempt}
            break
        if cosim.get("status") == "pass" or cosim.get("passed") is True:
            winner = {"id": cand.get("id"), "attempt": attempt}
            break

    if winner is not None:
        # Cosim-validated winner becomes the canonical selected pointer (csynth
        # rank-1 may have been a TB-broken lat_opt). Keep flash_seed untouched.
        if not dry_run:
            win_cand = next((c for c in ranked if c.get("id") == winner["id"]), None)
            if win_cand is not None:
                promote_rank1_kernel(cell_dir, bench, [win_cand], side=side)
        result = {
            "schema": "flash_df_ranked_cosim_result_v1",
            "benchmark": bench,
            "side": side,
            "status": "pass" if not dry_run else "dry_run",
            "passed": not dry_run,
            "winner_id": winner["id"],
            "winner": winner["attempt"],
            "attempts": attempts,
            "run_root": str(run_root),
            "finished_at": datetime.now(timezone.utc).isoformat(),
        }
    else:
        result = {
            "schema": "flash_df_ranked_cosim_result_v1",
            "benchmark": bench,
            "side": side,
            "status": "fail",
            "passed": False,
            "reason": "all_candidates_failed",
            "attempts": attempts,
            "run_root": str(run_root),
            "finished_at": datetime.now(timezone.utc).isoformat(),
        }
    out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell-dir", required=True)
    parser.add_argument("--bench", required=True)
    parser.add_argument("--side", required=True, choices=("flash", "dataflow"))
    parser.add_argument("--campaign-root", required=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--rebuild-rank",
        action="store_true",
        help="Re-collect/rank/promote before cosim",
    )
    args = parser.parse_args()

    result = run_ranked_cosim(
        cell_dir=Path(args.cell_dir),
        bench=args.bench,
        side=args.side,
        campaign_root=Path(args.campaign_root),
        force=args.force,
        dry_run=args.dry_run,
        rebuild_rank=args.rebuild_rank,
    )
    print(json.dumps(result, indent=2))
    status = result.get("status")
    if status in {"pass", "skipped", "dry_run"}:
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
