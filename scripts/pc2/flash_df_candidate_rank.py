"""Flash / dataflow candidate ranking for async ranked cosim.

Pool = {seed} ∪ {each accepted lat_opt round} (plus final ``latency_opt`` only
when it is not a duplicate of a round). Rank by ascending *worst-case* csynth
latency (min of max: ``latency_cycles_worst``). Cosim walks the full ranked
list (best → next → …).
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any, Optional


def _load_json_dict(path: Path) -> Optional[dict[str, Any]]:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def latency_cycles_from_report(report: Optional[dict[str, Any]]) -> Optional[float]:
    """Ranking key: worst-case csynth latency (max), not best/avg.

    Prefer ``latency_cycles_worst`` (HLS Worst-caseLatency). Fall back to
    ``latency_cycles`` only when worst is absent (legacy/partial reports).
    """
    if not isinstance(report, dict):
        return None
    lat = report.get("latency_cycles_worst")
    if lat is None:
        lat = report.get("latency_cycles")
    try:
        return float(lat) if lat is not None else None
    except (TypeError, ValueError):
        return None


def lat_opt_improved_vs_seed(
    seed_lat: Optional[float],
    post_lat: Optional[float],
) -> bool:
    """True only when lat-opt is strictly lower latency than the seed."""
    if seed_lat is None or post_lat is None:
        return False
    return float(post_lat) < float(seed_lat)


def _candidate(
    *,
    cid: str,
    variant: str,
    latency_cycles: float,
    code_path: Path,
    report_path: Path,
    report: dict[str, Any],
) -> dict[str, Any]:
    return {
        "id": cid,
        "variant": variant,
        "latency_cycles": float(latency_cycles),
        "code_path": str(code_path),
        "report_path": str(report_path) if report_path.is_file() else "",
        "report": dict(report),
    }


_ROUND_CPP_RE = re.compile(r"^(?P<prefix>.+)_r(?P<round>\d+)\.cpp$")


def _collect_lat_opt_round_candidates(
    *,
    cell_dir: Path,
    kernel_stem: str,
    id_prefix: str,
) -> tuple[list[dict[str, Any]], set[str]]:
    """Collect ``{stem}_rN.cpp`` round kernels; return (cands, bodies_seen)."""
    cell_dir = Path(cell_dir)
    candidates: list[dict[str, Any]] = []
    bodies: set[str] = set()
    for cpp in sorted(cell_dir.glob(f"{kernel_stem}_r*.cpp")):
        match = _ROUND_CPP_RE.match(cpp.name)
        if not match or match.group("prefix") != kernel_stem:
            continue
        round_idx = match.group("round")
        body = cpp.read_text(encoding="utf-8")
        if not body.strip():
            continue
        report_path = cell_dir / f"{kernel_stem}_r{round_idx}_report.json"
        report = _load_json_dict(report_path) or {}
        lat = latency_cycles_from_report(report)
        if lat is None:
            continue
        bodies.add(body)
        candidates.append(
            _candidate(
                cid=f"{id_prefix}:lat_opt_r{round_idx}",
                variant=f"lat_opt_r{round_idx}",
                latency_cycles=lat,
                code_path=cpp,
                report_path=report_path,
                report=report,
            )
        )
    return candidates, bodies


def collect_flash_side_candidates(cell_dir: Path, bench: str) -> list[dict[str, Any]]:
    """Collect flash seed + each accepted lat_opt round (+ final if unique).

    Seed priority (lat_opt must NEVER steal the seed slot):
      1) ``{bench}_flash_seed.cpp`` (preserved at lat_opt promote time)
      2) ``{bench}_final.cpp`` (+ final/flash_opt report)
      3) ``{bench}_selected.cpp`` only if it is not identical to lat_opt
    """
    cell_dir = Path(cell_dir)
    candidates: list[dict[str, Any]] = []
    lat_result = _load_json_dict(cell_dir / f"{bench}_latency_opt_result.json")

    seed_cpp, seed_report_path = _resolve_flash_seed_paths(cell_dir, bench)
    seed_report = (_load_json_dict(seed_report_path) if seed_report_path is not None else None) or {}
    seed_lat = latency_cycles_from_report(seed_report)
    if seed_lat is None and lat_result is not None:
        try:
            seed_lat = (
                float(lat_result["seed_latency_cycles"])
                if lat_result.get("seed_latency_cycles") is not None
                else None
            )
        except (TypeError, ValueError):
            seed_lat = None
    if seed_cpp is not None and seed_cpp.is_file() and seed_cpp.read_text(encoding="utf-8").strip():
        # Always keep the true seed in the cosim pool. Missing latency sorts last
        # (inf) so we still attempt TB validation instead of failing with an empty ranking.
        lat_for_rank = float(seed_lat) if seed_lat is not None else float("inf")
        candidates.append(
            _candidate(
                cid="flash:seed",
                variant="seed",
                latency_cycles=lat_for_rank,
                code_path=seed_cpp,
                report_path=seed_report_path or Path(""),
                report=seed_report or ({"latency_cycles": seed_lat} if seed_lat is not None else {}),
            )
        )

    round_cands, round_bodies = _collect_lat_opt_round_candidates(
        cell_dir=cell_dir,
        kernel_stem=f"{bench}_latency_opt",
        id_prefix="flash",
    )
    candidates.extend(round_cands)

    lat_cpp = cell_dir / f"{bench}_latency_opt.cpp"
    lat_report_path = cell_dir / f"{bench}_latency_opt_report.json"
    lat_report = _load_json_dict(lat_report_path) or {}
    lat_ok = bool(lat_result and lat_result.get("success"))
    lat_lat = latency_cycles_from_report(lat_report)
    if lat_lat is None and lat_result is not None:
        try:
            lat_lat = (
                float(lat_result["latency_cycles"])
                if lat_result.get("latency_cycles") is not None
                else None
            )
        except (TypeError, ValueError):
            lat_lat = None
    if (
        lat_ok
        and lat_cpp.is_file()
        and lat_lat is not None
        and lat_cpp.read_text(encoding="utf-8").strip()
    ):
        lat_body = lat_cpp.read_text(encoding="utf-8")
        # Skip duplicate of seed or of an already-collected accepted round.
        seed_body = (
            Path(candidates[0]["code_path"]).read_text(encoding="utf-8")
            if candidates and candidates[0].get("id") == "flash:seed"
            else None
        )
        if lat_body not in round_bodies and lat_body != seed_body:
            candidates.append(
                _candidate(
                    cid="flash:lat_opt",
                    variant="lat_opt",
                    latency_cycles=lat_lat,
                    code_path=lat_cpp,
                    report_path=lat_report_path,
                    report=lat_report or {"latency_cycles": lat_lat},
                )
            )
    return candidates


def _resolve_flash_seed_paths(cell_dir: Path, bench: str) -> tuple[Optional[Path], Optional[Path]]:
    """Return (seed_cpp, seed_report) for the true pre-lat_opt flash kernel."""
    cell_dir = Path(cell_dir)
    preserved = cell_dir / f"{bench}_flash_seed.cpp"
    preserved_report = cell_dir / f"{bench}_flash_seed_report.json"
    if preserved.is_file() and preserved.read_text(encoding="utf-8").strip():
        return preserved, preserved_report if preserved_report.is_file() else None

    final_cpp = cell_dir / f"{bench}_final.cpp"
    final_report = cell_dir / f"{bench}_final_report.json"
    if not final_report.is_file():
        alt = cell_dir / f"{bench}_flash_opt_report.json"
        if alt.is_file():
            final_report = alt
    lat_cpp = cell_dir / f"{bench}_latency_opt.cpp"
    selected_cpp = cell_dir / f"{bench}_selected.cpp"
    selected_report = cell_dir / f"{bench}_selected_report.json"

    # If selected was overwritten by lat_opt, prefer final as the true seed.
    if (
        final_cpp.is_file()
        and final_cpp.read_text(encoding="utf-8").strip()
        and lat_cpp.is_file()
        and selected_cpp.is_file()
        and selected_cpp.read_text(encoding="utf-8") == lat_cpp.read_text(encoding="utf-8")
        and selected_cpp.read_text(encoding="utf-8") != final_cpp.read_text(encoding="utf-8")
    ):
        return final_cpp, final_report if final_report.is_file() else None

    if selected_cpp.is_file() and selected_cpp.read_text(encoding="utf-8").strip():
        # selected is OK only when it is not the lat_opt body
        if not (
            lat_cpp.is_file()
            and selected_cpp.read_text(encoding="utf-8") == lat_cpp.read_text(encoding="utf-8")
        ):
            return selected_cpp, selected_report if selected_report.is_file() else None

    if final_cpp.is_file() and final_cpp.read_text(encoding="utf-8").strip():
        return final_cpp, final_report if final_report.is_file() else None

    if selected_cpp.is_file() and selected_cpp.read_text(encoding="utf-8").strip():
        return selected_cpp, selected_report if selected_report.is_file() else None
    return None, None


def collect_dataflow_side_candidates(cell_dir: Path, bench: str) -> list[dict[str, Any]]:
    """Collect dataflow seed + each accepted lat_opt round (+ final if unique)."""
    cell_dir = Path(cell_dir)
    candidates: list[dict[str, Any]] = []

    seed_cpp = cell_dir / f"{bench}_dataflow.cpp"
    seed_report_path = cell_dir / f"{bench}_dataflow_report.json"
    seed_report = _load_json_dict(seed_report_path) or {}
    seed_lat = latency_cycles_from_report(seed_report)
    seed_body = None
    if seed_cpp.is_file() and seed_lat is not None and seed_cpp.read_text(encoding="utf-8").strip():
        seed_body = seed_cpp.read_text(encoding="utf-8")
        candidates.append(
            _candidate(
                cid="dataflow:seed",
                variant="seed",
                latency_cycles=seed_lat,
                code_path=seed_cpp,
                report_path=seed_report_path,
                report=seed_report,
            )
        )

    round_cands, round_bodies = _collect_lat_opt_round_candidates(
        cell_dir=cell_dir,
        kernel_stem=f"{bench}_dataflow_latency_opt",
        id_prefix="dataflow",
    )
    candidates.extend(round_cands)

    lat_cpp = cell_dir / f"{bench}_dataflow_latency_opt.cpp"
    lat_report_path = cell_dir / f"{bench}_dataflow_latency_opt_report.json"
    lat_result = _load_json_dict(cell_dir / f"{bench}_dataflow_latency_opt_result.json")
    lat_report = _load_json_dict(lat_report_path) or {}
    lat_ok = bool(lat_result and lat_result.get("success"))
    lat_lat = latency_cycles_from_report(lat_report)
    if lat_lat is None and lat_result is not None:
        try:
            lat_lat = (
                float(lat_result["latency_cycles"])
                if lat_result.get("latency_cycles") is not None
                else None
            )
        except (TypeError, ValueError):
            lat_lat = None
    if (
        lat_ok
        and lat_cpp.is_file()
        and lat_lat is not None
        and lat_cpp.read_text(encoding="utf-8").strip()
    ):
        lat_body = lat_cpp.read_text(encoding="utf-8")
        if lat_body not in round_bodies and lat_body != seed_body:
            candidates.append(
                _candidate(
                    cid="dataflow:lat_opt",
                    variant="lat_opt",
                    latency_cycles=lat_lat,
                    code_path=lat_cpp,
                    report_path=lat_report_path,
                    report=lat_report or {"latency_cycles": lat_lat},
                )
            )
    return candidates


def rank_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sort by worst-case latency ascending (min of max); ties prefer seed."""
    return sorted(
        candidates,
        key=lambda c: (
            float(c.get("latency_cycles") if c.get("latency_cycles") is not None else float("inf")),
            0 if c.get("variant") == "seed" else 1,
            str(c.get("id") or ""),
        ),
    )


def write_ranking(
    cell_dir: Path,
    bench: str,
    side: str,
    ranked: list[dict[str, Any]],
) -> Path:
    """Persist ranking JSON (strip heavy report blobs for readability)."""
    cell_dir = Path(cell_dir)
    out = cell_dir / f"{bench}_{side}_candidate_ranking.json"
    slim = []
    for rank, c in enumerate(ranked, start=1):
        slim.append(
            {
                "rank": rank,
                "id": c.get("id"),
                "variant": c.get("variant"),
                "latency_cycles": c.get("latency_cycles"),
                "code_path": c.get("code_path"),
                "report_path": c.get("report_path"),
            }
        )
    payload = {
        "schema": "flash_df_candidate_ranking_v1",
        "benchmark": bench,
        "side": side,
        "candidates": slim,
        "rank1_id": slim[0]["id"] if slim else None,
        "rank1_latency_cycles": slim[0]["latency_cycles"] if slim else None,
    }
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return out


def promote_rank1_kernel(
    cell_dir: Path,
    bench: str,
    ranked: list[dict[str, Any]],
    *,
    side: str,
) -> Optional[Path]:
    """Copy rank-1 code into the downstream pointer path for this side."""
    if not ranked:
        return None
    cell_dir = Path(cell_dir)
    src = Path(ranked[0]["code_path"])
    if not src.is_file():
        return None
    if side == "flash":
        dest = cell_dir / f"{bench}_selected.cpp"
        report_dest = cell_dir / f"{bench}_selected_report.json"
    elif side == "dataflow":
        dest = cell_dir / f"{bench}_dataflow_selected.cpp"
        report_dest = cell_dir / f"{bench}_dataflow_selected_report.json"
    else:
        raise ValueError(f"unknown side {side!r}")
    if src.resolve() != dest.resolve():
        shutil.copy2(src, dest)
    report_src = Path(ranked[0].get("report_path") or "")
    if report_src.is_file():
        if report_src.resolve() != report_dest.resolve():
            shutil.copy2(report_src, report_dest)
    else:
        report_dest.write_text(
            json.dumps({"latency_cycles": ranked[0].get("latency_cycles")}, indent=2) + "\n",
            encoding="utf-8",
        )
    meta = {
        "schema": "flash_df_rank1_promotion_v1",
        "benchmark": bench,
        "side": side,
        "rank1_id": ranked[0].get("id"),
        "latency_cycles": ranked[0].get("latency_cycles"),
        "source_code_path": str(src),
        "dest_code_path": str(dest),
    }
    (cell_dir / f"{bench}_{side}_rank1_promotion.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    return dest


def rank_and_promote(
    cell_dir: Path,
    bench: str,
    *,
    side: str,
) -> list[dict[str, Any]]:
    """Collect → rank → write ranking → promote rank-1. Returns ranked list."""
    if side == "flash":
        cands = collect_flash_side_candidates(cell_dir, bench)
    elif side == "dataflow":
        cands = collect_dataflow_side_candidates(cell_dir, bench)
    else:
        raise ValueError(f"unknown side {side!r}")
    ranked = rank_candidates(cands)
    write_ranking(cell_dir, bench, side, ranked)
    promote_rank1_kernel(cell_dir, bench, ranked, side=side)
    return ranked
