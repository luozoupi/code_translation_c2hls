#!/usr/bin/env python3
"""Slice a skill union into Multistep per-step msss files.

Does not edit hls_full_optimization_skills_schema_1_1_package/multistep/
(the live 90 ∪ gemm_flatten_v2 slice). Reuses that assignment.json as the
step routing table. Later --source files win on id conflicts.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
PKG = REPO / "hls_full_optimization_skills_schema_1_1_package"
DEFAULT_ASSIGNMENT = PKG / "multistep" / "assignment.json"
STEPS = ("tiling", "pipeline", "unroll", "coalescing", "doublebuffer")

SKILLS_90 = PKG / "skills_ii_target_miss_solutions_added(90skills).json"
SKILLS_V1 = PKG / "skills_ii_target_miss_solutions_added(90skills)_gemm_flatten_v1.json"
SKILLS_V2 = PKG / "skills_ii_target_miss_solutions_added(90skills)_gemm_flatten_v2.json"
NO_RMW = PKG / "flash_no_RMW_m_axi_skill_entries.json"
UNION_90_V2 = PKG / "skills_ii_target_miss_solutions_added(90skills)_plus_gemm_flatten_v2.json"
MSSS_V1_NORMW_DIR = PKG / "multistep_gf_v1_normw"


def _load_skills(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    raw = data.get("skills", []) if isinstance(data, dict) else data
    out: list[dict[str, Any]] = []
    for entry in raw or []:
        if isinstance(entry, dict) and entry.get("id"):
            out.append(entry)
    return out


def merge_sources(sources: list[tuple[str, Path]]) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    by_id: dict[str, dict[str, Any]] = {}
    won: dict[str, str] = {}
    for label, path in sources:
        for skill in _load_skills(path):
            sid = str(skill["id"])
            by_id[sid] = skill
            won[sid] = label
    return by_id, won


def source_label(path: Path) -> str:
    name = path.name
    if "gemm_flatten_v2" in name:
        return "gemm_flatten_v2"
    if "gemm_flatten_v1" in name:
        return "gemm_flatten_v1"
    if "no_RMW" in name or "no-RMW" in name:
        return "no_rmw_overlay"
    if "90skills" in name and "gemm_flatten" not in name:
        return "90"
    return path.stem


def write_merged_pack(
    dest: Path,
    *,
    by_id: dict[str, dict[str, Any]],
    won: dict[str, str],
    sources: list[tuple[str, Path]],
    merge_rule: str,
) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    doc = {
        "saved_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "schema": "1.1",
        "derived_from": [path.name for _, path in sources],
        "merge_rule": merge_rule,
        "source_counts": dict(Counter(won.values())),
        "skill_count": len(by_id),
        "skills": [by_id[sid] for sid in sorted(by_id)],
    }
    dest.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")


def slice_msss(
    assignment: dict[str, Any],
    by_id: dict[str, dict[str, Any]],
    won: dict[str, str],
    sources: list[tuple[str, Path]],
    out_dir: Path,
    merge_rule: str,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    derived = [path.name for _, path in sources]
    steps_meta: dict[str, Any] = {}
    placed = 0
    missing: list[str] = []

    for step in STEPS:
        step_info = ((assignment.get("steps") or {}).get(step) or {})
        focus = str(step_info.get("focus") or "")
        buckets: dict[str, list[dict[str, Any]]] = {"skills": [], "avoids": []}
        assign_rows: dict[str, list[dict[str, Any]]] = {"skills": [], "avoids": []}
        for rec in assignment.get("skills") or []:
            sid = rec.get("id")
            if not sid:
                continue
            for place in rec.get("placements") or []:
                if place.get("step") != step:
                    continue
                role = "avoids" if place.get("role") == "avoids" else "skills"
                if sid not in by_id:
                    missing.append(f"{step}/{role}/{sid}")
                    continue
                buckets[role].append(by_id[sid])
                assign_rows[role].append(
                    {
                        "id": sid,
                        "band": place.get("band") or "primary",
                        "source": won[sid],
                    }
                )
                placed += 1

        for role, skills in buckets.items():
            primary = sum(1 for row in assign_rows[role] if row["band"] == "primary")
            overlap = sum(1 for row in assign_rows[role] if row["band"] != "primary")
            doc = {
                "saved_at": now,
                "schema": "1.1",
                "flow": "multistep",
                "step": step,
                "role": role,
                "focus": focus,
                "derived_from": derived,
                "merge_rule": merge_rule,
                "skill_count": len(skills),
                "primary_count": primary,
                "overlap_count": overlap,
                "assignment": assign_rows[role],
                "skills": skills,
            }
            (out_dir / f"{step}_{role}.json").write_text(
                json.dumps(doc, indent=2) + "\n", encoding="utf-8"
            )
        steps_meta[step] = {
            "focus": focus,
            "skills": len(buckets["skills"]),
            "avoids": len(buckets["avoids"]),
        }

    snap = {
        "saved_at": now,
        "schema": "1.1",
        "flow": "multistep",
        "derived_from": derived,
        "merge_rule": merge_rule,
        "source_counts": dict(Counter(won.values())),
        "union": len(by_id),
        "placed": placed,
        "missing_from_union": missing,
        "steps": steps_meta,
    }
    (out_dir / "assignment.json").write_text(
        json.dumps(snap, indent=2) + "\n", encoding="utf-8"
    )
    return snap


def build(
    sources: list[Path],
    *,
    assignment_path: Path = DEFAULT_ASSIGNMENT,
    out_dir: Path | None = None,
    merged_pack: Path | None = None,
    merge_rule: str = "",
) -> dict[str, Any]:
    labeled = [(source_label(path), path) for path in sources]
    if not merge_rule:
        names = " ∪ ".join(label for label, _ in labeled)
        last = labeled[-1][0] if labeled else "last"
        merge_rule = f"Union of {names}. On id conflicts, {last} replaces the earlier entry."
    by_id, won = merge_sources(labeled)
    if merged_pack is not None:
        write_merged_pack(
            merged_pack,
            by_id=by_id,
            won=won,
            sources=labeled,
            merge_rule=merge_rule,
        )
    snap: dict[str, Any] = {
        "union": len(by_id),
        "source_counts": dict(Counter(won.values())),
        "merged_pack": str(merged_pack) if merged_pack else None,
    }
    if out_dir is not None:
        assignment = json.loads(assignment_path.read_text(encoding="utf-8"))
        snap.update(
            slice_msss(assignment, by_id, won, labeled, out_dir, merge_rule)
        )
        snap["out_dir"] = str(out_dir)
    return snap


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preset",
        choices=("v1_normw", "90_v2", "all"),
        help="v1_normw: msss slice from gemm_flatten_v1 + no-RMW overlay. "
        "90_v2: packaged union of 90 + gemm_flatten_v2. all: both.",
    )
    parser.add_argument("--source", action="append", type=Path, default=[])
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--merged-pack", type=Path)
    parser.add_argument("--assignment", type=Path, default=DEFAULT_ASSIGNMENT)
    args = parser.parse_args()
    jobs: list[dict[str, Any]] = []
    if args.preset in (None,):
        pass
    if args.preset in {"v1_normw", "all"}:
        jobs.append(
            build(
                [SKILLS_V1, NO_RMW],
                assignment_path=args.assignment,
                out_dir=MSSS_V1_NORMW_DIR,
                merge_rule=(
                    "Union of gemm_flatten_v1 and flash_no_RMW overlay. "
                    "On id conflicts, the no-RMW overlay replaces the v1 entry."
                ),
            )
        )
    if args.preset in {"90_v2", "all"}:
        jobs.append(
            build(
                [SKILLS_90, SKILLS_V2],
                assignment_path=args.assignment,
                merged_pack=UNION_90_V2,
                merge_rule=(
                    "Union of the 90-skill package and gemm_flatten_v2. "
                    "On id conflicts, gemm_flatten_v2 replaces the 90-skill entry."
                ),
            )
        )
    if args.source:
        jobs.append(
            build(
                args.source,
                assignment_path=args.assignment,
                out_dir=args.out_dir,
                merged_pack=args.merged_pack,
            )
        )
    if not jobs:
        parser.error("use --preset and/or --source")
    for snap in jobs:
        print(json.dumps({k: snap[k] for k in snap if k != "missing_from_union"}, indent=2))
        missing = snap.get("missing_from_union") or []
        if missing:
            print(f"missing {len(missing)} assignment rows (ok if pack is a subset)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
