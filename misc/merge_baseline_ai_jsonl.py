#!/usr/bin/env python3
"""Merge naive baseline JSONL with Devstral PC2 flash AI JSONL for dev.llm4hls upload.

Example:
  python3 misc/merge_baseline_ai_jsonl.py
  python3 misc/merge_baseline_ai_jsonl.py --all-devstral-flash
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "misc"))

from export_schema_jsonl import validate_jsonl  # noqa: E402
from export_pc2_flash_matrix_jsonl import (  # noqa: E402
    PC2,
    discover_devstral_flash_dirs,
    export_artifact_dir,
)

SUITE_FOR_COMPARE = "hlsfactory_polybench_float_small"


def _normalize_group_path(group_path: list[str]) -> list[str]:
    return [part.replace("_", "-") for part in group_path]


def _normalize_baseline_record(record: dict) -> dict:
    out = json.loads(json.dumps(record))
    problem = out.setdefault("problem", {})
    problem["suite"] = SUITE_FOR_COMPARE
    if problem.get("group_path"):
        problem["group_path"] = _normalize_group_path(problem["group_path"])
    return out


def _read_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open() as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
    return records


def _export_devstral_ai(
    *,
    artifact_dirs: list[Path] | None,
    all_devstral_flash: bool,
    ai_jsonl: Path,
    refresh: bool,
) -> list[dict]:
    if ai_jsonl.is_file() and not refresh and not artifact_dirs and not all_devstral_flash:
        return _read_jsonl(ai_jsonl)

    if all_devstral_flash:
        dirs = discover_devstral_flash_dirs()
    elif artifact_dirs:
        dirs = artifact_dirs
    else:
        dirs = [
            PC2 / "flash_noskills_20260620_004507",
            PC2 / "flash_all_skills_no_avoids_global_20260620_113247",
            PC2 / "flash_skills_20260620_004507",
            PC2 / "flash_bn_skills_new_2_2_20260621_020847",
        ]

    records: list[dict] = []
    per_dir: dict[str, int] = {}
    for artifact_dir in dirs:
        if not artifact_dir.is_dir():
            continue
        batch = export_artifact_dir(artifact_dir)
        per_dir[artifact_dir.name] = len(batch)
        records.extend(batch)

    ai_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with ai_jsonl.open("w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    summary_path = ai_jsonl.with_suffix(".summary.json")
    summary_path.write_text(json.dumps({"artifact_dirs": per_dir, "records": len(records)}, indent=2) + "\n")
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline",
        type=Path,
        default=REPO / "misc/hlsfactory_baseline_u280_20260616_benchmarks.jsonl",
    )
    parser.add_argument(
        "--ai-jsonl",
        type=Path,
        default=REPO / "misc/devstral2_flash_pc2_schema.jsonl",
    )
    parser.add_argument(
        "--artifact-dir",
        action="append",
        dest="artifact_dirs",
        help="PC2 artifact dir name under artifacts/pc2/ (repeatable)",
    )
    parser.add_argument(
        "--all-devstral-flash",
        action="store_true",
        help="Include all 28-bench Devstral flash PC2 artifact dirs",
    )
    parser.add_argument(
        "--refresh-ai",
        action="store_true",
        help="Re-export AI JSONL even if it already exists",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO / "misc/hlsfactory_baseline_plus_devstral2_flash_20260616.jsonl",
    )
    args = parser.parse_args()

    baseline = [_normalize_baseline_record(r) for r in _read_jsonl(args.baseline)]
    artifact_dirs = [PC2 / name for name in args.artifact_dirs] if args.artifact_dirs else None
    refresh = args.refresh_ai or bool(artifact_dirs) or args.all_devstral_flash
    ai = _export_devstral_ai(
        artifact_dirs=artifact_dirs,
        all_devstral_flash=args.all_devstral_flash,
        ai_jsonl=args.ai_jsonl,
        refresh=refresh,
    )

    merged = baseline + ai
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        for rec in merged:
            f.write(json.dumps(rec) + "\n")

    validation = validate_jsonl(args.output, verbose=True)
    summary = {
        "baseline_records": len(baseline),
        "ai_records": len(ai),
        "merged_records": len(merged),
        "ai_jsonl": str(args.ai_jsonl),
        "suite": SUITE_FOR_COMPARE,
        "validation": validation,
    }
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "summary": summary}, indent=2))
    return 0 if validation.get("invalid", 0) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
