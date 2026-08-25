#!/usr/bin/env python3
"""Build a frozen schema-1.1 catalog from a base package and flash overlay."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import fields
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from skill_library import Skill


REQUIRED_FIELDS = {"id", "pattern", "strategy"}
RUNTIME_STAT_DEFAULTS = {
    "occurrences": 0,
    "sec_pass": 0,
    "mean_advantage": 0.0,
    "last_used_at": None,
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_catalog(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema") != "1.1":
        raise ValueError(f"{path} must be a schema-1.1 object")
    entries = payload.get("skills")
    if not isinstance(entries, list):
        raise ValueError(f"{path} does not contain a skills array")

    known_fields = {item.name for item in fields(Skill)}
    result: dict[str, dict[str, Any]] = {}
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict) or not REQUIRED_FIELDS <= set(entry):
            raise ValueError(f"{path}: malformed skill at index {index}")
        unknown = sorted(set(entry) - known_fields)
        if unknown:
            raise ValueError(
                f"{path}: skill at index {index} has unsupported fields: "
                f"{','.join(unknown)}"
            )
        skill_id = str(entry["id"])
        if skill_id in result:
            raise ValueError(f"{path}: duplicate skill id {skill_id!r}")
        result[skill_id] = dict(entry)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--overlay", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-ref", required=True)
    args = parser.parse_args()

    base_path = args.base.expanduser().resolve()
    overlay_path = args.overlay.expanduser().resolve()
    base = _read_catalog(base_path)
    overlay = _read_catalog(overlay_path)

    merged = dict(base)
    merged.update(overlay)
    overwritten_ids = sorted(set(base) & set(overlay))
    overlay_added_ids = sorted(set(overlay) - set(base))

    normalized: list[dict[str, Any]] = []
    for skill_id in sorted(merged):
        entry = dict(merged[skill_id])
        entry.update(RUNTIME_STAT_DEFAULTS)
        normalized.append(entry)

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {"schema": "1.1", "skills": normalized}
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    output.write_text(rendered, encoding="utf-8")

    positive_count = sum(
        str(entry.get("confidence", "")).strip().lower() != "avoid"
        and str(entry.get("kind", "")).strip().lower() != "avoid_rule"
        for entry in normalized
    )
    avoid_count = len(normalized) - positive_count
    manifest_payload = {
        "name": "skill_v3_no_rmw_fc27133",
        "schema": "1.1",
        "source_ref": args.source_ref,
        "merge_policy": (
            "base catalog followed by ID-keyed flash no-RMW overlay; "
            "overlay wins and runtime statistics are reset"
        ),
        "base": {
            "path": str(base_path),
            "sha256": _sha256(base_path),
            "skill_count": len(base),
        },
        "overlay": {
            "path": str(overlay_path),
            "sha256": _sha256(overlay_path),
            "skill_count": len(overlay),
        },
        "overwritten_ids": overwritten_ids,
        "overlay_added_ids": overlay_added_ids,
        "skill_count": len(normalized),
        "positive_skill_count": positive_count,
        "avoid_skill_count": avoid_count,
        "catalog_sha256": hashlib.sha256(rendered.encode("utf-8")).hexdigest(),
    }
    manifest = args.manifest.expanduser().resolve()
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        json.dumps(manifest_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"wrote {len(normalized)} skills ({positive_count} positive, "
        f"{avoid_count} avoid); overlaid {len(overwritten_ids)} IDs"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
