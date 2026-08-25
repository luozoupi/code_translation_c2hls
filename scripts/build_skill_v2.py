#!/usr/bin/env python3
"""Build the immutable skill_v2 snapshot from two compatible catalogs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


REQUIRED_FIELDS = {"id", "pattern", "strategy"}
RUNTIME_STAT_DEFAULTS = {
    "occurrences": 0,
    "sec_pass": 0,
    "mean_advantage": 0.0,
    "last_used_at": None,
}


def _read_catalog(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    entries = payload.get("skills") if isinstance(payload, dict) else payload
    if not isinstance(entries, list):
        raise ValueError(f"{path} does not contain a skills array")
    result: dict[str, dict[str, Any]] = {}
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict) or not REQUIRED_FIELDS <= set(entry):
            raise ValueError(f"{path}: malformed skill at index {index}")
        skill_id = str(entry["id"])
        if skill_id in result:
            raise ValueError(f"{path}: duplicate skill id {skill_id!r}")
        result[skill_id] = dict(entry)
    return result


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--primary",
        type=Path,
        required=True,
        help="Curated API-layout branch catalog; retained for shared IDs.",
    )
    parser.add_argument(
        "--supplement",
        type=Path,
        required=True,
        help="Local superset catalog; contributes IDs absent from primary.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--primary-ref", required=True)
    parser.add_argument("--supplement-ref", required=True)
    args = parser.parse_args()

    primary_path = args.primary.expanduser().resolve()
    supplement_path = args.supplement.expanduser().resolve()
    primary = _read_catalog(primary_path)
    supplement = _read_catalog(supplement_path)

    merged = dict(primary)
    added_ids = sorted(set(supplement) - set(primary))
    for skill_id in added_ids:
        merged[skill_id] = supplement[skill_id]
    primary_only_ids = sorted(set(primary) - set(supplement))

    # skill_v2 is an evaluation snapshot, not a continuation of mutable
    # online statistics.  Keep curated confidence, but reset runtime counters.
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

    manifest = {
        "name": "skill_v2",
        "schema": "1.1",
        "merge_policy": "primary shared IDs plus supplement-only IDs; runtime statistics reset",
        "primary": {
            "path": str(primary_path),
            "git_ref": args.primary_ref,
            "sha256": _sha256(primary_path),
            "skill_count": len(primary),
        },
        "supplement": {
            "path": str(supplement_path),
            "git_ref": args.supplement_ref,
            "sha256": _sha256(supplement_path),
            "skill_count": len(supplement),
        },
        "primary_only_ids": primary_only_ids,
        "supplement_added_ids": added_ids,
        "skill_count": len(normalized),
        "catalog_sha256": hashlib.sha256(rendered.encode("utf-8")).hexdigest(),
    }
    manifest_path = args.manifest.expanduser().resolve()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"wrote {len(normalized)} skills to {output}; "
        f"supplemented IDs: {','.join(added_ids) or '(none)'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
