#!/usr/bin/env python3
"""Audit a macro-parameterized HLSFactory tree and bind its shape registry."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


AUDITED_FILES = ("plain.cpp", "hls_baseline.cpp", "testbench.cpp")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_path(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _preprocess(source: Path) -> bytes:
    completed = subprocess.run(
        [
            "g++",
            "-E",
            "-P",
            "-I",
            str(source.parent),
            str(source),
            "-o",
            "-",
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return completed.stdout


def _canonical_name(name: str) -> str:
    return name.replace("-", "_")


def _read_metadata(directory: Path) -> dict[str, Any]:
    payload = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not payload.get("benchmark"):
        raise ValueError(f"invalid metadata: {directory / 'metadata.json'}")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--canonical-root", type=Path, required=True)
    parser.add_argument("--variant-root", type=Path, required=True)
    parser.add_argument("--canonical-registry", type=Path, required=True)
    parser.add_argument("--output-registry", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument("--source-ref", required=True)
    args = parser.parse_args()

    canonical_root = args.canonical_root.expanduser().resolve()
    variant_root = args.variant_root.expanduser().resolve()
    canonical_registry_path = args.canonical_registry.expanduser().resolve()
    canonical_registry = json.loads(
        canonical_registry_path.read_text(encoding="utf-8")
    )

    output_entries: dict[str, dict[str, Any]] = {}
    audit_rows: list[dict[str, Any]] = []
    for variant_dir in sorted(variant_root.glob("hlsfactory_*")):
        if not (variant_dir / "metadata.json").is_file():
            continue
        metadata = _read_metadata(variant_dir)
        variant_name = str(metadata["benchmark"])
        canonical_name = _canonical_name(variant_name)
        canonical_dir = canonical_root / canonical_name
        canonical_entry = (canonical_registry.get("benchmarks") or {}).get(
            canonical_name
        )
        if not canonical_dir.is_dir() or not isinstance(canonical_entry, dict):
            raise ValueError(
                f"no canonical benchmark/shape entry for {variant_name!r}"
            )

        file_audits: dict[str, dict[str, Any]] = {}
        for filename in AUDITED_FILES:
            canonical_source = canonical_dir / filename
            variant_source = variant_dir / filename
            if not canonical_source.is_file() or not variant_source.is_file():
                raise ValueError(f"missing audited source for {variant_name}:{filename}")
            canonical_preprocessed = _preprocess(canonical_source)
            variant_preprocessed = _preprocess(variant_source)
            equivalent = canonical_preprocessed == variant_preprocessed
            file_audits[filename] = {
                "equivalent": equivalent,
                "canonical_source_sha256": _sha256_path(canonical_source),
                "variant_source_sha256": _sha256_path(variant_source),
                "canonical_preprocessed_sha256": _sha256_bytes(
                    canonical_preprocessed
                ),
                "variant_preprocessed_sha256": _sha256_bytes(
                    variant_preprocessed
                ),
            }
            if not equivalent:
                raise ValueError(
                    f"preprocessed source differs for {variant_name}:{filename}"
                )

        variant_entry = deepcopy(canonical_entry)
        variant_entry["testbench_sha256"] = _sha256_path(
            variant_dir / "testbench.cpp"
        )
        output_entries[variant_name] = variant_entry
        audit_rows.append(
            {
                "variant_benchmark": variant_name,
                "canonical_benchmark": canonical_name,
                "equivalent": True,
                "files": file_audits,
            }
        )

    if len(output_entries) != len(canonical_registry.get("benchmarks") or {}):
        raise ValueError(
            "variant/canonical benchmark counts differ: "
            f"{len(output_entries)} != "
            f"{len(canonical_registry.get('benchmarks') or {})}"
        )

    generated_at = datetime.now(timezone.utc).isoformat()
    output_payload = {
        "schema_version": canonical_registry["schema_version"],
        "suite": f"HLSFactory enhanced branch {args.source_ref}",
        "policy": deepcopy(canonical_registry["policy"]),
        "benchmarks": output_entries,
        "provenance": {
            "source_ref": args.source_ref,
            "variant_root": str(variant_root),
            "canonical_registry": str(canonical_registry_path),
            "canonical_registry_sha256": _sha256_path(canonical_registry_path),
            "audit_method": (
                "g++ -E -P byte equality for plain.cpp, hls_baseline.cpp, "
                "and testbench.cpp"
            ),
            "generated_at": generated_at,
        },
    }
    audit_payload = {
        "schema_version": "c2hls.hlsfactory-preprocessor-equivalence.v1",
        "source_ref": args.source_ref,
        "canonical_root": str(canonical_root),
        "variant_root": str(variant_root),
        "generated_at": generated_at,
        "benchmark_count": len(audit_rows),
        "all_equivalent": all(row["equivalent"] for row in audit_rows),
        "benchmarks": audit_rows,
    }

    output_registry = args.output_registry.expanduser().resolve()
    audit_output = args.audit_output.expanduser().resolve()
    output_registry.parent.mkdir(parents=True, exist_ok=True)
    audit_output.parent.mkdir(parents=True, exist_ok=True)
    output_registry.write_text(
        json.dumps(output_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    audit_output.write_text(
        json.dumps(audit_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"wrote {len(output_entries)} hash-bound shape entries; "
        f"all preprocessed sources equivalent"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
