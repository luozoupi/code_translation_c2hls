"""Resolve audited output contracts by benchmark ID and exact public harness hash."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
AUDITED_REGISTRIES = (
    ROOT / "configs/hlsfactory_output_shapes.json",
    ROOT / "configs/hlsfactory_output_shapes_fc27133.json",
)


def resolve_shape_registry(meta, testbench_code, *, registry_path=None, environ=None):
    """An explicit path is binding; otherwise select exactly one audited match.

    Never normalize benchmark aliases or infer shapes from observed outputs.
    An unknown/changed harness fails closed, including when an override is wrong.
    """
    env = os.environ if environ is None else environ
    override = registry_path or env.get("C2HLS_HLSFACTORY_SHAPE_REGISTRY", "").strip()
    paths = (Path(override).expanduser().resolve(),) if override else AUDITED_REGISTRIES
    benchmark = str(meta.get("benchmark") or "")
    actual = hashlib.sha256((testbench_code or "").encode("utf-8")).hexdigest()
    matches, entries = [], []
    for path in paths:
        try:
            registry_bytes = path.read_bytes()
            registry = json.loads(registry_bytes)
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"HLSFactory shape registry unavailable: {exc}") from exc
        if registry.get("schema_version") != "c2hls.hlsfactory-output-shapes.v1":
            raise ValueError("unsupported HLSFactory shape-registry schema")
        entry = (registry.get("benchmarks") or {}).get(benchmark)
        if isinstance(entry, dict):
            entries.append(entry)
            if entry.get("testbench_sha256") == actual:
                matches.append(path)
    if not entries:
        raise ValueError(f"no authoritative HLSFactory shape entry for {benchmark!r}")
    if not matches:
        raise ValueError("HLSFactory testbench does not match audited shape contract: " + actual)
    if len(matches) != 1:
        raise ValueError(f"ambiguous audited HLSFactory shape contracts for {benchmark!r}")
    return matches[0]


def registry_identity(meta, testbench_code, *, registry_path=None, environ=None):
    path = resolve_shape_registry(meta, testbench_code, registry_path=registry_path, environ=environ)
    try:
        label = path.relative_to(ROOT).as_posix()
    except ValueError:
        label = str(path)
    return {"path": label, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "testbench_sha256": hashlib.sha256(testbench_code.encode("utf-8")).hexdigest()}
