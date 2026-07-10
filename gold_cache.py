"""Per-part cache for gold-reference validation results.

The validate_gold_reference flow in c2hls.py runs synth + csim + cosim on the
reference (gold) HLS source for every cell. Cosim alone can take 5-15 min per
bench. Across a 56-cell A/B sweep over 28 benches, the same gold runs twice
per bench — 28 wasted gold validations. This cache stores the validation
output keyed by (bench, part, inputs_hash) so cells after the first one for
a given bench skip the work.

Cache files live next to the source as `gold_reports_<part>.json`, one per
part, so flipping the target part starts a fresh cache (correct: gold
numbers change with the part). Set the env var `C2HLS_GOLD_CACHE_DISABLE=1`
to bypass the cache entirely (force a re-run).

Invalidation: inputs_hash covers the gold source code, the testbench source
(csim + cosim variants), every cosim extra file (gold shim, etc.), and the
meta fields that affect synth/csim/cosim (hls_top, cosim_size_overrides,
clock_ns, ...). If any of those changes, the hash mismatches and the entry
is treated as stale.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Optional

REPO = Path(__file__).resolve().parent

# Bench-input fields that affect the validation outcome and therefore the hash.
# Source-code fields:
_HASHED_INPUT_KEYS = (
    "ground_truth_code",
    "gold_hls_source_code",
    "testbench_code",
    "cosim_testbench_code",
    "header_code",
    "header_name",
)
# Meta fields that affect the validation outcome:
_HASHED_META_KEYS = (
    "hls_top",
    "translated_hls_top",
    "supports_csim",
    "supports_cosim",
    "cosim_depths",
    "cosim_size_overrides",
    "part",
    "clock_ns",
)


def _cache_path(part: str) -> Path:
    """`gold_reports_<part-slug>.json` next to this module."""
    slug = part.lower().replace("/", "_").replace(" ", "_")
    return REPO / f"gold_reports_{slug}.json"


def _hash_extra_files(extra_files) -> bytes:
    out = bytearray()
    for ef in extra_files or []:
        if isinstance(ef, dict):
            out += b"DICT\0"
            out += (ef.get("path") or "").encode()
            out += b"\0"
            content = ef.get("content") or ""
            out += content.encode() if isinstance(content, str) else content
            out += b"\0"
            out += b"1" if ef.get("compile", True) else b"0"
        else:
            out += b"PAIR\0"
            out += str(ef).encode()
        out += b"\n"
    return bytes(out)


def hash_inputs(inputs: dict) -> str:
    """Stable hash of every input that affects gold validation."""
    meta = inputs.get("meta") or {}
    h = hashlib.sha256()
    for key in _HASHED_INPUT_KEYS:
        v = inputs.get(key)
        h.update(key.encode()); h.update(b"\0")
        h.update((v or "").encode() if isinstance(v, str) else json.dumps(v, sort_keys=True, default=str).encode())
        h.update(b"\0")
    h.update(b"COSIM_EXTRA\0")
    h.update(_hash_extra_files(inputs.get("cosim_extra_files")))
    h.update(b"EXTRA\0")
    h.update(_hash_extra_files(inputs.get("extra_files")))
    for key in _HASHED_META_KEYS:
        v = meta.get(key)
        h.update(key.encode()); h.update(b"\0")
        h.update(json.dumps(v, sort_keys=True, default=str).encode())
        h.update(b"\0")
    return h.hexdigest()[:16]


def _load(part: str) -> dict:
    path = _cache_path(part)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def _store(part: str, cache: dict) -> None:
    path = _cache_path(part)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(cache, indent=2, default=str))
    tmp.replace(path)


def disabled() -> bool:
    return bool(os.environ.get("C2HLS_GOLD_CACHE_DISABLE"))


def lookup(bench_name: str, part: str, inputs_hash: str) -> Optional[dict]:
    """Return the cached validation dict, or None on miss / hash mismatch."""
    if disabled() or not bench_name:
        return None
    cache = _load(part)
    entry = cache.get(bench_name)
    if entry and entry.get("inputs_hash") == inputs_hash:
        return entry.get("validation")
    return None


def store(bench_name: str, part: str, inputs_hash: str, validation: dict) -> None:
    if disabled() or not bench_name or not validation:
        return
    cache = _load(part)
    cache[bench_name] = {"inputs_hash": inputs_hash, "validation": validation}
    _store(part, cache)
