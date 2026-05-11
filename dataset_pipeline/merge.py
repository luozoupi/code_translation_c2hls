"""Merge v2.0 trajectory records with the existing reference jsonls.

The reference jsonl files at
[results/references_philip/{hw,sw}_emu_*.jsonl] are the gold-standard
upstream measurements. This helper reads them in and joins by
(suite, group_path, variant.name) so a single output stream carries
both our generated points AND the upstream references for the same
(kernel, step) cell. Downstream rubric.py / report.py consumers don't
need to know about the join.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple


def _key(rec: Dict[str, Any]) -> Tuple[str, Tuple[str, ...], str, str]:
    """Group key used for joining: (suite, group_path, variant_name,
    report_type).

    Supports two on-disk variant placements: top-level (our v2 records and
    the agentic v1 jsonls) and nested under ``implementation.variant``
    (philip's reference jsonls)."""
    p = rec.get("problem") or {}
    v = rec.get("variant") or {}
    if not v:
        impl = rec.get("implementation") or {}
        nested = impl.get("variant") or {}
        if nested:
            v = nested
    return (
        p.get("suite", ""),
        tuple(p.get("group_path") or ()),
        v.get("name", ""),
        rec.get("report_type", ""),
    )


def read_jsonl(path: str) -> Iterator[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return
    with p.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def write_jsonl(records: Iterable[Dict[str, Any]], path: str) -> int:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with p.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, separators=(",", ":")))
            f.write("\n")
            n += 1
    return n


def merge_with_references(
    *,
    generated_jsonl: str,
    reference_paths: List[str],
    output_jsonl: str,
) -> Dict[str, Any]:
    """Read all reference jsonls + the generated jsonl, write a merged
    stream, and return a summary of join coverage. Reference rows pass
    through unchanged but get tagged with `implementation.origin =
    "rodinia_hls_benchmark"` if absent. Generated rows pass through
    unchanged.

    The merged stream is ordered: for each (kernel, step, report_type)
    group, references first, generated rows second."""
    by_key: Dict[Tuple[str, Tuple[str, ...], str, str], List[Dict[str, Any]]] = {}

    # Load references and tag missing origins.
    n_ref = 0
    for ref_path in reference_paths:
        for rec in read_jsonl(ref_path):
            n_ref += 1
            impl = rec.setdefault("implementation", {})
            impl.setdefault("origin", "rodinia_hls_benchmark")
            by_key.setdefault(_key(rec), []).append(rec)

    # Load generated.
    n_gen = 0
    for rec in read_jsonl(generated_jsonl):
        n_gen += 1
        impl = rec.setdefault("implementation", {})
        impl.setdefault("origin", "c2hls_orchestrator")
        by_key.setdefault(_key(rec), []).append(rec)

    # Write merged stream — references first within each group.
    def _origin_rank(rec: Dict[str, Any]) -> int:
        impl = rec.get("implementation") or {}
        return 0 if impl.get("origin", "").startswith("rodinia") else 1

    def _gen():
        for k in sorted(by_key.keys()):
            group = sorted(by_key[k], key=_origin_rank)
            for rec in group:
                yield rec

    n_total = write_jsonl(_gen(), output_jsonl)
    return {
        "reference_records": n_ref,
        "generated_records": n_gen,
        "merged_records": n_total,
        "joint_keys": len(by_key),
        "output": output_jsonl,
    }
