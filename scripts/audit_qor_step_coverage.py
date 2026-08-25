#!/usr/bin/env python3
"""Audit deterministic QoR-knob coverage across saved optimization steps."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from qor_design_space import discover_qor_knobs  # noqa: E402


DIRECTIVE_PATTERNS = {
    "dataflow": re.compile(r"(?im)^\s*#\s*pragma\s+HLS\s+DATAFLOW\b"),
    "bind_op": re.compile(r"(?im)^\s*#\s*pragma\s+HLS\s+BIND_OP\b"),
    "bind_storage": re.compile(r"(?im)^\s*#\s*pragma\s+HLS\s+BIND_STORAGE\b"),
    "resource": re.compile(r"(?im)^\s*#\s*pragma\s+HLS\s+RESOURCE\b"),
    "array_partition": re.compile(
        r"(?im)^\s*#\s*pragma\s+HLS\s+ARRAY_PARTITION\b"
    ),
    "array_reshape": re.compile(
        r"(?im)^\s*#\s*pragma\s+HLS\s+ARRAY_RESHAPE\b"
    ),
    "stream": re.compile(r"(?im)^\s*#\s*pragma\s+HLS\s+STREAM\b"),
    "allocation": re.compile(r"(?im)^\s*#\s*pragma\s+HLS\s+ALLOCATION\b"),
    "m_axi": re.compile(
        r"(?im)^\s*#\s*pragma\s+HLS\s+INTERFACE\b[^\n]*\bm_axi\b"
    ),
}

STEP_RELEVANT_KINDS = {
    "pipeline": {"pipeline_ii"},
    "unroll": {"unroll_factor"},
    "tiling": {"tile_size", "partition_factor", "reshape_factor"},
    "doublebuffer": {
        "tile_size",
        "stream_depth",
        "dataflow_enabled",
        "partition_enabled",
    },
    "coalescing": {
        "interface_max_widen_bitwidth",
        "interface_num_read_outstanding",
        "interface_num_write_outstanding",
        "interface_max_read_burst_length",
        "interface_max_write_burst_length",
    },
}


def _step_name(path: Path) -> str:
    return re.sub(r"^\d+_", "", path.stem).replace("double_buffer", "doublebuffer")


def _source_files(roots: Iterable[Path]) -> list[Path]:
    files: set[Path] = set()
    for root in roots:
        if root.is_file() and root.suffix in {".cpp", ".cc", ".cxx"}:
            files.add(root.resolve())
        elif root.exists():
            files.update(path.resolve() for path in root.rglob("steps/*.cpp"))
    return sorted(files)


def audit(roots: Iterable[Path]) -> dict:
    files = _source_files(roots)
    per_file = []
    aggregate = defaultdict(
        lambda: {
            "files": 0,
            "files_with_supported_knobs": 0,
            "files_with_step_relevant_knobs": 0,
            "knob_occurrences": Counter(),
            "knob_file_counts": Counter(),
            "directive_file_counts": Counter(),
        }
    )

    for path in files:
        code = path.read_text(errors="ignore")
        step = _step_name(path)
        knobs = discover_qor_knobs(code, max_knobs=None)
        kinds = sorted({knob.kind for knob in knobs})
        relevant_kinds = sorted(set(kinds) & STEP_RELEVANT_KINDS.get(step, set(kinds)))
        directives = sorted(
            name for name, pattern in DIRECTIVE_PATTERNS.items() if pattern.search(code)
        )
        per_file.append(
            {
                "path": str(path),
                "step": step,
                "knob_count": len(knobs),
                "knob_kinds": kinds,
                "step_relevant_knob_kinds": relevant_kinds,
                "directives": directives,
            }
        )

        item = aggregate[step]
        item["files"] += 1
        if knobs:
            item["files_with_supported_knobs"] += 1
        if relevant_kinds:
            item["files_with_step_relevant_knobs"] += 1
        item["knob_occurrences"].update(knob.kind for knob in knobs)
        item["knob_file_counts"].update(kinds)
        item["directive_file_counts"].update(directives)

    steps = {}
    for step, item in sorted(
        aggregate.items(), key=lambda pair: (-pair[1]["files"], pair[0])
    ):
        count = item["files"]
        covered = item["files_with_supported_knobs"]
        relevant = item["files_with_step_relevant_knobs"]
        steps[step] = {
            "files": count,
            "files_with_supported_knobs": covered,
            "coverage_pct": round(100.0 * covered / count, 1) if count else 0.0,
            "files_with_step_relevant_knobs": relevant,
            "step_relevant_coverage_pct": (
                round(100.0 * relevant / count, 1) if count else 0.0
            ),
            "knob_occurrences": dict(sorted(item["knob_occurrences"].items())),
            "knob_file_counts": dict(sorted(item["knob_file_counts"].items())),
            "directive_file_counts": dict(
                sorted(item["directive_file_counts"].items())
            ),
        }

    return {
        "schema_version": "c2hls.qor-step-coverage.v1",
        "roots": [str(path.resolve()) for path in roots],
        "file_count": len(files),
        "covered_file_count": sum(row["knob_count"] > 0 for row in per_file),
        "steps": steps,
        "files": per_file,
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "path",
                "step",
                "knob_count",
                "knob_kinds",
                "step_relevant_knob_kinds",
                "directives",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    **row,
                    "knob_kinds": ";".join(row["knob_kinds"]),
                    "step_relevant_knob_kinds": ";".join(
                        row["step_relevant_knob_kinds"]
                    ),
                    "directives": ";".join(row["directives"]),
                }
            )


def _markdown(payload: dict) -> str:
    lines = [
        "# QoR Step Coverage Audit",
        "",
        f"Scanned files: **{payload['file_count']}**",
        f"Files with at least one supported knob: **{payload['covered_file_count']}**",
        "",
        "| Step | Files | Any knob | Step-relevant | Relevant coverage | Supported knob instances | Directive signals |",
        "|---|---:|---:|---:|---:|---|---|",
    ]
    directive_signals = {"dataflow", "bind_op", "bind_storage", "resource"}
    for step, item in payload["steps"].items():
        knob_text = ", ".join(
            f"{kind}:{count}" for kind, count in item["knob_occurrences"].items()
        ) or "none"
        signal_text = ", ".join(
            f"{name}:{count}"
            for name, count in item["directive_file_counts"].items()
            if name in directive_signals
        ) or "none"
        lines.append(
            f"| {step} | {item['files']} | {item['files_with_supported_knobs']} | "
            f"{item['files_with_step_relevant_knobs']} | "
            f"{item['step_relevant_coverage_pct']:.1f}% | {knob_text} | {signal_text} |"
        )
    lines.extend(
        [
            "",
            "Current coverage includes pipeline II, unroll/partition/reshape factors, "
            "named tile/block sizes, stream depth, allocation limits, selected AXI "
            "burst/outstanding/widening options, and disable-only directive ablations.",
            "",
            "`DATAFLOW`, complete partition/reshape, `BIND_OP`, `BIND_STORAGE`, and "
            "legacy `RESOURCE` directives can be disabled as controlled ablations. "
            "Explicit binding latency can also be varied. Categorical implementation "
            "switches such as DSP versus fabric or BRAM versus URAM are not invented.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("roots", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    payload = audit(args.roots)
    if args.output_dir is None:
        print(json.dumps(payload, indent=2))
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "coverage.json").write_text(
        json.dumps(payload, indent=2) + "\n"
    )
    _write_csv(args.output_dir / "files.csv", payload["files"])
    (args.output_dir / "report.md").write_text(_markdown(payload))
    print(json.dumps({
        "output_dir": str(args.output_dir),
        "file_count": payload["file_count"],
        "covered_file_count": payload["covered_file_count"],
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
