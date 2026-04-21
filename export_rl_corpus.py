#!/usr/bin/env python3
"""Export the C-to-HLS corpus as RL training data.

Emits one JSONL file per supervision type in --output/.
This v1 produces translation_sft.jsonl only (user has chosen SFT-on-successful
runs); the other supervision types (step_sft, preference_pairs, metric_points)
ship in later versions.

Every record includes:
  - benchmark       — directory name
  - split           — "train" | "val" | "test" per the fixed split policy
  - source_repo     — rodinia-hls | ML4Accel-Dataset | unknown
  - status          — "ready" | "disabled"
  - gt_file         — filename of the selected ground-truth variant
  - gt_sha256       — content hash of that variant (stable artifact reference)
  - messages        — OpenAI-style [{role, content}, ...] for direct SFT use
  - final_score     — composite rubric score from results/, if available
  - csim_passed     — boolean, from results/ if a run exists
  - source          — "corpus" if built from GT variant, "run" if built from
                       orchestrator history

Splits (fixed up front — do not change without a cost-of-change decision):
  val:  StreamCluster, viterbi
  test: nw, spmv_crs
  train: everything else (ready)

Usage:
    python export_rl_corpus.py                            # default: artifacts/rl_corpus
    python export_rl_corpus.py --output artifacts/v1
    python export_rl_corpus.py --include-disabled         # emit rows for disabled benches (status=disabled)
    python export_rl_corpus.py --require-results          # skip benches without a results/<name>/ dir
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent
BENCHMARKS_DIR = REPO_ROOT / "benchmarks"
RESULTS_DIR = REPO_ROOT / "results"

VAL_SPLIT = {"StreamCluster", "viterbi"}
TEST_SPLIT = {"nw", "spmv_crs"}


def _split_for(benchmark: str) -> str:
    if benchmark in VAL_SPLIT:
        return "val"
    if benchmark in TEST_SPLIT:
        return "test"
    return "train"


def _source_repo(meta: dict) -> str:
    return meta.get("source_repo") or "unknown"


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _select_gt_file(bench_dir: Path, meta: dict) -> Optional[Path]:
    """Same policy the pipeline uses: preferred_gt_file > variants[-1] > baseline."""
    preferred = meta.get("preferred_gt_file")
    if preferred and (bench_dir / preferred).exists():
        return bench_dir / preferred
    for variant in reversed(meta.get("variants") or []):
        vfile = variant.get("file")
        if vfile and (bench_dir / vfile).exists():
            return bench_dir / vfile
    baseline = meta.get("gold_hls_baseline_file", "hls_baseline.cpp")
    bp = bench_dir / baseline
    return bp if bp.exists() else None


def _build_prompt(meta: dict, plain_code: str, header_name: str, header_code: str) -> str:
    """Leak-free prompt: plain.cpp + header + minimal structural cues.

    Does NOT include the GT code, GT interface ports, or any metric from a
    previous run. Matches the RL plan's anti-leakage requirement.
    """
    bench = meta.get("benchmark", "unknown")
    top = meta.get("hls_top") or meta.get("translated_hls_top") or "workload"
    parts = [
        f"Convert the following plain C/C++ kernel for benchmark `{bench}` into a",
        f"synthesizable Xilinx Vitis HLS implementation.",
        "",
        "Constraints:",
        f"- The top-level function must be `extern \"C\" void {top}(...)` with the same",
        f"  argument list as the testbench-visible signature.",
        "- Include the header exactly once and do not redeclare header-owned types.",
        "- Add `#pragma HLS INTERFACE` pragmas for every argument and `return`.",
        "- Add `#pragma HLS PIPELINE` / `UNROLL` / `ARRAY_PARTITION` where they",
        "  improve throughput without changing functional behaviour.",
        "- Preserve the algorithm's correctness.",
        "",
        f"Header ({header_name}):",
        "```cpp",
        header_code.strip(),
        "```",
        "",
        "Plain C kernel (plain.cpp):",
        "```cpp",
        plain_code.strip(),
        "```",
        "",
        "Return the complete HLS source in a single ```cpp code fence.",
    ]
    return "\n".join(parts)


@dataclass
class ExportResult:
    written: int
    skipped_disabled: int
    skipped_missing: int


def _build_record(bench_dir: Path) -> Optional[dict]:
    meta_path = bench_dir / "metadata.json"
    if not meta_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text())
    except json.JSONDecodeError:
        return None

    bench = bench_dir.name
    status = meta.get("status", "ready")
    gt_path = _select_gt_file(bench_dir, meta)
    plain_path = bench_dir / "plain.cpp"
    header_name = meta.get("header_file") or "kernel.h"
    header_path = bench_dir / header_name

    if status == "disabled":
        return {
            "benchmark": bench,
            "split": _split_for(bench),
            "source_repo": _source_repo(meta),
            "status": "disabled",
            "disabled_reason": meta.get("disabled_reason", ""),
        }

    if not (plain_path.exists() and header_path.exists() and gt_path is not None):
        return None  # caller counts this as skipped_missing

    plain_code = plain_path.read_text()
    header_code = header_path.read_text()
    gt_code = gt_path.read_text()

    record = {
        "benchmark": bench,
        "split": _split_for(bench),
        "source_repo": _source_repo(meta),
        "status": "ready",
        "header_file": header_name,
        "gt_file": gt_path.name,
        "gt_sha256": _sha256(gt_code),
        "plain_sha256": _sha256(plain_code),
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are an expert in FPGA High-Level Synthesis (HLS) using "
                    "Xilinx Vitis HLS. Your task is to convert plain C/C++ kernels "
                    "into synthesizable HLS code that preserves functional behaviour."
                ),
            },
            {
                "role": "user",
                "content": _build_prompt(meta, plain_code, header_name, header_code),
            },
            {
                "role": "assistant",
                "content": f"```cpp\n{gt_code.rstrip()}\n```",
            },
        ],
        "source": "corpus",  # built directly from GT, not from a run
    }

    # Best-effort enrich from the most recent run, if one exists.
    results_path = RESULTS_DIR / bench / f"{bench}_results.json"
    if results_path.exists():
        try:
            run = json.loads(results_path.read_text())
            csim = (run.get("csim") or {}).get("passed")
            record["csim_passed"] = bool(csim) if csim is not None else None
            if run.get("phase") == "complete" and run.get("success"):
                record["last_run_phase"] = "complete"
            else:
                record["last_run_phase"] = run.get("phase", "unknown")
        except json.JSONDecodeError:
            pass

    return record


def export(benchmarks_dir: Path, output_dir: Path,
           include_disabled: bool = False,
           require_results: bool = False) -> ExportResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "translation_sft.jsonl"
    manifest_path = output_dir / "manifest.json"

    written = 0
    skipped_disabled = 0
    skipped_missing = 0
    counts_by_split = {"train": 0, "val": 0, "test": 0}
    counts_by_repo: dict[str, int] = {}
    disabled_records: list[dict] = []

    with out_path.open("w") as f:
        for bench_dir in sorted(benchmarks_dir.iterdir()):
            if not bench_dir.is_dir():
                continue
            record = _build_record(bench_dir)
            if record is None:
                skipped_missing += 1
                continue
            if record["status"] == "disabled":
                skipped_disabled += 1
                disabled_records.append(record)
                if not include_disabled:
                    continue
            if require_results and "csim_passed" not in record:
                skipped_missing += 1
                continue
            f.write(json.dumps(record) + "\n")
            written += 1
            counts_by_split[record["split"]] = counts_by_split.get(record["split"], 0) + 1
            repo = record.get("source_repo", "unknown")
            counts_by_repo[repo] = counts_by_repo.get(repo, 0) + 1

    manifest = {
        "version": 1,
        "corpus_type": "translation_sft",
        "record_count": written,
        "counts_by_split": counts_by_split,
        "counts_by_source_repo": counts_by_repo,
        "disabled_benchmarks": [r["benchmark"] for r in disabled_records],
        "skipped_missing": skipped_missing,
        "fixed_splits": {
            "val": sorted(VAL_SPLIT),
            "test": sorted(TEST_SPLIT),
        },
        "output_file": str(out_path.relative_to(output_dir.parent) if output_dir.parent.exists()
                           else out_path.name),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    return ExportResult(written=written,
                        skipped_disabled=skipped_disabled,
                        skipped_missing=skipped_missing)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--benchmarks-dir", default=str(BENCHMARKS_DIR))
    p.add_argument("--output", default=str(REPO_ROOT / "artifacts" / "rl_corpus"))
    p.add_argument("--include-disabled", action="store_true",
                   help="Emit rows for disabled benches (with status='disabled'); default excludes them")
    p.add_argument("--require-results", action="store_true",
                   help="Only emit benchmarks that have a results/<name>/*_results.json on disk")
    args = p.parse_args()

    res = export(Path(args.benchmarks_dir),
                 Path(args.output),
                 include_disabled=args.include_disabled,
                 require_results=args.require_results)
    print(f"wrote {res.written} translation_sft records to {args.output}/translation_sft.jsonl")
    if res.skipped_disabled:
        print(f"  skipped {res.skipped_disabled} disabled benchmarks")
    if res.skipped_missing:
        print(f"  skipped {res.skipped_missing} benchmarks with missing files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
