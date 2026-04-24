#!/usr/bin/env python3
"""Measure GT synthesis stability by running each validated variant N times.

Addresses the data-quality concern that the same HLS source produces different
synthesis numbers across Vitis versions, server hardware, and workload
contention. For RL reward stability and cross-run reproducibility we want to
know which benchmarks produce deterministic GT numbers and which drift.

For each benchmark:
  1. Load inputs via c2hls._load_benchmark_inputs.
  2. Enumerate variants via c2hls._ground_truth_candidates.
  3. For each variant, run run_hls_synthesis_repeated(n_runs=N) on the
     current target (part + clock from env / .env / defaults).
  4. Record per-run reports + aggregate (mean, stdev, coefficient-of-variation,
     is_stable) for latency, Fmax, and resources.
  5. Persist to artifacts/stability/<bench>.json (one file per benchmark) and
     a top-level artifacts/stability_summary.json aggregating across benches.

Output schema per benchmark:
    {
      "benchmark": "aes",
      "part": "xcu50-fsvh2104-2-e",
      "clock_ns": 3.33,
      "vitis_version": "2025.2",
      "generated_at": "2026-04-24T15:10:00Z",
      "n_runs": 3,
      "stability_threshold_cv": 0.05,
      "variants": [
        {
          "variant_name": "aes_0_baseline",
          "file": "hls_baseline.cpp",
          "sha256": "...",
          "success": true,                     # all N runs synthesized
          "is_stable": true,                   # latency CV <= threshold
          "summary": { per-metric mean/stdev/cv },
          "runs": [ {success, error, report}, ... ]
        },
        ...
      ]
    }

Usage:
    python verify_corpus_stability.py                         # all 17, N=3
    python verify_corpus_stability.py --bench aes --n-runs 3  # single
    python verify_corpus_stability.py --validated-only        # skip variants
                                                              # that failed a
                                                              # first-pass
                                                              # synthesis check
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
BENCHMARKS_DIR = REPO_ROOT / "benchmarks"
ARTIFACTS_DIR = REPO_ROOT / "artifacts" / "stability"

# Delayed imports so --help works without dotenv.
def _load_dotenv_if_available() -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv(REPO_ROOT / ".env")
    except ImportError:
        pass


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _vitis_version() -> str:
    """Best-effort Vitis version from env/settings path.

    Tries in order:
      1. C2HLS_VITIS_VERSION        — explicit override
      2. C2HLS_VITIS_SETTINGS path  — version is a token like "2025.2"
      3. XILINX_VITIS path          — set by settings64.sh itself
    """
    import os
    explicit = os.getenv("C2HLS_VITIS_VERSION")
    if explicit:
        return explicit
    for env in ("C2HLS_VITIS_SETTINGS", "XILINX_VITIS", "XILINX_HLS"):
        val = os.getenv(env, "")
        for token in val.split("/"):
            # Tokens like "2025.2", "2024.1" — one dot, leading digit.
            if token.count(".") == 1 and token and token[0].isdigit():
                return token
    return "unknown"


def verify_variant(candidate: dict, inputs: dict, n_runs: int,
                   prefer_csim_check: bool = True) -> dict:
    """Run repeat-N synthesis on one GT variant. Returns the full variant record."""
    import hls_eval
    meta = inputs["meta"]
    hls_code = candidate["code"]
    header_name = candidate.get("header_name") or inputs.get("header_name") or "kernel.h"
    header_code = candidate.get("header_code", inputs.get("header_code", ""))
    top_function = meta.get("hls_top", "workload")

    logging.info("  variant %s (n_runs=%d)", candidate.get("variant_name", "?"), n_runs)
    t0 = time.time()
    outcome = hls_eval.run_hls_synthesis_repeated(
        hls_code, header_code,
        header_name=header_name,
        top_function=top_function,
        part=hls_eval.DEFAULT_PART,
        clock_ns=hls_eval.DEFAULT_CLOCK_NS,
        extra_files=inputs.get("extra_files", []),
        n_runs=n_runs,
    )
    elapsed = round(time.time() - t0, 1)

    return {
        "variant_name": candidate.get("variant_name", ""),
        "file": candidate.get("file", ""),
        "step_name": candidate.get("step_name", ""),
        "sha256": _sha256(hls_code),
        "success": outcome["success"],
        "is_stable": outcome["summary"].get("is_stable", False),
        "n_runs": outcome["n_runs"],
        "elapsed_sec": elapsed,
        "summary": outcome["summary"],
        "canonical_report": outcome["canonical_report"],
        "runs": [
            {
                "success": r["success"],
                "error": (r["error"] or "")[:300] if r.get("error") else "",
                "report": {k: v for k, v in (r.get("report") or {}).items()
                           if k != "work_dir"},  # strip non-deterministic
            }
            for r in outcome["runs"]
        ],
    }


def verify_benchmark(bench_dir: Path, n_runs: int,
                     validated_only: bool = False) -> dict:
    from c2hls import _load_benchmark_inputs, _ground_truth_candidates
    import hls_eval

    inputs = _load_benchmark_inputs(str(bench_dir))
    bench_name = inputs.get("bench_name", bench_dir.name)
    candidates = _ground_truth_candidates(inputs)
    if not candidates:
        return {"benchmark": bench_name, "variants": [], "skip_reason": "no_candidates"}

    logging.info("Benchmark %s: %d candidate variants, n_runs=%d",
                 bench_name, len(candidates), n_runs)

    variant_records = []
    for candidate in candidates:
        # Optional fast-path: skip variants that aren't likely to synth cleanly
        # by doing a single run first and checking success.
        if validated_only:
            probe = hls_eval.run_hls_synthesis(
                candidate["code"],
                candidate.get("header_code", inputs.get("header_code", "")),
                header_name=candidate.get("header_name") or "kernel.h",
                top_function=inputs["meta"].get("hls_top", "workload"),
                part=hls_eval.DEFAULT_PART,
                clock_ns=hls_eval.DEFAULT_CLOCK_NS,
                extra_files=inputs.get("extra_files", []),
            )
            if not probe.get("success"):
                variant_records.append({
                    "variant_name": candidate.get("variant_name", ""),
                    "file": candidate.get("file", ""),
                    "sha256": _sha256(candidate["code"]),
                    "success": False,
                    "is_stable": False,
                    "skip_reason": "probe_synth_failed",
                    "error": (probe.get("error") or "")[:300],
                })
                continue

        variant_records.append(verify_variant(candidate, inputs, n_runs))

    return {
        "benchmark": bench_name,
        "part": hls_eval.DEFAULT_PART,
        "clock_ns": hls_eval.DEFAULT_CLOCK_NS,
        "vitis_version": _vitis_version(),
        "generated_at": _now_utc(),
        "n_runs": n_runs,
        "stability_threshold_cv": hls_eval.STABILITY_CV_THRESHOLD,
        "variants": variant_records,
    }


def write_benchmark_record(record: dict, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{record['benchmark']}.json"
    out_path.write_text(json.dumps(record, indent=2, default=str) + "\n")
    return out_path


def write_summary(records: list, output_dir: Path) -> Path:
    """Aggregate per-benchmark records into a single top-level summary."""
    summary_path = output_dir.parent / "stability_summary.json"
    rows = []
    for rec in records:
        for v in rec.get("variants", []):
            lat = v.get("summary", {}).get("latency_ns") or {}
            fmax = v.get("summary", {}).get("fmax_mhz") or {}
            rows.append({
                "benchmark": rec["benchmark"],
                "variant": v.get("variant_name"),
                "file": v.get("file"),
                "success": v.get("success"),
                "is_stable": v.get("is_stable"),
                "n_runs": v.get("n_runs") or rec.get("n_runs"),
                "latency_ns_mean": lat.get("mean"),
                "latency_ns_cv": lat.get("cv"),
                "fmax_mhz_mean": fmax.get("mean"),
                "fmax_mhz_cv": fmax.get("cv"),
                "elapsed_sec": v.get("elapsed_sec"),
            })
    summary = {
        "generated_at": _now_utc(),
        "part": records[0]["part"] if records else None,
        "clock_ns": records[0]["clock_ns"] if records else None,
        "vitis_version": records[0]["vitis_version"] if records else None,
        "stability_threshold_cv": records[0]["stability_threshold_cv"] if records else None,
        "total_variants": len(rows),
        "stable_variants": sum(1 for r in rows if r["is_stable"]),
        "failed_variants": sum(1 for r in rows if r["success"] is False),
        "rows": rows,
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    return summary_path


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bench", help="Single benchmark name; default iterates all")
    p.add_argument("--benchmarks-dir", default=str(BENCHMARKS_DIR))
    p.add_argument("--output", default=str(ARTIFACTS_DIR),
                   help="Per-benchmark record directory")
    p.add_argument("--n-runs", type=int, default=3,
                   help="Number of synthesis repeats per variant (default: 3)")
    p.add_argument("--validated-only", action="store_true",
                   help="Probe each variant once first; skip N-run pass on variants that fail the probe")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    _load_dotenv_if_available()
    sys.path.insert(0, str(REPO_ROOT))  # allow `import c2hls`/`import hls_eval`

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    bench_root = Path(args.benchmarks_dir)
    if args.bench:
        targets = [bench_root / args.bench]
        if not targets[0].is_dir():
            print(f"error: benchmark not found: {args.bench}", file=sys.stderr)
            return 2
    else:
        targets = sorted(p for p in bench_root.iterdir() if p.is_dir())

    output_dir = Path(args.output)
    records = []
    for bench_dir in targets:
        logging.info("=== %s ===", bench_dir.name)
        record = verify_benchmark(bench_dir, args.n_runs,
                                  validated_only=args.validated_only)
        write_benchmark_record(record, output_dir)
        records.append(record)

    summary_path = write_summary(records, output_dir)
    stable = sum(1 for r in records for v in r.get("variants", []) if v.get("is_stable"))
    total = sum(len(r.get("variants", [])) for r in records)
    print(f"wrote {len(records)} benchmark records to {output_dir}/")
    print(f"  stability: {stable}/{total} variants met CV <= {records[0]['stability_threshold_cv'] if records else '?'} on latency_ns")
    print(f"  summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
