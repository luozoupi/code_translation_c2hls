#!/usr/bin/env python3
"""Run the load/compute/store focus group on an existing legal kernel.

Example::

    python3 scripts/pc2/run_post_flash_focus_group.py --pc2 \\
        --kernel-dir artifacts/pc2/.../dse_v2/pe64_simd16 \\
        --bench autosa_mm --show-prompts

    python3 scripts/pc2/run_post_flash_focus_group.py --pc2 \\
        --kernel-dir path/to/pe16_simd32 --bench autosa_mm
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "pc2"))

from c2hls_paths import BENCHMARKS_DIR, configure_site
from post_flash_focus_group import (
    focus_round_limit,
    prompt_text_for_docs,
    repair_round_limit,
    resolve_kernel_from_dir,
    run_focus_group_for_kernel_dir,
)


def _resolve_bench_dir(bench: str, explicit: str) -> Path:
    if explicit:
        path = Path(explicit).expanduser()
        if not path.is_absolute():
            path = REPO / path
        return path
    candidates = [
        BENCHMARKS_DIR / bench,
        REPO / "related_work/benchmarks/autosa_ready" / bench,
        REPO / "benchmarks_autosa_dse" / bench,
    ]
    for path in candidates:
        if (path / "metadata.json").is_file() or (path / "kernel.h").is_file():
            return path
    return BENCHMARKS_DIR / bench


def main() -> int:
    parser = argparse.ArgumentParser(description="Focus group latency pass on a legal kernel")
    parser.add_argument("--pc2", action="store_true", help="PC2 site paths")
    parser.add_argument("--kernel-dir", type=str, default="", help="Directory with bench.cpp + report")
    parser.add_argument("--bench", type=str, default="autosa_mm")
    parser.add_argument("--bench-dir", type=str, default="")
    parser.add_argument("--model", type=str, default=os.getenv("C2HLS_MODEL", ""))
    parser.add_argument("--rounds", type=int, default=0, help="Focus rounds (default 2)")
    parser.add_argument("--turns", type=int, default=0, help="Legality repairs per round (default 3)")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--show-prompts", action="store_true")
    args = parser.parse_args()

    if args.show_prompts:
        prompts = prompt_text_for_docs()
        print("=== ANALYST SYSTEM ===\n")
        print(prompts["analyst_system"])
        print("\n=== SPECIALIST SYSTEMS ===\n")
        for name, text in prompts["specialist_systems"].items():
            print(f"--- {name} ---\n{text}\n")
        print("\n=== CHAIR SYSTEM ===\n")
        print(prompts["chair_system"])
        print("\n=== REPAIR USER ===\n")
        print(prompts["repair_user"])
        return 0

    if not args.kernel_dir.strip():
        print("--kernel-dir is required (unless --show-prompts)", file=sys.stderr)
        return 1

    if args.pc2:
        configure_site("pc2")
    os.environ["C2HLS_FOCUS_GROUP"] = "1"
    if args.rounds > 0:
        os.environ["C2HLS_FOCUS_GROUP_ROUNDS"] = str(args.rounds)
    if args.turns > 0:
        os.environ["C2HLS_FOCUS_GROUP_REPAIR_ROUNDS"] = str(args.turns)

    kernel_dir = Path(args.kernel_dir).expanduser()
    if not kernel_dir.is_absolute():
        kernel_dir = REPO / kernel_dir
    if not kernel_dir.is_dir():
        print(f"kernel dir missing: {kernel_dir}", file=sys.stderr)
        return 1

    cpp, report = resolve_kernel_from_dir(kernel_dir, args.bench)
    print(
        f"kernel={cpp} report={report} rounds={focus_round_limit()} repairs={repair_round_limit()}"
    )
    if args.dry_run:
        return 0 if cpp else 1

    from c2hls import C2HLSOrchestrator, DEFAULT_MODEL_ID
    from post_flash_dse import dse_max_tokens

    model = args.model.strip() or DEFAULT_MODEL_ID
    orch = C2HLSOrchestrator(
        gpt_model=model,
        turns_limitation=repair_round_limit(),
        max_completion_tokens=dse_max_tokens(),
    )
    outcome = run_focus_group_for_kernel_dir(
        bench=args.bench,
        bench_dir=_resolve_bench_dir(args.bench, args.bench_dir),
        kernel_dir=kernel_dir,
        orchestrator=orch,
        skip_existing=not args.force,
    )
    print(f"success={outcome.success} out={outcome.out_dir} error={outcome.error}")
    if outcome.result:
        print(f"latency_cycles={outcome.result.get('latency_cycles')}")
    return 0 if outcome.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
