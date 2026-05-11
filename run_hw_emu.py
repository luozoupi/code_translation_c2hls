#!/usr/bin/env python3
"""Run nova-style hw_emu on a single bench's variant cpp.

Quick smoke test:
  source scripts/setup_emu_env.sh
  python run_hw_emu.py --nova-bench-dir /home/luo00466/rodinia-hls-nova/Benchmarks/pathfinder/pathfinder_0_baseline \\
                       --kernel-cpp benchmarks/pathfinder/hls_baseline.cpp \\
                       --kernel-basename pathfinder
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

import hls_eval


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--nova-bench-dir", required=True,
                   help="Path to the nova variant dir (must contain Makefile + src/)")
    p.add_argument("--kernel-cpp", required=True,
                   help="Path to the cpp file to substitute as the kernel source")
    p.add_argument("--kernel-basename", required=True,
                   help="Stem of the kernel cpp/header (e.g. 'pathfinder' for pathfinder.cpp)")
    p.add_argument("--timeout", type=int, default=7200,
                   help="hw_emu timeout in seconds (default 7200 = 2h)")
    p.add_argument("--output", default=None, help="Write the result dict to this JSON path")
    args = p.parse_args()

    cpp_path = Path(args.kernel_cpp)
    if not cpp_path.exists():
        print(f"error: kernel cpp not found: {cpp_path}", file=sys.stderr)
        return 2
    hls_code = cpp_path.read_text()

    print(f"Running hw_emu on:")
    print(f"  nova-bench-dir : {args.nova_bench_dir}")
    print(f"  kernel cpp     : {cpp_path}")
    print(f"  basename       : {args.kernel_basename}")
    print(f"  timeout        : {args.timeout}s")
    print()

    result = hls_eval.run_hw_emu_via_nova(
        args.nova_bench_dir,
        hls_code,
        kernel_basename=args.kernel_basename,
        timeout=args.timeout,
    )

    print(f"\nresult:")
    print(f"  ran     : {result['ran']}")
    print(f"  passed  : {result['passed']}")
    print(f"  success : {result['success']}")
    print(f"  kernel_runtime_us     : {result['kernel_runtime_us']}")
    print(f"  kernel_runtime_cycles : {result['kernel_runtime_cycles']}")
    if result.get("error"):
        print(f"  error   : {result['error']}")
    print(f"  work_dir: {result['work_dir']}")

    if args.output:
        Path(args.output).write_text(json.dumps(
            {k: v for k, v in result.items() if k != "log"},
            indent=2, default=str
        ))
        print(f"\nwrote {args.output}")

    return 0 if result["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
