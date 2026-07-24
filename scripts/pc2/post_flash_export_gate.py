#!/usr/bin/env python3
"""Gate post-flash dataflow until enough flash-selected kernels are exportable.

Prevents the early-watcher failure mode where campaign_status=complete after a
mass-fail, export yields 0/N, and a hollow dataflow session burns a GPU for hours.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from post_flash_mem_parallel import resolve_selected_kernel  # noqa: E402


def _cells_for_bench(variant_root: Path, bench: str) -> list[Path]:
    # variants/<variant>/<bench>/<cell>
    cells = [cell for cell in variant_root.glob(f"*/{bench}/*") if cell.is_dir()]
    # Prefer deepseek / external-llm cells over leftover failed model cells.
    cells.sort(
        key=lambda p: (
            0 if "deepseek" in p.name else 1,
            0 if "devstral" not in p.name else 2,
            p.name,
        )
    )
    return cells


def count_exportable(campaign_root: Path) -> tuple[int, int, list[str]]:
    """Return (exportable_benches, total_benches_seen, exportable_names)."""
    variant_root = campaign_root / "variants"
    if not variant_root.is_dir():
        return 0, 0, []
    benches = sorted({p.name for p in variant_root.glob("*/*") if p.is_dir()})
    ok: list[str] = []
    for bench in benches:
        for cell in _cells_for_bench(variant_root, bench):
            try:
                kernel = resolve_selected_kernel(cell, bench)
            except Exception:
                kernel = None
            path = kernel[0] if isinstance(kernel, tuple) else kernel
            if path:
                ok.append(bench)
                break
    return len(ok), len(benches), ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--campaign-root", type=Path, required=True)
    ap.add_argument("--min-exportable", type=int, default=1)
    ap.add_argument("--poll-sec", type=int, default=120)
    ap.add_argument(
        "--max-wait-sec",
        type=int,
        default=7200,
        help="Max seconds to wait after invocation for exportable kernels.",
    )
    ap.add_argument(
        "--count-only",
        action="store_true",
        help="Print exportable/total and exit 0 if exportable>=min else 1.",
    )
    args = ap.parse_args()
    root = args.campaign_root.resolve()
    deadline = time.time() + max(0, args.max_wait_sec)

    while True:
        n_ok, n_tot, names = count_exportable(root)
        print(
            f"exportable={n_ok}/{n_tot} min={args.min_exportable} "
            f"sample={names[:8]}{'...' if len(names) > 8 else ''}",
            flush=True,
        )
        if n_ok >= args.min_exportable:
            return 0
        if args.count_only or time.time() >= deadline:
            print(
                f"ERROR: only {n_ok}/{n_tot} exportable kernels "
                f"(need >= {args.min_exportable}); refusing hollow dataflow",
                file=sys.stderr,
            )
            return 1
        time.sleep(max(1, args.poll_sec))


if __name__ == "__main__":
    raise SystemExit(main())
