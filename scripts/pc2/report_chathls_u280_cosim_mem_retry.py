#!/usr/bin/env python3
"""Summarize ChatHLS U280 cosim memory-retry outcomes."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


TOOLING_MARKERS = (
    "Killed",
    "OOM",
    "SIGSEGV",
    "Signal SIGSEGV",
    "Cannot find the xsim executable snapshot",
    "xsim.dir",
)


def _first_error_line(text: str) -> str:
    for line in (text or "").splitlines():
        s = line.strip()
        if not s:
            continue
        if "ERROR" in s or "Killed" in s or "SIGSEGV" in s or "OOM" in s:
            return s[:240]
    for line in (text or "").splitlines():
        if line.strip():
            return line.strip()[:240]
    return ""


def _tooling_flags(text: str) -> list[str]:
    flags = []
    t = text or ""
    for m in TOOLING_MARKERS:
        if m in t:
            flags.append(m)
    # de-dupe preserve order
    seen = set()
    out = []
    for f in flags:
        if f not in seen:
            seen.add(f)
            out.append(f)
    return out


def _peak_rss_from_slurm(slurm_dir: Path, index: int | None) -> str:
    if index is None:
        return ""
    for ext in (".out", ".err"):
        # array job files may be cosim-<array>_<task>.out
        for p in slurm_dir.glob(f"cosim-*_{index}{ext}"):
            txt = p.read_text(errors="ignore")
            m = re.search(r"MaxRSS[=:\s]+([0-9.]+\s*[KMGT]?B?)", txt, re.I)
            if m:
                return m.group(1).replace(" ", "")
            m = re.search(r"peak.?rss[=:\s]+([0-9.]+)", txt, re.I)
            if m:
                return m.group(1)
    return ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", required=True, type=Path)
    ap.add_argument("--write-md", type=Path, default=None)
    args = ap.parse_args()
    run_root: Path = args.run_root
    manifest = json.loads((run_root / "manifest.json").read_text())
    cells = manifest.get("cells") or []
    rows = []
    for cell in cells:
        cell_id = cell.get("cell_id") or cell.get("id") or ""
        bench = (cell.get("bench") or "").removeprefix("chathls_")
        result_path = run_root / "cells" / cell_id / "cosim_result.json"
        if not result_path.is_file():
            rows.append(
                {
                    "bench": bench,
                    "cell_id": cell_id,
                    "status": "pending",
                    "passed": None,
                    "kernel_runtime_cycles": None,
                    "first_error": "",
                    "tooling_flags": [],
                    "peak_rss": "",
                    "result_path": str(result_path),
                }
            )
            continue
        d = json.loads(result_path.read_text())
        err = d.get("error") or ""
        idx = (d.get("provenance") or {}).get("index")
        rows.append(
            {
                "bench": bench,
                "cell_id": cell_id,
                "status": d.get("status"),
                "passed": bool(d.get("passed")),
                "kernel_runtime_cycles": d.get("kernel_runtime_cycles"),
                "first_error": _first_error_line(err),
                "tooling_flags": _tooling_flags(err),
                "peak_rss": _peak_rss_from_slurm(run_root / "slurm", idx),
                "runtime_seconds": d.get("runtime_seconds"),
                "result_path": str(result_path),
            }
        )

    print(f"run_root={run_root}")
    print(f"{'bench':<14} {'passed':>6} {'cycles':>12} {'tooling':<28} error")
    for r in rows:
        print(
            f"{r['bench']:<14} {str(r['passed']):>6} "
            f"{str(r['kernel_runtime_cycles'] if r['kernel_runtime_cycles'] is not None else '—'):>12} "
            f"{','.join(r['tooling_flags']) or '-':<28} "
            f"{r['first_error'][:100]}"
        )

    out_json = run_root / "memretry_report.json"
    out_json.write_text(json.dumps({"run_root": str(run_root), "rows": rows}, indent=2) + "\n")
    print(f"wrote {out_json}")

    if args.write_md:
        lines = [
            f"# ChatHLS U280 cosim memory retry — {run_root.name}",
            "",
            f"- run_root: `{run_root}`",
            "- retry benches: gemm, gemm_ncubed, kernel_symm, kernel_syrk",
            "- excluded resource fails (unchanged): matmul, kernel_3mm",
            "",
            "| Bench | passed | cycles | tooling flags | first error |",
            "|-------|--------|--------|---------------|-------------|",
        ]
        for r in rows:
            err = (r["first_error"] or "").replace("|", "\\|")
            lines.append(
                f"| {r['bench']} | {r['passed']} | {r['kernel_runtime_cycles']} | "
                f"{', '.join(r['tooling_flags']) or '—'} | {err} |"
            )
        args.write_md.parent.mkdir(parents=True, exist_ok=True)
        args.write_md.write_text("\n".join(lines) + "\n")
        print(f"wrote {args.write_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
