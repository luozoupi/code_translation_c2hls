#!/usr/bin/env python3
"""Seed selected.cpp from phase-B when flash produced no kernel.

Wave-1 hcl/catapult failed flash with "no code in flash LLM response" but
phase-B already passed csim+csynth. DSE can start from that ABI-correct seed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def _orchestrator_state(cell_dir: Path) -> dict:
    path = cell_dir / "pipelined" / "orchestrator_state.json"
    return json.loads(path.read_text(encoding="utf-8"))


def seed_phase_b(cell_dir: Path, bench: str, *, force: bool = False) -> dict:
    selected = cell_dir / f"{bench}_selected.cpp"
    if selected.is_file() and selected.stat().st_size > 0 and not force:
        return {"bench": bench, "skipped": True, "reason": "selected.cpp exists"}
    state = _orchestrator_state(cell_dir)
    code = (state.get("_flow_phase_b_code") or state.get("hls_code") or "").strip()
    if not code:
        raise SystemExit(f"{bench}: no phase-B code in orchestrator_state")
    if not code.endswith("\n"):
        code += "\n"
    report = state.get("_flow_phase_b_report") or state.get("synth_report") or {}
    selected.write_text(code, encoding="utf-8")
    (cell_dir / f"{bench}_phase_b.cpp").write_text(code, encoding="utf-8")
    (cell_dir / f"{bench}_final.cpp").write_text(code, encoding="utf-8")
    if isinstance(report, dict) and report:
        text = json.dumps(report, indent=2, default=str) + "\n"
        (cell_dir / f"{bench}_selected_report.json").write_text(text, encoding="utf-8")
        (cell_dir / f"{bench}_phase_b_report.json").write_text(text, encoding="utf-8")
    return {
        "bench": bench,
        "skipped": False,
        "latency_cycles": report.get("latency_cycles") if isinstance(report, dict) else None,
        "dsp": report.get("dsp") if isinstance(report, dict) else None,
        "bytes": len(code),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix-root", required=True)
    parser.add_argument(
        "--benches",
        default="autosa_mm_hcl,autosa_mm_catapult",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    root = Path(args.matrix_root)
    if not root.is_absolute():
        root = REPO / root
    benches = [b.strip() for b in args.benches.split(",") if b.strip()]
    summary = []
    for bench in benches:
        cells = list((root / "variants").rglob(f"{bench}_selected.cpp"))
        # also cells with orchestrator but no selected
        orch = list((root / "variants").rglob("orchestrator_state.json"))
        cell_dirs = []
        for path in orch:
            cell = path.parent.parent
            if cell.name.startswith("deepseek") or (cell / f"{bench}_multistep_results.json").is_file():
                parent_bench = cell.parent.name
                if parent_bench == bench:
                    cell_dirs.append(cell)
        if not cell_dirs:
            # fallback: variants/*/bench/cell
            for cand in (root / "variants").glob(f"*/{bench}/*"):
                if (cand / "pipelined" / "orchestrator_state.json").is_file():
                    cell_dirs.append(cand)
        seen = set()
        for cell_dir in cell_dirs:
            if cell_dir in seen:
                continue
            seen.add(cell_dir)
            row = seed_phase_b(cell_dir, bench, force=args.force)
            row["cell_dir"] = str(cell_dir)
            summary.append(row)
            print(row)
    if not summary:
        print("no cells seeded", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
