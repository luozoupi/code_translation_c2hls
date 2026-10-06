#!/usr/bin/env python3
"""Build machsuite_flash_cosim_speedup_vs_gold.csv for a flash campaign.

Gold sources (option C):
  1) Jul 10 MachSuite reference_validation / CSV (reuse when present)
  2) machsuite_gold_cosim_fill/*/cosim_result.json for missing benches
  3) optional --gold-fill-root override

Flash cycles: ranked cosim results under campaign flow/flash_ranked_cosim or
flash_selected_cosim, else multistep generated_step_history cosim.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
DEFAULT_LEGACY = (
    REPO
    / "artifacts/pc2/batch_parallel_machsuite_fd_20260710_machsuite_flash_dataflow"
)
DEFAULT_GOLD_FILL_ROOT = REPO / "artifacts/pc2/machsuite_gold_cosim_fill"
BENCHES = [
    "machsuite_aes_table",
    "machsuite_aes_tableless",
    "machsuite_backprop",
    "machsuite_bfs_bulk",
    "machsuite_bfs_queue",
    "machsuite_fft_transpose",
    "machsuite_gemm_blocked",
    "machsuite_gemm_ncubed",
    "machsuite_md_grid",
    "machsuite_md_knn",
    "machsuite_nw",
    "machsuite_sort_merge",
    "machsuite_sort_radix",
    "machsuite_spmv_crs",
    "machsuite_spmv_ellpack",
    "machsuite_stencil2D",
    "machsuite_stencil3D",
    "machsuite_viterbi",
]


def _load_json(path: Path) -> Any | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _parse_int(v: Any) -> int | None:
    if v is None or isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return int(v)
    s = str(v).strip()
    if not s or s.lower() in {"undef", "na", "n/a", "none", "-"}:
        return None
    try:
        return int(float(s))
    except ValueError:
        return None


def load_legacy_gold(legacy_root: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    csv_path = legacy_root / "reports" / "machsuite_flash_cosim_speedup_vs_gold.csv"
    if csv_path.is_file():
        with csv_path.open(encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                bench = row.get("bench") or ""
                if not bench.startswith("machsuite_"):
                    continue
                cyc = _parse_int(row.get("gold_cosim_kernel_runtime_cycles"))
                if cyc is not None:
                    out[bench] = {
                        "cycles": cyc,
                        "variant": row.get("gold_variant") or "hls_baseline.cpp",
                        "source": str(csv_path),
                    }
    for cell in legacy_root.glob("variants/*/*/*"):
        if not cell.is_dir():
            continue
        bench = cell.parent.name
        ref = _load_json(cell / "reference_validation.json")
        if not isinstance(ref, dict):
            continue
        cosim = ref.get("cosim") if isinstance(ref.get("cosim"), dict) else {}
        cyc = _parse_int(cosim.get("kernel_runtime_cycles"))
        passed = bool(cosim.get("passed"))
        if cyc is None:
            continue
        # Prefer passed=True; keep cycles even if not passed when CSV lacked them.
        prev = out.get(bench)
        if prev is None or (passed and not prev.get("passed", True)):
            out[bench] = {
                "cycles": cyc,
                "variant": ref.get("selected_variant_file")
                or ref.get("gold_variant")
                or "hls_baseline.cpp",
                "source": str(cell / "reference_validation.json"),
                "passed": passed,
            }
    return out


def load_gold_fill(root: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    if not root.is_dir():
        return out
    for cres in root.rglob("cosim_result.json"):
        data = _load_json(cres)
        if not isinstance(data, dict):
            continue
        # Infer bench from path / payload
        bench = data.get("bench") or ""
        if not bench:
            parts = cres.parts
            for p in parts:
                if p.startswith("machsuite_"):
                    bench = p
                    break
        if not bench.startswith("machsuite_"):
            # cell_id often contains bench
            cid = str(data.get("cell_id") or cres.parent.name)
            for b in BENCHES:
                if b in cid:
                    bench = b
                    break
        cyc = _parse_int(data.get("kernel_runtime_cycles"))
        if cyc is None and isinstance(data.get("cosim"), dict):
            cyc = _parse_int(data["cosim"].get("kernel_runtime_cycles"))
        if not bench or cyc is None:
            continue
        if not data.get("passed", True) and data.get("status") not in (None, "ok", "pass"):
            # keep failed gold only if no better exists later
            pass
        out[bench] = {
            "cycles": cyc,
            "variant": "hls_baseline.cpp",
            "source": str(cres),
            "passed": bool(data.get("passed", data.get("status") in ("ok", "pass", True))),
        }
    return out


def find_flash_cosim(campaign: Path, bench: str) -> tuple[bool, int | None, str]:
    # Ranked / selected cosim dirs
    patterns = [
        f"**/flash_ranked_cosim/**/{bench}/**/cosim_result.json",
        f"**/flash_selected_cosim/**/{bench}*/**/cosim_result.json",
        f"**/*{bench}*/cosim_result.json",
    ]
    candidates: list[Path] = []
    for pat in patterns:
        candidates.extend(campaign.glob(pat))
    # Prefer newest
    candidates = sorted({p.resolve() for p in candidates if p.is_file()}, key=lambda p: p.stat().st_mtime, reverse=True)
    for cres in candidates:
        data = _load_json(cres)
        if not isinstance(data, dict):
            continue
        cyc = _parse_int(data.get("kernel_runtime_cycles"))
        if cyc is None and isinstance(data.get("cosim"), dict):
            cyc = _parse_int(data["cosim"].get("kernel_runtime_cycles"))
        passed = bool(data.get("passed")) or data.get("status") in ("ok", "pass")
        if cyc is not None:
            try:
                rel = str(cres.resolve().relative_to(campaign.resolve()))
            except ValueError:
                rel = str(cres)
            return passed, cyc, rel

    # Fallback: multistep history
    for cell in campaign.glob(f"variants/*/{bench}/*"):
        if not cell.is_dir():
            continue
        ms = _load_json(cell / f"{bench}_multistep_results.json")
        if not isinstance(ms, dict):
            continue
        hist = ms.get("generated_step_history") or ms.get("steps") or []
        if isinstance(hist, list):
            for i, step in enumerate(reversed(hist)):
                if not isinstance(step, dict):
                    continue
                cosim = step.get("cosim")
                if isinstance(cosim, dict):
                    cyc = _parse_int(cosim.get("kernel_runtime_cycles"))
                    if cyc is not None:
                        passed = bool(cosim.get("passed"))
                        return passed, cyc, f"{cell.name}/multistep_history[-{i+1}].cosim"
    return False, None, ""


def geomean(vals: list[float]) -> float | None:
    if not vals:
        return None
    return math.exp(sum(math.log(v) for v in vals) / len(vals))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True)
    parser.add_argument(
        "--legacy-gold-root",
        default=str(DEFAULT_LEGACY),
        help="Jul 10 MachSuite campaign with reference_validation gold",
    )
    parser.add_argument(
        "--gold-fill-root",
        default=str(DEFAULT_GOLD_FILL_ROOT),
        help="Root containing filled gold cosim_result.json files",
    )
    parser.add_argument(
        "--out",
        default="",
        help="Output CSV (default: <campaign>/reports/machsuite_flash_cosim_speedup_vs_gold.csv)",
    )
    args = parser.parse_args()

    campaign = Path(args.campaign_root)
    out_path = Path(args.out) if args.out else campaign / "reports" / "machsuite_flash_cosim_speedup_vs_gold.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    gold = load_legacy_gold(Path(args.legacy_gold_root))
    fill = load_gold_fill(Path(args.gold_fill_root))
    for bench, info in fill.items():
        if bench not in gold or gold[bench].get("cycles") is None:
            gold[bench] = info

    rows: list[dict[str, Any]] = []
    speedups: list[float] = []
    for bench in BENCHES:
        g = gold.get(bench) or {}
        g_cyc = g.get("cycles")
        cosim_pass, f_cyc, f_path = find_flash_cosim(campaign, bench)
        # flash_pass: optimization produced a selected/final kernel
        flash_pass = bool(cosim_pass) or bool(f_path)
        if not flash_pass:
            for cell in campaign.glob(f"variants/*/{bench}/*"):
                if (cell / f"{bench}_selected_report.json").is_file() or list(cell.glob("*_final.cpp")):
                    flash_pass = True
                    break
        speedup = None
        include = False
        if g_cyc and cosim_pass and f_cyc and f_cyc > 0:
            speedup = g_cyc / f_cyc
            include = True
            speedups.append(speedup)
        rows.append(
            {
                "bench": bench,
                "flash_pass": flash_pass,
                "gold_variant": g.get("variant") or "hls_baseline.cpp",
                "gold_cosim_kernel_runtime_cycles": g_cyc if g_cyc is not None else "",
                "flash_cosim_kernel_runtime_cycles": f_cyc if f_cyc is not None else "",
                "speedup_gold_over_flash": f"{speedup:.6f}" if speedup is not None else "",
                "flash_cosim_path": f_path,
                "included_in_geomean": include,
            }
        )

    gm = geomean(speedups)
    fieldnames = [
        "bench",
        "flash_pass",
        "gold_variant",
        "gold_cosim_kernel_runtime_cycles",
        "flash_cosim_kernel_runtime_cycles",
        "speedup_gold_over_flash",
        "flash_cosim_path",
        "included_in_geomean",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(
                {
                    **r,
                    "flash_pass": str(bool(r["flash_pass"])),
                    "included_in_geomean": str(bool(r["included_in_geomean"])),
                }
            )
        w.writerow(
            {
                "bench": "__GEOMEAN__",
                "flash_pass": "",
                "gold_variant": "hls_baseline.cpp",
                "gold_cosim_kernel_runtime_cycles": "",
                "flash_cosim_kernel_runtime_cycles": "",
                "speedup_gold_over_flash": f"{gm:.6f}" if gm is not None else "",
                "flash_cosim_path": f"n={len(speedups)} included benches only",
                "included_in_geomean": "True",
            }
        )
        # footer meta rows (same style as template)
        fh.write("deepseek-v4-flash\n")
        fh.write(f"campaign={campaign.name}\n")
        fh.write("lat_opt=off,rag=off,rag2=off,dataflow=excluded\n")
        fh.write(
            "gold=reuse Jul10 reference_validation + machsuite_gold_cosim_fill for missing\n"
        )

    print(f"wrote {out_path}")
    print(f"geomean={gm} n={len(speedups)}")
    missing_gold = [b for b in BENCHES if b not in gold or gold[b].get("cycles") is None]
    if missing_gold:
        print("still_missing_gold:", ", ".join(missing_gold))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
