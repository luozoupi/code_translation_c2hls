#!/usr/bin/env python3
"""Analyze flash_hpc_positive matrix results and patch skills JSON for next round."""

from __future__ import annotations

import argparse
import json
import math
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PKG = REPO / "hls_full_optimization_skills_schema_1_1_package"

DP_BENCHES = frozenset({"hlsfactory_nussinov", "hlsfactory_floyd-warshall"})
STENCIL3D = frozenset({"hlsfactory_heat-3d"})
GEMM_FAMILY = frozenset(
    {
        "hlsfactory_gemm",
        "hlsfactory_syr2k",
        "hlsfactory_syrk",
        "hlsfactory_symm",
        "hlsfactory_trmm",
        "hlsfactory_2mm",
        "hlsfactory_3mm",
        "hlsfactory_lu",
        "hlsfactory_ludcmp",
        "hlsfactory_cholesky",
        "hlsfactory_gramschmidt",
    }
)


def gt_ratio(row: dict) -> float | None:
    vgt = (row.get("summary") or {}).get("vs_ground_truth") or {}
    lc = vgt.get("latency_cycles") or {}
    if lc.get("ratio") is not None:
        return float(lc["ratio"])
    sr = (row.get("summary") or {}).get("synth_report") or {}
    lat = sr.get("latency_cycles")
    gt = lc.get("ground_truth")
    if lat is not None and gt:
        return float(lat) / float(gt)
    return None


def load_matrix(pc2: Path, prefix: str, stamp: str) -> dict[str, dict]:
    path = pc2 / f"{prefix}_{stamp}" / "matrix.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    return {r["bench"]: r for r in json.loads(path.read_text())}


def classify(bench: str) -> str:
    if bench in DP_BENCHES:
        return "dp"
    if bench in STENCIL3D:
        return "stencil3d"
    if bench in GEMM_FAMILY:
        return "gemm"
    return "other"


def analyze_regressions(
    noskills: dict[str, dict], all_skills: dict[str, dict], *, threshold: float = 1.15
) -> list[dict]:
    out: list[dict] = []
    for bench, nrow in noskills.items():
        arow = all_skills.get(bench)
        if not arow:
            continue
        nr, ar = gt_ratio(nrow), gt_ratio(arow)
        if nr is None or ar is None or nr <= 0 or ar <= 0:
            continue
        if ar > nr * threshold:
            out.append(
                {
                    "bench": bench,
                    "class": classify(bench),
                    "noskills_ratio": nr,
                    "all_skills_ratio": ar,
                    "regression_factor": ar / nr,
                }
            )
    out.sort(key=lambda x: -x["regression_factor"])
    return out


def patch_skills(data: dict, regressions: list[dict], *, from_ver: str, to_ver: str) -> dict:
    patched = deepcopy(data)
    patched["saved_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S+0000")
    patched["description"] = (
        f"Flash/HPC positive-only skill library ({to_ver}). Auto-patched from {from_ver} "
        f"after {len(regressions)} all_skills regressions vs noskills."
    )

    by_id = {s["id"]: s for s in patched["skills"]}

    def add_guard(sid: str, text: str) -> None:
        if sid not in by_id:
            return
        guards = by_id[sid].setdefault("guards", [])
        if text not in guards:
            guards.append(text)

    dp_hits = [r for r in regressions if r["class"] == "dp"]
    st_hits = [r for r in regressions if r["class"] == "stencil3d"]
    gemm_hits = [r for r in regressions if r["class"] == "gemm"]

    if dp_hits:
        benches = ", ".join(r["bench"] for r in dp_hits)
        add_guard(
            "hpc-dp-recurrence-pipeline-in-place",
            f"round-{to_ver}: mandatory for {benches} — no full-table staging",
        )
        add_guard(
            "hpc-full-workspace-staging-when-fits",
            f"forbidden for regressing DP benches: {benches}",
        )

    if st_hits:
        benches = ", ".join(r["bench"] for r in st_hits)
        add_guard(
            "hpc-stencil-3d-pipeline-k-inner",
            f"round-{to_ver}: mandatory PIPELINE on k for {benches}",
        )

    if gemm_hits:
        benches = ", ".join(r["bench"] for r in gemm_hits)
        add_guard(
            "hpc-gemm-family-block-tile-default",
            f"round-{to_ver}: block tile only for {benches}; no local_C[N][N]",
        )

    # generic regression guard skill (one per round)
    new_id = f"hpc-round-{to_ver}-regression-guards"
    if not any(s.get("id") == new_id for s in patched["skills"]):
        worst = regressions[:8]
        patched["skills"].insert(
            1,
            {
                "id": new_id,
                "kind": "supporting_transformation",
                "confidence": "high",
                "origin": "auto_regression_patch",
                "pattern": "prior all_skills global injection caused latency regressions on specific benches",
                "strategy": (
                    "avoid repeating v1 mistakes: no full-workspace staging on DP/GEMM; "
                    "pipeline 3D stencil inner k; no dependence false; prefer block tiles"
                ),
                "bottleneck_kinds": ["latency_high"],
                "tags": ["regression-fix", "flash-positive", f"round-{to_ver}"],
                "guards": [],
                "required_steps": [
                    f"review regressions: {', '.join(r['bench'] for r in worst)}",
                    "apply class-specific mandatory-first skills before generic staging skills",
                ],
                "template": "",
            },
        )

    chk = by_id.get("hpc-flash-submit-checklist")
    if chk:
        step = f"round-{to_ver}: re-check regressing benches: {', '.join(r['bench'] for r in regressions[:6])}"
        if step not in chk.get("required_steps", []):
            chk.setdefault("required_steps", []).append(step)

    return patched


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stamp", required=True)
    parser.add_argument("--from-version", default="v2")
    parser.add_argument("--to-version", default="v3")
    parser.add_argument("--artifact-prefix", default="flash_hpc_positive_v2")
    parser.add_argument("--report", default="")
    args = parser.parse_args()

    from_ver = args.from_version.lstrip("v")
    to_ver = args.to_version.lstrip("v")
    pc2 = REPO / "artifacts" / "pc2"

    nosk = load_matrix(pc2, f"{args.artifact_prefix}_noskills", args.stamp)
    alls = load_matrix(pc2, f"{args.artifact_prefix}_all_skills", args.stamp)
    regressions = analyze_regressions(nosk, alls)

    src = PKG / f"skills_flash_hpc_positive_v{from_ver}.json"
    dst = PKG / f"skills_flash_hpc_positive_v{to_ver}.json"
    data = json.loads(src.read_text())
    patched = patch_skills(data, regressions, from_ver=from_ver, to_ver=to_ver)
    dst.write_text(json.dumps(patched, indent=2) + "\n")

    from skill_library import SkillLibrary

    SkillLibrary(store_path=dst).load()

    report_path = Path(args.report) if args.report else pc2 / f"flash_hpc_positive_patch_{from_ver}_to_{to_ver}_{args.stamp}.md"
    lines = [
        f"# Skills patch {from_ver} → {to_ver} (stamp {args.stamp})",
        "",
        f"Regressions (all_skills worse than noskills by >15%): **{len(regressions)}**",
        "",
        "| bench | class | noskills | all_skills | factor |",
        "|-------|-------|----------|------------|--------|",
    ]
    for r in regressions:
        lines.append(
            f"| {r['bench']} | {r['class']} | {r['noskills_ratio']:.4f} | "
            f"{r['all_skills_ratio']:.4f} | {r['regression_factor']:.2f}x |"
        )
    report_path.write_text("\n".join(lines) + "\n")

    print(f"regressions={len(regressions)}")
    print(f"patched={dst}")
    print(f"report={report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
