#!/usr/bin/env python3
"""Build manifest of multistep matrix cells for all-step cosim."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from flash_flow_artifacts import MULTISTEP_OPT_STEPS, resolve_cell_kernel_cpp
from scripts.pc2.flash_cosim_lib import (  # noqa: E402
    PC2_ARTIFACTS,
    CosimCell,
    _artifact_stamp_from_name,
    _cell_id_setup_tag,
    cosim_benchmarks_root,
    cosim_full_size_enabled,
    cosim_run_root,
    make_cell_id,
    write_manifest,
)

MULTISTEP_KERNEL_ROLES = ("phase_b", *MULTISTEP_OPT_STEPS, "selected")
DEFAULT_COSIM_ROOT = PC2_ARTIFACTS / "multistep_cosim"


def _normalize_role(role: str) -> str:
    raw = (role or "selected").strip().lower().replace("-", "_")
    if raw in ("phase_b", "translator", "translated"):
        return "phase_b"
    if raw == "selected":
        return "selected"
    if raw in MULTISTEP_OPT_STEPS:
        return raw
    raise ValueError(f"unknown multistep kernel role: {role}")


def discover_multistep_cells(
    *,
    artifacts_root: Path = PC2_ARTIFACTS,
    artifact_glob: str = "multistep_fixed_cosim_*",
    bench_filter: set[str] | None = None,
    artifact_filter: set[str] | None = None,
    matrix_status: str | None = "ok",
    kernel_roles: tuple[str, ...] = MULTISTEP_KERNEL_ROLES,
) -> list[CosimCell]:
    cells: list[CosimCell] = []
    index = 0
    for matrix_path in sorted(artifacts_root.glob(f"{artifact_glob}/matrix.json")):
        artifact_dir = matrix_path.parent
        artifact_basename = artifact_dir.name
        if artifact_filter and artifact_basename not in artifact_filter:
            continue
        rows = json.loads(matrix_path.read_text())
        if not isinstance(rows, list):
            continue
        for row in rows:
            bench = row.get("bench", "")
            if bench_filter and bench not in bench_filter:
                continue
            if matrix_status and row.get("status") != matrix_status:
                continue
            cell_dir = Path(row.get("cell_dir", ""))
            if not cell_dir.is_dir():
                continue
            manifest_path = cell_dir / f"{bench}_flow_manifest.json"
            flow_manifest = {}
            if manifest_path.is_file():
                try:
                    flow_manifest = json.loads(manifest_path.read_text())
                except json.JSONDecodeError:
                    flow_manifest = {}
            step_success = flow_manifest.get("step_success") or {}

            for role in kernel_roles:
                if role not in ("phase_b", "selected") and not step_success.get(role, True):
                    continue
                kernel_cpp = resolve_cell_kernel_cpp(cell_dir, bench, role)
                if kernel_cpp is None:
                    continue
                setup_tag = cell_dir.name
                cell_id = make_cell_id(
                    artifact_basename,
                    bench,
                    _cell_id_setup_tag(setup_tag, role),
                )
                bench_dir = cosim_benchmarks_root() / bench
                supports_cosim = False
                if (bench_dir / "metadata.json").exists():
                    try:
                        meta = json.loads((bench_dir / "metadata.json").read_text())
                        supports_cosim = bool(meta.get("supports_cosim"))
                    except (OSError, json.JSONDecodeError):
                        supports_cosim = False
                cells.append(
                    CosimCell(
                        index=index,
                        cell_id=cell_id,
                        artifact_dir=str(artifact_dir),
                        artifact_basename=artifact_basename,
                        artifact_stamp=_artifact_stamp_from_name(artifact_basename),
                        matrix_family=row.get("matrix_family", ""),
                        bench=bench,
                        setup_tag=setup_tag,
                        variant=row.get("variant", ""),
                        mode=row.get("mode", "multistep"),
                        model=row.get("model", ""),
                        curation_focus=row.get("curation_focus", ""),
                        skills_json=row.get("skills_json", ""),
                        cell_dir=str(cell_dir),
                        final_cpp=str(kernel_cpp.resolve()),
                        kernel_source=role,
                        source_matrix_status=row.get("status", ""),
                        supports_cosim=supports_cosim,
                    )
                )
                index += 1
    return cells


def multistep_cosim_run_root(stamp: str | None = None) -> Path:
    run_stamp = stamp or os.getenv("C2HLS_MULTISTEP_COSIM_STAMP", "").strip()
    if not run_stamp:
        from datetime import datetime, timezone

        run_stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    root = Path(os.getenv("C2HLS_MULTISTEP_COSIM_ROOT", str(DEFAULT_COSIM_ROOT))) / run_stamp
    root.mkdir(parents=True, exist_ok=True)
    return root


def main() -> int:
    # Multistep cosim always runs at full header problem size (matches csynth N).
    os.environ["C2HLS_FLASH_COSIM_FULL_SIZE"] = "1"
    os.environ["C2HLS_MULTISTEP_COSIM_FULL_SIZE"] = "1"

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stamp", default="")
    parser.add_argument("--artifact-glob", default="multistep_fixed_cosim_*")
    parser.add_argument("--artifact", action="append", default=[])
    parser.add_argument("--bench", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--allow-size-overrides",
        action="store_true",
        help="Debug only: allow cosim_size_overrides from benchmark metadata",
    )
    parser.add_argument(
        "--kernel-role",
        action="append",
        default=[],
        help=f"Limit roles (default all: {', '.join(MULTISTEP_KERNEL_ROLES)})",
    )
    args = parser.parse_args()

    if args.allow_size_overrides:
        os.environ.pop("C2HLS_FLASH_COSIM_FULL_SIZE", None)
        os.environ.pop("C2HLS_MULTISTEP_COSIM_FULL_SIZE", None)
    if args.stamp:
        os.environ["C2HLS_MULTISTEP_COSIM_STAMP"] = args.stamp

    roles = tuple(_normalize_role(r) for r in args.kernel_role) if args.kernel_role else MULTISTEP_KERNEL_ROLES
    cells = discover_multistep_cells(
        artifact_glob=args.artifact_glob,
        bench_filter=set(args.bench) if args.bench else None,
        artifact_filter=set(args.artifact) if args.artifact else None,
        kernel_roles=roles,
    )
    run_root = multistep_cosim_run_root(args.stamp or None)

    summary = {
        "run_root": str(run_root),
        "cell_count": len(cells),
        "kernel_roles": list(roles),
        "artifact_dirs": sorted({c.artifact_basename for c in cells}),
        "benches": sorted({c.bench for c in cells}),
        "cosim_size_mode": "full" if cosim_full_size_enabled() else "override",
    }
    print(json.dumps(summary, indent=2))

    if args.dry_run:
        return 0

    path = write_manifest(
        run_root,
        cells,
        extra={
            "schema": "multistep_cosim_manifest_v1",
            "artifact_glob": args.artifact_glob,
            "kernel_roles": list(roles),
            "cosim_size_mode": "full" if cosim_full_size_enabled() else "override",
        },
    )
    print(f"manifest: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
