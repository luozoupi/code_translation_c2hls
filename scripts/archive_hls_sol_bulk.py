#!/usr/bin/env python3
"""Archive bulky HLS sol artifacts in place to reclaim inodes.

See docs/superpowers/specs/2026-07-25-archive-hls-sol-bulk-design.md
"""

from __future__ import annotations

import fnmatch
import json
import os
import re
import shutil
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator

PROC_ROOT = Path("/proc")

TOOL_VERSION = "1"
ZIP_NAME = "sol_bulk.zip"
OK_NAME = ".sol_bulk_ok"
PARTIAL_SUFFIX = ".partial"

ACTIVE_PROC_RE = re.compile(
    r"(?:^|/)(vitis_hls|vitis-run|vitis_hls\.bin|xelab|xsim|xvlog|xvhdl)(?:$|\s)",
    re.I,
)

SIDECAR_GLOBS = (
    "*.aps",
    "*.directive",
    "*.cfg",
    "*_data.json",
    "vitis-comp.json",
)


@dataclass
class SolPlan:
    sol_dir: Path
    zip_paths: list[Path] = field(default_factory=list)
    delete_paths: list[Path] = field(default_factory=list)
    keep_paths: list[Path] = field(default_factory=list)


def _is_under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _iter_files(root: Path) -> Iterator[Path]:
    if not root.exists():
        return
    for dirpath, dirnames, filenames in os.walk(root):
        # do not descend into our own archive outputs
        dirnames[:] = [d for d in dirnames if d not in (ZIP_NAME,)]
        for name in filenames:
            if name in (ZIP_NAME, OK_NAME) or name.endswith(PARTIAL_SUFFIX):
                continue
            yield Path(dirpath) / name


def _match_any(name: str, patterns: Iterable[str]) -> bool:
    return any(fnmatch.fnmatch(name, pat) for pat in patterns)


def _rel(sol: Path, path: Path) -> str:
    return path.relative_to(sol).as_posix()


def classify_sol_members(sol_dir: Path) -> SolPlan:
    """Classify files under sol_dir into zip / delete / keep sets.

    Rules (spec):
    - Always keep: syn/**, .autopilot/db/*.xml, sol*.log, sim/report/**,
      csim/report/**, **/*.result.lat.rb (cosim cycle harvest)
    - Zip for backup: keep-set that is under syn/** PLUS all delete-set members
      AND non-kept .autopilot files (delete-set)
    - Delete after verify: .autopilot except db/*.xml; .debug; impl;
      sim except report and *.result.lat.rb; csim except report;
      sol-root sidecars (not sol*.log)
    """
    sol = sol_dir.resolve()
    plan = SolPlan(sol_dir=sol)
    zip_set: set[Path] = set()
    delete_set: set[Path] = set()
    keep_set: set[Path] = set()

    def mark_keep(p: Path) -> None:
        keep_set.add(p)

    def mark_zip(p: Path) -> None:
        zip_set.add(p)

    def mark_delete(p: Path) -> None:
        delete_set.add(p)
        zip_set.add(p)

    def _is_result_lat_rb(p: Path) -> bool:
        return p.name.endswith(".result.lat.rb")

    # syn/** : zip + keep, never delete
    syn = sol / "syn"
    for p in _iter_files(syn):
        mark_keep(p)
        mark_zip(p)

    # .autopilot/**
    ap = sol / ".autopilot"
    for p in _iter_files(ap):
        rel = _rel(sol, p)
        if rel.startswith(".autopilot/db/") and p.suffix.lower() == ".xml":
            mark_keep(p)
        else:
            mark_delete(p)

    # .debug/**, impl/**
    for sub in (".debug", "impl"):
        for p in _iter_files(sol / sub):
            mark_delete(p)

    # sim/** keep report + *.result.lat.rb; delete rest (+zip)
    sim = sol / "sim"
    for p in _iter_files(sim):
        rel = _rel(sol, p)
        if rel.startswith("sim/report/") or _is_result_lat_rb(p):
            mark_keep(p)
        else:
            mark_delete(p)

    # csim/** keep report + *.result.lat.rb; delete rest
    csim = sol / "csim"
    for p in _iter_files(csim):
        rel = _rel(sol, p)
        if rel.startswith("csim/report/") or _is_result_lat_rb(p):
            mark_keep(p)
        else:
            mark_delete(p)

    # sol-root sidecars
    if sol.is_dir():
        for p in sol.iterdir():
            if not p.is_file():
                continue
            name = p.name
            if name in (ZIP_NAME, OK_NAME) or name.endswith(PARTIAL_SUFFIX):
                continue
            if fnmatch.fnmatch(name, "sol*.log"):
                mark_keep(p)
                continue
            if _match_any(name, SIDECAR_GLOBS) or (
                name.endswith(".json") and p.parent == sol
            ):
                mark_delete(p)

    plan.keep_paths = sorted(keep_set)
    plan.zip_paths = sorted(zip_set)
    plan.delete_paths = sorted(delete_set)
    return plan


def batch_root_for_sol(sol_dir: Path, root: Path) -> Path:
    """Top-level directory under root that contains sol_dir."""
    sol = sol_dir.resolve()
    root = root.resolve()
    rel = sol.relative_to(root)
    return (root / rel.parts[0]).resolve()


def sol_age_ok(sol_dir: Path, min_age_hours: float) -> bool:
    mtime = sol_dir.stat().st_mtime
    return (time.time() - mtime) >= (min_age_hours * 3600.0)


def already_archived(sol_dir: Path) -> bool:
    return (sol_dir / ZIP_NAME).is_file() and (sol_dir / OK_NAME).is_file()


def _read_proc_cmdline(pid_dir: Path) -> str:
    try:
        raw = (pid_dir / "cmdline").read_bytes()
    except OSError:
        return ""
    return raw.replace(b"\x00", b" ").decode("utf-8", "replace")


def _proc_cwd(pid_dir: Path) -> Path | None:
    try:
        return (pid_dir / "cwd").resolve()
    except OSError:
        return None


def _proc_open_paths(pid_dir: Path) -> Iterator[Path]:
    fd_dir = pid_dir / "fd"
    if not fd_dir.is_dir():
        return
    try:
        for fd in fd_dir.iterdir():
            try:
                yield fd.resolve()
            except OSError:
                continue
    except OSError:
        return


def path_has_active_hls_process(batch_dir: Path) -> bool:
    """True if an HLS/sim process has cwd or open file under batch_dir."""
    batch = batch_dir.resolve()
    if not PROC_ROOT.is_dir():
        return False
    for pid_dir in PROC_ROOT.iterdir():
        if not pid_dir.name.isdigit():
            continue
        cmd = _read_proc_cmdline(pid_dir)
        if not ACTIVE_PROC_RE.search(cmd):
            continue
        cwd = _proc_cwd(pid_dir)
        if cwd is not None and _is_under(cwd, batch):
            return True
        for p in _proc_open_paths(pid_dir):
            if _is_under(p, batch):
                return True
    return False


def find_busy_batch_roots(root: Path) -> set[Path]:
    busy: set[Path] = set()
    root = root.resolve()
    if not PROC_ROOT.is_dir():
        return busy
    for pid_dir in PROC_ROOT.iterdir():
        if not pid_dir.name.isdigit():
            continue
        cmd = _read_proc_cmdline(pid_dir)
        if not ACTIVE_PROC_RE.search(cmd):
            continue
        candidates: list[Path] = []
        cwd = _proc_cwd(pid_dir)
        if cwd is not None:
            candidates.append(cwd)
        candidates.extend(_proc_open_paths(pid_dir))
        for p in candidates:
            try:
                rel = p.resolve().relative_to(root)
            except ValueError:
                continue
            busy.add((root / rel.parts[0]).resolve())
    return busy


def skip_reason_for_sol(
    sol_dir: Path,
    root: Path,
    min_age_hours: float,
    busy_batches: set[Path] | None = None,
) -> str | None:
    if already_archived(sol_dir):
        return "already_archived"
    if not sol_age_ok(sol_dir, min_age_hours):
        return "too_young"
    batch = batch_root_for_sol(sol_dir, root)
    if busy_batches is not None:
        if batch in busy_batches:
            return "active_process"
    elif path_has_active_hls_process(batch):
        return "active_process"
    plan = classify_sol_members(sol_dir)
    if not plan.delete_paths:
        return "nothing_to_reclaim"
    return None


class VerifyError(RuntimeError):
    pass


def verify_zip_members(zip_path: Path, members: list[Path], sol_dir: Path) -> None:
    with zipfile.ZipFile(zip_path, "r") as zf:
        bad = zf.testzip()
        if bad is not None:
            raise VerifyError(f"CRC failed for {bad}")
        infos = {i.filename: i for i in zf.infolist()}
        for p in members:
            rel = _rel(sol_dir, p)
            if rel not in infos:
                raise VerifyError(f"missing member {rel}")
            if infos[rel].file_size != p.stat().st_size:
                raise VerifyError(
                    f"size mismatch {rel}: zip={infos[rel].file_size} disk={p.stat().st_size}"
                )


def _write_zip(sol_dir: Path, members: list[Path]) -> Path:
    partial = sol_dir / (ZIP_NAME + PARTIAL_SUFFIX)
    if partial.exists():
        partial.unlink()
    with zipfile.ZipFile(partial, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in members:
            zf.write(p, arcname=_rel(sol_dir, p))
    return partial


def _delete_paths(paths: list[Path]) -> int:
    n = 0
    for p in sorted(paths, key=lambda x: len(x.parts), reverse=True):
        try:
            if p.is_file() or p.is_symlink():
                p.unlink()
                n += 1
        except OSError:
            pass
    return n


def _prune_empty_dirs(sol_dir: Path, roots: tuple[str, ...]) -> None:
    """Remove empty directories bottom-up under bulk roots.

    Always try ``rmdir`` (succeeds only if empty). Do not trust ``os.walk``'s
    ``dirnames`` list — it can be stale after children were removed.
    Multi-pass for NFS/Lustre delayed directory updates.
    """
    for _ in range(3):
        removed = False
        for name in roots:
            base = sol_dir / name
            if not base.exists():
                continue
            for dirpath, _dirnames, _filenames in os.walk(base, topdown=False):
                try:
                    Path(dirpath).rmdir()
                    removed = True
                except OSError:
                    pass
        if not removed:
            break


def _rmtree_if_exists(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)


def archive_one_sol(sol_dir: Path, dry_run: bool = False) -> dict:
    sol = sol_dir.resolve()
    plan = classify_sol_members(sol)
    rec = {
        "sol": str(sol),
        "zip_members": len(plan.zip_paths),
        "delete_members": len(plan.delete_paths),
        "keep_members": len(plan.keep_paths),
        "status": "",
        "error": None,
        "files_deleted": 0,
    }
    if not plan.delete_paths:
        rec["status"] = "skipped"
        rec["error"] = "nothing_to_reclaim"
        return rec
    if dry_run:
        rec["status"] = "would_archive"
        rec["bytes_delete_est"] = sum(
            p.stat().st_size for p in plan.delete_paths if p.is_file()
        )
        return rec
    partial = None
    try:
        members = [p for p in plan.zip_paths if p.is_file()]
        delete_files = [p for p in plan.delete_paths if p.is_file()]
        partial = _write_zip(sol, members)
        verify_zip_members(partial, members, sol)
        final = sol / ZIP_NAME
        partial.replace(final)
        partial = None
        ok = {
            "version": TOOL_VERSION,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "zip_members": len(members),
            "delete_members": len(delete_files),
            "zip_bytes": final.stat().st_size,
            "uncompressed_bytes": sum(p.stat().st_size for p in members),
        }
        (sol / OK_NAME).write_text(json.dumps(ok, indent=2) + "\n", encoding="utf-8")
        deleted = _delete_paths(delete_files)
        # Full trees with no keep-set: remove wholesale (avoids empty-dir leftovers).
        _rmtree_if_exists(sol / ".debug")
        _rmtree_if_exists(sol / "impl")
        _prune_empty_dirs(sol, (".autopilot", "sim", "csim"))
        rec["files_deleted"] = deleted
        rec["status"] = "archived"
        return rec
    except Exception as exc:  # noqa: BLE001 — per-sol isolation
        if partial is not None and partial.exists():
            try:
                partial.unlink()
            except OSError:
                pass
        final = sol / ZIP_NAME
        if final.exists() and not (sol / OK_NAME).exists():
            try:
                final.unlink()
            except OSError:
                pass
        rec["status"] = "failed"
        rec["error"] = str(exc)
        return rec


def discover_sol_dirs(root: Path, batch_prefix: str | None = None) -> list[Path]:
    root = root.resolve()
    out: list[Path] = []
    for dirpath, dirnames, _filenames in os.walk(root):
        if ".autopilot" in dirnames:
            sol = Path(dirpath)
            if batch_prefix is None or sol.relative_to(root).parts[0].startswith(
                batch_prefix
            ):
                out.append(sol)
            dirnames.remove(".autopilot")
    out.sort()
    return out


def main(argv: list[str] | None = None) -> int:
    import argparse

    repo_default = Path(__file__).resolve().parents[1] / "c2hls_tmp"
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=repo_default)
    p.add_argument("--min-age-hours", type=float, default=6.0)
    p.add_argument("--jobs", type=int, default=4)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--batch-prefix", type=str, default=None)
    p.add_argument("--strict", action="store_true")
    p.add_argument(
        "--report", type=Path, default=Path("archive_hls_sol_bulk_report.jsonl")
    )
    args = p.parse_args(argv)

    root = args.root.resolve()
    sols = discover_sol_dirs(root, batch_prefix=args.batch_prefix)
    busy = find_busy_batch_roots(root)
    to_run: list[Path] = []
    records: list[dict] = []

    for sol in sols:
        reason = skip_reason_for_sol(sol, root, args.min_age_hours, busy_batches=busy)
        if reason:
            records.append({"sol": str(sol), "status": "skipped", "error": reason})
            continue
        to_run.append(sol)
        if args.limit and len(to_run) >= args.limit:
            break

    # ThreadPool: zip/delete is mostly filesystem I/O; avoids ProcessPool
    # pickling failures when this file is run as ``python scripts/...py`` (__main__).
    jobs = max(1, args.jobs)
    if args.dry_run or jobs == 1:
        for sol in to_run:
            records.append(archive_one_sol(sol, dry_run=args.dry_run))
    else:
        with ThreadPoolExecutor(max_workers=jobs) as ex:
            futs = {
                ex.submit(archive_one_sol, sol, args.dry_run): sol for sol in to_run
            }
            for fut in as_completed(futs):
                records.append(fut.result())

    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")

    n_fail = sum(1 for r in records if r.get("status") == "failed")
    n_arch = sum(
        1 for r in records if r.get("status") in ("archived", "would_archive")
    )
    n_skip = sum(1 for r in records if r.get("status") == "skipped")
    print(
        f"done archived/would={n_arch} skipped={n_skip} failed={n_fail} report={args.report}",
        file=sys.stderr,
    )
    if args.strict and n_fail:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
