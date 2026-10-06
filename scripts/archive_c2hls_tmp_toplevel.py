#!/usr/bin/env python3
"""Zip whole top-level c2hls_tmp directories (age gate), verify, then remove.

Reclaims project file-count quota by replacing each aged tree with one .zip.
See also scripts/archive_hls_sol_bulk.py (per-sol bulk) — this is coarser.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

TOOL_VERSION = "1"
PARTIAL_SUFFIX = ".partial"
OK_SUFFIX = ".ok"

PROC_ROOT = Path("/proc")
ACTIVE_PROC_RE = re.compile(
    r"(?:^|/)(vitis_hls|vitis-run|vitis_hls\.bin|xelab|xsim|xvlog|xvhdl)(?:$|\s)",
    re.I,
)


def _is_under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


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


def _proc_open_paths(pid_dir: Path):
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


def find_busy_top_dirs(root: Path) -> set[Path]:
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
        candidates = []
        cwd = _proc_cwd(pid_dir)
        if cwd is not None:
            candidates.append(cwd)
        candidates.extend(_proc_open_paths(pid_dir))
        for p in candidates:
            try:
                rel = p.resolve().relative_to(root)
            except ValueError:
                continue
            if not rel.parts:
                continue
            busy.add((root / rel.parts[0]).resolve())
    return busy


def list_aged_top_dirs(
    root: Path,
    min_age_days: float | None = None,
    before_ts: float | None = None,
) -> list[Path]:
    """List top-level dirs with mtime on/before cutoff.

    Prefer ``before_ts`` when set (absolute). Otherwise use
    ``time.time() - min_age_days * 86400``.
    """
    root = root.resolve()
    if before_ts is not None:
        cutoff = before_ts
    else:
        if min_age_days is None:
            min_age_days = 20.0
        cutoff = time.time() - (min_age_days * 86400.0)
    out: list[Path] = []
    with os.scandir(root) as it:
        for e in it:
            if not e.is_dir(follow_symlinks=False):
                continue
            name = e.name
            if name.endswith(".zip") or name.startswith("."):
                continue
            try:
                st = e.stat(follow_symlinks=False)
            except OSError:
                continue
            if st.st_mtime <= cutoff:
                out.append(Path(e.path))
    out.sort(key=lambda p: p.name)
    return out


def before_date_end_ts(date_str: str) -> float:
    """Inclusive end-of-day local timestamp for YYYY-MM-DD."""
    day = datetime.strptime(date_str, "%Y-%m-%d")
    end = day.replace(hour=23, minute=59, second=59)
    return end.timestamp()


def _iter_files(src_dir: Path):
    for dirpath, _dirnames, filenames in os.walk(src_dir, followlinks=False):
        for name in filenames:
            yield Path(dirpath) / name


def zip_path_for(dir_path: Path) -> Path:
    return dir_path.parent / f"{dir_path.name}.zip"


def ok_path_for(zip_path: Path) -> Path:
    return Path(str(zip_path) + OK_SUFFIX)


def count_files(src_dir: Path) -> int:
    return sum(1 for _ in _iter_archive_files(src_dir))


def _iter_archive_files(src_dir: Path):
    """Yield regular files under src_dir suitable for archiving.

    Skips symlinks (including ones that resolve outside the tree, e.g. to
    system Python under ``.venv``), which previously broke ``relative_to``.
    """
    src = src_dir.resolve()
    for fp in _iter_files(src):
        try:
            if fp.is_symlink():
                continue
            if not fp.is_file():
                continue
            # Ensure path stays under src (no weird mounts)
            fp.resolve().relative_to(src)
        except (OSError, ValueError):
            continue
        yield fp


def write_zip(src_dir: Path, zip_path: Path) -> tuple[int, int]:
    """Write zip with members relative to parent (``dirname/...``). Returns (n_files, uncompressed)."""
    partial = Path(str(zip_path) + PARTIAL_SUFFIX)
    if partial.exists():
        partial.unlink()
    src = src_dir.resolve()
    n = 0
    uncompressed = 0
    with zipfile.ZipFile(
        partial, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True
    ) as zf:
        for fp in _iter_archive_files(src):
            try:
                st = fp.stat()
                arc = (Path(src.name) / fp.relative_to(src)).as_posix()
            except (OSError, ValueError):
                continue
            zf.write(fp, arcname=arc)
            n += 1
            uncompressed += st.st_size
    partial.replace(zip_path)
    return n, uncompressed


def verify_zip(zip_path: Path, expected_files: int) -> None:
    with zipfile.ZipFile(zip_path, "r") as zf:
        bad = zf.testzip()
        if bad is not None:
            raise RuntimeError(f"CRC failed for {bad}")
        # count file members only (skip dir entries if any)
        n = sum(1 for i in zf.infolist() if not i.is_dir() and not i.filename.endswith("/"))
        if n != expected_files:
            raise RuntimeError(f"member count mismatch zip={n} expected={expected_files}")


def archive_one_top_dir(
    src_dir: Path,
    *,
    dry_run: bool = False,
    busy: set[Path] | None = None,
) -> dict:
    src = src_dir.resolve()
    zpath = zip_path_for(src)
    opath = ok_path_for(zpath)
    rec: dict = {
        "dir": str(src),
        "zip": str(zpath),
        "status": "",
        "error": None,
        "files": 0,
        "zip_bytes": 0,
    }

    if not src.is_dir():
        if zpath.is_file() and opath.is_file():
            rec["status"] = "skipped"
            rec["error"] = "already_archived"
            return rec
        rec["status"] = "skipped"
        rec["error"] = "missing_dir"
        return rec

    if busy is not None:
        if src in busy:
            rec["status"] = "skipped"
            rec["error"] = "active_process"
            return rec
    elif path_has_active_hls_process(src):
        rec["status"] = "skipped"
        rec["error"] = "active_process"
        return rec

    n_files = count_files(src)
    rec["files"] = n_files
    if n_files == 0:
        # empty dir: still remove to reclaim the directory inode(s)
        if dry_run:
            rec["status"] = "would_archive_empty"
            return rec
        try:
            shutil.rmtree(src)
            rec["status"] = "removed_empty"
        except OSError as exc:
            rec["status"] = "failed"
            rec["error"] = str(exc)
        return rec

    if dry_run:
        rec["status"] = "would_archive"
        return rec

    # Resume: valid zip+ok already — just remove leftover tree
    if zpath.is_file() and opath.is_file():
        try:
            verify_zip(zpath, n_files)
            shutil.rmtree(src)
            rec["status"] = "removed_after_existing_zip"
            rec["zip_bytes"] = zpath.stat().st_size
            return rec
        except Exception:
            # fall through and rebuild
            try:
                opath.unlink(missing_ok=True)
                zpath.unlink(missing_ok=True)
            except OSError:
                pass

    try:
        written, uncompressed = write_zip(src, zpath)
        verify_zip(zpath, written)
        if written != n_files:
            # race: tree changed during zip
            raise RuntimeError(f"file count changed during zip: start={n_files} wrote={written}")
        ok = {
            "version": TOOL_VERSION,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "source_dir": src.name,
            "files": written,
            "uncompressed_bytes": uncompressed,
            "zip_bytes": zpath.stat().st_size,
        }
        opath.write_text(json.dumps(ok, indent=2) + "\n", encoding="utf-8")
        shutil.rmtree(src)
        rec["status"] = "archived"
        rec["files"] = written
        rec["zip_bytes"] = ok["zip_bytes"]
        return rec
    except Exception as exc:  # noqa: BLE001
        partial = Path(str(zpath) + PARTIAL_SUFFIX)
        for p in (partial,):
            try:
                p.unlink(missing_ok=True)
            except OSError:
                pass
        try:
            if zpath.exists() and not opath.exists():
                zpath.unlink(missing_ok=True)
        except OSError:
            pass
        try:
            if opath.exists() and not zpath.exists():
                opath.unlink(missing_ok=True)
        except OSError:
            pass
        rec["status"] = "failed"
        rec["error"] = str(exc)
        return rec


def main(argv: list[str] | None = None) -> int:
    repo_default = Path(__file__).resolve().parents[1] / "c2hls_tmp"
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=repo_default)
    p.add_argument("--min-age-days", type=float, default=20.0)
    p.add_argument(
        "--before-date",
        type=str,
        default=None,
        help="Archive dirs with mtime on/before this local date (YYYY-MM-DD, inclusive). Overrides --min-age-days.",
    )
    p.add_argument("--jobs", type=int, default=4)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--strict", action="store_true")
    p.add_argument(
        "--report",
        type=Path,
        default=Path("archive_c2hls_tmp_toplevel_report.jsonl"),
    )
    args = p.parse_args(argv)

    root = args.root.resolve()
    before_ts = before_date_end_ts(args.before_date) if args.before_date else None
    dirs = list_aged_top_dirs(
        root,
        min_age_days=None if before_ts is not None else args.min_age_days,
        before_ts=before_ts,
    )
    busy = find_busy_top_dirs(root)
    gate = (
        f"before_date={args.before_date} (inclusive EOD)"
        if args.before_date
        else f"min_age_days={args.min_age_days}"
    )
    print(
        f"found aged_dirs={len(dirs)} busy={len(busy)} {gate}",
        file=sys.stderr,
    )

    to_run = dirs
    if args.limit and args.limit > 0:
        to_run = dirs[: args.limit]

    records: list[dict] = []
    jobs = max(1, args.jobs)
    if args.dry_run or jobs == 1:
        for d in to_run:
            records.append(archive_one_top_dir(d, dry_run=args.dry_run, busy=busy))
    else:
        with ThreadPoolExecutor(max_workers=jobs) as ex:
            futs = {
                ex.submit(archive_one_top_dir, d, dry_run=False, busy=busy): d
                for d in to_run
            }
            for fut in as_completed(futs):
                records.append(fut.result())

    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")

    from collections import Counter

    c = Counter(r.get("status") for r in records)
    files = sum(int(r.get("files") or 0) for r in records if r.get("status") == "archived")
    files += sum(
        int(r.get("files") or 0)
        for r in records
        if r.get("status") == "removed_after_existing_zip"
    )
    n_fail = c.get("failed", 0)
    print(
        f"done status={dict(c)} files_in_archived={files} report={args.report}",
        file=sys.stderr,
    )
    if args.strict and n_fail:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
