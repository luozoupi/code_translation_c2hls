# Archive HLS sol bulk Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a resumable CLI that zips bulky finished HLS `sol*` trees under `c2hls_tmp` in place, verifies the zip, then deletes only reclaimable members while leaving pipeline keep-set files on disk.

**Architecture:** One Python module/CLI (`scripts/archive_hls_sol_bulk.py`) with pure helpers for classify / gates / zip-verify-delete, plus pytest coverage on synthetic fixtures. Discovery walks `.autopilot` parents; each sol gets `sol_bulk.zip` + `.sol_bulk_ok`.

**Tech Stack:** Python 3 stdlib (`pathlib`, `zipfile`, `argparse`, `concurrent.futures`, `json`, `os`, `time`); pytest; optional `/proc` for process open-file checks on Linux.

**Spec:** `docs/superpowers/specs/2026-07-25-archive-hls-sol-bulk-design.md`

---

## File map

| File | Responsibility |
|------|----------------|
| `scripts/archive_hls_sol_bulk.py` | All library helpers + CLI entrypoint |
| `tests/test_archive_hls_sol_bulk.py` | Unit + small integration tests |

No changes to `hls_feedback.py` / eval in v1.

---

### Task 1: Member classification (keep / zip / delete)

**Files:**
- Create: `scripts/archive_hls_sol_bulk.py`
- Test: `tests/test_archive_hls_sol_bulk.py`

- [ ] **Step 1: Write failing tests for classification**

```python
"""Tests for scripts/archive_hls_sol_bulk.py."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.archive_hls_sol_bulk import (
    classify_sol_members,
    ZIP_NAME,
    OK_NAME,
)


def _touch(p: Path, text: str = "x") -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


def _make_sol(tmp: Path) -> Path:
    sol = tmp / "batch_x" / "bench" / "hls_synth__a" / "hls_proj" / "sol1"
    _touch(sol / ".autopilot" / "db" / "burst.xml", "<burst/>")
    _touch(sol / ".autopilot" / "db" / "fe_messages.xml", "<fe/>")
    _touch(sol / ".autopilot" / "db" / "other.bin", "bin")
    _touch(sol / ".autopilot" / "huge.dat", "bulk")
    _touch(sol / ".debug" / "a.debug", "d")
    _touch(sol / "impl" / "ip" / "x.v", "v")
    _touch(sol / "syn" / "report" / "csynth.rpt", "rpt")
    _touch(sol / "syn" / "verilog" / "k.v", "sv")
    _touch(sol / "sim" / "report" / "k_cosim.rpt", "cr")
    _touch(sol / "sim" / "verilog" / "xelab.log", "xl")
    _touch(sol / "csim" / "report" / "csim.rpt", "csr")
    _touch(sol / "csim" / "build" / "a.o", "o")
    _touch(sol / "hls_config.cfg", "cfg")
    _touch(sol / "sol1.aps", "aps")
    _touch(sol / "sol1.directive", "dir")
    _touch(sol / "sol1_data.json", "{}")
    _touch(sol / "vitis-comp.json", "{}")
    _touch(sol / "sol1.log", "log")
    return sol


def test_classify_keep_zip_delete_sets(tmp_path: Path) -> None:
    sol = _make_sol(tmp_path)
    plan = classify_sol_members(sol)

    keep = {p.relative_to(sol).as_posix() for p in plan.keep_paths}
    zip_members = {p.relative_to(sol).as_posix() for p in plan.zip_paths}
    delete = {p.relative_to(sol).as_posix() for p in plan.delete_paths}

    assert "syn/report/csynth.rpt" in keep
    assert "syn/verilog/k.v" in keep
    assert ".autopilot/db/burst.xml" in keep
    assert ".autopilot/db/fe_messages.xml" in keep
    assert "sol1.log" in keep
    assert "sim/report/k_cosim.rpt" in keep
    assert "csim/report/csim.rpt" in keep

    # syn is zipped for backup but never deleted
    assert "syn/report/csynth.rpt" in zip_members
    assert "syn/report/csynth.rpt" not in delete

    assert ".autopilot/huge.dat" in zip_members
    assert ".autopilot/huge.dat" in delete
    assert ".autopilot/db/other.bin" in delete
    assert ".autopilot/db/burst.xml" not in delete

    assert ".debug/a.debug" in delete
    assert "impl/ip/x.v" in delete
    assert "sim/verilog/xelab.log" in delete
    assert "csim/build/a.o" in delete
    assert "hls_config.cfg" in delete
    assert "sol1.aps" in delete
    assert "sol1.log" not in delete
    assert ZIP_NAME not in delete
    assert OK_NAME not in delete


def test_classify_empty_delete_set_when_only_keep(tmp_path: Path) -> None:
    sol = tmp_path / "sol1"
    _touch(sol / ".autopilot" / "db" / "burst.xml")
    _touch(sol / "syn" / "report" / "csynth.rpt")
    _touch(sol / "sol1.log")
    plan = classify_sol_members(sol)
    assert plan.delete_paths == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
cd /scratch/hpc-prf-llmfpga/asa582/projects/c2hls
python -m pytest tests/test_archive_hls_sol_bulk.py::test_classify_keep_zip_delete_sets -v
```

Expected: `ImportError` or `classify_sol_members` not found.

- [ ] **Step 3: Implement classification helpers**

Create `scripts/archive_hls_sol_bulk.py` with at least:

```python
#!/usr/bin/env python3
"""Archive bulky HLS sol artifacts in place to reclaim inodes.

See docs/superpowers/specs/2026-07-25-archive-hls-sol-bulk-design.md
"""

from __future__ import annotations

import fnmatch
import json
import os
import re
import sys
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator

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
    - Always keep: syn/**, .autopilot/db/*.xml, sol*.log, sim/report/**, csim/report/**
    - Zip for backup: keep-set that is under syn/** PLUS all delete-set members
      AND non-kept .autopilot files (delete-set)
    - Delete after verify: .autopilot except db/*.xml; .debug; impl;
      sim except report; csim except report; sol-root sidecars (not sol*.log)
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

    # sim/** keep report, delete rest (+zip)
    sim = sol / "sim"
    for p in _iter_files(sim):
        rel = _rel(sol, p)
        if rel.startswith("sim/report/"):
            mark_keep(p)
        else:
            mark_delete(p)

    # csim/** same
    csim = sol / "csim"
    for p in _iter_files(csim):
        rel = _rel(sol, p)
        if rel.startswith("csim/report/"):
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
```

Ensure `scripts/` is importable in tests by adding to `conftest` **or** using path insert. Prefer making the module importable without package install:

At top of `tests/test_archive_hls_sol_bulk.py` (if `from scripts...` fails), use:

```python
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.archive_hls_sol_bulk import (  # noqa: E402
    classify_sol_members,
    ZIP_NAME,
    OK_NAME,
)
```

If `scripts` is not a package, either add empty `scripts/__init__.py` **or** import via importlib from file path. Prefer adding:

```python
# scripts/__init__.py  (empty)  — only if repo does not already treat scripts as non-package
```

Check: if other tests import `scripts.X`, follow that pattern; else load by path:

```python
import importlib.util

def _load():
    path = ROOT / "scripts" / "archive_hls_sol_bulk.py"
    spec = importlib.util.spec_from_file_location("archive_hls_sol_bulk", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod

mod = _load()
classify_sol_members = mod.classify_sol_members
ZIP_NAME = mod.ZIP_NAME
OK_NAME = mod.OK_NAME
```

Use the **importlib** approach to avoid adding `scripts/__init__.py` unless already present.

- [ ] **Step 4: Run classification tests**

```bash
python -m pytest tests/test_archive_hls_sol_bulk.py::test_classify_keep_zip_delete_sets tests/test_archive_hls_sol_bulk.py::test_classify_empty_delete_set_when_only_keep -v
```

Expected: PASS.

- [ ] **Step 5: Commit** (only if user requested commits; otherwise skip)

```bash
git add scripts/archive_hls_sol_bulk.py tests/test_archive_hls_sol_bulk.py
git commit -m "$(cat <<'EOF'
feat: classify HLS sol bulk members for in-place archive

EOF
)"
```

---

### Task 2: Age + process + already-done gates

**Files:**
- Modify: `scripts/archive_hls_sol_bulk.py`
- Test: `tests/test_archive_hls_sol_bulk.py`

- [ ] **Step 1: Write failing gate tests**

```python
import time
from scripts.archive_hls_sol_bulk import (
    sol_age_ok,
    already_archived,
    batch_root_for_sol,
    skip_reason_for_sol,
)


def test_sol_age_ok(tmp_path: Path) -> None:
    sol = tmp_path / "sol1"
    sol.mkdir()
    assert not sol_age_ok(sol, min_age_hours=6)
    old = time.time() - 7 * 3600
    os.utime(sol, (old, old))
    assert sol_age_ok(sol, min_age_hours=6)


def test_already_archived(tmp_path: Path) -> None:
    sol = tmp_path / "sol1"
    sol.mkdir()
    assert not already_archived(sol)
    (sol / ZIP_NAME).write_bytes(b"PK\x05\x06" + b"\x00" * 18)  # empty zip-ish may fail later
    # Prefer writing a real empty zip:
    import zipfile
    with zipfile.ZipFile(sol / ZIP_NAME, "w") as zf:
        zf.writestr("marker.txt", "ok")
    (sol / OK_NAME).write_text("{}", encoding="utf-8")
    assert already_archived(sol)


def test_batch_root_for_sol(tmp_path: Path) -> None:
    root = tmp_path / "c2hls_tmp"
    sol = root / "batch_a" / "b" / "hls_proj" / "sol1"
    sol.mkdir(parents=True)
    assert batch_root_for_sol(sol, root) == (root / "batch_a").resolve()
```

For process check, unit-test the helper with a monkeypatched `/proc` fixture **or** test `path_has_active_hls_process` returns False on empty fake proc:

```python
def test_path_has_active_hls_process_false_on_empty(tmp_path: Path, monkeypatch) -> None:
    from scripts import archive_hls_sol_bulk as m
    monkeypatch.setattr(m, "PROC_ROOT", tmp_path)
    assert m.path_has_active_hls_process(tmp_path / "batch") is False
```

- [ ] **Step 2: Run to see fail**

```bash
python -m pytest tests/test_archive_hls_sol_bulk.py -k "age_ok or already_archived or batch_root or active_hls" -v
```

- [ ] **Step 3: Implement gates**

```python
PROC_ROOT = Path("/proc")


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
```

Precompute `busy_batches` once per run for performance:

```python
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
```

- [ ] **Step 4: Run gate tests — expect PASS**

```bash
python -m pytest tests/test_archive_hls_sol_bulk.py -k "age_ok or already_archived or batch_root or active_hls or skip_reason" -v
```

- [ ] **Step 5: Commit** (if user requested)

---

### Task 3: Zip → verify → delete for one sol

**Files:**
- Modify: `scripts/archive_hls_sol_bulk.py`
- Test: `tests/test_archive_hls_sol_bulk.py`

- [ ] **Step 1: Write failing archive tests**

```python
from scripts.archive_hls_sol_bulk import archive_one_sol


def test_archive_one_sol_deletes_bulk_keeps_pipeline_files(tmp_path: Path) -> None:
    sol = _make_sol(tmp_path)
    # age gate bypassed by calling archive_one_sol directly
    result = archive_one_sol(sol, dry_run=False)
    assert result["status"] == "archived"

    assert (sol / "sol_bulk.zip").is_file()
    assert (sol / ".sol_bulk_ok").is_file()
    assert (sol / "syn" / "report" / "csynth.rpt").is_file()
    assert (sol / ".autopilot" / "db" / "burst.xml").is_file()
    assert (sol / "sol1.log").is_file()
    assert (sol / "sim" / "report" / "k_cosim.rpt").is_file()

    assert not (sol / ".autopilot" / "huge.dat").exists()
    assert not (sol / ".debug" / "a.debug").exists()
    assert not (sol / "impl" / "ip" / "x.v").exists()
    assert not (sol / "sim" / "verilog" / "xelab.log").exists()
    assert not (sol / "hls_config.cfg").exists()

    with zipfile.ZipFile(sol / "sol_bulk.zip") as zf:
        names = set(zf.namelist())
        assert "syn/report/csynth.rpt" in names
        assert ".autopilot/huge.dat" in names
        assert zf.testzip() is None


def test_archive_dry_run_no_changes(tmp_path: Path) -> None:
    sol = _make_sol(tmp_path)
    result = archive_one_sol(sol, dry_run=True)
    assert result["status"] == "would_archive"
    assert (sol / ".autopilot" / "huge.dat").is_file()
    assert not (sol / "sol_bulk.zip").exists()


def test_verify_refuses_delete_on_corrupt_zip(tmp_path: Path, monkeypatch) -> None:
    sol = _make_sol(tmp_path)
    import scripts.archive_hls_sol_bulk as m

    real_verify = m.verify_zip_members

    def bad_verify(zip_path, members):
        raise m.VerifyError("forced")

    monkeypatch.setattr(m, "verify_zip_members", bad_verify)
    result = archive_one_sol(sol, dry_run=False)
    assert result["status"] == "failed"
    assert (sol / ".autopilot" / "huge.dat").is_file()
```

- [ ] **Step 2: Run — expect fail**

```bash
python -m pytest tests/test_archive_hls_sol_bulk.py -k "archive_one_sol or dry_run or corrupt" -v
```

- [ ] **Step 3: Implement zip/verify/delete**

```python
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
    final = sol_dir / ZIP_NAME
    if partial.exists():
        partial.unlink()
    with zipfile.ZipFile(partial, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in members:
            zf.write(p, arcname=_rel(sol_dir, p))
    return partial


def _delete_paths(paths: list[Path]) -> int:
    n = 0
    # delete files first
    for p in sorted(paths, key=lambda x: len(x.parts), reverse=True):
        try:
            if p.is_file() or p.is_symlink():
                p.unlink()
                n += 1
        except OSError:
            pass
    # prune empty dirs under known bulk roots
    return n


def _prune_empty_dirs(sol_dir: Path, roots: tuple[str, ...]) -> None:
    for name in roots:
        base = sol_dir / name
        if not base.exists():
            continue
        for dirpath, dirnames, filenames in os.walk(base, topdown=False):
            if not dirnames and not filenames:
                try:
                    Path(dirpath).rmdir()
                except OSError:
                    pass


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
        rec["bytes_delete_est"] = sum(p.stat().st_size for p in plan.delete_paths if p.is_file())
        return rec
    partial = None
    try:
        # refresh sizes at zip time — files must still exist
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
        _prune_empty_dirs(sol, (".autopilot", ".debug", "impl", "sim", "csim"))
        # ensure keep xmls still present
        rec["files_deleted"] = deleted
        rec["status"] = "archived"
        return rec
    except Exception as exc:  # noqa: BLE001 — per-sol isolation
        if partial is not None and partial.exists():
            try:
                partial.unlink()
            except OSError:
                pass
        # do not remove final zip if verify failed before replace; if replace happened
        # but delete failed, leave zip+ok? Spec: on failure leave originals.
        # If OK not written yet, remove final zip if present without OK.
        final = sol / ZIP_NAME
        if final.exists() and not (sol / OK_NAME).exists():
            try:
                final.unlink()
            except OSError:
                pass
        rec["status"] = "failed"
        rec["error"] = str(exc)
        return rec
```

**Important:** `verify_zip_members` must run **before** deleting, while source files still exist (size check). After verify+rename, delete delete-set only.

- [ ] **Step 4: Run archive tests — expect PASS**

```bash
python -m pytest tests/test_archive_hls_sol_bulk.py -k "archive_one_sol or dry_run or corrupt" -v
```

- [ ] **Step 5: Commit** (if user requested)

---

### Task 4: Discovery, CLI, parallelism, JSONL report

**Files:**
- Modify: `scripts/archive_hls_sol_bulk.py`

- [ ] **Step 1: Write discovery + CLI smoke test**

```python
def test_discover_sol_dirs(tmp_path: Path) -> None:
    from scripts.archive_hls_sol_bulk import discover_sol_dirs
    a = tmp_path / "batch1" / "x" / "hls_proj" / "sol1" / ".autopilot"
    b = tmp_path / "batch2" / "y" / "hls_proj" / "sol1" / ".autopilot"
    a.mkdir(parents=True)
    b.mkdir(parents=True)
    found = discover_sol_dirs(tmp_path, batch_prefix="batch1")
    assert len(found) == 1
    assert found[0].name == "sol1"
```

- [ ] **Step 2: Implement discovery + main**

```python
def discover_sol_dirs(root: Path, batch_prefix: str | None = None) -> list[Path]:
    root = root.resolve()
    out: list[Path] = []
    for dirpath, dirnames, _filenames in os.walk(root):
        if ".autopilot" in dirnames:
            sol = Path(dirpath)
            if batch_prefix:
                try:
                    top = sol.relative_to(root).parts[0]
                except ValueError:
                    continue
                if not top.startswith(batch_prefix):
                    # still allow walking but skip collect
                    pass
                else:
                    out.append(sol)
                    dirnames.remove(".autopilot")
                    continue
            else:
                out.append(sol)
            dirnames.remove(".autopilot")  # do not walk into autopilot
    out.sort()
    return out


# Fix batch_prefix filtering cleanly:
def discover_sol_dirs(root: Path, batch_prefix: str | None = None) -> list[Path]:
    root = root.resolve()
    out: list[Path] = []
    for dirpath, dirnames, _filenames in os.walk(root):
        if ".autopilot" in dirnames:
            sol = Path(dirpath)
            if batch_prefix is None or sol.relative_to(root).parts[0].startswith(batch_prefix):
                out.append(sol)
            dirnames.remove(".autopilot")
    out.sort()
    return out


def _worker(args: tuple) -> dict:
    sol_s, dry_run = args
    return archive_one_sol(Path(sol_s), dry_run=dry_run)


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
    p.add_argument("--report", type=Path, default=Path("archive_hls_sol_bulk_report.jsonl"))
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

    # Process archive jobs
    jobs = max(1, args.jobs)
    if args.dry_run or jobs == 1:
        for sol in to_run:
            records.append(archive_one_sol(sol, dry_run=args.dry_run))
    else:
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            futs = {ex.submit(archive_one_sol, sol, args.dry_run): sol for sol in to_run}
            for fut in as_completed(futs):
                records.append(fut.result())

    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")

    n_fail = sum(1 for r in records if r.get("status") == "failed")
    n_arch = sum(1 for r in records if r.get("status") in ("archived", "would_archive"))
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
```

Fix the duplicated `discover_sol_dirs` in the plan when implementing — keep **one** clean version.

- [ ] **Step 3: Run unit tests for discovery**

```bash
python -m pytest tests/test_archive_hls_sol_bulk.py -v
```

Expected: all PASS.

- [ ] **Step 4: CLI dry-run smoke on tiny fixture**

```bash
python scripts/archive_hls_sol_bulk.py --root /tmp/fake --dry-run --jobs 1
```

(or point `--root` at a pytest-built tree via a small inline script). Prefer a test that invokes `main([...])`.

```python
def test_main_dry_run(tmp_path: Path) -> None:
    sol = _make_sol(tmp_path)
    old = time.time() - 7 * 3600
    os.utime(sol, (old, old))
    report = tmp_path / "r.jsonl"
    rc = main([
        "--root", str(tmp_path),
        "--dry-run",
        "--min-age-hours", "6",
        "--jobs", "1",
        "--report", str(report),
    ])
    assert rc == 0
    assert report.is_file()
```

- [ ] **Step 5: Commit** (if user requested)

---

### Task 5: Manual pilot checklist (operator)

Not code — run after implementation on the real tree.

- [ ] **Step 1: Dry-run full tree**

```bash
cd /scratch/hpc-prf-llmfpga/asa582/projects/c2hls
python scripts/archive_hls_sol_bulk.py \
  --root /scratch/hpc-prf-llmfpga/asa582/projects/c2hls/c2hls_tmp \
  --dry-run --jobs 1 \
  --report /tmp/archive_hls_sol_bulk_dry.jsonl
```

Inspect counts of `would_archive` vs `skipped` reasons.

- [ ] **Step 2: Live pilot one old finished batch**

```bash
python scripts/archive_hls_sol_bulk.py \
  --root .../c2hls_tmp \
  --batch-prefix batch_parallel_chathls_fd_ds_skills_20260717 \
  --limit 5 --jobs 2 \
  --report /tmp/archive_hls_sol_bulk_pilot.jsonl
```

Spot-check one sol: `syn/`, `.autopilot/db/*.xml`, `sol1.log` present; bulk gone; `sol_bulk.zip` tests clean (`python -c 'import zipfile; print(zipfile.ZipFile("sol_bulk.zip").testzip())'`).

- [ ] **Step 3: Full run** when pilot looks good

```bash
python scripts/archive_hls_sol_bulk.py --jobs 4 --report /tmp/archive_hls_sol_bulk_full.jsonl
```

Check `df -i /scratch` inode drop.

---

## Self-review vs spec

| Spec requirement | Task |
|------------------|------|
| Per-sol `sol_bulk.zip` | Task 3 |
| Keep syn, db xml, sol*.log, sim/csim report | Task 1 |
| Zip syn but never delete | Task 1 + 3 |
| Zip+delete autopilot bulk, debug, impl, sim/csim non-report, sidecars | Task 1 + 3 |
| Age ≥6h + process check | Task 2 |
| Verify before delete; partial zip; `.sol_bulk_ok` | Task 3 |
| dry-run, jobs, limit, batch-prefix, report JSONL | Task 4 |
| Idempotent skip if already archived | Task 2 |
| nothing_to_reclaim → skip | Task 2 |

**Placeholder scan:** none intentional.  
**Note:** Commit steps are optional per repo git policy — only run when the user asks to commit.

---

## Execution handoff

Plan saved to `docs/superpowers/plans/2026-07-25-archive-hls-sol-bulk.md`.

**Two execution options:**

1. **Subagent-Driven (recommended)** — fresh subagent per task, review between tasks  
2. **Inline Execution** — execute tasks in this session with checkpoints  

Which approach?
