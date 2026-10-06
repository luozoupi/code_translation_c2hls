"""Tests for scripts/archive_hls_sol_bulk.py."""

from __future__ import annotations

import importlib.util
import os
import sys
import time
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_archive_module():
    path = ROOT / "scripts" / "archive_hls_sol_bulk.py"
    spec = importlib.util.spec_from_file_location("archive_hls_sol_bulk", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["archive_hls_sol_bulk"] = mod
    spec.loader.exec_module(mod)
    return mod


_mod = _load_archive_module()
classify_sol_members = _mod.classify_sol_members
ZIP_NAME = _mod.ZIP_NAME
OK_NAME = _mod.OK_NAME
sol_age_ok = _mod.sol_age_ok
already_archived = _mod.already_archived
batch_root_for_sol = _mod.batch_root_for_sol
skip_reason_for_sol = _mod.skip_reason_for_sol
PROC_ROOT = _mod.PROC_ROOT
archive_one_sol = _mod.archive_one_sol
VerifyError = _mod.VerifyError
discover_sol_dirs = _mod.discover_sol_dirs
main = _mod.main


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
    _touch(sol / "sim" / "verilog" / "k.result.lat.rb", "$TOTAL_EXECUTE_TIME = \"42\";\n")
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
    assert "sim/verilog/k.result.lat.rb" in keep
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
    assert "sim/verilog/k.result.lat.rb" not in delete
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
    with zipfile.ZipFile(sol / ZIP_NAME, "w") as zf:
        zf.writestr("marker.txt", "ok")
    (sol / OK_NAME).write_text("{}", encoding="utf-8")
    assert already_archived(sol)


def test_batch_root_for_sol(tmp_path: Path) -> None:
    root = tmp_path / "c2hls_tmp"
    sol = root / "batch_a" / "b" / "hls_proj" / "sol1"
    sol.mkdir(parents=True)
    assert batch_root_for_sol(sol, root) == (root / "batch_a").resolve()


def test_path_has_active_hls_process_false_on_empty(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(_mod, "PROC_ROOT", tmp_path)
    assert _mod.path_has_active_hls_process(tmp_path / "batch") is False


def test_skip_reason_for_sol_already_archived(tmp_path: Path) -> None:
    sol = _make_sol(tmp_path)
    old = time.time() - 7 * 3600
    os.utime(sol, (old, old))
    with zipfile.ZipFile(sol / ZIP_NAME, "w") as zf:
        zf.writestr("marker.txt", "ok")
    (sol / OK_NAME).write_text("{}", encoding="utf-8")
    root = tmp_path
    assert skip_reason_for_sol(sol, root, min_age_hours=6) == "already_archived"


def test_skip_reason_for_sol_too_young(tmp_path: Path) -> None:
    sol = _make_sol(tmp_path)
    root = tmp_path
    assert skip_reason_for_sol(sol, root, min_age_hours=6) == "too_young"


def test_skip_reason_for_sol_active_process(tmp_path: Path) -> None:
    sol = _make_sol(tmp_path)
    old = time.time() - 7 * 3600
    os.utime(sol, (old, old))
    root = tmp_path
    batch = batch_root_for_sol(sol, root)
    assert skip_reason_for_sol(sol, root, min_age_hours=6, busy_batches={batch}) == (
        "active_process"
    )


def test_skip_reason_for_sol_nothing_to_reclaim(tmp_path: Path) -> None:
    sol = tmp_path / "batch_x" / "bench" / "hls_proj" / "sol1"
    _touch(sol / ".autopilot" / "db" / "burst.xml")
    _touch(sol / "syn" / "report" / "csynth.rpt")
    _touch(sol / "sol1.log")
    old = time.time() - 7 * 3600
    os.utime(sol, (old, old))
    root = tmp_path
    assert skip_reason_for_sol(sol, root, min_age_hours=6) == "nothing_to_reclaim"


def test_skip_reason_for_sol_none_when_ready(tmp_path: Path) -> None:
    sol = _make_sol(tmp_path)
    old = time.time() - 7 * 3600
    os.utime(sol, (old, old))
    root = tmp_path
    assert skip_reason_for_sol(sol, root, min_age_hours=6) is None


def test_archive_one_sol_deletes_bulk_keeps_pipeline_files(tmp_path: Path) -> None:
    sol = _make_sol(tmp_path)
    result = archive_one_sol(sol, dry_run=False)
    assert result["status"] == "archived"

    assert (sol / "sol_bulk.zip").is_file()
    assert (sol / ".sol_bulk_ok").is_file()
    assert (sol / "syn" / "report" / "csynth.rpt").is_file()
    assert (sol / ".autopilot" / "db" / "burst.xml").is_file()
    assert (sol / "sol1.log").is_file()
    assert (sol / "sim" / "report" / "k_cosim.rpt").is_file()
    assert (sol / "sim" / "verilog" / "k.result.lat.rb").is_file()

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

    def bad_verify(zip_path, members, sol_dir):
        raise VerifyError("forced")

    monkeypatch.setattr(_mod, "verify_zip_members", bad_verify)
    result = archive_one_sol(sol, dry_run=False)
    assert result["status"] == "failed"
    assert (sol / ".autopilot" / "huge.dat").is_file()


def test_discover_sol_dirs(tmp_path: Path) -> None:
    a = tmp_path / "batch1" / "x" / "hls_proj" / "sol1" / ".autopilot"
    b = tmp_path / "batch2" / "y" / "hls_proj" / "sol1" / ".autopilot"
    a.mkdir(parents=True)
    b.mkdir(parents=True)
    found = discover_sol_dirs(tmp_path, batch_prefix="batch1")
    assert len(found) == 1
    assert found[0].name == "sol1"


def test_main_dry_run(tmp_path: Path) -> None:
    sol = _make_sol(tmp_path)
    old = time.time() - 7 * 3600
    os.utime(sol, (old, old))
    report = tmp_path / "r.jsonl"
    rc = main(
        [
            "--root",
            str(tmp_path),
            "--dry-run",
            "--min-age-hours",
            "6",
            "--jobs",
            "1",
            "--report",
            str(report),
        ]
    )
    assert rc == 0
    assert report.is_file()
