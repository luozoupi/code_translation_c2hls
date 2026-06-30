"""Tests for post-cosim simulator log compaction."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from hls_log_compact import (
    COMPACT_BANNER,
    compact_cosim_work_dir_logs,
    compact_simulator_log_file,
)


def _write_log(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(lines), encoding="utf-8")


def test_compact_replaces_large_log_in_place(tmp_path: Path) -> None:
    log_path = tmp_path / "logs" / "hls_run_tcl.log"
    header = [f"INFO: header line {i}\n" for i in range(5)]
    middle = []
    for i in range(200):
        middle.append(f"Time: {i} ps  Iteration: 1  Process: /foo\n")
        middle.append("Warning: OPMODE Input Warning : repeated\n")
    footer = [
        "INFO: [COSIM 212-323] Starting verilog simulation.\n",
        "INFO: [Common 17-206] Exiting vitis_hls\n",
    ]
    _write_log(log_path, header + middle + footer)

    original_size = log_path.stat().st_size
    assert original_size > 1024

    stats = compact_simulator_log_file(
        log_path,
        min_bytes=1024,
        header_lines=3,
        footer_lines=2,
        max_warnings=2,
    )

    assert stats["compacted"] is True
    assert stats["warnings_total"] == 200
    assert stats["warnings_kept"] == 2
    assert log_path.stat().st_size < original_size

    compacted = log_path.read_text(encoding="utf-8")
    assert COMPACT_BANNER in compacted
    assert "header line 0" in compacted
    assert "Exiting vitis_hls" in compacted
    assert compacted.count("Warning: OPMODE Input Warning : repeated") >= 2


def test_compact_skips_small_logs(tmp_path: Path) -> None:
    log_path = tmp_path / "small.log"
    _write_log(log_path, ["INFO: tiny\n"])

    stats = compact_simulator_log_file(log_path, min_bytes=1024)
    assert stats["compacted"] is False
    assert log_path.read_text(encoding="utf-8") == "INFO: tiny\n"


def test_compact_cosim_work_dir_logs_finds_xsim(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("C2HLS_COSIM_COMPACT_LOGS", "1")
    monkeypatch.setenv("C2HLS_COSIM_COMPACT_LOG_MIN_BYTES", "1024")
    work_dir = tmp_path / "work"
    hls_log = work_dir / "logs" / "hls_run_tcl.log"
    xsim_log = work_dir / "hls_proj" / "sol1" / "sim" / "verilog" / "xsim.log"
    payload = ["INFO: start\n"] + ["Warning: noisy warning\n"] * 500 + ["INFO: end\n"]
    _write_log(hls_log, payload)
    _write_log(xsim_log, payload)

    results = compact_cosim_work_dir_logs(work_dir)
    assert len(results) == 2
    assert all(result["compacted"] for result in results)
    assert hls_log.stat().st_size < 4096
    assert xsim_log.stat().st_size < 4096


def test_compact_disabled_by_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("C2HLS_COSIM_COMPACT_LOGS", "0")
    log_path = tmp_path / "logs" / "hls_run_tcl.log"
    _write_log(log_path, ["Warning: noisy\n"] * 500)
    original_size = log_path.stat().st_size

    results = compact_cosim_work_dir_logs(tmp_path)
    assert results == []
    assert log_path.stat().st_size == original_size
