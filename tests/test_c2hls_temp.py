from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from c2hls_temp import (  # noqa: E402
    C2HLS_TMP_BENCH_ENV,
    C2HLS_TMP_ROOT_ENV,
    C2HLS_TMP_RUN_ENV,
    C2HLS_TMP_TAG_ENV,
    join_temp_tag,
    make_tempdir,
    sanitize_temp_tag,
)


@pytest.fixture(autouse=True)
def _clean_temp_env(monkeypatch, tmp_path):
    monkeypatch.setenv(C2HLS_TMP_ROOT_ENV, str(tmp_path))
    monkeypatch.delenv(C2HLS_TMP_RUN_ENV, raising=False)
    monkeypatch.delenv(C2HLS_TMP_BENCH_ENV, raising=False)
    monkeypatch.delenv(C2HLS_TMP_TAG_ENV, raising=False)
    yield


def test_sanitize_temp_tag():
    assert sanitize_temp_tag("chathls/gemm!!") == "chathls_gemm"
    # sanitize collapses runs of underscores (legacy behavior)
    assert sanitize_temp_tag("  a__b  ") == "a_b"


def test_join_temp_tag_collapses_separators():
    assert join_temp_tag("chathls_gemm", "flash", "synth") == "chathls_gemm_flash_synth"


def test_flat_make_tempdir_without_run(tmp_path):
    tag = join_temp_tag("chathls_gemm", "flash", "synth")
    path = Path(make_tempdir(prefix="hls_synth_", tag=tag))
    assert path.parent == tmp_path
    assert path.name == "hls_synth__chathls_gemm_flash_synth"
    assert path.is_dir()


def test_nested_run_bench_strips_leaf(tmp_path, monkeypatch):
    monkeypatch.setenv(C2HLS_TMP_RUN_ENV, "batch_parallel_chathls_fd_20260714")
    tag = join_temp_tag("chathls_gemm", "flash", "synth")
    path = Path(make_tempdir(prefix="hls_synth_", tag=tag))
    assert path == (
        tmp_path
        / "batch_parallel_chathls_fd_20260714"
        / "chathls_gemm"
        / "hls_synth__flash_synth"
    )
    assert path.is_dir()


def test_nested_explicit_bench_env(tmp_path, monkeypatch):
    monkeypatch.setenv(C2HLS_TMP_RUN_ENV, "camp_a")
    monkeypatch.setenv(C2HLS_TMP_BENCH_ENV, "chathls_covariance")
    tag = join_temp_tag("chathls_covariance", "dataflow", "a0", "csim")
    path = Path(make_tempdir(prefix="hls_csim_", tag=tag))
    assert path == (
        tmp_path / "camp_a" / "chathls_covariance" / "hls_csim__dataflow_a0_csim"
    )


def test_collision_suffix(tmp_path, monkeypatch):
    monkeypatch.setenv(C2HLS_TMP_RUN_ENV, "camp_b")
    monkeypatch.setenv(C2HLS_TMP_BENCH_ENV, "bench_x")
    tag = join_temp_tag("bench_x", "flash", "synth")
    p1 = Path(make_tempdir(prefix="hls_synth_", tag=tag))
    p2 = Path(make_tempdir(prefix="hls_synth_", tag=tag))
    assert p1.name == "hls_synth__flash_synth"
    assert p2.name == "hls_synth__flash_synth_001"
    assert p1.parent == p2.parent == tmp_path / "camp_b" / "bench_x"
