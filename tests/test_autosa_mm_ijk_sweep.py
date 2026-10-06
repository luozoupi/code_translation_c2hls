"""IJK size sweep: sibling benches, ready-root env, harvest scoring."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "pc2"))

SOURCE_H = REPO / "related_work/benchmarks/autosa_ready/autosa_mm/kernel.h"


def test_stage_ijk_bench_rewrites_header_and_leaves_source_64(tmp_path):
    from autosa_mm_ijk_sweep import stage_ijk_bench

    source_before = SOURCE_H.read_text(encoding="utf-8")
    dest = stage_ijk_bench(512, dest_root=tmp_path)
    source_after = SOURCE_H.read_text(encoding="utf-8")
    header = (dest / "kernel.h").read_text(encoding="utf-8")
    tb = (dest / "testbench.cpp").read_text(encoding="utf-8")
    meta = json.loads((dest / "metadata.json").read_text(encoding="utf-8"))
    assert source_after == source_before
    assert "#define I 64" in source_after
    assert "#define I 512" in header
    assert "#define J 512" in header
    assert "#define K 512" in header
    sig = header.find("autosa_mm(")
    defn = header.find("#define I 512")
    assert 0 <= defn < sig
    assert "malloc" in tb
    assert "data_t (*A)[K]" in tb
    assert "data_t A[I][K]," not in tb.split("int main", 1)[-1]
    assert 'printf("Failed with %d errors!\\n", err);' in tb
    assert 'printf("Passed!\\n");' in tb
    assert "return err ? 1 : 0;" in tb
    assert int(meta["csim_timeout_s"]) >= 7200


def test_ready_autosa_mm_testbench_returns_nonzero_on_mismatches():
    tb = (
        REPO / "related_work/benchmarks/autosa_ready/autosa_mm/testbench.cpp"
    ).read_text(encoding="utf-8")
    assert 'printf("Failed with %d errors!\\n", err);' in tb
    assert 'printf("Passed!\\n");' in tb
    assert "return err ? 1 : 0;" in tb


def test_resolve_autosa_ready_root_prefers_env(tmp_path, monkeypatch):
    from autosa_mm_ijk_sweep import resolve_autosa_ready_root

    sibling = tmp_path / "n512"
    (sibling / "autosa_mm").mkdir(parents=True)
    (sibling / "autosa_mm" / "metadata.json").write_text(
        json.dumps({"benchmark": "autosa_mm"}), encoding="utf-8"
    )
    monkeypatch.setenv("C2HLS_AUTOSA_READY_ROOT", str(sibling))
    assert resolve_autosa_ready_root() == sibling


def test_dse_resolve_bench_dir_uses_env(tmp_path, monkeypatch):
    import run_post_flash_dse as dse1
    import run_post_flash_dse_v2 as dse2

    sibling = tmp_path / "n1024"
    bench = sibling / "autosa_mm"
    bench.mkdir(parents=True)
    (bench / "metadata.json").write_text(
        json.dumps({"benchmark": "autosa_mm"}), encoding="utf-8"
    )
    monkeypatch.setenv("C2HLS_AUTOSA_READY_ROOT", str(sibling))
    assert dse1._resolve_bench_dir("autosa_mm") == bench
    assert dse2._resolve_bench_dir("autosa_mm") == bench


def test_flash_bench_map_uses_env(tmp_path, monkeypatch):
    from batch_parallel_autosa_lib import resolve_autosa_bench_map

    sibling = tmp_path / "n2048"
    bench = sibling / "autosa_mm"
    bench.mkdir(parents=True)
    (bench / "metadata.json").write_text(
        json.dumps({"benchmark": "autosa_mm"}), encoding="utf-8"
    )
    monkeypatch.setenv("C2HLS_AUTOSA_READY_ROOT", str(sibling))
    mapped = resolve_autosa_bench_map(["autosa_mm"])
    assert mapped["autosa_mm"] == bench


def test_harvest_picks_min_worst_latency_skips_fail_and_dse2_clone(tmp_path):
    from autosa_mm_ijk_sweep import harvest_size, pick_best

    rows = [
        {
            "cell_id": "flash_90_f1_r08",
            "family": "flash",
            "skill": "aav_n_90",
            "floor": 1,
            "rep": 8,
            "latency_cycles_worst": 3135,
            "dsp": 384,
            "csim": "pass",
            "dse_v2_trial": "",
        },
        {
            "cell_id": "flash_gf_f0_r02",
            "family": "flash",
            "skill": "aav_n_gf",
            "floor": 0,
            "rep": 2,
            "latency_cycles_worst": 4808,
            "dsp": 318,
            "csim": "pass",
            "dse_v2_trial": "",
        },
        {
            "cell_id": "flash_ns_f0_r01",
            "family": "flash",
            "skill": "noskills",
            "floor": 0,
            "rep": 1,
            "latency_cycles_worst": 100,
            "dsp": 10,
            "csim": "fail",
            "dse_v2_trial": "",
        },
        {
            "cell_id": "dse2_90_f1_r08",
            "family": "dse_v2",
            "skill": "aav_n_90",
            "floor": 1,
            "rep": 8,
            "latency_cycles_worst": 3135,
            "dsp": 384,
            "csim": "pass",
            "dse_v2_trial": "",
        },
        {
            "cell_id": "dse2_gf_f0_r01",
            "family": "dse_v2",
            "skill": "aav_n_gf",
            "floor": 0,
            "rep": 1,
            "latency_cycles_worst": 5602,
            "dsp": 384,
            "csim": "pass",
            "dse_v2_trial": "pe32_simd2",
        },
        {
            "cell_id": "dse1_gf_f0_r02",
            "family": "dse_v1",
            "skill": "aav_n_gf",
            "floor": 0,
            "rep": 2,
            "latency_cycles_worst": 5590,
            "dsp": 352,
            "csim": "pass",
            "dse_v2_trial": "",
        },
    ]
    flash = pick_best([r for r in rows if r["family"] == "flash"])
    dse1 = pick_best([r for r in rows if r["family"] == "dse_v1"])
    dse2 = pick_best(
        [r for r in rows if r["family"] == "dse_v2" and r.get("dse_v2_trial")]
    )
    assert flash["cell_id"] == "flash_90_f1_r08"
    assert dse1["cell_id"] == "dse1_gf_f0_r02"
    assert dse2["cell_id"] == "dse2_gf_f0_r01"
    empty = harvest_size(tmp_path)
    assert empty["best"]["flash"] is None
    assert empty["pending_counts"]["flash"] == 0
    assert empty["fail_counts"]["flash"] == 0
