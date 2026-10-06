"""autosa_mm variant sweep: manifest, prefixes, DAG, harvest, +mem."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "pc2"))


def test_manifest_is_43_by_10_with_unique_prefixes_and_dag():
    from autosa_mm_variant_sweep import (
        build_manifest,
        unique_config_count,
        unique_configs,
    )

    cells = build_manifest(reps=10, date="20260919", repo=REPO)
    assert len(cells) == 860
    assert len(unique_configs(cells)) == 86
    assert unique_config_count() == 86
    prefixes = [c["job_prefix"] for c in cells]
    assert len(prefixes) == len(set(prefixes))
    by_id = {c["cell_id"]: c for c in cells}
    assert len(by_id) == 860
    families = {c["family"] for c in cells}
    assert families == {"flash", "dse_v1", "dse_v2", "stream", "enf", "oneshot", "mem"}
    assert sum(1 for c in cells if c["family"] == "flash") == 60
    assert sum(1 for c in cells if c["family"] == "oneshot") == 10
    assert sum(1 for c in cells if c["family"] == "dse_v1") == 60
    assert sum(1 for c in cells if c["family"] == "dse_v2") == 60
    assert sum(1 for c in cells if c["family"] == "stream") == 120
    assert sum(1 for c in cells if c["family"] == "enf") == 120
    assert sum(1 for c in cells if c["family"] == "mem") == 430
    for cell in cells:
        parent = cell["parent_cell"]
        if cell["family"] in {"flash", "oneshot"}:
            assert parent == ""
            continue
        assert parent in by_id
        if cell["family"] in {"dse_v1", "dse_v2"}:
            assert by_id[parent]["family"] == "flash"
            assert by_id[parent]["skill"] == cell["skill"]
            assert by_id[parent]["floor"] == cell["floor"]
            assert by_id[parent]["rep"] == cell["rep"]
        if cell["family"] in {"stream", "enf"}:
            assert by_id[parent]["family"] in {"dse_v1", "dse_v2"}
            assert by_id[parent]["dse"] == cell["dse"]
            assert by_id[parent]["family"] != "mem"
        if cell["family"] == "mem":
            assert parent == cell["cell_id"][len("mem_"):]
            assert by_id[parent]["family"] != "mem"
            assert cell["mem"] == 1
            assert cell["job_prefix"].startswith("vm")
        assert "20260830_mmflow" not in cell["campaign_root"]
        assert "20260916_flash_" not in cell["campaign_root"]
        assert cell["campaign_root"].endswith(f"{cell['cell_id']}_camp")


def test_manifest_reps1_is_86_with_mem_siblings():
    from autosa_mm_variant_sweep import (
        _cell_rank,
        build_manifest,
        unique_configs,
    )

    cells = build_manifest(reps=1, date="20260919", repo=REPO)
    assert len(cells) == 86
    assert len(unique_configs(cells)) == 86
    by_id = {c["cell_id"]: c for c in cells}
    base = [c for c in cells if c["family"] != "mem"]
    mem = [c for c in cells if c["family"] == "mem"]
    assert len(base) == 43
    assert len(mem) == 43
    assert {c["cell_id"] for c in base} == {c["parent_cell"] for c in mem}
    assert by_id["mem_flash_90_f1_r01"]["job_prefix"] == "vmsf9101"
    assert by_id["mem_oneshot_r01"]["job_prefix"] == "vmsos01"
    assert by_id["dse1_ns_f0_r01"]["parent_cell"] == "flash_ns_f0_r01"
    assert by_id["stream_dse1_ns_f0_r01"]["parent_cell"] == "dse1_ns_f0_r01"
    assert _cell_rank(by_id["mem_flash_ns_f0_r01"]) == 0.5
    assert _cell_rank(by_id["mem_dse1_ns_f0_r01"]) == 1.5
    assert _cell_rank(by_id["mem_stream_dse1_ns_f0_r01"]) == 2.5
    assert _cell_rank(by_id["mem_enf_dse2_90_f1_r01"]) == 3.5


def test_floor_and_oneshot_launch_env():
    from autosa_mm_variant_sweep import build_manifest, launch_env_for_cell

    cells = build_manifest(reps=1, date="20260919", repo=REPO)
    by_id = {c["cell_id"]: c for c in cells}
    floor_off = launch_env_for_cell(by_id["flash_ns_f0_r01"])
    floor_on = launch_env_for_cell(by_id["flash_gf_f1_r01"])
    oneshot = launch_env_for_cell(by_id["oneshot_r01"])
    skills90 = launch_env_for_cell(by_id["flash_90_f0_r01"])
    assert floor_off["C2HLS_FLASH_MIN_DSP"] == ""
    assert floor_off["C2HLS_FLASH_ONLY"] == "1"
    assert floor_off["C2HLS_LLM_TIMEOUT"] == "3600"
    assert floor_off["C2HLS_LLM_EMPTY_RETRIES"] == "1"
    assert floor_on["C2HLS_FLASH_MIN_DSP"] == "300"
    assert floor_on["C2HLS_FLASH_DSP_REDO"] == ""
    assert skills90["C2HLS_FLASH_MIN_DSP"] == ""
    assert oneshot["C2HLS_ONE_SHOT"] == "1"
    assert oneshot["C2HLS_SKIP_PHASE_B"] == "1"
    assert oneshot["C2HLS_FLASH_OPT_PROMPT_MODE"] == "zero_shot"
    assert oneshot["C2HLS_POST_FLASH_DSE"] == "0"
    assert oneshot["C2HLS_POST_FLASH_STREAM"] == "0"
    assert oneshot["C2HLS_FLASH_MIN_DSP"] == ""
    dse2 = launch_env_for_cell(by_id["dse2_ns_f0_r01"])
    assert dse2["C2HLS_DSE_V2"] == "1"
    assert dse2["C2HLS_LLM_TIMEOUT"] == "3600"
    assert dse2["C2HLS_LLM_EMPTY_RETRIES"] == "1"
    assert dse2["C2HLS_POST_FLASH_STREAM"] == "0"
    stream_v2 = launch_env_for_cell(by_id["stream_dse2_ns_f0_r01"])
    assert stream_v2["C2HLS_POST_FLASH_STREAM"] == "1"
    enf = launch_env_for_cell(by_id["enf_dse1_gf_f1_r01"])
    assert enf["C2HLS_PP_LOAD_B_IN_DF"] == "1"
    assert enf["C2HLS_SKIP_FLASH"] == "1"
    mem = launch_env_for_cell(by_id["mem_flash_ns_f0_r01"])
    assert mem["C2HLS_MEM_ITER"] == "1"
    assert mem["C2HLS_MEM_ITER_ROUNDS"] == "50"
    assert mem["C2HLS_MEM_ITER_PROMPT_TOKENS"] == "32768"
    assert mem["C2HLS_MEM_ITER_MAX_TOKENS"] == "65536"
    assert mem["C2HLS_MEM_ITER_CONTEXT_TOKENS"] == "131072"
    assert mem["C2HLS_LLM_TIMEOUT"] == "3600"
    assert mem["C2HLS_LLM_TIMEOUT_RETRIES"] == "8"
    assert mem["C2HLS_MEM_PARENT_FAMILY"] == "flash"
    assert mem["C2HLS_ONE_SHOT"] == "0"
    mem_stream = launch_env_for_cell(by_id["mem_stream_dse2_ns_f0_r01"])
    assert mem_stream["C2HLS_MEM_PARENT_FAMILY"] == "stream"


def test_one_shot_extra_blocks_empty(monkeypatch):
    monkeypatch.setenv("C2HLS_ONE_SHOT", "1")
    monkeypatch.setenv("C2HLS_FLASH_MIN_DSP", "300")
    monkeypatch.setenv("C2HLS_FLASH_ONCHIP", "1")
    from c2hls import flash_step_guidance_extra_blocks

    assert flash_step_guidance_extra_blocks() == []
    monkeypatch.setenv("C2HLS_ONE_SHOT", "0")
    monkeypatch.delenv("C2HLS_FLASH_ONCHIP", raising=False)
    blocks = flash_step_guidance_extra_blocks()
    assert blocks
    joined = "\n".join(blocks)
    assert "300" in joined or "DSP" in joined.upper() or "dsp" in joined.lower()


def test_refuse_frozen_trees():
    import pytest
    from autosa_mm_variant_sweep import refuse_frozen

    with pytest.raises(ValueError, match="frozen"):
        refuse_frozen("/tmp/batch_parallel_autosa_mm_flow_aav_n_gf_20260830_mmflow")
    with pytest.raises(ValueError, match="frozen"):
        refuse_frozen("artifacts/pc2/batch_parallel_autosa_mm_flash_dsp_redo_20260916_redo3")
    with pytest.raises(ValueError, match="frozen"):
        refuse_frozen("artifacts/pc2/autosa_mm_variant_sweep_20260918/cells/flash_ns_f0_r01_camp")
    refuse_frozen("artifacts/pc2/autosa_mm_variant_sweep_20260919/cells/flash_ns_f0_r01_camp")


def test_parent_complete_and_harvest_parser(tmp_path):
    from autosa_mm_variant_sweep import (
        build_manifest,
        make_enforcement_seed,
        parent_complete,
        qor_from_report,
    )
    from harvest_autosa_mm_variant_sweep import harvest_cell

    cells = build_manifest(reps=1, date="20990101", repo=tmp_path)
    by_id = {c["cell_id"]: c for c in cells}
    flash = by_id["flash_ns_f0_r01"]
    dse = by_id["dse1_ns_f0_r01"]
    flash_root = tmp_path / Path(flash["campaign_root"]).relative_to(tmp_path)
    flash["campaign_root"] = str(flash_root)
    dse["campaign_root"] = str(tmp_path / Path(dse["campaign_root"]).relative_to(tmp_path))
    cell_dir = (
        flash_root
        / "variants"
        / "autosa_noskills"
        / "autosa_mm"
        / "deepseek-v4-flash__flash__autosa__noskills"
    )
    cell_dir.mkdir(parents=True)
    report = {
        "latency_cycles": 4965,
        "latency_cycles_worst": 4965,
        "dsp": 384,
        "bram": 12,
        "lut": 1000,
        "ff": 2000,
        "success": True,
        "csim": {"passed": True},
    }
    (cell_dir / "autosa_mm_flash_opt_report.json").write_text(
        json.dumps(report), encoding="utf-8"
    )
    (cell_dir / "autosa_mm_flash_opt.cpp").write_text("int autosa_mm() { return 0; }\n")
    qor = qor_from_report(report)
    assert qor["latency_cycles"] == 4965
    assert qor["latency_cycles_worst"] == 4965
    assert qor["dsp"] == 384
    assert qor["csim"] == "pass"
    assert "interval" not in qor
    assert parent_complete(dse, {"flash_ns_f0_r01": flash}) is True

    dse_root = Path(dse["campaign_root"])
    dse_cell = (
        dse_root
        / "variants"
        / "autosa_noskills"
        / "autosa_mm"
        / "deepseek-v4-flash__flash__autosa__noskills"
    )
    dse_cell.mkdir(parents=True)
    selected = dict(report)
    selected["latency_cycles"] = 13160
    selected["dsp"] = 352
    (dse_cell / "autosa_mm_dse_report.json").write_text(json.dumps(selected), encoding="utf-8")
    (dse_cell / "autosa_mm_selected_report.json").write_text(json.dumps(selected), encoding="utf-8")
    (dse_cell / "autosa_mm_selected.cpp").write_text("int autosa_mm() { return 1; }\n")
    enf = by_id["enf_dse1_ns_f0_r01"]
    assert parent_complete(enf, {"dse1_ns_f0_r01": dse}) is True
    seed = make_enforcement_seed(dse_root, tmp_path / "enf_seed")
    assert (seed / "autosa_mm_flash_opt.cpp").is_file()
    assert (seed / "autosa_mm_flash_opt_report.json").is_file()

    flash_rows = harvest_cell(flash, campaign_root=flash_root)
    assert len(flash_rows) == 1
    row = flash_rows[0]
    assert row["stage"] == "flash"
    assert row["latency_cycles"] == 4965
    assert row["latency_cycles_worst"] == 4965
    assert row["dsp"] == 384
    assert row["csim"] == "pass"
    assert "interval" not in row
    dse_rows = harvest_cell(dse, campaign_root=dse_root)
    stages = {r["stage"] for r in dse_rows}
    assert "dse" in stages
    dse_row = next(r for r in dse_rows if r["stage"] == "dse")
    assert dse_row["latency_cycles"] == 13160
    assert dse_row["dsp"] == 352

    mem = by_id["mem_flash_ns_f0_r01"]
    mem_root = tmp_path / Path(mem["campaign_root"]).relative_to(tmp_path)
    mem["campaign_root"] = str(mem_root)
    mem_cell = (
        mem_root
        / "variants"
        / "autosa_noskills"
        / "autosa_mm"
        / "deepseek-v4-flash__flash__autosa__noskills"
    )
    mem_cell.mkdir(parents=True)
    mem_report = {
        "latency_cycles": 4000,
        "latency_cycles_worst": 4100,
        "dsp": 320,
        "bram": 16,
        "lut": 900,
        "ff": 1800,
        "success": True,
        "csim": {"passed": True},
    }
    (mem_cell / "autosa_mm_mem_report.json").write_text(json.dumps(mem_report), encoding="utf-8")
    mem_rows = harvest_cell(mem, campaign_root=mem_root)
    assert len(mem_rows) == 1
    assert mem_rows[0]["stage"] == "mem"
    assert mem_rows[0]["latency_cycles_worst"] == 4100
    assert mem_rows[0]["mem"] == 1
    assert "interval" not in mem_rows[0]


def test_dse_v2_not_complete_from_copied_flash_selected(tmp_path):
    from autosa_mm_variant_sweep import build_manifest, cell_complete

    cells = build_manifest(reps=1, date="20990103", repo=tmp_path)
    by_id = {c["cell_id"]: c for c in cells}
    dse2 = by_id["dse2_ns_f0_r01"]
    root = tmp_path / Path(dse2["campaign_root"]).relative_to(tmp_path)
    dse2["campaign_root"] = str(root)
    cell_dir = (
        root
        / "variants"
        / "autosa_noskills"
        / "autosa_mm"
        / "deepseek-v4-flash__flash__autosa__noskills"
    )
    cell_dir.mkdir(parents=True)
    report = {
        "latency_cycles": 41125,
        "latency_cycles_worst": 41125,
        "dsp": 24,
        "success": True,
        "csim": {"passed": True},
    }
    (cell_dir / "autosa_mm_selected_report.json").write_text(json.dumps(report), encoding="utf-8")
    (cell_dir / "autosa_mm_flash_opt_report.json").write_text(json.dumps(report), encoding="utf-8")
    assert cell_complete(dse2) is False
    (cell_dir / "autosa_mm_dse_report.json").write_text(json.dumps(report), encoding="utf-8")
    (cell_dir / "autosa_mm_dse_v2_leaderboard.json").write_text(
        json.dumps({"schema": "post_flash_dse_v2_leaderboard_v1", "success": True}),
        encoding="utf-8",
    )
    assert cell_complete(dse2) is True


def test_reap_dead_submitted_cells_without_squeue_prefix():
    from autosa_mm_variant_sweep import reap_dead_cells

    cells = [
        {
            "cell_id": "oneshot_r01",
            "family": "oneshot",
            "job_prefix": "vsos01",
            "status": "submitted",
            "campaign_root": "/nope",
            "fail_count": 0,
        },
        {
            "cell_id": "mem_flash_90_f0_r01",
            "family": "mem",
            "job_prefix": "vmsf9001",
            "status": "submitted",
            "campaign_root": "/nope",
            "fail_count": 0,
        },
        {
            "cell_id": "mem_dse1_90_f0_r01",
            "family": "mem",
            "job_prefix": "vms19001",
            "status": "submitted",
            "campaign_root": "/nope",
            "fail_count": 0,
        },
    ]
    reap_dead_cells(cells, names=["vms19001"])
    by_id = {c["cell_id"]: c for c in cells}
    assert by_id["oneshot_r01"]["status"] == "pending"
    assert by_id["oneshot_r01"]["fail_count"] == 1
    assert "vanished" in (by_id["oneshot_r01"].get("error") or "")
    assert by_id["mem_flash_90_f0_r01"]["status"] == "pending"
    assert by_id["mem_dse1_90_f0_r01"]["status"] == "submitted"


def test_dry_run_prints_430_and_zero_sbatch(capsys, tmp_path, monkeypatch):
    from autosa_mm_variant_sweep import run_sweep

    monkeypatch.chdir(tmp_path)
    cells = run_sweep(
        date="20990101",
        reps=10,
        waves={"flash", "dse", "stream", "enf", "oneshot"},
        endpoint="http://login5:18092/v1",
        max_inflight=3,
        dry_run=True,
        submit=False,
        repo=tmp_path,
    )
    out = capsys.readouterr().out
    assert len(cells) == 860
    assert out.count("prefix=") == 430
    assert "dry-run: zero sbatch" in out
    assert "Submitted batch job" not in out
    manifest = tmp_path / "artifacts" / "pc2" / "autosa_mm_variant_sweep_20990101" / "cells.jsonl"
    assert manifest.is_file()
    lines = [ln for ln in manifest.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) == 860


def test_dry_run_mem_wave_prints_children_zero_sbatch(capsys, tmp_path, monkeypatch):
    from autosa_mm_variant_sweep import run_sweep

    monkeypatch.chdir(tmp_path)
    cells = run_sweep(
        date="20990102",
        reps=1,
        waves={"mem"},
        endpoint="http://login5:18092/v1",
        max_inflight=3,
        dry_run=True,
        submit=False,
        repo=tmp_path,
    )
    out = capsys.readouterr().out
    assert len(cells) == 86
    assert out.count("prefix=") == 43
    assert "mem_flash_" in out
    assert "dry-run: zero sbatch" in out
    assert "Submitted batch job" not in out


def test_sweep_job_prefix_lock(monkeypatch):
    from autosa_flash_lib import apply_mm_flow_flavor

    monkeypatch.setenv("C2HLS_SWEEP_JOB_PREFIX", "vsfn001")
    monkeypatch.setenv("C2HLS_SWEEP_ARTIFACT_PREFIX", "autosa_mm_variant_sweep_20260918/cells/flash_ns_f0_r01")
    snap = apply_mm_flow_flavor("noskills")
    assert snap["job_prefix"] == "vsfn001"
    assert os.environ["PC2_BATCH_JOB_PREFIX"] == "vsfn001"
    assert "mmns" != os.environ["PC2_BATCH_JOB_PREFIX"]
    assert os.environ["BATCH_PARALLEL_ARTIFACT_PREFIX"].endswith("flash_ns_f0_r01")


def test_inflight_counts_unique_prefixes_not_helper_rows():
    from autosa_mm_variant_sweep import inflight_count

    cells = [
        {"job_prefix": "vsfn001", "status": "submitted", "campaign_root": "/nope"},
        {"job_prefix": "vsfn101", "status": "submitted", "campaign_root": "/nope"},
    ]
    names = [
        "vsfn001-watch",
        "vsfn001-drain",
        "vsfn001-coord",
        "vsfn001-synth-n0-camp",
        "vsfn101-watch",
    ]
    assert inflight_count(cells, names=names) == 2
    assert inflight_count(cells, names=[]) == 0


def test_launcher_script_exists():
    text = (REPO / "scripts/pc2/start_autosa_mm_variant_sweep.sh").read_text(encoding="utf-8")
    assert "autosa_mm_variant_sweep.py" in text
    assert "dry-run" in text
    v2 = (REPO / "scripts/pc2/post_flash_dse_v2.sbatch.sh").read_text(encoding="utf-8")
    assert "run_post_flash_dse_v2.py" in v2
    harvest = (REPO / "scripts/pc2/harvest_autosa_mm_variant_sweep.py").read_text(encoding="utf-8")
    assert "sweep_qor.csv" in harvest
    assert "latency_cycles_worst" in harvest
    mem_sbatch = (REPO / "scripts/pc2/post_flash_mem_iter.sbatch.sh").read_text(encoding="utf-8")
    assert "run_post_flash_mem_iter.py" in mem_sbatch
    follow = (REPO / "scripts/pc2/start_autosa_mm_mem_follow.sbatch.sh").read_text(encoding="utf-8")
    assert "20260919" in follow
    assert "20260918" not in follow
    assert "--reps" in follow
