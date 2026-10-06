"""Skip-flash seed: enforcement starts from an existing flash_opt, not a new LLM."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "pc2"))

import flash_enforcement as fe


FLASH_139K = '#include "kernel.h"\n#define UF 8\nextern "C" void autosa_mm() {}\n'
FLASH_139K_REPORT = {"latency_cycles": 139484, "dsp": 10, "interval": 139485}
SELECTED_STREAM = "/* stream 4292/320 — must not be the seed */\n"
SELECTED_REPORT = {"latency_cycles": 4292, "dsp": 320}


@pytest.fixture
def seed_dir(tmp_path, monkeypatch):
    root = tmp_path / "mmflow_flash"
    root.mkdir()
    (root / "autosa_mm_flash_opt.cpp").write_text(FLASH_139K, encoding="utf-8")
    (root / "autosa_mm_flash_opt_report.json").write_text(
        json.dumps(FLASH_139K_REPORT) + "\n", encoding="utf-8"
    )
    (root / "autosa_mm_selected.cpp").write_text(SELECTED_STREAM, encoding="utf-8")
    (root / "autosa_mm_selected_report.json").write_text(
        json.dumps(SELECTED_REPORT) + "\n", encoding="utf-8"
    )
    monkeypatch.setenv("C2HLS_SKIP_FLASH", "1")
    monkeypatch.setenv("C2HLS_FLASH_SEED_DIR", str(root))
    return root


def test_load_flash_seed_reads_opt_not_selected(seed_dir):
    loaded = fe.load_flash_seed("autosa_mm")
    assert loaded is not None
    code, report = loaded
    assert "UF 8" in code
    assert "4292" not in code
    assert report["latency_cycles"] == 139484
    assert report["dsp"] == 10


def test_load_flash_seed_disabled_without_flag(seed_dir, monkeypatch):
    monkeypatch.delenv("C2HLS_SKIP_FLASH", raising=False)
    assert fe.skip_flash_enabled() is False
    orch = SimpleNamespace()
    assert fe.apply_flash_seed_to_orch(orch, "autosa_mm") is False


def test_apply_flash_seed_sets_baseline_and_copies(seed_dir, tmp_path):
    cell = tmp_path / "new_variant"
    orch = SimpleNamespace(_artifact_output_dir=None, _pipelined_ctx={})
    assert fe.apply_flash_seed_to_orch(orch, "autosa_mm", cell) is True
    assert orch.hls_code == FLASH_139K
    assert orch.synth_report["latency_cycles"] == 139484
    assert orch.synth_report["dsp"] == 10
    assert orch._pipelined_ctx["flash_step_result"]["seeded"] is True
    copied = json.loads((cell / "autosa_mm_flash_opt_report.json").read_text(encoding="utf-8"))
    assert copied["latency_cycles"] == 139484
    assert copied["dsp"] == 10
    assert (cell / "autosa_mm_flash_opt.cpp").read_text(encoding="utf-8") == FLASH_139K


def test_init_campaign_json_records_skip_flash(tmp_path, monkeypatch):
    from batch_parallel_config import BatchParallelConfig, init_campaign_json

    monkeypatch.setenv("C2HLS_SKIP_FLASH", "1")
    monkeypatch.setenv("C2HLS_FLASH_SEED_DIR", "/tmp/mmflow_flash")
    monkeypatch.setenv("C2HLS_ENFORCEMENT", "1")
    monkeypatch.setenv("C2HLS_ENFORCEMENT_ROUNDS", "20")
    monkeypatch.delenv("C2HLS_AUTOSA_FLOW", raising=False)
    root = tmp_path / "campaign"
    doc = init_campaign_json(root, BatchParallelConfig(job_prefix="mmenf"), stamp="t")
    assert doc.get("skip_flash") is True
    assert doc.get("skip_phase_b") is True
    assert doc.get("flash_seed_dir") == "/tmp/mmflow_flash"
    assert doc.get("enforcement") is True


def test_apply_campaign_restores_skip_flash_env(monkeypatch):
    monkeypatch.setenv("C2HLS_SKIP_FLASH", "0")
    monkeypatch.setenv("C2HLS_FLASH_SEED_DIR", "")
    monkeypatch.setenv("C2HLS_SKIP_PHASE_B", "0")
    fe.apply_enforcement_from_campaign(
        {
            "enforcement": True,
            "skip_flash": True,
            "flash_seed_dir": "/seed/flash",
        }
    )
    assert os.environ["C2HLS_SKIP_FLASH"] == "1"
    assert os.environ["C2HLS_FLASH_SEED_DIR"] == "/seed/flash"
    assert os.environ["C2HLS_SKIP_PHASE_B"] == "1"


def test_tier_a_synth_flash_accepts_seed_without_csynth(seed_dir, tmp_path):
    from batch_parallel_queue import BatchParallelJob
    from tier_a_batch_parallel_bench import TierABatchParallelBenchSession

    cell = tmp_path / "cell"
    cell.mkdir()
    orch = SimpleNamespace(
        hls_code="",
        synth_report=None,
        _pipelined_ctx={},
        _artifact_output_dir=str(cell),
        turns_limitation=4,
    )
    with patch.object(TierABatchParallelBenchSession, "__init__", lambda self, **kwargs: None):
        session = TierABatchParallelBenchSession(
            variant_key="autosa_aav_n_gf",
            bench="autosa_mm",
            bench_dir=tmp_path,
            cell_dir=cell,
            model_id="deepseek-v4-flash",
            turns=4,
        )
    session.bench = "autosa_mm"
    session.cell_dir = cell
    session._ensure_orchestrator = lambda: orch
    job = BatchParallelJob(
        id=1,
        variant="autosa_aav_n_gf",
        bench="autosa_mm",
        kind="synth",
        phase="flash",
        attempt=0,
        stage="synth",
        meta={},
    )
    with patch.object(session, "_synth_csim_only") as synth:
        with patch.object(fe, "attach_enforcement_after_flash") as attach:
            attach.return_value = {"attempted": True, "applied": False}
            followups = session._run_synth_flash(job)
    synth.assert_not_called()
    attach.assert_called_once()
    assert orch.synth_report["latency_cycles"] == 139484
    assert orch.synth_report["dsp"] == 10
    assert followups[0]["phase"] == "finalize"
    assert followups[0]["kind"] == "finalize"


def test_pipelined_codegen_skips_llm(seed_dir, tmp_path):
    from flash_pipelined_bench import FlashPipelinedBenchSession
    from flash_pipelined_queue import PipelinedJob

    cell = tmp_path / "cell"
    cell.mkdir()
    orch = SimpleNamespace(
        hls_code="",
        synth_report=None,
        _pipelined_ctx={},
        _artifact_output_dir=str(cell),
    )

    def _boom(*_a, **_k):
        raise AssertionError("flash LLM must not run")

    orch.pipelined_flash_codegen = _boom
    with patch.object(FlashPipelinedBenchSession, "__init__", lambda self, **kwargs: None):
        session = FlashPipelinedBenchSession(
            variant_key="autosa_aav_n_gf",
            bench="autosa_mm",
            bench_dir=tmp_path,
            cell_dir=cell,
            model_id="deepseek-v4-flash",
            turns=4,
        )
    session.bench = "autosa_mm"
    session.cell_dir = cell
    session.orchestrator = orch
    job = PipelinedJob(
        id=1,
        variant="autosa_aav_n_gf",
        bench="autosa_mm",
        kind="codegen",
        phase="flash",
        attempt=0,
        stage="optimize",
        meta={},
    )
    followups = session._run_codegen(job)
    assert followups[0]["kind"] == "synth"
    assert followups[0]["phase"] == "flash"
    assert orch.synth_report["latency_cycles"] == 139484


def test_launchers_mention_skip_flash():
    enf = (REPO / "scripts/pc2/start_autosa_flash_enforcement.sh").read_text(encoding="utf-8")
    mm = (REPO / "scripts/pc2/start_autosa_mm_enforcement.sh").read_text(encoding="utf-8")
    campaign = (REPO / "scripts/pc2/start_batch_parallel_campaign.sh").read_text(encoding="utf-8")
    variant = (REPO / "scripts/pc2/start_batch_parallel_variant.sh").read_text(encoding="utf-8")
    assert "--seed-flash" in enf
    assert "C2HLS_SKIP_FLASH" in enf
    assert "--load-b-in-df" in enf
    assert "C2HLS_PP_LOAD_B_IN_DF" in enf
    assert "C2HLS_PP_LOAD_B_IN_DF" in campaign
    assert "C2HLS_PP_LOAD_B_IN_DF" in variant
    assert "C2HLS_FLASH_SEED_DIR" in mm
    assert "C2HLS_SKIP_FLASH" in campaign
    assert "C2HLS_FLASH_SEED_DIR" in campaign
    assert "C2HLS_SKIP_FLASH" in variant
    assert "C2HLS_FLASH_SEED_DIR" in variant
