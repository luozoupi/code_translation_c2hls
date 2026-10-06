"""Gold-gate seed synth path for AutoSA-ready kernels (no LLM, no flash)."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "pc2"))

from autosa_skill_bins import (  # noqa: E402
    VARIANT_GOLD,
    job_short,
    kernels_for_wave,
    write_kernel_seed_synth_config,
)
from batch_parallel_autosa_lib import is_autosa_gold_workflow  # noqa: E402
from batch_parallel_config import seed_kwargs_for_workflow  # noqa: E402
from batch_parallel_dispatch import validate_variant  # noqa: E402

LAUNCHER = REPO / "scripts/pc2/start_autosa_kernel_seed_synth.sh"
PARALLEL = REPO / "scripts/pc2/start_autosa_kernel_seed_synth_parallel.sh"
CAMPAIGN = REPO / "scripts/pc2/start_batch_parallel_campaign.sh"


def test_seed_config_is_gold_not_flash(tmp_path):
    dest = tmp_path / "cfg.json"
    write_kernel_seed_synth_config(
        bench="autosa_mm_hcl", dest=dest, job_prefix="hclsd"
    )
    doc = json.loads(dest.read_text(encoding="utf-8"))
    assert doc["pilot"]["benches"] == ["autosa_mm_hcl"]
    assert doc["pilot"]["workflow"] == "autosa_gold"
    assert doc["pilot"]["variant"] == VARIANT_GOLD
    assert doc["pilot"]["corpus"] == "autosa_ready"
    assert doc["pilot"]["model"] == "none"
    assert doc["pilot"]["turns"] == 0
    assert doc["max_inflight_benches"] == 1


def test_gold_workflow_dispatch_and_seed():
    campaign = {"config": {"pilot": {"workflow": "autosa_gold"}}}
    assert is_autosa_gold_workflow(campaign)
    assert validate_variant(campaign, VARIANT_GOLD)
    assert not validate_variant(campaign, "autosa_onchip_gemm")
    kw = seed_kwargs_for_workflow("autosa_gold")
    assert kw["initial_kind"] == "synth"
    assert kw["initial_phase"] == "reference"
    assert kw["initial_stage"] == "gold_gate"


def test_seed_launcher_has_no_llm_and_skips_frozen_mm():
    text = LAUNCHER.read_text(encoding="utf-8")
    assert "--no-gpu" in text
    assert "C2HLS_FLASH_ONLY" in text
    assert "C2HLS_REFERENCE_ONLY" in text
    assert "write_kernel_seed_synth_config" in text
    assert "20260830_mmflow" in text
    assert "unset BATCH_PARALLEL_ARTIFACT_PREFIX" in text
    assert "--endpoint-url" not in text
    assert "start_hlsfactory_per_bench_proxies.sh" not in text
    campaign = CAMPAIGN.read_text(encoding="utf-8")
    assert "--no-gpu" in campaign
    assert "no_gpu" in campaign


def test_seed_parallel_has_no_proxies_and_all_wave():
    text = PARALLEL.read_text(encoding="utf-8")
    assert "start_autosa_kernel_seed_synth.sh" in text
    assert "start_hlsfactory_per_bench_proxies.sh" not in text
    assert "kernels_for_wave" in text
    assert "unset BATCH_PARALLEL_ARTIFACT_PREFIX" in text
    allk = kernels_for_wave("all")
    assert "autosa_mm" not in allk
    assert allk[0] == "autosa_mm_hcl"
    assert len(allk) == 20
    shorts = [f"{job_short(k)}sd"[:10] for k in allk]
    assert len(shorts) == len(set(shorts))


def test_configure_gold_strips_onchip(monkeypatch, tmp_path):
    monkeypatch.setenv("C2HLS_TMP_ROOT", str(tmp_path))
    from autosa_flash_lib import configure_autosa_gold_env

    monkeypatch.setenv("C2HLS_FLASH_ONCHIP", "1")
    monkeypatch.setenv("C2HLS_PACKAGED_SKILLS_JSON", "/tmp/x.json")
    configure_autosa_gold_env()
    assert os.environ["C2HLS_REFERENCE_ONLY"] == "1"
    assert os.environ["C2HLS_FLASH_ONLY"] == "0"
    assert "C2HLS_FLASH_ONCHIP" not in os.environ
    assert "C2HLS_PACKAGED_SKILLS_JSON" not in os.environ
    assert os.environ["BATCH_PARALLEL_VARIANT"] == VARIANT_GOLD
