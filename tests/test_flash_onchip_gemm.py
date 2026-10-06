"""Distilled on-chip GEMM flash pack (940-class), not the 90-skill dump."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import autosa_flow_gates as g

REPO = Path(__file__).resolve().parents[1]
PKG = REPO / "hls_full_optimization_skills_schema_1_1_package"
ONCHIP_JSON = PKG / "flash_onchip_wide_gemm_skill_entries.json"
LAUNCHER = REPO / "scripts/pc2/start_autosa_mm_flash_onchip.sh"

sys.path.insert(0, str(REPO / "scripts" / "pc2"))

_IDS = {
    "hls-onchip-stage-when-fits",
    "hls-onchip-axi-512-three-bundles",
    "hls-onchip-write-once-affine-c",
    "hls-onchip-flatten-c-output-groups",
    "hls-onchip-full-k-unroll-adder-tree",
    "hls-onchip-partition-match-unroll",
    "avoid-onchip-dataflow-when-io-matches-compute",
    "avoid-onchip-fused-ab-lockstep",
    "avoid-onchip-kernel0-systolic",
    "avoid-onchip-dsp-over-u280",
    "hls-onchip-row-uf-low-mac-dsp",
    "hls-onchip-tile-k-under-dsp-cap",
}


_ENV_KEYS = (
    "C2HLS_PACKAGED_SKILLS_JSON",
    "C2HLS_PACKAGED_SKILLS_ONLY",
    "C2HLS_FLASH_SKILL_ENTRIES_JSON",
    "C2HLS_FLASH_ONCHIP",
    "C2HLS_FORCE_SKILL_PROMPTS",
    "C2HLS_SKILL_MODE",
    "C2HLS_SKILL_PROMPT_MODE",
)


def test_onchip_pack_is_short_and_not_autosa():
    data = json.loads(ONCHIP_JSON.read_text(encoding="utf-8"))
    ids = [s["id"] for s in data["skills"]]
    assert set(ids) == _IDS
    assert len(ids) == 12
    blob = json.dumps(data).lower()
    assert "kernel0" in blob
    assert "do not emit kernel0" in blob or "zero kernel0" in blob
    assert "ping-pong" in blob
    assert "reject ping-pong" in blob or "do not emit #pragma hls dataflow" in blob
    assert "20260830_mmflow" in data["description"]
    assert "9024" in blob
    assert "row_uf" in blob
    assert "k_tile" in blob
    assert "1032" in blob or "adder tree" in blob


def test_flash_onchip_off_by_default(monkeypatch):
    monkeypatch.delenv("C2HLS_FLASH_ONCHIP", raising=False)
    assert g.flash_onchip() is False
    monkeypatch.setenv("C2HLS_FLASH_ONCHIP", "1")
    assert g.flash_onchip() is True
    monkeypatch.setenv("C2HLS_FLASH_ONCHIP", "0")
    assert g.flash_onchip() is False


def test_flash_onchip_guidance_rejects_dataflow_and_kernel0():
    text = g.flash_onchip_initial_guidance()
    low = text.lower()
    assert "dataflow" in low
    assert "ping-pong" in low
    assert "write-once" in low or "write once" in low
    assert "lanes=16" in low
    assert "pe_blk" in low
    assert "kernel0" in low
    assert "9024" in text
    assert "load_a" in low
    assert "load_b" in low
    assert "load_a_b" in low
    assert "acc0" in low or "acc0..acc7" in low or "independent acc" in low
    assert "9024" in text


def test_onchip_configure_uses_only_distilled_pack(monkeypatch, tmp_path):
    from autosa_flash_lib import SKILLS_90_GEMM_JSON, configure_autosa_flash_onchip_env

    monkeypatch.setenv("C2HLS_TMP_ROOT", str(tmp_path))
    saved = {k: os.environ.get(k) for k in _ENV_KEYS}
    monkeypatch.setenv("C2HLS_FLASH_SKILL_ENTRIES_JSON", "/tmp/overlay.json")
    monkeypatch.setenv("C2HLS_PACKAGED_SKILLS_JSON", str(SKILLS_90_GEMM_JSON))
    try:
        configure_autosa_flash_onchip_env()
        pack = os.environ["C2HLS_PACKAGED_SKILLS_JSON"]
        assert pack.endswith("flash_onchip_wide_gemm_skill_entries.json")
        assert os.environ["C2HLS_PACKAGED_SKILLS_ONLY"] == "1"
        assert "C2HLS_FLASH_SKILL_ENTRIES_JSON" not in os.environ
        assert os.environ["C2HLS_FLASH_ONCHIP"] == "1"
        assert os.environ["C2HLS_FORCE_SKILL_PROMPTS"] == "1"
    finally:
        for key, val in saved.items():
            if val is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = val


def test_onchip_variant_is_valid_and_tagged(tmp_path, monkeypatch):
    from autosa_flash_lib import setup_tag_for_variant
    from batch_parallel_autosa_lib import AUTOSA_VARIANTS, configure_autosa_campaign_env
    from batch_parallel_dispatch import validate_variant

    monkeypatch.setenv("C2HLS_TMP_ROOT", str(tmp_path))
    saved = {k: os.environ.get(k) for k in _ENV_KEYS}
    try:
        assert "autosa_onchip_gemm" in AUTOSA_VARIANTS
        assert setup_tag_for_variant("autosa_onchip_gemm") == "flash__autosa__onchip_gemm"
        campaign = {"config": {"pilot": {"workflow": "autosa_flash"}}}
        assert validate_variant(campaign, "autosa_onchip_gemm")
        configure_autosa_campaign_env("autosa_onchip_gemm")
        assert os.environ["C2HLS_PACKAGED_SKILLS_JSON"].endswith(
            "flash_onchip_wide_gemm_skill_entries.json"
        )
    finally:
        for key, val in saved.items():
            if val is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = val


def test_campaign_restores_flash_onchip(monkeypatch):
    from batch_parallel_config import apply_autosa_flow_from_campaign

    monkeypatch.delenv("C2HLS_FLASH_ONCHIP", raising=False)
    apply_autosa_flow_from_campaign({"flash_onchip": 1})
    assert os.environ["C2HLS_FLASH_ONCHIP"] == "1"
    monkeypatch.delenv("C2HLS_FLASH_ONCHIP", raising=False)


def test_onchip_launcher_distilled_pack_flash_only():
    text = LAUNCHER.read_text(encoding="utf-8")
    assert "flash_onchip_wide_gemm_skill_entries.json" in text
    assert "C2HLS_PACKAGED_SKILLS_ONLY=1" in text
    assert "C2HLS_FLASH_ONCHIP=1" in text
    assert "C2HLS_FLASH_PE_BLK" in text
    assert "C2HLS_FLASH_MIN_DSP" in text
    assert "C2HLS_FLASH_ONLY=1" in text
    assert "autosa_onchip_gemm" in text
    assert "20260830_mmflow" in text
    assert "C2HLS_FLASH_SKILL_ENTRIES_JSON" in text
    assert "unset BATCH_PARALLEL_ARTIFACT_PREFIX" in text
    assert "C2HLS_POST_FLASH_STREAM=0" in text
