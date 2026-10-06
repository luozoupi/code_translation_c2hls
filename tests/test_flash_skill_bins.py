"""Curated flash skill bins for AutoSA kernels other than frozen autosa_mm."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import autosa_flow_gates as g

REPO = Path(__file__).resolve().parents[1]
PKG = REPO / "hls_full_optimization_skills_schema_1_1_package"
sys.path.insert(0, str(REPO / "scripts" / "pc2"))

from autosa_skill_bins import (  # noqa: E402
    BIN_GEMM_FAMILY,
    BIN_GENERIC,
    BIN_ONCHIP,
    BIN_SYSTOLIC_IO,
    BIN_ZERO_SHOT,
    INVENTORY_JSON,
    SKILLS_GEMM_FAMILY,
    SKILLS_GENERIC,
    SKILLS_ONCHIP,
    SKILLS_SYSTOLIC_IO,
    apply_skill_bin,
    default_k_tile,
    default_max_dsp,
    default_min_dsp,
    default_pe_blk,
    default_row_uf,
    job_short,
    kernel_record,
    kernels_for_wave,
    load_inventory,
    needs_onchip_tile,
    normalize_bench,
    normalize_pack,
    pack_is_contaminated_90,
    prepare_kernel_id,
    variant_for_pack,
)

LAUNCHER = REPO / "scripts/pc2/start_autosa_kernel_flash.sh"
PARALLEL = REPO / "scripts/pc2/start_autosa_kernel_flash_parallel.sh"
_90 = PKG / "skills_ii_target_miss_solutions_added(90skills)_gemm_flatten_v1.json"

_ENV_KEYS = (
    "C2HLS_PACKAGED_SKILLS_JSON",
    "C2HLS_PACKAGED_SKILLS_ONLY",
    "C2HLS_FLASH_SKILL_ENTRIES_JSON",
    "C2HLS_FLASH_ONCHIP",
    "C2HLS_FLASH_SKILL_BIN",
    "C2HLS_FORCE_SKILL_PROMPTS",
    "C2HLS_SKILL_MODE",
    "C2HLS_SKILL_PROMPT_MODE",
    "C2HLS_FLASH_MIN_DSP",
    "C2HLS_FLASH_PE_BLK",
    "C2HLS_POST_FLASH_NO_SKILLS",
    "C2HLS_SKIP_PHASE_B",
    "C2HLS_FLASH_OPT_PROMPT_MODE",
    "BATCH_PARALLEL_VARIANT",
)


def _save() -> dict[str, str | None]:
    return {k: os.environ.get(k) for k in _ENV_KEYS}


def _restore(saved: dict[str, str | None]) -> None:
    for key, val in saved.items():
        if val is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = val


def test_90_pack_is_not_contaminated_and_curated_packs_stay_separate():
    assert _90.is_file()
    assert not pack_is_contaminated_90(_90)
    for path in (SKILLS_GENERIC, SKILLS_GEMM_FAMILY, SKILLS_SYSTOLIC_IO, SKILLS_ONCHIP):
        blob = path.read_text(encoding="utf-8").lower()
        assert "pe_blk must be 16" not in blob or path == SKILLS_ONCHIP
        assert "ping-pong tiles inside the gemm" not in blob
        data = json.loads(path.read_text(encoding="utf-8"))
        if path == SKILLS_ONCHIP:
            assert 9 <= len(data["skills"]) <= 12
        else:
            assert 4 <= len(data["skills"]) <= 9


def test_generic_pack_has_no_pe_blk_and_no_64x64_pong():
    data = json.loads(SKILLS_GENERIC.read_text(encoding="utf-8"))
    blob = json.dumps(data).lower()
    assert "this pack does not set pe_blk" in blob
    assert "ping-pong tiles inside the gemm" not in blob
    assert "no pe_blk" in data["description"].lower()
    ids = [s["id"] for s in data["skills"]]
    assert "hls-generic-pipeline-hot-ii1" in ids
    assert "avoid-generic-dataflow-as-magic" in ids


def test_gemm_family_keeps_b_layout_and_rejects_fused_ab():
    blob = SKILLS_GEMM_FAMILY.read_text(encoding="utf-8").lower()
    assert "b is j" in blob
    assert "from zero" in blob
    assert "fused" in blob
    assert "kernel.h" in blob
    ids = [s["id"] for s in json.loads(SKILLS_GEMM_FAMILY.read_text())["skills"]]
    assert "avoid-gemm-fused-ab-lockstep" in ids
    assert "hls-gemm-b-is-j-by-k" in ids


def test_systolic_io_is_hide_load_store_not_spend_chip():
    blob = SKILLS_SYSTOLIC_IO.read_text(encoding="utf-8").lower()
    assert "hide load-store" in blob or "hide load-store" in json.loads(
        SKILLS_SYSTOLIC_IO.read_text()
    )["description"].lower()
    assert "not spend-dsp" in json.loads(SKILLS_SYSTOLIC_IO.read_text())["description"].lower()
    assert "ram_2p" in blob
    assert "pe_kj" in blob


def test_inventory_skips_mm_and_orders_hcl_first():
    inv = load_inventory()
    assert inv["skip_kernels"] == ["autosa_mm"]
    assert inv["order"][0] == "autosa_mm_hcl"
    rec = kernel_record("autosa_mm_hcl")
    assert rec["spend_dsp_5k"] is True
    assert rec["dtype"] == "float"
    assert rec["I"] == rec["J"] == rec["K"] == 64
    lu = kernel_record("autosa_lu")
    assert lu["spend_dsp_5k"] is False
    dnn = kernel_record("autosa_dnn_ops")
    assert dnn["spend_dsp_5k"] is False
    assert dnn["max_legal_dsp_est"] == 1280
    assert kernel_record("autosa_mm")["skip"] is True


def test_normalize_and_prepare_ids():
    assert normalize_bench("mm_hcl") == "autosa_mm_hcl"
    assert normalize_bench("autosa_mm_hcl") == "autosa_mm_hcl"
    assert prepare_kernel_id("autosa_mm_hcl") == "mm_hcl"
    assert normalize_pack("spend-dsp") == BIN_ONCHIP
    assert variant_for_pack("generic") == "autosa_generic_hls"
    assert job_short("autosa_mm_hcl") == "hcl"
    assert default_min_dsp("autosa_mm_hcl", "onchip") == 5000
    assert default_max_dsp("autosa_mm_hcl", "onchip") == 9024
    assert default_max_dsp("autosa_dnn_ops", "onchip") == 1280
    assert default_max_dsp("autosa_lu", "onchip") == 400
    assert default_pe_blk("autosa_mm_hcl", "onchip") == 16
    assert default_pe_blk("autosa_mm_catapult", "onchip") == 32
    assert default_min_dsp("autosa_mm_hcl", "generic") is None
    assert default_min_dsp("autosa_lu", "onchip") == 50
    assert default_min_dsp("autosa_dnn_ops", "onchip") == 80
    assert default_row_uf("autosa_mm_int16", "onchip") == 8
    assert default_row_uf("autosa_mm_hcl", "onchip") is None
    assert needs_onchip_tile("autosa_large_mm") is True
    assert needs_onchip_tile("autosa_mm_hcl") is False
    assert default_k_tile("autosa_large_mm", "onchip") == 128


def test_apply_onchip_and_generic(monkeypatch, tmp_path):
    monkeypatch.setenv("C2HLS_TMP_ROOT", str(tmp_path))
    saved = _save()
    try:
        snap = apply_skill_bin("onchip")
        assert snap["variant"] == "autosa_onchip_gemm"
        assert os.environ["C2HLS_FLASH_ONCHIP"] == "1"
        assert os.environ["C2HLS_FLASH_SKILL_BIN"] == "onchip"
        assert os.environ["C2HLS_PACKAGED_SKILLS_JSON"].endswith(
            "flash_onchip_wide_gemm_skill_entries.json"
        )
        assert "C2HLS_FLASH_SKILL_ENTRIES_JSON" not in os.environ
        snap = apply_skill_bin("generic")
        assert snap["variant"] == "autosa_generic_hls"
        assert "C2HLS_FLASH_ONCHIP" not in os.environ
        assert os.environ["C2HLS_FLASH_SKILL_BIN"] == "generic"
        assert os.environ["C2HLS_PACKAGED_SKILLS_JSON"].endswith(
            "flash_generic_hls_skill_entries.json"
        )
        snap = apply_skill_bin("zero_shot")
        assert snap["variant"] == "autosa_zero_shot"
        assert os.environ["C2HLS_FLASH_OPT_PROMPT_MODE"] == "zero_shot"
        assert "C2HLS_PACKAGED_SKILLS_JSON" not in os.environ
    finally:
        _restore(saved)


def test_campaign_restores_skill_bin(monkeypatch):
    from batch_parallel_config import apply_autosa_flow_from_campaign

    monkeypatch.delenv("C2HLS_FLASH_SKILL_BIN", raising=False)
    apply_autosa_flow_from_campaign({"flash_skill_bin": "gemm_family"})
    assert os.environ["C2HLS_FLASH_SKILL_BIN"] == "gemm_family"
    monkeypatch.delenv("C2HLS_FLASH_SKILL_BIN", raising=False)


def test_dispatch_accepts_new_variants(tmp_path, monkeypatch):
    from batch_parallel_autosa_lib import AUTOSA_VARIANTS, configure_autosa_campaign_env
    from batch_parallel_dispatch import validate_variant
    from autosa_flash_lib import setup_tag_for_variant

    monkeypatch.setenv("C2HLS_TMP_ROOT", str(tmp_path))
    saved = _save()
    try:
        campaign = {"config": {"pilot": {"workflow": "autosa_flash"}}}
        for key in (
            "autosa_generic_hls",
            "autosa_gemm_family",
            "autosa_systolic_io",
            "autosa_onchip_gemm",
            "autosa_gold",
        ):
            assert key in AUTOSA_VARIANTS
            assert validate_variant(campaign, key)
        assert setup_tag_for_variant("autosa_generic_hls") == "flash__autosa__generic_hls"
        configure_autosa_campaign_env("autosa_gemm_family")
        assert os.environ["C2HLS_FLASH_SKILL_BIN"] == "gemm_family"
        assert os.environ["C2HLS_PACKAGED_SKILLS_JSON"].endswith(
            "flash_gemm_family_skill_entries.json"
        )
    finally:
        _restore(saved)


def test_guidance_bins(monkeypatch):
    monkeypatch.delenv("C2HLS_FLASH_SKILL_BIN", raising=False)
    gen = g.flash_generic_hls_initial_guidance().lower()
    assert "pe_blk" in gen
    assert "kernel.h" in gen
    gemm = g.flash_gemm_family_initial_guidance().lower()
    assert "j x k" in gemm or "jxk" in gemm
    assert "fuse" in gemm
    sysio = g.flash_systolic_io_initial_guidance().lower()
    assert "hide load-store" in sysio
    assert "940" in sysio
    assert g.flash_skill_bin() is None
    monkeypatch.setenv("C2HLS_FLASH_SKILL_BIN", "generic")
    assert g.flash_skill_bin() == "generic"


def test_kernel_launcher_is_flash_only_and_not_mmflow():
    text = LAUNCHER.read_text(encoding="utf-8")
    assert "--kernel" in text
    assert "--pack" in text
    assert "C2HLS_FLASH_ONLY" in text
    assert "C2HLS_POST_FLASH_STREAM=0" in text or 'os_env["C2HLS_POST_FLASH_STREAM"] = "0"' in text
    assert "20260830_mmflow" in text
    assert "autosa_mm_hcl" in text
    assert "unset BATCH_PARALLEL_ARTIFACT_PREFIX" in text
    assert "start_batch_parallel_campaign.sh" in text
    assert "C2HLS_FLASH_SKILL_BIN" in text
    assert "C2HLS_FLASH_MAX_DSP" in text
    assert "C2HLS_FLASH_ONCHIP_TILE" in text
    assert "C2HLS_FLASH_K_TILE" in text
    assert "C2HLS_FLASH_RETRY_TAG" in text
    campaign = (
        REPO / "scripts/pc2/start_batch_parallel_campaign.sh"
    ).read_text(encoding="utf-8")
    assert "C2HLS_FLASH_SKILL_BIN" in campaign
    assert "kernel_flash_configs" in text
    assert "BATCH_PARALLEL_CONFIG is not a file" in campaign



def test_write_kernel_flash_config_one_bench(tmp_path):
    from autosa_skill_bins import write_kernel_flash_config

    dest = tmp_path / "cfg.json"
    write_kernel_flash_config(bench="autosa_mm_hcl", dest=dest, job_prefix="hclonc7")
    doc = json.loads(dest.read_text(encoding="utf-8"))
    assert doc["pilot"]["benches"] == ["autosa_mm_hcl"]
    assert doc["pilot"]["workflow"] == "autosa_flash"
    assert doc["pilot"]["corpus"] == "autosa_ready"
    assert doc["max_inflight_benches"] == 1
    assert doc["job_prefix"] == "hclonc7"


def test_kernels_for_wave_skips_frozen_mm():
    gemm = kernels_for_wave("gemm64")
    assert "autosa_mm" not in gemm
    assert gemm[0] == "autosa_mm_hcl"
    assert "autosa_mm_block_sparse" in gemm
    assert "autosa_mm_hbm" in gemm
    assert "autosa_cnn" not in gemm
    rest = kernels_for_wave("rest")
    assert "autosa_cnn" in rest
    assert "autosa_lu" in rest
    assert "autosa_mm_hcl" not in rest
    assert "autosa_mm_hbm" not in rest
    allk = kernels_for_wave("all")
    assert "autosa_mm" not in allk
    assert allk[0] == "autosa_mm_hcl"
    assert "autosa_large_mttkrp" in allk
    assert len(allk) == 20
    shorts = [job_short(k) for k in allk]
    assert len(shorts) == len(set(shorts))


def test_parallel_launcher_uses_per_kernel_proxies():
    text = PARALLEL.read_text(encoding="utf-8")
    assert "start_hlsfactory_per_bench_proxies.sh" in text
    assert "start_autosa_kernel_flash.sh" in text
    assert "--endpoint-url" in text
    assert "18340" in text
    assert "18092" in text
    assert "CHATHLS_DEEPSEEK_QUEUE_WORKERS" not in text or "--workers 1" in text
    assert "unset BATCH_PARALLEL_ARTIFACT_PREFIX" in text
    assert "--wave" in text


def test_inventory_file_covers_all_autosa_ready_dirs():
    ready = REPO / "related_work/benchmarks/autosa_ready"
    names = sorted(p.name for p in ready.iterdir() if p.is_dir())
    inv = json.loads(INVENTORY_JSON.read_text(encoding="utf-8"))
    missing = [n for n in names if n not in inv["kernels"]]
    assert missing == [], missing
