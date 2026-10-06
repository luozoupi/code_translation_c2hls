"""HLSFactory + AutoSA onchip pack transfer launchers."""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PKG = REPO / "hls_full_optimization_skills_schema_1_1_package"
sys.path.insert(0, str(REPO / "scripts" / "pc2"))

from flash_fixed_cosim_lib import VARIANTS, verify_variant_skills  # noqa: E402
from hlsfactory_onchip_lib import (  # noqa: E402
    DEVICE_DSP,
    HLSFACTORY_GEMM_LIKE,
    SKILLS_ONCHIP,
    dsp_policy,
    hlsfactory_benches,
    hlsfactory_job_prefix,
    hlsfactory_job_short,
    is_hlsfactory_gemm_like,
    write_hlsfactory_onchip_config,
)

LAUNCHER = REPO / "scripts/pc2/start_hlsfactory_kernel_onchip.sh"
PARALLEL = REPO / "scripts/pc2/start_hlsfactory_kernel_onchip_parallel.sh"
_90 = PKG / "skills_ii_target_miss_solutions_added(90skills)_gemm_flatten_v1.json"


def test_onchip_variant_has_no_90_pack_and_no_overlay():
    variant = VARIANTS["onchip"]
    assert variant.skills_json == SKILLS_ONCHIP
    assert variant.flash_skill_overlay is False
    out = verify_variant_skills(variant)
    assert out["ok"], out
    assert out["skill_count"] == 12
    blob = SKILLS_ONCHIP.read_text(encoding="utf-8").lower()
    assert "pe_blk" in blob
    assert "9024" in blob
    assert "fused" in blob or "load_a_b" in blob
    assert "ping-pong tiles inside the gemm" not in blob
    assert pack_is_not_90()


def pack_is_not_90() -> bool:
    from autosa_skill_bins import pack_is_contaminated_90

    assert _90.is_file()
    assert not pack_is_contaminated_90(_90)
    assert not pack_is_contaminated_90(SKILLS_ONCHIP)
    return True


def test_hlsfactory_benches_and_unique_job_prefixes():
    benches = hlsfactory_benches()
    assert len(benches) == 28
    assert benches[0] == "hlsfactory_jacobi-1d"
    assert "hlsfactory_gemm" in benches
    shorts = [hlsfactory_job_short(b) for b in benches]
    assert len(shorts) == len(set(shorts))
    prefixes = [hlsfactory_job_prefix(b) for b in benches]
    assert len(prefixes) == len(set(prefixes))
    assert all(len(p) <= 10 for p in prefixes)
    assert is_hlsfactory_gemm_like("hlsfactory_gemm")
    assert not is_hlsfactory_gemm_like("hlsfactory_jacobi-1d")
    assert "hlsfactory_jacobi-1d" not in HLSFACTORY_GEMM_LIKE


def test_dsp_policy_never_forces_5000():
    for bench in ("hlsfactory_gemm", "hlsfactory_jacobi-1d", "hlsfactory_lu"):
        pol = dsp_policy(bench=bench)
        assert pol["flash_min_dsp"] is None
        assert pol["flash_max_dsp"] == DEVICE_DSP == 9024
        assert pol["dse_min_dsp"] == 1
        assert "5000" not in pol["reason"] or "no 5000" in pol["reason"].lower()


def test_write_onchip_config(tmp_path):
    dest = tmp_path / "cfg.json"
    write_hlsfactory_onchip_config(
        bench="hlsfactory_atax", dest=dest, job_prefix="hfatx"
    )
    doc = json.loads(dest.read_text(encoding="utf-8"))
    assert doc["pilot"]["benches"] == ["hlsfactory_atax"]
    assert doc["pilot"]["workflow"] == "flash"
    assert doc["pilot"]["variant"] == "onchip"
    assert doc["pilot"]["model"] == "deepseek-v4-flash"
    assert doc["max_inflight_benches"] == 1
    assert doc["cosim_nodes_per_variant"] == 0


def test_launchers_use_onchip_not_90_and_chain_compute():
    one = LAUNCHER.read_text(encoding="utf-8")
    par = PARALLEL.read_text(encoding="utf-8")
    assert "flash_onchip_wide_gemm_skill_entries.json" in one
    assert "skills_ii_target_miss_solutions_added(90skills)" not in one
    assert "C2HLS_FLASH_SKILL_BIN" in one
    assert "C2HLS_FLASH_ONCHIP" in one
    assert "C2HLS_POST_FLASH_DSE" in one
    assert "C2HLS_POST_FLASH_STREAM" in one
    assert "C2HLS_DSE_SKILL_ENTRIES_JSON" in one
    assert "C2HLS_STREAM_SKILL_ENTRIES_JSON" in one
    assert "C2HLS_DSE_MIN_DSP" in one
    assert "wait_hlsfactory_flash_then_dataflow" not in one
    assert "start_hlsfactory_per_bench_proxies.sh" in par
    assert "--workers 1" in par
    assert "18400" in par
    assert "18092" in par
    assert "unset BATCH_PARALLEL_ARTIFACT_PREFIX" in par
    assert "start_hlsfactory_kernel_onchip.sh" in par
