"""Curated flash skill bins for AutoSA-ready kernels (not the 90-skill dump).

Bins are a transfer test: zero-shot / generic HLS / GEMM-family / systolic I/O /
spend-DSP on-chip. Primary spend-DSP path is the distilled 8-skill on-chip pack.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
SKILLS_PKG = REPO / "hls_full_optimization_skills_schema_1_1_package"
INVENTORY_JSON = Path(__file__).resolve().parent / "autosa_kernel_inventory.json"

SKILLS_GENERIC = SKILLS_PKG / "flash_generic_hls_skill_entries.json"
SKILLS_GEMM_FAMILY = SKILLS_PKG / "flash_gemm_family_skill_entries.json"
SKILLS_SYSTOLIC_IO = SKILLS_PKG / "flash_systolic_io_skill_entries.json"
SKILLS_ONCHIP = SKILLS_PKG / "flash_onchip_wide_gemm_skill_entries.json"

BIN_ZERO_SHOT = "zero_shot"
BIN_GENERIC = "generic"
BIN_GEMM_FAMILY = "gemm_family"
BIN_SYSTOLIC_IO = "systolic_io"
BIN_ONCHIP = "onchip"

PACK_ALIASES = {
    "": BIN_ONCHIP,
    "skills": BIN_ONCHIP,
    "onchip": BIN_ONCHIP,
    "spend_dsp": BIN_ONCHIP,
    "spend-dsp": BIN_ONCHIP,
    "generic": BIN_GENERIC,
    "generic_hls": BIN_GENERIC,
    "gemm": BIN_GEMM_FAMILY,
    "gemm_family": BIN_GEMM_FAMILY,
    "systolic": BIN_SYSTOLIC_IO,
    "systolic_io": BIN_SYSTOLIC_IO,
    "sysio": BIN_SYSTOLIC_IO,
    "zero": BIN_ZERO_SHOT,
    "zero_shot": BIN_ZERO_SHOT,
    "zeroshot": BIN_ZERO_SHOT,
}

VARIANT_GENERIC = "autosa_generic_hls"
VARIANT_GEMM_FAMILY = "autosa_gemm_family"
VARIANT_SYSTOLIC_IO = "autosa_systolic_io"
VARIANT_ONCHIP = "autosa_onchip_gemm"
VARIANT_ZERO_SHOT = "autosa_zero_shot"
VARIANT_GOLD = "autosa_gold"

SETUP_TAG = {
    VARIANT_GENERIC: "flash__autosa__generic_hls",
    VARIANT_GEMM_FAMILY: "flash__autosa__gemm_family",
    VARIANT_SYSTOLIC_IO: "flash__autosa__systolic_io",
    VARIANT_ONCHIP: "flash__autosa__onchip_gemm",
    VARIANT_ZERO_SHOT: "flash__autosa__zero_shot",
    VARIANT_GOLD: "gold_gate",
}

_INVENTORY_CACHE: dict[str, Any] | None = None


def normalize_pack(pack: str) -> str:
    raw = (pack or BIN_ONCHIP).strip().lower().replace("-", "_")
    mapped = PACK_ALIASES.get(raw, raw)
    if mapped not in {
        BIN_ZERO_SHOT,
        BIN_GENERIC,
        BIN_GEMM_FAMILY,
        BIN_SYSTOLIC_IO,
        BIN_ONCHIP,
    }:
        raise ValueError(
            f"unknown skill pack {pack!r} (use onchip, generic, gemm_family, "
            "systolic_io, or zero_shot)"
        )
    return mapped


def normalize_bench(name: str) -> str:
    raw = (name or "").strip()
    if not raw:
        raise ValueError("kernel name is empty")
    if raw.startswith("autosa_"):
        return raw
    return f"autosa_{raw}"


def prepare_kernel_id(bench: str) -> str:
    bench = normalize_bench(bench)
    return bench[len("autosa_") :]


def load_inventory() -> dict[str, Any]:
    global _INVENTORY_CACHE
    if _INVENTORY_CACHE is None:
        _INVENTORY_CACHE = json.loads(INVENTORY_JSON.read_text(encoding="utf-8"))
    return _INVENTORY_CACHE


def kernel_record(bench: str) -> dict[str, Any]:
    bench = normalize_bench(bench)
    kernels = load_inventory().get("kernels") or {}
    rec = kernels.get(bench)
    if not isinstance(rec, dict):
        raise ValueError(f"kernel {bench!r} is not in autosa_kernel_inventory.json")
    return rec


def job_short(bench: str) -> str:
    rec = kernel_record(bench)
    short = str(rec.get("job_short") or "").strip()
    if short:
        return short
    return prepare_kernel_id(bench).replace("_", "")[:8]


def default_min_dsp(bench: str, pack: str) -> int | None:
    pack = normalize_pack(pack)
    if pack != BIN_ONCHIP:
        return None
    rec = kernel_record(bench)
    raw = rec.get("min_dsp_onchip")
    if raw is None:
        return 5000 if rec.get("spend_dsp_5k") else 50
    return max(1, int(raw))


def default_max_dsp(bench: str, pack: str) -> int | None:
    pack = normalize_pack(pack)
    if pack != BIN_ONCHIP:
        return None
    device = int(load_inventory().get("device_dsp") or 9024)
    rec = kernel_record(bench)
    legal = rec.get("max_legal_dsp_est")
    if legal is not None:
        return max(1, min(device, int(legal)))
    return device


def default_row_uf(bench: str, pack: str) -> int | None:
    pack = normalize_pack(pack)
    if pack != BIN_ONCHIP:
        return None
    rec = kernel_record(bench)
    raw = rec.get("row_uf_hint")
    if raw is None:
        return None
    return max(1, int(raw))


def default_k_tile(bench: str, pack: str) -> int | None:
    pack = normalize_pack(pack)
    if pack != BIN_ONCHIP:
        return None
    rec = kernel_record(bench)
    raw = rec.get("k_tile_hint")
    if raw is None:
        return None
    return max(8, int(raw))


def needs_onchip_tile(bench: str) -> bool:
    rec = kernel_record(bench)
    if rec.get("onchip_fits") is False:
        return True
    fam = str(rec.get("family") or "")
    if fam in {"large_gemm", "tensor"}:
        return True
    name = normalize_bench(bench)
    if name.startswith("autosa_large_"):
        return True
    try:
        return int(rec.get("K") or 0) > 64
    except (TypeError, ValueError):
        return False


def default_pe_blk(bench: str, pack: str) -> int | None:
    pack = normalize_pack(pack)
    if pack != BIN_ONCHIP:
        return None
    rec = kernel_record(bench)
    raw = rec.get("pe_blk_default")
    if raw is None:
        return None
    pe = int(raw)
    return pe if pe in {16, 32, 64} else None


def variant_for_pack(pack: str) -> str:
    pack = normalize_pack(pack)
    return {
        BIN_ZERO_SHOT: VARIANT_ZERO_SHOT,
        BIN_GENERIC: VARIANT_GENERIC,
        BIN_GEMM_FAMILY: VARIANT_GEMM_FAMILY,
        BIN_SYSTOLIC_IO: VARIANT_SYSTOLIC_IO,
        BIN_ONCHIP: VARIANT_ONCHIP,
    }[pack]


def skills_json_for_pack(pack: str) -> Path | None:
    pack = normalize_pack(pack)
    return {
        BIN_ZERO_SHOT: None,
        BIN_GENERIC: SKILLS_GENERIC,
        BIN_GEMM_FAMILY: SKILLS_GEMM_FAMILY,
        BIN_SYSTOLIC_IO: SKILLS_SYSTOLIC_IO,
        BIN_ONCHIP: SKILLS_ONCHIP,
    }[pack]


def apply_skill_bin(pack: str) -> dict[str, Any]:
    """Configure env for one curated bin. Does not submit jobs."""
    from autosa_flash_lib import (
        configure_autosa_flash_gemm_family_env,
        configure_autosa_flash_generic_env,
        configure_autosa_flash_onchip_env,
        configure_autosa_flash_systolic_io_env,
        configure_autosa_flash_zero_shot_env,
    )

    pack = normalize_pack(pack)
    os.environ["C2HLS_FLASH_SKILL_BIN"] = pack
    if pack == BIN_ZERO_SHOT:
        configure_autosa_flash_zero_shot_env()
        os.environ["C2HLS_FLASH_SKILL_BIN"] = pack
        os.environ.pop("C2HLS_FLASH_ONCHIP", None)
        os.environ["BATCH_PARALLEL_VARIANT"] = VARIANT_ZERO_SHOT
        return {"pack": pack, "variant": VARIANT_ZERO_SHOT, "skills": ""}
    if pack == BIN_ONCHIP:
        configure_autosa_flash_onchip_env()
        os.environ["BATCH_PARALLEL_VARIANT"] = VARIANT_ONCHIP
        return {"pack": pack, "variant": VARIANT_ONCHIP, "skills": str(SKILLS_ONCHIP)}
    if pack == BIN_GENERIC:
        configure_autosa_flash_generic_env()
    elif pack == BIN_GEMM_FAMILY:
        configure_autosa_flash_gemm_family_env()
    else:
        configure_autosa_flash_systolic_io_env()
        os.environ.pop("C2HLS_FLASH_MIN_DSP", None)
        os.environ.pop("C2HLS_FLASH_PE_BLK", None)
    variant = variant_for_pack(pack)
    os.environ["BATCH_PARALLEL_VARIANT"] = variant
    skills = skills_json_for_pack(pack)
    return {
        "pack": pack,
        "variant": variant,
        "skills": str(skills.resolve()) if skills is not None else "",
    }


def write_kernel_flash_config(*, bench: str, dest: Path, job_prefix: str) -> Path:
    bench = normalize_bench(bench)
    dest.parent.mkdir(parents=True, exist_ok=True)
    doc = {
        "job_prefix": job_prefix,
        "synth_nodes_per_variant": 1,
        "synth_workers_per_node": 1,
        "cosim_nodes_per_variant": 0,
        "cosim_workers_per_node": 0,
        "worker_cpus": 16,
        "worker_mem_gb": 64,
        "gpu_policy": "batch_park",
        "gpu_batch_threshold": 1,
        "gpu_batch_flush_s": 1800,
        "park_threshold_s": 1800,
        "long_cosim_park_s": 3600,
        "park_grace_s": 1800,
        "max_inflight_benches": 1,
        "bench_order": "listed",
        "bench_seeding": "short_first_waves",
        "poll_sec": 2.0,
        "coordinator_poll_sec": 15.0,
        "pilot": {
            "variant": os.environ.get("BATCH_PARALLEL_VARIANT", VARIANT_ONCHIP),
            "workflow": "autosa_flash",
            "corpus": "autosa_ready",
            "benches": [bench],
            "failure_policy": "ignore",
            "model": "deepseek-v4-flash",
            "turns": int(os.environ.get("C2HLS_TURNS", "7") or "7"),
        },
    }
    dest.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    return dest


def write_kernel_seed_synth_config(*, bench: str, dest: Path, job_prefix: str) -> Path:
    """Gold-gate only: csynth hls_baseline.cpp / kernel.h ABI. No LLM, no flash."""
    bench = normalize_bench(bench)
    dest.parent.mkdir(parents=True, exist_ok=True)
    doc = {
        "job_prefix": job_prefix,
        "synth_nodes_per_variant": 1,
        "synth_workers_per_node": 1,
        "cosim_nodes_per_variant": 0,
        "cosim_workers_per_node": 0,
        "worker_cpus": 16,
        "worker_mem_gb": 64,
        "gpu_policy": "batch_park",
        "gpu_batch_threshold": 1,
        "gpu_batch_flush_s": 1800,
        "park_threshold_s": 1800,
        "long_cosim_park_s": 3600,
        "park_grace_s": 1800,
        "max_inflight_benches": 1,
        "bench_order": "listed",
        "bench_seeding": "short_first_waves",
        "poll_sec": 2.0,
        "coordinator_poll_sec": 15.0,
        "pilot": {
            "variant": VARIANT_GOLD,
            "workflow": "autosa_gold",
            "corpus": "autosa_ready",
            "benches": [bench],
            "failure_policy": "ignore",
            "model": "none",
            "turns": 0,
        },
    }
    dest.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    return dest


def pack_is_contaminated_90(path: Path) -> bool:
    blob = path.read_text(encoding="utf-8").lower()
    return "pe_blk must be 16" in blob and "ping-pong tiles inside the gemm" in blob


def kernels_for_wave(family: str) -> list[str]:
    """Inventory order, minus skip_kernels. family='gemm64' | 'all' | 'rest'."""
    inv = load_inventory()
    order = [str(x) for x in (inv.get("order") or [])]
    skip = {str(x) for x in (inv.get("skip_kernels") or [])}
    kernels = inv.get("kernels") or {}
    want = (family or "gemm64").strip().lower()
    out: list[str] = []
    for name in order:
        rec = kernels.get(name) or {}
        if name in skip or rec.get("skip"):
            continue
        fam = str(rec.get("family") or "")
        if want == "all":
            out.append(name)
        elif want == "gemm64":
            if fam == "gemm64":
                out.append(name)
        elif want == "rest":
            if fam != "gemm64":
                out.append(name)
        elif fam == want:
            out.append(name)
    return out
