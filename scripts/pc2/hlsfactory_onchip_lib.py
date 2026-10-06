"""HLSFactory + AutoSA onchip pack (transfer). Not the 90-skill dump.

Flash uses flash_onchip_wide_gemm_skill_entries.json. The second stage is the
existing compute-rewrite + hide-load-store chain (C2HLS_POST_FLASH_DSE /
C2HLS_POST_FLASH_STREAM), with the same onchip JSON injected. Do not call that
stage DSE on slides.

DSP: never force min 5000 on HLSFactory. PolyBench / non-GEMM kernels cannot
legally fill 5000 DSP. Ceiling stays 9024. Compute-rewrite floor is 1 so tiny
stencils are not rejected after flash. Fused A+B reject stays on via
C2HLS_FLASH_ONCHIP=1.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
SKILLS_ONCHIP = (
    REPO
    / "hls_full_optimization_skills_schema_1_1_package"
    / "flash_onchip_wide_gemm_skill_entries.json"
)
HLSFACTORY_28_JSON = SCRIPT_DIR / "batch_parallel_hlsfactory_deepseek_u280.json"
DEVICE_DSP = 9024

# GEMM-like PolyBench. Still no 5000 flash floor: sizes are not AutoSA 64^3.
HLSFACTORY_GEMM_LIKE = frozenset(
    {
        "hlsfactory_gemm",
        "hlsfactory_2mm",
        "hlsfactory_3mm",
        "hlsfactory_syrk",
        "hlsfactory_syr2k",
        "hlsfactory_trmm",
        "hlsfactory_symm",
        "hlsfactory_gemver",
        "hlsfactory_gesummv",
        "hlsfactory_doitgen",
    }
)

# Unique <=4 char shorts so job prefix hf{short} stays under 10 chars.
_JOB_SHORT = {
    "hlsfactory_jacobi-1d": "j1d",
    "hlsfactory_gesummv": "gsv",
    "hlsfactory_correlation": "cor",
    "hlsfactory_fdtd-2d": "fd2",
    "hlsfactory_atax": "atx",
    "hlsfactory_bicg": "bcg",
    "hlsfactory_mvt": "mvt",
    "hlsfactory_2mm": "2mm",
    "hlsfactory_3mm": "3mm",
    "hlsfactory_cholesky": "cho",
    "hlsfactory_covariance": "cov",
    "hlsfactory_doitgen": "doi",
    "hlsfactory_durbin": "dur",
    "hlsfactory_floyd-warshall": "flw",
    "hlsfactory_gemm": "gem",
    "hlsfactory_gemver": "gvr",
    "hlsfactory_gramschmidt": "gra",
    "hlsfactory_heat-3d": "ht3",
    "hlsfactory_jacobi-2d": "j2d",
    "hlsfactory_lu": "hlu",
    "hlsfactory_ludcmp": "lud",
    "hlsfactory_nussinov": "nus",
    "hlsfactory_seidel-2d": "sei",
    "hlsfactory_symm": "sym",
    "hlsfactory_syr2k": "s2k",
    "hlsfactory_syrk": "syk",
    "hlsfactory_trisolv": "tri",
    "hlsfactory_trmm": "trm",
}


def hlsfactory_benches() -> list[str]:
    data = json.loads(HLSFACTORY_28_JSON.read_text(encoding="utf-8"))
    benches = [str(b) for b in ((data.get("pilot") or {}).get("benches") or [])]
    if not benches:
        raise ValueError(f"no benches in {HLSFACTORY_28_JSON}")
    return benches


def hlsfactory_job_short(bench: str) -> str:
    name = (bench or "").strip()
    if name in _JOB_SHORT:
        return _JOB_SHORT[name]
    slug = name.replace("hlsfactory_", "").replace("-", "")[:4]
    return slug or "hf"


def hlsfactory_job_prefix(bench: str) -> str:
    return f"hf{hlsfactory_job_short(bench)}"[:10]


def is_hlsfactory_gemm_like(bench: str) -> bool:
    return (bench or "").strip() in HLSFACTORY_GEMM_LIKE


def dsp_policy(*, bench: str) -> dict[str, Any]:
    """No FLASH_MIN_DSP. Max 9024. Compute-rewrite min DSP 1."""
    _ = bench
    return {
        "flash_min_dsp": None,
        "flash_max_dsp": DEVICE_DSP,
        "dse_min_dsp": 1,
        "reason": (
            "HLSFactory transfer: no 5000 DSP floor (PolyBench / non-GEMM cannot "
            "legally fill it). Ceiling 9024. Compute-rewrite floor 1."
        ),
    }


def write_hlsfactory_onchip_config(*, bench: str, dest: Path, job_prefix: str) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    turns = int(os.environ.get("C2HLS_TURNS", "7") or "7")
    doc = {
        "job_prefix": job_prefix,
        "combined_hls_nodes": True,
        "synth_nodes_per_variant": 1,
        "synth_workers_per_node": 1,
        "cosim_nodes_per_variant": 0,
        "cosim_workers_per_node": 0,
        "worker_cpus": 16,
        "worker_mem_gb": 64,
        "gpu_policy": "always_on",
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
            "variant": "onchip",
            "workflow": "flash",
            "corpus": "hlsfactory",
            "benches": [bench],
            "failure_policy": "ignore",
            "model": "deepseek-v4-flash",
            "turns": turns,
        },
    }
    dest.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    return dest
