"""Repository and machine path resolution for c2hls.

Repo-relative paths (benchmarks, skills package, scripts, …) are derived from
``REPO_ROOT`` and are always portable.

Machine-specific paths use one of two **sites**:

* **team** (default) — hardcoded paths for the team's development server.
* **pc2** — paths from ``local.env`` (gitignored) on the PC2 cluster.

Select a site with ``--pc2`` on the command line or ``C2HLS_SITE=pc2`` in the
environment. Example::

    ./c2hls.py --pc2 --bench nw
    C2HLS_SITE=pc2 python run_agentic_sweep.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

# --- In-repo paths (committed) ------------------------------------------------

BENCHMARKS_DIR = REPO_ROOT / "benchmarks"
BENCHMARKS_COSIM_DIR = REPO_ROOT / "benchmarks_cosim"
BENCHMARKS_EXTERNAL_DIR = REPO_ROOT / "benchmarks_external"
EXTERNAL_DATASETS_DIR = REPO_ROOT / "external_datasets"
SKILLS_PACKAGE_DIR = REPO_ROOT / "hls_full_optimization_skills_schema_1_1_package"
SKILLS_MUTABLE_DIR = REPO_ROOT / "skills"
ANALYSIS_DIR = REPO_ROOT / "analysis"
ARTIFACTS_DIR = REPO_ROOT / "artifacts"
FLASH_API_ARTIFACTS_DIR = ARTIFACTS_DIR / "flash_api"
SCHEMAS_DIR = REPO_ROOT / "schemas"
SCRIPTS_DIR = REPO_ROOT / "scripts"
RESULTS_DIR = REPO_ROOT / "results"
REPO_TMP_DIR = REPO_ROOT / "c2hls_tmp"
EMU_ENV_SCRIPT = SCRIPTS_DIR / "setup_emu_env.sh"

SKILLS_BASE_JSON = SKILLS_PACKAGE_DIR / "skills.json"
SKILLS_EXTENSION_JSON = SKILLS_PACKAGE_DIR / "skills_extension.json"
SKILLS_II_TARGET_MISS_73_JSON = (
    SKILLS_PACKAGE_DIR / "skills_ii_target_miss_solutions_added(73skills).json"
)
SKILLS_II_TARGET_MISS_90_JSON = (
    SKILLS_PACKAGE_DIR / "skills_ii_target_miss_solutions_added(90skills).json"
)
# Default for new-matrix tooling (latest library).
SKILLS_II_TARGET_MISS_JSON = SKILLS_II_TARGET_MISS_90_JSON

RESULTS_MATRIX_PHASE8 = REPO_ROOT / "results_matrix_u280_fullcosim"
RESULTS_MATRIX_EXTENDED = REPO_ROOT / "results_matrix_u280_fullcosim_extended"
RESULTS_MATRIX_MULTISTEP = REPO_ROOT / "results_matrix_u280_multistep_old_skills"

# --- Team server defaults (unchanged from upstream) ---------------------------

TEAM_DEFAULTS: dict[str, str] = {
    "C2HLS_VITIS_SETTINGS": "/mnt/data/luo00466/Xilinx/Vitis/2023.2/settings64.sh",
    "C2HLS_XRT_SETUP": "/mnt/data/luo00466/XRT_2023.2/opt/xilinx/xrt/setup.sh",
    "C2HLS_PLATFORM_REPO_PATHS": "/mnt/data/luo00466/U280_PLATFORM/opt/xilinx/platforms",
    "C2HLS_OPENCL_HEADERS": "/mnt/data/luo00466/opencl_headers",
    "C2HLS_RODINIA_HLS_DIR": "/home/luo00466/rodinia-hls/Benchmarks",
    "C2HLS_RODINIA_NOVA_DIR": "/home/luo00466/rodinia-hls-nova/Benchmarks",
    "C2HLS_ML4ACCEL_DIR": "/home/luo00466/ML4Accel-Dataset/fpga_ml_dataset/HLS_dataset",
    "C2HLS_CLAUDE_KEY_FILE": "/home/luo00466/claude-api-key.txt",
    "C2HLS_OPENAI_KEY_FILE": "/home/luo00466/gpt-key.txt",
    "C2HLS_TMP_ROOT": "/mnt/data/luo00466/tmp",
    "C2HLS_PYTHON": "/home/luo00466/.conda/envs/py310_2/bin/python",
}

# PC2 Otus defaults — applied after local.env on --pc2 (setdefault only).
PC2_DEFAULTS: dict[str, str] = {
    "C2HLS_VITIS_SETTINGS": "/opt/software/FPGA/Xilinx/Vitis/2023.2/settings64.sh",
    "C2HLS_XRT_SETUP": "/opt/software/FPGA/Xilinx/XRT/xrt_2.16/setup.sh",
    "C2HLS_PLATFORM_REPO_PATHS": (
        "/opt/software/FPGA/Xilinx/platforms/xilinx_u280_gen3x16_xdma_1_202211_1"
    ),
    "PC2_COMPUTE_MODULES": "fpga xilinx/xrt/2.16",
    "PC2_COMPUTE_U280_SWAP_TO": "xilinx/u280/xdma_202211_1",
    "PC2_GPU_MODULES": "lang system CUDA/12.6.0 Python/3.11.5-GCCcore-13.2.0",
    "C2HLS_MODEL": "mistralai/Devstral-2-123B-Instruct-2512",
    "PC2_LLM_MODEL": "mistralai/Devstral-2-123B-Instruct-2512",
    "OPENAI_API_KEY": "EMPTY",
    "PC2_GPU_PARTITION": "gpu_h100",
    "PC2_COMPUTE_PARTITION": "normal",
    "PC2_WALLTIME": "3:00:00",
    "PC2_SLURM_ACCOUNT": "hpc-prf-llmfpga",
    # PC2 pilot: csynth + csim only (skip cosim/hw_emu for faster runs).
    "C2HLS_RUN_COSIM": "0",
    "C2HLS_COSIM_REQUIRED": "0",
    "C2HLS_REFERENCE_COSIM": "0",
    "C2HLS_HW_EMU_FINAL": "0",
    # hlsfactory has one gold HLS kernel (reference gate); reuse its csynth
    # report for vs_ground_truth on steps like flash (no per-step GT variant).
    "C2HLS_GT_BASELINE_FALLBACK": "1",
}

SITES = frozenset({"team", "pc2"})
_ACTIVE_SITE: str | None = None


def bootstrap_site(argv: list[str] | None = None) -> str:
    """Detect site from ``C2HLS_SITE`` or ``--pc2`` in *argv* (default ``sys.argv``)."""
    if argv is None:
        argv = sys.argv
    site = os.environ.get("C2HLS_SITE", "").strip().lower()
    if "--pc2" in argv:
        site = "pc2"
        os.environ["C2HLS_SITE"] = "pc2"
    if site not in SITES:
        site = "team"
        os.environ.setdefault("C2HLS_SITE", "team")
    return site


def active_site() -> str:
    return _ACTIVE_SITE or bootstrap_site()


def _load_local_env_file() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(REPO_ROOT / "local.env")
    load_dotenv(REPO_ROOT / ".env")


def configure_site(site: str | None = None) -> str:
    """Apply path defaults for *site* (auto-detected when omitted). Idempotent."""
    global _ACTIVE_SITE
    if site is None:
        site = bootstrap_site()
    if _ACTIVE_SITE == site:
        return site
    _ACTIVE_SITE = site

    if site == "pc2":
        _load_local_env_file()
        for key, value in PC2_DEFAULTS.items():
            os.environ.setdefault(key, value)
        os.environ.setdefault("C2HLS_TMP_ROOT", str(REPO_TMP_DIR))
    else:
        for key, value in TEAM_DEFAULTS.items():
            os.environ.setdefault(key, value)

    apply_runtime_defaults()
    return site


def load_local_env() -> None:
    """Backward-compatible alias: configure the active site."""
    configure_site()


def add_site_argument(parser) -> None:
    parser.add_argument(
        "--pc2",
        action="store_true",
        help="Use PC2 cluster paths from local.env (default: team server paths)",
    )


def _env_path(name: str, default: str | None = None) -> Path | None:
    raw = os.getenv(name, default or "").strip()
    if not raw:
        return None
    return Path(raw).expanduser()


def optional_path(name: str) -> Path | None:
    return _env_path(name)


def required_path(name: str, hint: str = "") -> Path:
    p = optional_path(name)
    if p is None:
        msg = (
            f"{name} is not set. For PC2, copy local.env.example to local.env "
            "and configure it (or pass --pc2 only after local.env exists)."
        )
        if hint:
            msg = f"{msg} {hint}"
        raise RuntimeError(msg)
    return p


def rodinia_hls_benchmarks_dir() -> Path | None:
    return _env_path("C2HLS_RODINIA_HLS_DIR") or _env_path("RODINIA_HLS_DIR")


def rodinia_nova_benchmarks_dir() -> Path | None:
    return _env_path("C2HLS_RODINIA_NOVA_DIR") or _env_path("RODINIA_NOVA_DIR")


def ml4accel_dataset_dir() -> Path | None:
    return _env_path("C2HLS_ML4ACCEL_DIR") or _env_path("ML4ACCEL_DIR")


def ml4accel_repo_root() -> Path | None:
    root = _env_path("C2HLS_ML4ACCEL_ROOT")
    if root is not None:
        return root
    dataset = ml4accel_dataset_dir()
    if dataset is None:
        return None
    return dataset.parent.parent.parent


def claude_key_file() -> Path | None:
    return _env_path("C2HLS_CLAUDE_KEY_FILE")


def openai_key_file() -> Path | None:
    return _env_path("C2HLS_OPENAI_KEY_FILE")


def skills_config_path(*, extended: bool = False) -> str:
    """Colon-separated skills JSON path(s) under the repo package."""
    base = str(SKILLS_BASE_JSON)
    if extended:
        return f"{base}:{SKILLS_EXTENSION_JSON}"
    return base


def rodinia_variant_roots() -> list[tuple[Path, str]]:
    """(benchmarks_root, source_repo_name) pairs that exist on this host."""
    out: list[tuple[Path, str]] = []
    nova = rodinia_nova_benchmarks_dir()
    if nova and nova.is_dir():
        out.append((nova, "rodinia-hls-nova"))
    rodinia = rodinia_hls_benchmarks_dir()
    if rodinia and rodinia.is_dir():
        out.append((rodinia, "rodinia-hls"))
    return out


def apply_runtime_defaults(*, profile: str | None = None) -> None:
    """Set portable defaults; machine paths come from the active site."""
    os.environ.setdefault("C2HLS_EMU_ENV_SCRIPT", str(EMU_ENV_SCRIPT))
    os.environ.setdefault("C2HLS_DEVICE_PLATFORM", "xilinx_u280_gen3x16_xdma_1_202211_1")
    os.environ.setdefault("C2HLS_VITIS_VERSION", "2023.2")
    os.environ.setdefault("C2HLS_PART", "xcu280-fsvh2892-2L-e")
    os.environ.setdefault("C2HLS_CLOCK_NS", "3.33")
    os.environ.setdefault("C2HLS_FLOW_TARGET", "vitis")

    if profile == "sweep":
        tmp = os.getenv("C2HLS_SWEEP_TMP_ROOT") or os.getenv("C2HLS_TMP_ROOT")
        if tmp:
            os.environ.setdefault("C2HLS_TMP_ROOT", tmp)
