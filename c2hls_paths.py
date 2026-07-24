"""Repository and machine path resolution for c2hls.

Repo-relative paths (benchmarks, skills package, scripts, …) are derived from
``REPO_ROOT`` and are always portable.

Machine-specific paths use one of three **sites**:

* **team** (default) — team's development server; commercial API LLM inference.
* **pc2** — PC2 cluster; open-weight vLLM + module Vitis (``local.env``).
* **fir** — Alliance Canada Fir; open-weight vLLM + scratch Vitis (``fir.env``).

Select a site with ``--pc2`` / ``--fir`` or ``C2HLS_SITE``. Example::

    ./c2hls.py --pc2 --bench nw
    ./c2hls.py --fir --bench-dir benchmarks/hlsfactory_gemm --multistep --strategy flash
    C2HLS_SITE=fir python run_agentic_sweep.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

# --- In-repo paths (committed) ------------------------------------------------

BENCHMARKS_DIR = REPO_ROOT / "benchmarks"
BENCHMARKS_COSIM_DIR = REPO_ROOT / "benchmarks_cosim"
BENCHMARKS_AUTOSA_DSE_DIR = REPO_ROOT / "benchmarks_autosa_dse"
BENCHMARKS_EXTERNAL_DIR = REPO_ROOT / "benchmarks_external"
EXTERNAL_DATASETS_DIR = REPO_ROOT / "external_datasets"
SKILLS_PACKAGE_DIR = REPO_ROOT / "hls_full_optimization_skills_schema_1_1_package"
SKILLS_MUTABLE_DIR = REPO_ROOT / "skills"
ANALYSIS_DIR = REPO_ROOT / "analysis"
ARTIFACTS_DIR = REPO_ROOT / "artifacts"
PC2_ARTIFACTS_DIR = ARTIFACTS_DIR / "pc2"
FIR_ARTIFACTS_DIR = ARTIFACTS_DIR / "fir"
TEAM_ARTIFACTS_DIR = ARTIFACTS_DIR / "team"
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
# Flash: packaged base (73 or 90 skills) + standalone overlay at runtime.
FLASH_NO_RMW_M_AXI_SKILL_ENTRIES_JSON = (
    SKILLS_PACKAGE_DIR / "flash_no_RMW_m_axi_skill_entries.json"
)
# Default packaged base for new-matrix tooling.
SKILLS_II_TARGET_MISS_JSON = SKILLS_II_TARGET_MISS_90_JSON
VITIS_PRAGMAS_CURATED_MD = SKILLS_PACKAGE_DIR / "vitis_hls_2023_2_pragmas_curated.md"

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

# Fir (Alliance Canada) — Apptainer Vitis SIF on compute; vLLM via ../inference/.
_FIR_SCRATCH = "/scratch/asa582"
FIR_DEFAULTS: dict[str, str] = {
    "C2HLS_XILINX_SIF": f"{_FIR_SCRATCH}/containers/xilinx_vitis_2023.2.standalone.sif",
    "C2HLS_USE_CONTAINER": "1",
    "C2HLS_TMP_ROOT": f"{_FIR_SCRATCH}/tmp/c2hls",
    "C2HLS_VITIS_USER_HOME": f"{_FIR_SCRATCH}/tmp/vitis_user_home",
    "C2HLS_MODEL": "mistralai/Devstral-2-123B-Instruct-2512",
    "FIR_LLM_MODEL": "mistralai/Devstral-2-123B-Instruct-2512",
    "OPENAI_API_KEY": "EMPTY",
    "OPENAI_BASE_URL": "http://127.0.0.1:8000/v1",
    "FIR_INFERENCE_ROOT": f"{_FIR_SCRATCH}/workspaces/inference",
    "FIR_GPU_MODULES": "python/3.11.5 cuda/12.6",
    "FIR_GPU_PARTITION": "gpubase_bynode_b1",
    "FIR_COMPUTE_PARTITION": "cpubase_bynode_b1",
    "FIR_SLURM_ACCOUNT": "def-zhenman_gpu",
    "FIR_WALLTIME": "3:00:00",
    "FIR_LLM_PORT": "8000",
    "FIR_GPU_GPUS": "4",
    "FIR_VLLM_TENSOR_PARALLEL_SIZE": "4",
    # Fir pilot: csynth + csim only (same as PC2 flash runs).
    "C2HLS_RUN_COSIM": "0",
    "C2HLS_COSIM_REQUIRED": "0",
    "C2HLS_REFERENCE_COSIM": "0",
    "C2HLS_HW_EMU_FINAL": "0",
    "C2HLS_GT_BASELINE_FALLBACK": "1",
}

SITES = frozenset({"team", "pc2", "fir"})
_OPEN_WEIGHT_SITES = frozenset({"pc2", "fir"})
_ACTIVE_SITE: str | None = None


def is_open_weight_site(site: str | None = None) -> bool:
    """True for clusters that use self-hosted vLLM (not commercial API)."""
    if site is None:
        site = active_site()
    return site in _OPEN_WEIGHT_SITES


def bootstrap_site(argv: list[str] | None = None) -> str:
    """Detect site from ``C2HLS_SITE``, ``--pc2``, or ``--fir`` in *argv*."""
    if argv is None:
        argv = sys.argv
    site = os.environ.get("C2HLS_SITE", "").strip().lower()
    if "--pc2" in argv:
        site = "pc2"
        os.environ["C2HLS_SITE"] = "pc2"
    elif "--fir" in argv:
        site = "fir"
        os.environ["C2HLS_SITE"] = "fir"
    if site not in SITES:
        site = "team"
        os.environ.setdefault("C2HLS_SITE", "team")
    return site


def active_site() -> str:
    return _ACTIVE_SITE or bootstrap_site()


def site_artifacts_dir(site: str | None = None) -> Path:
    """Per-site artifact root (pc2 / fir / flash_api / team)."""
    site = (site or active_site()).lower()
    if site == "pc2":
        return PC2_ARTIFACTS_DIR
    if site == "fir":
        return FIR_ARTIFACTS_DIR
    if site == "flash_api":
        return FLASH_API_ARTIFACTS_DIR
    return TEAM_ARTIFACTS_DIR


def _parse_env_file(path: Path) -> None:
    """Load KEY=VALUE lines from *path* (no python-dotenv required)."""
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)


def _load_site_env_file(filename: str) -> None:
    _parse_env_file(REPO_ROOT / filename)
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(REPO_ROOT / filename, override=False)


def configure_site(site: str | None = None) -> str:
    """Apply path defaults for *site* (auto-detected when omitted). Idempotent."""
    global _ACTIVE_SITE
    if site is None:
        site = bootstrap_site()
    if _ACTIVE_SITE == site:
        return site
    _ACTIVE_SITE = site

    if site == "pc2":
        _load_site_env_file("local.env")
        _load_site_env_file(".env")
        for key, value in PC2_DEFAULTS.items():
            os.environ.setdefault(key, value)
        os.environ.setdefault("C2HLS_TMP_ROOT", str(REPO_TMP_DIR))
    elif site == "fir":
        _load_site_env_file("fir.env")
        for key, value in FIR_DEFAULTS.items():
            os.environ.setdefault(key, value)
    else:
        _load_site_env_file(".env")
        for key, value in TEAM_DEFAULTS.items():
            os.environ.setdefault(key, value)

    apply_runtime_defaults()
    return site


def load_local_env() -> None:
    """Backward-compatible alias: configure the active site."""
    configure_site()


def add_site_argument(parser) -> None:
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--pc2",
        action="store_true",
        help="PC2 cluster: open-weight vLLM + module Vitis (local.env)",
    )
    group.add_argument(
        "--fir",
        action="store_true",
        help="Alliance Fir: open-weight vLLM + scratch Vitis (fir.env)",
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
            f"{name} is not set. For PC2 copy local.env.example to local.env; "
            "for Fir copy fir.env.example to fir.env."
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
