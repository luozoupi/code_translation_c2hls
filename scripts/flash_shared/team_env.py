"""Team-server bootstrap for flash API batches (no PC2 / vLLM).

Uses the same path defaults as ``run_agentic_sweep.py`` via ``c2hls_paths.TEAM_DEFAULTS``.
Existing ``.env`` / shell exports win over defaults (``setdefault`` only).
"""

from __future__ import annotations

import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

# Flash pilot flags for team API runs (cosim controlled separately — see apply_flash_cosim_env).
FLASH_PILOT_ENV: dict[str, str] = {
    "C2HLS_PHASEB_MODE": "functional",
    "C2HLS_PHASE8_BASELINE_ALIGN": "0",
    "C2HLS_PHASE5_GT_PREPOP": "0",
    "C2HLS_HW_EMU_FINAL": "0",
    "C2HLS_HW_EMU_DISABLE_DEBUG_SYMBOLS": "1",
    "C2HLS_GT_BASELINE_FALLBACK": "1",
    "C2HLS_SYNTH_TIMEOUT": "1200",
    "C2HLS_CSIM_TIMEOUT": "180",
    "C2HLS_COSIM_TIMEOUT": "1200",
    "C2HLS_LLM_TIMEOUT": "900",
}

# Team default: full cosim (matches run_agentic_sweep). PC2 pilot skips via local.env / vLLM libs.
_FLASH_COSIM_ENABLED: dict[str, str] = {
    "C2HLS_RUN_COSIM": "1",
    "C2HLS_COSIM_REQUIRED": "1",
    "C2HLS_REFERENCE_COSIM": "1",
    "C2HLS_COSIM_TRACE_LEVEL": "none",
}

_FLASH_COSIM_SKIPPED: dict[str, str] = {
    "C2HLS_RUN_COSIM": "0",
    "C2HLS_COSIM_REQUIRED": "0",
    "C2HLS_REFERENCE_COSIM": "0",
    "C2HLS_COSIM_TRACE_LEVEL": "none",
}

_SKIP_COSIM_CHOICE: bool | None = None


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(REPO / ".env")
    load_dotenv(REPO / ".env.local")


def _ensure_writable_tmp() -> None:
    """Pick a scratch root that exists; fall back to in-repo ``c2hls_tmp/``."""
    from c2hls_paths import REPO_TMP_DIR

    candidates: list[Path] = []
    for key in ("C2HLS_SWEEP_TMP_ROOT", "C2HLS_TMP_ROOT"):
        raw = os.getenv(key, "").strip()
        if raw:
            candidates.append(Path(raw).expanduser())
    candidates.append(REPO_TMP_DIR)

    for cand in candidates:
        try:
            cand.mkdir(parents=True, exist_ok=True)
            test = cand / ".flash_api_write_test"
            test.write_text("ok")
            test.unlink(missing_ok=True)
            os.environ["C2HLS_TMP_ROOT"] = str(cand)
            return
        except OSError:
            continue
    raise RuntimeError(
        "No writable C2HLS_TMP_ROOT. Set C2HLS_TMP_ROOT or C2HLS_SWEEP_TMP_ROOT "
        f"(team default /mnt/data/luo00466/tmp may be unavailable on this host)."
    )


def resolve_skip_cosim(*, cli: bool | None = None) -> bool:
    """True when team flash API should skip cosim (csynth + csim only)."""
    global _SKIP_COSIM_CHOICE
    if cli is not None:
        return cli
    if _SKIP_COSIM_CHOICE is not None:
        return _SKIP_COSIM_CHOICE
    raw = os.getenv("C2HLS_FLASH_API_SKIP_COSIM", "").strip().lower()
    return raw in ("1", "true", "yes", "on")


def apply_flash_cosim_env(*, skip: bool) -> None:
    """Set cosim env for team flash API runs (not used by PC2 cosim-repair tooling)."""
    bundle = _FLASH_COSIM_SKIPPED if skip else _FLASH_COSIM_ENABLED
    for key, value in bundle.items():
        os.environ[key] = value


def flash_cosim_manifest() -> dict[str, str]:
    """Cosim mode snapshot for manifest.json."""
    skip = resolve_skip_cosim()
    return {
        "skip_cosim": "1" if skip else "0",
        "C2HLS_RUN_COSIM": os.getenv("C2HLS_RUN_COSIM", ""),
        "C2HLS_COSIM_REQUIRED": os.getenv("C2HLS_COSIM_REQUIRED", ""),
        "C2HLS_REFERENCE_COSIM": os.getenv("C2HLS_REFERENCE_COSIM", ""),
    }


def bootstrap_team_flash_env(*, skip_cosim: bool | None = None) -> None:
    """Team paths + flash experiment mode. Never touches PC2 / OPENAI_BASE_URL."""
    global _SKIP_COSIM_CHOICE
    if skip_cosim is not None:
        _SKIP_COSIM_CHOICE = skip_cosim

    _load_dotenv()

    os.environ.pop("C2HLS_SITE", None)
    os.environ.pop("OPENAI_BASE_URL", None)
    os.environ["C2HLS_FLASH_EXPERIMENT"] = "1"

    from c2hls_paths import apply_runtime_defaults, configure_site

    configure_site("team")
    apply_runtime_defaults(profile="sweep")
    _ensure_writable_tmp()

    from c2hls_temp import configure_temp_env

    configure_temp_env(create=True)

    for key, value in FLASH_PILOT_ENV.items():
        os.environ.setdefault(key, value)

    apply_flash_cosim_env(skip=resolve_skip_cosim())


def finalize_api_llm_env() -> None:
    """Let c2hls load keys from C2HLS_CLAUDE_KEY_FILE / C2HLS_OPENAI_KEY_FILE."""
    if os.environ.get("OPENAI_API_KEY", "").strip() in {"", "EMPTY"}:
        os.environ.pop("OPENAI_API_KEY", None)
    if os.environ.get("ANTHROPIC_API_KEY", "").strip() in {"", "EMPTY"}:
        os.environ.pop("ANTHROPIC_API_KEY", None)


def _is_hosted_openai_model(model_id: str) -> bool:
    model = (model_id or "").lower()
    return model.startswith(("gpt-", "o1", "o3", "o4", "codex-"))


def _is_claude_model(model_id: str) -> bool:
    return "claude" in (model_id or "").lower()


def preflight_api_run(model_id: str) -> list[str]:
    """Return human-readable blockers before spending API credits."""
    errors: list[str] = []

    vitis = os.getenv("C2HLS_VITIS_SETTINGS", "").strip()
    if not vitis:
        errors.append(
            "C2HLS_VITIS_SETTINGS is unset. Team default applies after configure_site('team'); "
            "set in .env if your Vitis install differs."
        )
    elif not Path(vitis).is_file():
        errors.append(
            f"C2HLS_VITIS_SETTINGS points to missing file: {vitis}\n"
            "  Fix: set in .env or export (team default is in c2hls_paths.TEAM_DEFAULTS)."
        )

    emu = os.getenv("C2HLS_EMU_ENV_SCRIPT", "").strip()
    if emu and not Path(emu).is_file():
        errors.append(f"C2HLS_EMU_ENV_SCRIPT not found: {emu}")

    if _is_claude_model(model_id):
        key_path = os.getenv("C2HLS_CLAUDE_KEY_FILE", "").strip()
        if key_path and not Path(key_path).is_file():
            errors.append(
                f"C2HLS_CLAUDE_KEY_FILE not found: {key_path}\n"
                "  Fix: export path to your Claude API key file (team default "
                "/home/luo00466/claude-api-key.txt)."
            )
        if not key_path and not os.getenv("ANTHROPIC_API_KEY", "").strip():
            errors.append(
                "Claude model requested but no API key: set C2HLS_CLAUDE_KEY_FILE "
                "or ANTHROPIC_API_KEY."
            )
    elif _is_hosted_openai_model(model_id):
        key_path = os.getenv("C2HLS_OPENAI_KEY_FILE", "").strip()
        if key_path and not Path(key_path).is_file():
            errors.append(f"C2HLS_OPENAI_KEY_FILE not found: {key_path}")
        if not key_path and not os.getenv("OPENAI_API_KEY", "").strip():
            errors.append(
                "OpenAI model requested but no API key: set C2HLS_OPENAI_KEY_FILE "
                "or OPENAI_API_KEY."
            )

    if os.getenv("OPENAI_BASE_URL", "").strip():
        errors.append(
            "OPENAI_BASE_URL is set (self-hosted vLLM). Unset it for commercial API runs:\n"
            "  unset OPENAI_BASE_URL"
        )

    return errors


def active_team_paths_summary() -> dict[str, str]:
    """Snapshot for manifest.json — shows what the run will use."""
    keys = (
        "C2HLS_VITIS_SETTINGS",
        "C2HLS_XRT_SETUP",
        "C2HLS_PLATFORM_REPO_PATHS",
        "C2HLS_TMP_ROOT",
        "C2HLS_CLAUDE_KEY_FILE",
        "C2HLS_OPENAI_KEY_FILE",
        "C2HLS_DEVICE_PLATFORM",
        "C2HLS_PART",
    )
    return {k: os.getenv(k, "") for k in keys if os.getenv(k, "")}
