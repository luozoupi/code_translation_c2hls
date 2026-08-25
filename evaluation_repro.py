"""Reproducibility and status guards for paper evaluation runs.

This module deliberately has no dependency on the C2HLS controller.  The
sweep driver can therefore compute a run identity *before* importing or
calling the controller and can reject stale artifacts without executing any
LLM or synthesis work.

The fingerprint is content based.  It covers the active implementation and
prompt sources, every benchmark file, model/decoding configuration, the
skill snapshot, toolchain target, and evaluation budgets.  Absolute output
paths and timestamps are excluded because neither changes the experiment.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Iterable, Mapping, MutableMapping, Sequence
from urllib.parse import urlsplit


FINGERPRINT_SCHEMA = "c2hls.run-fingerprint.v1"
PAPER_PROFILE = "hpca2027_reference_blind"

# These settings are invariants of the reference-isolated paper evaluation. They are
# assignments, not setdefault calls: a shell inherited from an oracle run
# must not silently turn reference guidance back on.
REFERENCE_BLIND_OVERRIDES: dict[str, str] = {
    "C2HLS_REFERENCE_BLIND": "1",
    "C2HLS_ORACLE_MODE": "0",
    "C2HLS_GT_AWARE_REVERT": "0",
    "C2HLS_PHASE8_BASELINE_ALIGN": "0",
    "C2HLS_SWEEP_BASELINE_ALIGN": "0",
    "C2HLS_PHASE5_GT_PREPOP": "0",
    "C2HLS_SWEEP_GT_PREPOP": "0",
    "C2HLS_COSIM_SKIP_SLOWER_THAN_GOLD": "0",
    "C2HLS_GOLD_RELATIVE_COSIM": "0",
    "C2HLS_REFERENCE_METRICS_IN_PROMPTS": "0",
    "C2HLS_REFERENCE_CODE_IN_PROMPTS": "0",
    "C2HLS_GT_COMPARISON_IN_CONTROL": "0",
    "C2HLS_SKILL_LIBRARY_PERSIST": "0",
    "C2HLS_SKILL_LIBRARY_FROZEN": "1",
    "C2HLS_SKILL_UPDATE_STATS": "0",
    "C2HLS_TRANSCRIPT_AUDIT": "1",
    "C2HLS_REFERENCE_BLIND_FAIL_ON_LEAK": "1",
    "C2HLS_COSIM_SELECTED_ONLY": "1",
    "C2HLS_FORCE_SELECTED_COSIM": "1",
    "C2HLS_COSIM_REQUIRED": "0",
    "C2HLS_REFERENCE_VALIDATE_MODE": "all",
    "C2HLS_REFERENCE_COSIM": "0",
    "C2HLS_REFERENCE_COSIM_SELECTED_ONLY": "1",
    "C2HLS_REFERENCE_COSIM_BASELINE": "1",
    # The optional feedback composer makes an additional 800-token LLM call.
    # It is outside the preregistered one-candidate/one-call accounting and
    # must not be inherited from an interactive shell during paper runs.
    "C2HLS_FEEDBACK_LLM": "0",
    "C2HLS_FEASIBILITY_SELECTION": "1",
    "C2HLS_CORRECTNESS_BEFORE_SYNTH": "1",
    "C2HLS_DISABLE_CORRECTNESS_REPAIR": "0",
    "C2HLS_SYNTH_REVERT_THRESHOLD": "0",
    "C2HLS_PHASE5_LLM_RETRY": "0",
    "C2HLS_CPU_GOLDEN_TIMEOUT": "180",
    "C2HLS_SYNTHESIS_EVAL_BUDGET": "5",
    "C2HLS_LLM_CANDIDATE_BUDGET": "5",
}

# Post-route implementation is optional (C2HLS_HW_EMU_FINAL remains a
# caller-controlled, preregistered matrix choice), but when it is requested
# these controls must not drift with the invoking shell.
PAPER_POST_ROUTE_OVERRIDES: dict[str, str] = {
    "C2HLS_ALLOW_WIDE_ABI": "0",
    "C2HLS_HW_EMU_DISABLE_DEBUG_SYMBOLS": "1",
    "C2HLS_HW_EMU_CLOCK_NS": "3.33",
    "C2HLS_HW_EMU_CLOCK_MHZ": "",
    "C2HLS_EMU_ENV_SCRIPT": str(
        Path(__file__).resolve().parent / "scripts" / "setup_emu_env.sh"
    ),
}
PAPER_PROFILE_OVERRIDES: dict[str, str] = {
    **REFERENCE_BLIND_OVERRIDES,
    **PAPER_POST_ROUTE_OVERRIDES,
}

_IMPLEMENTATION_SOURCES = (
    "bottleneck_router.py",
    "c2hls.py",
    "c2hls_temp.py",
    "candidate_cache.py",
    "evaluation_repro.py",
    "export_schema_jsonl.py",
    "golden_output.py",
    "hls_eval.py",
    "hls_feedback.py",
    "qor_design_space.py",
    "reference_isolation.py",
    "robustness.py",
    "rubric.py",
    "run_agentic_sweep.py",
    "skill_library.py",
    "smart_skill_router.py",
    "trajectory_alignment.py",
    "configs/hlsfactory_output_shapes.json",
    "configs/hlsfactory_output_shapes_fc27133.json",
    "configs/hlsfactory_development_suite.json",
)
_PROMPT_SOURCES = ("prompt_c2hls.py",)
_SKILL_SOURCES = (
    "skills/skills.json",
    "hls_full_optimization_skills_schema_1_1_package/skills.json",
)
_CONTROL_ENV_NAMES = (
    "C2HLS_ALLOW_WIDE_ABI",
    "C2HLS_ATTEMPTS_PER_CANDIDATE",
    "C2HLS_CANDIDATES_PER_STEP",
    "C2HLS_COMPILE_CHECK_TIMEOUT",
    "C2HLS_CORRECTNESS_BEFORE_SYNTH",
    "C2HLS_CPU_GOLDEN_TIMEOUT",
    "C2HLS_COSIM_REQUIRED",
    "C2HLS_COSIM_SELECTED_ONLY",
    "C2HLS_COSIM_SKIP_GOLD_RATIO",
    "C2HLS_COSIM_SKIP_SLOWER_THAN_GOLD",
    "C2HLS_COSIM_TIMEOUT",
    "C2HLS_COSIM_TRACE_LEVEL",
    "C2HLS_FORCE_SELECTED_COSIM",
    "C2HLS_CSIM_TIMEOUT",
    "C2HLS_DYNAMIC_ROUTING",
    "C2HLS_DISABLE_CORRECTNESS_REPAIR",
    "C2HLS_EXHAUSTIVE_CANDIDATE_ATTEMPTS",
    "C2HLS_FEEDBACK_LLM",
    "C2HLS_FEEDBACK_MODEL",
    "C2HLS_FEASIBILITY_SELECTION",
    "C2HLS_FLOW_TARGET",
    "C2HLS_FORCE_SKILL_PROMPTS",
    "C2HLS_GT_AWARE_REVERT",
    "C2HLS_GT_COMPARISON_IN_CONTROL",
    "C2HLS_HLSFACTORY_SHAPE_REGISTRY",
    "C2HLS_HW_EMU_FINAL",
    "C2HLS_HW_EMU_CLOCK_MHZ",
    "C2HLS_HW_EMU_CLOCK_NS",
    "C2HLS_HW_EMU_DISABLE_DEBUG_SYMBOLS",
    "C2HLS_HW_EMU_TIMEOUT",
    "C2HLS_LLM_SEED",
    "C2HLS_LLM_CANDIDATE_BUDGET",
    "C2HLS_LLM_TEMPERATURE",
    "C2HLS_LLM_TIMEOUT",
    "C2HLS_LLM_TOP_P",
    "C2HLS_DEEPSEEK_THINKING",
    "C2HLS_DEEPSEEK_REASONING_EFFORT",
    "C2HLS_XAI_REASONING_EFFORT",
    "C2HLS_MAX_COMPLETION_TOKENS",
    "C2HLS_MODEL_REVISION",
    "C2HLS_ORACLE_MODE",
    "C2HLS_PHASE5_GT_PREPOP",
    "C2HLS_PHASE5_LLM_RETRY",
    "C2HLS_PHASE7A",
    "C2HLS_PHASE8_BASELINE_ALIGN",
    "C2HLS_PHASEB_MODE",
    "C2HLS_QUALITY_REPAIR_MODEL",
    "C2HLS_QUALITY_REPAIR_TURNS",
    "C2HLS_QUALITY_SCORE_EPSILON",
    "C2HLS_QOR_DESIGN_SWEEP",
    "C2HLS_QOR_SWEEP_II_VALUES",
    "C2HLS_QOR_SWEEP_INTERACTIONS",
    "C2HLS_QOR_SWEEP_MAX_CANDIDATES",
    "C2HLS_QOR_SWEEP_MAX_INTERACTIONS",
    "C2HLS_QOR_SWEEP_MAX_KNOBS",
    "C2HLS_QOR_SWEEP_TILE_VALUES",
    "C2HLS_QOR_SWEEP_VALUES",
    "C2HLS_REFERENCE_BLIND",
    "C2HLS_REFERENCE_CACHE_REQUIRE_COSIM",
    "C2HLS_REFERENCE_CODE_IN_PROMPTS",
    "C2HLS_REFERENCE_COSIM",
    "C2HLS_REFERENCE_COSIM_BASELINE",
    "C2HLS_REFERENCE_COSIM_SELECTED_ONLY",
    "C2HLS_REFERENCE_METRICS_IN_PROMPTS",
    "C2HLS_REFERENCE_VALIDATE_MODE",
    "C2HLS_SWEEP_HW_EMU",
    "C2HLS_SWEEP_REFERENCE_CACHE_DIR",
    "C2HLS_SKILL_LIBRARY_FROZEN",
    "C2HLS_SKILL_LIBRARY_PERSIST",
    "C2HLS_SKILL_MODE",
    "C2HLS_SKILL_PROMPT_MODE",
    "C2HLS_SKILL_PROMPT_SCOPE",
    "C2HLS_SKILL_SNAPSHOT_SHA256",
    "C2HLS_SKILL_UPDATE_STATS",
    "C2HLS_STEP_REGRESSION_THRESHOLD",
    "C2HLS_STRATEGY",
    "C2HLS_SYNTHESIS_EVAL_BUDGET",
    "C2HLS_SYNTHESIS_MODEL",
    "C2HLS_SYNTH_REVERT_THRESHOLD",
    "C2HLS_SYNTH_TIMEOUT",
    "C2HLS_TOP_P",
    "C2HLS_TEMPERATURE",
    "C2HLS_TRANSLATOR_MODEL",
    "C2HLS_TURNS",
)

_VITIS_TRANSIENT_DIRECTORY_NAMES = {
    ".cache",
    "cache",
    "log",
    "logs",
    "temp",
    "tmp",
    "vitis_hls",
    "xsim",
}
_VITIS_TRANSIENT_SUFFIXES = {
    ".jou",
    ".lock",
    ".log",
    ".pb",
    ".pid",
    ".str",
    ".tmp",
    ".wdb",
    ".xmsgs",
}


def canonical_json(value: Any) -> str:
    """Return the sole serialization used to compute run identities."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json(value).encode("utf-8"))


def _bool_env(environ: Mapping[str, str], name: str, default: str = "0") -> bool:
    return str(environ.get(name, default)).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def apply_evaluation_profile(
    profile: str | None = None,
    *,
    environ: MutableMapping[str, str] | None = None,
) -> dict[str, Any]:
    """Apply a named evaluation profile and return an auditable change log.

    ``legacy`` and ``none`` preserve the historical environment.  The sweep
    driver defaults to :data:`PAPER_PROFILE`; callers that truly need the old
    reference-guided behavior must now request it explicitly.
    """

    env = os.environ if environ is None else environ
    requested = (profile or env.get("C2HLS_SWEEP_PROFILE") or PAPER_PROFILE).strip()
    normalized = requested.lower().replace("-", "_")
    if normalized in {"legacy", "none", "off"}:
        env["C2HLS_SWEEP_PROFILE"] = "legacy"
        return {
            "name": "legacy",
            "reference_blind": False,
            "forced_overrides": {},
        }
    if normalized not in {
        PAPER_PROFILE,
        "hpca2027",
        "paper",
        "reference_blind",
    }:
        raise ValueError(f"unknown C2HLS_SWEEP_PROFILE: {requested!r}")

    changed: dict[str, dict[str, str | None]] = {}
    for key, value in PAPER_PROFILE_OVERRIDES.items():
        previous = env.get(key)
        if previous != value:
            changed[key] = {"previous": previous, "effective": value}
        env[key] = value
    env["C2HLS_SWEEP_PROFILE"] = PAPER_PROFILE
    return {
        "name": PAPER_PROFILE,
        "reference_blind": True,
        "forced_overrides": changed,
        "invariants": dict(PAPER_PROFILE_OVERRIDES),
    }


def _record_file(path: Path, *, relative_to: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    try:
        rel = path.resolve().relative_to(relative_to.resolve()).as_posix()
    except ValueError:
        # Only the basename is retained for an explicitly configured external
        # snapshot.  The content digest is the identity; host-specific paths
        # must not make otherwise identical runs differ.
        rel = path.name
    return {"path": rel, "bytes": len(raw), "sha256": sha256_bytes(raw)}


def _manifest(paths: Iterable[Path], *, relative_to: Path) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for raw_path in paths:
        path = Path(raw_path)
        if not path.is_file():
            continue
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        records.append(_record_file(path, relative_to=relative_to))
    records.sort(key=lambda item: item["path"])
    return {
        "files": records,
        "file_count": len(records),
        "sha256": sha256_json(records),
    }


def _benchmark_files(benchmark_dir: Path) -> list[Path]:
    excluded_dirs = {".git", "__pycache__", "results", "result", "build"}
    excluded_suffixes = {".log", ".jou", ".str", ".tmp", ".pyc"}
    files: list[Path] = []
    for path in benchmark_dir.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(benchmark_dir)
        if any(part in excluded_dirs for part in rel.parts[:-1]):
            continue
        if path.suffix.lower() in excluded_suffixes:
            continue
        files.append(path)
    return sorted(files)


def _git_head(repo: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def _env_value(environ: Mapping[str, str], *names: str, default: Any = None) -> Any:
    for name in names:
        value = environ.get(name)
        if value not in (None, ""):
            return value
    return default


def _path_identity(path: Path, *, repo: Path) -> dict[str, Any]:
    """Bind a path without publishing host-specific absolute directory names."""

    expanded = Path(path).expanduser()
    if not expanded.is_absolute():
        return {
            "scope": "relative",
            "absolute": False,
            "basename": expanded.name,
            "path_sha256": sha256_bytes(str(expanded).encode("utf-8")),
        }
    resolved = expanded.resolve(strict=False)
    try:
        relative = resolved.relative_to(repo.resolve()).as_posix()
    except ValueError:
        return {
            "scope": "external",
            "absolute": True,
            "basename": resolved.name,
            "path_sha256": sha256_bytes(str(resolved).encode("utf-8")),
        }
    return {
        "scope": "repository",
        "absolute": True,
        "path": relative,
    }


def _relative_file_manifest(root: Path, paths: Iterable[Path]) -> dict[str, Any]:
    """Hash selected files by their lexical path below ``root``.

    Only content digests and relative names are retained.  Read failures are
    explicit and are part of the aggregate identity, so an unreadable input
    cannot be mistaken for an empty one.
    """

    root = Path(root)
    records: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    seen: set[str] = set()
    for path in sorted((Path(item) for item in paths), key=lambda item: item.as_posix()):
        try:
            relative = path.relative_to(root).as_posix()
        except ValueError:
            relative = path.name
        if relative in seen:
            continue
        seen.add(relative)
        try:
            raw = path.read_bytes()
        except OSError as exc:
            errors.append({"path": relative, "error": type(exc).__name__})
            continue
        records.append(
            {"path": relative, "bytes": len(raw), "sha256": sha256_bytes(raw)}
        )
    records.sort(key=lambda item: item["path"])
    errors.sort(key=lambda item: item["path"])
    digest_material = {"files": records, "errors": errors}
    return {
        "files": records,
        "file_count": len(records),
        "errors": errors,
        "sha256": sha256_json(digest_material),
    }


def _endpoint_identity(model_id: str, environ: Mapping[str, str]) -> dict[str, Any]:
    """Return the active provider endpoint identity without recording secrets."""

    lowered = (model_id or "").lower()
    if lowered.startswith("claude"):
        provider = "anthropic"
        source_env = "ANTHROPIC_BASE_URL"
        default_url = "https://api.anthropic.com"
    elif lowered.startswith(("gpt-", "o1", "o3", "o4", "codex-")):
        provider = "openai_hosted"
        source_env = "C2HLS_OPENAI_HOSTED_URL"
        default_url = "https://api.openai.com/v1"
    else:
        provider = "openai_compatible"
        source_env = "OPENAI_BASE_URL"
        default_url = "http://127.0.0.1:8000/v1"

    configured = str(environ.get(source_env, "")).strip()
    raw_url = configured or default_url
    try:
        parsed = urlsplit(raw_url)
        hostname = (parsed.hostname or "").lower()
        port = parsed.port
    except ValueError:
        return {
            "provider": provider,
            "source_env": source_env,
            "explicit": bool(configured),
            "valid": False,
            "unsafe_components": ["unparseable_url"],
            "endpoint_sha256": None,
        }

    unsafe_components: list[str] = []
    if parsed.username is not None or parsed.password is not None:
        unsafe_components.append("userinfo")
    if parsed.query:
        unsafe_components.append("query")
    if parsed.fragment:
        unsafe_components.append("fragment")

    scheme = parsed.scheme.lower()
    path = parsed.path or "/"
    host_class = (
        "loopback"
        if hostname in {"localhost", "127.0.0.1", "::1"}
        else "remote"
    )
    safe_components = {
        "scheme": scheme,
        "host_sha256": (
            sha256_bytes(hostname.encode("utf-8")) if hostname else None
        ),
        "port": port,
        "path_sha256": sha256_bytes(path.encode("utf-8")),
    }
    valid = bool(
        scheme in {"http", "https"}
        and hostname
        and not unsafe_components
    )
    return {
        "provider": provider,
        "source_env": source_env,
        "explicit": bool(configured),
        "valid": valid,
        "unsafe_components": unsafe_components,
        "host_class": host_class,
        **safe_components,
        "endpoint_sha256": sha256_json(safe_components),
    }


def _reference_cache_manifest(
    repo: Path,
    benchmark: str,
    environ: Mapping[str, str],
) -> dict[str, Any]:
    """Bind the configured cache location and this benchmark's cache entries."""

    configured = str(environ.get("C2HLS_REFERENCE_CACHE_DIR", "")).strip()
    if not configured:
        return {
            "enabled": False,
            "path": None,
            "state": "disabled",
            "files": [],
            "file_count": 0,
            "errors": [],
            "sha256": sha256_json([]),
        }

    configured_path = Path(configured).expanduser()
    root = configured_path if configured_path.is_absolute() else repo / configured_path
    root = root.resolve(strict=False)
    result: dict[str, Any] = {
        "enabled": True,
        "path": _path_identity(root, repo=repo),
        "configured_absolute": configured_path.is_absolute(),
    }
    if not root.exists():
        result.update(
            {
                "state": "directory_absent",
                "files": [],
                "file_count": 0,
                "errors": [],
                "sha256": sha256_json([]),
            }
        )
        return result
    if not root.is_dir():
        result.update(
            {
                "state": "not_directory",
                "files": [],
                "file_count": 0,
                "errors": [],
                "sha256": sha256_json([]),
            }
        )
        return result

    safe_benchmark = re.sub(r"[^A-Za-z0-9_.-]+", "_", benchmark).strip("_") or "benchmark"
    entry_pattern = re.compile(
        rf"{re.escape(safe_benchmark)}\.[0-9a-f]{{64}}\.json\Z",
        re.IGNORECASE,
    )
    try:
        matching = [
            path
            for path in root.iterdir()
            if path.is_file() and entry_pattern.fullmatch(path.name)
        ]
    except OSError as exc:
        result.update(
            {
                "state": "unreadable",
                "files": [],
                "file_count": 0,
                "errors": [{"path": ".", "error": type(exc).__name__}],
                "sha256": sha256_json(
                    {"files": [], "errors": [{"path": ".", "error": type(exc).__name__}]}
                ),
            }
        )
        return result

    manifest = _relative_file_manifest(root, matching)
    result.update(manifest)
    result["state"] = (
        "unreadable"
        if manifest["errors"]
        else "present"
        if manifest["file_count"]
        else "entry_absent"
    )
    return result


def _vitis_user_home_manifest(
    repo: Path,
    environ: Mapping[str, str],
) -> dict[str, Any]:
    """Bind deterministic configuration state read through Vitis's HOME."""

    raw_home = str(environ.get("C2HLS_VITIS_USER_HOME", "")).strip()
    if raw_home:
        home = Path(raw_home).expanduser()
        source = "C2HLS_VITIS_USER_HOME"
    else:
        raw_temp = str(
            environ.get("C2HLS_TMP_ROOT", "/mnt/data/luo00466/tmp")
        ).strip()
        temp_root = Path(raw_temp or "/mnt/data/luo00466/tmp").expanduser()
        home = temp_root / "vitis_user_home"
        source = "C2HLS_TMP_ROOT/default"

    result: dict[str, Any] = {
        "source": source,
        "configured_absolute": home.is_absolute(),
        "path": _path_identity(home, repo=repo),
    }
    if not home.is_absolute():
        result.update(
            {
                "state": "invalid_relative_path",
                "files": [],
                "file_count": 0,
                "errors": [],
                "sha256": sha256_json([]),
            }
        )
        return result
    home = home.resolve(strict=False)
    if not home.exists():
        result.update(
            {
                "state": "home_absent",
                "files": [],
                "file_count": 0,
                "errors": [],
                "sha256": sha256_json([]),
            }
        )
        return result
    if not home.is_dir():
        result.update(
            {
                "state": "home_not_directory",
                "files": [],
                "file_count": 0,
                "errors": [],
                "sha256": sha256_json([]),
            }
        )
        return result

    xilinx_root = home / ".Xilinx"
    if not xilinx_root.exists():
        result.update(
            {
                "state": "xilinx_state_absent",
                "files": [],
                "file_count": 0,
                "errors": [],
                "sha256": sha256_json([]),
            }
        )
        return result
    if not xilinx_root.is_dir():
        result.update(
            {
                "state": "xilinx_state_not_directory",
                "files": [],
                "file_count": 0,
                "errors": [],
                "sha256": sha256_json([]),
            }
        )
        return result

    paths: list[Path] = []
    walk_errors: list[dict[str, str]] = []

    def record_walk_error(exc: OSError) -> None:
        walk_errors.append({"path": ".", "error": type(exc).__name__})

    for current, directory_names, file_names in os.walk(
        xilinx_root, topdown=True, onerror=record_walk_error, followlinks=False
    ):
        current_path = Path(current)
        retained_directories: list[str] = []
        for name in sorted(directory_names):
            if name.lower() in _VITIS_TRANSIENT_DIRECTORY_NAMES:
                continue
            directory = current_path / name
            if directory.is_symlink():
                walk_errors.append(
                    {
                        "path": directory.relative_to(xilinx_root).as_posix(),
                        "error": "SymlinkDirectoryUnsupported",
                    }
                )
                continue
            retained_directories.append(name)
        directory_names[:] = retained_directories
        for name in sorted(file_names):
            path = current_path / name
            if path.suffix.lower() in _VITIS_TRANSIENT_SUFFIXES:
                continue
            if path.is_file():
                paths.append(path)

    manifest = _relative_file_manifest(xilinx_root, paths)
    if walk_errors:
        manifest["errors"] = sorted(
            [*manifest["errors"], *walk_errors], key=lambda item: item["path"]
        )
        manifest["sha256"] = sha256_json(
            {"files": manifest["files"], "errors": manifest["errors"]}
        )
    result.update(manifest)
    result["state"] = (
        "unreadable"
        if manifest["errors"]
        else "present"
        if manifest["file_count"]
        else "empty"
    )
    return result


def _post_route_manifest(repo: Path, environ: Mapping[str, str]) -> dict[str, Any]:
    """Bind optional post-route implementation controls and setup script."""

    raw_script = str(environ.get("C2HLS_EMU_ENV_SCRIPT", "")).strip()
    configured_script = Path(raw_script).expanduser() if raw_script else (
        repo / "scripts" / "setup_emu_env.sh"
    )
    script_path = (
        configured_script
        if configured_script.is_absolute()
        else repo / configured_script
    ).resolve(strict=False)
    script: dict[str, Any] = {
        "explicit": bool(raw_script),
        "configured_absolute": configured_script.is_absolute(),
        "path": _path_identity(script_path, repo=repo),
    }
    if not script_path.exists():
        script.update({"state": "absent", "bytes": None, "sha256": None})
    elif not script_path.is_file():
        script.update({"state": "not_file", "bytes": None, "sha256": None})
    else:
        try:
            raw = script_path.read_bytes()
        except OSError as exc:
            script.update(
                {
                    "state": "unreadable",
                    "bytes": None,
                    "sha256": None,
                    "error": type(exc).__name__,
                }
            )
        else:
            script.update(
                {
                    "state": "present",
                    "bytes": len(raw),
                    "sha256": sha256_bytes(raw),
                }
            )

    def flag(name: str, default: str = "0") -> dict[str, Any]:
        raw = str(environ.get(name, default))
        return {
            "configured": raw,
            "effective": raw.strip().lower() in {"1", "true", "yes"},
        }

    return {
        "hw_emu_final": flag("C2HLS_HW_EMU_FINAL"),
        "allow_wide_abi": flag("C2HLS_ALLOW_WIDE_ABI"),
        "disable_debug_symbols": flag("C2HLS_HW_EMU_DISABLE_DEBUG_SYMBOLS"),
        "clock_mhz": _env_value(environ, "C2HLS_HW_EMU_CLOCK_MHZ"),
        "clock_ns": _env_value(environ, "C2HLS_HW_EMU_CLOCK_NS"),
        "emu_environment_script": script,
    }


def _configured_decoding(environ: Mapping[str, str]) -> dict[str, Any]:
    return {
        "temperature": _env_value(
            environ, "C2HLS_LLM_TEMPERATURE", "C2HLS_TEMPERATURE",
            default="provider_default",
        ),
        "top_p": _env_value(
            environ, "C2HLS_LLM_TOP_P", "C2HLS_TOP_P",
            default="provider_default",
        ),
        "seed": _env_value(
            environ, "C2HLS_LLM_SEED", "C2HLS_SEED",
            default="provider_default",
        ),
        "max_completion_tokens": _env_value(
            environ, "C2HLS_MAX_COMPLETION_TOKENS", default="controller_default"
        ),
        "thinking": _env_value(
            environ, "C2HLS_DEEPSEEK_THINKING", default="provider_default"
        ),
        "reasoning_effort": _env_value(
            environ,
            "C2HLS_DEEPSEEK_REASONING_EFFORT",
            default="provider_default",
        ),
        "xai_reasoning_effort": _env_value(
            environ,
            "C2HLS_XAI_REASONING_EFFORT",
            default="provider_default",
        ),
    }


def skill_snapshot_manifest(
    repo: Path,
    *,
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Return the immutable content identity of the active skill sources."""

    env = os.environ if environ is None else environ
    repo = Path(repo).resolve()
    explicit_skill_path = _env_value(env, "C2HLS_SKILL_LIBRARY_PATH")
    if explicit_skill_path:
        skill_paths = [Path(explicit_skill_path).expanduser()]
        source_mode = "explicit_frozen_snapshot"
    else:
        skill_paths = [repo / name for name in _SKILL_SOURCES]
        source_mode = "default_merged_library"
    manifest = _manifest(skill_paths, relative_to=repo)
    manifest.update(
        {
            "source_mode": source_mode,
            "explicit_path_configured": bool(explicit_skill_path),
        }
    )
    return manifest


def _model_revision(model_id: str, environ: Mapping[str, str]) -> dict[str, Any]:
    explicit = _env_value(environ, "C2HLS_MODEL_REVISION")
    if explicit:
        return {"value": explicit, "source": "C2HLS_MODEL_REVISION", "resolved": True}
    # A versioned hosted model id is still a stable identity; local aliases
    # should set C2HLS_MODEL_REVISION to a commit or weights digest.
    return {"value": model_id, "source": "model_id_fallback", "resolved": False}


def _probe_vitis_version(
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Resolve the invoked Vitis binary version rather than trusting an env label."""
    env = dict(os.environ)
    if environ is not None:
        env.update({str(key): str(value) for key, value in environ.items()})
    executable = (
        shutil.which("vitis-run", path=env.get("PATH"))
        or shutil.which("vitis_hls", path=env.get("PATH"))
    )
    settings = Path(env.get("C2HLS_VITIS_SETTINGS", "")).expanduser()
    if executable:
        command = [executable, "--version"]
        public_command = [Path(executable).name, "--version"]
    elif settings.is_file():
        command = [
            "bash",
            "-c",
            'source "$1" >/dev/null 2>&1 && command -v vitis-run && vitis-run --version',
            "c2hls-vitis-probe",
            str(settings),
        ]
        public_command = ["source", "<C2HLS_VITIS_SETTINGS>", "then", "vitis-run", "--version"]
    else:
        return {
            "command": None,
            "executable": None,
            "ran": False,
            "version": None,
            "error": "vitis executable not found on PATH",
        }
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
            env=env,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {
            "command": public_command,
            "executable": str(Path(executable).resolve()) if executable else None,
            "ran": False,
            "version": None,
            "error": f"{type(exc).__name__}: {exc}",
        }
    output = (completed.stdout or "") + "\n" + (completed.stderr or "")
    if not executable and completed.stdout:
        first_line = completed.stdout.splitlines()[0].strip()
        if first_line.startswith("/"):
            executable = first_line
    match = re.search(r"(?<!\d)(20\d{2}\.\d+)(?!\d)", output)
    return {
        "command": public_command,
        "executable": str(Path(executable).resolve()) if executable else None,
        "executable_sha256": (
            sha256_file(Path(executable).resolve())
            if executable and Path(executable).is_file()
            else None
        ),
        "ran": True,
        "returncode": completed.returncode,
        "version": match.group(1) if match else None,
        "output_sha256": hashlib.sha256(output.encode("utf-8")).hexdigest(),
        "error": "" if completed.returncode == 0 else "version command failed",
    }


def build_run_fingerprint(
    *,
    repo: Path,
    benchmark_dir: Path,
    benchmark: str,
    model_id: str,
    model_label: str,
    skill_mode: str,
    steps: Sequence[str] | None,
    profile: Mapping[str, Any] | str,
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Build the complete pre-run identity used for safe resume."""

    env = os.environ if environ is None else environ
    repo = Path(repo).resolve()
    benchmark_dir = Path(benchmark_dir).resolve()
    profile_name = profile.get("name") if isinstance(profile, Mapping) else str(profile)

    implementation_paths = [repo / name for name in _IMPLEMENTATION_SOURCES]
    dataset_pipeline = repo / "dataset_pipeline"
    if dataset_pipeline.is_dir():
        implementation_paths.extend(sorted(dataset_pipeline.rglob("*.py")))
    implementation = _manifest(implementation_paths, relative_to=repo)
    prompts = _manifest((repo / name for name in _PROMPT_SOURCES), relative_to=repo)
    benchmark_inputs = _manifest(
        _benchmark_files(benchmark_dir), relative_to=benchmark_dir
    )

    skills = skill_snapshot_manifest(repo, environ=env)
    expected_skill_sha256 = _env_value(env, "C2HLS_SKILL_SNAPSHOT_SHA256")
    skills.update(
        {
            "mode": skill_mode,
            "prompt_injection": _bool_env(env, "C2HLS_FORCE_SKILL_PROMPTS"),
            "frozen": _bool_env(env, "C2HLS_SKILL_LIBRARY_FROZEN"),
            "persistence": _bool_env(env, "C2HLS_SKILL_LIBRARY_PERSIST", "1"),
            "online_statistics": _bool_env(env, "C2HLS_SKILL_UPDATE_STATS", "1"),
            "expected_sha256": expected_skill_sha256,
            "matches_expected": (
                skills.get("sha256") == expected_skill_sha256
                if expected_skill_sha256
                else None
            ),
        }
    )

    vitis_probe = _probe_vitis_version(env)
    endpoint = _endpoint_identity(model_id, env)
    reference_cache = _reference_cache_manifest(repo, benchmark, env)
    vitis_user_home = _vitis_user_home_manifest(repo, env)
    post_route = _post_route_manifest(repo, env)
    payload: dict[str, Any] = {
        "schema_version": FINGERPRINT_SCHEMA,
        "profile": profile_name,
        "benchmark": {
            "name": benchmark,
            "inputs": benchmark_inputs,
        },
        "implementation": {
            "git_head": _git_head(repo),
            "sources": implementation,
        },
        "prompts": prompts,
        "model": {
            "label": model_label,
            "id": model_id,
            "revision": _model_revision(model_id, env),
            "endpoint": endpoint,
            "agents": {
                "translator": _env_value(env, "C2HLS_TRANSLATOR_MODEL", default=model_id),
                "synthesis": _env_value(env, "C2HLS_SYNTHESIS_MODEL", default=model_id),
                "quality_repair": _env_value(env, "C2HLS_QUALITY_REPAIR_MODEL", default=model_id),
                "feedback": _env_value(env, "C2HLS_FEEDBACK_MODEL", default=model_id),
            },
        },
        "decoding": _configured_decoding(env),
        "skills": skills,
        "toolchain": {
            "vitis_version": _env_value(env, "C2HLS_VITIS_VERSION"),
            "vitis_version_probe": vitis_probe,
            "vitis_settings_sha256": (
                _record_file(Path(env["C2HLS_VITIS_SETTINGS"]), relative_to=repo)["sha256"]
                if env.get("C2HLS_VITIS_SETTINGS")
                and Path(env["C2HLS_VITIS_SETTINGS"]).is_file()
                else None
            ),
            "flow_target": _env_value(env, "C2HLS_FLOW_TARGET", default="vitis"),
            "part": _env_value(env, "C2HLS_PART"),
            "clock_ns": _env_value(env, "C2HLS_CLOCK_NS"),
            "device_platform": _env_value(env, "C2HLS_DEVICE_PLATFORM"),
            "vitis_user_home": vitis_user_home,
        },
        "post_route": post_route,
        "search": {
            "strategy": _env_value(env, "C2HLS_STRATEGY"),
            "dynamic_routing": _bool_env(env, "C2HLS_DYNAMIC_ROUTING"),
            "steps": list(steps) if steps is not None else None,
            "qor_design_sweep": _bool_env(env, "C2HLS_QOR_DESIGN_SWEEP"),
            "qor_interactions": _bool_env(
                env, "C2HLS_QOR_SWEEP_INTERACTIONS"
            ),
        },
        "budgets": {
            "turns": _env_value(env, "C2HLS_TURNS", default="4"),
            "candidate_budget": _env_value(
                env,
                "C2HLS_SYNTHESIS_EVAL_BUDGET",
                "C2HLS_SWEEP_SYNTHESIS_EVAL_BUDGET",
                default="unbounded",
            ),
            "llm_candidate_budget": _env_value(
                env, "C2HLS_LLM_CANDIDATE_BUDGET", default="unbounded"
            ),
            "candidates_per_step": _env_value(env, "C2HLS_CANDIDATES_PER_STEP"),
            "attempts_per_candidate": _env_value(env, "C2HLS_ATTEMPTS_PER_CANDIDATE"),
            "quality_repair_turns": _env_value(
                env, "C2HLS_QUALITY_REPAIR_TURNS", default="controller_default"
            ),
            "qor_max_knobs": _env_value(
                env, "C2HLS_QOR_SWEEP_MAX_KNOBS", default="4"
            ),
            "qor_max_candidates": _env_value(
                env, "C2HLS_QOR_SWEEP_MAX_CANDIDATES", default="8"
            ),
            "qor_max_interactions": _env_value(
                env, "C2HLS_QOR_SWEEP_MAX_INTERACTIONS", default="2"
            ),
            "synth_timeout_seconds": _env_value(env, "C2HLS_SYNTH_TIMEOUT"),
            "csim_timeout_seconds": _env_value(env, "C2HLS_CSIM_TIMEOUT"),
            "cosim_timeout_seconds": _env_value(env, "C2HLS_COSIM_TIMEOUT"),
        },
        "reference_isolation": {
            key: env.get(key) for key in sorted(REFERENCE_BLIND_OVERRIDES)
        },
        "reference_cache": reference_cache,
        "effective_configuration": {
            key: env.get(key) for key in _CONTROL_ENV_NAMES
        },
    }
    digest = sha256_json(payload)
    return {"schema_version": FINGERPRINT_SCHEMA, "sha256": digest, "payload": payload}


def fingerprint_matches(recorded: Any, expected: Any) -> bool:
    """Return true only for complete, same-schema, exact fingerprints."""

    if not isinstance(recorded, Mapping) or not isinstance(expected, Mapping):
        return False
    if recorded.get("schema_version") != FINGERPRINT_SCHEMA:
        return False
    if expected.get("schema_version") != FINGERPRINT_SCHEMA:
        return False
    rec_payload = recorded.get("payload")
    exp_payload = expected.get("payload")
    if not isinstance(rec_payload, Mapping) or not isinstance(exp_payload, Mapping):
        return False
    rec_digest = recorded.get("sha256")
    exp_digest = expected.get("sha256")
    if rec_digest != sha256_json(rec_payload) or exp_digest != sha256_json(exp_payload):
        return False
    return rec_digest == exp_digest and canonical_json(rec_payload) == canonical_json(exp_payload)


def fingerprint_completeness(fingerprint: Mapping[str, Any]) -> dict[str, Any]:
    """Report provenance gaps that disqualify a run from the paper matrix."""

    payload = fingerprint.get("payload")
    if not isinstance(payload, Mapping):
        return {"complete": False, "issues": ["fingerprint_payload_missing"]}
    issues: list[str] = []
    revision = (payload.get("model") or {}).get("revision") or {}
    if not revision.get("resolved"):
        issues.append("model_revision_unresolved")
    model = payload.get("model") or {}
    primary_model_id = model.get("id")
    endpoint = model.get("endpoint") or {}
    if not isinstance(endpoint, Mapping) or not endpoint.get("valid"):
        issues.append("model_endpoint_invalid")
    if isinstance(endpoint, Mapping) and endpoint.get("unsafe_components"):
        issues.append("model_endpoint_unsafe_components")
    if not isinstance(endpoint, Mapping) or not endpoint.get("endpoint_sha256"):
        issues.append("model_endpoint_identity_missing")
    agents = model.get("agents") or {}
    if not isinstance(agents, Mapping):
        issues.append("agent_model_identity_missing")
    else:
        # Paper runs intentionally use one immutable model identity for every
        # role.  This prevents inherited C2HLS_*_MODEL variables from silently
        # creating an unrevisioned mixed-model system.
        for agent_name in ("translator", "synthesis", "quality_repair", "feedback"):
            if agents.get(agent_name) != primary_model_id:
                issues.append(f"agent_model_override_forbidden:{agent_name}")
    decoding = payload.get("decoding") or {}
    for key in ("temperature", "top_p", "seed", "max_completion_tokens"):
        if decoding.get(key) in (None, "", "provider_default", "controller_default"):
            issues.append(f"decoding_{key}_not_explicit")
    toolchain = payload.get("toolchain") or {}
    for key in ("vitis_version", "part", "clock_ns"):
        if toolchain.get(key) in (None, ""):
            issues.append(f"toolchain_{key}_missing")
    probe = toolchain.get("vitis_version_probe") or {}
    actual_vitis = probe.get("version") if isinstance(probe, Mapping) else None
    if not isinstance(probe, Mapping) or not probe.get("ran"):
        issues.append("actual_vitis_probe_not_run")
    if not isinstance(probe, Mapping) or probe.get("returncode") != 0:
        issues.append("actual_vitis_probe_failed")
    if isinstance(probe, Mapping) and probe.get("error"):
        issues.append("actual_vitis_probe_error")
    if not isinstance(probe, Mapping) or not probe.get("executable"):
        issues.append("actual_vitis_executable_unresolved")
    if not isinstance(probe, Mapping) or not probe.get("executable_sha256"):
        issues.append("actual_vitis_executable_sha256_missing")
    if not actual_vitis:
        issues.append("actual_vitis_version_unresolved")
    elif str(actual_vitis) != str(toolchain.get("vitis_version")):
        issues.append("actual_vitis_version_mismatch")
    if not toolchain.get("vitis_settings_sha256"):
        issues.append("toolchain_vitis_settings_sha256_missing")
    vitis_user_home = toolchain.get("vitis_user_home") or {}
    if not isinstance(vitis_user_home, Mapping):
        issues.append("vitis_user_home_identity_missing")
    else:
        if not vitis_user_home.get("configured_absolute"):
            issues.append("vitis_user_home_path_not_absolute")
        if not vitis_user_home.get("path"):
            issues.append("vitis_user_home_path_identity_missing")
        if not vitis_user_home.get("sha256"):
            issues.append("vitis_user_home_state_digest_missing")
        if vitis_user_home.get("state") in {
            "home_not_directory",
            "invalid_relative_path",
            "unreadable",
            "xilinx_state_not_directory",
        }:
            issues.append("vitis_user_home_state_invalid")
    reference_cache = payload.get("reference_cache") or {}
    if not isinstance(reference_cache, Mapping):
        issues.append("reference_cache_identity_missing")
    else:
        if not reference_cache.get("sha256"):
            issues.append("reference_cache_state_digest_missing")
        if reference_cache.get("enabled") and not reference_cache.get("path"):
            issues.append("reference_cache_path_identity_missing")
        if reference_cache.get("state") in {"not_directory", "unreadable"}:
            issues.append("reference_cache_state_invalid")
    post_route = payload.get("post_route")
    if not isinstance(post_route, Mapping):
        issues.append("post_route_identity_missing")
    else:
        required_post_route_keys = {
            "allow_wide_abi",
            "clock_mhz",
            "clock_ns",
            "disable_debug_symbols",
            "emu_environment_script",
            "hw_emu_final",
        }
        if not required_post_route_keys.issubset(post_route):
            issues.append("post_route_controls_missing")
        hw_emu = post_route.get("hw_emu_final") or {}
        script = post_route.get("emu_environment_script") or {}
        if not isinstance(hw_emu, Mapping) or not isinstance(
            hw_emu.get("effective"), bool
        ):
            issues.append("post_route_hw_emu_setting_missing")
        if not isinstance(script, Mapping) or script.get("state") != "present":
            issues.append("post_route_emu_script_missing")
        if not isinstance(script, Mapping) or not script.get("configured_absolute"):
            issues.append("post_route_emu_script_path_not_absolute")
        if not isinstance(script, Mapping) or not script.get("path"):
            issues.append("post_route_emu_script_path_identity_missing")
        if not isinstance(script, Mapping) or not script.get("sha256"):
            issues.append("post_route_emu_script_digest_missing")
        if payload.get("profile") == PAPER_PROFILE:
            allow_wide = post_route.get("allow_wide_abi") or {}
            disable_debug = post_route.get("disable_debug_symbols") or {}
            if not isinstance(allow_wide, Mapping) or allow_wide.get("effective"):
                issues.append("paper_post_route_wide_abi_not_disabled")
            if not isinstance(disable_debug, Mapping) or not disable_debug.get(
                "effective"
            ):
                issues.append("paper_post_route_debug_symbols_not_disabled")
            if post_route.get("clock_ns") != "3.33":
                issues.append("paper_post_route_clock_ns_mismatch")
            if post_route.get("clock_mhz") not in (None, ""):
                issues.append("paper_post_route_clock_mhz_forbidden")
            script_path = script.get("path") if isinstance(script, Mapping) else {}
            if not isinstance(script_path, Mapping) or (
                script_path.get("scope") != "repository"
                or script_path.get("path") != "scripts/setup_emu_env.sh"
            ):
                issues.append("paper_post_route_emu_script_mismatch")
    budget = payload.get("budgets") or {}
    if budget.get("candidate_budget") in (None, "", "unbounded"):
        issues.append("synthesis_evaluation_budget_unbounded")
    if budget.get("llm_candidate_budget") in (None, "", "unbounded"):
        issues.append("llm_candidate_budget_unbounded")
    skills = payload.get("skills") or {}
    if skills.get("mode") == "skill_on" or skills.get("prompt_injection"):
        if not skills.get("explicit_path_configured"):
            issues.append("frozen_skill_path_missing")
        elif skills.get("source_mode") != "explicit_frozen_snapshot":
            issues.append("frozen_skill_source_not_explicit")
        if skills.get("file_count") != 1:
            issues.append("frozen_skill_snapshot_file_missing")
        if not skills.get("frozen"):
            issues.append("skill_library_not_frozen")
        if skills.get("persistence"):
            issues.append("skill_persistence_enabled")
        if skills.get("online_statistics"):
            issues.append("online_skill_statistics_enabled")
        if not skills.get("expected_sha256"):
            issues.append("skill_snapshot_expected_hash_missing")
        elif not skills.get("matches_expected"):
            issues.append("skill_snapshot_hash_mismatch")
    return {"complete": not issues, "issues": issues}


def _numeric_equal(actual: Any, expected: Any) -> bool:
    """Compare provider-reported numeric decoding values without type drift."""

    try:
        return float(actual) == float(expected)
    except (TypeError, ValueError):
        return actual == expected


def effective_llm_call_issues(
    result: Mapping[str, Any], fingerprint: Mapping[str, Any]
) -> list[str]:
    """Validate actual per-call identity and decoding against the fingerprint.

    A configured value is not evidence that a provider received it.  Each
    request event therefore has to bind the model revision, prompt, token cap,
    and effective decoding values used for that call.  Anthropic's documented
    lack of seed control is represented explicitly rather than pretending the
    configured trial seed was sent to the provider.
    """

    payload = fingerprint.get("payload") or {}
    model = payload.get("model") or {}
    expected_model = model.get("id")
    expected_revision = (model.get("revision") or {}).get("value")
    configured = payload.get("decoding") or {}
    baseline_contract = payload.get("paper_baseline")
    usage = result.get("llm_usage") or (result.get("run") or {}).get("llm_usage") or {}
    events = usage.get("events") if isinstance(usage, Mapping) else None
    calls = usage.get("calls") if isinstance(usage, Mapping) else None
    issues: list[str] = []
    if not isinstance(events, list):
        return ["effective_llm_call_records_missing"]
    if calls is None or int(calls) != len(events):
        issues.append("effective_llm_call_count_mismatch")

    baseline_schedule: dict[int, Mapping[str, Any]] | None = None
    baseline_seed_supported: bool | None = None
    if isinstance(baseline_contract, Mapping):
        seed_policy = baseline_contract.get("seed_policy")
        if seed_policy == "base_plus_candidate_index":
            baseline_seed_supported = True
        elif seed_policy == "unsupported_by_provider":
            baseline_seed_supported = False
        else:
            issues.append("baseline_seed_policy_invalid")

        raw_base_seed = baseline_contract.get("base_seed")
        if isinstance(raw_base_seed, int) and not isinstance(raw_base_seed, bool):
            base_seed = raw_base_seed
        else:
            base_seed = None
            issues.append("baseline_base_seed_invalid")
        if base_seed is not None and not _numeric_equal(
            base_seed, configured.get("seed")
        ):
            issues.append("baseline_base_seed_mismatch")

        raw_schedule = baseline_contract.get("candidate_seed_schedule")
        maximum = baseline_contract.get("max_llm_candidates")
        if (
            not isinstance(raw_schedule, list)
            or not isinstance(maximum, int)
            or isinstance(maximum, bool)
            or len(raw_schedule) != maximum
        ):
            issues.append("baseline_seed_schedule_invalid")
        elif base_seed is not None and baseline_seed_supported is not None:
            baseline_schedule = {}
            for candidate_index, raw_entry in enumerate(raw_schedule):
                expected_requested = base_seed + candidate_index
                expected_effective = (
                    expected_requested if baseline_seed_supported else None
                )
                if (
                    not isinstance(raw_entry, Mapping)
                    or raw_entry.get("candidate_index") != candidate_index
                    or raw_entry.get("requested_seed") != expected_requested
                    or raw_entry.get("effective_seed") != expected_effective
                    or raw_entry.get("seed_supported") is not baseline_seed_supported
                ):
                    issues.append(
                        f"baseline_seed_schedule_{candidate_index}_mismatch"
                    )
                    continue
                baseline_schedule[candidate_index] = raw_entry

    for index, event in enumerate(events):
        prefix = f"llm_call_{index}"
        if not isinstance(event, Mapping):
            issues.append(f"{prefix}:record_invalid")
            continue
        if event.get("model") != expected_model:
            issues.append(f"{prefix}:model_mismatch")
        if event.get("model_revision") != expected_revision:
            issues.append(f"{prefix}:model_revision_mismatch")
        if event.get("max_tokens") is None or not _numeric_equal(
            event.get("max_tokens"), configured.get("max_completion_tokens")
        ):
            issues.append(f"{prefix}:max_completion_tokens_mismatch")
        prompt_sha = str(event.get("prompt_sha256") or "")
        if not re.fullmatch(r"[0-9a-f]{64}", prompt_sha):
            issues.append(f"{prefix}:prompt_hash_missing")
        decoding = event.get("decoding") or {}
        if not isinstance(decoding, Mapping):
            issues.append(f"{prefix}:decoding_missing")
            continue
        omitted_decoding = decoding.get("mutually_exclusive_omission")
        for key in ("temperature", "top_p"):
            if key == omitted_decoding:
                if decoding.get(key) is not None or not _numeric_equal(
                    decoding.get(f"requested_{key}"),
                    configured.get(key),
                ):
                    issues.append(f"{prefix}:{key}_provider_omission_invalid")
                continue
            if not _numeric_equal(decoding.get(key), configured.get(key)):
                issues.append(f"{prefix}:{key}_mismatch")
        seed_supported = decoding.get("seed_supported")
        if not isinstance(seed_supported, bool):
            issues.append(f"{prefix}:seed_support_unreported")

        expected_seed = configured.get("seed")
        provider = str(event.get("provider") or "").strip().lower()
        if not provider:
            issues.append(f"{prefix}:provider_missing")
        expected_seed_supported: bool | None = provider not in {
            "anthropic",
            "deepseek",
        }
        if isinstance(baseline_contract, Mapping):
            candidate_index = event.get("candidate_index")
            if (
                isinstance(candidate_index, bool)
                or not isinstance(candidate_index, int)
                or baseline_schedule is None
                or candidate_index not in baseline_schedule
            ):
                issues.append(f"{prefix}:baseline_candidate_index_invalid")
            else:
                schedule_entry = baseline_schedule[candidate_index]
                expected_seed = schedule_entry.get("effective_seed")
                expected_seed_supported = bool(
                    schedule_entry.get("seed_supported")
                )
                if expected_seed_supported != (
                    provider not in {"anthropic", "deepseek"}
                ):
                    issues.append(f"{prefix}:provider_seed_policy_mismatch")
                if (
                    event.get("requested_seed")
                    != schedule_entry.get("requested_seed")
                    or event.get("effective_seed") != expected_seed
                    or event.get("seed_supported") is not expected_seed_supported
                ):
                    issues.append(f"{prefix}:baseline_seed_attribution_mismatch")

        if expected_seed_supported is False or (
            expected_seed_supported is None and seed_supported is False
        ):
            if (
                seed_supported is not False
                or decoding.get("seed") is not None
            ):
                issues.append(f"{prefix}:unsupported_seed_status_invalid")
        else:
            if seed_supported is not True:
                issues.append(f"{prefix}:seed_support_status_invalid")
            if not _numeric_equal(decoding.get("seed"), expected_seed):
                issues.append(f"{prefix}:seed_mismatch")
        if event.get("provider") == "deepseek":
            for key in ("thinking", "reasoning_effort"):
                if decoding.get(key) != configured.get(key):
                    issues.append(f"{prefix}:{key}_mismatch")
    return issues


def prompt_hashes(history_path: Path) -> dict[str, Any]:
    """Hash the exact prompts/responses without duplicating their contents."""

    try:
        data = json.loads(Path(history_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"available": False, "prompts": [], "responses": []}
    messages = data.get("messages") if isinstance(data, Mapping) else None
    if not isinstance(messages, list):
        return {"available": False, "prompts": [], "responses": []}
    prompts: list[dict[str, Any]] = []
    responses: list[dict[str, Any]] = []
    for index, message in enumerate(messages):
        if not isinstance(message, Mapping):
            continue
        role = str(message.get("role") or "unknown")
        content = str(message.get("content") or "")
        item = {
            "index": index,
            "role": role,
            "characters": len(content),
            "sha256": sha256_bytes(content.encode("utf-8")),
        }
        (responses if role == "assistant" else prompts).append(item)
    return {"available": True, "prompts": prompts, "responses": responses}


def _selected_test_summary(result: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    direct = result.get(key)
    if isinstance(direct, Mapping):
        return direct
    steps = result.get("steps")
    if isinstance(steps, list):
        for step in reversed(steps):
            if isinstance(step, Mapping) and isinstance(step.get(key), Mapping):
                return step[key]
    baseline = result.get(f"baseline_{key}")
    return baseline if isinstance(baseline, Mapping) else {}


def _execution_status(summary: Mapping[str, Any]) -> str:
    if not summary:
        return "not_run"
    ran = summary.get("ran")
    passed = summary.get("passed")
    status = str(summary.get("status") or "").strip().lower()
    error = str(summary.get("error") or summary.get("skip_reason") or "").lower()
    if passed is True or status in {"pass", "passed", "success"}:
        return "passed"
    if "timed out" in error or "timeout" in status:
        return "timeout"
    if status in {"error", "tool_error", "tool_failure"}:
        return "tool_failure"
    if ran is False:
        return "not_run"
    if passed is False and status in {"fail", "failed", "mismatch"}:
        return "failed"
    if error:
        return "tool_failure"
    return "unknown"


def _canonical_status(value: Any) -> str:
    status = str(value or "").strip().lower()
    return {
        "pass": "passed",
        "success": "passed",
        "fail": "failed",
        "mismatch": "failed",
        "error": "tool_failure",
        "tool_error": "tool_failure",
    }.get(status, status)


def _synthesis_status_from_events(result: Mapping[str, Any]) -> tuple[str, bool, bool]:
    summary = result.get("synthesis_evaluations")
    if not isinstance(summary, Mapping):
        run = result.get("run")
        summary = run.get("synthesis_evaluations") if isinstance(run, Mapping) else None
    events = summary.get("events") if isinstance(summary, Mapping) else None
    if not isinstance(events, list):
        events = []

    ran_events = []
    for raw in events:
        if not isinstance(raw, Mapping):
            continue
        explicitly_ran = raw.get("synthesis_ran")
        ran = (
            bool(explicitly_ran)
            if explicitly_ran is not None
            else raw.get("index") is not None or "success" in raw
        )
        if ran:
            ran_events.append(raw)

    if any(event.get("success") is True for event in ran_events):
        return "passed", False, False
    if not ran_events:
        # Compatibility fallback for older successful records that predate
        # typed synthesis-event telemetry.
        report = result.get("final_report") or result.get("synth_report")
        return ("passed", False, False) if isinstance(report, Mapping) and report else ("not_run", False, False)

    timeout = any(
        event.get("timed_out") is True
        or "timeout" in str(event.get("status") or "").lower()
        or "timed out" in str(event.get("error") or "").lower()
        for event in ran_events
    )
    tool_failure = any(
        event.get("tool_failure") is True
        or str(event.get("status") or "").lower()
        in {"tool_error", "tool_failure", "error"}
        for event in ran_events
    )
    if timeout:
        return "timeout", True, tool_failure
    if tool_failure:
        return "tool_failure", False, True
    return "failed", False, False


def derive_status_taxonomy(result: Mapping[str, Any]) -> dict[str, Any]:
    """Separate correctness, executed RTL simulation, prediction and tools."""

    csim = _selected_test_summary(result, "csim")
    cosim = _selected_test_summary(result, "cosim")
    policy = cosim.get("cosim_policy") if isinstance(cosim, Mapping) else {}
    policy = policy if isinstance(policy, Mapping) else {}
    predicted_skip = bool(
        policy.get("classification") == "predicted_timeout"
        or cosim.get("skip_reason") == "predicted_longer_than_gold"
    )
    correctness = result.get("correctness_status")
    if isinstance(correctness, Mapping):
        correctness = correctness.get("correctness_status") or correctness.get("status")
    correctness = _canonical_status(correctness)
    if not correctness:
        correctness = _execution_status(csim)

    synth_status, synth_timeout, synth_tool_failure = _synthesis_status_from_events(result)
    executed_cosim = "not_run" if predicted_skip else _execution_status(cosim)
    llm_usage = result.get("llm_usage")
    if not isinstance(llm_usage, Mapping):
        run = result.get("run")
        llm_usage = run.get("llm_usage") if isinstance(run, Mapping) else None
    llm_events = llm_usage.get("events") if isinstance(llm_usage, Mapping) else []
    provider_failure = any(
        isinstance(event, Mapping) and bool(event.get("error"))
        for event in (llm_events if isinstance(llm_events, list) else [])
    )
    tool_failure = bool(
        synth_tool_failure
        or executed_cosim == "tool_failure"
        or correctness == "tool_failure"
        or provider_failure
    )
    return {
        "schema_version": "c2hls.evaluation-status.v1",
        "correctness_status": correctness,
        "synthesis_status": synth_status,
        "cosim_execution_status": executed_cosim,
        "cosim_ran": bool(cosim.get("ran")) if cosim else False,
        "cosim_predicted_skip": predicted_skip,
        "cosim_prediction": policy.get("classification") if predicted_skip else None,
        "timeout": synth_timeout or executed_cosim == "timeout" or correctness == "timeout",
        "tool_failure": tool_failure,
        "provider_failure": provider_failure,
    }


def attach_run_provenance(
    result: dict[str, Any],
    *,
    fingerprint: Mapping[str, Any],
    profile: Mapping[str, Any],
    elapsed_seconds: float,
    history_path: Path,
    reference_audit: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Attach paper-required identity/provenance fields to a run result."""

    run = result.setdefault("run", {})
    if not isinstance(run, dict):
        run = {}
        result["run"] = run
    run["evaluation_profile"] = profile.get("name")
    run["reference_blind"] = bool(profile.get("reference_blind"))
    run["run_fingerprint"] = dict(fingerprint)
    configured_decoding = dict(
        fingerprint.get("payload", {}).get("decoding", {})
    )
    reported_decoding = run.get("decoding")
    if isinstance(reported_decoding, Mapping) and "effective" in reported_decoding:
        effective_decoding = reported_decoding.get("effective")
    elif isinstance(reported_decoding, Mapping) and any(
        key in reported_decoding for key in ("temperature", "top_p", "seed")
    ):
        effective_decoding = dict(reported_decoding)
    else:
        effective_decoding = None
    run["decoding"] = {
        "configured": configured_decoding,
        "effective": effective_decoding,
        "effective_reported": isinstance(effective_decoding, Mapping),
    }
    run["model_revision"] = dict(
        fingerprint.get("payload", {}).get("model", {}).get("revision", {})
    )
    run["toolchain"] = dict(
        fingerprint.get("payload", {}).get("toolchain", {})
    )
    run["prompt_hashes"] = prompt_hashes(history_path)
    run["elapsed_seconds"] = round(float(elapsed_seconds), 6)
    reproducibility = fingerprint_completeness(fingerprint)
    synthesis_evaluations = (
        result.get("synthesis_evaluations")
        if result.get("synthesis_evaluations") is not None
        else run.get("synthesis_evaluations", run.get("synthesis_count"))
    )
    if not isinstance(effective_decoding, Mapping):
        reproducibility["issues"].append("effective_decoding_unreported")
    reproducibility["issues"].extend(effective_llm_call_issues(result, fingerprint))
    if synthesis_evaluations is None:
        reproducibility["issues"].append("synthesis_evaluation_count_missing")
    reproducibility["issues"] = sorted(set(reproducibility["issues"]))
    reproducibility["complete"] = not reproducibility["issues"]
    run["reproducibility"] = reproducibility
    run["synthesis_evaluations"] = synthesis_evaluations
    run["llm_usage"] = result.get("llm_usage") or run.get("llm_usage")
    result["run_fingerprint"] = dict(fingerprint)
    evaluation_status = derive_status_taxonomy(result)
    result["evaluation_status"] = evaluation_status
    result["correctness_status"] = evaluation_status["correctness_status"]
    result["executed_cosim_status"] = evaluation_status[
        "cosim_execution_status"
    ]
    result["predicted_cosim_skip"] = evaluation_status[
        "cosim_predicted_skip"
    ]
    result["timeout_status"] = (
        "timeout" if evaluation_status["timeout"] else "none"
    )
    result["tool_failure_status"] = (
        "tool_failure" if evaluation_status["tool_failure"] else "none"
    )
    if reference_audit is not None:
        result["reference_isolation_audit"] = dict(reference_audit)
    return result
