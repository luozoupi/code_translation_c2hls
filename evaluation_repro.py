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
    "C2HLS_FEASIBILITY_SELECTION": "1",
    "C2HLS_CORRECTNESS_BEFORE_SYNTH": "1",
    "C2HLS_DISABLE_CORRECTNESS_REPAIR": "0",
    "C2HLS_SYNTH_REVERT_THRESHOLD": "0",
    "C2HLS_PHASE5_LLM_RETRY": "0",
    "C2HLS_CPU_GOLDEN_TIMEOUT": "180",
    "C2HLS_SYNTHESIS_EVAL_BUDGET": "5",
    "C2HLS_LLM_CANDIDATE_BUDGET": "5",
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
    "reference_isolation.py",
    "robustness.py",
    "rubric.py",
    "run_agentic_sweep.py",
    "skill_library.py",
    "trajectory_alignment.py",
    "configs/hlsfactory_output_shapes.json",
    "configs/hlsfactory_development_suite.json",
)
_PROMPT_SOURCES = ("prompt_c2hls.py",)
_SKILL_SOURCES = (
    "skills/skills.json",
    "hls_full_optimization_skills_schema_1_1_package/skills.json",
)
_CONTROL_ENV_NAMES = (
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
    "C2HLS_FEEDBACK_MODEL",
    "C2HLS_FEASIBILITY_SELECTION",
    "C2HLS_FLOW_TARGET",
    "C2HLS_FORCE_SKILL_PROMPTS",
    "C2HLS_GT_AWARE_REVERT",
    "C2HLS_GT_COMPARISON_IN_CONTROL",
    "C2HLS_HW_EMU_FINAL",
    "C2HLS_HW_EMU_TIMEOUT",
    "C2HLS_LLM_SEED",
    "C2HLS_LLM_CANDIDATE_BUDGET",
    "C2HLS_LLM_TEMPERATURE",
    "C2HLS_LLM_TIMEOUT",
    "C2HLS_LLM_TOP_P",
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
    "C2HLS_REFERENCE_BLIND",
    "C2HLS_REFERENCE_CACHE_REQUIRE_COSIM",
    "C2HLS_REFERENCE_CODE_IN_PROMPTS",
    "C2HLS_REFERENCE_COSIM",
    "C2HLS_REFERENCE_COSIM_BASELINE",
    "C2HLS_REFERENCE_COSIM_SELECTED_ONLY",
    "C2HLS_REFERENCE_METRICS_IN_PROMPTS",
    "C2HLS_REFERENCE_VALIDATE_MODE",
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
    for key, value in REFERENCE_BLIND_OVERRIDES.items():
        previous = env.get(key)
        if previous != value:
            changed[key] = {"previous": previous, "effective": value}
        env[key] = value
    env["C2HLS_SWEEP_PROFILE"] = PAPER_PROFILE
    return {
        "name": PAPER_PROFILE,
        "reference_blind": True,
        "forced_overrides": changed,
        "invariants": dict(REFERENCE_BLIND_OVERRIDES),
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
        },
        "search": {
            "strategy": _env_value(env, "C2HLS_STRATEGY"),
            "dynamic_routing": _bool_env(env, "C2HLS_DYNAMIC_ROUTING"),
            "steps": list(steps) if steps is not None else None,
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
            "synth_timeout_seconds": _env_value(env, "C2HLS_SYNTH_TIMEOUT"),
            "csim_timeout_seconds": _env_value(env, "C2HLS_CSIM_TIMEOUT"),
            "cosim_timeout_seconds": _env_value(env, "C2HLS_COSIM_TIMEOUT"),
        },
        "reference_isolation": {
            key: env.get(key) for key in sorted(REFERENCE_BLIND_OVERRIDES)
        },
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
    usage = result.get("llm_usage") or (result.get("run") or {}).get("llm_usage") or {}
    events = usage.get("events") if isinstance(usage, Mapping) else None
    calls = usage.get("calls") if isinstance(usage, Mapping) else None
    issues: list[str] = []
    if not isinstance(events, list):
        return ["effective_llm_call_records_missing"]
    if calls is None or int(calls) != len(events):
        issues.append("effective_llm_call_count_mismatch")
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
        for key in ("temperature", "top_p"):
            if not _numeric_equal(decoding.get(key), configured.get(key)):
                issues.append(f"{prefix}:{key}_mismatch")
        if decoding.get("seed_supported") is False:
            if event.get("provider") != "anthropic" or decoding.get("seed") is not None:
                issues.append(f"{prefix}:unsupported_seed_status_invalid")
        elif not _numeric_equal(decoding.get("seed"), configured.get("seed")):
            issues.append(f"{prefix}:seed_mismatch")
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
