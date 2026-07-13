#!/usr/bin/env python3
"""Validate and deterministically expand the HPCA 2027 experiment matrix.

This program is intentionally incapable of launching an experiment.  Its
only outputs are validation diagnostics, a summary, or row-level JSON/JSONL
describing the proposed invocation.  It validates runner mappings but never
calls an LLM endpoint, C2HLS, or Vitis.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


REPO = Path(__file__).resolve().parents[1]
DEFAULT_MATRIX = REPO / "configs" / "hpca2027_experiment_matrix.json"

SCHEMA_VERSION = "c2hls.hpca2027-experiment-matrix.v1"
PRIMARY_BENCHMARKS = (
    "StreamCluster",
    "hotspot",
    "kmeans",
    "knn",
    "lavaMD",
    "lud",
    "nw",
    "pathfinder",
    "srad",
)
MODEL_KEYS = ("qwen_27b", "claude_sonnet_4_6")
METHOD_KEYS = (
    "one_shot_best_of_five",
    "pragma_only",
    "flash_c2hls",
    "dynamic_no_skills",
    "dynamic_frozen_skills",
)
REPLICATION_METHODS = (
    "one_shot_best_of_five",
    "dynamic_no_skills",
    "dynamic_frozen_skills",
)
REPRESENTATIVE_STRATA = {
    "pathfinder": "memory_bound",
    "lavaMD": "compute_bound",
    "StreamCluster": "irregular",
}
STATISTICS_CONTRACT = {
    "performance_profile_tau_max": 10.0,
    "performance_profile_ticks": [1.0, 1.25, 2.0, 4.0, 10.0],
    "paired_bootstrap": {
        "statistic": "mean_paired_failure_aware_profile_auc_difference",
        "confidence": 0.95,
        "replicates": 10000,
        "seed": 2027,
    },
    "frozen_skill_gate": {
        "ci_rule": "lower_bound_strictly_above_zero",
        "correct_solve_rule": "not_lower_than_dynamic_no_skill",
    },
}

_TOLERANCE_DECLARATION_RE = re.compile(
    r"\b(?:float|double)\s+tol(?:erance)?\s*=", re.IGNORECASE
)
_FINITE_TOLERANCE_HELPER_RE = re.compile(
    r"\bstatic\s+bool\s+tolerance_mismatch\s*\(\s*"
    r"float\s+reference\s*,\s*float\s+actual\s*,\s*float\s+tolerance\s*\)"
    r"\s*\{(?P<body>.*?)\}",
    re.DOTALL,
)


def _strip_cpp_comments(text: str) -> str:
    return re.sub(r"//[^\n]*|/\*.*?\*/", "", text, flags=re.DOTALL)


def _finite_tolerance_policy_error(text: str) -> str:
    """Return why a float-tolerance testbench can accept NaN/Inf, if any.

    This is deliberately a narrow static contract for the checked-in Rodinia
    testbenches, not a general C++ parser.  A tolerance-using testbench must
    route every comparison through the shared-shaped helper, which first
    rejects non-finite reference and DUT values.
    """

    code = _strip_cpp_comments(text)
    if not _TOLERANCE_DECLARATION_RE.search(code):
        return ""
    helper = _FINITE_TOLERANCE_HELPER_RE.search(code)
    if helper is None:
        return "missing finite-safe tolerance_mismatch helper"
    body = helper.group("body")
    required_checks = (
        r"!\s*std::isfinite\s*\(\s*reference\s*\)",
        r"!\s*std::isfinite\s*\(\s*actual\s*\)",
    )
    if any(re.search(pattern, body) is None for pattern in required_checks):
        return "tolerance_mismatch must reject non-finite reference and DUT values"
    if not re.search(
        r"fabsf\s*\(\s*reference\s*-\s*actual\s*\)\s*>\s*"
        r"tolerance\s*\*\s*\(\s*fabsf\s*\(\s*reference\s*\)\s*\+\s*1\.0f\s*\)",
        body,
    ):
        return "tolerance_mismatch must preserve the existing relative-tolerance rule"

    outside_helper = code[: helper.start()] + code[helper.end() :]
    if re.search(
        r"(?:>\s*tol(?:erance)?\b|\btol(?:erance)?\b\s*<)",
        outside_helper,
        flags=re.IGNORECASE,
    ):
        return "raw tolerance comparison bypasses finite-value checks"
    if not re.search(r"\btolerance_mismatch\s*\(", outside_helper):
        return "tolerance_mismatch is defined but no output comparison uses it"
    return ""


@dataclass(frozen=True)
class Issue:
    severity: str
    code: str
    path: str
    message: str


def _issue(
    issues: list[Issue], severity: str, code: str, path: str, message: str
) -> None:
    issues.append(Issue(severity=severity, code=code, path=path, message=message))


def load_matrix(path: Path | str = DEFAULT_MATRIX) -> dict[str, Any]:
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError("experiment matrix root must be a JSON object")
    return value


def _ids(entries: Iterable[Mapping[str, Any]]) -> list[str]:
    return [str(entry.get("id", "")) for entry in entries]


def _selection_keys(selection: Any, available: Mapping[str, Any]) -> list[str]:
    if selection == "all":
        return list(available)
    if isinstance(selection, list):
        return [str(item) for item in selection]
    raise ValueError(f"selection must be `all` or a list, got {selection!r}")


def _benchmark_entries(
    matrix: Mapping[str, Any], set_name: str
) -> list[dict[str, Any]]:
    sets = matrix["benchmark_sets"]
    selected = sets[set_name]
    primary_by_id = {
        entry["id"]: dict(entry) for entry in sets["rodinia_primary"]
    }
    result: list[dict[str, Any]] = []
    for entry in selected:
        merged = dict(primary_by_id[entry["id"]])
        merged.update(entry)
        result.append(merged)
    return result


def _template_env_value(name: str) -> str:
    return "${" + name + "}"


def _execution_for_row(
    matrix: Mapping[str, Any],
    campaign: Mapping[str, Any],
    benchmark: Mapping[str, Any],
    model_key: str,
    method_key: str,
    seed: int,
    row_id: str,
    *,
    resolve_env: bool,
) -> dict[str, Any]:
    model = matrix["models"][model_key]
    method = matrix["methods"][method_key]
    required_env = sorted(
        set(matrix.get("required_env", []))
        | set(model.get("required_env", []))
        | set(method.get("required_env", []))
    )
    status = method["mapping_status"]
    if status != "supported":
        return {
            "mapping_status": status,
            "command": None,
            "environment": None,
            "required_env": required_env,
            "unresolved_required_env": required_env,
            "blocker": method.get("unsupported_reason", "unsupported method"),
            "required_implementation": method.get("required_implementation", ""),
        }

    profile = matrix["profiles"][campaign["profile"]]
    environment = {
        **matrix["common_runner_env"],
        **profile["env"],
        "C2HLS_SWEEP_BENCHES": benchmark["id"],
        "C2HLS_SWEEP_MODELS": model["model_id"],
        "C2HLS_SWEEP_STRATEGY": method["strategy"],
        "C2HLS_STRATEGY": method["strategy"],
        "C2HLS_LLM_SEED": str(seed),
        "C2HLS_SWEEP_STAMP": row_id,
        "C2HLS_MODEL_REVISION": _template_env_value(model["revision_env"]),
        "C2HLS_MODEL_LABEL": model_key,
        # Paper runs use one revisioned model for every agent.  Explicitly
        # overwrite inherited role-specific variables from interactive runs.
        "C2HLS_TRANSLATOR_MODEL": model["model_id"],
        "C2HLS_SYNTHESIS_MODEL": model["model_id"],
        "C2HLS_QUALITY_REPAIR_MODEL": model["model_id"],
        "C2HLS_FEEDBACK_MODEL": model["model_id"],
    }
    if method["runner"] == "run_paper_baseline.py":
        environment["C2HLS_BASELINE_METHOD"] = method_key
    if method["skills"] == "frozen":
        environment.update(
            {
                "C2HLS_SWEEP_SKILL_MODES": "on",
                "C2HLS_SKILL_MODE": "skill_on",
                "C2HLS_FORCE_SKILL_PROMPTS": "1",
                "C2HLS_SKILL_LIBRARY_PATH": _template_env_value(
                    "C2HLS_SKILL_LIBRARY_PATH"
                ),
                "C2HLS_SKILL_SNAPSHOT_SHA256": _template_env_value(
                    "C2HLS_SKILL_SNAPSHOT_SHA256"
                ),
            }
        )
    else:
        environment.update(
            {
                "C2HLS_SWEEP_SKILL_MODES": "off",
                "C2HLS_SKILL_MODE": "skill_off",
                "C2HLS_FORCE_SKILL_PROMPTS": "0",
            }
        )

    unresolved = [name for name in required_env if not os.environ.get(name)]
    if resolve_env:
        # Resolve only non-secret identity fields.  Credentials remain named
        # prerequisites and are never copied into generated matrix artifacts.
        revision_name = model["revision_env"]
        if os.environ.get(revision_name):
            environment["C2HLS_MODEL_REVISION"] = os.environ[revision_name]
        if method["skills"] == "frozen" and os.environ.get(
            "C2HLS_SKILL_SNAPSHOT_SHA256"
        ):
            environment["C2HLS_SKILL_SNAPSHOT_SHA256"] = os.environ[
                "C2HLS_SKILL_SNAPSHOT_SHA256"
            ]
        if method["skills"] == "frozen" and os.environ.get(
            "C2HLS_SKILL_LIBRARY_PATH"
        ):
            environment["C2HLS_SKILL_LIBRARY_PATH"] = os.environ[
                "C2HLS_SKILL_LIBRARY_PATH"
            ]

    return {
        "mapping_status": "supported",
        "command": (
            ["python", method["runner"], "--method", method_key]
            if method["runner"] == "run_paper_baseline.py"
            else ["python", method["runner"]]
        ),
        "environment": environment,
        "required_env": required_env,
        "unresolved_required_env": unresolved,
        "blocker": (
            "missing required environment: " + ", ".join(unresolved)
            if unresolved
            else ""
        ),
        "required_implementation": "",
    }


def expand_matrix(
    matrix: Mapping[str, Any], *, resolve_env: bool = False
) -> list[dict[str, Any]]:
    """Expand campaigns without launching or mutating external state."""

    rows: list[dict[str, Any]] = []
    models = matrix["models"]
    methods = matrix["methods"]
    for campaign in matrix["campaigns"]:
        benchmarks = _benchmark_entries(matrix, campaign["benchmark_set"])
        model_keys = _selection_keys(campaign["models"], models)
        method_keys = _selection_keys(campaign["methods"], methods)
        profile = matrix["profiles"][campaign["profile"]]
        for benchmark in benchmarks:
            for model_key in model_keys:
                for method_key in method_keys:
                    for seed in campaign["seeds"]:
                        row_id = "__".join(
                            (
                                campaign["id"],
                                benchmark["id"],
                                model_key,
                                method_key,
                                f"seed{seed}",
                            )
                        )
                        execution = _execution_for_row(
                            matrix,
                            campaign,
                            benchmark,
                            model_key,
                            method_key,
                            int(seed),
                            row_id,
                            resolve_env=resolve_env,
                        )
                        rows.append(
                            {
                                "row_id": row_id,
                                "campaign": campaign["id"],
                                "purpose": campaign["purpose"],
                                "profile": campaign["profile"],
                                "reference_blind": bool(profile["reference_blind"]),
                                "oracle_upper_bound": bool(
                                    profile.get("oracle_upper_bound", False)
                                ),
                                "benchmark": benchmark["id"],
                                "benchmark_role": benchmark.get(
                                    "stratum", benchmark.get("role", "")
                                ),
                                "benchmark_path": benchmark["path"],
                                "testbench": benchmark["testbench"],
                                "model_key": model_key,
                                "model_role": models[model_key]["role"],
                                "model_id": models[model_key]["model_id"],
                                "seed_control": models[model_key]["seed_control"],
                                "method": method_key,
                                "method_label": methods[method_key]["label"],
                                "seed": int(seed),
                                "budget": dict(matrix["budget"]),
                                "acceptance": dict(matrix["acceptance"]),
                                "execution": execution,
                            }
                        )
    return rows


def _validate_repo_benchmark(
    issues: list[Issue], repo: Path, entry: Mapping[str, Any], index: int,
    *, force_selected_cosim: bool = False,
) -> None:
    base_path = f"benchmark_sets.rodinia_primary[{index}]"
    bench_dir = repo / str(entry.get("path", ""))
    tb_path = repo / str(entry.get("testbench", ""))
    for filename in ("metadata.json", "plain.cpp"):
        if not (bench_dir / filename).is_file():
            _issue(
                issues,
                "error",
                "missing_benchmark_input",
                base_path,
                f"missing {bench_dir / filename}",
            )
    if not tb_path.is_file():
        _issue(
            issues,
            "error",
            "missing_golden_testbench",
            f"{base_path}.testbench",
            f"missing {tb_path}",
        )
        return
    raw_text = tb_path.read_text(errors="replace")
    text = raw_text.lower()
    evidence = {
        "golden/reference language": "golden reference" in text,
        "mismatch reporting": "mismatch" in text,
        "pass marker": "pass:" in text,
        "fail marker": "fail:" in text,
        "nonzero failure return": (
            "? 1 : 0" in text or "return errors" in text or "return (errors" in text
        ),
    }
    missing = [name for name, present in evidence.items() if not present]
    if entry.get("golden_check_required") is not True:
        missing.append("golden_check_required=true contract")
    if missing:
        _issue(
            issues,
            "error",
            "testbench_not_self_checking",
            f"{base_path}.testbench",
            "missing evidence: " + ", ".join(missing),
        )

    finite_tolerance_error = _finite_tolerance_policy_error(raw_text)
    if finite_tolerance_error:
        _issue(
            issues,
            "error",
            "nonfinite_tolerance_comparison",
            f"{base_path}.testbench",
            finite_tolerance_error,
        )

    metadata_path = bench_dir / "metadata.json"
    if metadata_path.is_file():
        try:
            metadata = json.loads(metadata_path.read_text())
        except json.JSONDecodeError as exc:
            _issue(
                issues,
                "error",
                "invalid_benchmark_metadata",
                base_path,
                str(exc),
            )
        else:
            source_repo = str(metadata.get("source_repo", "")).lower()
            if "rodinia-hls" not in source_repo:
                _issue(
                    issues,
                    "error",
                    "wrong_benchmark_source",
                    base_path,
                    f"expected Rodinia-HLS provenance, got {source_repo!r}",
                )
            if metadata.get("supports_csim") is not True:
                _issue(
                    issues,
                    "error",
                    "csim_not_enabled",
                    base_path,
                    "paper candidates require metadata supports_csim=true",
                )
            if metadata.get("testbench_file") != tb_path.name:
                _issue(
                    issues,
                    "error",
                    "testbench_not_wired",
                    base_path,
                    "metadata.testbench_file does not name the matrix testbench",
                )
            if metadata.get("supports_cosim") is not True and not force_selected_cosim:
                _issue(
                    issues,
                    "warning",
                    "selected_cosim_not_enabled",
                    base_path,
                    "selected-winner cosim is required by the paper but is not "
                    "enabled in current benchmark metadata",
                )


def validate_matrix(
    matrix: Mapping[str, Any], *, repo: Path | None = None
) -> list[Issue]:
    issues: list[Issue] = []

    if matrix.get("schema_version") != SCHEMA_VERSION:
        _issue(
            issues,
            "error",
            "schema_version",
            "schema_version",
            f"expected {SCHEMA_VERSION!r}",
        )

    budget = matrix.get("budget", {})
    for key in ("max_llm_candidates", "max_synthesis_evaluations"):
        if budget.get(key) != 5:
            _issue(
                issues,
                "error",
                "unmatched_budget",
                f"budget.{key}",
                "HPCA matched matrix requires an exact global cap of 5",
            )
    if budget.get("cosim_policy") != "selected_winner_only":
        _issue(
            issues,
            "error",
            "cosim_policy",
            "budget.cosim_policy",
            "cosim must execute only for the selected winner",
        )

    if matrix.get("statistics") != STATISTICS_CONTRACT:
        _issue(
            issues,
            "error",
            "statistics_contract",
            "statistics",
            "performance-profile and paired-bootstrap settings must remain "
            "exactly preregistered",
        )

    sets = matrix.get("benchmark_sets", {})
    primary = sets.get("rodinia_primary", [])
    if not isinstance(primary, list):
        primary = []
    primary_ids = _ids(primary)
    if tuple(primary_ids) != PRIMARY_BENCHMARKS:
        _issue(
            issues,
            "error",
            "primary_benchmark_set",
            "benchmark_sets.rodinia_primary",
            f"expected ordered set {list(PRIMARY_BENCHMARKS)!r}, got {primary_ids!r}",
        )
    if len(primary_ids) != len(set(primary_ids)):
        _issue(
            issues,
            "error",
            "duplicate_benchmark",
            "benchmark_sets.rodinia_primary",
            "benchmark identifiers must be unique",
        )
    representatives = sets.get("representative_three", [])
    representative_map = {
        str(item.get("id", "")): str(item.get("stratum", ""))
        for item in representatives
        if isinstance(item, Mapping)
    }
    if representative_map != REPRESENTATIVE_STRATA:
        _issue(
            issues,
            "error",
            "representative_strata",
            "benchmark_sets.representative_three",
            f"expected {REPRESENTATIVE_STRATA!r}, got {representative_map!r}",
        )

    models = matrix.get("models", {})
    if tuple(models) != MODEL_KEYS:
        _issue(
            issues,
            "error",
            "model_set",
            "models",
            f"expected exactly {list(MODEL_KEYS)!r}",
        )
    else:
        roles = {models[key].get("role") for key in MODEL_KEYS}
        if roles != {"primary_local", "commercial_anchor"}:
            _issue(
                issues,
                "error",
                "model_roles",
                "models",
                "matrix needs one primary_local model and one commercial_anchor",
            )
        for key in MODEL_KEYS:
            if not models[key].get("revision_env"):
                _issue(
                    issues,
                    "error",
                    "missing_model_revision",
                    f"models.{key}.revision_env",
                    "an immutable revision must be supplied at launch",
                )
        if models["qwen_27b"].get("seed_control") != "provider_enforced":
            _issue(
                issues,
                "error",
                "local_seed_control",
                "models.qwen_27b.seed_control",
                "local-model fixed seeds must be provider-enforced",
            )
        if models["qwen_27b"].get("model_id") != "qwen3.6-27b":
            _issue(
                issues,
                "error",
                "local_model_id",
                "models.qwen_27b.model_id",
                "primary local model must match deployed qwen3.6-27b identity",
            )
        if models["claude_sonnet_4_6"].get("seed_control") != (
            "unsupported_by_provider_record_as_repeated_trial"
        ):
            _issue(
                issues,
                "error",
                "commercial_seed_disclosure",
                "models.claude_sonnet_4_6.seed_control",
                "Claude seed limitation must remain explicit",
            )

    methods = matrix.get("methods", {})
    if tuple(methods) != METHOD_KEYS:
        _issue(
            issues,
            "error",
            "method_set",
            "methods",
            f"expected exactly {list(METHOD_KEYS)!r}",
        )
    for key, method in methods.items():
        status = method.get("mapping_status")
        if status not in {"supported", "unsupported"}:
            _issue(
                issues,
                "error",
                "mapping_status",
                f"methods.{key}.mapping_status",
                "must be supported or unsupported",
            )
        if status == "unsupported":
            if method.get("runner") is not None:
                _issue(
                    issues,
                    "error",
                    "unsafe_unsupported_mapping",
                    f"methods.{key}.runner",
                    "unsupported methods must have a null runner",
                )
            for field in ("unsupported_reason", "required_implementation"):
                if not method.get(field):
                    _issue(
                        issues,
                        "error",
                        "undocumented_unsupported_mapping",
                        f"methods.{key}.{field}",
                        "unsupported mappings require an explicit reason and remedy",
                    )
        if status == "supported":
            expected_runner = (
                "run_paper_baseline.py"
                if key in {"one_shot_best_of_five", "pragma_only"}
                else "run_agentic_sweep.py"
            )
            if method.get("runner") != expected_runner:
                _issue(
                    issues,
                    "error",
                    "unknown_runner_mapping",
                    f"methods.{key}.runner",
                    f"method requires fingerprinted runner {expected_runner!r}",
                )
            if not method.get("strategy"):
                _issue(
                    issues,
                    "error",
                    "missing_method_strategy",
                    f"methods.{key}.strategy",
                    "supported method requires an explicit strategy identity",
                )

    profiles = matrix.get("profiles", {})
    primary_profile = profiles.get("hpca2027_reference_blind", {})
    oracle_profile = profiles.get("legacy_oracle_reference_guided", {})
    if primary_profile.get("reference_blind") is not True:
        _issue(
            issues,
            "error",
            "primary_not_reference_blind",
            "profiles.hpca2027_reference_blind",
            "primary profile must be reference-blind",
        )
    required_blind_env = {
        "C2HLS_REFERENCE_BLIND": "1",
        "C2HLS_GT_COMPARISON_IN_CONTROL": "0",
        "C2HLS_PHASE8_BASELINE_ALIGN": "0",
        "C2HLS_PHASE5_GT_PREPOP": "0",
        "C2HLS_REFERENCE_CODE_IN_PROMPTS": "0",
        "C2HLS_REFERENCE_METRICS_IN_PROMPTS": "0",
        "C2HLS_FEEDBACK_LLM": "0",
    }
    actual_blind_env = primary_profile.get("env", {})
    for name, expected in required_blind_env.items():
        if actual_blind_env.get(name) != expected:
            _issue(
                issues,
                "error",
                "reference_isolation_invariant",
                f"profiles.hpca2027_reference_blind.env.{name}",
                f"expected {expected!r}",
            )
    if (
        oracle_profile.get("reference_blind") is not False
        or oracle_profile.get("oracle_upper_bound") is not True
    ):
        _issue(
            issues,
            "error",
            "oracle_profile_label",
            "profiles.legacy_oracle_reference_guided",
            "oracle must be explicitly non-blind and labelled as an upper bound",
        )

    common_env = matrix.get("common_runner_env", {})
    if "C2HLS_VITIS_SETTINGS" not in set(matrix.get("required_env", [])):
        _issue(
            issues,
            "error",
            "toolchain_probe_contract",
            "required_env",
            "C2HLS_VITIS_SETTINGS must be resolved so the invoked Vitis binary can be probed",
        )
    required_common_env = {
        "C2HLS_FEEDBACK_LLM": "0",
        "C2HLS_REFERENCE_CACHE_DIR": "artifacts/reference_validation_cache",
        "C2HLS_SWEEP_REFERENCE_CACHE_DIR": "artifacts/reference_validation_cache",
        "C2HLS_SWEEP_HW_EMU": "0",
        "C2HLS_HW_EMU_FINAL": "0",
        "C2HLS_CANDIDATES_PER_STEP": "5",
        "C2HLS_LLM_CANDIDATE_BUDGET": "5",
        "C2HLS_SYNTHESIS_EVAL_BUDGET": "5",
        "C2HLS_COSIM_SELECTED_ONLY": "1",
        "C2HLS_FORCE_SELECTED_COSIM": "1",
        "C2HLS_REFERENCE_COSIM": "0",
        "C2HLS_REFERENCE_COSIM_SELECTED_ONLY": "1",
        "C2HLS_REFERENCE_COSIM_BASELINE": "1",
        "C2HLS_CORRECTNESS_BEFORE_SYNTH": "1",
        "C2HLS_DISABLE_CORRECTNESS_REPAIR": "0",
        "C2HLS_SYNTH_REVERT_THRESHOLD": "0",
        "C2HLS_PHASE5_LLM_RETRY": "0",
        "C2HLS_CPU_GOLDEN_TIMEOUT": "180",
        "C2HLS_FEASIBILITY_SELECTION": "1",
        "C2HLS_SKILL_LIBRARY_FROZEN": "1",
        "C2HLS_SKILL_LIBRARY_PERSIST": "0",
        "C2HLS_SKILL_UPDATE_STATS": "0",
    }
    for name, expected in required_common_env.items():
        if common_env.get(name) != expected:
            _issue(
                issues,
                "error",
                "runner_contract",
                f"common_runner_env.{name}",
                f"expected {expected!r}",
            )

    campaigns = matrix.get("campaigns", [])
    campaign_by_id = {
        campaign.get("id"): campaign
        for campaign in campaigns
        if isinstance(campaign, Mapping)
    }
    expected_campaigns = {
        "primary_seed0",
        "representative_extra_seeds",
        "oracle_upper_bound_seed0",
    }
    if set(campaign_by_id) != expected_campaigns:
        _issue(
            issues,
            "error",
            "campaign_set",
            "campaigns",
            f"expected campaigns {sorted(expected_campaigns)!r}",
        )
    else:
        primary_campaign = campaign_by_id["primary_seed0"]
        replication_campaign = campaign_by_id["representative_extra_seeds"]
        oracle_campaign = campaign_by_id["oracle_upper_bound_seed0"]
        if primary_campaign.get("seeds") != [0]:
            _issue(
                issues,
                "error",
                "primary_seed",
                "campaigns.primary_seed0.seeds",
                "all nine primary kernels must run once at seed 0",
            )
        if replication_campaign.get("seeds") != [1, 2]:
            _issue(
                issues,
                "error",
                "replication_seeds",
                "campaigns.representative_extra_seeds.seeds",
                "the two additional fixed seeds must be 1 and 2",
            )
        if tuple(replication_campaign.get("methods", [])) != REPLICATION_METHODS:
            _issue(
                issues,
                "error",
                "replication_methods",
                "campaigns.representative_extra_seeds.methods",
                f"expected {list(REPLICATION_METHODS)!r}",
            )
        if oracle_campaign.get("benchmark_set") != "representative_three":
            _issue(
                issues,
                "error",
                "oracle_scope",
                "campaigns.oracle_upper_bound_seed0.benchmark_set",
                "oracle evaluation is limited to the three representative kernels",
            )
        if oracle_campaign.get("profile") != "legacy_oracle_reference_guided":
            _issue(
                issues,
                "error",
                "oracle_profile",
                "campaigns.oracle_upper_bound_seed0.profile",
                "oracle campaign must use the separately labelled guided profile",
            )

    try:
        rows = expand_matrix(matrix)
    except (KeyError, TypeError, ValueError) as exc:
        _issue(issues, "error", "expansion_failed", "campaigns", str(exc))
        rows = []
    if rows:
        row_ids = [row["row_id"] for row in rows]
        if len(row_ids) != len(set(row_ids)):
            _issue(
                issues,
                "error",
                "duplicate_row_id",
                "campaigns",
                "campaign expansion produced duplicate row identifiers",
            )
        counts = {
            "total_rows": len(rows),
            "supported_mapping_rows": sum(
                row["execution"]["mapping_status"] == "supported" for row in rows
            ),
            "unsupported_mapping_rows": sum(
                row["execution"]["mapping_status"] == "unsupported" for row in rows
            ),
            "primary_rows": sum(row["campaign"] == "primary_seed0" for row in rows),
            "replication_rows": sum(
                row["campaign"] == "representative_extra_seeds" for row in rows
            ),
            "oracle_rows": sum(
                row["campaign"] == "oracle_upper_bound_seed0" for row in rows
            ),
        }
        for name, actual in counts.items():
            expected = matrix.get("expected_counts", {}).get(name)
            if expected != actual:
                _issue(
                    issues,
                    "error",
                    "expanded_count",
                    f"expected_counts.{name}",
                    f"declared {expected!r}, expansion produced {actual}",
                )

    if repo is not None:
        force_selected_cosim = str(
            (matrix.get("common_runner_env") or {}).get(
                "C2HLS_FORCE_SELECTED_COSIM", "0"
            )
        ).lower() in {"1", "true", "yes", "on"}
        for index, entry in enumerate(primary):
            if isinstance(entry, Mapping):
                _validate_repo_benchmark(
                    issues,
                    repo,
                    entry,
                    index,
                    force_selected_cosim=force_selected_cosim,
                )
        for runner_name in {method.get("runner") for method in methods.values()}:
            if not runner_name:
                continue
            runner = repo / str(runner_name)
            if not runner.is_file():
                _issue(
                    issues,
                    "error",
                    "missing_runner",
                    "methods",
                    f"missing {runner}",
                )

    return issues


def summarize(matrix: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_campaign: dict[str, int] = {}
    by_mapping: dict[str, int] = {}
    for row in rows:
        by_campaign[row["campaign"]] = by_campaign.get(row["campaign"], 0) + 1
        status = row["execution"]["mapping_status"]
        by_mapping[status] = by_mapping.get(status, 0) + 1
    return {
        "schema_version": matrix.get("schema_version"),
        "dry_run_only": True,
        "rows": len(rows),
        "by_campaign": dict(sorted(by_campaign.items())),
        "by_mapping_status": dict(sorted(by_mapping.items())),
        "primary_benchmarks": _ids(matrix["benchmark_sets"]["rodinia_primary"]),
        "representative_three": _ids(
            matrix["benchmark_sets"]["representative_three"]
        ),
        "models": list(matrix["models"]),
        "methods": list(matrix["methods"]),
        "unsupported_methods": [
            key
            for key, value in matrix["methods"].items()
            if value["mapping_status"] == "unsupported"
        ],
    }


def _write_rows(path: Path, output_format: str, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if output_format == "jsonl":
        payload = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    else:
        payload = json.dumps(list(rows), indent=2, sort_keys=True) + "\n"
    path.write_text(payload)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate/dry-run-expand the HPCA 2027 matrix; never launches runs"
    )
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument(
        "--check-repo",
        action="store_true",
        help="also verify benchmark/testbench and runner files",
    )
    parser.add_argument("--out", type=Path, help="write expanded rows")
    parser.add_argument("--format", choices=("json", "jsonl"), default="jsonl")
    parser.add_argument(
        "--resolve-env",
        action="store_true",
        help="resolve revision/hash variables only; never serializes credentials",
    )
    parser.add_argument(
        "--fail-on-blocked",
        action="store_true",
        help="return nonzero if any row has an unsupported mapping or missing env",
    )
    args = parser.parse_args(argv)

    try:
        matrix = load_matrix(args.matrix)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(json.dumps({"valid": False, "error": str(exc)}, sort_keys=True))
        return 2

    issues = validate_matrix(matrix, repo=REPO if args.check_repo else None)
    errors = [issue for issue in issues if issue.severity == "error"]
    if errors:
        print(
            json.dumps(
                {
                    "valid": False,
                    "issues": [asdict(issue) for issue in issues],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 2

    rows = expand_matrix(matrix, resolve_env=args.resolve_env)
    summary = summarize(matrix, rows)
    summary["valid"] = True
    summary["issues"] = [asdict(issue) for issue in issues]
    summary["blocked_rows"] = sum(
        row["execution"]["mapping_status"] != "supported"
        or bool(row["execution"]["unresolved_required_env"])
        for row in rows
    )
    if args.out:
        _write_rows(args.out, args.format, rows)
        summary["expanded_output"] = str(args.out)
        summary["expanded_format"] = args.format
    print(json.dumps(summary, indent=2, sort_keys=True))

    if args.fail_on_blocked and summary["blocked_rows"]:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
