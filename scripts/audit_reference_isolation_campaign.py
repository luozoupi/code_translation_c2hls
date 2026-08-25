#!/usr/bin/env python3
"""Fail-closed static leakage audit for a reference-blind C2HLS campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import c2hls  # noqa: E402
from evaluation_repro import (  # noqa: E402
    PAPER_PROFILE,
    REFERENCE_BLIND_OVERRIDES,
    apply_evaluation_profile,
    skill_snapshot_manifest,
)
from reference_isolation import audit_messages  # noqa: E402


DEFAULT_BENCHMARKS = (
    "hlsfactory_2mm",
    "hlsfactory_3mm",
    "hlsfactory_atax",
    "hlsfactory_bicg",
    "hlsfactory_cholesky",
    "hlsfactory_correlation",
    "hlsfactory_covariance",
    "hlsfactory_durbin",
    "hlsfactory_fdtd-2d",
    "hlsfactory_floyd-warshall",
    "hlsfactory_gemm",
    "hlsfactory_gemver",
    "hlsfactory_gesummv",
    "hlsfactory_gramschmidt",
    "hlsfactory_heat-3d",
    "hlsfactory_jacobi-1d",
    "hlsfactory_jacobi-2d",
    "hlsfactory_lu",
    "hlsfactory_ludcmp",
    "hlsfactory_mvt",
    "hlsfactory_nussinov",
    "hlsfactory_seidel-2d",
    "hlsfactory_symm",
    "hlsfactory_syr2k",
    "hlsfactory_syrk",
    "hlsfactory_trisolv",
    "hlsfactory_trmm",
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_value(path: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(path), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def source_manifest(paths: list[Path]) -> dict[str, Any]:
    records = [
        {
            "path": str(path.relative_to(REPO)),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in paths
    ]
    encoded = json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    return {
        "file_count": len(records),
        "files": records,
        "sha256": hashlib.sha256(encoded).hexdigest(),
    }


def build_context(benchmark_dir: Path, metadata: dict[str, Any]) -> str:
    header_name = str(metadata.get("header_file") or "kernel.h")
    header_path = benchmark_dir / header_name
    testbench_name = str(metadata.get("testbench_file") or "")
    testbench_path = benchmark_dir / testbench_name if testbench_name else None
    plain_name = str(metadata.get("plain_c_file") or "plain.cpp")
    previous = os.environ.get(c2hls.REFERENCE_BLIND_ENV)
    os.environ[c2hls.REFERENCE_BLIND_ENV] = "1"
    try:
        return c2hls._build_benchmark_context(
            metadata,
            header_name,
            header_path.read_text() if header_path.is_file() else "",
            (benchmark_dir / plain_name).read_text(),
            testbench_path.read_text() if testbench_path and testbench_path.is_file() else "",
        )
    finally:
        if previous is None:
            os.environ.pop(c2hls.REFERENCE_BLIND_ENV, None)
        else:
            os.environ[c2hls.REFERENCE_BLIND_ENV] = previous


def reference_cache_data(
    cache_dir: Path, benchmark: str
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    entries: list[dict[str, Any]] = []
    files: list[dict[str, Any]] = []
    for path in sorted(cache_dir.glob(f"{benchmark}.*.json")):
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        reference_validation = payload.get("reference_validation")
        if not isinstance(reference_validation, dict):
            continue
        entries.append(reference_validation)
        files.append({
            "name": path.name,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    return {"cache_entries": entries}, files


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Reference-Isolation Campaign Audit",
        "",
        f"Status: **{report['status'].upper()}**",
        "",
        "## Campaign",
        "",
        f"- Benchmarks: {report['campaign']['benchmark_count']}",
        f"- Setups: {report['campaign']['setup_count']} ({', '.join(report['campaign']['setups'])})",
        f"- Strategies: {', '.join(report['campaign']['strategies'])}",
        f"- Reference enforcement: `{report['campaign']['reference_enforcement']}`",
        f"- COSIM: disabled",
        "",
        "## Checks",
        "",
        "| Check | Status | Findings |",
        "|---|---|---:|",
    ]
    for check in report["checks"]:
        lines.append(
            f"| {check['name']} | {'PASS' if check['passed'] else 'FAIL'} | "
            f"{check.get('finding_count', 0)} |"
        )
    disclosure = report["disclosure"]
    lines.extend([
        "",
        "## Interpretation",
        "",
        "- No exact expert path, expert-only identifier, or expert-code signature may appear in the static prompt or skill corpus.",
        "- Runtime transcript auditing is strict. Unmatched or explicitly labeled expert metrics fail an arm.",
        "- Exact metrics independently produced by candidate Vitis reports are retained as hashed provenance collisions, not leaks.",
        "- Reference source objects are scrubbed before controller generation begins.",
        "- Gold/reference reports remain available only for validation and post-run reporting.",
        "",
        "## Disclosure",
        "",
        f"- Skills with project-history origins: {disclosure['project_history_skill_count']} of {disclosure['skill_count']}.",
        f"- Inputs marked as pragma-stripped derivatives of benchmark HLS source: {disclosure['plain_derived_from_gold_count']} of {report['campaign']['benchmark_count']}.",
        "- Therefore this campaign measures in-domain skill reuse, not clean unseen-suite generalization.",
        "",
        "## Identity",
        "",
        f"- Controller commit: `{report['identity']['controller_commit']}`",
        f"- Controller source manifest SHA-256: `{report['identity']['controller_source_manifest']['sha256']}`",
        f"- Controller worktree dirty: `{str(report['identity']['controller_worktree_dirty']).lower()}`",
        f"- Benchmark commit: `{report['identity']['benchmark_commit']}`",
        f"- Benchmark worktree dirty: `{str(report['identity']['benchmark_worktree_dirty']).lower()}`",
        f"- Skill file SHA-256: `{report['identity']['skill_file_sha256']}`",
        f"- Skill manifest SHA-256: `{report['identity']['skill_manifest_sha256']}`",
    ])
    if report["failures"]:
        lines.extend(["", "## Failures", ""])
        lines.extend(f"- {failure}" for failure in report["failures"])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmarks-dir", type=Path, required=True)
    parser.add_argument("--skill-library", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--reference-cache-dir",
        type=Path,
        default=REPO / "artifacts" / "reference_validation_cache",
    )
    parser.add_argument(
        "--benchmarks",
        default=",".join(DEFAULT_BENCHMARKS),
        help="Comma-separated benchmark directory names.",
    )
    parser.add_argument("--prompt-source", type=Path, action="append", default=[])
    args = parser.parse_args()

    benchmarks = tuple(item.strip() for item in args.benchmarks.split(",") if item.strip())
    prompt_sources = args.prompt_source or [REPO / "prompt_c2hls.py"]
    static_text = "\n".join(
        [args.skill_library.read_text(), *(path.read_text() for path in prompt_sources)]
    )
    skills_payload = json.loads(args.skill_library.read_text())
    skills = skills_payload.get("skills") or []
    origin_counts = Counter(str(skill.get("origin") or "unknown") for skill in skills)

    profile_env: dict[str, str] = {"C2HLS_SWEEP_PROFILE": PAPER_PROFILE}
    profile = apply_evaluation_profile(environ=profile_env)
    invariant_failures = [
        f"{key}={profile_env.get(key)!r}, expected {expected!r}"
        for key, expected in REFERENCE_BLIND_OVERRIDES.items()
        if profile_env.get(key) != expected
    ]

    static_findings: list[dict[str, Any]] = []
    context_findings: list[dict[str, Any]] = []
    missing_inputs: list[str] = []
    plain_derived_count = 0
    benchmark_records: list[dict[str, Any]] = []
    reference_cache_files: list[dict[str, Any]] = []
    for benchmark in benchmarks:
        benchmark_dir = args.benchmarks_dir / benchmark
        metadata_path = benchmark_dir / "metadata.json"
        if not metadata_path.is_file():
            missing_inputs.append(f"missing metadata: {metadata_path}")
            continue
        metadata = json.loads(metadata_path.read_text())
        plain_derived = bool(
            (metadata.get("provenance") or {}).get("plain_derived_from_gold_hls")
        )
        plain_derived_count += int(plain_derived)
        reference_data, cache_files = reference_cache_data(
            args.reference_cache_dir, benchmark
        )
        if not cache_files:
            missing_inputs.append(
                f"missing reference metric audit cache for {benchmark}"
            )
        reference_cache_files.extend(
            {"benchmark": benchmark, **record} for record in cache_files
        )
        static_audit = audit_messages(
            [{"role": "system", "content": static_text}],
            benchmark_dir=benchmark_dir,
            reference_data=reference_data,
        )
        context = build_context(benchmark_dir, metadata)
        context_audit = audit_messages(
            [{"role": "user", "content": context}],
            benchmark_dir=benchmark_dir,
            reference_data=reference_data,
        )
        if not static_audit["passed"]:
            static_findings.append({
                "benchmark": benchmark,
                "finding_counts": static_audit["finding_counts"],
            })
        if not context_audit["passed"]:
            context_findings.append({
                "benchmark": benchmark,
                "finding_counts": context_audit["finding_counts"],
            })
        benchmark_records.append({
            "benchmark": benchmark,
            "metadata_sha256": sha256_file(metadata_path),
            "static_corpus_audit_passed": static_audit["passed"],
            "benchmark_context_audit_passed": context_audit["passed"],
            "plain_derived_from_gold_hls": plain_derived,
            "reference_metric_cache_file_count": len(cache_files),
        })

    skill_env = {
        "C2HLS_SKILL_LIBRARY_PATH": str(args.skill_library.resolve()),
        "C2HLS_SKILL_LIBRARY_FROZEN": "1",
    }
    skill_manifest = skill_snapshot_manifest(REPO, environ=skill_env)
    controller_sources = source_manifest([
        REPO / "c2hls.py",
        REPO / "evaluation_repro.py",
        REPO / "hls_eval.py",
        REPO / "reference_isolation.py",
        REPO / "run_agentic_sweep.py",
        REPO / "skill_library.py",
        REPO / "smart_skill_router.py",
        REPO / "prompt_c2hls.py",
    ])
    controller_status = git_value(REPO, "status", "--porcelain")
    benchmark_repo = args.benchmarks_dir.parent
    benchmark_status = git_value(benchmark_repo, "status", "--porcelain")
    checks = [
        {
            "name": "reference-blind profile invariants",
            "passed": not invariant_failures and profile.get("reference_blind") is True,
            "finding_count": len(invariant_failures),
        },
        {
            "name": "static prompts and skill corpus vs expert sources",
            "passed": not static_findings,
            "finding_count": len(static_findings),
        },
        {
            "name": "rendered public benchmark contexts vs expert sources",
            "passed": not context_findings,
            "finding_count": len(context_findings),
        },
        {
            "name": "complete frozen benchmark and skill inputs",
            "passed": not missing_inputs and skill_manifest.get("file_count") == 1,
            "finding_count": len(missing_inputs),
        },
        {
            "name": "strict runtime transcript enforcement",
            "passed": profile_env.get("C2HLS_REFERENCE_BLIND_FAIL_ON_LEAK") == "1",
            "finding_count": 0 if profile_env.get("C2HLS_REFERENCE_BLIND_FAIL_ON_LEAK") == "1" else 1,
        },
    ]
    failures = [
        *invariant_failures,
        *missing_inputs,
        *(f"static corpus overlap for {item['benchmark']}: {item['finding_counts']}" for item in static_findings),
        *(f"benchmark-context overlap for {item['benchmark']}: {item['finding_counts']}" for item in context_findings),
    ]
    passed = all(check["passed"] for check in checks)
    project_history_skill_count = sum(
        count
        for origin, count in origin_counts.items()
        if origin.startswith("project_")
    )
    report = {
        "schema_version": "c2hls.reference-isolation-campaign-audit.v1",
        "status": "passed" if passed else "failed",
        "campaign": {
            "benchmark_count": len(benchmarks),
            "benchmarks": list(benchmarks),
            "strategies": ["flash", "dynamic"],
            "setups": [
                "skillless",
                "matched",
                "smart_best_fit",
                "smart_exhaustive",
                "all_positive",
            ],
            "setup_count": 10,
            "reference_enforcement": "strict",
            "cosim": False,
        },
        "identity": {
            "controller_commit": git_value(REPO, "rev-parse", "HEAD"),
            "controller_worktree_dirty": bool(controller_status),
            "controller_worktree_status_sha256": hashlib.sha256(
                controller_status.encode()
            ).hexdigest(),
            "controller_source_manifest": controller_sources,
            "benchmark_commit": git_value(benchmark_repo, "rev-parse", "HEAD"),
            "benchmark_worktree_dirty": bool(benchmark_status),
            "benchmark_worktree_status_sha256": hashlib.sha256(
                benchmark_status.encode()
            ).hexdigest(),
            "skill_file": str(args.skill_library.resolve()),
            "skill_file_sha256": sha256_file(args.skill_library),
            "skill_manifest_sha256": skill_manifest.get("sha256"),
            "prompt_sources": [
                {"path": str(path.resolve()), "sha256": sha256_file(path)}
                for path in prompt_sources
            ],
            "reference_metric_cache": {
                "directory": str(args.reference_cache_dir.resolve()),
                "file_count": len(reference_cache_files),
                "files_sha256": hashlib.sha256(
                    json.dumps(
                        reference_cache_files,
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode()
                ).hexdigest(),
            },
        },
        "checks": checks,
        "profile": profile,
        "benchmarks": benchmark_records,
        "static_findings": static_findings,
        "context_findings": context_findings,
        "failures": failures,
        "disclosure": {
            "skill_count": len(skills),
            "skill_origin_counts": dict(sorted(origin_counts.items())),
            "project_history_skill_count": project_history_skill_count,
            "plain_derived_from_gold_count": plain_derived_count,
            "interpretation": "in_domain_skill_reuse_not_unseen_suite_generalization",
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.output_dir / "audit.md").write_text(markdown_report(report))
    print(json.dumps({
        "status": report["status"],
        "benchmark_count": len(benchmarks),
        "check_count": len(checks),
        "failure_count": len(failures),
        "output_dir": str(args.output_dir),
    }, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
