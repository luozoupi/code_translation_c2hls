#!/usr/bin/env python3
"""Post-flash DATAFLOW batch (task functions + #pragma HLS DATAFLOW).

Reads flash matrix cells, takes each bench's selected/final kernel, and runs the
DATAFLOW refactor LLM step. Validation uses the original benchmark testbench
internally for csim only — it is not shown to the LLM. Top-level kernel
signature must stay unchanged. csim + csynth (cosim off).

Example::

    python3 scripts/pc2/run_post_flash_dataflow.py --pc2 \\
        --matrix-root artifacts/pc2/flash_all_new_skills_avoids_global_20260623_024548 \\
        --dry-run

    python3 scripts/pc2/run_post_flash_dataflow.py --pc2 \\
        --matrix-root artifacts/pc2/flash_all_new_skills_avoids_global_20260623_024548 \\
        --benches hlsfactory_gemm,hlsfactory_atax

    # Print prompts only:
    python3 scripts/pc2/run_post_flash_dataflow.py --show-prompts
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "pc2"))

from c2hls_paths import BENCHMARKS_DIR, configure_site
from post_flash_dataflow import (
    DEFAULT_PROMPT_POLICY,
    PROMPT_POLICIES,
    configure_post_flash_env,
    discover_matrix_cells,
    format_results_root_name,
    kernel_bundle_dir_name,
    prompt_text_for_docs,
    recover_kernels_from_history,
    repair_round_limit,
    resolve_prompt_policy,
    resolve_selected_kernel,
    run_dataflow_for_cell,
    validate_recovered_dataflow_cell,
)
from dataflow_contract_check import contract_round_limit


def _split_csv(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _resolve_bench_dir(bench: str) -> Path:
    candidates = [
        BENCHMARKS_DIR / bench,
        REPO / "benchmarks_autosa_dse" / bench,
        REPO / "benchmarks_autosa" / bench,
        REPO / "related_work/benchmarks/HLSFactory_benchmarks/chathls_ready" / bench,
        REPO / "related_work/benchmarks/HLSFactory_benchmarks/tier_B_ready" / bench,
        REPO / "related_work/benchmarks/HLSFactory_benchmarks/tier_A_ready" / bench,
    ]
    for path in candidates:
        if (path / "metadata.json").is_file():
            return path
    raise ValueError(f"unknown benchmark: {bench}")


def _preflight_llm() -> None:
    """Fail fast if OPENAI_BASE_URL is missing or returns 401."""
    import urllib.error
    import urllib.request

    base = os.getenv("OPENAI_BASE_URL", "").strip().rstrip("/")
    if not base:
        raise RuntimeError(
            "OPENAI_BASE_URL is not set. Submit with ./scripts/pc2/start_post_flash_dataflow.sh "
            "--submit (supervised GPU session), or export OPENAI_BASE_URL to a live vLLM endpoint."
        )
    api_key = os.getenv("OPENAI_API_KEY", "EMPTY")
    url = f"{base}/models"
    req = urllib.request.Request(
        url,
        headers={"Authorization": f"Bearer {api_key}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            if resp.status != 200:
                raise RuntimeError(f"LLM preflight {url} returned HTTP {resp.status}")
    except urllib.error.HTTPError as exc:
        if exc.code == 401:
            raise RuntimeError(
                f"LLM preflight 401 Unauthorized for {url}. "
                "Stale or wrong endpoint — use start_post_flash_dataflow.sh --submit "
                "(starts fresh vLLM on gpu_h100) instead of a hardcoded gpu host."
            ) from exc
        raise RuntimeError(f"LLM preflight failed: HTTP {exc.code} for {url}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"LLM preflight cannot reach {url}: {exc}. "
            "Is the GPU vLLM job running?"
        ) from exc
    print(f"LLM preflight ok: {base}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Post-flash DATAFLOW refactor step")
    parser.add_argument("--pc2", action="store_true", help="PC2 site paths + vLLM")
    parser.add_argument("--matrix-root", type=str, default="")
    parser.add_argument("--benches", type=str, default="", help="Comma-separated filter")
    parser.add_argument("--model", type=str, default=os.getenv("C2HLS_MODEL", ""))
    parser.add_argument("--turns", type=int, default=repair_round_limit())
    parser.add_argument(
        "--contract-turns",
        type=int,
        default=contract_round_limit(),
        help="Max contract-check fix rounds before csynth (separate from --turns)",
    )
    parser.add_argument(
        "--no-contract-check",
        action="store_true",
        help="Skip hybrid contract check (static + LLM auditor) before validation",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true", help="Re-run even if result exists")
    parser.add_argument("--show-prompts", action="store_true", help="Print system/user prompts and exit")
    parser.add_argument(
        "--recover-kernels",
        action="store_true",
        help="Re-extract *_dataflow.cpp from existing *_dataflow_history.json (no LLM/HLS)",
    )
    parser.add_argument(
        "--validate-recovered",
        action="store_true",
        help="Run csim + csynth on existing *_dataflow.cpp (no LLM)",
    )
    parser.add_argument(
        "--no-prepare-kernel",
        action="store_true",
        help="With --validate-recovered: skip workload/INTERFACE prepare rewrite",
    )
    parser.add_argument(
        "--no-package-results",
        action="store_true",
        help="Skip building kernel bundle + reports after the batch",
    )
    parser.add_argument(
        "--results-suffix",
        type=str,
        default="parallel_fix",
        help="Suffix for auto-created results directory name",
    )
    parser.add_argument(
        "--prompt-policy",
        type=str,
        choices=list(PROMPT_POLICIES),
        default=os.getenv("C2HLS_POST_FLASH_PROMPT_POLICY", DEFAULT_PROMPT_POLICY),
        help=(
            "Prompt layout policy: system_skills=rules+skills in system (legacy); "
            "user_skills=rules in system, skills+task brief in user"
        ),
    )
    parser.add_argument(
        "--rag",
        action="store_true",
        help="Enable UG1399 documentation RAG (off by default). Default mode: flashopt.",
    )
    parser.add_argument(
        "--rag2",
        action="store_true",
        help="Enable RAG2 dual-policy retrieval (incompatible with --scrape).",
    )
    parser.add_argument(
        "--rag-mode",
        choices=["flashopt", "repair", "both", "everywhere"],
        default=None,
        help="RAG/RAG2 injection scope. Default: flashopt (--rag) or both (--rag2).",
    )
    parser.add_argument(
        "--rag-corpus",
        type=str,
        default=None,
        help="RAG index directory (default: artifacts/rag/ug1399).",
    )
    parser.add_argument(
        "--rag2-opt-corpus",
        type=str,
        default=None,
        help="RAG2 opt index directory (default: artifacts/rag/rag2_opt).",
    )
    parser.add_argument(
        "--rag2-repair-corpus",
        type=str,
        default=None,
        help="RAG2 repair index directory (default: artifacts/rag/rag2_repair).",
    )
    parser.add_argument(
        "--rag-top-k",
        type=int,
        default=None,
        help="Number of chunks to retrieve (default: 4).",
    )
    parser.add_argument(
        "--scrape",
        action="store_true",
        help="With --rag: analysis→keyword PDF/doc scrape before action prompts.",
    )
    parser.add_argument(
        "--scrape-corpus",
        type=str,
        default=None,
        help="Colon/comma-separated PDF/HTML/TXT paths for --scrape.",
    )
    args = parser.parse_args()

    if args.rag2 and args.scrape:
        parser.error("--rag2 is incompatible with --scrape")
    if args.scrape and not args.rag:
        parser.error("--scrape requires --rag")
    if args.rag_mode and not (args.rag or args.rag2):
        parser.error("--rag-mode requires --rag or --rag2")
    if args.rag2:
        os.environ["C2HLS_RAG2"] = "1"
        os.environ["C2HLS_RAG_SCRAPE"] = "0"
        if args.rag_mode:
            os.environ["C2HLS_RAG_MODE"] = args.rag_mode
        if args.rag_top_k is not None:
            os.environ["C2HLS_RAG_TOP_K"] = str(args.rag_top_k)
        if args.rag2_opt_corpus:
            os.environ["C2HLS_RAG2_OPT_CORPUS"] = args.rag2_opt_corpus
        if args.rag2_repair_corpus:
            os.environ["C2HLS_RAG2_REPAIR_CORPUS"] = args.rag2_repair_corpus
        from c2hls_rag import load_index
        from c2hls_rag2 import rag2_config_from_env

        rag2_cfg = rag2_config_from_env(
            enabled=True,
            mode=args.rag_mode,
            opt_corpus_dir=args.rag2_opt_corpus,
            repair_corpus_dir=args.rag2_repair_corpus,
            top_k=args.rag_top_k,
        )
        load_index(rag2_cfg.opt_corpus_dir)
        load_index(rag2_cfg.repair_corpus_dir)
    if args.rag:
        os.environ["C2HLS_RAG"] = "1"
        if args.rag_mode:
            os.environ["C2HLS_RAG_MODE"] = args.rag_mode
        if args.rag_corpus:
            os.environ["C2HLS_RAG_CORPUS"] = args.rag_corpus
        if args.rag_top_k is not None:
            os.environ["C2HLS_RAG_TOP_K"] = str(args.rag_top_k)
        if args.scrape:
            os.environ["C2HLS_RAG_SCRAPE"] = "1"
            if args.scrape_corpus:
                os.environ["C2HLS_RAG_SCRAPE_CORPUS"] = args.scrape_corpus
        from c2hls_rag import get_index, rag_config_from_env

        cfg = rag_config_from_env(
            enabled=True,
            mode=args.rag_mode,
            corpus_dir=args.rag_corpus,
            top_k=args.rag_top_k,
            scrape_enabled=args.scrape if args.scrape else None,
            scrape_corpus=args.scrape_corpus,
        )
        if args.scrape and not cfg.scrape_corpus_paths:
            parser.error("--scrape requires --scrape-corpus with existing files")
        if not args.scrape and not args.rag2:
            get_index(cfg)

    try:
        prompt_policy = resolve_prompt_policy(args.prompt_policy)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    if args.show_prompts:
        prompts = prompt_text_for_docs(prompt_policy)
        print(f"=== PROMPT POLICY: {prompts['prompt_policy']} ===\n")
        print("=== SYSTEM ===\n")
        print(prompts["system"])
        print(
            f"\n=== SKILLS ({prompts.get('skill_count', '?')} from "
            f"{prompts.get('skills_path', '?')}) — "
            f"in_system={prompts.get('skills_in_system')} "
            f"in_user={prompts.get('skills_in_user')} ===\n"
        )
        print("\n=== INITIAL USER (sample) ===\n")
        print(prompts["initial_user"])
        print("\n=== REPAIR USER (sample) ===\n")
        print(prompts["repair_user"])
        print("\n=== CONTRACT AUDIT SYSTEM ===\n")
        print(prompts.get("contract_audit_system", ""))
        print("\n=== CONTRACT AUDIT USER (sample) ===\n")
        print(prompts.get("contract_audit_user", ""))
        print("\n=== CONTRACT FIX USER (sample) ===\n")
        print(prompts.get("contract_fix_user", ""))
        return 0

    if not args.matrix_root.strip():
        print("--matrix-root is required (unless --show-prompts)", file=sys.stderr)
        return 1

    if args.pc2:
        configure_site("pc2")
    configure_post_flash_env()
    # Validate mode always needs cosim when requested; local.env defaults RUN_COSIM=0.
    if args.validate_recovered:
        os.environ["C2HLS_RUN_COSIM"] = "1"
    os.environ["C2HLS_DATAFLOW_REPAIR_ROUNDS"] = str(args.turns)
    os.environ["C2HLS_DATAFLOW_CONTRACT_ROUNDS"] = str(args.contract_turns)
    if args.no_contract_check:
        os.environ["C2HLS_DATAFLOW_CONTRACT_CHECK"] = "0"

    matrix_root = Path(args.matrix_root).expanduser()
    if not matrix_root.is_absolute():
        matrix_root = REPO / matrix_root
    if not matrix_root.is_dir():
        print(f"matrix root missing: {matrix_root}", file=sys.stderr)
        return 1

    bench_filter = set(_split_csv(args.benches)) if args.benches else None
    cells = discover_matrix_cells(matrix_root)

    if args.recover_kernels:
        recovered = recover_kernels_from_history(matrix_root, bench_filter=bench_filter)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        out_path = matrix_root / f"post_flash_dataflow_recover_{stamp}.json"
        out_path.write_text(json.dumps(recovered, indent=2) + "\n", encoding="utf-8")
        ok = sum(1 for row in recovered if row.get("recovered"))
        print(f"recover: {ok}/{len(recovered)} kernels -> {out_path}")
        for row in recovered:
            if row.get("recovered"):
                print(f"  {row['bench']}: {row['kernel']}")
        return 0 if ok > 0 else 1

    if args.validate_recovered:
        if bench_filter:
            cells = [c for c in cells if c["bench"] in bench_filter]
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        summary: list[dict[str, Any]] = []
        for cell in cells:
            bench = cell["bench"]
            cell_dir = Path(cell["cell_dir"])
            paths = cell_dir / f"{bench}_dataflow.cpp"
            if not paths.is_file():
                print(f"SKIP {bench}: no *_dataflow.cpp", flush=True)
                summary.append({"bench": bench, "skipped": True, "reason": "no dataflow cpp"})
                continue
            print(f"VALIDATE {bench}", flush=True)
            t0 = time.time()
            try:
                outcome = validate_recovered_dataflow_cell(
                    bench=bench,
                    bench_dir=_resolve_bench_dir(bench),
                    cell_dir=cell_dir,
                    skip_existing=not args.force,
                    prepare_kernel=not args.no_prepare_kernel,
                )
            except Exception as exc:
                print(f"ERROR {bench}: {exc}", flush=True)
                summary.append({"bench": bench, "error": str(exc)})
                continue
            elapsed = round(time.time() - t0, 1)
            row = {
                "bench": bench,
                "elapsed_s": elapsed,
                "success": outcome.success,
                "error": outcome.error,
                "latency_cycles": (outcome.result or {}).get("latency_cycles"),
            }
            summary.append(row)
            print(f"DONE {bench} elapsed={elapsed}s success={outcome.success}", flush=True)

        out_summary = matrix_root / f"post_flash_dataflow_validate_{stamp}.json"
        out_summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(f"summary: {out_summary}")
        ok = sum(1 for row in summary if row.get("success"))
        attempted = sum(1 for row in summary if not row.get("skipped"))
        print(f"passed: {ok}/{attempted}")
        return 0 if ok == attempted and attempted > 0 else 1

    if bench_filter:
        cells = [c for c in cells if c["bench"] in bench_filter]

    plan: list[dict[str, Any]] = []
    for cell in cells:
        bench = cell["bench"]
        cell_dir = Path(cell["cell_dir"])
        kpath, role = resolve_selected_kernel(cell_dir, bench)
        plan.append({
            "bench": bench,
            "cell_dir": str(cell_dir),
            "kernel": str(kpath) if kpath else None,
            "kernel_role": role,
            "flash_status": cell.get("status"),
        })

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    plan_path = matrix_root / f"post_flash_dataflow_plan_{stamp}.json"
    plan_path.write_text(json.dumps({
        "matrix_root": str(matrix_root),
        "repair_rounds": repair_round_limit(),
        "contract_rounds": contract_round_limit(),
        "contract_check": not args.no_contract_check,
        "prompt_policy": prompt_policy,
        "results_suffix": args.results_suffix.strip(),
        "cells": plan,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }, indent=2) + "\n", encoding="utf-8")
    print(f"plan: {plan_path} ({len(plan)} cells, prompt_policy={prompt_policy})")

    if args.dry_run:
        for row in plan:
            print(f"  {row['bench']}: kernel={row['kernel']}")
        return 0

    _preflight_llm()

    from c2hls import C2HLSOrchestrator, DEFAULT_MODEL_ID

    model = args.model.strip() or DEFAULT_MODEL_ID
    orch = C2HLSOrchestrator(gpt_model=model, turns_limitation=args.turns)

    summary: list[dict[str, Any]] = []
    for cell in plan:
        bench = cell["bench"]
        if not cell.get("kernel"):
            print(f"SKIP {bench}: no kernel", flush=True)
            summary.append({"bench": bench, "skipped": True, "reason": "no kernel"})
            continue
        print(f"START {bench}", flush=True)
        t0 = time.time()
        try:
            outcome = run_dataflow_for_cell(
                bench=bench,
                bench_dir=_resolve_bench_dir(bench),
                cell_dir=Path(cell["cell_dir"]),
                orchestrator=orch,
                skip_existing=not args.force,
                prompt_policy=prompt_policy,
            )
        except Exception as exc:
            print(f"ERROR {bench}: {exc}", flush=True)
            summary.append({"bench": bench, "error": str(exc)})
            continue
        elapsed = round(time.time() - t0, 1)
        row = {
            "bench": bench,
            "elapsed_s": elapsed,
            "success": outcome.success,
            "error": outcome.error,
        }
        summary.append(row)
        print(f"DONE {bench} elapsed={elapsed}s success={outcome.success}", flush=True)

    out_summary = matrix_root / f"post_flash_dataflow_summary_{stamp}.json"
    out_summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    meta_path = matrix_root / f"post_flash_dataflow_summary_meta_{stamp}.json"
    meta_path.write_text(json.dumps({
        "prompt_policy": prompt_policy,
        "results_suffix": args.results_suffix.strip(),
        "summary_path": str(out_summary),
    }, indent=2) + "\n", encoding="utf-8")
    print(f"summary: {out_summary} (prompt_policy={prompt_policy})")
    ok = sum(1 for row in summary if row.get("success"))
    attempted = sum(1 for row in summary if not row.get("skipped"))
    print(f"passed: {ok}/{attempted}")

    results_root: Path | None = None
    if not args.no_package_results and attempted > 0:
        suffix = args.results_suffix.strip() or "aligned_prompt"
        results_root = matrix_root / format_results_root_name(
            stamp,
            results_suffix=suffix,
            prompt_policy=prompt_policy,
        )
        print(f"packaging results -> {results_root}", flush=True)
        from build_post_flash_dataflow_results import build_results

        flash_bundle = REPO / "artifacts/pc2/flash_selected_bundle" / matrix_root.name
        old_bundle = matrix_root / "post_flash_dataflow_kernel_bundle"
        build_out = build_results(
            matrix_root=matrix_root,
            flash_bundle_root=flash_bundle,
            results_root=results_root,
            summary_path=out_summary,
            old_kernel_bundle=old_bundle if old_bundle.is_dir() else None,
            prompt_policy=prompt_policy,
            force=True,
        )
        print(f"report: {build_out['report_md']}", flush=True)

    # Full-batch supervised sessions exit 0 even with per-bench failures so
    # auto-stop can scancel the GPU. Streaming single-bench runs (--benches)
    # must return non-zero on failure so the watcher does not mark them done.
    batch_done = len(summary) == len(plan)
    if not batch_done:
        return 1
    if bench_filter and attempted > 0 and ok < attempted:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
