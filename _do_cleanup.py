#!/usr/bin/env python3
"""Reversible archive/cleanup for e:/courses/UMN/c2hls (+ code_translation_c2hls).

Archive-only: everything moves into an `_archive/` tree (never deleted, except
__pycache__). Every move is logged to _archive/ARCHIVE_MANIFEST.tsv and a
generated _archive/UNDO.sh reverses it. A safety-grep prevents moving any .py
that is imported by KEPT framework/analysis code.

DRY-RUN by default (prints planned moves). Pass --apply to actually move.
See C:/Users/renji/.claude/plans/hi-i-have-isntalled-snazzy-aurora.md
"""
from __future__ import annotations
import os, re, sys, shutil
from pathlib import Path

APPLY = "--apply" in sys.argv
CHILD = Path(__file__).resolve().parent                 # code_translation_c2hls
PARENT = CHILD.parent                                    # e:/courses/UMN/c2hls

# ---------------------------------------------------------------- PROTECTED sets
PROTECTED_CHILD = {
    # entry points + imported modules
    "c2hls.py", "matrix_sweep.py", "prompt_c2hls.py", "hls_eval.py", "gold_cache.py",
    "c2hls_temp.py", "hls_feedback.py", "skill_library.py", "skills.py",
    "bottleneck_router.py", "robustness.py", "candidate_cache.py", "prepare_benchmarks.py",
    # runtime cfg / secrets
    "api-key.txt", "alcf-token.txt", ".env",
    # gold cache (read on reruns)
    "gold_reports_vitis_csynth.json", "gold_reports_vitis_full.json",
    "gold_reports_xcu280-fsvh2892-2l-e.json",
    # docs / deliverable kept at root
    "README.md", "requirements.txt", "jsonl_schema.md", "schema_records.jsonl",
    # this script
    "_do_cleanup.py",
}
PROTECTED_CHILD_DIRS = {
    "dataset_pipeline", "analysis", "scripts", "tests", "benchmarks", "skills",
    "hls_full_optimization_skills_schema_1_1_package", "artifacts", "_artix7",
    "analysis_tools", "_archive",
}
KEPT5 = {  # -> analysis_tools/ (handled separately, not archived)
    "_emit_all_arms_schema.py", "_compare_ms_ablation.py", "_summary_table.py",
    "_success_rates.py", "_count_tokens.py",
}
# result dirs that stay at root (active / referenced by kept analysis tools)
ACTIVE_RESULT_DIRS = {
    "results_matrix_u280_ENH_allpositive_flash_OPUS",
    "results_matrix_u280_ENH_allpositive_flash_SKILLS_OPUS",
    "results_matrix_u280_ENH_allpositive_multistep_skills_OPUS",
    "results_matrix_u280_ENH_allpositive_multistep_skills_symmrerun_OPUS",
    "results_matrix_u280_ENH_curated_flash_OPUS",
    "results_matrix_u280_ENH_curated_multistep_skills_OPUS",
    "results_matrix_u280_ENH_curated_multistep_skills_symmrerun_OPUS",
    "results_matrix_u280_ENH_oneshot_OPUS",
    "results_matrix_u280_ENH_allpositive_jacobi2d_rerun",
}
SUPERSEDED_RESULT_DIRS = {
    "results_matrix_u280_fullcosim", "results_matrix_u280_fullcosim_extended",
    "results_matrix_u280_fullcosim_OPUS", "results_matrix_u280_fullcosim_allpositive_OPUS",
    "results_matrix_u280_fullcosim_extended_OPUS", "results_matrix_u280_fullcosim_extended_BUGGED",
    "results_matrix_u280_fullcosim_noguard", "results_matrix_u280_fullcosim_noguard_OPUS",
    "results_matrix_u280_multistep", "results_matrix_u280_multistep_base_OPUS",
    "results_matrix_u280_multistep_old_skills", "results_matrix_u280_fixed_protocol",
    "results_matrix_u280_ENH_curated_multistep_OPUS", "results_matrix_skills_ext_v2",
    "results", "results_multistep", "_smoke_enhanced",
}

PROTECTED_PARENT = {
    "api-key.txt", "skills.json", "inference_auth_token.py", ".env",
}
PROTECTED_PARENT_DIRS = {
    "code_translation_c2hls", "hls_full_optimization_skills_schema_1_1_package",
    "HLSFactory", "rodinia-hls-nova-1", "c2hls_ahmed", "argo-shim",
    "vitis_install_test", "hpca2027-latex-template", "aes_csim_diag", "logs", "_archive",
}

# ---------------------------------------------------------------- safety-grep
def build_importer_text() -> str:
    """Concatenate all KEPT framework/analysis .py so we can detect inbound imports."""
    parts = []
    scan_dirs = [CHILD / d for d in ("dataset_pipeline", "analysis", "scripts", "tests")]
    scan_files = [CHILD / n for n in PROTECTED_CHILD if n.endswith(".py")]
    scan_files += [CHILD / "analysis_tools" / n for n in KEPT5]  # future location
    scan_files += [CHILD / n for n in KEPT5]                     # current location
    scan_files += [PARENT / "inference_auth_token.py"]
    for d in scan_dirs:
        if d.is_dir():
            for p in d.rglob("*.py"):
                try: parts.append(p.read_text(encoding="utf-8", errors="ignore"))
                except OSError: pass
    for p in scan_files:
        if p.is_file():
            try: parts.append(p.read_text(encoding="utf-8", errors="ignore"))
            except OSError: pass
    return "\n".join(parts)

IMPORTER_TEXT = build_importer_text()

def is_imported(stem: str) -> bool:
    return bool(re.search(rf"(?m)^\s*(?:import\s+{re.escape(stem)}\b|from\s+{re.escape(stem)}\s+import)", IMPORTER_TEXT))

# ---------------------------------------------------------------- categorizers
def cat_child_file(name: str):
    if name in PROTECTED_CHILD or name in KEPT5:
        return None
    if name.endswith((".log", ".out")):
        return "logs"
    if name.endswith((".csv", ".html")):
        return "reports"
    py = name.endswith(".py")
    if py and (name in ("test.py", "test_phase_c.py") or re.match(r"_(smoke_|test_|probe_|wsl_smoke_loader|framework_provenance|extract_|skills_taxonomy|strip_)", name)):
        return "scripts/smoke_probe"
    if py and (re.match(r"(_check_|_debug_|_diag_|_inspect_|_verify_|_validate_|_list_|inspect_|_investigate_|_noguard_opus_status|_opus_queue_status|validate_)", name) or name == "timeout_check.py"):
        return "scripts/validation"
    if py and (re.match(r"(_compare_|_analyze_|_audit_|_count_|_build_|_recalc_|_scan_|_augment_|_backfill_|_find_|_rebaseline_|_resynth_|_off_vs_on_diff|_phase9_summary|_fullcosim_summary|compare_|aggregate_matrix|dump_matrix_csv|score_matrix|export_ml4accel)", name)):
        return "scripts/analysis"
    if py and name.startswith("run_"):
        return "scripts/experiments"
    if name.endswith(".sh"):
        return "scripts/launchers"
    if name.endswith(".jsonl") and (name.startswith("u280_") or name.startswith("cosim_")):
        return "schema_exports"
    if name == "install_2023.2_config.txt":
        return "schema_exports"
    return None

def cat_parent_file(name: str):
    if name in PROTECTED_PARENT:
        return None
    low = name.lower()
    if low.endswith((".pdf", ".pptx")):
        return "docs"
    if (low.startswith("miniconda3") and low.endswith(".sh")) or low.startswith("installlibs.sh") \
       or low.endswith((".tar.gz", ".zip")) or low.endswith(".png"):
        return "installers"
    if name.endswith(".jsonl") or name == "schema_manifest.json" or name == "vitis_hls.log":
        return "schema_exports"
    if low.endswith((".md", ".csv", ".log")) or name.endswith(".summary.json") \
       or name.startswith("agentic_") or name.startswith("hlsfactory_") or name == "matrix_full.csv":
        return "reports"
    if name.endswith(".json"):  # remaining loose parent json = reports
        return "reports"
    if name == "analyze_multistep.py":
        return None if is_imported("analyze_multistep") else "reports"
    return None

# ---------------------------------------------------------------- move engine
manifest: list[tuple[str, str]] = []
skipped: list[str] = []

def plan_move(src: Path, arch_root: Path, subdir: str):
    dst_dir = arch_root / subdir
    dst = dst_dir / src.name
    # safety-grep for .py
    if src.suffix == ".py" and is_imported(src.stem):
        skipped.append(f"{src}  (imported by kept code)")
        return
    print(f"  {src.relative_to(PARENT)}  ->  _archive/{subdir}/")
    if APPLY:
        dst_dir.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
    manifest.append((str(dst), str(src)))

def process_level(root: Path, protected_files: set, protected_dirs: set, file_cat, extra_dirs=None):
    arch = root / "_archive"
    print(f"\n### {root}  ({'APPLY' if APPLY else 'DRY-RUN'}) ###")
    # files
    for p in sorted(root.iterdir()):
        if p.is_file():
            sub = file_cat(p.name)
            if sub:
                plan_move(p, arch, sub)
    # result/data dirs (child only, via extra_dirs mapping)
    for dname, sub in (extra_dirs or {}).items():
        d = root / dname
        if d.is_dir():
            print(f"  {d.relative_to(PARENT)}/  ->  _archive/{sub}/")
            if APPLY:
                (arch / sub).mkdir(parents=True, exist_ok=True)
                shutil.move(str(d), str(arch / sub / dname))
            manifest.append((str(arch / sub / dname), str(d)))

def delete_pycache(root: Path, skip_top_dirs: set | None = None):
    skip_top_dirs = skip_top_dirs or set()
    for pc in root.rglob("__pycache__"):
        if "_archive" in pc.parts:
            continue
        rel = pc.relative_to(root)
        if rel.parts and rel.parts[0] in skip_top_dirs:  # don't descend into subrepos
            continue
        print(f"  delete {pc.relative_to(PARENT)}")
        if APPLY:
            shutil.rmtree(pc, ignore_errors=True)

def write_undo_and_manifest(root: Path):
    arch = root / "_archive"
    if APPLY:
        arch.mkdir(parents=True, exist_ok=True)
    man = arch / "ARCHIVE_MANIFEST.tsv"
    undo = arch / "UNDO.sh"
    # append-safe: merge with any existing manifest so re-runs stay cumulative/undoable
    existing: list[tuple[str, str]] = []
    if man.exists():
        for line in man.read_text(encoding="utf-8").splitlines():
            if not line.strip() or line.startswith("dst\t"):
                continue
            if "\t" in line:
                d, s = line.split("\t", 1)
                existing.append((d, s))
    combined = existing + manifest
    if APPLY:
        with man.open("w", encoding="utf-8") as f:
            f.write("dst\tsrc\n")
            for dst, src in combined:
                f.write(f"{dst}\t{src}\n")
        lines = ["#!/bin/bash", "# Reverse every move in ARCHIVE_MANIFEST.tsv.",
                 "# Usage: bash UNDO.sh [--dry-run]", 'DRY=0; [ "$1" = "--dry-run" ] && DRY=1',
                 'cd "$(dirname "$0")"']
        for dst, src in reversed(combined):
            d = dst.replace("\\", "/"); s = src.replace("\\", "/")
            lines.append(f'if [ -e "{d}" ]; then echo "{d} -> {s}"; [ $DRY -eq 0 ] && mkdir -p "$(dirname "{s}")" && mv "{d}" "{s}"; fi')
        undo.write_text("\n".join(lines) + "\n", encoding="utf-8")
        (arch / "SKIPPED_REFERENCED.txt").write_text("\n".join(skipped) + "\n", encoding="utf-8")
    print(f"\n{root.name}: {len(manifest)} new moves this run, {len(combined)} total in manifest, {len(skipped)} skipped")

# ---------------------------------------------------------------- run
CHILD_RESULT_MAP = {d: "results_superseded" for d in SUPERSEDED_RESULT_DIRS}

# CHILD level
manifest.clear(); skipped.clear()
process_level(CHILD, PROTECTED_CHILD, PROTECTED_CHILD_DIRS, cat_child_file, extra_dirs=CHILD_RESULT_MAP)
delete_pycache(CHILD)
write_undo_and_manifest(CHILD)

# PARENT level (files only; subrepos untouched)
manifest.clear(); skipped.clear()
process_level(PARENT, PROTECTED_PARENT, PROTECTED_PARENT_DIRS, cat_parent_file, extra_dirs=None)
delete_pycache(PARENT, skip_top_dirs=PROTECTED_PARENT_DIRS)  # don't touch subrepos / child (done separately)
write_undo_and_manifest(PARENT)

print("\nDRY-RUN complete. Re-run with --apply to execute." if not APPLY else "\nAPPLY complete.")
