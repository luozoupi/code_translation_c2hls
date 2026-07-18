"""Emit the canonical schema-1.0 JSONL (schema_records.jsonl) across the
enhanced-framework arms for BOTH models — Opus 4.8 (6 arms) and Sonnet 4.6
(Core-3: one-shot + curated/all-positive multistep) — merged into one file,
using the SAME per-cell emitter as analysis/_emit_schema_records.py
(candidate hls_synth/sw_run/rtl_sim + gold). GPT-5.5 arms fold in the same way
once that campaign completes (add its dirs to ARMS).

Each arm dir is homogeneous (one model/mode/skills), so instead of the
_iter_cells() convention we drive _emit_for_cell() directly with an explicit
(model, mode, skills, setup_label) per arm — this lets us:
  * tag the one-shot arm as mode="oneshot" (its cell subdir says "flash")
  * give the two Opus PRE-FIX skill-less flash draws distinct setup labels
    (skilless_A / skilless_B) so their origin_versions don't collide
  * fold Opus symm in from the *_symmrerun_OPUS dirs (base symm cell failed the
    1h-cosim cap; the rerun with C2HLS_SKIP_BASELINE_COSIM is the good one).
    Sonnet symm was handled in-arm (1h cap), so no rerun dir there.

origin_version = enh__<model>__<mode>__<skill_setup> distinguishes every arm and
model. Gold (hlsfactory_benchmark) records are deduped ONCE across ALL arms/models
via a shared emitted_gold_keys set. Passes validate_schema.py /
scripts/validate_jsonl_semantics.py.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
if not (HERE / "c2hls.py").exists():
    HERE = HERE.parent   # running from analysis_tools/ -> project root is the parent
# Import the CANONICAL emitter explicitly by path (a wrong same-named module may
# sit in cwd; importlib from the analysis/ path avoids shadowing).
_spec = importlib.util.spec_from_file_location(
    "canon_emit", HERE / "analysis" / "_emit_schema_records.py")
E = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(E)

# Mark provenance: these are the ENHANCED framework runs, distinct from the prior
# 630ce11 flash campaign already living in schemas/.
E.ORCHESTRATOR_GIT_COMMIT = "enh"

OPUS = "claude-opus-4-8"
SONNET = "claude-sonnet-4-6"
OUT = HERE / "schema_records.jsonl"

# (dir, model, mode, skills, setup_label, symm_override_dir_or_None)
ARMS = [
    # ---- Opus 4.8 (6 arms, complete) ----
    ("results_matrix_u280_ENH_oneshot_OPUS",                  OPUS, "oneshot",   "off", "",             None),
    ("results_matrix_u280_ENH_curated_flash_OPUS",            OPUS, "flash",     "on",  "skilless_A",   None),
    ("results_matrix_u280_ENH_allpositive_flash_OPUS",        OPUS, "flash",     "on",  "skilless_B",   None),
    ("results_matrix_u280_ENH_allpositive_flash_SKILLS_OPUS", OPUS, "flash",     "on",  "all_positive", None),
    ("results_matrix_u280_ENH_curated_multistep_skills_OPUS", OPUS, "multistep", "on",  "curated",
     "results_matrix_u280_ENH_curated_multistep_skills_symmrerun_OPUS"),
    ("results_matrix_u280_ENH_allpositive_multistep_skills_OPUS", OPUS, "multistep", "on", "all_positive",
     "results_matrix_u280_ENH_allpositive_multistep_skills_symmrerun_OPUS"),
    # ---- Sonnet 4.6 (Core-3, complete) ----
    ("results_matrix_u280_ENH_oneshot_SONNET",                    SONNET, "oneshot",   "off", "",             None),
    ("results_matrix_u280_ENH_curated_multistep_skills_SONNET",   SONNET, "multistep", "on",  "curated",      None),
    ("results_matrix_u280_ENH_allpositive_multistep_skills_SONNET", SONNET, "multistep", "on", "all_positive", None),
]


def wallclock_by_bench(arm_dir: Path) -> dict[str, float]:
    """bench -> wallclock_s from matrix.json (each arm dir is single-cell-per-bench)."""
    out: dict[str, float] = {}
    mj = arm_dir / "matrix.json"
    if not mj.exists():
        return out
    try:
        cells = json.loads(mj.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return out
    for c in cells if isinstance(cells, list) else []:
        b, wc = c.get("bench"), c.get("wallclock_s")
        if b and wc is not None:
            try:
                out[b] = float(wc)
            except (TypeError, ValueError):
                pass
    return out


def only_cell(bench_dir: Path, model: str) -> Path | None:
    """The single cell subdir for this bench, e.g. opus__flash__skills / sonnet__multistep__skills."""
    pref = E._model_short(model) + "__"
    subs = sorted(p for p in bench_dir.iterdir() if p.is_dir() and p.name.startswith(pref))
    return subs[0] if subs else None


def emit_cell(bench: str, cell_dir: Path, model: str, mode: str, skills: str, label: str,
              gold_keys: set, wc_map: dict) -> list[dict]:
    return E._emit_for_cell(
        bench, cell_dir, model, mode, skills, gold_keys,
        wallclock_s=wc_map.get(bench), setup_label=label,
    )


def main() -> int:
    records: list[dict] = []
    gold_keys: set = set()   # shared across ALL arms/models -> gold emitted once total
    for (dirn, model, mode, skills, label, symm_dir) in ARMS:
        arm = HERE / dirn
        if not arm.is_dir():
            print(f"skip missing arm dir: {dirn}")
            continue
        wc = wallclock_by_bench(arm)
        n0 = len(records)
        for bench_dir in sorted(p for p in arm.iterdir() if p.is_dir() and p.name.startswith("hlsfactory_")):
            bench = bench_dir.name
            # Opus multistep symm comes from the rerun dir (base symm was the failed run)
            if mode == "multistep" and bench == "hlsfactory_symm" and symm_dir:
                continue
            cell = only_cell(bench_dir, model)
            if cell is None:
                continue
            records.extend(emit_cell(bench, cell, model, mode, skills, label, gold_keys, wc))
        # symm from the rerun dir for arms that have one (Opus multistep)
        if mode == "multistep" and symm_dir:
            sarm = HERE / symm_dir
            sbench_dir = sarm / "hlsfactory_symm"
            if sbench_dir.is_dir():
                cell = only_cell(sbench_dir, model)
                if cell is not None:
                    swc = wallclock_by_bench(sarm)
                    records.extend(emit_cell("hlsfactory_symm", cell, model, mode, skills, label, gold_keys, swc))
        print(f"  {dirn}: +{len(records) - n0} records")

    with OUT.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, separators=(", ", ": ")) + "\n")

    # summary
    from collections import Counter
    by_type = Counter(r["report_type"] for r in records)
    by_ver = Counter((r["implementation"].get("origin_version"),
                      r["implementation"].get("origin")) for r in records)
    print(f"\nWrote {len(records)} records -> {OUT.name}")
    print("by report_type:", dict(by_type))
    print("\norigin_version                          origin                 n")
    print("-" * 70)
    for (ver, origin), n in sorted(by_ver.items(), key=lambda x: (str(x[0][0]), str(x[0][1]))):
        print(f"  {str(ver):38} {str(origin):22} {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
