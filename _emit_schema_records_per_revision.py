"""(B)-flavor emitter — produces per-revision variants instead of just "final".

Variants now denote the SEQUENCE OF CODE REVISIONS the orchestrator went
through during a single run (the collaborator's intended semantics):

  MULTISTEP cells: one variant per optimization step
    variant.index = step_i
    variant.name  = f"step_{i}_{step_name}"   e.g. step_0_tiling, step_1_pipeline,
                                                   step_2_unroll, step_3_doublebuffer,
                                                   step_4_coalescing

  FLASH cells: Phase B + each accepted quality-repair turn
    variant.index = 0,  variant.name = "phase_b_initial"   (always present)
    variant.index = 1+, variant.name = f"quality_repair_{turn}"  (one per accepted repair)
    If no repair fired, only variant 0 is emitted; we relabel it "final" so
    flash-no-repair cells still resolve to a single "final" revision.

Baseline (Direct Vitis) records remain {index: 0, name: "baseline"} —
one per (bench, report_type), deduped.

For each variant, emits:
  - hls_synth (csynth report)
  - sw_run    (csim status)
  - rtl_sim   (cosim status + cycles when passed)

Usage:
  python3 _emit_schema_records_per_revision.py <results_dir> <out_jsonl> [<setup_label>]
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _emit_schema_records import (  # noqa
    ORCHESTRATOR_GIT_COMMIT,
    SUITE,
    _compose_origin_version,
    _hls_synth_record,
    _sw_run_record,
    _rtl_sim_record,
    _status_from_flags,
    _iter_cells,
    _load_wallclock_map,
    MODEL_DIR_TO_ID,
)


def _emit_step_variants_multistep(
    bench: str, cell_dir: Path, model: str, skills: str,
    setup_label: str, emitted_gold_keys: set,
    wallclock_s: float | None,
) -> list[dict]:
    """For multistep, emit one variant per step (5 typically). Variant index =
    the step's position in summary.steps; variant.name = f"step_{i}_{name}".
    Baseline gold is emitted at most once across all cells (shared dedup set).
    """
    rj = cell_dir / f"{bench}_multistep_results.json"
    if not rj.exists():
        return []
    try:
        data = json.loads(rj.read_text())
    except (OSError, json.JSONDecodeError):
        return []

    mode = "multistep"
    steps = data.get("steps") or []
    cand_origin_version = _compose_origin_version(mode, skills, setup_label)

    # Shared A/B knob block for all of this cell's variants
    base_ab_meta = {
        "model": model,
        "mode": mode,
        "skills": skills,
    }
    skills_log = data.get("skills_log") or {}
    skills_provenance = "native"
    if not skills_log:
        sidecar = cell_dir / f"{bench}_skills_log.backfilled.json"
        if sidecar.exists():
            try:
                skills_log = json.loads(sidecar.read_text()) or {}
                skills_provenance = skills_log.get("provenance") or "backfilled"
            except (OSError, json.JSONDecodeError):
                skills_log = {}
    skills_applied = skills_log.get("unique_skill_ids") or []
    if skills_applied:
        base_ab_meta["skills_applied"] = list(skills_applied)
        base_ab_meta["skills_provenance"] = skills_provenance
    cfg_sha1 = skills_log.get("skills_config_sha1")
    if cfg_sha1:
        base_ab_meta["skills_config_sha1"] = cfg_sha1
    base_ab_meta["skills_variant"] = (
        "none" if skills == "off"
        else ((setup_label or "base_skills").replace("_skills", "") or "base")
    )

    records: list[dict] = []
    part = "xcu280-fsvh2892-2L-e"
    clock_ns = 3.33

    # Per-step candidate records
    for i, step in enumerate(steps):
        if not isinstance(step, dict):
            continue
        step_name = step.get("step_name") or step.get("name") or f"step{i}"
        synth = step.get("report") or {}
        # Use step-local part/clock if present, else fall back to defaults
        step_part = synth.get("part") or part
        step_clock = synth.get("requested_clock_period_ns") or clock_ns

        variant = {"index": i, "name": f"step_{i}_{step_name}"}
        # Carry the same `phase` on all 3 report_types so downstream pivots
        # by origin_meta.phase work identically for hls_synth/sw_run/rtl_sim.
        # (Earlier code only set phase on hls_synth — a symmetry gap.)
        phase_str = f"multistep_step_{i}_{step_name}"
        cand_meta_hls = dict(base_ab_meta, phase=phase_str,
                             multistep_step=step_name, multistep_step_index=i)
        cand_meta_run = dict(base_ab_meta, phase=phase_str,
                             multistep_step=step_name, multistep_step_index=i)

        if synth:
            cand_status = "pass" if synth.get("latency_cycles") is not None else "fail"
            records.append(_hls_synth_record(
                bench, "c2hls_orchestrator", cand_meta_hls, variant, synth,
                cand_status, step_part, step_clock,
                origin_version=cand_origin_version,
                runtime_seconds=None,   # step-level wallclock not tracked
            ))

        cand_csim = step.get("csim") or {}
        csim_status = _status_from_flags(cand_csim)
        if csim_status is not None:
            records.append(_sw_run_record(
                bench, "c2hls_orchestrator", cand_meta_run, variant, csim_status, step_part,
                origin_version=cand_origin_version,
            ))

        cand_cosim = step.get("cosim") or {}
        cosim_status = _status_from_flags(cand_cosim)
        if cosim_status is not None:
            records.append(_rtl_sim_record(
                bench, "c2hls_orchestrator", cand_meta_run, variant, cosim_status, step_part,
                cand_cosim.get("measured"), step_clock,
                origin_version=cand_origin_version,
            ))

    # Gold baseline (deduped via emitted_gold_keys)
    ref = data.get("reference_validation") or {}
    gold_report = ref.get("report") or {}
    gold_variant = {"index": 0, "name": "baseline"}
    if gold_report:
        key = (bench, "hls_synth")
        if key not in emitted_gold_keys:
            emitted_gold_keys.add(key)
            gold_status = "pass" if gold_report.get("latency_cycles") is not None else "fail"
            records.append(_hls_synth_record(
                bench, "hlsfactory_benchmark", None, gold_variant, gold_report,
                gold_status, part, clock_ns,
            ))
    gold_csim_status = _status_from_flags(ref.get("csim"))
    if gold_csim_status is not None:
        key = (bench, "sw_run")
        if key not in emitted_gold_keys:
            emitted_gold_keys.add(key)
            records.append(_sw_run_record(
                bench, "hlsfactory_benchmark", None, gold_variant, gold_csim_status, part,
            ))
    gold_cosim = ref.get("cosim") or {}
    gold_cosim_status = _status_from_flags(gold_cosim)
    if gold_cosim_status is not None:
        key = (bench, "rtl_sim")
        if key not in emitted_gold_keys:
            emitted_gold_keys.add(key)
            records.append(_rtl_sim_record(
                bench, "hlsfactory_benchmark", None, gold_variant, gold_cosim_status, part,
                gold_cosim.get("measured"), clock_ns,
            ))

    return records


def _emit_phase_b_plus_final_flash(
    bench: str, cell_dir: Path, model: str, skills: str,
    setup_label: str, emitted_gold_keys: set,
    wallclock_s: float | None,
) -> list[dict]:
    """For flash, emit Phase B initial (variant 0) + final after quality_repair
    (variant 1) if a repair was accepted; otherwise just variant 0 named 'final'.
    Phase B's csim/cosim may be None (e.g. if cosim timed out before
    quality_repair pivoted). In that case we still emit an hls_synth record
    (Phase B synth fields ARE present in optimization_history[0].report) and
    skip the missing sw_run/rtl_sim records.
    """
    rj = cell_dir / f"{bench}_results.json"
    if not rj.exists():
        return []
    try:
        data = json.loads(rj.read_text())
    except (OSError, json.JSONDecodeError):
        return []

    mode = "flash"
    cand_origin_version = _compose_origin_version(mode, skills, setup_label)

    base_ab_meta = {"model": model, "mode": mode, "skills": skills}
    skills_log = data.get("skills_log") or {}
    skills_provenance = "native"
    if not skills_log:
        sidecar = cell_dir / f"{bench}_skills_log.backfilled.json"
        if sidecar.exists():
            try:
                skills_log = json.loads(sidecar.read_text()) or {}
                skills_provenance = skills_log.get("provenance") or "backfilled"
            except (OSError, json.JSONDecodeError):
                skills_log = {}
    skills_applied = skills_log.get("unique_skill_ids") or []
    if skills_applied:
        base_ab_meta["skills_applied"] = list(skills_applied)
        base_ab_meta["skills_provenance"] = skills_provenance
    cfg_sha1 = skills_log.get("skills_config_sha1")
    if cfg_sha1:
        base_ab_meta["skills_config_sha1"] = cfg_sha1
    base_ab_meta["skills_variant"] = (
        "none" if skills == "off"
        else ((setup_label or "base_skills").replace("_skills", "") or "base")
    )

    qr = data.get("quality_repair") or {}
    qr_applied = bool(qr.get("applied"))
    oh = data.get("optimization_history") or []
    phase_b_entry = next((e for e in oh if isinstance(e, dict) and e.get("phase") == "B"), None)

    final_synth = data.get("synth_report") or {}
    part = final_synth.get("part") or "xcu280-fsvh2892-2L-e"
    clock_ns = final_synth.get("requested_clock_period_ns") or 3.33

    records: list[dict] = []

    # ---- Phase B (always variant 0) ----
    if qr_applied:
        pb_variant = {"index": 0, "name": "phase_b_initial"}
        pb_meta_hls = dict(base_ab_meta, phase="phase_b_initial",
                           accepted_into_final=False,
                           repair_pivoted_away=True)
        pb_meta_run = dict(base_ab_meta, phase="phase_b_initial",
                           accepted_into_final=False, repair_pivoted_away=True)
    else:
        # No repair fired — Phase B IS the final. Tag it as "final" for
        # downstream tools that key on variant.name=="final".
        pb_variant = {"index": 0, "name": "final"}
        pb_meta_hls = dict(base_ab_meta, phase="phase_b_final")
        pb_meta_run = dict(base_ab_meta, phase="phase_b_final")

    # Phase B's synth_report comes from optimization_history[0].report when
    # repair pivoted (final synth_report belongs to the repair). When no
    # repair, top-level synth_report IS Phase B.
    if qr_applied and phase_b_entry:
        pb_synth = phase_b_entry.get("report") or {}
        # Phase B csim/cosim aren't preserved separately; if they were
        # passed before repair, they'd be in optimization_history too —
        # but the orchestrator only logs the final state. Skip those records
        # rather than fabricating data.
        pb_csim = None
        pb_cosim = None
    else:
        pb_synth = final_synth
        pb_csim = data.get("csim") or {}
        pb_cosim = data.get("cosim") or {}

    if pb_synth:
        pb_status = "pass" if pb_synth.get("latency_cycles") is not None else "fail"
        records.append(_hls_synth_record(
            bench, "c2hls_orchestrator", pb_meta_hls, pb_variant, pb_synth,
            pb_status, part, clock_ns,
            origin_version=cand_origin_version,
            runtime_seconds=wallclock_s,
        ))
    if pb_csim is not None:
        csim_status = _status_from_flags(pb_csim)
        if csim_status is not None:
            records.append(_sw_run_record(
                bench, "c2hls_orchestrator", pb_meta_run, pb_variant, csim_status, part,
                origin_version=cand_origin_version,
                runtime_seconds=wallclock_s,
            ))
    if pb_cosim is not None:
        cosim_status = _status_from_flags(pb_cosim)
        if cosim_status is not None:
            records.append(_rtl_sim_record(
                bench, "c2hls_orchestrator", pb_meta_run, pb_variant, cosim_status, part,
                pb_cosim.get("measured"), clock_ns,
                origin_version=cand_origin_version,
                runtime_seconds=wallclock_s,
            ))

    # ---- Quality-repair final (variant 1) ----
    if qr_applied:
        final_variant = {"index": 1, "name": "final"}
        attempts_n = len(qr.get("attempts") or [])
        f_meta_hls = dict(base_ab_meta, phase="quality_repair_final",
                          repair_attempts=attempts_n,
                          final_score=qr.get("final_score"))
        f_meta_run = dict(base_ab_meta, phase="quality_repair_final",
                          repair_attempts=attempts_n)
        if final_synth:
            f_status = "pass" if final_synth.get("latency_cycles") is not None else "fail"
            records.append(_hls_synth_record(
                bench, "c2hls_orchestrator", f_meta_hls, final_variant, final_synth,
                f_status, part, clock_ns,
                origin_version=cand_origin_version,
                runtime_seconds=wallclock_s,
            ))
        final_csim = data.get("csim") or {}
        csim_status = _status_from_flags(final_csim)
        if csim_status is not None:
            records.append(_sw_run_record(
                bench, "c2hls_orchestrator", f_meta_run, final_variant, csim_status, part,
                origin_version=cand_origin_version,
                runtime_seconds=wallclock_s,
            ))
        final_cosim = data.get("cosim") or {}
        cosim_status = _status_from_flags(final_cosim)
        if cosim_status is not None:
            records.append(_rtl_sim_record(
                bench, "c2hls_orchestrator", f_meta_run, final_variant, cosim_status, part,
                final_cosim.get("measured"), clock_ns,
                origin_version=cand_origin_version,
                runtime_seconds=wallclock_s,
            ))

    # ---- Gold baseline (deduped) ----
    ref = data.get("reference_validation") or {}
    gold_report = ref.get("report") or {}
    gold_variant = {"index": 0, "name": "baseline"}
    if gold_report:
        key = (bench, "hls_synth")
        if key not in emitted_gold_keys:
            emitted_gold_keys.add(key)
            gold_status = "pass" if gold_report.get("latency_cycles") is not None else "fail"
            records.append(_hls_synth_record(
                bench, "hlsfactory_benchmark", None, gold_variant, gold_report,
                gold_status, part, clock_ns,
            ))
    gold_csim_status = _status_from_flags(ref.get("csim"))
    if gold_csim_status is not None:
        key = (bench, "sw_run")
        if key not in emitted_gold_keys:
            emitted_gold_keys.add(key)
            records.append(_sw_run_record(
                bench, "hlsfactory_benchmark", None, gold_variant, gold_csim_status, part,
            ))
    gold_cosim = ref.get("cosim") or {}
    gold_cosim_status = _status_from_flags(gold_cosim)
    if gold_cosim_status is not None:
        key = (bench, "rtl_sim")
        if key not in emitted_gold_keys:
            emitted_gold_keys.add(key)
            records.append(_rtl_sim_record(
                bench, "hlsfactory_benchmark", None, gold_variant, gold_cosim_status, part,
                gold_cosim.get("measured"), clock_ns,
            ))

    return records


def main() -> int:
    if len(sys.argv) not in (3, 4):
        print(__doc__)
        return 2
    root = Path(sys.argv[1])
    out_path = Path(sys.argv[2])
    setup_label = sys.argv[3] if len(sys.argv) == 4 else os.environ.get(
        "C2HLS_SCHEMA_SETUP_LABEL", "")
    setup_label = (setup_label or "").strip()
    if not root.exists():
        print(f"no such dir: {root}")
        return 1
    wallclock_map = _load_wallclock_map(root)
    records: list[dict] = []
    emitted_gold_keys: set[tuple[str, str]] = set()
    for bench, cell_dir, model, mode, skills in _iter_cells(root):
        wc = wallclock_map.get((bench, model, mode, skills))
        if mode == "multistep":
            records.extend(_emit_step_variants_multistep(
                bench, cell_dir, model, skills, setup_label, emitted_gold_keys, wc))
        else:
            records.extend(_emit_phase_b_plus_final_flash(
                bench, cell_dir, model, skills, setup_label, emitted_gold_keys, wc))

    with out_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, separators=(", ", ": ")) + "\n")

    from collections import Counter
    print(f"Wrote {len(records)} records to {out_path}")
    rt_counts = Counter(r["report_type"] for r in records)
    for k, v in sorted(rt_counts.items()):
        print(f"  {k}: {v}")
    # Per-(origin_version, variant.name) variant count
    print("\nPer-origin_version variant.name distribution:")
    vc = Counter()
    for r in records:
        impl = r.get("implementation") or {}
        ver = impl.get("origin_version") or "(baseline)"
        v = (impl.get("variant") or {}).get("name")
        vc[(ver, v)] += 1
    for (ver, v), n in sorted(vc.items()):
        print(f"  {ver:<48} {v!r:<28} {n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
