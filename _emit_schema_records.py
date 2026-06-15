"""Emit JSONL records matching the shape of `schema_records.jsonl` for an
orchestrator sweep. For each cell that produced a `*_results.json`, we emit:

  1. an `hls_synth` record for the c2hls_orchestrator candidate (Phase B / quality)
  2. an `sw_run`    record for that candidate's csim (if it ran)
  3. an `rtl_sim`   record for that candidate's cosim (if it ran), populated
     with kernel_runtime_cycles / kernel_clock_freq_mhz pulled from the cosim
     report parser
  4. an `hls_synth` record for the gold baseline (reference_validation.report)
  5. an `sw_run`    record for the gold csim (if it ran)
  6. an `rtl_sim`   record for the gold cosim (if it ran)

Schema-contract addressing (2026-06-15 fix per collaborator feedback):
  - `implementation.origin_version` distinguishes SEPARATE ORCHESTRATOR RUNS
    (different mode + skill setup = different version). Format:
        <git_sha>__<mode>__<skill_setup>
    e.g.  630ce11__flash__base_skills
          630ce11__flash__no_skills
          630ce11__flash__extended_skills
          630ce11__multistep__base_skills
  - `implementation.variant` denotes the SEQUENCE OF CODE REVISIONS WITHIN
    one orchestrator run as it slowly optimizes. For now we emit only the
    final accepted revision per cell, so variant = {"index": 0, "name": "final"}.
    Per-step / per-quality-repair variants are emitted by the (B)-flavor
    extension (see _emit_schema_records_per_step.py).
  - `implementation.origin_meta` still carries the per-cell A/B tuple
    (model, mode, skills, skills_applied, skills_provenance, skills_config_sha1)
    for downstream filtering.

The third positional CLI arg <setup_label> picks the skill-setup label baked
into origin_version for skills=on cells. For skills=off cells the label is
always "no_skills" regardless of <setup_label>. Common values:
    base_skills        — the curated 55-skill skills.json
    extended_skills    — base + skills_extension.json (3 hard_guards)
    no_skills          — no skill block (for sweeps where every cell is off)

Usage:
  python3 _emit_schema_records.py <results_dir> <out_jsonl> [<setup_label>]
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

SCHEMA_VERSION = "1.0"
SUITE = "hlsfactory_polybench_float_small"
VITIS_VERSION = "2023.2"
# Git commit hash of the c2hls_orchestrator at the time of this matrix run.
# Used to populate implementation.origin_version for c2hls_orchestrator records.
ORCHESTRATOR_GIT_COMMIT = "630ce11"

# Per-device AvailableResources (BRAM_18K block count, DSP, FF, LUT, URAM).
# Values match Vitis HLS report's `AvailableResources` block convention used
# by the canonical schema_records.jsonl. Numeric strings to mirror canonical.
DEVICE_AVAILABLE_RESOURCES = {
    # Alveo U280
    "xcu280-fsvh2892-2L-e": {
        "BRAM_18K": "4032",
        "DSP": "9024",
        "FF": "2607360",
        "LUT": "1303680",
        "URAM": "960",
    },
    # Alveo U50 (kept so old runs against U50 still emit populated caps)
    "xcu50-fsvh2104-2-e": {
        "BRAM_18K": "2688",
        "DSP": "5952",
        "FF": "1743360",
        "LUT": "871680",
        "URAM": "640",
    },
}


# Per-device ProductFamily strings as written by Vitis HLS into the
# UserAssignments block of csynth.xml. Verified from a real 2023.2 csynth.xml
# for xcu280: ProductFamily="virtexuplusHBM" (mixed case as Vitis emits it).
DEVICE_PRODUCT_FAMILY = {
    "xcu280-fsvh2892-2L-e": "virtexuplusHBM",
    "xcu50-fsvh2104-2-e": "virtexuplusHBM",
}


def _available_resources(part: str | None) -> dict:
    caps = DEVICE_AVAILABLE_RESOURCES.get(part or "")
    if caps is None:
        return {"BRAM_18K": None, "DSP": None, "FF": None, "LUT": None, "URAM": None}
    return dict(caps)


def _product_family(part: str | None) -> str | None:
    return DEVICE_PRODUCT_FAMILY.get(part or "")


def _clock_uncertainty(clock_ns: float | int | None) -> str | None:
    """Vitis HLS default ClockUncertainty = 27% of TargetClockPeriod, formatted
    to 2 decimal places. Verified from a real U280 csynth.xml (clock=3.33 ->
    "0.90"). Returns the same string convention Vitis writes into the XML."""
    if clock_ns is None:
        return None
    try:
        c = float(clock_ns)
    except (TypeError, ValueError):
        return None
    return f"{c * 0.27:.2f}"


def _f(x):
    if x is None or x == "":
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _i(x):
    if x is None or x == "":
        return None
    try:
        return int(x)
    except (TypeError, ValueError):
        try:
            return int(float(x))
        except (TypeError, ValueError):
            return None


def _str_or_none(x):
    return None if x is None else str(x)


def _run_block(target: str, part: str, runtime_seconds=None) -> dict:
    return {
        "target": target,
        "device": part,
        "vitis_version": VITIS_VERSION,
        "runtime_seconds": runtime_seconds,
    }


def _problem_block(bench: str) -> dict:
    # Drop the "hlsfactory_" prefix for group_path so it reads like the
    # rodinia_hls / ml4accel records in the schema sample.
    name = bench[len("hlsfactory_"):] if bench.startswith("hlsfactory_") else bench
    return {"suite": SUITE, "group_path": [name]}


def _user_assignments(part: str, clock_ns: float | int | None) -> dict:
    return {
        "unit": "ns",
        "ProductFamily": _product_family(part),
        "Part": part,
        "TopModelName": "workload",
        "TargetClockPeriod": _str_or_none(clock_ns),
        "ClockUncertainty": _clock_uncertainty(clock_ns),
        "FlowTarget": "vitis",
    }


def _format_realtime_latency(lat_ns: float | None) -> str | None:
    """Auto-scale latency in ns to us / ms / s the way Vitis HLS does in
    the report XML (e.g. canonical schema_records.jsonl).
      lat_ns <  1_000_000        -> "X.XXX us"   (us = ns/1e3)
      lat_ns <  1_000_000_000    -> "X.XXX ms"   (ms = ns/1e6)
      otherwise                  -> "X.XXX s"    (s  = ns/1e9)
    """
    if lat_ns is None:
        return None
    if lat_ns < 1_000_000:
        return f"{lat_ns / 1_000.0:.3f} us"
    if lat_ns < 1_000_000_000:
        return f"{lat_ns / 1_000_000.0:.3f} ms"
    return f"{lat_ns / 1_000_000_000.0:.3f} s"


def _performance_estimates(report: dict) -> dict:
    lat = _i(report.get("latency_cycles"))
    lat_ns = _f(report.get("latency_ns"))
    interval = _i(report.get("interval"))
    est_clk = report.get("estimated_clock_period_ns")
    real_us = _format_realtime_latency(lat_ns)
    return {
        "SummaryOfTimingAnalysis": {
            "unit": "ns",
            "EstimatedClockPeriod": _str_or_none(est_clk),
        },
        "SummaryOfOverallLatency": {
            "unit": "clock cycles",
            "Best-caseLatency": _str_or_none(lat),
            "Average-caseLatency": _str_or_none(lat),
            "Worst-caseLatency": _str_or_none(lat),
            "Best-caseRealTimeLatency": real_us,
            "Average-caseRealTimeLatency": real_us,
            "Worst-caseRealTimeLatency": real_us,
            "Interval-min": _str_or_none(interval),
            "Interval-max": _str_or_none(interval),
        },
    }


def _area_estimates(report: dict, part: str | None = None) -> dict:
    return {
        "Resources": {
            "BRAM_18K": _str_or_none(report.get("bram")),
            "DSP": _str_or_none(report.get("dsp")),
            "FF": _str_or_none(report.get("ff")),
            "LUT": _str_or_none(report.get("lut")),
            "URAM": _str_or_none(report.get("uram")),
        },
        "AvailableResources": _available_resources(part),
    }


def _hls_synth_record(bench: str, origin: str, origin_meta: dict | None,
                      variant: dict, report: dict, status: str, part: str,
                      clock_ns: float | int | None,
                      origin_version: str | None = None,
                      runtime_seconds: float | None = None) -> dict:
    # Contract V6: on fail/timeout the four subsections must be present but
    # null, instead of populated with all-None inner fields.
    if status == "pass":
        synth_payload = {
            "status": status,
            "ReportVersion": {"Version": VITIS_VERSION},
            "UserAssignments": _user_assignments(part, clock_ns),
            "PerformanceEstimates": _performance_estimates(report),
            "AreaEstimates": _area_estimates(report, part),
        }
    else:
        synth_payload = {
            "status": status,
            "ReportVersion": None,
            "UserAssignments": None,
            "PerformanceEstimates": None,
            "AreaEstimates": None,
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "report_type": "hls_synth",
        "run": _run_block("vitis.csynth", part, runtime_seconds=runtime_seconds),
        "problem": _problem_block(bench),
        "implementation": {
            "origin": origin,
            "origin_version": origin_version,
            "origin_meta": origin_meta,
            "variant": variant,
        },
        "hls_synth": synth_payload,
    }


def _sw_run_record(bench: str, origin: str, origin_meta: dict | None,
                   variant: dict, status: str, part: str,
                   origin_version: str | None = None,
                   runtime_seconds: float | None = None) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "report_type": "sw_run",
        "run": _run_block("vitis.csim", part, runtime_seconds=runtime_seconds),
        "problem": _problem_block(bench),
        "implementation": {
            "origin": origin,
            "origin_version": origin_version,
            "origin_meta": origin_meta,
            "variant": variant,
        },
        "sw_run": {"status": status},
    }


def _rtl_sim_record(bench: str, origin: str, origin_meta: dict | None,
                    variant: dict, status: str, part: str,
                    measured: dict | None, clock_ns: float | int | None,
                    origin_version: str | None = None,
                    runtime_seconds: float | None = None,
                    target: str = "vitis.cosim") -> dict:
    rtl = {
        "status": status,
        "kernel_runtime_cycles": None,
        "kernel_runtime_us": None,
        "kernel_clock_freq_mhz": None,
    }
    if measured:
        cycles = _i(measured.get("latency_cycles_avg") or measured.get("latency_cycles_min"))
        rtl["kernel_runtime_cycles"] = cycles
        # Only compute derived runtime/freq when we have a real cycle
        # measurement; otherwise a failed/timed-out cosim would leak a
        # frequency value derived from the requested clock period.
        if cycles is not None and clock_ns:
            rtl["kernel_runtime_us"] = round(cycles * float(clock_ns) / 1000.0, 3)
            rtl["kernel_clock_freq_mhz"] = round(1000.0 / float(clock_ns), 3)
    # Contract V1: for vitis.cosim target, kernel_runtime_us and
    # kernel_clock_freq_mhz must be null regardless of what was measured.
    if target == "vitis.cosim":
        rtl["kernel_runtime_us"] = None
        rtl["kernel_clock_freq_mhz"] = None
    return {
        "schema_version": SCHEMA_VERSION,
        "report_type": "rtl_sim",
        "run": _run_block(target, part, runtime_seconds=runtime_seconds),
        "problem": _problem_block(bench),
        "implementation": {
            "origin": origin,
            "origin_version": origin_version,
            "origin_meta": origin_meta,
            "variant": variant,
        },
        "rtl_sim": rtl,
    }


def _status_from_flags(block: dict | None) -> str | None:
    if not block:
        return None
    if block.get("passed") is True:
        return "pass"
    # Treat explicit timeout signal from the runner as the contract's "timeout"
    # status. Cosim/csim blocks may surface a textual `status` field, a boolean
    # `timed_out`, or both.
    status_text = (block.get("status") or "").lower()
    if status_text == "timeout" or block.get("timed_out") is True:
        return "timeout"
    ran = block.get("ran")
    if ran is False or ran is None:
        return None
    return "fail"


def _compose_origin_version(mode: str, skills: str, setup_label: str) -> str:
    """Build origin_version = <git_sha>__<mode>__<skill_setup>.

    For skills=off, the skill part is always 'no_skills' regardless of the
    sweep's setup_label (skills_off cells don't load the configured skill
    package; they don't load anything).
    """
    if skills == "off":
        skill_part = "no_skills"
    else:
        skill_part = (setup_label or "base_skills").strip()
    return f"{ORCHESTRATOR_GIT_COMMIT}__{mode}__{skill_part}"


def _emit_for_cell(bench: str, cell_dir: Path, model: str, mode: str,
                   skills: str,
                   emitted_gold_keys: set[tuple[str, str]] | None = None,
                   wallclock_s: float | None = None,
                   setup_label: str = "") -> list[dict]:
    if mode == "multistep":
        rj = cell_dir / f"{bench}_multistep_results.json"
    else:
        rj = cell_dir / f"{bench}_results.json"
    if not rj.exists():
        return []
    try:
        data = json.loads(rj.read_text())
    except (OSError, json.JSONDecodeError):
        return []

    # For multistep, the "current" candidate metrics live in final_report +
    # the cosim attached to the last successful optimization step. For flash,
    # they live in synth_report + top-level cosim.
    if mode == "multistep":
        synth = data.get("final_report") or {}
        steps = data.get("steps") or []
        chosen_step = None
        for s in steps:
            if s.get("success") and (s.get("cosim") or {}).get("ran") is not None:
                chosen_step = s
        cand_csim = (chosen_step or {}).get("csim") or data.get("baseline_csim")
        cand_cosim = (chosen_step or {}).get("cosim") or data.get("baseline_cosim")
    else:
        synth = data.get("synth_report") or {}
        cand_csim = data.get("csim")
        cand_cosim = data.get("cosim")
    part = synth.get("part") or "xcu280-fsvh2892-2L-e"
    clock_ns = synth.get("requested_clock_period_ns") or 3.33

    # Per user: KEEP A/B knobs (model/mode/skills) on all three record types
    # as an intentional schema extension. Only hls_synth.origin_meta carries
    # `phase`; sw_run/rtl_sim origin_meta is {model, mode, skills} only.
    # Gold-baseline records use the same A/B knob block (no `kind` field --
    # canonical never has one; gold is distinguished via implementation.origin
    # = "hlsfactory_benchmark" and variant.name = "baseline").
    ab_meta = {
        "model": model,
        "mode": mode,
        "skills": skills,
    }
    # Surface the in-orchestrator skills_log summary into origin_meta:
    # `skills_applied` = deduped list of skill ids actually rendered into a
    # prompt during this run, and `skills_config_sha1` for reproducibility.
    # Only added when populated, so old phase-7 cells without a skills_log
    # don't get bogus empty arrays. Falls back to a backfilled sidecar
    # (<bench>_skills_log.backfilled.json, written by _backfill_skills_log.py)
    # when the native field is absent — sidecar payloads carry a `provenance`
    # field that we surface so consumers know it's a replay vs a live record.
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
        ab_meta["skills_applied"] = list(skills_applied)
        ab_meta["skills_provenance"] = skills_provenance
    cfg_sha1 = skills_log.get("skills_config_sha1")
    if cfg_sha1:
        ab_meta["skills_config_sha1"] = cfg_sha1
    cand_meta_hls = dict(ab_meta, phase=data.get("phase"))
    cand_meta_run = dict(ab_meta)
    # GOLD_META: gold baseline is bench-source-determined; A/B knobs
    # (model/mode/skills) describe the orchestrator and have no meaning
    # for the gold record. Match canonical's rodinia/ml4accel baselines
    # which have origin_meta=null.
    gold_meta_hls = None
    gold_meta_run = None
    # Schema-correct distinction: skill setup goes into origin_version (one
    # value per orchestrator run), and variant denotes the code revision
    # within that run. Until per-step emission lands (B-flavor extension),
    # every cell emits exactly one accepted revision, named "final".
    cand_origin_version = _compose_origin_version(mode, skills, setup_label)
    cand_variant = {"index": 0, "name": "final"}
    gold_variant = {"index": 0, "name": "baseline"}

    # Surface the skill-setup tri-state into origin_meta so downstream
    # consumers can tuple-check (mode, skills_variant) without parsing the
    # origin_version string. Legacy `skills` field stays on/off for
    # backward compat with existing readers.
    if skills == "off":
        skills_variant_tag = "none"
    else:
        skills_variant_tag = (setup_label or "base_skills").strip()
        skills_variant_tag = skills_variant_tag.replace("_skills", "") or "base"
    ab_meta["skills_variant"] = skills_variant_tag

    # Gold dedup: gold synthesis/cosim is a deterministic function of the
    # benchmark source — emitting it once per (bench, report_type) across all
    # A/B cells is sufficient. If the caller didn't pass a shared set, fall
    # back to a local one (preserves the legacy per-cell behavior).
    if emitted_gold_keys is None:
        emitted_gold_keys = set()

    records: list[dict] = []

    # Candidate hls_synth (only if Phase B produced a report)
    if synth:
        cand_status = "pass" if synth.get("latency_cycles") is not None else "fail"
        records.append(_hls_synth_record(
            bench, "c2hls_orchestrator", cand_meta_hls, cand_variant, synth,
            cand_status, part, clock_ns,
            origin_version=cand_origin_version,
            runtime_seconds=wallclock_s,
        ))

    # Candidate csim
    csim_status = _status_from_flags(cand_csim)
    if csim_status is not None:
        records.append(_sw_run_record(
            bench, "c2hls_orchestrator", cand_meta_run, cand_variant, csim_status, part,
            origin_version=cand_origin_version,
            runtime_seconds=wallclock_s,
        ))

    # Candidate cosim
    cosim = cand_cosim or {}
    cosim_status = _status_from_flags(cosim)
    if cosim_status is not None:
        records.append(_rtl_sim_record(
            bench, "c2hls_orchestrator", cand_meta_run, cand_variant, cosim_status, part,
            cosim.get("measured"), clock_ns,
            origin_version=cand_origin_version,
            runtime_seconds=wallclock_s,
        ))

    # Gold baseline — emit at most once per (bench, report_type) across all
    # A/B cells, since gold is deterministic w.r.t. the benchmark source.
    ref = data.get("reference_validation") or {}
    gold_report = ref.get("report") or {}
    gold_synth_status = "pass" if gold_report.get("latency_cycles") is not None else "fail"
    if gold_report:
        key = (bench, "hls_synth")
        if key not in emitted_gold_keys:
            emitted_gold_keys.add(key)
            records.append(_hls_synth_record(
                bench, "hlsfactory_benchmark", gold_meta_hls, gold_variant, gold_report,
                gold_synth_status, part, clock_ns,
            ))

    gold_csim_status = _status_from_flags(ref.get("csim"))
    if gold_csim_status is not None:
        key = (bench, "sw_run")
        if key not in emitted_gold_keys:
            emitted_gold_keys.add(key)
            records.append(_sw_run_record(
                bench, "hlsfactory_benchmark", gold_meta_run, gold_variant, gold_csim_status, part,
            ))

    gold_cosim = ref.get("cosim") or {}
    gold_cosim_status = _status_from_flags(gold_cosim)
    if gold_cosim_status is not None:
        key = (bench, "rtl_sim")
        if key not in emitted_gold_keys:
            emitted_gold_keys.add(key)
            records.append(_rtl_sim_record(
                bench, "hlsfactory_benchmark", gold_meta_run, gold_variant, gold_cosim_status, part,
                gold_cosim.get("measured"), clock_ns,
            ))

    return records


MODEL_DIR_TO_ID = {
    "sonnet": "claude-sonnet-4-6",
    "haiku": "claude-haiku-4-5-20251001",
}


def _iter_cells(root: Path):
    for bench_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        bench = bench_dir.name
        for cell_dir in sorted(p for p in bench_dir.iterdir() if p.is_dir()):
            parts = cell_dir.name.split("__")
            if len(parts) != 3:
                continue
            model_lbl, mode, skills_tag = parts
            model = MODEL_DIR_TO_ID.get(model_lbl, model_lbl)
            skills = "on" if skills_tag == "skills" else "off"
            yield bench, cell_dir, model, mode, skills


def _load_wallclock_map(root: Path) -> dict[tuple[str, str, str, str], float]:
    """Build (bench, model, mode, skills) -> wallclock_s from matrix.json.

    matrix.json is the per-cell summary written by the matrix runner and
    carries wall-clock duration per (bench, model, mode, skills) cell. The
    schema contract says `run.runtime_seconds` is wall time of the run, so
    we surface this per candidate record.
    """
    out: dict[tuple[str, str, str, str], float] = {}
    matrix_path = root / "matrix.json"
    if not matrix_path.exists():
        return out
    try:
        cells = json.loads(matrix_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return out
    if not isinstance(cells, list):
        return out
    for cell in cells:
        if not isinstance(cell, dict):
            continue
        bench = cell.get("bench")
        model = cell.get("model")
        mode = cell.get("mode")
        skills = cell.get("skills")
        wc = cell.get("wallclock_s")
        if bench and model and mode and skills and wc is not None:
            try:
                out[(bench, model, mode, skills)] = float(wc)
            except (TypeError, ValueError):
                pass
    return out


def main() -> int:
    if len(sys.argv) not in (3, 4):
        print(__doc__)
        return 2
    root = Path(sys.argv[1])
    out_path = Path(sys.argv[2])
    setup_label = sys.argv[3] if len(sys.argv) == 4 else os.environ.get(
        "C2HLS_SCHEMA_SETUP_LABEL",
        os.environ.get("C2HLS_SCHEMA_VARIANT_SUFFIX", ""),  # deprecated alias
    )
    setup_label = (setup_label or "").strip()
    if not root.exists():
        print(f"no such dir: {root}")
        return 1
    wallclock_map = _load_wallclock_map(root)
    records: list[dict] = []
    # Shared across all cells so gold records are emitted only once per
    # (bench, report_type), not once per A/B cell.
    emitted_gold_keys: set[tuple[str, str]] = set()
    for bench, cell_dir, model, mode, skills in _iter_cells(root):
        wc = wallclock_map.get((bench, model, mode, skills))
        records.extend(_emit_for_cell(
            bench, cell_dir, model, mode, skills, emitted_gold_keys,
            wallclock_s=wc, setup_label=setup_label,
        ))
    with out_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, separators=(", ", ": ")) + "\n")
    print(f"Wrote {len(records)} records to {out_path}")
    # Per-type summary
    counts: dict[str, int] = {}
    for r in records:
        counts[r["report_type"]] = counts.get(r["report_type"], 0) + 1
    for k, v in sorted(counts.items()):
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
