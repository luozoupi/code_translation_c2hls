"""Phase 1 smoke test: covers Pillar 1 (fine-grained feedback parser),
Pillar 9 MVP (no-op trap, csim-gating, xrt.ini auto-inject), and Pillar 8
(dataset_pipeline v2.0 schema + recorder + replay + merge).

Runs offline — does NOT require a Vitis installation. Uses an existing
synthesis artifact (rodinia-hls-nova/Benchmarks/pathfinder/.../csynth.{xml,rpt})
as the substrate for the parser tests, and `results_multistep/` for the
replay tests.

Run:
    cd /home/luo00466/code_translation-c2hls
    python tests/test_phase1_smoke.py [--out artifacts/phase1_smoke_<timestamp>.md]

Exit code 0 = all assertions passed; non-zero = at least one failure.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import inspect
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import hls_eval               # noqa: E402
import hls_feedback as hf     # noqa: E402
import c2hls                  # noqa: E402
from dataset_pipeline import (  # noqa: E402
    SCHEMA_VERSION,
    classify_step_effect,
    classify_validation_status,
    merge_with_references,
    record_step_outcome,
    replay_existing_results,
)
from dataset_pipeline.schema import RunMeta  # noqa: E402


# Path to a known good synthesis artifact already on disk. Used as the
# input for parser tests; if these files ever move, update the constants.
_SAMPLE_ROOT = (
    "/home/luo00466/rodinia-hls-nova/Benchmarks/pathfinder/pathfinder_0_baseline/"
    "_x.hw_emu.xilinx_u280_gen3x16_xdma_1_202211_1/pathfinder/"
    "pathfinder.hw_emu.xilinx_u280_gen3x16_xdma_1_202211_1/workload/workload/"
    "solution/syn/report"
)
SAMPLE_XML = f"{_SAMPLE_ROOT}/csynth.xml"
SAMPLE_RPT = f"{_SAMPLE_ROOT}/csynth.rpt"


class Tee:
    """Write console output AND collect for the markdown artifact."""

    def __init__(self):
        self.buf: List[str] = []

    def __call__(self, *args: Any, level: str = "info"):
        line = " ".join(str(a) for a in args)
        print(line)
        self.buf.append(line)

    def section(self, title: str) -> None:
        print()
        print(f"=== {title} ===")
        self.buf.append("")
        self.buf.append(f"## {title}")
        self.buf.append("")

    def md(self) -> str:
        return "\n".join(self.buf) + "\n"


def _check(label: str, ok: bool, detail: str, results: List[Dict[str, Any]],
           tee: Tee) -> bool:
    results.append({"check": label, "ok": bool(ok), "detail": detail})
    icon = "OK  " if ok else "FAIL"
    tee(f"[{icon}] {label}: {detail}")
    return ok


def test_pillar1_feedback_parser(results: List[Dict[str, Any]], tee: Tee) -> None:
    tee.section("Pillar 1: fine-grained HLS feedback parser")

    if not (Path(SAMPLE_XML).exists() and Path(SAMPLE_RPT).exists()):
        _check("sample-artifact-present", False,
               f"missing fixture: {SAMPLE_XML}", results, tee)
        return

    fb = hf.build_feedback(xml_path=SAMPLE_XML, rpt_path=SAMPLE_RPT)
    summary = fb.get("summary") or {}

    _check("scopes-emitted", summary.get("scope_count", 0) >= 8,
           f"scope_count={summary.get('scope_count')} (expect >=8)", results, tee)
    _check("loops-emitted", summary.get("loop_count", 0) >= 5,
           f"loop_count={summary.get('loop_count')} (expect >=5)", results, tee)
    _check("loops-pipelined-detected", summary.get("pipelined_loops", 0) >= 4,
           f"pipelined_loops={summary.get('pipelined_loops')} (expect >=4)", results, tee)

    scopes = fb.get("scopes") or []
    ids = [s["scope_id"] for s in scopes]
    _check("no-duplicate-scope-ids", len(ids) == len(set(ids)),
           f"unique={len(set(ids))}/total={len(ids)}", results, tee)

    # KERNEL_OUTER must be a non-pipelined hot loop
    kernel_outer = next((s for s in scopes if s["scope_id"].endswith("/KERNEL_OUTER")), None)
    _check("kernel-outer-found", kernel_outer is not None,
           "scope id ending /KERNEL_OUTER", results, tee)
    if kernel_outer is not None:
        _check(
            "kernel-outer-shape",
            kernel_outer.get("kind") == "loop"
            and kernel_outer.get("trip_count") == 1023
            and kernel_outer.get("latency_cycles") == 2111472
            and (kernel_outer.get("pipelined") or "").lower().startswith("no"),
            f"trip={kernel_outer.get('trip_count')} lat={kernel_outer.get('latency_cycles')} "
            f"piped={kernel_outer.get('pipelined')}",
            results, tee,
        )

    # KERNEL_INNER must be pipelined II=1
    kernel_inner = next((s for s in scopes if s["scope_id"].endswith("/KERNEL_INNER")), None)
    if kernel_inner is not None:
        _check(
            "kernel-inner-pipelined-ii1",
            (kernel_inner.get("pipelined") or "").lower().startswith("yes")
            and kernel_inner.get("pipeline_ii") == 1,
            f"piped={kernel_inner.get('pipelined')} ii={kernel_inner.get('pipeline_ii')}",
            results, tee,
        )

    bottlenecks = fb.get("bottlenecks") or []
    kinds = [b.get("kind") for b in bottlenecks]
    _check("bottleneck-kinds-present", "non_pipelined_hot_loop" in kinds,
           f"kinds={kinds}", results, tee)
    _check("bottlenecks-have-severity",
           all(b.get("severity") in {"high", "medium", "low"} for b in bottlenecks),
           f"severities={[b.get('severity') for b in bottlenecks]}", results, tee)


def test_pillar9_no_op_detector(results: List[Dict[str, Any]], tee: Tee) -> None:
    tee.section("Pillar 9: no-op trap detector")

    prev = {"latency_cycles": 1048816, "interval": 1048817, "bram": 30,
            "dsp": 14, "ff": 7864, "lut": 5257}

    no_op = dict(prev)
    reasons = c2hls._step_no_op_reasons(no_op, prev)
    _check("no-op-fires-on-identical-tuple", bool(reasons),
           f"reasons={reasons[:1] if reasons else []}", results, tee)

    improved = dict(prev, latency_cycles=524000, latency_ns=1747.4)
    no_op_neg = c2hls._step_no_op_reasons(improved, prev)
    _check("no-op-quiet-when-improved", not no_op_neg,
           f"reasons={no_op_neg}", results, tee)

    sparse = {"latency_cycles": 1024}
    no_op_sparse = c2hls._step_no_op_reasons(sparse, sparse)
    _check("no-op-quiet-when-tuple-too-sparse", not no_op_sparse,
           "fewer than 3 populated fields → no false positive", results, tee)


def test_pillar9_csim_gating(results: List[Dict[str, Any]], tee: Tee) -> None:
    tee.section("Pillar 9: csim-gating (validation_status)")

    cases = [
        ({"available": True, "ran": True, "passed": True}, "validated"),
        ({"available": False, "ran": False, "passed": False}, "unscored"),
        ({"available": True, "ran": False, "passed": False}, "unscored"),
        ({"available": True, "ran": True, "passed": False}, "failed"),
        (None, "unscored"),
    ]
    for csim, expected in cases:
        actual = classify_validation_status(csim)
        _check(
            f"csim-status::{expected}",
            actual == expected,
            f"csim={csim} → got '{actual}', expected '{expected}'",
            results, tee,
        )

    # Synth-fail short-circuits all csim cases.
    actual = classify_validation_status(
        {"available": True, "ran": True, "passed": True},
        synth_status="fail",
    )
    _check("synth-fail-overrides-csim", actual == "failed",
           f"got '{actual}' (synth_status=fail)", results, tee)


def test_pillar9_xrt_ini_wired(results: List[Dict[str, Any]], tee: Tee) -> None:
    tee.section("Pillar 9: xrt.ini auto-inject")

    src = inspect.getsource(hls_eval._stage_nova_workdir)
    has_xrt_ini = "xrt.ini" in src
    has_debug_off = "debug_mode=off" in src
    has_profile = "profile=true" in src.lower()
    _check("xrt-ini-write-present", has_xrt_ini,
           "xrt.ini literal in _stage_nova_workdir", results, tee)
    _check("xrt-ini-content-debug-off", has_debug_off,
           "debug_mode=off line present", results, tee)
    _check("xrt-ini-content-profile", has_profile,
           "profile=true line present", results, tee)


def test_pillar8_step_effect_classification(results: List[Dict[str, Any]], tee: Tee) -> None:
    tee.section("Pillar 8: step_effect classification")

    prev = {"latency_ns": 3493.0, "latency_cycles": 1048816, "interval": 1048817,
            "bram": 30, "dsp": 14, "ff": 7864, "lut": 5257, "fmax_mhz": 300}

    cases = [
        # name, new_report, expected_effect, success, csim_passed, error
        ("first-step-no-parent", {"latency_ns": 1747.4}, "improved", True, None, None),
        ("identical-tuple", dict(prev), "no_op", True, None, None),
        # A real regression has to differ on integer tuple fields (lat_cycles,
        # interval, bram, dsp, ff, lut) AND a >10% latency_ns growth — otherwise
        # the no-op detector legitimately short-circuits first.
        ("regressed-latency",
         dict(prev, latency_ns=8000.0, latency_cycles=2400000, interval=2400001),
         "regressed", True, None, None),
        ("absorbed",
         dict(prev, latency_ns=3490.0, latency_cycles=1048800, lut=5258),
         "absorbed", True, None, None),
        ("synth-failed", None, "synth_failed", False, None, "Synthesis failed"),
        ("translation-failed", None, "translation_failed", False, None,
         "Translation produced no code"),
        ("csim-failed",
         dict(prev, latency_ns=1747.4, latency_cycles=524000, interval=524001),
         "csim_failed", True, False, None),
    ]

    for name, new_report, expected, success, csim_passed, error in cases:
        parent = None if name == "first-step-no-parent" else prev
        got = classify_step_effect(
            new_report, parent,
            success=success, csim_passed=csim_passed, error=error,
        )
        _check(f"step-effect::{name}", got == expected,
               f"got '{got}', expected '{expected}'", results, tee)


def test_pillar8_replay_existing_results(results: List[Dict[str, Any]], tee: Tee,
                                          out_dir: Path) -> None:
    tee.section("Pillar 8: dataset_pipeline replay-existing-results")

    rm = RunMeta(
        target="vitis.csynth", vitis_version="2023.2",
        device="xcu280-fsvh2892-2L-e", flow_target="vitis", clock_ns=3.33,
    )
    out_jsonl = out_dir / "phase1_smoke_traj_v2.jsonl"
    summary = replay_existing_results(
        results_dirs=[str(REPO_ROOT / "results_multistep")],
        output_jsonl=str(out_jsonl),
        run_meta=rm,
        origin_version="phase1-smoke-test",
    )
    _check("replay-no-skipped", not summary["skipped_files"],
           f"skipped={summary['skipped_files']}", results, tee)
    _check("replay-records-emitted", summary["records_written"] >= 5,
           f"records={summary['records_written']} (expect >=5)", results, tee)
    _check("replay-multiple-kernels", summary["kernel_count"] >= 1,
           f"kernel_count={summary['kernel_count']} ({summary['kernels']})", results, tee)

    # Validate every emitted record is shape-correct.
    bad_rows = 0
    types: Dict[str, int] = {}
    effects: Dict[str, int] = {}
    statuses: Dict[str, int] = {}
    schema_versions: Dict[str, int] = {}
    with out_jsonl.open() as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                bad_rows += 1
                continue
            schema_versions[rec.get("schema_version", "?")] = (
                schema_versions.get(rec.get("schema_version", "?"), 0) + 1
            )
            types[rec.get("report_type", "?")] = types.get(rec.get("report_type", "?"), 0) + 1
            impl = rec.get("implementation") or {}
            effects[impl.get("step_effect", "?")] = effects.get(impl.get("step_effect", "?"), 0) + 1
            statuses[impl.get("validation_status", "?")] = statuses.get(
                impl.get("validation_status", "?"), 0,
            ) + 1
    _check("no-malformed-rows", bad_rows == 0,
           f"bad_rows={bad_rows}", results, tee)
    _check("schema-version-2-only", set(schema_versions.keys()) <= {"2.0"},
           f"versions={schema_versions}", results, tee)
    tee(f"  step_effects: {effects}")
    tee(f"  validation_statuses: {statuses}")
    tee(f"  report_types: {types}")


def test_pillar8_merge_with_references(results: List[Dict[str, Any]], tee: Tee,
                                        out_dir: Path) -> None:
    tee.section("Pillar 8: merge_with_references")

    refs = [
        str(REPO_ROOT / "results/references_philip/hw_emu_vitis_2023.2"
            "__device_xilinx_u280_gen3x16_xdma_1_202211_1.jsonl"),
        str(REPO_ROOT / "results/references_philip/sw_emu_vitis_2023.2"
            "__device_xilinx_u280_gen3x16_xdma_1_202211_1.jsonl"),
    ]
    refs_present = [p for p in refs if Path(p).exists()]
    _check("reference-jsonls-present", len(refs_present) == 2,
           f"present={len(refs_present)}/2", results, tee)

    gen = out_dir / "phase1_smoke_traj_v2.jsonl"
    if not gen.exists():
        _check("merge-input-present", False,
               f"missing {gen}; replay step must run first", results, tee)
        return

    merged = out_dir / "phase1_smoke_traj_v2_merged.jsonl"
    summary = merge_with_references(
        generated_jsonl=str(gen),
        reference_paths=refs_present,
        output_jsonl=str(merged),
    )
    _check(
        "merge-records-conserved",
        summary["merged_records"] == summary["reference_records"] + summary["generated_records"],
        f"merged={summary['merged_records']} = ref({summary['reference_records']}) "
        f"+ gen({summary['generated_records']})",
        results, tee,
    )
    _check("merge-joint-keys-positive", summary["joint_keys"] > 0,
           f"joint_keys={summary['joint_keys']}", results, tee)


def test_pillar8_record_step_outcome_unit(results: List[Dict[str, Any]], tee: Tee) -> None:
    tee.section("Pillar 8: record_step_outcome unit case")

    rm = RunMeta(target="vitis.csynth", vitis_version="2023.2",
                 device="xcu280-fsvh2892-2L-e", clock_ns=3.33)

    parent = {"latency_cycles": 1048816, "latency_ns": 3493.0, "interval": 1048817,
              "bram": 30, "dsp": 14, "ff": 7864, "lut": 5257, "fmax_mhz": 300}

    # 1) An "improved" step with csim pass and feedback.
    improved_report = dict(parent, latency_cycles=524000, latency_ns=1747.4,
                           interval=524001, fmax_mhz=300)
    improved_report["feedback"] = {
        "schema": "1.0",
        "scopes": [],
        "scheduler_blame": [],
        "bottlenecks": [],
        "summary": {"scope_count": 0, "loop_count": 0, "bottleneck_count": 0},
    }
    step_result = {
        "success": True,
        "step_name": "pipeline",
        "report": improved_report,
        "csim": {"available": True, "ran": True, "passed": True},
    }
    recs = record_step_outcome(
        step_result=step_result, suite="rodinia_hls", group_path=["knn"],
        variant_index=2, variant_name="pipeline", run_meta=rm,
        parent_report=parent, origin_version="unit-test",
    )
    _check("unit-improved-record-count", len(recs) == 1,
           f"got {len(recs)} record(s)", results, tee)
    if recs:
        impl = recs[0]["implementation"]
        _check("unit-improved-step-effect", impl["step_effect"] == "improved",
               f"effect={impl['step_effect']}", results, tee)
        _check("unit-improved-validation", impl["validation_status"] == "validated",
               f"validation={impl['validation_status']}", results, tee)
        _check("unit-improved-feedback-attached", "feedback" in recs[0],
               "feedback key present in record", results, tee)
        _check("unit-improved-hashes-present",
               impl.get("parent_hash") and impl.get("candidate_hash"),
               f"parent={impl.get('parent_hash')} cand={impl.get('candidate_hash')}",
               results, tee)

    # 2) A no-op step (Pillar 9 path) with no csim → unscored.
    no_op_step = {
        "success": False,
        "step_name": "unroll",
        "error": "no_op_persisted",
        "no_op_reasons": ["identical_synth_tuple", "all unchanged"],
        "rejected_report": dict(parent),
        "reverted_to_prev": True,
    }
    recs = record_step_outcome(
        step_result=no_op_step, suite="rodinia_hls", group_path=["knn"],
        variant_index=3, variant_name="unroll", run_meta=rm,
        parent_report=parent, origin_version="unit-test",
    )
    _check("unit-no-op-step-effect", recs and recs[0]["implementation"]["step_effect"] == "no_op",
           f"effect={recs[0]['implementation']['step_effect'] if recs else None}",
           results, tee)
    _check("unit-no-op-unscored",
           recs and recs[0]["implementation"]["validation_status"] != "validated",
           f"status={recs[0]['implementation']['validation_status'] if recs else None}",
           results, tee)


def test_existing_pipeline_still_imports(results: List[Dict[str, Any]], tee: Tee) -> None:
    tee.section("Regression: existing modules still import / invariants hold")

    _check("hls-eval-import", hasattr(hls_eval, "run_hls_synthesis"),
           "run_hls_synthesis present", results, tee)
    _check("c2hls-orchestrator-import", hasattr(c2hls, "C2HLSOrchestrator"),
           "C2HLSOrchestrator class present", results, tee)
    _check("rubric-import", __import__("rubric") is not None,
           "rubric module imports", results, tee)

    # canonical_report must drop feedback before hashing
    rep = {"latency_cycles": 1, "feedback": {"x": 1}}
    canon = hls_eval.canonical_report(rep)
    _check("canonical-report-drops-feedback", "feedback" not in canon,
           f"keys={list(canon.keys())}", results, tee)


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 1 smoke test")
    timestamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    default_out = REPO_ROOT / "artifacts" / f"phase1_smoke_{timestamp}.md"
    parser.add_argument("--out", type=Path, default=default_out,
                        help="Markdown report output path")
    parser.add_argument("--keep-tmp", action="store_true",
                        help="Keep the temp jsonl files on disk for inspection")
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(tempfile.mkdtemp(prefix="phase1_smoke_"))
    tee = Tee()
    results: List[Dict[str, Any]] = []

    tee(f"# Phase 1 Smoke Test ({timestamp})")
    tee(f"REPO_ROOT={REPO_ROOT}")
    tee(f"sample artifact: {SAMPLE_XML}")
    tee(f"tmp dir: {tmp_dir}")

    test_pillar1_feedback_parser(results, tee)
    test_pillar9_no_op_detector(results, tee)
    test_pillar9_csim_gating(results, tee)
    test_pillar9_xrt_ini_wired(results, tee)
    test_pillar8_step_effect_classification(results, tee)
    test_pillar8_record_step_outcome_unit(results, tee)
    test_pillar8_replay_existing_results(results, tee, tmp_dir)
    test_pillar8_merge_with_references(results, tee, tmp_dir)
    test_existing_pipeline_still_imports(results, tee)

    tee.section("Summary")
    total = len(results)
    failed = [r for r in results if not r["ok"]]
    tee(f"checks_total: {total}")
    tee(f"checks_passed: {total - len(failed)}")
    tee(f"checks_failed: {len(failed)}")
    if failed:
        tee("failed_checks:")
        for r in failed:
            tee(f"  - {r['check']}: {r['detail']}")

    args.out.write_text(tee.md(), encoding="utf-8")
    print()
    print(f"report written: {args.out}")
    if not args.keep_tmp:
        try:
            import shutil
            shutil.rmtree(tmp_dir, ignore_errors=True)
        except OSError:
            pass
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
