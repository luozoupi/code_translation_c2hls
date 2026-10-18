#!/usr/bin/env python3
"""Run a deterministic QoR step-control smoke from a saved valid result."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-json", type=Path, required=True)
    parser.add_argument("--bench-dir", type=Path, required=True)
    parser.add_argument("--origin-step", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--part", default="xcu280-fsvh2892-2L-e")
    parser.add_argument("--clock-ns", type=float, default=3.33)
    parser.add_argument("--vitis-version", default="2023.2")
    parser.add_argument("--shape-registry", type=Path)
    parser.add_argument("--max-knobs", type=int, default=1)
    parser.add_argument("--max-candidates", type=int, default=1)
    parser.add_argument("--factor-values", default="1,2,4,8,16")
    parser.add_argument("--ii-values", default="1,2,4,8")
    parser.add_argument("--tile-values", default="4,8,16,32,64")
    parser.add_argument("--widen-values", default="64,128,256,512,1024")
    parser.add_argument("--expected-knob-kind")
    parser.add_argument("--expected-knob-name")
    parser.add_argument(
        "--gemm-linked-lanes-parent",
        action="store_true",
        help=(
            "replace GEMM's compiler-overridden unroll with a linked lane "
            "control and freshly validate the transformed parent"
        ),
    )
    parser.add_argument("--parent-lanes", type=int, default=2)
    parser.add_argument(
        "--gemm-linked-lanes-architecture",
        choices=("direct_reduction", "striped_accumulators"),
        default="direct_reduction",
    )
    parser.add_argument(
        "--gemm-step-control-parent",
        choices=("pipeline", "tiling", "doublebuffer", "coalescing", "resource"),
        help="build and freshly validate a step-native linked GEMM parent",
    )
    parser.add_argument(
        "--two-mm-linked-lanes-parent",
        action="store_true",
        help="build and freshly validate a coupled two-reduction 2MM lane parent",
    )
    parser.add_argument(
        "--two-mm-step-control-parent",
        choices=("pipeline", "tiling", "doublebuffer", "coalescing", "resource"),
        help="build and freshly validate a step-native linked 2MM parent",
    )
    parser.add_argument("--parent-control-value", type=int)
    parser.add_argument("--fixed-lanes", type=int, default=8)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    os.environ["C2HLS_QOR_DESIGN_SWEEP"] = "1"
    os.environ["C2HLS_QOR_SWEEP_MAX_KNOBS"] = str(args.max_knobs)
    os.environ["C2HLS_QOR_SWEEP_MAX_CANDIDATES"] = str(args.max_candidates)
    os.environ["C2HLS_QOR_SWEEP_INTERACTIONS"] = "0"
    os.environ["C2HLS_QOR_SWEEP_VALUES"] = args.factor_values
    os.environ["C2HLS_QOR_SWEEP_II_VALUES"] = args.ii_values
    os.environ["C2HLS_QOR_SWEEP_TILE_VALUES"] = args.tile_values
    os.environ["C2HLS_QOR_SWEEP_WIDEN_VALUES"] = args.widen_values
    if args.expected_knob_name:
        os.environ["C2HLS_QOR_SWEEP_KNOB_NAME"] = args.expected_knob_name
    if args.shape_registry is not None:
        os.environ["C2HLS_HLSFACTORY_SHAPE_REGISTRY"] = str(
            args.shape_registry.resolve()
        )
    os.environ.setdefault("OPENAI_API_KEY", "EMPTY")
    os.environ.setdefault("OPENAI_BASE_URL", "http://127.0.0.1:1/v1")

    import c2hls
    from qor_design_space import (
        STEP_PREFERRED_KINDS,
        code_sha256,
        discover_qor_knobs,
        extract_qor_metrics,
    )
    from qor_linked_controls import (
        build_2mm_linked_lanes_parent,
        build_2mm_step_control_parent,
        build_gemm_linked_lanes_parent,
        build_gemm_step_control_parent,
        localize_2mm_dimensions,
        validate_2mm_lane_values,
        validate_2mm_step_control_values,
        validate_gemm_lane_values,
        validate_gemm_step_control_values,
    )

    source = json.loads(args.result_json.read_text())
    inputs = c2hls._load_benchmark_inputs(str(args.bench_dir))
    golden = c2hls._prepare_independent_golden(inputs)
    if not golden.get("success"):
        raise RuntimeError(golden.get("error") or "independent golden failed")

    source_hls_code = str(source.get("hls_code") or "")
    report = source.get("final_report") or {}
    csim = source.get("csim") or {}
    if not source_hls_code or not report or csim.get("passed") is not True:
        raise ValueError("saved result must contain code, CSynth report, and passing CSim")
    linked_control = None
    hls_code = source_hls_code
    linked_modes = sum(bool(value) for value in (
        args.gemm_linked_lanes_parent,
        args.gemm_step_control_parent,
        args.two_mm_linked_lanes_parent,
        args.two_mm_step_control_parent,
    ))
    if linked_modes > 1:
        raise ValueError("choose exactly one linked-parent mode")
    if args.gemm_linked_lanes_parent:
        hls_code, linked_control = build_gemm_linked_lanes_parent(
            source_hls_code,
            parent_lanes=args.parent_lanes,
            architecture=args.gemm_linked_lanes_architecture,
        )
    elif args.gemm_step_control_parent:
        if args.parent_control_value is None:
            raise ValueError("--parent-control-value is required for a GEMM step control")
        hls_code, linked_control = build_gemm_step_control_parent(
            source_hls_code,
            step=args.gemm_step_control_parent,
            parent_value=args.parent_control_value,
            fixed_lanes=args.fixed_lanes,
        )
    elif args.two_mm_linked_lanes_parent:
        hls_code, linked_control = build_2mm_linked_lanes_parent(
            source_hls_code,
            parent_lanes=args.parent_lanes,
        )
    elif args.two_mm_step_control_parent:
        if args.parent_control_value is None:
            raise ValueError(
                "--parent-control-value is required for a 2MM step control"
            )
        hls_code, linked_control = build_2mm_step_control_parent(
            source_hls_code,
            step=args.two_mm_step_control_parent,
            parent_value=args.parent_control_value,
            fixed_lanes=args.fixed_lanes,
        )
    if linked_control and linked_control.get("benchmark") == "2mm":
        hls_code, dimension_localization = localize_2mm_dimensions(
            hls_code,
            inputs["header_code"],
        )
        linked_control["dimension_localization"] = dimension_localization
        linked_control["linked_parent_code_sha256"] = hashlib.sha256(
            hls_code.encode("utf-8")
        ).hexdigest()

    def values(raw: str) -> tuple[int, ...]:
        return tuple(int(item.strip()) for item in raw.split(",") if item.strip())

    if (
        linked_control
        and args.gemm_linked_lanes_parent
        and args.gemm_linked_lanes_architecture == "striped_accumulators"
    ):
        linked_control["domain_validation"] = validate_gemm_lane_values(
            inputs["header_code"],
            (*values(args.factor_values), int(args.parent_lanes)),
        )
    elif linked_control and args.gemm_step_control_parent:
        domain_values = {
            "pipeline": values(args.ii_values),
            "tiling": values(args.tile_values),
            "doublebuffer": values(args.factor_values),
            "coalescing": values(args.widen_values),
            "resource": values(args.factor_values),
        }[args.gemm_step_control_parent]
        linked_control["domain_validation"] = validate_gemm_step_control_values(
            inputs["header_code"],
            args.gemm_step_control_parent,
            (*domain_values, int(args.parent_control_value)),
            fixed_lanes=args.fixed_lanes,
        )
    elif linked_control and args.two_mm_linked_lanes_parent:
        linked_control["domain_validation"] = validate_2mm_lane_values(
            inputs["header_code"],
            (*values(args.factor_values), int(args.parent_lanes)),
        )
    elif linked_control and args.two_mm_step_control_parent:
        domain_values = {
            "pipeline": values(args.ii_values),
            "tiling": values(args.tile_values),
            "doublebuffer": values(args.factor_values),
            "coalescing": values(args.widen_values),
            "resource": values(args.factor_values),
        }[args.two_mm_step_control_parent]
        linked_control["domain_validation"] = validate_2mm_step_control_values(
            inputs["header_code"],
            args.two_mm_step_control_parent,
            (*domain_values, int(args.parent_control_value)),
            fixed_lanes=args.fixed_lanes,
        )

    origin_step = args.origin_step.strip().lower().replace(
        "double_buffer", "doublebuffer"
    )
    probe_knobs = discover_qor_knobs(
        hls_code,
        factor_values=values(args.factor_values),
        ii_values=values(args.ii_values),
        tile_values=values(args.tile_values),
        stream_depth_values=values(args.factor_values),
        widen_values=values(args.widen_values),
        max_knobs=(None if args.expected_knob_name else args.max_knobs),
        preferred_kinds=STEP_PREFERRED_KINDS.get(origin_step, ()),
    )
    if args.expected_knob_name:
        probe_knobs = [
            knob for knob in probe_knobs if knob.name == args.expected_knob_name
        ][:args.max_knobs]
    if args.expected_knob_kind and (
        not probe_knobs or probe_knobs[0].kind != args.expected_knob_kind
    ):
        actual = probe_knobs[0].kind if probe_knobs else None
        raise ValueError(
            f"expected first QoR knob kind {args.expected_knob_kind!r}, got {actual!r}"
        )
    if args.expected_knob_name and (
        not probe_knobs or probe_knobs[0].name != args.expected_knob_name
    ):
        actual = probe_knobs[0].name if probe_knobs else None
        raise ValueError(
            f"expected first QoR knob name {args.expected_knob_name!r}, got {actual!r}"
        )

    orchestrator = c2hls.C2HLSOrchestrator(
        gpt_model="local-qor-smoke",
        quality_repair_turns=0,
    )
    orchestrator.c_code = inputs["c_code"]
    orchestrator.hls_code = hls_code
    orchestrator.synth_report = {} if linked_control else report
    orchestrator.generated_csim = None if linked_control else csim
    orchestrator.generated_cosim = None
    orchestrator.header_code = inputs["header_code"]
    orchestrator.header_name = inputs["header_name"]
    orchestrator.testbench_code = inputs["testbench_code"]
    orchestrator.extra_files = inputs["extra_files"]
    orchestrator.translated_hls_top = inputs["meta"].get("hls_top", "workload")
    orchestrator.part = args.part
    orchestrator.clock_ns = args.clock_ns
    orchestrator.vitis_version = args.vitis_version
    orchestrator.benchmark_name = inputs["bench_name"]
    orchestrator.benchmark_context = inputs["benchmark_context"]
    orchestrator.independent_golden_output = golden["output"]
    orchestrator.independent_golden_specs = golden["specs"]
    orchestrator.independent_golden_provenance = golden["provenance"]
    orchestrator.synthesis_eval_budget = (
        args.max_candidates + (1 if linked_control else 0)
    )
    orchestrator.qor_parent_origin = {
        "step_name": args.origin_step,
        "step_index": None,
        "source": (
            "freshly_validated_linked_control_parent"
            if linked_control
            else "saved_valid_result_smoke"
        ),
    }

    linked_parent_evaluation = None
    if linked_control:
        evaluation = orchestrator._evaluate_qor_candidate(
            hls_code,
            "[QoR linked control parent]",
            {
                "stage": "linked_parent",
                "linked_control": linked_control,
            },
        )
        linked_parent_evaluation = {
            "success": evaluation.get("success") is True,
            "feasible": (evaluation.get("feasibility") or {}).get("feasible") is True,
            "feasibility": evaluation.get("feasibility") or {},
            "csim": c2hls._sanitize_test_summary(evaluation.get("csim")),
            "metrics": extract_qor_metrics(evaluation.get("report")),
            "report": evaluation.get("report") or {},
            "error": str(evaluation.get("error") or ""),
            "event": evaluation.get("event") or {},
        }
        if linked_parent_evaluation["feasible"] is not True:
            reasons = linked_parent_evaluation["feasibility"].get("reasons") or []
            detail = linked_parent_evaluation["error"] or ", ".join(
                str(reason) for reason in reasons
            )
            if not detail:
                detail = json.dumps(linked_parent_evaluation["metrics"], sort_keys=True)
            raise RuntimeError(
                "fresh linked parent failed CSim/CSynth feasibility: "
                + detail
            )
        orchestrator.synth_report = evaluation["report"]
        orchestrator.generated_csim = evaluation.get("csim")

    design_sweep = c2hls.QualityRepairAgent(orchestrator).run_design_sweep()
    payload = {
        "schema_version": "c2hls.saved-qor-step-smoke.v1",
        "benchmark": inputs["bench_name"],
        "reference_blind": True,
        "model_calls": 0,
        "source_result": {
            "path": str(args.result_json.resolve()),
            "sha256": _sha256(args.result_json),
            "code_sha256": code_sha256(source_hls_code),
        },
        "linked_control": linked_control,
        "linked_parent_evaluation": linked_parent_evaluation,
        "independent_golden": golden["provenance"],
        "toolchain": {
            "vitis_version": args.vitis_version,
            "part": args.part,
            "clock_ns": args.clock_ns,
            "shape_registry": (
                str(args.shape_registry.resolve())
                if args.shape_registry is not None
                else None
            ),
        },
        "requested_values": {
            "factor": args.factor_values,
            "pipeline_ii": args.ii_values,
            "tile": args.tile_values,
            "widen": args.widen_values,
        },
        "expected_knob": {
            "kind": args.expected_knob_kind,
            "name": args.expected_knob_name,
        },
        "design_sweep": design_sweep,
        "selected_code_sha256": code_sha256(orchestrator.hls_code),
        "selected_hls_code": orchestrator.hls_code,
        "selected_report": orchestrator.synth_report,
        "selected_csim": c2hls._sanitize_test_summary(orchestrator.generated_csim),
        "synthesis_evaluations": orchestrator._synthesis_evaluation_summary(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "attempted": design_sweep.get("attempted"),
        "candidate_count": design_sweep.get("candidate_count"),
        "feasible_candidate_count": design_sweep.get("feasible_candidate_count"),
        "winner_candidate_id": design_sweep.get("winner_candidate_id"),
        "applied": design_sweep.get("applied"),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
