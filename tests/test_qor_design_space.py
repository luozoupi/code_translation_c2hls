from __future__ import annotations

import hashlib
import os
import re
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import c2hls  # noqa: E402
import export_schema_jsonl  # noqa: E402
import hls_feedback  # noqa: E402
import qor_design_space as qds  # noqa: E402
import run_agentic_sweep  # noqa: E402
from qor_linked_controls import (  # noqa: E402
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


PART = "xcu280-fsvh2892-2L-e"
CLOCK_NS = 3.33


def _report(cycles: int, *, dsp: int = 8, ii: int = 1) -> dict:
    return {
        "latency_cycles": cycles,
        "latency_cycles_worst": cycles,
        "bram": 4,
        "dsp": dsp,
        "ff": 1000,
        "lut": 2000,
        "uram": 0,
        "slack_ns": 0.05,
        "requested_clock_period_ns": CLOCK_NS,
        "estimated_clock_period_ns": 3.20,
        "feedback": {
            "scopes": [
                {"scope_id": "loop_i", "pipeline_ii": ii, "trip_count": 64}
            ]
        },
    }


def _passing_csim() -> dict:
    return {
        "status": "passed",
        "ran": True,
        "success": True,
        "passed": True,
        "error": "",
    }


def _two_mm_source() -> str:
    return """\
#include "2mm.h"
extern "C" {
void kernel_2mm() {
#pragma HLS INTERFACE m_axi port=tmp offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem2
#pragma HLS INTERFACE m_axi port=C offset=slave bundle=gmem3
#pragma HLS INTERFACE m_axi port=D offset=slave bundle=gmem4
    const int ni = NI;
    const int nj = NJ;
    const int nk = NK;
    const int nl = NL;
    double local_A[NI][NK];
    double local_B[NK][NJ];
    double local_C[NJ][NL];
    double local_D[NI][NL];
    double local_tmp[NI][NJ];
#pragma HLS ARRAY_PARTITION variable=local_A cyclic factor=4 dim=2
#pragma HLS ARRAY_PARTITION variable=local_B cyclic factor=4 dim=1
#pragma HLS ARRAY_PARTITION variable=local_C cyclic factor=4 dim=2
#pragma HLS ARRAY_PARTITION variable=local_tmp cyclic factor=4 dim=2
load_A:
    for (int i = 0; i < ni; i++) {}
load_B:
    for (int k = 0; k < nk; k++) {
        for (int j = 0; j < nj; j++) {
#pragma HLS PIPELINE II=1
            local_B[k][j] = B[k][j];
        }
    }
load_C:
    for (int j = 0; j < nj; j++) {}
load_D:
    for (int i = 0; i < ni; i++) {}
    // ---- Phase 1: tmp = alpha * A * B ----
phase1:
    for (int i = 0; i < ni; i++) {
        for (int j = 0; j < nj; j++) {
#pragma HLS PIPELINE II=1
            double partial[4];
#pragma HLS ARRAY_PARTITION variable=partial complete dim=1
            for (int p = 0; p < 4; p++) {
#pragma HLS UNROLL
                partial[p] = 0.0;
            }
            for (int k = 0; k < nk; k += 4) {
#pragma HLS PIPELINE II=1
                for (int p = 0; p < 4; p++) {
#pragma HLS UNROLL
                    if (k + p < nk) {
                        partial[p] += local_A[i][k + p] * local_B[k + p][j];
                    }
                }
            }
            double sum = 0.0;
            for (int p = 0; p < 4; p++) {
#pragma HLS UNROLL
                sum += partial[p];
            }
            local_tmp[i][j] = alpha * sum;
        }
    }
    // ---- Phase 2: D = beta * D + tmp * C ----
phase2:
    for (int i = 0; i < ni; i++) {
        for (int l = 0; l < nl; l++) {
#pragma HLS PIPELINE II=1
            double partial[4];
#pragma HLS ARRAY_PARTITION variable=partial complete dim=1
            for (int p = 0; p < 4; p++) {
#pragma HLS UNROLL
                partial[p] = 0.0;
            }
            for (int j = 0; j < nj; j += 4) {
#pragma HLS PIPELINE II=1
                for (int p = 0; p < 4; p++) {
#pragma HLS UNROLL
                    if (j + p < nj) {
                        partial[p] += local_tmp[i][j + p] * local_C[j + p][l];
                    }
                }
            }
            double sum = 0.0;
            for (int p = 0; p < 4; p++) {
#pragma HLS UNROLL
                sum += partial[p];
            }
            local_D[i][l] = beta * local_D[i][l] + sum;
        }
    }
    // ---- Store phase: write outputs back ----
}
}
"""


class QorDesignSpaceTests(unittest.TestCase):
    def test_2mm_lane_and_step_domains_use_public_dimensions(self):
        header = """\
void kernel_2mm(double A[40 + 0][70 + 0], double B[70 + 0][50 + 0],
                double C[50 + 0][80 + 0], double D[40 + 0][80 + 0]);
"""
        lanes = validate_2mm_lane_values(
            header, (1, 2, 4, 5, 7, 10, 14, 25, 35, 50)
        )
        self.assertEqual((40, 50, 70, 80), (
            lanes["ni"], lanes["nj"], lanes["nk"], lanes["nl"]
        ))
        self.assertTrue(lanes["boundary_guarded_remainders"])
        with self.assertRaisesRegex(ValueError, r"exceed min\(NK, NJ\)=50"):
            validate_2mm_lane_values(header, (1, 51))

        tiled = validate_2mm_step_control_values(
            header, "tiling", (1, 2, 5, 7, 10, 14, 35, 70), fixed_lanes=1
        )
        self.assertEqual(70, tiled["nk"])
        with self.assertRaisesRegex(ValueError, "must divide NK=70"):
            validate_2mm_step_control_values(
                header, "doublebuffer", (1, 3, 10), fixed_lanes=1
            )

        localized, localization = localize_2mm_dimensions(
            _two_mm_source(), header
        )
        self.assertNotRegex(localized, r"\b(?:NI|NJ|NK|NL)\b")
        self.assertEqual(70, localization["dimensions"]["nk"])
        self.assertGreater(localization["replacement_counts"]["NK"], 0)

    def test_2mm_linked_lane_control_couples_both_reductions(self):
        linked, contract = build_2mm_linked_lanes_parent(
            _two_mm_source(), parent_lanes=4
        )
        self.assertIn("#define MM2_LANES 4", linked)
        self.assertEqual(10, contract["dependency_count"])
        self.assertEqual(2, contract["derived_dependency_count"])
        self.assertIn(
            "variable=local_C cyclic factor=MM2_LANES dim=1", linked
        )
        self.assertNotIn("double partial[4]", linked)
        self.assertEqual(2, linked.count("k + p < nk") + linked.count("j + p < nj"))
        self.assertNotRegex(
            linked,
            r"#pragma HLS PIPELINE II=1\s+double partial\[MM2_LANES\]",
        )
        knobs = qds.discover_qor_knobs(
            linked,
            factor_values=(1, 2, 4, 5, 7, 10),
            max_knobs=1,
            preferred_kinds=qds.STEP_PREFERRED_KINDS["unroll"],
        )
        self.assertEqual("MM2_LANES", knobs[0].name)
        self.assertEqual("compute_lanes", knobs[0].kind)
        self.assertEqual(10, len(knobs[0].dependent_lines))

    def test_2mm_step_controls_are_exact_and_source_coupled(self):
        controls = (
            ("pipeline", 1, 4, "MM2_COMPUTE_II", "pipeline_ii", 2),
            ("tiling", 10, 1, "MM2_K_TILE", "tile_size", 0),
            (
                "doublebuffer", 10, 1, "MM2_BUFFER_DEPTH",
                "buffer_tile_depth", 0,
            ),
            (
                "coalescing", 512, 4, "MM2_AXI_BITS",
                "interface_max_widen_bitwidth", 5,
            ),
            (
                "resource", 4, 4, "MM2_MUL_IMPL",
                "bind_op_implementation", 10,
            ),
        )
        for step, parent, lanes, name, kind, dependency_count in controls:
            with self.subTest(step=step):
                linked, contract = build_2mm_step_control_parent(
                    _two_mm_source(),
                    step=step,
                    parent_value=parent,
                    fixed_lanes=lanes,
                )
                self.assertIn(f"#define {name} {parent}", linked)
                self.assertEqual(dependency_count, contract["dependency_count"])
                knobs = qds.discover_qor_knobs(
                    linked,
                    factor_values=(1, 2, 4, 5, 10),
                    ii_values=(1, 2, 3, 4, 5, 8),
                    tile_values=(1, 2, 5, 7, 10, 14, 35, 70),
                    stream_depth_values=(1, 2, 5, 7, 10, 14, 35, 70),
                    widen_values=(64, 128, 256, 512, 1024),
                    preferred_kinds=qds.STEP_PREFERRED_KINDS[step],
                )
                knob = next(item for item in knobs if item.name == name)
                self.assertEqual(kind, knob.kind)
                self.assertGreater(len(knob.dependent_source_lines), 0)
                if step == "tiling":
                    self.assertIn("k_tile += MM2_K_TILE", linked)
                elif step == "doublebuffer":
                    self.assertIn("double b_ping[MM2_BUFFER_DEPTH][NJ]", linked)
                    self.assertNotIn("double local_B[NK][NJ]", linked)
                elif step == "resource":
                    self.assertEqual(
                        2,
                        linked.count(
                            "BIND_OP variable=lane_product op=dmul impl=fabric"
                        ),
                    )

    def test_gemm_lane_domain_requires_factors_of_nk(self):
        domain = validate_gemm_lane_values(
            "#define NK 80\n", (1, 2, 4, 5, 8, 10, 16, 20)
        )
        self.assertEqual(80, domain["nk"])
        self.assertTrue(domain["all_values_divide_nk"])
        with self.assertRaisesRegex(ValueError, "do not divide NK=80"):
            validate_gemm_lane_values("#define NK 80\n", (1, 3, 8))

    def test_gemm_linked_lane_control_coordinates_compute_and_memory(self):
        source = """\
#include "gemm.h"
void kernel_gemm() {
double local_A[80][80];
double local_B[80][80];
#pragma HLS ARRAY_PARTITION variable=local_A cyclic factor=2 dim=2
#pragma HLS ARRAY_PARTITION variable=local_B cyclic factor=2 dim=1
    // Load B
    for (int k = 0; k < 80; ++k) {}
    // Load C
    for (int i = 0; i < 80; ++i) {}
// Compute: C = alpha * A * B + beta * C
for (int i = 0; i < 80; ++i) {
  for (int j = 0; j < 80; ++j) {
#pragma HLS PIPELINE II=1
    double sum = 0.0;
    for (int k = 0; k < 80; ++k) {
#pragma HLS UNROLL factor=2
      sum += local_A[i][k] * local_B[k][j];
    }
  }
}
// Store C
}
"""
        linked, contract = build_gemm_linked_lanes_parent(source, parent_lanes=2)
        self.assertIn("#define GEMM_LANES 2", linked)
        self.assertEqual(4, linked.count("GEMM_LANES"))
        compute = linked.split("// Compute:", 1)[1]
        self.assertNotIn("#pragma HLS PIPELINE", compute)
        self.assertEqual(3, contract["dependency_count"])

        knobs = qds.discover_qor_knobs(
            linked,
            factor_values=(1, 2, 4, 8),
            max_knobs=1,
            preferred_kinds=qds.STEP_PREFERRED_KINDS["unroll"],
        )
        self.assertEqual(1, len(knobs))
        knob = knobs[0]
        self.assertEqual("compute_lanes", knob.kind)
        self.assertEqual("GEMM_LANES", knob.name)
        self.assertEqual(
            ["array_partition", "array_partition", "unroll"],
            list(knob.dependent_controls),
        )
        candidate = qds.apply_knob_values(linked, [(knob, 4)])
        self.assertIn("#define GEMM_LANES 4", candidate)
        self.assertEqual(4, candidate.count("GEMM_LANES"))

        striped, striped_contract = build_gemm_linked_lanes_parent(
            source,
            parent_lanes=2,
            architecture="striped_accumulators",
        )
        self.assertIn("double partial_sums[GEMM_LANES]", striped)
        self.assertIn("k += GEMM_LANES", striped)
        self.assertNotIn("sum += local_A", striped)
        self.assertEqual("striped_accumulators", striped_contract["architecture"])
        self.assertEqual(5, striped_contract["dependency_count"])
        self.assertEqual(1, striped_contract["derived_dependency_count"])
        self.assertEqual(
            "array_partition_complete",
            striped_contract["derived_dependencies"][0]["control"],
        )
        source_roles = {
            item["role"] for item in striped_contract["source_dependencies"]
        }
        self.assertTrue({"hls_directive", "array_extent", "loop_stride"}.issubset(
            source_roles
        ))
        striped_knob = qds.discover_qor_knobs(
            striped,
            factor_values=(1, 2, 4, 8),
            max_knobs=1,
            preferred_kinds=qds.STEP_PREFERRED_KINDS["unroll"],
        )[0]
        self.assertEqual(5, len(striped_knob.dependent_lines))
        self.assertEqual(10, len(striped_knob.dependent_source_lines))
        self.assertTrue({"array_extent", "loop_bound", "loop_stride"}.issubset(
            set(striped_knob.dependent_source_roles)
        ))

    def test_gemm_step_controls_are_exact_and_source_coupled(self):
        source = """\
#include "gemm.h"
extern "C" {
void kernel_gemm() {
#pragma HLS INTERFACE m_axi port=C offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem
double local_A[NI][NK];
double local_B[NK][NJ];
#pragma HLS ARRAY_PARTITION variable=local_A cyclic factor=2 dim=2
#pragma HLS ARRAY_PARTITION variable=local_B cyclic factor=2 dim=1
    // Load B
    for (int k = 0; k < 80; ++k) {}
    // Load C
    for (int i = 0; i < 80; ++i) {}
// Compute: C = alpha * A * B + beta * C
for (int i = 0; i < 80; ++i) {
  for (int j = 0; j < 80; ++j) {
#pragma HLS PIPELINE II=1
    double sum = 0.0;
    for (int k = 0; k < 80; ++k) {
#pragma HLS UNROLL factor=2
      sum += local_A[i][k] * local_B[k][j];
    }
  }
}
// Store C
}
}
"""
        controls = (
            ("pipeline", 5, "GEMM_COMPUTE_II", "pipeline_ii", 1),
            ("tiling", 16, "GEMM_K_TILE", "tile_size", 0),
            (
                "doublebuffer",
                8,
                "GEMM_BUFFER_DEPTH",
                "buffer_tile_depth",
                0,
            ),
            (
                "coalescing",
                512,
                "GEMM_AXI_BITS",
                "interface_max_widen_bitwidth",
                3,
            ),
            ("resource", 4, "GEMM_MUL_IMPL", "bind_op_implementation", 5),
        )
        for step, parent, name, kind, dependency_count in controls:
            with self.subTest(step=step):
                linked, contract = build_gemm_step_control_parent(
                    source,
                    step=step,
                    parent_value=parent,
                    fixed_lanes=8,
                )
                self.assertIn(f"#define {name} {parent}", linked)
                self.assertEqual(dependency_count, contract["dependency_count"])
                knobs = qds.discover_qor_knobs(
                    linked,
                    factor_values=(1, 2, 4, 8, 16),
                    ii_values=(1, 2, 4, 5, 8),
                    tile_values=(8, 16, 20, 40, 80),
                    stream_depth_values=(1, 2, 4, 8, 16),
                    widen_values=(64, 128, 256, 512, 1024),
                    preferred_kinds=qds.STEP_PREFERRED_KINDS[step],
                )
                knob = next(item for item in knobs if item.name == name)
                self.assertEqual(kind, knob.kind)
                self.assertGreater(len(knob.dependent_source_lines), 0)
                if step == "resource":
                    self.assertIn(
                        "#pragma HLS BIND_OP variable=lane_product op=dmul "
                        "impl=fabric // GEMM_MUL_IMPL=1",
                        linked,
                    )
                    self.assertIn(
                        "#pragma HLS BIND_OP variable=lane_product op=dmul "
                        "impl=primitivedsp // GEMM_MUL_IMPL=5",
                        linked,
                    )
                    self.assertIn(
                        "lane_product = local_A[i][k + lane]",
                        linked,
                    )

    def test_gemm_step_control_domains_reject_invalid_values(self):
        header = "#define NK 80\n"
        tiling = validate_gemm_step_control_values(
            header, "tiling", (8, 16, 40, 80), fixed_lanes=8
        )
        self.assertEqual([8, 16, 40, 80], tiling["validated_values"])
        with self.assertRaisesRegex(ValueError, "must divide NK=80"):
            validate_gemm_step_control_values(
                header, "tiling", (8, 20), fixed_lanes=8
            )
        with self.assertRaisesRegex(ValueError, "powers of two"):
            validate_gemm_step_control_values(
                header, "coalescing", (64, 192, 512), fixed_lanes=8
            )

    def test_linked_lane_realization_accepts_partial_unroll(self):
        log = """\
INFO: [HLS 214-186] Unrolling loop 'VITIS_LOOP_65_9' (/tmp/kernel.cpp:65:30) partially with a factor of 4
"""
        transformations = hls_feedback.parse_vitis_schedule_transformations(log)
        self.assertEqual("partial_unroll", transformations[0]["kind"])
        report = {"feedback": {"schedule_transformations": transformations}}
        realization = qds.compiler_knob_realization(report, [{
            "knob_id": "lanes",
            "kind": "compute_lanes",
            "name": "GEMM_LANES",
            "line": 3,
            "dependent_lines": [30, 31, 66],
            "dependent_controls": ["array_partition", "array_partition", "unroll"],
            "to": 4,
        }])
        knob = realization["knobs"][0]
        self.assertEqual("realized", knob["status"])
        self.assertEqual(4, knob["effective_value"])
        self.assertEqual("partially", knob["unroll_mode"])
        self.assertFalse(realization["any_overridden"])

        implied_log = """\
INFO: [HLS 214-291] Loop 'VITIS_LOOP_65_9' is marked as complete unroll implied by the pipeline pragma (/tmp/kernel.cpp:65:30)
INFO: [HLS 214-186] Unrolling loop 'VITIS_LOOP_65_9' (/tmp/kernel.cpp:65:30) completely with a factor of 4
"""
        implied_report = {"feedback": {"schedule_transformations": (
            hls_feedback.parse_vitis_schedule_transformations(implied_log)
        )}}
        implied = qds.compiler_knob_realization(implied_report, [{
            "knob_id": "lanes",
            "kind": "compute_lanes",
            "name": "GEMM_LANES",
            "line": 3,
            "dependent_lines": [30, 31, 66],
            "dependent_controls": ["array_partition", "array_partition", "unroll"],
            "to": 4,
        }])
        self.assertEqual(
            "realized_by_enclosing_pipeline", implied["knobs"][0]["status"]
        )
        self.assertFalse(implied["any_overridden"])

        rolled_report = {"feedback": {"schedule_transformations": [{
            "kind": "pipeline_result",
            "loop": "VITIS_LOOP_10_1",
            "target_ii": 1,
            "final_ii": 1,
        }]}}
        rolled = qds.compiler_knob_realization(rolled_report, [{
            "knob_id": "lanes",
            "kind": "compute_lanes",
            "name": "GEMM_LANES",
            "line": 3,
            "dependent_lines": [30, 31, 66],
            "dependent_controls": ["array_partition", "array_partition", "unroll"],
            "to": 1,
        }])
        rolled_knob = rolled["knobs"][0]
        self.assertEqual("realized_rolled", rolled_knob["status"])
        self.assertEqual(1.0, rolled_knob["effective_value"])
        self.assertTrue(rolled["all_observed"])

    def test_one_cycle_top_level_interval_overhead_is_not_a_bottleneck(self):
        base = {
            "scope_id": "kernel",
            "kind": "module",
            "name": "kernel",
            "latency_cycles": 100,
        }
        normal = hls_feedback.derive_bottleneck_records(
            [{**base, "interval": 101}], []
        )
        self.assertFalse(any(
            item["kind"] == "interval_exceeds_latency" for item in normal
        ))

        stalled = hls_feedback.derive_bottleneck_records(
            [{**base, "interval": 150}], []
        )
        restart_gap = next(
            item for item in stalled if item["kind"] == "interval_exceeds_latency"
        )
        self.assertEqual(50, restart_gap["metric"]["interval_overhead_cycles"])

    def test_compiler_realization_distinguishes_requested_and_effective_controls(self):
        log = """\
INFO: [HLS 214-291] Loop 'VITIS_LOOP_64_9' is marked as complete unroll implied by the pipeline pragma (/tmp/kernel.cpp:64:30)
INFO: [HLS 214-186] Unrolling loop 'VITIS_LOOP_64_9' (/tmp/kernel.cpp:64:30) completely with a factor of 80
INFO: [HLS 200-1470] Pipelining result : Target II = 4, Final II = 3, Depth = 3, loop 'VITIS_LOOP_33_1'
"""
        transformations = hls_feedback.parse_vitis_schedule_transformations(log)
        report = {"feedback": {"schedule_transformations": transformations}}
        unroll = qds.compiler_knob_realization(report, [{
            "knob_id": "unroll",
            "kind": "unroll_factor",
            "name": "unroll@L65",
            "line": 65,
            "to": 4,
        }])
        self.assertTrue(unroll["any_overridden"])
        self.assertEqual(
            "overridden_by_enclosing_pipeline", unroll["knobs"][0]["status"]
        )
        self.assertEqual(80, unroll["knobs"][0]["effective_value"])

        pipeline = qds.compiler_knob_realization(report, [{
            "knob_id": "pipeline",
            "kind": "pipeline_ii",
            "name": "pipeline_ii@L34",
            "line": 34,
            "to": 4,
        }])
        self.assertFalse(pipeline["any_overridden"])
        self.assertEqual("target_realized", pipeline["knobs"][0]["status"])
        self.assertEqual(4, pipeline["knobs"][0]["target_ii"])
        self.assertEqual(3, pipeline["knobs"][0]["achieved_ii"])
        self.assertFalse(pipeline["knobs"][0]["target_achieved"])

        linked_pipeline = qds.compiler_knob_realization(report, [{
            "knob_id": "linked_pipeline",
            "kind": "pipeline_ii",
            "name": "GEMM_COMPUTE_II",
            "line": 3,
            "dependent_lines": [34],
            "dependent_controls": ["pipeline"],
            "to": 4,
        }])
        self.assertTrue(linked_pipeline["knobs"][0]["observed"])
        self.assertEqual(4, linked_pipeline["knobs"][0]["target_ii"])

    def test_schema_export_keeps_qor_evidence_but_drops_code(self):
        compact = export_schema_jsonl._qor_design_sweep({
            "quality_repair": {
                "design_sweep": {
                    "enabled": True,
                    "winner_candidate_id": "qor-ofat-1",
                    "candidates": [
                        {
                            "candidate_id": "qor-ofat-1",
                            "code": "void workload() {}",
                            "metrics": {"latency_cycles_worst": 90},
                        }
                    ],
                }
            }
        })
        self.assertTrue(compact["enabled"])
        self.assertEqual("qor-ofat-1", compact["winner_candidate_id"])
        self.assertNotIn("code", compact["candidates"][0])

    def test_discovers_diverse_typed_knobs_and_multiple_interface_options(self):
        code = """\
#define TILE_M 16
static const int BLOCK_N = 8;
const int ORDINARY = 7;
void workload(int *a) {
#pragma HLS PIPELINE II=2
#pragma HLS PIPELINE II=4
#pragma HLS PIPELINE
#pragma HLS UNROLL
#pragma HLS ARRAY_PARTITION variable=a factor=4 dim=1
#pragma HLS INTERFACE m_axi port=a max_widen_bitwidth=512 num_read_outstanding=16 max_read_burst_length=64
}
"""
        limited = qds.discover_qor_knobs(code, max_knobs=4)
        self.assertEqual(
            ["pipeline_ii", "unroll_factor", "partition_factor", "tile_size"],
            [knob.kind for knob in limited],
        )

        all_knobs = qds.discover_qor_knobs(code)
        names = {knob.name for knob in all_knobs}
        kinds = {knob.kind for knob in all_knobs}
        self.assertIn("TILE_M", names)
        self.assertIn("BLOCK_N", names)
        self.assertNotIn("ORDINARY", names)
        self.assertIn("interface_max_widen_bitwidth", kinds)
        self.assertIn("interface_num_read_outstanding", kinds)
        self.assertIn("interface_max_read_burst_length", kinds)
        self.assertTrue(
            any(
                knob.kind == "pipeline_ii" and knob.current_label == "auto"
                for knob in all_knobs
            )
        )

    def test_discovers_step_native_tile_axi_and_resource_toggles(self):
        code = """\
void workload(float *a, float *b) {
const int tk = 16; // Tile size for the K dimension
#pragma HLS INTERFACE m_axi port=a offset=slave bundle=gmem
#pragma HLS DATAFLOW
#pragma HLS ARRAY_PARTITION variable=b complete dim=1
#pragma HLS BIND_OP variable=x op=mul impl=dsp
#pragma HLS BIND_STORAGE variable=b type=ram_2p impl=bram
#pragma HLS RESOURCE variable=a core=RAM_2P_BRAM
}
"""
        knobs = qds.discover_qor_knobs(code)
        kinds = {knob.kind for knob in knobs}
        self.assertTrue({
            "tile_size",
            "interface_max_widen_bitwidth",
            "dataflow_enabled",
            "partition_enabled",
            "bind_op_enabled",
            "bind_storage_enabled",
            "resource_enabled",
        }.issubset(kinds))

        tile = next(knob for knob in knobs if knob.kind == "tile_size")
        widened = next(
            knob for knob in knobs
            if knob.kind == "interface_max_widen_bitwidth"
        )
        dataflow = next(knob for knob in knobs if knob.kind == "dataflow_enabled")
        candidate = qds.apply_knob_values(
            code,
            [(tile, 8), (widened, 512), (dataflow, 0)],
        )
        self.assertIn("const int tk = 8", candidate)
        self.assertIn("max_widen_bitwidth=512", candidate)
        self.assertIn("#if 0\n#pragma HLS DATAFLOW\n#endif", candidate)

    def test_resource_latency_and_step_preference_are_typed(self):
        code = """\
void workload(float *a) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=4
#pragma HLS ARRAY_PARTITION variable=a cyclic factor=4 dim=1
#pragma HLS INTERFACE m_axi port=a offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=b offset=slave bundle=gmem
#pragma HLS BIND_OP variable=x op=mul impl=dsp latency=2
#pragma HLS BIND_STORAGE variable=a type=ram_2p impl=bram latency=2
#pragma HLS RESOURCE variable=a core=RAM_2P_BRAM latency=2
}
"""
        all_knobs = qds.discover_qor_knobs(code)
        kinds = {knob.kind for knob in all_knobs}
        self.assertTrue({
            "bind_op_latency",
            "bind_storage_latency",
            "resource_latency",
        }.issubset(kinds))
        self.assertNotIn("bind_op_enabled", kinds)

        focused = qds.discover_qor_knobs(
            code,
            max_knobs=2,
            preferred_kinds=qds.STEP_PREFERRED_KINDS["coalescing"],
        )
        self.assertEqual(2, len(focused))
        self.assertEqual(
            ["interface_max_widen_bitwidth", "interface_max_widen_bitwidth"],
            [knob.kind for knob in focused],
        )

    def test_ofat_and_interaction_candidates_always_use_frozen_parent(self):
        parent = """\
#define TILE_M 8
void workload(int *a) {
#pragma HLS PIPELINE II=2
#pragma HLS ARRAY_PARTITION variable=a factor=2 dim=1
}
"""
        knobs = qds.discover_qor_knobs(
            parent,
            factor_values=(1, 2, 4),
            ii_values=(1, 2, 4),
            tile_values=(4, 8, 16),
        )
        candidates = qds.build_ofat_candidates(parent, knobs, max_candidates=6)
        self.assertEqual(6, len(candidates))
        self.assertTrue(all(len(item["changed_knobs"]) == 1 for item in candidates))
        self.assertTrue(all(item["code"] != parent for item in candidates))
        self.assertEqual(parent, "#define TILE_M 8\nvoid workload(int *a) {\n#pragma HLS PIPELINE II=2\n#pragma HLS ARRAY_PARTITION variable=a factor=2 dim=1\n}\n")

        preferred = {
            knobs[0].knob_id: knobs[0].candidate_values[0],
            knobs[1].knob_id: knobs[1].candidate_values[0],
        }
        interactions = qds.build_interaction_candidates(
            parent,
            knobs,
            preferred,
            max_candidates=1,
        )
        self.assertEqual(1, len(interactions))
        self.assertEqual(2, len(interactions[0]["changed_knobs"]))
        self.assertEqual(hashlib.sha256(interactions[0]["code"].encode()).hexdigest(),
                         interactions[0]["code_sha256"])

    def test_pareto_and_trends_include_parent_observation(self):
        code = "void workload() {\n#pragma HLS PIPELINE II=1\n}\n"
        knob = qds.discover_qor_knobs(code, ii_values=(1, 2, 4))[0]
        parent = {
            "candidate_id": "frozen_parent",
            "feasible": True,
            "metrics": qds.extract_qor_metrics(_report(100, dsp=8, ii=1)),
        }
        records = [
            {
                "candidate_id": "ii2",
                "feasible": True,
                "changed_knobs": [{"knob_id": knob.knob_id, "to": 2}],
                "metrics": qds.extract_qor_metrics(_report(120, dsp=7, ii=2)),
            },
            {
                "candidate_id": "ii4",
                "feasible": True,
                "changed_knobs": [{"knob_id": knob.knob_id, "to": 4}],
                "metrics": qds.extract_qor_metrics(_report(160, dsp=6, ii=4)),
            },
        ]
        trends = qds.summarize_knob_trends(records, [knob], parent=parent)
        self.assertEqual([1.0, 2.0, 4.0], trends[0]["tested_values"])
        self.assertEqual(0, trends[0]["monotonicity_violations"])
        self.assertAlmostEqual(1.0, trends[0]["spearman_value_vs_worst_cycles"])
        self.assertTrue(trends[0]["observations"][0]["is_parent"])
        self.assertTrue(trends[0]["observable_synthesis_response"])

        pareto_records = [
            parent,
            records[0],
            {
                "candidate_id": "dominated",
                "feasible": True,
                "metrics": qds.extract_qor_metrics(_report(180, dsp=12)),
            },
        ]
        frontier = qds.pareto_candidate_ids(pareto_records)
        self.assertNotIn("dominated", frontier)
        self.assertIn("frozen_parent", frontier)

    def test_trend_marks_source_changes_with_identical_synthesis_as_inactive(self):
        code = "void workload() {\n#pragma HLS UNROLL factor=2\n}\n"
        knob = qds.discover_qor_knobs(code, factor_values=(1, 2, 4))[0]
        metrics = qds.extract_qor_metrics(_report(100, dsp=8, ii=3))
        parent = {
            "candidate_id": "frozen_parent",
            "code_sha256": "parent",
            "feasible": True,
            "metrics": metrics,
        }
        records = [{
            "candidate_id": "factor4",
            "code_sha256": "candidate",
            "feasible": True,
            "changed_knobs": [{"knob_id": knob.knob_id, "to": 4}],
            "metrics": metrics,
        }]
        trend = qds.summarize_knob_trends(records, [knob], parent=parent)[0]
        self.assertFalse(trend["observable_synthesis_response"])
        self.assertEqual(
            "source_changed_no_observable_synthesis_response",
            trend["realization_status"],
        )


class _FakeOrchestrator:
    def __init__(self):
        self.gpt_model = "unused"
        self.hls_code = "#define TILE_M 8\nvoid workload() {}\n"
        self.synth_report = _report(100)
        self.generated_csim = _passing_csim()
        self.generated_cosim = None
        self.testbench_code = "int main() { return 0; }"
        self.part = PART
        self.clock_ns = CLOCK_NS
        self.vitis_version = "2023.2"
        self.seen_parent_codes: list[str] = []

    @staticmethod
    def _candidate_report_sha256(report):
        return c2hls.C2HLSOrchestrator._candidate_report_sha256(report)

    @staticmethod
    def _best_so_far_score(report):
        return c2hls.C2HLSOrchestrator._best_so_far_score(report)

    def _evaluate_qor_candidate(self, code, label, metadata):
        self.seen_parent_codes.append(self.hls_code)
        value = int(re.search(r"TILE_M\s+(\d+)", code).group(1))
        cycles = {4: 120, 16: 70}[value]
        report = _report(cycles, dsp=8 + value // 4)
        return {
            "success": True,
            "code": code,
            "report": report,
            "csim": _passing_csim(),
            "cosim": None,
            "feasibility": {
                "feasible": True,
                "correctness_ok": True,
                "resource_fit": True,
                "timing_met": True,
                "reasons": [],
            },
            "error": "",
            "event": {
                "qor_evaluation_index": len(self.seen_parent_codes) - 1,
                "failure_class": None,
            },
        }


class _ResourceTieOrchestrator(_FakeOrchestrator):
    def _evaluate_qor_candidate(self, code, label, metadata):
        self.seen_parent_codes.append(self.hls_code)
        value = int(re.search(r"TILE_M\s+(\d+)", code).group(1))
        report = _report(100, dsp={4: 4, 16: 12}[value])
        return {
            "success": True,
            "code": code,
            "report": report,
            "csim": _passing_csim(),
            "cosim": None,
            "feasibility": {
                "feasible": True,
                "correctness_ok": True,
                "resource_fit": True,
                "timing_met": True,
                "reasons": [],
            },
            "error": "",
            "event": {
                "qor_evaluation_index": len(self.seen_parent_codes) - 1,
                "failure_class": None,
            },
        }


class _CompilerOverrideOrchestrator(_FakeOrchestrator):
    def __init__(self):
        super().__init__()
        self.hls_code = """\
void workload(double a[8]) {
  for (int i = 0; i < 8; ++i) {
#pragma HLS PIPELINE II=1
    for (int j = 0; j < 8; ++j) {
#pragma HLS UNROLL factor=2
      a[i] += j;
    }
  }
}
"""

    def _evaluate_qor_candidate(self, code, label, metadata):
        self.seen_parent_codes.append(self.hls_code)
        report = _report(100, dsp=8, ii=3)
        report["feedback"]["schedule_transformations"] = (
            hls_feedback.parse_vitis_schedule_transformations("""\
INFO: [HLS 214-291] Loop 'VITIS_LOOP_4_2' is marked as complete unroll implied by the pipeline pragma (/tmp/kernel.cpp:4:5)
INFO: [HLS 214-186] Unrolling loop 'VITIS_LOOP_4_2' (/tmp/kernel.cpp:4:5) completely with a factor of 8
""")
        )
        return {
            "success": True,
            "code": code,
            "report": report,
            "csim": _passing_csim(),
            "cosim": None,
            "feasibility": {
                "feasible": True,
                "correctness_ok": True,
                "resource_fit": True,
                "timing_met": True,
                "reasons": [],
            },
            "error": "",
            "event": {
                "qor_evaluation_index": len(self.seen_parent_codes) - 1,
                "failure_class": None,
            },
        }


class QualityRepairIntegrationTests(unittest.TestCase):
    def test_design_sweep_stops_testing_compiler_overridden_knob(self):
        orch = _CompilerOverrideOrchestrator()
        orch.qor_parent_origin = {"step_name": "unroll"}
        agent = c2hls.QualityRepairAgent(orch)
        environment = {
            c2hls.QOR_DESIGN_SWEEP_ENV: "1",
            c2hls.QOR_SWEEP_MAX_KNOBS_ENV: "1",
            c2hls.QOR_SWEEP_MAX_CANDIDATES_ENV: "3",
            c2hls.QOR_SWEEP_VALUES_ENV: "1,2,4,8",
            c2hls.QOR_SWEEP_INTERACTIONS_ENV: "0",
        }
        with patch.dict(os.environ, environment, clear=False):
            summary = agent.run_design_sweep()

        self.assertEqual(1, summary["candidate_count"])
        self.assertEqual(2, summary["skipped_candidate_count"])
        self.assertEqual(1, summary["compiler_override_candidate_count"])
        self.assertEqual(1, len(orch.seen_parent_codes))
        self.assertIn("compiler-overridden", summary["winner_explanation"])

    def test_design_sweep_with_no_explicit_knobs_retains_parent(self):
        orch = _FakeOrchestrator()
        orch.hls_code = "void workload() {}\n"
        agent = c2hls.QualityRepairAgent(orch)
        with patch.dict(
            os.environ,
            {c2hls.QOR_DESIGN_SWEEP_ENV: "1"},
            clear=False,
        ):
            summary = agent.run_design_sweep()
        self.assertTrue(summary["attempted"])
        self.assertFalse(summary["applied"])
        self.assertEqual(0, summary["candidate_count"])
        self.assertEqual("frozen_parent", summary["winner_candidate_id"])
        self.assertTrue(summary["parent"]["pareto_frontier"])

    def test_design_sweep_refuses_parent_without_passing_csim(self):
        orch = _FakeOrchestrator()
        orch.generated_csim = None
        agent = c2hls.QualityRepairAgent(orch)
        with patch.dict(
            os.environ,
            {c2hls.QOR_DESIGN_SWEEP_ENV: "1"},
            clear=False,
        ):
            summary = agent.run_design_sweep()
        self.assertFalse(summary["attempted"])
        self.assertIn("passing CSim", summary["reason"])
        self.assertEqual([], orch.seen_parent_codes)

    def test_design_sweep_promotes_best_feasible_candidate_after_frozen_evaluation(self):
        orch = _FakeOrchestrator()
        orch.qor_parent_origin = {
            "step_name": "tiling",
            "step_index": 0,
            "source": "step",
        }
        parent_code = orch.hls_code
        agent = c2hls.QualityRepairAgent(orch)
        environment = {
            c2hls.QOR_DESIGN_SWEEP_ENV: "1",
            c2hls.QOR_SWEEP_MAX_KNOBS_ENV: "1",
            c2hls.QOR_SWEEP_MAX_CANDIDATES_ENV: "2",
            c2hls.QOR_SWEEP_TILE_VALUES_ENV: "4,8,16",
            c2hls.QOR_SWEEP_VALUES_ENV: "1,2,4",
            c2hls.QOR_SWEEP_II_VALUES_ENV: "1,2,4",
            c2hls.QOR_SWEEP_INTERACTIONS_ENV: "0",
        }
        with patch.dict(os.environ, environment, clear=False):
            summary = agent.run_design_sweep()

        self.assertTrue(summary["attempted"])
        self.assertTrue(summary["applied"])
        self.assertEqual(2, summary["candidate_count"])
        self.assertEqual(
            list(qds.STEP_PREFERRED_KINDS["tiling"]),
            summary["configuration"]["preferred_knob_kinds"],
        )
        self.assertEqual("tiling", summary["parent_origin"]["step_name"])
        self.assertEqual(70.0, summary["winner_metrics"]["latency_cycles_worst"])
        self.assertIn("TILE_M 16", orch.hls_code)
        self.assertTrue(all(code == parent_code for code in orch.seen_parent_codes))
        self.assertIsNone(orch.generated_cosim)
        winner = next(
            item
            for item in summary["candidates"]
            if item["candidate_id"] == summary["winner_candidate_id"]
        )
        changed_id = winner["changed_knobs"][0]["knob_id"]
        self.assertNotIn(changed_id, winner["fixed_knob_values"])
        self.assertEqual(16, winner["effective_knob_values"][changed_id])

    def test_design_sweep_uses_resources_to_break_exact_latency_tie(self):
        orch = _ResourceTieOrchestrator()
        agent = c2hls.QualityRepairAgent(orch)
        environment = {
            c2hls.QOR_DESIGN_SWEEP_ENV: "1",
            c2hls.QOR_SWEEP_MAX_KNOBS_ENV: "1",
            c2hls.QOR_SWEEP_MAX_CANDIDATES_ENV: "2",
            c2hls.QOR_SWEEP_TILE_VALUES_ENV: "4,8,16",
            c2hls.QOR_SWEEP_INTERACTIONS_ENV: "0",
        }
        with patch.dict(os.environ, environment, clear=False):
            summary = agent.run_design_sweep()

        self.assertTrue(summary["applied"])
        self.assertEqual(100.0, summary["winner_metrics"]["latency_cycles_worst"])
        self.assertEqual(4.0, summary["winner_metrics"]["dsp"])
        self.assertIn("TILE_M 4", orch.hls_code)
        self.assertIn("aggregate-resource tie-break", summary["winner_explanation"])

    def test_qor_candidate_forces_csim_before_csynth_and_never_cosim(self):
        orch = c2hls.C2HLSOrchestrator.__new__(c2hls.C2HLSOrchestrator)
        orch.qor_synthesis_eval_count = 0
        orch.qor_synthesis_eval_events = []
        orch.synthesis_eval_count = 0
        orch.synthesis_eval_events = []
        orch.synthesis_eval_budget = 3
        orch.llm_usage_events = []
        orch.llm_candidate_request_count = 0
        orch.llm_candidate_budget = 5
        orch.selected_winner_cosim_count = 0
        orch.post_route_implementation_count = 0
        orch.header_code = ""
        orch.header_name = "kernel.h"
        orch.extra_files = []
        orch.translated_hls_top = "workload"
        orch.part = PART
        orch.clock_ns = CLOCK_NS
        orch.testbench_code = "int main() { return 0; }"
        orch.cosim_depths = {}
        orch.independent_golden_output = ""
        orch.independent_golden_specs = {}
        outcome = {
            "synth": {
                "success": True,
                "ran": True,
                "report": _report(90),
                "error": "",
            },
            "csim": _passing_csim(),
            "cosim": None,
        }
        with (
            patch.object(c2hls, "compile_check_cpp", return_value=(True, "")),
            patch.object(
                c2hls,
                "_run_synth_csim_cosim",
                return_value=outcome,
            ) as runner,
        ):
            result = orch._evaluate_qor_candidate(
                "void workload() {}", "qor", {"candidate_id": "qor-1"}
            )

        self.assertTrue(result["success"])
        self.assertEqual(1, orch.qor_synthesis_eval_count)
        self.assertTrue(runner.call_args.kwargs["correctness_first_override"])
        self.assertFalse(runner.call_args.kwargs["run_cosim_check"])
        self.assertEqual("csynth_latency_cycles_worst",
                         result["event"]["latency_source"])
        self.assertFalse(result["event"]["selected_for_executed_cosim"])
        root = {
            "llm_usage": orch._llm_usage_summary(),
            "synthesis_evaluations": orch._synthesis_evaluation_summary(),
            **orch._tool_call_attribution(),
            "selected_code_sha256": None,
            "cosim_target_code_sha256": None,
        }
        contract = run_agentic_sweep._candidate_telemetry_contract(root)
        self.assertTrue(contract["complete"])
        self.assertTrue(contract["qor_attribution_complete"])
        self.assertEqual(1, contract["qor_synthesis_count"])
        self.assertEqual(1, root["synthesis_evaluation_count"])
        self.assertEqual(0, root["llm_candidate_synthesis_evaluation_count"])

    def test_qor_candidate_records_csim_timeout_without_counting_synthesis(self):
        orch = c2hls.C2HLSOrchestrator.__new__(c2hls.C2HLSOrchestrator)
        orch.qor_synthesis_eval_count = 0
        orch.qor_synthesis_eval_events = []
        orch.synthesis_eval_count = 0
        orch.synthesis_eval_events = []
        orch.synthesis_eval_budget = 1
        orch.llm_usage_events = []
        orch.llm_candidate_request_count = 0
        orch.header_code = ""
        orch.header_name = "kernel.h"
        orch.extra_files = []
        orch.translated_hls_top = "workload"
        orch.part = PART
        orch.clock_ns = CLOCK_NS
        orch.testbench_code = "int main() { return 0; }"
        orch.cosim_depths = {}
        orch.independent_golden_output = ""
        orch.independent_golden_specs = {}
        outcome = {
            "synth": {
                "success": False,
                "ran": False,
                "skipped": True,
                "status": "not_run",
                "report": {},
                "error": "Synthesis skipped after CSim timeout",
            },
            "csim": {
                "status": "timeout",
                "ran": True,
                "success": False,
                "passed": False,
                "error": "Csim timed out after 180s",
            },
            "cosim": None,
        }
        with (
            patch.object(c2hls, "compile_check_cpp", return_value=(True, "")),
            patch.object(c2hls, "_run_synth_csim_cosim", return_value=outcome),
        ):
            result = orch._evaluate_qor_candidate(
                "void workload() {}", "qor", {"candidate_id": "qor-timeout"}
            )

        self.assertFalse(result["success"])
        self.assertEqual("csim_timeout", result["event"]["failure_class"])
        self.assertEqual("timeout", result["event"]["status"])
        self.assertTrue(result["event"]["timed_out"])
        self.assertFalse(result["event"]["tool_failure"])
        self.assertEqual(0, orch.qor_synthesis_eval_count)

    def test_sweep_runner_adds_qor_budget_after_paper_profile(self):
        env = {
            "C2HLS_QOR_DESIGN_SWEEP": "1",
            "C2HLS_QOR_SWEEP_MAX_CANDIDATES": "8",
            "C2HLS_SYNTHESIS_EVAL_BUDGET": "5",
        }
        profile = {"name": "paper_reference_blind_v1"}
        with patch.dict(os.environ, env, clear=True):
            run_agentic_sweep._apply_post_profile_qor_budget(profile)
            self.assertEqual("13", os.environ["C2HLS_SYNTHESIS_EVAL_BUDGET"])
        self.assertEqual(
            13,
            profile["post_profile_overrides"]["qor_synthesis_budget"][
                "effective_total"
            ],
        )


if __name__ == "__main__":
    unittest.main()
