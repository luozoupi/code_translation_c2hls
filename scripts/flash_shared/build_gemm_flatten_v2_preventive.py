#!/usr/bin/env python3
"""Build gemm_flatten_v2 from gemm_flatten_v1. Does not write v1 or the 90 pack.

Fixes the two flash gaps vs the manual 16-wide / independent-acc rewrite:
  1. max_widen=512 with a 32-bit data_t requires LANES=16 load/store and
     matching partition — not k+=4 / factor=4.
  2. float MAC on one acc[i][j] every k is II=4. Advance k by 4 in one
     PIPELINE body and update named acc0..acc3 each iteration. switch(k&3)
     does not close II (HLS still reports distance=1).

Default mmflow / aav_n_gf still points at v1.
"""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PKG = REPO / "hls_full_optimization_skills_schema_1_1_package"
V1_NAME = "skills_ii_target_miss_solutions_added(90skills)_gemm_flatten_v1.json"
V2_NAME = "skills_ii_target_miss_solutions_added(90skills)_gemm_flatten_v2.json"
V1_PATH = PKG / V1_NAME
V2_PATH = PKG / V2_NAME

NEW_IDS = (
    "hls-axi-load-step-matches-widen-lanes",
    "avoid-axi-widen-512-with-narrow-unroll",
    "hls-fp-mac-k-step-independent-acc-banks",
    "avoid-fp-mac-recurrence-single-acc",
    "avoid-switch-k-mod-named-acc-banks",
)

PATCH_IDS = (
    "axi-burst-coalescing-narrow-safe",
    "axi-burst-widening-512",
    "prompt-unroll",
    "hls-coalescing-512-compound-transform",
    "hls-coalescing-compute-lane-parallelism",
    "hls-coalescing-contiguous-access-rewrite",
    "hls-coalescing-lane-parallel-reduction",
    "hls-coalescing-partition-lane-buffers",
    "hls-avoid-coalescing-interface-only",
    "hls-pipeline-handle-true-recurrence",
    "hls-unroll-reduction-partial-sums",
    "ii-reduction-lane-partial-tree",
    "hls-avoid-serial-fp-acc-under-full-k-unroll",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _skill_by_id(skills: list[dict], sid: str) -> dict:
    for skill in skills:
        if skill.get("id") == sid:
            return skill
    raise KeyError(sid)


def _append_unique(seq: list, items: list) -> list:
    out = list(seq)
    for item in items:
        if item not in out:
            out.append(item)
    return out


def _u280_meta() -> dict:
    return {
        "applicable_versions": ["2023.2"],
        "applicable_fpgas": ["xcu280-fsvh2892-2L-e"],
        "mean_advantage": 0.0,
        "occurrences": 0,
        "last_used_at": None,
        "sec_pass": 0,
        "origin": "flash_redo3_vs_manual_wide_load_ii4_20260917",
    }


def _new_skills() -> list[dict]:
    meta = _u280_meta()
    return [
        {
            "id": "hls-axi-load-step-matches-widen-lanes",
            "kind": "transformation_operator",
            "confidence": "high",
            **meta,
            "pattern": (
                "m_axi has max_widen_bitwidth=512 on a 32-bit (or similarly narrow) "
                "pointer ABI, but the load/store pipeline still steps by 4 (or any "
                "factor smaller than LANES) and the local buffer is banked with that "
                "same small factor on the consecutive dimension"
            ),
            "strategy": (
                "LANES = 512 / element_bitwidth. For data_t float/int32 that is 16, "
                "not 4. The load/store inner loop must step by LANES with UNROLL, "
                "and ARRAY_PARTITION cyclic factor on the consecutive dim must equal "
                "LANES. max_widen_bitwidth=512 without rewriting the walk is incomplete. "
                "Compute row/col banking (16x32) is a different factor and must not "
                "replace the consecutive-dim LANES bank."
            ),
            "required_steps": [
                "compute LANES = 512 / (8 * sizeof(data_t))",
                "rewrite load/store: for (k = 0; k < K; k += LANES) PIPELINE II=1, inner UNROLL l < LANES",
                "ARRAY_PARTITION cyclic factor=LANES on the consecutive (k) dim of the local buffer",
                "keep compute-side row banking separate (it feeds parallel MACs, not the AXI beat)",
                "target load trip ≈ (rows * cols) / LANES, not one element per cycle",
            ],
            "guards": [
                "do not change the public scalar pointer ABI to ap_uint<512>",
                "do not use unroll 2/4/8 as the AXI load default when LANES=16",
                "partition factor on the consecutive dim must match LANES, not a smaller guess",
                "this skill does not require fused A+B; separate load_A / load_B is also legal",
            ],
            "bottleneck_kinds": [
                "underutilized_wide_memory",
                "compute_not_scaled_after_widening",
                "memory_bandwidth",
                "axi_burst_failed",
            ],
            "tags": [
                "hls",
                "coalescing",
                "512-bit",
                "LANES",
                "array-partition",
                "preventive",
                "gemm_flatten_v2",
                "kernel-independent",
            ],
            "template": (
                "const int LANES = 512 / (8 * sizeof(data_t));  // 16 for float\n"
                "#pragma HLS ARRAY_PARTITION variable=A_loc cyclic factor=LANES dim=2\n"
                "load_k: for (int k = 0; k < K; k += LANES) {\n"
                "#pragma HLS PIPELINE II=1\n"
                "  load_l: for (int l = 0; l < LANES; ++l) {\n"
                "#pragma HLS UNROLL\n"
                "    A_loc[i][k + l] = A[i][k + l];\n"
                "  }\n"
                "}\n"
                "// BAD: k += 4 with factor=4 while max_widen_bitwidth=512 on 32-bit data_t."
            ),
        },
        {
            "id": "avoid-axi-widen-512-with-narrow-unroll",
            "kind": "avoid_rule",
            "confidence": "avoid",
            **meta,
            "pattern": (
                "INTERFACE max_widen_bitwidth=512 is present, but the load/store "
                "pipeline still uses k += 2/4/8 (or UNROLL factor 2/4/8) and the "
                "consecutive-dim partition factor is smaller than LANES = 512 / "
                "element_bitwidth"
            ),
            "strategy": (
                "Reject that kernel as coalesced. Widening the adapter without a "
                "LANES-wide walk leaves trip at 1024 for a 64x64 float instead of 256. "
                "Rewrite the load/store body and matching partition before claiming "
                "512-bit I/O is done."
            ),
            "required_steps": [
                "detect max_widen_bitwidth=512 together with k += 4 (or factor=4) on a 32-bit ABI",
                "do not accept INTERFACE-only widening as complete coalescing",
                "route to hls-axi-load-step-matches-widen-lanes",
                "re-synth and confirm load trip dropped by LANES (64x64 float: 1024 → 256)",
            ],
            "guards": [
                "LANES is AXI packing, not PE_BLK / tile width",
                "do not widen by changing the top pointer type",
            ],
            "bottleneck_kinds": [
                "underutilized_wide_memory",
                "compute_not_scaled_after_widening",
                "memory_bandwidth",
            ],
            "tags": [
                "avoid",
                "coalescing",
                "512-bit",
                "preventive",
                "gemm_flatten_v2",
                "kernel-independent",
            ],
            "template": (
                "// BAD: 512-bit pragma + 4-wide walk\n"
                "#pragma HLS INTERFACE m_axi port=A max_widen_bitwidth=512\n"
                "#pragma HLS ARRAY_PARTITION variable=A_loc cyclic factor=4 dim=2\n"
                "for (int k = 0; k < K; k += 4) { PIPELINE II=1; UNROLL l<4; A_loc[i][k+l]=A[i][k+l]; }\n"
                "// GOOD: LANES=16 walk + factor=16 on the consecutive dim."
            ),
        },
        {
            "id": "hls-fp-mac-k-step-independent-acc-banks",
            "kind": "transformation_operator",
            "confidence": "high",
            **meta,
            "pattern": (
                "a pipelined k-reduction updates the same float/double acc[i][j] "
                "(or C[i][j]) every iteration, so Vitis reports II=4 from the FP "
                "add/MAC recurrence even when the spatial tile is already 16x32"
            ),
            "strategy": (
                "Do not accept II=4 on that reduction. Advance k by the FP recurrence "
                "distance (typically 4) inside one PIPELINE II=1 iteration and update "
                "named accumulators acc0, acc1, acc2, acc3 in that same body, then "
                "tree-add them at store. Each bank is written once per iteration, so "
                "the scheduler sees no carried dep. Full UNROLL of k into an adder "
                "tree is also legal when DSP stays under the device cap. "
                "Do NOT use switch(k & 3) / acc[k%4] — HLS still reports distance=1."
            ),
            "required_steps": [
                "keep spatial MACs (tile rows x cols) as they are",
                "declare named acc0, acc1, acc2, acc3 with complete partition",
                "red_k: for (k = 0; k < K; k += 4) PIPELINE II=1",
                "in that body, UNROLL the spatial MAC into acc0..acc3 using A/B at k, k+1, k+2, k+3",
                "store: result = (acc0+acc1)+(acc2+acc3) — do not fold back into one acc during k",
                "re-synth until red_k Final II=1",
            ],
            "guards": [
                "do not use switch(k&3) or acc[k%4][i][j] — that form stayed II=4",
                "do not add DEPENDENCE inter false on a true FP MAC recurrence",
                "do not treat more PE_BLK / more DSPs as the II=4 fix",
                "preserve csim vs the reference accumulation within the harness tolerance",
            ],
            "bottleneck_kinds": [
                "ii_target_miss",
                "pipeline_blocked",
                "true_loop_carried_dep",
                "recurrence_limited_ii",
                "reduction_loop",
            ],
            "tags": [
                "hls",
                "ii",
                "fp-mac",
                "partial-sums",
                "preventive",
                "gemm_flatten_v2",
                "kernel-independent",
            ],
            "template": (
                "data_t acc0[TI][TJ], acc1[TI][TJ], acc2[TI][TJ], acc3[TI][TJ];\n"
                "#pragma HLS ARRAY_PARTITION variable=acc0 complete dim=0\n"
                "#pragma HLS ARRAY_PARTITION variable=acc1 complete dim=0\n"
                "#pragma HLS ARRAY_PARTITION variable=acc2 complete dim=0\n"
                "#pragma HLS ARRAY_PARTITION variable=acc3 complete dim=0\n"
                "red_k: for (int k = 0; k < K; k += 4) {\n"
                "#pragma HLS PIPELINE II=1\n"
                "  // update acc0 from k, acc1 from k+1, acc2 from k+2, acc3 from k+3\n"
                "  // in this same iteration (UNROLL spatial i,j). Not switch(k&3).\n"
                "}\n"
                "// store: C[...] = (acc0+acc1)+(acc2+acc3);"
            ),
        },
        {
            "id": "avoid-fp-mac-recurrence-single-acc",
            "kind": "avoid_rule",
            "confidence": "avoid",
            **meta,
            "pattern": (
                "PIPELINE II=1 on a k-loop that does acc[i][j] += a*b (or C[i][j] +=) "
                "every k. Csynth Final II=4 from the float MAC recurrence. Spatial "
                "unroll (16x32) may already be present; DSP may already be hundreds. "
                "csim and csynth still pass"
            ),
            "strategy": (
                "Reject the kernel as optimized. II=4 on the reduction is not a legal "
                "endpoint. Route to hls-fp-mac-k-step-independent-acc-banks. More DSPs "
                "or a wider tile under II=4 does not close II."
            ),
            "required_steps": [
                "detect red_k / compute_k PIPELINE with acc[i][j] += every k",
                "read csynth: pipelined yes, Final II=4, HLS 200-880 carried dependence",
                "do not accept csim+csynth success as done",
                "rewrite with k += 4 and named acc0..acc3 updated in one iteration",
            ],
            "guards": [
                "DEPENDENCE inter false is not the fix",
                "DSP fill / PE_BLK widening under II=4 is not the fix",
                "inherently serial integer recurrences are out of scope",
            ],
            "bottleneck_kinds": [
                "ii_target_miss",
                "pipeline_blocked",
                "true_loop_carried_dep",
                "recurrence_limited_ii",
            ],
            "tags": [
                "avoid",
                "ii",
                "fp-mac",
                "preventive",
                "gemm_flatten_v2",
                "kernel-independent",
            ],
            "template": (
                "// BAD: one acc bank, k++\n"
                "red_k: for (int k = 0; k < K; ++k) {\n"
                "#pragma HLS PIPELINE II=1\n"
                "  acc[i][j] += a_frag[i] * b_frag[j];  // Final II=4\n"
                "}\n"
                "// GOOD: k += 4, acc0..acc3 all written in this iteration."
            ),
        },
        {
            "id": "avoid-switch-k-mod-named-acc-banks",
            "kind": "avoid_rule",
            "confidence": "avoid",
            **meta,
            "pattern": (
                "the rewrite splits acc0..acc3 but still uses for (k=0; k<K; ++k) "
                "PIPELINE with switch(k&3) / acc[k%4], so only one bank is written "
                "per iteration. HLS reports carried dependence distance=1 and Final II=4"
            ),
            "strategy": (
                "Reject switch(k&3) as the II=4 fix. Distance-4 is not visible to the "
                "scheduler in that form (observed: II=4 on acc2). Emit k += 4 and "
                "update all four named banks in the same pipeline iteration."
            ),
            "required_steps": [
                "detect switch(k & 3) or acc[k%4] under a k++ PIPELINE",
                "do not treat four named arrays as success unless k advances by 4 in the body",
                "route to hls-fp-mac-k-step-independent-acc-banks",
            ],
            "guards": [
                "named acc0..acc3 are necessary but not sufficient",
                "do not add DEPENDENCE inter false to paper over the switch",
            ],
            "bottleneck_kinds": [
                "ii_target_miss",
                "pipeline_blocked",
                "true_loop_carried_dep",
            ],
            "tags": [
                "avoid",
                "ii",
                "fp-mac",
                "preventive",
                "gemm_flatten_v2",
                "kernel-independent",
            ],
            "template": (
                "// BAD: k++ plus switch — still II=4\n"
                "for (int k = 0; k < K; ++k) {\n"
                "#pragma HLS PIPELINE II=1\n"
                "  switch (k & 3) { case 0: acc0[i][j] += ...; break; /* acc1/2/3 */ }\n"
                "}\n"
                "// GOOD: k += 4; write acc0, acc1, acc2, acc3 in this iteration."
            ),
        },
    ]


def _patch_existing(skills: list[dict]) -> None:
    lanes_note = (
        "LANES = 512 / element_bitwidth (16 for 32-bit). Load/store must step by "
        "LANES with matching consecutive-dim partition. k+=4 / factor=4 is not done "
        "when max_widen_bitwidth=512 on a 32-bit ABI."
    )
    mac_note = (
        "A float MAC into one acc[i][j] every k is II=4. Advance k by 4 in one "
        "PIPELINE iteration and update named acc0..acc3 there. Do not switch(k&3). "
        "Do not accept II=4 as the legal endpoint for that GEMM reduction."
    )

    s = _skill_by_id(skills, "axi-burst-coalescing-narrow-safe")
    s["required_steps"] = _append_unique(
        s["required_steps"],
        [
            "set load/store inner step to LANES, not 2/4/8, when max_widen_bitwidth=512",
            "ARRAY_PARTITION cyclic factor=LANES on the consecutive dim of the staging buffer",
        ],
    )
    s["guards"] = _append_unique(
        s["guards"],
        [
            "do not leave k+=4 / factor=4 after requesting 512-bit widening on 32-bit data_t",
        ],
    )
    s["strategy"] = s["strategy"].rstrip() + " " + lanes_note

    s = _skill_by_id(skills, "axi-burst-widening-512")
    s["required_steps"] = _append_unique(
        s["required_steps"],
        [
            "rewrite the load/store loop to step by LANES = 512 / element_bitwidth",
            "match consecutive-dim partition factor to LANES (16 for float, not 4)",
        ],
    )
    s["strategy"] = s["strategy"].rstrip() + " " + lanes_note
    s["guards"] = _append_unique(
        s["guards"],
        ["k+=4 with factor=4 is an incomplete 512-bit load on 32-bit data_t"],
    )

    s = _skill_by_id(skills, "prompt-unroll")
    s["guards"] = _append_unique(
        s["guards"],
        [
            "AXI load/store unroll is LANES=512/element_bitwidth (16 for float), not the generic 2/4/8 default",
            "do not match partition factor=4 to a 4-wide load when the bus is 512-bit / 32-bit",
        ],
    )
    s["strategy"] = (
        s["strategy"].rstrip()
        + " For m_axi 512-bit widening, the load/store unroll and consecutive-dim "
        "partition are LANES, not a conservative 4."
    )

    s = _skill_by_id(skills, "hls-coalescing-512-compound-transform")
    s["required_steps"] = _append_unique(
        s["required_steps"],
        [
            "the load/store loop itself must iterate base += LANES; k+=4 is not LANES=16",
            "partition the consecutive dim by LANES even if compute banking uses a different factor",
        ],
    )
    s["guards"] = _append_unique(
        s["guards"],
        ["do not stop after max_widen_bitwidth=512 plus a 4-wide fused load"],
    )
    s["strategy"] = s["strategy"].rstrip() + " " + lanes_note

    s = _skill_by_id(skills, "hls-coalescing-compute-lane-parallelism")
    s["pattern"] = (
        s["pattern"].rstrip()
        + "; also applies when load already uses max_widen_bitwidth=512 but still "
        "walks four floats per cycle"
    )
    s["required_steps"] = _append_unique(
        s["required_steps"],
        ["apply the same LANES factor to the load/store walk, not only to compute"],
    )
    s["strategy"] = s["strategy"].rstrip() + " " + lanes_note

    s = _skill_by_id(skills, "hls-coalescing-contiguous-access-rewrite")
    s["required_steps"] = _append_unique(
        s["required_steps"],
        [
            "after isolating load/store, step the inner index by LANES when 512-bit widening is requested",
        ],
    )
    s["template"] = (
        "const int LANES = 512 / (8 * sizeof(data_t));\n"
        "load_loop: for (int i = 0; i < N; i += LANES) {\n"
        "#pragma HLS PIPELINE II=1\n"
        "    for (int l = 0; l < LANES; ++l) {\n"
        "#pragma HLS UNROLL\n"
        "        if (i + l < N) local[i + l] = in[i + l];\n"
        "    }\n"
        "}\n"
        "// BAD: for (int i = 0; i < N; ++i) PIPELINE local[i] = in[i];  // one float / cycle\n"
    )
    s["strategy"] = (
        s["strategy"].rstrip()
        + " Isolated load/store loops must still walk LANES elements per II=1 beat "
        "when max_widen_bitwidth=512; a scalar i++ pipeline is not coalesced."
    )

    s = _skill_by_id(skills, "hls-coalescing-lane-parallel-reduction")
    s["pattern"] = (
        s["pattern"].rstrip()
        + "; on GEMM this is also a k-loop that does acc[i][j] += a*b every k "
        "(Final II=4) after spatial MACs already exist"
    )
    s["required_steps"] = _append_unique(
        s["required_steps"],
        [
            "for a k-reduction, step k by the FP recurrence (typically 4) and update named acc0..acc3 in the same iteration",
            "do not implement partial banks with switch(k&3) under k++",
        ],
    )
    s["strategy"] = s["strategy"].rstrip() + " " + mac_note
    s["guards"] = _append_unique(
        s["guards"],
        ["switch(k&3) named banks still schedule as II=4"],
    )

    s = _skill_by_id(skills, "hls-coalescing-partition-lane-buffers")
    s["required_steps"] = _append_unique(
        s["required_steps"],
        [
            "consecutive-dim (k) partition factor equals the load unroll / LANES, not the compute tile width",
            "16-row compute banking does not substitute for factor=16 on k when the load is 16-wide",
        ],
    )
    s["strategy"] = (
        s["strategy"].rstrip()
        + " Match the consecutive-dim bank count to the AXI load unroll. Flash "
        "16x32 compute with factor=4 on k is a mismatch when the load is meant to be 16-wide."
    )

    s = _skill_by_id(skills, "hls-avoid-coalescing-interface-only")
    s["pattern"] = (
        s["pattern"].rstrip()
        + "; includes max_widen_bitwidth=512 plus k+=4 / partition factor=4 on 32-bit data_t"
    )
    s["strategy"] = (
        s["strategy"].rstrip()
        + " A 4-wide load after a 512-bit pragma is still interface-only coalescing."
    )

    s = _skill_by_id(skills, "hls-pipeline-handle-true-recurrence")
    s["required_steps"] = [
        "identify whether the dependence is true or false",
        "if it is a float/double MAC/add into one acc[i][j] every k, do not accept II=4",
        "rewrite with k += recurrence_distance (typically 4) and named acc0..acc3 updated in that same PIPELINE iteration, or fully unroll k into an adder tree when DSP fits",
        "do not use switch(k&3) / acc[k%4] — HLS still reports distance=1",
        "do not add DEPENDENCE inter false on a true FP MAC",
        "only accept the legal II for inherently serial non-reduction recurrences (true DP / feedback that cannot be restriped)",
        "preserve numerical and dependency semantics required by csim",
    ]
    s["strategy"] = (
        "Do not force II=1 by suppressing a true dependence. For a GEMM/float MAC "
        "reduction, do not accept II=4: restructure with independent accumulators "
        "updated in one k+=4 pipeline body (or a k adder tree). switch(k&3) is not "
        "that restructure. Accept legal II only when the recurrence cannot be restriped."
    )
    s["guards"] = _append_unique(
        s["guards"],
        [
            "float MAC on one acc every k is not 'inherently serial' — restripe it",
            "more spatial MACs under II=4 do not close II",
        ],
    )
    s["template"] = (
        "// BAD (Final II=4):\n"
        "for (int k = 0; k < K; ++k) {\n"
        "#pragma HLS PIPELINE II=1\n"
        "    acc[i][j] += a[k] * b[k];\n"
        "}\n"
        "// GOOD: k += 4; write acc0, acc1, acc2, acc3 in this iteration; tree-add at store.\n"
    )
    s["tags"] = _append_unique(s.get("tags") or [], ["gemm_flatten_v2", "preventive"])

    s = _skill_by_id(skills, "hls-unroll-reduction-partial-sums")
    s["required_steps"] = _append_unique(
        s["required_steps"],
        [
            "on a k-loop MAC, the lane factor is the FP recurrence (typically 4) and all partials update in one iteration via k += factor",
        ],
    )
    s["strategy"] = s["strategy"].rstrip() + " " + mac_note
    s["guards"] = _append_unique(s["guards"], ["do not encode partials as switch(k&3) under k++"])

    s = _skill_by_id(skills, "ii-reduction-lane-partial-tree")
    s["required_steps"] = _append_unique(
        s["required_steps"],
        [
            "the main loop index must step by the lane factor so each partial is written once per iteration",
        ],
    )
    s["strategy"] = s["strategy"].rstrip() + " " + mac_note

    s = _skill_by_id(skills, "hls-avoid-serial-fp-acc-under-full-k-unroll")
    s["pattern"] = (
        s["pattern"].rstrip()
        + "; also a PIPELINE on k with acc[i][j] += a*b every k (Final II=4) after "
        "spatial unroll — csim+csynth pass is not success"
    )
    s["strategy"] = (
        s["strategy"].rstrip()
        + " Also reject II=4 on a k-pipeline that updates one acc bank every k. "
        "Route to k+=4 named acc0..acc3 in one iteration, not switch(k&3), and not "
        "more PE_BLK under II=4."
    )
    s["required_steps"] = _append_unique(
        s["required_steps"],
        [
            "detect PIPELINE on k with a single acc[i][j] += MAC and Final II=4",
            "reject that as optimized even if DSP is already hundreds",
        ],
    )
    s["tags"] = _append_unique(s.get("tags") or [], ["gemm_flatten_v2", "preventive"])


def build() -> dict:
    src = json.loads(V1_PATH.read_text(encoding="utf-8"))
    skills = deepcopy(src["skills"])
    v1_ids = [s["id"] for s in skills]
    for sid in PATCH_IDS:
        if sid not in v1_ids:
            raise KeyError(f"v1 missing patch target {sid}")
    _patch_existing(skills)
    added = _new_skills()
    for skill in added:
        if skill["id"] in v1_ids:
            raise ValueError(f"new id collides with v1: {skill['id']}")
        skills.append(skill)
    out = {
        "saved_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S+0000"),
        "schema": src.get("schema", "1.1"),
        "derived_from": V1_NAME,
        "derived_from_sha256": _sha256(V1_PATH),
        "derived_from_sha256_note": (
            "copy of gemm_flatten_v1 with preventive coalescing / FP-MAC II=4 "
            "fixes; v1 and the 90 pack are not modified"
        ),
        "change_summary": [
            "Patched coalescing / unroll / recurrence skills so 512-bit widen on 32-bit ABI requires LANES=16 load and matching k-dim partition, not k+=4 / factor=4",
            "Patched true-recurrence / partial-sum skills so float MAC II=4 is restriped with k+=4 named acc0..acc3 in one iteration, not accepted as legal II",
            "Added hls-axi-load-step-matches-widen-lanes and avoid-axi-widen-512-with-narrow-unroll",
            "Added hls-fp-mac-k-step-independent-acc-banks, avoid-fp-mac-recurrence-single-acc, avoid-switch-k-mod-named-acc-banks",
            "switch(k&3) is an avoid: manual csynth of that form stayed II=4",
        ],
        "skill_count": len(skills),
        "skills": skills,
        "metadata": {
            **(src.get("metadata") or {}),
            "gemm_flatten_v2_preventive": True,
            "parent_skill_count": len(src["skills"]),
            "added_skill_ids": list(NEW_IDS),
            "patched_skill_ids": list(PATCH_IDS),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        },
    }
    return out


def main() -> int:
    if not V1_PATH.is_file():
        raise SystemExit(f"missing {V1_PATH}")
    v1_sha_before = _sha256(V1_PATH)
    doc = build()
    V2_PATH.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    v1_sha_after = _sha256(V1_PATH)
    if v1_sha_before != v1_sha_after:
        raise SystemExit("refusing: v1 hash changed while writing v2")
    print(f"wrote {V2_PATH}")
    print(f"v1_sha256={v1_sha_after}")
    print(f"skill_count={doc['skill_count']} (+{len(NEW_IDS)} from v1 {doc['metadata']['parent_skill_count']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
