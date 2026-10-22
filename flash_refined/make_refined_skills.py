#!/usr/bin/env python3
"""Write the flash_refined skill pack: a NEW file derived from pack 5; the source pack is only read.

Source: <c2hls>/hls_full_optimization_skills_schema_1_1_package/
        skills_ii_target_miss_solutions_added(90skills)_plus_gemm_flatten_v2.json   (134 skills)
Output: flash_refined/skills_ii_target_miss_solutions_added(90skills)_plus_gemm_flatten_v2_refined.json

Changes (every edit is an exact-substring replacement, asserted, and listed in the output's "refined" block):
  * hold-back wording rewritten: wide-bus-only-if-allowed, conservative/fixed-small unroll,
    "conservative BRAM budget", "separate bundles only when the platform supports it";
  * "resource explosion" defined as exceeding device capacity or failing timing (not growth vs the start);
  * two new skills: size the parallel datapath to the device budget, and avoid an idle device on a
    compute-bound kernel.
Legality guards (true dependences, bounded staging, safe DEPENDENCE pragmas, ...) are unchanged.
"""
import copy
import json
from datetime import datetime, timezone
from pathlib import Path

C2HLS = Path("/scratch/asa582/workspaces/code_translation_c2hls")
SRC = C2HLS / "hls_full_optimization_skills_schema_1_1_package" / \
    "skills_ii_target_miss_solutions_added(90skills)_plus_gemm_flatten_v2.json"
OUT_DIR = Path(__file__).resolve().parent
OUT_NAMES = {1: "skills_ii_target_miss_solutions_added(90skills)_plus_gemm_flatten_v2_refined.json",
             2: "skills_ii_target_miss_solutions_added(90skills)_plus_gemm_flatten_v2_refined_v2.json"}

# (skill id, field, old substring, new substring)
EDITS = [
    ("axi-burst-widening-512", "strategy",
     "if a packed wide-bus ABI is explicitly allowed, use ap_uint<512> words",
     "use ap_uint<512> (or hls::vector) words inside the kernel for 512-bit beats while the top-level argument types stay unchanged"),
    ("axi-burst-coalescing-narrow-safe", "template",
     "Keep pointer element types unchanged unless the active variant explicitly allows a packed wide-bus ABI",
     "Keep top-level pointer element types unchanged; inside the kernel pack 16 floats per 512-bit beat (max_widen_bitwidth=512, ap_uint<512> or hls::vector)"),
    ("hls-avoid-single-gmem-for-independent-streams", "guards",
     "separate only when host/platform support exists",
     "Vitis on Alveo U280 supports one m_axi bundle per pointer port; give independent streams distinct bundles"),
    ("hls-pipeline-unroll-small-inner-loops", "required_steps",
     "unroll inner loops conservatively or completely if small",
     "unroll inner loops completely if small, otherwise to the factor the device budget (DSP, buffer ports, timing) supports"),
    ("hls-tile-compute-inner-parallelism", "required_steps",
     "apply conservative UNROLL to the independent inner loop",
     "apply UNROLL to the independent inner loop, sized to the device budget and the local buffer ports"),
    ("hls-prefer-full-workspace-staging-when-fits", "required_steps",
     "compare against conservative BRAM budget for the target FPGA",
     "compare against the target FPGA's on-chip capacity (U280: 4032 BRAM_18K, 960 URAM)"),
    ("prompt-unroll", "strategy",
     "Reserve partial/conservative factors for large trip counts or proven resource pressure",
     "Use partial factors for large trip counts, sized so the whole design uses most of the device budget without reaching 100% of any resource"),
]
EXPLOSION_OLD = "resource explosion"
EXPLOSION_NEW = "resource explosion (exceeding device capacity or failing timing, not growth relative to the starting design)"

NEW_SKILLS = [
    {
        "id": "hls-size-parallel-datapath-to-device-budget",
        "kind": "compound_transformation",
        "confidence": "high",
        "pattern": "compute-bound kernel with large static trip counts (e.g. matrix multiply, convolution, stencil) whose "
                   "parallel datapath uses a small fraction of the target device's DSPs",
        "strategy": "choose the number of parallel operators from the device budget instead of a fixed small factor: "
                    "estimate DSP per operation, take the largest parallelism that keeps every resource below 100% "
                    "(aim for most of the DSP budget), shape it into spatial lanes/PEs and vector lanes that the "
                    "memory interfaces and on-chip buffers can feed every cycle, and overlap loads and stores with compute",
        "required_steps": [
            "estimate DSP per operation for the data type and part (fp32 multiply-add on UltraScale+ is about 5 DSP)",
            "compute the operator count that fits the DSP budget with margin, P = floor(fill * DSP_capacity / DSP_per_op)",
            "factor P into loop dimensions with static trip counts (non-power-of-two factors are allowed when they divide or are guarded)",
            "size on-chip buffers and partitions so every operator gets its operands each cycle",
            "feed buffers with 512-bit bursts on distinct m_axi bundles and overlap transfers with compute (ping-pong or DATAFLOW)",
            "check BRAM/URAM/LUT/FF and timing; if any resource reaches 100% or timing fails, reduce P one step",
        ],
        "guards": [
            "every resource must stay strictly below 100% of the device",
            "keep the top-level ABI unchanged",
            "do not unroll across true loop-carried dependencies; use partial sums or lane accumulators for reductions",
        ],
        "template": "// U280: DSP=9024, fp32 MAC ~5 DSP -> up to ~1700 MACs/cycle at 95%\n"
                    "// choose P_i * P_j * P_k ~= that budget, e.g. rows x cols of outputs x k-lanes per cycle",
        "tags": ["device-budget", "parallelism", "dsp-fill", "flash-refined"],
        "bottleneck_kinds": ["latency_high", "ii_target_miss", "non_pipelined_hot_loop", "compute_bound"],
    },
    {
        "id": "avoid-idle-device-on-compute-bound-kernel",
        "kind": "avoid_rule",
        "confidence": "avoid",
        "pattern": "a compute-bound kernel is optimized with a datapath that uses a small share of the device "
                   "(for example under a quarter of the DSPs) while latency is dominated by compute",
        "strategy": "do not stop at a modest unroll or a small PE array when the device has headroom; scale the "
                    "parallel datapath toward the device budget (hls-size-parallel-datapath-to-device-budget)",
        "required_steps": [
            "compare csynth DSP/LUT/BRAM utilization against device capacity",
            "if compute dominates latency and utilization is low, increase spatial/vector parallelism",
        ],
        "guards": ["growth relative to the starting design is not a reason to stop; only device capacity and timing are"],
        "template": "",
        "tags": ["device-budget", "avoid", "flash-refined"],
        "bottleneck_kinds": ["latency_high", "compute_bound"],
    },
]


# Version 2 adds the compute/transfer balance rule (a filled datapath must not starve on memory).
BALANCE_SKILLS = [
    {
        "id": "hls-balance-compute-and-transfer-after-scaling",
        "kind": "compound_transformation",
        "confidence": "high",
        "pattern": "the parallel datapath has been scaled toward the device budget, so each tile's compute now takes "
                   "few cycles, while operand loads and result stores per tile take as long or longer",
        "strategy": "after sizing the datapath, budget transfers per tile against compute per tile: keep operands "
                    "resident on chip across the tiles that reuse them (load each operand once per reuse window), move "
                    "every array, including the output read-modify-write, in 512-bit beats at II=1, and overlap loads "
                    "and stores with compute (DATAFLOW stages or ping-pong buffers) so the datapath never waits",
        "required_steps": [
            "estimate per-tile compute cycles (work per tile / parallel operators) and per-tile transfer cycles (words moved / words per cycle)",
            "if transfers per tile approach or exceed compute per tile, enlarge the reuse window: keep the operand that is reused across tiles resident on chip instead of reloading it every tile",
            "make every load and store loop move one 512-bit beat per cycle (II=1, unrolled 16-float lanes, max_widen_bitwidth=512); an II of 16 on a 512-bit port means one float per beat",
            "overlap the next tile's loads and the previous tile's stores with the current tile's compute (DATAFLOW load/compute/store or ping-pong buffers)",
            "check csynth: latency should be close to total work / parallel operators; if a load or store loop dominates, the design is transfer-bound",
        ],
        "guards": [
            "every resource must stay strictly below 100% of the device",
            "keep the top-level ABI unchanged",
            "DATAFLOW needs a legal producer/consumer structure (single producer and consumer per buffer or stream)",
        ],
        "template": "// per tile: compute = (TI*TJ*K)/(PE_I*PE_J*KL) cycles; transfer = words/16 beats\n"
                    "// keep transfer < compute: B strip resident across row tiles, C moved in 512-bit beats, DATAFLOW overlap",
        "tags": ["device-budget", "balance", "dataflow", "flash-refined"],
        "bottleneck_kinds": ["memory_bandwidth", "latency_high", "ii_target_miss"],
    },
    {
        "id": "avoid-transfer-bound-filled-datapath",
        "kind": "avoid_rule",
        "confidence": "avoid",
        "pattern": "DSP is filled but latency is far above work / parallel operators, and the longest loops are "
                   "loads or stores (operands reloaded every tile, output moved one element per beat, no overlap)",
        "strategy": "do not stop after filling DSP: rebalance transfers (hls-balance-compute-and-transfer-after-scaling) "
                    "before accepting the design",
        "required_steps": ["compare csynth latency against total work / parallel operators",
                           "find the loops that dominate latency; if they are load/store loops, rebalance"],
        "guards": ["more DSP does not help a transfer-bound design"],
        "template": "",
        "tags": ["device-budget", "balance", "avoid", "flash-refined"],
        "bottleneck_kinds": ["memory_bandwidth", "latency_high"],
    },
]


def main() -> None:
    src = json.loads(SRC.read_text())
    out = copy.deepcopy(src)
    skills = {s["id"]: s for s in out["skills"]}
    log = []
    for sid, field, old, new in EDITS:
        v = skills[sid][field]
        if isinstance(v, list):
            hits = [i for i, x in enumerate(v) if isinstance(x, str) and old in x]
            assert len(hits) == 1, (sid, field, old)
            v[hits[0]] = v[hits[0]].replace(old, new)
        else:
            assert isinstance(v, str) and v.count(old) == 1, (sid, field, old)
            skills[sid][field] = v.replace(old, new)
        log.append({"skill": sid, "field": field, "old": old, "new": new})
    n_expl = 0

    def walk(o):
        nonlocal n_expl
        if isinstance(o, dict):
            for k, v in o.items():
                if isinstance(v, str) and EXPLOSION_OLD in v and EXPLOSION_NEW not in v:
                    o[k] = v.replace(EXPLOSION_OLD, EXPLOSION_NEW)
                    n_expl += 1
                else:
                    walk(v)
        elif isinstance(o, list):
            for i, v in enumerate(o):
                if isinstance(v, str) and EXPLOSION_OLD in v and EXPLOSION_NEW not in v:
                    o[i] = v.replace(EXPLOSION_OLD, EXPLOSION_NEW)
                    n_expl += 1
                else:
                    walk(v)

    walk(out["skills"])
    base = src["skills"][0]
    import sys
    version = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    out_path = OUT_DIR / OUT_NAMES[version]
    for ns in NEW_SKILLS + (BALANCE_SKILLS if version >= 2 else []):
        assert ns["id"] not in skills
        s = {k: copy.deepcopy(base.get(k)) for k in base}
        s.update(ns)
        s["origin"] = "flash_refined"
        s["occurrences"], s["mean_advantage"], s["sec_pass"], s["last_used_at"] = 0, 0.0, 0, None
        out["skills"].append(s)
    out["saved_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    out["refined"] = {"source": str(SRC), "edits": log, "resource_explosion_clarified": n_expl,
                      "added": [s["id"] for s in NEW_SKILLS + (BALANCE_SKILLS if version >= 2 else [])],
                      "version": version,
                      "note": "derived file; the source pack is unchanged"}
    out_path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {out_path.name}: {len(out['skills'])} skills ({len(src['skills'])} source), "
          f"{len(log)} edits, {n_expl} 'resource explosion' clarifications, +{len(out['refined']['added'])} new")


if __name__ == "__main__":
    main()
