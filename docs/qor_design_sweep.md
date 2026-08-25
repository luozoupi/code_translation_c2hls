# QoR Design Sweep

## Purpose

`QualityRepairAgent` has an opt-in, reference-blind design-space stage for
testing explicit HLS design choices instead of asking an LLM to make another
uncontrolled rewrite. The stage answers two questions:

1. Which tested parameter value gives the lowest feasible CSynth latency?
2. How do latency, achieved II, timing, and resource use move as that parameter
   changes while the rest of the source remains fixed?

It does not assume that a larger optimization factor must improve latency.
Unroll, partition, and tile changes often cross memory, timing, and resource
limits. Monotonicity is measured as evidence, not imposed as a selection rule.

## Execution Contract

1. Freeze the current best HLS source and its CSim/CSynth evidence.
2. Discover explicit controls: pipeline II, unroll factor, array
   partition/reshape factor, stream depth, allocation limit, selected AXI
   interface options, named tile/block constants, binding latency, and safe
   disable-only toggles for existing HLS directives.
3. Record which optimization step produced the frozen best state and spend the
   bounded knob budget on that step's control family first. Then generate
   bounded
   one-factor-at-a-time candidates from the same frozen parent.
4. Compile-check each candidate, run CSim first, and run Vitis CSynth only when
   CSim passes. COSIM is not run during this local search.
5. Optionally combine pairs of values that improved independently, still from
   the frozen parent.
6. Reject candidates lacking passing correctness, timing, resource-fit, or
   positive-latency evidence. Select minimum worst-case CSynth cycles, using a
   tiny resource sum only to break exact latency ties.
7. Promote the winner only when it improves the parent. The framework's normal
   selected-winner COSIM policy may validate it later, outside this stage.

Every candidate records its changed and fixed knob values, source/report
hashes, CSim result, CSynth metrics, feasibility reasons, Pareto membership,
and deterministic winner explanation. Per-knob evidence includes Spearman
correlations against cycles, achieved II, and each resource plus directional
monotonicity violations.

## Controls

```bash
export C2HLS_QOR_DESIGN_SWEEP=1
export C2HLS_QOR_SWEEP_MAX_KNOBS=4
export C2HLS_QOR_SWEEP_MAX_CANDIDATES=8
export C2HLS_QOR_SWEEP_VALUES=1,2,4,8,16
export C2HLS_QOR_SWEEP_II_VALUES=1,2,4,8
export C2HLS_QOR_SWEEP_TILE_VALUES=4,8,16,32,64
export C2HLS_QOR_SWEEP_INTERACTIONS=1
export C2HLS_QOR_SWEEP_MAX_INTERACTIONS=2
```

Direct CLI example:

```bash
python c2hls.py \
  --bench-dir benchmarks_external/HLSFactory/polybench_float_small/hlsfactory_2mm \
  --multistep \
  --qor-design-sweep \
  --qor-sweep-max-knobs 4 \
  --qor-sweep-max-candidates 8 \
  --qor-sweep-interactions
```

Sweep-driver aliases use the `C2HLS_SWEEP_QOR_*` prefix. The driver adds the
QoR candidate cap to the existing synthesis budget after applying the paper
profile. Set `C2HLS_SWEEP_SYNTHESIS_EVAL_BUDGET` to an explicit total when a
different combined cap is required. For direct `c2hls.py` runs with a bounded
`C2HLS_SYNTHESIS_EVAL_BUDGET`, reserve capacity for both the normal LLM
candidates and the QoR candidates.

## Result Fields

- `quality_repair.design_sweep`: complete parent, candidate, trend, Pareto,
  winner, and configuration evidence.
- `qor_design_sweep`: multistep root alias for the same evidence.
- `synthesis_evaluations.qor_design_events`: deterministic tool-call events.
- `qor_synthesis_evaluation_count`: QoR CSynth calls, separated from LLM
  candidate syntheses but included in total tool cost.
- `run.qor_design_sweep`: effective controls and policies.
- Website JSONL: compact evidence under
  `implementation.origin_meta.qor_design_sweep`; generated code bodies are
  omitted.

The stage runs once after the trajectory has selected its best snapshot. It
does not multiply synthesis cost by running a full experiment after every LLM
step. The recorded `parent_origin` and `preferred_knob_kinds` explain which
step-aware controls received priority.

Supported step-aware families are:

| Winning step | Prioritized controls |
|---|---|
| Pipeline | Pipeline II |
| Unroll | Unroll factor |
| Tiling | Tile/block size and partition/reshape factor |
| Double buffer | Existing dataflow, stream depth, tile size, and partition/reshape controls |
| Coalescing | AXI widening, outstanding-transaction, and burst-length controls |
| Resource repair | Allocation, binding/resource latency, and existing binding/partition toggles |

Bare existing `m_axi` interfaces can receive an explicit
`max_widen_bitwidth`. Existing `DATAFLOW`, complete partition/reshape,
`BIND_OP`, `BIND_STORAGE`, and legacy `RESOURCE` directives can be disabled for
an ablation. Every such edit remains subject to CSim, CSynth, timing, and device
resource-fit gates.

The stage only mutates source-local controls or disables directives already in
the source. It does not invent a new buffer topology or switch categorical
implementations such as DSP versus fabric or BRAM versus URAM. In particular,
hard-coded ping-pong arrays and `% 2` indexing need a coordinated rewrite, not
an unsafe numeric buffer-count substitution. If no safe knob exists, the stage
records `no safely parameterized QoR knobs discovered` and retains the parent.

## Work Storage

Run active Vitis CSim/CSynth work directories on local storage through
`C2HLS_TMP_ROOT` and `C2HLS_VITIS_USER_HOME`. Archive completed work directories
to `/mnt/data2/luo00466/c2hls_rl` afterward. Running Vitis directly against the
NAS can make compilation and timeout cleanup block on network-filesystem I/O;
the subprocess wrapper bounds cleanup, but local scratch remains the supported
execution path.
