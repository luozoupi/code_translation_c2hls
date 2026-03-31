"""
FPGA Synthesis Quality Rubric for C-to-HLS Translation Evaluation.

Evaluates generated HLS code against ground-truth baselines using 9 metrics
that reflect real FPGA engineering concerns:

  M1. Synthesis Success       ( 5%)  — did the code synthesise at all?
  M2. Csim (Functional)       (10%)  — C-simulation pass/fail (correctness vs testbench)
  M3. Cosim (RTL)             (10%)  — co-simulation pass/fail (RTL matches C behaviour)
  M4. Latency (ns)            (23%)  — latency in ns vs GT (lower is better)
  M5. Clock Frequency / Fmax  (10%)  — timing closure quality (higher is better)
  M6. Resource Efficiency     (17%)  — BRAM/DSP/FF/LUT vs GT, weighted by scarcity
  M7. Area-Delay Product      (15%)  — combined efficiency: latency × normalised area
  M8. Device Feasibility      (10%)  — hard resource (BRAM+DSP) pressure vs device limits

Target device: xc7a100t-csg324-1 (Artix-7 100T)
Clock target: 4 ns (250 MHz)
"""

import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ── Target device resource limits (xc7a100t-csg324-1, Artix-7 100T) ──

DEVICE_LIMITS = {
    "bram": 270,      # BRAM_18K (135 × BRAM_36K × 2)
    "dsp":  240,       # DSP48E1
    "ff":   126800,    # flip-flops
    "lut":  63400,     # 6-input LUTs
    "uram": 0,         # not available on Artix-7
}

DEFAULT_CLOCK_NS = 4  # target clock period (ns)

# ── Metric weights (sum to 1.0) ──────────────────────────────────────

METRIC_WEIGHTS = {
    "synthesis":   0.05,  # M1: binary pass/fail
    "csim":        0.10,  # M2: C-simulation correctness
    "cosim":       0.10,  # M3: co-simulation RTL correctness
    "latency":     0.23,  # M4: latency (ns) vs GT
    "fmax":        0.10,  # M5: timing closure
    "resources":   0.17,  # M6: resource usage vs GT
    "adp":         0.15,  # M7: area-delay product vs GT
    "feasibility": 0.10,  # M8: hard resource pressure vs device
}

# Within M4, sub-weights reflect resource scarcity on the target FPGA.
# BRAM and DSP are hard macros (fixed count), FF/LUT are soft fabric.
RESOURCE_SUB_WEIGHTS = {
    "bram": 0.30,   # scarce hard macro
    "dsp":  0.30,   # scarce hard macro
    "ff":   0.20,   # abundant fabric
    "lut":  0.20,   # abundant fabric
}


# ── Scoring functions ─────────────────────────────────────────────────

def _ratio_score(ratio: float, lower_is_better: bool = True) -> float:
    """Convert a gen/gt ratio to a 0–100 score.

    Piecewise: matching GT (ratio=1) anchors at 85. Beating GT scales to 100.
    Overhead decays exponentially:
        ratio = 1.0  →  85     (matches GT)
        ratio = 0.5  →  92.5   (50% better than GT)
        ratio = 2.0  →  42.2   (2× worse)
        ratio = 5.0  →   5.3   (5× worse)
    """
    if ratio is None or ratio <= 0:
        return 0.0
    if not lower_is_better:
        ratio = 1.0 / ratio
    ratio = max(ratio, 0.01)

    if ratio <= 1.0:
        # Better than or equal to GT
        return round(85.0 + 15.0 * (1.0 - ratio), 2)
    else:
        # Worse than GT — exponential decay
        score = 85.0 * math.exp(-0.7 * (ratio - 1.0))
        return round(max(score, 0.0), 2)


def _feasibility_score(gen_report: dict) -> float:
    """M7: Score based on resource pressure against device limits.

    Evaluates whether the design is physically feasible on the target.
    Checks ALL resources (BRAM, DSP, FF, LUT) — any single resource
    exceeding device limits makes the design infeasible.
    Uses the most-stressed resource (max utilisation %) to score:
        ≤ 50%    →  100  (comfortable margin)
        50–80%   →  100→60  (getting tight)
        80–100%  →  60→20   (at risk)
        > 100%   →  0       (infeasible / won't fit)
    """
    pcts = []
    for key in ["bram", "dsp", "ff", "lut"]:
        pct = _util_pct(gen_report.get(key), key)
        if pct is not None:
            pcts.append(pct)
    max_pct = max(pcts) if pcts else 0.0

    if max_pct > 100:
        return 0.0
    elif max_pct > 80:
        return round(20.0 + 40.0 * (100.0 - max_pct) / 20.0, 2)
    elif max_pct > 50:
        return round(60.0 + 40.0 * (80.0 - max_pct) / 30.0, 2)
    else:
        return 100.0


def _adp_score(gen_report: dict, gt_report: dict) -> float:
    """M5: Area-Delay Product — the standard FPGA efficiency metric.

    ADP = latency_ns × normalised_area
    where normalised_area = Σ (weight_i × resource_i / device_limit_i)

    Using latency_ns accounts for both cycle count and clock frequency,
    so a design running at a higher Fmax is properly credited.
    Lower ADP is better — a design that is 2× faster but uses 2× resources
    has the same ADP (neutral trade-off). Better designs reduce ADP.
    """
    gen_lat = _try_float(gen_report.get("latency_ns")) or _try_int(gen_report.get("latency_cycles"))
    gt_lat = _try_float(gt_report.get("latency_ns")) or _try_int(gt_report.get("latency_cycles"))
    if gen_lat is None or gt_lat is None or gt_lat == 0:
        return 50.0

    def _normalised_area(report):
        total = 0.0
        for key, weight in RESOURCE_SUB_WEIGHTS.items():
            val = _try_float(report.get(key))
            limit = DEVICE_LIMITS.get(key, 1)
            if val is not None and limit > 0:
                total += weight * (val / limit)
        return total

    gen_area = _normalised_area(gen_report)
    gt_area = _normalised_area(gt_report)
    if gt_area == 0 or gen_area == 0:
        return 50.0

    gen_adp = gen_lat * gen_area
    gt_adp = gt_lat * gt_area
    ratio = gen_adp / gt_adp if gt_adp > 0 else 10.0

    return _ratio_score(ratio, lower_is_better=True)


def _util_pct(value, resource_key: str) -> Optional[float]:
    """Resource utilisation as % of device capacity."""
    limit = DEVICE_LIMITS.get(resource_key, 0)
    if limit <= 0:
        return None
    try:
        return round(100.0 * float(value) / limit, 2)
    except (TypeError, ValueError):
        return None


def _safe_ratio(gen_val, gt_val) -> Optional[float]:
    try:
        g = float(gen_val)
        t = float(gt_val)
        if t == 0:
            return 1.0 if g == 0 else 10.0
        return g / t
    except (TypeError, ValueError):
        return None


def _try_int(v) -> Optional[int]:
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _try_float(v) -> Optional[float]:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


# ── Data classes ──────────────────────────────────────────────────────

@dataclass
class StepScore:
    step_name: str
    synthesised: bool

    # M2: Csim (functional correctness)
    csim_ran: bool = False         # was csim attempted?
    csim_passed: bool = False      # did csim pass?
    csim_score: float = 0.0        # 0 or 100

    # M3: Cosim (RTL correctness)
    cosim_ran: bool = False
    cosim_passed: bool = False
    cosim_score: float = 0.0

    # M4: Latency (ns — accounts for both cycle count and clock frequency)
    gen_latency: Optional[float] = None     # latency_ns
    gt_latency: Optional[float] = None      # latency_ns
    gen_latency_cycles: Optional[int] = None
    gt_latency_cycles: Optional[int] = None
    latency_ratio: Optional[float] = None
    latency_score: float = 0.0

    # M3: Fmax / Timing
    gen_fmax: Optional[float] = None
    gt_fmax: Optional[float] = None
    fmax_ratio: Optional[float] = None
    fmax_score: float = 0.0
    timing_slack_ns: Optional[float] = None  # positive = timing met

    # M5: Resources (ratios vs GT)
    resource_ratios: dict = field(default_factory=dict)
    resource_score: float = 0.0

    # M5: ADP
    adp_score: float = 50.0

    # M6: Device feasibility
    util_pct: dict = field(default_factory=dict)  # {bram: %, dsp: %, ff: %, lut: %}
    feasibility_score: float = 100.0

    # Composite (excludes synthesis weight — that's at benchmark level)
    composite: float = 0.0


@dataclass
class BenchmarkScore:
    benchmark: str
    steps: list  # list[StepScore]

    synthesis_rate: float = 0.0      # 0–100
    csim_rate: float = 0.0           # 0–100 pass rate among attempted steps
    cosim_rate: float = 0.0          # 0–100 pass rate among attempted steps
    csim_coverage: float = 0.0       # 0–100 attempted among synthesised steps
    cosim_coverage: float = 0.0      # 0–100 attempted among synthesised steps
    avg_latency: float = 0.0
    avg_fmax: float = 0.0
    avg_resources: float = 0.0
    avg_adp: float = 50.0
    avg_feasibility: float = 100.0

    composite: float = 0.0          # weighted 0–100


# ── Grade thresholds ──────────────────────────────────────────────────

GRADE_THRESHOLDS = [
    (90, "A",  "Excellent — matches or beats GT; device-feasible; efficient ADP"),
    (75, "B",  "Good — close to GT with minor overhead; timing met"),
    (60, "C",  "Acceptable — functional but notable resource or latency gaps"),
    (40, "D",  "Below average — large overhead, timing issues, or poor ADP"),
    ( 0, "F",  "Poor — synthesis failures, infeasible, or extreme overhead"),
]


def letter_grade(score: float) -> tuple:
    for threshold, letter, desc in GRADE_THRESHOLDS:
        if score >= threshold:
            return letter, desc
    return "F", GRADE_THRESHOLDS[-1][2]


# ── Core scoring ──────────────────────────────────────────────────────

def score_step(step_name: str, gen_report: dict, gt_report: dict,
               synthesised: bool = True,
               csim_result: dict = None, cosim_result: dict = None) -> StepScore:
    ss = StepScore(step_name=step_name, synthesised=synthesised)

    if not synthesised or gen_report is None:
        ss.adp_score = 0.0
        ss.feasibility_score = 0.0
        return ss

    # M2: Csim — binary pass/fail (100 if passed, 0 if failed/not run)
    if csim_result is not None:
        ss.csim_ran = True
        ss.csim_passed = csim_result.get("passed", False)
        ss.csim_score = 100.0 if ss.csim_passed else 0.0
    # If csim was not run, score stays 0 — this penalises missing verification

    # M3: Cosim — binary pass/fail
    if cosim_result is not None:
        ss.cosim_ran = True
        ss.cosim_passed = cosim_result.get("passed", False)
        ss.cosim_score = 100.0 if ss.cosim_passed else 0.0

    # M4: Latency — use latency_ns (accounts for cycle count × clock period)
    ss.gen_latency = _try_float(gen_report.get("latency_ns"))
    ss.gt_latency = _try_float(gt_report.get("latency_ns"))
    ss.gen_latency_cycles = _try_int(gen_report.get("latency_cycles"))
    ss.gt_latency_cycles = _try_int(gt_report.get("latency_cycles"))
    # Fall back to cycles if ns not available
    if ss.gen_latency is None and ss.gen_latency_cycles is not None:
        ss.gen_latency = float(ss.gen_latency_cycles)
    if ss.gt_latency is None and ss.gt_latency_cycles is not None:
        ss.gt_latency = float(ss.gt_latency_cycles)
    ss.latency_ratio = _safe_ratio(ss.gen_latency, ss.gt_latency)
    ss.latency_score = _ratio_score(ss.latency_ratio) if ss.latency_ratio else 0.0

    # M3: Fmax + timing slack
    ss.gen_fmax = _try_float(gen_report.get("fmax_mhz"))
    ss.gt_fmax = _try_float(gt_report.get("fmax_mhz"))
    ss.fmax_ratio = _safe_ratio(ss.gen_fmax, ss.gt_fmax)
    ss.fmax_score = _ratio_score(ss.fmax_ratio, lower_is_better=False) if ss.fmax_ratio else 0.0
    if ss.gen_fmax and ss.gen_fmax > 0:
        estimated_period = 1000.0 / ss.gen_fmax
        ss.timing_slack_ns = round(DEFAULT_CLOCK_NS - estimated_period, 2)

    # M5: Resource efficiency vs GT
    res_scores = []
    for key, weight in RESOURCE_SUB_WEIGHTS.items():
        r = _safe_ratio(gen_report.get(key), gt_report.get(key))
        if r is not None:
            ss.resource_ratios[key] = round(r, 3)
            res_scores.append((_ratio_score(r), weight))
    if res_scores:
        total_w = sum(w for _, w in res_scores)
        ss.resource_score = round(sum(s * w for s, w in res_scores) / total_w, 2) if total_w else 0.0

    # M5: ADP
    ss.adp_score = _adp_score(gen_report, gt_report)

    # M6: Device feasibility
    for key in ["bram", "dsp", "ff", "lut"]:
        pct = _util_pct(gen_report.get(key), key)
        if pct is not None:
            ss.util_pct[key] = pct
    ss.feasibility_score = _feasibility_score(gen_report)

    # Composite (synthesis weight handled at benchmark level)
    ss.composite = round(
        METRIC_WEIGHTS["csim"]        * ss.csim_score
        + METRIC_WEIGHTS["cosim"]     * ss.cosim_score
        + METRIC_WEIGHTS["latency"]   * ss.latency_score
        + METRIC_WEIGHTS["fmax"]      * ss.fmax_score
        + METRIC_WEIGHTS["resources"] * ss.resource_score
        + METRIC_WEIGHTS["adp"]       * ss.adp_score
        + METRIC_WEIGHTS["feasibility"] * ss.feasibility_score,
        2,
    )
    return ss


def score_benchmark(benchmark: str, steps: list) -> BenchmarkScore:
    bs = BenchmarkScore(benchmark=benchmark, steps=steps)
    if not steps:
        return bs

    n = len(steps)
    n_synth = sum(1 for s in steps if s.synthesised)
    bs.synthesis_rate = round(100.0 * n_synth / n, 2) if n else 0.0

    # Csim/cosim rates: among steps that were synthesised
    csim_attempted = [s for s in steps if s.csim_ran]
    cosim_attempted = [s for s in steps if s.cosim_ran]
    bs.csim_rate = round(100.0 * sum(1 for s in csim_attempted if s.csim_passed) / len(csim_attempted), 2) if csim_attempted else 0.0
    bs.cosim_rate = round(100.0 * sum(1 for s in cosim_attempted if s.cosim_passed) / len(cosim_attempted), 2) if cosim_attempted else 0.0
    bs.csim_coverage = round(100.0 * len(csim_attempted) / n_synth, 2) if n_synth else 0.0
    bs.cosim_coverage = round(100.0 * len(cosim_attempted) / n_synth, 2) if n_synth else 0.0

    scored = [s for s in steps if s.synthesised]
    if scored:
        k = len(scored)
        bs.avg_latency = round(sum(s.latency_score for s in scored) / k, 2)
        bs.avg_fmax = round(sum(s.fmax_score for s in scored) / k, 2)
        bs.avg_resources = round(sum(s.resource_score for s in scored) / k, 2)
        bs.avg_adp = round(sum(s.adp_score for s in scored) / k, 2)
        bs.avg_feasibility = round(sum(s.feasibility_score for s in scored) / k, 2)

    bs.composite = round(
        METRIC_WEIGHTS["synthesis"]   * bs.synthesis_rate
        + METRIC_WEIGHTS["csim"]      * bs.csim_rate
        + METRIC_WEIGHTS["cosim"]     * bs.cosim_rate
        + METRIC_WEIGHTS["latency"]   * bs.avg_latency
        + METRIC_WEIGHTS["fmax"]      * bs.avg_fmax
        + METRIC_WEIGHTS["resources"] * bs.avg_resources
        + METRIC_WEIGHTS["adp"]       * bs.avg_adp
        + METRIC_WEIGHTS["feasibility"] * bs.avg_feasibility,
        2,
    )
    return bs


# ── Report formatter ──────────────────────────────────────────────────

def format_report(benchmarks: list, title: str = "FPGA Synthesis Quality Rubric") -> str:
    W = 120
    lines = []
    lines.append(f"{'=' * W}")
    lines.append(f"  {title}")
    lines.append(f"{'=' * W}")
    lines.append(f"  Target: xc7a100t-csg324-1 (Artix-7 100T)  |  Clock: {DEFAULT_CLOCK_NS} ns  |  "
                 f"BRAM={DEVICE_LIMITS['bram']}  DSP={DEVICE_LIMITS['dsp']}  "
                 f"FF={DEVICE_LIMITS['ff']}  LUT={DEVICE_LIMITS['lut']}")
    lines.append(f"  Weights: Synth={METRIC_WEIGHTS['synthesis']:.0%}  "
                 f"Csim={METRIC_WEIGHTS['csim']:.0%}  "
                 f"Cosim={METRIC_WEIGHTS['cosim']:.0%}  "
                 f"Latency={METRIC_WEIGHTS['latency']:.0%}  "
                 f"Fmax={METRIC_WEIGHTS['fmax']:.0%}  "
                 f"Resources={METRIC_WEIGHTS['resources']:.0%}  "
                 f"ADP={METRIC_WEIGHTS['adp']:.0%}  "
                 f"Feasibility={METRIC_WEIGHTS['feasibility']:.0%}")
    lines.append("")

    for bs in benchmarks:
        grade, desc = letter_grade(bs.composite)
        lines.append(f"{'─' * W}")
        lines.append(f"  {bs.benchmark:24s}   Score: {bs.composite:5.1f}/100   Grade: {grade}")
        lines.append(f"  {desc}")
        lines.append(f"  Synth: {bs.synthesis_rate:3.0f}%   "
                     f"Csim: {bs.csim_rate:3.0f}%   "
                     f"Cosim: {bs.cosim_rate:3.0f}%   "
                     f"Lat: {bs.avg_latency:5.1f}   "
                     f"Fmax: {bs.avg_fmax:5.1f}   "
                     f"Res: {bs.avg_resources:5.1f}   "
                     f"ADP: {bs.avg_adp:5.1f}   "
                     f"Feasibility: {bs.avg_feasibility:5.1f}")
        lines.append("")

        # Detailed per-step table
        hdr = (f"    {'Step':<14s} {'Syn':>4s} {'Csim':>4s} {'Cosm':>4s}"
               f" {'LatNs':>12s} {'LatR':>6s} {'Lat':>5s}"
               f" {'Fmax':>6s} {'FmxR':>5s} {'Fmx':>5s} {'Slack':>6s}"
               f" {'BRAM%':>6s} {'DSP%':>6s} {'FF%':>5s} {'LUT%':>5s} {'Res':>5s}"
               f" {'ADP':>5s} {'Feas':>5s}"
               f" {'Total':>5s}")
        lines.append(hdr)
        lines.append(f"    {'─' * (len(hdr) - 4)}")

        for s in bs.steps:
            syn = " OK" if s.synthesised else "FAIL"
            csim_s = "PASS" if s.csim_passed else "FAIL"
            cosim_s = "PASS" if s.cosim_passed else "FAIL"
            lat_ns = f"{s.gen_latency:,.0f}" if s.gen_latency else "—"
            lat_r = f"{s.latency_ratio:.2f}" if s.latency_ratio else "—"
            fmax_v = f"{s.gen_fmax:.1f}" if s.gen_fmax else "—"
            fmx_r = f"{s.fmax_ratio:.2f}" if s.fmax_ratio else "—"
            slack = f"{s.timing_slack_ns:+.1f}" if s.timing_slack_ns is not None else "—"

            bram_u = f"{s.util_pct.get('bram', 0):.1f}" if s.synthesised else "—"
            dsp_u = f"{s.util_pct.get('dsp', 0):.1f}" if s.synthesised else "—"
            ff_u = f"{s.util_pct.get('ff', 0):.1f}" if s.synthesised else "—"
            lut_u = f"{s.util_pct.get('lut', 0):.1f}" if s.synthesised else "—"

            row = (f"    {s.step_name:<14s} {syn:>4s} {csim_s:>4s} {cosim_s:>4s}"
                   f" {lat_ns:>12s} {lat_r:>6s} {s.latency_score:>5.1f}"
                   f" {fmax_v:>6s} {fmx_r:>5s} {s.fmax_score:>5.1f} {slack:>6s}"
                   f" {bram_u:>6s} {dsp_u:>6s} {ff_u:>5s} {lut_u:>5s} {s.resource_score:>5.1f}"
                   f" {s.adp_score:>5.1f} {s.feasibility_score:>5.1f}"
                   f" {s.composite:>5.1f}")
            lines.append(row)

        lines.append("")

    # Overall summary
    lines.append(f"{'=' * W}")
    if benchmarks:
        overall = sum(b.composite for b in benchmarks) / len(benchmarks)
        grade, desc = letter_grade(overall)
        synth_only = [b for b in benchmarks if b.synthesis_rate > 0]
        overall_synth = sum(b.composite for b in synth_only) / len(synth_only) if synth_only else 0
        lines.append(f"  Overall (all):       {overall:5.1f}/100  Grade: {grade}")
        if len(synth_only) < len(benchmarks):
            g2, d2 = letter_grade(overall_synth)
            lines.append(f"  Overall (synth ok):  {overall_synth:5.1f}/100  Grade: {g2}  "
                         f"({len(synth_only)}/{len(benchmarks)} synthesised)")
    lines.append(f"{'=' * W}")
    return "\n".join(lines)


# ── Loaders ───────────────────────────────────────────────────────────

def load_singleshot_results(results_dir: str) -> list:
    results_dir = Path(results_dir)
    benchmarks = []
    for bench_dir in sorted(results_dir.iterdir()):
        if not bench_dir.is_dir():
            continue
        results_file = bench_dir / f"{bench_dir.name}_results.json"
        if not results_file.exists():
            continue
        with open(results_file) as f:
            data = json.load(f)

        comp = data.get("comparison", {}) or {}
        reference_validation = data.get("reference_validation", {}) or {}
        reference_ready = reference_validation.get("benchmark_ready", comp.get("success", False))
        generated_synth = data.get("generated_status") == "passed" or bool(data.get("synth_report"))
        csim_result = data.get("csim")
        cosim_result = data.get("cosim")

        if comp.get("success") and comp.get("ground_truth_report"):
            gen_report = comp.get("generated_report", {})
            gt_report = comp.get("ground_truth_report", {})
            ss = score_step(
                "baseline",
                gen_report,
                gt_report,
                synthesised=generated_synth,
                csim_result=csim_result,
                cosim_result=cosim_result,
            )
        else:
            ss = StepScore(
                step_name="baseline",
                synthesised=generated_synth,
                adp_score=0.0,
                feasibility_score=0.0,
            )
            if csim_result is not None:
                ss.csim_ran = True
                ss.csim_passed = csim_result.get("passed", False)
                ss.csim_score = 100.0 if ss.csim_passed else 0.0
            if cosim_result is not None:
                ss.cosim_ran = True
                ss.cosim_passed = cosim_result.get("passed", False)
                ss.cosim_score = 100.0 if ss.cosim_passed else 0.0
            if not reference_ready:
                ss.step_name = "baseline_invalid_reference"

        benchmarks.append(score_benchmark(bench_dir.name, [ss]))
    return benchmarks


def load_multistep_results(results_dir: str) -> list:
    results_dir = Path(results_dir)
    benchmarks = []
    for bench_dir in sorted(results_dir.iterdir()):
        if not bench_dir.is_dir():
            continue
        ms_files = list(bench_dir.glob("*_multistep_results.json"))
        if not ms_files:
            continue
        with open(ms_files[0]) as f:
            data = json.load(f)
        step_scores = []
        for step_data in data.get("steps", []):
            step_name = step_data.get("step_name", "unknown")
            synthesised = step_data.get("success", False)
            gen_report = step_data.get("report", {})
            gt_report = step_data.get("gt_report", {})
            csim_result = step_data.get("csim")
            cosim_result = step_data.get("cosim")
            if gt_report:
                ss = score_step(step_name, gen_report, gt_report,
                                synthesised=synthesised,
                                csim_result=csim_result,
                                cosim_result=cosim_result)
            else:
                ss = StepScore(step_name=step_name, synthesised=synthesised,
                               adp_score=0.0, feasibility_score=0.0)
                # Still score csim/cosim even without GT
                if csim_result is not None:
                    ss.csim_ran = True
                    ss.csim_passed = csim_result.get("passed", False)
                    ss.csim_score = 100.0 if ss.csim_passed else 0.0
                if cosim_result is not None:
                    ss.cosim_ran = True
                    ss.cosim_passed = cosim_result.get("passed", False)
                    ss.cosim_score = 100.0 if ss.cosim_passed else 0.0
            step_scores.append(ss)
        if step_scores:
            benchmarks.append(score_benchmark(bench_dir.name, step_scores))
    return benchmarks


# ── CLI ───────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="FPGA synthesis quality rubric for HLS translation evaluation")
    parser.add_argument("--results", type=str, default="results",
                        help="Path to results directory")
    parser.add_argument("--multistep", action="store_true",
                        help="Score multistep results")
    parser.add_argument("--json", action="store_true",
                        help="Output JSON")
    args = parser.parse_args()

    if args.multistep:
        benchmarks = load_multistep_results(args.results)
        title = "FPGA Synthesis Quality — Multi-Step Optimisation"
    else:
        benchmarks = load_singleshot_results(args.results)
        title = "FPGA Synthesis Quality — Single-Shot Translation"

    if not benchmarks:
        print(f"No results found in {args.results}")
        sys.exit(1)

    if args.json:
        out = []
        for bs in benchmarks:
            out.append({
                "benchmark": bs.benchmark,
                "composite": bs.composite,
                "grade": letter_grade(bs.composite)[0],
                "synthesis_rate": bs.synthesis_rate,
                "csim_rate": bs.csim_rate,
                "cosim_rate": bs.cosim_rate,
                "avg_latency": bs.avg_latency,
                "avg_fmax": bs.avg_fmax,


                "avg_resources": bs.avg_resources,
                "avg_adp": bs.avg_adp,
                "avg_feasibility": bs.avg_feasibility,
                "steps": [
                    {
                        "step": s.step_name,
                        "synthesised": s.synthesised,
                        "csim_ran": s.csim_ran,
                        "csim_passed": s.csim_passed,
                        "csim_score": s.csim_score,
                        "cosim_ran": s.cosim_ran,
                        "cosim_passed": s.cosim_passed,
                        "cosim_score": s.cosim_score,
                        "latency_ns": s.gen_latency,
                        "gt_latency_ns": s.gt_latency,
                        "latency_cycles": s.gen_latency_cycles,
                        "gt_latency_cycles": s.gt_latency_cycles,
                        "latency_ratio": s.latency_ratio,
                        "latency_score": s.latency_score,
                        "fmax_mhz": s.gen_fmax,
                        "gt_fmax_mhz": s.gt_fmax,
                        "fmax_ratio": s.fmax_ratio,
                        "fmax_score": s.fmax_score,
                        "timing_slack_ns": s.timing_slack_ns,


                        "resource_ratios": s.resource_ratios,
                        "resource_score": s.resource_score,
                        "device_util_pct": s.util_pct,
                        "adp_score": s.adp_score,
                        "feasibility_score": s.feasibility_score,
                        "composite": s.composite,
                    }
                    for s in bs.steps
                ],
            })
        print(json.dumps(out, indent=2))
    else:
        print(format_report(benchmarks, title=title))


if __name__ == "__main__":
    main()
