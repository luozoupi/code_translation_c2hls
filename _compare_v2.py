"""Corrected Opus/Sonnet + guardrail-ablation comparison.

Fixes the multistep over-credit bug the verifier caught: a multistep cell's
result is the FINAL delivered kernel (last step = coalescing). If that kernel
fails csim or cosim, the delivered design is broken -> the cell has NO valid
as-delivered result (it is a correctness failure, not a speedup).

Metrics reported per multistep setup:
  - validity rate: how many of 26 final kernels pass csim+cosim
  - as-delivered geomean cosim/gold: over final-valid cells only
  - best-state geomean (hypothetical): last csim+cosim-valid step (what a
    'revert to best valid step' framework could deliver)

Flash cells: cosim gated on cosim.passed=true.
"""
from __future__ import annotations
import json
from math import exp, log
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SETUPS = {
    "S_off":   ("results_matrix_u280_fullcosim",          "sonnet","flash","noskills"),
    "S_base":  ("results_matrix_u280_fullcosim",          "sonnet","flash","skills"),
    "S_ext":   ("results_matrix_u280_fullcosim_extended", "sonnet","flash","skills"),
    "S_noguard":("results_matrix_u280_fullcosim_noguard", "sonnet","flash","skills"),
    "S_multi": ("results_matrix_u280_multistep_old_skills","sonnet","multistep","skills"),
    "O_off":   ("results_matrix_u280_fullcosim_OPUS",          "opus","flash","noskills"),
    "O_base":  ("results_matrix_u280_fullcosim_OPUS",          "opus","flash","skills"),
    "O_ext":   ("results_matrix_u280_fullcosim_extended_OPUS", "opus","flash","skills"),
    "O_noguard":("results_matrix_u280_fullcosim_noguard_OPUS", "opus","flash","skills"),
    "O_multi": ("results_matrix_u280_multistep_base_OPUS",     "opus","multistep","skills"),
}
BENCHES = [f"hlsfactory_{b}" for b in [
    "2mm","3mm","atax","bicg","cholesky","correlation","covariance","doitgen",
    "durbin","fdtd-2d","floyd-warshall","gemm","gemver","gesummv","gramschmidt",
    "jacobi-1d","jacobi-2d","lu","ludcmp","mvt","nussinov","symm","syr2k","syrk",
    "trisolv","trmm"]]

def _load(p):
    try: return json.loads(p.read_text())
    except Exception: return None

def _cell(setup, bench):
    d, model, mode, tag = SETUPS[setup]
    return ROOT / d / bench / f"{model}__{mode}__{tag}", mode

def flash_cosim(setup, bench):
    cell, _ = _cell(setup, bench)
    j = _load(cell / f"{bench}_results.json")
    if not j: return None
    c = j.get("cosim") or {}
    if not c.get("passed"): return None
    return (c.get("measured") or {}).get("latency_cycles_avg")

def multi_cosim(setup, bench, mode_metric):
    """mode_metric: 'delivered' (final step, gated) or 'best' (last valid step)."""
    cell, _ = _cell(setup, bench)
    j = _load(cell / f"{bench}_multistep_results.json")
    if not j: return None
    steps = j.get("steps") or []
    if not steps: return None
    if mode_metric == "delivered":
        fin = steps[-1]
        if (fin.get("csim") or {}).get("passed") and (fin.get("cosim") or {}).get("passed"):
            return ((fin.get("cosim") or {}).get("measured") or {}).get("latency_cycles_avg")
        return None
    else:  # best
        for s in reversed(steps):
            if (s.get("csim") or {}).get("passed") and (s.get("cosim") or {}).get("passed"):
                cyc = ((s.get("cosim") or {}).get("measured") or {}).get("latency_cycles_avg")
                if cyc: return cyc
        return None

def cosim(setup, bench, multi_metric="delivered"):
    _, _, mode, _ = SETUPS[setup]
    return multi_cosim(setup, bench, multi_metric) if mode == "multistep" else flash_cosim(setup, bench)

def gold(bench):
    for s in ("S_off","O_off","S_base","O_base"):
        cell,_ = _cell(s, bench)
        j = _load(cell / f"{bench}_results.json")
        if j:
            m = ((j.get("reference_validation") or {}).get("cosim") or {}).get("measured") or {}
            if m.get("latency_cycles_avg"): return m["latency_cycles_avg"]
    return None

def geomean(xs):
    xs=[x for x in xs if x and x>0]
    return exp(sum(log(x) for x in xs)/len(xs)) if xs else None

GOLD={b:gold(b) for b in BENCHES}

labels={"S_off":"Sonnet flash no-skills","O_off":"Opus flash no-skills",
 "S_base":"Sonnet flash base","O_base":"Opus flash base",
 "S_ext":"Sonnet flash ext(guards)","O_ext":"Opus flash ext(guards)",
 "S_noguard":"Sonnet flash ext-NOGUARD","S_multi":"Sonnet multistep","O_multi":"Opus multistep"}

print("="*96)
print("  TABLE 1 — cosim/gold geomean (lower=faster than Direct Vitis). Flash gated on cosim.passed;")
print("            multistep = AS-DELIVERED (final kernel must pass csim+cosim).")
print("="*96)
print(f"  {'setup':<30}{'valid_n':>9}{'geomean':>10}{'wins<.95':>10}{'loss>1.05':>11}")
print("  "+"-"*70)
for s in ["S_off","O_off","S_base","O_base","S_ext","O_ext","S_noguard","S_multi","O_multi"]:
    rat=[cosim(s,b)/GOLD[b] for b in BENCHES if cosim(s,b) and GOLD[b]]
    g=geomean(rat)
    print(f"  {labels[s]:<30}{len(rat):>9}{(f'{g:.3f}x' if g else '-'):>10}"
          f"{sum(1 for r in rat if r<0.95):>10}{sum(1 for r in rat if r>1.05):>11}")

print()
print("="*96)
print("  TABLE 1b — multistep validity + best-state hypothetical")
print("="*96)
for s in ["S_multi","O_multi"]:
    deliv=[(b,cosim(s,b,'delivered')) for b in BENCHES]
    nvalid=sum(1 for _,c in deliv if c)
    rat_d=[c/GOLD[b] for b,c in deliv if c and GOLD[b]]
    best=[(b,cosim(s,b,'best')) for b in BENCHES]
    rat_b=[c/GOLD[b] for b,c in best if c and GOLD[b]]
    gd=geomean(rat_d); gb=geomean(rat_b)
    print(f"  {labels[s]:<22} final-valid={nvalid}/26  as-delivered geomean={gd:.3f}x (n={len(rat_d)})  "
          f"best-state geomean={gb:.3f}x (n={len(rat_b)})")

print()
print("="*96)
print("  TABLE 2 — Opus vs Sonnet head-to-head (geomean cosim ratio; <1 = Opus faster). matched cells only.")
print("="*96)
for o,sn,nm,mm in [("O_off","S_off","flash no-skills","delivered"),
                   ("O_base","S_base","flash base","delivered"),
                   ("O_ext","S_ext","flash ext(guards)","delivered"),
                   ("O_multi","S_multi","multistep as-delivered","delivered"),
                   ("O_multi","S_multi","multistep best-state","best")]:
    rat=[cosim(o,b,mm)/cosim(sn,b,mm) for b in BENCHES if cosim(o,b,mm) and cosim(sn,b,mm)]
    g=geomean(rat)
    print(f"  {nm:<26} n={len(rat):<3} Opus/Sonnet={ (f'{g:.3f}x' if g else '-'):<9} "
          f"Opus-wins={sum(1 for r in rat if r<0.95):<3} Sonnet-wins={sum(1 for r in rat if r>1.05)}")

print()
print("="*96)
print("  TABLE 3 — GUARDRAIL ABLATION 2x3 (flash, cosim/gold geomean; lower=better)")
print("="*96)
print(f"  {'model':<10}{'base':>22}{'ext WITH guards':>22}{'ext NO-guard':>22}")
print("  "+"-"*76)
for model, (sb, se, sn) in [("Sonnet",("S_base","S_ext","S_noguard")),
                            ("Opus",  ("O_base","O_ext","O_noguard"))]:
    cells=[]
    for s in (sb, se, sn):
        rat=[cosim(s,b)/GOLD[b] for b in BENCHES if cosim(s,b) and GOLD[b]]
        g=geomean(rat)
        cells.append(f"{g:.3f}x (n={len(rat)})")
    print(f"  {model:<10}{cells[0]:>22}{cells[1]:>22}{cells[2]:>22}")
print()
print("  Per-model guard effect (ext-guards/base) and recovery (noguard/base):")
for model,(sb,se,sn) in [("Sonnet",("S_base","S_ext","S_noguard")),("Opus",("O_base","O_ext","O_noguard"))]:
    gb=geomean([cosim(sb,b)/GOLD[b] for b in BENCHES if cosim(sb,b) and GOLD[b]])
    ge=geomean([cosim(se,b)/GOLD[b] for b in BENCHES if cosim(se,b) and GOLD[b]])
    gn=geomean([cosim(sn,b)/GOLD[b] for b in BENCHES if cosim(sn,b) and GOLD[b]])
    print(f"    {model:<8} base={gb:.3f}x  guards={ge:.3f}x ({ge/gb:.2f}x vs base)  noguard={gn:.3f}x ({gn/gb:.2f}x vs base)")
