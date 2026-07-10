"""THE skill-breadth ablation (clean): curated multistep (routed ~1-2 recipes/step)
vs all-positive multistep (all 41/step). Both multistep, both inject skills, so
this isolates BREADTH. as-delivered cosim = best_so_far_promotion step's cosim."""
import json, glob, os, math

VIT_CS = json.load(open("gold_reports_vitis_csynth.json"))
VIT_FULL = json.load(open("gold_reports_vitis_full.json")) if os.path.exists("gold_reports_vitis_full.json") else {}
def gold(b):
    fb = VIT_FULL.get(b)
    if fb and fb.get("cosim_cycles"): return fb["cosim_cycles"]
    cs = VIT_CS.get(b)
    return cs["latency_cycles"] if (cs and cs.get("success") and cs.get("latency_cycles")) else None

def _cyc(co):
    if not co or not co.get("passed"): return None
    m = co.get("measured") or {}
    return m.get("latency_cycles_avg") or m.get("latency_cycles_min") or co.get("kernel_runtime_cycles")

def ms_arm(dirn):
    out = {}
    for rf in glob.glob(f"{dirn}/hlsfactory_*/**/*multistep_results.json", recursive=True):
        b = rf.split(os.sep)[1]
        try: d = json.load(open(rf))
        except Exception: continue
        steps = d.get("steps") or []
        bsp = d.get("best_so_far_promotion") or {}
        idx = bsp.get("from_step_index")
        c = None
        if isinstance(idx, int) and 0 <= idx < len(steps):
            c = _cyc(steps[idx].get("cosim"))
        if c is None:
            cands = [x for x in (_cyc(st.get("cosim")) for st in steps) if x]
            c = min(cands) if cands else _cyc(d.get("baseline_cosim"))
        if c: out[b] = c
    return out

CUR = ms_arm("results_matrix_u280_ENH_curated_multistep_skills_OPUS")
ALP = ms_arm("results_matrix_u280_ENH_allpositive_multistep_skills_OPUS")
# fold in the symm rerun (baseline-cosim-skipped) results
CUR.update(ms_arm("results_matrix_u280_ENH_curated_multistep_skills_symmrerun_OPUS"))
ALP.update(ms_arm("results_matrix_u280_ENH_allpositive_multistep_skills_symmrerun_OPUS"))
common = sorted(set(CUR) & set(ALP))
def gm(xs):
    xs=[x for x in xs if x and x>0]
    return math.exp(sum(math.log(x) for x in xs)/len(xs)) if xs else float("nan")
print(f"curated-ms cells={len(CUR)}  allpos-ms cells={len(ALP)}  common={len(common)}")
print(f'{"bench":15}{"curated":>11}{"allpos":>11}{"gold":>12}{"cur/allpos":>11}{"sp_cur":>8}{"sp_allp":>8}')
print("-"*76)
h2h,spc,spa=[],[],[]
for b in common:
    c=CUR[b]; a=ALP[b]; g=gold(b)
    r=c/a if (c and a) else None  # >1 => all-positive faster (fewer cycles)
    if r: h2h.append(r)
    if g and c: spc.append(g/c)
    if g and a: spa.append(g/a)
    bs=b.replace("hlsfactory_","")
    print(f'{bs:15}{c:>11}{a:>11}{str(g):>12}{(f"{r:.2f}x" if r else "-"):>11}{(f"{g/c:.0f}" if (g and c) else "-"):>8}{(f"{g/a:.0f}" if (g and a) else "-"):>8}')
print("-"*76)
print(f"HEAD-TO-HEAD geomean (curated/allpos cyc; >1 => all-positive faster): {gm(h2h):.2f}x  (n={len(h2h)})")
print(f"speedup vs gold: curated {gm(spc):.1f}x | all-positive {gm(spa):.1f}x")
wins=sum(1 for r in h2h if r>1.05); losses=sum(1 for r in h2h if r<0.95)
print(f"all-positive vs curated per-bench: allpos wins {wins}, ties {len(h2h)-wins-losses}, losses {losses}")
