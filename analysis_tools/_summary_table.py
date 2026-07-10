"""Consolidated table: skill-injection setup x mode -> speedup vs gold + tokens.
cosim as-delivered (flash: top-level cosim; multistep: best_so_far step). Speedup
geomean computed on the COMMON bench set (present in all arms) for comparability,
plus each arm's full-coverage geomean."""
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

def flash_arm(dirn):
    out = {}
    for rf in glob.glob(f"{dirn}/hlsfactory_*/**/*_results.json", recursive=True):
        b = rf.split(os.sep)[1]
        try: c = _cyc(json.load(open(rf)).get("cosim"))
        except Exception: c = None
        if c: out[b] = c
    return out
def ms_arm(dirn):
    out = {}
    for rf in glob.glob(f"{dirn}/hlsfactory_*/**/*multistep_results.json", recursive=True):
        b = rf.split(os.sep)[1]
        try: d = json.load(open(rf))
        except Exception: continue
        steps = d.get("steps") or []; bsp = d.get("best_so_far_promotion") or {}; idx = bsp.get("from_step_index")
        c = _cyc(steps[idx].get("cosim")) if (isinstance(idx,int) and 0<=idx<len(steps)) else None
        if c is None:
            cc=[x for x in (_cyc(s.get("cosim")) for s in steps) if x]; c=min(cc) if cc else _cyc(d.get("baseline_cosim"))
        if c: out[b]=c
    return out
def tokens_per_cell(dirn):
    tt=cells=0
    for f in set(glob.glob(f"{dirn}/hlsfactory_*/**/*_results.json", recursive=True)):
        try: lu=json.load(open(f)).get("llm_usage") or {}
        except: continue
        if lu: cells+=1; tt+=lu.get("total_tokens") or 0
    return (tt//cells if cells else 0), cells

ARMS = [
 ("one-shot (skills OFF)","flash",   "results_matrix_u280_ENH_oneshot_OPUS", flash_arm),
 ("skill-less A","flash",            "results_matrix_u280_ENH_curated_flash_OPUS", flash_arm),
 ("skill-less B","flash",            "results_matrix_u280_ENH_allpositive_flash_OPUS", flash_arm),
 ("all-positive (41)","flash",       "results_matrix_u280_ENH_allpositive_flash_SKILLS_OPUS", flash_arm),
 ("curated (routed)","multistep",    "results_matrix_u280_ENH_curated_multistep_skills_OPUS", ms_arm),
 ("all-positive (41)","multistep",   "results_matrix_u280_ENH_allpositive_multistep_skills_OPUS", ms_arm),
]
data = {name+"|"+mode: fn(d) for name,mode,d,fn in ARMS}
common = None
for k,v in data.items():
    bs=set(b for b in v if gold(b))
    common = bs if common is None else (common & bs)
def gm(xs):
    xs=[x for x in xs if x and x>0]
    return math.exp(sum(math.log(x) for x in xs)/len(xs)) if xs else float("nan")
print(f"common bench set ({len(common)}): {sorted(b.replace('hlsfactory_','') for b in common)}")
print()
print(f'{"setup":20}{"mode":11}{"cells":>6}{"sp_gold(common)":>16}{"sp_gold(full)":>14}{"tok/cell":>10}')
print("-"*80)
for name,mode,d,fn in ARMS:
    v=data[name+"|"+mode]
    spc=gm([gold(b)/v[b] for b in common if b in v])
    spf=gm([gold(b)/v[b] for b in v if gold(b)])
    tpc,cells=tokens_per_cell(d)
    print(f'{name:20}{mode:11}{cells:>6}{spc:>15.1f}x{spf:>13.1f}x{tpc:>10,}')
