"""Per-arm success rates: attempted (cell dirs), csim-pass, cosim-pass (usable),
and which benches failed. multistep uses best-so-far as-delivered cosim."""
import json, glob, os

def _cyc(co):
    if not co or not co.get("passed"): return None
    m = co.get("measured") or {}
    return m.get("latency_cycles_avg") or m.get("latency_cycles_min") or co.get("kernel_runtime_cycles")

def audit(dirn, multistep=False):
    pat = "*multistep_results.json" if multistep else "*_results.json"
    dirs = set(os.path.dirname(os.path.dirname(p)) for p in glob.glob(f"{dirn}/hlsfactory_*/**/*_results.json", recursive=True))
    attempted = len([d for d in glob.glob(f"{dirn}/hlsfactory_*/") ])
    csim = cosim = 0; failed = []
    benches = sorted(glob.glob(f"{dirn}/hlsfactory_*/"))
    for bd in benches:
        b = os.path.basename(bd.rstrip("/")).replace("hlsfactory_", "")
        rf = glob.glob(f"{bd}/**/{pat}", recursive=True)
        if not rf:
            failed.append(b + "(no-result)"); continue
        d = json.load(open(rf[0]))
        cs = (d.get("csim") or {}).get("passed") or (d.get("baseline_csim") or {}).get("passed")
        if multistep:
            steps = d.get("steps") or []; bsp = d.get("best_so_far_promotion") or {}; idx = bsp.get("from_step_index")
            c = _cyc(steps[idx].get("cosim")) if (isinstance(idx,int) and 0<=idx<len(steps)) else None
            if c is None:
                cc=[x for x in (_cyc(s.get("cosim")) for s in steps) if x]; c=min(cc) if cc else _cyc(d.get("baseline_cosim"))
        else:
            c = _cyc(d.get("cosim"))
        if cs: csim += 1
        if c: cosim += 1
        else: failed.append(b)
    return attempted, csim, cosim, failed

ARMS = [
 ("one-shot (skills off) flash","results_matrix_u280_ENH_oneshot_OPUS",False),
 ("skill-less A flash","results_matrix_u280_ENH_curated_flash_OPUS",False),
 ("skill-less B flash","results_matrix_u280_ENH_allpositive_flash_OPUS",False),
 ("all-positive flash SKILLS","results_matrix_u280_ENH_allpositive_flash_SKILLS_OPUS",False),
 ("curated multistep (skills)","results_matrix_u280_ENH_curated_multistep_skills_OPUS",True),
 ("all-positive multistep (skills)","results_matrix_u280_ENH_allpositive_multistep_skills_OPUS",True),
]
print(f'{"arm":34}{"attempt":>8}{"csim_ok":>8}{"cosim_ok":>9}{"cosim_rate":>11}')
print("-"*74)
for name,d,ms in ARMS:
    a,cs,co,fl = audit(d,ms)
    rate = f"{100*co/a:.0f}%" if a else "-"
    print(f'{name:34}{a:>8}{cs:>8}{co:>9}{rate:>11}')
    if fl: print(f'      failed/no-cosim: {fl}')
print()
print("flash-curated-WITH-skills dir exists?", bool(glob.glob("results_matrix_u280_ENH_curated_flash_SKILLS*")))
