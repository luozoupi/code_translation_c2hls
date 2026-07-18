"""Clean Opus-vs-Sonnet comparison: as-delivered = BEST cosim-passing kernel per
cell (multistep: min over cosim-verified steps; flash/oneshot: top-level cosim),
scored vs the CORRECTED vitis-flow gold (gold_reports_vitis_full.json cosim,
csynth fallback) — the same principled method as _compare_ms_ablation.py, not the
schema's embedded (vivado-flow) gold. Head-to-head per shared arm.

Run from the project root: python3 analysis_tools/_compare_opus_sonnet.py
"""
import json, glob, os, math

VIT_CS = json.load(open("gold_reports_vitis_csynth.json"))
VIT_FULL = json.load(open("gold_reports_vitis_full.json")) if os.path.exists("gold_reports_vitis_full.json") else {}

def gold(b):
    fb = VIT_FULL.get(b) or {}
    if fb.get("cosim_cycles"): return fb["cosim_cycles"]
    cs = VIT_CS.get(b) or {}
    return cs["latency_cycles"] if (cs.get("success") and cs.get("latency_cycles")) else None

def _cyc(co):
    if not co or not co.get("passed"): return None
    m = co.get("measured") or {}
    return m.get("latency_cycles_avg") or m.get("latency_cycles_min") or co.get("kernel_runtime_cycles")

def _verified(csim, co):
    return bool((csim or {}).get("passed") and (co or {}).get("passed") and _cyc(co))

def mine(dirs, multistep):
    """bench -> best cosim-verified cycles across the given dirs (folds symmrerun)."""
    out = {}
    for dirn in dirs:
        pat = "*_multistep_results.json" if multistep else "*_results.json"
        for rf in glob.glob(f"{dirn}/hlsfactory_*/**/{pat}", recursive=True):
            if not multistep and "multistep" in os.path.basename(rf):
                continue
            b = rf.replace("\\", "/").split("/")[1]
            try: d = json.load(open(rf))
            except Exception: continue
            best = None
            if multistep:
                for s in d.get("steps") or []:
                    if _verified(s.get("csim"), s.get("cosim")):
                        c = _cyc(s.get("cosim"))
                        best = c if best is None else min(best, c)
            else:
                if _verified(d.get("csim"), d.get("cosim")):
                    best = _cyc(d.get("cosim"))
            if best:
                out[b] = min(out.get(b, best), best)
    return out

R = "results_matrix_u280_ENH_"
ARMS = [
    ("oneshot",            False, [R+"oneshot_OPUS"],   [R+"oneshot_SONNET"]),
    ("multistep curated",  True,  [R+"curated_multistep_skills_OPUS", R+"curated_multistep_skills_symmrerun_OPUS"],
                                  [R+"curated_multistep_skills_SONNET"]),
    ("multistep allpos",   True,  [R+"allpositive_multistep_skills_OPUS", R+"allpositive_multistep_skills_symmrerun_OPUS"],
                                  [R+"allpositive_multistep_skills_SONNET"]),
]

def gm(xs):
    xs = [x for x in xs if x and x > 0]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")

print(f"{'arm':20}{'opus vs gold':>16}{'sonnet vs gold':>16}{'common':>8}{'O/S (common)':>14}")
print("-" * 74)
for name, ms, odirs, sdirs in ARMS:
    O, S = mine(odirs, ms), mine(sdirs, ms)
    o_sp = [gold(b)/O[b] for b in O if gold(b)]
    s_sp = [gold(b)/S[b] for b in S if gold(b)]
    common = sorted(set(O) & set(S) & {b for b in O if gold(b)})
    oc = gm([gold(b)/O[b] for b in common])
    sc = gm([gold(b)/S[b] for b in common])
    ratio = oc / sc if sc and sc == sc else float("nan")
    print(f"{name:20}{gm(o_sp):>13.1f}x{'('+str(len(o_sp))+')':>4}"
          f"{gm(s_sp):>13.1f}x{'('+str(len(s_sp))+')':>4}"
          f"{len(common):>6}  {ratio:>10.2f}x")
print("-" * 74)
print("O/S (common) > 1 => Opus faster on the shared benches; < 1 => Sonnet faster.")
print("gold = vitis-flow cosim (gold_reports_vitis_full.json), csynth fallback.")
