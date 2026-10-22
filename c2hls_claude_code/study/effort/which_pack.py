#!/usr/bin/env python3
"""Identify which skills pack a run used by comparing its saved skills_source.json with candidate pack files."""
import json, glob, sys, os, hashlib
P = "/scratch/asa582/workspaces/code_translation_c2hls/hls_full_optimization_skills_schema_1_1_package/"
F = "/scratch/asa582/workspaces/c2hls_claude_code/flash_refined/"
cands = glob.glob(P + "*90skills*.json") + glob.glob(F + "*90skills*.json")
def load(p):
    d = json.load(open(p)); s = d["skills"] if isinstance(d, dict) and "skills" in d else d
    return s
def ids(s): return sorted(x.get("id") or x.get("skill_id") or x.get("name") for x in s)
def h(s): return hashlib.sha1(json.dumps(s, sort_keys=True).encode()).hexdigest()[:10]
C = {os.path.basename(p): (ids(load(p)), h(load(p))) for p in cands}
for c, (i, hh) in sorted(C.items()): print(f"pack {len(i):4d} skills  {hh}  {c}")
print()
for run in sys.argv[1:]:
    s = load(run); i, hh = ids(s), h(s)
    exact = [c for c, (ci, ch) in C.items() if ch == hh]
    same_ids = [c for c, (ci, ch) in C.items() if ci == i]
    best = max(C.items(), key=lambda kv: len(set(kv[1][0]) & set(i)) / max(1, len(set(kv[1][0]) | set(i))))
    j = len(set(best[1][0]) & set(i)) / len(set(best[1][0]) | set(i))
    print(f"{run.split('/runs')[-1][:70]:70s} n={len(i)} exact={exact or '-'} ids={same_ids or '-'} nearest={best[0]} (jaccard {j:.2f})")
