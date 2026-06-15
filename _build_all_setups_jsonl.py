"""Build the unified all-setups schema JSONL from the per-sweep JSONLs.

Inputs (all already emitted with the new (A)-style origin_version scheme):
  u280__flash_baseAB_skills_schema.jsonl       (flash sweep: base + no_skills A/B)
  u280__flash_extended_skills_schema.jsonl     (flash + extended hard-guards)
  u280__multistep_base_skills_schema.jsonl     (multistep + base skills)

Output:
  u280__all_setups_schema.jsonl

Baseline (Direct Vitis) records are deduped across inputs on
(group_path, report_type). Candidate records are deduped on
(group_path, report_type, origin_version, variant.index) — under the new
scheme each (origin_version, variant.index) tuple is unique per orchestrator
run so collisions only happen if the same input file is concatenated twice.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCHEMAS = ROOT / "schemas"
INPUTS = [
    SCHEMAS / "u280__flash_baseAB_skills_schema.jsonl",
    SCHEMAS / "u280__flash_extended_skills_schema.jsonl",
    SCHEMAS / "u280__multistep_base_skills_schema.jsonl",
]
OUT = SCHEMAS / "u280__all_setups_schema.jsonl"
SCHEMAS.mkdir(exist_ok=True)


def load(p): return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def variant_name_of(r): return ((r.get("implementation") or {}).get("variant") or {}).get("name")
def origin_of(r): return (r.get("implementation") or {}).get("origin")
def version_of(r): return (r.get("implementation") or {}).get("origin_version")
def gp_of(r): return tuple((r.get("problem") or {}).get("group_path") or [])


records = []
seen_baseline = set()    # (group_path, report_type)
seen_candidate = set()   # (group_path, report_type, origin_version, variant.index)

def add(r):
    rt = r["report_type"]
    gp = gp_of(r)
    if variant_name_of(r) == "baseline":
        key = (gp, rt)
        if key in seen_baseline:
            return
        seen_baseline.add(key)
    else:
        ver = version_of(r)
        vi = ((r.get("implementation") or {}).get("variant") or {}).get("index")
        key = (gp, rt, ver, vi)
        if key in seen_candidate:
            print(f"WARN: dup candidate {key} -- skipping")
            return
        seen_candidate.add(key)
    records.append(r)


for p in INPUTS:
    if not p.exists():
        print(f"skip: {p} missing")
        continue
    for r in load(p):
        add(r)

with OUT.open("w", encoding="utf-8") as f:
    for r in records:
        f.write(json.dumps(r, separators=(", ", ": ")) + "\n")

# Summary
from collections import Counter
counts = Counter()
benches = set()
for r in records:
    counts[(r["report_type"], origin_of(r), version_of(r), variant_name_of(r))] += 1
    gp = gp_of(r)
    if gp: benches.add(gp[0])

print(f"\nWrote {len(records)} records to {OUT}")
print(f"Benches: {len(benches)}")
print(f"\n{'report_type':<10} {'origin':<22} {'origin_version':<48} {'variant.name':<14}  n")
print("-" * 105)
for (rt, o, ver, vn), n in sorted(counts.items()):
    print(f"  {rt:<10} {o!r:<22} {ver!r:<48} {vn!r:<14}  {n}")
