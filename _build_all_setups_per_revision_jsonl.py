"""Build the combined per-revision (B) schema JSONL.

Inputs (each emitted by _emit_schema_records_per_revision.py):
  u280__flash_baseAB_per_revision_schema.jsonl
  u280__flash_extended_per_revision_schema.jsonl
  u280__multistep_base_skills_per_revision_schema.jsonl

Output:
  u280__all_setups_per_revision_schema.jsonl

Dedup rules:
  - baseline (group_path, report_type)
  - candidate (group_path, report_type, origin_version, variant.index)
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCHEMAS = ROOT / "schemas"
INPUTS = [
    SCHEMAS / "u280__flash_baseAB_per_revision_schema.jsonl",
    SCHEMAS / "u280__flash_extended_per_revision_schema.jsonl",
    SCHEMAS / "u280__multistep_base_skills_per_revision_schema.jsonl",
]
OUT = SCHEMAS / "u280__all_setups_per_revision_schema.jsonl"
SCHEMAS.mkdir(exist_ok=True)

def load(p): return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
def variant_name_of(r): return ((r.get("implementation") or {}).get("variant") or {}).get("name")
def origin_of(r): return (r.get("implementation") or {}).get("origin")
def version_of(r): return (r.get("implementation") or {}).get("origin_version")
def index_of(r): return ((r.get("implementation") or {}).get("variant") or {}).get("index")
def gp_of(r): return tuple((r.get("problem") or {}).get("group_path") or [])

records = []
seen_baseline = set()
seen_candidate = set()

def add(r):
    rt = r["report_type"]
    gp = gp_of(r)
    if variant_name_of(r) == "baseline":
        key = (gp, rt)
        if key in seen_baseline:
            return
        seen_baseline.add(key)
    else:
        ver = version_of(r); vi = index_of(r)
        key = (gp, rt, ver, vi)
        if key in seen_candidate:
            print(f"WARN dup: {key}")
            return
        seen_candidate.add(key)
    records.append(r)

for p in INPUTS:
    if not p.exists():
        print(f"skip missing: {p}")
        continue
    for r in load(p): add(r)

with OUT.open("w", encoding="utf-8") as f:
    for r in records:
        f.write(json.dumps(r, separators=(", ", ": ")) + "\n")

from collections import Counter
counts = Counter()
benches = set()
for r in records:
    counts[(r["report_type"], origin_of(r), version_of(r), variant_name_of(r), index_of(r))] += 1
    gp = gp_of(r)
    if gp: benches.add(gp[0])

print(f"\nWrote {len(records)} records to {OUT}")
print(f"Benches: {len(benches)}")
print(f"\n{'rt':<10} {'origin':<22} {'origin_version':<40} {'idx':>4} {'variant.name':<24} n")
print("-" * 115)
for (rt, o, ver, vn, vi), n in sorted(counts.items(), key=lambda x: (x[0][0], str(x[0][2]), x[0][4] if x[0][4] is not None else -1)):
    print(f"  {rt:<10} {o!r:<22} {ver!r:<40} {vi if vi is not None else '-':>4} {vn!r:<24} {n}")
