"""Experiment A (plain HLSmith on the fixed bench, study/exp_hlsmith/runs): same extraction as extract.py -> rows_expA.json."""
import glob, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import extract
from registry import ROOT

rows = []
for d in sorted(glob.glob(ROOT + "/study/exp_hlsmith/runs/*_pl_*/")):
    r = dict(family="F5-expA", dir=d, tag=os.path.basename(d.rstrip("/")), excluded=None)
    rows.append(extract.process(r, dict(family=r["family"], tag=r["tag"], excluded=None)))
json.dump(rows, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "rows_expA.json"), "w"), indent=1, default=str)
print(len(rows))
