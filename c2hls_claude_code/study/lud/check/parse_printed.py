#!/usr/bin/env python3
"""Compare the matrices printed by AutoSA's LU host (L_golden/U_golden vs L/U) in vitis_csim.log.
The printed L/U are the real outputs (L shown from L[j][i], j<i; U upper). AutoSA's own error count instead compares
L[i][j] for j<=i, a region the algorithm never writes."""
import re, sys
def mats(path):
    out, cur, name = {}, None, None
    for line in open(path, errors="replace"):
        s = line.strip()
        if s in ("A:", "L_golden:", "U_golden:", "L:", "U:"):
            name = s[:-1]; out[name] = []; continue
        if name and re.match(r"^-?[0-9.]+(\s+-?[0-9.naif]+)*$", s):
            out[name].append([float(x) for x in s.split()])
        elif name and s and not re.match(r"^-?[0-9]", s):
            name = None
    return out
for p in sys.argv[1:]:
    m = mats(p)
    def diff(a, b):
        A, B = m.get(a, []), m.get(b, [])
        if not A or len(A) != len(B): return "missing"
        bad = [(i, j) for i in range(len(A)) for j in range(len(A[i])) if abs(A[i][j] - B[i][j]) > 1e-3 * max(1, abs(A[i][j]))]
        return f"{len(bad)} bad" + (f" e.g. {bad[:4]}" if bad else "")
    print(p.split("/dse/")[-1].split("/")[0], "| L:", diff("L_golden", "L"), "| U:", diff("U_golden", "U"))
