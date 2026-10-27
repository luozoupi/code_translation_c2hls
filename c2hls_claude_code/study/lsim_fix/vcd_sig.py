#!/usr/bin/env python3
"""Per-cycle values of chosen signals from an xsim VCD (sampled at the falling edge of the top ap_clk, i.e. the settled
values of the cycle that the next rising edge acts on).

  vcd_sig.py ALL.vcd.gz SCOPE_REGEX SIG[,SIG...] [--from C] [--to C] [--changes]
SCOPE_REGEX matches the dotted scope path (e.g. 'comp0_U0$'); SIG are exact names in that scope ('*' lists them).
Cycle 0 is the first rising edge after ap_rst_n goes high. --changes prints only cycles where a value changed.
"""
import argparse, gzip, re, sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("vcd")
    ap.add_argument("scope")
    ap.add_argument("sigs")
    ap.add_argument("--from", dest="c0", type=int, default=0)
    ap.add_argument("--to", dest="c1", type=int, default=10 ** 12)
    ap.add_argument("--changes", action="store_true")
    a = ap.parse_args()
    want = a.sigs.split(",")
    srx = re.compile(a.scope)
    f = gzip.open(a.vcd, "rt") if a.vcd.endswith(".gz") else open(a.vcd)
    stack, ids, clk, rst = [], {}, None, None
    names = {}
    for line in f:
        t = line.split()
        if not t:
            continue
        if t[0] == "$scope":
            stack.append(t[2])
        elif t[0] == "$upscope":
            stack.pop()
        elif t[0] == "$var":
            code, name = t[3], t[4]
            path = ".".join(stack)
            if len(stack) == 2 and name == "ap_clk":
                clk = code
            if len(stack) == 2 and name == "ap_rst_n":
                rst = code
            if srx.search(path):
                names.setdefault(path, []).append(name)
                if name in want:
                    ids.setdefault(code, []).append(name)
        elif t[0] == "$enddefinitions":
            break
    if want == ["*"]:
        for p, ns in names.items():
            print(p, " ".join(ns))
        return
    hit = sorted({n for v in ids.values() for n in v})
    if not hit:
        sys.exit(f"no signals {want} in scopes matching {a.scope}: {list(names)[:10]}")
    print("cycle " + " ".join(want), flush=True)
    val = {w: "x" for w in want}
    cur_clk, rst_v, cyc, last = "0", "0", -1, None
    for line in f:
        c = line[0]
        if c == "#":
            continue
        if c in "01xzXZ":
            v, code = c, line[1:].strip()
        elif c == "b":
            v, code = line[1:].split()
            v = str(int(v, 2)) if set(v) <= {"0", "1"} else v
        else:
            continue
        if code == clk:
            if cur_clk == "0" and v == "1" and rst_v == "1":
                cyc += 1
            if cur_clk == "1" and v == "0" and cyc >= 0:
                if cyc > a.c1:
                    break
                if cyc >= a.c0:
                    row = [val[w] for w in want]
                    if not a.changes or row != last:
                        print(cyc, " ".join(row))
                    last = row
            cur_clk = v
        if code == rst:
            rst_v = v
        for n in ids.get(code, ()):
            val[n] = v


if __name__ == "__main__":
    main()
