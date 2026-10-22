#!/usr/bin/env python3
"""Per-transaction AXI timing from a cosim VCD written by vcd_rerun.sh.

  parse_axi_vcd.py CONFIG_DIR [--json OUT]

Signals are sampled at each rising ap_clk edge (values just before the edge, as the RTL sees them). Cycle 0 is
the first cycle with ap_start = 1. Per m_axi port it lists AR/AW handshakes (cycle, beat address, len), R/W beats
(first, last, gaps) and B handshakes, then prints derived quantities: AR -> first R of that burst, R-beat gaps,
last W of a burst -> AW (CONSERVATIVE throttling), AW -> B, and requests in flight.
"""
import argparse, gzip, json, re, sys
from collections import defaultdict
from pathlib import Path


def read_vcd(path):
    ids, names = {}, {}
    scope = []
    op = gzip.open if str(path).endswith(".gz") else open
    with op(path, "rt") as f:
        lines = iter(f)
        for line in lines:
            t = line.split()
            if not t:
                continue
            if t[0] == "$scope":
                scope.append(t[2])
            elif t[0] == "$upscope":
                scope.pop()
            elif t[0] == "$var":
                code, name = t[3], t[4]
                full = "/".join(scope[1:] + [name]) if len(scope) > 1 else name
                ids.setdefault(code, []).append(full)
            elif t[0] == "$enddefinitions":
                break
        cur = {}
        changes = []  # (time, code, value)
        time = 0
        for line in lines:
            line = line.strip()
            if not line:
                continue
            c = line[0]
            if c == "#":
                time = int(line[1:])
            elif c in "01xzXZ" and len(line) > 1 and line[1:] in ids:
                changes.append((time, line[1:], c))
            elif c in "bB":
                v, code = line[1:].split()
                changes.append((time, code, v))
    return ids, changes


def sample(ids, changes, want):
    """Return list of dicts (one per rising ap_clk edge) with the values of `want` just before that edge."""
    code_of = {}
    for code, names in ids.items():
        for n in names:
            parts = n.split("/")
            # top-level DUT ports by short name; adapter ports as "<inst>/<port>"
            short = parts[-1] if not (len(parts) >= 2 and parts[-2].endswith("_m_axi_U")) else "/".join(parts[-2:])
            if short in want and short not in code_of:
                code_of[short] = code
    clk = code_of["ap_clk"]
    inv = defaultdict(list)
    for k, v in code_of.items():
        inv[v].append(k)
    cur = {k: "0" for k in code_of}
    out = []
    i = 0
    by_time = defaultdict(list)
    for t, code, v in changes:
        if code in inv:
            by_time[t].append((code, v))
    for t in sorted(by_time):
        evs = by_time[t]
        clk_rise = any(code == clk and v == "1" for code, v in evs) and cur["ap_clk"] == "0"
        if clk_rise:
            out.append(dict(cur))
        for code, v in evs:
            for k in inv[code]:
                cur[k] = v
    return out


def val(s):
    try:
        return int(s, 2)
    except ValueError:
        return None


def analyse(cycles, ports):
    t0 = next(i for i, c in enumerate(cycles) if c.get("ap_start") == "1")
    done = [i - t0 for i, c in enumerate(cycles) if c.get("ap_done") == "1"]
    res = {"ap_start": 0, "ap_done": done[0] if done else None, "ports": {}}
    for p in ports:
        g = lambda c, s: c.get(f"m_axi_{p}_{s}")
        ar, r, aw, w, b = [], [], [], [], []
        for i, c in enumerate(cycles):
            cy = i - t0
            if g(c, "ARVALID") == "1" and g(c, "ARREADY") == "1":
                ar.append((cy, val(g(c, "ARADDR")) // 64, val(g(c, "ARLEN")) + 1))
            if g(c, "RVALID") == "1" and g(c, "RREADY") == "1":
                r.append((cy, g(c, "RLAST") == "1"))
            if g(c, "AWVALID") == "1" and g(c, "AWREADY") == "1":
                aw.append((cy, val(g(c, "AWADDR")) // 64, val(g(c, "AWLEN")) + 1))
            if g(c, "WVALID") == "1" and g(c, "WREADY") == "1":
                w.append((cy, g(c, "WLAST") == "1"))
            if g(c, "BVALID") == "1" and g(c, "BREADY") == "1":
                b.append(cy)
        u = lambda c, s: c.get(f"{p}_m_axi_U/I_{s}")
        uar, ur, uaw, uw, ub = [], [], [], [], []
        for i, c in enumerate(cycles):
            cy = i - t0
            if u(c, "ARVALID") == "1" and u(c, "ARREADY") == "1":
                uar.append(cy)
            if u(c, "RVALID") == "1" and u(c, "RREADY") == "1":
                ur.append(cy)
            if u(c, "AWVALID") == "1" and u(c, "AWREADY") == "1":
                uaw.append(cy)
            if u(c, "WVALID") == "1" and u(c, "WREADY") == "1":
                uw.append(cy)
            if u(c, "BVALID") == "1" and u(c, "BREADY") == "1":
                ub.append(cy)
        d = {"user": {"readreqs": uar, "reads": ur, "writereqs": uaw, "writes": uw, "writeresps": ub},
             "ar": ar, "aw": aw, "b": b, "r_first": r[0][0] if r else None, "r_last": r[-1][0] if r else None,
             "r_beats": len(r), "w_first": w[0][0] if w else None, "w_last": w[-1][0] if w else None, "w_beats": len(w)}
        # first R beat of each burst
        rb, k = [], 0
        for cy, last in r:
            if k == 0 or prev_last:
                rb.append(cy)
            prev_last = last
            k += 1
        d["ar_to_r"] = [rb[j] - ar[j][0] for j in range(min(len(rb), len(ar)))]
        d["r_gaps"] = sorted({r[j + 1][0] - r[j][0] for j in range(len(r) - 1)})
        wl = [cy for cy, last in w if last]
        d["wlast_to_aw"] = [aw[j][0] - wl[j] for j in range(min(len(aw), len(wl)))]
        d["aw_to_b"] = [b[j] - aw[j][0] for j in range(min(len(aw), len(b)))]
        d["wlast_to_b"] = [b[j] - wl[j] for j in range(min(len(wl), len(b)))]
        res["ports"][p] = d
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cfg")
    ap.add_argument("--json")
    a = ap.parse_args()
    ids, changes = read_vcd(Path(a.cfg) / "axi.vcd.gz")
    names = {n.split("/")[-1] for ns in ids.values() for n in ns if "_m_axi_U/" not in n}
    ports = sorted({m.group(1) for n in names for m in [re.match(r"m_axi_(\w+?)_ARVALID$", n)] if m})
    want = {"ap_clk", "ap_start", "ap_done"} | {n for n in names if n.startswith("m_axi_")}
    want |= {"/".join(n.split("/")[-2:]) for ns in ids.values() for n in ns if "_m_axi_U/I_" in n}
    res = analyse(sample(ids, changes, want), ports)
    if a.json:
        Path(a.json).write_text(json.dumps(res) + "\n")
    print(f"{a.cfg}: ap_done at {res['ap_done']}")
    for p, d in res["ports"].items():
        if not (d["ar"] or d["aw"]):
            continue
        print(f"  {p}: AR n={len(d['ar'])} first={d['ar'][:3]} last={d['ar'][-1:]}  R beats={d['r_beats']} "
              f"first={d['r_first']} last={d['r_last']} gaps={d['r_gaps'][:6]}")
        print(f"      AR->R first beat per burst: {sorted(set(d['ar_to_r']))[:8]} (first {d['ar_to_r'][:4]})")
        print(f"      AW n={len(d['aw'])} first={d['aw'][:3]} last={d['aw'][-1:]}  W beats={d['w_beats']} "
              f"first={d['w_first']} last={d['w_last']}  B={d['b'][:3]}..{d['b'][-2:]}")
        print(f"      Wlast->AW {sorted(set(d['wlast_to_aw']))[:6]}  AW->B {sorted(set(d['aw_to_b']))[:6]}  "
              f"Wlast->B {sorted(set(d['wlast_to_b']))[:6]}")


if __name__ == "__main__":
    main()
