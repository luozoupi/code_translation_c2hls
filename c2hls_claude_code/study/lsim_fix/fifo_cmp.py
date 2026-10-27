#!/usr/bin/env python3
"""Per-FIFO write/read cycles: LightningSim (v2 FRP rule) against the cosim VCD, to find where the two first part.

  fifo_cmp.py SOLUTION_DIR OUT.json NPTR [COSIM]      (same argv as frp_eval.py; the VCD is OUT's dir/all.vcd.gz)
VCD events: a write is a cycle with if_write & if_full_n, a read one with if_read & if_empty_n, on each <fifo>_U.
FIFOCMP_BIG=1: all LSIM depths unbounded. FIFOCMP_DUMP=<regex>: also store the shifted raw cycles of matching FIFOs.
Both time axes are shifted so the first write of any FIFO is cycle 0. OUT.json per FIFO: n writes/reads in each,
the first index where the cycles differ and the differences there, and the difference at the last event.
"""
import asyncio, dataclasses, gzip, json, os, re, sys
from pathlib import Path

os.environ["LSIM_FRP"] = "0"
sys.path.insert(0, str(Path(__file__).resolve().parent))


def vcd_fifo_events(path):
    """{fifo base name: (writes, reads)} from every scope ending in <name>_U that has if_write/if_read."""
    f = gzip.open(path, "rt") if str(path).endswith(".gz") else open(path)
    stack, clk, rst, sig = [], None, None, {}
    want = ("if_write", "if_full_n", "if_read", "if_empty_n")
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
            if len(stack) == 2 and name == "ap_clk":
                clk = code
            if len(stack) == 2 and name == "ap_rst_n":
                rst = code
            if name in want and stack and stack[-1].endswith("_U") and "_m_axi_U" not in ".".join(stack):
                sig.setdefault(code, []).append((stack[-1][:-2], name))
        elif t[0] == "$enddefinitions":
            break
    val = {}
    ev = {}
    cur, rst_v, cyc = "0", "0", -1
    for line in f:
        c = line[0]
        if c in "01xzXZ":
            v, code = c, line[1:].strip()
        elif c == "b":
            v, code = line[1:].split()
            v = v[-1]
        else:
            continue
        if code == clk:
            if cur == "0" and v == "1" and rst_v == "1":
                # rising edge: the values settled during the previous cycle act now
                cyc += 1
                for fifo, s in val.items():
                    if s.get("if_write") == "1" and s.get("if_full_n") == "1":
                        ev.setdefault(fifo, ([], []))[0].append(cyc)
                    if s.get("if_read") == "1" and s.get("if_empty_n") == "1":
                        ev.setdefault(fifo, ([], []))[1].append(cyc)
            cur = v
        if code == rst:
            rst_v = v
        for fifo, name in sig.get(code, ()):
            val.setdefault(fifo, {})[name] = v
    return ev


def main():
    from frp_rtl import analyse
    from lightningsim.model import Solution
    from lightningsim.runner import Runner
    sol, out, nptr = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
    rtl = analyse(sol)
    trace = asyncio.run(Runner(Solution(sol)).run())
    p = trace.params
    frp, dep, names = {}, {}, {}
    for f in trace.fifos:
        n = f.name.replace(".", "_")
        hits = [x for x in rtl["fifos"] if x == n] or [x for x in rtl["fifos"] if n.endswith(x) or x.endswith(n)]
        if not hits:
            continue
        m = max(hits, key=len)
        names[f.id] = m
        dep[f.id] = rtl["fifos"][m]["depth"]
        if m in rtl["frp_reads"]:
            frp[f.id] = rtl["frp_reads"][m]
    if os.environ.get("FIFOCMP_BIG") == "1":   # unbounded FIFOs, for designs that deadlock in LSIM
        # depth 1000, and width 1 where the real FIFO is a shift register so its read-after-write delay is kept
        srl = {f.id for f in trace.fifos if dep.get(f.id, p.fifo_depths.get(f.id)) is None
               or dep.get(f.id, p.fifo_depths.get(f.id)) <= 2
               or (p.fifo_widths.get(f.id) or 0) * dep.get(f.id, p.fifo_depths.get(f.id)) <= 1024}
        dep = {f.id: 1000 for f in trace.fifos}
        q = dataclasses.replace(p, fifo_raw_extra=frp, fifo_depths={**p.fifo_depths, **dep},
                                fifo_widths={**p.fifo_widths, **{i: 1 for i in srl}})
    else:
        q = dataclasses.replace(p, fifo_raw_extra=frp, fifo_depths={**p.fifo_depths, **dep})
    sim = trace.compiled.execute(q)
    ls = {}
    for fifo, io in sim.fifo_io.items():
        if fifo.id in names:
            ls[names[fifo.id]] = (list(io.writes), list(io.reads))
    cs = vcd_fifo_events(out.parent / "all.vcd.gz")
    z_l = min((w[0] for w, _ in ls.values() if w), default=0)
    z_c = min((w[0] for w, _ in cs.values() if w), default=0)
    res = {}
    dump = re.compile(os.environ["FIFOCMP_DUMP"]) if os.environ.get("FIFOCMP_DUMP") else None
    for name in sorted(set(ls) | set(cs)):
        lw, lr = ls.get(name, ([], []))
        cw, cr = cs.get(name, ([], []))
        row = {"depth": rtl["fifos"].get(name, {}).get("depth"), "frp_r": rtl["frp_reads"].get(name)}
        for kind, a, b in (("w", lw, cw), ("r", lr, cr)):
            a = [x - z_l for x in a]
            b = [x - z_c for x in b]
            row[f"n{kind}"] = [len(a), len(b)]
            first = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)
            row[f"{kind}_first_diff"] = None if first is None else [first, a[first], b[first], a[first] - b[first]]
            row[f"{kind}_last"] = [a[-1], b[-1], a[-1] - b[-1]] if a and b else None
        res[name] = row
        if dump and dump.search(name):   # raw cycles (shifted), first and last 40 of each list
            row["raw"] = {k: v[:40] + v[-40:] for k, v in (("lsim_w", [x - z_l for x in lw]), ("lsim_r", [x - z_l for x in lr]),
                                                         ("cos_w", [x - z_c for x in cw]), ("cos_r", [x - z_c for x in cr]))}
    out.write_text(json.dumps({"end": sim.top_module.end, "fifos": res}, indent=1) + "\n")
    order = sorted((r["w_first_diff"][1], n) for n, r in res.items() if r["w_first_diff"])
    print(out.name, "end", sim.top_module.end, "fifos", len(res), "first write divergences:",
          [(n, res[n]["w_first_diff"]) for _, n in order[:6]])


if __name__ == "__main__":
    main()
