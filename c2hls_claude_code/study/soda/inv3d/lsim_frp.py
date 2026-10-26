#!/usr/bin/env python3
"""LightningSim with a model of Vitis free-running pipelines (FRP) on their input FIFOs.

  lsim_frp.py SOLUTION_DIR OUT.json

An FRP pipeline (frp_pipeline_valid + frp_fifoout in the RTL) issues an iteration only when every input FIFO's
num_data_valid covers the reads it already requested (ap_frp_data_req_*) plus this one, and reads at stage 1. So an
item must be in the FIFO one cycle earlier than for a stall pipeline: read-after-write delay 2 instead of 1.
LightningSim's RAM FIFOs already use RAW 2 (WAR 1, same as shift registers), so the FRP input FIFOs are given a width
that makes LightningSim treat them as RAM. Width only feeds the FIFO type and the BRAM estimate, not the timing
otherwise. The trace is resolved once; each variant is a re-execution.
Variants: base (stock LightningSim), frp_in (FRP input FIFOs RAW 2).
"""
import asyncio, json, re, sys
from pathlib import Path
from lightningsim.model import Solution
from lightningsim.runner import Runner
from lightningsim.trace_file import SimulationParameters

sol, out = Path(sys.argv[1]), Path(sys.argv[2])
frp_in = set()
for v in (sol / "syn" / "verilog").glob("*.v"):
    frp_in |= set(re.findall(r"reg\s+\[\d+:0\]\s+ap_frp_data_req_(\w+?)(?<!_op\d)(?<!_op\d\d)(?<!_op\d\d\d);", v.read_text(errors="replace")))
trace = asyncio.run(Runner(Solution(sol)).run())
p = trace.params
by_name = {f.name: f.id for f in trace.fifos}
hit = {n: i for n, i in by_name.items() if any(n == x or n.endswith(x) for x in frp_in)}
res = {"frp_input_fifos_rtl": sorted(frp_in), "frp_input_fifos_matched": sorted(hit), "fifos": sorted(by_name),
       # depth <= 2 stays a shift register (RAW 1) whatever the width: the trick does not cover those FIFOs
       "frp_input_fifos_depth_le2": sorted(n for n, i in hit.items() if (p.fifo_depths.get(i) or 0) <= 2)}

def run(widths):
    q = SimulationParameters(fifo_depths=p.fifo_depths, fifo_widths=widths, axi_delays=p.axi_delays,
                             ap_ctrl_chain_top_port_count=p.ap_ctrl_chain_top_port_count)
    return trace.compiled.execute(q).top_module.end

res["base"] = run(p.fifo_widths)
w = dict(p.fifo_widths)
for i in hit.values():
    w[i] = max(w.get(i, 0), 1 << 16)   # width*depth > 1024 and depth > 2 -> FifoType::Ram -> RAW 2
res["frp_in"] = run(w)
out.write_text(json.dumps(res, indent=1) + "\n")
print({k: res[k] for k in ("base", "frp_in")}, "matched", len(hit), "of", len(frp_in))
