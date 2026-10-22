#!/usr/bin/env python3
"""Dump LightningSim's per-event m_axi times for one solution (run inside the Vitis container, see run_lsim.sh).

  lsim_io.py SOLUTION_DIR OUT.json

Uses LSIM_AXI_MODEL / LSIM_AXI_IFCFG / LSIM_AXI_CONST like the CLI. Output: {"end": top end, "interfaces":
{name: {readreqs, reads, writereqs, writes, writeresps: [cycle, ...]}}} for comparison with the kernel-side
(I_*) handshakes that parse_axi_vcd.py extracts from a cosim VCD.
"""
import asyncio, json, sys
from pathlib import Path
from lightningsim.model import Solution
from lightningsim.runner import Runner

trace = asyncio.run(Runner(Solution(Path(sys.argv[1]))).run())
sim = trace.compiled.execute(trace.params)
names = {i.address: i.name for i in trace.axi_interfaces}
out = {"end": sim.top_module.end, "interfaces": {}}
for iface, io in sim.axi_io.items():
    out["interfaces"][names.get(iface.address, hex(iface.address))] = {
        "readreqs": [x.time for x in io.readreqs], "reads": [x.time for x in io.reads],
        "writereqs": [x.time for x in io.writereqs], "writes": [x.time for x in io.writes],
        "writeresps": list(io.writeresps)}
Path(sys.argv[2]).write_text(json.dumps(out) + "\n")
print("end", out["end"], {k: {e: len(v) for e, v in d.items()} for k, d in out["interfaces"].items()})
