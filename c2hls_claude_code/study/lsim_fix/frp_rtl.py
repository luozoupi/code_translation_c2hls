#!/usr/bin/env python3
"""RTL facts LightningSim needs for Vitis free-running pipelines (FRP), read from sol1/syn/verilog.

  frp_rtl.py SOLUTION_DIR      -> prints the JSON below
fifos: {instance base name: {"depth": D, "type": "S"|"A", "width": W}}   (every <top>_fifo_wW_dD_S|A instance)
frp_reads: {FIFO instance base name: r}   r = cycles from FRP issue to the read of that FIFO (the stage of
  `<port>_read = 1'b1`: frp_pipeline_valid valid_out[r] or ap_enable_reg_pp0_iter<r>)
An FRP pipeline issues an iteration only when every input FIFO's num_data_valid covers the reads it has requested and not
done yet plus this one, so the item read at stage r must be in the FIFO r cycles before the read (checked in the
f_k3_l_frp waveform: issue at the cycle num_data_valid becomes 1, read 3 cycles later; Vitis had grown that FIFO from
depth 2 to 5 to keep II=1).
"""
import json, re, sys
from pathlib import Path

FIFO_INST = re.compile(r"^\s*\w+_fifo_w(\d+)_d(\d+)_([SA])\s+(\w+)_U\s*\(", re.M)
INST = re.compile(r"^\s*(\w+)\s+(\w+)\s*\((.*?)\);", re.M | re.S)
CONN = re.compile(r"\.(\w+)\s*\(\s*([\w\[\]:']*)\s*\)")
READ = re.compile(r"^\s*if \(([^\n]*)\) begin\s*\n\s*(\w+)_read = 1'b1;", re.M)
STAGE = (re.compile(r"valid_out\[\d+'d(\d+)\]"), re.compile(r"ap_enable_reg_pp0_iter(\d+)\b"))


def analyse(sol: Path):
    vdir = sol / "syn" / "verilog"
    src = {p.stem: p.read_text(errors="replace") for p in vdir.glob("*.v")}
    fifos = {}
    for text in src.values():
        for w, d, t, name in FIFO_INST.findall(text):
            fifos[name] = {"depth": int(d), "type": t, "width": int(w)}
    # read stage per port of every FRP module
    frp_ports = {}
    for mod, text in src.items():
        if "frp_pipeline_valid " not in text and "frp_pipeline_valid_U" not in text:
            continue
        ports = {}
        for cond, port in READ.findall(text):
            r = None
            for rx in STAGE:
                m = rx.search(cond)
                if m:
                    r = int(m.group(1))
                    break
            if r is not None:
                ports[port] = r
        if ports:
            frp_ports[mod] = ports
    # parent connections: FRP instance port <p>_read -> net -> FIFO instance whose .if_read is that net
    frp_reads = {}
    for text in src.values():
        insts = INST.findall(text)
        by_read_net = {}
        for mtype, iname, body in insts:
            if iname.endswith("_U") and "_fifo_w" in mtype:
                for port, net in CONN.findall(body):
                    if port == "if_read":
                        by_read_net[net] = iname[:-2]
        for mtype, iname, body in insts:
            if mtype not in frp_ports:
                continue
            conns = dict(CONN.findall(body))
            for port, r in frp_ports[mtype].items():
                net = conns.get(f"{port}_read")
                if net in by_read_net:
                    frp_reads[by_read_net[net]] = r
    return {"fifos": fifos, "frp_reads": frp_reads, "frp_modules": {m: p for m, p in frp_ports.items()}}


if __name__ == "__main__":
    print(json.dumps(analyse(Path(sys.argv[1])), indent=1))
