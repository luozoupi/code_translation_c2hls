#!/usr/bin/env python3
"""Per-interface m_axi adapter configuration of a Vitis HLS 2023.2 solution, for the v2023 LightningSim AXI model.

  axi_cfg.py SOLUTION_DIR OUT.json

Reads each `<top>_<bundle>_m_axi #(...)` instance in syn/verilog/<top>.v (MAX_*_BURST_LENGTH,
NUM_*_OUTSTANDING, bus and user data widths) and the argument -> bundle map from syn/report/csynth.rpt.
Bursts and the write buffer (WBUFF_DEPTH = MAX_WRITE_BURST_LENGTH * BUS_DW / USER_DW) are in user beats.
The JSON is keyed by bundle, by m_axi_<bundle> and by every argument mapped to the bundle, since the trace
names interfaces by one of these.
"""
import json, re, sys
from pathlib import Path


def params_of(text, inst_start):
    body = text[inst_start: text.index(")\n", text.index("NUM_WRITE_OUTSTANDING", inst_start)) + 1]
    return dict(re.findall(r"\.(\w+)\(\s*([^()]*?)\s*\)", body))


def resolve(val, text):
    if re.fullmatch(r"\d+", val):
        return int(val)
    m = re.search(rf"parameter\s+{re.escape(val)}\s*=\s*(\d+)", text)
    return int(m.group(1)) if m else None


def main():
    sol, out = Path(sys.argv[1]), Path(sys.argv[2])
    tops = [p for p in (sol / "syn/verilog").glob("*.v") if re.search(r"_m_axi #\(", p.read_text(errors="replace"))
            and not p.name.endswith("_m_axi.v")]
    cfg = {}
    for top_v in tops:
        text = top_v.read_text(errors="replace")
        top = top_v.stem
        for m in re.finditer(rf"\b{re.escape(top)}_(\w+?)_m_axi #\(", text):
            bundle = m.group(1)
            p = params_of(text, m.start())
            bus = resolve(p.get("C_M_AXI_DATA_WIDTH", "32"), text) or 32
            user = resolve(p.get("USER_DW", str(bus)), text) or bus
            ratio = max(bus // user, 1)
            entry = {
                "read_burst_beats": int(p["MAX_READ_BURST_LENGTH"]) * ratio,
                "read_outstanding": int(p["NUM_READ_OUTSTANDING"]),
                "write_burst_beats": int(p["MAX_WRITE_BURST_LENGTH"]) * ratio,
                "write_outstanding": int(p["NUM_WRITE_OUTSTANDING"]),
                "write_buffer_beats": int(p["MAX_WRITE_BURST_LENGTH"]) * ratio,
                "bus_dw": bus, "user_dw": user, "conservative": int(p.get("CONSERVATIVE", "0")),
                "bundle": bundle, "top": top,
            }
            cfg[bundle] = entry
            cfg[f"m_axi_{bundle}"] = entry
    rpt = sol / "syn/report/csynth.rpt"
    if rpt.exists():
        for arg, iface in re.findall(r"^\|\s*(\w+)\s*\|\s*m_axi_(\w+)\s*\|\s*interface\s*\|", rpt.read_text(), re.M):
            if iface in cfg:
                cfg.setdefault(arg, cfg[iface])
    out.write_text(json.dumps(cfg, indent=1) + "\n")
    print(f"{len({e['bundle'] for e in cfg.values()})} m_axi bundles -> {out}")


if __name__ == "__main__":
    main()
