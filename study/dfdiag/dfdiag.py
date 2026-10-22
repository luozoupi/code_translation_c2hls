#!/usr/bin/env python3
"""dfdiag: does an HLS design really overlap load / compute / store?

Evidence, never the DATAFLOW label alone:
  * source     : DATAFLOW pragmas and where they sit (in a loop?), explicit [2] buffers / ping-pong indexing,
                 hls::stream_of_blocks, hls::stream channels.
  * csynth log : regions Vitis actually turned into dataflow ("Applying dataflow ... extracted N process"),
                 channels it built (PIPO = ping-pong memories, FIFO), FIFO depth warnings.
  * csynth rpt : per-module latency and interval, per-instance (process) latency, loop tables (trip, II).
For every dataflow region R with processes P_i:
  S = sum(lat P_i), M = max(lat P_i), L = lat R, II = interval R.
  intra-invocation overlap  = (S - L) / (S - M)          1.0 = all processes run together, 0.0 = one after another
  across iterations (R inside a loop of N trips, loop latency T):
      pipelined if T ~= N*II + (L - II)   (next iteration's region starts while the previous one drains)
      serial    if T ~= N*L
Classes (design level):
  NO_DATAFLOW, INEFFECTIVE (pragma present, no real overlap), TASK_CONCURRENCY (processes overlap, single
  invocation over the whole problem, e.g. stream pipelines), PINGPONG (region in a tile loop, iterations overlap
  through PIPO/double buffers), and an overlap grade: PARTIAL or FULL (latency within 10% of the bottleneck stage's
  total work, plus fill). Work outside every dataflow region (e.g. a serial preload) is reported as serial overhead.

Usage: dfdiag.py <work_dir> [--src kernel.cpp] [--json out.json]
  work_dir = the Vitis synthesis dir holding hls_proj/sol1/syn/report and logs/hls_run_tcl.log
"""
from __future__ import annotations

import argparse, json, re, sys
from pathlib import Path

NUM = r"(\d+|-)"


def _int(x):
    try:
        return int(x)
    except (TypeError, ValueError):
        return None


def parse_rpt(path: Path) -> dict:
    """Module csynth .rpt: overall latency/interval/pipeline type, instance (process) table, loop table."""
    text = path.read_text(errors="replace")
    mod = {"name": path.name.replace("_csynth.rpt", ""), "instances": [], "loops": []}
    m = re.search(r"\|\s*Latency \(cycles\)\s*\|.*?\n.*?\n.*?\n\s*\|\s*(\d+)\|\s*(\d+)\|[^|]*\|[^|]*\|\s*(\d+)\|\s*(\d+)\|\s*([\w ]+?)\s*\|", text, re.S)
    if m:
        mod.update(lat_min=int(m[1]), lat_max=int(m[2]), ii_min=int(m[3]), ii_max=int(m[4]), ptype=m[5].strip())
    det = text.split("+ Detail:", 1)[-1]
    inst = re.search(r"\* Instance:\s*\n(.*?)\n\s*\n", det, re.S)
    if inst:
        for line in inst.group(1).splitlines():
            c = [x.strip() for x in line.strip().strip("|").split("|")]
            if len(c) >= 9 and c[2].isdigit():
                mod["instances"].append({"inst": c[0], "module": c[1], "lat_min": _int(c[2]), "lat_max": _int(c[3]),
                                         "ii_min": _int(c[6]), "ii_max": _int(c[7]), "ptype": c[8]})
    loop = re.search(r"\* Loop:\s*\n(.*?)\n\s*\n", det, re.S)
    if loop:
        for line in loop.group(1).splitlines():
            c = [x.strip() for x in line.strip().strip("|").split("|")]
            if len(c) >= 8 and c[0].startswith("-") or (len(c) >= 8 and c[0].startswith("o")):
                name = c[0].lstrip("-o ").strip()
                mod["loops"].append({"loop": name, "depth": len(c[0]) - len(c[0].lstrip("-o ")),
                                     "lat_min": _int(c[1]), "lat_max": _int(c[2]), "iter_lat": _int(c[3]),
                                     "ii": _int(c[4]), "trip": max((int(x) for x in re.findall(r"\d+", c[6])), default=None),
                                     "pipelined": c[7]})
    return mod


def parse_log(path: Path) -> dict:
    t = path.read_text(errors="replace") if path.exists() else ""
    return {
        "dataflow_funcs": [{"func": f, "processes": int(n)} for f, n in
                           re.findall(r"Applying dataflow to function '([^']+)'.*?extracted (\d+) process", t)],
        "pipo": sorted(set(re.findall(r"Implementing PIPO (\S+)", t))),
        "fifo": sorted(set(re.findall(r"Implementing FIFO '([^']+)'", t))),
        "fifo_depth_raised": re.findall(r"Increasing the depth of FIFO (\S+) \(from (\S+) to (\S+)\) to (\d+)", t),
        "dataflow_warnings": sorted(set(re.findall(r"WARNING: \[HLS 200-(?:656|1449|786|979|976|1263)\][^\n]{0,160}", t)))[:12],
    }


def parse_src(path: Path | None) -> dict:
    if not path or not path.exists():
        return {}
    s = path.read_text(errors="replace")
    code = re.sub(r"//.*", "", re.sub(r"/\*.*?\*/", "", s, flags=re.S))
    lines = code.split("\n")
    df_lines = [i for i, l in enumerate(lines) if re.search(r"#pragma\s+HLS\s+DATAFLOW", l, re.I)]
    in_loop = []
    for i in df_lines:  # nearest enclosing 'for' before the pragma, at most 3 lines up, with an opening brace
        prev = "\n".join(lines[max(0, i - 3):i])
        in_loop.append(bool(re.search(r"\bfor\s*\(.*\)\s*\{\s*$", prev, re.M)))
    return {
        "dataflow_pragmas": len(df_lines),
        "dataflow_in_loop": any(in_loop),
        "explicit_pingpong": sorted(set(re.findall(r"\b(\w+)\s*\[\s*2\s*\]\s*\[", code))),
        "pingpong_index": bool(re.search(r"(\&\s*1\b|%\s*2\b|\^\s*1\b|\bping\b|\bpong\b)", code)),
        "stream_of_blocks": len(re.findall(r"stream_of_blocks\s*<", code)),
        "streams": len(re.findall(r"hls::stream<[^;&()]*>\s+\w+\s*[;(\[]", code)),
    }


def analyse(work: Path, src: Path | None) -> dict:
    rep = work / "hls_proj/sol1/syn/report"
    mods = {}
    for p in rep.glob("*_csynth.rpt"):
        m = parse_rpt(p)
        mods[m["name"]] = m
    log = parse_log(work / "logs/hls_run_tcl.log")
    srci = parse_src(src or (work / "kernel.cpp"))
    top_name = next((n for n in mods if n == "autosa_mm"), None) or max(mods, key=lambda n: mods[n].get("lat_max") or 0)
    top = mods[top_name]
    total = top.get("lat_max") or 0
    regions = []
    for name, m in mods.items():
        if (m.get("ptype") or "").startswith("dataflow") and m["instances"]:
            procs = [(i["module"], i["lat_max"] or 0) for i in m["instances"]]
            S = sum(l for _, l in procs)
            M = max(l for _, l in procs)
            L = m.get("lat_max") or 0
            ii = m.get("ii_max") or L
            overlap = (S - L) / (S - M) if S > M else 1.0
            # where is this region invoked? find a parent instance + enclosing loop
            calls, loop = 1, None
            for pn, pm in mods.items():
                if any(i["module"] == name for i in pm["instances"]):
                    for lp in pm["loops"]:
                        if lp["trip"] and lp["lat_max"] and lp["iter_lat"] and abs(lp["iter_lat"] - L) <= max(8, 0.02 * L):
                            loop = {"parent": pn, **lp}
                            calls = lp["trip"]
                        elif lp["trip"] and lp["iter_lat"] and lp["iter_lat"] > 1.02 * L and not loop:
                            calls = lp["trip"]   # region is one stage of a longer serial loop body
                    if not loop:
                        for lp in pm["loops"]:   # dataflow pipelined loops report no iteration latency
                            if (lp["trip"] and lp["lat_max"] and lp["iter_lat"] is None
                                    and lp["lat_max"] <= 1.05 * lp["trip"] * L and lp["lat_max"] >= L):
                                loop = {"parent": pn, **lp}
                                calls = lp["trip"]
                                break
            across = None
            if loop and calls > 1:
                T = loop["lat_max"]
                pipe_T, serial_T = calls * ii + (L - ii), calls * L
                across = {"calls": calls, "loop_latency": T, "if_pipelined": pipe_T, "if_serial": serial_T,
                          "pipelined": abs(T - pipe_T) <= 0.05 * T and pipe_T < 0.95 * serial_T}
            work_stage = max(procs, key=lambda x: x[1])
            regions.append({"region": name, "processes": [{"module": a, "latency": b} for a, b in procs],
                            "latency": L, "interval": ii, "sum_processes": S, "max_process": M,
                            "intra_overlap": round(overlap, 3), "bottleneck_process": work_stage[0],
                            "across_iterations": across,
                            "region_total": (across["loop_latency"] if across else L * calls)})
    # serial work outside dataflow regions
    covered = sum(r["region_total"] for r in regions)
    serial_outside = max(0, total - covered) if regions else total
    # classification
    findings, cls, grade = [], "NO_DATAFLOW", None
    if srci.get("dataflow_pragmas") and not regions:
        cls = "INEFFECTIVE"
        findings.append("DATAFLOW pragma in the source, but csynth built no dataflow region (pragma ignored or region "
                        "not canonical).")
    for r in regions:
        names = " ".join(p["module"] for p in r["processes"]).lower()
        has_compute = any(k in names for k in ("comp", "mac", "pe", "gemm", "mm", "dot", "red"))
        if not has_compute:
            findings.append(f"region {r['region']}: no compute process ({names}); it only overlaps loads/stores "
                            f"with each other, compute stays serial.")
        if r["intra_overlap"] < 0.2 and not (r["across_iterations"] and r["across_iterations"]["pipelined"]):
            findings.append(f"region {r['region']}: latency {r['latency']:,} ~ sum of processes {r['sum_processes']:,} "
                            f"(overlap {r['intra_overlap']:.2f}) and no overlap across calls: serialized.")
    if regions:
        effective = [r for r in regions if (r["intra_overlap"] >= 0.2 or (r["across_iterations"] or {}).get("pipelined"))
                     and any(k in " ".join(p["module"] for p in r["processes"]).lower()
                             for k in ("comp", "mac", "pe", "gemm", "mm", "dot", "red"))]
        if not effective:
            cls = "INEFFECTIVE"
        elif any((r["across_iterations"] or {}).get("pipelined") for r in effective):
            cls = "PINGPONG"
        else:
            cls = "TASK_CONCURRENCY"
        if effective:
            r = max(effective, key=lambda x: x["region_total"])
            calls = (r["across_iterations"] or {}).get("calls", 1)
            bottleneck_total = r["max_process"] * calls if calls > 1 else r["max_process"]
            ideal = bottleneck_total
            grade = "FULL" if total <= 1.10 * ideal else "PARTIAL"
            bn = r["bottleneck_process"]
            if not any(k in bn.lower() for k in ("comp", "mac", "pe", "gemm", "dot", "red")):
                findings.append(f"bottleneck stage is {bn} ({r['max_process']:,} per call), not compute: the overlap "
                                f"is real but a memory stage sets the pace.")
            if serial_outside > 0.05 * total:
                findings.append(f"{serial_outside:,} cycles ({100 * serial_outside / total:.1f}%) are outside every "
                                f"dataflow region and run serially (e.g. a preload).")
            if r["across_iterations"] and not r["across_iterations"]["pipelined"] and calls > 1:
                findings.append(f"region {r['region']} is called {calls}x but calls do not overlap "
                                f"(loop {r['across_iterations']['loop_latency']:,} ~ {calls} x latency).")
    # narrow memory loops (II > 1 on loops that look like transfers)
    for name, m in mods.items():
        for lp in m["loops"]:
            if lp["ii"] and lp["ii"] > 1 and re.search(r"(load|read|rd|store|write|wr|drain|fill)", lp["loop"], re.I):
                findings.append(f"{name}/{lp['loop']}: transfer loop at II={lp['ii']} (trip {lp['trip']}): likely one "
                                f"word per beat instead of a 512-bit beat.")
            if lp["ii"] and lp["ii"] > 1 and re.search(r"(comp|mac|red|dot|pe)", lp["loop"], re.I):
                findings.append(f"{name}/{lp['loop']}: compute loop at II={lp['ii']}: accumulator recurrence or port "
                                f"conflict leaves {100 * (1 - 1 / lp['ii']):.0f}% of the MAC hardware idle.")
    # top-level serial phases (instances x calls, loops): what runs one after another at the top
    serial_phases = []
    for i in top["instances"]:
        serial_phases.append((i["module"], i["lat_max"] or 0))
    for lp in top["loops"]:
        if lp["depth"] <= 1:
            serial_phases.append((lp["loop"], lp["lat_max"] or 0))
    serial_phases.sort(key=lambda x: -x[1])
    pp_evidence = {"pipo_channels": log["pipo"], "explicit_buffers": srci.get("explicit_pingpong", []),
                   "stream_of_blocks": srci.get("stream_of_blocks", 0)}
    if cls == "PINGPONG" and not (log["pipo"] or srci.get("explicit_pingpong") or srci.get("stream_of_blocks")):
        findings.append("iterations overlap but no PIPO/explicit double buffer was found; overlap must come from "
                        "streams (FIFO) instead.")
    return {"work_dir": str(work), "top": top_name, "latency": total, "interval": top.get("ii_max"),
            "class": cls, "overlap_grade": grade, "serial_phases": serial_phases[:8], "serial_outside_regions": serial_outside,
            "regions": regions, "pingpong_evidence": pp_evidence, "log": {k: log[k] for k in ("dataflow_funcs", "fifo_depth_raised", "dataflow_warnings")},
            "source": srci, "findings": findings}


def explain(d: dict) -> str:
    out = [f"{d['top']}: latency {d['latency']:,}  class={d['class']}" + (f"  overlap={d['overlap_grade']}" if d['overlap_grade'] else "")]
    for r in d["regions"]:
        a = r["across_iterations"]
        out.append(f"  region {r['region']}: L={r['latency']:,} II={r['interval']:,} sum={r['sum_processes']:,} "
                   f"max={r['max_process']:,} intra_overlap={r['intra_overlap']} bottleneck={r['bottleneck_process']}"
                   + (f" | x{a['calls']} loop={a['loop_latency']:,} (pipelined {a['if_pipelined']:,} / serial {a['if_serial']:,}) "
                      f"-> {'iterations overlap' if a['pipelined'] else 'iterations serial'}" if a else ""))
        out.append("     " + ", ".join(f"{p['module']}={p['latency']:,}" for p in r["processes"]))
    if d["regions"]:
        out.append(f"  outside regions (serial): {d['serial_outside_regions']:,}")
    ev = d["pingpong_evidence"]
    if d.get("serial_phases"):
        out.append("  serial phases (top level): " + ", ".join(f"{n}={c:,} ({100 * c / max(1, d['latency']):.0f}%)" for n, c in d["serial_phases"]))
    out.append(f"  ping-pong evidence: PIPO={len(ev['pipo_channels'])} explicit={ev['explicit_buffers']} stream_of_blocks={ev['stream_of_blocks']}")
    out += [f"  - {f}" for f in d["findings"]]
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("work_dir")
    ap.add_argument("--src")
    ap.add_argument("--json")
    a = ap.parse_args()
    d = analyse(Path(a.work_dir), Path(a.src) if a.src else None)
    if a.json:
        Path(a.json).write_text(json.dumps(d, indent=1))
    print(explain(d))


if __name__ == "__main__":
    main()
