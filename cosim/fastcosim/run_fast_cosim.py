#!/usr/bin/env python3
"""TAPA/PASTA fast cosim (tapa-fast-cosim 0.0.20220816.dev1, vendored and patched) for a plain Vitis HLS autosa_mm.

  run_fast_cosim.py NAME KERNEL.cpp [--size N]    (--size only for self-tests; real runs are the 1024^3 bench)

1. csynth (c2hls hls_eval.run_hls_synthesis, as verify/*/run_eval.py) -> hls_proj/sol1/syn/verilog
2. gen_data (bench data + golden, static_assert on the size) -> A.bin B.bin C.bin C_golden.bin
3. tapa-fast-cosim testbench (AXI-Lite master, one AXI RAM per m_axi bundle) + Vivado xsim, in the Vitis container
4. C_out.bin vs C_golden.bin with the bench's rule (|golden - C| > 0.001 is an error)
Only NAME/ (summary, log excerpt, generated testbench) is kept; the work directory is deleted.
"""
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "vendor"))
sys.path.insert(0, "/scratch/asa582/workspaces/code_translation_c2hls")
BENCH = Path("/scratch/asa582/workspaces/code_translation_c2hls/artifacts/pc2/autosa_mm_ijk_benches/n1024/autosa_mm")
TMP = Path(os.environ.get("C2HLS_TMP_ROOT", "/scratch/asa582/tmp/c2hls"))
PART, CLOCK_NS, TOP = "xcu280-fsvh2892-2L-e", 3.33, "autosa_mm"
TB_CLOCK_PERIOD_NS = 4  # tapa-fast-cosim testbench: HALF_CLOCK_PERIOD = 2


def header(size: int) -> str:
    h = (BENCH / "kernel.h").read_text()
    if size != 1024:  # self-test only
        h = h.replace("#define I 1024\n#define J 1024\n#define K 1024", f"#define I {size}\n#define J {size}\n#define K {size}")
    assert f"#define I {size}" in h
    return h


def bundles(src: str) -> dict:
    """port -> bundle from '#pragma HLS INTERFACE m_axi port=X ... bundle=Y'."""
    out = {}
    for line in src.splitlines():
        if re.search(r"#pragma\s+HLS\s+INTERFACE\s+(mode=)?m_axi\b", line, re.I):
            port, bundle = re.search(r"\bport=(\w+)", line), re.search(r"\bbundle=(\w+)", line)
            if port and bundle:
                out[port.group(1)] = bundle.group(1)
    return out


def run(cmd: str, timeout: int, log: Path) -> tuple[int, float]:
    t0 = time.time()
    with open(log, "w") as fp:
        try:
            rc = subprocess.run(["bash", "-lc", cmd], stdout=fp, stderr=subprocess.STDOUT, timeout=timeout).returncode
        except subprocess.TimeoutExpired:
            rc = -9
    return rc, round(time.time() - t0, 1)


def compare(out_bin: Path, golden_bin: Path) -> dict:
    if not out_bin.exists():
        return {"dumped": False, "errors": None}
    a, b = out_bin.read_bytes(), golden_bin.read_bytes()
    if len(a) != len(b):
        return {"dumped": True, "errors": None, "why": f"size {len(a)} != {len(b)}"}
    n = len(b) // 4
    got, ref = struct.unpack(f"<{n}f", a), struct.unpack(f"<{n}f", b)
    bad = [i for i in range(n) if abs(ref[i] - got[i]) > 0.001]
    return {"dumped": True, "errors": len(bad), "first_bad": [(i, ref[i], got[i]) for i in bad[:5]]}


def main() -> int:
    name, kernel = sys.argv[1], Path(sys.argv[2]).resolve()
    size = int(sys.argv[sys.argv.index("--size") + 1]) if "--size" in sys.argv else 1024
    out = HERE / name
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    work = TMP / f"fastcosim_{name}_{stamp}"
    work.mkdir(parents=True)
    src, hdr = kernel.read_text(), header(size)
    shutil.copy(kernel, out / "kernel.cpp")
    s = {"name": name, "kernel": str(kernel), "size": size, "work_dir_deleted": str(work), "wall_s": {}}
    try:
        # 1. csynth -> RTL
        import hls_eval
        t0 = time.time()
        r = hls_eval.run_hls_synthesis(src, hdr, header_name="kernel.h", top_function=TOP, part=PART,
                                       clock_ns=CLOCK_NS, work_dir=str(work / "hls"), allow_compile_jobs=False)
        s["wall_s"]["csynth"] = round(time.time() - t0, 1)
        if not r.get("success"):
            s["error"] = f"csynth failed: {(r.get('error') or '')[:1500]}"
            return finish(s, out, work)
        rep = r.get("report") or {}
        s["csynth"] = {k: rep.get(k) for k in ("latency_cycles", "latency_cycles_worst", "dsp", "estimated_clock_period_ns")}
        rtl_src = work / "hls" / "hls_proj" / "sol1" / "syn" / "verilog"
        rtl = work / "rtl"
        shutil.copytree(rtl_src, rtl)

        # 2. data + golden (bench data, size asserted at compile time)
        data = work / "data"
        data.mkdir()
        (data / "kernel.h").write_text(hdr)
        rc, s["wall_s"]["gen_data"] = run(
            f"g++ -O2 -I{data} -DEXPECT_N={size} {HERE / 'gen_data.cpp'} -o {data / 'gen_data'} && {data / 'gen_data'} {data}",
            3600, work / "gen_data.log")
        s["tb_size_line"] = next((l for l in (work / "gen_data.log").read_text().splitlines() if l.startswith("TB_SIZE")), None)
        if rc != 0 or s["tb_size_line"] != f"TB_SIZE I={size} J={size} K={size}":
            s["error"] = "gen_data failed: " + (work / "gen_data.log").read_text()[-1500:]
            return finish(s, out, work)

        # 3. tapa-fast-cosim testbench + Vivado xsim
        from tapa_fast_cosim.main import parse_register_addr, parse_m_axi_interfaces, get_cosim_tb, set_default_nettype
        from tapa_fast_cosim.templates import get_axi_ram_module, get_srl_fifo_template
        from tapa_fast_cosim.vivado import get_vivado_tcl
        port_to_bundle = bundles(src)
        assert set(port_to_bundle) == {"A", "B", "C"} and len(set(port_to_bundle.values())) == 3, port_to_bundle
        set_default_nettype(str(rtl))
        axi_list = parse_m_axi_interfaces(str(rtl / f"{TOP}.v"))
        assert {a.name for a in axi_list} == set(port_to_bundle.values()), ([a.name for a in axi_list], port_to_bundle)
        bundle_to_port = {b: p for p, b in port_to_bundle.items()}
        tbdir = work / "tb"
        (tbdir / "run").mkdir(parents=True)
        (tbdir / "tb.v").write_text(get_cosim_tb(TOP, str(rtl / f"{TOP}_control_s_axi.v"), axi_list, {}))
        (tbdir / "fifo_srl_tb.v").write_text(get_srl_fifo_template())
        s["axi"] = {}
        for axi in axi_list:
            port = bundle_to_port[axi.name]
            nbytes = (data / f"{port}.bin").stat().st_size
            beats = nbytes // (axi.data_width // 8)
            assert beats * (axi.data_width // 8) == nbytes
            s["axi"][axi.name] = {"port": port, "data_width": axi.data_width, "beats": beats}
            (tbdir / f"axi_ram_{axi.name}.v").write_text(get_axi_ram_module(axi, str(data / f"{port}.bin"), beats))
        tcl = get_vivado_tcl({"verilog_path": str(rtl), "part_num": PART}, str(tbdir), False)
        (tbdir / "run" / "run_cosim.tcl").write_text("\n".join(tcl) + "\n")
        for f in ("tb.v", "run/run_cosim.tcl"):
            (out / Path(f).name).write_text((tbdir / f).read_text())
        sif = os.environ["XILINX_SIF"]
        inner = (f'export LIBRARY_PATH="/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu${{LIBRARY_PATH:+:$LIBRARY_PATH}}"; '
                 f"ulimit -s unlimited 2>/dev/null || true; cd {tbdir / 'run'} && HOME=$PWD vivado -mode batch -source run_cosim.tcl")
        rc, s["wall_s"]["vivado"] = run(f"apptainer exec {sif} bash -c '{inner}'",
                                        int(os.environ.get("FASTCOSIM_TIMEOUT", "580000")), work / "vivado.log")
        s["vivado_rc"] = rc
        log = (work / "vivado.log").read_text(errors="replace")
        t_start = re.search(r"FASTCOSIM_START (\d+) cycle=(\d+)", log)
        t_done = re.search(r"FASTCOSIM_DONE (\d+) cycle=(\d+)", log)
        # Cycles come from the testbench's own clock-edge counter (PATCH 5), independent of $time units and
        # print formats. The $time difference (1 ns unit, one clock = TB_CLOCK_PERIOD_NS) is kept as a cross-check.
        if t_start and t_done:
            s["rtl_cycles"] = int(t_done.group(2)) - int(t_start.group(2))
            s["rtl_cycles_from_time"] = (int(t_done.group(1)) - int(t_start.group(1))) // TB_CLOCK_PERIOD_NS
            s["rtl_cycles_consistent"] = abs(s["rtl_cycles"] - s["rtl_cycles_from_time"]) <= 2
        s["timeout"] = "FASTCOSIM_TIMEOUT" in log
        keep = [l for l in log.splitlines() if re.search(r"FASTCOSIM|ERROR|CRITICAL|Time \(s\)|launch_simulation|xelab|SIGSEGV|\$finish", l)]
        (out / "vivado_log_excerpt.txt").write_text("\n".join(keep[-300:] + ["", "----- tail -----"] + log.splitlines()[-150:]) + "\n")

        # 4. compare with the bench's rule
        s["compare"] = compare(data / "C_out.bin", data / "C_golden.bin")
        s["correct"] = bool(t_done) and s["compare"].get("errors") == 0 and not s["timeout"]
    except Exception as exc:  # noqa: BLE001
        s["error"] = f"{type(exc).__name__}: {exc}"
    return finish(s, out, work)


def finish(s: dict, out: Path, work: Path) -> int:
    s.setdefault("correct", False)
    shutil.rmtree(work, ignore_errors=True)
    (out / "fastcosim_summary.json").write_text(json.dumps(s, indent=2, default=str) + "\n")
    print(json.dumps(s, indent=2, default=str))
    return 0 if s["correct"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
