#!/usr/bin/env python3
"""RTL correctness check of a vanilla Vitis HLS autosa_mm kernel with Verilator (TAPA-style fast cosim).

  run_vcosim.py NAME KERNEL.cpp [--size N] [--threads T] [--heartbeat C] [--early-stop-after M]
  (--size only for self-tests; real runs are the 1024^3 bench)

1. csynth (c2hls hls_eval, flow_target vitis) -> syn/verilog
2. bench data + golden (../fastcosim/gen_data.cpp: the bench testbench's rand() data and golden, static_assert size)
3. RTL prep (rtl_prep.py, port of TAPA): AMD floating-point IP -> behavioral SV + DPI (fp_dpi.c); NBA fix
4. testbench (tb_gen.py, AXI model ported from TAPA): AXI-Lite start, one memory per m_axi bundle
5. Verilator build + run (cwd = rtl dir for $readmemh), then C_out.bin vs golden with the bench rule |diff| > 0.001
Everything bulky lives in $C2HLS_TMP_ROOT (node-local in the sbatch script) and is deleted; <here>/runs/NAME keeps
summary.json, logs excerpts, tb.cpp, the generated IP stand-ins and the kernel.
"""
import json, os, re, shutil, struct, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
C2HLS = Path("/scratch/asa582/workspaces/code_translation_c2hls")
BENCH = C2HLS / "artifacts/pc2/autosa_mm_ijk_benches/n1024/autosa_mm"
TMP = Path(os.environ.get("C2HLS_TMP_ROOT", "/scratch/asa582/tmp/c2hls"))
VERILATOR = Path(os.environ.get("VERILATOR_BIN", "/scratch/asa582/tools/verilator-v5.044/bin/verilator"))
PART, CLOCK_NS, TOP = "xcu280-fsvh2892-2L-e", 3.33, "autosa_mm"
sys.path.insert(0, str(C2HLS))
sys.path.insert(0, str(HERE))
import rtl_prep  # noqa: E402
import tb_gen  # noqa: E402

WARN = ["-Wno-fatal", "-Wno-PINMISSING", "-Wno-WIDTH", "-Wno-UNUSEDSIGNAL", "-Wno-UNDRIVEN", "-Wno-UNOPTFLAT",
        "-Wno-STMTDLY", "-Wno-CASEINCOMPLETE", "-Wno-SYMRSVDWORD", "-Wno-COMBDLY", "-Wno-TIMESCALEMOD",
        "-Wno-MULTIDRIVEN", "-Wno-lint", "-Wno-style"]


def opt(flag, default, cast=str):
    return cast(sys.argv[sys.argv.index(flag) + 1]) if flag in sys.argv else default


def header(size: int) -> str:
    h = (BENCH / "kernel.h").read_text()
    if size != 1024:
        h = h.replace("#define I 1024\n#define J 1024\n#define K 1024", f"#define I {size}\n#define J {size}\n#define K {size}")
    assert f"#define I {size}" in h
    return h


def arg_bundles(src: str) -> dict:
    out = {}
    for line in src.splitlines():
        if re.search(r"#\s*pragma\s+HLS\s+INTERFACE", line, re.I) and re.search(r"\bm_axi\b", line):
            port, bundle = re.search(r"\bport\s*=\s*(\w+)", line), re.search(r"\bbundle\s*=\s*(\w+)", line)
            if port:
                out[port.group(1)] = bundle.group(1) if bundle else "gmem"
    return out


def run(cmd, log: Path, cwd=None, timeout=None):
    t = time.time()
    with open(log, "w") as fp:
        rc = subprocess.run(cmd, stdout=fp, stderr=subprocess.STDOUT, cwd=cwd, timeout=timeout).returncode
    return rc, round(time.time() - t, 1)


def main() -> int:
    name, kernel = sys.argv[1], Path(sys.argv[2]).resolve()
    size = opt("--size", 1024, int)
    threads = opt("--threads", 1, int)
    heartbeat = opt("--heartbeat", 100000, int)
    early_stop = opt("--early-stop-after", 4096, int)
    safe = "--safe" in sys.argv  # conservative Verilator: no -O3, X values randomized (+verilator+seed), not fast-assigned
    out = HERE / "runs" / name
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    work = TMP / f"vcosim_{name}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    work.mkdir(parents=True)
    src = kernel.read_text()
    shutil.copy(kernel, out / "kernel.cpp")
    s = {"name": name, "kernel": str(kernel), "size": size, "threads": threads, "safe_mode": safe, "wall_s": {}, "correct": False,
         "work_dir_deleted": str(work)}
    try:
        # 1. csynth
        import hls_eval
        t = time.time()
        syn = hls_eval.run_hls_synthesis(src, header(size), header_name="kernel.h", top_function=TOP, part=PART,
                                         clock_ns=CLOCK_NS, work_dir=str(work / "hls"), allow_compile_jobs=False)
        s["wall_s"]["csynth"] = round(time.time() - t, 1)
        rep = syn.get("report") or {}
        s["csynth"] = {k: rep.get(k) for k in ("latency_cycles", "latency_cycles_worst", "dsp", "lut")}
        vdirs = list(work.rglob("syn/verilog"))
        if not syn.get("success") or not vdirs:
            s["error"] = "csynth failed: " + (syn.get("error") or "")[:1500]
            return finish(s, out, work)
        vdir = vdirs[0]

        # 2. data + golden
        data = work / "data"
        data.mkdir()
        (data / "kernel.h").write_text(header(size))
        rc, s["wall_s"]["gen_data"] = run(
            ["bash", "-c", f"g++ -O2 -I{data} -DEXPECT_N={size} {HERE.parent / 'fastcosim/gen_data.cpp'} "
                           f"-o {data / 'gen_data'} && {data / 'gen_data'} {data}"], work / "gen_data.log")
        if rc:
            s["error"] = "gen_data failed: " + (work / "gen_data.log").read_text()[-1500:]
            return finish(s, out, work)

        # 3. RTL prep
        rtl = work / "rtl"
        rtl.mkdir()
        for f in vdir.iterdir():
            if f.suffix in (".v", ".sv", ".vh", ".dat", ".tcl"):
                shutil.copy(f, rtl / f.name)
        ips = rtl_prep.generate_fp_ip_models(rtl)
        s["fp_ip"] = {k: (v if isinstance(v, str) else f"{v[0]} latency {v[1]}") for k, v in ips.items()}
        s["nba_files_fixed"] = rtl_prep.fix_combinational_nba(rtl)
        bad = [k for k, v in ips.items() if v == "unsupported"]
        if bad:
            s["error"] = f"IP without a Verilog model (not fp32/fp64 add/sub/mul): {bad}"
            return finish(s, out, work)
        (out / "ip_models").mkdir()
        for ip, v in ips.items():
            if not isinstance(v, str):
                shutil.copy(rtl / f"{ip}.sv", out / "ip_models" / f"{ip}.sv")

        # 4. ports, register map, testbench
        top_v = (rtl / f"{TOP}.v").read_text()
        params = dict(re.findall(r"parameter\s+(\w+)\s*=\s*(\d+)\s*;", top_v))
        bundles = {}
        for m in re.finditer(r"output\s+\[(\w+)\s*-\s*1\s*:\s*0\]\s+m_axi_(\w+)_WDATA", top_v):
            bundles[m.group(2)] = int(params[m.group(1)]) // 8
        ctrl = (rtl / f"{TOP}_control_s_axi.v").read_text()
        regs = {m.group(1).upper() + m.group(2): int(m.group(3), 16)
                for m in re.finditer(r"ADDR_(\w+?)_DATA_([01])\s*=\s*\d+'h([0-9a-fA-F]+)", ctrl)}
        ab = arg_bundles(src)
        elems = {"A": size * size, "B": size * size, "C": size * size}
        ports = []
        for i, arg in enumerate(("A", "B", "C")):
            b = ab.get(arg)
            assert b in bundles, f"arg {arg}: bundle {b} not among RTL m_axi bundles {sorted(bundles)}"
            assert list(ab.values()).count(b) == 1, f"bundle {b} shared by several args; not supported here"
            ports.append({"bundle": b, "arg": arg, "nbytes": bundles[b], "size": elems[arg] * 4,
                          "base": (i + 1) * 0x1000_0000, "reg_lo": regs[f"{arg}0"], "reg_hi": regs[f"{arg}1"]})
        s["ports"] = ports
        tb = work / "tb.cpp"
        tb.write_text(tb_gen.generate_tb(TOP, ports, c_idx=2))
        shutil.copy(tb, out / "tb.cpp")

        # 5. build
        dpi_o = work / "fp_dpi.o"
        subprocess.run(["gcc", "-O2", "-ffp-contract=off", "-fno-fast-math", "-c", str(HERE / "fp_dpi.c"), "-o", str(dpi_o)],
                       check=True)
        jobs = str(os.cpu_count() or 8)
        vfiles = sorted(str(f) for f in rtl.iterdir() if f.suffix in (".v", ".sv"))
        cmd = [str(VERILATOR), "--cc", "--exe", "--build", "-j", jobs, "--top-module", TOP, "--no-timing",
               *(["-O3", "--x-assign", "fast"] if not safe else ["--x-assign", "unique", "--x-initial", "unique"]),
               "--Mdir", str(work / "obj_dir"), "-y", str(rtl),
               "-LDFLAGS", str(dpi_o), "-MAKEFLAGS", "OPT_FAST=-O2", *WARN, str(tb), *vfiles]
        if threads > 1:
            cmd[1:1] = ["--threads", str(threads)]
        rc, s["wall_s"]["verilator_build"] = run(cmd, work / "verilator_build.log")
        if rc:
            s["error"] = "verilator build failed"
            return finish(s, out, work)
        exe = work / "obj_dir" / f"V{TOP}"

        # 6. run (cwd = rtl for $readmemh "./x.dat")
        lat = max(rep.get("latency_cycles") or 0, rep.get("latency_cycles_worst") or 0)
        max_cycles = 5 * lat + 1_000_000 if lat else 2_000_000_000
        s["max_cycles"] = max_cycles
        rc, s["wall_s"]["simulate"] = run([str(exe), str(data), str(data), str(max_cycles), str(heartbeat),
                                           str(early_stop), *(["+verilator+rand+reset+2", "+verilator+seed+7"] if safe else [])],
                                          out / "simulate.log", cwd=rtl)
        s["sim_rc"] = rc  # simulate.log (heartbeats, mismatches) is written straight to the results dir: visible live
        log = (out / "simulate.log").read_text(errors="replace")
        end = re.search(r"VCOSIM_END status=(\w+) cycles=(\d+) wall_s=([\d.]+) c_read=(\d) c_rewritten=(\d) "
                        r"c_first_writes=(\d+) c_early_bad=(\d+) first_bad_idx=(-?\d+)", log)
        if end:
            s.update(status=end.group(1), rtl_cycles=int(end.group(2)), sim_wall_s=float(end.group(3)),
                     c_read=bool(int(end.group(4))), c_rewritten=bool(int(end.group(5))),
                     c_first_writes=int(end.group(6)), c_early_bad=int(end.group(7)))
            s["cycles_per_s"] = round(s["rtl_cycles"] / max(s["sim_wall_s"], 1e-9), 1)

        # 7. check C with the bench rule
        cout = data / "C_out.bin"
        if s.get("status") == "done" and cout.exists():
            got = struct.unpack(f"<{size * size}f", cout.read_bytes())
            ref = struct.unpack(f"<{size * size}f", (data / "C_golden.bin").read_bytes())
            bad = [i for i in range(len(ref)) if abs(ref[i] - got[i]) > 0.001]
            s["errors"] = len(bad)
            s["first_bad"] = [(i, ref[i], got[i]) for i in bad[:5]]
            s["bench_line"] = f"Failed with {len(bad)} errors!" if bad else "Passed!"
            s["correct"] = not bad
        else:
            s["bench_line"] = f"not checked (status={s.get('status')}, rc={rc})"
    except Exception as exc:  # noqa: BLE001
        s["error"] = f"{type(exc).__name__}: {exc}"
    return finish(s, out, work)


def finish(s, out, work) -> int:
    for f in ("verilator_build.log", "simulate.log", "gen_data.log"):
        p = work / f
        if p.exists():
            lines = p.read_text(errors="replace").splitlines()
            keep = [l for l in lines if re.search(r"VCOSIM|%Error|%Warning-(?!WIDTH)|error:|Error", l)]
            (out / (f + ".excerpt")).write_text("\n".join(keep[-200:] + ["----- tail -----"] + lines[-80:]) + "\n")
    shutil.rmtree(work, ignore_errors=True)
    (out / "summary.json").write_text(json.dumps(s, indent=2) + "\n")
    print(json.dumps(s, indent=2))
    return 0 if s.get("correct") else 1


if __name__ == "__main__":
    raise SystemExit(main())
