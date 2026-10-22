#!/usr/bin/env python3
"""Build one autosa_mm design at size N for the LightningSim calibration, keeping the Vitis project as one tarball.

  run_design.py NAME KERNEL.cpp OUT_ROOT --size N [--cosim] [--vcd]

--cosim: c2hls hls_eval.run_cosim (csynth + Vitis 2023.2 cosim, same settings and container handling as
         ../run_cosim_one.py) -> the reference cycle count. Without --cosim: csynth only (enough for LightningSim).
--vcd:   cosim with -trace_level port, then vcd_rerun.sh dumps the AXI/ap_ctrl ports (small sizes only).
The testbench is ../run_cosim_one.py's (bench testbench.cpp with reporting changes only), sized by kernel.h.
Writes OUT_ROOT/NAME_nN/{proj.tar.gz, summary.json, *cosim.rpt, csynth.rpt, axi.vcd.gz}.
"""
import argparse, json, os, re, shlex, shutil, subprocess, sys, tarfile, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, "/scratch/asa582/workspaces/code_translation_c2hls")
import run_cosim_one as rco  # noqa: E402

PART, CLOCK = "xcu280-fsvh2892-2L-e", 3.33


def csynth_only(work, src, tb, hdr, top="autosa_mm"):
    (work / "kernel.cpp").write_text(src)
    (work / "testbench.cpp").write_text(tb)
    (work / "kernel.h").write_text(hdr)
    (work / "run.tcl").write_text(f"""open_project hls_proj
set_top {top}
add_files kernel.cpp
add_files kernel.h
add_files -tb testbench.cpp
add_files -tb kernel.h
open_solution "sol1" -flow_target vitis
set_part {{{PART}}}
create_clock -period {CLOCK} -name default
csynth_design
exit
""")
    r = subprocess.run(f"cd {shlex.quote(str(work))} && vitis-run --tcl --input_file run.tcl", shell=True,
                       capture_output=True, text=True)
    (work / "vitis.log").write_text(r.stdout + r.stderr)
    return {"vitis_rc": r.returncode}


def cosim(work, src, tb, hdr, trace):
    import hls_eval
    os.environ["C2HLS_COSIM_TRACE_LEVEL"] = trace
    hls_eval.DEFAULT_COSIM_TRACE_LEVEL = trace
    original_run = hls_eval._run_vitis_cmd

    def run_in_container(cmd, timeout):
        if "sim.sh" in cmd and os.environ.get("XILINX_SIF"):
            inner = ('export LIBRARY_PATH="/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu${LIBRARY_PATH:+:$LIBRARY_PATH}"; '
                     "ulimit -s unlimited 2>/dev/null || true; " + cmd)
            cmd = f"apptainer exec {shlex.quote(os.environ['XILINX_SIF'])} bash -lc {shlex.quote(inner)}"
        return original_run(cmd, timeout)

    hls_eval._run_vitis_cmd = run_in_container
    r = hls_eval.run_cosim(src, tb, hdr, header_name="kernel.h", top_function="autosa_mm", part=PART,
                           clock_ns=CLOCK, work_dir=str(work))
    log = str(r.get("log") or "")
    (work / "vitis.log").write_text(log[-2_000_000:])
    return {k: r.get(k) for k in ("success", "passed", "kernel_runtime_cycles", "error")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("kernel")
    ap.add_argument("out_root")
    ap.add_argument("--size", type=int, required=True)
    ap.add_argument("--cosim", action="store_true")
    ap.add_argument("--vcd", action="store_true")
    ap.add_argument("--top", default="autosa_mm")
    ap.add_argument("--tb", help="testbench file (default: ../run_cosim_one.py's autosa_mm bench)")
    ap.add_argument("--hdr", help="kernel.h file (default: the bench's, sized by --size)")
    a = ap.parse_args()
    tag = f"{a.name}_n{a.size}"
    out = Path(a.out_root) / tag
    out.mkdir(parents=True, exist_ok=True)
    work = Path(os.environ.get("C2HLS_TMP_ROOT", "/scratch/asa582/tmp/c2hls")) / f"lsim_{tag}"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    result_file = out / "tb_results.txt"
    result_file.unlink(missing_ok=True)
    src = Path(a.kernel).read_text()
    tb = Path(a.tb).read_text() if a.tb else rco.testbench(a.size, result_file)
    hdr = Path(a.hdr).read_text() if a.hdr else rco.header(a.size)
    if a.cosim and a.top != "autosa_mm":
        raise SystemExit("--cosim supports the autosa_mm top only")
    t0 = time.time()
    s = {"name": a.name, "kernel": str(Path(a.kernel).resolve()), "size": a.size, "mode": "cosim" if a.cosim else "csynth"}
    s.update(cosim(work, src, tb, hdr, "port" if a.vcd else "none") if a.cosim else csynth_only(work, src, tb, hdr, a.top))
    s["wall_s"] = round(time.time() - t0)
    if a.cosim:
        verdicts = result_file.read_text().splitlines() if result_file.exists() else []
        s["tb_verdicts"] = verdicts
        m = re.match(r"I=(\d+) J=(\d+) K=(\d+) err=(\d+)$", verdicts[-1]) if verdicts else None
        s["rtl_check_passed"] = bool(m) and len(verdicts) >= 2 and int(m.group(4)) == 0
    if a.vcd:
        r = subprocess.run(["bash", str(HERE / "vcd_rerun.sh"), str(work)], capture_output=True, text=True)
        (out / "vcd_rerun.log").write_text(r.stdout + r.stderr)
        v = work / "hls_proj/sol1/sim/verilog"
        for f in ("vcd_xsim.log", "run_xsim.sh", "vcd.tcl"):
            if (v / f).exists():
                shutil.copy(v / f, out / f"vcd_{f}")
        if (work / "axi.vcd.gz").exists():
            shutil.copy(work / "axi.vcd.gz", out / "axi.vcd.gz")
    for pat in ("hls_proj/sol1/sim/report/*cosim.rpt", "hls_proj/sol1/syn/report/csynth.rpt",
                "hls_proj/sol1/sim/verilog/*.result.lat.rb"):
        for f in work.glob(pat):
            shutil.copy(f, out / f.name)
    tail = (work / "vitis.log").read_text(errors="replace").splitlines()[-120:] if (work / "vitis.log").exists() else []
    (out / "vitis_tail.log").write_text("\n".join(tail) + "\n")
    with tarfile.open(out / "proj.tar.gz", "w:gz") as t:
        def keep(ti):
            p = ti.name
            if ("/sol1/sim/" in p and "/sim/report" not in p) or p.endswith((".wdb", ".vcd")) or "/impl/" in p:
                return None
            return ti
        t.add(work / "hls_proj", arcname="hls_proj", filter=keep)
        for f in ("kernel.cpp", "kernel.h", "testbench.cpp"):
            if (work / f).exists():
                t.add(work / f, arcname=f)
    shutil.rmtree(work, ignore_errors=True)
    (out / "summary.json").write_text(json.dumps(s, indent=2) + "\n")
    print(json.dumps(s))


if __name__ == "__main__":
    main()
