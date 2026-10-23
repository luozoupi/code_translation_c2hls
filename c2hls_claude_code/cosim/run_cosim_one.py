#!/usr/bin/env python3
"""C/RTL cosim of one autosa_mm kernel on the n1024 bench (I = J = K = 1024), keeping only small files.

  run_cosim_one.py NAME KERNEL.cpp [--size N]   (--size only for harness self-tests; real runs are 1024)

The testbench is the bench's testbench.cpp with reporting changes only (the comparison, tolerance and data are
unchanged): it static_asserts the problem size, prints it, returns nonzero on a mismatch, and appends one
verdict line per run to RESULT_FILE. Vitis runs the testbench once to capture inputs and once more on the RTL
outputs, so the LAST verdict line is the RTL check. Cosim runs with trace_level none (no waveform). After the
run, reports and a log excerpt are copied to <here>/<NAME>/ and the Vitis work directory is deleted.
"""
import json, os, re, shutil, sys, time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
BENCH = Path("/scratch/hpc-prf-llmfpga/asa582/projects/workspaces/code_translation_c2hls/artifacts/pc2/autosa_mm_ijk_benches/n1024/autosa_mm")
TMP = Path(os.environ.get("C2HLS_TMP_ROOT", "/scratch/hpc-prf-llmfpga/asa582/projects/tmp/c2hls"))
sys.path.insert(0, "/scratch/hpc-prf-llmfpga/asa582/projects/workspaces/code_translation_c2hls")


def testbench(size: int, result_file: Path) -> str:
    tb = (BENCH / "testbench.cpp").read_text()
    pre = (f'#include <stdio.h>\nstatic_assert(I == {size} && J == {size} && K == {size}, "cosim must run the {size}^3 bench");\n'
           f'#define COSIM_RESULT_FILE "{result_file}"\n')
    tb = tb.replace('extern "C" void autosa_mm', pre + 'extern "C" void autosa_mm', 1)
    tb = tb.replace("  autosa_mm(A, B, C);", '  printf("TB_SIZE I=%d J=%d K=%d\\n", I, J, K);\n  autosa_mm(A, B, C);', 1)
    old_tail = """  if (err)
    printf("Failed with %d errors!\\n", err);
  else
    printf("Passed!\\n");

  return 0;"""
    new_tail = """  if (err)
    printf("Failed with %d errors!\\n", err);
  else
    printf("Passed!\\n");
  {
    FILE *rf = fopen(COSIM_RESULT_FILE, "a");
    if (rf) { fprintf(rf, "I=%d J=%d K=%d err=%d\\n", I, J, K, err); fclose(rf); }
  }
  return err ? 1 : 0;"""
    assert tb.count(old_tail) == 1 and "TB_SIZE" in tb
    return tb.replace(old_tail, new_tail)


def header(size: int) -> str:
    h = (BENCH / "kernel.h").read_text()
    if size != 1024:  # self-test only
        h = h.replace("#define I 1024\n#define J 1024\n#define K 1024", f"#define I {size}\n#define J {size}\n#define K {size}")
    assert f"#define I {size}" in h
    return h


def main() -> int:
    name, kernel = sys.argv[1], Path(sys.argv[2]).resolve()
    size = int(sys.argv[sys.argv.index("--size") + 1]) if "--size" in sys.argv else 1024
    out = HERE / name
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    work = TMP / f"cosim_{name}_{stamp}"
    work.mkdir(parents=True)
    result_file = out / "tb_results.txt"
    result_file.unlink(missing_ok=True)
    src = kernel.read_text()
    tb, hdr = testbench(size, result_file), header(size)
    shutil.copy(kernel, out / "kernel.cpp")
    (out / "testbench_cosim.cpp").write_text(tb)
    import hls_eval, shlex
    # c2hls's xelab "-mt off" path (C2HLS_COSIM_XELAB_MT_OFF=1) avoids the xelab SIGSEGV (XSIM 43-3316) on large
    # designs, but runs sim.sh outside the Vitis container. Run that one command inside the same container the
    # vitis-run wrapper uses (scripts/fir/bin/vitis-run); every other Vitis call is unchanged.
    original_run = hls_eval._run_vitis_cmd

    def run_in_container(cmd, timeout):
        if "sim.sh" in cmd and os.environ.get("XILINX_SIF"):
            inner = ('export LIBRARY_PATH="/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu${LIBRARY_PATH:+:$LIBRARY_PATH}"; '
                     "ulimit -s unlimited 2>/dev/null || true; " + cmd)
            cmd = f"apptainer exec {shlex.quote(os.environ['XILINX_SIF'])} bash -lc {shlex.quote(inner)}"
        return original_run(cmd, timeout)

    hls_eval._run_vitis_cmd = run_in_container
    t0 = time.time()
    r = hls_eval.run_cosim(src, tb, hdr, header_name="kernel.h", top_function="autosa_mm",
                           part="xcu280-fsvh2892-2L-e", clock_ns=3.33, work_dir=str(work))
    wall = round(time.time() - t0)
    log = str(r.get("log") or "")
    verdicts = result_file.read_text().splitlines() if result_file.exists() else []
    m = re.match(r"I=(\d+) J=(\d+) K=(\d+) err=(\d+)$", verdicts[-1]) if verdicts else None
    rtl_ok = bool(m) and len(verdicts) >= 2 and int(m.group(4)) == 0 and all(int(m.group(i)) == size for i in (1, 2, 3))
    s = {"name": name, "kernel": str(kernel), "size": size, "tb_verdicts": verdicts,
         "rtl_check_passed": rtl_ok, "vitis_success": bool(r.get("success")), "vitis_passed": bool(r.get("passed")),
         "kernel_runtime_cycles": r.get("kernel_runtime_cycles"), "error": (r.get("error") or "")[:2000],
         "wallclock_s": wall, "work_dir_deleted": str(work)}
    s["correct"] = rtl_ok and s["vitis_success"]
    # Keep reports and a log excerpt only.
    keep_rpts = [x for x in work.rglob("*.rpt") if "/.autopilot/" not in str(x)] + list(work.rglob("*.result.lat.rb"))
    for rpt in keep_rpts:
        if rpt.stat().st_size < 2_000_000:
            dest = out / "reports" / rpt.relative_to(work)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(rpt, dest)
    lines = log.splitlines()
    keep = [l for l in lines if re.search(r"TB_SIZE|Passed!|Failed with|ERROR|CRITICAL|co-simulation|COSIM|lat|\$finish|xsim", l)]
    (out / "cosim_log_excerpt.txt").write_text("\n".join(keep[-400:] + ["", "----- tail -----"] + lines[-200:]) + "\n")
    shutil.rmtree(work, ignore_errors=True)
    (out / "cosim_summary.json").write_text(json.dumps(s, indent=2) + "\n")
    print(json.dumps(s, indent=2))
    return 0 if s["correct"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
