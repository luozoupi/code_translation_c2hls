#!/usr/bin/env python3
"""SODA target hand-patched for Vitis HLS 2023.2: designs/<cfg> -> designs_patched/<cfg>.

  patch_soda.py CFG [CFG ...]
Why: Vitis 2023.2 does not pipeline sodac's data-driven `for (bool enable = true; enable;)` module loops that have no
line buffer (~4x slower cosim; study/soda/dse/diag/, PROGRESS.md 2026-10-18/19). The patch, per module loop:
  * the loop runs a fixed trip count instead of exiting on the end-of-stream flag. The count is measured, not derived:
    an instrumented g++ build of the unpatched kernel at this config's size counts the iterations that pass the
    emptiness guard, per module function (it includes the end-of-stream element). All instances of one function must
    agree, otherwise the config is rejected;
  * the `if (!fifo.empty() ...)` guard becomes `if (true)` (blocking reads): with a fixed count an iteration that
    found a FIFO momentarily empty would be lost and the dataflow would deadlock (diag/v_fixed_cosim).
The datapath, FIFOs, bursts and pragmas are unchanged. A patched design is valid only at the size it was measured
for (config.json argv). Report it as "SODA hand-patched for Vitis 2023.2 (fixed trip counts)".
Then a g++ build of the patched kernel must pass SODA's own golden (INFO: PASS).
"""
import json, re, shutil, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
INC = "/scratch/hpc-prf-llmfpga/asa582/projects/tmp/c2hls/vitis_hls_2023.2_include"
LOOP = re.compile(r"for \(bool enable = true; enable;\)\n(\s*\{\n(?:#pragma[^\n]*\n)+)(\s*)if \(([^\n]*)\)\n(\s*\{\n)")


def funcs(src):
    """(name, start, end) of every `void ModuleNFunc(` ... `} // ModuleNFunc` body."""
    return [(m.group(1), m.start(), src.index(f"}} // {m.group(1)}", m.end()))
            for m in re.finditer(r"^void (Module\d+Func)\(", src, re.M)]


def build_run(w, argv):
    r = subprocess.run(f"g++ -O2 -std=c++17 kernel.cpp host.cpp -o t -DSODA_TEST_MAIN -DSODA_CPP_BINDING -I. -I{INC} "
                       f"&& ./t {argv}; rc=$?; rm -f t; exit $rc", shell=True, cwd=w, capture_output=True, text=True)
    return r.returncode, r.stdout + r.stderr


def patch(cfg):
    src_dir, out = HERE / "designs" / cfg, HERE / "designs_patched" / cfg
    c = json.loads((src_dir / "config.json").read_text())
    ker = (src_dir / "kernel.cpp").read_text()
    out.mkdir(parents=True, exist_ok=True)
    for f in ("host.cpp", "frt.h"):
        shutil.copy(src_dir / f, out / f)
    # 1. count: ++n after the guard, print per function instance at loop exit
    cnt = ker
    for name, s, e in reversed(funcs(ker)):
        body = cnt[s:e]
        body, k = LOOP.subn(lambda m: f"for (bool enable = true; enable;)\n{m.group(1)}{m.group(2)}if ({m.group(3)})\n"
                                      f"{m.group(4)}      ++patch_n;\n", body)
        assert k == 1, (cfg, name, k)
        body = body.replace("for (bool enable = true; enable;)", "long patch_n = 0;\n  for (bool enable = true; enable;)", 1)
        body += f'  fprintf(stderr, "PATCHCOUNT {name} %ld\\n", patch_n);\n'
        cnt = cnt[:s] + body + cnt[e:]
    w = out / "_count"
    w.mkdir(exist_ok=True)
    (w / "kernel.cpp").write_text("#include <cstdio>\n" + cnt)
    for f in ("host.cpp", "frt.h"):
        shutil.copy(src_dir / f, w / f)
    rc, log = build_run(w, c["argv"])
    shutil.rmtree(w)
    assert rc == 0 and "INFO: PASS" in log, (cfg, "count run failed", log[-800:])
    counts = {}
    for name, n in re.findall(r"PATCHCOUNT (\w+) (\d+)", log):
        counts.setdefault(name, set()).add(int(n))
    bad = {k: sorted(v) for k, v in counts.items() if len(v) != 1}
    assert not bad, (cfg, "instances of one function disagree", bad)
    counts = {k: v.pop() for k, v in counts.items()}
    assert set(counts) == {n for n, _, _ in funcs(ker)}, (cfg, "a module never ran", counts)
    # 2. patched kernel
    pk = ker
    for name, s, e in reversed(funcs(ker)):
        t = counts[name]
        body, k = LOOP.subn(lambda m: f"bool enable = true;  // patch: fixed trip count {t} (measured at {c['argv']})\n"
                                      f"  for (uint64_t t_patch = 0; t_patch < {t}ULL; ++t_patch)\n{m.group(1)}"
                                      f"{m.group(2)}if (true)  // patch: blocking reads (was: {m.group(3)})\n{m.group(4)}",
                            pk[s:e])
        assert k == 1, (cfg, name, k)
        pk = pk[:s] + body + pk[e:]
    code = re.sub(r"//[^\n]*", "", pk)
    assert "enable;)" not in code and ".empty()" not in code, (cfg, "unpatched loop or guard left")
    (out / "kernel.cpp").write_text(pk)
    rc, log = build_run(out, c["argv"])
    ok = rc == 0 and "INFO: PASS" in log
    (out / "gpp_check.log").write_text(log[-3000:])
    (out / "config.json").write_text(json.dumps(dict(c, patched=True, patch_counts=counts, patch_gpp_pass=ok), indent=1) + "\n")
    return ok, counts


if __name__ == "__main__":
    bad = 0
    for cfg in sys.argv[1:]:
        try:
            ok, counts = patch(cfg)
            print(cfg, "gpp=" + ("PASS" if ok else "FAIL"), counts, flush=True)
            bad += not ok
        except AssertionError as e:
            print(cfg, "REJECTED", e, flush=True)
            bad += 1
    sys.exit(1 if bad else 0)
