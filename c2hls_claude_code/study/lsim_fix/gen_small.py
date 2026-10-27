#!/usr/bin/env python3
"""Small SODA designs for cycle-level comparison of LightningSim with cosim (waveforms are affordable at this size).

  gen_small.py            -> soda_small/designs/<name> (as generated), soda_small/designs_patched/<name>
                             (patch_soda.py: fixed trip counts + blocking reads), and <name>_stp copies of the patched
                             3D ones with `style = stp` on every pipeline
Same sodac flags as study/soda/dse/gen_dse.py (burst 512, one tile); sizes 64x64 (2D) and 16x16x16 (3D).
"""
import json, re, shutil, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DSE = HERE.parent / "soda" / "dse"
sys.path.insert(0, str(DSE))
import gen_dse, patch_soda  # noqa: E402

OUT = HERE / "soda_small"
gen_dse.HERE = OUT
gen_dse.SODAC = "/scratch/hpc-prf-llmfpga/asa582/projects/packages/soda-venv-otus/bin/sodac"   # Otus venv, same pinned versions as Fir's
patch_soda.HERE = OUT

PICK = {("blur", 8, False), ("sobel2d", 8, False), ("erosion", 8, False), ("erosion", 8, True), ("xcorr", 8, False),
        ("xcorr", 8, True), ("jacobi2d", 4, False), ("seidel2d", 4, False), ("heat3d", 2, False),
        ("jacobi3d", 2, False), ("heat3d", 1, False)}


def main():
    done = []
    for c in gen_dse.configs():
        if c["burst"] != 512 or (c["kernel"], c["unroll"], c["cr"]) not in PICK:
            continue
        c = dict(c, sizes=[64, 64] if len(c["sizes"]) == 2 else [16, 16, 16])
        d, c = gen_dse.gen(c)
        ok = gen_dse.check(d, c)
        print(c["name"], "gen gpp=" + ("PASS" if ok else "FAIL"), flush=True)
        try:
            pok, counts = patch_soda.patch(c["name"])
        except AssertionError as e:
            print(c["name"], "patch REJECTED", e, flush=True)
            continue
        print(c["name"], "patch gpp=" + ("PASS" if pok else "FAIL"), flush=True)
        if c["kernel"].endswith("3d"):
            src, dst = OUT / "designs_patched" / c["name"], OUT / "designs_patched" / (c["name"] + "_stp")
            shutil.copytree(src, dst, dirs_exist_ok=True)
            k = (dst / "kernel.cpp").read_text()
            k, n = re.subn(r"^(\s*#pragma HLS pipeline II = [0-9]+)\s*$", r"\1 style = stp", k, flags=re.M)
            (dst / "kernel.cpp").write_text(k)
            print(c["name"] + "_stp", n, "pipelines set to stp", flush=True)
        done.append(c["name"])
    print("done", len(done))


if __name__ == "__main__":
    main()
