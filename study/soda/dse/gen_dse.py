#!/usr/bin/env python3
"""SODA design space for the study: sodac kernel + SODA's own test main as the Vitis testbench, one dir per config.

  gen_dse.py [--only NAME,...] [--check]     -> designs/<name>/{kernel.cpp, host.cpp, frt.h, config.json}
Space: burst 512, single tile (tile = image width; SODA's multi-tile host is buggy), study iterate count,
unroll {1,2,4,8,16} for float kernels and {8,16,32} for 16-bit kernels, computation reuse on/off for
contrast/erosion/xcorr. Plus LAAFD's setting (arXiv 2602.06085): burst 128, unroll 4, every kernel,
named <k>_u4_b128[_cr][_laafd], at LAAFD's sizes. Problem size = bench_soda metadata (2D 1024^2, contrast 512^2, 3D 128^3).
Edits to sodac output (cosim plumbing only, no datapath change):
  kernel.cpp: `depth = N` on each m_axi pragma (Vitis cosim needs a depth; N = the host's buffer size in words);
  host.cpp:   each buffer allocation padded by 1 MiB so the cosim adapter never reads past it.
--check: g++ build of kernel + host at the study size (C simulation of the dataflow) and run SODA's golden check.
"""
import argparse, json, math, re, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BS = HERE.parents[1] / "bench_soda"
SODAC = "/scratch/asa582/workspaces/soda_src/venv/bin/sodac"
INC = "/scratch/asa582/tmp/c2hls/vitis_hls_2023.2_include"
INT16 = {"blur", "erosion", "sobel2d", "xcorr"}
CR = {"contrast", "erosion", "xcorr"}


def configs():
    out = []
    for k in sorted(p.name for p in BS.iterdir() if (p / "metadata.json").exists()):
        m = json.loads((BS / k / "metadata.json").read_text())
        for u in ((8, 16, 32) if k in INT16 else (1, 2, 4, 8, 16)):
            for cr in ((False, True) if k in CR else (False,)):
                out.append(dict(name=f"{k}_u{u}{'_cr' if cr else ''}", kernel=k, unroll=u, cr=cr, burst=512,
                                iterate=m["soda"]["iterate"], sizes=m["soda"]["sizes"],
                                src=m["algorithm_source_path"]))
        # LAAFD setting: 128-bit data path, unroll 4, LAAFD's sizes (2D 1024^2, 3D 256^3). Where that differs from
        # the bench size (contrast, 3D kernels) the name ends in _laafd; the bench-size dirs from 2026-10-17 stay.
        sz = [1024, 1024] if len(m["soda"]["sizes"]) == 2 else [256, 256, 256]
        tag = "_laafd" if sz != m["soda"]["sizes"] else ""
        for cr in ((False, True) if k in CR else (False,)):
            out.append(dict(name=f"{k}_u4_b128{'_cr' if cr else ''}{tag}", kernel=k, unroll=4, cr=cr, burst=128,
                            iterate=m["soda"]["iterate"], sizes=sz, src=m["algorithm_source_path"]))
    return out


def gen(c):
    d = HERE / "designs" / c["name"]
    d.mkdir(parents=True, exist_ok=True)
    tile = [str(s) for s in c["sizes"][:-1]]
    cmd = [SODAC, c["src"], "--burst-width", str(c["burst"]), "--unroll-factor", str(c["unroll"]), "--iterate", str(c["iterate"]),
           "--tile-size", *tile, "--xocl-kernel", str(d / "kernel.cpp"), "--frt-host", str(d / "host.cpp")]
    if c["cr"]:
        cmd += ["--computation-reuse", "yes"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    (d / "sodac.log").write_text(" ".join(cmd) + "\n" + r.stdout + r.stderr)
    assert r.returncode == 0, f"{c['name']}: sodac failed, see {d}/sodac.log"
    host = (d / "host.cpp").read_text()
    # buffer words per port, from the host's own formula with one tile (tile = extents)
    dist = int(re.search(r"kStencilDistance = (\d+);", host).group(1))
    n = math.prod(c["sizes"])
    depth = {}
    for name, w in re.findall(r"constexpr int kWidth_(\w+) = (\d+);", host):
        epc = c["burst"] // int(w)
        depth[name] = math.ceil(n / epc) + math.ceil(dist / epc) + 1
    host, k = re.subn(r"aligned_alloc\(4096, round_up\((buf_size_\w+), 4096\)\)",
                      r"aligned_alloc(4096, round_up(\1 + (1 << 20), 4096))", host)
    assert k >= 2, "allocation pattern"
    (d / "host.cpp").write_text(host)
    ker = (d / "kernel.cpp").read_text()

    def dep(m):
        port = m.group(1)
        t = re.match(r"bank_\d+_(\w+)$", port).group(1)
        return f"{m.group(0)} depth = {depth[t]}"
    ker, k = re.subn(r"#pragma HLS interface m_axi port = (\w+) offset = slave bundle = \w+", dep, ker)
    assert k >= 2, "m_axi pragma pattern"
    (d / "kernel.cpp").write_text(ker)
    (d / "frt.h").write_text("#pragma once\n")
    top = re.search(r'extern "C" \{\s*void (\w+_kernel)\(', ker).group(1)
    c = dict(c, top=top, depth=depth, stencil_distance=dist, argv=" ".join(["x"] + [str(s) for s in c["sizes"]]))
    (d / "config.json").write_text(json.dumps(c, indent=1) + "\n")
    return d, c


def check(d, c):
    b = d / "gpp_check"
    r = subprocess.run(f"g++ -O2 -std=c++17 kernel.cpp host.cpp -o gpp_check -DSODA_TEST_MAIN -DSODA_CPP_BINDING "
                       f"-I. -I{INC} && ./gpp_check {c['argv']}", shell=True, cwd=d, capture_output=True, text=True)
    b.unlink(missing_ok=True)
    tail = (r.stdout + r.stderr)[-3000:]
    (d / "gpp_check.log").write_text(tail)
    return "INFO: PASS" in tail and r.returncode == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    cs = configs()
    if a.only:
        cs = [c for c in cs if c["name"] in a.only.split(",")]
    for c in cs:
        d, c = gen(c)
        msg = f"{c['name']}: top={c['top']} depth={c['depth']}"
        if a.check:
            msg += " gpp=" + ("PASS" if check(d, c) else "FAIL")
        print(msg, flush=True)


if __name__ == "__main__":
    main()
