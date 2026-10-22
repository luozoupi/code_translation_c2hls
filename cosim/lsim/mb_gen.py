#!/usr/bin/env python3
"""Generate m_axi microbenchmarks (kernel, testbench, Vitis Tcl) for characterising Vitis 2023.2 cosim AXI timing.

  mb_gen.py OUT_DIR [--only NAME[,NAME...]] [--trace port]

Every kernel uses 512-bit ap_uint ports, one m_axi bundle per pointer (as in our autosa_mm designs), s_axilite
control and compile-time trip counts, so csynth latency is fixed and comparable with cosim and LightningSim.
Kinds:
  rd  read N beats in a pipelined loop (II = ii), XOR-reduce, write one beat
  wr  write N beats
  cp  copy N beats in one pipelined loop
  df  DATAFLOW load(A) -> stream -> proc (II = ii) -> stream -> store(C)
`off` shifts the start beat, so a burst crosses a 4 KB boundary when off*64 + len crosses it.
"""
import argparse, json, re
from pathlib import Path

PART, CLOCK = "xcu280-fsvh2892-2L-e", 3.33


def configs():
    out = []

    def add(kind, n, lat=None, mbl=16, nout=16, ii=1, off=0, dw=512):
        name = f"{kind}_n{n}" + (f"_lat{lat}" if lat is not None else "") + (f"_mbl{mbl}" if mbl != 16 else "") \
            + (f"_out{nout}" if nout != 16 else "") + (f"_ii{ii}" if ii != 1 else "") + (f"_off{off}" if off else "") \
            + (f"_dw{dw}" if dw != 512 else "")
        out.append(dict(name=name, kind=kind, n=n, lat=lat, mbl=mbl, nout=nout, ii=ii, off=off, dw=dw))

    for kind in ("rd", "wr", "cp", "df"):
        for n in (1, 2, 16, 17, 64, 256, 1024, 4096):
            add(kind, n)
    for kind in ("rd", "wr", "cp"):
        for mbl in (64, 256):
            add(kind, 1024, mbl=mbl)
        for lat in (1, 64):
            add(kind, 1024, lat=lat)
    for kind in ("rd", "wr"):
        for nout in (1, 4):
            add(kind, 1024, nout=nout)
        add(kind, 256, off=60)
    for kind in ("rd", "df"):
        for ii in (2, 4):
            add(kind, 1024, ii=ii)
    add("df", 4096, mbl=64)  # the burst setting our autosa_mm designs use
    # Set 2: narrow (32-bit) ports and dependent read-modify-write chains through memory.
    for kind in ("rd", "wr", "cp"):
        for n in (16, 1024):
            add(kind, n, dw=32)
    for dw in (32, 512):
        for n in (16, 64):
            add("chain", n, dw=dw)
    # Set 3: single-beat requests in a II=1 loop (burst inference fails: access under a runtime condition), the
    # pattern csynth reports as "Widened 1 / Inferred 16 floats" in the autosa_mm designs.
    for lat in (None, 1):
        add("rdsc", 1024, lat=lat)
        add("wrsc", 1024, lat=lat)
    add("rdsc", 1024, nout=4)
    return out


def axi_pragma(port, bundle, c, read, write):
    p = f"#pragma HLS INTERFACE m_axi port={port} offset=slave bundle={bundle} depth={c['n'] + c['off']}"
    if c["lat"] is not None:
        p += f" latency={c['lat']}"
    if read:
        p += f" max_read_burst_length={c['mbl']} num_read_outstanding={c['nout']}"
    if write:
        p += f" max_write_burst_length={c['mbl']} num_write_outstanding={c['nout']}"
    return p


def kernel(c):
    n, off, ii = c["n"], c["off"], c["ii"]
    head = f"""#include <ap_int.h>
#include <hls_stream.h>
typedef ap_uint<{c.get('dw', 512)}> w_t;
#define N {n}
#define OFF {off}
"""
    ctl = "#pragma HLS INTERFACE s_axilite port=return bundle=control\n"
    if c["kind"] == "rd":
        return head + f"""void mb(w_t *A, w_t *C) {{
{axi_pragma('A', 'gmem0', c, True, False)}
{axi_pragma('C', 'gmem1', c, False, True)}
#pragma HLS INTERFACE s_axilite port=A bundle=control
#pragma HLS INTERFACE s_axilite port=C bundle=control
{ctl}  w_t acc = 0;
  for (int i = 0; i < N; i++) {{
#pragma HLS PIPELINE II={ii}
    acc ^= A[OFF + i];
  }}
  C[OFF] = acc;
}}
"""
    if c["kind"] == "wr":
        return head + f"""void mb(w_t *A, w_t *C) {{
{axi_pragma('A', 'gmem0', c, True, False)}
{axi_pragma('C', 'gmem1', c, False, True)}
#pragma HLS INTERFACE s_axilite port=A bundle=control
#pragma HLS INTERFACE s_axilite port=C bundle=control
{ctl}  w_t seed = A[0];
  for (int i = 0; i < N; i++) {{
#pragma HLS PIPELINE II={ii}
    C[OFF + i] = seed + i;
  }}
}}
"""
    if c["kind"] in ("rdsc", "wrsc"):
        body = ("    if (i != m) acc ^= A[OFF + i];" if c["kind"] == "rdsc" else "    if (i != m) C[OFF + i] = seed + i;")
        pre = "  w_t acc = 0;" if c["kind"] == "rdsc" else "  w_t seed = A[0];"
        post = "  C[OFF] = acc;" if c["kind"] == "rdsc" else ""
        return head + f"""void mb(w_t *A, w_t *C, int m) {{
{axi_pragma('A', 'gmem0', c, True, False)}
{axi_pragma('C', 'gmem1', c, False, True)}
#pragma HLS INTERFACE s_axilite port=A bundle=control
#pragma HLS INTERFACE s_axilite port=C bundle=control
#pragma HLS INTERFACE s_axilite port=m bundle=control
{ctl}{pre}
  for (int i = 0; i < N; i++) {{
#pragma HLS PIPELINE II=1
{body}
  }}
{post}
}}
"""
    if c["kind"] == "chain":
        return head + f"""void mb(w_t *A, w_t *C) {{
{axi_pragma('A', 'gmem0', c, True, False)}
{axi_pragma('C', 'gmem1', c, True, True)}
#pragma HLS INTERFACE s_axilite port=A bundle=control
#pragma HLS INTERFACE s_axilite port=C bundle=control
{ctl}  for (int i = 0; i < N; i++) {{
    C[OFF] = C[OFF] + A[OFF + i];
  }}
}}
"""
    if c["kind"] == "cp":
        return head + f"""void mb(w_t *A, w_t *C) {{
{axi_pragma('A', 'gmem0', c, True, False)}
{axi_pragma('C', 'gmem1', c, False, True)}
#pragma HLS INTERFACE s_axilite port=A bundle=control
#pragma HLS INTERFACE s_axilite port=C bundle=control
{ctl}  for (int i = 0; i < N; i++) {{
#pragma HLS PIPELINE II={ii}
    C[OFF + i] = A[OFF + i] + 1;
  }}
}}
"""
    assert c["kind"] == "df"
    return head + f"""static void load(w_t *A, hls::stream<w_t> &s) {{
  for (int i = 0; i < N; i++) {{
#pragma HLS PIPELINE II=1
    s.write(A[OFF + i]);
  }}
}}
static void proc(hls::stream<w_t> &s, hls::stream<w_t> &t) {{
  for (int i = 0; i < N; i++) {{
#pragma HLS PIPELINE II={ii}
    t.write(s.read() + 1);
  }}
}}
static void store(hls::stream<w_t> &t, w_t *C) {{
  for (int i = 0; i < N; i++) {{
#pragma HLS PIPELINE II=1
    C[OFF + i] = t.read();
  }}
}}
void mb(w_t *A, w_t *C) {{
{axi_pragma('A', 'gmem0', c, True, False)}
{axi_pragma('C', 'gmem1', c, False, True)}
#pragma HLS INTERFACE s_axilite port=A bundle=control
#pragma HLS INTERFACE s_axilite port=C bundle=control
{ctl}#pragma HLS DATAFLOW
  hls::stream<w_t> s("s"), t("t");
#pragma HLS STREAM variable=s depth=2
#pragma HLS STREAM variable=t depth=2
  load(A, s);
  proc(s, t);
  store(t, C);
}}
"""


def testbench(c):
    n, off = c["n"], c["off"]
    kind = c["kind"]
    if kind == "rd":
        check = "  w_t acc = 0; for (int i = 0; i < N; i++) acc ^= A[OFF + i];\n  int err = (C[OFF] != acc);\n"
    elif kind == "rdsc":
        check = "  w_t acc = 0; for (int i = 0; i < N; i++) acc ^= A[OFF + i];\n  int err = (C[OFF] != acc);\n"
    elif kind == "wrsc":
        check = "  int err = 0; for (int i = 0; i < N; i++) err += (C[OFF + i] != A[0] + i);\n"
    elif kind == "chain":
        check = "  w_t acc = 0; for (int i = 0; i < N; i++) acc += A[OFF + i];\n  int err = (C[OFF] != acc);\n"
    elif kind == "wr":
        check = "  int err = 0; for (int i = 0; i < N; i++) err += (C[OFF + i] != A[0] + i);\n"
    else:
        check = "  int err = 0; for (int i = 0; i < N; i++) err += (C[OFF + i] != A[OFF + i] + 1);\n"
    return f"""#include <stdio.h>
#include <stdlib.h>
#include <ap_int.h>
typedef ap_uint<{c.get('dw', 512)}> w_t;
#define N {n}
#define OFF {off}
void mb(w_t *A, w_t *C{", int m" if kind in ("rdsc", "wrsc") else ""});
static w_t A[N + OFF], C[N + OFF];
int main() {{
  srand(1);
  for (int i = 0; i < N + OFF; i++) {{
    w_t v = 0;
    for (int w = 0; w < {c.get('dw', 512) // 32}; w++) v.range(32 * w + 31, 32 * w) = rand();
    A[i] = v; C[i] = 0;
  }}
  mb(A, C{", -1" if kind in ("rdsc", "wrsc") else ""});
{check}  if (err) printf("Failed with %d errors!\\n", err); else printf("Passed!\\n");
  return err ? 1 : 0;
}}
"""


def tcl(trace):
    return f"""open_project hls_proj
set_top mb
add_files kernel.cpp
add_files -tb tb.cpp
open_solution "sol1" -flow_target vitis
set_part {{{PART}}}
create_clock -period {CLOCK} -name default
csynth_design
set ::env(LIBRARY_PATH) {{/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu}}
cosim_design -trace_level {trace}
exit
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--only", default="")
    ap.add_argument("--trace", default="none")
    a = ap.parse_args()
    only = set(filter(None, re.split("[,:]", a.only)))
    out = Path(a.out)
    cs = [c for c in configs() if not only or c["name"] in only]
    for c in cs:
        d = out / c["name"]
        d.mkdir(parents=True, exist_ok=True)
        (d / "kernel.cpp").write_text(kernel(c))
        (d / "tb.cpp").write_text(testbench(c))
        (d / "run.tcl").write_text(tcl(a.trace))
        (d / "config.json").write_text(json.dumps(c, indent=2) + "\n")
    print("\n".join(c["name"] for c in cs))


if __name__ == "__main__":
    main()
