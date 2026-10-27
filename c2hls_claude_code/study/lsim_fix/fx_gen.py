#!/usr/bin/env python3
"""Microbenchmarks for the LightningSim FRP and non-blocking-read work (study/lsim_fix/).

  fx_gen.py OUT_DIR [--only NAME[,NAME...]]      -> OUT_DIR/<name>/{kernel.cpp, tb.cpp, run.tcl, config.json}; prints names

Same conventions as cosim/lsim/mb_gen.py (512-bit m_axi A -> C, one bundle each, s_axilite control, U280, 3.33 ns), so the
cosim report mapping is P=8, K=51. Families:
  f   load(A) -> K streams -> comp (pipeline II=cii, style frp|stp) -> stream -> store(C)
      lat s: int multiply-add per input; lat l: float multiply-add chain (long pipeline, like SODA's Module6Func)
      pii/sii: producer/consumer II (bubbles into, back-pressure out of the FRP pipeline); d: input FIFO depth;
      cond: inputs 1..K-1 are read only on odd iterations; chain2: two comp stages in a row
  p   load -> K streams -> poll (SODA pattern: `if (!s0.empty() && ...) { read all; last flag ends the loop }`) -> store
      pipe: poll loop has `pipeline II=1` (Vitis may still leave it unpipelined, as for SODA)
"""
import argparse, json, re
from pathlib import Path

PART, CLOCK = "xcu280-fsvh2892-2L-e", 3.33


def configs():
    out = []

    def f(k, lat, style, d=2, pii=1, sii=1, cii=1, cond=False, chain=1, n=2048):
        name = (f"f_k{k}_{lat}_{style}" + (f"_d{d}" if d != 2 else "") + (f"_pii{pii}" if pii != 1 else "")
                + (f"_sii{sii}" if sii != 1 else "") + (f"_cii{cii}" if cii != 1 else "") + ("_cond" if cond else "")
                + (f"_chain{chain}" if chain != 1 else ""))
        out.append(dict(name=name, fam="f", k=k, lat=lat, style=style, d=d, pii=pii, sii=sii, cii=cii, cond=cond,
                        chain=chain, n=n))

    for style in ("frp", "stp"):
        f(1, "s", style)
        f(1, "l", style)
        f(3, "l", style)
        f(7, "l", style)
        f(7, "l", style, d=16)
        f(3, "l", style, pii=3)
        f(3, "l", style, sii=3)
        f(1, "s", style, sii=3)
        f(3, "l", style, cii=2)
        f(3, "l", style, cond=True)
        f(3, "l", style, chain=2)

    def p(k, pipe=True, pii=1, sii=1, chain=1, d=2, n=2048):
        name = (f"p_k{k}" + ("_pipe" if pipe else "_nopipe") + (f"_d{d}" if d != 2 else "")
                + (f"_pii{pii}" if pii != 1 else "") + (f"_sii{sii}" if sii != 1 else "")
                + (f"_chain{chain}" if chain != 1 else ""))
        out.append(dict(name=name, fam="p", k=k, pipe=pipe, pii=pii, sii=sii, chain=chain, d=d, n=n))

    p(1)
    p(3)
    p(1, pipe=False)
    p(1, pii=4)
    p(1, sii=3)
    p(1, chain=3)
    p(3, pii=4, chain=2)
    p(1, d=16, pii=4)
    return out


HEAD = """#include <ap_int.h>
#include <hls_stream.h>
typedef ap_uint<512> w_t;
#define N {n}
#define K {k}
"""

TOP_PRAGMAS = """#pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem0 depth=N max_read_burst_length=16 num_read_outstanding=16
#pragma HLS INTERFACE m_axi port=C offset=slave bundle=gmem1 depth=N max_write_burst_length=16 num_write_outstanding=16
#pragma HLS INTERFACE s_axilite port=A bundle=control
#pragma HLS INTERFACE s_axilite port=C bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control
#pragma HLS DATAFLOW
"""


def step_expr(c, x):
    """acc update for one input: short = int, long = float chain."""
    return f"acc = acc * 3 + {x};" if c["lat"] == "s" else f"acc = acc * 1.5f + (float){x};"


def kernel_f(c):
    acc_t = "int" if c["lat"] == "s" else "float"
    reads = []
    for i in range(c["k"]):
        guard = f"if ({'1' if i == 0 or not c['cond'] else '(i & 1)'}) " if c["cond"] else ""
        reads.append(f"      {guard}{{ int x = s[{i}].read(); {step_expr(c, 'x')} }}")
    comp_body = "\n".join(reads)
    stages = []
    for j in range(c["chain"]):
        src = "s" if j == 0 else f"m{j}"
        stages.append(f"""static void comp{j}(hls::stream<int> {'s[K]' if j == 0 else '&s0'}, hls::stream<int> &o) {{
  for (int i = 0; i < N; i++) {{
#pragma HLS PIPELINE II={c['cii']} style={c['style']}
    {acc_t} acc = i;
{comp_body if j == 0 else '      { int x = s0.read(); ' + step_expr(c, 'x') + ' }'}
    o.write((int)acc);
  }}
}}
""")
    calls = []
    for j in range(c["chain"]):
        src = "s" if j == 0 else f"m{j}"
        dst = "o" if j == c["chain"] - 1 else f"m{j + 1}"
        calls.append(f"  comp{j}({src}, {dst});")
    mids = "".join(f"  hls::stream<int> m{j}(\"m{j}\");\n#pragma HLS STREAM variable=m{j} depth=2\n"
                   for j in range(1, c["chain"]))
    return HEAD.format(**c) + f"""static void load(w_t *A, hls::stream<int> s[K]) {{
  for (int i = 0; i < N; i++) {{
#pragma HLS PIPELINE II={c['pii']}
    w_t v = A[i];
    for (int k = 0; k < K; k++) {{
#pragma HLS UNROLL
      if ({'1' if not c['cond'] else 'k == 0 || (i & 1)'}) s[k].write((int)v.range(32 * k + 31, 32 * k));
    }}
  }}
}}
{''.join(stages)}static void store(hls::stream<int> &o, w_t *C) {{
  for (int i = 0; i < N; i++) {{
#pragma HLS PIPELINE II={c['sii']}
    C[i] = (w_t)(unsigned)o.read();
  }}
}}
void mb(w_t *A, w_t *C) {{
{TOP_PRAGMAS}  hls::stream<int> s[K];
#pragma HLS STREAM variable=s depth={c['d']}
  hls::stream<int> o("o");
#pragma HLS STREAM variable=o depth=2
{mids}  load(A, s);
{chr(10).join(calls)}
  store(o, C);
}}
"""


def golden_f(c):
    acc_t = "int" if c["lat"] == "s" else "float"
    lines = [f"    {acc_t} acc = i;"]
    for k in range(c["k"]):
        g = "" if not c["cond"] or k == 0 else "if (i & 1) "
        lines.append(f"    {g}{{ int x = (int)A[i].range(32 * {k} + 31, 32 * {k}); {step_expr(c, 'x')} }}")
    lines.append("    int r = (int)acc;")
    for _ in range(1, c["chain"]):
        lines.append(f"    {{ {acc_t} acc = i; {step_expr(c, 'r')} r = (int)acc; }}")
    return "\n".join(lines)


def kernel_p(c):
    # SODA pattern: one element carries a last flag (bit 32); the poll loop runs until it has seen it.
    stages = []
    for j in range(c["chain"]):
        kin = c["k"] if j == 0 else 1
        cond = " && ".join(f"!s[{i}].empty()" for i in range(kin)) if j == 0 else "!s0.empty()"
        reads = "\n".join(f"      ap_uint<33> v{i} = s[{i}].read();" for i in range(kin)) if j == 0 else \
            "      ap_uint<33> v0 = s0.read();"
        summ = " + ".join(f"(int)v{i}.range(31, 0)" for i in range(kin))
        stages.append(f"""static void poll{j}(hls::stream<ap_uint<33> > {'s[K]' if j == 0 else '&s0'}, hls::stream<ap_uint<33> > &o) {{
  for (bool enable = true; enable;) {{
{'#pragma HLS PIPELINE II=1' if c['pipe'] else ''}
    if ({cond}) {{
{reads}
      enable = !v0[32];
      ap_uint<33> r = (unsigned)({summ} + 1);
      r[32] = v0[32];
      o.write(r);
    }}
  }}
}}
""")
    calls = []
    for j in range(c["chain"]):
        src = "s" if j == 0 else f"m{j}"
        dst = "o" if j == c["chain"] - 1 else f"m{j + 1}"
        calls.append(f"  poll{j}({src}, {dst});")
    mids = "".join(f"  hls::stream<ap_uint<33> > m{j}(\"m{j}\");\n#pragma HLS STREAM variable=m{j} depth={c['d']}\n"
                   for j in range(1, c["chain"]))
    return HEAD.format(**c) + f"""static void load(w_t *A, hls::stream<ap_uint<33> > s[K]) {{
  for (int i = 0; i < N; i++) {{
#pragma HLS PIPELINE II={c['pii']}
    w_t v = A[i];
    for (int k = 0; k < K; k++) {{
#pragma HLS UNROLL
      ap_uint<33> e = (unsigned)v.range(32 * k + 31, 32 * k);
      e[32] = (i == N - 1);
      s[k].write(e);
    }}
  }}
}}
{''.join(stages)}static void store(hls::stream<ap_uint<33> > &o, w_t *C) {{
  for (int i = 0; i < N; i++) {{
#pragma HLS PIPELINE II={c['sii']}
    C[i] = (w_t)(unsigned)o.read().range(31, 0);
  }}
}}
void mb(w_t *A, w_t *C) {{
{TOP_PRAGMAS}  hls::stream<ap_uint<33> > s[K];
#pragma HLS STREAM variable=s depth={c['d']}
  hls::stream<ap_uint<33> > o("o");
#pragma HLS STREAM variable=o depth=2
{mids}  load(A, s);
{chr(10).join(calls)}
  store(o, C);
}}
"""


def golden_p(c):
    s = " + ".join(f"(int)A[i].range(32 * {k} + 31, 32 * {k})" for k in range(c["k"]))
    lines = [f"    int r = {s} + 1;"] + ["    r = r + 1;"] * (c["chain"] - 1)
    return "\n".join(lines)


def testbench(c):
    gold = golden_f(c) if c["fam"] == "f" else golden_p(c)
    return f"""#include <stdio.h>
#include <stdlib.h>
#include <ap_int.h>
typedef ap_uint<512> w_t;
#define N {c['n']}
void mb(w_t *A, w_t *C);
static w_t A[N], C[N];
int main() {{
  srand(1);
  for (int i = 0; i < N; i++) {{
    w_t v = 0;
    for (int w = 0; w < 16; w++) v.range(32 * w + 31, 32 * w) = rand() & 0xffff;
    A[i] = v; C[i] = 0;
  }}
  mb(A, C);
  int err = 0;
  for (int i = 0; i < N; i++) {{
{gold}
    err += (C[i] != (w_t)(unsigned)r);
  }}
  if (err) printf("Failed with %d errors!\\n", err); else printf("Passed!\\n");
  return err ? 1 : 0;
}}
"""


def tcl():
    return f"""open_project hls_proj
set_top mb
add_files kernel.cpp
add_files -tb tb.cpp
open_solution "sol1" -flow_target vitis
set_part {{{PART}}}
create_clock -period {CLOCK} -name default
csim_design
csynth_design
cosim_design -trace_level port
exit
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    only = set(filter(None, re.split("[,:]", a.only)))
    out = Path(a.out)
    cs = [c for c in configs() if not only or c["name"] in only]
    for c in cs:
        d = out / c["name"]
        d.mkdir(parents=True, exist_ok=True)
        (d / "kernel.cpp").write_text(kernel_f(c) if c["fam"] == "f" else kernel_p(c))
        (d / "tb.cpp").write_text(testbench(c))
        (d / "run.tcl").write_text(tcl())
        (d / "config.json").write_text(json.dumps(c, indent=2) + "\n")
    print("\n".join(c["name"] for c in cs))


if __name__ == "__main__":
    main()
