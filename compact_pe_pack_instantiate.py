"""Packed-AXI 2-D mesh GEMM (family B). Top: autosa_mm_pack(A_t16*, B_t16*, C_t16*).

Not AutoSA kernel0. Packed 512-bit DRAM, PE grid PE_I x PE_J, SIMD MAC,
Crow[LAT_J] per PE, one packed C word per PE row.
"""

from __future__ import annotations

from compact_pe_instantiate import (
    _check_simd,
    _decl_a_vars,
    _decl_b_vars,
    _dot_partial,
    _indent,
    _vec_name,
    emit_pack,
    emit_unpack,
)
from post_flash_pe_recipe import PeRecipe

_INTERFACE = """\
#pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem_A max_read_burst_length=64 num_read_outstanding=16
#pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem_B max_read_burst_length=64 num_read_outstanding=16
#pragma HLS INTERFACE m_axi port=C offset=slave bundle=gmem_C max_read_burst_length=64 max_write_burst_length=64 num_write_outstanding=16
#pragma HLS INTERFACE s_axilite port=A bundle=control
#pragma HLS INTERFACE s_axilite port=B bundle=control
#pragma HLS INTERFACE s_axilite port=C bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control"""

AXI_LANES = 16


def emit_pack_pe_calls(pe_i: int, pe_j: int, indent: str = "    ") -> str:
    lines = []
    for i in range(pe_i):
        for j in range(pe_j):
            lines.append(
                f"{indent}pack_pe(fifo_A[{i}][{j}], fifo_B[{i}][{j}], "
                f"fifo_B[{i + 1}][{j}], fifo_C[{i}][{j}]);"
            )
    return "\n".join(lines)


def emit_pack_header() -> str:
    return """\
#ifndef KERNEL_H
#define KERNEL_H
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <ap_int.h>

typedef float data_t;
#define I 64
#define J 64
#define K 64
typedef ap_uint<512> A_t16;
typedef ap_uint<512> B_t16;
typedef ap_uint<512> C_t16;

#ifdef __cplusplus
extern "C" void autosa_mm_pack(A_t16 *A, B_t16 *B, C_t16 *C);
#endif
#endif
"""


def emit_pack_testbench() -> str:
    return r'''#include "kernel.h"

static void pack16_word(A_t16 &w, data_t v[16]) {
    for (int n = 0; n < 16; ++n) {
        union { unsigned u; float f; } c;
        c.f = (float)v[n];
        w.range(32 * n + 31, 32 * n) = c.u;
    }
}

static void unpack16_word(A_t16 w, data_t v[16]) {
    for (int n = 0; n < 16; ++n) {
        union { unsigned u; float f; } c;
        c.u = (unsigned)w.range(32 * n + 31, 32 * n);
        v[n] = (data_t)c.f;
    }
}

int main() {
    data_t Af[I][K], Bf[J][K], Cf[I][J], Cg[I][J];
    A_t16 A[I * K / 16];
    B_t16 B[J * K / 16];
    C_t16 C[I * J / 16];

    for (int i = 0; i < I; ++i)
        for (int k = 0; k < K; ++k)
            Af[i][k] = (data_t)rand() / RAND_MAX;
    for (int j = 0; j < J; ++j)
        for (int k = 0; k < K; ++k)
            Bf[j][k] = (data_t)rand() / RAND_MAX;

    for (int i = 0; i < I; ++i) {
        for (int k0 = 0; k0 < K; k0 += 16) {
            data_t v[16];
            for (int n = 0; n < 16; ++n)
                v[n] = Af[i][k0 + n];
            pack16_word(A[i * (K / 16) + k0 / 16], v);
        }
    }
    for (int j = 0; j < J; ++j) {
        for (int k0 = 0; k0 < K; k0 += 16) {
            data_t v[16];
            for (int n = 0; n < 16; ++n)
                v[n] = Bf[j][k0 + n];
            pack16_word(B[j * (K / 16) + k0 / 16], v);
        }
    }

    autosa_mm_pack(A, B, C);

    for (int i = 0; i < I; ++i) {
        for (int j0 = 0; j0 < J; j0 += 16) {
            data_t v[16];
            unpack16_word(C[i * (J / 16) + j0 / 16], v);
            for (int n = 0; n < 16; ++n)
                Cf[i][j0 + n] = v[n];
        }
    }

    int err = 0;
    for (int i = 0; i < I; ++i) {
        for (int j = 0; j < J; ++j) {
            Cg[i][j] = 0;
            for (int k = 0; k < K; ++k)
                Cg[i][j] += Af[i][k] * Bf[j][k];
            if (fabs((float)Cg[i][j] - (float)Cf[i][j]) > 0.001f)
                err++;
        }
    }
    if (err)
        printf("Failed with %d errors!\n", err);
    else
        printf("Passed!\n");
    return err ? 1 : 0;
}
'''


def _emit_axi_packers() -> str:
    return """\
static A_t16 pack16(data_t v[16]) {
#pragma HLS INLINE
    A_t16 w;
    for (int n = 0; n < 16; ++n) {
#pragma HLS UNROLL
        union { unsigned u; float f; } c;
        c.f = (float)v[n];
        w.range(32 * n + 31, 32 * n) = c.u;
    }
    return w;
}

static void unpack16(A_t16 w, data_t v[16]) {
#pragma HLS INLINE
    for (int n = 0; n < 16; ++n) {
#pragma HLS UNROLL
        union { unsigned u; float f; } c;
        c.u = (unsigned)w.range(32 * n + 31, 32 * n);
        v[n] = (data_t)c.f;
    }
}"""


def _emit_pack_pe(simd: int) -> str:
    vec = _vec_name(simd)
    a_list = ", ".join(f"a{i}" for i in range(simd))
    b_list = ", ".join(f"b{i}" for i in range(simd))
    a_decl = _decl_a_vars(simd, init_zero=True)
    kj = _indent(
        f"""pe_kj: for (int t = 0; t < (K / SIMD) * LAT_J; ++t) {{
#pragma HLS PIPELINE II=1
    const int k_tile = t / LAT_J;
    const int jj = t - k_tile * LAT_J;
    if (jj == 0) {{
        unpack{simd}(fifo_A.read(), {a_list});
    }}
    {vec} bw = fifo_B_in.read();
    fifo_B_out.write(bw);
    {_decl_b_vars(simd)}
    unpack{simd}(bw, {b_list});
{_dot_partial(simd, 4)}
    data_t acc = (k_tile == 0) ? partial : (Crow[jj] + partial);
    Crow[jj] = acc;
}}""",
        8,
    )
    return f"""static void pack_pe(hls::stream<{vec}> &fifo_A,
                   hls::stream<{vec}> &fifo_B_in,
                   hls::stream<{vec}> &fifo_B_out,
                   hls::stream<A_t16> &fifo_C) {{
#pragma HLS INLINE off
    data_t Crow[LAT_J];
#pragma HLS BIND_STORAGE variable=Crow type=ram_2p impl=bram

    for (int ii = 0; ii < LAT_I; ++ii) {{
        {a_decl}
{kj}
        data_t cv[16];
        for (int n = 0; n < 16; ++n) {{
#pragma HLS UNROLL
            cv[n] = 0;
        }}
        for (int jj = 0; jj < LAT_J; ++jj) {{
#pragma HLS PIPELINE II=1
            cv[jj] = Crow[jj];
        }}
        fifo_C.write(pack16(cv));
    }}
}}"""


def _emit_load_A(simd: int) -> str:
    vec = _vec_name(simd)
    chunks = AXI_LANES // simd
    slice_packs = []
    for s in range(chunks):
        args = ", ".join(f"av[{s * simd + t}]" for t in range(simd))
        slice_packs.append(
            f"                fifo_A[pi][pj].write(pack{simd}({args}));"
        )
    inner = "\n".join(slice_packs)
    return f"""static void load_A(A_t16 *A, hls::stream<{vec}> fifo_A[PE_I][PE_J]) {{
#pragma HLS INLINE off
    for (int i = 0; i < I; ++i) {{
        const int pi = i / LAT_I;
        for (int k16 = 0; k16 < K; k16 += 16) {{
            A_t16 w = A[i * (K / 16) + k16 / 16];
            data_t av[16];
            unpack16(w, av);
            for (int pj = 0; pj < PE_J; ++pj) {{
#pragma HLS UNROLL
{inner}
            }}
        }}
    }}
}}"""


def _emit_load_B(simd: int) -> str:
    vec = _vec_name(simd)
    chunks = AXI_LANES // simd
    args = ", ".join(f"bv[s * {simd} + {t}]" for t in range(simd))
    return f"""static void load_B(B_t16 *B, hls::stream<{vec}> fifo_B[PE_I + 1][PE_J]) {{
#pragma HLS INLINE off
    B_t16 Bmem[J * K / 16];
#pragma HLS BIND_STORAGE variable=Bmem type=ram_2p impl=bram
    for (int n = 0; n < J * K / 16; ++n) {{
#pragma HLS PIPELINE II=1
        Bmem[n] = B[n];
    }}
    for (int ii = 0; ii < LAT_I; ++ii) {{
        (void)ii;
        for (int k16 = 0; k16 < K; k16 += 16) {{
            for (int s = 0; s < {chunks}; ++s) {{
                for (int pj = 0; pj < PE_J; ++pj) {{
                    for (int jj = 0; jj < LAT_J; ++jj) {{
#pragma HLS PIPELINE II=1
                        const int j = pj * LAT_J + jj;
                        data_t bv[16];
                        unpack16(Bmem[j * (K / 16) + k16 / 16], bv);
                        fifo_B[0][pj].write(pack{simd}({args}));
                    }}
                }}
            }}
        }}
    }}
}}"""


def _emit_drain_B(simd: int) -> str:
    vec = _vec_name(simd)
    chunks = AXI_LANES // simd
    return f"""static void drain_B(hls::stream<{vec}> fifo_B[PE_I + 1][PE_J]) {{
#pragma HLS INLINE off
    for (int ii = 0; ii < LAT_I; ++ii) {{
        (void)ii;
        for (int k16 = 0; k16 < K; k16 += 16) {{
            for (int s = 0; s < {chunks}; ++s) {{
                for (int pj = 0; pj < PE_J; ++pj) {{
                    for (int jj = 0; jj < LAT_J; ++jj) {{
#pragma HLS PIPELINE II=1
                        (void)fifo_B[PE_I][pj].read();
                    }}
                }}
            }}
        }}
    }}
}}"""


def _emit_store_C() -> str:
    return """static void store_C(C_t16 *C, hls::stream<A_t16> fifo_C[PE_I][PE_J]) {
#pragma HLS INLINE off
    for (int i = 0; i < I; ++i) {
        const int pi = i / LAT_I;
        data_t row[J];
        for (int pj = 0; pj < PE_J; ++pj) {
            data_t cv[16];
            unpack16(fifo_C[pi][pj].read(), cv);
            for (int jj = 0; jj < LAT_J; ++jj) {
#pragma HLS UNROLL
                row[pj * LAT_J + jj] = cv[jj];
            }
        }
        for (int j0 = 0; j0 < J; j0 += 16) {
#pragma HLS PIPELINE II=1
            data_t v[16];
            for (int n = 0; n < 16; ++n) {
#pragma HLS UNROLL
                v[n] = row[j0 + n];
            }
            C[i * (J / 16) + j0 / 16] = pack16(v);
        }
    }
}"""


def instantiate_pack(rec: PeRecipe) -> str:
    """Return packed-AXI autosa_mm_pack C++ for `rec`."""
    pe_i = int(rec.pe_i)
    pe_j = int(rec.pe_j)
    simd = int(rec.simd)
    _check_simd(simd)
    if AXI_LANES % simd:
        raise ValueError(f"simd={simd} does not divide AXI 16-lane words")
    vec = _vec_name(simd)
    calls = emit_pack_pe_calls(pe_i, pe_j)

    top_body = f"""    hls::stream<{vec}> fifo_A[PE_I][PE_J];
    hls::stream<{vec}> fifo_B[PE_I + 1][PE_J];
    hls::stream<A_t16> fifo_C[PE_I][PE_J];

#pragma HLS STREAM variable=fifo_A depth=256
#pragma HLS STREAM variable=fifo_B depth=256
#pragma HLS STREAM variable=fifo_C depth=64
#pragma HLS ARRAY_PARTITION variable=fifo_A complete
#pragma HLS ARRAY_PARTITION variable=fifo_B complete
#pragma HLS ARRAY_PARTITION variable=fifo_C complete

#pragma HLS DATAFLOW

    load_A(A, fifo_A);
    load_B(B, fifo_B);

{calls}

    drain_B(fifo_B);
    store_C(C, fifo_C);"""

    return "\n".join(
        [
            "// Generated by compact_pe_pack_instantiate.py (deterministic; not LLM, not kernel0).",
            '#include "kernel.h"',
            "#include <hls_stream.h>",
            "#include <ap_int.h>",
            "",
            f"#define PE_I {pe_i}",
            f"#define PE_J {pe_j}",
            f"#define LAT_I (I / PE_I)",
            f"#define LAT_J (J / PE_J)",
            f"#define SIMD {simd}",
            "",
            "typedef ap_uint<512> axi512_t;",
            f"typedef ap_uint<{simd * 32}> {vec};",
            "",
            emit_pack(simd),
            "",
            emit_unpack(simd),
            "",
            _emit_axi_packers(),
            "",
            _emit_load_A(simd),
            "",
            _emit_load_B(simd),
            "",
            _emit_pack_pe(simd),
            "",
            _emit_drain_B(simd),
            "",
            _emit_store_C(),
            "",
            'extern "C" void autosa_mm_pack(A_t16 *A, B_t16 *B, C_t16 *C) {',
            _INTERFACE,
            "",
            top_body,
            "}",
            "",
        ]
    )
