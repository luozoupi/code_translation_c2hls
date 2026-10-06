"""Tiled IO-mesh GEMM (family C). Top: autosa_mm_pack(A_t16*, B_t16*, C_t16*).

io4 = paper Design 4 (i,j space): local Crow, B shift along I, K/J tiles,
L2 B ping/pong. io5 = paper Design 5 (i,k space): C systolic along K,
pe_j = PEs along K.

Not AutoSA kernel0. Deterministic; no LLM.
"""

from __future__ import annotations

from compact_pe_instantiate import (
    _check_simd,
    _decl_a_vars,
    _decl_b_vars,
    _dot_partial,
    _vec_name,
    emit_pack,
    emit_unpack,
)
from compact_pe_pack_instantiate import AXI_LANES, _INTERFACE, _emit_axi_packers
from post_flash_pe_recipe import PeRecipe

STREAM_DEPTH = 1024


def emit_io4_pe_calls(pe_i: int, pe_j: int, indent: str = "    ") -> str:
    lines = []
    for i in range(pe_i):
        for j in range(pe_j):
            lines.append(
                f"{indent}io_pe(fifo_A[{i}][{j}], fifo_B[{i}][{j}], "
                f"fifo_B[{i + 1}][{j}], fifo_C[{i}][{j}]);"
            )
    return "\n".join(lines)


def emit_io5_pe_calls(pe_i: int, pe_j: int, indent: str = "    ") -> str:
    lines = []
    for i in range(pe_i):
        for j in range(pe_j):
            lines.append(
                f"{indent}io_pe(fifo_A[{i}][{j}], fifo_B[{i}][{j}], "
                f"fifo_B[{i + 1}][{j}], fifo_C_k[{i}][{j}], "
                f"fifo_C_k[{i}][{j + 1}], fifo_C_store[{i}][{j}], {j});"
            )
    return "\n".join(lines)


def _simd_pack_args(simd: int, arr: str, offset_expr: str) -> str:
    return ", ".join(f"{arr}[({offset_expr}) + {t}]" for t in range(simd))


def _emit_io4_load_A(simd: int) -> str:
    vec = _vec_name(simd)
    chunks = AXI_LANES // simd
    inner_lines = []
    for s in range(chunks):
        args = _simd_pack_args(simd, "local_A", str(s * simd))
        inner_lines.append(
            f"                    fifo_A[pi][pj].write(pack{simd}({args}));"
        )
    inner = "\n".join(inner_lines)
    return f"""static void load_A(A_t16 *A, hls::stream<{vec}> fifo_A[PE_I][PE_J]) {{
#pragma HLS INLINE off
    for (int t_j = 0; t_j < N_J; ++t_j) {{
        (void)t_j;
        for (int t_k = 0; t_k < N_K; ++t_k) {{
            for (int i = 0; i < I; ++i) {{
                const int pi = i / LAT_I;
                for (int k16 = 0; k16 < K_PART; k16 += 16) {{
                    A_t16 w = A[i * (K / 16) + (t_k * K_PART + k16) / 16];
                    data_t local_A[16];
                    unpack16(w, local_A);
                    for (int pj = 0; pj < PE_J; ++pj) {{
#pragma HLS UNROLL
{inner}
                    }}
                }}
            }}
        }}
    }}
}}"""


def _emit_io4_load_B(simd: int) -> str:
    vec = _vec_name(simd)
    chunks = AXI_LANES // simd
    args = _simd_pack_args(simd, "bv", f"s * {simd}")
    return f"""static void load_B(B_t16 *B, hls::stream<{vec}> fifo_B[PE_I + 1][PE_J]) {{
#pragma HLS INLINE off
    B_t16 B_ping[J_PART * K_PART / 16];
    B_t16 B_pong[J_PART * K_PART / 16];
#pragma HLS BIND_STORAGE variable=B_ping type=ram_2p impl=bram
#pragma HLS BIND_STORAGE variable=B_pong type=ram_2p impl=bram
    int tile_id = 0;
    for (int t_j = 0; t_j < N_J; ++t_j) {{
        for (int t_k = 0; t_k < N_K; ++t_k) {{
            const int use_ping = ((tile_id & 1) == 0);
            for (int j_loc = 0; j_loc < J_PART; ++j_loc) {{
                for (int k16 = 0; k16 < K_PART; k16 += 16) {{
#pragma HLS PIPELINE II=1
                    const int src = (t_j * J_PART + j_loc) * (K / 16)
                        + (t_k * K_PART + k16) / 16;
                    const int dst = j_loc * (K_PART / 16) + k16 / 16;
                    if (use_ping)
                        B_ping[dst] = B[src];
                    else
                        B_pong[dst] = B[src];
                }}
            }}
            for (int ii = 0; ii < LAT_I; ++ii) {{
                (void)ii;
                for (int k16 = 0; k16 < K_PART; k16 += 16) {{
                    for (int s = 0; s < {chunks}; ++s) {{
                        for (int jj = 0; jj < LAT_J; ++jj) {{
#pragma HLS PIPELINE II=1
                            for (int pj = 0; pj < PE_J; ++pj) {{
#pragma HLS UNROLL
                                const int j_loc = pj * LAT_J + jj;
                                const int dst = j_loc * (K_PART / 16) + k16 / 16;
                                data_t bv[16];
                                unpack16(use_ping ? B_ping[dst] : B_pong[dst], bv);
                                fifo_B[0][pj].write(pack{simd}({args}));
                            }}
                        }}
                    }}
                }}
            }}
            tile_id++;
        }}
    }}
}}"""


def _emit_io4_pe(simd: int) -> str:
    vec = _vec_name(simd)
    a_list = ", ".join(f"a{i}" for i in range(simd))
    b_list = ", ".join(f"b{i}" for i in range(simd))
    a_decl = _decl_a_vars(simd, init_zero=True)
    local_store = "\n".join(
        f"                        local_A[k_tile][{i}] = a{i};" for i in range(simd)
    )
    return f"""static void io_pe(hls::stream<{vec}> &fifo_A,
                  hls::stream<{vec}> &fifo_B_in,
                  hls::stream<{vec}> &fifo_B_out,
                  hls::stream<A_t16> &fifo_C) {{
#pragma HLS INLINE off
    data_t Crow[LAT_I][LAT_J];
#pragma HLS BIND_STORAGE variable=Crow type=ram_2p impl=bram
    data_t local_A[K_PART / SIMD][SIMD];
#pragma HLS ARRAY_PARTITION variable=local_A complete dim=2

    for (int t_j = 0; t_j < N_J; ++t_j) {{
        (void)t_j;
        for (int t_k = 0; t_k < N_K; ++t_k) {{
            for (int ii = 0; ii < LAT_I; ++ii) {{
                {a_decl}
                pe_kj: for (int t = 0; t < (K_PART / SIMD) * LAT_J; ++t) {{
#pragma HLS PIPELINE II=1
                    const int k_tile = t / LAT_J;
                    const int jj = t - k_tile * LAT_J;
                    if (jj == 0) {{
                        unpack{simd}(fifo_A.read(), {a_list});
{local_store}
                    }}
                    {vec} bw = fifo_B_in.read();
                    fifo_B_out.write(bw);
                    {_decl_b_vars(simd)}
                    unpack{simd}(bw, {b_list});
{_dot_partial(simd, 20)}
                    data_t acc = (t_k == 0 && k_tile == 0)
                        ? partial : (Crow[ii][jj] + partial);
                    Crow[ii][jj] = acc;
                }}
                if (t_k == N_K - 1) {{
                    data_t cv[16];
                    for (int n = 0; n < 16; ++n) {{
#pragma HLS UNROLL
                        cv[n] = 0;
                    }}
                    for (int jj = 0; jj < LAT_J; ++jj) {{
#pragma HLS UNROLL
                        cv[jj] = Crow[ii][jj];
                    }}
                    fifo_C.write(pack16(cv));
                }}
            }}
        }}
    }}
}}"""


def _emit_io4_drain_B(simd: int) -> str:
    vec = _vec_name(simd)
    chunks = AXI_LANES // simd
    return f"""static void drain_B(hls::stream<{vec}> fifo_B[PE_I + 1][PE_J]) {{
#pragma HLS INLINE off
    for (int t_j = 0; t_j < N_J; ++t_j) {{
        (void)t_j;
        for (int t_k = 0; t_k < N_K; ++t_k) {{
            (void)t_k;
            for (int ii = 0; ii < LAT_I; ++ii) {{
                (void)ii;
                for (int k16 = 0; k16 < K_PART; k16 += 16) {{
                    for (int s = 0; s < {chunks}; ++s) {{
                        for (int jj = 0; jj < LAT_J; ++jj) {{
#pragma HLS PIPELINE II=1
                            for (int pj = 0; pj < PE_J; ++pj) {{
#pragma HLS UNROLL
                                (void)fifo_B[PE_I][pj].read();
                            }}
                        }}
                    }}
                }}
            }}
        }}
    }}
}}"""


def _emit_io4_store_C() -> str:
    return """static void store_C(C_t16 *C, hls::stream<A_t16> fifo_C[PE_I][PE_J]) {
#pragma HLS INLINE off
    for (int t_j = 0; t_j < N_J; ++t_j) {
        for (int i = 0; i < I; ++i) {
            const int pi = i / LAT_I;
            data_t row[J_PART];
            for (int pj = 0; pj < PE_J; ++pj) {
                data_t cv[16];
                unpack16(fifo_C[pi][pj].read(), cv);
                for (int jj = 0; jj < LAT_J; ++jj) {
#pragma HLS UNROLL
                    row[pj * LAT_J + jj] = cv[jj];
                }
            }
            for (int j0 = 0; j0 < J_PART; j0 += 16) {
#pragma HLS PIPELINE II=1
                data_t v[16];
                for (int n = 0; n < 16; ++n) {
#pragma HLS UNROLL
                    v[n] = row[j0 + n];
                }
                C[i * (J / 16) + (t_j * J_PART + j0) / 16] = pack16(v);
            }
        }
    }
}"""


def _instantiate_io4(rec: PeRecipe) -> str:
    pe_i = int(rec.pe_i)
    pe_j = int(rec.pe_j)
    simd = int(rec.simd)
    _check_simd(simd)
    if AXI_LANES % simd:
        raise ValueError(f"simd={simd} does not divide AXI 16-lane words")
    vec = _vec_name(simd)
    calls = emit_io4_pe_calls(pe_i, pe_j)
    top_body = f"""    hls::stream<{vec}> fifo_A[PE_I][PE_J];
    hls::stream<{vec}> fifo_B[PE_I + 1][PE_J];
    hls::stream<A_t16> fifo_C[PE_I][PE_J];

#pragma HLS STREAM variable=fifo_A depth={STREAM_DEPTH}
#pragma HLS STREAM variable=fifo_B depth={STREAM_DEPTH}
#pragma HLS STREAM variable=fifo_C depth={STREAM_DEPTH}
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
            "// Generated by compact_pe_io_instantiate.py (deterministic; not LLM, not kernel0).",
            '#include "kernel.h"',
            "#include <hls_stream.h>",
            "#include <ap_int.h>",
            "",
            f"#define PE_I {pe_i}",
            f"#define PE_J {pe_j}",
            f"#define SIMD {simd}",
            f"#define K_PART {int(rec.k_part)}",
            f"#define J_PART {int(rec.j_part)}",
            f"#define LAT_I {int(rec.lat_i)}",
            f"#define LAT_J {int(rec.lat_j)}",
            "#define N_J (J / J_PART)",
            "#define N_K (K / K_PART)",
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
            _emit_io4_load_A(simd),
            "",
            _emit_io4_load_B(simd),
            "",
            _emit_io4_pe(simd),
            "",
            _emit_io4_drain_B(simd),
            "",
            _emit_io4_store_C(),
            "",
            'extern "C" void autosa_mm_pack(A_t16 *A, B_t16 *B, C_t16 *C) {',
            _INTERFACE,
            "",
            top_body,
            "}",
            "",
        ]
    )


def _emit_io5_load_A(simd: int) -> str:
    vec = _vec_name(simd)
    args = _simd_pack_args(simd, "av", "off")
    return f"""static void load_A(A_t16 *A, hls::stream<{vec}> fifo_A[PE_I][PE_J]) {{
#pragma HLS INLINE off
    for (int t_j = 0; t_j < N_J; ++t_j) {{
        (void)t_j;
        for (int t_k = 0; t_k < N_K; ++t_k) {{
            for (int i = 0; i < I; ++i) {{
                const int pi = i / LAT_I;
                for (int pk = 0; pk < PE_J; ++pk) {{
#pragma HLS UNROLL
                    const int k0 = t_k * K_PART + pk * SIMD;
                    A_t16 w = A[i * (K / 16) + k0 / 16];
                    data_t av[16];
                    data_t local_A[16];
                    unpack16(w, av);
                    for (int n = 0; n < 16; ++n) {{
#pragma HLS UNROLL
                        local_A[n] = av[n];
                    }}
                    const int off = k0 % 16;
                    fifo_A[pi][pk].write(pack{simd}({args}));
                }}
            }}
        }}
    }}
}}"""


def _emit_io5_load_B(simd: int) -> str:
    vec = _vec_name(simd)
    args = _simd_pack_args(simd, "bv", "off")
    return f"""static void load_B(B_t16 *B, hls::stream<{vec}> fifo_B[PE_I + 1][PE_J]) {{
#pragma HLS INLINE off
    B_t16 B_ping[J_PART * K_PART / 16];
    B_t16 B_pong[J_PART * K_PART / 16];
#pragma HLS BIND_STORAGE variable=B_ping type=ram_2p impl=bram
#pragma HLS BIND_STORAGE variable=B_pong type=ram_2p impl=bram
    int tile_id = 0;
    for (int t_j = 0; t_j < N_J; ++t_j) {{
        for (int t_k = 0; t_k < N_K; ++t_k) {{
            const int use_ping = ((tile_id & 1) == 0);
            for (int j_loc = 0; j_loc < J_PART; ++j_loc) {{
                for (int k16 = 0; k16 < K_PART; k16 += 16) {{
#pragma HLS PIPELINE II=1
                    const int src = (t_j * J_PART + j_loc) * (K / 16)
                        + (t_k * K_PART + k16) / 16;
                    const int dst = j_loc * (K_PART / 16) + k16 / 16;
                    if (use_ping)
                        B_ping[dst] = B[src];
                    else
                        B_pong[dst] = B[src];
                }}
            }}
            for (int j_loc = 0; j_loc < J_PART; ++j_loc) {{
                for (int ii = 0; ii < LAT_I; ++ii) {{
                    (void)ii;
#pragma HLS PIPELINE II=1
                    for (int pk = 0; pk < PE_J; ++pk) {{
#pragma HLS UNROLL
                        const int k0 = pk * SIMD;
                        const int dst = j_loc * (K_PART / 16) + k0 / 16;
                        const int off = k0 % 16;
                        data_t bv[16];
                        unpack16(use_ping ? B_ping[dst] : B_pong[dst], bv);
                        fifo_B[0][pk].write(pack{simd}({args}));
                    }}
                }}
            }}
            tile_id++;
        }}
    }}
}}"""


def _emit_io5_pe(simd: int) -> str:
    vec = _vec_name(simd)
    a_list = ", ".join(f"a{i}" for i in range(simd))
    b_list = ", ".join(f"b{i}" for i in range(simd))
    a_decl = _decl_a_vars(simd, init_zero=True)
    local_store = "\n".join(
        f"                    local_A[ii][{i}] = a{i};" for i in range(simd)
    )
    local_reload = "\n".join(
        f"                    a{i} = local_A[ii][{i}];" for i in range(simd)
    )
    return f"""static void io_pe(hls::stream<{vec}> &fifo_A,
                  hls::stream<{vec}> &fifo_B_in,
                  hls::stream<{vec}> &fifo_B_out,
                  hls::stream<data_t> &fifo_C_in,
                  hls::stream<data_t> &fifo_C_out,
                  hls::stream<A_t16> &fifo_C_store,
                  int pe_k) {{
#pragma HLS INLINE off
    data_t Cmem[LAT_I][J_PART];
#pragma HLS BIND_STORAGE variable=Cmem type=ram_2p impl=bram
    data_t local_A[LAT_I][SIMD];
#pragma HLS ARRAY_PARTITION variable=local_A complete dim=2

    for (int t_j = 0; t_j < N_J; ++t_j) {{
        (void)t_j;
        for (int t_k = 0; t_k < N_K; ++t_k) {{
            {a_decl}
            pe_kj: for (int t = 0; t < J_PART * LAT_I; ++t) {{
#pragma HLS PIPELINE II=1
                const int ii = t % LAT_I;
                const int j_loc = t / LAT_I;
                const int j0 = j_loc / LAT_J;
                const int jj = j_loc - j0 * LAT_J;
                if (j0 == 0 && jj == 0) {{
                    unpack{simd}(fifo_A.read(), {a_list});
{local_store}
                }} else {{
{local_reload}
                }}
                {vec} bw = fifo_B_in.read();
                fifo_B_out.write(bw);
                {_decl_b_vars(simd)}
                unpack{simd}(bw, {b_list});
{_dot_partial(simd, 16)}
                data_t cin = fifo_C_in.read();
                data_t acc = cin + partial;
                fifo_C_out.write(acc);
                data_t prev = (t_k == 0) ? (data_t)0 : Cmem[ii][j_loc];
                Cmem[ii][j_loc] = prev + acc;
            }}
        }}
        for (int ii = 0; ii < LAT_I; ++ii) {{
            for (int j0 = 0; j0 < J_PART; j0 += 16) {{
#pragma HLS PIPELINE II=1
                data_t cv[16];
                for (int n = 0; n < 16; ++n) {{
#pragma HLS UNROLL
                    cv[n] = Cmem[ii][j0 + n];
                }}
                if (pe_k == PE_J - 1)
                    fifo_C_store.write(pack16(cv));
            }}
        }}
    }}
}}"""


def _emit_io5_feed_C() -> str:
    return """static void feed_C_zero(hls::stream<data_t> fifo_C_k[PE_I][PE_J + 1]) {
#pragma HLS INLINE off
    for (int t_j = 0; t_j < N_J; ++t_j) {
        (void)t_j;
        for (int t_k = 0; t_k < N_K; ++t_k) {
            (void)t_k;
            for (int t = 0; t < J_PART * LAT_I; ++t) {
#pragma HLS PIPELINE II=1
                (void)t;
                for (int pi = 0; pi < PE_I; ++pi) {
#pragma HLS UNROLL
                    fifo_C_k[pi][0].write((data_t)0);
                }
            }
        }
    }
}"""


def _emit_io5_drain_C() -> str:
    return """static void drain_C_k(hls::stream<data_t> fifo_C_k[PE_I][PE_J + 1]) {
#pragma HLS INLINE off
    for (int t_j = 0; t_j < N_J; ++t_j) {
        (void)t_j;
        for (int t_k = 0; t_k < N_K; ++t_k) {
            (void)t_k;
            for (int t = 0; t < J_PART * LAT_I; ++t) {
#pragma HLS PIPELINE II=1
                (void)t;
                for (int pi = 0; pi < PE_I; ++pi) {
#pragma HLS UNROLL
                    (void)fifo_C_k[pi][PE_J].read();
                }
            }
        }
    }
}"""


def _emit_io5_drain_B(simd: int) -> str:
    vec = _vec_name(simd)
    return f"""static void drain_B(hls::stream<{vec}> fifo_B[PE_I + 1][PE_J]) {{
#pragma HLS INLINE off
    for (int t_j = 0; t_j < N_J; ++t_j) {{
        (void)t_j;
        for (int t_k = 0; t_k < N_K; ++t_k) {{
            (void)t_k;
            for (int j_loc = 0; j_loc < J_PART; ++j_loc) {{
                for (int ii = 0; ii < LAT_I; ++ii) {{
                    (void)ii;
#pragma HLS PIPELINE II=1
                    for (int pk = 0; pk < PE_J; ++pk) {{
#pragma HLS UNROLL
                        (void)fifo_B[PE_I][pk].read();
                    }}
                }}
            }}
        }}
    }}
}}"""


def _emit_io5_store_C() -> str:
    return """static void store_C(C_t16 *C, hls::stream<A_t16> fifo_C_store[PE_I][PE_J]) {
#pragma HLS INLINE off
    for (int t_j = 0; t_j < N_J; ++t_j) {
        for (int i = 0; i < I; ++i) {
            const int pi = i / LAT_I;
            for (int j0 = 0; j0 < J_PART; j0 += 16) {
#pragma HLS PIPELINE II=1
                C[i * (J / 16) + (t_j * J_PART + j0) / 16] =
                    fifo_C_store[pi][PE_J - 1].read();
            }
        }
    }
}"""


def _instantiate_io5(rec: PeRecipe) -> str:
    pe_i = int(rec.pe_i)
    pe_j = int(rec.pe_j)
    simd = int(rec.simd)
    _check_simd(simd)
    if AXI_LANES % simd:
        raise ValueError(f"simd={simd} does not divide AXI 16-lane words")
    vec = _vec_name(simd)
    calls = emit_io5_pe_calls(pe_i, pe_j)
    top_body = f"""    hls::stream<{vec}> fifo_A[PE_I][PE_J];
    hls::stream<{vec}> fifo_B[PE_I + 1][PE_J];
    hls::stream<data_t> fifo_C_k[PE_I][PE_J + 1];
    hls::stream<A_t16> fifo_C_store[PE_I][PE_J];

#pragma HLS STREAM variable=fifo_A depth={STREAM_DEPTH}
#pragma HLS STREAM variable=fifo_B depth={STREAM_DEPTH}
#pragma HLS STREAM variable=fifo_C_k depth={STREAM_DEPTH}
#pragma HLS STREAM variable=fifo_C_store depth={STREAM_DEPTH}
#pragma HLS ARRAY_PARTITION variable=fifo_A complete
#pragma HLS ARRAY_PARTITION variable=fifo_B complete
#pragma HLS ARRAY_PARTITION variable=fifo_C_k complete
#pragma HLS ARRAY_PARTITION variable=fifo_C_store complete

#pragma HLS DATAFLOW

    load_A(A, fifo_A);
    load_B(B, fifo_B);
    feed_C_zero(fifo_C_k);

{calls}

    drain_B(fifo_B);
    drain_C_k(fifo_C_k);
    store_C(C, fifo_C_store);"""
    return "\n".join(
        [
            "// Generated by compact_pe_io_instantiate.py (deterministic; not LLM, not kernel0).",
            '#include "kernel.h"',
            "#include <hls_stream.h>",
            "#include <ap_int.h>",
            "",
            f"#define PE_I {pe_i}",
            f"#define PE_J {pe_j}",
            f"#define SIMD {simd}",
            f"#define K_PART {int(rec.k_part)}",
            f"#define J_PART {int(rec.j_part)}",
            f"#define LAT_I {int(rec.lat_i)}",
            f"#define LAT_J {int(rec.lat_j)}",
            "#define N_J (J / J_PART)",
            "#define N_K (K / K_PART)",
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
            _emit_io5_load_A(simd),
            "",
            _emit_io5_load_B(simd),
            "",
            _emit_io5_feed_C(),
            "",
            _emit_io5_pe(simd),
            "",
            _emit_io5_drain_B(simd),
            "",
            _emit_io5_drain_C(),
            "",
            _emit_io5_store_C(),
            "",
            'extern "C" void autosa_mm_pack(A_t16 *A, B_t16 *B, C_t16 *C) {',
            _INTERFACE,
            "",
            top_body,
            "}",
            "",
        ]
    )


def instantiate_io(rec: PeRecipe) -> str:
    """Return compact io4/io5 autosa_mm_pack C++ for `rec`."""
    layout = rec.layout
    if layout == "io4":
        return _instantiate_io4(rec)
    if layout == "io5":
        return _instantiate_io5(rec)
    raise ValueError(f"unsupported io layout={layout!r}")
