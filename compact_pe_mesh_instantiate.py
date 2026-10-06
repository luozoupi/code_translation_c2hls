"""2-D compact-stream mesh GEMM kernel instantiation from a PeRecipe.

Deterministic emitter for layout=mesh recipes. Uses mesh_pe (not mm_pe),
PE_I x PE_J grid, column-owned B rows, per-PE C output.
"""

from __future__ import annotations

from compact_pe_instantiate import (
    _INTERFACE,
    _check_simd,
    _decl_a_vars,
    _decl_b_vars,
    _dot_partial,
    _indent,
    _pack_write,
    _vec_name,
    emit_pack,
    emit_unpack,
)
from post_flash_pe_recipe import PeRecipe


def emit_mesh_pe_calls(pe_i: int, pe_j: int, indent: str = "    ") -> str:
    lines = []
    for i in range(pe_i):
        for j in range(pe_j):
            lines.append(
                f"{indent}mesh_pe(fifo_A[{i}][{j}], fifo_B[{i}][{j}], "
                f"fifo_B[{i + 1}][{j}], fifo_C[{i}][{j}]);"
            )
    return "\n".join(lines)


def _mesh_kj_loop(simd: int, j_macro: str) -> str:
    vec = _vec_name(simd)
    a_list = ", ".join(f"a{i}" for i in range(simd))
    b_list = ", ".join(f"b{i}" for i in range(simd))
    body = f"""pe_kj: for (int t = 0; t < (K / SIMD) * ({j_macro} / PE_J); ++t) {{
#pragma HLS PIPELINE II=1
    const int k_tile = t / ({j_macro} / PE_J);
    const int jj = t - k_tile * ({j_macro} / PE_J);

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

    if (k_tile == (K / SIMD) - 1) {{
        fifo_C.write(acc);
    }}
}}"""
    return _indent(body, 8)


def _emit_mesh_pe(simd: int, j_macro: str) -> str:
    vec = _vec_name(simd)
    a_decl = _decl_a_vars(simd, init_zero=True)
    kj = _mesh_kj_loop(simd, j_macro)
    body = (
        f"    data_t Crow[{j_macro} / PE_J];\n"
        "#pragma HLS BIND_STORAGE variable=Crow type=ram_2p impl=bram\n"
        "\n"
        "    for (int tile = 0; tile < I / PE_I; ++tile) {\n"
        "        const int i0 = tile * PE_I;\n"
        "        (void)i0;\n"
        f"        {a_decl}\n"
        f"{kj}\n"
        "    }"
    )
    return f"""static void mesh_pe(hls::stream<{vec}> &fifo_A,
                  hls::stream<{vec}> &fifo_B_in,
                  hls::stream<{vec}> &fifo_B_out,
                  hls::stream<data_t> &fifo_C) {{
#pragma HLS INLINE off
{body}
}}"""


def _emit_mesh_load_A(simd: int) -> str:
    vec = _vec_name(simd)
    expr = "A[i0 + i][k0 + {i}]"
    write = _pack_write(simd, "fifo_A[i][j]", expr, indent=20)
    return f"""static void load_A(data_t A[I][K], hls::stream<{vec}> fifo_A[PE_I][PE_J]) {{
#pragma HLS INLINE off
    for (int tile = 0; tile < I / PE_I; ++tile) {{
        const int i0 = tile * PE_I;
        for (int k0 = 0; k0 < K; k0 += SIMD) {{
            for (int i = 0; i < PE_I; ++i) {{
                for (int j = 0; j < PE_J; ++j) {{
#pragma HLS PIPELINE II=1
{write}
                }}
            }}
        }}
    }}
}}"""


def _emit_mesh_load_B(simd: int, j_macro: str) -> str:
    vec = _vec_name(simd)
    expr = f"B[j * ({j_macro} / PE_J) + jj][k0 + {{i}}]"
    write = _pack_write(simd, "fifo_B[0][j]", expr, indent=20)
    return f"""static void load_B(data_t B[J][K], hls::stream<{vec}> fifo_B[PE_I + 1][PE_J]) {{
#pragma HLS INLINE off
    for (int tile = 0; tile < I / PE_I; ++tile) {{
        const int i0 = tile * PE_I;
        (void)i0;
        for (int k0 = 0; k0 < K; k0 += SIMD) {{
            for (int j = 0; j < PE_J; ++j) {{
                for (int jj = 0; jj < {j_macro} / PE_J; ++jj) {{
#pragma HLS PIPELINE II=1
{write}
                }}
            }}
        }}
    }}
}}"""


def _emit_mesh_store_C(j_macro: str) -> str:
    return f"""static void store_C(data_t C[I][J], hls::stream<data_t> fifo_C[PE_I][PE_J]) {{
#pragma HLS INLINE off
    for (int tile = 0; tile < I / PE_I; ++tile) {{
        const int i0 = tile * PE_I;
        for (int i = 0; i < PE_I; ++i) {{
            for (int j = 0; j < PE_J; ++j) {{
                for (int jj = 0; jj < {j_macro} / PE_J; ++jj) {{
#pragma HLS PIPELINE II=1
                    C[i0 + i][j * ({j_macro} / PE_J) + jj] = fifo_C[i][j].read();
                }}
            }}
        }}
    }}
}}"""


def instantiate_mesh(rec: PeRecipe) -> str:
    """Return compact-stream mesh autosa_mm C++ for `rec`. Deterministic; no LLM."""
    pe_i = int(rec.pe_i)
    pe_j = int(rec.pe_j)
    simd = int(rec.simd)
    pack_bits = int(rec.pack_bits)
    j_macro = rec.j_macro
    _check_simd(simd)
    if pack_bits != simd * 32:
        raise ValueError(f"pack_bits={pack_bits} does not match simd={simd} * 32")

    vec = _vec_name(simd)
    calls = emit_mesh_pe_calls(pe_i, pe_j)

    drain_B = f"""static void drain_B(hls::stream<{vec}> fifo_B[PE_I + 1][PE_J]) {{
#pragma HLS INLINE off
    for (int tile = 0; tile < I / PE_I; ++tile) {{
        const int i0 = tile * PE_I;
        (void)i0;
        for (int k0 = 0; k0 < K; k0 += SIMD) {{
            for (int j = 0; j < PE_J; ++j) {{
                for (int jj = 0; jj < {j_macro} / PE_J; ++jj) {{
#pragma HLS PIPELINE II=1
                    (void)fifo_B[PE_I][j].read();
                }}
            }}
        }}
    }}
}}"""

    simd_define = f"#define SIMD {simd}"
    if simd == 8:
        simd_define = "#define SIMD   8"

    top_body = f"""    hls::stream<{vec}> fifo_A[PE_I][PE_J];
    hls::stream<{vec}> fifo_B[PE_I + 1][PE_J];
    hls::stream<data_t> fifo_C[PE_I][PE_J];

#pragma HLS STREAM variable=fifo_A depth=16
#pragma HLS STREAM variable=fifo_B depth=16
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
            "// Generated by compact_pe_mesh_instantiate.py (deterministic; not LLM).",
            '#include "kernel.h"',
            "#include <hls_stream.h>",
            "#include <ap_int.h>",
            "",
            f"#define PE_I {pe_i}",
            f"#define PE_J {pe_j}",
            simd_define,
            "",
            f"typedef ap_uint<{pack_bits}> {vec};",
            "",
            emit_pack(simd),
            "",
            emit_unpack(simd),
            "",
            _emit_mesh_load_A(simd),
            "",
            _emit_mesh_load_B(simd, j_macro),
            "",
            _emit_mesh_pe(simd, j_macro),
            "",
            drain_B,
            "",
            _emit_mesh_store_C(j_macro),
            "",
            'extern "C" void autosa_mm(data_t A[I][K], data_t B[J][K], data_t C[I][J]) {',
            _INTERFACE,
            "",
            top_body,
            "}",
            "",
        ]
    )
