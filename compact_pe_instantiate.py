"""Deterministic compact-stream GEMM kernel instantiation from a PeRecipe.

No LLM. Emits HLS C++ parameterized by PE_NUM, SIMD, pack width, and
i-tile placement (inside_tasks vs around_dataflow). The checked-in 32x8
reference is:

    hls_full_optimization_skills_schema_1_1_package/compact_pe_stream_skeleton.cpp
"""

from __future__ import annotations

from post_flash_pe_recipe import PeRecipe

_SUPPORTED_SIMD = (2, 4, 8)

_INTERFACE = """\
#pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem0 max_read_burst_length=64 num_read_outstanding=16
#pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem1 max_read_burst_length=64 num_read_outstanding=16
#pragma HLS INTERFACE m_axi port=C offset=slave bundle=gmem2 max_read_burst_length=64 max_write_burst_length=64 num_read_outstanding=16 num_write_outstanding=16
#pragma HLS INTERFACE s_axilite port=A bundle=control
#pragma HLS INTERFACE s_axilite port=B bundle=control
#pragma HLS INTERFACE s_axilite port=C bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control"""

# i_tiles==1 must never emit this exact header (16x4 test).
_FORBIDDEN_I0_WRAP = "for (int i0 = 0; i0 < I; i0 += PE_NUM)"


def _vec_name(simd: int) -> str:
    return f"vec{simd}_bits"


def _check_simd(simd: int) -> None:
    if simd not in _SUPPORTED_SIMD:
        raise ValueError(f"unsupported simd={simd}; expected one of {_SUPPORTED_SIMD}")


def _indent(text: str, spaces: int) -> str:
    pad = " " * spaces
    out = []
    for line in text.splitlines():
        if not line or line.lstrip().startswith("#pragma"):
            out.append(line)
        else:
            out.append(pad + line)
    return "\n".join(out)


def emit_pack(simd: int) -> str:
    """Emit packN packing simd 32-bit float slices into ap_uint<simd*32>."""
    _check_simd(simd)
    vec = _vec_name(simd)
    if simd == 8:
        params = (
            "data_t v0, data_t v1, data_t v2, data_t v3,\n"
            "                       data_t v4, data_t v5, data_t v6, data_t v7"
        )
    else:
        params = ", ".join(f"data_t v{i}" for i in range(simd))
    unions = ", ".join(f"c{i}" for i in range(simd))
    float_assigns = "\n".join(f"    c{i}.f = (float)v{i};" for i in range(simd))
    range_assigns = []
    for i in range(simd):
        hi = (i + 1) * 32 - 1
        lo = i * 32
        pad = " " * max(0, 4 - len(str(hi)))
        range_assigns.append(f"    w.range({hi}, {lo}){pad} = c{i}.u;")
    return (
        f"static {vec} pack{simd}({params}) {{\n"
        f"#pragma HLS INLINE\n"
        f"    union {{ unsigned u; float f; }} {unions};\n"
        f"{float_assigns}\n"
        f"\n"
        f"    {vec} w;\n"
        f"{chr(10).join(range_assigns)}\n"
        f"    return w;\n"
        f"}}"
    )


def emit_unpack(simd: int) -> str:
    """Emit unpackN extracting simd 32-bit float slices from ap_uint<simd*32>."""
    _check_simd(simd)
    vec = _vec_name(simd)
    if simd == 8:
        outs = (
            "data_t &v0, data_t &v1, data_t &v2, data_t &v3,\n"
            "                    data_t &v4, data_t &v5, data_t &v6, data_t &v7"
        )
    else:
        outs = ", ".join(f"data_t &v{i}" for i in range(simd))
    unions = ", ".join(f"c{i}" for i in range(simd))
    range_reads = []
    for i in range(simd):
        hi = (i + 1) * 32 - 1
        lo = i * 32
        range_reads.append(f"    c{i}.u = (unsigned)w.range({hi}, {lo});")
    float_outs = "\n".join(f"    v{i} = (data_t)c{i}.f;" for i in range(simd))
    return (
        f"static void unpack{simd}({vec} w, {outs}) {{\n"
        f"#pragma HLS INLINE\n"
        f"    union {{ unsigned u; float f; }} {unions};\n"
        f"{chr(10).join(range_reads)}\n"
        f"\n"
        f"{float_outs}\n"
        f"}}"
    )


def emit_mm_pe_calls(pe: int, *, indent: str = "        ") -> str:
    """Emit `pe` explicit mm_pe() invocations (DATAFLOW cannot loop on mm_pe)."""
    lines = []
    for i in range(pe):
        lines.append(
            f"{indent}mm_pe(fifo_A[{i}], fifo_B[{i}], fifo_B[{i + 1}], fifo_C[{i}]);"
        )
    return "\n".join(lines)


def _pack_args(simd: int, expr: str) -> str:
    args = [expr.format(i=i) for i in range(simd)]
    if simd <= 4:
        return ", ".join(args)
    return ", ".join(args[:4]) + ",\n                " + ", ".join(args[4:])


def _pack_write(simd: int, dest: str, expr: str, indent: int) -> str:
    args = _pack_args(simd, expr)
    # Second line of SIMD=8 args is relative to the first arg column.
    if simd == 8:
        inner = " " * (indent + 4)
        args = (
            ", ".join(expr.format(i=i) for i in range(4))
            + ",\n"
            + inner
            + ", ".join(expr.format(i=i) for i in range(4, 8))
        )
        return (
            f"{' ' * indent}{dest}.write(pack{simd}(\n"
            f"{inner}{args}));"
        )
    return f"{' ' * indent}{dest}.write(pack{simd}({args}));"


def _decl_a_vars(simd: int, *, init_zero: bool) -> str:
    if init_zero:
        return "data_t " + ", ".join(f"a{i} = 0" for i in range(simd)) + ";"
    return "data_t " + ", ".join(f"a{i}" for i in range(simd)) + ";"


def _decl_b_vars(simd: int) -> str:
    return "data_t " + ", ".join(f"b{i}" for i in range(simd)) + ";"


def _dot_partial(simd: int, indent: int) -> str:
    pad = " " * indent
    terms = [f"a{i} * b{i}" for i in range(simd)]
    if simd <= 4:
        return f"{pad}data_t partial = " + " + ".join(terms) + ";"
    row0 = " + ".join(terms[:4])
    row1 = " + ".join(terms[4:])
    inner = " " * (indent + 4)
    return (
        f"{pad}data_t partial =\n"
        f"{inner}{row0} +\n"
        f"{inner}{row1};"
    )


def _kj_loop(simd: int, extra_indent: int) -> str:
    vec = _vec_name(simd)
    a_list = ", ".join(f"a{i}" for i in range(simd))
    b_list = ", ".join(f"b{i}" for i in range(simd))
    body = f"""pe_kj: for (int t = 0; t < (K / SIMD) * J; ++t) {{
#pragma HLS PIPELINE II=1
    const int k_tile = t / J;
    const int j = t - k_tile * J;

    if (j == 0) {{
        unpack{simd}(fifo_A.read(), {a_list});
    }}

    {vec} bw = fifo_B_in.read();
    fifo_B_out.write(bw);

    {_decl_b_vars(simd)}
    unpack{simd}(bw, {b_list});

{_dot_partial(simd, 4)}

    data_t acc = (k_tile == 0) ? partial : (Crow[j] + partial);
    Crow[j] = acc;

    if (k_tile == (K / SIMD) - 1) {{
        fifo_C.write(acc);
    }}
}}"""
    return _indent(body, extra_indent)


def _emit_mm_pe_body(simd: int, *, replay_i_tiles: bool) -> str:
    a_decl = _decl_a_vars(simd, init_zero=replay_i_tiles)
    if replay_i_tiles:
        kj = _kj_loop(simd, extra_indent=8)
        return (
            "    data_t Crow[J];\n"
            "#pragma HLS BIND_STORAGE variable=Crow type=ram_2p impl=bram\n"
            "\n"
            "    for (int tile = 0; tile < I / PE_NUM; ++tile) {\n"
            "        const int i0 = tile * PE_NUM;\n"
            "        (void)i0;\n"
            f"        {a_decl}\n"
            f"{kj}\n"
            "    }"
        )
    kj = _kj_loop(simd, extra_indent=4)
    return (
        "    data_t Crow[J];\n"
        "#pragma HLS BIND_STORAGE variable=Crow type=ram_2p impl=bram\n"
        "\n"
        f"    {a_decl}\n"
        "\n"
        f"{kj}"
    )


def _emit_load_A(simd: int, *, around_dataflow: bool) -> str:
    vec = _vec_name(simd)
    expr = "A[i0 + p][k0 + {i}]"
    if around_dataflow:
        write = _pack_write(simd, "fifo_A[p]", expr, indent=12)
        return f"""static void load_A(data_t A[I][K], hls::stream<{vec}> fifo_A[PE_NUM], int i0) {{
#pragma HLS INLINE off
    for (int k0 = 0; k0 < K; k0 += SIMD) {{
        for (int p = 0; p < PE_NUM; ++p) {{
#pragma HLS PIPELINE II=1
{write}
        }}
    }}
}}"""
    write = _pack_write(simd, "fifo_A[p]", expr, indent=16)
    return f"""static void load_A(data_t A[I][K], hls::stream<{vec}> fifo_A[PE_NUM]) {{
#pragma HLS INLINE off
    for (int tile = 0; tile < I / PE_NUM; ++tile) {{
        const int i0 = tile * PE_NUM;
        for (int k0 = 0; k0 < K; k0 += SIMD) {{
            for (int p = 0; p < PE_NUM; ++p) {{
#pragma HLS PIPELINE II=1
{write}
            }}
        }}
    }}
}}"""


def _emit_load_B(simd: int, *, around_dataflow: bool) -> str:
    vec = _vec_name(simd)
    expr = "B[j][k0 + {i}]"
    if around_dataflow:
        write = _pack_write(simd, "fifo_B", expr, indent=12)
        return f"""static void load_B(data_t B[J][K], hls::stream<{vec}> &fifo_B) {{
#pragma HLS INLINE off
    for (int k0 = 0; k0 < K; k0 += SIMD) {{
        for (int j = 0; j < J; ++j) {{
#pragma HLS PIPELINE II=1
{write}
        }}
    }}
}}"""
    write = _pack_write(simd, "fifo_B", expr, indent=16)
    return f"""static void load_B(data_t B[J][K], hls::stream<{vec}> &fifo_B) {{
#pragma HLS INLINE off
    for (int tile = 0; tile < I / PE_NUM; ++tile) {{
        const int i0 = tile * PE_NUM;
        (void)i0;
        for (int k0 = 0; k0 < K; k0 += SIMD) {{
            for (int j = 0; j < J; ++j) {{
#pragma HLS PIPELINE II=1
{write}
            }}
        }}
    }}
}}"""


def _emit_drain_B(simd: int, *, around_dataflow: bool) -> str:
    vec = _vec_name(simd)
    if around_dataflow:
        return f"""static void drain_B(hls::stream<{vec}> &fifo_B) {{
#pragma HLS INLINE off
    for (int k0 = 0; k0 < K; k0 += SIMD) {{
        for (int j = 0; j < J; ++j) {{
#pragma HLS PIPELINE II=1
            (void)fifo_B.read();
        }}
    }}
}}"""
    return f"""static void drain_B(hls::stream<{vec}> &fifo_B) {{
#pragma HLS INLINE off
    for (int tile = 0; tile < I / PE_NUM; ++tile) {{
        const int i0 = tile * PE_NUM;
        (void)i0;
        for (int k0 = 0; k0 < K; k0 += SIMD) {{
            for (int j = 0; j < J; ++j) {{
#pragma HLS PIPELINE II=1
                (void)fifo_B.read();
            }}
        }}
    }}
}}"""


def _emit_store_C(*, around_dataflow: bool) -> str:
    if around_dataflow:
        return """static void store_C(data_t C[I][J], hls::stream<data_t> fifo_C[PE_NUM], int i0) {
#pragma HLS INLINE off
    for (int p = 0; p < PE_NUM; ++p) {
        for (int j = 0; j < J; ++j) {
#pragma HLS PIPELINE II=1
            C[i0 + p][j] = fifo_C[p].read();
        }
    }
}"""
    return """static void store_C(data_t C[I][J], hls::stream<data_t> fifo_C[PE_NUM]) {
#pragma HLS INLINE off
    for (int tile = 0; tile < I / PE_NUM; ++tile) {
        const int i0 = tile * PE_NUM;
        for (int p = 0; p < PE_NUM; ++p) {
            for (int j = 0; j < J; ++j) {
#pragma HLS PIPELINE II=1
                C[i0 + p][j] = fifo_C[p].read();
            }
        }
    }
}"""


def _emit_mm_pe(simd: int, *, around_dataflow: bool) -> str:
    vec = _vec_name(simd)
    body = _emit_mm_pe_body(simd, replay_i_tiles=not around_dataflow)
    return f"""static void mm_pe(hls::stream<{vec}> &fifo_A,
                  hls::stream<{vec}> &fifo_B_in,
                  hls::stream<{vec}> &fifo_B_out,
                  hls::stream<data_t> &fifo_C) {{
#pragma HLS INLINE off
{body}
}}"""


def _fifo_depths(*, around_dataflow: bool) -> tuple[int, int, int]:
    # Proven 32x8 ping-pong uses fifo_B depth=8; locked 16x4 uses 16.
    if around_dataflow:
        return 16, 8, 64
    return 16, 16, 64


def instantiate_mm(rec: PeRecipe) -> str:
    """Return compact-stream autosa_mm C++ for `rec`. Deterministic; no LLM."""
    pe = int(rec.pe)
    simd = int(rec.simd)
    pack_bits = int(rec.pack_bits)
    _check_simd(simd)

    layout = getattr(rec, "layout", "chain")
    if layout == "pack":
        from compact_pe_pack_instantiate import instantiate_pack

        return instantiate_pack(rec)

    if layout in ("io4", "io5"):
        from compact_pe_io_instantiate import instantiate_io

        return instantiate_io(rec)

    if pack_bits != simd * 32:
        raise ValueError(f"pack_bits={pack_bits} does not match simd={simd} * 32")

    if layout == "mesh":
        from compact_pe_mesh_instantiate import instantiate_mesh

        return instantiate_mesh(rec)

    around = rec.i_tiles > 1
    vec = _vec_name(simd)
    d_a, d_b, d_c = _fifo_depths(around_dataflow=around)
    call_indent = "        " if around else "    "
    calls = emit_mm_pe_calls(pe, indent=call_indent)

    if around:
        top_body = f"""    hls::stream<{vec}> fifo_A[PE_NUM];
    hls::stream<{vec}> fifo_B[PE_NUM + 1];
    hls::stream<data_t> fifo_C[PE_NUM];

#pragma HLS STREAM variable=fifo_A depth={d_a}
#pragma HLS STREAM variable=fifo_B depth={d_b}
#pragma HLS STREAM variable=fifo_C depth={d_c}
#pragma HLS ARRAY_PARTITION variable=fifo_A complete
#pragma HLS ARRAY_PARTITION variable=fifo_B complete
#pragma HLS ARRAY_PARTITION variable=fifo_C complete

    for (int i0 = 0; i0 < I; i0 += PE_NUM) {{
#pragma HLS DATAFLOW
        load_A(A, fifo_A, i0);
        load_B(B, fifo_B[0]);

{calls}

        drain_B(fifo_B[{pe}]);
        store_C(C, fifo_C, i0);
    }}"""
    else:
        top_body = f"""    hls::stream<{vec}> fifo_A[PE_NUM];
    hls::stream<{vec}> fifo_B[PE_NUM + 1];
    hls::stream<data_t> fifo_C[PE_NUM];

#pragma HLS STREAM variable=fifo_A depth={d_a}
#pragma HLS STREAM variable=fifo_B depth={d_b}
#pragma HLS STREAM variable=fifo_C depth={d_c}
#pragma HLS ARRAY_PARTITION variable=fifo_A complete
#pragma HLS ARRAY_PARTITION variable=fifo_B complete
#pragma HLS ARRAY_PARTITION variable=fifo_C complete

#pragma HLS DATAFLOW

    load_A(A, fifo_A);
    load_B(B, fifo_B[0]);

{calls}

    drain_B(fifo_B[{pe}]);
    store_C(C, fifo_C);"""

    simd_define = f"#define SIMD {simd}"
    if simd == 8:
        simd_define = "#define SIMD   8"

    code = "\n".join(
        [
            "// Generated by compact_pe_instantiate.py (deterministic; not LLM).",
            '#include "kernel.h"',
            "#include <hls_stream.h>",
            "#include <ap_int.h>",
            "",
            f"#define PE_NUM {pe}",
            simd_define,
            "",
            f"typedef ap_uint<{pack_bits}> {vec};",
            "",
            emit_pack(simd),
            "",
            emit_unpack(simd),
            "",
            _emit_load_A(simd, around_dataflow=around),
            "",
            _emit_load_B(simd, around_dataflow=around),
            "",
            _emit_mm_pe(simd, around_dataflow=around),
            "",
            _emit_drain_B(simd, around_dataflow=around),
            "",
            _emit_store_C(around_dataflow=around),
            "",
            'extern "C" void autosa_mm(data_t A[I][K], data_t B[J][K], data_t C[I][J]) {',
            _INTERFACE,
            "",
            top_body,
            "}",
            "",
        ]
    )
    if not around and _FORBIDDEN_I0_WRAP in code:
        raise RuntimeError(
            "i_tiles==1 kernel must not contain "
            f"{_FORBIDDEN_I0_WRAP!r}"
        )
    return code
