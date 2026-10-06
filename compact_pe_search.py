# compact_pe_search.py
from __future__ import annotations

from post_flash_pe_recipe import PeRecipe

U280_DSP = 9024
I = J = K = 64
PE_CHOICES = (8, 16, 32, 64)
SIMD_CHOICES = (2, 4, 8)


def candidate_id(rec: PeRecipe) -> str:
    return f"pe{rec.pe}_simd{rec.simd}"


def enumerate_mm_recipes(*, dsp_cap: float = 0.85) -> list[PeRecipe]:
    out: list[PeRecipe] = []
    max_dsp = dsp_cap * U280_DSP
    for pe in PE_CHOICES:
        if I % pe:
            continue
        for simd in SIMD_CHOICES:
            if K % simd:
                continue
            expected = pe * simd * 5
            if expected > max_dsp:
                continue
            # PE arrays of 16 or fewer absorb remaining I rows inside tasks
            # (locked 16×4 style: load_A/store_C scan full I, no outer i0 wrap).
            # Larger PE arrays tile I around DATAFLOW (proven 32×8 style).
            i_tiles = 1 if pe <= 16 else I // pe
            rec = PeRecipe(
                bench="autosa_mm",
                pe=pe,
                simd=simd,
                data_kind="float",
                i_macro="I",
                j_macro="J",
                k_macro="K",
                pack_bits=simd * 32,
                expected_dsp=expected,
                min_dsp=max(1, int(expected * 0.6)),
                pe_kj=(K // simd) * J,
                i_tiles=i_tiles,
                tile_loop="around_dataflow" if i_tiles > 1 else "inside_tasks",
                note=f"search point {pe}x{simd}",
            )
            out.append(rec)
    return out


MESH_PE_CHOICES = (4, 8, 16, 32)


def candidate_id_mesh(rec: PeRecipe) -> str:
    return f"mesh{rec.pe_i}x{rec.pe_j}_simd{rec.simd}"


def candidate_id_pack(rec: PeRecipe) -> str:
    return f"pack{rec.pe_i}x{rec.pe_j}_simd{rec.simd}"


def candidate_id_io(rec: PeRecipe) -> str:
    return f"{rec.layout}_{rec.pe_i}x{rec.pe_j}_s{rec.simd}_k{rec.k_part}_j{rec.j_part}"


def search_candidate_id(rec: PeRecipe) -> str:
    if rec.layout in ("io4", "io5"):
        return candidate_id_io(rec)
    if rec.layout == "pack":
        return candidate_id_pack(rec)
    if rec.layout == "mesh":
        return candidate_id_mesh(rec)
    return candidate_id(rec)


def enumerate_mm_mesh_recipes(
    *, dsp_cap: float = 0.85, max_pe: int = 128
) -> list[PeRecipe]:
    out: list[PeRecipe] = []
    max_dsp = dsp_cap * U280_DSP
    for pe_i in MESH_PE_CHOICES:
        if I % pe_i:
            continue
        for pe_j in MESH_PE_CHOICES:
            if J % pe_j:
                continue
            if pe_i * pe_j > max_pe:
                continue
            for simd in SIMD_CHOICES:
                if K % simd:
                    continue
                expected = pe_i * pe_j * simd * 5
                if expected > max_dsp:
                    continue
                rec = PeRecipe(
                    bench="autosa_mm",
                    pe=pe_i * pe_j,
                    simd=simd,
                    data_kind="float",
                    i_macro="I",
                    j_macro="J",
                    k_macro="K",
                    pack_bits=simd * 32,
                    expected_dsp=expected,
                    min_dsp=max(1, int(expected * 0.6)),
                    pe_kj=(K // simd) * (J // pe_j),
                    i_tiles=I // pe_i,
                    tile_loop="inside_tasks",
                    note=f"mesh {pe_i}x{pe_j} simd{simd}",
                    layout="mesh",
                    pe_i=pe_i,
                    pe_j=pe_j,
                )
                out.append(rec)
    return out


def enumerate_mm_pack_recipes(
    *, dsp_cap: float = 0.85, max_pe: int = 128
) -> list[PeRecipe]:
    """Same 30-point mesh grid, packed AXI ABI (family B)."""
    out: list[PeRecipe] = []
    max_dsp = dsp_cap * U280_DSP
    for pe_i in MESH_PE_CHOICES:
        if I % pe_i:
            continue
        for pe_j in MESH_PE_CHOICES:
            if J % pe_j:
                continue
            if pe_i * pe_j > max_pe:
                continue
            for simd in SIMD_CHOICES:
                if K % simd:
                    continue
                expected = pe_i * pe_j * simd * 5
                if expected > max_dsp:
                    continue
                lat_j = J // pe_j
                rec = PeRecipe(
                    bench="autosa_mm",
                    pe=pe_i * pe_j,
                    simd=simd,
                    data_kind="float",
                    i_macro="I",
                    j_macro="J",
                    k_macro="K",
                    pack_bits=512,
                    expected_dsp=expected,
                    min_dsp=max(1, int(expected * 0.6)),
                    pe_kj=(K // simd) * lat_j,
                    i_tiles=1,
                    tile_loop="inside_tasks",
                    note=f"pack {pe_i}x{pe_j} simd{simd}",
                    layout="pack",
                    pe_i=pe_i,
                    pe_j=pe_j,
                )
                out.append(rec)
    return out


# (layout, pe_i, pe_j, simd, k_part, j_part)
_IO_POINTS = (
    ("io4", 8, 4, 8, 32, 32),
    ("io4", 8, 4, 8, 32, 64),
    ("io4", 16, 8, 8, 32, 64),
    ("io5", 8, 4, 8, 32, 32),
    ("io5", 8, 4, 8, 32, 64),
    ("io5", 16, 8, 8, 64, 64),
)


def enumerate_mm_io_recipes(
    *, dsp_cap: float = 0.85, max_pe: int = 128
) -> list[PeRecipe]:
    """Family C: paper Design 4 (io4) and Design 5 (io5) in one grid."""
    out: list[PeRecipe] = []
    max_dsp = dsp_cap * U280_DSP
    for layout, pe_i, pe_j, simd, k_part, j_part in _IO_POINTS:
        if I % pe_i:
            continue
        if J % j_part or K % k_part:
            continue
        pe = pe_i * pe_j
        if pe > max_pe:
            continue
        lat_i = I // pe_i
        if layout == "io4":
            if j_part % pe_j:
                continue
            lat_j = j_part // pe_j
            if pe_j * lat_j != j_part:
                continue
            pe_kj = (k_part // simd) * lat_j
        else:
            if pe_j * simd != k_part:
                continue
            lat_j = 4 if j_part % 4 == 0 else 8
            if j_part % lat_j:
                continue
            pe_kj = j_part * lat_i
        expected = pe * simd * 5
        if expected > max_dsp:
            continue
        out.append(
            PeRecipe(
                bench="autosa_mm",
                pe=pe,
                simd=simd,
                data_kind="float",
                i_macro="I",
                j_macro="J",
                k_macro="K",
                pack_bits=512,
                expected_dsp=expected,
                min_dsp=max(1, int(expected * 0.6)),
                pe_kj=pe_kj,
                i_tiles=1,
                tile_loop="inside_tasks",
                note=f"{layout} {pe_i}x{pe_j} s{simd} k{k_part} j{j_part}",
                layout=layout,
                pe_i=pe_i,
                pe_j=pe_j,
                k_part=k_part,
                j_part=j_part,
                lat_i=lat_i,
                lat_j=lat_j,
            )
        )
    return out
