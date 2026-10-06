"""Per-kernel PE x SIMD x pack recipe for AutoSA 64^3 GEMM DSE/stream.

The original post-flash skills were locked to autosa_mm (float, PE=16, SIMD=4,
ap_uint<128>, DSP~320). Wave-1 rank1 arrays are not that:

  bench                 rank1 DSP   PE x SIMD   data_t
  autosa_mm             320         16 x 4      float
  autosa_mm_hcl         320         32 x 2      float
  autosa_mm_hcl_intel   320         16 x 4      float
  autosa_mm_intel       640         32 x 4      float
  autosa_mm_getting_started  640    32 x 4      float
  autosa_mm_int16        64         32 x 2      unsigned short
  autosa_mm_catapult     96          8 x 4      unsigned int (I_P/J_P/K_P)

Overlay (same autosa_mm ABI, selected with C2HLS_PE_RECIPE):
  autosa_mm_32x8       1280        32 x 8     float, pack 256, pe_kj=512, around_dataflow

Do not clone AutoSA netlists. Compact nested-loop / stream PE only.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from typing import Optional


@dataclass(frozen=True)
class PeRecipe:
    bench: str
    pe: int
    simd: int
    data_kind: str  # float | uint16 | uint32
    i_macro: str
    j_macro: str
    k_macro: str
    pack_bits: int
    expected_dsp: int
    min_dsp: int
    pe_kj: int
    i_tiles: int = 1
    tile_loop: str = "inside_tasks"  # inside_tasks | around_dataflow
    note: str = ""
    layout: str = "chain"  # chain | mesh | pack | io4 | io5
    pe_i: int = 0  # 0 => use pe (chain)
    pe_j: int = 1
    k_part: int = 0
    j_part: int = 0
    lat_i: int = 0
    lat_j: int = 0
    ti: int = 0
    tj: int = 0
    tk: int = 0
    load_trip: int = 0
    compute_trip: int = 0
    store_trip: int = 0
    outer_tiles: int = 0

    @property
    def pack_lanes(self) -> int:
        return self.simd

    @property
    def elem_bits(self) -> int:
        return {"float": 32, "uint16": 16, "uint32": 32}[self.data_kind]


_RECIPES: dict[str, PeRecipe] = {
    "autosa_mm": PeRecipe(
        bench="autosa_mm",
        pe=16,
        simd=4,
        data_kind="float",
        i_macro="I",
        j_macro="J",
        k_macro="K",
        pack_bits=128,
        expected_dsp=320,
        min_dsp=200,
        pe_kj=1024,
        note="locked mm: 16x4 float, DSP~320, pe_kj=1024",
    ),
    "autosa_mm_hcl": PeRecipe(
        bench="autosa_mm_hcl",
        pe=32,
        simd=2,
        data_kind="float",
        i_macro="I",
        j_macro="J",
        k_macro="K",
        pack_bits=64,
        expected_dsp=320,
        min_dsp=200,
        pe_kj=2048,
        note="rank1 array_part[32,64,4] latency[1,32] simd[2] => PE=32 SIMD=2",
    ),
    "autosa_mm_hcl_intel": PeRecipe(
        bench="autosa_mm_hcl_intel",
        pe=16,
        simd=4,
        data_kind="float",
        i_macro="I",
        j_macro="J",
        k_macro="K",
        pack_bits=128,
        expected_dsp=320,
        min_dsp=200,
        pe_kj=1024,
        note="same 16x4 float class as autosa_mm",
    ),
    "autosa_mm_intel": PeRecipe(
        bench="autosa_mm_intel",
        pe=32,
        simd=4,
        data_kind="float",
        i_macro="I",
        j_macro="J",
        k_macro="K",
        pack_bits=128,
        expected_dsp=640,
        min_dsp=400,
        pe_kj=1024,
        i_tiles=2,
        tile_loop="around_dataflow",
        note="rank1 ~32 PE x SIMD=4, DSP~640. I/PE=2: wrap DATAFLOW in i-tile loop (do not replay 2048 beats inside mm_pe).",
    ),
    "autosa_mm_getting_started": PeRecipe(
        bench="autosa_mm_getting_started",
        pe=32,
        simd=4,
        data_kind="float",
        i_macro="I",
        j_macro="J",
        k_macro="K",
        pack_bits=128,
        expected_dsp=640,
        min_dsp=400,
        pe_kj=1024,
        i_tiles=2,
        tile_loop="around_dataflow",
        note="I/PE=2. Tile loop MUST wrap DATAFLOW so ping-pong II~1024, latency~2194. Tile loop inside mm_pe doubles to ~4307.",
    ),
    "autosa_mm_int16": PeRecipe(
        bench="autosa_mm_int16",
        pe=32,
        simd=2,
        data_kind="uint16",
        i_macro="I",
        j_macro="J",
        k_macro="K",
        pack_bits=32,
        expected_dsp=64,
        min_dsp=40,
        pe_kj=2048,
        note="unsigned short; pack 2 x u16 as ap_uint<32>; do not use float union; DSP~64",
    ),
    "autosa_mm_catapult": PeRecipe(
        bench="autosa_mm_catapult",
        pe=8,
        simd=4,
        data_kind="uint32",
        i_macro="I_P",
        j_macro="J_P",
        k_macro="K_P",
        pack_bits=128,
        expected_dsp=96,
        min_dsp=48,
        pe_kj=1024,
        note="unsigned int; macros I_P/J_P/K_P not I/J/K; PE=8 SIMD=4; DSP~96",
    ),
    "autosa_mm_32x8": PeRecipe(
        bench="autosa_mm_32x8",
        pe=32,
        simd=8,
        data_kind="float",
        i_macro="I",
        j_macro="J",
        k_macro="K",
        pack_bits=256,
        expected_dsp=1280,
        min_dsp=800,
        pe_kj=512,
        i_tiles=2,
        tile_loop="around_dataflow",
        note=(
            "overlay on autosa_mm ABI: 32 PE x SIMD=8, ap_uint<256>, DSP~1280. "
            "I/PE=2: wrap DATAFLOW in the top. pe_kj=512. Do not copy PE_NUM 16 / pack4. "
            "Not the locked 16x4 slide recipe."
        ),
    ),
}

DEFAULT_RECIPE = _RECIPES["autosa_mm"]
# Locked PE×SIMD recipes were authored against the 64³ autosa_mm header.
RECIPE_REFERENCE_N = 64


def recipe_for(bench: Optional[str]) -> PeRecipe:
    override = os.getenv("C2HLS_PE_RECIPE", "").strip()
    if override and override in _RECIPES:
        return _RECIPES[override]
    if not bench:
        return DEFAULT_RECIPE
    key = bench.strip()
    return _RECIPES.get(key, DEFAULT_RECIPE)


def _env_positive_int(name: str) -> Optional[int]:
    raw = os.getenv(name, "").strip()
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError:
        return None
    return value if value > 0 else None


def _parse_seed_pe_simd(seed_code: Optional[str]) -> tuple[Optional[int], Optional[int]]:
    if not seed_code:
        return None, None
    from post_flash_focus_group import parse_pe_simd_from_kernel

    return parse_pe_simd_from_kernel(seed_code)


def _positive_int(value: Optional[int]) -> Optional[int]:
    if isinstance(value, int) and value > 0:
        return value
    return None


def _parse_seed_tiles(
    seed_code: Optional[str],
) -> tuple[Optional[int], Optional[int], Optional[int]]:
    if not seed_code:
        return None, None, None
    from post_flash_focus_group import parse_tile_sizes

    found = parse_tile_sizes(seed_code)
    return (
        _positive_int(found.get("TI")),
        _positive_int(found.get("TJ")),
        _positive_int(found.get("TK")),
    )


def onchip_tiles_active(rec: PeRecipe) -> bool:
    return rec.ti > 0 and rec.tj > 0 and rec.tk > 0


def overlay_pe_simd(rec: PeRecipe, pe: int, simd: int) -> PeRecipe:
    """Replace PE and SIMD. Pack width is SIMD × element bits.

    DSP floors scale with PE×SIMD from the locked recipe so a wider seed is
    not accepted at the 16×4 DSP budget. Trip counts are filled later by
    ``scale_recipe_to_problem``.
    """
    if pe == rec.pe and simd == rec.simd:
        return rec
    old = rec.pe * rec.simd
    new = pe * simd
    if old > 0 and new > 0:
        expected = max(1, rec.expected_dsp * new // old)
        min_dsp = max(1, rec.min_dsp * new // old)
    else:
        expected, min_dsp = rec.expected_dsp, rec.min_dsp
    return replace(
        rec,
        pe=pe,
        simd=simd,
        pack_bits=simd * rec.elem_bits,
        expected_dsp=expected,
        min_dsp=min_dsp,
        note=(
            f"PE={pe} SIMD={simd} from the seed kernel or "
            "C2HLS_STREAM_PE/C2HLS_STREAM_SIMD"
        ),
    )


def resolve_stream_recipe(
    bench: Optional[str],
    i: int,
    j: int,
    k: int,
    *,
    seed_code: Optional[str] = None,
) -> PeRecipe:
    """Locked bench recipe, then seed PE×SIMD, then numeric env.

    ``C2HLS_STREAM_PE`` / ``C2HLS_STREAM_SIMD`` win when set. Otherwise a
    named ``C2HLS_PE_RECIPE`` wins over the seed. Otherwise ``const int PE``
    / ``const int SIMD`` (or ``#define``) in the seed kernel replace the
    locked pair. With no seed and no override, the bench recipe is unchanged
    (autosa_mm stays 16×4).

    When the seed (or ``C2HLS_STREAM_TI`` / ``TJ`` / ``TK``) defines all three
    on-chip tiles, the stream nest follows those tiles. A named
    ``C2HLS_PE_RECIPE`` still ignores seed tiles unless a numeric tile
    override is set. A 64³ kernel with no TI/TJ/TK stays on the locked
    systolic trips.
    """
    base = recipe_for(bench)
    env_pe = _env_positive_int("C2HLS_STREAM_PE")
    env_simd = _env_positive_int("C2HLS_STREAM_SIMD")
    named = os.getenv("C2HLS_PE_RECIPE", "").strip() in _RECIPES
    seed_pe: Optional[int] = None
    seed_simd: Optional[int] = None
    if not named or env_pe is not None or env_simd is not None:
        seed_pe, seed_simd = _parse_seed_pe_simd(seed_code)
    if named and env_pe is None and env_simd is None:
        seed_pe, seed_simd = None, None
    pe = env_pe if env_pe is not None else (seed_pe if seed_pe is not None else base.pe)
    simd = (
        env_simd if env_simd is not None else (seed_simd if seed_simd is not None else base.simd)
    )
    changed = pe != base.pe or simd != base.simd
    rec = overlay_pe_simd(base, pe, simd) if changed else base
    rec = scale_recipe_to_problem(rec, i, j, k, force=changed)
    env_ti = _env_positive_int("C2HLS_STREAM_TI")
    env_tj = _env_positive_int("C2HLS_STREAM_TJ")
    env_tk = _env_positive_int("C2HLS_STREAM_TK")
    seed_ti: Optional[int] = None
    seed_tj: Optional[int] = None
    seed_tk: Optional[int] = None
    named_blocks_seed_tiles = (
        named and env_ti is None and env_tj is None and env_tk is None
    )
    if not named_blocks_seed_tiles:
        seed_ti, seed_tj, seed_tk = _parse_seed_tiles(seed_code)
    ti = env_ti if env_ti is not None else seed_ti
    tj = env_tj if env_tj is not None else seed_tj
    tk = env_tk if env_tk is not None else seed_tk
    if ti and tj and tk:
        rec = apply_onchip_tile_recipe(rec, i, j, k, ti, tj, tk)
    return rec


def apply_onchip_tile_recipe(
    rec: PeRecipe,
    i: int,
    j: int,
    k: int,
    ti: int,
    tj: int,
    tk: int,
) -> PeRecipe:
    """Switch the stream nest to the seed's on-chip TI×TJ×TK tiles.

    B is one TJ×TK tile reused by the PEs. Compute stays the seed's PE-row ×
    SIMD-lane nest inside TK and TJ. There is no loop over full K.
    """
    simd = rec.simd if rec.simd > 0 else 1
    compute_trip = (int(tk) // simd) * int(tj)
    load_trip = int(tj) * int(tk)
    store_trip = int(rec.pe) * int(tj)
    outer_i = int(i) // int(ti) if ti else 0
    outer_j = int(j) // int(tj) if tj else 0
    outer = max(outer_i, 0) * max(outer_j, 0)
    note = (
        f"On-chip tiles TI={int(ti)} TJ={int(tj)} TK={int(tk)} from the seed. "
        f"i0 steps by TI, j0 steps by TJ, k only in 0..TK-1. "
        "No loop over full K. Do not widen the partial GEMM. "
        f"load_B is one TJ×TK tile (element trip {load_trip}) per j0 or per (i0,j0). "
        f"Compute is the seed PE-row × SIMD-lane nest inside the tile "
        f"(trip {compute_trip}). Store trip {store_trip} with i < PE. "
        f"Outer tile pairs {outer}. DATAFLOW ping-pong overlaps load, compute, and store."
    )
    return replace(
        rec,
        ti=int(ti),
        tj=int(tj),
        tk=int(tk),
        load_trip=load_trip,
        compute_trip=compute_trip,
        store_trip=store_trip,
        outer_tiles=outer,
        pe_kj=compute_trip,
        tile_loop="around_dataflow",
        note=note,
    )


def scale_recipe_to_problem(
    rec: PeRecipe,
    i: int,
    j: int,
    k: int,
    *,
    force: bool = False,
) -> PeRecipe:
    """Recompute trip counts from kernel.h I/J/K.

    PE, SIMD, pack width, and tile_loop stay on the recipe object passed in.
    At the authored 64³ size the recipe object is returned unchanged, including
    autosa_mm's historical i_tiles=1, unless ``force`` (a seed or env PE×SIMD
    override). Any other size uses pe_kj=(K/SIMD)*J and i_tiles=I/PE_NUM.
    """
    ref_size = (
        int(i) == RECIPE_REFERENCE_N
        and int(j) == RECIPE_REFERENCE_N
        and int(k) == RECIPE_REFERENCE_N
    )
    if ref_size and not force:
        return rec
    if rec.simd <= 0 or rec.pe <= 0:
        return rec
    pe_kj = (int(k) // rec.simd) * int(j)
    i_tiles = int(i) // rec.pe
    if i_tiles < 1:
        i_tiles = rec.i_tiles
    note = rec.note or ""
    if not ref_size:
        suffix = (
            f" Problem size from kernel.h: I={int(i)} J={int(j)} K={int(k)}"
            " (not a fixed 64^3)."
        )
        if suffix.strip() not in note:
            note = (note + suffix).strip()
    return replace(rec, pe_kj=pe_kj, i_tiles=i_tiles, note=note)


def _format_onchip_tile_prompt(
    rec: PeRecipe,
    *,
    step: str,
    i: Optional[int],
    j: Optional[int],
    k: Optional[int],
) -> str:
    pack_how = {
        "float": f"{rec.simd} floats via bit_cast/union of uint32 (NOT for integer data_t)",
        "uint16": f"{rec.simd} unsigned short via raw bits (no float union)",
        "uint32": f"{rec.simd} unsigned int via raw bits (no float union)",
    }[rec.data_kind]
    lines = [f"## Mandatory PE recipe for {rec.bench} ({step})"]
    if i is not None and j is not None and k is not None:
        lines.append(
            f"Problem size is I={int(i)} J={int(j)} K={int(k)} from kernel.h. "
            "Do not assume a 64^3 matrix."
        )
    lines.append(
        f"Authoritative pair is PE_NUM={rec.pe} SIMD={rec.simd}. "
        "Do not substitute another pair."
    )
    lines.extend([
        f"- PE_NUM={rec.pe}",
        f"- SIMD={rec.simd}",
        f"- TI={rec.ti}",
        f"- TJ={rec.tj}",
        f"- TK={rec.tk}",
        (
            f"- loop macros: {rec.i_macro}, {rec.j_macro}, {rec.k_macro} "
            "(read kernel.h; catapult is I_P/J_P/K_P)"
        ),
        f"- data_t kind: {rec.data_kind}",
        f"- pack SIMD as ap_uint<{rec.pack_bits}> ({pack_how})",
        f"- outer tile pairs (I/TI)*(J/TJ) = {rec.outer_tiles}",
        "- i0 steps by TI. j0 steps by TJ. k only inside 0..TK-1.",
        "- Do not add a K-tile loop the seed does not have. Do not widen the partial GEMM.",
        "- A and B are indexed with k in 0..TK-1, not across the full K extent.",
        (
            "- load_B loads one TJ×TK tile per j0 (reused across i0) or per (i0, j0). "
            f"Element trip TJ*TK = {rec.load_trip}."
        ),
        (
            "- That B tile is reused by the PEs. It is not replayed for every i0 "
            "across all columns and every SIMD slice of the full K extent."
        ),
        (
            "- Do not nest load_B as an i0 loop stepping by PE_NUM, a k0 loop walking "
            "the full K extent, and a j loop over all columns."
        ),
        (
            "- compute stays the seed PE-row × SIMD-lane nest inside the tile: "
            "k0 steps by SIMD while k0 < TK, j < TJ, p < PE unrolled, s < SIMD unrolled. "
            f"Compute trip (TK/SIMD)*TJ = {rec.compute_trip}."
        ),
        (
            f"- store trip PE*TJ = {rec.store_trip}. "
            "The seed stores i < PE, not i < TI. Keep that."
        ),
        (
            "- DATAFLOW / ping-pong overlaps load, compute, and store. "
            "Put DATAFLOW inside the j0 loop so the next tile's load overlaps this tile."
        ),
        (
            f"- Do not build {rec.pe} mm_pe calls whose body walks every SIMD slice "
            "of the full K extent and every column."
        ),
        f"- expected DSP ≈ {rec.expected_dsp} (reject if DSP < {rec.min_dsp})",
        (
            f"- instantiate {rec.pe} explicit mm_pe() calls only for the tile nest above; "
            "DATAFLOW cannot call mm_pe in a C for-loop"
        ),
        "- keep the exact extern C top name and parameter list",
        (
            "- if INTERFACE uses one bundle=gmem for A/B/C, split to "
            "gmem0/gmem1/gmem2 (do not change ports)"
        ),
        f"- note: {rec.note}",
    ])
    if rec.data_kind != "float":
        lines.append(
            "- integer GEMM: do not emit `0.0f` or float pack/unpack; C is data_t accumulation"
        )
    return "\n".join(lines) + "\n"


def format_recipe_prompt(
    bench: str,
    *,
    step: str = "dse",
    i: Optional[int] = None,
    j: Optional[int] = None,
    k: Optional[int] = None,
    recipe: Optional[PeRecipe] = None,
) -> str:
    if recipe is not None:
        rec = recipe
    else:
        rec = recipe_for(bench)
        if i is not None and j is not None and k is not None:
            rec = scale_recipe_to_problem(rec, int(i), int(j), int(k))
    if onchip_tiles_active(rec):
        return _format_onchip_tile_prompt(rec, step=step, i=i, j=j, k=k)
    pack_how = {
        "float": f"{rec.simd} floats via bit_cast/union of uint32 (NOT for integer data_t)",
        "uint16": f"{rec.simd} unsigned short via raw bits (no float union)",
        "uint32": f"{rec.simd} unsigned int via raw bits (no float union)",
    }[rec.data_kind]
    mm_pe = rec.pe
    sized = i is not None and j is not None and k is not None
    lines = [
        f"## Mandatory PE recipe for {rec.bench} ({step})",
    ]
    if sized and (int(i), int(j), int(k)) != (
        RECIPE_REFERENCE_N,
        RECIPE_REFERENCE_N,
        RECIPE_REFERENCE_N,
    ):
        lines.append(
            f"Problem size is I={int(i)} J={int(j)} K={int(k)} from kernel.h. "
            "Do not assume a 64^3 matrix."
        )
    if rec.pe == 16 and rec.simd == 4:
        lines.extend([
            "Do **not** default to autosa_mm PE=16 SIMD=4 unless this block says so.",
            "Do not copy PE_NUM 16 / SIMD 4 / pack4 from the skill template unless this recipe is 16x4.",
        ])
    else:
        lines.append(
            f"Authoritative pair is PE_NUM={rec.pe} SIMD={rec.simd}. "
            "Do not substitute another pair."
        )
    lines.extend([
        f"- PE_NUM={rec.pe}",
        f"- SIMD={rec.simd}",
        f"- loop macros: {rec.i_macro}, {rec.j_macro}, {rec.k_macro} (read kernel.h; catapult is I_P/J_P/K_P)",
        f"- data_t kind: {rec.data_kind}",
        f"- pack SIMD as ap_uint<{rec.pack_bits}> ({pack_how})",
        f"- pe_kj trip = ({rec.k_macro}/SIMD)*{rec.j_macro} = {rec.pe_kj}",
        f"- i_tiles = {rec.i_macro}/PE_NUM = {rec.i_tiles} (tile_loop={rec.tile_loop})",
        f"- expected DSP ≈ {rec.expected_dsp} (reject if DSP < {rec.min_dsp})",
        f"- instantiate {mm_pe} explicit mm_pe() calls; DATAFLOW cannot call mm_pe in a C for-loop",
        "- keep the exact extern C top name and parameter list",
        "- if INTERFACE uses one bundle=gmem for A/B/C, split to gmem0/gmem1/gmem2 (do not change ports)",
        f"- note: {rec.note}",
    ])
    if rec.tile_loop == "around_dataflow":
        lines.append(
            f"- CRITICAL: i-tile loop ({rec.i_tiles} trips) wraps DATAFLOW in the top function. "
            "mm_pe / load_B / drain_B have NO tile/i0 loop — one pe_kj of "
            f"{rec.pe_kj} beats per DATAFLOW firing. Ping-pong STREAM depth>=8. "
            f"Target latency ~{rec.pe_kj + (rec.i_tiles - 1) * rec.pe_kj} plus pipeline fill, "
            "not 2x pe_kj serialized inside tasks."
        )
    if rec.simd == 8 and rec.pack_bits == 256:
        lines.append(
            "- SIMD=8 pack: typedef ap_uint<256> vec8_bits; pack8/unpack8 eight floats "
            f"as 32-bit slices 0..255. pe_kj=(K/8)*J={rec.pe_kj}. 32 explicit mm_pe() calls."
        )
    if rec.data_kind != "float":
        lines.append(
            "- integer GEMM: do not emit `0.0f` or float pack/unpack; C is data_t accumulation"
        )
    return "\n".join(lines) + "\n"


def known_benches() -> tuple[str, ...]:
    return tuple(_RECIPES.keys())
