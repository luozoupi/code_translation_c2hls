from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from compact_pe_instantiate import instantiate_mm
from compact_pe_search import (
    candidate_id_io,
    enumerate_mm_io_recipes,
    enumerate_mm_mesh_recipes,
    enumerate_mm_pack_recipes,
    enumerate_mm_recipes,
    search_candidate_id,
)

REPO = Path(__file__).resolve().parents[1]

MUST_INCLUDE = (
    "io4_8x4_s8_k32_j32",
    "io4_8x4_s8_k32_j64",
    "io4_16x8_s8_k32_j64",
    "io5_8x4_s8_k32_j32",
    "io5_8x4_s8_k32_j64",
    "io5_16x8_s8_k64_j64",
)


def _by_id():
    return {candidate_id_io(r): r for r in enumerate_mm_io_recipes()}


def _report(dsp: int) -> dict:
    return {
        "dsp": dsp,
        "latency_cycles": 1200,
        "interval": 1100,
        "feedback": {"scopes": [{"name": "pe_kj", "pipeline_ii": 1}]},
    }


def test_io_grid_is_six_points_with_must_include_ids():
    recs = enumerate_mm_io_recipes()
    ids = [candidate_id_io(r) for r in recs]
    assert len(recs) == 6
    assert set(ids) == set(MUST_INCLUDE)
    for r in recs:
        assert r.pack_bits == 512
        assert r.tile_loop == "inside_tasks"
        assert r.i_tiles == 1
        assert r.pe == r.pe_i * r.pe_j
        assert r.expected_dsp == r.pe_i * r.pe_j * r.simd * 5
        assert r.pe_i * r.pe_j <= 128
        assert r.expected_dsp <= 0.85 * 9024
        assert r.simd == 8
        assert 64 % r.k_part == 0 and 64 % r.j_part == 0
        assert r.pe_i * r.lat_i == 64


def test_io5_cand9_analog_identities():
    r = _by_id()["io5_8x4_s8_k32_j32"]
    assert r.layout == "io5"
    assert r.pe_i == 8 and r.pe_j == 4 and r.simd == 8
    assert r.k_part == 32 and r.j_part == 32
    assert r.lat_i == 8 and r.lat_j == 4
    assert r.pe == 32 and r.expected_dsp == 1280
    assert r.pe_j * r.simd == r.k_part
    assert search_candidate_id(r) == "io5_8x4_s8_k32_j32"


def test_io4_cand5_analog_identities():
    r = _by_id()["io4_16x8_s8_k32_j64"]
    assert r.layout == "io4"
    assert r.pe_i == 16 and r.pe_j == 8
    assert r.k_part == 32 and r.j_part == 64
    assert r.lat_i == 4 and r.lat_j == 8
    assert r.pe == 128 and r.expected_dsp == 5120
    assert r.pe_j * r.lat_j == r.j_part
    assert search_candidate_id(r) == "io4_16x8_s8_k32_j64"


def test_io4_32pe_j_tile_identities():
    r = _by_id()["io4_8x4_s8_k32_j32"]
    assert r.layout == "io4"
    assert r.pe == 32 and r.expected_dsp == 1280
    assert r.lat_i == 8 and r.lat_j == 8
    assert r.pe_j * r.lat_j == r.j_part
    r64 = _by_id()["io4_8x4_s8_k32_j64"]
    assert r64.lat_j == 16 and r64.pe == 32


def test_io5_16x8_uses_k64_not_k32():
    ids = {candidate_id_io(r) for r in enumerate_mm_io_recipes()}
    assert "io5_16x8_s8_k32_j64" not in ids
    r = _by_id()["io5_16x8_s8_k64_j64"]
    assert r.k_part == 64 and r.pe_j * r.simd == r.k_part
    assert r.lat_i == 4 and r.lat_j == 4
    assert r.pe == 128 and r.expected_dsp == 5120


def test_search_candidate_id_io_does_not_steal_other_families():
    io5 = _by_id()["io5_8x4_s8_k32_j32"]
    assert search_candidate_id(io5) == "io5_8x4_s8_k32_j32"
    pack = next(r for r in enumerate_mm_pack_recipes() if r.pe_i == 8 and r.pe_j == 4 and r.simd == 8)
    assert search_candidate_id(pack) == "pack8x4_simd8"
    mesh = next(r for r in enumerate_mm_mesh_recipes() if r.pe_i == 8 and r.pe_j == 4 and r.simd == 8)
    assert search_candidate_id(mesh) == "mesh8x4_simd8"
    chain = next(r for r in enumerate_mm_recipes() if r.pe == 16 and r.simd == 4)
    assert search_candidate_id(chain) == "pe16_simd4"


def test_io4_8x4_emits_tiled_ping_pong_local_c():
    rec = _by_id()["io4_8x4_s8_k32_j32"]
    code = instantiate_mm(rec)
    assert "#define PE_I 8" in code
    assert "#define PE_J 4" in code
    assert "#define K_PART 32" in code
    assert "#define J_PART 32" in code
    assert "B_ping" in code and "B_pong" in code
    assert "local_A" in code
    assert "autosa_mm_pack(" in code
    assert "void kernel0(" not in code
    assert "kernel_kernel.cpp" not in code
    assert "PE_wrapper" not in code
    assert code.count("io_pe(") == 33
    assert "pack_pe(" not in code
    assert "mesh_pe(" not in code
    assert "mm_pe(" not in code
    assert "Crow" in code
    assert "fifo_C_in" not in code
    assert "t_j" in code and "t_k" in code
    assert "Bmem[J * K / 16]" not in code
    assert "#pragma HLS DATAFLOW" in code
    assert "for (int i0 = 0; i0 < I; i0 += PE_I)" not in code
    assert "for (int i0 = 0; i0 < I; i0 += PE_NUM)" not in code
    assert "ap_uint<512>" in code


def test_io5_8x4_emits_c_flow_and_32_io_pe():
    rec = _by_id()["io5_8x4_s8_k32_j32"]
    code = instantiate_mm(rec)
    assert code.count("io_pe(") == 33
    assert "#define PE_I 8" in code
    assert "#define PE_J 4" in code
    assert "#define K_PART 32" in code
    assert "fifo_C_in" in code and "fifo_C_out" in code
    assert "B_ping" in code and "B_pong" in code
    assert "autosa_mm_pack(" in code
    assert "void kernel0(" not in code
    assert "pack_pe(" not in code
    assert "PE_wrapper" not in code


def test_io5_b_replay_is_j_outer_ii_inner_matching_pe_flatten():
    """csim 20260831_io: io5 failed ~4000/4096 C errors because load_B was ii-outer
    while pe_kj used ii = t % LAT_I (j outer). Pairing A[ii] with the wrong B[j].
    """
    import post_flash_stream as pfs

    rec = _by_id()["io5_8x4_s8_k32_j32"]
    code = instantiate_mm(rec)
    assert "const int ii = t % LAT_I;" in code
    assert "const int j_loc = t / LAT_I;" in code
    load_b = pfs._extract_fn_body(code, "load_B")
    drain_b = pfs._extract_fn_body(code, "drain_B")
    scatter = (
        "for (int j_loc = 0; j_loc < J_PART; ++j_loc) {\n"
        "                for (int ii = 0; ii < LAT_I; ++ii) {"
    )
    assert scatter in load_b, load_b[-800:]
    assert scatter in drain_b, drain_b


def test_io4_16x8_has_128_io_pe():
    rec = _by_id()["io4_16x8_s8_k32_j64"]
    code = instantiate_mm(rec)
    assert code.count("io_pe(") == 129
    assert "#define PE_I 16" in code
    assert "#define PE_J 8" in code
    assert "#define K_PART 32" in code
    assert "#define J_PART 64" in code


def test_instantiated_io_passes_architecture_ok_for_recipe():
    import post_flash_stream as pfs

    rec4 = _by_id()["io4_8x4_s8_k32_j32"]
    rec5 = _by_id()["io5_8x4_s8_k32_j32"]
    assert pfs.architecture_ok_for_recipe(instantiate_mm(rec4), _report(1280), rec4) is True
    assert pfs.architecture_ok_for_recipe(instantiate_mm(rec5), _report(1280), rec5) is True


def test_io_architecture_ok_rejects_wrong_pe_count_and_kernel0():
    import post_flash_stream as pfs

    rec = _by_id()["io5_8x4_s8_k32_j32"]
    report = _report(1280)
    code = instantiate_mm(rec).replace("io_pe(", "pack_pe(", 1)
    assert pfs.architecture_ok_for_recipe(code, report, rec) is False
    k0 = instantiate_mm(rec).replace("autosa_mm_pack(", "kernel0(", 1)
    assert pfs.architecture_ok_for_recipe(k0, report, rec) is False


def test_validate_io_uses_packed_top_and_header(tmp_path, monkeypatch):
    from compact_pe_validate import validate_candidate

    rec = _by_id()["io5_8x4_s8_k32_j32"]
    captured = {}

    def fake_run(**kwargs):
        captured.update(kwargs)
        return {
            "synth": {
                "success": True,
                "report": {
                    "latency_cycles": 1800,
                    "interval": 1100,
                    "dsp": 1280,
                    "feedback": {"scopes": [{"name": "pe_kj", "pipeline_ii": 1}]},
                },
            },
            "csim": {"success": True},
            "cosim": None,
        }

    monkeypatch.setattr(
        "compact_pe_validate.compile_check_cpp", lambda *a, **k: (True, "")
    )
    monkeypatch.setattr("compact_pe_validate._run_synth_csim_cosim", fake_run)
    validate_candidate(
        rec, tmp_path, header_code="// scalar", testbench_code="int main(){}"
    )
    assert captured["top_function"] == "autosa_mm_pack"
    assert "autosa_mm_pack" in captured["header_code"]
    assert "A_t16" in captured["header_code"]
    result = json.loads((tmp_path / "io5_8x4_s8_k32_j32" / "result.json").read_text())
    assert result["cand_id"] == "io5_8x4_s8_k32_j32"
    assert result["layout"] == "io5"
    assert result["csynth_latency"] == 1800


def test_io_search_main_ranks_only_io(tmp_path, monkeypatch):
    from compact_pe_io_search_main import main

    recs = [
        _by_id()["io5_8x4_s8_k32_j32"],
        _by_id()["io4_16x8_s8_k32_j64"],
    ]
    monkeypatch.setattr(
        "compact_pe_io_search_main.enumerate_mm_io_recipes", lambda: recs
    )
    kernels = {
        "io5_8x4_s8_k32_j32": "// io5 cand9 analog\n",
        "io4_16x8_s8_k32_j64": "// io4 cand5 analog\n",
    }
    results = {
        "io5_8x4_s8_k32_j32": {
            "cand_id": "io5_8x4_s8_k32_j32",
            "hls_csim_pass": True,
            "hls_csynth_pass": True,
            "architecture_ok": True,
            "csynth_latency": 1846,
            "csynth_dsp": 1280,
            "reason": "",
        },
        "io4_16x8_s8_k32_j64": {
            "cand_id": "io4_16x8_s8_k32_j64",
            "hls_csim_pass": True,
            "hls_csynth_pass": True,
            "architecture_ok": True,
            "csynth_latency": 2033,
            "csynth_dsp": 5120,
            "reason": "",
        },
    }

    def _validate(rec, out_root, header_code, testbench_code, **_kwargs):
        cid = search_candidate_id(rec)
        cand_dir = Path(out_root) / cid
        cand_dir.mkdir(parents=True, exist_ok=True)
        (cand_dir / "kernel.cpp").write_text(kernels[cid], encoding="utf-8")
        row = dict(results[cid])
        (cand_dir / "result.json").write_text(
            json.dumps(row, indent=2) + "\n", encoding="utf-8"
        )
        return row

    monkeypatch.setattr("compact_pe_io_search_main.validate_candidate", _validate)
    out = tmp_path / "compact_pe_io_search_test"
    rc = main(["--stamp", "20260831_io", "--out", str(out)])
    assert rc == 0
    text = (out / "ranking.jsonl").read_text(encoding="utf-8")
    first = json.loads(text.splitlines()[0])
    assert first["cand_id"] == "io5_8x4_s8_k32_j32"
    assert first["queue_rank"] == 1
    assert (out / "selected.cpp").read_text(encoding="utf-8") == kernels["io5_8x4_s8_k32_j32"]
    assert "pack8x4_simd8" not in text
    assert "mesh8x4" not in text
    assert "pe16_simd4" not in text
    assert "autosa_mm_pack" in (out / "kernel.h").read_text()


def test_io_launcher_dry_run_prints_ids_and_exits_zero():
    script = REPO / "scripts" / "pc2" / "start_autosa_mm_io_search.sh"
    proc = subprocess.run(
        [str(script), "--dry-run"],
        cwd=str(REPO),
        capture_output=True,
        text=True,
        check=False,
    )
    stdout = proc.stdout or ""
    combined = stdout + (proc.stderr or "")
    assert proc.returncode == 0, combined
    for cid in MUST_INCLUDE:
        assert cid in stdout
    assert "compact_pe_io_search_" in stdout
    assert "mmio" in stdout
    assert "pack8x4_simd8" not in stdout
    assert "mesh8x4_simd8" not in stdout
    assert "pe16_simd4" not in stdout
    assert "Submitted batch job" not in combined
