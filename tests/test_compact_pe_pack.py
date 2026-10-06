from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from compact_pe_instantiate import instantiate_mm
from compact_pe_search import (
    candidate_id_pack,
    enumerate_mm_mesh_recipes,
    enumerate_mm_pack_recipes,
    enumerate_mm_recipes,
    search_candidate_id,
)

REPO = Path(__file__).resolve().parents[1]


def test_pack_grid_includes_autosa_analogs():
    recs = {candidate_id_pack(r): r for r in enumerate_mm_pack_recipes()}
    r = recs["pack8x4_simd8"]
    assert r.layout == "pack"
    assert r.pe_i == 8 and r.pe_j == 4 and r.simd == 8
    assert r.pe == 32 and r.pack_bits == 512
    assert r.expected_dsp == 1280
    assert r.pe_kj == (64 // 8) * (64 // 4)
    assert r.i_tiles == 1
    assert r.tile_loop == "inside_tasks"
    r5 = recs["pack16x8_simd8"]
    assert r5.pe == 128 and r5.expected_dsp == 5120
    assert r5.pe_i == 16 and r5.pe_j == 8
    assert r5.pack_bits == 512


def test_pack_caps_match_mesh_grid():
    pack = enumerate_mm_pack_recipes()
    mesh = enumerate_mm_mesh_recipes()
    assert len(pack) == len(mesh)
    for r in pack:
        assert r.pe_i * r.pe_j <= 128
        assert r.expected_dsp <= 0.85 * 9024
        assert r.layout == "pack"
        assert r.pack_bits == 512
        assert 64 % r.pe_i == 0 and 64 % r.pe_j == 0 and 64 % r.simd == 0
    ids = {candidate_id_pack(r) for r in pack}
    assert "pack32x16_simd2" not in ids
    assert 20 <= len(pack) <= 40


def test_search_candidate_id_pack_dispatch():
    pack = next(
        r for r in enumerate_mm_pack_recipes() if candidate_id_pack(r) == "pack8x4_simd8"
    )
    assert search_candidate_id(pack) == "pack8x4_simd8"
    chain = next(r for r in enumerate_mm_recipes() if r.pe == 16 and r.simd == 4)
    assert search_candidate_id(chain) == "pe16_simd4"


def test_pack_8x4_has_32_pack_pe_and_512bit_axi():
    rec = next(
        r for r in enumerate_mm_pack_recipes() if candidate_id_pack(r) == "pack8x4_simd8"
    )
    code = instantiate_mm(rec)
    assert "#define PE_I 8" in code
    assert "#define PE_J 4" in code
    assert "ap_uint<512>" in code
    assert "autosa_mm_pack(" in code
    assert "void kernel0(" not in code
    assert "kernel_kernel.cpp" not in code
    assert code.count("pack_pe(") == 33
    assert "mesh_pe(" not in code
    assert "mm_pe(" not in code
    assert "#pragma HLS DATAFLOW" in code
    assert "for (int i0 = 0; i0 < I; i0 += PE_I)" not in code
    assert "for (int i0 = 0; i0 < I; i0 += PE_NUM)" not in code


def test_pack_16x8_has_128_pack_pe():
    rec = next(
        r for r in enumerate_mm_pack_recipes() if candidate_id_pack(r) == "pack16x8_simd8"
    )
    code = instantiate_mm(rec)
    assert code.count("pack_pe(") == 129
    assert "#define PE_I 16" in code
    assert "#define PE_J 8" in code
    assert "ap_uint<512>" in code


def test_instantiated_pack_8x4_passes_architecture_ok_for_recipe():
    import post_flash_stream as pfs

    rec = next(
        r for r in enumerate_mm_pack_recipes() if candidate_id_pack(r) == "pack8x4_simd8"
    )
    code = instantiate_mm(rec)
    report = {
        "dsp": 1280,
        "latency_cycles": 1200,
        "interval": 1100,
        "feedback": {"scopes": [{"name": "pe_kj", "pipeline_ii": 1}]},
    }
    assert pfs.architecture_ok_for_recipe(code, report, rec) is True


def test_pack_architecture_ok_rejects_wrong_pe_count_and_kernel0():
    import post_flash_stream as pfs

    rec = next(
        r for r in enumerate_mm_pack_recipes() if candidate_id_pack(r) == "pack8x4_simd8"
    )
    report = {
        "dsp": 1280,
        "latency_cycles": 1200,
        "interval": 1100,
        "feedback": {"scopes": [{"name": "pe_kj", "pipeline_ii": 1}]},
    }
    code = instantiate_mm(rec).replace("pack_pe(", "mesh_pe(", 1)
    assert pfs.architecture_ok_for_recipe(code, report, rec) is False
    k0 = instantiate_mm(rec).replace("autosa_mm_pack(", "kernel0(", 1)
    assert pfs.architecture_ok_for_recipe(k0, report, rec) is False


def test_validate_pack_uses_packed_top_and_header(tmp_path, monkeypatch):
    from compact_pe_validate import validate_candidate

    rec = next(
        r for r in enumerate_mm_pack_recipes() if candidate_id_pack(r) == "pack8x4_simd8"
    )
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
    result = json.loads((tmp_path / "pack8x4_simd8" / "result.json").read_text())
    assert result["cand_id"] == "pack8x4_simd8"
    assert result["layout"] == "pack"
    assert result["csynth_latency"] == 1800


def test_pack_search_main_ranks_only_pack(tmp_path, monkeypatch):
    from compact_pe_pack_search_main import main

    recs = [
        r
        for r in enumerate_mm_pack_recipes()
        if candidate_id_pack(r) in ("pack8x4_simd8", "pack16x8_simd8")
    ]
    monkeypatch.setattr(
        "compact_pe_pack_search_main.enumerate_mm_pack_recipes", lambda: recs
    )
    kernels = {
        "pack8x4_simd8": "// pack 8x4\n",
        "pack16x8_simd8": "// pack 16x8\n",
    }
    results = {
        "pack8x4_simd8": {
            "cand_id": "pack8x4_simd8",
            "hls_csim_pass": True,
            "hls_csynth_pass": True,
            "architecture_ok": True,
            "csynth_latency": 1846,
            "csynth_dsp": 1280,
            "reason": "",
        },
        "pack16x8_simd8": {
            "cand_id": "pack16x8_simd8",
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

    monkeypatch.setattr("compact_pe_pack_search_main.validate_candidate", _validate)
    out = tmp_path / "compact_pe_pack_search_test"
    rc = main(["--stamp", "20260831_pack", "--out", str(out)])
    assert rc == 0
    lines = (out / "ranking.jsonl").read_text(encoding="utf-8").splitlines()
    first = json.loads(lines[0])
    assert first["cand_id"] == "pack8x4_simd8"
    assert first["queue_rank"] == 1
    assert (out / "selected.cpp").read_text(encoding="utf-8") == kernels["pack8x4_simd8"]
    assert "pe16_simd4" not in (out / "ranking.jsonl").read_text()
    assert "mesh8x4" not in (out / "ranking.jsonl").read_text()
    assert "autosa_mm_pack" in (out / "kernel.h").read_text()


def test_pack_launcher_dry_run_prints_ids_and_exits_zero():
    script = REPO / "scripts" / "pc2" / "start_autosa_mm_pack_search.sh"
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
    assert "pack8x4_simd8" in stdout
    assert "pack16x8_simd8" in stdout
    assert "compact_pe_pack_search_" in stdout
    assert "mmpack" in stdout
    assert "pe16_simd4" not in stdout
    assert "mesh8x4_simd8" not in stdout
    assert "Submitted batch job" not in combined


def test_family_a_launcher_does_not_print_pack_ids():
    script = REPO / "scripts" / "pc2" / "start_autosa_mm_pe_search.sh"
    proc = subprocess.run(
        [str(script), "--dry-run"],
        cwd=str(REPO),
        capture_output=True,
        text=True,
        check=False,
    )
    stdout = proc.stdout or ""
    assert proc.returncode == 0, stdout + (proc.stderr or "")
    assert "pack8x4_simd8" not in stdout
    assert "pe16_simd4" in stdout
