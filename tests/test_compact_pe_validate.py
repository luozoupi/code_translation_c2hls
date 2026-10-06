# tests/test_compact_pe_validate.py
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from compact_pe_instantiate import instantiate_mm
from compact_pe_search import candidate_id, enumerate_mm_recipes
from compact_pe_validate import validate_candidate
from post_flash_stream import architecture_ok_for_recipe


def test_validate_writes_result_json(tmp_path, monkeypatch):
    rec = next(r for r in enumerate_mm_recipes() if candidate_id(r) == "pe16_simd4")

    def fake_run(**_kwargs):
        return {
            "synth": {
                "success": True,
                "report": {
                    "latency_cycles": 4292,
                    "interval": 4173,
                    "dsp": 320,
                    "feedback": {"scopes": [{"name": "pe_kj", "pipeline_ii": 1}]},
                },
            },
            "csim": {"success": True},
            "cosim": None,
        }

    monkeypatch.setattr("compact_pe_validate._run_synth_csim_cosim", fake_run)
    monkeypatch.setattr(
        "compact_pe_validate.compile_check_cpp", lambda *a, **k: (True, "")
    )
    out = validate_candidate(rec, tmp_path, header_code="//h", testbench_code="int main(){}")
    result = json.loads((tmp_path / "pe16_simd4" / "result.json").read_text())
    assert result["hls_csim_pass"] is True
    assert result["hls_csynth_pass"] is True
    assert result["csynth_latency"] == 4292
    assert out["csynth_latency"] == 4292


def test_architecture_ok_for_recipe_uses_candidate_not_env(monkeypatch):
    rec = next(r for r in enumerate_mm_recipes() if candidate_id(r) == "pe32_simd8")
    code = instantiate_mm(rec)
    report = {
        "dsp": 1280,
        "latency_cycles": 1200,
        "interval": 1100,
        "feedback": {"scopes": [{"name": "pe_kj", "pipeline_ii": 1}]},
    }
    monkeypatch.setenv("C2HLS_PE_RECIPE", "autosa_mm")
    assert architecture_ok_for_recipe(code, report, rec) is True


def test_validate_skips_existing_result_json(tmp_path, monkeypatch):
    rec = next(r for r in enumerate_mm_recipes() if candidate_id(r) == "pe16_simd4")
    cand_dir = tmp_path / "pe16_simd4"
    cand_dir.mkdir()
    existing = {
        "cand_id": "pe16_simd4",
        "hls_csim_pass": True,
        "hls_csynth_pass": True,
        "csynth_latency": 1,
    }
    (cand_dir / "result.json").write_text(json.dumps(existing), encoding="utf-8")

    def boom(**_kwargs):
        raise AssertionError("must not re-run HLS")

    monkeypatch.setattr("compact_pe_validate._run_synth_csim_cosim", boom)
    monkeypatch.setattr(
        "compact_pe_validate.compile_check_cpp", lambda *a, **k: (True, "")
    )
    out = validate_candidate(rec, tmp_path, header_code="//h", testbench_code="int main(){}")
    assert out["csynth_latency"] == 1
    assert json.loads((cand_dir / "result.json").read_text())["csynth_latency"] == 1


def test_validate_mesh_writes_mesh_id_dir(tmp_path, monkeypatch):
    from compact_pe_validate import validate_candidate
    from compact_pe_search import enumerate_mm_mesh_recipes, candidate_id_mesh

    rec = next(
        r for r in enumerate_mm_mesh_recipes() if candidate_id_mesh(r) == "mesh8x4_simd8"
    )

    def fake_run(**_kwargs):
        return {
            "synth": {
                "success": True,
                "report": {
                    "latency_cycles": 2000,
                    "interval": 1100,
                    "dsp": 1280,
                    "feedback": {"scopes": [{"name": "pe_kj", "pipeline_ii": 1}]},
                },
            },
            "csim": {"success": True},
            "cosim": None,
        }

    monkeypatch.setattr("compact_pe_validate.compile_check_cpp", lambda *a, **k: (True, ""))
    monkeypatch.setattr("compact_pe_validate._run_synth_csim_cosim", fake_run)
    validate_candidate(rec, tmp_path, header_code="//h", testbench_code="int main(){}")
    result = json.loads((tmp_path / "mesh8x4_simd8" / "result.json").read_text())
    assert result["cand_id"] == "mesh8x4_simd8"
    assert result["layout"] == "mesh"
    assert result["pe_i"] == 8 and result["pe_j"] == 4
    assert result["hls_csim_pass"] is True
    assert result["csynth_latency"] == 2000
