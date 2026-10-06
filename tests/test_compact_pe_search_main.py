# tests/test_compact_pe_search_main.py
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from compact_pe_search import candidate_id, enumerate_mm_recipes, search_candidate_id
from compact_pe_search_main import main

REPO = Path(__file__).resolve().parents[1]


def _recs(*ids: str):
    by_id = {candidate_id(r): r for r in enumerate_mm_recipes()}
    return [by_id[i] for i in ids]


def _pass_row(cand_id: str, latency: int, dsp: int) -> dict:
    return {
        "cand_id": cand_id,
        "hls_csim_pass": True,
        "hls_csynth_pass": True,
        "architecture_ok": True,
        "csynth_latency": latency,
        "csynth_dsp": dsp,
        "reason": "",
    }


def _fail_csim_row(cand_id: str) -> dict:
    return {
        "cand_id": cand_id,
        "hls_csim_pass": False,
        "hls_csynth_pass": True,
        "architecture_ok": True,
        "csynth_latency": 100,
        "csynth_dsp": 80,
        "reason": "csim failed",
    }


def _fake_validate(kernels: dict[str, str], results: dict[str, dict]):
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

    return _validate


def test_main_ranks_lower_latency_first_and_copies_selected(tmp_path, monkeypatch):
    recs = _recs("pe16_simd4", "pe32_simd8")
    monkeypatch.setattr("compact_pe_search_main.enumerate_mm_recipes", lambda: recs)
    monkeypatch.setattr("compact_pe_search_main.enumerate_mm_mesh_recipes", lambda: [])
    kernels = {
        "pe16_simd4": "// kernel pe16_simd4\n",
        "pe32_simd8": "// kernel pe32_simd8\n",
    }
    results = {
        "pe16_simd4": _pass_row("pe16_simd4", 4292, 320),
        "pe32_simd8": _pass_row("pe32_simd8", 4583, 1280),
    }
    monkeypatch.setattr(
        "compact_pe_search_main.validate_candidate",
        _fake_validate(kernels, results),
    )

    header = tmp_path / "kernel.h"
    tb = tmp_path / "testbench.cpp"
    header.write_text("//h\n", encoding="utf-8")
    tb.write_text("int main(){}\n", encoding="utf-8")
    out = tmp_path / "compact_pe_search_test"

    rc = main(
        [
            "--stamp",
            "20260830_pesearch",
            "--out",
            str(out),
            "--header",
            str(header),
            "--testbench",
            str(tb),
        ]
    )
    assert rc == 0

    ranking_path = out / "ranking.jsonl"
    lines = ranking_path.read_text(encoding="utf-8").splitlines()
    assert lines
    first = json.loads(lines[0])
    assert first["cand_id"] == "pe16_simd4"
    assert first["queue_rank"] == 1
    assert first["csynth_latency"] == 4292

    selected = out / "selected.cpp"
    assert selected.is_file()
    assert selected.read_text(encoding="utf-8") == kernels["pe16_simd4"]
    report = json.loads((out / "selected_report.json").read_text(encoding="utf-8"))
    assert report["cand_id"] == "pe16_simd4"
    assert report["queue_rank"] == 1


def test_main_omits_csim_fail_from_ranking(tmp_path, monkeypatch):
    recs = _recs("pe16_simd4", "pe32_simd8", "pe8_simd2")
    monkeypatch.setattr("compact_pe_search_main.enumerate_mm_recipes", lambda: recs)
    monkeypatch.setattr("compact_pe_search_main.enumerate_mm_mesh_recipes", lambda: [])
    kernels = {
        "pe16_simd4": "// kernel pe16_simd4\n",
        "pe32_simd8": "// kernel pe32_simd8\n",
        "pe8_simd2": "// kernel pe8_simd2\n",
    }
    results = {
        "pe16_simd4": _pass_row("pe16_simd4", 4292, 320),
        "pe32_simd8": _pass_row("pe32_simd8", 4583, 1280),
        "pe8_simd2": _fail_csim_row("pe8_simd2"),
    }
    monkeypatch.setattr(
        "compact_pe_search_main.validate_candidate",
        _fake_validate(kernels, results),
    )

    header = tmp_path / "kernel.h"
    tb = tmp_path / "testbench.cpp"
    header.write_text("//h\n", encoding="utf-8")
    tb.write_text("int main(){}\n", encoding="utf-8")
    out = tmp_path / "out"

    rc = main(
        [
            "--stamp",
            "omit_fail",
            "--out",
            str(out),
            "--header",
            str(header),
            "--testbench",
            str(tb),
        ]
    )
    assert rc == 0
    ranked_ids = [
        json.loads(line)["cand_id"]
        for line in (out / "ranking.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert "pe8_simd2" not in ranked_ids
    assert ranked_ids[0] == "pe16_simd4"


def test_main_ranks_mesh_ahead_of_slower_chain(tmp_path, monkeypatch):
    from compact_pe_search import enumerate_mm_mesh_recipes, candidate_id_mesh

    chain = _recs("pe16_simd4")
    mesh = [
        r for r in enumerate_mm_mesh_recipes() if candidate_id_mesh(r) == "mesh8x4_simd8"
    ]
    monkeypatch.setattr("compact_pe_search_main.enumerate_mm_recipes", lambda: chain)
    monkeypatch.setattr("compact_pe_search_main.enumerate_mm_mesh_recipes", lambda: mesh)
    kernels = {
        "pe16_simd4": "// chain\n",
        "mesh8x4_simd8": "// kernel mesh\n",
    }
    results = {
        "pe16_simd4": _pass_row("pe16_simd4", 4292, 320),
        "mesh8x4_simd8": _pass_row("mesh8x4_simd8", 2000, 1280),
    }
    monkeypatch.setattr(
        "compact_pe_search_main.validate_candidate",
        _fake_validate(kernels, results),
    )
    header = tmp_path / "kernel.h"
    tb = tmp_path / "testbench.cpp"
    header.write_text("//h\n")
    tb.write_text("int main(){}\n")
    out = tmp_path / "out"
    rc = main(["--stamp", "mesh", "--out", str(out), "--header", str(header), "--testbench", str(tb)])
    assert rc == 0
    first = json.loads((out / "ranking.jsonl").read_text().splitlines()[0])
    assert first["cand_id"] == "mesh8x4_simd8"
    assert (out / "selected.cpp").read_text() == kernels["mesh8x4_simd8"]


def test_launcher_dry_run_prints_ids_and_exits_zero():
    script = REPO / "scripts" / "pc2" / "start_autosa_mm_pe_search.sh"
    proc = subprocess.run(
        [str(script), "--dry-run"],
        cwd=str(REPO),
        capture_output=True,
        text=True,
        check=False,
    )
    stdout = proc.stdout or ""
    stderr = proc.stderr or ""
    combined = stdout + stderr
    assert proc.returncode == 0, combined
    assert "pe16_simd4" in stdout
    assert "pe32_simd8" in stdout
    assert "mesh8x4_simd8" in stdout
    assert "mesh16x8_simd8" in stdout
    assert "compact_pe_search_" in stdout
    assert "Submitted batch job" not in combined
