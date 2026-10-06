"""Stage-box ranked cosim: 43 buckets, next-best replicate on fail."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "pc2"))

from autosa_mm_variant_sweep import build_manifest  # noqa: E402


def _write(path: Path, text: str | dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(text, dict):
        path.write_text(json.dumps(text), encoding="utf-8")
    else:
        path.write_text(text, encoding="utf-8")


def _stage_for(family: str) -> str:
    from autosa_mm_stage_ranked_cosim import stage_for_family

    return stage_for_family(family)


def _fake_harvest(cells: list[dict], *, latency_fn) -> list[dict]:
    rows: list[dict] = []
    for cell in cells:
        family = cell["family"]
        if family == "mem":
            continue
        stage = _stage_for(family)
        lat, dsp = latency_fn(cell)
        rows.append(
            {
                "family": family,
                "skill": cell["skill"],
                "floor": int(cell["floor"]),
                "dse": cell.get("dse") or "",
                "stream": int(cell.get("stream") or 0),
                "enf": int(cell.get("enf") or 0),
                "rep": int(cell["rep"]),
                "stage": stage,
                "latency_cycles": lat,
                "dsp": dsp,
                "campaign_root": cell["campaign_root"],
            }
        )
        if family == "stream":
            rows.append(
                {
                    "family": family,
                    "skill": cell["skill"],
                    "floor": int(cell["floor"]),
                    "dse": cell.get("dse") or "",
                    "stream": 1,
                    "enf": 0,
                    "rep": int(cell["rep"]),
                    "stage": "flash",
                    "latency_cycles": 1,
                    "dsp": 1,
                    "campaign_root": cell["campaign_root"],
                }
            )
    return rows


def test_build_buckets_is_43_non_mem_boxes():
    from autosa_mm_stage_ranked_cosim import build_buckets, bucket_id

    cells = build_manifest(reps=2, date="20260919", repo=REPO)
    harvest = _fake_harvest(
        cells, latency_fn=lambda c: (1000 + int(c["rep"]), 10)
    )
    buckets = build_buckets(cells, harvest, require_kernel=False)
    assert len(buckets) == 43
    ids = [b["bucket_id"] for b in buckets]
    assert len(ids) == len(set(ids))
    families = {b["family"] for b in buckets}
    assert families == {"flash", "oneshot", "dse_v1", "dse_v2", "stream", "enf"}
    assert sum(1 for b in buckets if b["family"] == "flash") == 6
    assert sum(1 for b in buckets if b["family"] == "dse_v1") == 6
    assert sum(1 for b in buckets if b["family"] == "dse_v2") == 6
    assert sum(1 for b in buckets if b["family"] == "stream") == 12
    assert sum(1 for b in buckets if b["family"] == "enf") == 12
    assert sum(1 for b in buckets if b["family"] == "oneshot") == 1
    assert "oneshot" in ids
    assert "flash_ns_f0" in ids
    assert "stream_dse2_gf_f1" in ids
    assert all(len(b["candidates"]) == 2 for b in buckets)
    oneshot = next(b for b in buckets if b["bucket_id"] == "oneshot")
    assert bucket_id(oneshot["candidates"][0]) == "oneshot"


def test_stream_ranked_on_stream_latency_not_copied_flash():
    from autosa_mm_stage_ranked_cosim import build_buckets

    cells = [
        {
            "cell_id": "stream_dse1_ns_f0_r01",
            "family": "stream",
            "skill": "noskills",
            "floor": 0,
            "dse": "v1",
            "stream": 1,
            "enf": 0,
            "rep": 1,
            "campaign_root": "/tmp/stream_r01_camp",
        },
        {
            "cell_id": "stream_dse1_ns_f0_r02",
            "family": "stream",
            "skill": "noskills",
            "floor": 0,
            "dse": "v1",
            "stream": 1,
            "enf": 0,
            "rep": 2,
            "campaign_root": "/tmp/stream_r02_camp",
        },
    ]
    harvest = [
        {
            "family": "stream",
            "skill": "noskills",
            "floor": 0,
            "dse": "v1",
            "stream": 1,
            "enf": 0,
            "rep": 1,
            "stage": "flash",
            "latency_cycles": 1,
            "dsp": 1,
            "campaign_root": "/tmp/stream_r01_camp",
        },
        {
            "family": "stream",
            "skill": "noskills",
            "floor": 0,
            "dse": "v1",
            "stream": 1,
            "enf": 0,
            "rep": 1,
            "stage": "stream",
            "latency_cycles": 5000,
            "dsp": 320,
            "campaign_root": "/tmp/stream_r01_camp",
        },
        {
            "family": "stream",
            "skill": "noskills",
            "floor": 0,
            "dse": "v1",
            "stream": 1,
            "enf": 0,
            "rep": 2,
            "stage": "flash",
            "latency_cycles": 9,
            "dsp": 1,
            "campaign_root": "/tmp/stream_r02_camp",
        },
        {
            "family": "stream",
            "skill": "noskills",
            "floor": 0,
            "dse": "v1",
            "stream": 1,
            "enf": 0,
            "rep": 2,
            "stage": "stream",
            "latency_cycles": 1218,
            "dsp": 320,
            "campaign_root": "/tmp/stream_r02_camp",
        },
    ]
    buckets = build_buckets(cells, harvest, require_kernel=False)
    assert len(buckets) == 1
    ranked = buckets[0]["candidates"]
    assert ranked[0]["cell_id"] == "stream_dse1_ns_f0_r02"
    assert ranked[0]["latency_cycles"] == 1218
    assert ranked[1]["cell_id"] == "stream_dse1_ns_f0_r01"
    assert ranked[1]["latency_cycles"] == 5000


def test_kernel_resolution_per_family(tmp_path: Path):
    from autosa_mm_stage_ranked_cosim import resolve_kernel_cpp

    def _cell(name: str, files: dict[str, str]) -> Path:
        cell = tmp_path / name / "variants" / "v" / "autosa_mm" / "setup"
        for fname, body in files.items():
            _write(cell / fname, body)
        return tmp_path / name

    flash = _cell(
        "flash_camp",
        {
            "autosa_mm_flash_opt.cpp": "flash_opt",
            "autosa_mm_flash_opt_report.json": "{}",
            "autosa_mm_selected.cpp": "selected",
        },
    )
    dse1 = _cell(
        "dse1_camp",
        {
            "autosa_mm_dse.cpp": "dse",
            "autosa_mm_dse_report.json": "{}",
            "autosa_mm_selected.cpp": "selected",
        },
    )
    dse2 = _cell(
        "dse2_camp",
        {
            "autosa_mm_selected.cpp": "dse2_selected",
            "autosa_mm_selected_report.json": "{}",
            "autosa_mm_flash_opt.cpp": "flash_copy",
        },
    )
    stream = _cell(
        "stream_camp",
        {
            "autosa_mm_stream.cpp": "stream",
            "autosa_mm_stream_report.json": "{}",
            "autosa_mm_selected.cpp": "selected",
        },
    )
    enf = _cell(
        "enf_camp",
        {
            "autosa_mm_selected.cpp": "enf_selected",
            "autosa_mm_enforcement.json": "{}",
            "autosa_mm_flash_opt.cpp": "seed",
        },
    )
    oneshot = _cell(
        "oneshot_camp",
        {
            "autosa_mm_selected.cpp": "oneshot_sel",
            "autosa_mm_selected_report.json": "{}",
        },
    )
    assert resolve_kernel_cpp(flash, "flash").read_text() == "flash_opt"
    assert resolve_kernel_cpp(dse1, "dse_v1").read_text() == "dse"
    assert resolve_kernel_cpp(dse2, "dse_v2").read_text() == "dse2_selected"
    assert resolve_kernel_cpp(stream, "stream").read_text() == "stream"
    assert resolve_kernel_cpp(enf, "enf").read_text() == "enf_selected"
    assert resolve_kernel_cpp(oneshot, "oneshot").read_text() == "oneshot_sel"


def test_refuse_frozen_write_blocks_20260918_allows_sibling():
    from autosa_mm_stage_ranked_cosim import refuse_frozen_write

    with pytest.raises(ValueError, match="frozen"):
        refuse_frozen_write(
            "artifacts/pc2/autosa_mm_variant_sweep_20260918/cells/flash_ns_f0_r01_camp"
        )
    with pytest.raises(ValueError, match="frozen"):
        refuse_frozen_write(
            "/tmp/autosa_mm_variant_sweep_20260918_cosim/result.json"
        )
    refuse_frozen_write("artifacts/pc2/autosa_mm_stage_ranked_cosim_20260921")
    refuse_frozen_write("artifacts/pc2/autosa_mm_variant_sweep_20260919/cells/x")


def _make_ranked_bucket(tmp_path: Path) -> dict:
    camp1 = tmp_path / "r01_camp"
    camp2 = tmp_path / "r02_camp"
    cell1 = camp1 / "variants" / "v" / "autosa_mm" / "setup"
    cell2 = camp2 / "variants" / "v" / "autosa_mm" / "setup"
    _write(cell1 / "autosa_mm_flash_opt.cpp", "rank1")
    _write(cell1 / "autosa_mm_flash_opt_report.json", {"latency_cycles": 100})
    _write(cell2 / "autosa_mm_flash_opt.cpp", "rank2")
    _write(cell2 / "autosa_mm_flash_opt_report.json", {"latency_cycles": 200})
    return {
        "bucket_id": "flash_ns_f0",
        "family": "flash",
        "skill": "noskills",
        "floor": 0,
        "dse": "",
        "candidates": [
            {
                "cell_id": "flash_ns_f0_r01",
                "family": "flash",
                "campaign_root": str(camp1),
                "latency_cycles": 100,
                "dsp": 10,
                "code_path": str(cell1 / "autosa_mm_flash_opt.cpp"),
            },
            {
                "cell_id": "flash_ns_f0_r02",
                "family": "flash",
                "campaign_root": str(camp2),
                "latency_cycles": 200,
                "dsp": 10,
                "code_path": str(cell2 / "autosa_mm_flash_opt.cpp"),
            },
        ],
    }


def test_walks_to_second_on_fail(tmp_path: Path):
    from autosa_mm_stage_ranked_cosim import run_bucket_cosim

    out = tmp_path / "autosa_mm_stage_ranked_cosim_20260921"
    bucket = _make_ranked_bucket(tmp_path)
    calls: list[str] = []

    def _fake_cosim(cell_obj, run_root, *, force=False, dry_run=False):
        calls.append(Path(cell_obj.final_cpp).read_text())
        if calls[-1] == "rank1":
            return {"status": "fail", "passed": False, "error": "boom"}
        return {"status": "pass", "passed": True}

    result = run_bucket_cosim(
        bucket=bucket,
        out_root=out,
        cosim_fn=_fake_cosim,
        force=True,
    )
    assert result["status"] == "pass"
    assert result["winner_id"] == "flash_ns_f0_r02"
    assert calls == ["rank1", "rank2"]
    result_path = out / "buckets" / "flash_ns_f0" / "result.json"
    ranking_path = out / "buckets" / "flash_ns_f0" / "ranking.json"
    assert result_path.is_file()
    assert ranking_path.is_file()
    frozen_camp = Path(bucket["candidates"][0]["campaign_root"])
    assert not any(frozen_camp.rglob("*cosim*"))


def test_run_bucket_refuses_frozen_out_root(tmp_path: Path):
    from autosa_mm_stage_ranked_cosim import run_bucket_cosim

    frozen_out = tmp_path / "autosa_mm_variant_sweep_20260918"
    bucket = _make_ranked_bucket(tmp_path)
    with pytest.raises(ValueError, match="frozen"):
        run_bucket_cosim(
            bucket=bucket,
            out_root=frozen_out,
            cosim_fn=lambda *a, **k: {"status": "pass", "passed": True},
        )


def test_parallel_candidates_reduce_keeps_lowest_rank_pass(tmp_path: Path):
    from autosa_mm_stage_ranked_cosim import reduce_bucket_result, run_candidate_cosim

    out = tmp_path / "autosa_mm_stage_ranked_cosim_20260921"
    bucket = _make_ranked_bucket(tmp_path)
    calls: list[str] = []

    def _fake_cosim(cell_obj, run_root, *, force=False, dry_run=False):
        calls.append(Path(cell_obj.final_cpp).read_text())
        return {"status": "pass", "passed": True}

    rank2 = run_candidate_cosim(
        bucket=bucket,
        cell_id="flash_ns_f0_r02",
        out_root=out,
        cosim_fn=_fake_cosim,
        force=True,
    )
    assert rank2["status"] == "pass"
    assert rank2["winner_id"] == "flash_ns_f0_r02"
    rank1 = run_candidate_cosim(
        bucket=bucket,
        cell_id="flash_ns_f0_r01",
        out_root=out,
        cosim_fn=_fake_cosim,
        force=True,
    )
    assert rank1["winner_id"] == "flash_ns_f0_r01"
    reduced = reduce_bucket_result(bucket=bucket, out_root=out)
    assert reduced["winner_id"] == "flash_ns_f0_r01"
    assert calls == ["rank2", "rank1"]
    cand_dir = out / "buckets" / "flash_ns_f0" / "candidates"
    assert (cand_dir / "flash_ns_f0_r01" / "result.json").is_file()
    assert (cand_dir / "flash_ns_f0_r02" / "result.json").is_file()


def test_remaining_candidates_skip_already_attempted(tmp_path: Path):
    from autosa_mm_stage_ranked_cosim import remaining_candidates, run_bucket_cosim

    out = tmp_path / "autosa_mm_stage_ranked_cosim_20260921"
    bucket = _make_ranked_bucket(tmp_path)

    def _fake_cosim(cell_obj, run_root, *, force=False, dry_run=False):
        return {"status": "pass", "passed": True}

    run_bucket_cosim(bucket=bucket, out_root=out, cosim_fn=_fake_cosim, force=True)
    left = remaining_candidates([bucket], out)
    ids = [c["cell_id"] for c in left]
    assert ids == ["flash_ns_f0_r02"]


def test_stage_bench_enables_cosim_and_copies_plain(tmp_path: Path):
    from autosa_mm_stage_ranked_cosim import refuse_frozen_write, stage_autosa_mm_bench

    src = tmp_path / "src_mm"
    _write(src / "kernel.h", "#define I 64\n")
    _write(src / "testbench.cpp", "int main(){}\n")
    _write(src / "plain.cpp", "void autosa_mm(){}\n")
    _write(
        src / "metadata.json",
        {"benchmark": "autosa_mm", "hls_top": "autosa_mm", "supports_cosim": False},
    )
    out = tmp_path / "autosa_mm_stage_ranked_cosim_20260921"
    dest = stage_autosa_mm_bench(out, source=src)
    meta = json.loads((dest / "metadata.json").read_text(encoding="utf-8"))
    assert meta["supports_cosim"] is True
    assert meta["cosim_testbench_file"] == "testbench.cpp"
    assert (dest / "kernel.h").is_file()
    assert (dest / "testbench.cpp").is_file()
    assert (dest / "plain.cpp").is_file()
    with pytest.raises(ValueError, match="frozen"):
        refuse_frozen_write(tmp_path / "autosa_mm_variant_sweep_20260918" / "bench")
