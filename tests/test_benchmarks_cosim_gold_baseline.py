"""Gold baseline selection for benchmarks_cosim corpus."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "bench",
    sorted(p.name for p in (REPO / "benchmarks_cosim").glob("hlsfactory_*") if p.is_dir()),
)
def test_cosim_corpus_uses_hls_baseline_cosim_for_gold(bench: str) -> None:
    from c2hls import _ground_truth_candidates, _load_benchmark_inputs, _resolve_gold_baseline_file

    bench_dir = REPO / "benchmarks_cosim" / bench
    meta = json.loads((bench_dir / "metadata.json").read_text(encoding="utf-8"))
    cosim_file = meta.get("cosim_kernel_file", "hls_baseline_cosim.cpp")

    assert _resolve_gold_baseline_file(meta, bench_dir) == cosim_file

    inputs = _load_benchmark_inputs(str(bench_dir))
    assert inputs["ground_truth_code"] == (bench_dir / cosim_file).read_text()

    candidates = _ground_truth_candidates(inputs)
    assert len(candidates) == 1
    assert candidates[0]["file"] == cosim_file
    assert candidates[0]["code"] == inputs["ground_truth_code"]
    assert "baseline" in inputs["gt_variants"]
    assert inputs["gt_variants"]["baseline"] == inputs["ground_truth_code"]


def test_doitgen_legacy_hls_baseline_not_selected_for_cosim_corpus() -> None:
    from c2hls import _resolve_gold_baseline_file

    bench_dir = REPO / "benchmarks_cosim" / "hlsfactory_doitgen"
    meta = json.loads((bench_dir / "metadata.json").read_text(encoding="utf-8"))

    assert _resolve_gold_baseline_file(meta, bench_dir) == "hls_baseline_cosim.cpp"
    assert _resolve_gold_baseline_file(meta, bench_dir) != "hls_baseline.cpp"


def _normalize_algo(text: str) -> str:
    import re

    text = re.sub(r"//[^\n]*", "", text)
    text = re.sub(r"#pragma[^\n]*\n", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


@pytest.mark.parametrize(
    "bench",
    sorted(p.name for p in (REPO / "benchmarks_cosim").glob("hlsfactory_*") if p.is_dir()),
)
def test_cosim_gold_matches_cosim_baseline(bench: str) -> None:
    """Cosim oracle must be the same kernel as hls_baseline_cosim.cpp."""
    bench_dir = REPO / "benchmarks_cosim" / bench
    meta = json.loads((bench_dir / "metadata.json").read_text(encoding="utf-8"))
    if not meta.get("supports_cosim"):
        pytest.skip("no cosim")

    cosim_file = meta.get("cosim_kernel_file", "hls_baseline_cosim.cpp")
    gold_file = meta.get("gold_hls_source_file", "gold_hls_source.cpp")
    plain_file = meta.get("plain_c_file", "plain.cpp")

    cosim_body = (bench_dir / cosim_file).read_text()
    gold_body = (bench_dir / gold_file).read_text()
    # Generated gold has a short header; kernel body must match baseline cosim.
    assert gold_body.endswith(cosim_body if cosim_body.endswith("\n") else cosim_body + "\n")
    assert _normalize_algo(gold_body) == _normalize_algo(cosim_body)
    assert _normalize_algo((bench_dir / plain_file).read_text()) == _normalize_algo(cosim_body)
    assert meta.get("gold_derived_from") == cosim_file
    assert meta.get("provenance", {}).get("gold_derived_from_cosim_kernel") is True
