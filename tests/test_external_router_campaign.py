from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import pytest

from scripts.augment_setup_router_corpus import _external_records
from scripts.freeze_external_phase_b_manifest import (
    _load_split,
    _select_entries,
)
from scripts.run_corrected_setup_matrix import (
    _entry_metadata,
    _select_benchmarks,
)
from setup_router import setup_registry


def test_external_split_rejects_lineage_leakage(tmp_path):
    path = tmp_path / "split.json"
    path.write_text(
        json.dumps(
            {
                "entries": {
                    "train_case": {
                        "problem": "train",
                        "benchmark_lineage": "machsuite:bfs",
                        "split": "train",
                        "representative": True,
                    },
                    "test_case": {
                        "problem": "test",
                        "benchmark_lineage": "machsuite:bfs",
                        "split": "test",
                        "representative": True,
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="lineage leakage"):
        _load_split(path)


def test_external_selection_excludes_paired_holdouts(tmp_path):
    path = tmp_path / "split.json"
    path.write_text(
        json.dumps(
            {
                "entries": {
                    "hlseval_machsuite_bfs_queue": {
                        "problem": "machsuite_bfs",
                        "benchmark_lineage": "machsuite:bfs",
                        "split": "test",
                        "representative": True,
                    },
                    "hlseval_machsuite_bfs_bulk": {
                        "problem": "machsuite_bfs_bulk",
                        "benchmark_lineage": "machsuite:bfs",
                        "split": "paired_holdout",
                        "representative": False,
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    payload = _load_split(path)
    assert list(_select_entries(payload, "")) == [
        "hlseval_machsuite_bfs_queue"
    ]
    assert list(_select_entries(payload, "machsuite_bfs")) == [
        "hlseval_machsuite_bfs_queue"
    ]


def test_corrected_matrix_selects_external_suffix_and_metadata():
    available = [
        "hlseval_machsuite_aes_aes",
        "hlseval_machsuite_spmv_crs",
    ]
    assert _select_benchmarks(available, "spmv_crs") == [
        "hlseval_machsuite_spmv_crs"
    ]
    entries = {
        "hlseval_machsuite_spmv_crs": {
            "problem": "machsuite_spmv",
            "benchmark_lineage": "machsuite:spmv",
            "split": "validation",
        }
    }
    assert _entry_metadata(
        "hlseval_machsuite_spmv_crs", entries
    ) == {
        "problem": "machsuite_spmv",
        "benchmark_lineage": "machsuite:spmv",
        "split": "validation",
    }


def test_external_corpus_requires_and_ranks_all_corrected_setups(
    tmp_path,
):
    benchmark = "hlseval_machsuite_aes_aes"
    code_dir = tmp_path / "code"
    code_dir.mkdir()
    (code_dir / f"{benchmark}.cpp").write_text(
        "void aes256_encrypt_ecb() {}\n",
        encoding="utf-8",
    )
    phase_manifest = tmp_path / "manifest.json"
    phase_manifest.write_text(
        json.dumps(
            {
                "status": "complete",
                "entries": {
                    benchmark: {
                        "problem": "machsuite_aes",
                        "benchmark_lineage": "machsuite:aes",
                        "split": "train",
                        "dataset": "HLS-Eval/MachSuite",
                        "code_path": f"code/{benchmark}.cpp",
                        "code_sha256": "phase-code-hash",
                        "csim": {"ran": True, "passed": True},
                        "csynth_report": {
                            "latency_cycles": 1000,
                            "interval": 1001,
                            "bram": 1,
                            "dsp": 1,
                            "ff": 1,
                            "lut": 1,
                        },
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    matrix_path = tmp_path / "matrix.jsonl"
    candidates = []
    for index, setup in enumerate(setup_registry(), start=1):
        candidates.append(
            {
                "setup_id": setup.setup_id,
                "setup_fingerprint": setup.fingerprint,
                "valid": True,
                "latency_cycles": 1000 + index,
                "code_sha256": f"code-{index}",
                "result_path": "",
            }
        )
    matrix_path.write_text(
        json.dumps(
            {
                "benchmark": benchmark,
                "success": True,
                "candidates": candidates,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    records = _external_records(
        Namespace(
            phase_b_manifest=phase_manifest,
            matrix_records=matrix_path,
            benchmarks_dir=(
                Path(__file__).resolve().parents[1]
                / "benchmarks_external"
                / "hls_eval"
            ),
        )
    )
    assert len(records) == 10
    assert all(
        record["eligibility"]["complete_crossed_matrix"]
        for record in records
    )
    assert sum(
        record["labels"]["is_best_setup"] for record in records
    ) == 1
    assert {
        record["labels"]["setup_rank"] for record in records
    } == set(range(1, 11))
