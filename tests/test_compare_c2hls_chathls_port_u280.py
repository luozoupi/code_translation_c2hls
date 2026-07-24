from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "pc2"))

from compare_c2hls_chathls_port_u280 import (  # noqa: E402
    best_c2hls_metrics,
    fmt_ratio,
    merge_chathls_rows,
    read_chathls_latency_csv,
    read_chathls_resources_csv,
)


def test_fmt_ratio_handles_missing_and_values():
    assert fmt_ratio(None, 100.0) == "N/A"
    assert fmt_ratio(200.0, None) == "N/A"
    assert fmt_ratio(150.0, 100.0) == "1.500×"


def test_read_chathls_csvs_merge_resources(tmp_path):
    latency_csv = tmp_path / "latency.csv"
    resources_csv = tmp_path / "resources.csv"
    with latency_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["bench", "passed_optimization", "csynth_best_cycles"],
        )
        writer.writeheader()
        writer.writerow({
            "bench": "hlsfactory_atax",
            "passed_optimization": "True",
            "csynth_best_cycles": "915",
        })
        writer.writerow({
            "bench": "machsuite_gemm_ncubed",
            "passed_optimization": "False",
            "csynth_best_cycles": "4627",
        })
    with resources_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["bench", "LUT", "DSP"])
        writer.writeheader()
        writer.writerow({"bench": "hlsfactory_atax", "LUT": "8090", "DSP": "16"})
        writer.writerow({"bench": "machsuite_gemm_ncubed", "LUT": "72060", "DSP": "704"})

    latency_rows = read_chathls_latency_csv(latency_csv)
    resource_rows = read_chathls_resources_csv(resources_csv)
    merged = merge_chathls_rows(latency_rows, resource_rows)

    assert merged["hlsfactory_atax"].latency == 915.0
    assert merged["hlsfactory_atax"].lut == 8090
    assert merged["hlsfactory_atax"].dsp == 16
    assert merged["machsuite_gemm_ncubed"].latency is None
    assert merged["machsuite_gemm_ncubed"].passed_optimization is False
    assert merged["machsuite_gemm_ncubed"].lut == 72060


def test_best_c2hls_metrics_from_flash_selected(tmp_path):
    campaign = tmp_path / "campaign"
    bench = "hlsfactory_atax"
    report_dir = campaign / "flash_selected" / bench / "selected"
    report_dir.mkdir(parents=True)
    (report_dir / "synth_report.json").write_text(
        json.dumps({"latency_cycles": 1200, "lut": 5000, "dsp": 8}),
        encoding="utf-8",
    )

    metrics = best_c2hls_metrics(campaign, bench)
    assert metrics.latency == 1200.0
    assert metrics.lut == 5000
    assert metrics.dsp == 8
    assert metrics.source == "flash_selected/synth_report"


def test_best_c2hls_metrics_missing_campaign_returns_empty(tmp_path):
    metrics = best_c2hls_metrics(tmp_path / "missing", "hlsfactory_atax")
    assert metrics.latency is None
    assert metrics.lut is None
    assert metrics.dsp is None
