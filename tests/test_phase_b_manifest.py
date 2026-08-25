import json
from pathlib import Path

import pytest

from phase_b_manifest import (
    SCHEMA_VERSION,
    canonical_json_sha256,
    load_phase_b_seed,
    text_sha256,
    toolchain_fingerprint,
)


def _manifest(tmp_path: Path) -> Path:
    code = "void workload() {}\n"
    code_dir = tmp_path / "code"
    code_dir.mkdir()
    (code_dir / "bench.cpp").write_text(code, encoding="utf-8")
    report = {
        "latency_cycles": 17,
        "requested_clock_period_ns": 3.33,
    }
    csim = {"ran": True, "passed": True, "status": "passed"}
    payload = {
        "schema_version": SCHEMA_VERSION,
        "toolchain": toolchain_fingerprint(
            vitis_version="2023.2",
            part="xcu280-fsvh2892-2L-e",
            clock_ns=3.33,
        ),
        "entries": {
            "bench": {
                "benchmark": "bench",
                "input_c_sha256": text_sha256("plain"),
                "header_sha256": text_sha256("header"),
                "code_path": "code/bench.cpp",
                "code_sha256": text_sha256(code),
                "csim": csim,
                "csim_sha256": canonical_json_sha256(csim),
                "csynth_report": report,
                "csynth_report_sha256": canonical_json_sha256(report),
            }
        },
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_frozen_phase_b_enforces_hashes_and_toolchain(
    tmp_path: Path,
) -> None:
    seed = load_phase_b_seed(
        _manifest(tmp_path),
        benchmark="bench",
        input_c="plain",
        header_code="header",
        expected_part="xcu280-fsvh2892-2L-e",
        expected_clock_ns=3.33,
        expected_vitis_version="2023.2",
    )

    assert seed["csim"]["passed"] is True
    assert seed["csynth_report"]["latency_cycles"] == 17
    assert seed["provenance"]["code_sha256"] == text_sha256(seed["code"])


def test_frozen_phase_b_rejects_code_tampering(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    (tmp_path / "code" / "bench.cpp").write_text(
        "void workload() { int changed = 1; }\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="code hash mismatch"):
        load_phase_b_seed(
            manifest,
            benchmark="bench",
            input_c="plain",
            header_code="header",
            expected_part="xcu280-fsvh2892-2L-e",
            expected_clock_ns=3.33,
            expected_vitis_version="2023.2",
        )
