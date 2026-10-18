"""Strict frozen Phase-B seed manifests for controlled setup comparisons."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "c2hls.phase-b-seed-manifest.v1"


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def text_sha256(value: str) -> str:
    return hashlib.sha256((value or "").encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def toolchain_fingerprint(
    *,
    vitis_version: str,
    part: str,
    clock_ns: float,
    flow_target: str = "vitis",
) -> dict[str, Any]:
    configuration = {
        "vitis_version": str(vitis_version),
        "part": str(part),
        "clock_ns": float(clock_ns),
        "flow_target": str(flow_target),
    }
    return {
        "configuration": configuration,
        "sha256": canonical_json_sha256(configuration),
    }


def _entry_index(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    entries = payload.get("entries")
    if isinstance(entries, dict):
        return {
            str(name): dict(entry)
            for name, entry in entries.items()
            if isinstance(entry, dict)
        }
    if isinstance(entries, list):
        return {
            str(entry.get("benchmark")): dict(entry)
            for entry in entries
            if isinstance(entry, dict) and entry.get("benchmark")
        }
    raise ValueError("Phase-B manifest entries must be an object or array")


def load_phase_b_seed(
    manifest_path: str | Path,
    *,
    benchmark: str,
    input_c: str,
    header_code: str,
    expected_part: str,
    expected_clock_ns: float,
    expected_vitis_version: str,
    expected_flow_target: str = "vitis",
) -> dict[str, Any]:
    """Load one seed and reject any input, payload, or toolchain drift."""

    path = Path(manifest_path).expanduser().resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"Phase-B manifest must use {SCHEMA_VERSION!r}, got "
            f"{payload.get('schema_version')!r}"
        )

    expected_toolchain = toolchain_fingerprint(
        vitis_version=expected_vitis_version,
        part=expected_part,
        clock_ns=expected_clock_ns,
        flow_target=expected_flow_target,
    )
    observed_toolchain = payload.get("toolchain") or {}
    if observed_toolchain != expected_toolchain:
        raise ValueError(
            "frozen Phase-B toolchain mismatch: "
            f"expected {expected_toolchain}, got {observed_toolchain}"
        )

    entries = _entry_index(payload)
    if benchmark not in entries:
        raise KeyError(
            f"frozen Phase-B manifest has no entry for {benchmark!r}"
        )
    entry = entries[benchmark]
    if entry.get("input_c_sha256") != text_sha256(input_c):
        raise ValueError(
            f"frozen Phase-B plain-source hash mismatch for {benchmark}"
        )
    if entry.get("header_sha256") != text_sha256(header_code):
        raise ValueError(
            f"frozen Phase-B header hash mismatch for {benchmark}"
        )

    code_path = (path.parent / str(entry.get("code_path") or "")).resolve()
    if not code_path.is_file():
        raise FileNotFoundError(
            f"frozen Phase-B code is missing for {benchmark}: {code_path}"
        )
    code = code_path.read_text(encoding="utf-8")
    if entry.get("code_sha256") != text_sha256(code):
        raise ValueError(
            f"frozen Phase-B code hash mismatch for {benchmark}"
        )

    report = entry.get("csynth_report")
    csim = entry.get("csim")
    if not isinstance(report, dict) or not report:
        raise ValueError(
            f"frozen Phase-B CSynth report is missing for {benchmark}"
        )
    if entry.get("csynth_report_sha256") != canonical_json_sha256(report):
        raise ValueError(
            f"frozen Phase-B CSynth report hash mismatch for {benchmark}"
        )
    if not isinstance(csim, dict) or csim.get("passed") is not True:
        raise ValueError(
            f"frozen Phase-B CSim must be an executed pass for {benchmark}"
        )
    if entry.get("csim_sha256") != canonical_json_sha256(csim):
        raise ValueError(
            f"frozen Phase-B CSim hash mismatch for {benchmark}"
        )
    latency = report.get("latency_cycles")
    if (
        not isinstance(latency, int)
        or isinstance(latency, bool)
        or latency <= 0
    ):
        raise ValueError(
            f"frozen Phase-B report lacks exact positive cycles for {benchmark}"
        )

    entry_payload = {
        **entry,
        "code_path": str(code_path),
        "code": code,
    }
    provenance = {
        "schema_version": SCHEMA_VERSION,
        "manifest_path": str(path),
        "manifest_sha256": file_sha256(path),
        "benchmark": benchmark,
        "entry_sha256": canonical_json_sha256(entry),
        "code_path": str(code_path),
        "code_sha256": entry["code_sha256"],
        "csim_sha256": entry["csim_sha256"],
        "csynth_report_sha256": entry["csynth_report_sha256"],
        "input_c_sha256": entry["input_c_sha256"],
        "header_sha256": entry["header_sha256"],
        "toolchain": expected_toolchain,
        "source_artifacts": entry.get("source_artifacts") or {},
    }
    return {
        "code": code,
        "csim": dict(csim),
        "csynth_report": dict(report),
        "entry": entry_payload,
        "provenance": provenance,
    }
