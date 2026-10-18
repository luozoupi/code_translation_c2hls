#!/usr/bin/env python3
"""Check frozen paper bytes and recompute the main numerical evidence offline.

This command does not run Vitis, query a model, or change the evidence bundle.
The copied historical generators remain unmodified beside it.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent


def verify_reference(root: Path = ROOT) -> dict:
    reference = json.loads((root / "reference.json").read_text())
    if reference.get("schema") != "hlsmith.paper-reference.v1":
        raise ValueError("Unsupported paper reference schema")
    index_bytes = json.dumps(reference["copied_files_sha256"], sort_keys=True,
                             separators=(",", ":")).encode()
    if hashlib.sha256(index_bytes).hexdigest() != reference.get("bundle_sha256"):
        raise ValueError("Paper reference file index changed")
    for name, expected in reference["copied_files_sha256"].items():
        path = root / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Frozen paper reference changed: {name}")
    return reference


def replay() -> dict:
    reference = verify_reference()
    sys.path.insert(0, str(ROOT / "scripts"))
    import generate_sigmetrics_commercial_validation as commercial
    import generate_skill_v3_rtl as rtl

    panels, _, _ = commercial.load_data()
    commercial_summary = commercial.summary(panels)
    expected_commercial = json.loads((ROOT / "generated/sigmetrics/commercial_validation_manifest.json").read_text())
    if commercial_summary != expected_commercial["models"]:
        raise ValueError("Commercial cohort or computed numerical summary changed")
    bundle = rtl.load_bundle(ROOT / "generated/sigmetrics/skill_v3_rtl_evidence")
    open_summary = rtl.summarize(rtl.build_rows(bundle))
    expected_open = json.loads((ROOT / "generated/sigmetrics/skill_v3_rtl_manifest.json").read_text())
    if open_summary != expected_open["summary"]:
        raise ValueError("Open-model cohort or computed numerical summary changed")
    return {
        "status": "passed",
        "paper_source_commit": reference["paper_source_commit"],
        "files_verified": len(reference["copied_files_sha256"]),
        "commercial": {
            name: {key: values[key] for key in ("paired", "rtl_off_over_selected_gm")}
            for name, values in commercial_summary.items()
        },
        "open_models": {
            key: open_summary[key] for key in ("selected_pass_1h", "off_pass_1h", "paired_1h", "strict_wins_1h", "ties_1h")
        },
        "historical_manifest_caveats": reference["historical_manifest_caveats"],
        "scope": "Frozen historical evidence replay; no new model generation or hardware execution.",
    }


if __name__ == "__main__":
    print(json.dumps(replay(), indent=2, sort_keys=True))
