#!/usr/bin/env python3
"""Check AutoSA skill text against the held-out mm1024 rows.

The latency rule used here is the one the tiling skill states:
abs(HLS kernel min - model) / model <= 0.10, applied per sa_sizes.
The U280 300 MHz HLS jobs produced no module-max rows, so no skill
is allowed to claim that comparison is done.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

PKG = Path(__file__).resolve().parent
REPO = PKG.parent
CSV = (
    REPO.parent
    / "AutoSA"
    / "artifacts"
    / "dse"
    / "mm1024_paper_vs_previous.csv"
)
# workspace layout: c2hls and AutoSA are siblings
if not CSV.is_file():
    CSV = Path(
        "/scratch/hpc-prf-llmfpga/asa582/projects/AutoSA/artifacts/dse/"
        "mm1024_paper_vs_previous.csv"
    )

PACKS = [
    "autosa_spacetime_flags_skill_entries.json",
    "autosa_tiling_grammar_skill_entries.json",
    "autosa_latency_reading_skill_entries.json",
    "autosa_resource_model_skill_entries.json",
    "autosa_generated_architecture_skill_entries.json",
]

U280_MODULE_MAX = [
    PKG.parents[1].parent
    / "AutoSA"
    / "artifacts"
    / "dse"
    / name
    for name in (
        "mm1024_u280_st0.csv",
        "mm1024_u280_st3.csv",
        "mm1024_u280_st4.csv",
    )
]
# sibling path is unreliable; use the absolute campaign artifacts
U280_MODULE_MAX = [
    Path("/scratch/hpc-prf-llmfpga/asa582/projects/AutoSA/artifacts/dse") / name
    for name in (
        "mm1024_u280_st0.csv",
        "mm1024_u280_st3.csv",
        "mm1024_u280_st4.csv",
    )
]


def load_skills() -> dict[str, dict]:
    sys.path.insert(0, str(REPO))
    from skill_library import _coerce_skill_entry

    found: dict[str, dict] = {}
    for name in PACKS:
        doc = json.loads((PKG / name).read_text(encoding="utf-8"))
        if doc.get("schema") != "1.1":
            raise SystemExit(f"{name} schema is {doc.get('schema')}")
        for entry in doc["skills"]:
            sk = _coerce_skill_entry(entry)
            if sk is None:
                raise SystemExit(f"skill failed schema coerce: {entry.get('id')}")
            blob = " ".join(
                [
                    sk.pattern,
                    sk.strategy,
                    " ".join(sk.required_steps),
                    " ".join(sk.guards),
                ]
            )
            found[sk.id] = {"skill": sk, "text": blob, "confidence": sk.confidence}
    return found


def holdout_rows() -> list[dict]:
    rows = []
    with CSV.open(encoding="utf-8") as f:
        for rec in csv.DictReader(f):
            if rec["record_kind"] != "previous_hls_u280":
                continue
            if rec["space_time"] not in {"0", "3", "4"}:
                continue
            rows.append(rec)
    return rows


def within_10(rec: dict) -> bool:
    return abs(float(rec["rel_err_min"])) <= 0.10


def main() -> int:
    skills = load_skills()
    rows = holdout_rows()
    shape = [r for r in rows if r["sa_shape"] == "128x8" and r["space_time"] == "0"]
    if len(shape) != 9:
        raise SystemExit(f"expected 9 of 128x8, found {len(shape)}")
    near = {int(r["candidate_id"]) for r in shape if within_10(r)}
    far = {int(r["candidate_id"]) for r in shape if not within_10(r)}
    text = skills["avoid-autosa-shape-is-one-config"]["text"]
    missing_near = sorted(c for c in near if str(c) not in text)
    missing_far = sorted(c for c in far if str(c) not in text)
    claimed_all = "every 128x8" in text and "within 10%" in text and "do not mark every 128x8" not in text

    st3 = [r for r in rows if r["space_time"] == "3"]
    st4 = [r for r in rows if r["space_time"] == "4"]
    st3_near = sum(1 for r in st3 if within_10(r))
    st4_near = sum(1 for r in st4 if within_10(r))
    st3_over_dsp = [
        int(r["candidate_id"])
        for r in st3
        if int(float(r["csynth_dsp"])) > 9024
    ]
    model_dsp_zero = all(str(r.get("model_dsp", "0")) in {"0", "0.0", ""} for r in rows)

    latency = skills["autosa-latency-max-of-modules"]
    paper_claim_high = latency["confidence"] == "high"
    module_rows = 0
    for path in U280_MODULE_MAX:
        if not path.is_file():
            continue
        lines = path.read_text(encoding="utf-8").splitlines()
        module_rows += max(0, len(lines) - 1)

    checks = [
        ("nine 128x8 rows", len(shape) == 9),
        ("near set is 17,20,23,26", near == {17, 20, 23, 26}),
        ("far set is 15,25,28,31,32", far == {15, 25, 28, 31, 32}),
        ("skill names every near candidate", not missing_near),
        ("skill names every far candidate", not missing_far),
        ("skill does not call every 128x8 within 10%", not claimed_all),
        ("skill states the 0.10 rule", "0.10" in text),
        ("space-time 3 is not all within 10%", st3_near < len(st3)),
        ("space-time 3 skill notes 13x16x8 absent", "13x16x8" in skills["autosa-pe-shape-from-sa-sizes"]["text"] and "no earlier" in skills["autosa-pe-shape-from-sa-sizes"]["text"]),
        ("space-time 4 kernel-min skill stays medium", skills["autosa-st4-kernel-min-band"]["confidence"] == "medium"),
        ("space-time 4 within 10% count is 8 of 10", st4_near == 8 and len(st4) == 10),
        ("paper module-max skill stays medium", latency["confidence"] == "medium" and not paper_claim_high),
        ("U280 module-max CSV has no data rows", module_rows == 0),
        ("model DSP is 0 on these rows", model_dsp_zero),
        ("resource skill forbids calling DSP 0 the paper search", "not the paper resource search" in skills["avoid-autosa-model-dsp-zero"]["text"]),
        ("DSP above 9024 is named", "9024" in skills["avoid-autosa-model-dsp-zero"]["text"] and "10240" in skills["avoid-autosa-model-dsp-zero"]["text"]),
        ("five space-time 3 rows exceed 9024 DSP", len(st3_over_dsp) == 5),
    ]
    failed = [name for name, ok in checks if not ok]
    lines = [
        "# AutoSA skill holdout",
        "",
        "Rule, taken from `avoid-autosa-shape-is-one-config`: a row is near the model when `abs(HLS kernel min - model) / model <= 0.10`. The comparison is per `sa_sizes`.",
        "",
        f"Held-out 128x8 candidates near: {sorted(near)}.",
        f"Held-out 128x8 candidates not near: {sorted(far)}.",
        f"Space-time 3 rows within 10% of kernel min: {st3_near} of {len(st3)}. DSP above 9024: {st3_over_dsp}.",
        f"Space-time 4 rows within 10% of kernel min: {st4_near} of {len(st4)}.",
        "",
        "The U280 searches 3464449, 3464452, and 3464455 completed. HLS jobs 3464450, 3464453, and 3464456 failed after writing 0 candidates, because validation looked under `default_cap` while the search wrote `u280_paper`. `mm1024_u280_st{0,3,4}.csv` each have a header and no data rows. Module-max versus the model is unmeasured, so `autosa-latency-max-of-modules` and `autosa-st4-kernel-min-band` stay at confidence medium.",
        "",
        "## Checks",
        "",
    ]
    for name, ok in checks:
        lines.append(f"- {'pass' if ok else 'FAIL'}: {name}")
    lines.append("")
    lines.append(f"Failed checks: {len(failed)}.")
    report = PKG / "autosa_skill_holdout.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(report)
    for name, ok in checks:
        print(("OK " if ok else "FAIL"), name)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
