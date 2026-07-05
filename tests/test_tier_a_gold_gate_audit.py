"""Static audit for tier_A_ready gold-gate corpus (no Vitis)."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "pc2"))

from tier_a_gold_gate_audit import (  # noqa: E402
    ALLOWED_VIOLATIONS,
    TIER_A_READY_ROOT,
    audit_forgebench_support_staging,
    audit_hls_eval_csim_extra_files,
    iter_tier_a_benches,
    run_tier_a_gold_gate_audit,
    unexpected_violations,
)

TIER_A_25 = json.loads(
    (REPO / "scripts/pc2/batch_parallel_tier_a_25.json").read_text(encoding="utf-8")
)["pilot"]["benches"]

TIER_A_30_REMAINING = [
    line.strip()
    for line in (REPO / "scripts/pc2/tier_a_30_remaining_benches.txt").read_text(encoding="utf-8").splitlines()
    if line.strip()
]


class TierAGoldGateAuditTests(unittest.TestCase):
    def test_tier_a_ready_has_25_campaign_benches(self) -> None:
        present = set(iter_tier_a_benches())
        missing = [b for b in TIER_A_25 if b not in present]
        self.assertEqual(missing, [], f"missing tier_A_ready dirs: {missing}")

    def test_tier_a_ready_has_30_remaining_campaign_benches(self) -> None:
        present = set(iter_tier_a_benches())
        missing = [b for b in TIER_A_30_REMAINING if b not in present]
        self.assertEqual(missing, [], f"missing tier_A_ready dirs: {missing}")

    def test_tier_a_remaining_30_pass_static_audit(self) -> None:
        violations = run_tier_a_gold_gate_audit(TIER_A_30_REMAINING)
        surprise = unexpected_violations(violations, ALLOWED_VIOLATIONS)
        if surprise:
            lines = [f"{v.bench} [{v.kind}] {v.detail}" for v in surprise]
            self.fail("unexpected audit violations:\n" + "\n".join(lines))

    def test_forgebench_corpus_support_files_on_disk(self) -> None:
        violations: list = []
        for bench in iter_tier_a_benches():
            violations.extend(audit_forgebench_support_staging(bench))
        self.assertEqual(violations, [], violations)

    def test_forgebench_hls_eval_csim_stages_support_txt(self) -> None:
        violations: list = []
        for bench in iter_tier_a_benches():
            meta_path = TIER_A_READY_ROOT / bench / "metadata.json"
            if not meta_path.is_file():
                continue
            if json.loads(meta_path.read_text(encoding="utf-8")).get("dataset") != "forgebench":
                continue
            violations.extend(audit_hls_eval_csim_extra_files(bench))
        self.assertEqual(violations, [], violations)

    def test_full_audit_no_unexpected_violations(self) -> None:
        violations = run_tier_a_gold_gate_audit(TIER_A_25)
        surprise = unexpected_violations(violations, ALLOWED_VIOLATIONS)
        if surprise:
            lines = [f"{v.bench} [{v.kind}] {v.detail}" for v in surprise]
            self.fail("unexpected audit violations:\n" + "\n".join(lines))


if __name__ == "__main__":
    unittest.main()
