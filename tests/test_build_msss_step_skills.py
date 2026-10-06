#!/usr/bin/env python3
"""msss v1+no-RMW slice and 90+v2 union pack."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "flash_shared"))

from build_msss_step_skills import (  # noqa: E402
    MSSS_V1_NORMW_DIR,
    SKILLS_V1,
    UNION_90_V2,
    build,
    source_label,
)

LIVE_MULTISTEP = REPO / "hls_full_optimization_skills_schema_1_1_package" / "multistep"

V2_ONLY = "hls-fp-mac-k-step-independent-acc-banks"
NO_RMW_ID = "hls-load-compute-store-no-rmw-m_axi"


class BuildMsssStepSkillsTest(unittest.TestCase):
    def test_source_labels(self) -> None:
        self.assertEqual(source_label(SKILLS_V1), "gemm_flatten_v1")
        self.assertEqual(
            source_label(Path("flash_no_RMW_m_axi_skill_entries.json")),
            "no_rmw_overlay",
        )

    def test_v1_normw_slice_uses_overlay_and_drops_v2_only(self) -> None:
        import tempfile

        overlay = (
            REPO
            / "hls_full_optimization_skills_schema_1_1_package"
            / "flash_no_RMW_m_axi_skill_entries.json"
        )
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "msss_v1"
            snap = build([SKILLS_V1, overlay], out_dir=out)
            tiling = json.loads((out / "tiling_skills.json").read_text(encoding="utf-8"))
            ids = {sk["id"] for sk in tiling["skills"]}
            self.assertIn("prompt-tiling", ids)
            self.assertIn(NO_RMW_ID, ids)
            coal = json.loads((out / "coalescing_skills.json").read_text(encoding="utf-8"))
            coal_ids = {sk["id"] for sk in coal["skills"]}
            self.assertNotIn(V2_ONLY, coal_ids)
            self.assertNotIn("hls-axi-load-step-matches-widen-lanes", coal_ids)
            self.assertGreater(snap["union"], 90)
            self.assertTrue((out / "tiling_avoids.json").is_file())
            self.assertTrue((out / "pipeline_avoids.json").is_file())

    def test_90_v2_union_has_v2_only_and_90_only(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            pack = Path(tmp) / "union.json"
            snap = build(
                [
                    REPO / "hls_full_optimization_skills_schema_1_1_package" / "skills_ii_target_miss_solutions_added(90skills).json",
                    REPO / "hls_full_optimization_skills_schema_1_1_package" / "skills_ii_target_miss_solutions_added(90skills)_gemm_flatten_v2.json",
                ],
                merged_pack=pack,
            )
            self.assertEqual(snap["union"], 134)
            ids = {sk["id"] for sk in json.loads(pack.read_text(encoding="utf-8"))["skills"]}
            self.assertIn(V2_ONLY, ids)
            self.assertIn("prompt-tiling", ids)

    def test_generated_v1_normw_dir_does_not_replace_live_v2_slice(self) -> None:
        self.assertTrue((MSSS_V1_NORMW_DIR / "tiling_skills.json").is_file())
        self.assertTrue(UNION_90_V2.is_file())
        live = json.loads((LIVE_MULTISTEP / "tiling_skills.json").read_text(encoding="utf-8"))
        v1 = json.loads((MSSS_V1_NORMW_DIR / "tiling_skills.json").read_text(encoding="utf-8"))
        self.assertTrue(any("gemm_flatten_v2" in str(x) for x in live.get("derived_from") or []))
        self.assertTrue(any("gemm_flatten_v1" in str(x) for x in v1.get("derived_from") or []))
        self.assertFalse(any("gemm_flatten_v2" in str(x) for x in v1.get("derived_from") or []))
        union = json.loads(UNION_90_V2.read_text(encoding="utf-8"))
        self.assertEqual(union["skill_count"], 134)
        self.assertIn(V2_ONLY, {sk["id"] for sk in union["skills"]})


if __name__ == "__main__":
    unittest.main()
