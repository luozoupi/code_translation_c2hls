#!/usr/bin/env python3
"""Tests for flash standalone overlay on packaged 90-skills base."""

from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from prompt_c2hls import q_optimize_flash

PKG = REPO / "hls_full_optimization_skills_schema_1_1_package"
FLASH_ENTRIES_PATH = PKG / "flash_no_RMW_m_axi_skill_entries.json"
SKILLS_90_BASE = PKG / "skills_ii_target_miss_solutions_added(90skills).json"

FLASH_ENTRIES_IDS = {
    "avoid-no-rmw-m_axi-direct",
    "hls-load-compute-store-no-rmw-m_axi",
    "hls-loop-structure-audit-flash",
    "avoid-pipeline-parent-with-nested-child-loop",
    "avoid-loop-tripcount-flatten-variable-bounds",
    "hls-rewrite-variable-trip-fixed-bound-guard",
    "avoid-fp-scalar-recurrence-in-pipelined-loop",
    "hls-fp-reduction-partial-accumulators",
    "avoid-outer-memory-loop-pipeline-over-row",
    "hls-pipeline-innermost-memory-copy-loop",
    "hls-array-partition-match-unroll-factor",
    "avoid-noncanonical-dataflow-region",
    "hls-dataflow-ping-pong-gate-flash",
    "avoid-dataflow-nonunit-tile-loop",
    "avoid-fake-ping-pong-without-tile-overlap",
    "avoid-ping-pong-cross-tile-deps-triangular",
    "hls-canonical-dataflow-lcs-functions",
    "avoid-local-output-rmw-in-pipelined-unroll",
    "hls-unroll-column-partition-matrix-vector",
    "hls-single-load-dual-local-layout",
    "hls-distinct-gmem-bundle-per-port",
    "avoid-dataflow-without-stream-or-overlap-proof",
    "avoid-fused-multi-array-load-outer-pipeline",
    "hls-gemm-chain-local-reuse",
    "hls-fixed-tile-bound-not-runtime-tile-size",
    "avoid-m_axi-accumulate-in-tiled-phase",
    "hls-dataflow-one-writer-one-reader-local",
    "avoid-dataflow-temporal-loop-as-parallel-tasks",
    "avoid-m_axi-read-inside-dataflow-outer-loop",
    "hls-dataflow-performance-partition-unroll",
    "hls-dataflow-fused-compute-phases",
    "hls-dual-layout-fused-load-dataflow",
    "hls-dataflow-merge-parallel-consumers",
}


class FlashStandaloneOverlayTest(unittest.TestCase):
    def test_flash_entries_standalone_file(self) -> None:
        data = json.loads(FLASH_ENTRIES_PATH.read_text(encoding="utf-8"))
        ids = {s["id"] for s in data.get("skills", [])}
        self.assertEqual(len(ids), len(FLASH_ENTRIES_IDS))
        self.assertTrue(FLASH_ENTRIES_IDS.issubset(ids))
        self.assertIn("NOT merged", data["description"])

    def test_90_base_does_not_contain_flash_overlay_skills(self) -> None:
        data = json.loads(SKILLS_90_BASE.read_text(encoding="utf-8"))
        ids = {s["id"] for s in data.get("skills", [])}
        self.assertNotIn("hls-loop-structure-audit-flash", ids)
        self.assertNotIn("avoid-outer-memory-loop-pipeline-over-row", ids)

    def test_flash_prompts_do_not_require_rmw_for_noskills(self) -> None:
        lower = q_optimize_flash.lower()
        self.assertNotIn("read-modify-write", lower)
        self.assertNotIn("no rmw on m_axi", lower)

    def test_noskills_variant_uses_no_skills_file(self) -> None:
        from flash_shared.new_skills_lib import VARIANTS, skills_json_for_variant

        noskills = VARIANTS["noskills_new"]
        self.assertIsNone(skills_json_for_variant(noskills))

    def test_avoids_variant_uses_90_base_not_merged_file(self) -> None:
        from flash_shared.new_skills_lib import VARIANTS, skills_json_for_variant

        variant = VARIANTS["all_new_skills_avoids_global"]
        path = skills_json_for_variant(variant)
        self.assertIsNotNone(path)
        self.assertTrue(str(path).endswith("(90skills).json"))
        self.assertNotIn("no_RMW_m_axi", str(path))

    def test_skills_variant_sets_packaged_90_and_flash_overlay(self) -> None:
        from flash_shared.new_skills_lib import VARIANTS, configure_new_skills_env

        variant = VARIANTS["all_new_skills_avoids_global"]
        saved = {
            k: os.environ.get(k)
            for k in (
                "C2HLS_PACKAGED_SKILLS_JSON",
                "C2HLS_FLASH_SKILL_ENTRIES_JSON",
                "C2HLS_PACKAGED_SKILLS_ONLY",
            )
        }
        try:
            configure_new_skills_env(variant, inference="api")
            packaged = os.environ.get("C2HLS_PACKAGED_SKILLS_JSON", "")
            flash = os.environ.get("C2HLS_FLASH_SKILL_ENTRIES_JSON", "")
            self.assertTrue(packaged.endswith("(90skills).json"))
            self.assertTrue(flash.endswith("flash_no_RMW_m_axi_skill_entries.json"))
        finally:
            for key, val in saved.items():
                if val is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = val

    def test_skill_library_stacks_90_base_plus_flash_overlay(self) -> None:
        from skill_library import make_default_library

        saved = {
            k: os.environ.get(k)
            for k in (
                "C2HLS_PACKAGED_SKILLS_JSON",
                "C2HLS_PACKAGED_SKILLS_ONLY",
                "C2HLS_FLASH_SKILL_ENTRIES_JSON",
            )
        }
        try:
            os.environ["C2HLS_PACKAGED_SKILLS_JSON"] = str(SKILLS_90_BASE)
            os.environ["C2HLS_PACKAGED_SKILLS_ONLY"] = "1"
            os.environ["C2HLS_FLASH_SKILL_ENTRIES_JSON"] = str(FLASH_ENTRIES_PATH)
            lib = make_default_library(persist=False)
            ids = {sk.id for sk in lib.all()}
            # from 90 base
            self.assertGreater(len(ids), 80)
            # from flash overlay
            self.assertIn("hls-loop-structure-audit-flash", ids)
            self.assertIn("avoid-outer-memory-loop-pipeline-over-row", ids)
        finally:
            for key, val in saved.items():
                if val is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = val


if __name__ == "__main__":
    unittest.main()
