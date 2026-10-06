#!/usr/bin/env python3
"""aav_n multistep uses gemm_flatten_v1 + standalone no_RMW overlay."""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "pc2"))

from multistep_fixed_cosim_lib import VARIANTS, configure_fixed_cosim_multistep_env, variant_env_snapshot

PKG = REPO / "hls_full_optimization_skills_schema_1_1_package"
GEMM = PKG / "skills_ii_target_miss_solutions_added(90skills)_gemm_flatten_v1.json"
OVERLAY = PKG / "flash_no_RMW_m_axi_skill_entries.json"

_ENV_KEYS = (
    "C2HLS_PACKAGED_SKILLS_JSON",
    "C2HLS_FLASH_SKILL_ENTRIES_JSON",
    "C2HLS_PACKAGED_SKILLS_ONLY",
    "C2HLS_SKILL_PROMPT_MODE",
    "C2HLS_FORCE_SKILL_PROMPTS",
    "C2HLS_RAG",
    "C2HLS_RAG2",
    "C2HLS_POST_FLASH_LATENCY_OPT",
)


class MultistepAavNSkillsTest(unittest.TestCase):
    def test_aav_n_points_at_gemm_flatten_and_overlay(self) -> None:
        variant = VARIANTS["aav_n"]
        self.assertEqual(variant.skills_json, GEMM)
        self.assertTrue(variant.flash_skill_overlay)
        snap = variant_env_snapshot(variant)
        self.assertEqual(snap["skills_json_mode"], "packaged_base_plus_flash_overlay")
        self.assertTrue(str(snap["skills_json"]).endswith("gemm_flatten_v1.json"))

    def test_configure_sets_packaged_plus_overlay_and_rag_off(self) -> None:
        saved = {k: os.environ.get(k) for k in _ENV_KEYS}
        for k in ("C2HLS_PACKAGED_SKILLS_JSON", "C2HLS_FLASH_SKILL_ENTRIES_JSON"):
            os.environ.pop(k, None)
        try:
            configure_fixed_cosim_multistep_env(VARIANTS["aav_n"])
            packaged = os.environ.get("C2HLS_PACKAGED_SKILLS_JSON", "")
            overlay = os.environ.get("C2HLS_FLASH_SKILL_ENTRIES_JSON", "")
            self.assertTrue(packaged.endswith("gemm_flatten_v1.json"))
            self.assertTrue(overlay.endswith("flash_no_RMW_m_axi_skill_entries.json"))
            self.assertEqual(os.environ.get("C2HLS_SKILL_PROMPT_MODE"), "all_skills_avoids_global")
            self.assertEqual(os.environ.get("C2HLS_RAG"), "0")
            self.assertEqual(os.environ.get("C2HLS_RAG2"), "0")
            self.assertEqual(os.environ.get("C2HLS_POST_FLASH_LATENCY_OPT"), "0")
            self.assertTrue(OVERLAY.is_file())
            self.assertTrue(GEMM.is_file())
        finally:
            for key, val in saved.items():
                if val is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = val

    def test_noskills_does_not_enable_overlay(self) -> None:
        saved = {k: os.environ.get(k) for k in _ENV_KEYS}
        try:
            configure_fixed_cosim_multistep_env(VARIANTS["noskills"])
            self.assertFalse((os.environ.get("C2HLS_FLASH_SKILL_ENTRIES_JSON") or "").strip())
            self.assertIsNone(VARIANTS["noskills"].skills_json)
        finally:
            for key, val in saved.items():
                if val is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = val


if __name__ == "__main__":
    unittest.main()
