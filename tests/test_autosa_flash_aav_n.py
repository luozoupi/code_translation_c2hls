"""AutoSA ready flash aav_n: 90-skill overlay with avoid rules in the prompt."""

from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "pc2"))

_ENV_KEYS = (
    "C2HLS_SKILL_PROMPT_MODE",
    "C2HLS_PACKAGED_SKILLS_JSON",
    "C2HLS_PACKAGED_SKILLS_ONLY",
    "C2HLS_FLASH_SKILL_ENTRIES_JSON",
    "C2HLS_FORCE_SKILL_PROMPTS",
    "C2HLS_SKILL_MODE",
)


def _save_env() -> dict[str, str | None]:
    return {k: os.environ.get(k) for k in _ENV_KEYS}


def _restore_env(saved: dict[str, str | None]) -> None:
    for key, val in saved.items():
        if val is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = val


class AutosaFlashAavNTests(unittest.TestCase):
    def setUp(self) -> None:
        self._saved = _save_env()

    def tearDown(self) -> None:
        _restore_env(self._saved)

    def test_aav_n_configure_injects_avoids_and_overlay(self) -> None:
        from autosa_flash_lib import configure_autosa_flash_aav_n_env

        configure_autosa_flash_aav_n_env()
        self.assertEqual(
            os.environ["C2HLS_SKILL_PROMPT_MODE"], "all_skills_avoids_global"
        )
        self.assertTrue(
            os.environ["C2HLS_PACKAGED_SKILLS_JSON"].endswith("(90skills).json")
        )
        self.assertTrue(
            os.environ["C2HLS_FLASH_SKILL_ENTRIES_JSON"].endswith(
                "flash_no_RMW_m_axi_skill_entries.json"
            )
        )
        self.assertEqual(os.environ["C2HLS_FORCE_SKILL_PROMPTS"], "1")

    def test_nav_n_configure_still_drops_avoids(self) -> None:
        from autosa_flash_lib import configure_autosa_flash_nav_n_env

        configure_autosa_flash_nav_n_env()
        self.assertEqual(
            os.environ["C2HLS_SKILL_PROMPT_MODE"], "all_skills_no_avoids_global"
        )

    def test_setup_tags_and_cell_dir(self) -> None:
        from autosa_flash_lib import SETUP_TAG, SETUP_TAG_AAV_N, setup_tag_for_variant
        from batch_parallel_autosa_lib import autosa_cell_dir

        self.assertEqual(SETUP_TAG, "flash__autosa__nav_n")
        self.assertEqual(SETUP_TAG_AAV_N, "flash__autosa__aav_n")
        self.assertEqual(setup_tag_for_variant("autosa_aav_n"), SETUP_TAG_AAV_N)
        self.assertEqual(setup_tag_for_variant("autosa_nav_n"), SETUP_TAG)
        cell = autosa_cell_dir(
            Path("/tmp/variants/autosa_aav_n"),
            "autosa_mm",
            "deepseek-v4-flash",
            variant_key="autosa_aav_n",
        )
        self.assertEqual(
            cell,
            Path(
                "/tmp/variants/autosa_aav_n/autosa_mm/"
                "deepseek-v4-flash__flash__autosa__aav_n"
            ),
        )

    def test_dispatch_accepts_aav_n_and_configures_avoids(self) -> None:
        from batch_parallel_autosa_lib import configure_autosa_campaign_env
        from batch_parallel_dispatch import validate_variant

        campaign = {"config": {"pilot": {"workflow": "autosa_flash"}}}
        self.assertTrue(validate_variant(campaign, "autosa_aav_n"))
        self.assertTrue(validate_variant(campaign, "autosa_nav_n"))
        self.assertFalse(validate_variant(campaign, "aav_n"))
        configure_autosa_campaign_env("autosa_aav_n")
        self.assertEqual(
            os.environ["C2HLS_SKILL_PROMPT_MODE"], "all_skills_avoids_global"
        )
        self.assertTrue(
            os.environ["C2HLS_PACKAGED_SKILLS_JSON"].endswith("(90skills).json")
        )
        self.assertFalse("gemm_flatten" in os.environ["C2HLS_PACKAGED_SKILLS_JSON"])

    def test_aav_n_gf_uses_gemm_flatten_pack_and_overlay(self) -> None:
        from autosa_flash_lib import (
            SETUP_TAG_AAV_N_GF,
            configure_autosa_flash_aav_n_gf_env,
            setup_tag_for_variant,
        )
        from batch_parallel_autosa_lib import autosa_cell_dir, configure_autosa_campaign_env
        from batch_parallel_dispatch import validate_variant

        configure_autosa_flash_aav_n_gf_env()
        self.assertEqual(
            os.environ["C2HLS_SKILL_PROMPT_MODE"], "all_skills_avoids_global"
        )
        self.assertTrue(
            os.environ["C2HLS_PACKAGED_SKILLS_JSON"].endswith("gemm_flatten_v1.json")
        )
        self.assertTrue(
            os.environ["C2HLS_FLASH_SKILL_ENTRIES_JSON"].endswith(
                "flash_no_RMW_m_axi_skill_entries.json"
            )
        )
        self.assertEqual(setup_tag_for_variant("autosa_aav_n_gf"), SETUP_TAG_AAV_N_GF)
        campaign = {"config": {"pilot": {"workflow": "autosa_flash"}}}
        self.assertTrue(validate_variant(campaign, "autosa_aav_n_gf"))
        configure_autosa_campaign_env("autosa_aav_n_gf")
        self.assertTrue(
            os.environ["C2HLS_PACKAGED_SKILLS_JSON"].endswith("gemm_flatten_v1.json")
        )
        cell = autosa_cell_dir(
            Path("/tmp/variants/autosa_aav_n_gf"),
            "autosa_mm",
            "deepseek-v4-flash",
            variant_key="autosa_aav_n_gf",
        )
        self.assertEqual(
            cell,
            Path(
                "/tmp/variants/autosa_aav_n_gf/autosa_mm/"
                "deepseek-v4-flash__flash__autosa__aav_n_gf"
            ),
        )

    def test_aav_n_90_uses_plain_90_pack_without_overlay(self) -> None:
        from autosa_flash_lib import (
            SETUP_TAG_AAV_N_90,
            SKILLS_90_JSON,
            configure_autosa_flash_aav_n_90_env,
            setup_tag_for_variant,
        )
        from batch_parallel_autosa_lib import autosa_cell_dir, configure_autosa_campaign_env
        from batch_parallel_dispatch import validate_variant

        os.environ["C2HLS_FLASH_SKILL_ENTRIES_JSON"] = "/tmp/overlay.json"
        os.environ["C2HLS_PACKAGED_SKILLS_JSON"] = "/tmp/gemm_flatten_v1.json"
        configure_autosa_flash_aav_n_90_env()
        self.assertEqual(
            os.environ["C2HLS_SKILL_PROMPT_MODE"], "all_skills_avoids_global"
        )
        self.assertEqual(
            os.environ["C2HLS_PACKAGED_SKILLS_JSON"], str(SKILLS_90_JSON.resolve())
        )
        self.assertTrue(
            os.environ["C2HLS_PACKAGED_SKILLS_JSON"].endswith("(90skills).json")
        )
        self.assertFalse("gemm_flatten" in os.environ["C2HLS_PACKAGED_SKILLS_JSON"])
        self.assertNotIn("C2HLS_FLASH_SKILL_ENTRIES_JSON", os.environ)
        self.assertEqual(setup_tag_for_variant("autosa_aav_n_90"), SETUP_TAG_AAV_N_90)
        campaign = {"config": {"pilot": {"workflow": "autosa_flash"}}}
        self.assertTrue(validate_variant(campaign, "autosa_aav_n_90"))
        configure_autosa_campaign_env("autosa_aav_n_90")
        self.assertNotIn("C2HLS_FLASH_SKILL_ENTRIES_JSON", os.environ)
        cell = autosa_cell_dir(
            Path("/tmp/variants/autosa_aav_n_90"),
            "autosa_mm",
            "deepseek-v4-flash",
            variant_key="autosa_aav_n_90",
        )
        self.assertEqual(
            cell,
            Path(
                "/tmp/variants/autosa_aav_n_90/autosa_mm/"
                "deepseek-v4-flash__flash__autosa__aav_n_90"
            ),
        )

    def test_wave1_config_skips_mm_and_is_64_gemm(self) -> None:
        import json

        cfg = json.loads(
            (REPO / "scripts/pc2/batch_parallel_autosa_wave1_aav_n_gf.json").read_text()
        )
        benches = cfg["pilot"]["benches"]
        self.assertNotIn("autosa_mm", benches)
        self.assertEqual(cfg["pilot"]["variant"], "autosa_aav_n_gf")
        expected = {
            "autosa_mm_hcl",
            "autosa_mm_hcl_intel",
            "autosa_mm_intel",
            "autosa_mm_int16",
            "autosa_mm_catapult",
            "autosa_mm_getting_started",
        }
        self.assertEqual(set(benches), expected)
        targets = json.loads(
            (REPO / "scripts/pc2/autosa_rank1_u280_targets.json").read_text()
        )
        for bench in benches:
            row = targets["targets"][bench]
            self.assertEqual(row["wave"], 1)
            self.assertIsInstance(row["rank1_cycles"], int)
            self.assertGreater(row["rank1_cycles"], 0)


if __name__ == "__main__":
    unittest.main()
