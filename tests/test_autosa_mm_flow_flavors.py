"""autosa_mm flow flavors: zero-shot and no-skills vs frozen with-skills."""

from __future__ import annotations

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
    "C2HLS_POST_FLASH_NO_SKILLS",
    "C2HLS_SKIP_PHASE_B",
    "C2HLS_ONE_SHOT",
    "C2HLS_FLASH_OPT_PROMPT_MODE",
    "C2HLS_POST_FLASH_DSE",
    "C2HLS_DSE_CHAIN_FLASH",
    "C2HLS_POST_FLASH_STREAM",
    "C2HLS_STREAM_CHAIN_FLASH",
    "C2HLS_TURNS",
    "C2HLS_MM_FLOW_FLAVOR",
    "C2HLS_PE_RECIPE",
    "C2HLS_DSE_SKILL_ENTRIES_JSON",
    "C2HLS_STREAM_SKILL_ENTRIES_JSON",
    "BATCH_PARALLEL_VARIANT",
    "PC2_BATCH_JOB_PREFIX",
    "BATCH_PARALLEL_ARTIFACT_PREFIX",
    "C2HLS_FLASH_ONLY",
    "C2HLS_FLASH_MIN_DSP",
    "C2HLS_DSE_V2",
    "C2HLS_DSE_V2_CHAIN_FLASH",
    "C2HLS_SWEEP_JOB_PREFIX",
    "C2HLS_SWEEP_ARTIFACT_PREFIX",
)


def _save_env() -> dict[str, str | None]:
    return {k: os.environ.get(k) for k in _ENV_KEYS}


def _restore_env(saved: dict[str, str | None]) -> None:
    for key, val in saved.items():
        if val is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = val


class AutosaMmFlowFlavorTests(unittest.TestCase):
    def setUp(self) -> None:
        self._saved = _save_env()
        for key in _ENV_KEYS:
            os.environ.pop(key, None)

    def tearDown(self) -> None:
        _restore_env(self._saved)

    def test_setup_tags_for_new_variants(self) -> None:
        from autosa_flash_lib import setup_tag_for_variant

        self.assertEqual(setup_tag_for_variant("autosa_noskills"), "flash__autosa__noskills")
        self.assertEqual(setup_tag_for_variant("autosa_zero_shot"), "flash__autosa__zero_shot")
        self.assertEqual(setup_tag_for_variant("autosa_one_shot"), "flash__autosa__one_shot")
        self.assertEqual(setup_tag_for_variant("autosa_aav_n_gf"), "flash__autosa__aav_n_gf")
        self.assertEqual(setup_tag_for_variant("autosa_aav_n_90"), "flash__autosa__aav_n_90")

    def test_launcher_script_exposes_flavor_and_keeps_default_mmflow(self) -> None:
        text = (REPO / "scripts/pc2/start_autosa_mm_flow.sh").read_text(encoding="utf-8")
        self.assertIn("--flavor", text)
        self.assertIn("zero_shot", text)
        self.assertIn("noskills", text)
        self.assertIn("one_shot", text)
        self.assertIn("mmzs", text)
        self.assertIn("mmns", text)
        self.assertIn("mm1s", text)
        self.assertIn("mmflow", text)
        self.assertIn("aav_n_90", text)
        self.assertIn("mmdsev290", text)
        self.assertIn("90only", text)
        campaign_sh = (REPO / "scripts/pc2/start_batch_parallel_campaign.sh").read_text(encoding="utf-8")
        self.assertIn("C2HLS_SKIP_PHASE_B", campaign_sh)
        self.assertIn("C2HLS_POST_FLASH_NO_SKILLS", campaign_sh)
        self.assertIn("C2HLS_FLASH_OPT_PROMPT_MODE", campaign_sh)
        self.assertIn("C2HLS_FLASH_MAX_TOKENS", campaign_sh)
        self.assertIn("C2HLS_CPP_CONTINUATIONS", campaign_sh)
        self.assertIn("C2HLS_FLASH_MAX_TOKENS", text)
        self.assertIn("65536", text)
        self.assertIn("--no-thinking", text)
        self.assertIn("C2HLS_THINKING", text)
        replay = (REPO / "scripts/pc2/start_autosa_mm_flash16k_v41.sh").read_text(encoding="utf-8")
        self.assertIn("C2HLS_FLASH_ONLY=1", replay)
        self.assertIn("C2HLS_FLASH_MAX_TOKENS=16384", replay)
        self.assertIn("C2HLS_CPP_CONTINUATIONS=0", replay)
        self.assertIn("C2HLS_SKILL_PROMPT_ORDER_JSON", replay)
        self.assertIn("--thinking-on", replay)
        self.assertIn("api_default", replay)

    def test_noskills_configure_strips_skill_files(self) -> None:
        from autosa_flash_lib import configure_autosa_flash_noskills_env

        os.environ["C2HLS_PACKAGED_SKILLS_JSON"] = "/tmp/skills.json"
        os.environ["C2HLS_FLASH_SKILL_ENTRIES_JSON"] = "/tmp/overlay.json"
        os.environ["C2HLS_SKILL_PROMPT_MODE"] = "all_skills_avoids_global"
        configure_autosa_flash_noskills_env()
        self.assertEqual(os.environ["C2HLS_SKILL_MODE"], "skill_off")
        self.assertEqual(os.environ["C2HLS_FORCE_SKILL_PROMPTS"], "0")
        self.assertEqual(os.environ["C2HLS_POST_FLASH_NO_SKILLS"], "1")
        self.assertNotIn("C2HLS_PACKAGED_SKILLS_JSON", os.environ)
        self.assertNotIn("C2HLS_FLASH_SKILL_ENTRIES_JSON", os.environ)
        self.assertNotIn("C2HLS_SKILL_PROMPT_MODE", os.environ)

    def test_zero_shot_configure_skips_phase_b_and_skills(self) -> None:
        from autosa_flash_lib import configure_autosa_flash_zero_shot_env

        configure_autosa_flash_zero_shot_env()
        self.assertEqual(os.environ["C2HLS_SKILL_MODE"], "skill_off")
        self.assertEqual(os.environ["C2HLS_FORCE_SKILL_PROMPTS"], "0")
        self.assertEqual(os.environ["C2HLS_SKIP_PHASE_B"], "1")
        self.assertEqual(os.environ["C2HLS_FLASH_OPT_PROMPT_MODE"], "zero_shot")
        self.assertEqual(os.environ["C2HLS_POST_FLASH_NO_SKILLS"], "1")
        self.assertNotIn("C2HLS_PACKAGED_SKILLS_JSON", os.environ)

    def test_dispatch_accepts_noskills_and_zero_shot_variants(self) -> None:
        from batch_parallel_autosa_lib import configure_autosa_campaign_env
        from batch_parallel_dispatch import validate_variant

        campaign = {"config": {"pilot": {"workflow": "autosa_flash"}}}
        self.assertTrue(validate_variant(campaign, "autosa_noskills"))
        self.assertTrue(validate_variant(campaign, "autosa_zero_shot"))
        self.assertTrue(validate_variant(campaign, "autosa_aav_n_gf"))
        self.assertTrue(validate_variant(campaign, "autosa_aav_n_90"))
        configure_autosa_campaign_env("autosa_noskills")
        self.assertEqual(os.environ["C2HLS_SKILL_MODE"], "skill_off")
        self.assertNotIn("C2HLS_PACKAGED_SKILLS_JSON", os.environ)
        configure_autosa_campaign_env("autosa_zero_shot")
        self.assertEqual(os.environ["C2HLS_SKIP_PHASE_B"], "1")
        self.assertEqual(os.environ["C2HLS_FLASH_OPT_PROMPT_MODE"], "zero_shot")

    def test_apply_noskills_keeps_dse_stream_and_sets_mmns(self) -> None:
        from autosa_flash_lib import apply_mm_flow_flavor

        snap = apply_mm_flow_flavor("noskills")
        self.assertEqual(snap["flavor"], "noskills")
        self.assertEqual(os.environ["BATCH_PARALLEL_VARIANT"], "autosa_noskills")
        self.assertEqual(os.environ["PC2_BATCH_JOB_PREFIX"], "mmns")
        self.assertEqual(
            os.environ["BATCH_PARALLEL_ARTIFACT_PREFIX"],
            "batch_parallel_autosa_mm_flow_noskills",
        )
        self.assertEqual(os.environ["C2HLS_POST_FLASH_DSE"], "1")
        self.assertEqual(os.environ["C2HLS_POST_FLASH_STREAM"], "1")
        self.assertEqual(os.environ["C2HLS_POST_FLASH_NO_SKILLS"], "1")
        self.assertNotIn("C2HLS_PACKAGED_SKILLS_JSON", os.environ)

    def test_apply_zero_shot_disables_dse_stream_and_sets_turns_1(self) -> None:
        from autosa_flash_lib import apply_mm_flow_flavor

        snap = apply_mm_flow_flavor("zero_shot")
        self.assertEqual(snap["flavor"], "zero_shot")
        self.assertEqual(os.environ["BATCH_PARALLEL_VARIANT"], "autosa_zero_shot")
        self.assertEqual(os.environ["PC2_BATCH_JOB_PREFIX"], "mmzs")
        self.assertEqual(
            os.environ["BATCH_PARALLEL_ARTIFACT_PREFIX"],
            "batch_parallel_autosa_mm_flow_zero_shot",
        )
        self.assertEqual(os.environ["C2HLS_POST_FLASH_DSE"], "0")
        self.assertEqual(os.environ["C2HLS_POST_FLASH_STREAM"], "0")
        self.assertEqual(os.environ["C2HLS_TURNS"], "1")
        self.assertEqual(os.environ["C2HLS_SKIP_PHASE_B"], "1")
        self.assertEqual(os.environ["C2HLS_FLASH_OPT_PROMPT_MODE"], "zero_shot")

    def test_apply_zero_shot_rejects_pe_recipe(self) -> None:
        from autosa_flash_lib import apply_mm_flow_flavor

        os.environ["C2HLS_PE_RECIPE"] = "autosa_mm_32x8"
        with self.assertRaises(ValueError):
            apply_mm_flow_flavor("zero_shot")

    def test_apply_one_shot_skips_phase_b_and_downstream(self) -> None:
        from autosa_flash_lib import apply_mm_flow_flavor
        from prompt_c2hls import flash_optimization_prompt, q_optimize_zero_shot_direct

        os.environ["C2HLS_FLASH_MIN_DSP"] = "300"
        os.environ["C2HLS_PACKAGED_SKILLS_JSON"] = "/tmp/skills.json"
        snap = apply_mm_flow_flavor("one_shot")
        self.assertEqual(snap["flavor"], "one_shot")
        self.assertEqual(os.environ["BATCH_PARALLEL_VARIANT"], "autosa_one_shot")
        self.assertEqual(os.environ["PC2_BATCH_JOB_PREFIX"], "mm1s")
        self.assertEqual(
            os.environ["BATCH_PARALLEL_ARTIFACT_PREFIX"],
            "batch_parallel_autosa_mm_flow_one_shot",
        )
        self.assertEqual(os.environ["C2HLS_ONE_SHOT"], "1")
        self.assertEqual(os.environ["C2HLS_SKIP_PHASE_B"], "1")
        self.assertEqual(os.environ["C2HLS_FLASH_OPT_PROMPT_MODE"], "zero_shot")
        self.assertEqual(os.environ["C2HLS_POST_FLASH_DSE"], "0")
        self.assertEqual(os.environ["C2HLS_POST_FLASH_STREAM"], "0")
        self.assertEqual(os.environ["C2HLS_FLASH_ONLY"], "1")
        self.assertNotIn("C2HLS_PACKAGED_SKILLS_JSON", os.environ)
        self.assertNotIn("C2HLS_FLASH_MIN_DSP", os.environ)
        prompt = flash_optimization_prompt(zero_shot=True, skip_phase_b=True)
        self.assertEqual(prompt, q_optimize_zero_shot_direct)

    def test_dispatch_accepts_one_shot_variant(self) -> None:
        from batch_parallel_dispatch import validate_variant

        campaign = {"config": {"pilot": {"workflow": "autosa_flash"}}}
        self.assertTrue(validate_variant(campaign, "autosa_one_shot"))

    def test_apply_one_shot_rejects_pe_recipe(self) -> None:
        from autosa_flash_lib import apply_mm_flow_flavor

        os.environ["C2HLS_PE_RECIPE"] = "autosa_mm_32x8"
        with self.assertRaises(ValueError):
            apply_mm_flow_flavor("oneshot")

    def test_apply_skills_keeps_gemm_pack_if_already_set(self) -> None:
        from autosa_flash_lib import SKILLS_90_GEMM_JSON, apply_mm_flow_flavor

        pack = str(SKILLS_90_GEMM_JSON.resolve())
        os.environ["C2HLS_PACKAGED_SKILLS_JSON"] = pack
        os.environ["C2HLS_SKILL_PROMPT_MODE"] = "all_skills_avoids_global"
        snap = apply_mm_flow_flavor("skills")
        self.assertEqual(snap["flavor"], "skills")
        self.assertEqual(os.environ["C2HLS_PACKAGED_SKILLS_JSON"], pack)
        self.assertEqual(os.environ["C2HLS_SKILL_PROMPT_MODE"], "all_skills_avoids_global")
        self.assertEqual(os.environ.get("BATCH_PARALLEL_VARIANT", "autosa_aav_n_gf"), "autosa_aav_n_gf")
        self.assertEqual(os.environ.get("PC2_BATCH_JOB_PREFIX", "mmflow"), "mmflow")

    def test_apply_aav_n_90_drops_overlay_and_gemm_flatten_pack(self) -> None:
        from autosa_flash_lib import SKILLS_90_JSON, apply_mm_flow_flavor

        os.environ["C2HLS_PACKAGED_SKILLS_JSON"] = "/tmp/gemm_flatten_v1.json"
        os.environ["C2HLS_FLASH_SKILL_ENTRIES_JSON"] = "/tmp/overlay.json"
        os.environ["BATCH_PARALLEL_VARIANT"] = "autosa_aav_n_gf"
        snap = apply_mm_flow_flavor("90only")
        self.assertEqual(snap["flavor"], "aav_n_90")
        self.assertEqual(os.environ["BATCH_PARALLEL_VARIANT"], "autosa_aav_n_90")
        self.assertEqual(os.environ["PC2_BATCH_JOB_PREFIX"], "mm90")
        self.assertEqual(
            os.environ["BATCH_PARALLEL_ARTIFACT_PREFIX"],
            "batch_parallel_autosa_mm_flow_aav_n_90",
        )
        self.assertEqual(
            os.environ["C2HLS_PACKAGED_SKILLS_JSON"], str(SKILLS_90_JSON.resolve())
        )
        self.assertNotIn("gemm_flatten", os.environ["C2HLS_PACKAGED_SKILLS_JSON"])
        self.assertEqual(os.environ["C2HLS_PACKAGED_SKILLS_ONLY"], "1")
        self.assertNotIn("C2HLS_FLASH_SKILL_ENTRIES_JSON", os.environ)
        self.assertEqual(os.environ["C2HLS_POST_FLASH_DSE"], "1")
        self.assertEqual(os.environ["C2HLS_POST_FLASH_STREAM"], "1")

    def test_unknown_flavor_raises(self) -> None:
        from autosa_flash_lib import apply_mm_flow_flavor

        with self.assertRaises(ValueError):
            apply_mm_flow_flavor("rag2")


class AutosaMmFlowNoSkillsPromptTests(unittest.TestCase):
    def test_dse_skills_block_empty_when_flag_set(self) -> None:
        import post_flash_dse as pfd

        prev = os.environ.get("C2HLS_POST_FLASH_NO_SKILLS")
        os.environ["C2HLS_POST_FLASH_NO_SKILLS"] = "1"
        try:
            block, meta = pfd.build_dse_skills_prompt_block()
            self.assertEqual(block, "")
            self.assertEqual(meta["skill_count"], 0)
            self.assertEqual(meta["skill_ids"], [])
            self.assertEqual(meta["skills_path"], "")
            prompts = pfd.prompt_text_for_docs()
            self.assertNotIn("hls-dse-gemm-multi-pe-latency-hiding", prompts["initial_user"])
            self.assertNotIn("using the DSE skills below", prompts["initial_user"])
            self.assertNotIn("## DSE skills", prompts["initial_user"])
        finally:
            if prev is None:
                os.environ.pop("C2HLS_POST_FLASH_NO_SKILLS", None)
            else:
                os.environ["C2HLS_POST_FLASH_NO_SKILLS"] = prev

    def test_stream_skills_block_empty_when_flag_set(self) -> None:
        import post_flash_stream as pfs

        prev = os.environ.get("C2HLS_POST_FLASH_NO_SKILLS")
        os.environ["C2HLS_POST_FLASH_NO_SKILLS"] = "1"
        try:
            block, meta = pfs.build_stream_skills_prompt_block()
            self.assertEqual(block, "")
            self.assertEqual(meta["skill_count"], 0)
            self.assertEqual(meta["skill_ids"], [])
            prompts = pfs.prompt_text_for_docs()
            user = prompts["initial_user"]
            self.assertNotIn("hls-stream-pe-array-dataflow", user)
            self.assertNotIn("skills below", user.lower())
            self.assertNotIn("Stream skills (follow these", user)
        finally:
            if prev is None:
                os.environ.pop("C2HLS_POST_FLASH_NO_SKILLS", None)
            else:
                os.environ["C2HLS_POST_FLASH_NO_SKILLS"] = prev

    def test_dse_default_still_loads_skill_json(self) -> None:
        import post_flash_dse as pfd

        prev = os.environ.get("C2HLS_POST_FLASH_NO_SKILLS")
        os.environ.pop("C2HLS_POST_FLASH_NO_SKILLS", None)
        try:
            block, meta = pfd.build_dse_skills_prompt_block()
            self.assertGreater(meta["skill_count"], 0)
            self.assertIn("hls-dse-gemm-multi-pe-latency-hiding", meta["skill_ids"])
            self.assertIn("hls-dse-gemm-multi-pe-latency-hiding", block)
        finally:
            if prev is None:
                os.environ.pop("C2HLS_POST_FLASH_NO_SKILLS", None)
            else:
                os.environ["C2HLS_POST_FLASH_NO_SKILLS"] = prev


if __name__ == "__main__":
    unittest.main()
