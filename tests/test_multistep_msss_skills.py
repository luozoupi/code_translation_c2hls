#!/usr/bin/env python3
"""msss injects per-step skill files; aav_n still uses the global dump."""

from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "pc2"))

from batch_parallel_dispatch import (  # noqa: E402
    is_autosa_multistep_workflow,
    resolve_bench_map,
    validate_variant,
)
from batch_parallel_config import BatchParallelConfig, seed_kwargs_for_workflow  # noqa: E402
from batch_parallel_multistep_lib import (  # noqa: E402
    AUTOSA_MS_AAV_N_VARIANT,
    AUTOSA_MS_MSSS_VARIANT,
    autosa_repeat_source,
    configure_autosa_multistep_campaign_env,
    resolve_autosa_multistep_bench_map,
)
from skill_library import msss_skills_for_step  # noqa: E402

_ENV_KEYS = (
    "C2HLS_PACKAGED_SKILLS_JSON",
    "C2HLS_FLASH_SKILL_ENTRIES_JSON",
    "C2HLS_PACKAGED_SKILLS_ONLY",
    "C2HLS_SKILL_PROMPT_MODE",
    "C2HLS_FORCE_SKILL_PROMPTS",
    "C2HLS_STRATEGY",
    "C2HLS_RAG",
    "C2HLS_RAG2",
    "C2HLS_POST_FLASH_LATENCY_OPT",
    "C2HLS_MULTISTEP_SKIP_FINAL_COSIM",
    "C2HLS_MSSS_SKILLS_DIR",
    "C2HLS_AUTOSA_MS_AAV_PACK",
    "C2HLS_AUTOSA_MS_MSSS_PACK",
)


class MultistepMsssSkillsTest(unittest.TestCase):
    def test_global_modes_include_msss(self) -> None:
        src = (REPO / "c2hls.py").read_text(encoding="utf-8")
        self.assertIn('"msss"', src)
        self.assertIn("GLOBAL_SKILL_PROMPT_MODES", src)
        self.assertIn('skill_mode == "msss"', src)
        self.assertIn("msss_skills_for_step", src)
        self.assertLess(src.index('skill_mode == "msss"'), src.index('skill_mode == "all_skills_avoids_global"'))

    def test_tiling_file_excludes_coalescing_only_skill(self) -> None:
        skills = msss_skills_for_step("tiling")
        ids = {sk.id for sk in skills}
        self.assertIn("prompt-tiling", ids)
        self.assertNotIn("axi-burst-coalescing-narrow-safe", ids)
        self.assertTrue(any(sk.confidence == "avoid" for sk in skills))

    def test_aav_n_keeps_global_prompt_mode(self) -> None:
        saved = {k: os.environ.get(k) for k in _ENV_KEYS}
        for k in _ENV_KEYS:
            os.environ.pop(k, None)
        try:
            configure_autosa_multistep_campaign_env(AUTOSA_MS_AAV_N_VARIANT)
            self.assertEqual(os.environ["C2HLS_STRATEGY"], "static")
            self.assertEqual(os.environ["C2HLS_SKILL_PROMPT_MODE"], "all_skills_avoids_global")
            self.assertTrue(os.environ["C2HLS_PACKAGED_SKILLS_JSON"].endswith("gemm_flatten_v1.json"))
            self.assertTrue(
                os.environ["C2HLS_FLASH_SKILL_ENTRIES_JSON"].endswith(
                    "flash_no_RMW_m_axi_skill_entries.json"
                )
            )
            self.assertEqual(os.environ["C2HLS_RAG"], "0")
            self.assertEqual(os.environ["C2HLS_RAG2"], "0")
            self.assertEqual(os.environ["C2HLS_POST_FLASH_LATENCY_OPT"], "0")
        finally:
            for key, val in saved.items():
                if val is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = val

    def test_msss_uses_step_files_not_global_dump(self) -> None:
        saved = {k: os.environ.get(k) for k in _ENV_KEYS}
        for k in _ENV_KEYS:
            os.environ.pop(k, None)
        try:
            configure_autosa_multistep_campaign_env(AUTOSA_MS_MSSS_VARIANT)
            self.assertEqual(os.environ["C2HLS_STRATEGY"], "static")
            self.assertEqual(os.environ["C2HLS_SKILL_PROMPT_MODE"], "msss")
            self.assertFalse((os.environ.get("C2HLS_PACKAGED_SKILLS_JSON") or "").strip())
            self.assertFalse((os.environ.get("C2HLS_FLASH_SKILL_ENTRIES_JSON") or "").strip())
            self.assertEqual(os.environ["C2HLS_FORCE_SKILL_PROMPTS"], "1")
            self.assertEqual(os.environ["C2HLS_RAG"], "0")
            self.assertEqual(os.environ["C2HLS_RAG2"], "0")
            self.assertEqual(os.environ["C2HLS_POST_FLASH_LATENCY_OPT"], "0")
        finally:
            for key, val in saved.items():
                if val is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = val

    def test_aav_n_90_v2_pack_dumps_union_without_overlay(self) -> None:
        saved = {k: os.environ.get(k) for k in _ENV_KEYS}
        for k in _ENV_KEYS:
            os.environ.pop(k, None)
        try:
            os.environ["C2HLS_AUTOSA_MS_AAV_PACK"] = "90_v2"
            configure_autosa_multistep_campaign_env(AUTOSA_MS_AAV_N_VARIANT)
            self.assertEqual(os.environ["C2HLS_SKILL_PROMPT_MODE"], "all_skills_avoids_global")
            self.assertTrue(
                os.environ["C2HLS_PACKAGED_SKILLS_JSON"].endswith(
                    "skills_ii_target_miss_solutions_added(90skills)_plus_gemm_flatten_v2.json"
                )
            )
            self.assertFalse((os.environ.get("C2HLS_FLASH_SKILL_ENTRIES_JSON") or "").strip())
        finally:
            for key, val in saved.items():
                if val is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = val

    def test_msss_v1_normw_pack_points_at_v1_slice(self) -> None:
        saved = {k: os.environ.get(k) for k in _ENV_KEYS}
        for k in _ENV_KEYS:
            os.environ.pop(k, None)
        try:
            os.environ["C2HLS_AUTOSA_MS_MSSS_PACK"] = "v1_normw"
            configure_autosa_multistep_campaign_env(AUTOSA_MS_MSSS_VARIANT)
            self.assertEqual(os.environ["C2HLS_SKILL_PROMPT_MODE"], "msss")
            self.assertTrue(os.environ["C2HLS_MSSS_SKILLS_DIR"].endswith("multistep_gf_v1_normw"))
            self.assertFalse((os.environ.get("C2HLS_PACKAGED_SKILLS_JSON") or "").strip())
            skills = msss_skills_for_step("tiling")
            ids = {sk.id for sk in skills}
            self.assertIn("hls-load-compute-store-no-rmw-m_axi", ids)
            self.assertNotIn("hls-fp-mac-k-step-independent-acc-banks", ids)
        finally:
            for key, val in saved.items():
                if val is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = val

    def test_campaign_configs_are_ten_vs_ten(self) -> None:
        expected = [f"autosa_mm_r{i:02d}" for i in range(1, 11)]
        for name in (
            "batch_parallel_autosa_mm_multistep_aav_n.json",
            "batch_parallel_autosa_mm_multistep_msss.json",
            "batch_parallel_autosa_mm_multistep_msss_v1n.json",
            "batch_parallel_autosa_mm_multistep_aav_n_90v2.json",
            "batch_parallel_autosa_mm1024_multistep_msss_v1n.json",
            "batch_parallel_autosa_mm1024_multistep_aav_n_90v2.json",
        ):
            doc = json.loads((REPO / "scripts" / "pc2" / name).read_text(encoding="utf-8"))
            self.assertEqual(doc["pilot"]["benches"], expected)
            self.assertEqual(doc["synth_nodes_per_variant"], 1)
            self.assertEqual(doc["max_inflight_benches"], 1)

    def test_dedicated_proxy_raises_max_queue(self) -> None:
        dedicated = (REPO / "scripts" / "pc2" / "start_dedicated_deepseek_proxy.sh").read_text(
            encoding="utf-8"
        )
        chathls = (
            REPO.parent
            / "test-chathls"
            / "ChatHLS-ACL-26"
            / "scripts"
            / "pc2"
            / "start_deepseek_queue_proxy.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("CHATHLS_DEEPSEEK_MAX_QUEUE", dedicated)
        self.assertIn("CHATHLS_DEEPSEEK_MAX_QUEUE:-32", dedicated)
        self.assertIn("--max-queue", chathls)
        self.assertIn("CHATHLS_DEEPSEEK_MAX_QUEUE", chathls)

    def test_repeat_alias_and_workflow(self) -> None:
        self.assertEqual(autosa_repeat_source("autosa_mm_r01"), "autosa_mm")
        campaign = {"config": {"pilot": {"workflow": "autosa_multistep"}}}
        self.assertTrue(is_autosa_multistep_workflow(campaign))
        self.assertTrue(validate_variant(campaign, AUTOSA_MS_AAV_N_VARIANT))
        self.assertTrue(validate_variant(campaign, AUTOSA_MS_MSSS_VARIANT))
        self.assertEqual(seed_kwargs_for_workflow("autosa_multistep"), {})
        ready = REPO / "related_work" / "benchmarks" / "autosa_ready" / "autosa_mm"
        if ready.is_dir():
            mapped = resolve_autosa_multistep_bench_map(["autosa_mm_r01", "autosa_mm_r02"])
            self.assertEqual(mapped["autosa_mm_r01"], ready)
            self.assertEqual(mapped["autosa_mm_r02"], ready)
            cfg = BatchParallelConfig()
            self.assertEqual(
                resolve_bench_map(campaign, cfg, ["autosa_mm_r01"])["autosa_mm_r01"],
                ready,
            )


if __name__ == "__main__":
    unittest.main()
