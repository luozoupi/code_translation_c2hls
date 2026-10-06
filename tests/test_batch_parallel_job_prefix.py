"""Tests for batch_parallel Slurm job-name prefixes."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "pc2"))

from batch_parallel_config import (
    BatchParallelConfig,
    apply_autosa_flow_from_campaign,
    campaign_job_prefix,
    init_campaign_json,
    load_config,
)


class BatchParallelJobPrefixTests(unittest.TestCase):
    def test_full_flash_config_uses_bpfcosim(self) -> None:
        import os

        config = REPO / "scripts/pc2/batch_parallel_full_aav_n_park.json"
        prev = os.environ.get("BATCH_PARALLEL_CONFIG")
        os.environ["BATCH_PARALLEL_CONFIG"] = str(config)
        try:
            cfg = load_config()
            self.assertEqual(cfg.job_prefix, "bpfcosim")
        finally:
            if prev is None:
                os.environ.pop("BATCH_PARALLEL_CONFIG", None)
            else:
                os.environ["BATCH_PARALLEL_CONFIG"] = prev

    def test_campaign_job_prefix_top_level(self) -> None:
        doc = {"job_prefix": "bpfcosim", "config": {"job_prefix": "bpcplx"}}
        self.assertEqual(campaign_job_prefix(doc), "bpfcosim")

    def test_init_campaign_json_stores_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cfg = BatchParallelConfig(job_prefix="bpfcosim")
            doc = init_campaign_json(root, cfg, stamp="test_stamp")
            self.assertEqual(doc["job_prefix"], "bpfcosim")
            saved = json.loads((root / "campaign.json").read_text())
            self.assertEqual(saved["job_prefix"], "bpfcosim")
            self.assertEqual(saved["config"]["job_prefix"], "bpfcosim")

    def test_init_campaign_json_copies_enforcement_from_env(self) -> None:
        import os

        prev_on = os.environ.get("C2HLS_ENFORCEMENT")
        prev_rounds = os.environ.get("C2HLS_ENFORCEMENT_ROUNDS")
        os.environ["C2HLS_ENFORCEMENT"] = "1"
        os.environ["C2HLS_ENFORCEMENT_ROUNDS"] = "20"
        try:
            with tempfile.TemporaryDirectory() as td:
                root = Path(td)
                cfg = BatchParallelConfig(job_prefix="flshenf")
                doc = init_campaign_json(root, cfg, stamp="enf_stamp")
                self.assertTrue(doc.get("enforcement"))
                self.assertEqual(doc.get("enforcement_rounds"), 20)
                saved = json.loads((root / "campaign.json").read_text())
                self.assertTrue(saved["enforcement"])
                self.assertEqual(saved["enforcement_rounds"], 20)
        finally:
            if prev_on is None:
                os.environ.pop("C2HLS_ENFORCEMENT", None)
            else:
                os.environ["C2HLS_ENFORCEMENT"] = prev_on
            if prev_rounds is None:
                os.environ.pop("C2HLS_ENFORCEMENT_ROUNDS", None)
            else:
                os.environ["C2HLS_ENFORCEMENT_ROUNDS"] = prev_rounds

    def test_init_campaign_json_copies_synth_timeout_from_env(self) -> None:
        import os

        prev = os.environ.get("C2HLS_SYNTH_TIMEOUT")
        os.environ["C2HLS_SYNTH_TIMEOUT"] = "3600"
        try:
            with tempfile.TemporaryDirectory() as td:
                root = Path(td)
                cfg = BatchParallelConfig(job_prefix="mmenf")
                doc = init_campaign_json(root, cfg, stamp="to_stamp")
                self.assertEqual(doc.get("synth_timeout"), 3600)
                saved = json.loads((root / "campaign.json").read_text())
                self.assertEqual(saved["synth_timeout"], 3600)
        finally:
            if prev is None:
                os.environ.pop("C2HLS_SYNTH_TIMEOUT", None)
            else:
                os.environ["C2HLS_SYNTH_TIMEOUT"] = prev

    def test_apply_campaign_synth_timeout_overrides_meta(self) -> None:
        import os

        from batch_parallel_config import apply_campaign_synth_timeout

        prev = os.environ.get("C2HLS_SYNTH_TIMEOUT")
        os.environ["C2HLS_SYNTH_TIMEOUT"] = "14400"
        try:
            apply_campaign_synth_timeout({"synth_timeout": 3600})
            self.assertEqual(os.environ["C2HLS_SYNTH_TIMEOUT"], "3600")
        finally:
            if prev is None:
                os.environ.pop("C2HLS_SYNTH_TIMEOUT", None)
            else:
                os.environ["C2HLS_SYNTH_TIMEOUT"] = prev

    def test_mm_flow_config_uses_mmflow(self) -> None:
        import os

        config = REPO / "scripts/pc2/batch_parallel_autosa_mm_flow.json"
        prev = os.environ.get("BATCH_PARALLEL_CONFIG")
        prev_prefix = os.environ.get("PC2_BATCH_JOB_PREFIX")
        os.environ["BATCH_PARALLEL_CONFIG"] = str(config)
        os.environ.pop("PC2_BATCH_JOB_PREFIX", None)
        try:
            cfg = load_config()
            self.assertEqual(cfg.job_prefix, "mmflow")
        finally:
            if prev is None:
                os.environ.pop("BATCH_PARALLEL_CONFIG", None)
            else:
                os.environ["BATCH_PARALLEL_CONFIG"] = prev
            if prev_prefix is None:
                os.environ.pop("PC2_BATCH_JOB_PREFIX", None)
            else:
                os.environ["PC2_BATCH_JOB_PREFIX"] = prev_prefix

    def test_init_campaign_json_copies_autosa_flow_from_env(self) -> None:
        import os

        keys = (
            "C2HLS_AUTOSA_FLOW",
            "C2HLS_ENFORCEMENT",
            "C2HLS_POST_FLASH_DSE",
            "C2HLS_DSE_CHAIN_FLASH",
            "C2HLS_POST_FLASH_STREAM",
            "C2HLS_STREAM_CHAIN_FLASH",
        )
        prev = {k: os.environ.get(k) for k in keys}
        os.environ["C2HLS_AUTOSA_FLOW"] = "1"
        os.environ["C2HLS_ENFORCEMENT"] = "0"
        os.environ["C2HLS_POST_FLASH_DSE"] = "1"
        os.environ["C2HLS_DSE_CHAIN_FLASH"] = "1"
        os.environ["C2HLS_POST_FLASH_STREAM"] = "1"
        os.environ["C2HLS_STREAM_CHAIN_FLASH"] = "1"
        try:
            with tempfile.TemporaryDirectory() as td:
                root = Path(td)
                cfg = BatchParallelConfig(job_prefix="mmflow")
                doc = init_campaign_json(root, cfg, stamp="flow_stamp")
                self.assertTrue(doc.get("autosa_flow"))
                self.assertTrue(doc.get("post_flash_dse"))
                self.assertTrue(doc.get("dse_chain_flash"))
                self.assertTrue(doc.get("post_flash_stream"))
                self.assertTrue(doc.get("stream_chain_flash"))
                self.assertIsNot(doc.get("enforcement"), True)
                saved = json.loads((root / "campaign.json").read_text())
                self.assertTrue(saved["autosa_flow"])
                self.assertTrue(saved["post_flash_dse"])
                self.assertTrue(saved["dse_chain_flash"])
                self.assertTrue(saved["post_flash_stream"])
                self.assertTrue(saved["stream_chain_flash"])
                self.assertIsNot(saved.get("enforcement"), True)
        finally:
            for key, value in prev.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    def test_init_campaign_json_honors_dse_off_for_zero_shot(self) -> None:
        import os

        keys = (
            "C2HLS_AUTOSA_FLOW",
            "C2HLS_POST_FLASH_DSE",
            "C2HLS_DSE_CHAIN_FLASH",
            "C2HLS_POST_FLASH_STREAM",
            "C2HLS_STREAM_CHAIN_FLASH",
            "C2HLS_SKIP_PHASE_B",
            "C2HLS_MM_FLOW_FLAVOR",
            "C2HLS_TURNS",
        )
        prev = {k: os.environ.get(k) for k in keys}
        os.environ["C2HLS_AUTOSA_FLOW"] = "1"
        os.environ["C2HLS_POST_FLASH_DSE"] = "0"
        os.environ["C2HLS_DSE_CHAIN_FLASH"] = "0"
        os.environ["C2HLS_POST_FLASH_STREAM"] = "0"
        os.environ["C2HLS_STREAM_CHAIN_FLASH"] = "0"
        os.environ["C2HLS_SKIP_PHASE_B"] = "1"
        os.environ["C2HLS_MM_FLOW_FLAVOR"] = "zero_shot"
        os.environ["C2HLS_TURNS"] = "1"
        try:
            with tempfile.TemporaryDirectory() as td:
                root = Path(td)
                cfg = BatchParallelConfig(job_prefix="mmzs")
                doc = init_campaign_json(root, cfg, stamp="zs_stamp")
                self.assertTrue(doc.get("autosa_flow"))
                self.assertFalse(doc.get("post_flash_dse"))
                self.assertFalse(doc.get("post_flash_stream"))
                self.assertTrue(doc.get("skip_phase_b"))
                self.assertEqual(doc.get("mm_flow_flavor"), "zero_shot")
                self.assertEqual(doc.get("turns"), 1)
        finally:
            for key, value in prev.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    def test_init_campaign_json_copies_pe_recipe_from_env(self) -> None:
        import os

        prev = os.environ.get("C2HLS_PE_RECIPE")
        os.environ["C2HLS_PE_RECIPE"] = "autosa_mm_32x8"
        try:
            with tempfile.TemporaryDirectory() as td:
                root = Path(td)
                cfg = BatchParallelConfig(job_prefix="mm32x8")
                doc = init_campaign_json(root, cfg, stamp="x8_stamp")
                self.assertEqual(doc.get("pe_recipe"), "autosa_mm_32x8")
                saved = json.loads((root / "campaign.json").read_text())
                self.assertEqual(saved["pe_recipe"], "autosa_mm_32x8")
                os.environ.pop("C2HLS_PE_RECIPE", None)
                apply_autosa_flow_from_campaign(saved)
                self.assertEqual(os.environ.get("C2HLS_PE_RECIPE"), "autosa_mm_32x8")
        finally:
            if prev is None:
                os.environ.pop("C2HLS_PE_RECIPE", None)
            else:
                os.environ["C2HLS_PE_RECIPE"] = prev

    def test_init_campaign_json_copies_flash_token_budget(self) -> None:
        import os

        keys = ("C2HLS_FLASH_MAX_TOKENS", "C2HLS_LLM_MAX_TOKENS", "C2HLS_CPP_CONTINUATIONS")
        prev = {k: os.environ.get(k) for k in keys}
        os.environ["C2HLS_FLASH_MAX_TOKENS"] = "65536"
        os.environ["C2HLS_LLM_MAX_TOKENS"] = "65536"
        os.environ["C2HLS_CPP_CONTINUATIONS"] = "8"
        try:
            with tempfile.TemporaryDirectory() as td:
                root = Path(td)
                cfg = BatchParallelConfig(job_prefix="mmzs")
                doc = init_campaign_json(root, cfg, stamp="tok_stamp")
                self.assertEqual(doc.get("flash_max_tokens"), 65536)
                self.assertEqual(doc.get("cpp_continuations"), 8)
                for key in keys:
                    os.environ.pop(key, None)
                apply_autosa_flow_from_campaign(doc)
                self.assertEqual(os.environ.get("C2HLS_FLASH_MAX_TOKENS"), "65536")
                self.assertEqual(os.environ.get("C2HLS_CPP_CONTINUATIONS"), "8")
        finally:
            for key, value in prev.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    def test_init_campaign_json_copies_thinking_mode(self) -> None:
        import os

        prev = os.environ.get("C2HLS_THINKING")
        os.environ["C2HLS_THINKING"] = "disabled"
        try:
            with tempfile.TemporaryDirectory() as td:
                root = Path(td)
                cfg = BatchParallelConfig(job_prefix="mmflow")
                doc = init_campaign_json(root, cfg, stamp="nthink")
                self.assertEqual(doc.get("thinking"), "disabled")
                os.environ.pop("C2HLS_THINKING", None)
                apply_autosa_flow_from_campaign(doc)
                self.assertEqual(os.environ.get("C2HLS_THINKING"), "disabled")
        finally:
            if prev is None:
                os.environ.pop("C2HLS_THINKING", None)
            else:
                os.environ["C2HLS_THINKING"] = prev


if __name__ == "__main__":
    unittest.main()
