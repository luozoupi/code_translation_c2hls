"""Bare HLS-opt prompts experiment arm (C2HLS_BARE_OPT_PROMPTS)."""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import c2hls  # noqa: E402 requires project .venv (openai); run via .venv/bin/python -m pytest
import post_flash_dataflow as pfd  # noqa: E402


class _EnvVarScope:
    """Save/restore a set of env vars around a test."""

    def __init__(self, keys):
        self._keys = list(keys)
        self._saved: dict[str, str | None] = {}

    def __enter__(self):
        for key in self._keys:
            self._saved[key] = os.environ.get(key)
        return self

    def __exit__(self, exc_type, exc, tb):
        for key, value in self._saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


class BareOptPromptsFlashGateTests(unittest.TestCase):
    _ENV_KEYS = ("C2HLS_BARE_OPT_PROMPTS", "C2HLS_FLASH_OPT_PROMPT_MODE")

    def test_bare_env_forces_zero_shot_flash_prompt(self) -> None:
        with _EnvVarScope(self._ENV_KEYS):
            os.environ.pop("C2HLS_FLASH_OPT_PROMPT_MODE", None)
            os.environ["C2HLS_BARE_OPT_PROMPTS"] = "1"
            self.assertTrue(c2hls._bare_opt_prompts_enabled())
            self.assertTrue(c2hls._flash_opt_prompt_zero_shot())

    def test_bare_env_accepts_common_truthy_spellings(self) -> None:
        with _EnvVarScope(self._ENV_KEYS):
            os.environ.pop("C2HLS_FLASH_OPT_PROMPT_MODE", None)
            for value in ("true", "Yes", "ON"):
                os.environ["C2HLS_BARE_OPT_PROMPTS"] = value
                self.assertTrue(c2hls._flash_opt_prompt_zero_shot(), value)

    def test_without_bare_or_zero_shot_mode_flash_prompt_is_not_zero_shot(self) -> None:
        with _EnvVarScope(self._ENV_KEYS):
            os.environ.pop("C2HLS_BARE_OPT_PROMPTS", None)
            os.environ.pop("C2HLS_FLASH_OPT_PROMPT_MODE", None)
            self.assertFalse(c2hls._flash_opt_prompt_zero_shot())

    def test_zero_shot_mode_alone_still_works(self) -> None:
        with _EnvVarScope(self._ENV_KEYS):
            os.environ.pop("C2HLS_BARE_OPT_PROMPTS", None)
            os.environ["C2HLS_FLASH_OPT_PROMPT_MODE"] = "zero_shot"
            self.assertTrue(c2hls._flash_opt_prompt_zero_shot())


class BareOptPromptsDataflowBlockTests(unittest.TestCase):
    _ENV_KEYS = ("C2HLS_BARE_OPT_PROMPTS", "C2HLS_DATAFLOW_NO_SKILLS")

    def test_bare_block_omits_skills_and_dataflow_mandate_language(self) -> None:
        with _EnvVarScope(self._ENV_KEYS):
            os.environ.pop("C2HLS_DATAFLOW_NO_SKILLS", None)
            os.environ["C2HLS_BARE_OPT_PROMPTS"] = "1"
            block, meta = pfd.build_dataflow_skills_prompt_block()
            self.assertNotIn("FLASH HLS OPTIMIZATION SKILLS", block)
            self.assertNotIn("#pragma HLS DATAFLOW", block)
            self.assertIn("Bare mode", block)
            self.assertTrue(meta.get("bare"))
            self.assertEqual(meta.get("skill_count"), 0)

    def test_bare_block_omits_mandate_even_when_noskills_also_set(self) -> None:
        with _EnvVarScope(self._ENV_KEYS):
            os.environ["C2HLS_DATAFLOW_NO_SKILLS"] = "1"
            os.environ["C2HLS_BARE_OPT_PROMPTS"] = "1"
            block, meta = pfd.build_dataflow_skills_prompt_block()
            self.assertNotIn("FLASH HLS OPTIMIZATION SKILLS", block)
            self.assertNotIn("#pragma HLS DATAFLOW", block)
            self.assertTrue(meta.get("bare"))

    def test_noskills_without_bare_keeps_existing_dataflow_rule_block(self) -> None:
        with _EnvVarScope(self._ENV_KEYS):
            os.environ.pop("C2HLS_BARE_OPT_PROMPTS", None)
            os.environ["C2HLS_DATAFLOW_NO_SKILLS"] = "1"
            block, meta = pfd.build_dataflow_skills_prompt_block()
            self.assertIn("#pragma HLS DATAFLOW", block)
            self.assertIn("No packaged skills", block)
            self.assertTrue(meta.get("noskills"))
            self.assertNotIn("bare", meta)


if __name__ == "__main__":
    unittest.main()
