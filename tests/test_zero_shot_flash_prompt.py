"""Tests for zero-shot flash prompt selection."""

from __future__ import annotations

import unittest

from prompt_c2hls import (
    Instruction_c2hls,
    Instruction_c2hls_flash,
    Instruction_c2hls_zero_shot,
    flash_optimization_prompt,
    llm_messages,
    llm_system_instruction,
    q_optimize_zero_shot_direct,
    q_optimize_zero_shot_phaseb,
    q_translate_zero_shot,
)

_ZERO_SHOT_USER_OPENING = (
    "You are an expert HLS engineer. Translate and optimize the following C kernel "
    "into synthesizable Vitis HLS C++."
)


class TestZeroShotFlashPrompt(unittest.TestCase):
    def test_zero_shot_system_instruction(self) -> None:
        self.assertIn("expert in FPGA High-Level Synthesis", Instruction_c2hls_zero_shot)
        self.assertIn("translated_hls_top", Instruction_c2hls_zero_shot)
        self.assertNotIn("Key HLS optimization techniques", Instruction_c2hls_zero_shot)
        self.assertEqual(
            llm_system_instruction(zero_shot=True, step_name="flash"),
            Instruction_c2hls_zero_shot,
        )
        self.assertEqual(
            llm_system_instruction(zero_shot=True, translate=True),
            Instruction_c2hls_zero_shot,
        )

    def test_phaseb_template_has_expert_user_prefix(self) -> None:
        self.assertNotIn("{synth_report}", q_optimize_zero_shot_phaseb)
        rendered = q_optimize_zero_shot_phaseb.format(
            header_code="// hdr",
            current_code="void k() {}",
        )
        self.assertIn(_ZERO_SHOT_USER_OPENING, rendered)
        self.assertIn("void k() {}", rendered)
        self.assertNotIn("benchmark_context", rendered)

    def test_direct_template_has_expert_user_prefix(self) -> None:
        self.assertNotIn("{synth_report}", q_optimize_zero_shot_direct)
        rendered = q_optimize_zero_shot_direct.format(
            header_code="// hdr",
            current_code="int main() { return 0; }",
        )
        self.assertIn(_ZERO_SHOT_USER_OPENING, rendered)
        self.assertIn("C code", rendered)

    def test_translate_zero_shot_has_expert_user_prefix(self) -> None:
        rendered = q_translate_zero_shot.format(
            header_code="// hdr",
            c_code="int main() { return 0; }",
        )
        self.assertIn(_ZERO_SHOT_USER_OPENING, rendered)
        self.assertNotIn("benchmark_context", rendered)

    def test_flash_optimization_prompt_selector(self) -> None:
        self.assertIs(
            flash_optimization_prompt(zero_shot=True, skip_phase_b=False),
            q_optimize_zero_shot_phaseb,
        )
        self.assertIs(
            flash_optimization_prompt(zero_shot=True, skip_phase_b=True),
            q_optimize_zero_shot_direct,
        )

    def test_non_zero_shot_still_uses_full_expert_instructions(self) -> None:
        self.assertIn("expert", llm_system_instruction(zero_shot=False, translate=True).lower())
        self.assertIn("expert", llm_system_instruction(zero_shot=False, step_name="flash").lower())
        self.assertNotEqual(Instruction_c2hls, "")
        self.assertNotEqual(Instruction_c2hls_flash, "")
        self.assertNotEqual(Instruction_c2hls, Instruction_c2hls_zero_shot)

    def test_llm_messages_include_zero_shot_system(self) -> None:
        self.assertEqual(
            llm_messages(system="", user="hi"),
            [{"role": "user", "content": "hi"}],
        )
        msgs = llm_messages(system=Instruction_c2hls_zero_shot, user="hi")
        self.assertEqual(len(msgs), 2)
        self.assertEqual(msgs[0]["role"], "system")
        self.assertEqual(msgs[0]["content"], Instruction_c2hls_zero_shot)


if __name__ == "__main__":
    unittest.main()
