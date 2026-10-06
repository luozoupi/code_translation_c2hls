#!/usr/bin/env python3
"""Complete ```cpp fences only; truncated replies must continue, not be accepted."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from c2hls import (
    extract_cpp_code,
    cpp_fence_is_truncated,
    stitch_cpp_continuation,
    looks_like_complete_hls_kernel,
    _cpp_continuation_limit,
    _flash_max_completion_tokens,
    _CLOSED_CPP_RETRY_TRIES,
)
from prompt_c2hls import (
    CLOSED_CPP_OUTPUT_CONTRACT,
    Instruction_c2hls_multistep,
    q_optimize_flash,
    q_optimize_tiling,
    q_optimize_pipeline,
    q_optimize_unroll,
    q_optimize_doublebuffer,
    q_optimize_coalescing,
    hls_synthesis_fix,
    hls_correctness_repair_fix,
    c_compilation_fix,
)


class ExtractCppCodeCompleteOnlyTest(unittest.TestCase):
    def test_closed_cpp_fence(self) -> None:
        reply = "Here is the kernel:\n```cpp\nint foo() { return 1; }\n```\nDone."
        self.assertEqual(extract_cpp_code(reply), "int foo() { return 1; }")

    def test_unclosed_cpp_fence_rejected(self) -> None:
        reply = (
            "Optimized version:\n"
            "```cpp\n"
            '#include "kernel_kernel.h"\n'
            "void kernel0() {\n"
            "  hls::stream<int> fifo_A;\n"
        )
        self.assertTrue(cpp_fence_is_truncated(reply))
        self.assertIsNone(extract_cpp_code(reply))

    def test_empty_or_prose_only_returns_none(self) -> None:
        self.assertIsNone(extract_cpp_code(""))
        self.assertIsNone(extract_cpp_code("I cannot optimize this kernel."))
        self.assertIsNone(extract_cpp_code("```cpp\n```"))
        self.assertFalse(cpp_fence_is_truncated("I cannot optimize this kernel."))

    def test_stitch_continuation_closes_fence(self) -> None:
        part1 = "```cpp\nint foo() {\n  int x = 1;\n"
        part2 = "  return x;\n}\n```"
        stitched = stitch_cpp_continuation(part1, part2)
        self.assertEqual(extract_cpp_code(stitched), "int foo() {\n  int x = 1;\n  return x;\n}")
        self.assertFalse(cpp_fence_is_truncated(stitched))

    def test_stitch_strips_reopened_fence(self) -> None:
        part1 = "```cpp\nint foo() {\n"
        part2 = "```cpp\n  return 1;\n}\n```"
        stitched = stitch_cpp_continuation(part1, part2)
        self.assertEqual(extract_cpp_code(stitched), "int foo() {\n  return 1;\n}")

    def test_flash_prompt_requires_complete_closed_kernel(self) -> None:
        lower = q_optimize_flash.lower()
        self.assertIn("closing", lower)
        self.assertIn("complete", lower)
        self.assertIn("must", lower)
        self.assertIn("continu", lower)


    def test_unlabeled_cot_sketch_is_not_a_kernel(self) -> None:
        reply = (
            "Plan:\n"
            "```\n"
            "for i\n"
            "  for j\n"
            "    ...\n"
            "    C_local[i][j] = acc;\n"
            "```\n"
            "Now I will write the real kernel.\n"
        )
        self.assertFalse(looks_like_complete_hls_kernel("for i\n  for j\n    C_local[i][j] = acc;"))
        self.assertIsNone(extract_cpp_code(reply))
        self.assertFalse(cpp_fence_is_truncated(reply))

    def test_labeled_kernel_preferred_over_unlabeled_sketch(self) -> None:
        reply = (
            "Sketch:\n"
            "```\n"
            "for i\n"
            "  for j\n"
            "```\n"
            "Kernel:\n"
            "```cpp\n"
            "void kernel0(int A[4], int B[4], int C[4]) {\n"
            "  C[0] = A[0] * B[0];\n"
            "}\n"
            "```\n"
        )
        code = extract_cpp_code(reply)
        self.assertIsNotNone(code)
        self.assertIn("void kernel0", code)

    def test_flash_token_floor_matches_dse_budget(self) -> None:
        import os

        prev_flash = os.environ.get("C2HLS_FLASH_MAX_TOKENS")
        prev_llm = os.environ.get("C2HLS_LLM_MAX_TOKENS")
        os.environ.pop("C2HLS_FLASH_MAX_TOKENS", None)
        os.environ.pop("C2HLS_LLM_MAX_TOKENS", None)
        try:
            self.assertGreaterEqual(_flash_max_completion_tokens(8192), 65536)
        finally:
            if prev_flash is None:
                os.environ.pop("C2HLS_FLASH_MAX_TOKENS", None)
            else:
                os.environ["C2HLS_FLASH_MAX_TOKENS"] = prev_flash
            if prev_llm is None:
                os.environ.pop("C2HLS_LLM_MAX_TOKENS", None)
            else:
                os.environ["C2HLS_LLM_MAX_TOKENS"] = prev_llm

    def test_explicit_16384_is_not_raised_to_floor(self) -> None:
        import os

        prev_flash = os.environ.get("C2HLS_FLASH_MAX_TOKENS")
        prev_llm = os.environ.get("C2HLS_LLM_MAX_TOKENS")
        os.environ["C2HLS_FLASH_MAX_TOKENS"] = "16384"
        os.environ.pop("C2HLS_LLM_MAX_TOKENS", None)
        try:
            self.assertEqual(_flash_max_completion_tokens(8192), 16384)
            self.assertEqual(_flash_max_completion_tokens(65536), 16384)
        finally:
            if prev_flash is None:
                os.environ.pop("C2HLS_FLASH_MAX_TOKENS", None)
            else:
                os.environ["C2HLS_FLASH_MAX_TOKENS"] = prev_flash
            if prev_llm is None:
                os.environ.pop("C2HLS_LLM_MAX_TOKENS", None)
            else:
                os.environ["C2HLS_LLM_MAX_TOKENS"] = prev_llm

    def test_cpp_continuations_zero_is_one_shot(self) -> None:
        import os

        prev = os.environ.get("C2HLS_CPP_CONTINUATIONS")
        os.environ["C2HLS_CPP_CONTINUATIONS"] = "0"
        try:
            self.assertEqual(_cpp_continuation_limit(4000, 16384), 0)
        finally:
            if prev is None:
                os.environ.pop("C2HLS_CPP_CONTINUATIONS", None)
            else:
                os.environ["C2HLS_CPP_CONTINUATIONS"] = prev


class CompleteCppViaContinuationTest(unittest.TestCase):
    def test_continuation_loop_assembles_full_kernel(self) -> None:
        from c2hls import C2HLSOrchestrator

        orch = MagicMock(spec=C2HLSOrchestrator)
        orch.max_completion_tokens = 8192
        orch.history = []
        orch._append_history = MagicMock()

        replies = [
            "```cpp\nint foo() {\n  int x = 1;\n",
            "  return x;\n}\n```",
        ]

        def _call(messages, max_tokens=None):
            return replies.pop(0)

        orch._call_llm = _call

        # Bind the real method
        method = C2HLSOrchestrator._call_llm_for_complete_cpp.__get__(orch, C2HLSOrchestrator)
        messages = [{"role": "user", "content": "optimize"}]
        code = method(messages, max_tokens=1024, max_continuations=4)
        self.assertEqual(code, "int foo() {\n  int x = 1;\n  return x;\n}")
        self.assertEqual(replies, [])

    def test_continuation_when_reply_is_cot_without_kernel(self) -> None:
        from c2hls import C2HLSOrchestrator

        orch = MagicMock(spec=C2HLSOrchestrator)
        orch.max_completion_tokens = 8192
        orch.history = []
        orch._append_history = MagicMock()

        replies = [
            "I will pipeline the inner loop.\n```\nfor i\n  for j\n```\n",
            "```cpp\nvoid kernel0(int A[2]) {\n  A[0] = 1;\n}\n```",
        ]

        def _call(messages, max_tokens=None):
            return replies.pop(0)

        orch._call_llm = _call
        method = C2HLSOrchestrator._call_llm_for_complete_cpp.__get__(orch, C2HLSOrchestrator)
        code = method([{"role": "user", "content": "optimize"}], max_tokens=1024, max_continuations=4)
        self.assertIn("void kernel0", code)
        self.assertEqual(replies, [])


class ClosedCppRetryAndPromptContractTest(unittest.TestCase):
    def test_retry_accepts_closed_kernel_on_later_try(self) -> None:
        from c2hls import C2HLSOrchestrator

        orch = MagicMock(spec=C2HLSOrchestrator)
        orch.max_completion_tokens = 8192
        orch.history = []
        orch._append_history = MagicMock()
        replies = [None, "int foo() { return 1; }"]

        def _complete(messages, max_tokens=None, max_continuations=None, baseline_chars=0):
            return replies.pop(0)

        orch._call_llm_for_complete_cpp = _complete
        method = C2HLSOrchestrator._call_llm_until_closed_cpp.__get__(
            orch, C2HLSOrchestrator
        )
        messages = [{"role": "user", "content": "repair"}]
        code = method(messages, max_tokens=1024, max_tries=3)
        self.assertEqual(code, "int foo() { return 1; }")
        self.assertEqual(replies, [])
        self.assertEqual(orch._append_history.call_count, 1)

    def test_three_empty_replies_return_none(self) -> None:
        from c2hls import C2HLSOrchestrator

        orch = MagicMock(spec=C2HLSOrchestrator)
        orch.max_completion_tokens = 8192
        orch.history = []
        orch._append_history = MagicMock()
        calls = {"n": 0}

        def _complete(messages, max_tokens=None, max_continuations=None, baseline_chars=0):
            calls["n"] += 1
            return None

        orch._call_llm_for_complete_cpp = _complete
        method = C2HLSOrchestrator._call_llm_until_closed_cpp.__get__(
            orch, C2HLSOrchestrator
        )
        code = method([{"role": "user", "content": "repair"}], max_tokens=1024)
        self.assertIsNone(code)
        self.assertEqual(calls["n"], _CLOSED_CPP_RETRY_TRIES)

    def test_pipelined_repair_uses_closed_cpp_helper(self) -> None:
        src = (REPO / "c2hls.py").read_text(encoding="utf-8")
        start = src.index("def _pipelined_opt_step_repair_codegen")
        end = src.index("\n    def ", start + 1)
        region = src[start:end]
        self.assertIn("_request_closed_cpp_repair", region)
        self.assertNotIn("extract_cpp_code(reply)", region)
        self.assertNotIn("self._call_llm(self.messages)", region)

    def test_step_and_repair_prompts_require_closed_cpp_no_analysis(self) -> None:
        self.assertIn("OUTPUT CONTRACT", CLOSED_CPP_OUTPUT_CONTRACT)
        self.assertIn("Do not explain", CLOSED_CPP_OUTPUT_CONTRACT)
        for text in (
            Instruction_c2hls_multistep,
            q_optimize_tiling,
            q_optimize_pipeline,
            q_optimize_unroll,
            q_optimize_doublebuffer,
            q_optimize_coalescing,
            hls_synthesis_fix,
            hls_correctness_repair_fix,
            c_compilation_fix,
        ):
            self.assertIn("OUTPUT CONTRACT", text)
            self.assertIn("opened AND closed", text)
            self.assertNotIn("ONE sentence", text)


if __name__ == "__main__":
    unittest.main()
