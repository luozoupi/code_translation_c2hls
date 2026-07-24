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
)
from prompt_c2hls import q_optimize_flash


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


if __name__ == "__main__":
    unittest.main()
