import os
import unittest
from unittest.mock import patch

import c2hls


class CompletionTokenConfigTests(unittest.TestCase):
    def test_default_limit(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop(c2hls.MAX_COMPLETION_TOKENS_ENV, None)
            self.assertEqual(c2hls._completion_token_limit(), 8192)

    def test_environment_limit(self):
        with patch.dict(
            os.environ,
            {c2hls.MAX_COMPLETION_TOKENS_ENV: "16384"},
        ):
            self.assertEqual(c2hls._completion_token_limit(), 16384)

    def test_explicit_limit_takes_precedence(self):
        with patch.dict(
            os.environ,
            {c2hls.MAX_COMPLETION_TOKENS_ENV: "16384"},
        ):
            self.assertEqual(c2hls._completion_token_limit(4096), 4096)

    def test_orchestrator_reads_environment_limit(self):
        with (
            patch.dict(
                os.environ,
                {
                    c2hls.MAX_COMPLETION_TOKENS_ENV: "16384",
                    "OPENAI_API_KEY": "EMPTY",
                },
            ),
            patch.object(c2hls, "OpenAI"),
        ):
            orchestrator = c2hls.C2HLSOrchestrator(
                gpt_model="deepseek-v4-flash"
            )
        self.assertEqual(orchestrator.max_completion_tokens, 16384)

    def test_invalid_limit_fails_fast(self):
        for value in ("0", "-1", "invalid"):
            with self.subTest(value=value), patch.dict(
                os.environ,
                {c2hls.MAX_COMPLETION_TOKENS_ENV: value},
            ):
                with self.assertRaisesRegex(ValueError, "positive integer"):
                    c2hls._completion_token_limit()


if __name__ == "__main__":
    unittest.main()
