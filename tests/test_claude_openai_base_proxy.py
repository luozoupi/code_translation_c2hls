"""Claude via OPENAI_BASE_URL must use OpenAI client (PC2 login proxy)."""

from __future__ import annotations

import os
import unittest
from unittest import mock


class TestClaudeViaOpenAIBaseProxy(unittest.TestCase):
    def setUp(self) -> None:
        self._prev = {
            k: os.environ.get(k)
            for k in ("OPENAI_BASE_URL", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "C2HLS_MODEL")
        }

    def tearDown(self) -> None:
        for k, v in self._prev.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def test_claude_with_openai_base_uses_openai_client(self) -> None:
        os.environ["OPENAI_BASE_URL"] = "http://login5:18192/v1"
        os.environ["OPENAI_API_KEY"] = "sk-test-proxy-key-xxxxxxxx"
        os.environ.pop("ANTHROPIC_API_KEY", None)

        import c2hls

        with mock.patch.object(c2hls, "OpenAI") as mock_openai:
            mock_openai.return_value = mock.Mock(name="openai_client")
            orch = c2hls.C2HLSOrchestrator.__new__(c2hls.C2HLSOrchestrator)
            # Call only the client-selection portion of __init__ via re-running logic:
            c2hls.C2HLSOrchestrator.__init__(
                orch,
                max_completion_tokens=128,
                gpt_model="claude-sonnet-5",
                turns_limitation=1,
            )
            self.assertFalse(orch.use_anthropic)
            self.assertTrue(orch.use_openai_base_proxy)
            mock_openai.assert_called()
            kind, _client = orch._client_for_model("claude-sonnet-5")
            self.assertEqual(kind, "openai")

    def test_claude_without_openai_base_uses_anthropic_when_available(self) -> None:
        os.environ.pop("OPENAI_BASE_URL", None)
        os.environ["ANTHROPIC_API_KEY"] = "sk-ant-test-key-xxxxxxxxxxxxxxxx"

        import c2hls

        if not c2hls.HAS_ANTHROPIC:
            self.skipTest("anthropic package not installed")

        with mock.patch.object(c2hls.anthropic, "Anthropic") as mock_ant:
            mock_ant.return_value = mock.Mock(name="anthropic_client")
            orch = c2hls.C2HLSOrchestrator(
                max_completion_tokens=128,
                gpt_model="claude-sonnet-5",
                turns_limitation=1,
            )
            self.assertTrue(orch.use_anthropic)
            kind, _client = orch._client_for_model("claude-sonnet-5")
            self.assertEqual(kind, "anthropic")


if __name__ == "__main__":
    unittest.main()
