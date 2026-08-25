import os
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import c2hls


class XAIConfigurationTests(unittest.TestCase):
    def test_grok_model_detection(self):
        self.assertTrue(c2hls._is_xai_model("grok-4.5"))
        self.assertFalse(c2hls._is_xai_model("deepseek-v4-flash"))

    def test_low_reasoning_is_sent_and_recorded(self):
        usage = SimpleNamespace(
            prompt_tokens=10,
            completion_tokens=3,
            total_tokens=15,
            prompt_tokens_details=SimpleNamespace(cached_tokens=0),
            completion_tokens_details=SimpleNamespace(reasoning_tokens=2),
        )
        response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="OK"))],
            usage=usage,
        )
        client = MagicMock()
        client.chat.completions.create.return_value = response

        with (
            patch.object(c2hls, "OpenAI", return_value=client),
            patch.dict(
                os.environ,
                {
                    "OPENAI_API_KEY": "EMPTY",
                    "OPENAI_BASE_URL": "https://api.x.ai/v1",
                    c2hls.XAI_REASONING_EFFORT_ENV: "low",
                    c2hls.LLM_TEMPERATURE_ENV: "0",
                    c2hls.LLM_TOP_P_ENV: "1",
                    c2hls.LLM_SEED_ENV: "42",
                },
            ),
        ):
            orchestrator = c2hls.C2HLSOrchestrator(
                gpt_model="grok-4.5",
                max_completion_tokens=64,
            )
            result = orchestrator._call_llm_with_model(
                [{"role": "user", "content": "Reply OK"}],
                agent_name="translator",
            )

        self.assertEqual(result, "OK")
        kwargs = client.chat.completions.create.call_args.kwargs
        self.assertEqual(kwargs["reasoning_effort"], "low")
        self.assertEqual(kwargs["max_tokens"], 64)
        self.assertEqual(kwargs["seed"], 42)
        event = orchestrator.llm_usage_events[-1]
        self.assertEqual(event["provider"], "xai")
        self.assertEqual(event["reasoning_tokens"], 2)
        self.assertEqual(event["decoding"]["reasoning_effort"], "low")

    def test_invalid_reasoning_effort_fails_before_request(self):
        client = MagicMock()
        with (
            patch.object(c2hls, "OpenAI", return_value=client),
            patch.dict(
                os.environ,
                {
                    "OPENAI_API_KEY": "EMPTY",
                    c2hls.XAI_REASONING_EFFORT_ENV: "off",
                },
            ),
        ):
            orchestrator = c2hls.C2HLSOrchestrator(gpt_model="grok-4.5")
            with self.assertRaisesRegex(ValueError, "low, medium, or high"):
                orchestrator._call_llm_with_model(
                    [{"role": "user", "content": "Reply OK"}]
                )
        client.chat.completions.create.assert_not_called()


if __name__ == "__main__":
    unittest.main()
