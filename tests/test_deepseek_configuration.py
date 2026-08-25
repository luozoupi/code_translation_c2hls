import os
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import c2hls


class DeepSeekConfigurationTests(unittest.TestCase):
    def test_v4_model_detection(self):
        self.assertTrue(c2hls._is_deepseek_model("deepseek-v4-flash"))
        self.assertTrue(c2hls._is_deepseek_model("deepseek-v4-pro"))
        self.assertFalse(c2hls._is_deepseek_model("grok-4.5"))

    def test_pro_nonthinking_configuration_is_sent_and_recorded(self):
        usage = SimpleNamespace(
            prompt_tokens=10,
            completion_tokens=3,
            total_tokens=13,
            prompt_tokens_details=SimpleNamespace(cached_tokens=0),
            completion_tokens_details=SimpleNamespace(reasoning_tokens=0),
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
                    "OPENAI_API_KEY": "test-key",
                    "OPENAI_BASE_URL": "https://api.deepseek.com",
                    c2hls.DEEPSEEK_THINKING_ENV: "disabled",
                    c2hls.DEEPSEEK_REASONING_EFFORT_ENV: "high",
                    c2hls.LLM_TEMPERATURE_ENV: "0",
                    c2hls.LLM_TOP_P_ENV: "1",
                    c2hls.LLM_SEED_ENV: "42",
                },
            ),
        ):
            orchestrator = c2hls.C2HLSOrchestrator(
                gpt_model="deepseek-v4-pro",
                max_completion_tokens=16384,
            )
            result = orchestrator._call_llm_with_model(
                [{"role": "user", "content": "Reply OK"}],
                agent_name="translator",
            )

        self.assertEqual(result, "OK")
        kwargs = client.chat.completions.create.call_args.kwargs
        self.assertEqual(kwargs["model"], "deepseek-v4-pro")
        self.assertEqual(kwargs["max_tokens"], 16384)
        self.assertEqual(
            kwargs["extra_body"],
            {
                "thinking": {"type": "disabled"},
                "reasoning_effort": "high",
            },
        )
        self.assertEqual(kwargs["temperature"], 0.0)
        self.assertNotIn("top_p", kwargs)
        self.assertNotIn("seed", kwargs)

        event = orchestrator.llm_usage_events[-1]
        self.assertEqual(event["provider"], "deepseek")
        self.assertEqual(event["model"], "deepseek-v4-pro")
        self.assertEqual(event["decoding"]["thinking"], "disabled")
        self.assertEqual(event["decoding"]["reasoning_effort"], "high")
        self.assertEqual(event["decoding"]["mutually_exclusive_omission"], "top_p")


if __name__ == "__main__":
    unittest.main()
