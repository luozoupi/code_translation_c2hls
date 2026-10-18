"""A prompt profile must agree with the requests attributed to its run."""
from unittest.mock import MagicMock
import pytest
import c2hls


def test_mixed_profile_instances_fail_before_any_provider_request(monkeypatch):
    client = MagicMock()
    monkeypatch.setattr(c2hls, 'OpenAI', lambda **kwargs: client)
    monkeypatch.setenv('OPENAI_API_KEY', 'test-key')
    first = c2hls.C2HLSOrchestrator(gpt_model='deepseek-flash', prompt_profile='paper-sigmetrics-2026')
    try:
        c2hls.C2HLSOrchestrator(gpt_model='deepseek-flash', prompt_profile='researcher-v1')
        with pytest.raises(ValueError, match='separate process'):
            first._call_llm_with_model([{'role': 'user', 'content': 'No request should be sent'}])
        client.chat.completions.create.assert_not_called()
    finally:
        c2hls._activate_prompt_profile('paper-sigmetrics-2026')
