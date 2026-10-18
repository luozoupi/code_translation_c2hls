"""Provider extraction keeps inherited APIs and pipeline patches effective."""
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import c2hls as core
from hlsmith.backends.client import LLMBackendMixin


@pytest.mark.parametrize('first_import', ['hlsmith.backends', 'hlsmith.backends.client', 'hlsmith.pipeline'])
def test_backend_import_orders_preserve_inherited_api(first_import):
    code = f'''
import importlib
importlib.import_module({first_import!r})
import c2hls
from hlsmith import pipeline
from hlsmith.backends.client import LLMBackendMixin
assert c2hls is pipeline
assert issubclass(pipeline.C2HLSOrchestrator, LLMBackendMixin)
for method in ('_call_llm', '_usage_value', '_configured_decoding', '_record_llm_usage',
               '_llm_usage_summary', '_client_for_model', '_call_llm_with_model'):
    assert getattr(pipeline.C2HLSOrchestrator, method) is getattr(LLMBackendMixin, method)
'''
    environment = os.environ.copy()
    environment['PYTHONDONTWRITEBYTECODE'] = '1'
    subprocess.run([sys.executable, '-c', code], cwd=Path(__file__).resolve().parents[1],
                   env=environment, check=True, timeout=15, capture_output=True, text=True)


def test_mixed_provider_clients_still_use_patched_core_factories(monkeypatch):
    default_client, hosted_client = Mock(), Mock()
    factory = Mock(side_effect=[default_client, hosted_client])
    monkeypatch.setattr(core, 'OpenAI', factory)
    monkeypatch.setattr(core, '_load_openai_api_key', lambda: 'hosted-test-key')
    monkeypatch.setenv('DEEPSEEK_API_KEY', 'deepseek-test-key')
    monkeypatch.setenv('DEEPSEEK_BASE_URL', 'https://api.deepseek.com')
    orch = core.C2HLSOrchestrator(gpt_model='deepseek-flash')
    assert orch._client_for_model('deepseek-flash') == ('deepseek', default_client)
    assert orch._client_for_model('gpt-5.6-luna') == ('openai', hosted_client)
    assert orch._client_for_model('gpt-5.6-luna') == ('openai', hosted_client)
    assert factory.call_count == 2
    assert factory.call_args.kwargs['base_url'] == core.OPENAI_HOSTED_BASE_URL
    assert factory.call_args.kwargs['api_key'] == 'hosted-test-key'


def test_anthropic_client_initialization_uses_core_patch(monkeypatch):
    factory = Mock(return_value=object())
    monkeypatch.setattr(core, 'HAS_ANTHROPIC', True)
    monkeypatch.setattr(core, 'anthropic', SimpleNamespace(Anthropic=factory), raising=False)
    monkeypatch.setattr(core, '_load_anthropic_api_key', lambda: 'anthropic-test-key')
    orch = core.C2HLSOrchestrator(gpt_model='claude-sonnet-4-6')
    assert orch._client_for_model(orch.gpt_model) == ('anthropic', factory.return_value)
    assert factory.call_args.kwargs['api_key'] == 'anthropic-test-key'


def test_budgeted_request_hook_and_candidate_accounting_still_share_core(monkeypatch):
    response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='reply'))],
                               usage=SimpleNamespace(prompt_tokens=10, completion_tokens=3, total_tokens=13))
    request = Mock(return_value=response)
    client = Mock()
    monkeypatch.setattr(core, 'OpenAI', Mock(return_value=client))
    monkeypatch.setattr(core, 'send_budgeted_request', request)
    monkeypatch.setenv('DEEPSEEK_API_KEY', 'deepseek-test-key')
    monkeypatch.setenv('C2HLS_DEEPSEEK_THINKING', 'disabled')
    orch = core.C2HLSOrchestrator(gpt_model='deepseek-flash')
    assert orch._call_llm([{'role': 'user', 'content': 'test'}]) == 'reply'
    assert request.call_count == 1
    assert request.call_args.kwargs['kwargs']['extra_body']['thinking'] == {'type': 'disabled'}
    assert not client.chat.completions.create.called
    summary = orch._llm_usage_summary()
    assert summary['calls'] == summary['candidate_requests'] == 1
    assert summary['total_tokens'] == 13
    assert len(orch.synthesis_eval_events) == 1
    assert summary['events'][0]['candidate_evaluation_index'] == 0


def test_candidate_ledger_and_profile_activation_remain_on_controller():
    assert '_start_candidate_evaluation_event' in core.C2HLSOrchestrator.__dict__
    assert '_finalize_candidate_evaluation' in core.C2HLSOrchestrator.__dict__
    assert '__init__' in core.C2HLSOrchestrator.__dict__
    assert '_call_llm_with_model' not in core.C2HLSOrchestrator.__dict__
