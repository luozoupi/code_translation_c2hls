"""Agent module boundaries preserve shared helpers and active prompt access."""
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from typing import get_type_hints
from unittest.mock import Mock

import pytest

import c2hls as core
from hlsmith.agents import FeedbackAgent, QualityRepairAgent, SynthesisAgent, TranslatorAgent
from hlsmith.agents.base import _AgentBase


@pytest.mark.parametrize('first_import', [
    'c2hls', 'hlsmith.pipeline', 'hlsmith.agents', 'hlsmith.agents.base',
    'hlsmith.agents.translator', 'hlsmith.agents.synthesis',
    'hlsmith.agents.quality', 'hlsmith.agents.feedback',
])
def test_fresh_import_orders_share_the_same_classes(first_import):
    code = f'''
import importlib
importlib.import_module({first_import!r})
import c2hls
from hlsmith import pipeline
from hlsmith import agents
assert c2hls is pipeline
for name in agents.__all__:
    assert getattr(agents, name) is getattr(pipeline, name)
'''
    environment = os.environ.copy()
    environment['PYTHONDONTWRITEBYTECODE'] = '1'
    subprocess.run([sys.executable, '-c', code], cwd=Path(__file__).resolve().parents[1],
                   env=environment, check=True, timeout=15, capture_output=True, text=True)


def test_translator_uses_patched_pipeline_compiler_and_current_prompt(monkeypatch):
    compiler, history = Mock(return_value=(True, '')), Mock()
    monkeypatch.setattr(core, 'compile_check_cpp', compiler)
    monkeypatch.setattr(core, 'Instruction_c2hls', 'current versioned system prompt')
    orch = SimpleNamespace(gpt_model='test-model', extra_files=[], _append_history=history)
    translator = TranslatorAgent(orch)
    assert translator.run_phase_a('void workload() {}')
    compiler.assert_called_once_with('void workload() {}', '', 'kernel.h', extra_files=[])
    assert history.call_args_list[0].args == ('system', 'current versioned system prompt')


def test_all_agents_resolve_shared_helpers_through_the_pipeline(monkeypatch):
    repair = Mock(return_value='patched guidance')
    monkeypatch.setattr(core, '_build_repair_guidance', repair)
    orch = SimpleNamespace(gpt_model='test-model')
    assert SynthesisAgent(orch)._compose_repair_guidance('compiler error') == 'patched guidance'
    assert FeedbackAgent(orch).render_for_compile_error('compiler error') == 'patched guidance'
    monkeypatch.setattr(core, '_qor_design_sweep_enabled', lambda: True)
    assert QualityRepairAgent.design_sweep_enabled() is True
    monkeypatch.setattr(core, 'extract_cpp_code', lambda reply: 'patched code')
    orch.messages, orch._append_history = [], Mock()
    orch._call_llm_with_model = Mock(return_value='model reply')
    assert _AgentBase(orch)._request_code_revision('repair') == 'patched code'


def test_orchestrator_type_annotation_still_resolves_after_extraction():
    assert get_type_hints(_AgentBase.__init__)['orch'] is core.C2HLSOrchestrator
