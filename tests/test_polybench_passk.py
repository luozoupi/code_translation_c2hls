"""Protocol and accounting tests; no model/provider or Vitis calls."""
from copy import deepcopy
import hashlib
import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import run_polybench_passk as runner
from cosim_walltime import TIMING_SCOPE


def config():
    return {'k': 5, 'model': 'deepseek-flash', 'max_model_requests_per_attempt': 5,
            'aggregate_input_token_cap_per_attempt': 100000,
            'max_output_tokens_per_request': 8192, 'max_selection_syntheses_per_attempt': 5}


def valid_result():
    code = 'void kernel() {}'
    digest = hashlib.sha256(code.encode()).hexdigest()
    correctness = {'passed': True}
    return {'success': True, 'rtl_executed_pass': True, 'hls_code': code,
            'selected_code_sha256': digest, 'cosim_target_code_sha256': digest,
            'candidate_feasibility': {'feasible': True},
            'reference_isolation_audit': {'passed': True},
            'csim': {'passed': True, 'success': True, 'correctness': correctness},
            'cosim': {'passed': True, 'success': True, 'ran': True, 'kernel_runtime_cycles': 100,
                      'correctness': correctness, 'target_code_sha256': digest,
                      'classification': 'completed', 'phase_markers_complete': True,
                      'validation_log_scan': {'complete': True},
                      'timing_scope': TIMING_SCOPE, 'timed_out': False}}


def test_reference_timing_requires_same_phase_input_toolchain_and_success():
    reference = {'status': 'pass', 'exact_input_match': True, 'toolchain_match': True,
                 'censored': False, 'timing_scope': TIMING_SCOPE, 'cosim_design_wall_seconds': 90}
    assert runner.phase_limit(reference)['seconds'] == 180
    for key, value in [('status', 'timeout'), ('exact_input_match', False),
                       ('toolchain_match', False), ('censored', True),
                       ('timing_scope', 'whole_worker'), ('cosim_design_wall_seconds', True),
                       ('cosim_design_wall_seconds', float('nan'))]:
        assert runner.phase_limit({**reference, key: value})['seconds'] == 3600
    assert runner.phase_limit({'kernel_runtime_cycles': 1})['seconds'] == 3600


def test_rtl_pass_requires_positive_measured_cycles_and_matching_generated_code():
    assert runner.classify(valid_result()) == 'pass'
    for value in (None, 0, True, float('nan')):
        result = valid_result()
        result['cosim']['kernel_runtime_cycles'] = value
        assert runner.classify(result) == 'evidence_error'
    result = valid_result()
    result['cosim']['target_code_sha256'] = 'different'
    assert runner.classify(result) == 'evidence_error'


def test_functional_or_feasibility_failure_never_counts_as_pass():
    result = valid_result()
    result['candidate_feasibility']['feasible'] = False
    assert runner.classify(result) == 'fail'
    result = valid_result()
    result['csim']['passed'] = False
    assert runner.classify(result) == 'fail'
    result = valid_result()
    result['cosim']['correctness'] = {'passed': False}
    assert runner.classify(result) == 'evidence_error'
    result = valid_result()
    result['cosim'].update(passed=False, success=False, timed_out=True, status='timeout')
    assert runner.classify(result) == 'timeout'


def test_phase_markers_and_guard_completion_required_for_pass():
    for field, value in [('phase_markers_complete', False), ('classification', 'tool_error'),
                         ('timing_scope', 'kernel_cycles'), ('timed_out', True)]:
        result = valid_result()
        result['cosim'][field] = value
        assert runner.classify(result) == 'evidence_error'


def accounted():
    event = {'model': 'deepseek-flash', 'usage_available': True, 'input_tokens': 30,
             'output_tokens': 10, 'cached_tokens': 0, 'prompt_sha256': 'abc', 'max_tokens': 8192}
    reservation = {**event, 'status': 'settled', 'overrun': False}
    result = {'llm_usage': {'calls': 1, 'events': [event]}, 'reference_isolation_audit': {'passed': True},
              'tool_flows': {'counts': {'csynth': 5, 'rtl_cosim': 1}}}
    return result, [reservation]


def test_exact_provider_accounting_accepts_bounded_result():
    result, reservations = accounted()
    assert runner.accounting_issues(result, reservations, config(), requests_started=1) == []
    result['steps'] = [{'exception_type': 'AttemptAllowanceExhausted', 'budget_exhausted': True}]
    assert runner.accounting_issues(result, reservations, config(), requests_started=1) == []


def test_unknown_reservation_or_missing_event_stops_retry():
    result, reservations = accounted()
    reservations[0]['status'] = 'reserved'
    result['llm_usage'] = {'calls': 0, 'events': []}
    issues = runner.accounting_issues(result, reservations, config(), requests_started=1)
    assert 'unknown_or_overrun_usage' in issues
    assert 'request_accounting_mismatch' in issues
    assert reservations[0]['status'] == 'reserved'


def test_cost_and_synthesis_mismatches_are_not_silently_retried():
    result, reservations = accounted()
    reservations[0]['input_tokens'] += 1
    result['tool_flows']['counts']['csynth'] = 6
    result['exception_type'] = 'APIConnectionError'
    issues = runner.accounting_issues(result, reservations, config(), requests_started=1)
    assert set(issues) >= {'token_or_prompt_accounting_mismatch', 'synthesis_budget_exceeded', 'execution_exception'}


class FakeAgent:
    def __init__(self):
        self.llm_usage_events = []
        self.max_completion_tokens = 8192
        self.agents = []

    def _call_llm_with_model(self, messages, model=None, max_tokens=None, agent_name='orchestrator'):
        self.agents.append(agent_name)
        self.llm_usage_events.append({'input_tokens': 10, 'usage_available': True})
        return 'generated'


def test_central_guard_caps_all_agent_roles_not_only_candidate_repairs():
    module = SimpleNamespace(C2HLSOrchestrator=FakeAgent)
    instances = []
    with runner.capped_agent(module, config(), instances):
        agent = module.C2HLSOrchestrator()
        for role in ['translator', 'synthesis', 'planner', 'feedback', 'reviewer']:
            assert agent._call_llm_with_model([{'role': 'user', 'content': 'task'}], agent_name=role) == 'generated'
        with pytest.raises(runner.AttemptAllowanceExhausted, match='maximum_model_requests'):
            agent._call_llm_with_model([], agent_name='reviewer')
        assert agent.passk_requests_started == 5 and len(agent.agents) == 5
    assert module.C2HLSOrchestrator is FakeAgent


def test_input_reservation_guard_stops_before_provider_count():
    module = SimpleNamespace(C2HLSOrchestrator=FakeAgent)
    small = {**config(), 'aggregate_input_token_cap_per_attempt': 10}
    with runner.capped_agent(module, small, []):
        agent = module.C2HLSOrchestrator()
        with pytest.raises(runner.AttemptAllowanceExhausted, match='input_reservation_limit'):
            agent._call_llm_with_model([{'role': 'user', 'content': 'task'}])
        assert agent.passk_requests_started == 0 and agent.agents == []


def test_public_metadata_drops_expert_and_reduced_cosim_fields():
    metadata = {'benchmark': 'hlsfactory_doitgen', 'source_repo': 'HLSFactory',
                'plain_c_file': 'plain.cpp', 'gold_hls_source_file': 'gold.cpp',
                'variants': [{'file': 'expert.cpp'}], 'cosim_size_overrides': {'N': 8},
                'cosim_testbench_file': 'reduced.cpp', 'cosim_support_files': ['gold.cpp']}
    public = runner.public_metadata(metadata)
    assert not set(public) & {'gold_hls_source_file', 'variants', 'cosim_size_overrides',
                              'cosim_testbench_file', 'cosim_support_files'}


def test_attempt_order_keeps_first_success_and_unresolved_error_terminal(tmp_path):
    group = {'id': 'hlsfactory_gemm__flash', 'benchmark': 'hlsfactory_gemm'}
    runner.save(tmp_path / 'manifest.json', {'schema_version': 'test'})
    runner.save(tmp_path / 'input_preflight.json', {'complete': True,
                'manifest_sha256': runner.sha(tmp_path / 'manifest.json'),
                'records': [{'benchmark': group['benchmark'], 'success': True}]})
    runner.require_attempt_ready(tmp_path, config(), group, 1)
    prior = tmp_path / 'attempts' / group['id'] / '1' / 'result.json'
    for status in ('fail', 'timeout'):
        runner.save(prior, {'passk_attempt': {'status': status}})
        runner.require_attempt_ready(tmp_path, config(), group, 2)
    for status in ('pass', 'execution_error', 'evidence_error'):
        runner.save(prior, {'passk_attempt': {'status': status}})
        with pytest.raises(ValueError, match='Previous attempt'):
            runner.require_attempt_ready(tmp_path, config(), group, 2)


def test_blocked_native_input_stays_blocked(tmp_path):
    group = {'id': 'doitgen__flash', 'benchmark': 'hlsfactory_doitgen'}
    runner.save(tmp_path / 'manifest.json', {})
    runner.save(tmp_path / 'input_preflight.json', {'complete': True,
                'manifest_sha256': runner.sha(tmp_path / 'manifest.json'),
                'records': [{'benchmark': group['benchmark'], 'success': False}]})
    with pytest.raises(ValueError, match='blocked'):
        runner.require_attempt_ready(tmp_path, config(), group, 1)


def test_fingerprint_controls_cover_phase_policy_and_implementation():
    import evaluation_repro
    assert 'cosim_walltime.py' in evaluation_repro._IMPLEMENTATION_SOURCES
    assert 'C2HLS_COSIM_WALL_TIMEOUT_SECONDS' in evaluation_repro._CONTROL_ENV_NAMES


def test_real_fingerprint_and_benchmark_api_signatures_match_runner():
    # Bind signatures only: a fingerprint build can probe Vitis and is not called here.
    import evaluation_repro
    import c2hls
    inspect.signature(evaluation_repro.build_run_fingerprint).bind(
        repo=Path('/tmp/frozen'), benchmark_dir=Path('/tmp/public'), benchmark='hlsfactory_gemm',
        model_id='deepseek-flash', model_label='deepseek_flash', skill_mode='smart_exhaustive_v2',
        steps=None, profile={'name': runner.SCHEMA}, environ={},
    )
    inspect.signature(c2hls.run_benchmark_multistep).bind(
        '/tmp/public', '/tmp/output', gpt_model='deepseek-flash', turns_limitation=3, quality_repair_turns=0,
    )
