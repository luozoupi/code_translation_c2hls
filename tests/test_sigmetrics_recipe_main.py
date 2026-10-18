from __future__ import annotations
import copy
import importlib.util
import shutil
import textwrap
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('recipe_main', ROOT / 'scripts/run_sigmetrics_recipe_main.py')
main = importlib.util.module_from_spec(spec)
spec.loader.exec_module(main)


def test_registration_is_full_factorial_and_arm_positions_are_balanced():
    benchmarks = [f'kernel{i}' for i in range(12)]
    cells = main.make_cells(benchmarks)
    assert len(cells) == 216
    assert len({c['cell_id'] for c in cells}) == 216
    assert {(c['benchmark'], c['model'], c['arm'], c['repeat']) for c in cells} == {
        (b, m, a, r) for b in benchmarks for m in main.MODELS for a in main.ARMS for r in range(1, 4)}
    for benchmark in benchmarks:
        first_arms = [next(c['arm'] for c in cells if c['benchmark'] == benchmark and c['repeat'] == r) for r in range(1, 4)]
        assert set(first_arms) == set(main.ARMS)


def valid_parent():
    stage = {'passed': True, 'validation_suite': {'executed_cases': 5, 'all_cases_passed': True}}
    return {'origin': 'unchanged_public_c', 'model_calls': 0, 'success': True,
            'csim': copy.deepcopy(stage), 'cosim': copy.deepcopy(stage),
            'tool_flows': {'counts': {'csim': 5, 'csynth': 1, 'rtl_cosim': 5}}}


@pytest.mark.parametrize('stage', ['csim', 'cosim'])
def test_parent_needs_both_five_case_suites(stage):
    parent = valid_parent()
    assert main.parent_gate(parent) == []
    parent[stage]['validation_suite']['executed_cases'] = 4
    assert stage + '_missing_five_passes' in main.parent_gate(parent)


def test_parent_rejects_model_construction_and_hidden_tool_flows():
    parent = valid_parent()
    parent['model_calls'] = 1
    parent['tool_flows']['counts']['csynth'] = 2
    assert set(main.parent_gate(parent)) == {'parent_not_unchanged_public_c', 'parent_tool_counts_mismatch'}


def accounting_fixture():
    event = dict(input_tokens=10, output_tokens=5, cached_tokens=0, prompt_sha256='prompt')
    result = {'llm_usage': {'events': [event.copy() for _ in range(5)]},
              'synthesis_evaluations': {'count': 5}, 'candidate_ledger': {'complete': True},
              'reference_isolation_audit': {'passed': True}, 'selected_code_sha256': 'code', 'cosim_target_code_sha256': 'code',
              'rtl_executed_pass': True, 'cosim': {'passed': True, 'validation_suite': {'executed_cases': 5, 'all_cases_passed': True}},
              'tool_flows': {'counts': {'csynth': 5, 'rtl_cosim': 5}, 'total_flow_invocations': 36,
                             'events': [{'kind': 'rtl_cosim', 'code_sha256': 'code'} for _ in range(5)]}}
    rows = [dict(event, status='settled', charged_or_reserved_usd=.001) for _ in range(5)]
    return result, rows


def test_five_calls_reconcile_and_failed_rtl_is_a_valid_outcome():
    result, rows = accounting_fixture()
    assert main.reconcile(result, rows, 'prompt')['complete']
    result['rtl_executed_pass'] = False
    result['cosim']['passed'] = False
    result['cosim']['validation_suite'].update(executed_cases=2, all_cases_passed=False)
    result['tool_flows']['counts']['rtl_cosim'] = 2
    result['tool_flows']['events'] = result['tool_flows']['events'][:2]
    assert main.reconcile(result, rows, 'prompt')['complete']


def test_extra_request_prompt_drift_and_unsettled_costs_fail_closed():
    result, rows = accounting_fixture()
    result['llm_usage']['events'].append(rows[0])
    rows.append(rows[0])
    rows[1] = dict(rows[1], status='reserved')
    issues = main.reconcile(result, rows, 'different')['issues']
    assert {'registered_search_cap_exceeded', 'first_prompt_not_frozen_preview', 'unsettled_or_overrun_provider_request'} <= set(issues)


def test_nested_controller_exceptions_are_engineering_failures():
    result, rows = accounting_fixture()
    result['steps'] = [{'exception_type': 'BudgetExhausted', 'error': 'campaign budget exhausted'}]
    assert not main.reconcile(result, rows, 'prompt')['complete']


def test_local_allowance_checked_before_request_count_changes():
    from sigmetrics_main_policy import check_request_allowance, request_input_bound, SearchAllowanceExhausted
    owner = SimpleNamespace(llm_usage_events=[{'input_tokens': 499990}], llm_candidate_request_count=1)
    with pytest.raises(SearchAllowanceExhausted, match='aggregate'):
        check_request_allowance(owner, [{'role': 'user', 'content': 'x'}])
    assert owner.llm_candidate_request_count == 1
    owner.llm_usage_events = []
    messages = [{'role': 'user', 'content': 'x' * 250000}]
    assert request_input_bound(messages) > 250000
    with pytest.raises(SearchAllowanceExhausted, match='next_request'):
        check_request_allowance(owner, messages)


@pytest.fixture
def amended_method(tmp_path):
    import c2hls
    for filename in ('c2hls.py', 'prompt_c2hls.py'):
        shutil.copy2(ROOT / filename, tmp_path / filename)
    before = main.sha(ROOT / 'c2hls.py')
    main.patch_snapshot(tmp_path)
    assert main.sha(ROOT / 'c2hls.py') == before
    code = (tmp_path / 'c2hls.py').read_text()
    start = code.index('    def _optimization_step_attempt_single(')
    end = code.index('\n    def ', start + 5)
    namespace = dict(vars(c2hls))
    body = code[start:end].split('\n    @staticmethod')[0]
    exec(compile(textwrap.dedent(body), '<frozen-main-method>', 'exec'), namespace)
    return namespace['_optimization_step_attempt_single'], namespace


def mock_orchestrator(namespace, monkeypatch, replies):
    monkeypatch.setenv('C2HLS_EXHAUSTIVE_CANDIDATE_ATTEMPTS', '1')
    monkeypatch.setenv('C2HLS_ATTEMPTS_PER_CANDIDATE', '5')
    monkeypatch.setenv('C2HLS_SKILL_PROMPT_SCOPE', 'skillless')
    monkeypatch.setenv('C2HLS_QOR_PROFILE_REVIEW', '0')
    monkeypatch.setenv('C2HLS_FEASIBILITY_SELECTION', '1')
    from c2hls import C2HLSOrchestrator
    orch = C2HLSOrchestrator.__new__(C2HLSOrchestrator)
    orch.turns_limitation = 5
    orch.synth_report = {'latency_cycles': 1000}
    orch._baseline_report = {}
    orch.hls_code = 'void workload() {}'
    orch.header_code = ''
    orch.header_name = 'kernel.h'
    orch.part = 'xcu280-fsvh2892-2L-e'
    orch.clock_ns = 3.33
    orch.skill_library = None
    orch.skill_library_provenance = {}
    orch.vitis_version = '2023.2'
    orch.reference_blind = True
    orch.extra_files = []
    orch.testbench_code = 'testbench'
    orch.benchmark_context = ''
    orch._call_llm = MagicMock(side_effect=replies)
    orch._append_history = MagicMock()
    orch._pending_candidate_evaluation_index = MagicMock(return_value=None)
    orch._preflight_generated_hls_code = lambda code, label: code
    orch._finalize_candidate_evaluation = MagicMock()
    orch._best_so_far_score = lambda report: report['latency_cycles']
    orch.synthesis = SimpleNamespace(_compose_repair_guidance=lambda *args, **kwargs: '')
    namespace['_build_profile_signal'] = lambda *args, **kwargs: {}
    namespace['_lint_in_place_stencil_guardrails'] = lambda *args: []
    namespace['compile_check_cpp'] = lambda *args, **kwargs: (True, '')
    namespace['_paper_candidate_feasibility'] = lambda *args, **kwargs: {'feasible': True}
    namespace['format_report_summary'] = lambda report: str(report)
    namespace['compare_reports'] = lambda *args: {}
    return orch


def passed(cycles):
    return {'synth': {'success': True, 'report': {'latency_cycles': cycles}}, 'csim': {'ran': True, 'passed': True}, 'cosim': None}


def test_fifth_synthesis_failure_keeps_prior_best_without_sixth_call(amended_method, monkeypatch):
    method, namespace = amended_method
    replies = ['```cpp\nvoid workload() {}\n```'] * 5
    orch = mock_orchestrator(namespace, monkeypatch, replies)
    orch._synth_and_test = MagicMock(side_effect=[passed(100), passed(90), passed(70), passed(80), {'synth': {'success': False, 'error': 'synthesis failed'}, 'csim': None, 'cosim': None}])
    result = method(orch, 'flash')
    assert result['success'] is True
    assert result['report']['latency_cycles'] == 70
    assert orch._call_llm.call_count == 5
    assert orch._synth_and_test.call_count == 5


def test_local_token_limit_keeps_prior_best_without_a_new_call(amended_method, monkeypatch):
    from sigmetrics_main_policy import SearchAllowanceExhausted
    method, namespace = amended_method
    orch = mock_orchestrator(namespace, monkeypatch, ['```cpp\nvoid workload() {}\n```', SearchAllowanceExhausted('test limit')])
    orch._synth_and_test = MagicMock(return_value=passed(70))
    result = method(orch, 'flash')
    assert result['success'] is True
    assert result['report']['latency_cycles'] == 70
    assert orch._synth_and_test.call_count == 1


def test_real_billing_failure_is_not_swallowed(amended_method, monkeypatch):
    from experiment_budget import BudgetExhausted
    method, namespace = amended_method
    orch = mock_orchestrator(namespace, monkeypatch, ['```cpp\nvoid workload() {}\n```', BudgetExhausted('unsettled usage')])
    orch._synth_and_test = MagicMock(return_value=passed(70))
    with pytest.raises(BudgetExhausted):
        method(orch, 'flash')
