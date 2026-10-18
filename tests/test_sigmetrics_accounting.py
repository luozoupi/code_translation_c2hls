from copy import deepcopy
import pytest
from experiment_telemetry import capture_tool_calls, record_tool, tool_scope, tool_summary, unified_candidate_ledger
from skill_library import Skill, render_skill_for_prompt, same_content_recipe_atoms


def test_failed_tools_and_cosim_resynthesis_are_counted():
    @record_tool('csynth')
    def synth(code):
        return {'success': True}
    @record_tool('rtl_cosim')
    def rtl(code):
        raise TimeoutError('timed out')
    with capture_tool_calls() as events:
        with tool_scope('search'):
            synth('void kernel() {}')
        with pytest.raises(TimeoutError):
            rtl('void kernel() {}')
    result = tool_summary(events)
    assert result['total_flow_invocations'] == 2
    assert result['vitis_synthesis_flow_invocations'] == 2
    assert events[1]['exception_type'] == 'TimeoutError'
    assert events[0]['scope'] == 'search'
    assert all('code' not in event and 'log' not in event for event in events)


def fixture():
    return {'selected_code_sha256': 'parent', 'frozen_phase_b': {'code_sha256': 'parent'},
            'llm_usage': {'calls': 1, 'candidate_requests': 1, 'total_tokens': 12, 'usage_missing_calls': 0,
                          'events': [{'candidate_evaluation_index': 0, 'total_tokens': 12}]},
            'synthesis_evaluations': {'count': 2, 'events': [
                {'candidate_evaluation_index': 0, 'code_sha256': 'llm', 'synthesis_ran': True}],
                'qor_design_events': [{'qor_evaluation_index': 0, 'code_sha256': 'dse', 'synthesis_ran': True}]}}


def test_mixed_origins_reconcile_without_fabricated_llm_links():
    result = unified_candidate_ledger(fixture())
    assert result['complete'], result['issues']
    assert result['origin_counts'] == {'llm': 1, 'deterministic_search': 1, 'frozen_parent': 1}
    assert result['candidates'][0]['llm_event_indices'] == [0]
    assert result['candidates'][1]['llm_event_indices'] is None
    assert result['candidates'][2]['selected_source_match']
    assert result['candidates'][2]['cost_scope'] == 'previous_parent_construction'


def test_missing_usage_and_wrong_synthesis_totals_fail_closed():
    result = fixture()
    result['llm_usage']['candidate_requests'] = 2
    result['synthesis_evaluations']['count'] = 3
    ledger = unified_candidate_ledger(result)
    assert not ledger['complete']
    assert 'provider_attempts_missing_usage_events' in ledger['issues']
    assert 'synthesis_count_disagrees_with_events' in ledger['issues']


def test_same_content_preserves_every_guard_step_and_template_line():
    skill = Skill(id='example', pattern='Distinct pattern', strategy='Distinct strategy',
                  guards=[f'Guard {i}: do not reorder recurrence' for i in range(12)],
                  required_steps=[f'Step {i}: transform loop' for i in range(14)],
                  template='\n'.join(f'line_{i}();' for i in range(30)))
    outputs = [render_skill_for_prompt(skill, prompt_mode=mode) for mode in ['same_content_flat','same_content_structured']]
    assert outputs[0] != outputs[1]
    for _, atom in same_content_recipe_atoms(skill):
        assert all(atom in text for text in outputs)
    assert all('...' not in text for text in outputs)
