import copy
import hashlib
from pathlib import Path

import pytest

import c2hls
from experiment_telemetry import capture_tool_calls, record_tool, tool_summary
from validation_suite import _ACTIVE, load_suite, run_validation_cases


@pytest.fixture
def suite():
    return {'benchmark': 'synthetic', 'manifest_sha256': 'manifest', 'native_validation_sha256': 'native',
            'header_sha256': hashlib.sha256(b'header').hexdigest(),
            'cases': [{'case_id': i, 'testbench_code': str(i), 'testbench_sha256': hashlib.sha256(str(i).encode()).hexdigest(),
                       'golden_output_text': 'golden'+str(i), 'golden_output_specs': {'x': {}},
                       'golden_output_sha256': 'goldenhash'+str(i)} for i in range(5)]}


def invoke(suite, function):
    token = _ACTIVE.set(suite)
    try:
        with capture_tool_calls() as events:
            result = function(hls_code='candidate', testbench_code='0', header_code='header')
        return result, tool_summary(events)
    finally:
        _ACTIVE.reset(token)


def flow(failure=None, missing_cycles=None):
    @run_validation_cases('cosim')
    @record_tool('rtl_cosim')
    def call(hls_code, testbench_code, header_code='', golden_output_text='', golden_output_specs=None, work_dir=None):
        case = int(testbench_code)
        assert golden_output_text == 'golden'+testbench_code
        return {'passed': case != failure, 'success': case != failure,
                'kernel_runtime_cycles': None if case == missing_cycles else [2, 4, 9, 3, 1][case], 'log': 'private log'}
    return call


def test_all_cases_and_worst_latency_count_resynthesis(suite):
    result, accounting = invoke(suite, flow())
    assert result['passed'] and result['kernel_runtime_cycles'] == 9
    assert result['validation_suite']['executed_cases'] == 5
    assert accounting['vitis_synthesis_flow_invocations'] == 5
    assert [e['scope'] for e in accounting['events']] == [f'cosim:case{i}' for i in range(5)]
    summary = c2hls._summarize_test_result(result, True)
    assert summary['validation_suite'] == result['validation_suite']


def test_failure_preserved_and_fixed_stop_rule(suite):
    result, accounting = invoke(suite, flow(failure=2))
    assert not result['passed'] and result['validation_suite']['executed_cases'] == 3
    assert accounting['counts']['rtl_cosim'] == 3


def test_missing_rtl_cycles_fails(suite):
    result, _ = invoke(suite, flow(missing_cycles=0))
    assert not result['passed'] and result['validation_suite']['executed_cases'] == 1


def test_wrong_header_fails_before_tool(suite):
    suite['header_sha256'] = 'changed'
    with pytest.raises(ValueError, match='header'):
        invoke(suite, flow())


def test_public_oracle_mode_never_loads_expert_cache(monkeypatch):
    monkeypatch.setenv('C2HLS_REFERENCE_VALIDATE_MODE', 'public_only')
    monkeypatch.setenv('C2HLS_REFERENCE_BLIND', '1')
    def forbidden(*args):
        raise AssertionError('expert cache accessed')
    monkeypatch.setattr(c2hls, '_load_reference_validation_cache', forbidden)
    inputs = {'independent_golden_provenance': {'required': True, 'status': 'passed'},
              'independent_golden_output': 'golden', 'independent_golden_specs': {'x': {}}}
    result = c2hls.validate_gold_reference(inputs)
    assert result['benchmark_ready'] and result['ground_truth_status'] == 'not_evaluated'
    assert result['synthesis']['status'] == 'not_run' and not result['report']
    inputs['independent_golden_provenance']['status'] = 'failed'
    assert not c2hls.validate_gold_reference(inputs)['benchmark_ready']
    inputs['independent_golden_provenance']['status'] = 'passed'
    monkeypatch.setenv('C2HLS_REFERENCE_BLIND', '0')
    assert not c2hls.validate_gold_reference(inputs)['benchmark_ready']


def test_saved_public_oracle_result_does_not_claim_expert_validation():
    reference={'reference_source':'independent_public_oracle','ground_truth_status':'not_evaluated',
               'benchmark_ready':True,'report':{},'workflow':[],
               'synthesis':{'status':'not_run','ran':False,'success':False}}
    result=c2hls.sanitize_saved_result_record({'success':True,'ground_truth_status':'not_evaluated'},reference)
    assert result['ground_truth_status']=='not_evaluated'
    assert result['reference_validation']['ground_truth_status']=='not_evaluated'
    assert result['ground_truth_report']=={}


def test_legacy_reference_result_keeps_valid_status():
    reference={'reference_source':'local_vitis','benchmark_ready':True,'report':{},'workflow':[],
               'synthesis':{'status':'passed','ran':True,'success':True}}
    result=c2hls.sanitize_saved_result_record({'success':True},reference)
    assert result['ground_truth_status']=='valid'


def test_public_oracle_coverage_keeps_generated_tests_without_expert_tests():
    meta={'testbench_file':'testbench.cpp','supports_csim':True,'supports_cosim':True}
    coverage=c2hls._build_coverage(meta,{'reference_source':'independent_public_oracle'},None,None)
    assert not coverage['ground_truth_csim_available']
    assert not coverage['ground_truth_cosim_available']
    assert coverage['generated_csim_available'] and coverage['generated_cosim_available']
