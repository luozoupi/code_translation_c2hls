"""Pipeline checks with saved responses, real native checks and synthetic QoR.

These tests make no model requests and never execute Vitis or a GPU job.
"""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from hlsmith_minimal import core


PRIVATE_OUTPUT = 'PRIVATE_ORACLE_OUTPUT_907351'
PRIVATE_DIAGNOSTIC = 'PRIVATE_TESTBENCH_DIAGNOSTIC_681239'


@pytest.fixture
def project(tmp_path):
    root = tmp_path / 'project'
    root.mkdir()
    header = '''#ifndef CORE_TEST_KERNEL_H
#define CORE_TEST_KERNEL_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
uint32_t kernel(uint32_t x);
#ifdef __cplusplus
}
#endif
#endif
'''
    correct = '#include "kernel.h"\nuint32_t kernel(uint32_t x) { return x + 1u; }\n'
    other_correct = '#include "kernel.h"\nuint32_t kernel(uint32_t x) { return (uint32_t)(1u + x); }\n'
    wrong = '#include "kernel.h"\nuint32_t kernel(uint32_t x) { return x + 2u; }\n'
    (root / 'kernel.h').write_text(header)
    (root / 'original.c').write_text(correct)
    (root / 'private_vectors.h').write_text(
        '#define PRIVATE_X 7u\n#define PRIVATE_LABEL "' + PRIVATE_OUTPUT + '"\n')
    (root / 'tb.c').write_text(
        '#include <stdio.h>\n#include "kernel.h"\n#include "private_vectors.h"\n'
        'int main(void) { printf("%s %u\\n", PRIVATE_LABEL, kernel(PRIVATE_X)); return 0; }\n')
    config = {'source': 'original.c', 'testbench': 'tb.c',
              'headers': {'kernel.h': 'kernel.h'},
              'test_files': {'private_vectors.h': 'private_vectors.h'},
              'top': 'kernel', 'validation': {'atol': 0, 'rtol': 0}}
    path = root / 'kernel.json'
    path.write_text(json.dumps(config))
    return SimpleNamespace(root=root, path=path, config=config, correct=correct,
                           other_correct=other_correct, wrong=wrong)


def responses(project, sources):
    path = project.root / 'responses.json'
    path.write_text(json.dumps({'origin': 'synthetic offline test, not provider output',
                                'responses': sources}))
    return path


def captured_messages(out):
    return [json.loads(path.read_text())['messages']
            for path in sorted((out / 'model').glob('*/request.json'))]


def synthetic_synthesis(project, monkeypatch, measurements):
    """Only compiler QoR is mocked; candidates still run the real native oracle."""
    project.config['vitis'] = {'executable': 'core-test-vitis', 'part': 'test-part',
                               'clock_ns': 3.33, 'resource_limits': {'dsp': 8, 'lut': 100}}
    project.path.write_text(json.dumps(project.config))
    actual_which = core.shutil.which
    monkeypatch.setattr(core.shutil, 'which',
        lambda name: '/synthetic/vitis' if name == 'core-test-vitis' else actual_which(name))
    called = []

    def synthesize(candidate, **kwargs):
        measurement = measurements[len(called)]
        called.append(core.sha(candidate))
        metrics = {'latency_cycles_worst': measurement['cycles'],
                   'requested_clock_period_ns': 3.33, 'estimated_clock_period_ns': 2.5,
                   'dsp': measurement.get('dsp', 4), 'lut': 40, 'ff': 50, 'bram': 0, 'uram': 0}
        return {'passed': True, 'status': 'passed', 'measurement_origin': 'synthetic_test_only',
                'metrics': {'hls_estimate': metrics}, 'input_hashes': {'candidate': core.sha(candidate)}}

    monkeypatch.setattr(core.tools, 'synthesize_vitis', synthesize)
    return called


def test_repairs_wrong_output_then_selects_first_passing_candidate_without_oracle_in_prompt(project, tmp_path):
    out = tmp_path / 'run'
    result = core.run(project.path, out, responses=responses(project,
                      [project.wrong, project.correct, project.other_correct]), max_calls=3)
    assert result['status'] == 'passed'
    assert len(result['attempts']) == 2
    assert not result['attempts'][0]['native']['passed']
    assert result['selected']['attempt'] == 1
    assert Path(result['selected']['path']).read_text() == project.correct
    assert result['selected']['sha256'] == result['attempts'][1]['source_sha256']
    assert result['selected']['independent_native_passed']
    assert len(result['api_accounting']['calls']) == 2
    assert all(call['provider_calls'] == 0 for call in result['api_accounting']['calls'])
    requests = captured_messages(out)
    assert len(requests) == 2
    for messages in requests:
        serialized = json.dumps(messages)
        assert PRIVATE_OUTPUT not in serialized
        assert 'private_vectors.h' not in serialized
        assert 'int main(void)' not in serialized
        assert 'uint32_t kernel' in serialized
    assert result['input_manifest']['private_vectors.h']['sha256'] == core.sha(project.root / 'private_vectors.h')


def test_wrong_explicit_candidate_is_rejected_before_optional_rtl(project, tmp_path, monkeypatch):
    candidate = tmp_path / 'wrong.cpp'
    candidate.write_text(project.wrong)

    def unexpected_rtl(*args, **kwargs):
        pytest.fail('RTL must not run for a native nonpass')

    monkeypatch.setattr(core.tools, 'simulate_verilator', unexpected_rtl)
    result = core.validate(project.path, candidate, tmp_path / 'check', rtl_config=tmp_path / 'unused.json')
    assert result['status'] == 'failed'
    assert not result['native']['passed']
    assert result['rtl'] is None


def test_failed_and_slower_improvements_keep_the_evaluated_winner(project, tmp_path, monkeypatch):
    calls = synthetic_synthesis(project, monkeypatch, [{'cycles': 100}, {'cycles': 140}])
    out = tmp_path / 'optimized'
    result = core.run(project.path, out, responses=responses(project,
                      [project.correct, project.wrong, project.other_correct]),
                      max_calls=3, optimize=2, synthesize=True)
    assert result['status'] == 'passed'
    assert len(result['attempts']) == 3
    assert not result['attempts'][1]['native']['passed']
    assert len(calls) == 2
    assert result['selected']['attempt'] == 0
    assert result['selected']['objective']['value'] == 100
    assert Path(result['selected']['path']).read_text() == project.correct
    assert result['selected']['sha256'] == calls[0] == result['attempts'][0]['source_sha256']
    final = json.loads((out / 'final-validation.json').read_text())
    assert final['passed'] and final['input_hashes']['candidate'] == result['selected']['sha256']


def test_faster_candidate_over_resource_budget_does_not_replace_winner(project, tmp_path, monkeypatch):
    synthetic_synthesis(project, monkeypatch, [{'cycles': 100}, {'cycles': 10, 'dsp': 9}])
    result = core.run(project.path, tmp_path / 'budgeted', responses=responses(project,
                      [project.correct, project.other_correct]), max_calls=2, optimize=1, synthesize=True)
    assert result['status'] == 'passed'
    assert result['selected']['attempt'] == 0
    assert result['selected']['objective']['value'] == 100
    assert not result['attempts'][1]['eligibility']['passed']
    assert 'dsp_exceeds_budget' in result['attempts'][1]['eligibility']['reasons']


def test_fractional_hardware_cycle_count_cannot_qualify_a_candidate(project, tmp_path, monkeypatch):
    synthetic_synthesis(project, monkeypatch, [{'cycles': 10.5}])
    result = core.run(project.path, tmp_path / 'fractional',
                      responses=responses(project, [project.correct]), max_calls=1, synthesize=True)
    assert result['status'] != 'passed'
    assert result['selected'] is None


def test_private_compiler_diagnostic_never_enters_repair_prompt(project, tmp_path, monkeypatch):
    nonpass = {'status': 'failed', 'passed': False, 'reference': {'status': 'failed', 'stdout': ''}, 'candidate': {},
               'comparison': None, 'commands': [{'argv': ['gcc', '-c', 'testbench.c'],
                 'status': 'failed', 'stderr': PRIVATE_DIAGNOSTIC}]}
    monkeypatch.setattr(core.tools, 'validate_native', lambda *args, **kwargs: deepcopy(nonpass))
    out = tmp_path / 'private-diagnostic'
    result = core.run(project.path, out,
                      responses=responses(project, [project.correct, project.other_correct]), max_calls=2)
    assert result['status'] != 'passed'
    assert len(result['api_accounting']['calls']) == 1
    assert result['attempts'][0]['status'] == 'blocked_input'
    for messages in captured_messages(out):
        assert PRIVATE_DIAGNOSTIC not in json.dumps(messages)


def test_unavailable_native_backend_stops_without_a_false_pass(project, tmp_path, monkeypatch):
    monkeypatch.setattr(core.tools, 'validate_native', lambda *args, **kwargs:
                        {'status': 'unavailable', 'passed': False, 'commands': []})
    out = tmp_path / 'missing-tool'
    result = core.run(project.path, out,
                      responses=responses(project, [project.correct, project.other_correct]), max_calls=2)
    assert result['status'] == 'failed'
    assert result['selected'] is None
    assert len(result['api_accounting']['calls']) == 1
    assert len(result['attempts']) == 1


def test_final_independent_check_failure_prevents_success(project, tmp_path, monkeypatch):
    actual_validate = core.tools.validate_native

    def validate(*args, **kwargs):
        result = actual_validate(*args, **kwargs)
        if Path(kwargs['out']).name == 'final-validation':
            result.update(passed=False, status='failed')
        return result

    monkeypatch.setattr(core.tools, 'validate_native', validate)
    result = core.run(project.path, tmp_path / 'final-failure',
                      responses=responses(project, [project.correct]), max_calls=1)
    assert result['status'] == 'failed_final_validation'
    assert not result['selected']['independent_native_passed']
