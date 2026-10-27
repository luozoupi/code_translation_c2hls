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
                'metrics': {'hls_estimate': metrics}, 'input_hashes': {'candidate': core.sha(candidate)},
                'workspace': str(kwargs['out']), 'solution_dir': str(Path(kwargs['out']) / 'hls_project/solution')}

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


def synthetic_trace(project, monkeypatch, estimates):
    """Exercise controller integration without importing or executing LightningSim."""
    project.config['lightningsim'] = {'executable': 'core-test-lightningsim'}
    project.path.write_text(json.dumps(project.config))
    actual_which = core.shutil.which
    monkeypatch.setattr(core.shutil, 'which',
        lambda name: '/synthetic/lightningsim' if name == 'core-test-lightningsim' else actual_which(name))
    calls = []

    def estimate(solution, **kwargs):
        value = estimates[len(calls)]
        calls.append({'solution': str(solution), **kwargs})
        return {'backend': 'lightningsim', 'status': 'estimated' if value is not None else 'failed',
                'estimate_available': value is not None, 'passed': False, 'correctness_checked': False,
                'measurement_origin': 'trace_estimate', 'metrics': {'trace_estimate_cycles': value},
                'provenance': kwargs.get('provenance', {})}

    monkeypatch.setattr(core.trace_backend, 'estimate', estimate)
    return calls


def test_trace_is_diagnostic_and_selected_trace_belongs_to_hls_winner(project, tmp_path, monkeypatch):
    synthesis_calls = synthetic_synthesis(project, monkeypatch, [{'cycles': 100}, {'cycles': 140}])
    trace_calls = synthetic_trace(project, monkeypatch, [9000, 5])
    out = tmp_path / 'trace-selection'
    result = core.run(project.path, out, responses=responses(project,
                      [project.correct, project.other_correct]), max_calls=2, optimize=1,
                      synthesize=True, lightningsim=True)
    assert result['status'] == 'passed'
    assert len(trace_calls) == len(synthesis_calls) == len(result['api_accounting']['calls']) == 2
    assert result['selected']['attempt'] == 0
    assert result['selected']['objective']['value'] == 100
    assert result['hardware']['trace_estimate_cycles'] == 9000
    assert result['hardware']['rtl_cycles'] is None
    assert result['lightningsim']['passed'] is False
    assert result['lightningsim']['correctness_checked'] is False
    assert trace_calls[0]['provenance']['candidate_sha256'] == result['selected']['sha256']
    summary = core.report(out)
    assert summary['trace_estimate_cycles'] == 9000 and summary['rtl_cycles'] is None
    for messages in captured_messages(out):
        text = json.dumps(messages)
        assert 'trace_estimate_cycles' not in text
        assert PRIVATE_OUTPUT not in text


def test_requested_trace_failure_preserves_native_verdict_but_marks_estimation_incomplete(project, tmp_path, monkeypatch):
    synthetic_synthesis(project, monkeypatch, [{'cycles': 100}])
    trace_calls = synthetic_trace(project, monkeypatch, [None])
    out = tmp_path / 'trace-failure'
    result = core.run(project.path, out, responses=responses(project, [project.correct]),
                      max_calls=2, synthesize=True, lightningsim=True)
    assert len(trace_calls) == len(result['api_accounting']['calls']) == 1
    assert result['status'] == 'incomplete_estimation'
    assert result['selected']['independent_native_passed']
    assert result['lightningsim']['estimate_available'] is False
    assert result['lightningsim']['passed'] is False
    assert result['hardware']['trace_estimate_cycles'] is None
    assert core.report(out)['native_passed'] is True


def test_trace_requirements_are_checked_before_model_initialization(project, tmp_path, monkeypatch):
    def unexpected_model(*args, **kwargs):
        pytest.fail('preflight failure must precede model initialization')

    monkeypatch.setattr(core, 'Model', unexpected_model)
    with pytest.raises(ValueError, match='requires --synthesize'):
        core.run(project.path, tmp_path / 'without-synthesis', lightningsim=True)
    synthetic_synthesis(project, monkeypatch, [{'cycles': 100}])
    actual_which = core.shutil.which
    monkeypatch.setattr(core.shutil, 'which', lambda name: None if name == 'lightningsim' else actual_which(name))
    with pytest.raises(ValueError, match='LightningSim executable unavailable'):
        core.run(project.path, tmp_path / 'without-trace-tool', synthesize=True, lightningsim=True)
    assert not (tmp_path / 'without-synthesis').exists()
    assert not (tmp_path / 'without-trace-tool').exists()


def test_external_solution_estimate_does_not_establish_candidate_rtl_binding(project, tmp_path, monkeypatch):
    trace_calls = synthetic_trace(project, monkeypatch, [23])
    candidate = tmp_path / 'candidate.cpp'
    candidate.write_text(project.correct)
    out = tmp_path / 'external-trace'
    result = core.validate(project.path, candidate, out,
                           lightningsim_solution=tmp_path / 'external/solution',
                           lightningsim_workspace=tmp_path / 'external')
    assert result['status'] == 'passed'
    assert result['native']['passed'] and result['rtl'] is None
    assert result['lightningsim']['passed'] is False
    assert result['lightningsim']['correctness_checked'] is False
    assert trace_calls[0]['provenance']['candidate_binding'] == 'not_established'
    assert result['lightningsim']['provenance']['candidate_binding'] == 'not_established'
    assert core.report(out)['trace_estimate_cycles'] == 23
    assert core.report(out)['rtl_cycles'] is None


def test_default_synthesis_does_not_register_private_testbench_or_run_trace(project, tmp_path, monkeypatch):
    synthetic_synthesis(project, monkeypatch, [{'cycles': 100}])
    synthesis = core.tools.synthesize_vitis
    calls = []

    def capture(candidate, **kwargs):
        calls.append(kwargs)
        return synthesis(candidate, **kwargs)

    monkeypatch.setattr(core.tools, 'synthesize_vitis', capture)
    monkeypatch.setattr(core.trace_backend, 'estimate', lambda *args, **kwargs:
                        pytest.fail('default synthesis must not run LightningSim'))
    result = core.run(project.path, tmp_path / 'default-synthesis',
                      responses=responses(project, [project.correct]), max_calls=1, synthesize=True)
    assert result['status'] == 'passed' and len(calls) == 1
    assert 'testbench' not in calls[0] and 'test_files' not in calls[0]


def synthetic_bambu(project, monkeypatch, outcomes):
    """Only generated-RTL results are mocked; the original-C gate stays real."""
    project.config['bambu'] = {'executable': 'core-test-bambu', 'timeout': 120, 'clock_ns': 3.33}
    project.path.write_text(json.dumps(project.config))
    actual_which = core.shutil.which
    monkeypatch.setattr(core.shutil, 'which', lambda name:
        '/synthetic/' + name if name in {'core-test-bambu', 'verilator'} else actual_which(name))
    for module, name in [(core.tools, 'synthesize_vitis'), (core.tools, 'simulate_verilator'),
                         (core.trace_backend, 'estimate')]:
        monkeypatch.setattr(module, name, lambda *args, **kwargs:
                            pytest.fail('Bambu dispatch must not execute the Vitis, explicit-RTL or LightningSim path'))
    calls = []

    def evaluate(candidate, testbench, **kwargs):
        native = json.loads((Path(kwargs['out']).parent / 'native.json').read_text())
        assert native['passed'], 'original-C comparison must precede generated-RTL work'
        passed, cycles = outcomes[len(calls)]
        calls.append({'candidate_sha256': core.sha(candidate),
                      'testbench_sha256': core.sha(testbench), **kwargs})
        return {'backend': 'bambu', 'status': 'passed' if passed else 'failed', 'passed': passed,
                'measurement_origin': 'synthetic_test_only', 'builtin_validation_enabled': passed,
                'input_hashes': {'candidate.cpp': core.sha(candidate), 'testbench.c': core.sha(testbench)},
                'requested_clock_ns': kwargs.get('clock_ns'), 'metrics': {'rtl_cycles': cycles},
                'qualification': 'Synthetic pipeline-test result, not an actual RTL measurement'}

    monkeypatch.setattr(core.bambu_backend, 'evaluate', evaluate)
    return calls


def test_bambu_runs_only_after_original_c_gate_and_does_not_add_model_calls(project, tmp_path, monkeypatch):
    calls = synthetic_bambu(project, monkeypatch, [(True, 51)])
    result = core.run(project.path, tmp_path / 'bambu-repair',
                      responses=responses(project, [project.wrong, project.correct]), max_calls=2, backend='bambu')
    assert result['status'] == 'passed' and result['backend'] == 'bambu'
    assert len(calls) == 1 and len(result['api_accounting']['calls']) == 2
    assert all(row['provider_calls'] == 0 for row in result['api_accounting']['calls'])
    assert result['selected']['attempt'] == 1
    assert result['selected']['independent_native_passed'] and result['selected']['generated_rtl_passed']
    assert calls[0]['candidate_sha256'] == result['selected']['sha256']
    assert result['bambu']['input_hashes']['candidate.cpp'] == result['selected']['sha256']
    assert result['hardware']['rtl_cycles'] == 51
    assert result['hardware']['resource_eligibility'] is None
    assert result['hardware']['physical_timing_fit'] is None
    assert result['hardware']['physical_implementation'] is None
    assert result['bambu']['requested_clock_ns'] == 3.33
    assert core.report(tmp_path / 'bambu-repair')['candidate_rtl_binding'] == 'bambu_generated_and_cosim_checked'
    assert 'estimated_clock_period_ns' not in result['bambu']['metrics']


def test_bambu_rtl_failure_cannot_select_a_native_passing_candidate(project, tmp_path, monkeypatch):
    calls = synthetic_bambu(project, monkeypatch, [(False, None)])
    result = core.run(project.path, tmp_path / 'bambu-failure',
                      responses=responses(project, [project.correct]), max_calls=1, backend='bambu')
    assert len(calls) == 1 and result['attempts'][0]['native']['passed']
    assert result['status'] == 'failed' and result['selected'] is None
    assert not result['attempts'][0]['bambu']['passed']
    assert not result['attempts'][0]['eligibility']['passed']
    assert result['hardware']['rtl_cycles'] is None


def test_bambu_failure_and_slower_proposal_keep_winner_and_its_rtl_receipt(project, tmp_path, monkeypatch):
    calls = synthetic_bambu(project, monkeypatch, [(True, 90), (False, None), (True, 110)])
    third = project.other_correct + '/* Distinct final proposal. */\n'
    out = tmp_path / 'bambu-best'
    result = core.run(project.path, out,
                      responses=responses(project, [project.correct, project.other_correct, third]),
                      max_calls=3, optimize=2, backend='bambu')
    assert len(calls) == len(result['api_accounting']['calls']) == 3
    assert result['status'] == 'passed'
    assert result['selected']['attempt'] == 0
    assert result['selected']['objective'] == {'metric': 'rtl_cosim_total_cycles', 'value': 90}
    assert result['selected']['sha256'] == calls[0]['candidate_sha256']
    assert result['bambu']['input_hashes']['candidate.cpp'] == result['selected']['sha256']
    assert result['bambu']['metrics']['rtl_cycles'] == result['hardware']['rtl_cycles'] == 90
    assert core.report(out)['resource_eligibility'] is None
    assert core.report(out)['rtl_passed'] is True


def test_bambu_explicit_validation_stops_on_original_native_mismatch(project, tmp_path, monkeypatch):
    calls = synthetic_bambu(project, monkeypatch, [])
    monkeypatch.setattr(core, 'Model', lambda *args, **kwargs:
                        pytest.fail('explicit validation must not initialize a model client'))
    candidate = tmp_path / 'wrong.cpp'
    candidate.write_text(project.wrong)
    result = core.validate(project.path, candidate, tmp_path / 'bambu-native-failure', backend='bambu')
    assert result['status'] == 'failed' and not result['native']['passed']
    assert not calls and 'bambu' not in result


def test_bambu_verilator_preflight_precedes_model_and_output_creation(project, tmp_path, monkeypatch):
    project.config['bambu'] = {'executable': 'core-test-bambu'}
    project.path.write_text(json.dumps(project.config))
    actual_which = core.shutil.which
    monkeypatch.setattr(core.shutil, 'which', lambda name:
        '/synthetic/bambu' if name == 'core-test-bambu' else None if name == 'verilator' else actual_which(name))
    monkeypatch.setattr(core, 'Model', lambda *args, **kwargs:
                        pytest.fail('missing co-simulator must be detected before any model request'))
    out = tmp_path / 'missing-verilator'
    with pytest.raises(ValueError, match='Verilator'):
        core.run(project.path, out, backend='bambu')
    assert not out.exists()


def test_plain_c_with_private_supplied_testbench_repairs_and_saves_reusable_inputs(project, tmp_path):
    out = tmp_path / 'plain'
    result = core.run(project.root / 'original.c', out, top='kernel',
        testbench=project.root / 'tb.c', headers=[project.root / 'kernel.h'], backend='native',
        responses=responses(project, [project.wrong, project.correct]), max_calls=2)
    assert result['status'] == 'passed' and result['selected']['attempt'] == 1
    assert result['testbench']['kind'] == 'provided'
    assert result['testbench']['reference_preflight_passed']
    assert len(result['api_accounting']['calls']) == 2
    for messages in captured_messages(out):
        assert PRIVATE_OUTPUT not in json.dumps(messages)
        assert 'private_vectors.h' not in json.dumps(messages)
    for name in ('reference.c', 'testbench.c', 'kernel.h', 'private_vectors.h', 'kernel.json'):
        assert (out / 'inputs' / name).is_file()
    prepared = core.load_project(out / 'inputs/kernel.json')
    assert all(p.is_file() for p in [prepared['source'], prepared['testbench'], *prepared['test_files'].values()])
    checked = core.validate(out / 'inputs/kernel.json', out / 'selected.cpp', tmp_path / 'rechecked')
    assert checked['status'] == 'passed'


def test_plain_c_generated_harness_observes_real_outputs_and_stays_private(project, tmp_path):
    out = tmp_path / 'auto'
    result = core.run(project.root / 'original.c', out, top='kernel', backend='native',
        responses=responses(project, [project.wrong, project.correct]), max_calls=2)
    assert result['status'] == 'passed' and result['selected']['attempt'] == 1
    assert result['testbench']['kind'] == 'generated' and result['testbench']['cases'] == 3
    assert result['testbench']['reference_preflight_passed']
    frozen = core.sha(out / 'inputs/testbench.c')
    assert frozen == result['testbench']['sha256']
    for messages in captured_messages(out):
        text = json.dumps(messages)
        assert 'int main(void)' not in text and '__hlsmith_case' not in text
    final = json.loads((out / 'final-validation.json').read_text())
    assert final['input_hashes']['testbench'] == frozen
    assert final['input_hashes']['include_files']['hlsmith_task.h']


def test_plain_c_failed_original_preflight_never_initializes_model(project, tmp_path, monkeypatch):
    (project.root / 'original.c').write_text('invalid C source\n')
    def forbidden(*args, **kwargs):
        raise AssertionError('model initialized before the original C check passed')
    monkeypatch.setattr(core, 'Model', forbidden)
    out = tmp_path / 'bad-input'
    result = core.run(project.root / 'original.c', out, top='kernel', backend='native',
        testbench=project.root / 'tb.c', headers=[project.root / 'kernel.h'])
    assert result['status'] == 'blocked_input'
    assert result['api_accounting']['calls'] == []
    assert result['api_accounting']['charged_or_reserved_usd'] == 0
    assert not (out / 'model').exists()


def test_generated_harness_bad_buffer_extent_blocks_before_model(tmp_path, monkeypatch):
    source = tmp_path / 'bounds.c'
    source.write_text('void kernel(int *output) {output[3]=7;}\n')
    monkeypatch.setattr(core, 'Model', lambda *args, **kwargs:
                        pytest.fail('model initialized for an invalid generated testbench domain'))
    out = tmp_path / 'bad-extent'
    result = core.run(source, out, top='kernel', sizes=['output=1'], backend='native')
    assert result['status'] == 'blocked_input' and result['api_accounting']['calls'] == []
    preflight = json.loads((out / 'reference-check.json').read_text())
    assert preflight['sanitizers'] == ['address', 'undefined'] and not preflight['passed']
    assert 'AddressSanitizer' in preflight['reference']['stderr']
    checked = core.validate(source, source, tmp_path / 'bad-extent-validate',
                            top='kernel', sizes=['output=1'])
    assert checked['status'] == 'blocked_input' and not checked['native']['passed']
