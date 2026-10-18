"""Fixed archived selection and launch accounting, without model or HLS jobs."""
from argparse import Namespace
import json
import os
from pathlib import Path
import sys

import pytest

from hlsmith import saved_designs as gate


@pytest.fixture(autouse=True)
def restore_worker_controls():
    environment, paths = os.environ.copy(), sys.path[:]
    yield
    os.environ.clear()
    os.environ.update(environment)
    sys.path[:] = paths


@pytest.fixture
def saved_registration(tmp_path):
    original = tmp_path / 'original'
    bench = original / 'benchmark'
    bench.mkdir(parents=True)
    for name, text in [('plain.cpp', 'int workload(int x) { return x; }\n'),
                       ('kernel.h', 'int workload(int);\n'), ('testbench.cpp', 'int main() { return 0; }\n')]:
        (bench / name).write_text(text)
    gate.tools.save(bench / 'metadata.json', {'benchmark': 'hlsfactory_gemm', 'source_repo': 'HLSFactory',
                    'header_file': 'kernel.h', 'testbench_file': 'testbench.cpp',
                    'cosim_size_overrides': {'N': 2}, 'cosim_testbench_file': 'reduced.cpp'})
    kernel = original / 'selected.cpp'
    kernel.write_text('int workload(int x) { return x; }\n')
    registry = tmp_path / 'configs' / 'shapes.json'
    gate.tools.save(registry, {'schema_version': 'c2hls.hlsfactory-output-shapes.v1',
                             'benchmarks': {'hlsfactory_gemm': {'outputs': {'A': {'shape': [1], 'kind': 'float'}}}}})
    base = {'benchmark': 'hlsfactory_gemm', 'kernel_sha256': gate.tools.sha(kernel),
            'staged_kernel': str(kernel), 'staged_benchmark': str(bench),
            'input_hashes': {p.name: gate.tools.sha(p) for p in bench.iterdir()},
            'golden_registry': {'path': 'configs/shapes.json', 'sha256': gate.tools.sha(registry),
                                'testbench_sha256': gate.tools.sha(bench / 'testbench.cpp')}}
    commercial = [{'validation_id': f'c{i:03}', 'paper_slots': [f'role{i}'] + ([f'extra{i}'] if i < 12 else []),
                   **base} for i in range(144)]
    recovery = [{'validation_id': f'o{i:03}', **base} for i in range(180)]
    slots = [{'model': 'frozen-model', 'workflow': 'flash', 'benchmark': 'hlsfactory_gemm',
              'arm': 'selected' if i % 2 else 'off', 'validation_id': f'o{i % 180:03}',
              'kernel_sha256': base['kernel_sha256']} for i in range(211)]
    slots += [{'model': 'frozen-model', 'workflow': 'multistep', 'benchmark': f'blocked{i}',
               'arm': 'off', 'validation_id': None} for i in range(5)]
    for name, data in [('commercial', {'rows': commercial}), ('recovery', {'rows': recovery}), ('identity', {'rows': slots})]:
        gate.tools.save(tmp_path / (name + '.json'), data)
    for name in gate.COHORTS:
        root = tmp_path / name
        root.mkdir()
        (root / 'c2hls.py').write_text('# synthetic implementation identity\n')
    args = Namespace(commercial_manifest=tmp_path / 'commercial.json', recovery_manifest=tmp_path / 'recovery.json',
                     open_identity=tmp_path / 'identity.json', registry_root=tmp_path,
                     frozen_source=tmp_path / 'frozen', clean_source=tmp_path / 'clean', output=tmp_path / 'run')
    gate.register(args)
    return args


def version_fixture(tmp_path, monkeypatch):
    settings, binary = tmp_path / 'settings.sh', tmp_path / 'vitis-run'
    settings.write_text('# no tool execution\n')
    binary.write_text('# verified fixture\n')
    monkeypatch.setattr(gate.tools, 'probe_vitis_versions', lambda settings: {
        name: {'status': 'verified', 'observed_version': '2023.2', 'attempts': [{}],
               'binary': str(binary), 'binary_sha256': gate.tools.sha(binary)}
        for name in ('vitis-run', 'vitis_hls')})
    monkeypatch.setattr(gate.tools, 'active_vitis', lambda: [])
    return settings


def test_registration_preserves_fixed_designs_roles_and_blocked_denominator(saved_registration):
    output = saved_registration.output
    manifest = gate.verify(output)
    assert manifest['counts'] == gate.EXPECTED
    assert len(manifest['jobs']) == 648
    assert len(manifest['designs']) == 324
    assert sum(len(d['paper_roles']) for d in manifest['designs']) == 156
    assert sum(s['status'] == 'blocked_identity' for s in manifest['open_slots']) == 5
    # The original public harness is byte-identical; no reduced testbench is copied.
    for design in manifest['designs']:
        bench = Path(design['path']) / 'benchmark'
        assert gate.tools.sha(bench / 'testbench.cpp') == design['input_hashes']['testbench.cpp']
        assert not (bench / 'reduced.cpp').exists()
    report = gate.report(output)
    assert report['status_counts'] == {'pending': 648}
    assert report['paired_denominator'] == 324
    assert report['paired_rtl_passes'] == 0
    assert report['job_hours_charged'] == report['provider_calls'] == 0
    assert not (output / 'jobs').exists()
    assert not (output / 'launch.json').exists()
    with pytest.raises(FileExistsError):
        gate.register(saved_registration)


def test_registration_rejects_missing_or_conflicting_selected_identity(saved_registration):
    args = saved_registration
    commercial = gate.tools.read(args.commercial_manifest)
    recovery = gate.tools.read(args.recovery_manifest)
    identity = gate.tools.read(args.open_identity)
    identity['rows'][0]['kernel_sha256'] = 'wrong selected code'
    with pytest.raises(ValueError, match='selected-source identity mismatch'):
        gate.fixed_selection(commercial, recovery, identity)
    identity['rows'].pop()
    identity['rows'][0]['kernel_sha256'] = recovery['rows'][0]['kernel_sha256']
    with pytest.raises(ValueError, match='incomplete fixed paper denominator'):
        gate.fixed_selection(commercial, recovery, identity)


@pytest.mark.parametrize('change', ['kernel', 'harness', 'evaluator', 'added_input', 'job_order'])
def test_registered_bytes_and_job_denominator_are_pinned(saved_registration, change):
    output = saved_registration.output
    manifest = gate.verify(output)
    root = Path(manifest['designs'][0]['path'])
    if change == 'kernel':
        (root / 'selected.cpp').write_text('changed source')
    elif change == 'harness':
        (root / 'benchmark' / 'testbench.cpp').write_text('changed harness')
    elif change == 'evaluator':
        (saved_registration.clean_source / 'c2hls.py').write_text('changed evaluator')
    elif change == 'added_input':
        (root / 'benchmark' / 'extra.h').write_text('unregistered extra input')
    else:
        state = gate.tools.read(output / 'state.json')
        state['jobs'].reverse()
        gate.tools.save(output / 'state.json', state)
    with pytest.raises(ValueError):
        gate.verify(output)


def test_public_preflight_report_and_launch_require_explicit_cap(saved_registration, tmp_path, monkeypatch, capsys):
    output = saved_registration.output
    settings = version_fixture(tmp_path, monkeypatch)
    assert gate.main(['preflight', '--output', str(output), '--vitis-settings', str(settings)]) == 0
    checked = json.loads(capsys.readouterr().out)
    assert checked['complete']
    assert checked['provider_calls'] == checked['vitis_job_calls'] == 0
    assert checked['version_queries'] == 2
    assert gate.main(['report', '--output', str(output)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report['registered_cells'] == 648
    assert report['status_counts'] == {'pending': 648}
    with pytest.raises(SystemExit) as result:
        gate.main(['launch', '--output', str(output), '--vitis-settings', str(settings)])
    assert result.value.code == 2
    assert not (output / 'launch.json').exists()


def test_aggregate_cap_blocks_whole_cell_before_any_process(saved_registration, tmp_path, monkeypatch):
    output = saved_registration.output
    settings = version_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(gate.tools, 'run_process', lambda *a, **k: pytest.fail('cap allowed a tool worker'))
    report = gate.launch(Namespace(output=output, max_compute_hours=0.01, vitis_settings=settings,
                                   scratch_root=tmp_path / 'scratch'))
    assert report['campaign_status'] == 'stopped_compute_cap'
    assert report['job_hours_charged'] == 0
    assert report['status_counts'] == {'pending': 648}
    assert not (output / 'jobs').exists()


def test_crashed_cell_reservation_is_not_refunded_or_relaunched(saved_registration, tmp_path, monkeypatch):
    output = saved_registration.output
    settings = version_fixture(tmp_path, monkeypatch)
    def crash(*args, **kwargs):
        raise RuntimeError('simulated launcher loss after reservation')
    monkeypatch.setattr(gate.tools, 'run_process', crash)
    args = Namespace(output=output, max_compute_hours=2, vitis_settings=settings, scratch_root=tmp_path / 'scratch')
    with pytest.raises(RuntimeError, match='launcher loss'):
        gate.launch(args)
    state = gate.tools.read(output / 'state.json')
    assert state['active_job'] == gate.verify(output)['jobs'][0]['id']
    assert state['job_seconds_charged'] == gate.TIMEOUTS['whole_cell']
    with pytest.raises(ValueError, match='interrupted cell requires evidence review'):
        gate.launch(args)


def test_worker_requires_active_launch_and_report_rejects_bare_pass(saved_registration):
    output = saved_registration.output
    job = gate.verify(output)['jobs'][0]
    with pytest.raises(FileNotFoundError):
        gate.worker(Namespace(output=output, job=job['id']))
    gate.tools.save(output / 'jobs' / job['id'] / 'measurement.json', {'job': job, 'status': 'passed',
                    'rtl': {'ran': True, 'passed': True, 'kernel_runtime_cycles': 10}})
    with pytest.raises(ValueError, match='lacks independent'):
        gate.report(output)


@pytest.mark.parametrize('case,status,expected_calls', [
    ('qualified', 'passed', ['native', 'csim', 'rtl']),
    ('native_bad', 'failed_native', ['native']),
    ('csim_unchecked', 'failed_csim', ['native', 'csim']),
    ('rtl_wrong_oracle', 'failed_rtl', ['native', 'csim', 'rtl']),
    ('rtl_zero_cycles', 'failed_rtl', ['native', 'csim', 'rtl']),
    ('synthesis_missing', 'unqualified_rtl_synthesis', ['native', 'csim', 'rtl']),
])
def test_independent_validation_controller_never_promotes_unchecked_tool_pass(
        saved_registration, tmp_path, monkeypatch, case, status, expected_calls):
    import hashlib
    import c2hls
    import hls_eval
    output = saved_registration.output
    manifest = gate.verify(output)
    job = manifest['jobs'][0]
    root = Path(__file__).resolve().parents[1]
    manifest['sources'] = {'frozen': str(root), 'clean': str(root)}
    monkeypatch.setattr(gate, 'verify', lambda output: manifest)
    folder = output / 'jobs' / job['id']
    folder.mkdir(parents=True)
    settings = version_fixture(tmp_path, monkeypatch)
    versions = gate.tools.probe_vitis_versions(settings)
    gate.tools.save(output / 'launch.json', {'manifest_sha256': gate.tools.sha(output / 'manifest.json'),
                    'limits': {'compute_seconds': 10000, 'settings': str(settings),
                               'settings_sha256': gate.tools.sha(settings), 'scratch_root': str(tmp_path / 'scratch'),
                               'tool_versions': versions, 'common_runner': versions['vitis-run']}})
    state = gate.tools.read(output / 'state.json')
    state.update(status='running', active_job=job['id'], job_seconds_charged=4800)
    gate.tools.save(output / 'state.json', state)
    dump = '==BEGIN DUMP_ARRAYS==\nbegin dump: A\n1.0\nend dump: A\n==END DUMP_ARRAYS==\n'
    digest = hashlib.sha256(dump.encode()).hexdigest()
    def golden(inputs, *, shape_registry_path):
        assert inputs['meta']['independent_golden_required'] is True
        assert inputs['testbench_code'] == (tmp_path / 'original/benchmark/testbench.cpp').read_text()
        assert shape_registry_path == manifest['designs'][0]['registry']
        return {'success': True, 'output': dump, 'specs': {'A': {'shape': [1], 'kind': 'float'}},
                'provenance': {'status': 'passed', 'output_sha256': digest}}
    monkeypatch.setattr(c2hls, '_prepare_independent_golden', golden)
    calls = []
    passed = {'success': True, 'passed': True, 'correctness': {'passed': True}, 'golden_output_sha256': digest}
    def native(*args, **kwargs):
        calls.append('native')
        return {'success': True, 'passed': True, 'output': dump.replace('1.0', '9.0') if case == 'native_bad' else dump}
    def csim(*args, **kwargs):
        calls.append('csim')
        assert kwargs['golden_output_text'] == dump
        return {'success': True, 'passed': True} if case == 'csim_unchecked' else dict(passed)
    def rtl(*args, **kwargs):
        calls.append('rtl')
        assert kwargs['golden_output_text'] == dump
        work = Path(kwargs['work_dir'])
        work.mkdir(parents=True)
        (work / 'kernel.cpp').write_text(args[0])
        (work / 'testbench.cpp').write_text(args[1])
        (work / kwargs['header_name']).write_text(kwargs['header_code'])
        return {**passed, 'golden_output_sha256': 'wrong' if case == 'rtl_wrong_oracle' else digest,
                'kernel_runtime_cycles': 0 if case == 'rtl_zero_cycles' else 1234}
    monkeypatch.setattr(hls_eval, 'run_native_testbench', native)
    monkeypatch.setattr(hls_eval, 'run_csim', csim)
    monkeypatch.setattr(hls_eval, 'run_cosim', rtl)
    archived = gate.tools.read(root / 'tests/fixtures/v6_replay/snapshot.json')['cases'][0]['report']
    def synthesis(evaluator, result):
        if case == 'synthesis_missing':
            return {}, {'status': 'missing'}
        report = tmp_path / 'scratch' / job['id'] / 'rtl/hls_proj/sol1/syn/report/csynth.xml'
        gate.tools.save(report, archived)
        tcl = tmp_path / 'scratch' / job['id'] / 'rtl/run_cosim.tcl'
        tcl.write_text('create_clock -period 3.33 -name default\n')
        return archived, {'status': 'recorded', 'path': str(report), 'sha256': gate.tools.sha(report),
                          'requested_clock': {'status': 'verified', 'period_ns': 3.33,
                                              'sources': [{'kind': 'vitis_tcl', 'path': str(tcl),
                                                           'sha256': gate.tools.sha(tcl), 'period_ns': 3.33}]}}
    monkeypatch.setattr(gate.tools, 'fresh_rtl_synthesis', synthesis)
    gate.worker(Namespace(output=output, job=job['id']))
    measured = gate.tools.read(folder / 'measurement.json')
    assert calls == expected_calls
    assert measured['status'] == status
    assert measured['provider_calls'] == 0
    assert measured['golden_output_sha256'] == digest
    assert measured['target'] == gate.TARGET
    if case == 'qualified':
        assert gate.report(output)['paired_rtl_passes'] == 0  # Its clean counterpart is still pending.
        for field in ('design_source_sha256', 'input_hashes', 'target', 'common_runner'):
            changed = {**measured, field: 'foreign measurement'}
            gate.tools.save(folder / 'measurement.json', changed)
            with pytest.raises((ValueError, AttributeError), match='independent|runner'):
                gate.report(output)
        gate.tools.save(folder / 'measurement.json', measured)
        tcl = Path(measured['rtl_synthesis_evidence']['requested_clock']['sources'][0]['path'])
        original_tcl = tcl.read_bytes()
        tcl.write_text('create_clock -period 5.0 -name default\n')
        with pytest.raises(ValueError, match='requested clock provenance changed'):
            gate.report(output)
        tcl.write_bytes(original_tcl)
        Path(measured['rtl_synthesis_evidence']['path']).write_text('changed synthesis report')
        with pytest.raises(ValueError, match='fresh synthesis evidence changed'):
            gate.report(output)
