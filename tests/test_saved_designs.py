"""Fixed archived selection and launch accounting, without model or HLS jobs."""
from argparse import Namespace
import json
import os
from pathlib import Path
import sys
import signal
import shutil
import subprocess
import threading

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


def saved_failure(output, job, status='failed_native'):
    """Save a real-shaped terminal nonpass, never an invented checked pass."""
    receipt = gate.tools.read(output / 'launch.json')
    controller = receipt['controller']
    gate.tools.save(output / 'jobs' / job['id'] / 'measurement.json', {
        'job': job, 'status': status, 'rtl': {'ran': False}, 'provider_calls': 0,
        'dispatch_controller': controller, 'execution_epoch': controller['execution_epoch'],
        'max_workers': controller['max_workers'], 'controller_sha256': controller['sha256']})


def test_parallel_reservations_bound_concurrency_and_only_parent_writes_state(saved_registration, tmp_path, monkeypatch):
    output = saved_registration.output
    manifest = gate.verify(output)
    jobs = {job['id']: job for job in manifest['jobs']}
    settings = version_fixture(tmp_path, monkeypatch)
    barrier = threading.Barrier(3)
    mutex = threading.Lock()
    live, peak, started = set(), 0, []
    parent = threading.get_ident()
    original_save = gate.tools.save
    def save(path, value):
        if Path(path).name in {'state.json', 'launch.json'}:
            assert threading.get_ident() == parent
        original_save(path, value)
    monkeypatch.setattr(gate.tools, 'save', save)
    def run(command, **kwargs):
        nonlocal peak
        identity = command[command.index('--job') + 1]
        assert command[1] == str(Path(gate.__file__).resolve())
        assert kwargs['env']['PYTHONPATH'] == str(saved_registration.clean_source)
        assert '--execution-epoch' in command
        with mutex:
            live.add(identity)
            started.append(identity)
            peak = max(peak, len(live))
        try:
            barrier.wait(timeout=5)
            for _ in range(10):
                state = gate.tools.read(output / 'state.json')
                assert identity in gate._active_job_ids(state)
                assert state['job_seconds_charged'] <= 18000
                reserved = sum(row.get('reserved_seconds', 0) for row in state['jobs'] if row['status'] == 'running')
                assert reserved <= state['job_seconds_charged']
            saved_failure(output, jobs[identity])
            return {'status': 'complete', 'returncode': 0, 'elapsed_seconds': 4800}
        finally:
            with mutex:
                live.remove(identity)
    monkeypatch.setattr(gate.tools, 'run_process', run)
    report = gate.launch(Namespace(output=output, max_compute_hours=5, workers=4,
                    vitis_settings=settings, scratch_root=tmp_path / 'scratch'))
    assert len(started) == peak == 3  # Four slots, but only three full reservations fit.
    assert report['campaign_status'] == 'stopped_compute_cap'
    assert report['job_hours_charged'] == 4
    assert report['execution_mode_status_counts']['parallel'] == {'failed_native': 3}
    state = gate.tools.read(output / 'state.json')
    assert not gate._active_job_ids(state)
    assert state['job_seconds_charged'] == 14400


def test_resume_keeps_completed_bytes_budget_and_serial_parallel_strata(saved_registration, tmp_path, monkeypatch):
    output = saved_registration.output
    manifest = gate.verify(output)
    jobs = {job['id']: job for job in manifest['jobs']}
    settings = version_fixture(tmp_path, monkeypatch)
    started = []
    def run(command, **kwargs):
        identity = command[command.index('--job') + 1]
        started.append(identity)
        saved_failure(output, jobs[identity], 'failed_native' if len(started) == 1 else 'rtl_timeout')
        return {'status': 'complete', 'returncode': 0, 'elapsed_seconds': 4800}
    monkeypatch.setattr(gate.tools, 'run_process', run)
    args = Namespace(output=output, max_compute_hours=4800 / 3600, vitis_settings=settings,
                     scratch_root=tmp_path / 'scratch')  # Existing programmatic caller: no workers attribute.
    first = gate.launch(args)
    old_path = output / 'jobs' / started[0] / 'measurement.json'
    old_bytes = old_path.read_bytes()
    assert first['dispatch_controller']['max_workers'] == 1
    args.max_compute_hours, args.workers = 9600 / 3600, 2
    second = gate.launch(args)
    assert started == [job['id'] for job in manifest['jobs'][:2]]
    assert old_path.read_bytes() == old_bytes
    assert second['job_hours_charged'] == 9600 / 3600
    assert second['paired_status_differences'] == second['mixed_execution_pairs'] == 1
    assert second['paired_execution_modes']['serial/parallel'] == 1
    assert len(second['execution_epochs']) == 2
    assert len(list((output / 'launches').glob('*.json'))) == 2
    args.scratch_root = tmp_path / 'different-scratch'
    with pytest.raises(ValueError, match='same scratch root and common runner'):
        gate.launch(args)
    assert len(started) == 2


def test_parallel_signal_drains_owned_cells_and_dispatches_no_replacements(saved_registration, tmp_path, monkeypatch):
    output = saved_registration.output
    jobs = {job['id']: job for job in gate.verify(output)['jobs']}
    settings = version_fixture(tmp_path, monkeypatch)
    handlers = {}
    original_signal = signal.signal
    def install(signum, handler):
        handlers[signum] = handler
        return original_signal(signum, handler)
    monkeypatch.setattr(gate.signal, 'signal', install)
    barrier = threading.Barrier(2)
    started, finished = [], []
    def run(command, **kwargs):
        identity = command[command.index('--job') + 1]
        started.append(identity)
        order = barrier.wait(timeout=5)
        if order == 0:
            handlers[signal.SIGTERM](signal.SIGTERM, None)
        barrier.wait(timeout=5)
        saved_failure(output, jobs[identity])
        finished.append(identity)
        return {'status': 'complete', 'returncode': 0, 'elapsed_seconds': 200}
    monkeypatch.setattr(gate.tools, 'run_process', run)
    report = gate.launch(Namespace(output=output, max_compute_hours=10, workers=2,
                    vitis_settings=settings, scratch_root=tmp_path / 'scratch'))
    assert len(started) == len(finished) == 2
    assert report['campaign_status'] == 'stopped_by_signal'
    assert report['job_hours_charged'] == 400 / 3600
    assert not gate._active_job_ids(gate.tools.read(output / 'state.json'))


def test_whole_cell_timeout_fallback_has_reportable_dispatch_provenance(saved_registration, tmp_path, monkeypatch):
    output = saved_registration.output
    settings = version_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(gate.tools, 'run_process', lambda *a, **k: {
        'status': 'timeout', 'returncode': None, 'elapsed_seconds': 50})
    report = gate.launch(Namespace(output=output, max_compute_hours=4800 / 3600, workers=2,
                    vitis_settings=settings, scratch_root=tmp_path / 'scratch'))
    assert report['status_counts']['whole_cell_timeout'] == 1
    assert report['execution_mode_status_counts']['parallel'] == {'whole_cell_timeout': 1}
    path = next((output / 'jobs').glob('*/measurement.json'))
    measured = gate.tools.read(path)
    original = json.loads(json.dumps(measured))
    measured.pop('dispatch_controller')
    gate.tools.save(path, measured)
    with pytest.raises(ValueError, match='metadata lacks its controller'):
        gate.report(output)
    gate.tools.save(path, original)
    state = gate.tools.read(output / 'state.json')
    state['jobs'][0]['max_workers'] = 999
    gate.tools.save(output / 'state.json', state)
    with pytest.raises(ValueError, match='execution metadata changed'):
        gate.report(output)


def test_running_or_unmeasured_partner_is_not_a_paired_status_difference(saved_registration):
    output = saved_registration.output
    manifest = gate.verify(output)
    first = manifest['jobs'][0]
    gate.tools.save(output / 'jobs' / first['id'] / 'measurement.json', {
        'job': first, 'status': 'failed_native', 'rtl': {'ran': False}})
    state = gate.tools.read(output / 'state.json')
    state['jobs'][0]['status'] = 'complete'
    state['jobs'][1]['status'] = 'running'
    gate.tools.save(output / 'state.json', state)
    report = gate.report(output)
    assert report['paired_status_differences'] == 0
    assert not report['pairs'][0]['both_outcomes_observed']
    assert report['pairs'][0]['mixed_execution_modes'] is None


def test_owned_vitis_uses_start_identity_and_ignores_exited_snapshot_entries(monkeypatch):
    from hlsmith.toolchain import cosim_runtime_monitor as monitor
    parent = os.getpid()
    table = {parent: {'ppid': 1, 'start_ticks': 100, 'name': 'python', 'state': 'R'},
             123: {'ppid': parent, 'start_ticks': 101, 'name': 'xsim', 'state': 'R'},
             456: {'ppid': 1, 'start_ticks': 102, 'name': 'vitis_hls', 'state': 'R'}}
    monkeypatch.setattr(gate.tools, 'active_vitis', lambda: [123, 456, 789])
    monkeypatch.setattr(monitor, '_process_table', lambda: table)
    identities = {}
    assert gate._external_vitis(identities) == [456]
    # The same numeric PID later belongs to an unrelated process: no ownership.
    table[123] = {'ppid': 1, 'start_ticks': 999, 'name': 'xsim', 'state': 'R'}
    assert gate._external_vitis(identities) == [123, 456]
    table[456]['state'] = 'Z'
    assert gate._external_vitis(identities) == [123]


def test_pending_workspace_collision_fails_before_reservation(saved_registration, tmp_path, monkeypatch):
    output = saved_registration.output
    job = gate.verify(output)['jobs'][0]
    (output / 'jobs' / job['id']).mkdir(parents=True)
    settings = version_fixture(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match='workspace already exists'):
        gate.launch(Namespace(output=output, max_compute_hours=10, workers=4,
                    vitis_settings=settings, scratch_root=tmp_path / 'scratch'))
    assert gate.tools.read(output / 'state.json')['job_seconds_charged'] == 0


@pytest.mark.parametrize('workers', [0, -1, True, 1.5])
def test_worker_limit_requires_positive_integer(saved_registration, tmp_path, workers):
    with pytest.raises(ValueError, match='positive integer'):
        gate.launch(Namespace(output=saved_registration.output, max_compute_hours=10,
                    workers=workers, vitis_settings=tmp_path / 'unused', scratch_root=tmp_path / 'scratch'))


def test_waits_for_reservation_settlement_before_stopping_at_cap(saved_registration, tmp_path, monkeypatch):
    output = saved_registration.output
    jobs = {job['id']: job for job in gate.verify(output)['jobs']}
    settings = version_fixture(tmp_path, monkeypatch)
    started = []
    def run(command, **kwargs):
        identity = command[command.index('--job') + 1]
        started.append(identity)
        saved_failure(output, jobs[identity])
        return {'status': 'complete', 'returncode': 0, 'elapsed_seconds': 100 if len(started) == 1 else 4800}
    monkeypatch.setattr(gate.tools, 'run_process', run)
    report = gate.launch(Namespace(output=output, max_compute_hours=5000 / 3600, workers=2,
                    vitis_settings=settings, scratch_root=tmp_path / 'scratch'))
    assert len(started) == 2
    assert report['campaign_status'] == 'stopped_compute_cap'
    assert report['job_hours_charged'] == 4900 / 3600


def test_worker_exception_drains_siblings_and_retains_unknown_reservation(saved_registration, tmp_path, monkeypatch):
    output = saved_registration.output
    manifest = gate.verify(output)
    jobs = {job['id']: job for job in manifest['jobs']}
    settings = version_fixture(tmp_path, monkeypatch)
    barrier = threading.Barrier(2)
    completed = []
    def run(command, **kwargs):
        identity = command[command.index('--job') + 1]
        barrier.wait(timeout=5)
        if identity == manifest['jobs'][0]['id']:
            raise RuntimeError('unknown worker billing')
        saved_failure(output, jobs[identity])
        completed.append(identity)
        return {'status': 'complete', 'returncode': 0, 'elapsed_seconds': 200}
    monkeypatch.setattr(gate.tools, 'run_process', run)
    args = Namespace(output=output, max_compute_hours=10, workers=2,
                    vitis_settings=settings, scratch_root=tmp_path / 'scratch')
    with pytest.raises(RuntimeError, match='unknown worker billing'):
        gate.launch(args)
    state = gate.tools.read(output / 'state.json')
    assert completed == [manifest['jobs'][1]['id']]
    assert state['job_seconds_charged'] == 5000
    assert gate._active_job_ids(state) == [manifest['jobs'][0]['id']]
    assert state['jobs'][1]['status'] == 'complete'
    with pytest.raises(ValueError, match='interrupted cell'):
        gate.launch(args)


@pytest.mark.parametrize('cohort', ['frozen', 'clean'])
def test_absolute_controller_script_imports_pinned_evaluators(saved_registration, tmp_path, monkeypatch, cohort):
    """A new dispatcher must not preload its own numerical hlsmith package."""
    output = saved_registration.output
    clean, frozen = saved_registration.clean_source, saved_registration.frozen_source
    package = clean / 'hlsmith'
    (package / 'research').mkdir(parents=True)
    (package / 'toolchain').mkdir()
    for relative in ('__init__.py', 'research/__init__.py', 'toolchain/__init__.py'):
        (package / relative).write_text('')
    shutil.copy2(Path(gate.tools.__file__), package / 'experiments.py')
    root = Path(gate.__file__).resolve().parent
    shutil.copy2(root / 'research/cleanup_statistics.py', package / 'research/cleanup_statistics.py')
    (package / 'saved_designs.py').write_text("raise AssertionError('old dispatcher selected')\n")
    numerical = ("def _prepare_independent_golden(inputs, **kwargs):\n"
                 "    return {'success': False, 'output': '', 'provenance': {'evaluator_path': __file__}}\n")
    (package / 'pipeline.py').write_text(numerical)
    (package / 'toolchain/hls_eval.py').write_text('EVALUATOR_PATH = __file__\n')
    (clean / 'c2hls.py').write_text("import sys\nfrom hlsmith import pipeline\nsys.modules[__name__] = pipeline\n")
    (clean / 'hls_eval.py').write_text("import sys\nfrom hlsmith.toolchain import hls_eval\nsys.modules[__name__] = hls_eval\n")
    (frozen / 'c2hls.py').write_text(numerical)
    (frozen / 'hls_eval.py').write_text('EVALUATOR_PATH = __file__\n')
    for directory in (clean, frozen):
        (directory / 'golden_output.py').write_text('def compare_hlsfactory_dumps(*args, **kwargs):\n    raise AssertionError("unexpected native work")\n')
    manifest = gate.tools.read(output / 'manifest.json')
    manifest['source_files'] = {name: gate.tools.source_manifest(path) for name, path in manifest['sources'].items()}
    gate.tools.save(output / 'manifest.json', manifest)
    job = next(job for job in manifest['jobs'] if job['cohort'] == cohort)
    (output / 'jobs' / job['id']).mkdir(parents=True)
    controller_file = tmp_path / 'new-controller.py'
    shutil.copy2(gate.__file__, controller_file)
    epoch = 'a' * 32
    controller = {'path': str(controller_file), 'sha256': gate.tools.sha(controller_file),
                  'max_workers': 4, 'execution_epoch': epoch}
    settings = version_fixture(tmp_path, monkeypatch)
    versions = gate.tools.probe_vitis_versions(settings)
    limits = {'compute_seconds': 10000, 'settings': str(settings), 'settings_sha256': gate.tools.sha(settings),
              'scratch_root': str(tmp_path / 'scratch'), 'tool_versions': versions,
              'common_runner': versions['vitis-run'], 'max_workers': 4}
    gate.tools.save(output / 'launch.json', {'manifest_sha256': gate.tools.sha(output / 'manifest.json'),
                    'limits': limits, 'controller': controller})
    state = gate.tools.read(output / 'state.json')
    state.update(status='running', active_jobs=[job['id']], execution_epoch=epoch,
                 manifest_sha256=gate.tools.sha(output / 'manifest.json'), job_seconds_charged=4800)
    gate.tools.save(output / 'state.json', state)
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(('C2HLS_', 'HLSMITH_', 'OPENAI_', 'ANTHROPIC_', 'DEEPSEEK_'))}
    env.update(PYTHONPATH=str(clean), PYTHONDONTWRITEBYTECODE='1', PYTHON_DOTENV_DISABLED='1')
    command = [sys.executable, str(controller_file), 'worker', '--output', str(output),
               '--job', job['id'], '--execution-epoch', epoch]
    result = subprocess.run(command, cwd=clean, env=env, capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    measured = gate.tools.read(output / 'jobs' / job['id'] / 'measurement.json')
    assert measured['status'] == 'blocked_golden'
    selected_root = frozen if cohort == 'frozen' else clean
    assert Path(measured['golden_provenance']['evaluator_path']).is_relative_to(selected_root)
    assert measured['dispatch_controller'] == controller
    assert measured['max_workers'] == 4
    # A stale/missing epoch cannot enter the numerical evaluator.
    bad = subprocess.run(command[:-1] + ['b' * 32], cwd=clean, env=env,
                         capture_output=True, text=True, timeout=20)
    assert bad.returncode != 0 and 'dispatch epoch differs' in bad.stderr
