"""No API or Vitis calls: registration, limits, identities and owned cleanup."""
from argparse import Namespace
from collections import Counter
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from hlsmith import experiments as study


@pytest.fixture(autouse=True)
def restore_worker_environment():
    original = os.environ.copy()
    yield
    os.environ.clear()
    os.environ.update(original)


def authorize_test_worker(output, job):
    """Create a bounded launch fixture; it never launches provider/tool work."""
    state = study.read(output / 'state.json')
    state.update(status='running', active_job=job['id'])
    study.save(output / 'state.json', state)
    settings = output / 'fake-settings.sh'
    settings.write_text('# test fixture only\n')
    limits = {'api_usd': 1, 'compute_seconds': 1000000, 'max_input_tokens_per_cell': 500000,
              'scratch_root': str(output / 'scratch'), 'vitis_settings': str(settings),
              'vitis_settings_sha256': study.sha(settings)}
    study.save(output / 'launch.json', {'manifest_sha256': study.sha(output / 'manifest.json'), 'limits': limits})
    study.save(output / 'api_budget.json', {'limit_usd': 1, 'reservations': []})


def test_full_matrix_has_all_cells_and_rotates_cohorts():
    jobs = study.make_jobs()
    assert len(jobs) == len({j['id'] for j in jobs}) == 1638
    assert Counter(j['cohort'] for j in jobs) == {c['id']: 546 for c in study.COHORTS}
    assert Counter(j['repeat'] for j in jobs) == {1: 546, 2: 546, 3: 546}
    assert Counter(j['arm'] for j in jobs) == {a['id']: 234 for a in study.ARMS}
    for start in range(0, len(jobs), 3):
        group = jobs[start:start + 3]
        assert len({(j['benchmark'], j['arm'], j['repeat']) for j in group}) == 1
        assert {j['cohort'] for j in group} == {c['id'] for c in study.COHORTS}
    assert jobs[0]['cohort'] != jobs[3]['cohort']


def test_original_flash_entrypoint_and_arm_limits_are_preserved():
    for arm in study.ARMS:
        assert arm['entrypoint'] == ('run_benchmark_multistep' if arm['id'].startswith('ms_') else 'run_benchmark')
        assert arm['turns'] == (1 if arm['id'] == 'oneshot' else 5)
        assert arm['quality_repair_turns'] == (0 if arm['id'] == 'oneshot' else 4)
        assert arm['generation_timeout_seconds'] == (7200 if arm['id'] == 'oneshot' else 14400)


@pytest.fixture
def registration(tmp_path):
    frozen, clean = tmp_path / 'v6', tmp_path / 'clean'
    for root in (frozen, clean):
        root.mkdir()
        (root / 'c2hls.py').write_text('# synthetic source\n')
    (clean / 'hlsmith' / 'prompts').mkdir(parents=True)
    (clean / 'hlsmith' / 'prompts' / 'researcher_v1.py').write_text('# synthetic profile\n')
    inputs, registry = tmp_path / 'benchmarks', {}
    for name in study.BENCHMARKS:
        directory = inputs / name
        directory.mkdir(parents=True)
        (directory / 'plain.cpp').write_text('int workload(int x) { return x; }\n')
        (directory / 'kernel.h').write_text('int workload(int);\n')
        (directory / 'testbench.cpp').write_text('int main() { return 0; }\n')
        (directory / 'gold_hls_source.cpp').write_text('private expert source\n')
        study.save(directory / 'metadata.json', {'benchmark': name, 'source_repo': 'HLSFactory',
                   'plain_c_file': 'plain.cpp', 'header_file': 'kernel.h', 'testbench_file': 'testbench.cpp',
                   'cosim_testbench_file': 'testbench_cosim.cpp', 'cosim_size_overrides': {'N': 8},
                   'gold_hls_source_file': 'gold_hls_source.cpp', 'supports_cosim': True})
        registry[name] = {'testbench_sha256': study.sha(directory / 'testbench.cpp'),
                          'outputs': {'A': {'shape': [1], 'logical_shape': [1], 'kind': 'float'}}}
    shape_path, skill_path = tmp_path / 'shapes.json', tmp_path / 'skills.json'
    study.save(shape_path, {'schema_version': 'c2hls.hlsfactory-output-shapes.v1', 'benchmarks': registry})
    study.save(skill_path, {'skills': []})
    payload = tmp_path / 'historical'
    for model in ('sonnet', 'opus', 'gpt55'):
        for arm in study.ARMS:
            study.save(payload / 'results' / f'{model}_{arm["id"]}' / 'matrix.json',
                       [{'wallclock_s': 10, 'summary': {'llm_usage': {'input_tokens': 100, 'output_tokens': 20}}}
                        for name in study.BENCHMARKS])
    output = tmp_path / 'registration'
    study.register(Namespace(output=output, frozen_source=frozen, clean_source=clean,
                   benchmark_root=inputs, skill_library=skill_path, shape_registry=shape_path,
                   historical_payload=payload))
    return output


def test_registration_needs_no_credentials_or_tools_and_stages_public_inputs(registration, monkeypatch):
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    monkeypatch.delenv('DEEPSEEK_API_KEY', raising=False)
    manifest = study.verify(registration)
    assert manifest['provider']['model'] == 'deepseek-flash'
    assert manifest['provider']['declared_release'] == 'DeepSeek-V4.1-Flash-1017'
    assert not manifest['provider']['immutable_weights_claim']
    assert not (registration / 'api_budget.json').exists()
    assert study.preflight(registration)['complete']
    for entry in manifest['benchmark_entries']:
        root = Path(entry['path'])
        assert not (root / 'gold_hls_source.cpp').exists()
        meta = study.read(root / 'metadata.json')
        assert meta['supports_cosim'] is False
        assert 'cosim_size_overrides' not in meta
        assert 'cosim_testbench_file' not in meta


def test_estimate_uses_entire_historical_denominator(registration):
    estimate = study.verify(registration)['estimates']
    assert estimate['input_tokens'] == 163800
    assert estimate['output_tokens'] == 32760
    assert estimate['generation_job_hours'] == pytest.approx(16380 / 3600)
    assert estimate['api_estimate_usd_without_cache_credit'] == pytest.approx(.088452)


def test_report_keeps_pending_cells_and_never_infers_a_measured_win(registration):
    report = study.report(registration)
    assert report['registered_cells'] == 1638
    assert report['status_counts'] == {'pending': 1638}
    assert report['paired_paper_denominator'] == 546
    assert report['paired_rtl_passes'] == 0
    assert report['paired_rtl_geomean_speedup'] is None
    uncertainty = report['uncertainty']
    assert uncertainty['conclusion'] == 'inconclusive_incomplete_study'
    assert uncertainty['bootstrap']['registered_kernel_clusters'] == 26
    assert all(cohort['registered_cells'] == cohort['unobserved_cells'] == 546
               for cohort in uncertainty['cohorts'].values())
    assert all(pair['validation_pass_difference_ci95'] is None
               and pair['geomean_rtl_cycle_ratio_ci95'] is None
               for pair in uncertainty['comparisons'])


def test_public_cli_registration_estimate_preflight_and_report(registration, monkeypatch, capsys):
    """Exercise real CLI dispatch and files; substitute only version queries."""
    root = registration.parent
    output = root / 'cli-registration'
    arguments = ['register', '--output', str(output)]
    for flag, path in (
        ('frozen-source', root / 'v6'), ('clean-source', root / 'clean'),
        ('benchmark-root', root / 'benchmarks'), ('skill-library', root / 'skills.json'),
        ('shape-registry', root / 'shapes.json'), ('historical-payload', root / 'historical'),
    ):
        arguments.extend(['--' + flag, str(path)])
    capsys.readouterr()
    assert study.main(arguments) == 0
    registered = json.loads(capsys.readouterr().out)
    assert registered['registered'] == str(output)
    assert registered['jobs'] == 1638

    assert study.main(['estimate', '--output', str(output)]) == 0
    estimate = json.loads(capsys.readouterr().out)
    assert estimate['jobs'] == 1638
    assert estimate['estimates']['historical_rows'] == 546

    settings = root / 'version-only-settings.sh'
    settings.write_text('# version probe fixture; no Vitis jobs\n')
    probed = []
    def versions(path):
        probed.append(path)
        return {name: {'status': 'verified', 'observed_version': '2023.2',
                       'attempts': [{'status': 'verified'}]}
                for name in ('vitis-run', 'vitis_hls')}
    monkeypatch.setattr(study, 'probe_vitis_versions', versions)
    assert study.main(['preflight', '--output', str(output), '--vitis-settings', str(settings)]) == 0
    preflight = json.loads(capsys.readouterr().out)
    assert probed == [settings]
    assert preflight['complete']
    assert preflight['vitis_version_queries'] == 2
    assert preflight['provider_calls'] == preflight['vitis_job_calls'] == 0

    assert study.main(['report', '--output', str(output)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report['registered_cells'] == 1638
    assert report['status_counts'] == {'pending': 1638}
    assert report['paired_paper_denominator'] == report['researcher_paired_denominator'] == 546
    assert report['paired_rtl_passes'] == report['researcher_paired_rtl_passes'] == 0
    assert report['api_charged_or_reserved_usd'] == report['job_hours_used'] == 0
    assert study.read(output / 'report.json') == report
    assert all((output / name).is_file() for name in ('cells.csv', 'paired_paper.csv', 'paired_researcher.csv'))
    assert not (output / 'launch.json').exists()
    assert not (output / 'api_budget.json').exists()
    assert not (output / 'jobs').exists()


def test_changed_source_or_registered_input_fails_verification(registration):
    manifest = study.verify(registration)
    (Path(manifest['sources']['clean']) / 'c2hls.py').write_text('# changed\n')
    with pytest.raises(ValueError, match='registered input changed'):
        study.verify(registration)


def test_environment_uses_explicit_provider_and_rejects_inherited_controls(registration, monkeypatch):
    monkeypatch.setenv('C2HLS_TRANSLATOR_MODEL', 'gpt-wrong-provider')
    monkeypatch.setenv('OPENAI_BASE_URL', 'https://wrong.example')
    monkeypatch.setenv('HLS_TCL_RUNNER', 'unregistered_runner')
    monkeypatch.setenv('HLSMITH_OUTPUT_ROOT', '/unregistered/output')
    manifest = study.verify(registration)
    limits = {'scratch_root': '/tmp/scratch', 'max_input_tokens_per_cell': 100000, 'vitis_settings': '/tools/settings64.sh'}
    env = study.cell_environment(manifest, manifest['jobs'][0], limits, registration)
    assert env['OPENAI_BASE_URL'] == 'https://api.deepseek.com'
    assert env['C2HLS_DEEPSEEK_THINKING'] == 'disabled'
    assert 'C2HLS_LLM_SEED' not in env
    assert 'C2HLS_TRANSLATOR_MODEL' not in env
    assert 'C2HLS_LLM_CANDIDATE_BUDGET' not in env
    assert 'C2HLS_SYNTHESIS_EVAL_BUDGET' not in env
    assert env['C2HLS_COSIM_REQUIRED'] == '0'
    assert env['C2HLS_FORCE_SELECTED_COSIM'] == '0'
    assert env['HLS_TCL_RUNNER'] == 'vitis-run --tcl --input_file'
    assert env['PYTHON_DOTENV_DISABLED'] == '1'
    assert 'HLSMITH_OUTPUT_ROOT' not in env


def test_all_cohorts_use_same_verified_vitis_runner(registration):
    manifest = study.verify(registration)
    binary = '/verified Vitis/2023.2/bin/vitis-run'
    identity = {'binary': binary, 'binary_sha256': 'fixture-identity', 'observed_version': '2023.2',
                'arguments': ['--tcl', '--input_file']}
    limits = {'scratch_root': '/tmp/scratch', 'max_input_tokens_per_cell': 100000,
              'vitis_settings': '/tools/settings64.sh', 'common_runner': identity}
    for job in manifest['jobs'][:3]:
        env = study.cell_environment(manifest, job, limits, registration)
        assert env['HLS_TCL_RUNNER'] == "'/verified Vitis/2023.2/bin/vitis-run' --tcl --input_file"
        assert env['PATH'].split(os.pathsep)[0] == str(Path(binary).parent)


@pytest.mark.parametrize('value', [0, -1, float('inf'), float('nan'), True, False])
def test_invalid_launch_limits_rejected(value):
    with pytest.raises(ValueError):
        study.positive(value)


def test_paid_launch_requires_all_numeric_caps():
    with pytest.raises(SystemExit) as result:
        study.main(['launch', '--output', '/tmp/registration'])
    assert result.value.code == 2


def test_private_worker_cannot_bypass_launch_registration(registration):
    with pytest.raises(FileNotFoundError):
        study.worker(Namespace(output=registration, job=study.make_jobs()[0]['id'], stage='all'))


def test_known_generation_timeout_is_preserved_without_reselection(registration, monkeypatch):
    job = study.make_jobs()[0]
    authorize_test_worker(registration, job)
    monkeypatch.setenv('DEEPSEEK_API_KEY', 'unit-test-placeholder-never-used')
    folder = registration / 'jobs' / job['id']
    folder.mkdir(parents=True)
    calls = []
    def timeout(*args, **kwargs):
        calls.append(args)
        return {'status': 'timeout', 'returncode': None, 'elapsed_seconds': 1}
    monkeypatch.setattr(study, 'run_process', timeout)
    monkeypatch.setattr(study.signal, 'signal', lambda *args: None)
    study.worker(Namespace(output=registration, job=job['id'], stage='all'))
    assert len(calls) == 1
    assert study.read(folder / 'measurement.json')['status'] == 'generation_timeout'


def test_pid_reuse_does_not_grant_ownership():
    owned = {10: 100}
    table = {10: {'start_ticks': 999, 'ppid': 1}, 11: {'start_ticks': 1, 'ppid': 10}}
    assert not study.owned_processes(10, owned, table)
    assert owned == {10: 100}


def test_owned_cleanup_reaches_detached_children_without_signaling_peer(tmp_path):
    child_file = tmp_path / 'child'
    code = ('import subprocess,sys,time; from pathlib import Path; '
            "p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(20)'],start_new_session=True); "
            f'Path({str(child_file)!r}).write_text(str(p.pid)); time.sleep(20)')
    peer = subprocess.Popen([sys.executable, '-c', 'import time;time.sleep(20)'])
    try:
        result = study.run_process([sys.executable, '-c', code], cwd=tmp_path, env=os.environ.copy(),
                                   timeout=1, log=tmp_path / 'job.log')
        assert result['status'] == 'timeout'
        assert peer.poll() is None
        child = int(child_file.read_text())
        stat = Path('/proc') / str(child) / 'stat'
        assert not stat.exists() or stat.read_text().rsplit(')', 1)[1].split()[0] == 'Z'
    finally:
        peer.terminate()
        peer.wait()


def test_state_cannot_silently_truncate_or_reorder_registered_jobs(registration):
    state = study.read(registration / 'state.json')
    state['jobs'].pop()
    study.save(registration / 'state.json', state)
    with pytest.raises(ValueError, match='state job identities'):
        study.verify(registration)


def test_new_source_module_is_not_silently_outside_pinned_source_set(registration):
    manifest = study.verify(registration)
    (Path(manifest['sources']['clean']) / 'new_runtime.py').write_text('# new implementation\n')
    with pytest.raises(ValueError, match='registered input changed'):
        study.verify(registration)


def test_worker_rechecks_source_identity_before_any_stage(registration):
    manifest = study.verify(registration)
    (Path(manifest['sources']['frozen']) / 'c2hls.py').write_text('# changed after launch preflight\n')
    with pytest.raises(ValueError, match='registered input changed'):
        study.worker(Namespace(output=registration, job=manifest['jobs'][0]['id'], stage='all'))


@pytest.mark.parametrize('value', [True, 0, -1, 1.5, float('inf'), float('nan'), '123', None])
def test_rtl_cycles_require_executed_positive_finite_integers(value):
    assert study.measured_cycles({'kernel_runtime_cycles': value}) is None


def test_independent_pass_requires_actual_comparator_evidence():
    assert not study.independently_passed({'success': True, 'passed': True})
    assert not study.independently_passed({'success': True, 'passed': True, 'correctness': {'passed': False}})
    passed = {'success': True, 'passed': True, 'correctness': {'passed': True}, 'golden_output_sha256': 'oracle-a'}
    assert study.independently_passed(passed, 'oracle-a')
    assert not study.independently_passed(passed, 'oracle-b')


@pytest.fixture
def validation_case(registration, monkeypatch):
    """Exercise the real validation controller with archived synthesis evidence."""
    import c2hls
    import hls_eval
    root = Path(__file__).resolve().parents[1]
    archived = study.read(root / 'tests/fixtures/v6_replay/snapshot.json')['cases'][0]
    manifest = study.verify(registration)
    manifest['sources']['clean'] = str(root)
    job = manifest['jobs'][0]
    authorize_test_worker(registration, job)
    folder = registration / 'jobs' / job['id']
    folder.mkdir(parents=True)
    selected = folder / 'selected.cpp'
    selected.write_text(archived['kernel'])
    selection = {'source_sha256': study.sha(selected), 'synthesis_report': archived['report'],
                 'synthesis_report_sha256': study.content_sha(archived['report'])}
    study.save(folder / 'selection.json', selection)
    dump = '==BEGIN DUMP_ARRAYS==\nbegin dump: A\n1.0 2.0\nend dump: A\n==END DUMP_ARRAYS==\n'
    import hashlib
    oracle_sha = hashlib.sha256(dump.encode()).hexdigest()
    golden = {'success': True, 'output': dump, 'specs': {'A': {'shape': [2], 'kind': 'float'}},
              'provenance': {'required': True, 'status': 'passed', 'output_sha256': oracle_sha}}
    monkeypatch.setattr(study, 'verify', lambda output: manifest)
    monkeypatch.setattr(c2hls, '_load_benchmark_inputs', lambda path: {
        'header_code': archived['header'], 'header_name': archived['header_name'], 'extra_files': [],
        'testbench_code': archived['testbench'], 'meta': archived['metadata']})
    monkeypatch.setattr(c2hls, '_prepare_independent_golden', lambda inputs: golden)
    monkeypatch.setattr(hls_eval, 'run_native_testbench', lambda *args, **kwargs: {'success': True, 'passed': True, 'output': dump})
    calls = []
    passed = {'success': True, 'passed': True, 'correctness': {'passed': True}, 'golden_output_sha256': oracle_sha}
    def csim(*args, **kwargs):
        calls.append('csim')
        return dict(passed)
    def cosim(*args, **kwargs):
        calls.append('rtl')
        return {**passed, 'kernel_runtime_cycles': 1234}
    monkeypatch.setattr(hls_eval, 'run_csim', csim)
    monkeypatch.setattr(hls_eval, 'run_cosim', cosim)
    monkeypatch.setattr(study, 'fresh_rtl_synthesis', lambda evaluator, rtl: (archived['report'], {'status': 'recorded'}))
    return Namespace(output=registration, job=job['id'], stage='validate'), folder, selection, calls


def test_archived_report_and_independent_cpu_csim_rtl_qualify_passing_candidate(validation_case):
    args, folder, _, calls = validation_case
    study.worker(args)
    record = study.read(folder / 'measurement.json')
    assert calls == ['csim', 'rtl']
    assert record['status'] == 'passed'
    assert record['feasibility']['feasible']
    assert record['rtl_feasibility']['feasible']


@pytest.mark.parametrize('change', [
    {'slack_ns': -0.1}, {'dsp': 10**9}, {'uram': None},
    {'requested_clock_period_ns': 10}, {'latency_cycles_worst': True},
])
def test_timing_resource_and_report_evidence_fail_before_tool_dispatch(validation_case, change):
    args, folder, selection, calls = validation_case
    selection['synthesis_report'] = {**selection['synthesis_report'], **change}
    selection['synthesis_report_sha256'] = study.content_sha(selection['synthesis_report'])
    study.save(folder / 'selection.json', selection)
    study.worker(args)
    assert not calls
    assert study.read(folder / 'measurement.json')['status'] == 'failed_feasibility'


def test_fresh_rtl_synthesis_must_also_satisfy_target(validation_case, monkeypatch):
    args, folder, selection, calls = validation_case
    monkeypatch.setattr(study, 'fresh_rtl_synthesis', lambda evaluator, rtl: (
        {**selection['synthesis_report'], 'slack_ns': -0.1}, {'status': 'recorded'}))
    study.worker(args)
    assert calls == ['csim', 'rtl']
    assert study.read(folder / 'measurement.json')['status'] == 'unqualified_rtl_synthesis'


def test_selected_report_tampering_is_rejected_without_reselecting(validation_case):
    args, folder, selection, calls = validation_case
    selection['synthesis_report']['slack_ns'] = 999
    study.save(folder / 'selection.json', selection)
    with pytest.raises(ValueError, match='selected synthesis report changed'):
        study.worker(args)
    assert not calls


def test_unrecorded_fresh_synthesis_report_is_missing_evidence(tmp_path):
    assert study.fresh_rtl_synthesis(None, {'work_dir': str(tmp_path)})[1]['status'] == 'missing'


def test_report_rejects_a_bare_tool_pass_without_required_evidence(registration):
    manifest = study.verify(registration)
    job = manifest['jobs'][0]
    study.save(registration / 'jobs' / job['id'] / 'measurement.json', {
        'job': job, 'status': 'passed', 'rtl': {'success': True, 'passed': True, 'kernel_runtime_cycles': 123}})
    with pytest.raises(ValueError, match='lacks independent/timing/resource evidence'):
        study.report(registration)


def test_report_tracks_researcher_profile_pairs_with_full_denominator(registration):
    import hashlib
    manifest = study.verify(registration)
    oracle_sha = hashlib.sha256(b'oracle').hexdigest()
    passed = {'success': True, 'passed': True, 'correctness': {'passed': True}, 'golden_output_sha256': oracle_sha}
    for job, cycles in zip(manifest['jobs'][:3], (400, 200, 100)):
        folder = registration / 'jobs' / job['id']
        folder.mkdir(parents=True)
        (folder / 'selected.cpp').write_text('void workload() {}')
        study.save(folder / 'measurement.json', {
            'job': job, 'status': 'passed', 'selected_source_sha256': study.sha(folder / 'selected.cpp'),
            'native_comparison': {'passed': True}, 'feasibility': {'feasible': True}, 'rtl_feasibility': {'feasible': True},
            'golden_output_sha256': oracle_sha, 'golden_provenance': {'status': 'passed', 'output_sha256': oracle_sha},
            'csim': passed, 'rtl': {**passed, 'kernel_runtime_cycles': cycles}})
    report = study.report(registration)
    assert report['paired_paper_denominator'] == report['researcher_paired_denominator'] == 546
    assert report['paired_rtl_passes'] == report['researcher_paired_rtl_passes'] == 1
    assert report['paired_rtl_geomean_speedup'] == report['researcher_paired_rtl_geomean_speedup'] == 2
    assert report['complete_paper_reproduction'] is False
    assert (registration / 'paired_researcher.csv').is_file()


def test_private_worker_requires_numeric_launch_receipt_even_with_active_job(registration):
    job = study.make_jobs()[0]
    state = study.read(registration / 'state.json')
    state.update(status='running', active_job=job['id'])
    study.save(registration / 'state.json', state)
    study.save(registration / 'launch.json', {'manifest_sha256': study.sha(registration / 'manifest.json')})
    with pytest.raises(ValueError, match='explicit registered API, compute and input limits'):
        study.worker(Namespace(output=registration, job=job['id'], stage='all'))


def test_preflight_rejects_hash_correct_but_empty_shape_registry(registration):
    manifest = study.verify(registration)
    registry = study.read(manifest['shape_registry'])
    registry['benchmarks'][study.BENCHMARKS[0]]['outputs'] = {}
    study.save(manifest['shape_registry'], registry)
    manifest['shape_registry_sha256'] = study.sha(manifest['shape_registry'])
    study.save(registration / 'manifest.json', manifest)
    state = study.read(registration / 'state.json')
    state['manifest_sha256'] = study.sha(registration / 'manifest.json')
    study.save(registration / 'state.json', state)
    report = study.preflight(registration)
    assert not report['complete']
    assert any('empty authoritative output contract' in issue for issue in report['issues'])


@pytest.mark.parametrize('version', ['2023.2', '2024.1'])
def test_bounded_probe_uses_reported_tool_version_not_installation_path(tmp_path, monkeypatch, version):
    binary = tmp_path / 'tool'
    binary.write_text('# version-only fixture\n')
    settings = tmp_path / 'settings64.sh'
    settings.write_text('# fixture\n')
    calls = []
    def probe(command, **kwargs):
        assert kwargs['timeout'] == 10
        assert Path(kwargs['cwd']).is_dir()
        calls.append(command)
        label = 'vitis-run' if command[-3] == 'vitis-run' else 'Vitis HLS'
        return subprocess.CompletedProcess(command, 0, f'{binary}\n{label} v{version}\n', '/installation/2023.2/label-only')
    monkeypatch.setattr(study.subprocess, 'run', probe)
    result = study.probe_vitis_versions(settings)
    assert len(calls) == (2 if version == '2023.2' else 4)
    for tool in result.values():
        assert tool['observed_version'] == version
        assert tool['status'] == ('verified' if version == '2023.2' else 'unverified')


def test_version_mismatch_blocks_preflight_without_running_a_job(registration, monkeypatch):
    settings = registration / 'settings64.sh'
    settings.write_text('# no tools launched\n')
    monkeypatch.setattr(study, 'probe_vitis_versions', lambda path: {
        name: {'status': 'unverified', 'observed_version': '2024.1', 'attempts': []}
        for name in ('vitis-run', 'vitis_hls')})
    report = study.preflight(registration, settings)
    assert not report['complete']
    assert report['vitis_job_calls'] == report['provider_calls'] == 0
    assert sum('version not verified' in issue for issue in report['issues']) == 2
