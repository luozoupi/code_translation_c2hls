"""Recovery invariants; synthetic campaigns only, no provider or EDA calls."""
import importlib.util
import json
import os
import sys
from pathlib import Path
import threading

import pytest

SPEC = importlib.util.spec_from_file_location('main_recovery', Path(__file__).resolve().parents[1] / 'scripts/resume_sigmetrics_recipe_main.py')
recovery = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(recovery)


@pytest.fixture
def campaign(tmp_path):
    original = tmp_path / 'original'
    original.mkdir()
    source = original / 'source'
    source.mkdir()
    frozen = source / 'frozen.txt'
    frozen.write_text('immutable evaluator')
    cells = [dict(cell_id=f'kernel__model__none__r{i}', benchmark='kernel', model='model', arm='none', repeat=i,
                  status='complete' if i < 71 else 'blocked_parent' if i < 161 else 'accounting_or_execution_failure' if i == 161 else 'pending') for i in range(216)]
    qpath = original / 'qualification/kernel/qualification.json'
    recovery.save(qpath, {'files': recovery.records([frozen])})
    prices = {'model': {'input': .2, 'output': 1.2}}
    ledger_path = tmp_path / 'ledger.json'
    reservation = dict(id='unknown', run_id='main__' + cells[161]['cell_id'], model='model', status='reserved', charged_or_reserved_usd=.025, reserved_usd=.025)
    recovery.save(ledger_path, {'limit_usd': 300, 'prices_per_million': prices, 'reservations': [reservation]})
    config = dict(schema_version='c2hls.sigmetrics-recipe-main.v1', cells=[dict(c, status='pending') for c in cells], files=recovery.records([frozen]), source=str(source), api_budget=str(ledger_path), shared_limit_usd=300, rates_per_million=prices,
                  aggregate_input_token_cap=500000, llm_call_cap=5, max_response_tokens=16384, cell_job_timeout_seconds=43200,
                  resource_thresholds={'scratch_free_gib': 60, 'memory_available_gib': 64, 'load_per_cpu': .8})
    recovery.save(original / 'manifest.json', config)
    recovery.save(original / 'launch_receipt.json', {'main_command': [sys.executable], 'main_manifest_sha256': recovery.sha(original / 'manifest.json')})
    state = dict(status='stopped_on_failure', manifest_sha256=recovery.sha(original / 'manifest.json'), cells=cells, parents=[{'benchmark': 'kernel', 'status': 'ready', 'qualification_sha256': recovery.sha(qpath)}])
    recovery.save(original / 'state.json', state)
    recovery.save(original / 'summary.json', {'status': 'stopped_on_failure'})
    for c in cells:
        if c['status'] in ('complete', 'accounting_or_execution_failure'):
            recovery.save(original / 'cells' / c['cell_id'] / 'result.json', {'preserved_outcome': c['status']})
    output = tmp_path / 'continuation'
    return original, output, config, state


def prepare(campaign):
    original, output, config, state = campaign
    recovery.prepare(original, output)
    return original, output, config, state


def test_prepare_preserves_denominator_unknown_reservation_and_snapshots(campaign):
    original, output, config, state = prepare(campaign)
    request, continuation, _ = recovery.verify(output, pristine=True)
    assert len(request['selected_cells']) == 54
    assert read_summary(output)['cells'] == {'complete': 71, 'blocked_parent': 90, 'accounting_or_execution_failure': 1, 'pending': 54}
    assert recovery.read(config['api_budget'])['reservations'][0]['status'] == 'reserved'
    assert recovery.read(original / 'state.json') == state
    assert recovery.sha(request['source_runner']) == request['source_runner_sha256']
    assert request['calendar_alias_caveat']
    assert request['python_executable'] == str(Path(sys.executable).resolve())


def read_summary(output):
    return recovery.read(output / 'summary.json')


@pytest.mark.parametrize('artifact', ['directory', 'log', 'request'])
def test_never_started_rejects_every_execution_marker(campaign, artifact):
    original, output, config, state = campaign
    c = state['cells'][162]
    if artifact == 'directory':
        (original / 'cells' / c['cell_id']).mkdir()
    elif artifact == 'log':
        (original / (c['cell_id'] + '.log')).write_text('partial')
    else:
        ledger = recovery.read(config['api_budget'])
        ledger['reservations'].append({'id': 'extra', 'run_id': 'main__' + c['cell_id'], 'status': 'settled', 'charged_or_reserved_usd': 0})
        recovery.save(config['api_budget'], ledger)
    with pytest.raises(ValueError, match='pending cell has'):
        recovery.prepare(original, output)
    assert not output.exists()


@pytest.mark.parametrize('target', ['result', 'state', 'evaluator', 'qualification'])
def test_preserved_evidence_tamper_blocks_run(campaign, target):
    original, output, config, state = prepare(campaign)
    paths = {'result': original / 'cells' / state['cells'][0]['cell_id'] / 'result.json',
             'state': original / 'state.json', 'evaluator': Path(config['source']) / 'frozen.txt',
             'qualification': original / 'qualification/kernel/qualification.json'}
    paths[target].write_text('changed')
    with pytest.raises(ValueError, match='changed'):
        recovery.verify(output)


def test_unknown_usage_cannot_be_silently_settled(campaign):
    original, output, config, state = prepare(campaign)
    ledger = recovery.read(config['api_budget'])
    ledger['reservations'][0].update(status='settled', charged_or_reserved_usd=0)
    recovery.save(config['api_budget'], ledger)
    with pytest.raises(ValueError, match='unknown usage must remain reserved'):
        recovery.verify(output)


def test_insufficient_budget_prevents_preparation(campaign):
    original, output, config, state = campaign
    ledger = recovery.read(config['api_budget'])
    ledger['reservations'][0]['charged_or_reserved_usd'] = 299
    recovery.save(config['api_budget'], ledger)
    with pytest.raises(ValueError, match='budget headroom'):
        recovery.prepare(original, output)


def test_original_controller_lock_excludes_competing_launch(campaign):
    original, output, config, state = campaign
    with recovery.original_lock(original):
        with pytest.raises(BlockingIOError):
            recovery.prepare(original, output)
    assert not output.exists()


def test_prepared_restart_guard_rejects_running_state(campaign):
    original, output, config, state = prepare(campaign)
    cont = recovery.read(output / 'state.json')
    cont['status'] = 'running'
    recovery.save(output / 'state.json', cont)
    with pytest.raises(ValueError, match='already started'):
        recovery.verify(output, pristine=True)


class FakeRunner:
    def __init__(self, original, config, fail=False):
        self.original, self.config, self.fail = original, config, fail
        self.started = []
        self.lock = threading.Lock()

    def start_ticks(self, pid):
        return '123'

    def resource_gate(self, config):
        return {'allowed': True, 'reasons': []}

    def base_environment(self, config, identity):
        return {}

    def execute(self, command, source, log, env, timeout):
        identity = command[-1]
        with self.lock:
            self.started.append(identity)
        Path(log).write_text('synthetic execution; no network or EDA calls')
        if self.fail:
            return {'returncode': 2, 'timed_out': False}
        recovery.save(self.original / 'cells' / identity / 'result.json', {
            'registration': {'spec': {'cell_id': identity}, 'manifest_sha256': recovery.sha(self.original / 'manifest.json')},
            'accounting_reconciliation': {'complete': True, 'provider_requests': 0},
        })
        return {'returncode': 0, 'timed_out': False}


def fake_run(campaign, monkeypatch, fail=False):
    original, output, config, state = prepare(campaign)
    request = recovery.read(output / 'manifest.json')
    runner = FakeRunner(original, config, fail)
    monkeypatch.setattr(recovery, '__file__', request['source_runner'])
    monkeypatch.setattr(recovery, 'load_frozen', lambda config: runner)
    monkeypatch.setattr(recovery, 'execute_owned', runner.execute)
    code = recovery.run(output)
    return original, output, config, state, runner, code


def test_continuation_finishes_only_pending_cells_preserves_original(campaign, monkeypatch):
    original, output, config, state, runner, code = fake_run(campaign, monkeypatch)
    assert code == 0
    assert len(runner.started) == len(set(runner.started)) == 54
    assert set(runner.started) == {c['cell_id'] for c in state['cells'] if c['status'] == 'pending'}
    assert recovery.read(original / 'state.json') == state
    summary = read_summary(output)
    assert summary['cells'] == {'complete': 125, 'blocked_parent': 90, 'accounting_or_execution_failure': 1}
    assert summary['registered_cells'] == 216
    assert summary['status'] == 'complete_with_quarantined_failure'
    assert recovery.read(config['api_budget'])['reservations'][0]['status'] == 'reserved'
    recovery.verify(output)
    with pytest.raises(ValueError, match='already started'):
        recovery.run(output)


def test_new_failure_halts_dispatch_and_drains_active(campaign, monkeypatch):
    original, output, config, state, runner, code = fake_run(campaign, monkeypatch, fail=True)
    assert code == 2
    assert 1 <= len(runner.started) <= 2
    assert read_summary(output)['status'] == 'stopped_on_failure'
    assert read_summary(output)['continuation_cells'].get('running', 0) == 0
    assert recovery.read(original / 'state.json') == state


def test_success_exit_without_reconciliation_is_failure(campaign):
    original, output, config, state = prepare(campaign)
    identity = state['cells'][162]['cell_id']
    recovery.save(original / 'cells' / identity / 'result.json', {
        'registration': {'spec': {'cell_id': identity}, 'manifest_sha256': recovery.sha(original / 'manifest.json')},
        'accounting_reconciliation': {'complete': False}})
    with pytest.raises(ValueError, match='without accounting'):
        recovery.result_status(original, identity, {'returncode': 0}, config)


def test_outer_timeout_kills_detached_descendants_preserves_unrelated(tmp_path):
    import subprocess
    import sys
    import time
    unrelated = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])
    child_code = 'import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(30)'
    parent_code = ('import subprocess,sys,signal,time,json; '
                   'signal.signal(signal.SIGTERM, signal.SIG_IGN); '
                   f'p=subprocess.Popen([sys.executable,"-c",{child_code!r}],start_new_session=True); '
                   'print(json.dumps({"detached_pid":p.pid}),flush=True); time.sleep(30)')
    log = tmp_path / 'outer.log'
    try:
        result = recovery.execute_owned([sys.executable, '-c', parent_code], tmp_path, log,
                                        dict(os.environ), 0.5, grace=0.1, poll=0.02)
        assert result['returncode'] == 124
        assert result['tracked_process_count'] >= 2
        assert result['cleanup_survivors'] == []
        detached = json.loads(log.read_text().splitlines()[0])['detached_pid']
        row = recovery.process_table().get(detached)
        assert row is None or row['state'] == 'Z'
        assert unrelated.poll() is None
    finally:
        unrelated.terminate()
        unrelated.wait(timeout=3)


def test_pid_reuse_never_acquires_descendants_or_signals(monkeypatch):
    row = recovery.process_table()[os.getpid()]
    stale = dict(row, start_ticks='not-current-start-ticks')
    unrelated_child = {'pid': 999999999, 'ppid': os.getpid(), 'start_ticks': '1', 'state': 'S'}
    owned = {os.getpid(): stale}
    recovery.capture_descendants(owned, {os.getpid(): row, unrelated_child['pid']: unrelated_child})
    assert unrelated_child['pid'] not in owned
    def forbidden(*args):
        pytest.fail('reused PID must not be signaled')
    monkeypatch.setattr(recovery.os, 'kill', forbidden)
    assert recovery.signal_owned(stale, 15) is False
