"""Process allowlisting must fail closed without altering frozen backfill results."""
import fcntl
import json
from pathlib import Path

import pytest

from scripts import resume_sigmetrics_cosim_coexist as coexist


def put_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def fake_process(proc, pid, parent, ticks, argv):
    folder = proc / str(pid)
    folder.mkdir(parents=True, exist_ok=True)
    fields = ['S', str(parent)] + ['0'] * 17 + [str(ticks)]
    (folder / 'stat').write_text(f'{pid} (process name) ' + ' '.join(fields))
    (folder / 'cmdline').write_bytes(b'\0'.join(arg.encode() for arg in argv) + b'\0')


@pytest.fixture
def registered_main(tmp_path):
    campaign = tmp_path / 'main'
    runner = campaign / 'source' / coexist.MAIN_RUNNER
    runner.parent.mkdir(parents=True)
    runner.write_text('# frozen main runner\n')
    put_json(campaign / 'manifest.json', {'schema_version': coexist.MAIN_SCHEMA})
    registration = {'campaign': str(campaign), 'source_runner': str(runner),
                    'manifest_sha256': coexist.sha(campaign / 'manifest.json'),
                    'source_runner_sha256': coexist.sha(runner)}
    identity = {'pid': 100, 'start_ticks': '12345',
                'manifest_sha256': registration['manifest_sha256'],
                'source_runner_sha256': registration['source_runner_sha256']}
    put_json(campaign / 'controller_identity.json', identity)
    proc = tmp_path / 'proc'
    argv = ['/usr/bin/python3', '-u', str(runner), 'run', '--output', str(campaign)]
    fake_process(proc, 100, 1, '12345', argv)
    fake_process(proc, 101, 100, '12346', ['worker'])
    fake_process(proc, 102, 101, '12347', ['vitis_hls'])
    fake_process(proc, 200, 1, '11111', ['unrelated'])
    fake_process(proc, 201, 200, '11112', ['xsim'])
    return campaign, registration, identity, proc, argv


def test_only_authenticated_descendants_are_exempted(registered_main):
    _, registration, _, proc, _ = registered_main
    allowed, reason = coexist.allowed_main_descendants(registration, [102, 201], proc)
    assert allowed == {102}
    assert reason is None


@pytest.mark.parametrize('damage', ['pid_reuse', 'missing_identity', 'wrong_manifest_hash',
                                     'wrong_source_hash', 'wrong_action', 'wrong_campaign',
                                     'argument_decoy', 'source_changed', 'manifest_changed'])
def test_invalid_registration_never_exempts_tools(registered_main, damage):
    campaign, registration, identity, proc, argv = registered_main
    if damage == 'pid_reuse':
        fake_process(proc, 100, 1, '99999', argv)
    elif damage == 'missing_identity':
        (campaign / 'controller_identity.json').unlink()
    elif damage in ('wrong_manifest_hash', 'wrong_source_hash'):
        identity['manifest_sha256' if damage == 'wrong_manifest_hash' else 'source_runner_sha256'] = 'bad'
        put_json(campaign / 'controller_identity.json', identity)
    elif damage == 'wrong_action':
        argv[3] = 'prepare'
        fake_process(proc, 100, 1, '12345', argv)
    elif damage == 'wrong_campaign':
        argv[-1] += '_other'
        fake_process(proc, 100, 1, '12345', argv)
    elif damage == 'argument_decoy':
        fake_process(proc, 100, 1, '12345', ['/bin/bash', '-c', *argv[2:]])
    elif damage == 'source_changed':
        Path(registration['source_runner']).write_text('# changed\n')
    else:
        (campaign / 'manifest.json').write_text('{}')
    allowed, reason = coexist.allowed_main_descendants(registration, [102, 201], proc)
    assert allowed == set()
    assert reason


def test_parent_cycle_is_not_a_descendant():
    assert not coexist.descendant_of(102, 100, {102: 103, 103: 102})


def test_controller_exits_during_process_scan(registered_main, monkeypatch):
    _, registration, _, proc, _ = registered_main
    monkeypatch.setattr(coexist, 'same_process', lambda *args: False)
    allowed, reason = coexist.allowed_main_descendants(registration, [102], proc)
    assert allowed == set()
    assert 'exited' in reason


@pytest.fixture
def frozen_campaign(tmp_path, registered_main):
    _, registration, _, _, _ = registered_main
    output, source, raw = tmp_path / 'backfill', tmp_path / 'frozen', tmp_path / 'raw'
    output.mkdir()
    source.mkdir()
    (source / 'worker.py').write_text('# frozen validator\n')
    put_json(raw / 'measurements' / 'timeout.json', {'cosim': {'status': 'timeout', 'kernel_runtime_cycles': None}})
    put_json(output / 'manifest.json', {'raw_root': str(raw), 'code_hashes': {'worker.py': coexist.sha(source / 'worker.py')}})
    put_json(output / 'launch_receipt.json', {'source_snapshot': str(source)})
    history = tmp_path / 'before'
    put_json(history / 'state.json', {'active': ['old'], 'pid': 300})
    old_source = tmp_path / 'old_controller.py'
    old_source.write_text('# old controller\n')
    request = {'schema_version': coexist.SCHEMA, 'scheduler_source_sha256': coexist.sha(coexist.__file__),
               'output': str(output), 'requested_workers': 4, 'timeout_seconds': 3600,
               'manifest_sha256': coexist.sha(output / 'manifest.json'), 'frozen_source': str(source),
               'completed_measurement_hashes': {'timeout.json': coexist.sha(raw / 'measurements' / 'timeout.json')},
               'old_controller_source': str(old_source), 'old_controller_source_sha256': coexist.sha(old_source),
               'before_handover_snapshot': str(history), 'before_handover_hashes': {'state.json': coexist.sha(history / 'state.json')},
               'allowed_main': registration, 'old_controller_pid': 300, 'old_controller_start_ticks': '555'}
    return output, source, raw, request


def test_preflight_is_read_only_and_preserves_timeouts(frozen_campaign):
    output, _, raw, request = frozen_campaign
    files = [p for p in output.parent.rglob('*') if p.is_file()]
    before = {p: p.read_bytes() for p in files}
    coexist.verify_campaign(output, request)
    assert before == {p: p.read_bytes() for p in files}
    assert coexist.read(raw / 'measurements' / 'timeout.json')['cosim']['status'] == 'timeout'


@pytest.mark.parametrize('damage', ['measurement', 'validator', 'manifest', 'old_controller', 'snapshot', 'wrapper_hash', 'timeout'])
def test_campaign_drift_prevents_resume(frozen_campaign, damage):
    output, source, raw, request = frozen_campaign
    paths = {'measurement': raw / 'measurements' / 'timeout.json', 'validator': source / 'worker.py',
             'manifest': output / 'manifest.json', 'old_controller': Path(request['old_controller_source']),
             'snapshot': Path(request['before_handover_snapshot']) / 'state.json'}
    if damage == 'wrapper_hash':
        request['scheduler_source_sha256'] = 'bad'
    elif damage == 'timeout':
        request['timeout_seconds'] = 7200
    else:
        paths[damage].write_text(paths[damage].read_text() + '\n')
    with pytest.raises(ValueError):
        coexist.verify_campaign(output, request)


def test_second_handover_controller_cannot_enter(frozen_campaign):
    output, _, _, request = frozen_campaign
    with (output / 'coexist_handover.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            coexist.run(output, request, output / 'request.json')


def test_unclean_drain_cannot_import_or_start_workers(frozen_campaign, monkeypatch):
    output, _, _, request = frozen_campaign
    request_path = output / 'request.json'
    put_json(request_path, request)
    put_json(output / 'state.json', {'pid': 300, 'active': ['still-running'], 'status': 'draining'})
    monkeypatch.setattr(coexist, 'same_process', lambda *args: False)
    with pytest.raises(RuntimeError, match='clean drain'):
        coexist.run(output, request, request_path)
    assert coexist.read(output / 'coexist_scheduler_status.json')['status'] == 'error'
    assert coexist.read(output / 'state.json')['active'] == ['still-running']
