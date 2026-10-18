"""Outer watchdog cleanup must cover owned detached sessions without touching peers."""
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from scripts import recover_sigmetrics_timeout_replay as recovery


def wait_for(path, timeout=3):
    deadline = time.monotonic() + timeout
    while not path.exists() and time.monotonic() < deadline:
        time.sleep(.01)
    assert path.exists(), f'child readiness missing: {path}'
    return int(path.read_text())


def live(pid):
    info = recovery._cleanup_process_info(pid)
    return info is not None and info['state'] != 'Z'


def make_worker(folder):
    folder.mkdir(parents=True)
    ready = folder / 'detached.pid'
    child = (
        'import os, signal, time; from pathlib import Path; '
        'signal.signal(signal.SIGTERM, signal.SIG_IGN); '
        f'Path({str(ready)!r}).write_text(str(os.getpid())); time.sleep(300)'
    )
    script = folder / 'worker.py'
    script.write_text(
        'import signal, subprocess, sys, time\n'
        'signal.signal(signal.SIGTERM, signal.SIG_IGN)\n'
        f'subprocess.Popen([sys.executable, "-c", {child!r}], start_new_session=True)\n'
        'time.sleep(300)\n'
    )
    return script, ready


def test_detached_descendants_are_killed_and_peer_is_untouched(tmp_path):
    script, ready = make_worker(tmp_path / 'worker')
    process = subprocess.Popen([sys.executable, str(script)], start_new_session=True)
    unrelated = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(300)'])
    tree = recovery.OwnedProcessTree(process)
    child_pid = None
    try:
        child_pid = wait_for(ready)
        tree.capture()
        assert os.getpgid(child_pid) != os.getpgid(process.pid)
        report = tree.terminate(grace_seconds=.05, kill_wait_seconds=.5)
        assert report['remaining_owned_pids'] == []
        assert process.returncode is not None
        assert not live(child_pid)
        assert unrelated.poll() is None
        assert any(x['pid'] == child_pid and x['signal'] == signal.SIGKILL for x in report['signaled'])
    finally:
        tree.terminate(grace_seconds=.01, kill_wait_seconds=.2)
        unrelated.terminate()
        unrelated.wait(timeout=3)


def test_stale_pid_identity_is_never_signaled(monkeypatch):
    sent = []
    monkeypatch.setattr(recovery.os, 'kill', lambda *args: sent.append(args))
    if hasattr(signal, 'pidfd_send_signal'):
        monkeypatch.setattr(recovery.signal, 'pidfd_send_signal', lambda *args: sent.append(args))
    info = recovery._cleanup_process_info(os.getpid())
    assert not recovery._signal_owned_identity(os.getpid(), str(int(info['start_ticks']) + 1), signal.SIGTERM)
    assert sent == []


def test_tracked_orphan_is_still_owned_after_worker_exits(tmp_path):
    ready = tmp_path / 'child.pid'
    child_code = f'import os,time; from pathlib import Path; Path({str(ready)!r}).write_text(str(os.getpid())); time.sleep(300)'
    parent_code = f'import subprocess,sys,time; subprocess.Popen([sys.executable,"-c",{child_code!r}],start_new_session=True); time.sleep(.3)'
    process = subprocess.Popen([sys.executable, '-c', parent_code], start_new_session=True)
    tree = recovery.OwnedProcessTree(process)
    try:
        child_pid = wait_for(ready)
        tree.capture()
        process.wait(timeout=3)
        assert live(child_pid)
        report = tree.terminate(grace_seconds=.1, kill_wait_seconds=.3)
        assert not live(child_pid)
        assert report['remaining_owned_pids'] == []
    finally:
        tree.terminate(grace_seconds=.01, kill_wait_seconds=.2)


def task_setup(tmp_path):
    source = tmp_path / 'source'
    script, ready = make_worker(source / 'scripts')
    script.rename(source / 'scripts' / 'run_historical_cosim.py')
    output = tmp_path / 'output'
    output.mkdir()
    task = {'id': 'new-test-' + tmp_path.name, 'historical_output': str(tmp_path / 'historical'),
            'timeout_seconds': .12, 'kernel_sha256': 'fake'}
    manifest = {'historical_source': str(source), 'outer_worker_overhead_seconds': 0,
                'observation_interval_seconds': .03}
    return source, ready, output, task, manifest


def short_cleanup(monkeypatch):
    original = recovery.OwnedProcessTree.terminate
    monkeypatch.setattr(recovery.OwnedProcessTree, 'terminate',
                        lambda self: original(self, grace_seconds=.03, kill_wait_seconds=.3))


def test_monitor_constructor_failure_cleans_worker_and_detached_child(tmp_path, monkeypatch):
    import cosim_runtime_monitor
    source, ready, output, task, manifest = task_setup(tmp_path)
    short_cleanup(monkeypatch)
    class BrokenMonitor:
        def __init__(self, *args, **kwargs):
            wait_for(ready)
            raise RuntimeError('monitor setup failed')
    monkeypatch.setattr(cosim_runtime_monitor, 'CosimRuntimeMonitor', BrokenMonitor)
    with pytest.raises(RuntimeError, match='monitor setup failed'):
        recovery.run_task(output, task, manifest)
    assert not live(int(ready.read_text()))
    report = json.loads((output / 'telemetry' / task['id'] / 'cleanup.json').read_text())
    assert report['remaining_owned_pids'] == []


def test_sampler_failure_does_not_end_validation_before_outer_deadline(tmp_path, monkeypatch):
    import cosim_runtime_monitor
    source, ready, output, task, manifest = task_setup(tmp_path)
    short_cleanup(monkeypatch)
    task['timeout_seconds'] = .5
    sampled, setup_finished, cleanup_started = [], [], []
    terminate = recovery.OwnedProcessTree.terminate
    def recorded_cleanup(self):
        cleanup_started.append(time.monotonic())
        return terminate(self)
    monkeypatch.setattr(recovery.OwnedProcessTree, 'terminate', recorded_cleanup)
    class BrokenSampler:
        def __init__(self, *args, **kwargs):
            wait_for(ready)
            setup_finished.append(time.monotonic())
        def sample(self):
            sampled.append(time.monotonic())
            raise RuntimeError('read unavailable')
        def finish(self, outcome):
            return {'outcome': outcome}
    monkeypatch.setattr(cosim_runtime_monitor, 'CosimRuntimeMonitor', BrokenSampler)
    result = recovery.run_task(output, task, manifest)
    assert len(sampled) > 1
    # The first ownership scan occurs after the deadline is established, so
    # the interval between samples omits legitimate time spent in that scan.
    # Cleanup must begin only after the full cap following monitor setup.
    assert cleanup_started[0] - setup_finished[0] >= task['timeout_seconds']
    assert result['execution']['process_timed_out']
    assert result['status'] == 'execution_error'  # fake worker never creates a result
    assert not live(int(ready.read_text()))
    assert result['execution']['owned_process_cleanup']['remaining_owned_pids'] == []
