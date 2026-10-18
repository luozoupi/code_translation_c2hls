#!/usr/bin/env python3
"""Resume the frozen backfill while allowing one authenticated main-study controller.

This wrapper never signals an existing process. An operator first prepares the
request, asks the old controller to drain, and then runs this wrapper. Validation
code, saved measurements, worker count, and the 3600-second timeout are preserved.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


SCHEMA = 'c2hls.sigmetrics-cosim-coexist.v1'
MAIN_SCHEMA = 'c2hls.sigmetrics-recipe-main.v1'
MAIN_RUNNER = 'scripts/run_sigmetrics_recipe_main.py'


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    temp.replace(path)


def process_info(pid, proc_root=Path('/proc')):
    """Return stable PID identity, argv, and parent without reading environment."""
    root = Path(proc_root) / str(pid)
    try:
        fields = (root / 'stat').read_text().rsplit(')', 1)[1].split()
        argv = (root / 'cmdline').read_bytes().rstrip(b'\0').split(b'\0')
        return {'pid': int(pid), 'start_ticks': fields[19], 'ppid': int(fields[1]),
                'argv': [arg.decode(errors='replace') for arg in argv]}
    except (FileNotFoundError, ProcessLookupError):
        return None


def same_process(pid, ticks, proc_root=Path('/proc')):
    info = process_info(pid, proc_root)
    return info is not None and info['start_ticks'] == str(ticks)


def verify_campaign(output, request):
    if sha(__file__) != request['scheduler_source_sha256']:
        raise ValueError('coexistence scheduler source changed')
    if request['schema_version'] != SCHEMA:
        raise ValueError('unknown coexistence request schema')
    if Path(request['output']).resolve() != output.resolve():
        raise ValueError('backfill campaign differs from the request')
    if request['requested_workers'] != 4 or request['timeout_seconds'] != 3600:
        raise ValueError('backfill worker count or timeout changed')
    if sha(output / 'manifest.json') != request['manifest_sha256']:
        raise ValueError('backfill manifest changed')
    manifest = read(output / 'manifest.json')
    source = Path(read(output / 'launch_receipt.json')['source_snapshot']).resolve()
    if str(source) != request['frozen_source']:
        raise ValueError('frozen source path changed')
    for name, expected in manifest['code_hashes'].items():
        if sha(source / name) != expected:
            raise ValueError(f'frozen validator changed: {name}')
    raw = Path(manifest['raw_root'])
    for name, expected in request['completed_measurement_hashes'].items():
        if sha(raw / 'measurements' / name) != expected:
            raise ValueError(f'completed measurement changed: {name}')
    if sha(request['old_controller_source']) != request['old_controller_source_sha256']:
        raise ValueError('old controller source changed')
    for name, expected in request['before_handover_hashes'].items():
        if sha(Path(request['before_handover_snapshot']) / name) != expected:
            raise ValueError(f'before-handover snapshot changed: {name}')
    verify_main_registration(request['allowed_main'])
    return manifest, source


def verify_main_registration(registration):
    campaign = Path(registration['campaign']).resolve()
    runner = (campaign / 'source' / MAIN_RUNNER).resolve()
    if str(runner) != registration['source_runner']:
        raise ValueError('registered main runner path differs')
    manifest = campaign / 'manifest.json'
    if read(manifest).get('schema_version') != MAIN_SCHEMA:
        raise ValueError('unexpected main-study schema')
    if sha(manifest) != registration['manifest_sha256']:
        raise ValueError('registered main manifest changed')
    if sha(runner) != registration['source_runner_sha256']:
        raise ValueError('registered main runner changed')
    return campaign, runner


def authenticate_main(registration, proc_root=Path('/proc')):
    """Missing/dead/invalid controllers receive no process-tree exemption."""
    try:
        campaign, runner = verify_main_registration(registration)
        identity = read(campaign / 'controller_identity.json')
        for field in ('manifest_sha256', 'source_runner_sha256'):
            if identity[field] != registration[field]:
                raise ValueError(f'main controller {field} mismatch')
        info = process_info(int(identity['pid']), proc_root)
        if info is None or info['start_ticks'] != str(identity['start_ticks']):
            return None, 'registered main controller is not alive with the recorded identity'
        argv = info['argv']
        # The registered Python file must be the executable script, not a string
        # supplied as an unrelated program's argument.
        script_index = 2 if len(argv) > 1 and argv[1] == '-u' else 1
        if len(argv) <= script_index or argv[script_index] != str(runner):
            raise ValueError('main controller is not executing the registered frozen runner')
        tail = argv[script_index + 1:]
        if not tail or tail[0] != 'run':
            raise ValueError('main controller is not running the registered campaign')
        for flag, value in (('--output', str(campaign)),):
            if tail.count(flag) != 1 or tail.index(flag) + 1 >= len(tail) or tail[tail.index(flag) + 1] != value:
                raise ValueError(f'main controller command has no exact {flag} {value}')
        return info, None
    except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
        return None, f'{type(exc).__name__}: {exc}'


def descendant_of(pid, ancestor, parents):
    seen = set()
    while pid in parents and pid not in seen and pid != ancestor:
        seen.add(pid)
        pid = parents[pid]
    return pid == ancestor


def allowed_main_descendants(registration, candidates, proc_root=Path('/proc')):
    info, reason = authenticate_main(registration, proc_root)
    if info is None:
        return set(), reason
    parents = {}
    for path in Path(proc_root).iterdir():
        if path.name.isdigit():
            try:
                fields = (path / 'stat').read_text().rsplit(')', 1)[1].split()
                parents[int(path.name)] = int(fields[1])
            except (OSError, IndexError, ValueError):
                continue
    # Recheck after the process-table read to fail closed on controller exit/PID reuse.
    if not same_process(info['pid'], info['start_ticks'], proc_root):
        return set(), 'registered main controller exited during process-tree inspection'
    allowed = {pid for pid in candidates if descendant_of(pid, info['pid'], parents)}
    return allowed, None


def prepare(output, request_path, campaign):
    if request_path.exists():
        raise FileExistsError('change request already exists; use a new request path')
    status = read(output / 'parallel_scheduler_status.json')
    state = read(output / 'state.json')
    if status['workers'] != 4 or status['timeout_seconds'] != 3600:
        raise ValueError('expected existing four-worker 3600-second scheduler')
    info = process_info(status['pid'])
    if info is None or state['pid'] != info['pid']:
        raise ValueError('old controller identity could not be authenticated')
    old_source = Path(status['scheduler_source']).resolve()
    if str(old_source) not in info['argv'] or sha(old_source) != status['scheduler_source_sha256']:
        raise ValueError('old controller command or source does not match its status')
    manifest = read(output / 'manifest.json')
    source = Path(read(output / 'launch_receipt.json')['source_snapshot']).resolve()
    campaign = campaign.resolve()
    runner = (campaign / 'source' / MAIN_RUNNER).resolve()
    registration = {'campaign': str(campaign), 'source_runner': str(runner),
                    'manifest_sha256': sha(campaign / 'manifest.json'),
                    'source_runner_sha256': sha(runner)}
    verify_main_registration(registration)
    request_path.parent.mkdir(parents=True, exist_ok=True)
    history = request_path.parent / ('before_handover_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    history.mkdir(exist_ok=False)
    for name in ('state.json', 'summary.json', 'parallel_scheduler_status.json', 'launch_receipt.json'):
        (history / name).write_bytes((output / name).read_bytes())
    request = {'schema_version': SCHEMA, 'created_at': now(), 'output': str(output),
               'scheduler_source_sha256': sha(__file__),
               'requested_workers': 4, 'timeout_seconds': 3600,
               'manifest_sha256': sha(output / 'manifest.json'), 'frozen_source': str(source),
               'old_controller_pid': info['pid'], 'old_controller_start_ticks': info['start_ticks'],
               'old_controller_argv': info['argv'], 'old_controller_source': str(old_source),
               'old_controller_source_sha256': sha(old_source),
               'before_handover_snapshot': str(history),
               'before_handover_hashes': {p.name: sha(p) for p in sorted(history.iterdir())},
               'completed_measurement_hashes': {p.name: sha(p) for p in sorted((Path(manifest['raw_root']) / 'measurements').glob('*.json'))},
               'allowed_main': registration}
    verify_campaign(output, request)
    write(request_path, request)
    return request


def run(output, request, request_path):
    # A separate handover lock prevents two waiting wrappers from racing. The
    # frozen runner additionally owns queue.lock for its complete invocation.
    lock = (output / 'coexist_handover.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    manifest, source = verify_campaign(output, request)
    status_path = output / 'coexist_scheduler_status.json'
    status = {'started_at': now(), 'pid': os.getpid(),
              'start_ticks': process_info(os.getpid())['start_ticks'], 'workers': 4,
              'timeout_seconds': 3600, 'status': 'waiting_for_previous_controller',
              'scheduler_source': str(Path(__file__).resolve()), 'scheduler_source_sha256': sha(__file__),
              'change_request': str(request_path), 'change_request_sha256': sha(request_path),
              'previous_controller_pid': request['old_controller_pid'], 'allowed_main': request['allowed_main']}
    write(status_path, status)
    cancelled = False
    def stop_waiting(signum, frame):
        nonlocal cancelled
        cancelled = True
    signal.signal(signal.SIGTERM, stop_waiting)
    signal.signal(signal.SIGINT, stop_waiting)
    try:
        deadline = time.monotonic() + 4800
        while same_process(request['old_controller_pid'], request['old_controller_start_ticks']):
            if cancelled:
                status.update(status='cancelled_before_resume', finished_at=now())
                write(status_path, status)
                return
            if time.monotonic() >= deadline:
                raise TimeoutError('old controller has not drained; no new workers started')
            time.sleep(5)
        if cancelled:
            status.update(status='cancelled_before_resume', finished_at=now())
            write(status_path, status)
            return
        if sha(request_path) != status['change_request_sha256']:
            raise ValueError('change request changed during handover')
        state = read(output / 'state.json')
        if state['pid'] != request['old_controller_pid'] or state.get('active') or state['status'] not in ('stopped', 'complete'):
            raise RuntimeError('old controller did not record a clean drain')
        manifest, source = verify_campaign(output, request)
        history = Path(request['before_handover_snapshot']).parent / ('after_drain_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
        history.mkdir(exist_ok=False)
        for name in ('state.json', 'summary.json', 'parallel_scheduler_status.json'):
            (history / name).write_bytes((output / name).read_bytes())
        drain_measurements = {p.name: sha(p) for p in sorted((Path(manifest['raw_root']) / 'measurements').glob('*.json'))}
        write(history / 'completed_measurement_hashes.json', drain_measurements)
        sys.path.insert(0, str(source))
        spec = importlib.util.spec_from_file_location('frozen_cosim_scheduler', source / 'scripts/run_historical_cosim.py')
        runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runner)
        if runner.REPO.resolve() != source:
            raise RuntimeError('scheduler imported a different evaluator tree')
        runner.verify_code(manifest)
        original_external = runner.external_vitis
        original_render = runner.render
        def external_vitis(parent_pid):
            external = original_external(parent_pid)
            allowed, reason = allowed_main_descendants(request['allowed_main'], external)
            write(output / 'coexist_resource_gate.json', {'checked_at': now(), 'allowed_main_tool_pids': sorted(allowed),
                  'external_vitis': [pid for pid in external if pid not in allowed], 'authentication_note': reason})
            return [pid for pid in external if pid not in allowed]
        def render(*args, **kwargs):
            summary = original_render(*args, **kwargs)
            summary.update(scheduler_workers=4, scheduler_source_sha256=status['scheduler_source_sha256'],
                           coexist_change_request_sha256=status['change_request_sha256'])
            runner.atomic(output / 'summary.json', summary)
            report = output / 'report.md'
            report.write_text(report.read_text().replace('at most two workers.',
                              'up to four workers; only the registered main study may run concurrently.'))
            return summary
        runner.external_vitis = external_vitis
        runner.render = render
        status.update(status='running', resumed_at=now(), after_drain_snapshot=str(history),
                      completed_before_resume=read(output / 'summary.json')['finished_new_validations'])
        write(status_path, status)
        print('COEXIST_RESUME', json.dumps(status), flush=True)
        runner.run(output, 4, 3600, 0)
        verify_campaign(output, request)
        for name, expected in drain_measurements.items():
            if sha(Path(manifest['raw_root']) / 'measurements' / name) != expected:
                raise ValueError(f'completed drain measurement changed: {name}')
        terminal = read(output / 'state.json')
        status.update(status=terminal['status'], finished_at=now(), returncode=0)
        write(status_path, status)
    except Exception as exc:
        status.update(status='error', finished_at=now(), error=f'{type(exc).__name__}: {exc}')
        write(status_path, status)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--change-request', type=Path, required=True)
    parser.add_argument('--main-campaign', type=Path)
    action = parser.add_mutually_exclusive_group()
    action.add_argument('--prepare', action='store_true')
    action.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    output, request_path = args.output.resolve(), args.change_request.resolve()
    if args.prepare:
        if not args.main_campaign:
            parser.error('--prepare requires --main-campaign')
        request = prepare(output, request_path, args.main_campaign)
    else:
        request = read(request_path)
    manifest, _ = verify_campaign(output, request)
    if args.prepare or args.check_only:
        identity, note = authenticate_main(request['allowed_main'])
        print(json.dumps({'verified': True, 'workers': 4, 'timeout_seconds': 3600,
                          'frozen_files': len(manifest['code_hashes']),
                          'preserved_results': len(request['completed_measurement_hashes']),
                          'main_controller_authenticated': identity is not None,
                          'main_controller_note': note, 'change_request': str(request_path)}))
        return
    run(output, request, request_path)


if __name__ == '__main__':
    main()
