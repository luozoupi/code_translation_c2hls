#!/usr/bin/env python3
"""Change queue parallelism while retaining the campaign's frozen validator.

The launcher drains the earlier controller and calls the frozen scheduler's
existing run(output, workers, timeout, limit) interface. Workers still execute
the original frozen file. This file only controls scheduling and labels reports.
"""
import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, data):
    temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(data, indent=2, sort_keys=True) + '\n')
    temp.replace(path)


def verify_campaign(output, request):
    receipt = read(output / 'launch_receipt.json')
    manifest_path = output / 'manifest.json'
    if sha(manifest_path) != request['manifest_sha256']:
        raise ValueError('campaign manifest changed')
    manifest = read(manifest_path)
    source = Path(receipt['source_snapshot'])
    for name, expected in manifest['code_hashes'].items():
        if sha(source / name) != expected:
            raise ValueError(f'frozen validator changed: {name}')
    raw = Path(manifest['raw_root'])
    for name, expected in request['completed_measurement_hashes'].items():
        if sha(raw / 'measurements' / name) != expected:
            raise ValueError(f'completed measurement changed: {name}')
    return manifest, source


def same_process(pid, start_ticks):
    try:
        return (Path('/proc') / str(pid) / 'stat').read_text().rsplit(')', 1)[1].split()[19] == start_ticks
    except FileNotFoundError:
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--change-request', type=Path, required=True)
    parser.add_argument('--workers', type=int, choices=range(1, 5), required=True)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    output = args.output.resolve()
    request = read(args.change_request.resolve())
    if args.workers != request['requested_workers']:
        raise ValueError('worker count differs from the recorded user request')
    manifest, source = verify_campaign(output, request)
    if args.check_only:
        print(json.dumps({'verified': True, 'workers': args.workers,
                          'frozen_files': len(manifest['code_hashes']),
                          'preserved_results': len(request['completed_measurement_hashes'])}))
        return
    status_path = output / 'parallel_scheduler_status.json'
    status = {'started_at': now(), 'pid': os.getpid(), 'workers': args.workers,
              'scheduler_source_sha256': sha(Path(__file__)),
              'scheduler_source': str(Path(__file__).resolve()),
              'manifest_sha256': request['manifest_sha256'],
              'frozen_source': str(source), 'status': 'waiting_for_previous_controller',
              'previous_controller_pid': request['old_controller_pid'],
              'timeout_seconds': 3600}
    write(status_path, status)
    try:
        deadline = time.monotonic() + 4800
        while same_process(request['old_controller_pid'], request['old_controller_start_ticks']):
            if time.monotonic() >= deadline:
                raise TimeoutError('previous controller did not finish draining; no new workers started')
            time.sleep(10)
        old_state = read(output / 'state.json')
        if old_state['pid'] != request['old_controller_pid'] or old_state.get('active') or old_state['status'] not in ('stopped', 'complete'):
            raise RuntimeError('prior controller did not record a clean drain; inspect workers before resuming')
        manifest, source = verify_campaign(output, request)
        history = output / 'execution_history' / 'before_four_workers_20260929'
        history.mkdir(parents=True, exist_ok=False)
        for name in ('state.json', 'summary.json', 'launcher_status.json'):
            (history / name).write_bytes((output / name).read_bytes())
        sys.path.insert(0, str(source))
        spec = importlib.util.spec_from_file_location('frozen_cosim_scheduler', source / 'scripts/run_historical_cosim.py')
        runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runner)
        if runner.REPO.resolve() != source.resolve():
            raise RuntimeError('scheduler imported a different evaluator tree')
        runner.verify_code(manifest)
        original_render = runner.render

        def render_with_schedule(*render_args, **render_kwargs):
            summary = original_render(*render_args, **render_kwargs)
            summary['scheduler_workers'] = args.workers
            summary['scheduler_source_sha256'] = status['scheduler_source_sha256']
            runner.atomic(output / 'summary.json', summary)
            report = output / 'report.md'
            text = report.read_text().replace('at most two workers.',
                f'up to {args.workers} workers in the current invocation (previously two).')
            report.write_text(text)
            return summary

        runner.render = render_with_schedule
        status.update(status='running', resumed_at=now(),
                      completed_before_resume=read(output / 'summary.json')['finished_new_validations'])
        write(status_path, status)
        print('RESUME', json.dumps(status), flush=True)
        runner.run(output, args.workers, 3600, 0)
        terminal = read(output / 'state.json')
        status.update(status=terminal['status'], finished_at=now(), returncode=0)
        write(status_path, status)
    except Exception as exc:
        status.update(status='error', finished_at=now(), error=f'{type(exc).__name__}: {exc}')
        write(status_path, status)
        raise


if __name__ == '__main__':
    main()
