#!/usr/bin/env python3
"""Freeze and schedule one Figure 7/Table 7 validation lane after current recovery.

The frozen historical evaluator is unchanged. This campaign admits only saved
designs with verified inputs, excludes existing outcomes and reserved July work,
and never waits on the stopped main-study controller. No model calls are made.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
VITIS_NAMES = {'vitis_hls', 'vitis-run', 'xsim', 'xsimk', 'vivado', 'xelab'}
TERMINAL = {'complete', 'complete_with_quarantined_failure', 'complete_with_failures'}


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    temp.replace(path)


def process_table(proc_root=Path('/proc')):
    """Enumerate /proc once; disappearing process directories are expected races."""
    rows = {}
    with os.scandir(proc_root) as entries:
        names = [entry.name for entry in entries if entry.name.isdecimal()]
    for name in names:
        try:
            fields = (proc_root / name / 'stat').read_text().rsplit(')', 1)[1].split()
            pid = int(name)
            rows[pid] = {'pid': pid, 'ppid': int(fields[1]),
                         'start_ticks': fields[19], 'state': fields[0]}
        except (OSError, ValueError, IndexError):
            continue
    return rows


def load_owned(path):
    owned = load_module('figure7_owned_process', path)
    # Only the new controller replaces its own /proc enumerator. The historical
    # main/recovery code and their frozen helpers remain byte-for-byte intact.
    owned.process_table = process_table
    return owned


def alive(identity):
    try:
        row = process_table().get(identity['pid'])
        if row is None or row['state'] == 'Z' or row['start_ticks'] != str(identity['start_ticks']):
            return False
        proc = Path('/proc') / str(identity['pid'])
        argv = (proc / 'cmdline').read_bytes().split(b'\0')
        return (Path('/proc/sys/kernel/random/boot_id').read_text().strip() == identity['boot_id']
                and str(identity['source_runner']).encode() in argv)
    except (OSError, KeyError, IndexError, TypeError):
        return False


def dependency_gate(config):
    folder = Path(config['wait_for_recovery'])
    summary = read(folder / 'summary.json')
    identity = read(folder / 'controller_identity.json')
    live = alive(identity)
    # A stale running state or an error stop is not permission to bypass recovery.
    allowed = summary['status'] in TERMINAL and not live
    return {'allowed': allowed, 'campaign': str(folder), 'status': summary['status'],
            'controller_alive': live,
            'reason': None if allowed else 'preserve_frozen_recovery_peer_admission_policy'}


def resource_gate(output):
    unrelated = []
    for pid, row in process_table().items():
        if row['state'] == 'Z':
            continue
        try:
            name = (Path('/proc') / str(pid) / 'comm').read_text().strip()
        except OSError:
            continue
        if name in VITIS_NAMES:
            unrelated.append(pid)
    mem = {line.split(':')[0]: int(line.split()[1])
           for line in Path('/proc/meminfo').read_text().splitlines()}
    observed = {'scratch_free_gib': shutil.disk_usage('/tmp').free / 2**30,
                'output_free_gib': shutil.disk_usage(output).free / 2**30,
                'available_memory_gib': mem['MemAvailable'] / 2**20,
                'load_per_cpu': os.getloadavg()[0] / (os.cpu_count() or 1)}
    reasons = []
    if min(observed['scratch_free_gib'], observed['output_free_gib']) < 100:
        reasons.append('storage_below_100_GiB')
    if observed['available_memory_gib'] < 128:
        reasons.append('memory_below_128_GiB')
    if observed['load_per_cpu'] >= .7:
        reasons.append('load_above_0.7_per_cpu')
    if unrelated:
        reasons.append('unrelated_vitis')
    return {'allowed': not reasons, 'reasons': reasons, 'observed': observed,
            'unrelated_vitis_pids': unrelated}


def summarize(state):
    return {'status': state['status'], 'updated_at': state['updated_at'],
            'distinct_validations': len(state['tasks']),
            'counts': dict(Counter(row['status'] for row in state['tasks'])),
            'active': [row for row in state['tasks'] if row['status'] == 'running'],
            'dependency_gate': state.get('dependency_gate'),
            'resource_gate': state.get('resource_gate'), 'failure': state.get('failure'),
            'model_calls': 0}


def persist(output, state):
    state['updated_at'] = now()
    save(output / 'state.json', state)
    save(output / 'summary.json', summarize(state))


def prepare(source, output, raw_root, python):
    july = load_module('figure7_july_prepare', ROOT / 'scripts/prepare_july_routing_backfill.py')
    fingerprint = ROOT / 'results_sweeps/historical_registry_recovery_20260928/manifest.json'
    frozen = ROOT / 'artifacts/sigmetrics2027/registry_recovery_20260928/source'
    original = july.HPCA / 'results_sweeps/july_cosim_backfill_20260928/manifest.json'
    existing = july.default_existing(original, ROOT)
    existing.append(ROOT / 'results_sweeps/july_routing_cosim_20261006/manifest.json')
    report = july.prepare(source, output, raw_root, frozen, fingerprint, existing)
    # One-way registration: do not revise this protocol after a controller starts.
    owned_source = output / 'source/scripts'
    owned_source.mkdir(parents=True)
    source_files = [Path(__file__), ROOT / 'scripts/prepare_july_routing_backfill.py',
                    ROOT / 'scripts/resume_sigmetrics_recipe_main.py']
    for path in source_files:
        shutil.copyfile(path, owned_source / path.name)
    pinned = [source, output/'manifest.json', output/'input_audit.json']
    pinned.extend(owned_source.iterdir())
    recovery = ROOT / 'results_sweeps/sigmetrics_timeout_recovery_20261006'
    pinned.append(recovery/'manifest.json')
    config = {'schema_version': 'c2hls.figure7-priority-cosim-queue.v1',
              'created_at': now(), 'campaign': str(output), 'python': str(python),
              'python_sha256': july.file_sha(python), 'source': str(owned_source.parent),
              'source_runner': str(owned_source / Path(__file__).name),
              'owned_process_helper': str(owned_source/'resume_sigmetrics_recipe_main.py'),
              'wait_for_recovery': str(recovery), 'workers': 1,
              'outer_worker_timeout_seconds': 5400,
              'policy': 'Independent lane after recovery completes; no dependency on stopped main. Never modify old queues, repeat outcomes, or duplicate reserved July identities. One 3600-second historical invocation per eligible design. No model calls.',
              'pinned_files': {str(path): july.file_sha(path) for path in pinned}}
    save(output/'queue_manifest.json', config)
    state = read(output/'state.json')
    state.update(status='queued', queued_at=now(), failure=None,
                 queue_manifest_sha256=july.file_sha(output/'queue_manifest.json'))
    persist(output, state)
    report['source_runner'] = config['source_runner']
    report['queue_manifest_sha256'] = state['queue_manifest_sha256']
    print(json.dumps(report, indent=2, sort_keys=True))


def verify(output):
    config = read(output/'queue_manifest.json')
    july = load_module('figure7_july_verify', Path(config['source'])/'scripts/prepare_july_routing_backfill.py')
    for name, digest in config['pinned_files'].items():
        if july.file_sha(name) != digest:
            raise ValueError('Frozen queue input changed: ' + name)
    if july.file_sha(config['python']) != config['python_sha256']:
        raise ValueError('Python executable changed')
    manifest = read(output/'manifest.json')
    july.verify_files(manifest['frozen_source'], manifest['code_hashes'])
    for row in manifest['rows']:
        july.audit_row(row, manifest['toolchain'], Path(manifest['frozen_source']), require_result=True)
    return config, manifest, july


def classify_measurement(measurement, execution):
    c = measurement.get('cosim') or {}
    if execution.get('returncode') != 0 or execution.get('cleanup_survivors') or measurement.get('archive_error'):
        return 'execution_error'
    status = c.get('status')
    if status == 'not_run':
        return {'candidate_csim_failed': 'failed_csim', 'invalid_native_golden': 'blocked_input'}.get(
            c.get('skip_reason'), 'execution_error')
    if status == 'pass' and not (c.get('passed') is True and c.get('success') is True
                                and (c.get('kernel_runtime_cycles') or 0) > 0
                                and (c.get('correctness') or {}).get('passed') is True):
        return 'evidence_error'
    return status if status in ('pass', 'fail', 'timeout') else 'execution_error'


def execute_task(output, config, manifest, task, owned, july):
    validation_id = task['validation_id']
    measurement = Path(manifest['raw_root'])/'measurements'/(validation_id+'.json')
    scratch = Path('/tmp')/output.name/validation_id
    if measurement.exists() or scratch.exists():
        raise ValueError('Refusing previously started validation identity: ' + validation_id)
    row = next(row for row in manifest['rows'] if row['validation_id'] == validation_id)
    july.audit_row(row, manifest['toolchain'], Path(manifest['frozen_source']), require_result=True)
    command = [config['python'], manifest['frozen_worker'], 'worker', '--output', str(output),
               '--validation-id', validation_id, '--timeout', str(manifest['timeout_seconds'])]
    env = {k: v for k, v in os.environ.items() if not k.startswith('C2HLS_')}
    execution = owned.execute_owned(command, manifest['frozen_source'],
                                    output/'logs'/(validation_id+'.log'), env,
                                    config['outer_worker_timeout_seconds'])
    if not measurement.exists():
        return {'status': 'execution_error', 'error': 'missing_measurement', 'execution': execution}
    record = read(measurement)
    if record.get('validation_id') != validation_id or record.get('kernel_sha256') != row['kernel_sha256']:
        return {'status': 'evidence_error', 'error': 'measurement_identity_mismatch', 'execution': execution}
    return {'status': classify_measurement(record, execution), 'execution': execution,
            'recorded_cosim_status': (record.get('cosim') or {}).get('status'),
            'measurement_path': str(measurement), 'measurement_sha256': july.file_sha(measurement)}


def run(output):
    config, manifest, july = verify(output)
    if Path(__file__).resolve() != Path(config['source_runner']).resolve():
        raise ValueError('Run the frozen scheduler recorded in queue_manifest.json')
    state = read(output/'state.json')
    if state['status'] != 'queued' or any(t['status'] != 'pending' for t in state['tasks']):
        raise ValueError('Started campaigns require explicit recovery; no implicit reruns')
    if state['queue_manifest_sha256'] != july.file_sha(output/'queue_manifest.json'):
        raise ValueError('Queue manifest hash mismatch')
    locks = []
    for name in ('controller.lock', 'queue.lock'):
        lock = (output/name).open('a')
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        locks.append(lock)
    owned = load_owned(config['owned_process_helper'])
    row = process_table()[os.getpid()]
    identity = {'pid': os.getpid(), 'start_ticks': row['start_ticks'],
                'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                'campaign': str(output), 'source_runner': config['source_runner'],
                'source_runner_sha256': july.file_sha(config['source_runner']),
                'manifest_sha256': state['queue_manifest_sha256'], 'registered_at': now()}
    save(output/'controller_identity.json', identity)
    (output/'logs').mkdir(exist_ok=True)
    stop = {}
    def request_stop(signum, frame):
        stop.update(signal=signum, at=now(), policy='drain active validation, stop new dispatches')
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, request_stop)
    while True:
        pending = next((t for t in state['tasks'] if t['status'] == 'pending'), None)
        if stop or pending is None:
            state['status'] = 'stopped' if stop else 'complete'
            state['stop_request'] = stop or None
            persist(output, state)
            return
        state['dependency_gate'] = dependency_gate(config)
        state['resource_gate'] = resource_gate(output)
        if not state['dependency_gate']['allowed'] or not state['resource_gate']['allowed']:
            state['status'] = ('waiting_for_recovery' if not state['dependency_gate']['allowed']
                               else 'waiting_for_resources')
            persist(output, state)
            time.sleep(30)
            continue
        state['status'] = 'running'
        pending.update(status='running', started_at=now())
        persist(output, state)
        try:
            outcome = execute_task(output, config, manifest, pending, owned, july)
        except Exception as exc:
            outcome = {'status': 'execution_error', 'error': f'{type(exc).__name__}: {exc}'}
        pending.update(outcome, finished_at=now())
        if outcome['status'] in ('execution_error', 'evidence_error'):
            state.update(status='stopped_on_error', failure={'validation_id': pending['validation_id'], **outcome})
            persist(output, state)
            return
        persist(output, state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    prep = commands.add_parser('prepare')
    prep.add_argument('--source', type=Path, required=True)
    prep.add_argument('--output', type=Path, required=True)
    prep.add_argument('--raw-root', type=Path, required=True)
    prep.add_argument('--python', type=Path, default=Path(sys.executable))
    for command in ('verify', 'run'):
        child = commands.add_parser(command)
        child.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare(args.source.resolve(), args.output.resolve(), args.raw_root.resolve(), args.python.resolve())
    elif args.command == 'verify':
        config, manifest, _ = verify(args.output.resolve())
        print(json.dumps({'verified': True, 'candidate_rows': len(manifest['rows']),
                          'worker_count': config['workers'], 'model_calls': 0}))
    else:
        run(args.output.resolve())


if __name__ == '__main__':
    main()
