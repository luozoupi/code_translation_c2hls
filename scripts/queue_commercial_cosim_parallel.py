#!/usr/bin/env python3
"""Continue a drained timeout recovery alongside two commercial RTL backfills.

Scheduling-only amendment: frozen evaluators, saved inputs, original task IDs,
co-simulation limits, and completed outcomes are unchanged. Preparation and run
never signal another controller. An operator requests the old controller's
existing graceful drain separately, after reviewing this registration.
"""
from __future__ import annotations
import argparse
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import copy
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
ERRORS = {'execution_error', 'evidence_error'}
VITIS = {'vitis_hls', 'vitis-run', 'xsim', 'xsimk', 'vivado', 'xelab'}


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    temporary.replace(path)


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def check_files(files):
    for path, digest in files.items():
        if sha(path) != digest:
            raise ValueError('Frozen evidence changed: ' + path)


def completed_files(state):
    files = {}
    for task in state['tasks']:
        if task['status'] in ('pending', 'running'):
            continue
        path = task.get('measurement_path')
        if path:
            if sha(path) != task['measurement_sha256']:
                raise ValueError('Completed measurement changed: ' + path)
            files[path] = sha(path)
            record = read(path)
            # Pin saved archive receipts and archives whenever a measurement
            # carries them, as well as its final numerical result record.
            def evidence(value):
                if isinstance(value, dict):
                    for child in value.values():
                        evidence(child)
                elif isinstance(value, list):
                    for child in value:
                        evidence(child)
                elif isinstance(value, str) and value.startswith('/'):
                    candidate = Path(value)
                    if candidate.is_file():
                        files[str(candidate)] = sha(candidate)
            evidence(record)
    return files


def require_clean_drain(before, after):
    """Active jobs may finish; no terminal outcome can change or be retried."""
    if after['status'] not in ('stopped', 'complete') or after.get('failure'):
        raise ValueError('Previous controller has not recorded a clean drain')
    if any(t['status'] == 'running' for t in after['tasks']):
        raise ValueError('Previous controller still has active tasks')
    old = {t['id']: t for t in before['tasks']}
    current = {t['id']: t for t in after['tasks']}
    if (len(old) != len(before['tasks']) or len(current) != len(after['tasks'])
            or old.keys() != current.keys()):
        raise ValueError('Recovery task identity changed')
    for identity, task in old.items():
        successor = current[identity]
        for key in ('id', 'historical_output', 'kernel_sha256', 'timeout_seconds', 'cohort'):
            if task[key] != successor[key]:
                raise ValueError('Recovery task protocol changed: ' + identity)
        if task['status'] not in ('pending', 'running') and task != successor:
            raise ValueError('Completed recovery outcome changed: ' + identity)
        if task['status'] == 'running' and successor['status'] in ('pending', 'execution_error'):
            raise ValueError('An active recovery job did not finish cleanly: ' + identity)
    return [copy.deepcopy(t) for t in after['tasks'] if t['status'] == 'pending']


def require_pending_pristine(tasks):
    for task in tasks:
        hist = Path(task['historical_output'])
        measurement = hist/'raw/measurements'/(task['id'] + '.json')
        scratch = Path('/tmp')/hist.name/task['id']
        if measurement.exists() or scratch.exists():
            raise ValueError('Pending task already has execution evidence: ' + task['id'])


def prepare(output, recovery, commercial, python):
    if output.exists():
        raise ValueError('Output exists; refusing to replace a campaign')
    fig = module('commercial_queue_prepare', ROOT/'scripts/queue_figure7_cosim_backfill.py')
    commercial_config, commercial_manifest, _ = fig.verify(commercial)
    commercial_state = read(commercial/'state.json')
    if commercial_state['status'] != 'queued' or any(t['status'] != 'pending' for t in commercial_state['tasks']):
        raise ValueError('Commercial child must be an untouched prepared queue')
    if (commercial/'parallel_registration.json').exists():
        raise ValueError('Commercial child already registered')
    before = read(recovery/'state.json')
    recovery_manifest = read(recovery/'manifest.json')
    identity = read(recovery/'controller_identity.json')
    if before['status'] not in ('running', 'stopped', 'complete') or before.get('failure'):
        raise ValueError('Recovery controller is not eligible for a clean scheduling handoff')
    if before['manifest_sha256'] != sha(recovery/'manifest.json'):
        raise ValueError('Recovery manifest identity changed')
    check_files(recovery_manifest['files'])
    if identity['manifest_sha256'] != before['manifest_sha256'] or sha(identity['source_runner']) != identity['source_runner_sha256']:
        raise ValueError('Recovery controller identity changed')
    if str(python.resolve()) != str(Path(commercial_config['python']).resolve()):
        raise ValueError('Use the registered commercial Python executable')
    output.mkdir(parents=True)
    source = output/'source/scripts'
    source.mkdir(parents=True)
    shutil.copyfile(__file__, source/Path(__file__).name)
    save(output/'recovery_before_handoff.json', before)
    save(output/'recovery_controller_before_handoff.json', identity)
    files = completed_files(before)
    for path in (source/Path(__file__).name, output/'recovery_before_handoff.json',
                 output/'recovery_controller_before_handoff.json', recovery/'manifest.json',
                 commercial/'manifest.json', commercial/'queue_manifest.json', commercial/'input_audit.json'):
        files[str(path)] = sha(path)
    config = {'schema_version': 'c2hls.commercial-parallel-cosim.v1', 'created_at': now(),
              'recovery': str(recovery), 'commercial': str(commercial),
              'python': str(python.resolve()), 'python_sha256': sha(python),
              'source_runner': str(source/Path(__file__).name),
              'recovery_runner': identity['source_runner'],
              'commercial_runner': commercial_config['source_runner'],
              'workers': {'recovery': 2, 'commercial': 2},
              'minimum_free_gib': 200, 'minimum_available_memory_gib': 128,
              'maximum_load_per_cpu': .7, 'files': files,
              'policy': 'After the authenticated prior controller durably records its graceful-drain request, admit up to two commercial workers alongside its already-active jobs. After its clean exit, take locks and continue only never-started recovery IDs with unchanged evaluator and 10800-second limits; commercial invocation limits remain 3600 seconds. Preserve all terminal outcomes, original manifest, and snapshots; update old operational state/summary/identity only after ownership handoff. No model calls or signals to predecessor. Stop the affected lane on execution or evidence uncertainty.'}
    save(output/'manifest.json', config)
    state = {'status': 'queued', 'created_at': now(), 'updated_at': now(),
             'manifest_sha256': sha(output/'manifest.json'), 'tasks': [], 'lane_failures': {},
             'recovery_baseline_counts': None, 'recovery_legacy_counts': recovery_manifest['legacy_historical_counts'],
             'commercial_distinct_validations': len(commercial_state['tasks'])}
    persist(output, state)
    save(commercial/'parallel_registration.json', {'registered_at': now(), 'campaign': str(output),
         'manifest_sha256': state['manifest_sha256'], 'canonical_state': str(output/'state.json'),
         'canonical_summary': str(output/'summary.json'),
         'note': 'Prepared child state remains immutable; only combined scheduler may dispatch child validations.'})
    print(json.dumps({'prepared': True, 'commercial_tasks': len(commercial_state['tasks']),
                      'source_runner': config['source_runner'], 'workers': config['workers'], 'model_calls': 0}, indent=2))


def summarize(state):
    counts = {lane: dict(Counter(t['status'] for t in state['tasks'] if t['lane'] == lane))
              for lane in ('recovery', 'commercial')}
    combined = None
    if state.get('recovery_baseline_counts') is not None:
        combined = Counter(state['recovery_legacy_counts'])
        combined.update(state['recovery_baseline_counts'])
        combined.update(counts['recovery'])
    return {'status': state['status'], 'updated_at': state['updated_at'], 'counts': counts,
            'combined_three_hour_replay': dict(combined) if combined is not None else None,
            'commercial_distinct_validations': state['commercial_distinct_validations'],
            'active': [t for t in state['tasks'] if t['status'] == 'running'],
            'resource_gate': state.get('resource_gate'), 'drain_gate': state.get('drain_gate'),
            'lane_failures': state['lane_failures'], 'model_calls': 0}


def persist(output, state):
    state['updated_at'] = now()
    save(output/'state.json', state)
    save(output/'summary.json', summarize(state))


def verify(output):
    config = read(output/'manifest.json')
    state = read(output/'state.json')
    if state['manifest_sha256'] != sha(output/'manifest.json'):
        raise ValueError('Combined registration changed')
    check_files(config['files'])
    if sha(config['python']) != config['python_sha256']:
        raise ValueError('Python executable changed')
    fig = module('commercial_frozen_queue', config['commercial_runner'])
    commercial_config, commercial_manifest, july = fig.verify(Path(config['commercial']))
    recovery_manifest = read(Path(config['recovery'])/'manifest.json')
    check_files(recovery_manifest['files'])
    return config, state, fig, commercial_config, commercial_manifest, july, recovery_manifest


def descendant(pid, ancestor, table):
    seen = set()
    while pid in table and pid not in seen and pid != ancestor:
        seen.add(pid)
        pid = table[pid]['ppid']
    return pid == ancestor


def drain_admission(identity, previous, fig):
    """Durable stop_requested admits the still-running controller as a peer."""
    if not fig.alive(identity):
        return {'alive': False, 'commercial_allowed': False, 'peer_pid': None}
    requested = previous.get('stop_requested') or {}
    allowed = (previous['status'] == 'running' and not previous.get('failure')
               and requested.get('signal') in (signal.SIGINT, signal.SIGTERM)
               and previous.get('pid') == identity['pid']
               and str(previous.get('pid_start_ticks')) == str(identity['start_ticks'])
               and previous.get('boot_id') == identity['boot_id'])
    return {'alive': True, 'commercial_allowed': allowed,
            'peer_pid': identity['pid'] if allowed else None,
            'reason': 'authenticated_durable_drain' if allowed else 'awaiting_recorded_drain_request'}


def resource_gate(output, config, fig, peer_identity=None):
    table = fig.process_table()
    peer_pid = None
    if peer_identity:
        peer_row = table.get(peer_identity['pid'])
        if (peer_row and peer_row['state'] != 'Z'
                and str(peer_row['start_ticks']) == str(peer_identity['start_ticks'])):
            peer_pid = peer_identity['pid']
    unrelated = []
    for pid, row in table.items():
        if (row['state'] == 'Z' or descendant(pid, os.getpid(), table)
                or (peer_pid is not None and descendant(pid, peer_pid, table))):
            continue
        try:
            name = (Path('/proc')/str(pid)/'comm').read_text().strip()
        except OSError:
            continue
        if name in VITIS:
            unrelated.append(pid)
    mem = {r.split(':')[0]: int(r.split()[1]) for r in Path('/proc/meminfo').read_text().splitlines()}
    observed = {'scratch_free_gib': shutil.disk_usage('/tmp').free/2**30,
                'output_free_gib': shutil.disk_usage(output).free/2**30,
                'memory_available_gib': mem['MemAvailable']/2**20,
                'load_per_cpu': os.getloadavg()[0]/(os.cpu_count() or 1)}
    reasons = []
    if min(observed['scratch_free_gib'], observed['output_free_gib']) < config['minimum_free_gib']:
        reasons.append('storage_headroom')
    if observed['memory_available_gib'] < config['minimum_available_memory_gib']:
        reasons.append('memory_headroom')
    if observed['load_per_cpu'] >= config['maximum_load_per_cpu']:
        reasons.append('cpu_load')
    if unrelated:
        reasons.append('unrelated_vitis')
    return {'allowed': not reasons, 'reasons': reasons, 'observed': observed,
            'unrelated_vitis_pids': sorted(unrelated)}


def recovery_operational_state(baseline, tasks, identity, stop=None):
    """Only a locked successor calls this after preserving the drained state."""
    merged = copy.deepcopy(baseline)
    completed = {t['id']: t for t in tasks if t['lane'] == 'recovery'}
    for index, task in enumerate(merged['tasks']):
        if task['id'] in completed:
            if task['status'] != 'pending':
                raise ValueError('Refusing to replace a completed recovery outcome')
            merged['tasks'][index] = {k: v for k, v in completed[task['id']].items() if k != 'lane'}
    statuses = [t['status'] for t in merged['tasks']]
    failed = next((t for t in merged['tasks'] if t['status'] in ERRORS), None)
    if failed:
        status = 'stopped_on_error'
    elif stop and 'running' not in statuses:
        status = 'stopped'
    else:
        status = 'running' if any(s in ('pending', 'running') for s in statuses) else 'complete'
    merged.update(status=status, pid=identity['pid'], pid_start_ticks=identity['start_ticks'],
                  boot_id=identity['boot_id'], updated_at=now(), stop_requested=stop,
                  continuation_campaign=identity['campaign'],
                  failure={'task': failed['id'], 'detail': failed} if failed else None)
    if status == 'complete':
        merged['finished_at'] = now()
    return merged


def run(output):
    config, state, fig, cc, cm, july, rm = verify(output)
    if Path(__file__).resolve() != Path(config['source_runner']).resolve() or state['status'] != 'queued':
        raise ValueError('Use untouched registration and its frozen controller; recovery must be explicit')
    recovery, commercial = Path(config['recovery']), Path(config['commercial'])
    locks = []
    def lock(path):
        handle = path.open('a')
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        locks.append(handle)
    lock(output/'controller.lock')
    lock(commercial/'controller.lock')
    lock(commercial/'queue.lock')
    child_state = read(commercial/'state.json')
    registration = read(commercial/'parallel_registration.json')
    if (child_state['status'] != 'queued' or any(t['status'] != 'pending' for t in child_state['tasks'])
            or registration['campaign'] != str(output)
            or registration['manifest_sha256'] != state['manifest_sha256']):
        raise ValueError('Commercial child changed or has another registered owner')
    row = fig.process_table()[os.getpid()]
    new_identity = {'pid': os.getpid(), 'start_ticks': row['start_ticks'],
         'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
         'campaign': str(output), 'source_runner': config['source_runner'],
         'source_runner_sha256': sha(config['source_runner']), 'manifest_sha256': sha(output/'manifest.json')}
    save(output/'controller_identity.json', new_identity)
    stop = {}
    def stopping(signum, _frame):
        stop.update(signal=signum, requested_at=now(), policy='stop dispatch and drain active jobs')
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, stopping)
    identity = read(output/'recovery_controller_before_handoff.json')
    state['tasks'] = [dict(t, lane='commercial') for t in child_state['tasks']]
    recovery_runner = module('unchanged_recovery_worker', config['recovery_runner'])
    owned = fig.load_owned(cc['owned_process_helper'])
    execution_root = output/'recovery_execution'
    execution_root.mkdir()
    (commercial/'logs').mkdir(exist_ok=True)
    preserved, baseline = {}, None
    handoff = False
    active = {}

    def persist_both():
        persist(output, state)
        if handoff:
            operational = recovery_operational_state(baseline, state['tasks'], new_identity, stop or None)
            save(recovery/'state.json', operational)
            save(recovery/'summary.json', recovery_runner.summarize(operational, rm))

    def accept_handoff():
        nonlocal preserved, baseline, handoff
        lock(recovery/'controller.lock')
        lock(Path(rm['legacy'])/'controller.lock')
        after = read(recovery/'state.json')
        pending = require_clean_drain(read(output/'recovery_before_handoff.json'), after)
        require_pending_pristine(pending)
        preserved = completed_files(after)
        snapshots = {'state.json': 'recovery_after_drain.json',
                     'summary.json': 'recovery_summary_after_drain.json',
                     'controller_identity.json': 'recovery_identity_after_drain.json'}
        for original, snapshot in snapshots.items():
            shutil.copyfile(recovery/original, output/snapshot)
            preserved[str(output/snapshot)] = sha(output/snapshot)
        save(output/'preserved_after_drain.json', preserved)
        state['recovery_baseline_counts'] = dict(Counter(t['status'] for t in after['tasks']
             if t['cohort'] == 'recovery' and t['status'] != 'pending'))
        if any(t['cohort'] != 'recovery' or t['timeout_seconds'] != 10800 for t in pending):
            raise ValueError('Only original three-hour recovery tasks may continue')
        state['tasks'].extend(dict(t, lane='recovery') for t in pending)
        baseline = after
        receipt = {'accepted_at': now(), 'new_campaign': str(output),
                   'manifest_sha256': state['manifest_sha256'],
                   'before_drain_snapshot': str(output/'recovery_before_handoff.json'),
                   'after_drain_snapshot': str(output/'recovery_after_drain.json'),
                   'preserved_evidence': str(output/'preserved_after_drain.json'),
                   'pending_ids': [t['id'] for t in pending],
                   'policy': 'Operational state/summary/identity now mirror the authenticated successor; original manifest, evaluator, limits and completed outcomes remain unchanged.'}
        save(recovery/'continuation_receipt_20261008.json', receipt)
        save(output/'handoff_receipt.json', receipt)
        # Consumers of the old queue continue seeing live ownership until this
        # combined controller exits, even if its recovery lane finishes first.
        save(recovery/'controller_identity.json', dict(new_identity,
             campaign=str(recovery), manifest_sha256=sha(recovery/'manifest.json'),
             continuation_campaign=str(output), scheduling_amendment_sha256=state['manifest_sha256']))
        handoff = True
        state.update(status='running', drain_gate={'alive': False, 'clean_drain': True, 'accepted_at': now()})
        persist_both()

    with ThreadPoolExecutor(max_workers=sum(config['workers'].values())) as pool:
        while True:
            peer_pid = None
            admitted = handoff
            if not handoff:
                previous = read(recovery/'state.json')
                state['drain_gate'] = drain_admission(identity, previous, fig)
                if not state['drain_gate']['alive']:
                    accept_handoff()
                    admitted = True
                else:
                    admitted = state['drain_gate']['commercial_allowed']
                    peer_pid = state['drain_gate']['peer_pid']
                    state['status'] = 'commercial_running_during_recovery_drain' if admitted else 'waiting_for_recovery_drain'
            state['resource_gate'] = resource_gate(output, config, fig, identity if peer_pid else None)
            for lane, slots in config['workers'].items():
                count = sum(t['lane'] == lane for t in active.values())
                for task in state['tasks']:
                    if (stop or not admitted or lane in state['lane_failures']
                            or not state['resource_gate']['allowed'] or count >= slots):
                        break
                    if task['lane'] != lane or task['status'] != 'pending':
                        continue
                    task.update(status='running', started_at=now())
                    persist_both()
                    if lane == 'recovery':
                        future = pool.submit(recovery_runner.run_task, execution_root, task.copy(), rm)
                    else:
                        future = pool.submit(fig.execute_task, commercial, cc, cm, task.copy(), owned, july)
                    active[future] = task
                    count += 1
            state['stop_request'] = stop or None
            persist_both()
            if not active:
                dispatchable = any(t['status'] == 'pending' and t['lane'] not in state['lane_failures'] for t in state['tasks'])
                if stop or (handoff and not dispatchable):
                    break
                time.sleep(10)
                continue
            finished, _ = wait(active, timeout=10, return_when=FIRST_COMPLETED)
            for future in finished:
                task = active.pop(future)
                try:
                    outcome = future.result()
                except Exception as exc:
                    outcome = {'status': 'execution_error', 'error': f'{type(exc).__name__}: {exc}'}
                task.update(outcome, finished_at=now())
                if task['status'] in ERRORS:
                    state['lane_failures'][task['lane']] = dict(task)
                print('VALIDATION_DONE', task['lane'], task['benchmark'], task['status'], flush=True)
            persist_both()
    check_files(preserved)
    check_files(config['files'])
    state.update(status='stopped' if stop else ('complete_with_lane_error' if state['lane_failures'] else 'complete'), finished_at=now())
    persist_both()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    prep = commands.add_parser('prepare')
    prep.add_argument('--output', type=Path, required=True)
    prep.add_argument('--commercial', type=Path, required=True)
    prep.add_argument('--recovery', type=Path, default=ROOT/'results_sweeps/sigmetrics_timeout_recovery_20261006')
    prep.add_argument('--python', type=Path, default=Path(sys.executable))
    for command in ('verify', 'run'):
        child = commands.add_parser(command)
        child.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare(args.output.resolve(), args.recovery.resolve(), args.commercial.resolve(), args.python.resolve())
    elif args.command == 'verify':
        config, *_ = verify(args.output.resolve())
        print(json.dumps({'verified': True, 'workers': config['workers'], 'model_calls': 0}))
    else:
        run(args.output.resolve())


if __name__ == '__main__':
    main()
