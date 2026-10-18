#!/usr/bin/env python3
"""Replay only recorded 3600-second RTL timeouts in an isolated campaign."""
from __future__ import annotations

import argparse
import collections
import copy
import fcntl
import hashlib
import importlib.util
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / 'results_sweeps/sigmetrics_recipe_main_20260930'
HISTORY = ROOT / 'results_sweeps/historical_registry_recovery_20260928'
DEFAULT_OUTPUT = ROOT / 'results_sweeps/sigmetrics_timeout_replay_20261001'
TIMEOUT = 10800
VITIS_NAMES = {'vitis_hls', 'vitis-run', 'xsim', 'xsimk', 'vivado'}


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    temporary.replace(path)


def start_ticks(pid):
    try:
        return Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[19]
    except (OSError, IndexError):
        return None


def load_frozen_main():
    config = read(MAIN / 'manifest.json')
    source = Path(config['source'])
    path = source / 'scripts/run_sigmetrics_recipe_main.py'
    sys.path.insert(0, str(source))
    spec = importlib.util.spec_from_file_location('frozen_sigmetrics_main_for_replay', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.verify_files(config['files'])
    return config, module


def is_original_timeout(measurement):
    cosim = measurement.get('cosim') or {}
    return cosim.get('status') == 'timeout' and cosim.get('error') == 'Cosim timed out after 3600s'


def prepare(output):
    if output.exists():
        raise FileExistsError(f'replay output already exists: {output}')
    main, main_module = load_frozen_main()
    mstate = read(MAIN / 'state.json')
    if mstate['manifest_sha256'] != sha(MAIN / 'manifest.json'):
        raise ValueError('main-study manifest identity changed')
    original = read(HISTORY / 'manifest.json')
    frozen_history = Path(read(HISTORY / 'launch_receipt.json')['source_snapshot'])
    hist_worker = frozen_history / 'scripts/run_historical_cosim.py'
    hist_raw = Path(original['raw_root'])
    parents = []
    for item in mstate['parents']:
        if item['status'] != 'failed_validation':
            continue
        old = Path(item['path']) / 'measurement.json'
        measurement = read(old)
        if (measurement.get('csim') or {}).get('passed') is not True or (measurement.get('synthesis') or {}).get('success') is not True:
            continue
        if (measurement.get('cosim') or {}).get('error') != 'Cosim timed out after 3600s':
            continue
        parents.append({'kind': 'parent', 'id': item['benchmark'], 'benchmark': item['benchmark'],
                        'prior_measurement': str(old), 'prior_sha256': sha(old), 'estimate_cycles': (measurement.get('synthesis') or {}).get('report', {}).get('latency_cycles'), 'status': 'pending'})
    if not parents:
        raise ValueError('no eligible parent timeouts')
    parents.sort(key=lambda x: (x['estimate_cycles'] if x['estimate_cycles'] is not None else float('inf'), x['id']))
    historical = copy.deepcopy(original)
    historical['rows'] = []
    historical['raw_root'] = str(output / 'historical_timeout_replay_20261001' / 'raw')
    historical['replay_policy'] = {'reason': 'original_3600s_timeout', 'timeout_seconds': TIMEOUT,
                                   'prior_manifest_sha256': sha(HISTORY / 'manifest.json')}
    historic_tasks = {}
    for row in original['rows']:
        if row.get('inventory_status') != 'queued':
            continue
        old_id = row['validation_id']
        old_path = hist_raw / 'measurements' / f'{old_id}.json'
        if not old_path.is_file() or not is_original_timeout(read(old_path)):
            continue
        new_id = hashlib.sha256(f'c2hls-rtl-timeout-replay-v1:{TIMEOUT}:{old_id}:{sha(hist_worker)}'.encode()).hexdigest()
        new_row = copy.deepcopy(row)
        new_row['replay_of_validation_id'] = old_id
        new_row['replay_timeout_seconds'] = TIMEOUT
        new_row['validation_id'] = new_id
        historical['rows'].append(new_row)
        historic_tasks[new_id] = {'kind': 'historical', 'id': new_id, 'benchmark': row['benchmark'],
                                  'prior_id': old_id, 'prior_measurement': str(old_path),
                                  'prior_sha256': sha(old_path), 'estimate_cycles': row['csynth_cycles'], 'status': 'pending'}
    if len(historic_tasks) != 102 or len(historical['rows']) != 169:
        raise ValueError(f'unexpected timed-out historical cohort: {len(historic_tasks)} identities, {len(historical["rows"])} rows')
    output.mkdir(parents=True)
    hist_dir = output / 'historical_timeout_replay_20261001'
    (hist_dir / 'raw' / 'measurements').mkdir(parents=True)
    (hist_dir / 'raw' / 'logs').mkdir()
    (hist_dir / 'raw' / 'vitis_archives').mkdir()
    (output / 'parents').mkdir()
    (output / 'logs').mkdir()
    save(hist_dir / 'manifest.json', historical)
    manifest = {'schema_version': 'c2hls.sigmetrics-timeout-replay.v1', 'prepared_at': now(),
                'timeout_seconds': TIMEOUT, 'parent_workers': 2, 'historical_workers': 2,
                'main_manifest_sha256': sha(MAIN / 'manifest.json'),
                'historical_manifest_sha256': sha(HISTORY / 'manifest.json'),
                'historical_replay_manifest_sha256': sha(hist_dir / 'manifest.json'),
                'main_source': main['source'], 'historical_source': str(frozen_history),
                'main_parent_builder_sha256': sha(Path(main['source']) / 'scripts/build_sigmetrics_parent.py'),
                'historical_worker_sha256': sha(hist_worker), 'runner_sha256': sha(__file__),
                'tasks': parents + sorted(historic_tasks.values(), key=lambda x: (x['estimate_cycles'] if x['estimate_cycles'] is not None else float('inf'), x['id'])),
                'note': 'Sensitivity replay only; original frozen outcomes and 216-cell denominator are unchanged.'}
    save(output / 'manifest.json', manifest)
    state = {'status': 'prepared', 'updated_at': now(), 'tasks': manifest['tasks']}
    save(output / 'state.json', state)
    save(output / 'summary.json', summarize(state))
    print(json.dumps({'prepared': str(output), 'parents': len(parents),
                      'historical_validations': len(historic_tasks), 'historical_rows': len(historical['rows']),
                      'timeout_seconds': TIMEOUT}), flush=True)


def verify(output):
    manifest = read(output / 'manifest.json')
    if manifest['runner_sha256'] != sha(__file__) or manifest['timeout_seconds'] != TIMEOUT:
        raise ValueError('replay runner or policy changed')
    if sha(MAIN / 'manifest.json') != manifest['main_manifest_sha256']:
        raise ValueError('main manifest changed')
    if sha(HISTORY / 'manifest.json') != manifest['historical_manifest_sha256']:
        raise ValueError('historical manifest changed')
    if sha(output / 'historical_timeout_replay_20261001' / 'manifest.json') != manifest['historical_replay_manifest_sha256']:
        raise ValueError('replay historical manifest changed')
    main, _ = load_frozen_main()
    if sha(Path(main['source']) / 'scripts/build_sigmetrics_parent.py') != manifest['main_parent_builder_sha256']:
        raise ValueError('frozen parent builder changed')
    if sha(Path(manifest['historical_source']) / 'scripts/run_historical_cosim.py') != manifest['historical_worker_sha256']:
        raise ValueError('frozen historical worker changed')
    for task in manifest['tasks']:
        if sha(task['prior_measurement']) != task['prior_sha256']:
            raise ValueError('original timeout measurement changed: ' + task['id'])
    return manifest, main


def summary_counts(tasks, kind):
    return dict(collections.Counter(x['status'] for x in tasks if x['kind'] == kind))


def summarize(state):
    return {'status': state['status'], 'updated_at': state['updated_at'],
            'parent_counts': summary_counts(state['tasks'], 'parent'),
            'historical_counts': summary_counts(state['tasks'], 'historical'),
            'resource_gate': state.get('resource_gate'), 'stop_requested': state.get('stop_requested')}


def process_table():
    rows = subprocess.check_output(['ps', '-eo', 'pid=,ppid=,comm='], text=True)
    parsed = {}
    for line in rows.splitlines():
        pid, ppid, name = line.split(None, 2)
        parsed[int(pid)] = (int(ppid), name)
    return parsed


def descendant(pid, root, table):
    visited = set()
    while pid in table and pid not in visited and pid != root:
        visited.add(pid)
        pid = table[pid][0]
    return pid == root


def resource_gate(output):
    gib = 1024 ** 3
    memory = {r.split(':')[0]: int(r.split()[1]) * 1024 for r in Path('/proc/meminfo').read_text().splitlines()}
    observed = {'scratch_free_gib': shutil.disk_usage('/tmp').free / gib,
                'memory_available_gib': memory['MemAvailable'] / gib,
                'load_per_cpu': os.getloadavg()[0] / (os.cpu_count() or 1)}
    reasons = [k for k, limit in [('scratch_free_gib', 100), ('memory_available_gib', 128)] if observed[k] < limit]
    if observed['load_per_cpu'] >= 0.7:
        reasons.append('load_per_cpu')
    table = process_table()
    identity = read(MAIN / 'controller_identity.json')
    main_pid = int(identity['pid'])
    main_live = start_ticks(main_pid) == str(identity['start_ticks'])
    main_state = read(MAIN / 'state.json')
    if main_state['status'] == 'running' and not main_live:
        reasons.append('main_controller_identity_lost')
    unknown = [pid for pid, (_, name) in table.items() if name in VITIS_NAMES
               and not descendant(pid, os.getpid(), table)
               and not (main_live and descendant(pid, main_pid, table))]
    if unknown:
        reasons.append('unrelated_vitis')
    return {'allowed': not reasons, 'reasons': reasons, 'observed': observed,
            'main_controller_live': main_live, 'unrelated_vitis_pids': unknown}


def execute(command, cwd, log_path, env, limit):
    with Path(log_path).open('x') as stream:
        process = subprocess.Popen(command, cwd=cwd, stdout=stream, stderr=subprocess.STDOUT,
                                   env=env, start_new_session=True)
        try:
            return {'returncode': process.wait(timeout=limit), 'process_timed_out': False}
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            return {'returncode': 124, 'process_timed_out': True}


def run_task(output, task, main, main_module):
    if task['kind'] == 'parent':
        source = Path(main['source'])
        scratch = Path('/tmp') / output.name / ('parent_' + task['id'])
        env = main_module.base_environment(main, 'replay_' + task['id'] + '__parent')
        env.update(C2HLS_COSIM_TIMEOUT=str(TIMEOUT), C2HLS_TMP_ROOT=str(scratch),
                   C2HLS_VITIS_USER_HOME=str(scratch / 'vitis_home'), C2HLS_COSIM_TRACE_LEVEL='none')
        target = output / 'parents' / task['id']
        command = [sys.executable, '-u', str(source / 'scripts/build_sigmetrics_parent.py'),
                   '--inputs', main['inputs'], '--benchmark', task['benchmark'], '--output', str(target)]
        limit = 60000
    else:
        source = Path(read(HISTORY / 'launch_receipt.json')['source_snapshot'])
        env = {k: v for k, v in os.environ.items() if not k.startswith('C2HLS_')}
        hist_output = output / 'historical_timeout_replay_20261001'
        command = [sys.executable, '-u', str(source / 'scripts/run_historical_cosim.py'), 'worker',
                   '--output', str(hist_output), '--validation-id', task['id'], '--timeout', str(TIMEOUT)]
        target = hist_output / 'raw' / 'measurements' / (task['id'] + '.json')
        limit = 12200
    execution = execute(command, source, output / 'logs' / (task['kind'] + '_' + task['id'] + '.log'), env, limit)
    measurement_path = target / 'measurement.json' if task['kind'] == 'parent' else target
    if not measurement_path.is_file():
        return {'status': 'execution_error', 'execution': execution, 'measurement_path': None}
    measurement = read(measurement_path)
    if task['kind'] == 'parent':
        status = 'pass' if measurement.get('success') is True else ('timeout' if 'timed out' in str((measurement.get('cosim') or {}).get('error', '')).lower() else 'fail')
        if status == 'pass' and not (target / 'phase_b_manifest.json').is_file():
            status = 'execution_error'
    else:
        status = (measurement.get('cosim') or {}).get('status', 'execution_error')
        if measurement.get('validation_id') != task['id'] or measurement.get('kernel_sha256') is None:
            status = 'execution_error'
    return {'status': status, 'execution': execution, 'measurement_path': str(measurement_path),
            'measurement_sha256': sha(measurement_path)}


def run(output):
    lock = (output / 'controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    manifest, main = verify(output)
    state = read(output / 'state.json')
    if state['status'] != 'prepared':
        raise ValueError('replay already started; no implicit rerolls')
    main_state = read(MAIN / 'state.json')
    if main_state['status'] != 'running':
        raise ValueError('frozen main study is not running')
    _, main_module = load_frozen_main()
    stopping = {}
    def on_stop(signum, _frame):
        stopping.update(signal=signum, received_at=now())
    signal.signal(signal.SIGTERM, on_stop)
    signal.signal(signal.SIGINT, on_stop)
    state.update(status='running', started_at=now(), pid=os.getpid(), pid_start_ticks=start_ticks(os.getpid()))
    save(output / 'state.json', state)
    active = {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        while True:
            gate = resource_gate(output)
            state['resource_gate'] = gate
            if stopping:
                state['stop_requested'] = stopping
            for kind, slots in [('parent', 2), ('historical', 2)]:
                active_kind = sum(t['kind'] == kind for t in active.values())
                for task in state['tasks']:
                    if active_kind >= slots or stopping or not gate['allowed']:
                        break
                    if task['kind'] != kind or task['status'] != 'pending':
                        continue
                    if sha(task['prior_measurement']) != task['prior_sha256']:
                        raise ValueError('original measurement changed: ' + task['id'])
                    task.update(status='running', started_at=now())
                    active[pool.submit(run_task, output, task.copy(), main, main_module)] = task
                    active_kind += 1
            state['updated_at'] = now()
            save(output / 'state.json', state)
            save(output / 'summary.json', summarize(state))
            if not active:
                if stopping:
                    break
                if all(t['status'] != 'pending' for t in state['tasks']):
                    break
                time.sleep(20)
                continue
            finished, _ = wait(active, timeout=20, return_when=FIRST_COMPLETED)
            for future in finished:
                task = active.pop(future)
                try:
                    result = future.result()
                except Exception as exc:
                    result = {'status': 'execution_error', 'exception': f'{type(exc).__name__}: {exc}'}
                task.update(result, finished_at=now())
                print('REPLAY_DONE', task['kind'], task['id'], task['status'], flush=True)
    state.update(status='stopped' if stopping else 'complete', updated_at=now(), finished_at=now())
    save(output / 'state.json', state)
    save(output / 'summary.json', summarize(state))
    print('REPLAY_CAMPAIGN_DONE', json.dumps(summarize(state)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'run', 'status'))
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    if args.action == 'prepare':
        prepare(output)
    elif args.action == 'run':
        run(output)
    else:
        print(json.dumps(summarize(read(output / 'state.json')), indent=2))


if __name__ == '__main__':
    main()
