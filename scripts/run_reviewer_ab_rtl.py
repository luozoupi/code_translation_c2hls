#!/usr/bin/env python3
"""Independent, one-worker RTL check of the frozen reviewer A/B first repeat."""
from __future__ import annotations

import argparse
import collections
import fcntl
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path('/home/luo00466/code_translation-c2hls-hpca2027/results_sweeps/deepseek_flash_reviewer_ab27_20260925')
HISTORY = ROOT / 'results_sweeps/historical_registry_recovery_20260928'
MAIN = ROOT / 'results_sweeps/sigmetrics_recipe_main_20260930'
REPLAY = ROOT / 'results_sweeps/sigmetrics_timeout_replay_20261001'
OUTPUT = ROOT / 'results_sweeps/reviewer_ab_rtl_r1_20261001'
VITIS_NAMES = {'vitis_hls', 'vitis-run', 'xsim', 'xsimk', 'vivado'}
TIMEOUT = 3600


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + '\n')
    tmp.replace(path)


def ticks(pid):
    try:
        return Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[19]
    except (OSError, IndexError):
        return None


def frozen_worker():
    receipt = read(HISTORY / 'launch_receipt.json')
    source = Path(receipt['source_snapshot'])
    manifest = read(HISTORY / 'manifest.json')
    for name, sha in manifest['code_hashes'].items():
        if digest(source / name) != sha:
            raise ValueError('frozen evaluator changed: ' + name)
    return source, manifest


def benchmark_files(bench):
    meta = read(bench / 'metadata.json')
    names = {'metadata.json', 'plain.cpp', meta.get('header_file') or 'kernel.h',
             meta.get('testbench_file') or 'testbench.cpp', *(meta.get('support_files') or [])}
    names.update(str(p.relative_to(bench)) for p in bench.glob('*.h'))
    names.update(str(p.relative_to(bench)) for p in (bench / 'support').rglob('*') if p.is_file())
    if any(Path(name).is_absolute() or '..' in Path(name).parts for name in names):
        raise ValueError('unsafe benchmark input path')
    return meta, {n: digest(bench / n) for n in sorted(names)}


def prepare(output):
    if output.exists():
        raise FileExistsError(output)
    source, historical = frozen_worker()
    upstream_manifest = read(SOURCE / 'manifest.json')
    arms = read(SOURCE / 'arm_measurements.json')
    chosen = [a for a in arms if a['repeat'] == 1]
    if len(chosen) != 108 or len({a['pair_id'] for a in chosen}) != 54:
        raise ValueError('first-repeat cohort changed')
    registry_rel = 'configs/hlsfactory_output_shapes_fc27133.json'
    registry = read(source / registry_rel)
    registry_sha = digest(source / registry_rel)
    raw = Path('/mnt/data2/luo00466/c2hls_rl/reviewer_ab_rtl_r1_20261001/raw')
    if raw.exists():
        raise FileExistsError(raw)
    staged = []
    rows = []
    by_id = {}
    try:
        raw.mkdir(parents=True)
        for a in chosen:
            if not a['audit_ok'] or not a['feasible']:
                raise ValueError('chosen arm is not audit-valid and feasible: ' + a['cell_id'])
            result_path = Path(a['result_path'])
            result = read(result_path)
            code = result['hls_code']
            code_sha = hashlib.sha256(code.encode()).hexdigest()
            if code_sha != a['selected_code_sha256'] or result['benchmark'] != a['benchmark']:
                raise ValueError('selected source mismatch: ' + a['cell_id'])
            if a['selected_code_sha256'] == a['seed_code_sha256']:
                if result['baseline_report']['latency_cycles_worst'] != a['cycles']:
                    raise ValueError('selected parent report mismatch: ' + a['cell_id'])
            else:
                events = (result.get('synthesis_evaluations') or {}).get('events') or []
                if not any(e.get('code_sha256') == code_sha and e.get('success') is True
                           and e.get('synthesized_latency_cycles') == a['cycles'] for e in events):
                    raise ValueError('selected synthesis event mismatch: ' + a['cell_id'])
            golden = result.get('independent_golden') or {}
            if golden.get('status') != 'passed' or (result.get('csim') or {}).get('passed') is not True:
                raise ValueError('prior native/CSim failed: ' + a['cell_id'])
            bench = Path(upstream_manifest['config']['benchmarks_root']) / a['benchmark']
            meta, hashes = benchmark_files(bench)
            saved = {f['path']: f['sha256'] for f in result['run_fingerprint']['payload']['benchmark']['inputs']['files']}
            if any(saved.get(n) != sha for n, sha in hashes.items()):
                raise ValueError('historical harness changed: ' + a['cell_id'])
            tb_name = meta.get('testbench_file') or 'testbench.cpp'
            reg = registry['benchmarks'][a['benchmark']]
            if hashes[tb_name] != reg['testbench_sha256']:
                raise ValueError('registry/testbench mismatch: ' + a['cell_id'])
            if golden['output_sha256'] != result['csim']['golden_output_sha256']:
                raise ValueError('prior native/CSim golden mismatch: ' + a['cell_id'])
            identity = {'benchmark': a['benchmark'], 'code_sha256': code_sha,
                        'input_hashes': hashes, 'registry_sha256': registry_sha, 'timeout': TIMEOUT}
            validation_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
            target = raw / 'inputs' / validation_id
            if validation_id not in by_id:
                for name, sha in hashes.items():
                    dest = target / 'benchmark' / name
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(bench / name, dest)
                    if digest(dest) != sha:
                        raise ValueError('staged harness hash mismatch')
                (target / 'kernel.cpp').write_text(code)
                if digest(target / 'kernel.cpp') != code_sha:
                    raise ValueError('staged code hash mismatch')
                staged.append(str(target))
                by_id[validation_id] = []
            by_id[validation_id].append(a['cell_id'])
            rows.append({'validation_id': validation_id, 'inventory_status': 'queued',
                         'record_id': a['cell_id'], 'cell_id': a['cell_id'],
                         'pair_id': a['pair_id'], 'benchmark': a['benchmark'],
                         'strategy': a['strategy'], 'arm': a['arm'], 'repeat': 1,
                         'model': upstream_manifest['config']['model'], 'kind': 'agentic',
                         'skill_mode': upstream_manifest['config']['skill_mode'],
                         'campaign': 'deepseek_flash_reviewer_ab27_20260925',
                         'source_summary': str(SOURCE / 'arm_measurements.json'),
                         'source_result': str(result_path), 'source_result_sha256': digest(result_path),
                         'source_evidence': 'saved_selected_code_and_matching_synthesis_event_or_parent',
                         'historical_reference_audit_passed': True,
                         'historical_golden_output_sha256': golden['output_sha256'],
                         'csynth_cycles': a['cycles'], 'kernel_sha256': code_sha,
                         'staged_kernel': str(target / 'kernel.cpp'),
                         'staged_benchmark': str(target / 'benchmark'), 'input_hashes': hashes,
                         'golden_registry': {'path': registry_rel, 'sha256': registry_sha,
                                             'testbench_sha256': hashes[tb_name]}})
        for pair in {a['pair_id'] for a in chosen}:
            pair_rows = [r for r in rows if r['pair_id'] == pair]
            if {r['arm'] for r in pair_rows} != {'review_off', 'review_on'} or len(pair_rows) != 2:
                raise ValueError('unpaired cohort entry: ' + pair)
        tasks = [{'validation_id': key, 'cells': cells, 'status': 'pending',
                  'csynth_cycles': min(r['csynth_cycles'] for r in rows if r['validation_id'] == key)}
                 for key, cells in by_id.items()]
        tasks.sort(key=lambda t: (t['csynth_cycles'], t['validation_id']))
        raw.joinpath('measurements').mkdir()
        raw.joinpath('logs').mkdir()
        raw.joinpath('vitis_archives').mkdir()
        manifest = {'schema_version': 'c2hls.reviewer-ab-rtl-r1.v1', 'prepared_at': now(),
                    'cohort': 'all 27 kernels x flash/dynamic x review off/on x first repeat',
                    'source_manifest_sha256': digest(SOURCE / 'manifest.json'),
                    'arm_measurements_sha256': digest(SOURCE / 'arm_measurements.json'),
                    'frozen_worker': str(source / 'scripts/run_historical_cosim.py'),
                    'frozen_worker_sha256': digest(source / 'scripts/run_historical_cosim.py'),
                    'code_hashes': historical['code_hashes'], 'toolchain': historical['toolchain'],
                    'raw_root': str(raw), 'rows': rows, 'timeout_seconds': TIMEOUT}
        output.mkdir(parents=True)
        save(output / 'manifest.json', manifest)
        state = {'status': 'prepared', 'updated_at': now(), 'tasks': tasks}
        save(output / 'state.json', state)
        save(output / 'summary.json', summarize(state))
        print(json.dumps({'output': str(output), 'pairs': 54, 'arms': len(rows),
                          'distinct_rtl_jobs': len(tasks), 'raw': str(raw)}), flush=True)
    except Exception:
        for path in staged:
            shutil.rmtree(path, ignore_errors=True)
        if raw.exists():
            shutil.rmtree(raw)
        if output.exists():
            shutil.rmtree(output)
        raise


def summarize(state):
    return {'status': state['status'], 'updated_at': state['updated_at'],
            'jobs': dict(collections.Counter(t['status'] for t in state['tasks'])),
            'resource_gate': state.get('resource_gate'), 'stop_requested': state.get('stop_requested')}


def process_table():
    table = {}
    for line in subprocess.check_output(['ps', '-eo', 'pid=,ppid=,comm='], text=True).splitlines():
        pid, ppid, name = line.split(None, 2)
        table[int(pid)] = (int(ppid), name)
    return table


def descendant(pid, root, table):
    seen = set()
    while pid in table and pid not in seen and pid != root:
        seen.add(pid)
        pid = table[pid][0]
    return pid == root


def gate():
    gib = 1024 ** 3
    memory = {line.split(':')[0]: int(line.split()[1]) * 1024
              for line in Path('/proc/meminfo').read_text().splitlines()}
    observed = {'scratch_free_gib': shutil.disk_usage('/tmp').free / gib,
                'memory_available_gib': memory['MemAvailable'] / gib,
                'load_per_cpu': os.getloadavg()[0] / (os.cpu_count() or 1)}
    reasons = []
    if observed['scratch_free_gib'] < 100 or observed['memory_available_gib'] < 128:
        reasons.append('scratch_or_memory')
    if observed['load_per_cpu'] >= 0.7:
        reasons.append('cpu_load')
    main_id = read(MAIN / 'controller_identity.json')
    main_state = read(MAIN / 'state.json')
    replay_state = read(REPLAY / 'state.json')
    main_pid = int(main_id['pid'])
    replay_pid = int(replay_state['pid'])
    main_live = ticks(main_pid) == str(main_id['start_ticks'])
    replay_live = ticks(replay_pid) == str(replay_state['pid_start_ticks'])
    if main_state['status'] == 'running' and not main_live:
        reasons.append('main_controller_identity_lost')
    if replay_state['status'] == 'running' and not replay_live:
        reasons.append('replay_controller_identity_lost')
    occupied = sum(c['status'] == 'running' for c in main_state['cells'])
    occupied += sum(p['status'] == 'running' for p in main_state['parents'])
    occupied += sum(t['status'] == 'running' for t in replay_state['tasks'])
    if occupied >= 6:
        reasons.append('six_job_cap')
    # The frozen replay controller pauses replacement jobs while this worker
    # runs. Give its pending historical queue the chance to refill first.
    replay_history = [t for t in replay_state['tasks'] if t['kind'] == 'historical']
    if (replay_state['status'] == 'running'
            and any(t['status'] == 'pending' for t in replay_history)
            and sum(t['status'] == 'running' for t in replay_history) < 2):
        reasons.append('replay_historical_refill_priority')
    table = process_table()
    known = [os.getpid()]
    if main_live:
        known.append(main_pid)
    if replay_live:
        known.append(replay_pid)
    unknown = [pid for pid, (_, name) in table.items() if name in VITIS_NAMES
               and not any(descendant(pid, root, table) for root in known)]
    if unknown:
        reasons.append('unrelated_vitis')
    return {'allowed': not reasons, 'reasons': reasons, 'observed': observed,
            'other_active_jobs': occupied, 'unrelated_vitis_pids': unknown}


def verify(output):
    manifest = read(output / 'manifest.json')
    if digest(SOURCE / 'manifest.json') != manifest['source_manifest_sha256']:
        raise ValueError('original reviewer manifest changed')
    if digest(SOURCE / 'arm_measurements.json') != manifest['arm_measurements_sha256']:
        raise ValueError('original reviewer measurements changed')
    source, _ = frozen_worker()
    if digest(source / 'scripts/run_historical_cosim.py') != manifest['frozen_worker_sha256']:
        raise ValueError('worker changed')
    for row in manifest['rows']:
        if digest(row['source_result']) != row['source_result_sha256']:
            raise ValueError('original result changed: ' + row['cell_id'])
        if digest(row['staged_kernel']) != row['kernel_sha256']:
            raise ValueError('staged kernel changed: ' + row['cell_id'])
        for name, sha in row['input_hashes'].items():
            if digest(Path(row['staged_benchmark']) / name) != sha:
                raise ValueError('staged harness changed: ' + row['cell_id'])
    return manifest, source


def run(output):
    lock = (output / 'controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    manifest, source = verify(output)
    state = read(output / 'state.json')
    if state['status'] not in {'prepared', 'stopped'} or any(t['status'] == 'running' for t in state['tasks']):
        raise ValueError('cohort is active or has unresolved running work')
    stop = {}
    def on_stop(signum, _frame):
        stop.update(signal=signum, received_at=now())
    signal.signal(signal.SIGTERM, on_stop)
    signal.signal(signal.SIGINT, on_stop)
    state.setdefault('restarts', []).append({'at': now(), 'previous_status': state['status']})
    state.update(status='running', started_at=now(), pid=os.getpid(), pid_start_ticks=ticks(os.getpid()), stop_requested=None)
    save(output / 'state.json', state)
    raw = Path(manifest['raw_root'])
    last_finished = None
    for task in state['tasks']:
        if task['status'] != 'pending':
            continue
        # Give the frozen replay controller at least one full 20-second poll
        # without a third-session Vitis process before admitting another job.
        while last_finished is not None and time.monotonic() - last_finished < 35 and not stop:
            time.sleep(5)
        if stop:
            break
        while True:
            resource = gate()
            state.update(resource_gate=resource, updated_at=now())
            if stop:
                state['stop_requested'] = stop
            save(output / 'state.json', state)
            save(output / 'summary.json', summarize(state))
            if resource['allowed'] or stop:
                break
            time.sleep(20)
        if stop:
            break
        task.update(status='running', started_at=now())
        save(output / 'state.json', state)
        key = task['validation_id']
        command = [sys.executable, '-u', manifest['frozen_worker'], 'worker',
                   '--output', str(output), '--validation-id', key, '--timeout', str(TIMEOUT)]
        with (raw / 'logs' / f'{key}.log').open('x') as stream:
            proc = subprocess.Popen(command, cwd=source, stdout=stream,
                                    stderr=subprocess.STDOUT, start_new_session=True)
            try:
                rc = proc.wait(timeout=5400)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
                rc = 124
        measurement = raw / 'measurements' / f'{key}.json'
        if measurement.is_file():
            measured = read(measurement)
            cosim = measured.get('cosim') or {}
            status = cosim.get('status', 'execution_error')
            if measured.get('kernel_sha256') != next(r['kernel_sha256'] for r in manifest['rows'] if r['validation_id'] == key):
                status = 'execution_error'
            task.update(status=status, measurement_path=str(measurement),
                        measurement_sha256=digest(measurement), returncode=rc, finished_at=now())
        else:
            task.update(status='execution_error', returncode=rc, finished_at=now())
        state['updated_at'] = now()
        save(output / 'state.json', state)
        save(output / 'summary.json', summarize(state))
        print('RTL_DONE', key, task['status'], task['cells'], flush=True)
        last_finished = time.monotonic()
    if stop:
        state['stop_requested'] = stop
    state.update(status='stopped' if stop else 'complete', updated_at=now(), finished_at=now())
    save(output / 'state.json', state)
    save(output / 'summary.json', summarize(state))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'verify', 'gate', 'run', 'wait-resume', 'status'))
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    if args.action == 'prepare':
        prepare(output)
    elif args.action == 'verify':
        manifest, _ = verify(output)
        print(json.dumps({'verified_rows': len(manifest['rows']),
                          'distinct_jobs': len({r['validation_id'] for r in manifest['rows']})}))
    elif args.action == 'gate':
        print(json.dumps(gate(), indent=2))
    elif args.action == 'run':
        run(output)
    elif args.action == 'wait-resume':
        while True:
            current = read(output / 'state.json')['status']
            if current == 'stopped':
                break
            if current == 'complete':
                return
            if current != 'running':
                raise ValueError('unexpected state while waiting for drain: ' + current)
            time.sleep(15)
        run(output)
    else:
        print(json.dumps(summarize(read(output / 'state.json')), indent=2))


if __name__ == '__main__':
    main()
