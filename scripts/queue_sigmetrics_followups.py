#!/usr/bin/env python3
"""Drain existing studies, then run one July backfill and one pass-within-K lane.

Only this queue's frozen workers and scratch are owned by this scheduler. No
existing study is stopped, modified, reprioritized, or given a new time limit.
"""
from __future__ import annotations
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 'c2hls.sigmetrics-followup-queue.v1'
VITIS_NAMES = {'vitis_hls', 'vitis-run', 'xsim', 'xsimk', 'vivado', 'xelab'}


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(data, indent=2, sort_keys=True) + '\n')
    temp.replace(path)


def helper(source):
    path = Path(source) / 'scripts/resume_sigmetrics_recipe_main.py'
    spec = importlib.util.spec_from_file_location('queue_owned_process', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def alive(identity):
    try:
        proc = Path('/proc') / str(identity['pid'])
        fields = (proc / 'stat').read_text().rsplit(')', 1)[1].split()
        return (Path('/proc/sys/kernel/random/boot_id').read_text().strip() == identity['boot_id']
                and fields[19] == str(identity['start_ticks']) and fields[0] != 'Z'
                and str(identity['source_runner']).encode() in (proc / 'cmdline').read_bytes().split(b'\0'))
    except (OSError, KeyError, IndexError):
        return False


def dependencies(config):
    observed = []
    for folder in config['wait_for_campaigns']:
        p = Path(folder)
        summary = read(p / 'summary.json')
        identity = read(p / 'controller_identity.json')
        live = alive(identity)
        terminal = summary['status'] in ('complete', 'complete_with_quarantined_failure', 'complete_with_failures')
        observed.append({'campaign': folder, 'status': summary['status'], 'controller_alive': live,
                         'drained': terminal and not live})
    return {'allowed': all(x['drained'] for x in observed), 'campaigns': observed}


def resource_gate(owned_helper):
    table = owned_helper.process_table()
    own = {os.getpid(): table[os.getpid()]} if os.getpid() in table else {}
    owned_helper.capture_descendants(own, table)
    unrelated = []
    for pid, row in table.items():
        if pid in own or row['state'] == 'Z':
            continue
        try:
            name = Path(f'/proc/{pid}/comm').read_text().strip()
        except OSError:
            continue
        if name in VITIS_NAMES:
            unrelated.append(pid)
    mem = {}
    for line in Path('/proc/meminfo').read_text().splitlines():
        pieces = line.split()
        mem[pieces[0].rstrip(':')] = int(pieces[1])
    observed = {'scratch_free_gib': shutil.disk_usage('/tmp').free / 1024**3,
                'output_free_gib': shutil.disk_usage(ROOT).free / 1024**3,
                'memory_available_gib': mem['MemAvailable'] / 1024**2,
                'load_per_cpu': os.getloadavg()[0] / (os.cpu_count() or 1)}
    reasons = []
    if min(observed['scratch_free_gib'], observed['output_free_gib']) < 100:
        reasons.append('free_storage_below_100_GiB')
    if observed['memory_available_gib'] < 128:
        reasons.append('available_memory_below_128_GiB')
    if observed['load_per_cpu'] > 0.7:
        reasons.append('host_load_above_0.7_per_cpu')
    if unrelated:
        reasons.append('unrelated_vitis_processes')
    return {'allowed': not reasons, 'reasons': reasons, 'observed': observed, 'unrelated_vitis_pids': unrelated}


def next_group(groups):
    eligible = [g for g in groups if g['status'] == 'pending']
    return min(eligible, key=lambda g: (g['next_attempt'], g['id'])) if eligible else None


def apply_attempt(group, result, k):
    if group['status'] != 'running' or result['attempt'] != group['next_attempt']:
        raise ValueError('Unexpected attempt identity or duplicate completion')
    group['attempts'].append(result)
    if result['status'] == 'pass':
        group.update(status='pass', passed_on_attempt=result['attempt'])
    elif result['status'] in ('execution_error', 'evidence_error'):
        group.update(status=result['status'])
    elif result['status'] in ('fail', 'timeout'):
        if result['attempt'] >= k:
            group.update(status='exhausted')
        else:
            group.update(status='pending', next_attempt=result['attempt'] + 1)
    else:
        raise ValueError('Invalid attempt outcome')


def summary(state, config):
    historical = state['historical_tasks']
    groups = state['passk_groups']
    curves = {}
    for mode in config['passk_modes']:
        selected = [g for g in groups if g['mode'] == mode]
        curves[mode] = {'denominator_all_packaged_kernels': len(selected),
                        'observed_successes_within_k': {str(k): sum(g.get('passed_on_attempt', 999) <= k for g in selected)
                                                       for k in range(1, config['k']+1)},
                        'group_counts': dict(Counter(g['status'] for g in selected)),
                        'interpretation': 'Interim observed successes, not an IID pass@k estimator; unattempted and input-blocked kernels stay visible.'}
    return {'status': state['status'], 'updated_at': state['updated_at'],
            'historical_distinct_validations': len(historical),
            'historical_counts': dict(Counter(t['status'] for t in historical)),
            'passk_group_counts': dict(Counter(g['status'] for g in groups)),
            'passk_attempt_counts': dict(Counter(a['status'] for g in groups for a in g['attempts'])),
            'success_within_k': curves, 'paused_lanes': state['paused_lanes'],
            'dependency_gate': state.get('dependency_gate'), 'resource_gate': state.get('resource_gate'),
            'active': [{'lane': 'backfill', 'id': t['validation_id']} for t in historical if t['status']=='running'] +
                      [{'lane': 'passk', 'id': g['id'], 'attempt': g['next_attempt']} for g in groups if g['status']=='running']}


def persist(output, state, config):
    state['updated_at'] = now()
    save(output / 'state.json', state)
    save(output / 'summary.json', summary(state, config))


def prepare(output, historical, passk):
    if output.exists():
        raise FileExistsError(output)
    h, p = read(historical / 'manifest.json'), read(passk / 'manifest.json')
    hs = read(historical / 'state.json')
    ps = read(passk / 'state.json')
    preflight = read(passk / 'input_preflight.json')
    if not preflight['complete'] or len(preflight['records']) != 28:
        raise ValueError('All 28 public-oracle input checks must finish before queuing')
    if preflight.get('manifest_sha256') != sha(passk/'manifest.json'):
        raise ValueError('Input preflight does not match this frozen passK protocol')
    if hs['status'] != 'prepared' or ps['status'] != 'prepared':
        raise ValueError('Refusing to requeue a campaign that has started')
    if any((path/'queue_registration.json').exists() for path in (historical, passk)):
        raise ValueError('Campaign already belongs to another queue')
    output.mkdir(parents=True)
    source = output / 'source/scripts'
    source.mkdir(parents=True)
    shutil.copy2(Path(__file__), source / Path(__file__).name)
    shutil.copy2(ROOT / 'scripts/resume_sigmetrics_recipe_main.py', source / 'resume_sigmetrics_recipe_main.py')
    failed_inputs = {r['benchmark']: r for r in preflight['records'] if not r['success']}
    groups = ps['groups']
    for group in groups:
        if group['benchmark'] in failed_inputs:
            group.update(status='blocked_input', input_preflight=failed_inputs[group['benchmark']])
    pinned = [historical/'manifest.json', historical/'input_audit.json', passk/'manifest.json', passk/'input_preflight.json']
    pinned += list(source.glob('*.py'))
    config = {'schema_version': SCHEMA, 'prepared_at': now(), 'historical': str(historical), 'passk': str(passk),
              'historical_worker': h['frozen_worker'], 'historical_source': h['frozen_source'],
              'source': str(source.parent), 'python': p['python'], 'k': p['k'], 'passk_modes': p['modes'],
              'workers': {'backfill': 1, 'passk': 1},
              'wait_for_campaigns': [str(ROOT/'results_sweeps'/n) for n in
                                    ('sigmetrics_main_continuation_20261006', 'sigmetrics_timeout_recovery_20261006')],
              'policy': 'Start only after both existing campaigns complete and controllers exit; one worker per lane, no peer interruption. Stop a lane on execution/accounting uncertainty; retain and count measured nonpasses. No automatic reboot restart of started tasks.',
              'files': {str(path): sha(path) for path in pinned}}
    save(output / 'manifest.json', config)
    state = {'status': 'queued', 'updated_at': now(), 'historical_tasks': hs['tasks'],
             'passk_groups': groups, 'paused_lanes': {}, 'manifest_sha256': sha(output/'manifest.json')}
    persist(output, state, config)
    for path in (historical, passk):
        save(path/'queue_registration.json', {'coordinator': str(output),
             'coordinator_manifest_sha256': sha(output/'manifest.json'),
             'canonical_state': str(output/'state.json'), 'canonical_summary': str(output/'summary.json'),
             'registered_at': now(), 'status': 'queued_under_coordinator',
             'note': 'Local prepared state is the immutable launch inventory; live task states are owned by this coordinator.'})
    print(json.dumps({'queued_historical': len(hs['tasks']), 'passk_groups': len(groups),
                      'input_blocked_groups': sum(g['status']=='blocked_input' for g in groups),
                      'source_runner': str(source/Path(__file__).name)}))


def verify(output):
    config = read(output/'manifest.json')
    for path, digest in config['files'].items():
        if sha(path) != digest:
            raise ValueError('Queued input changed: ' + path)
    h = read(Path(config['historical'])/'manifest.json')
    for name, digest in h['code_hashes'].items():
        if sha(Path(h['frozen_source'])/name) != digest:
            raise ValueError('Historical evaluator changed: ' + name)
    p = read(Path(config['passk'])/'manifest.json')
    for path, digest in p['frozen_files'].items():
        if sha(path) != digest:
            raise ValueError('PassK frozen input changed: ' + path)
    return config


def archive_attempt(passk, group, index):
    scratch = passk/'scratch'/(group['id']+f'__attempt{index}')
    if not scratch.exists():
        return None
    destination = passk/'vitis_archives'/(group['id']+f'__attempt{index}.tar.zst')
    destination.parent.mkdir(exist_ok=True)
    if destination.exists():
        raise FileExistsError(destination)
    partial = destination.with_suffix('.partial')
    subprocess.run(['tar','--zstd','-cf',str(partial),'-C',str(scratch),'.'], check=True, timeout=900)
    subprocess.run(['tar','--zstd','--compare','-f',str(partial),'-C',str(scratch)], check=True, timeout=900,
                   stdout=subprocess.DEVNULL)
    digest = sha(partial)
    partial.replace(destination)
    receipt = {'path': str(destination), 'sha256': digest, 'source_scratch': str(scratch), 'tar_compare_passed': True}
    save(destination.with_suffix('.receipt.json'), receipt)
    shutil.rmtree(scratch)
    return receipt


def historical_status(measurement, execution):
    c = measurement.get('cosim') or {}
    if execution['returncode'] != 0 or execution.get('cleanup_survivors') or measurement.get('archive_error'):
        return 'execution_error'
    status = c.get('status')
    if status == 'not_run':
        # A failed C-output gate means RTL was not executed. It must consume this
        # validation identity without becoming an RTL failure or a silent retry.
        if c.get('skip_reason') == 'candidate_csim_failed':
            return 'failed_csim'
        if c.get('skip_reason') == 'invalid_native_golden':
            return 'blocked_input'
        return 'execution_error'
    if status == 'pass' and not (c.get('passed') is True and c.get('success') is True
                                and (c.get('kernel_runtime_cycles') or 0)>0
                                and (c.get('correctness') or {}).get('passed') is True):
        return 'evidence_error'
    return status if status in ('pass','fail','timeout') else 'execution_error'


def run_historical(output, config, task, owned):
    h = read(Path(config['historical'])/'manifest.json')
    vid = task['validation_id']
    measurement = Path(h['raw_root'])/'measurements'/(vid+'.json')
    if measurement.exists():
        raise FileExistsError('Refusing to rerun existing measurement')
    scratch = Path('/tmp')/Path(config['historical']).name/vid
    if scratch.exists():
        raise FileExistsError('Refusing to reuse interrupted historical scratch')
    command = [config['python'], config['historical_worker'], 'worker', '--output', config['historical'],
               '--validation-id', vid, '--timeout', str(h['timeout_seconds'])]
    env = {k:v for k,v in os.environ.items() if not k.startswith('C2HLS_')}
    result = owned.execute_owned(command, config['historical_source'], output/'logs'/('backfill_'+vid+'.log'), env, 5400)
    if not measurement.exists():
        return {'status': 'execution_error', 'execution': result, 'error': 'missing_measurement'}
    m = read(measurement); c = m.get('cosim') or {}
    row = next(r for r in h['rows'] if r['validation_id'] == vid)
    if m.get('validation_id') != vid or m.get('kernel_sha256') != row['kernel_sha256']:
        raise ValueError('Historical measurement identity mismatch')
    return {'status': historical_status(m, result), 'recorded_cosim_status': c.get('status'),
            'cosim_skip_reason': c.get('skip_reason'), 'execution': result,
            'measurement_path': str(measurement), 'measurement_sha256': sha(measurement)}


def run_passk(output, config, group, owned):
    passk = Path(config['passk'])
    p = read(passk/'manifest.json')
    index = group['next_attempt']
    resultpath = passk/'attempts'/group['id']/str(index)/'result.json'
    if resultpath.parent.exists():
        raise FileExistsError('Refusing to rerun a started attempt')
    command = [p['python'], str(Path(p['source'])/'scripts/run_polybench_passk.py'), 'attempt', '--output', str(passk),
               '--group', group['id'], '--attempt', str(index)]
    env = {k:v for k,v in os.environ.items() if not k.startswith('C2HLS_')}
    execution = owned.execute_owned(command, p['source'], output/'logs'/(group['id']+f'__attempt{index}.log'), env,
                                    p['timeouts']['attempt_outer'])
    if execution.get('cleanup_survivors'):
        return {'attempt': index, 'status': 'execution_error', 'execution': execution,
                'error': 'owned_processes_remain_scratch_preserved'}
    if not resultpath.exists():
        return {'attempt': index, 'status': 'execution_error', 'execution': execution, 'error': 'missing_attempt_result'}
    result = read(resultpath)
    outcome = result['passk_attempt']
    if outcome['group_id'] != group['id'] or outcome['attempt'] != index:
        raise ValueError('PassK attempt identity mismatch')
    if execution['returncode'] != 0 and outcome['status'] not in ('execution_error','evidence_error'):
        outcome = dict(outcome, status='execution_error')
    archive = archive_attempt(passk, group, index)
    return {**outcome, 'execution': execution, 'result_path': str(resultpath), 'result_sha256': sha(resultpath), 'archive': archive}


def run(output):
    config = verify(output)
    if Path(__file__).resolve() != Path(config['source'])/'scripts/queue_sigmetrics_followups.py':
        raise ValueError('Launch the frozen scheduler recorded in the queue manifest')
    owned = helper(config['source'])
    state = read(output/'state.json')
    if state['status'] != 'queued':
        raise ValueError('Only an unstarted queued campaign can launch; interrupted attempts require explicit recovery')
    with (output/'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        info = owned.process_table()[os.getpid()]
        save(output/'controller_identity.json', {'pid': os.getpid(), 'start_ticks': info['start_ticks'],
             'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
             'source_runner': str(Path(__file__).resolve()), 'source_runner_sha256': sha(__file__),
             'manifest_sha256': sha(output/'manifest.json')})
        stopping = False
        def stop(*_):
            nonlocal stopping
            stopping = True
        for sig in (signal.SIGINT, signal.SIGTERM): signal.signal(sig, stop)
        (output/'logs').mkdir(exist_ok=True)
        # Hold lane locks too: these manifests may not run under another controller.
        lane_locks = []
        for path in (Path(config['historical']), Path(config['passk'])):
            handle = (path/'controller.lock').open('a')
            fcntl.flock(handle, fcntl.LOCK_EX|fcntl.LOCK_NB); lane_locks.append(handle)
        legacy_queue_lock = (Path(config['historical'])/'queue.lock').open('a')
        fcntl.flock(legacy_queue_lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        lane_locks.append(legacy_queue_lock)
        state.update(status='waiting_for_existing_campaigns', started_at=now(), pid=os.getpid())
        with ThreadPoolExecutor(max_workers=2) as pool:
            active = {}
            admitted = False
            while True:
                if not admitted:
                    state['dependency_gate'] = dependencies(config)
                    admitted = state['dependency_gate']['allowed']
                state['resource_gate'] = resource_gate(owned)
                if admitted and state['resource_gate']['allowed'] and not stopping:
                    state['status'] = 'running'
                    busy = {lane for lane, _ in active.values()}
                    if 'backfill' not in busy and 'backfill' not in state['paused_lanes']:
                        task = next((t for t in state['historical_tasks'] if t['status']=='pending'), None)
                        if task:
                            task.update(status='running', started_at=now())
                            active[pool.submit(run_historical, output, config, dict(task), owned)] = ('backfill', task)
                    if 'passk' not in busy and 'passk' not in state['paused_lanes']:
                        group = next_group(state['passk_groups'])
                        if group:
                            group.update(status='running', started_at=now())
                            active[pool.submit(run_passk, output, config, dict(group), owned)] = ('passk', group)
                persist(output, state, config)
                remaining = (any(t['status']=='pending' for t in state['historical_tasks']) and 'backfill' not in state['paused_lanes']) or (
                    any(g['status']=='pending' for g in state['passk_groups']) and 'passk' not in state['paused_lanes'])
                if not active and (stopping or (admitted and not remaining)):
                    break
                if not active:
                    time.sleep(30)
                    continue
                done, _ = wait(active, timeout=10, return_when=FIRST_COMPLETED)
                for future in done:
                    lane, target = active.pop(future)
                    try:
                        result = future.result()
                    except Exception as exc:
                        result = {'status': 'execution_error', 'exception_type': type(exc).__name__}
                        if lane=='passk': result['attempt'] = target['next_attempt']
                    result['finished_at'] = now()
                    if lane=='backfill': target.update(result)
                    else: apply_attempt(target, result, config['k'])
                    if result['status'] in ('execution_error','evidence_error'):
                        state['paused_lanes'][lane] = result
                    print(json.dumps({'lane':lane,'id':target.get('id',target.get('validation_id')),'result':result['status']}),flush=True)
                persist(output, state, config)
        state.update(status='stopped' if stopping else ('stopped_on_lane_error' if state['paused_lanes'] else 'complete'), finished_at=now())
        persist(output, state, config)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['prepare','verify','run','status'])
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--historical',type=Path,default=ROOT/'results_sweeps/july_routing_cosim_20261006')
    p.add_argument('--passk',type=Path,default=ROOT/'results_sweeps/polybench_passk_20261006_v2')
    args=p.parse_args();output=args.output.resolve()
    if args.action=='prepare':prepare(output,args.historical.resolve(),args.passk.resolve())
    elif args.action=='verify':verify(output);print('{"verified":true}')
    elif args.action=='run':run(output)
    else:
        value=read(output/'summary.json')
        if (output/'controller_identity.json').exists(): value['controller_alive']=alive(read(output/'controller_identity.json'))
        print(json.dumps(value,indent=2))


if __name__=='__main__':main()
