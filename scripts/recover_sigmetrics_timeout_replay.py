#!/usr/bin/env python3
"""Continue interrupted RTL validations with immutable source evidence and sidecars.

Old measurements and state are read-only. Recovery uses the original 10800-second
per-case policy. A distinct, prospectively selected diagnostic uses 21600 seconds.
Neither cohort changes the frozen main-study denominator or validator.
"""
from __future__ import annotations
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
VITIS = {'vitis_hls', 'vitis-run', 'xsim', 'xsimk', 'vivado'}
TERMINAL = {'pass', 'fail', 'timeout', 'not_run', 'execution_error'}


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    temp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def process_info(pid):
    try:
        base = Path('/proc') / str(pid)
        fields = (base / 'stat').read_text().rsplit(')', 1)[1].split()
        return {'start_ticks': fields[19], 'ppid': int(fields[1]),
                'argv': (base / 'cmdline').read_bytes().rstrip(b'\0').decode(errors='replace').split('\0')}
    except (OSError, ValueError, IndexError):
        return None


def boot_id():
    return Path('/proc/sys/kernel/random/boot_id').read_text().strip()


def check_hashes(files):
    for path, expected in files.items():
        if sha(path) != expected:
            raise ValueError(f'evidence changed: {path}')


def require_dead(state):
    info = process_info(state.get('pid', -1))
    if info and (not state.get('pid_start_ticks') or info['start_ticks'] == str(state['pid_start_ticks'])):
        raise RuntimeError('previous controller may still be alive')


def inventory(legacy, state, historical_manifest):
    """Return incomplete identities; terminal measurements can never be rerun."""
    raw = Path(historical_manifest['raw_root']) / 'measurements'
    known = {t['id']: t for t in state['tasks'] if t['kind'] == 'historical'}
    kernels = {r['validation_id']: r['kernel_sha256'] for r in historical_manifest['rows']}
    if len(known) != sum(t['kind'] == 'historical' for t in state['tasks']):
        raise ValueError('duplicate historical identity')
    if set(known) != {r['validation_id'] for r in historical_manifest['rows']}:
        raise ValueError('state and manifest identities differ')
    if {p.stem for p in raw.glob('*.json')} - set(known):
        raise ValueError('unregistered measurement')
    finished, unfinished, hashes = [], [], {}
    for key, task in known.items():
        path = raw / f'{key}.json'
        if path.exists():
            measurement = read(path)
            status = (measurement.get('cosim') or {}).get('status')
            if measurement.get('validation_id') != key or measurement.get('kernel_sha256') != kernels[key] or status not in TERMINAL:
                raise ValueError(f'invalid terminal measurement: {key}')
            if task['status'] in TERMINAL and task['status'] != status:
                raise ValueError(f'state/result disagreement: {key}')
            if task.get('measurement_sha256') and sha(path) != task['measurement_sha256']:
                raise ValueError(f'measurement hash mismatch: {key}')
            hashes[str(path)] = sha(path)
            finished.append(dict(task, status=status, measurement_path=str(path)))
        elif task['status'] in ('pending', 'running'):
            unfinished.append(copy.deepcopy(task))
        else:
            raise ValueError(f'terminal task has no saved result: {key}')
    for task in state['tasks']:
        if task['kind'] == 'parent':
            path = Path(task.get('measurement_path', legacy / 'parents' / task['id'] / 'measurement.json'))
            if task['status'] not in TERMINAL or not path.is_file():
                raise ValueError('this recovery requires all parents terminal')
            if task.get('measurement_sha256') and sha(path) != task['measurement_sha256']:
                raise ValueError('parent measurement changed')
            hashes[str(path)] = sha(path)
    return finished, unfinished, hashes


def select_diagnostics(finished, count=1):
    candidates = [t for t in finished if t['status'] == 'timeout']
    return sorted(candidates, key=lambda t: (t.get('estimate_cycles') or float('inf'), t['id']))[:count]


def prepare(legacy, output, peer_campaigns=(), diagnostic_count=1):
    if output.exists():
        raise FileExistsError('use a fresh recovery output; never overwrite an attempt')
    old = read(legacy / 'manifest.json')
    state = read(legacy / 'state.json')
    require_dead(state)
    if old['timeout_seconds'] != 10800:
        raise ValueError('expected frozen three-hour replay')
    repo = legacy.parent.parent
    hist_path = legacy / 'historical_timeout_replay_20261001' / 'manifest.json'
    historical = read(hist_path)
    source = Path(old['historical_source'])
    files = {str(legacy / name): sha(legacy / name) for name in ['manifest.json', 'state.json', 'summary.json']}
    files.update({str(repo / 'scripts/run_sigmetrics_timeout_replay.py'): old['runner_sha256'],
                  str(hist_path): old['historical_replay_manifest_sha256'],
                  str(repo / 'results_sweeps/sigmetrics_recipe_main_20260930/manifest.json'): old['main_manifest_sha256'],
                  str(repo / 'results_sweeps/historical_registry_recovery_20260928/manifest.json'): old['historical_manifest_sha256']})
    files.update({str(source / name): digest for name, digest in historical['code_hashes'].items()})
    files[str(source / 'scripts/run_historical_cosim.py')] = old['historical_worker_sha256']
    for t in old['tasks']:
        files[t['prior_measurement']] = t['prior_sha256']
    check_hashes(files)
    finished, unfinished, hashes = inventory(legacy, state, historical)
    files.update(hashes)
    diagnostics = select_diagnostics(finished, diagnostic_count)
    if diagnostic_count and len(diagnostics) != diagnostic_count:
        raise ValueError('not enough completed timeouts for diagnostic selection')
    cohorts = {'recovery': (unfinished, 10800, 2), 'diagnostic': (diagnostics, 21600, 1)}
    peers = []
    for peer in peer_campaigns:
        peer_manifest = peer / 'manifest.json'
        peers.append({'campaign': str(peer), 'manifest_sha256': sha(peer_manifest)})
    output.mkdir(parents=True)
    snapshot = output / 'source'
    (snapshot / 'scripts').mkdir(parents=True)
    shutil.copyfile(__file__, snapshot / 'scripts' / Path(__file__).name)
    shutil.copyfile(ROOT / 'cosim_runtime_monitor.py', snapshot / 'cosim_runtime_monitor.py')
    for name in ['manifest.json', 'state.json', 'summary.json']:
        target = output / 'before_recovery' / name
        target.parent.mkdir(exist_ok=True)
        shutil.copyfile(legacy / name, target)
        files[str(target)] = files[str(legacy / name)]
    tasks = []
    for cohort, (selected, timeout, slots) in cohorts.items():
        hist_output = output / 'cohorts' / f'{cohort}_{output.name}'
        (hist_output / 'raw' / 'measurements').mkdir(parents=True)
        for name in ['logs', 'vitis_archives']:
            (hist_output / 'raw' / name).mkdir()
        selected_ids = {t['id'] for t in selected}
        id_map = {old_id: hashlib.sha256(f'c2hls-recovery-v1:{cohort}:{timeout}:{old_id}'.encode()).hexdigest() for old_id in selected_ids}
        rows = []
        for original in historical['rows']:
            if original['validation_id'] not in selected_ids:
                continue
            row = copy.deepcopy(original)
            prior_id = row['validation_id']
            row.update(validation_id=id_map[prior_id], continuation_of_validation_id=prior_id,
                       replay_timeout_seconds=timeout, continuation_cohort=cohort)
            rows.append(row)
            files[row['staged_kernel']] = row['kernel_sha256']
            for name, digest in row['input_hashes'].items():
                files[str(Path(row['staged_benchmark']) / name)] = digest
        new_manifest = copy.deepcopy(historical)
        new_manifest['replay_policy'] = dict(new_manifest.get('replay_policy', {}), timeout_seconds=timeout)
        new_manifest.update(raw_root=str(hist_output / 'raw'), rows=rows,
                            continuation_policy={'cohort': cohort, 'timeout_seconds': timeout,
                                'original_manifest_sha256': sha(hist_path), 'no_completed_recovery_rerolls': True})
        save(hist_output / 'manifest.json', new_manifest)
        files[str(hist_output / 'manifest.json')] = sha(hist_output / 'manifest.json')
        row_by_id = {r['validation_id']: r for r in rows}
        for task in selected:
            ident = id_map[task['id']]
            tasks.append({'id': ident, 'prior_id': task['id'], 'benchmark': task['benchmark'],
                          'cohort': cohort, 'timeout_seconds': timeout, 'status': 'pending',
                          'prior_status': task['status'], 'historical_output': str(hist_output),
                          'kernel_sha256': row_by_id[ident]['kernel_sha256']})
    frozen = {str(p): sha(p) for p in snapshot.rglob('*.py')}
    files.update(frozen)
    check_hashes(files)
    manifest = {'schema_version': 'c2hls.cosim-recovery.v1', 'created_at': now(),
                'legacy': str(legacy), 'historical_source': str(source), 'files': files,
                'tasks': tasks, 'legacy_historical_counts': dict(Counter(t['status'] for t in finished)),
                'legacy_parent_counts': dict(Counter(t['status'] for t in state['tasks'] if t['kind'] == 'parent')),
                'workers': {'recovery': 2, 'diagnostic': 1}, 'peer_campaigns': peers,
                'observation_interval_seconds': 10, 'outer_worker_overhead_seconds': 1400,
                'diagnostic_selection': 'lowest recorded synthesis cycles among already-terminal three-hour timeouts, tie by ID',
                'diagnostic_timeout_basis': 'prospective twofold escalation of a censored three-hour cap; not a calibrated completion prediction',
                'policy': 'No source, testbench, clock, tool, tracing or validation changes. Keep cohorts separate; monitor never decides correctness or terminates jobs.',
                'registration_note': 'User-approved recovery after Oct4 reboot; two interrupted validations receive explicit new attempts, all completed outcomes preserved.'}
    save(output / 'manifest.json', manifest)
    state = {'status': 'prepared', 'updated_at': now(), 'manifest_sha256': sha(output / 'manifest.json'), 'tasks': tasks}
    save(output / 'state.json', state)
    save(output / 'summary.json', summarize(state, manifest))
    print(json.dumps(summarize(state, manifest)), flush=True)


def summarize(state, manifest):
    counts = {c: dict(Counter(t['status'] for t in state['tasks'] if t['cohort'] == c)) for c in manifest['workers']}
    combined = Counter(manifest['legacy_historical_counts'])
    combined.update(counts['recovery'])
    return {'status': state['status'], 'updated_at': state['updated_at'], 'cohorts': counts,
            'combined_three_hour_replay': dict(combined), 'parent_counts': manifest['legacy_parent_counts'],
            'active': [{k:t[k] for k in ['id', 'benchmark', 'cohort', 'started_at']} for t in state['tasks'] if t['status'] == 'running'],
            'resource_gate': state.get('resource_gate'), 'failure': state.get('failure')}


def verify(output):
    manifest, state = read(output / 'manifest.json'), read(output / 'state.json')
    if state['manifest_sha256'] != sha(output / 'manifest.json'):
        raise ValueError('recovery registration changed')
    check_hashes(manifest['files'])
    if str(Path(__file__).resolve()) not in manifest['files']:
        raise ValueError('run the frozen recovery controller from output/source/scripts')
    return manifest, state


def peer_pid(peer):
    try:
        folder = Path(peer['campaign'])
        identity = read(folder / 'controller_identity.json')
        if sha(folder / 'manifest.json') != peer['manifest_sha256'] or identity['manifest_sha256'] != peer['manifest_sha256']:
            return None
        if identity['boot_id'] != boot_id() or identity['campaign'] != str(folder):
            return None
        info = process_info(identity['pid'])
        if not info or info['start_ticks'] != str(identity['start_ticks']):
            return None
        if sha(identity['source_runner']) != identity['source_runner_sha256']:
            return None
        argv = info['argv']; index = 2 if len(argv)>1 and argv[1]=='-u' else 1
        if argv[index:index+2] != [identity['source_runner'], 'run']:
            return None
        if '--output' not in argv or argv[argv.index('--output')+1] != str(folder):
            return None
        return identity['pid']
    except (OSError, ValueError, KeyError, IndexError, TypeError):
        return None


def descendant(pid, ancestor, table):
    seen = set()
    while pid in table and pid not in seen and pid != ancestor:
        seen.add(pid); pid = table[pid][0]
    return pid == ancestor


def gate(manifest):
    gib=1024**3
    mem={r.split(':')[0]:int(r.split()[1])*1024 for r in Path('/proc/meminfo').read_text().splitlines()}
    observed={'scratch_free_gib':shutil.disk_usage('/tmp').free/gib,
              'memory_available_gib':mem['MemAvailable']/gib,'load_per_cpu':os.getloadavg()[0]/(os.cpu_count() or 1)}
    reasons=[key for key,limit in [('scratch_free_gib',100),('memory_available_gib',128)] if observed[key]<limit]
    if observed['load_per_cpu']>=0.7: reasons.append('load_per_cpu')
    table={}
    for line in subprocess.check_output(['ps','-eo','pid=,ppid=,comm='],text=True).splitlines():
        pid,ppid,name=line.split(None,2);table[int(pid)]=(int(ppid),name)
    allowed=[os.getpid()]+[pid for p in manifest.get('peer_campaigns',[]) if (pid:=peer_pid(p)) is not None]
    unknown=[pid for pid,(_,name) in table.items() if name in VITIS and not any(descendant(pid,a,table) for a in allowed)]
    if unknown: reasons.append('unrelated_vitis')
    return {'allowed':not reasons,'reasons':reasons,'observed':observed,'unrelated_vitis_pids':unknown}


def classify_measurement(task, path, execution):
    if not path.is_file():
        return {'status':'execution_error','execution':execution,'error':'worker did not save a measurement'}
    m=read(path);cosim=m.get('cosim') or {}
    if m.get('validation_id')!=task['id'] or m.get('kernel_sha256')!=task['kernel_sha256']:
        raise ValueError('worker output identity mismatch')
    status=cosim.get('status')
    if status not in TERMINAL: raise ValueError('invalid terminal status')
    if status=='pass' and (cosim.get('passed') is not True or not (cosim.get('kernel_runtime_cycles') or 0)>0):
        raise ValueError('pass without functional and cycle evidence')
    if execution['returncode']!=0 or execution['process_timed_out'] or m.get('archive_error'):
        return {'status':'execution_error','execution':execution,'measurement_status':status,
                'measurement_path':str(path),'measurement_sha256':sha(path),'archive_error':m.get('archive_error')}
    return {'status':status,'execution':execution,'measurement_path':str(path),'measurement_sha256':sha(path)}



def _cleanup_process_info(pid):
    """Read stable identity without collecting potentially sensitive arguments."""
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        return {'ppid': int(fields[1]), 'start_ticks': fields[19], 'state': fields[0]}
    except (OSError, ValueError, IndexError):
        return None


def _cleanup_process_table():
    result = {}
    for path in Path('/proc').iterdir():
        if path.name.isdigit():
            info = _cleanup_process_info(int(path.name))
            if info is not None:
                result[int(path.name)] = info
    return result


def _signal_owned_identity(pid, start_ticks, signum):
    """Signal an owned identity, using pidfds when available to prevent PID reuse."""
    descriptor = None
    try:
        if hasattr(os, 'pidfd_open') and hasattr(signal, 'pidfd_send_signal'):
            descriptor = os.pidfd_open(pid)
        info = _cleanup_process_info(pid)
        if info is None or info['start_ticks'] != start_ticks or info['state'] == 'Z':
            return False
        if descriptor is not None:
            signal.pidfd_send_signal(descriptor, signum)
        else:
            os.kill(pid, signum)
        return True
    except ProcessLookupError:
        return False
    finally:
        if descriptor is not None:
            os.close(descriptor)


class OwnedProcessTree:
    """Track only this worker and its descendants, including detached sessions.

    Remember identities while ancestry exists so orphaned descendants remain
    eligible for cleanup. Never send signals to a process group or a reused PID.
    """
    def __init__(self, process):
        self.process = process
        info = _cleanup_process_info(process.pid)
        self.identities = {process.pid: info['start_ticks']} if info else {}
        self.capture()

    def capture(self):
        table = _cleanup_process_table()
        owned = {pid for pid, ticks in self.identities.items()
                 if pid in table and table[pid]['start_ticks'] == ticks}
        while True:
            children = {pid for pid, info in table.items() if info['ppid'] in owned}
            if children <= owned:
                break
            owned.update(children)
        self.identities.update({pid: table[pid]['start_ticks'] for pid in owned})
        return {pid: table[pid] for pid in owned if table[pid]['state'] != 'Z'}

    def terminate(self, grace_seconds=5, kill_wait_seconds=5):
        """Bound all waits, clean detached descendants, and reap the direct child."""
        errors = []
        signaled = []
        for signum, duration in ((signal.SIGTERM, grace_seconds), (signal.SIGKILL, kill_wait_seconds)):
            deadline = time.monotonic() + duration
            sent = set()
            while True:
                # Capture every known descendant before signaling any parent.
                live = self.capture()
                for pid in sorted(live, key=lambda item: item == self.process.pid):
                    identity = (pid, self.identities[pid])
                    if identity in sent:
                        continue
                    sent.add(identity)
                    try:
                        if _signal_owned_identity(pid, identity[1], signum):
                            signaled.append({'pid': pid, 'start_ticks': identity[1], 'signal': int(signum)})
                    except OSError as exc:
                        errors.append(f'{type(exc).__name__}: {exc}')
                self.process.poll()  # reap a terminated direct child promptly
                if not self.capture() or time.monotonic() >= deadline:
                    break
                time.sleep(min(.05, max(0, deadline - time.monotonic())))
        try:
            self.process.wait(timeout=.2)
        except subprocess.TimeoutExpired:
            errors.append('direct worker did not exit within cleanup bound')
        return {'signaled': signaled, 'remaining_owned_pids': sorted(self.capture()), 'errors': errors}


def run_task(output, task, manifest):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    from cosim_runtime_monitor import CosimRuntimeMonitor
    hist=Path(task['historical_output']);source=Path(manifest['historical_source'])
    measurement=hist/'raw'/'measurements'/(task['id']+'.json')
    if measurement.exists(): raise FileExistsError('refusing to replace a measurement')
    scratch=Path('/tmp')/hist.name/task['id']
    if scratch.exists(): raise FileExistsError(f'unexpected preexisting scratch: {scratch}')
    log=output/'logs'/(task['id']+'.log');log.parent.mkdir(exist_ok=True)
    telemetry=output/'telemetry'/task['id'];telemetry.mkdir(parents=True,exist_ok=False)
    command=[sys.executable,'-u',str(source/'scripts/run_historical_cosim.py'),'worker',
             '--output',str(hist),'--validation-id',task['id'],'--timeout',str(task['timeout_seconds'])]
    env={k:v for k,v in os.environ.items() if not k.startswith('C2HLS_')}
    with log.open('x') as stream:
        process=subprocess.Popen(command,cwd=source,stdout=stream,stderr=subprocess.STDOUT,env=env,start_new_session=True)
        owned=OwnedProcessTree(process)
        cleanup=None
        try:
            monitor=CosimRuntimeMonitor([scratch],telemetry/'samples.jsonl',root_pids=[process.pid])
            deadline=time.monotonic()+task['timeout_seconds']+manifest['outer_worker_overhead_seconds']
            timed_out=False; monitoring_errors=[]
            while process.poll() is None:
                owned.capture()
                try: monitor.sample()
                except Exception as exc: monitoring_errors.append(f'{type(exc).__name__}: {exc}')
                if time.monotonic()>=deadline:
                    timed_out=True
                    cleanup=owned.terminate()
                    break
                try: process.wait(timeout=manifest['observation_interval_seconds'])
                except subprocess.TimeoutExpired: pass
            # A normally exited worker must not leave detached Vitis descendants.
            if cleanup is None:
                cleanup=owned.terminate()
            execution={'returncode':process.returncode,'process_timed_out':timed_out,
                       'owned_process_cleanup':cleanup}
            try: report=monitor.finish(outcome=execution)
            except Exception as exc: report={'monitor_error':f'{type(exc).__name__}: {exc}'}
            report['monitoring_errors']=monitoring_errors
            save(telemetry/'summary.json',report)
            if cleanup['remaining_owned_pids']:
                raise RuntimeError('owned Vitis descendants remain after bounded cleanup')
            result=classify_measurement(task,measurement,execution)
            result['telemetry_path']=str(telemetry)
            return result
        finally:
            # Also covers monitor construction, finish/save, and classification
            # exceptions. Ordinary sample errors above never terminate a job.
            final_cleanup=owned.terminate()
            save(telemetry/'cleanup.json',final_cleanup if cleanup is None else
                 {'initial_cleanup':cleanup,'final_cleanup':final_cleanup})



def run(output):
    manifest,state=verify(output)
    if state['status']!='prepared':
        raise ValueError('prepare a separate recovery attempt after interruption; no implicit rerolls')
    with (output/'controller.lock').open('a') as lock, (Path(manifest['legacy'])/'controller.lock').open('a') as old_lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(old_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        require_dead(read(Path(manifest['legacy'])/'state.json'))
        stopping={}
        def stop(signum,_frame): stopping.update(signal=signum,requested_at=now())
        signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
        info=process_info(os.getpid())
        save(output/'controller_identity.json',{'pid':os.getpid(),'start_ticks':info['start_ticks'],
             'boot_id':boot_id(),'source_runner':str(Path(__file__).resolve()),'source_runner_sha256':sha(__file__),
             'manifest_sha256':sha(output/'manifest.json'),'campaign':str(output)})
        state.update(status='running',started_at=now(),pid=os.getpid(),pid_start_ticks=info['start_ticks'],boot_id=boot_id())
        active={};failed=False
        with ThreadPoolExecutor(max_workers=sum(manifest['workers'].values())) as pool:
            while True:
                state['resource_gate']=gate(manifest)
                for cohort,slots in manifest['workers'].items():
                    count=sum(t['cohort']==cohort for t in active.values())
                    for task in state['tasks']:
                        if count>=slots or stopping or failed or not state['resource_gate']['allowed']: break
                        if task['cohort']!=cohort or task['status']!='pending': continue
                        task.update(status='running',started_at=now())
                        active[pool.submit(run_task,output,task.copy(),manifest)]=task;count+=1
                state.update(updated_at=now(),stop_requested=stopping or None)
                save(output/'state.json',state);save(output/'summary.json',summarize(state,manifest))
                if not active:
                    if stopping or failed or all(t['status']!='pending' for t in state['tasks']): break
                    time.sleep(10);continue
                finished,_=wait(active,timeout=10,return_when=FIRST_COMPLETED)
                for future in finished:
                    task=active.pop(future)
                    try: result=future.result()
                    except Exception as exc: result={'status':'execution_error','error':f'{type(exc).__name__}: {exc}'}
                    task.update(result,finished_at=now())
                    if result['status']=='execution_error':
                        failed=True;state['failure']={'task':task['id'],'detail':result}
                    print('VALIDATION_DONE',task['cohort'],task['benchmark'],task['status'],flush=True)
        state.update(status='stopped_on_error' if failed else ('stopped' if stopping else 'complete'),updated_at=now(),finished_at=now())
        check_hashes(manifest['files'])
        save(output/'state.json',state);save(output/'summary.json',summarize(state,manifest))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','verify','run','status'])
    parser.add_argument('--legacy',type=Path,default=ROOT/'results_sweeps/sigmetrics_timeout_replay_20261001')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--peer-campaign',type=Path,action='append',default=[])
    parser.add_argument('--diagnostic-count',type=int,choices=range(0,4),default=1)
    args=parser.parse_args();output=args.output.resolve()
    if args.action=='prepare': prepare(args.legacy.resolve(),output,[p.resolve() for p in args.peer_campaign],args.diagnostic_count)
    elif args.action=='run': run(output)
    elif args.action=='verify':
        manifest,state=verify(output);print(json.dumps({'verified':True,'tasks':len(state['tasks'])}))
    else:
        state,manifest=read(output/'state.json'),read(output/'manifest.json')
        summary=summarize(state,manifest)
        info=process_info(state.get('pid',-1))
        live=bool(info and state.get('boot_id')==boot_id() and info['start_ticks']==str(state.get('pid_start_ticks')))
        summary['controller_live']=live
        if summary['status']=='running' and not live: summary['observed_status']='interrupted'
        print(json.dumps(summary,indent=2))

if __name__=='__main__': main()
