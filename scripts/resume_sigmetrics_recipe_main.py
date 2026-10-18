#!/usr/bin/env python3
"""Explicit scheduling amendment: finish only never-started frozen main-study cells.

Original results/state and uncertain API usage are preserved. This scheduler never
settles a reservation, rerolls a started cell, or changes inference/validation code.
"""
from __future__ import annotations
import argparse
import collections
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from contextlib import contextmanager
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

SCHEMA = 'c2hls.sigmetrics-main-continuation.v1'
ALIAS_CAVEAT = ('Continuation occurs in a later calendar interval; provider aliases '
                'do not guarantee immutable model weights. Original identities, '
                'budgets, prompts, selection and validation policies are unchanged.')


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(data, indent=2) + '\n')
    temp.replace(path)


def records(paths):
    return [{'path': str(p.resolve()), 'sha256': sha(p)} for p in sorted(set(map(Path, paths)))]


def verify_files(rows):
    for row in rows:
        if sha(row['path']) != row['sha256']:
            raise ValueError('preserved file changed: ' + row['path'])


@contextmanager
def original_lock(original):
    with (Path(original) / 'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def pending_pristine(original, cells, ledger):
    """Never-started includes both durable artifacts and request reservations."""
    for cell in cells:
        identity = cell['cell_id']
        if (original / 'cells' / identity).exists() or (original / (identity + '.log')).exists():
            raise ValueError('pending cell has execution artifacts: ' + identity)
        if any(row['run_id'].endswith('__' + identity) for row in ledger['reservations']):
            raise ValueError('pending cell has provider requests: ' + identity)


def verify_ledger(ledger, config, existing_rows):
    if ledger['limit_usd'] != config['shared_limit_usd'] or ledger['prices_per_million'] != config['rates_per_million']:
        raise ValueError('registered API cap/rates changed')
    ids = [row['id'] for row in ledger['reservations']]
    if len(ids) != len(set(ids)):
        raise ValueError('duplicate reservation identity')
    current = {row['id']: row for row in ledger['reservations']}
    for old in existing_rows:
        if current.get(old['id']) != old:
            raise ValueError('original reservation changed; unknown usage must remain reserved')
    if any(row.get('overrun') for row in current.values()):
        raise ValueError('provider reservation overrun')
    if sum(row['charged_or_reserved_usd'] for row in current.values()) > config['shared_limit_usd']:
        raise ValueError('shared budget exhausted')


def remaining_bound(cells, config):
    return sum((config['aggregate_input_token_cap'] * config['rates_per_million'][c['model']]['input']
                + config['llm_call_cap'] * config['max_response_tokens'] * config['rates_per_million'][c['model']]['output']) / 1e6
               for c in cells)


def prepare(original, output):
    original, output = original.resolve(), output.resolve()
    if output.exists():
        raise ValueError('continuation exists; refusing overwrite')
    with original_lock(original):
        config, state = read(original / 'manifest.json'), read(original / 'state.json')
        receipt = read(original / 'launch_receipt.json')
        python_executable = Path(receipt['main_command'][0]).resolve()
        if receipt['main_manifest_sha256'] != sha(original / 'manifest.json') or not python_executable.is_file():
            raise ValueError('original launch receipt/runtime identity mismatch')
        if config['schema_version'] != 'c2hls.sigmetrics-recipe-main.v1' or state['status'] != 'stopped_on_failure':
            raise ValueError('expected the stopped frozen main campaign')
        if state['manifest_sha256'] != sha(original / 'manifest.json'):
            raise ValueError('original manifest identity mismatch')
        if len(config['cells']) != 216 or len(state['cells']) != 216:
            raise ValueError('original denominator must remain 216')
        if len({c['cell_id'] for c in config['cells']}) != 216:
            raise ValueError('duplicate registered cell identity')
        if [c['cell_id'] for c in state['cells']] != [c['cell_id'] for c in config['cells']]:
            raise ValueError('original cell identities/order changed')
        counts = collections.Counter(c['status'] for c in state['cells'])
        if counts != {'complete': 71, 'blocked_parent': 90, 'accounting_or_execution_failure': 1, 'pending': 54}:
            raise ValueError('state differs from the audited 71/90/1/54 stop')
        selected = [c for c in state['cells'] if c['status'] == 'pending']
        ready = {p['benchmark'] for p in state['parents'] if p['status'] == 'ready'}
        if any(c['benchmark'] not in ready for c in selected):
            raise ValueError('pending cell parent is not qualified')
        verify_files(config['files'])
        ledger = read(config['api_budget'])
        verify_ledger(ledger, config, ledger['reservations'])
        pending_pristine(original, selected, ledger)
        unknown = [r for r in ledger['reservations'] if r['status'] != 'settled']
        failures = [c for c in state['cells'] if c['status'] == 'accounting_or_execution_failure']
        failed_ids = {c['cell_id'] for c in failures}
        if len(unknown) != 1 or unknown[0]['status'] != 'reserved' or not any(unknown[0]['run_id'].endswith('__' + i) for i in failed_ids):
            raise ValueError('only the audited failed-cell reservation may be unknown')
        bound = remaining_bound(selected, config)
        if sum(r['charged_or_reserved_usd'] for r in ledger['reservations']) + bound > config['shared_limit_usd']:
            raise ValueError('insufficient conservative budget headroom')
        preserved = [original / name for name in ('state.json', 'summary.json', 'manifest.json', 'launch_receipt.json')]
        preserved.append(python_executable)
        for cell in state['cells']:
            if cell['status'] != 'pending':
                folder = original / 'cells' / cell['cell_id']
                if cell['status'] in ('complete', 'accounting_or_execution_failure') and not (folder / 'result.json').is_file():
                    raise ValueError('recorded result missing: ' + cell['cell_id'])
                preserved.extend(p for p in folder.rglob('*') if p.is_file())
                log = original / (cell['cell_id'] + '.log')
                if log.exists():
                    preserved.append(log)
        for parent in state['parents']:
            if parent['status'] == 'ready':
                qpath = original / 'qualification' / parent['benchmark'] / 'qualification.json'
                if sha(qpath) != parent['qualification_sha256']:
                    raise ValueError('ready parent qualification changed')
                qualification = read(qpath)
                verify_files(qualification['files'])
                preserved.append(qpath)
                preserved.extend(Path(r['path']) for r in qualification['files'])
        output.mkdir(parents=True)
        (output / 'source').mkdir()
        scheduler = output / 'source' / Path(__file__).name
        shutil.copyfile(Path(__file__).resolve(), scheduler)
        snapshots = output / 'snapshots'
        snapshots.mkdir()
        for name in ('state.json', 'summary.json', 'manifest.json'):
            shutil.copyfile(original / name, snapshots / name)
        shutil.copyfile(config['api_budget'], snapshots / 'api_budget.json')
        request = dict(schema_version=SCHEMA, prepared_at=now(), original=str(original),
                       original_manifest_sha256=sha(original / 'manifest.json'),
                       original_state_sha256=sha(original / 'state.json'), python_executable=str(python_executable), source_runner=str(scheduler),
                       source_runner_sha256=sha(scheduler), workers=2, selected_cells=selected,
                       preserved_files=records(preserved), snapshot_files=records(snapshots.iterdir()),
                       original_reservations=ledger['reservations'], quarantined_cells=failures,
                       quarantined_reservations=unknown, remaining_maximum_estimated_usd=bound,
                       amendment='Resume only never-started cells; retain original failure and reservation; stop new submissions on any new execution/accounting failure.',
                       calendar_alias_caveat=ALIAS_CAVEAT)
        save(output / 'manifest.json', request)
        continuation = dict(status='prepared', updated_at=now(), manifest_sha256=sha(output / 'manifest.json'),
                            original=str(original), cells=selected, calendar_alias_caveat=ALIAS_CAVEAT)
        save(output / 'state.json', continuation)
        save(output / 'summary.json', summarize(request, continuation, state))
    return {'prepared': True, 'selected_cells': len(selected), 'remaining_maximum_estimated_usd': bound,
            'frozen_runner': str(scheduler), 'manifest_sha256': sha(output / 'manifest.json')}


def verify(output, *, pristine=False):
    request, state = read(output / 'manifest.json'), read(output / 'state.json')
    if request['schema_version'] != SCHEMA or state['manifest_sha256'] != sha(output / 'manifest.json'):
        raise ValueError('continuation manifest changed')
    if request['workers'] != 2 or len(request['selected_cells']) != 54:
        raise ValueError('audited scheduling limits changed')
    if sha(request['source_runner']) != request['source_runner_sha256']:
        raise ValueError('frozen recovery scheduler changed')
    verify_files(request['preserved_files'])
    verify_files(request['snapshot_files'])
    original = Path(request['original'])
    if sha(original / 'manifest.json') != request['original_manifest_sha256'] or sha(original / 'state.json') != request['original_state_sha256']:
        raise ValueError('original registration/state changed')
    config = read(original / 'manifest.json')
    verify_files(config['files'])
    ledger = read(config['api_budget'])
    verify_ledger(ledger, config, request['original_reservations'])
    if [c['cell_id'] for c in state['cells']] != [c['cell_id'] for c in request['selected_cells']]:
        raise ValueError('continuation cell identities/order changed')
    if pristine:
        if state['status'] != 'prepared' or any(c['status'] != 'pending' for c in state['cells']):
            raise ValueError('continuation already started; no implicit rerolls')
        pending_pristine(original, state['cells'], ledger)
    return request, state, config


def summarize(request, state, original_state):
    updates = {c['cell_id']: c['status'] for c in state['cells']}
    aggregate = collections.Counter(updates.get(c['cell_id'], c['status']) for c in original_state['cells'])
    return dict(status=state['status'], updated_at=state['updated_at'], registered_cells=len(original_state['cells']),
                original_cells=dict(collections.Counter(c['status'] for c in original_state['cells'])),
                continuation_cells=dict(collections.Counter(c['status'] for c in state['cells'])),
                cells=dict(aggregate), parents=dict(collections.Counter(p['status'] for p in original_state['parents'])),
                resource_gate=state.get('resource_gate'), stop_requested=state.get('stop_requested'),
                quarantined_unknown_usage=len(request['quarantined_reservations']),
                calendar_alias_caveat=ALIAS_CAVEAT)


def load_frozen(config):
    path = Path(config['source']) / 'scripts/run_sigmetrics_recipe_main.py'
    spec = importlib.util.spec_from_file_location('frozen_main_continuation_cells', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if module.ROOT.resolve() != Path(config['source']).resolve():
        raise ValueError('frozen runner root mismatch')
    return module


def result_status(original, identity, execution, config):
    if execution['returncode'] != 0:
        return 'accounting_or_execution_failure'
    path = original / 'cells' / identity / 'result.json'
    if not path.is_file():
        raise ValueError('successful exit without result')
    result = read(path)
    if result.get('registration', {}).get('spec', {}).get('cell_id') != identity:
        raise ValueError('result identity mismatch')
    if result.get('registration', {}).get('manifest_sha256') != sha(original / 'manifest.json'):
        raise ValueError('result campaign identity mismatch')
    if result.get('accounting_reconciliation', {}).get('complete') is not True:
        raise ValueError('successful exit without accounting reconciliation')
    rows = [r for r in read(config['api_budget'])['reservations'] if r['run_id'].endswith('__' + identity)]
    if any(r['status'] != 'settled' or r.get('overrun') for r in rows):
        raise ValueError('new cell has unsettled/overrun reservations')
    if len(rows) != result['accounting_reconciliation']['provider_requests']:
        raise ValueError('new cell reservation count mismatch')
    return 'complete'



def process_table():
    """Read process identity and ancestry, without inspecting environment/secrets."""
    rows = {}
    for path in Path('/proc').glob('[0-9]*/stat'):
        try:
            fields = path.read_text().rsplit(')', 1)[1].split()
            rows[int(path.parent.name)] = {'pid': int(path.parent.name), 'ppid': int(fields[1]),
                                          'start_ticks': fields[19], 'state': fields[0]}
        except (FileNotFoundError, ProcessLookupError, PermissionError, ValueError, IndexError):
            continue
    return rows


def capture_descendants(owned, table):
    """Retain previously observed descendants even after their parents exit."""
    live = {pid for pid, row in owned.items() if pid in table and table[pid]['start_ticks'] == row['start_ticks']}
    while True:
        new = {pid for pid, row in table.items() if row['ppid'] in live and pid not in live}
        if not new:
            break
        for pid in new:
            owned[pid] = dict(table[pid])
        live.update(new)
    return owned


def live_owned(owned, table):
    return [row for pid, row in owned.items() if pid in table
            and table[pid]['start_ticks'] == row['start_ticks'] and table[pid]['state'] != 'Z']


def signal_owned(row, signum):
    """PID reuse is never sufficient authority to signal a process."""
    try:
        fields = (Path('/proc') / str(row['pid']) / 'stat').read_text().rsplit(')', 1)[1].split()
        if fields[19] != row['start_ticks'] or fields[0] == 'Z':
            return False
        os.kill(row['pid'], signum)
        return True
    except (FileNotFoundError, ProcessLookupError):
        return False


def cleanup_owned(process, owned, *, grace=20, poll=0.2):
    # Capture descendants before killing ancestors: Vitis/XSim have new sessions,
    # and killing just the Python process group cannot reach those sessions.
    table = process_table()
    capture_descendants(owned, table)
    targets = live_owned(owned, table)
    for row in sorted(targets, key=lambda row: row['pid'] == process.pid):
        signal_owned(row, signal.SIGTERM)
    deadline = time.monotonic() + grace
    while time.monotonic() < deadline:
        process.poll()  # Reap our child so zombies do not prolong cleanup.
        table = process_table()
        old = set(owned)
        capture_descendants(owned, table)
        for pid in set(owned) - old:
            signal_owned(owned[pid], signal.SIGTERM)
        if not live_owned(owned, table):
            break
        time.sleep(min(poll, max(0, deadline - time.monotonic())))
    table = process_table()
    capture_descendants(owned, table)
    for row in sorted(live_owned(owned, table), key=lambda row: row['pid'] == process.pid):
        signal_owned(row, signal.SIGKILL)
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        pass
    deadline = time.monotonic() + 2
    survivors = live_owned(owned, process_table())
    while survivors and time.monotonic() < deadline:
        time.sleep(min(poll, max(0, deadline - time.monotonic())))
        survivors = live_owned(owned, process_table())
    return [{'pid': row['pid'], 'start_ticks': row['start_ticks']} for row in survivors]


def execute_owned(command, cwd, log, env, timeout, *, grace=20, poll=1):
    """Keep frozen cell behavior but bound all detached descendants on failure."""
    owned = {}
    with Path(log).open('x') as stream:
        process = subprocess.Popen(command, cwd=cwd, stdout=stream, stderr=subprocess.STDOUT,
                                   env=env, start_new_session=True)
        table = process_table()
        if process.pid in table:
            owned[process.pid] = dict(table[process.pid])
        deadline = time.monotonic() + timeout
        try:
            while True:
                capture_descendants(owned, process_table())
                code = process.poll()
                if code is not None:
                    survivors = live_owned(owned, process_table())
                    if survivors:
                        remaining = cleanup_owned(process, owned, grace=grace)
                        return {'returncode': code or 3, 'timed_out': False,
                                'orphaned_descendants_after_cell_exit': True, 'cleanup_survivors': remaining}
                    return {'returncode': code, 'timed_out': False}
                if time.monotonic() >= deadline:
                    remaining = cleanup_owned(process, owned, grace=grace)
                    return {'returncode': 124, 'timed_out': True, 'cleanup_survivors': remaining,
                            'tracked_process_count': len(owned)}
                time.sleep(min(poll, max(0, deadline - time.monotonic())))
        except BaseException:
            # Interrupts or bookkeeping exceptions must not strand EDA sessions.
            remaining = cleanup_owned(process, owned, grace=grace)
            if remaining:
                print(json.dumps({'cleanup_survivors': remaining, 'cell_log': str(log)}), file=sys.stderr, flush=True)
            raise

def run(output):
    request = read(output / 'manifest.json')
    original = Path(request['original'])
    with original_lock(original):
        request, state, config = verify(output, pristine=True)
        if Path(__file__).resolve() != Path(request['source_runner']).resolve():
            raise ValueError('launch the frozen recovery scheduler from output/source')
        runner = load_frozen(config)
        source = Path(config['source'])
        original_state = read(output / 'snapshots/state.json')
        identity = dict(pid=os.getpid(), start_ticks=runner.start_ticks(os.getpid()),
                        boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                        source_runner=request['source_runner'], source_runner_sha256=request['source_runner_sha256'],
                        manifest_sha256=sha(output / 'manifest.json'), campaign=str(output), registered_at=now())
        save(output / 'controller_identity.json', identity)
        stop_request, active = {}, {}
        def stop(signum, frame):
            stop_request.update(signal=signum, received_at=now(), policy='stop submissions and drain active cells')
        previous_signals = {sig: signal.signal(sig, stop) for sig in (signal.SIGTERM, signal.SIGINT)}
        state.update(status='running', started_at=now(), pid=os.getpid())
        def persist():
            state['updated_at'] = now()
            save(output / 'state.json', state)
            save(output / 'summary.json', summarize(request, state, original_state))
        def launch(cell):
            verify_files(config['files'])
            pending_pristine(original, [cell], read(config['api_budget']))
            command = [request['python_executable'], str(source / 'scripts/run_sigmetrics_recipe_main.py'), 'cell', '--output', str(original), '--cell-id', cell['cell_id']]
            return execute_owned(command, source, original / (cell['cell_id'] + '.log'),
                                  runner.base_environment(config, cell['cell_id']), config['cell_job_timeout_seconds'])
        failed = False
        persist()
        try:
            with ThreadPoolExecutor(max_workers=2) as pool:
                while True:
                    if stop_request:
                        state['stop_requested'] = dict(stop_request)
                    try:
                        gate = runner.resource_gate(config)
                        state['resource_gate'] = gate
                        verify_ledger(read(config['api_budget']), config, request['original_reservations'])
                    except Exception as exc:
                        failed = True
                        state['scheduler_error'] = f'{type(exc).__name__}: {exc}'
                        gate = {'allowed': False}
                    pending = [c for c in state['cells'] if c['status'] == 'pending']
                    while not failed and not stop_request and gate['allowed'] and len(active) < 2 and pending:
                        cell = pending.pop(0)
                        cell.update(status='running', started_at=now())
                        active[pool.submit(launch, cell)] = cell
                    persist()
                    if not active:
                        if not failed and not stop_request and pending and not gate['allowed']:
                            time.sleep(10)
                            continue
                        break
                    finished, _ = wait(active, timeout=10, return_when=FIRST_COMPLETED)
                    for future in finished:
                        cell = active.pop(future)
                        try:
                            execution = future.result()
                            status = result_status(original, cell['cell_id'], execution, config)
                        except Exception as exc:
                            execution = {'returncode': 3, 'exception_type': type(exc).__name__, 'error': str(exc)}
                            status = 'accounting_or_execution_failure'
                        cell.update(status=status, finished_at=now(), execution=execution)
                        if status != 'complete':
                            failed = True
                    persist()
            try:
                verify_files(request['preserved_files'])
                verify_files(request['snapshot_files'])
                verify_ledger(read(config['api_budget']), config, request['original_reservations'])
            except Exception as exc:
                failed = True
                state['integrity_error'] = f'{type(exc).__name__}: {exc}'
            state.update(status='stopped_on_failure' if failed else 'drained_on_request' if stop_request else 'complete_with_quarantined_failure', finished_at=now())
            persist()
        finally:
            for sig, handler in previous_signals.items():
                signal.signal(sig, handler)
        return 2 if failed else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'verify', 'run', 'status'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--original', type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    if args.action == 'prepare':
        if args.original is None:
            parser.error('prepare requires --original')
        print(json.dumps(prepare(args.original, output)))
    elif args.action == 'verify':
        request, state, _ = verify(output, pristine=read(output / 'state.json')['status'] == 'prepared')
        print(json.dumps({'verified': True, 'status': state['status'], 'selected_cells': len(request['selected_cells'])}))
    elif args.action == 'status':
        print(json.dumps(read(output / 'summary.json'), indent=2))
    else:
        return run(output)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
