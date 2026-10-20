"""Register and independently revalidate fixed archived paper designs.

This study never generates, repairs, ranks or replaces a design and never calls
a model. Registration copies public inputs and fixed sources into a fresh run.
Only an explicit bounded launch can execute native, CSim or RTL work.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import signal
import sys
import uuid

from hlsmith import experiments as tools

SCHEMA = 'hlsmith.saved-paper-designs.v1'
TARGET = {'vitis_version': '2023.2', 'part': 'xcu280-fsvh2892-2L-e', 'clock_ns': 3.33}
COHORTS = ('frozen', 'clean')
TIMEOUTS = {'native': 180, 'csim': 180, 'rtl': 3600, 'whole_cell': 4800,
            'process': 4790, 'cleanup_allowance': 10}
EXPECTED = {'commercial_designs': 144, 'commercial_roles': 156, 'open_designs': 180,
            'open_slots': 216, 'open_linked_slots': 211, 'open_blocked_slots': 5}


def source_text(path):
    """Preserve source newlines exactly when passing text to an evaluator."""
    return Path(path).read_bytes().decode('utf-8')


def common_runner_identity(limits):
    record = limits['common_runner']
    verified = limits['tool_versions']['vitis-run']
    keys = ('binary', 'binary_sha256', 'observed_version')
    if (verified['status'] != 'verified' or verified['observed_version'] != '2023.2'
            or any(record.get(k) != verified.get(k) for k in keys)):
        raise ValueError('common runner differs from the verified Vitis identity')
    return {k: record[k] for k in keys}


def _copy_checked(source, destination, expected):
    source, destination = Path(source), Path(destination)
    if tools.sha(source) != expected:
        raise ValueError('archived input hash mismatch: ' + str(source))
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    if tools.sha(destination) != expected:
        raise ValueError('staged copy hash mismatch: ' + str(destination))


def fixed_selection(commercial, recovery, identity):
    """Resolve the paper's frozen roles, retaining unresolved comparators."""
    commercial_rows = commercial['rows']
    if len({r['validation_id'] for r in commercial_rows}) != len(commercial_rows):
        raise ValueError('commercial distinct validation IDs required')
    by_id = {}
    for row in recovery['rows']:
        key = row['validation_id']
        if key in by_id and any(row[k] != by_id[key][k] for k in ('kernel_sha256', 'input_hashes', 'golden_registry')):
            raise ValueError('conflicting archived open validation identity: ' + key)
        by_id[key] = row
    open_rows, slots = {}, []
    for row in identity['rows']:
        slot = {k: row.get(k) for k in ('model', 'workflow', 'benchmark', 'arm', 'validation_id',
                                      'kernel_sha256', 'status_1h', 'status_with_replay', 'rtl_cycles')}
        key = row.get('validation_id')
        slot['status'] = 'linked' if key else 'blocked_identity'
        slots.append(slot)
        if key:
            source = by_id[key]
            if row['kernel_sha256'] != source['kernel_sha256']:
                raise ValueError('paper selected-source identity mismatch: ' + key)
            open_rows[key] = source
    counts = {'commercial_designs': len(commercial_rows),
              'commercial_roles': sum(len(r['paper_slots']) for r in commercial_rows),
              'open_designs': len(open_rows), 'open_slots': len(slots),
              'open_linked_slots': sum(s['status'] == 'linked' for s in slots),
              'open_blocked_slots': sum(s['status'] == 'blocked_identity' for s in slots)}
    if counts != EXPECTED:
        raise ValueError('incomplete fixed paper denominator: ' + repr(counts))
    return commercial_rows, list(open_rows.values()), slots, counts


def register(args):
    paths = {k: Path(getattr(args, k)).resolve() for k in
             ('commercial_manifest', 'recovery_manifest', 'open_identity', 'registry_root', 'frozen_source', 'clean_source')}
    commercial, recovery, identity = (tools.read(paths[k]) for k in
                                      ('commercial_manifest', 'recovery_manifest', 'open_identity'))
    commercial_rows, open_rows, slots, counts = fixed_selection(commercial, recovery, identity)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    sources = {name: str(paths[name + '_source']) for name in COHORTS}
    source_files = {name: tools.source_manifest(root) for name, root in sources.items()}
    for name in ('commercial_manifest', 'recovery_manifest', 'open_identity'):
        _copy_checked(paths[name], output / 'evidence' / (name + '.json'), tools.sha(paths[name]))
    designs = []
    for family, rows in (('commercial', commercial_rows), ('open', open_rows)):
        for row in sorted(rows, key=lambda r: r['validation_id']):
            key = family + '__' + row['validation_id']
            directory = output / 'designs' / key
            _copy_checked(row['staged_kernel'], directory / 'selected.cpp', row['kernel_sha256'])
            inputs = directory / 'benchmark'
            for name, digest in row['input_hashes'].items():
                rel = Path(name)
                if rel.is_absolute() or '..' in rel.parts or 'gold' in rel.name.lower():
                    raise ValueError('non-public archived input: ' + name)
                _copy_checked(Path(row['staged_benchmark']) / rel, inputs / rel, digest)
            meta = tools.read(inputs / 'metadata.json')
            if str(meta.get('source_repo', '')).lower() != 'hlsfactory':
                raise ValueError('independent HLSFactory oracle required')
            testbench = inputs / (meta.get('testbench_file') or 'testbench.cpp')
            registry = row['golden_registry']
            if tools.sha(testbench) != registry['testbench_sha256']:
                raise ValueError('full public harness differs from the frozen registry')
            registry_path = output / 'registries' / (registry['sha256'] + '.json')
            if not registry_path.exists():
                _copy_checked(paths['registry_root'] / registry['path'], registry_path, registry['sha256'])
            designs.append({'id': key, 'family': family, 'benchmark': row['benchmark'],
                            'validation_id': row['validation_id'], 'source_sha256': row['kernel_sha256'],
                            'path': str(directory), 'input_hashes': row['input_hashes'],
                            'registry': str(registry_path), 'registry_sha256': registry['sha256'],
                            'expected_golden_sha256': row.get('historical_golden_output_sha256')
                                or row.get('current_preflight_native_golden_output_sha256'),
                            'paper_roles': row.get('paper_slots', []),
                            'archived_source_result': row.get('source_result'),
                            'archived_source_result_sha256': row.get('source_result_sha256'),
                            'generation_reference_isolation': row.get('historical_reference_audit_passed'),
                            'archived_synthesis_cycles': row.get('published_synthesis_cycles', row.get('csynth_cycles'))})
    jobs = [{'id': d['id'] + '__' + cohort, 'design': d['id'], 'cohort': cohort,
             'maximum_seconds': TIMEOUTS['whole_cell']} for d in designs for cohort in COHORTS]
    files = {str(p.resolve()): tools.sha(p) for folder in ('designs', 'registries', 'evidence')
             for p in (output / folder).rglob('*') if p.is_file()}
    manifest = {'schema': SCHEMA, 'created_at': tools.now(), 'counts': counts, 'designs': designs,
                'open_slots': slots, 'jobs': jobs, 'sources': sources, 'source_files': source_files,
                'staged_files': files, 'target': TARGET, 'timeouts': TIMEOUTS,
                'maximum_job_hours': sum(j['maximum_seconds'] for j in jobs) / 3600,
                'method': 'fixed saved-source independent validation with frozen and clean evaluators; no generation, repair or reselection',
                'validation': 'unchanged full public plain C/header/testbench/support files; ignore reduced cosim overrides',
                'limitations': ['Five unresolved open comparator identities remain blocked in both cohorts.',
                               'New output checks do not resolve historical generation reference exposure.',
                               'This gate does not regenerate original model cohorts, router, adapters or GRPO.'],
                'registration_only': True, 'provider_calls': 0}
    tools.save(output / 'manifest.json', manifest)
    tools.save(output / 'state.json', {'manifest_sha256': tools.sha(output / 'manifest.json'), 'status': 'registered',
                                     'job_seconds_charged': 0, 'jobs': [{'id': j['id'], 'status': 'pending'} for j in jobs]})
    return {'registered': str(output), 'counts': counts, 'cells': len(jobs), 'provider_calls': 0,
            'vitis_job_calls': 0, 'maximum_job_hours': manifest['maximum_job_hours']}


def verify(output):
    output = Path(output).resolve()
    manifest, state = tools.read(output / 'manifest.json'), tools.read(output / 'state.json')
    if (manifest.get('schema') != SCHEMA or state['manifest_sha256'] != tools.sha(output / 'manifest.json')
            or manifest['counts'] != EXPECTED or manifest['target'] != TARGET or manifest['timeouts'] != TIMEOUTS):
        raise ValueError('saved-design registration identity changed')
    expected_jobs = [{'id': d['id'] + '__' + c, 'design': d['id'], 'cohort': c,
                      'maximum_seconds': TIMEOUTS['whole_cell']} for d in manifest['designs'] for c in COHORTS]
    if manifest['jobs'] != expected_jobs or [j['id'] for j in state['jobs']] != [j['id'] for j in expected_jobs]:
        raise ValueError('fixed saved-design job identities changed')
    charged = state['job_seconds_charged']
    if isinstance(charged, bool) or not isinstance(charged, (int, float)) or not math.isfinite(charged) or charged < 0:
        raise ValueError('invalid aggregate compute accounting')
    for name, root in manifest['sources'].items():
        if tools.source_manifest(root) != manifest['source_files'][name]:
            raise ValueError('saved-design evaluator source changed: ' + name)
    for path, expected in manifest['staged_files'].items():
        if tools.sha(path) != expected:
            raise ValueError('saved source/public input changed: ' + path)
    actual_files = {str(p.resolve()) for folder in ('designs', 'registries', 'evidence')
                    for p in (output / folder).rglob('*') if p.is_file()}
    if actual_files != set(manifest['staged_files']):
        raise ValueError('saved-design staged file set changed')
    return manifest


def preflight(output, settings=None):
    manifest = verify(output)
    versions = tools.probe_vitis_versions(settings) if settings is not None else {}
    issues = [name + ' is not verified Vitis 2023.2' for name, record in versions.items()
              if record['status'] != 'verified']
    if shutil.which('g++') is None:
        issues.append('g++ is unavailable')
    result = {'complete': not issues, 'issues': issues, 'cells': len(manifest['jobs']),
              'counts': manifest['counts'], 'tool_versions': versions,
              'version_queries': sum(len(v['attempts']) for v in versions.values()),
              'provider_calls': 0, 'vitis_job_calls': 0, 'active_vitis_pids': tools.active_vitis(),
              'launch_requirement': 'explicit positive aggregate worker-hour cap and verified Vitis settings'}
    tools.save(Path(output) / 'preflight.json', result)
    return result


def _active_job_ids(state):
    active = state.get('active_jobs')
    if active is None:
        return [state['active_job']] if state.get('active_job') else []
    if (not isinstance(active, list) or not all(isinstance(job, str) for job in active)
            or len(active) != len(set(active))
            or (state.get('active_job') and state['active_job'] not in active)):
        raise ValueError('invalid active saved-design job identities')
    return active


def _controller_identity(receipt, *, current=False):
    controller = receipt.get('controller')
    if controller is None:
        return None  # Historical serial receipt.
    workers = controller.get('max_workers')
    try:
        valid_epoch = uuid.UUID(controller.get('execution_epoch', '')).hex == controller.get('execution_epoch')
    except (ValueError, AttributeError, TypeError):
        valid_epoch = False
    if (isinstance(workers, bool) or not isinstance(workers, int) or workers < 1
            or not valid_epoch
            or receipt.get('limits', {}).get('max_workers', workers) != workers
            or tools.sha(controller['path']) != controller.get('sha256')
            or (current and Path(controller['path']).resolve() != Path(__file__).resolve())):
        raise ValueError('saved-design dispatch controller identity changed')
    return controller


def _external_vitis(identities):
    """Exclude only observed descendants with matching PID/start identities."""
    pids = tools.active_vitis()
    if not pids:
        return []
    from hlsmith.toolchain.cosim_runtime_monitor import _process_table
    table = _process_table()
    owned = tools.owned_processes(os.getpid(), identities, table)
    names = {'vitis_hls', 'vitis-run', 'xsim', 'xsimk', 'v++', 'xelab', 'xvlog'}
    return [pid for pid in pids if pid in table and table[pid]['name'] in names
            and table[pid]['state'] != 'Z' and pid not in owned]


def launch(args):
    output = Path(args.output).resolve()
    maximum = tools.positive(args.max_compute_hours) * 3600
    workers = getattr(args, 'workers', 1)
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError('workers must be a positive integer')
    lock = (output / 'launch.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    manifest = verify(output)
    state = tools.read(output / 'state.json')
    if _active_job_ids(state) or any(row['status'] == 'running' for row in state['jobs']):
        raise ValueError('interrupted cell requires evidence review; no automatic rerun or budget refund')
    pending = [(job, row) for job, row in zip(manifest['jobs'], state['jobs']) if row['status'] == 'pending']
    if any((output / 'jobs' / job['id']).exists() for job, _ in pending):
        raise ValueError('pending saved-design workspace already exists; never overwrite or rerun it')
    checked = preflight(output, args.vitis_settings)
    if not checked['complete'] or checked['active_vitis_pids']:
        raise ValueError('tool version/resource preflight failed')
    scratch = Path(args.scratch_root).resolve()
    scratch.mkdir(parents=True, exist_ok=True)
    limits = {'compute_seconds': maximum, 'settings': str(Path(args.vitis_settings).resolve()),
              'settings_sha256': tools.sha(args.vitis_settings), 'scratch_root': str(scratch),
              'tool_versions': checked['tool_versions'],
              'common_runner': {k: checked['tool_versions']['vitis-run'][k]
                                for k in ('binary', 'binary_sha256', 'observed_version')}}
    previous = tools.read(output / 'launch.json') if (output / 'launch.json').is_file() else None
    if previous and any(row['status'] != 'pending' for row in state['jobs']):
        if (previous['limits']['scratch_root'] != limits['scratch_root']
                or common_runner_identity(previous['limits']) != common_runner_identity(limits)):
            raise ValueError('resumed saved designs require the same scratch root and common runner')
    controller = {'path': str(Path(__file__).resolve()), 'sha256': tools.sha(__file__),
                  'max_workers': workers, 'execution_epoch': uuid.uuid4().hex}
    limits['max_workers'] = workers
    receipt = {'manifest_sha256': tools.sha(output / 'manifest.json'), 'created_at': tools.now(),
               'limits': limits, 'controller': controller, 'provider_calls': 0}
    tools.save(output / 'launches' / (controller['execution_epoch'] + '.json'), receipt)
    tools.save(output / 'launch.json', receipt)
    stopping = False
    def stop(signum, frame):
        nonlocal stopping
        stopping = True
    handlers = {signum: signal.signal(signum, stop) for signum in (signal.SIGTERM, signal.SIGINT)}
    state['status'] = 'running'
    state['execution_epoch'] = controller['execution_epoch']
    active, uncertain = {}, []
    cursor, stop_reason, failure = 0, None, None
    owned = {}

    def save_state():
        ids = [job['id'] for job, _ in active.values()] + uncertain
        if ids:
            state['active_jobs'] = ids
        else:
            state.pop('active_jobs', None)
        # Keep scalar compatibility for old inspection tools and serial runs.
        if len(ids) == 1:
            state['active_job'] = ids[0]
        else:
            state.pop('active_job', None)
        tools.save(output / 'state.json', state)

    try:
        save_state()
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix='saved-design') as pool:
            while cursor < len(pending) or active:
                changed = False
                for future in [future for future in active if future.done()]:
                    changed = True
                    job, row = active.pop(future)
                    try:
                        result = future.result()
                        elapsed = tools.positive(result['elapsed_seconds'])
                    except Exception as error:
                        # Unknown work retains its full reservation and blocks resume.
                        uncertain.append(job['id'])
                        failure = failure or error
                        stop_reason = 'stopped_execution_error'
                        continue
                    state['job_seconds_charged'] -= job['maximum_seconds'] - elapsed
                    row.update(result, elapsed_seconds=elapsed)
                    folder = output / 'jobs' / job['id']
                    if not (folder / 'measurement.json').exists():
                        tools.save(folder / 'measurement.json', {
                            'job': job, 'status': 'whole_cell_timeout' if result['status'] == 'timeout' else 'execution_error',
                            'rtl': {'ran': False}, 'provider_calls': 0, 'dispatch_controller': controller,
                            'execution_epoch': controller['execution_epoch'], 'max_workers': workers,
                            'controller_sha256': controller['sha256']})
                    if elapsed > job['maximum_seconds']:
                        stop_reason = 'stopped_accounting_overrun'
                    elif result['status'] != 'complete':
                        stop_reason = stop_reason or 'stopped_execution_error'
                if stopping:
                    stop_reason = stop_reason or 'stopped_by_signal'
                if stop_reason is None and _external_vitis(owned):
                    stop_reason = 'stopped_external_vitis'
                if changed:
                    save_state()
                while stop_reason is None and not stopping and len(active) < workers and cursor < len(pending):
                    job, row = pending[cursor]
                    if state['job_seconds_charged'] + job['maximum_seconds'] > maximum:
                        # Other cells may settle below their reservations. Wait for
                        # them before deciding that the aggregate allowance is spent.
                        if not active:
                            stop_reason = 'stopped_compute_cap'
                        break
                    folder = output / 'jobs' / job['id']
                    folder.mkdir(parents=True, exist_ok=False)
                    state['job_seconds_charged'] += job['maximum_seconds']
                    row.update(status='running', reserved_seconds=job['maximum_seconds'],
                               execution_epoch=controller['execution_epoch'], max_workers=workers,
                               controller_sha256=controller['sha256'])
                    # Publish authorization and the full reservation atomically
                    # before a worker can read state or perform any work.
                    state['active_jobs'] = [item[0]['id'] for item in active.values()] + [job['id']]
                    if len(state['active_jobs']) == 1:
                        state['active_job'] = job['id']
                    else:
                        state.pop('active_job', None)
                    tools.save(output / 'state.json', state)
                    environment = {k: v for k, v in os.environ.items()
                                   if not k.startswith(('C2HLS_', 'HLSMITH_', 'OPENAI_', 'ANTHROPIC_', 'DEEPSEEK_'))}
                    environment.update(PYTHONDONTWRITEBYTECODE='1', PYTHON_DOTENV_DISABLED='1',
                                       PYTHONPATH=manifest['sources']['clean'])
                    future = pool.submit(tools.run_process,
                        [sys.executable, controller['path'], 'worker', '--output', str(output),
                         '--job', job['id'], '--execution-epoch', controller['execution_epoch']],
                        cwd=manifest['sources']['clean'], env=environment,
                        timeout=TIMEOUTS['process'], log=folder / 'worker.log')
                    active[future] = (job, row)
                    cursor += 1
                if active:
                    wait(active, timeout=.2, return_when=FIRST_COMPLETED)
                elif stop_reason:
                    break
        state['status'] = stop_reason or 'complete'
        save_state()
        if failure is not None:
            raise failure
    finally:
        # Signals stop dispatch and drain active bounded watchdogs; their owned
        # descendant cleanup finishes before the parent releases its launch lock.
        for signum, handler in handlers.items():
            signal.signal(signum, handler)
    return report(output)


def worker(args):
    output = Path(args.output).resolve()
    manifest = verify(output)
    state, receipt = tools.read(output / 'state.json'), tools.read(output / 'launch.json')
    if (state.get('status') != 'running' or args.job not in _active_job_ids(state)
            or receipt['manifest_sha256'] != tools.sha(output / 'manifest.json')):
        raise ValueError('worker requires an active bounded launch')
    controller = _controller_identity(receipt, current=True)
    if controller is not None and (getattr(args, 'execution_epoch', None) != controller['execution_epoch']
                                   or state.get('execution_epoch') != controller['execution_epoch']):
        raise ValueError('worker dispatch epoch differs from its active launch')
    limits = receipt['limits']
    if (state['job_seconds_charged'] > tools.positive(limits['compute_seconds'])
            or tools.sha(limits['settings']) != limits['settings_sha256']):
        raise ValueError('compute cap or Vitis settings identity mismatch')
    for tool in limits['tool_versions'].values():
        if tools.sha(tool['binary']) != tool['binary_sha256']:
            raise ValueError('verified Vitis binary changed')
    job = next(j for j in manifest['jobs'] if j['id'] == args.job)
    design = next(d for d in manifest['designs'] if d['id'] == job['design'])
    folder = output / 'jobs' / args.job
    if (folder / 'measurement.json').exists():
        raise ValueError('saved-design measurement already exists; never reuse it')
    scratch = Path(limits['scratch_root']) / args.job
    scratch.mkdir(parents=True, exist_ok=False)
    (scratch / 'user_home').mkdir()
    runner = common_runner_identity(limits)
    binary = runner['binary']
    for key in tuple(os.environ):
        if key.startswith(('C2HLS_', 'HLSMITH_', 'OPENAI_', 'ANTHROPIC_', 'DEEPSEEK_')):
            os.environ.pop(key)
    os.environ.update(C2HLS_VITIS_VERSION='2023.2', C2HLS_PART=TARGET['part'], C2HLS_CLOCK_NS='3.33',
                      C2HLS_FLOW_TARGET='vitis', C2HLS_VITIS_SETTINGS=limits['settings'],
                      C2HLS_CSIM_TIMEOUT='180', C2HLS_COSIM_TIMEOUT='3600', C2HLS_CPU_GOLDEN_TIMEOUT='180',
                      C2HLS_TMP_ROOT=str(scratch), TMPDIR=str(scratch), PYTHON_DOTENV_DISABLED='1',
                      PYTHONDONTWRITEBYTECODE='1', C2HLS_COSIM_TRACE_LEVEL='none', XILINX_LOCAL_USER_DATA='no',
                      C2HLS_VITIS_USER_HOME=str(scratch / 'user_home'),
                      HLS_TCL_RUNNER=shlex.quote(binary) + ' --tcl --input_file')
    os.environ['PATH'] = str(Path(binary).parent) + os.pathsep + os.environ.get('PATH', '')
    source = manifest['sources'][job['cohort']]
    sys.path.insert(0, source)
    import c2hls
    import hls_eval
    from golden_output import compare_hlsfactory_dumps
    if any(not Path(module.__file__).resolve().is_relative_to(Path(source).resolve()) for module in (c2hls, hls_eval)):
        raise ValueError('registered evaluator import identity mismatch')
    bench = Path(design['path']) / 'benchmark'
    meta = tools.read(bench / 'metadata.json')
    meta = {**meta, 'independent_golden_required': True}
    header, tb, plain = (meta.get('header_file') or 'kernel.h', meta.get('testbench_file') or 'testbench.cpp',
                         meta.get('plain_c_file') or 'plain.cpp')
    extras = [{'path': name, 'content': source_text(bench / name)} for name in design['input_hashes']
              if name not in {'metadata.json', header, tb, plain}]
    inputs = {'meta': meta, 'c_code': source_text(bench / plain), 'header_name': header,
              'header_code': source_text(bench / header), 'testbench_code': source_text(bench / tb), 'extra_files': extras}
    golden = c2hls._prepare_independent_golden(inputs, shape_registry_path=design['registry'])
    record = {'job': job, 'design_source_sha256': design['source_sha256'], 'status': 'blocked_golden',
              'golden_provenance': golden.get('provenance', {}), 'common_runner': runner,
              'input_hashes': design['input_hashes'], 'target': TARGET, 'provider_calls': 0,
              'csim': {'ran': False}, 'rtl': {'ran': False}}
    if controller is not None:
        record.update(dispatch_controller=controller, execution_epoch=controller['execution_epoch'],
                      max_workers=controller['max_workers'], controller_sha256=controller['sha256'])
    golden_sha = hashlib.sha256(golden.get('output', '').encode()).hexdigest()
    record['golden_output_sha256'] = golden_sha
    if golden.get('success') is True and design['expected_golden_sha256'] and golden_sha != design['expected_golden_sha256']:
        record['status'] = 'golden_identity_mismatch'
    if golden.get('error'):
        record['golden_error'] = golden['error']
    if (golden.get('success') is True and golden.get('output') and
            (not design['expected_golden_sha256'] or golden_sha == design['expected_golden_sha256'])):
        common = {'header_code': inputs['header_code'], 'header_name': header, 'extra_files': extras}
        code = source_text(Path(design['path']) / 'selected.cpp')
        if hashlib.sha256(code.encode('utf-8')).hexdigest() != design['source_sha256']:
            raise ValueError('selected source text differs from the registered source bytes')
        native = hls_eval.run_native_testbench(code, inputs['testbench_code'], **common, timeout=180)
        comparison = compare_hlsfactory_dumps(golden['output'], native.get('output', ''), specs=golden['specs']).to_dict()
        record.update(native={k: v for k, v in native.items() if k != 'output'}, native_comparison=comparison, status='failed_native')
        if native.get('passed') is True and comparison.get('passed') is True:
            common.update(top_function=meta.get('translated_hls_top') or meta.get('hls_top') or 'workload',
                          part=TARGET['part'], clock_ns=3.33, golden_output_text=golden['output'], golden_output_specs=golden['specs'])
            record['csim'] = hls_eval.run_csim(code, inputs['testbench_code'], **common, work_dir=str(scratch / 'csim'))
            record['status'] = 'csim_timeout' if tools.test_timed_out(record['csim']) else 'failed_csim'
            if tools.independently_passed(record['csim'], golden_sha):
                record['rtl'] = hls_eval.run_cosim(code, inputs['testbench_code'], **common,
                                                interface_depths=meta.get('cosim_depths') or {}, work_dir=str(scratch / 'rtl'))
                rtl = record['rtl']
                record['status'] = ('passed' if tools.independently_passed(rtl, golden_sha) and tools.measured_cycles(rtl) is not None
                                    else 'rtl_timeout' if tools.test_timed_out(rtl) else 'failed_rtl')
                if record['status'] == 'passed':
                    depths = meta.get('cosim_depths') or {}
                    materialized_code = hls_eval._inject_interface_depths(code, depths) if depths else code
                    materialized_path = scratch / 'rtl' / 'kernel.cpp'
                    expected_sha = hashlib.sha256(materialized_code.encode('utf-8')).hexdigest()
                    record['rtl_source_evidence'] = {
                        'selected_input_sha256': design['source_sha256'], 'path': str(materialized_path),
                        'expected_materialized_sha256': expected_sha,
                        'materialized_sha256': tools.sha(materialized_path) if materialized_path.is_file() else None,
                        'adaptation': 'evaluator interface-depth annotations from frozen public metadata'}
                    public_materialized = {name: ('testbench.cpp' if name == tb else name)
                                           for name in design['input_hashes'] if name not in {'metadata.json', plain}}
                    record['rtl_input_evidence'] = {name: {
                        'path': str(scratch / 'rtl' / relative), 'expected_sha256': design['input_hashes'][name],
                        'sha256': tools.sha(scratch / 'rtl' / relative) if (scratch / 'rtl' / relative).is_file() else None}
                        for name, relative in public_materialized.items()}
                    if (record['rtl_source_evidence']['materialized_sha256'] != expected_sha
                            or (rtl.get('target_code_sha256') is not None and rtl['target_code_sha256'] != design['source_sha256'])):
                        record['status'] = 'rtl_source_identity_mismatch'
                    if any(e['sha256'] != e['expected_sha256'] for e in record['rtl_input_evidence'].values()):
                        record['status'] = 'rtl_public_input_identity_mismatch'
                    synthesis, evidence = tools.fresh_rtl_synthesis(hls_eval, rtl)
                    record.update(rtl_synthesis=synthesis, rtl_synthesis_evidence=evidence,
                                  rtl_synthesis_sha256=tools.content_sha(synthesis),
                                  feasibility=tools.selected_feasibility(c2hls, synthesis, TARGET))
                    if record['status'] == 'passed' and not record['feasibility']['feasible']:
                        record['status'] = 'unqualified_rtl_synthesis'
    tools.save(folder / 'measurement.json', record)


def report(output):
    manifest = verify(output)
    state = tools.read(Path(output) / 'state.json')
    rows = []
    designs = {d['id']: d for d in manifest['designs']}
    latest_receipt = tools.read(Path(output) / 'launch.json') if (Path(output) / 'launch.json').is_file() else None
    latest_controller = _controller_identity(latest_receipt) if latest_receipt else None
    epochs = {}
    for job, progress in zip(manifest['jobs'], state['jobs']):
        path = Path(output) / 'jobs' / job['id'] / 'measurement.json'
        measured = tools.read(path) if path.exists() else {}
        if measured and measured.get('job') != job:
            raise ValueError('saved-design measured job identity mismatch')
        controller = measured.get('dispatch_controller')
        if measured and controller is None and any(
                record.get(key) is not None for record in (measured, progress)
                for key in ('execution_epoch', 'max_workers', 'controller_sha256')):
            raise ValueError('saved-design execution metadata lacks its controller')
        if controller is not None:
            epoch = controller.get('execution_epoch')
            if epoch not in epochs:
                _controller_identity({'controller': controller})
                dispatch = tools.read(Path(output) / 'launches' / (epoch + '.json'))
                _controller_identity(dispatch)
                if (dispatch.get('controller') != controller
                        or dispatch['manifest_sha256'] != tools.sha(Path(output) / 'manifest.json')
                        or latest_receipt is None
                        or dispatch['limits']['scratch_root'] != latest_receipt['limits']['scratch_root']
                        or common_runner_identity(dispatch['limits']) != common_runner_identity(latest_receipt['limits'])):
                    raise ValueError('saved-design execution epoch provenance changed')
                epochs[epoch] = controller
            if (epochs[epoch] != controller or measured.get('execution_epoch') != epoch
                    or measured.get('max_workers') != controller['max_workers']
                    or measured.get('controller_sha256') != controller['sha256']
                    or progress.get('execution_epoch') != epoch
                    or progress.get('max_workers') != controller['max_workers']
                    or progress.get('controller_sha256') != controller['sha256']):
                raise ValueError('saved-design execution metadata changed')
        if measured.get('status') == 'passed':
            golden_sha = measured.get('golden_output_sha256')
            design = designs[job['design']]
            if (measured.get('design_source_sha256') != design['source_sha256']
                    or measured.get('input_hashes') != design['input_hashes'] or measured.get('target') != TARGET
                    or not golden_sha or measured.get('golden_provenance', {}).get('output_sha256') != golden_sha
                    or measured.get('golden_provenance', {}).get('status') != 'passed'
                    or not measured.get('native_comparison', {}).get('passed')
                    or not tools.independently_passed(measured.get('csim', {}), golden_sha)
                    or not tools.independently_passed(measured.get('rtl', {}), golden_sha)
                    or tools.measured_cycles(measured.get('rtl', {})) is None
                    or not measured.get('feasibility', {}).get('feasible')):
                raise ValueError('passing saved design lacks independent output/resource/timing evidence')
            receipt = tools.read(Path(output) / 'launch.json')
            if (receipt['manifest_sha256'] != tools.sha(Path(output) / 'manifest.json')
                    or measured.get('common_runner') != common_runner_identity(receipt['limits'])):
                raise ValueError('passing saved design common runner differs from its launch receipt')
            evidence = measured.get('rtl_source_evidence') or {}
            work = Path(receipt['limits']['scratch_root']) / job['id'] / 'rtl'
            if (evidence.get('selected_input_sha256') != design['source_sha256']
                    or not evidence.get('materialized_sha256')
                    or evidence.get('materialized_sha256') != evidence.get('expected_materialized_sha256')
                    or Path(evidence['path']).resolve() != (work / 'kernel.cpp').resolve()
                    or tools.sha(evidence['path']) != evidence['materialized_sha256']
                    or (measured['rtl'].get('target_code_sha256') is not None
                        and measured['rtl']['target_code_sha256'] != design['source_sha256'])):
                raise ValueError('passing saved design RTL source association changed')
            meta = tools.read(Path(design['path']) / 'benchmark' / 'metadata.json')
            plain, tb = meta.get('plain_c_file') or 'plain.cpp', meta.get('testbench_file') or 'testbench.cpp'
            inputs = measured.get('rtl_input_evidence') or {}
            expected_inputs = set(design['input_hashes']) - {'metadata.json', plain}
            if set(inputs) != expected_inputs or any(
                    e.get('expected_sha256') != design['input_hashes'][name]
                    or e.get('sha256') != design['input_hashes'][name]
                    or Path(e['path']).resolve() != (work / ('testbench.cpp' if name == tb else name)).resolve()
                    or tools.sha(e['path']) != design['input_hashes'][name]
                    for name, e in inputs.items()):
                raise ValueError('passing saved design materialized public input changed')
            evidence = measured.get('rtl_synthesis_evidence') or {}
            if (evidence.get('status') != 'recorded' or not evidence.get('sha256')
                    or Path(evidence['path']).resolve().parent != (work / 'hls_proj/sol1/syn/report').resolve()
                    or tools.sha(evidence['path']) != evidence['sha256']
                    or tools.content_sha(measured.get('rtl_synthesis', {})) != measured.get('rtl_synthesis_sha256')):
                raise ValueError('passing saved design fresh synthesis evidence changed')
            clock = evidence.get('requested_clock') or {}
            allowed_sources = {'vitis_xml': Path(evidence['path']).resolve(),
                               'parsed_synthesis_report': Path(evidence['path']).resolve(),
                               'vitis_tcl': (work / 'run_cosim.tcl').resolve()}
            requested = measured['rtl_synthesis'].get('requested_clock_period_ns')
            if (clock.get('status') != 'verified' or clock.get('period_ns') != requested
                    or not clock.get('sources') or any(
                        source.get('kind') not in allowed_sources
                        or Path(source['path']).resolve() != allowed_sources[source['kind']]
                        or tools.sha(source['path']) != source.get('sha256')
                        or source.get('period_ns') != requested for source in clock['sources'])):
                raise ValueError('passing saved design requested clock provenance changed')
        maximum_workers = controller['max_workers'] if controller else progress.get('max_workers')
        mode = ('parallel' if maximum_workers > 1 else 'serial') if maximum_workers else ('legacy_serial' if measured else 'not_started')
        rows.append({**job, 'status': measured.get('status', progress['status']),
                     'measurement_observed': bool(measured) and measured.get('status') not in (None, 'pending', 'running'),
                     'execution_mode': mode, 'execution_epoch': controller['execution_epoch'] if controller else progress.get('execution_epoch'),
                     'max_workers': maximum_workers or (1 if measured else None),
                     'controller_sha256': controller['sha256'] if controller else progress.get('controller_sha256'),
                     'materialized_source_sha256': measured.get('rtl_source_evidence', {}).get('materialized_sha256'),
                     'rtl_cycles': tools.measured_cycles(measured.get('rtl', {})) if measured.get('status') == 'passed' else None})
    pairs = []
    for index in range(0, len(rows), 2):
        frozen, clean = rows[index:index + 2]
        pairs.append({'design': frozen['design'], 'frozen_status': frozen['status'], 'clean_status': clean['status'],
                      'frozen_execution_mode': frozen['execution_mode'], 'clean_execution_mode': clean['execution_mode'],
                      'frozen_execution_epoch': frozen['execution_epoch'], 'clean_execution_epoch': clean['execution_epoch'],
                      'both_outcomes_observed': frozen['measurement_observed'] and clean['measurement_observed'],
                      'mixed_execution_modes': (frozen['execution_mode'] != clean['execution_mode']
                          if frozen['measurement_observed'] and clean['measurement_observed'] else None),
                      'materialized_sources_match': (frozen['materialized_source_sha256'] == clean['materialized_source_sha256']
                          if frozen['materialized_source_sha256'] and clean['materialized_source_sha256'] else None),
                      'frozen_over_clean_rtl_cycles': (frozen['rtl_cycles'] / clean['rtl_cycles']
                          if frozen['status'] == clean['status'] == 'passed' else None)})
    result = {'schema': SCHEMA + '.report', 'counts': manifest['counts'], 'registered_cells': len(rows),
              'status_counts': dict(Counter(r['status'] for r in rows)),
              'cohort_counts': {c: dict(Counter(r['status'] for r in rows if r['cohort'] == c)) for c in COHORTS},
              'blocked_open_slots_per_cohort': [s for s in manifest['open_slots'] if s['status'] == 'blocked_identity'],
              'paired_denominator': len(pairs), 'paired_rtl_passes': sum(p['frozen_over_clean_rtl_cycles'] is not None for p in pairs),
              'paired_status_differences': sum(p['frozen_status'] != p['clean_status']
                   for p in pairs if p['both_outcomes_observed']),
              'paired_cycle_differences': sum(p['frozen_over_clean_rtl_cycles'] != 1
                   for p in pairs if p['frozen_over_clean_rtl_cycles'] is not None),
              'paired_materialized_source_differences': sum(p['materialized_sources_match'] is False for p in pairs),
              'mixed_execution_pairs': sum(p['mixed_execution_modes'] is True for p in pairs),
              'paired_execution_modes': dict(Counter(p['frozen_execution_mode'] + '/' + p['clean_execution_mode'] for p in pairs)),
              'execution_mode_status_counts': {mode: dict(Counter(r['status'] for r in rows if r['execution_mode'] == mode))
                                             for mode in sorted({r['execution_mode'] for r in rows})},
              'dispatch_controller': latest_controller, 'execution_epochs': epochs,
              'pairs': pairs, 'cells': rows, 'job_hours_charged': state['job_seconds_charged'] / 3600,
              'campaign_status': state['status'], 'provider_calls': 0, 'complete_paper_reproduction': False,
              'limitations': manifest['limitations']}
    tools.save(Path(output) / 'report.json', result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest='action', required=True)
    registration = actions.add_parser('register')
    for name in ('commercial-manifest', 'recovery-manifest', 'open-identity', 'registry-root', 'frozen-source', 'clean-source', 'output'):
        registration.add_argument('--' + name, type=Path, required=True)
    for action in ('preflight', 'report', 'launch', 'worker'):
        command = actions.add_parser(action)
        command.add_argument('--output', type=Path, required=True)
        if action in ('preflight', 'launch'):
            command.add_argument('--vitis-settings', type=Path, required=action == 'launch')
        if action == 'launch':
            command.add_argument('--max-compute-hours', type=float, required=True)
            command.add_argument('--scratch-root', type=Path, required=True)
            command.add_argument('--workers', type=int, default=1)
        if action == 'worker':
            command.add_argument('--job', required=True)
            command.add_argument('--execution-epoch')
    args = parser.parse_args(argv)
    if args.action == 'register':
        result = register(args)
    elif args.action == 'preflight':
        result = preflight(args.output, args.vitis_settings)
    elif args.action == 'launch':
        result = launch(args)
    elif args.action == 'worker':
        worker(args)
        return 0
    else:
        result = report(args.output)
    print(json.dumps(result, indent=2))
    return 2 if args.action == 'preflight' and not result['complete'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
