"""Register and independently revalidate fixed archived paper designs.

This study never generates, repairs, ranks or replaces a design and never calls
a model. Registration copies public inputs and fixed sources into a fresh run.
Only an explicit bounded launch can execute native, CSim or RTL work.
"""
from __future__ import annotations

import argparse
from collections import Counter
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


def launch(args):
    output = Path(args.output).resolve()
    maximum = tools.positive(args.max_compute_hours) * 3600
    lock = (output / 'launch.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    manifest = verify(output)
    state = tools.read(output / 'state.json')
    if state.get('active_job'):
        raise ValueError('interrupted cell requires evidence review; no automatic rerun or budget refund')
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
    tools.save(output / 'launch.json', {'manifest_sha256': tools.sha(output / 'manifest.json'),
                                      'created_at': tools.now(), 'limits': limits, 'provider_calls': 0})
    stopping = False
    def stop(signum, frame):
        nonlocal stopping
        stopping = True
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    state['status'] = 'running'
    for job, row in zip(manifest['jobs'], state['jobs']):
        if row['status'] != 'pending':
            continue
        if stopping:
            state['status'] = 'stopped_by_signal'
            break
        if state['job_seconds_charged'] + job['maximum_seconds'] > maximum:
            state['status'] = 'stopped_compute_cap'
            break
        if tools.active_vitis():
            state['status'] = 'stopped_external_vitis'
            break
        # Reserve the entire cell before spawning. A crash never refunds unknown work.
        state['job_seconds_charged'] += job['maximum_seconds']
        state.update(active_job=job['id'])
        row.update(status='running', reserved_seconds=job['maximum_seconds'])
        tools.save(output / 'state.json', state)
        folder = output / 'jobs' / job['id']
        folder.mkdir(parents=True, exist_ok=False)
        environment = {k: v for k, v in os.environ.items()
                       if not k.startswith(('C2HLS_', 'HLSMITH_', 'OPENAI_', 'ANTHROPIC_', 'DEEPSEEK_'))}
        environment.update(PYTHONDONTWRITEBYTECODE='1', PYTHON_DOTENV_DISABLED='1',
                           PYTHONPATH=manifest['sources']['clean'])
        result = tools.run_process([sys.executable, '-m', 'hlsmith.saved_designs', 'worker',
                                    '--output', str(output), '--job', job['id']],
                                   cwd=manifest['sources']['clean'], env=environment,
                                   timeout=TIMEOUTS['process'], log=folder / 'worker.log')
        elapsed = tools.positive(result['elapsed_seconds'])
        state['job_seconds_charged'] -= job['maximum_seconds'] - elapsed
        row.update(result, elapsed_seconds=elapsed)
        if not (folder / 'measurement.json').exists():
            tools.save(folder / 'measurement.json', {'job': job, 'status': 'whole_cell_timeout' if result['status'] == 'timeout'
                       else 'execution_error', 'rtl': {'ran': False}, 'provider_calls': 0})
        state.pop('active_job', None)
        tools.save(output / 'state.json', state)
        if elapsed > job['maximum_seconds']:
            state['status'] = 'stopped_accounting_overrun'
            break
        if result['status'] != 'complete':
            state['status'] = 'stopped_execution_error'
            break
    else:
        state['status'] = 'complete'
    tools.save(output / 'state.json', state)
    return report(output)


def worker(args):
    output = Path(args.output).resolve()
    manifest = verify(output)
    state, receipt = tools.read(output / 'state.json'), tools.read(output / 'launch.json')
    if (state.get('status') != 'running' or state.get('active_job') != args.job
            or receipt['manifest_sha256'] != tools.sha(output / 'manifest.json')):
        raise ValueError('worker requires an active bounded launch')
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
    for job, progress in zip(manifest['jobs'], state['jobs']):
        path = Path(output) / 'jobs' / job['id'] / 'measurement.json'
        measured = tools.read(path) if path.exists() else {}
        if measured and measured.get('job') != job:
            raise ValueError('saved-design measured job identity mismatch')
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
        rows.append({**job, 'status': measured.get('status', progress['status']),
                     'materialized_source_sha256': measured.get('rtl_source_evidence', {}).get('materialized_sha256'),
                     'rtl_cycles': tools.measured_cycles(measured.get('rtl', {})) if measured.get('status') == 'passed' else None})
    pairs = []
    for index in range(0, len(rows), 2):
        frozen, clean = rows[index:index + 2]
        pairs.append({'design': frozen['design'], 'frozen_status': frozen['status'], 'clean_status': clean['status'],
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
                   for p in pairs if p['frozen_status'] != 'pending' and p['clean_status'] != 'pending'),
              'paired_cycle_differences': sum(p['frozen_over_clean_rtl_cycles'] != 1
                   for p in pairs if p['frozen_over_clean_rtl_cycles'] is not None),
              'paired_materialized_source_differences': sum(p['materialized_sources_match'] is False for p in pairs),
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
        if action == 'worker':
            command.add_argument('--job', required=True)
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
