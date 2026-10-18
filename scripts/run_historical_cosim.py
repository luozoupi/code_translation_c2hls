#!/usr/bin/env python3
"""Bounded, resumable COSIM backfill with immutable source data and table overlays."""
from __future__ import annotations

import argparse
import csv
import fcntl
import io
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.scan_historical_cosim import OUT, REPO, TOOLCHAIN, atomic, file_sha, passed, positive, read


def now():
    return datetime.now(timezone.utc).isoformat()


def verify(row):
    registry = row.get('golden_registry')
    if not registry or file_sha(REPO / registry['path']) != registry['sha256']:
        raise ValueError('missing or changed frozen golden registry; create a versioned campaign')
    if file_sha(row['staged_kernel']) != row['kernel_sha256']:
        raise ValueError('staged kernel changed')
    for name, sha in row['input_hashes'].items():
        if file_sha(Path(row['staged_benchmark']) / name) != sha:
            raise ValueError(f'staged benchmark input changed: {name}')


from golden_output import compare_hlsfactory_cosim_dumps as compare_cosim_dumps


def verify_code(manifest):
    for name, sha in manifest['code_hashes'].items():
        if file_sha(REPO / name) != sha:
            raise ValueError(f'frozen evaluator changed: {name}')


@lru_cache(maxsize=4096)
def staged_meta(path):
    return read(Path(path) / 'metadata.json')


def schema_record(row, cosim):
    if not cosim.get('ran'):
        return None
    from export_schema_jsonl import SCHEMA_VERSION, TARGET_COSIM, _build_run, _build_problem, _build_implementation, _validate_record
    meta = staged_meta(row['staged_benchmark'])
    rec = dict(schema_version=SCHEMA_VERSION, report_type='rtl_sim',
               run=_build_run(TARGET_COSIM, TOOLCHAIN['part'], cosim.get('runtime_seconds'), TOOLCHAIN),
               problem=_build_problem(meta),
               implementation=_build_implementation(meta, variant_name=row.get('selected_step') or 'selected',
                   origin_override='c2hls_orchestrator' if row['kind'] == 'agentic' else 'c2hls',
                   origin_version=row.get('model') or 'historical', origin_meta=dict(
                       source_summary=row['source_summary'], source_result=row['source_result'],
                       historical_campaign=row['campaign'], source_code_sha256=row['kernel_sha256'],
                       backfill_validation_id=row['validation_id'], model=row.get('model'),
                       strategy=row.get('strategy'), skills=row.get('skill_mode'),
                       historical_reference_audit_passed=row.get('historical_reference_audit_passed'),
                       source_evidence=row.get('source_evidence'), validation_scope=cosim.get('validation_scope'),
                       cosim_skip_reason=cosim.get('skip_reason'), cosim_work_dir=cosim.get('work_dir'))),
               rtl_sim={k: cosim.get(k) for k in ('status', 'ran', 'passed', 'kernel_runtime_cycles',
                   'kernel_runtime_us', 'kernel_clock_freq_mhz', 'target_code_sha256', 'skip_reason', 'error')})
    errors = _validate_record(rec)
    if errors:
        raise ValueError(f'COSIM schema errors: {errors}')
    return rec


def row_measurement(row, results):
    state = row['inventory_status']
    existing = row.get('existing_cosim')
    value = results.get(row.get('validation_id'))
    if existing:
        return {**existing, 'validation_scope': 'historical_cosim', 'backfill_status': 'existing_cosim'}
    if value and value.get('cosim'):
        return {**value['cosim'], 'backfill_status': value['cosim']['status']}
    reason = row.get('reason') or (state if state != 'queued' else 'queued_for_backfill')
    return dict(status='not_run', ran=False, passed=False, kernel_runtime_cycles=None,
                target_code_sha256=row.get('kernel_sha256'), skip_reason=reason,
                backfill_status=state, validation_scope='not_executed')


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=fields)
    writer.writeheader()
    writer.writerows([{k: json.dumps(v, sort_keys=True) if isinstance(v, (list, dict)) else v
                      for k, v in row.items()} for row in rows])
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(out.getvalue())
    temp.replace(path)


def render(output, manifest, results, active=()):
    known = {r.get('validation_id') for r in manifest['rows']}
    results = {key: value for key, value in results.items() if key in known}
    records, schemas = [], []
    for row in manifest['rows']:
        measure = row_measurement(row, results)
        rec = {k: v for k, v in row.items() if k not in ('existing_cosim', 'input_hashes')}
        rec.update(cosim_backfill_status=measure['backfill_status'], cosim_status=measure['status'],
                   cosim_ran=measure['ran'], cosim_passed=measure['passed'],
                   cosim_cycles=measure.get('kernel_runtime_cycles') if passed(measure) else None,
                   cosim_error=measure.get('error'), cosim_skip_reason=measure.get('skip_reason'),
                   cosim_work_dir=measure.get('work_dir'), validation_scope=measure.get('validation_scope'))
        records.append(rec)
        if row.get('staged_benchmark') and measure.get('ran'):
            schemas.append(schema_record(row, measure))
    write_csv(output / 'inventory_and_results.csv', records)
    for name, values in [('inventory_and_results.jsonl', records), ('schema_records.jsonl', schemas)]:
        temp = output / (name + '.tmp')
        temp.write_text(''.join(json.dumps(r, sort_keys=True, allow_nan=False) + '\n' for r in values))
        temp.replace(output / name)
    from export_schema_jsonl import validate_jsonl
    validation = validate_jsonl(output / 'schema_records.jsonl')
    if validation['invalid']:
        raise ValueError(validation)
    summary = dict(updated_at=now(), historical_rows=len(records), unique_queued_validations=len({
        r['validation_id'] for r in manifest['rows'] if r['inventory_status'] == 'queued'}),
        finished_new_validations=len(results), active=list(active),
        row_status_counts=dict(Counter(r['cosim_backfill_status'] for r in records)),
        measurement_status_counts=dict(Counter(r['cosim']['status'] for r in results.values())),
        schema_validation=validation)
    atomic(output / 'summary.json', summary)
    by_campaign = {}
    for row in records:
        by_campaign.setdefault(row['campaign'], []).append(row)
    groups = []
    for campaign, rows in by_campaign.items():
        slug = campaign.replace('/', '__')
        write_csv(output / 'by_campaign' / (slug + '.csv'), rows)
        groups.append(dict(campaign=campaign, rows=len(rows), statuses=dict(Counter(r['cosim_backfill_status'] for r in rows))))
    write_csv(output / 'campaign_summary.csv', groups)
    # Companion tables keep every original synthesis value and setup label intact.
    lookup = {}
    for row in records:
        key = row['source_result']
        previous = lookup.get(key)
        if previous is None or row['cosim_passed']:
            lookup[key] = row
    overlays = []
    for dirname, filename in [
        ('sonnet46_skillv2_latency_matrix_20260726', 'sonnet46_skillv2_latency_records_long.csv'),
        ('skill_v3_qwen_gemma_matrix_20260731', 'all_rows.csv'),
        ('qwen_gemma_oneshot_agentic_sft_20260729', 'comparison_records.csv')]:
        source = REPO / 'results_sweeps' / dirname / filename
        if not source.exists():
            continue
        with source.open() as f:
            table = list(csv.DictReader(f))
        for row in table:
            match = lookup.get(row.get('source_result_path'))
            for field in ('cosim_backfill_status', 'cosim_cycles', 'cosim_error', 'cosim_skip_reason', 'validation_id', 'validation_scope'):
                row[field] = match.get(field) if match else ('not_in_inventory' if field == 'cosim_backfill_status' else None)
        dest = source.with_name(source.stem + '.cosim_backfill.csv')
        write_csv(dest, table)
        overlays.append(str(dest))
    lines = ['# July Historical COSIM Backfill', '',
             'Date window: July 1-31, 2026 (campaign dates; actual completion may extend into August).',
             'Saved final/selected kernels only. No generation calls or reference Vitis reruns.',
             'Vitis 2023.2, U280 xcu280-fsvh2892-2L-e, 3.33 ns, at most two workers.',
             'CSYNTH estimates and measured RTL cycles are separate columns. Missing cycles are never imputed.',
             'Queued/skipped validations are reported in tables only; website RTL records require an executed pass/fail/timeout.',
             f'Predicted skip: selected CSYNTH cycles > {manifest["gold_skip_ratio"]:g}x same-run reference CSYNTH cycles.',
             'Historical reference-isolation flags are retained; COSIM does not resolve generation leakage.',
             'New passes require native independent golden, candidate CSim, and RTL COSIM output checks.', '',
             '```json', json.dumps(summary, indent=2), '```', '', '## Updated Companion Tables', '']
    lines += ['- ' + p for p in overlays]
    (output / 'report.md').write_text('\n'.join(lines) + '\n')
    return summary


def load_results(raw):
    return {p.stem: read(p) for p in (raw / 'measurements').glob('*.json')}


def archive_work(work, raw, validation_id):
    archive = raw / 'vitis_archives' / (validation_id + '.tar.zst')
    archive.parent.mkdir(parents=True, exist_ok=True)
    pending = archive.with_suffix(archive.suffix + '.partial')
    subprocess.run(['tar', '--zstd', '-cf', str(pending), '-C', str(work.parent), work.name], check=True, timeout=600)
    subprocess.run(['tar', '--zstd', '-df', str(pending), '-C', str(work.parent)], check=True, timeout=600, capture_output=True)
    sha = file_sha(pending)
    pending.replace(archive)
    atomic(archive.with_suffix('.receipt.json'), dict(archive=str(archive), sha256=sha,
           source=str(work), tar_compare='passed', completed_at=now()))
    shutil.rmtree(work)
    return str(archive)


def worker(output, validation_id, timeout):
    manifest = read(output / 'manifest.json')
    verify_code(manifest)
    row = next(r for r in manifest['rows'] if r.get('validation_id') == validation_id and r['inventory_status'] == 'queued')
    verify(row)
    raw = Path(manifest['raw_root'])
    work = Path('/tmp') / output.name / validation_id
    work.mkdir(parents=True, exist_ok=True)
    vitis_user_home = work / 'user_home'
    vitis_user_home.mkdir(exist_ok=True)
    os.environ.update(C2HLS_VITIS_VERSION='2023.2', C2HLS_PART=TOOLCHAIN['part'], C2HLS_CLOCK_NS='3.33',
        C2HLS_FLOW_TARGET='vitis', C2HLS_VITIS_SETTINGS='/mnt/data/luo00466/Xilinx/Vitis/2023.2/settings64.sh',
        C2HLS_TMP_ROOT=str(work), C2HLS_COSIM_TIMEOUT=str(timeout), C2HLS_CSIM_TIMEOUT='180',
        C2HLS_COSIM_TRACE_LEVEL='none', XILINX_LOCAL_USER_DATA='no',
        C2HLS_VITIS_USER_HOME=str(vitis_user_home), TMPDIR=str(work))
    from c2hls import _prepare_independent_golden
    from hls_eval import run_csim, run_cosim
    bench = Path(row['staged_benchmark'])
    meta = read(bench / 'metadata.json')
    header_name = meta.get('header_file') or 'kernel.h'
    tb_name = meta.get('testbench_file') or 'testbench.cpp'
    extras = [dict(path=name, content=(bench / name).read_text()) for name in row['input_hashes']
              if name not in {'metadata.json', 'plain.cpp', header_name, tb_name}]
    inputs = dict(meta=meta, c_code=(bench / 'plain.cpp').read_text(),
                  header_name=header_name, header_code=(bench / header_name).read_text(),
                  testbench_code=(bench / tb_name).read_text(), extra_files=extras)
    result = dict(validation_id=validation_id, started_at=now(), row_id=row['record_id'],
                  kernel_sha256=row['kernel_sha256'], toolchain=TOOLCHAIN)
    started = time.monotonic()
    cosim = dict(status='not_run', ran=False, passed=False, kernel_runtime_cycles=None,
                 target_code_sha256=row['kernel_sha256'], validation_scope='independent_golden_csim_rtl_cosim')
    try:
        golden = _prepare_independent_golden(inputs, shape_registry_path=REPO / row['golden_registry']['path'])
        result['golden'] = golden['provenance']
        expected_golden = row.get('historical_golden_output_sha256')
        if expected_golden and golden.get('success') and golden['provenance'].get('output_sha256') != expected_golden:
            raise ValueError('native golden differs from the historical golden output')
        if not golden.get('success'):
            cosim.update(skip_reason='invalid_native_golden', error=golden.get('error'))
        else:
            kwargs = dict(header_name=header_name, top_function=meta.get('translated_hls_top') or meta.get('hls_top') or 'workload',
                          part=TOOLCHAIN['part'], clock_ns=3.33, extra_files=extras,
                          golden_output_text=golden['output'], golden_output_specs=golden['specs'])
            code = Path(row['staged_kernel']).read_text()
            csim = run_csim(code, inputs['testbench_code'], inputs['header_code'], work_dir=str(work / 'csim'), **kwargs)
            (work / 'csim.log').write_text(csim.pop('log', ''))
            result['csim'] = csim
            if not passed(csim):
                cosim.update(skip_reason='candidate_csim_failed', error=csim.get('error'))
            else:
                cosim['ran'] = True
                measure = run_cosim(code, inputs['testbench_code'], inputs['header_code'], work_dir=str(work / 'cosim'),
                                    interface_depths=meta.get('cosim_depths') or {}, **kwargs)
                log = measure.pop('log', '')
                (work / 'cosim.log').write_text(log)
                ok = passed(measure) and positive(measure.get('kernel_runtime_cycles')) is not None
                error = measure.get('error') or ('' if ok else 'COSIM produced no valid measured cycles')
                cosim.update(measure, status='pass' if ok else ('timeout' if 'timed out' in error.lower() else 'fail'),
                             ran=True, passed=bool(ok), error=error)
                if not ok:
                    cosim['kernel_runtime_cycles'] = None
                    cosim['kernel_runtime_us'] = None
    except Exception as exc:
        cosim.update(status='fail' if cosim['ran'] else 'not_run',
                     skip_reason=None if cosim['ran'] else 'worker_error', error=f'{type(exc).__name__}: {exc}')
    cosim['runtime_seconds'] = round(time.monotonic() - started, 3)
    result.update(cosim=cosim, finished_at=now())
    try:
        archive = archive_work(work, raw, validation_id)
        cosim['work_dir'] = archive
    except Exception as exc:
        result['archive_error'] = str(exc)
    schema_record(row, cosim)
    atomic(raw / 'measurements' / (validation_id + '.json'), result)
    print('COSIM_DONE', validation_id, cosim['status'], cosim.get('kernel_runtime_cycles'), flush=True)


def external_vitis(parent_pid):
    procs = subprocess.check_output(['ps', '-eo', 'pid=,ppid=,comm='], text=True)
    parents, names = {}, {}
    for line in procs.splitlines():
        pid, ppid, name = line.split(None, 2)
        parents[int(pid)], names[int(pid)] = int(ppid), name
    external = []
    for pid, name in names.items():
        if name not in ('vitis_hls', 'vitis-run', 'xsim', 'xsimk', 'vivado'):
            continue
        seen, cur = set(), pid
        while cur in parents and cur not in seen and cur != parent_pid:
            seen.add(cur)
            cur = parents[cur]
        if cur != parent_pid:
            external.append(pid)
    return external


def run(output, workers, timeout, limit):
    manifest = read(output / 'manifest.json')
    verify_code(manifest)
    raw = Path(manifest['raw_root'])
    lock = (output / 'queue.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    results = load_results(raw)
    cases = {}
    for row in manifest['rows']:
        if row['inventory_status'] == 'queued' and row['validation_id'] not in results:
            cases.setdefault(row['validation_id'], row)
    pending = sorted(cases, key=lambda key: (cases[key]['csynth_cycles'], key))
    if limit:
        pending = pending[:limit]
    (raw / 'logs').mkdir(parents=True, exist_ok=True)
    active, completed = {}, 0
    stopping = False
    def stop(signum, frame):
        nonlocal stopping
        stopping = True
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    def launch(key):
        cmd = [sys.executable, '-u', str(Path(__file__).resolve()), 'worker', '--output', str(output),
               '--validation-id', key, '--timeout', str(timeout)]
        with (raw / 'logs' / (key + '.log')).open('a') as log:
            return subprocess.run(cmd, cwd=REPO, stdout=log, stderr=subprocess.STDOUT).returncode
    summary = render(output, manifest, results, active)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        while pending or active:
            for key, future in list(active.items()):
                if not future.done():
                    continue
                rc = future.result()
                path = raw / 'measurements' / (key + '.json')
                if path.exists():
                    results[key] = read(path)
                else:
                    results[key] = dict(validation_id=key, cosim=dict(status='not_run', ran=False, passed=False,
                        kernel_runtime_cycles=None, error=f'worker exited {rc}; see durable log',
                        target_code_sha256=cases[key]['kernel_sha256']))
                    atomic(path, results[key])
                del active[key]
                completed += 1
            gate = dict(external_vitis=external_vitis(os.getpid()), free_gib=shutil.disk_usage('/tmp').free / 2**30,
                        load_ratio=os.getloadavg()[0] / (os.cpu_count() or 1))
            while pending and not stopping and len(active) < workers and not gate['external_vitis'] and gate['free_gib'] >= 30 and gate['load_ratio'] < 0.7:
                key = pending.pop(0)
                verify(cases[key])
                active[key] = pool.submit(launch, key)
                print('START', key, cases[key]['benchmark'], cases[key]['model'], flush=True)
            summary = render(output, manifest, results, active)
            atomic(output / 'state.json', dict(updated_at=now(), pid=os.getpid(), pending=len(pending), active=list(active),
                   completed_this_invocation=completed, resource_gate=gate, status='draining' if stopping else 'running'))
            if stopping and not active:
                break
            if pending or active:
                time.sleep(20)
    atomic(output / 'state.json', dict(updated_at=now(), pid=os.getpid(), pending=len(pending), active=[],
           status='stopped' if stopping else ('smoke_complete' if limit else 'complete'), completed_this_invocation=completed))
    print('QUEUE_DONE', json.dumps(summary), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['run', 'worker', 'report'])
    parser.add_argument('--output', type=Path, default=OUT)
    parser.add_argument('--workers', type=int, choices=[1, 2], default=2)
    parser.add_argument('--timeout', type=int, default=3600)
    parser.add_argument('--validation-id')
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args()
    if args.command == 'worker':
        worker(args.output, args.validation_id, args.timeout)
    elif args.command == 'report':
        manifest = read(args.output / 'manifest.json')
        print(json.dumps(render(args.output, manifest, load_results(Path(manifest['raw_root']))), indent=2))
    else:
        run(args.output, args.workers, args.timeout, args.limit)


if __name__ == '__main__':
    main()
