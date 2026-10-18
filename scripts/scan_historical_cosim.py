#!/usr/bin/env python3
"""Inventory and stage saved July kernels without changing historical artifacts."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
OLD = Path('/home/luo00466/code_translation-c2hls')
CAMPAIGN = 'july_cosim_registry_v2_20260928'
OUT = REPO / 'results_sweeps' / CAMPAIGN
RAW = Path('/mnt/data2/luo00466/c2hls_rl') / CAMPAIGN
TOOLCHAIN = dict(vitis_version='2023.2', part='xcu280-fsvh2892-2L-e',
                 clock_ns=3.33, flow_target='vitis')


def digest(value):
    if isinstance(value, bytes):
        return hashlib.sha256(value).hexdigest()
    return digest(json.dumps(value, sort_keys=True, separators=(',', ':')).encode())


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()



def frozen_evaluator_hashes():
    names = {'scripts/scan_historical_cosim.py', 'scripts/run_historical_cosim.py',
             'configs/hlsfactory_output_shapes.json',
             'configs/hlsfactory_output_shapes_fc27133.json'}
    # Include indirect evaluator dependencies (telemetry, budgets, suite, feedback).
    names.update(path.name for path in REPO.glob('*.py'))
    return {name: file_sha(REPO / name) for name in sorted(names)}


def validation_identity(code_sha256, input_hashes, golden_registry):
    return dict(schema_version='c2hls.historical-validation-identity.v2',
                code_sha256=code_sha256, inputs=input_hashes, toolchain=TOOLCHAIN,
                golden_registry=golden_registry, evaluator_hashes=frozen_evaluator_hashes())


def read(path):
    return json.loads(Path(path).read_text())


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    temp.replace(path)


def positive(value):
    try:
        v = float(value)
        return v if 0 < v < float('inf') else None
    except (TypeError, ValueError):
        return None


def passed(value):
    if value is True:
        return True
    if not isinstance(value, dict):
        return str(value).lower() in ('pass', 'passed')
    if value.get('passed') is False or value.get('success') is False:
        return False
    return value.get('passed') is True or value.get('status') in ('pass', 'passed')


class Resolver:
    """Recover only small source/result files from checksum-verified bundles."""
    def __init__(self, raw):
        self.raw = Path(raw)
        self.mappings = []
        self.cosim_supplements = {}

    def recover_archives(self):
        for repo in (REPO, OLD):
            for rel in ('results_sweeps', 'artifacts/vllm_vitis_smoke',
                        'artifacts/vllm_cosim_compare'):
                src = repo / rel
                if not (src / 'payload.tar.zst').is_file():
                    continue
                meta = read(src / 'metadata.json')
                payload = src / 'payload.tar.zst'
                target = self.raw / 'recovered' / repo.name / rel
                receipt = target.parent / (target.name + '.receipt.json')
                expected = meta['archive']['sha256']
                if not receipt.exists() or read(receipt).get('payload_sha256') != expected:
                    if file_sha(payload) != expected:
                        raise ValueError(f'archive checksum mismatch: {payload}')
                    print(f'RECOVER {src}', flush=True)
                    count = 0
                    proc = subprocess.Popen(['zstd', '-dc', str(payload)], stdout=subprocess.PIPE)
                    try:
                        with tarfile.open(fileobj=proc.stdout, mode='r|') as archive:
                            for item in archive:
                                parts = PurePosixPath(item.name).parts
                                if not parts or parts[0] != src.name or '..' in parts:
                                    raise ValueError(f'unsafe archive member: {item.name}')
                                if not item.isfile() or item.size > 20 * 1024**2:
                                    continue
                                name = parts[-1]
                                if not (name.endswith(('.json', '.cpp', '.h'))):
                                    continue
                                # Histories are not needed to recover the saved selected code.
                                if name.endswith('_history.json'):
                                    continue
                                dest = target.joinpath(*parts[1:])
                                dest.parent.mkdir(parents=True, exist_ok=True)
                                with archive.extractfile(item) as f, dest.open('wb') as out:
                                    shutil.copyfileobj(f, out)
                                count += 1
                        if proc.wait(timeout=60):
                            raise RuntimeError(f'archive decompression failed: {payload}')
                    finally:
                        if proc.poll() is None:
                            proc.kill()
                            proc.wait()
                    atomic(receipt, dict(source=str(src), payload_sha256=expected, files=count))
                self.mappings.append((str(src), target))

    def resolve(self, value):
        p = Path(value)
        if p.exists():
            return p
        for src, dest in self.mappings:
            if str(p).startswith(src + '/'):
                candidate = dest / str(p)[len(src) + 1:]
                if candidate.exists():
                    return candidate
        # These directories were moved intact during the September cleanup.
        if str(p).startswith('/home/luo00466/tmp_c2hls_'):
            candidate = Path('/mnt/data1/luo00466') / p.relative_to('/home/luo00466')
            if candidate.exists():
                return candidate
        return p

    def load_supplements(self):
        p = REPO / 'artifacts/qwen_cosim_enrichment/strong_gains_20260725_cosim/results.jsonl'
        if p.is_file():
            for line in p.read_text().splitlines():
                row = json.loads(line)
                if passed(row.get('cosim')):
                    self.cosim_supplements[row['source_result']] = row
        for source, base in self.mappings:
            if not source.endswith('vllm_cosim_compare'):
                continue
            for p in base.rglob('summary.json'):
                data = read(p)
                for r in data.get('rows', []):
                    if r.get('vllm_cosim_status') != 'pass' or not r.get('generated_code_path'):
                        continue
                    work = self.resolve(str(Path(r.get('vllm_cosim_work_dir') or '/nonexistent') / 'kernel.cpp'))
                    if not work.is_file():
                        continue
                    self.cosim_supplements[r['generated_code_path']] = dict(
                        toolchain=data.get('vitis') or {}, benchmark_dir=None,
                        cosim=dict(status='pass', success=True, passed=True, ran=True,
                            kernel_runtime_cycles=r.get('vllm_cosim_cycles'),
                            kernel_runtime_us=r.get('vllm_cosim_runtime_us'),
                            kernel_clock_freq_mhz=r.get('vllm_cosim_clock_freq_mhz'),
                            work_dir=r.get('vllm_cosim_work_dir'), target_code_sha256=file_sha(work)))


def eligible_date(value, start, end):
    dates = re.findall(r'20\d{6}', str(value))
    return any(start <= x < end for x in dates)


def descriptors(resolver, start, end):
    for repo in (REPO, OLD):
        for source in sorted((repo / 'artifacts').glob('*.summary.json')):
            if not eligible_date(source.name, start, end):
                continue
            data = read(source)
            for i, row in enumerate(data.get('rows', [])):
                if not str(row.get('bench', '')).startswith('hlsfactory_'):
                    continue
                yield dict(source_summary=str(source), source_row=i,
                           source_result=(row.get('current') or {}).get('json', ''),
                           benchmark=row['bench'], benchmark_dir=row.get('bench_dir'),
                           campaign=source.stem, model=row.get('model_id', row.get('model')),
                           skill_mode=row.get('skill_mode'), strategy=('flash' if 'flash' in source.name else 'multistep'),
                           historical_best=(row.get('current') or {}).get('best'), kind='agentic')
        original = repo / 'artifacts/vllm_vitis_smoke'
        base = next((dest for src, dest in resolver.mappings if src == str(original)), original)
        for source in sorted(base.rglob('summary.json')):
            if not eligible_date(str(source.relative_to(base)), start, end):
                continue
            data = read(source)
            for i, row in enumerate(data.get('rows', [])):
                bench = row.get('benchmark') or row.get('bench')
                if not str(bench or '').startswith('hlsfactory_'):
                    continue
                code_path = row.get('generated_code_path') or row.get('code_path') or row.get('generated_cpp') or ''
                if code_path and not Path(code_path).is_absolute():
                    code_path = str(repo / code_path)
                yield dict(source_summary=str(original / source.relative_to(base)), source_row=i,
                           source_result=code_path,
                           benchmark=bench, benchmark_dir=row.get('benchmark_dir') or row.get('bench_dir'),
                           campaign=str(source.parent.relative_to(base)), model=data.get('model') or row.get('model'),
                           strategy='one_shot', skill_mode='none', kind='one_shot', direct_row=row,
                           direct_summary={k: v for k, v in data.items() if k != 'rows'}, source_repo=str(repo))
    for rel in ('corrected_setup_matrix_v2/corrected_matrix_records.jsonl',
                'skill_dose_pilot_v1/skill_dose_records.jsonl',
                'skill_dose_smoke_v1/skill_dose_records.jsonl'):
        source = REPO / 'artifacts' / rel
        if not source.exists():
            continue
        campaign_meta = read(source.parent / 'experiment_manifest.json')
        if not eligible_date(str(campaign_meta.get('created_at', '')).replace('-', ''), start, end):
            continue
        for i, line in enumerate(source.read_text().splitlines()):
            r = json.loads(line)
            for item in r.get('candidates', [r]):
                setup = str(item.get('setup_id') or '')
                yield dict(source_summary=str(source), source_row=i, source_result=item['result_path'],
                           benchmark=r['benchmark'], benchmark_dir=None, campaign=source.parent.name,
                           model=r.get('model'), strategy='flash' if 'flash' in setup else r.get('strategy', 'multistep'),
                           skill_mode=setup.split(':')[-1] if setup else f"dose_{r.get('requested_positive_skill_count')}_{r.get('prompt_policy')}_sample{r.get('sample_index')}",
                           kind='agentic', historical_best=None)


def benchmark_files(bench_dir):
    meta = read(bench_dir / 'metadata.json')
    names = {'metadata.json', 'plain.cpp', meta.get('header_file') or 'kernel.h',
             meta.get('testbench_file') or 'testbench.cpp', *(meta.get('support_files') or [])}
    names.update(str(p.relative_to(bench_dir)) for p in bench_dir.glob('*.h'))
    names.update(str(p.relative_to(bench_dir)) for p in (bench_dir / 'support').rglob('*') if p.is_file())
    for name in names:
        if Path(name).is_absolute() or '..' in Path(name).parts:
            raise ValueError(f'unsafe benchmark input path: {name}')
    return meta, {n: file_sha(bench_dir / n) for n in sorted(names) if (bench_dir / n).is_file()}


def selected_source(result, resolver, result_path=None):
    code = result.get('hls_code') or result.get('final_code')
    report = result.get('final_report') or {}
    path = resolver.resolve(str(Path(report.get('work_dir') or '/nonexistent') / 'kernel.cpp'))
    evidence = 'saved_final_code'
    if not code and result_path:
        final = Path(result_path).with_name(str(result.get('benchmark')) + '_final.cpp')
        if final.is_file():
            code, evidence = final.read_text(), 'saved_final_cpp'
    if not code and path.is_file():
        code, evidence = path.read_text(), 'selected_report_source'
    if not code:
        raise ValueError('saved_selected_code_missing')
    sha = digest(code.encode())
    expected = result.get('selected_code_sha256')
    if expected and sha != expected:
        raise ValueError('selected_code_hash_mismatch')
    if path.is_file() and file_sha(path) != sha:
        raise ValueError('selected_code_report_source_mismatch')
    return code, sha, ('recorded_selected_hash' if expected else evidence + ('_and_synth_source' if path.is_file() else ''))


def measured_cosim(result, code_hash, resolver):
    values = [result.get('cosim') or {}, result.get('baseline_cosim') or {}]
    values += [s.get('cosim') or {} for s in result.get('steps', [])]
    for item in values:
        if not passed(item) or not positive(item.get('kernel_runtime_cycles')):
            continue
        target = item.get('target_code_sha256')
        if not target and item is result.get('cosim'):
            target = result.get('cosim_target_code_sha256')
        path = resolver.resolve(str(Path(item.get('work_dir') or '/nonexistent') / 'kernel.cpp'))
        if not target and path.is_file():
            target = file_sha(path)
        if target == code_hash:
            return {**item, 'status': 'pass', 'ran': True, 'target_code_sha256': code_hash}
    return None


def stage(desc, resolver, raw, ratio):
    row = dict(desc)
    row.pop('direct_row', None)
    row.pop('direct_summary', None)
    row['record_id'] = digest([desc['source_summary'], desc['source_row'], desc['source_result']])[:20]
    row['inventory_status'] = 'blocked'
    try:
        result_path = resolver.resolve(desc['source_result'])
        if desc['kind'] == 'one_shot':
            direct = desc['direct_row']
            if not result_path.is_file():
                raise ValueError('one_shot_code_missing')
            code = result_path.read_text()
            sha, evidence = digest(code.encode()), 'one_shot_saved_code'
            result = dict(direct)
            result['final_report'] = direct.get('csynth') or direct.get('synth') or {}
            run = direct.get('vitis') or desc['direct_summary'].get('vitis') or {}
            run = {**run, 'vitis_version': run.get('version') or run.get('vitis_version')}
            source_jsonl = Path(direct.get('source_jsonl') or '/nonexistent')
            if not source_jsonl.is_absolute():
                source_jsonl = Path(desc['source_repo']) / source_jsonl
            source_jsonl = resolver.resolve(source_jsonl)
            if source_jsonl.is_file():
                generation = json.loads(source_jsonl.read_text().splitlines()[int(direct['source_line_index'])])
                run['model'] = (generation.get('request') or {}).get('model')
                row['benchmark_dir'] = ((generation.get('record') or {}).get('metadata') or {}).get('benchmark_dir')
        else:
            if not result_path.is_file():
                raise ValueError('result_json_missing')
            result = read(result_path)
            code, sha, evidence = selected_source(result, resolver, result_path)
            run = result.get('run') or {}
        row['resolved_result'] = str(result_path)
        row['source_result_sha256'] = file_sha(result_path)
        row['kernel_sha256'] = sha
        row['source_evidence'] = evidence
        row['model'] = run.get('model') or row['model']
        report = result.get('final_report') or {}
        row['csynth_cycles'] = positive(report.get('latency_cycles_worst')) or positive(report.get('latency_cycles'))
        expected_cycles = positive((desc.get('historical_best') or {}).get('cycles'))
        if expected_cycles and expected_cycles != positive(report.get('latency_cycles')):
            raise ValueError('summary_selected_cycles_mismatch')
        row['historical_success'] = result.get('success')
        row['historical_reference_audit_passed'] = (result.get('reference_isolation_audit') or {}).get('passed')
        row['historical_csim_passed'] = passed(result.get('csim'))
        row['historical_golden_output_sha256'] = (result.get('independent_golden') or {}).get('output_sha256')
        row['historical_feasibility'] = result.get('candidate_feasibility')
        row['selected_step'] = (result.get('best_so_far_promotion') or {}).get('from_step_name') or (row.get('historical_best') or {}).get('step') or result.get('phase')
        row['historical_toolchain'] = {k: run.get(k) for k in TOOLCHAIN}
        row['toolchain_match'] = all(str(run.get(k)) == str(v) for k, v in TOOLCHAIN.items())
        if any(run.get(k) is not None and str(run[k]) != str(v) for k, v in TOOLCHAIN.items()):
            raise ValueError('historical_toolchain_mismatch')
        row['toolchain_evidence'] = 'complete' if row['toolchain_match'] else 'partial_revalidated_on_backfill'
        if not row['csynth_cycles']:
            raise ValueError('no_successful_selected_csynth')
        candidates = [Path(row['benchmark_dir'])] if row.get('benchmark_dir') else []
        candidates += [REPO / 'benchmarks_external/HLSFactory/polybench_float_small' / desc['benchmark']]
        bench = next((p for p in candidates if (p / 'metadata.json').is_file()), None)
        if bench is None:
            raise ValueError('benchmark_inputs_missing')
        meta, hashes = benchmark_files(bench)
        saved_inputs = (((result.get('run_fingerprint') or {}).get('payload') or {}).get('benchmark') or {}).get('inputs') or {}
        recorded_hashes = {f['path']: f['sha256'] for f in saved_inputs.get('files', [])}
        mismatches = [name for name, sha in hashes.items() if name != 'metadata.json'
                      and recorded_hashes.get(name) and recorded_hashes[name] != sha]
        if mismatches:
            raise ValueError('historical_harness_changed:' + ','.join(mismatches))
        row['historical_harness_evidence'] = 'run_fingerprint' if recorded_hashes else 'original_benchmark_path_current_snapshot'
        if not meta.get('supports_cosim'):
            raise ValueError('benchmark_does_not_support_cosim')
        if 'plain.cpp' not in hashes or (meta.get('testbench_file') or 'testbench.cpp') not in hashes:
            raise ValueError('testbench_or_plain_source_missing')
        row['benchmark_dir'] = str(bench)
        from hlsfactory_registry import registry_identity
        row['golden_registry'] = registry_identity(meta, (bench / (meta.get('testbench_file') or 'testbench.cpp')).read_text())
        # Include the entire harness, not just benchmark name, in the reuse key.
        identity = validation_identity(sha, hashes, row['golden_registry'])
        row['validation_id'] = digest(identity)
        row['input_hashes'] = hashes
        staged = raw / 'inputs' / row['validation_id']
        staged.mkdir(parents=True, exist_ok=True)
        for name in hashes:
            dest = staged / 'benchmark' / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            if not dest.exists():
                shutil.copy2(bench / name, dest)
            if file_sha(dest) != hashes[name]:
                raise ValueError('staged_input_hash_mismatch')
        kernel = staged / 'kernel.cpp'
        if not kernel.exists():
            kernel.write_text(code)
        if file_sha(kernel) != sha:
            raise ValueError('staged_kernel_hash_mismatch')
        row['staged_kernel'] = str(kernel)
        row['staged_benchmark'] = str(staged / 'benchmark')
        existing = measured_cosim(result, sha, resolver)
        supplement = resolver.cosim_supplements.get(desc['source_result'])
        if not existing and supplement:
            measure = supplement.get('cosim') or {}
            recorded_tc = supplement.get('toolchain') or (supplement.get('schema_record') or {}).get('run') or {}
            recorded_tc = {**recorded_tc, 'vitis_version': recorded_tc.get('vitis_version') or recorded_tc.get('version'),
                           'part': recorded_tc.get('part') or recorded_tc.get('device')}
            if (measure.get('target_code_sha256') == sha and passed(measure)
                    and positive(measure.get('kernel_runtime_cycles'))
                    and all(str(recorded_tc.get(k)) == str(v) for k, v in TOOLCHAIN.items())):
                existing = {**measure, 'supplement_source_result': desc['source_result']}
        row['existing_cosim'] = existing
        row['prior_cosim_status'] = (result.get('cosim') or {}).get('status')
        ref = result.get('ground_truth_report') or {}
        row['reference_csynth_cycles'] = positive(ref.get('latency_cycles_worst')) or positive(ref.get('latency_cycles'))
        row['gold_skip_ratio'] = ratio
        if existing:
            row['inventory_status'] = 'existing_cosim'
        elif row['reference_csynth_cycles'] and row['csynth_cycles'] > ratio * row['reference_csynth_cycles']:
            row['inventory_status'] = 'predicted_longer_than_gold'
        else:
            row['inventory_status'] = 'queued'
        row['reason'] = ''
    except (ValueError, OSError, KeyError, TypeError) as exc:
        row['reason'] = str(exc)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUT)
    parser.add_argument('--raw-root', type=Path, default=RAW)
    parser.add_argument('--start', default='20260701')
    parser.add_argument('--end', default='20260801')
    parser.add_argument('--gold-skip-ratio', type=float, default=10.0)
    parser.add_argument('--benches', default='', help='Optional comma-separated smoke subset')
    parser.add_argument('--one-per-benchmark', action='store_true')
    args = parser.parse_args()
    if (args.output / 'manifest.json').exists():
        raise SystemExit('Manifest already exists; do not rescan a frozen campaign.')
    resolver = Resolver(args.raw_root)
    resolver.recover_archives()
    resolver.load_supplements()
    rows = []
    benches = set(args.benches.split(',')) if args.benches else set()
    seen = set()
    for i, desc in enumerate(descriptors(resolver, args.start, args.end)):
        if benches and desc['benchmark'] not in benches:
            continue
        if args.one_per_benchmark and desc['benchmark'] in seen:
            continue
        row = stage(desc, resolver, args.raw_root, args.gold_skip_ratio)
        rows.append(row)
        if row['inventory_status'] == 'queued':
            seen.add(desc['benchmark'])
        if i % 100 == 0:
            print('SCAN', i, dict(Counter(r['inventory_status'] for r in rows)), flush=True)
    manifest = dict(schema_version='c2hls.historical-cosim-backfill.v2',
                    created_at=datetime.now(timezone.utc).isoformat(),
                    date_window=dict(start=args.start, end_exclusive=args.end, basis='campaign_date'),
                    subset=dict(benchmarks=sorted(benches), one_per_benchmark=args.one_per_benchmark),
                    raw_root=str(args.raw_root), toolchain=TOOLCHAIN,
                    gold_skip_ratio=args.gold_skip_ratio, rows=rows,
                    code_hashes=frozen_evaluator_hashes())
    atomic(args.output / 'manifest.json', manifest)
    print('INVENTORY', len(rows), dict(Counter(r['inventory_status'] for r in rows)), flush=True)


if __name__ == '__main__':
    main()
