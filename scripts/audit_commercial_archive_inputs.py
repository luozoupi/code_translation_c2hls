#!/usr/bin/env python3
"""Audit archived prompt inputs and stage a separately identified public harness.

This is a native-only preflight. It never invokes Vitis or a model. The archive
does not contain its original testbench: this program must not manufacture an
exact historical-harness claim from matching public headers and plain source.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def recover_prompt_inputs(history):
    """Only the initial input prompt counts, never later candidate code blocks."""
    inputs = []
    for message in history.get('messages', []):
        text = message.get('content')
        if message.get('role') != 'user' or not isinstance(text, str):
            continue
        if 'Here is the plain C kernel:' not in text:
            continue
        parts = []
        for heading in ('Here is the header:', 'Here is the plain C kernel:'):
            match = re.search(re.escape(heading) + r'\s*```(?:cpp|c\+\+|c)?\n(.*?)\n```', text, re.S)
            if not match:
                raise ValueError('incomplete original input prompt')
            parts.append(match.group(1))
        inputs.append(tuple(parts))
    if not inputs or len(set(inputs)) != 1:
        raise ValueError('missing or conflicting original input prompts')
    return inputs[0]


def validate_prompt_matches(history, header, plain):
    recovered_header, recovered_plain = recover_prompt_inputs(history)
    if recovered_header.strip() != header.strip() or recovered_plain.strip() != plain.strip():
        raise ValueError('original prompt header/plain differs from public input')
    return {
        'header_matches_public_ignoring_outer_whitespace': True,
        'plain_matches_public_ignoring_outer_whitespace': True,
        'prompt_header_sha256': hashlib.sha256(recovered_header.encode()).hexdigest(),
        'prompt_plain_sha256': hashlib.sha256(recovered_plain.encode()).hexdigest(),
    }


def native_run(code_path, bench, output, timeout=180):
    """Keep native compile and run logs; no source editing or repair."""
    output.mkdir(parents=True, exist_ok=False)
    meta = json.loads((bench / 'metadata.json').read_text())
    compiler = shutil.which('g++')
    command = [compiler, '-std=c++17', '-O0', '-Wno-unknown-pragmas', '-I' + str(bench),
               str(code_path), str(bench / meta['testbench_file']), '-lm', '-o', str(output / 'native')]
    # Native candidate code may use the installed Vitis C++ simulation headers.
    vendor = Path('/mnt/data/luo00466/Xilinx/Vitis_HLS/2023.2/include')
    if vendor.is_dir():
        command.insert(4, '-I' + str(vendor))
    record = {'command': command, 'source_sha256': sha(code_path), 'timeout_seconds': timeout,
              'work_dir': str(output), 'vitis_invoked': False}
    try:
        proc = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
        (output / 'compile.log').write_text(proc.stdout + proc.stderr)
        record.update(compile_returncode=proc.returncode, compile_log=str(output / 'compile.log'))
        if proc.returncode:
            return {**record, 'status': 'compile_failed', 'success': False}
        proc = subprocess.run([str(output / 'native')], capture_output=True, text=True, timeout=timeout)
        text = proc.stdout + proc.stderr
        (output / 'output.txt').write_text(text)
        return {**record, 'status': 'executed' if proc.returncode == 0 else 'execution_failed',
                'success': proc.returncode == 0, 'returncode': proc.returncode,
                'output': str(output / 'output.txt'), 'output_sha256': sha(output / 'output.txt')}
    except subprocess.TimeoutExpired as exc:
        return {**record, 'status': 'timeout', 'success': False,
                'timeout_stage': 'compile' if exc.cmd == command else 'execute'}


def prepare(payload, public_root, output, workers):
    if output.exists():
        raise ValueError('refusing to overwrite an input audit')
    output.mkdir(parents=True)
    histories, issues, bench_histories = [], [], defaultdict(list)
    for path in sorted((payload / 'results').rglob('*_history.json')):
        benchmark = path.name.removesuffix('_history.json')
        local = public_root / benchmark
        try:
            meta = json.loads((local / 'metadata.json').read_text())
            check = validate_prompt_matches(json.loads(path.read_text()),
                        (local / meta['header_file']).read_text(), (local / 'plain.cpp').read_text())
            record = {'history': str(path), 'history_sha256': sha(path), 'benchmark': benchmark, **check}
            histories.append(record)
            bench_histories[benchmark].append(record)
        except (OSError, ValueError, KeyError) as exc:
            issues.append({'history': str(path), 'error': str(exc)})
    registry_source = REPO / 'configs/hlsfactory_output_shapes_fc27133.json'
    registry = json.loads(registry_source.read_text())
    shutil.copyfile(registry_source, output / 'hlsfactory_output_shapes_fc27133.json')
    benches = []
    for benchmark, records in sorted(bench_histories.items()):
        local, staged = public_root / benchmark, output / 'benchmarks' / benchmark
        original_meta = json.loads((local / 'metadata.json').read_text())
        meta = {key: original_meta[key] for key in ('benchmark', 'source_repo', 'header_file',
                'plain_c_file', 'testbench_file', 'kernel_top', 'hls_top', 'translated_hls_top') if key in original_meta}
        meta.update(supports_csim=True, supports_cosim=True, support_files=[], include_dirs=[],
                    cosim_depths={}, independent_golden_required=True)
        if original_meta.get('support_files'):
            raise ValueError('public support files require explicit review')
        staged.mkdir(parents=True)
        names = ['plain.cpp', meta['header_file'], meta['testbench_file']]
        for name in names:
            shutil.copyfile(local / name, staged / name)
        write(staged / 'metadata.json', meta)
        names += ['metadata.json']
        contract = registry['benchmarks'][benchmark]
        if sha(staged / meta['testbench_file']) != contract['testbench_sha256']:
            raise ValueError('public testbench hash mismatches shape registry')
        benches.append({'benchmark': benchmark, 'staged_benchmark': str(staged),
            'input_hashes': {name: sha(staged / name) for name in names},
            'public_root': str(local), 'public_metadata_sha256': sha(local / 'metadata.json'),
            'archive_prompt_history_count': len(records), 'prompt_inputs_match': True,
            'historical_testbench_identity': 'unavailable_in_archive',
            'validation_protocol': 'new_independent_full_size_public_testbench_validation',
            'size_overrides': {}, 'outputs': contract['outputs'],
            'golden_registry': {'path': 'configs/hlsfactory_output_shapes_fc27133.json',
                'sha256': sha(registry_source), 'testbench_sha256': contract['testbench_sha256']}})
    from golden_output import compare_hlsfactory_dumps
    def preflight(row):
        b = Path(row['staged_benchmark'])
        result = native_run(b / 'plain.cpp', b, output / 'native_golden' / row['benchmark'])
        if result['success']:
            text = Path(result['output']).read_text()
            specs = {key: {'shape': value['shape'], 'kind': value['kind']} for key, value in row['outputs'].items()}
            comparison = compare_hlsfactory_dumps(text, text, specs=specs).to_dict()
            result.update(comparison=comparison, success=comparison['passed'])
        row['native_golden'] = result
        return row
    with ThreadPoolExecutor(max_workers=workers) as pool:
        benches = list(pool.map(preflight, benches))
    report = {'schema_version': 'c2hls.commercial-input-audit.v1',
        'created_at': datetime.now(timezone.utc).isoformat(), 'archive_payload': str(payload),
        'history_count': len(histories), 'history_input_checks': histories, 'issues': issues,
        'benchmarks': benches, 'native_golden_passes': sum(r['native_golden']['success'] for r in benches),
        'registry_sha256': sha(registry_source), 'script_sha256': sha(__file__),
        'original_harness_not_archived': True, 'vitis_or_api_invoked': False}
    write(output / 'report.json', report)
    print(json.dumps({k: report[k] for k in ('history_count', 'issues', 'native_golden_passes')}))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--payload', type=Path, required=True)
    p.add_argument('--public-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--workers', type=int, default=4)
    args = p.parse_args()
    prepare(args.payload.resolve(), args.public_root.resolve(), args.output.resolve(), args.workers)


if __name__ == '__main__':
    main()
