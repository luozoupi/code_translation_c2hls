#!/usr/bin/env python3
"""Build a captured-boundary replay fixture from fixed commercial selections.

This reads archived evidence only. A candidate is included only when its saved
assistant response extracts to the unchanged selected source. Missing records
remain explicit exclusions; no replacement response or tool result is invented.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.compare_v6_replay import load_fixture
from hlsmith.validation.source_contracts import extract_cpp_code


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def text_sha(value):
    return hashlib.sha256(value.encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def checked(path, expected):
    if not expected or sha(path) != expected:
        raise ValueError(f"Archived identity changed: {path}")
    return Path(path)


def validate_measurement(row, measurement, toolchain):
    """Bind saved tool records to the selected source, harness and target."""
    expected_toolchain = {'vitis_version': '2023.2', 'part': 'xcu280-fsvh2892-2L-e',
                          'clock_ns': 3.33, 'flow_target': 'vitis'}
    if toolchain != expected_toolchain or measurement.get('toolchain') != toolchain:
        raise ValueError('Measurement toolchain differs from the registered replay target')
    for key, expected in (('kernel_sha256', row['kernel_sha256']),
                          ('validation_id', row['validation_id']),
                          ('row_id', row['record_id'])):
        if measurement.get(key) != expected:
            raise ValueError('Measurement/source association differs: ' + key)
    golden = measurement.get('golden') or {}
    registry = golden.get('shape_registry') or {}
    expected_registry = row['golden_registry']
    if (registry.get('sha256') != expected_registry['sha256']
            or registry.get('testbench_sha256') != expected_registry['testbench_sha256']):
        raise ValueError('Measurement public harness/shape registry association differs')
    expected_output = row['current_preflight_native_golden_output_sha256']
    if not expected_output or golden.get('output_sha256') != expected_output:
        raise ValueError('Measurement independent golden-output association differs')
    for name in ('csim', 'cosim'):
        tool = measurement.get(name)
        if not isinstance(tool, dict):
            raise ValueError('Measurement lacks saved tool boundary: ' + name)
        if tool.get('target_code_sha256') not in (None, row['kernel_sha256']):
            raise ValueError('Saved tool source association differs: ' + name)
        if tool.get('golden_output_sha256') not in (None, expected_output):
            raise ValueError('Saved tool golden-output association differs: ' + name)


def build(manifest_path, output, selection_fixture):
    manifest_path = Path(manifest_path).resolve()
    manifest = read(manifest_path)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    cases, source_cases, exclusions = [], [], []
    if len({row['record_id'] for row in manifest['rows']}) != len(manifest['rows']):
        raise ValueError('Duplicate registered replay record identity')
    if len({row['validation_id'] for row in manifest['rows']}) != len(manifest['rows']):
        raise ValueError('Duplicate registered replay validation identity')
    for row in manifest['rows']:
        identity = row.get('record_id') or row.get('validation_id')
        try:
            result_path = checked(row['source_result'], row['source_result_sha256'])
            history_path = checked(row['history_json'], row['history_sha256'])
            kernel_path = checked(row['staged_kernel'], row['kernel_sha256'])
            benchmark = Path(row['staged_benchmark'])
            for name, expected in row['input_hashes'].items():
                checked(benchmark / name, expected)
            measurement_path = Path(manifest['raw_root']) / 'measurements' / (row['validation_id'] + '.json')
            measurement = read(measurement_path)
            validate_measurement(row, measurement, manifest['toolchain'])
            result = read(result_path)
            kernel = kernel_path.read_text()
            messages = read(history_path)['messages']
            matches = [(index, message['content']) for index, message in enumerate(messages)
                       if message.get('role') == 'assistant' and isinstance(message.get('content'), str)
                       and (extract_cpp_code(message['content']) or '').strip() == kernel.strip()]
            if not matches:
                exclusions.append({'id': identity, 'reason': 'No saved assistant response extracts to the fixed selected source',
                                   'kernel_sha256': row['kernel_sha256']})
            report = result.get('synth_report') or result.get('final_report')
            if not isinstance(report, dict):
                exclusions.append({'id': identity, 'reason': 'No archived selected synthesis report'})
                continue
            index, reply = matches[-1] if matches else (None, None)
            meta = read(benchmark / 'metadata.json')
            case = {'id': identity, 'model': row['model'], 'setup': row.get('source_setup'),
                          'benchmark': row['benchmark'], 'metadata': meta,
                          'plain': (benchmark / meta['plain_c_file']).read_text(),
                          'header': (benchmark / meta['header_file']).read_text(),
                          'header_name': meta['header_file'],
                          'testbench': (benchmark / meta['testbench_file']).read_text(),
                          'kernel': kernel, 'reply': reply, 'report': report,
                          'historical_csim': result.get('csim'), 'independent_measurement': measurement,
                          'provenance': {'source_result': str(result_path), 'source_result_sha256': sha(result_path),
                                         'history': str(history_path), 'history_sha256': sha(history_path),
                                         'reply_message_index': index, 'reply_sha256': text_sha(reply) if reply is not None else None,
                                         'kernel_sha256': text_sha(kernel),
                                         'report_sha256': text_sha(json.dumps(report, sort_keys=True, ensure_ascii=False, separators=(',', ':'))),
                                         'independent_measurement': str(measurement_path),
                                         'independent_measurement_sha256': sha(measurement_path),
                                         'independent_measurement_content_sha256': text_sha(json.dumps(measurement, sort_keys=True, ensure_ascii=False, separators=(',', ':'))),
                                         'public_input_hashes': row['input_hashes'], 'paper_slots': row['paper_slots']}}
            source_cases.append(case)
            if matches:
                cases.append(case)
        except FileNotFoundError as error:
            exclusions.append({'id': identity, 'reason': 'Archived dependency missing', 'path': str(error.filename)})
    # Preserve the independently captured seven-arm selection restoration probe.
    selected = load_fixture(Path(selection_fixture))['selection_candidates']
    snapshot = {'schema': 'hlsmith.saved-boundary-replay.v1', 'cases': cases,
                'source_cases': source_cases, 'selection_candidates': selected}
    (output / 'snapshot.json').write_text(json.dumps(snapshot, sort_keys=True, indent=2) + '\n')
    (output / 'manifest.json').write_text(json.dumps({'schema': 'hlsmith.replay-fixture-manifest.v1',
        'source_manifest': str(manifest_path), 'source_manifest_sha256': sha(manifest_path),
        'files': {'snapshot.json': sha(output / 'snapshot.json')}}, indent=2) + '\n')
    summary = {'registered_designs': len(manifest['rows']), 'replayed_designs': len(cases),
               'replayed_paper_roles': sum(len(case['provenance']['paper_slots']) for case in cases),
               'source_report_designs': len(source_cases),
               'source_report_paper_roles': sum(len(case['provenance']['paper_slots']) for case in source_cases),
               'exclusions': exclusions, 'fixture_sha256': sha(output / 'snapshot.json'),
               'scope': 'Archived-boundary controller replay; no full search, compilation, model or tool execution'}
    (output / 'coverage.json').write_text(json.dumps(summary, indent=2) + '\n')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--selection-fixture', type=Path,
                        default=Path(__file__).resolve().parents[1] / 'tests/fixtures/v6_replay')
    args = parser.parse_args()
    print(json.dumps(build(args.manifest, args.output, args.selection_fixture), indent=2))


if __name__ == '__main__':
    main()
