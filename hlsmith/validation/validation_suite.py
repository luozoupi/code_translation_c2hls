"""Opt-in, hash-bound multi-input validation with a prespecified stop rule."""
from __future__ import annotations

import contextvars
import hashlib
import inspect
import json
from contextlib import contextmanager
from functools import wraps
from pathlib import Path

from experiment_telemetry import tool_scope

_ACTIVE = contextvars.ContextVar('c2hls_validation_suite', default=None)


def digest(value):
    return hashlib.sha256(value).hexdigest()


def load_suite(manifest_path, benchmark):
    # Imported here to avoid the controller/evaluator import cycle.
    from c2hls import _authoritative_hlsfactory_output_specs
    path = Path(manifest_path).resolve()
    payload = path.read_bytes()
    manifest = json.loads(payload)
    if manifest.get('schema_version') != 'c2hls.sigmetrics-inputs.v1':
        raise ValueError('unknown validation input schema')
    native_path = path.parent / 'native_validation.json'
    native = json.loads(native_path.read_bytes())
    if native['manifest_sha256'] != digest(payload):
        raise ValueError('native validation does not match the input manifest')
    entries = [entry for entry in manifest['entries'] if entry['benchmark'] == benchmark]
    if len(entries) != 1:
        raise ValueError('benchmark must have exactly one input entry')
    entry = entries[0]
    directory = Path(entry['benchmark_dir'])
    metadata = json.loads((directory / 'metadata.json').read_text())
    if metadata.get('benchmark') != benchmark:
        raise ValueError('benchmark metadata identity mismatch')
    for filename, key in [('plain.cpp', 'input_c_sha256'), (metadata['header_file'], 'header_sha256')]:
        if digest((directory / filename).read_bytes()) != entry[key]:
            raise ValueError('public source/header hash mismatch')
    if [case['case_id'] for case in entry['cases']] != list(range(5)):
        raise ValueError('exactly the five prespecified input cases are required')
    cases = []
    for case in entry['cases']:
        records = [r for r in native['cases'] if r['benchmark'] == benchmark and r['case_id'] == case['case_id']]
        if len(records) != 1 or records[0]['success'] is not True:
            raise ValueError('missing successful native oracle')
        record = records[0]
        tb = Path(case['testbench_path']).read_bytes()
        registry = Path(case['shape_registry_path'])
        golden = Path(record['golden_output_path']).read_bytes()
        provenance = record['provenance']
        if digest(tb) != case['testbench_sha256'] or digest(tb) != record['testbench_sha256']:
            raise ValueError('testbench hash mismatch')
        if digest(registry.read_bytes()) != provenance['shape_registry']['sha256']:
            raise ValueError('shape registry hash mismatch')
        if digest(golden) != provenance['output_sha256']:
            raise ValueError('native golden hash mismatch')
        specs, _, _ = _authoritative_hlsfactory_output_specs(metadata, tb.decode(), registry_path=registry)
        cases.append({**case, 'testbench_code': tb.decode(), 'golden_output_text': golden.decode(),
                      'golden_output_specs': specs, 'golden_output_sha256': digest(golden)})
    return {'benchmark': benchmark, 'manifest_path': str(path), 'manifest_sha256': digest(payload),
            'native_validation_sha256': digest(native_path.read_bytes()),
            'header_sha256': entry['header_sha256'], 'cases': cases}


@contextmanager
def use_validation_suite(manifest_path, benchmark):
    suite = load_suite(manifest_path, benchmark)
    token = _ACTIVE.set(suite)
    try:
        yield suite
    finally:
        _ACTIVE.reset(token)


def run_validation_cases(stage):
    """Run in fixed order; fail at the first failed case, without reselection.

    Every COSIM invocation resynthesizes and is counted by the inner tool
    decorator. The reported latency is the maximum across the five RTL runs.
    """
    def decorate(function):
        signature = inspect.signature(function)

        @wraps(function)
        def wrapped(*args, **kwargs):
            suite = _ACTIVE.get()
            if suite is None:
                return function(*args, **kwargs)
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            if digest(bound.arguments['header_code'].encode()) != suite['header_sha256']:
                raise ValueError('candidate header differs from frozen validation suite')
            supplied_tb = digest(bound.arguments['testbench_code'].encode())
            if supplied_tb != suite['cases'][0]['testbench_sha256']:
                raise ValueError('candidate testbench differs from frozen case zero')
            results = []
            for case in suite['cases']:
                arguments = dict(bound.arguments)
                for key in ['testbench_code', 'golden_output_text', 'golden_output_specs']:
                    arguments[key] = case[key]
                if arguments.get('work_dir'):
                    arguments['work_dir'] = str(Path(arguments['work_dir']) / f"case{case['case_id']}")
                with tool_scope(f"{stage}:case{case['case_id']}"):
                    result = function(**arguments)
                if stage == 'cosim' and result.get('passed'):
                    cycles = result.get('kernel_runtime_cycles')
                    if not isinstance(cycles, int) or isinstance(cycles, bool) or cycles <= 0:
                        result = {**result, 'success': False, 'passed': False,
                                  'error': 'passing RTL case lacks measured positive cycles'}
                results.append((case, result))
                if not (result.get('success') is True and result.get('passed') is True):
                    break
            passed = len(results) == 5 and all(r.get('success') is True and r.get('passed') is True for _, r in results)
            selected = max((r for _, r in results), key=lambda r: r.get('kernel_runtime_cycles') or 0) if passed else results[-1][1]
            output = dict(selected)
            output.update(success=passed, passed=passed, validation_suite={
                'manifest_sha256': suite['manifest_sha256'], 'native_validation_sha256': suite['native_validation_sha256'],
                'benchmark': suite['benchmark'], 'planned_cases': 5, 'executed_cases': len(results),
                'all_cases_passed': passed, 'stop_rule': 'first_failed_case',
                'performance_aggregation': 'maximum_measured_cycles_across_all_five_cases' if stage == 'cosim' else None,
                'cases': [{**{key: value for key, value in result.items() if key != 'log'},
                           'case_id': case['case_id'], 'testbench_sha256': case['testbench_sha256'],
                           'golden_output_sha256': case['golden_output_sha256']} for case, result in results]})
            return output
        return wrapped
    return decorate
