"""Exercise the unchanged registered evaluator without invoking Vitis."""
import importlib.util
import json
from pathlib import Path
import sys
import types

import pytest

REPO = Path(__file__).resolve().parents[1]
FROZEN = REPO / 'artifacts/sigmetrics2027/registry_recovery_20260928/source/scripts/run_historical_cosim.py'


@pytest.mark.parametrize('csim_passed', [False, True])
def test_frozen_worker_stops_before_rtl_unless_typed_csim_passes(tmp_path, monkeypatch, csim_passed):
    spec = importlib.util.spec_from_file_location('commercial_frozen_worker_test', FROZEN)
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    raw, bench = tmp_path / 'raw', tmp_path / 'benchmark'
    bench.mkdir()
    for name, text in {'plain.cpp': 'PLAIN', 'kernel.h': 'HEADER', 'testbench.cpp': 'PUBLIC TESTBENCH'}.items():
        (bench / name).write_text(text)
    (bench / 'metadata.json').write_text(json.dumps({'benchmark': 'hlsfactory_test',
        'translated_hls_top': 'kernel_test', 'header_file': 'kernel.h', 'testbench_file': 'testbench.cpp'}))
    source = tmp_path / 'selected.cpp'
    source.write_text('EXACT SELECTED SOURCE')
    row = {'validation_id': 'case', 'inventory_status': 'queued',
        'staged_kernel': str(source), 'staged_benchmark': str(bench), 'kernel_sha256': 'kernel-hash',
        'record_id': 'record', 'input_hashes': dict.fromkeys(['plain.cpp', 'kernel.h', 'testbench.cpp', 'metadata.json'], 'hash'),
        'golden_registry': {'path': 'configs/hlsfactory_output_shapes_fc27133.json'}}
    manifest = {'rows': [row], 'raw_root': str(raw)}
    monkeypatch.setattr(worker, 'read', lambda path: manifest if Path(path).name == 'manifest.json'
                        else json.loads(Path(path).read_text()))
    monkeypatch.setattr(worker, 'verify_code', lambda manifest: None)
    monkeypatch.setattr(worker, 'verify', lambda row: None)
    monkeypatch.setattr(worker, 'schema_record', lambda row, record: None)
    monkeypatch.setattr(worker, 'archive_work', lambda work, raw, vid: 'MOCK ARCHIVE')
    # Keep every scratch directory under pytest's own output, not the real queue.
    original_path = worker.Path
    monkeypatch.setattr(worker, 'Path', lambda value: tmp_path if value == '/tmp' else original_path(value))
    monkeypatch.setattr(worker.os, 'environ', dict(worker.os.environ))
    golden = {'success': True, 'provenance': {'output_sha256': 'gold'},
              'output': 'GOLDEN OUTPUT', 'specs': {'A': {'shape': [1], 'kind': 'float', 'atol': 1e-6, 'rtol': 1e-5}}}
    calls = []
    def run_csim(code, tb, header, **kwargs):
        calls.append(('csim', code, tb, header, kwargs))
        return {'success': csim_passed, 'passed': csim_passed,
                'correctness': {'passed': csim_passed}, 'error': '' if csim_passed else 'Independent golden-output comparison failed'}
    def run_cosim(code, tb, header, **kwargs):
        calls.append(('cosim', code, tb, header, kwargs))
        return {'success': True, 'passed': True, 'correctness': {'passed': True}, 'kernel_runtime_cycles': 12}
    monkeypatch.setitem(sys.modules, 'c2hls', types.SimpleNamespace(_prepare_independent_golden=lambda inputs, **kwargs: golden))
    monkeypatch.setitem(sys.modules, 'hls_eval', types.SimpleNamespace(run_csim=run_csim, run_cosim=run_cosim))
    worker.worker(tmp_path / 'queue', 'case', 3600)
    record = json.loads((raw / 'measurements/case.json').read_text())
    assert [c[0] for c in calls] == (['csim', 'cosim'] if csim_passed else ['csim'])
    for _, code, tb, header, kwargs in calls:
        assert (code, tb, header) == ('EXACT SELECTED SOURCE', 'PUBLIC TESTBENCH', 'HEADER')
        assert kwargs['golden_output_text'] == 'GOLDEN OUTPUT'
        assert kwargs['golden_output_specs'] == golden['specs']
        assert kwargs['top_function'] == 'kernel_test'
    assert record['cosim']['status'] == ('pass' if csim_passed else 'not_run')
    if not csim_passed:
        assert record['cosim']['skip_reason'] == 'candidate_csim_failed'


def test_frozen_csim_tool_success_is_overridden_by_numeric_mismatch():
    evaluator_path = FROZEN.parents[1] / 'hls_eval.py'
    spec = importlib.util.spec_from_file_location('commercial_frozen_hls_eval_test', evaluator_path)
    evaluator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluator)
    dump = '==BEGIN DUMP_ARRAYS==\nbegin dump: A\n{}\nend   dump: A\n==END   DUMP_ARRAYS==\n'
    result = evaluator._apply_independent_golden(
        {'passed': True, 'success': True, 'log': dump.format('10000.0')},
        golden_output_text=dump.format('1.0'),
        golden_output_specs={'A': {'shape': [1], 'kind': 'float', 'atol': 1e-6, 'rtol': 1e-5}},
        stage='csim')
    assert result['passed'] is False
    assert result['success'] is False
    assert result['correctness']['reason'] == 'float_mismatch'


def test_rtl_evidence_requires_postcheck_dump():
    evaluator_path = FROZEN.parents[1] / 'hls_eval.py'
    spec = importlib.util.spec_from_file_location('commercial_frozen_hls_eval_test', evaluator_path)
    evaluator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluator)
    dump = '==BEGIN DUMP_ARRAYS==\nbegin dump: A\n1.0\nend   dump: A\n==END   DUMP_ARRAYS==\n'
    result = evaluator._apply_independent_golden(
        {'passed': True, 'success': True, 'log': dump}, golden_output_text=dump,
        golden_output_specs={'A': {'shape': [1], 'kind': 'float'}}, stage='cosim')
    assert result['passed'] is False
    assert result['correctness']['reason'] == 'missing_or_malformed_rtl_postcheck_dump'
