"""Scientific accounting checks for the read-only reporting pipeline."""
import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('backfill_report', Path(__file__).parents[1] / 'scripts/report_cosim_backfill.py')
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


def evidence():
    row = {'validation_id': 'v', 'kernel_sha256': 'k', 'historical_golden_output_sha256': 'g'}
    m = {'validation_id': 'v', 'kernel_sha256': 'k',
         'cosim': {'status': 'pass', 'passed': True, 'success': True, 'ran': True,
                   'kernel_runtime_cycles': 100, 'target_code_sha256': 'k',
                   'golden_output_sha256': 'g',
                   'correctness': {'passed': True, 'rtl_postcheck_dump_count': 1, 'checks': [{'passed': True}]}},
         'csim': {'passed': True, 'success': True, 'golden_output_sha256': 'g', 'correctness': {'passed': True}},
         'golden': {'status': 'passed', 'output_sha256': 'g'}}
    return row, m


def test_requires_measured_rtl_and_independent_output_checks():
    row, m = evidence()
    assert report.classify(row, m)['rtl_cycles'] == 100
    m['cosim']['correctness']['rtl_postcheck_dump_count'] = 0
    assert report.classify(row, m)['status'] == 'evidence_error'
    assert report.classify(row, m)['rtl_cycles'] is None


@pytest.mark.parametrize('cycles', [None, 0, -1, True, float('nan'), float('inf')])
def test_invalid_cycles_never_become_latency_bars(cycles):
    row, m = evidence()
    m['cosim']['kernel_runtime_cycles'] = cycles
    assert report.classify(row, m)['status'] == 'evidence_error'
    assert report.classify(row, m)['rtl_cycles'] is None


@pytest.mark.parametrize('status', ['fail', 'timeout'])
def test_nonpass_never_plots_even_with_stale_cycles(status):
    row, m = evidence()
    m['cosim']['status'] = status
    m['cosim']['passed'] = False
    assert report.classify(row, m)['status'] == status
    assert report.classify(row, m)['rtl_cycles'] is None


def test_kernel_and_golden_identity_required():
    row, m = evidence()
    m['cosim']['target_code_sha256'] = 'different'
    assert report.classify(row, m)['status'] == 'evidence_error'
    row, m = evidence()
    m['golden']['output_sha256'] = 'different'
    assert report.classify(row, m)['status'] == 'evidence_error'


def test_stale_running_is_interrupted_and_missing_terminal_is_error():
    row, _ = evidence()
    assert report.classify(row, None, {'status': 'running'}, live=False)['status'] == 'interrupted'
    assert report.classify(row, None, {'status': 'running'}, live=True)['status'] == 'running'
    assert report.classify(row, None, {'status': 'pass'})['status'] == 'evidence_error'


def test_shared_validation_is_not_counted_twice():
    row = {'validation_id': 'v', 'benchmark': 'b', 'kernel_sha256': 'k', 'input_hashes': {'tb': 'h'}}
    assert len(report.group_rows([row, dict(row)])) == 1
    other = dict(row, kernel_sha256='other')
    with pytest.raises(ValueError, match='Conflicting'):
        report.group_rows([row, other])


def test_recovery_preserves_denominator_and_terminal_originals():
    old = {'validation_id': 'a', 'recorded_status': None, 'status': 'interrupted',
           'kernel_sha256': 'k', 'input_hashes': {'tb': 'h'}}
    new = dict(old, validation_id='b', prior_replay_id='a', status='pass', recorded_status='pass')
    assert report.merge_recovery([old], [new]) == [new]
    with pytest.raises(ValueError, match='terminal'):
        report.merge_recovery([dict(old, recorded_status='timeout')], [new])
    with pytest.raises(ValueError, match='Duplicate'):
        report.merge_recovery([old], [new, new])


def pair():
    base = {'benchmark': 'b', 'strategy': 'flash', 'input_hashes': {'tb': 'h'},
            'toolchain': {'vitis': '2023.2'}, 'repeat': 1, 'status': 'pass'}
    return [dict(base, arm='review_off', validation_id='off', rtl_cycles=100),
            dict(base, arm='review_on', validation_id='on', rtl_cycles=25)]


def test_paired_rtl_ratio_and_missing_pairs():
    rows = pair()
    assert report.pair_reviewers(rows)[0]['speedup'] == 4
    rows[1]['status'] = 'timeout'
    rows[1]['rtl_cycles'] = None
    p = report.pair_reviewers(rows)[0]
    assert p['speedup'] is None and p['on_status'] == 'timeout'


@pytest.mark.parametrize('key,value', [('input_hashes', {'tb': 'different'}), ('toolchain', {'vitis': 'other'}), ('repeat', 2)])
def test_pairs_must_match_inputs_toolchain_and_repeat(key, value):
    rows = pair()
    rows[1][key] = value
    assert report.pair_reviewers(rows)[0]['speedup'] is None


def test_infrastructure_error_does_not_become_pass():
    row, m = evidence()
    assert report.classify(row, m, {'status': 'execution_error'})['rtl_cycles'] is None


def test_snapshot_caches_exact_input_and_preserves_original(tmp_path):
    src = tmp_path / 'original.json'
    src.write_text('{"n":1}')
    out = tmp_path / 'report'
    out.mkdir()
    snap = report.Snapshot(out)
    assert snap.read(src) == {'n': 1}
    src.write_text('{"n":2}')
    assert snap.read(src) == {'n': 1}
    assert src.read_text() == '{"n":2}'
    preserved = out / snap.sources[str(src.resolve())]['snapshot']
    assert preserved.read_text() == '{"n":1}'


def test_measure_rejects_actual_toolchain_drift(tmp_path):
    row, m = evidence()
    row.update(benchmark='b', input_hashes={'tb': 'h'})
    m['toolchain'] = {'vitis_version': 'different'}
    raw = tmp_path / 'raw'
    (raw / 'measurements').mkdir(parents=True)
    (raw / 'measurements/v.json').write_text(json.dumps(m))
    output = tmp_path / 'output'
    output.mkdir()
    result = report.measure(report.Snapshot(output), 'test', [row], raw,
                            toolchain={'vitis_version': '2023.2'})[0]
    assert result['status'] == 'evidence_error'
    assert result['rtl_cycles'] is None
