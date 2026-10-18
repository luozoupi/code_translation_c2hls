import json
from pathlib import Path

import pytest

from scripts.scan_historical_cosim import (
    Resolver, TOOLCHAIN, digest, eligible_date, measured_cosim, passed,
    selected_source, stage,
)
from scripts.run_historical_cosim import compare_cosim_dumps, row_measurement, schema_record, verify


@pytest.fixture
def case(tmp_path, monkeypatch):
    bench = tmp_path / 'bench'
    bench.mkdir()
    meta = dict(benchmark='hlsfactory_atax', source_repo='HLSFactory',
                supports_cosim=True, header_file='kernel.h', testbench_file='testbench.cpp')
    for name, text in {'metadata.json': json.dumps(meta), 'plain.cpp': 'void kernel() {}',
                       'kernel.h': 'void kernel();', 'testbench.cpp': 'int main() { return 0; }'}.items():
        (bench / name).write_text(text)
    registry = tmp_path / 'registry.json'
    registry.write_text(json.dumps(dict(schema_version='c2hls.hlsfactory-output-shapes.v1',
        benchmarks={'hlsfactory_atax': dict(testbench_sha256=digest((bench / 'testbench.cpp').read_bytes()),
                                          outputs={'y': dict(shape=[1], logical_shape=[1], kind='float')})})))
    monkeypatch.setenv('C2HLS_HLSFACTORY_SHAPE_REGISTRY', str(registry))
    result = tmp_path / 'result.json'
    data = dict(hls_code='void kernel() {}', final_report=dict(latency_cycles=100),
                run=TOOLCHAIN, success=True, csim=dict(passed=True),
                ground_truth_report=dict(latency_cycles=20))
    result.write_text(json.dumps(data))
    desc = dict(source_summary=str(tmp_path / 'July.summary.json'), source_row=0,
                source_result=str(result), benchmark='hlsfactory_atax', benchmark_dir=str(bench),
                campaign='20260701', model='test-model', skill_mode='none',
                strategy='flash', kind='agentic', historical_best=None)
    return desc, data, Resolver(tmp_path / 'raw'), tmp_path / 'raw'


def test_july_exclusive_date_bounds():
    assert eligible_date('run_20260701', '20260701', '20260801')
    assert eligible_date('run_20260731', '20260701', '20260801')
    assert not eligible_date('run_20260630', '20260701', '20260801')
    assert not eligible_date('run_20260801', '20260701', '20260801')


def test_stage_exact_code_and_frozen_inputs(case):
    desc, data, resolver, raw = case
    row = stage(desc, resolver, raw, 10)
    assert row['inventory_status'] == 'queued'
    assert row['csynth_cycles'] == 100
    assert row['kernel_sha256'] == digest(data['hls_code'].encode())
    verify(row)
    Path(row['staged_benchmark'], 'testbench.cpp').write_text('changed')
    with pytest.raises(ValueError, match='benchmark input changed'):
        verify(row)


def test_dedup_requires_identical_harness(case):
    desc, data, resolver, raw = case
    a = stage(desc, resolver, raw, 10)
    b = stage({**desc, 'source_row': 1}, resolver, raw, 10)
    assert a['record_id'] != b['record_id']
    assert a['validation_id'] == b['validation_id']
    Path(desc['benchmark_dir'], 'testbench.cpp').write_text('different inputs')
    c = stage(desc, resolver, raw, 10)
    assert c['inventory_status'] == 'blocked'
    assert 'does not match audited shape contract' in c['reason']


def test_predicted_skip_does_not_fake_timeout_or_cycles(case):
    desc, _, resolver, raw = case
    row = stage(desc, resolver, raw, 2)
    assert row['inventory_status'] == 'predicted_longer_than_gold'
    measure = row_measurement(row, {})
    assert measure['status'] == 'not_run'
    assert measure['kernel_runtime_cycles'] is None
    assert schema_record(row, measure) is None


def test_source_hash_mismatch_rejected(case):
    desc, data, resolver, raw = case
    data['selected_code_sha256'] = 'wrong'
    with pytest.raises(ValueError, match='hash_mismatch'):
        selected_source(data, resolver)


def test_report_source_mismatch_rejected(case, tmp_path):
    desc, data, resolver, raw = case
    (tmp_path / 'kernel.cpp').write_text('different design')
    data['final_report']['work_dir'] = str(tmp_path)
    with pytest.raises(ValueError, match='report_source_mismatch'):
        selected_source(data, resolver)


def test_no_baseline_substitution_when_code_missing(case):
    desc, data, resolver, raw = case
    data.pop('hls_code')
    with pytest.raises(ValueError, match='saved_selected_code_missing'):
        selected_source(data, resolver)


def test_recover_exact_final_source(case):
    desc, data, resolver, raw = case
    code = data.pop('hls_code')
    data['benchmark'] = desc['benchmark']
    path = Path(desc['source_result'])
    path.with_name(desc['benchmark'] + '_final.cpp').write_text(code)
    recovered, sha, evidence = selected_source(data, resolver, path)
    assert recovered == code
    assert evidence == 'saved_final_cpp'


def test_cosim_must_match_selected_kernel(case):
    desc, data, resolver, raw = case
    sha = digest(data['hls_code'].encode())
    data['cosim'] = dict(passed=True, kernel_runtime_cycles=20, target_code_sha256='different')
    assert measured_cosim(data, sha, resolver) is None
    data['cosim']['target_code_sha256'] = sha
    assert measured_cosim(data, sha, resolver)['kernel_runtime_cycles'] == 20


def test_failed_status_overrides_pass_marker():
    assert not passed(dict(passed=True, success=False))


def test_schema_for_executed_pass(case):
    desc, data, resolver, raw = case
    row = stage(desc, resolver, raw, 10)
    measure = dict(status='pass', ran=True, passed=True, kernel_runtime_cycles=123,
                   kernel_runtime_us=0.40959, kernel_clock_freq_mhz=1000/3.33,
                   target_code_sha256=row['kernel_sha256'])
    record = schema_record(row, measure)
    assert record['rtl_sim']['kernel_runtime_cycles'] == 123
    assert record['run']['vitis_version'] == '2023.2'


def test_existing_results_resume_without_new_cycles(case):
    desc, data, resolver, raw = case
    row = stage(desc, resolver, raw, 10)
    result = dict(cosim=dict(status='timeout', ran=True, passed=False, kernel_runtime_cycles=None))
    measure = row_measurement(row, {row['validation_id']: result})
    assert measure['status'] == 'timeout'
    assert measure['kernel_runtime_cycles'] is None


def test_toolchain_mismatch_blocks(case):
    desc, data, resolver, raw = case
    data['run'] = {**TOOLCHAIN, 'clock_ns': 5.0}
    Path(desc['source_result']).write_text(json.dumps(data))
    assert stage(desc, resolver, raw, 10)['reason'] == 'historical_toolchain_mismatch'


DUMP = '==BEGIN DUMP_ARRAYS==\nbegin dump: y\n1.000000 2.000000\nend   dump: y\n==END   DUMP_ARRAYS==\n'
SPECS = {'y': dict(shape=[2], kind='float', atol=1e-6, rtol=1e-5)}


def test_vitis_c_test_and_rtl_dumps_are_both_checked():
    log = 'Starting C TB testing\n' + DUMP + 'Starting C post checking\n' + DUMP
    checked = compare_cosim_dumps(log, DUMP, SPECS)
    assert checked['passed']
    assert checked['dump_count'] == 2
    assert checked['rtl_postcheck_dump_count'] == 1


@pytest.mark.parametrize('which', ['initial', 'post'])
def test_bad_dump_cannot_be_hidden_by_good_dump(which):
    bad = DUMP.replace('2.000000', '3.000000')
    first, second = (bad, DUMP) if which == 'initial' else (DUMP, bad)
    assert not compare_cosim_dumps(first + 'Starting C post checking\n' + second, DUMP, SPECS)['passed']


def test_missing_rtl_postcheck_is_not_a_pass():
    assert not compare_cosim_dumps(DUMP, DUMP, SPECS)['passed']


def test_incomplete_dump_is_not_a_pass():
    log = DUMP + 'Starting C post checking\n' + DUMP.split('==END')[0]
    assert not compare_cosim_dumps(log, DUMP, SPECS)['passed']


def test_changed_historical_summary_blocks(case):
    desc, _, resolver, raw = case
    desc['historical_best'] = dict(cycles=42)
    assert stage(desc, resolver, raw, 10)['reason'] == 'summary_selected_cycles_mismatch'


def test_registry_change_invalidates_frozen_row(case):
    desc, _, resolver, raw = case
    row = stage(desc, resolver, raw, 10)
    registry = Path(row['golden_registry']['path'])
    registry.write_text(registry.read_text() + "\n")
    with pytest.raises(ValueError, match='changed frozen golden registry'):
        verify(row)
    replacement = stage(desc, resolver, raw, 10)
    assert replacement['validation_id'] != row['validation_id']
