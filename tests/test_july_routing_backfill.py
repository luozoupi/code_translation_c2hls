import copy
import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/prepare_july_routing_backfill.py'
spec = importlib.util.spec_from_file_location('july_queue', SCRIPT)
queue = importlib.util.module_from_spec(spec)
spec.loader.exec_module(queue)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def case(tmp_path):
    source = tmp_path / 'source'
    bench = tmp_path / 'bench'
    bench.mkdir()
    for name, content in {'metadata.json': json.dumps({'benchmark': 'hlsfactory_test',
            'header_file': 'kernel.h', 'testbench_file': 'testbench.cpp'}),
            'plain.cpp': 'int f(){return 1;}', 'kernel.h': 'int f();',
            'testbench.cpp': 'int main(){return f();}'}.items():
        (bench / name).write_text(content)
    kernel = tmp_path / 'kernel.cpp'
    kernel.write_text('int f(){return 1;}')
    result = tmp_path / 'result.json'
    save(result, {'selected_code_sha256': queue.file_sha(kernel)})
    save(source / queue.REGISTRY_FILES[0], {'benchmarks': {'hlsfactory_test': {
        'testbench_sha256': queue.file_sha(bench / 'testbench.cpp')}}})
    save(source / queue.REGISTRY_FILES[1], {'benchmarks': {}})
    worker = source / 'scripts/run_historical_cosim.py'
    worker.parent.mkdir(parents=True)
    worker.write_text('# frozen worker\n')
    row = dict(record_id='r1', kind='agentic', campaign='routing_20260701',
               benchmark='hlsfactory_test', strategy='flash', skill_mode='matched',
               model='model', historical_csim_passed=True, inventory_status='queued',
               source_result=str(result), resolved_result=str(result),
               source_result_sha256=queue.file_sha(result), kernel_sha256=queue.file_sha(kernel),
               staged_kernel=str(kernel), staged_benchmark=str(bench), validation_id='old-id',
               input_hashes={p.name: queue.file_sha(p) for p in bench.iterdir()})
    tc = dict(vitis_version='2023.2', clock_ns=3.33, part='part', flow_target='vitis')
    return tmp_path, source, row, tc


def test_identity_uses_all_inputs_toolchain_and_registry(case):
    _, source, row, tc = case
    contract = queue.registry_contract(row, source)
    identity = queue.semantic_identity(row, tc, contract)
    other = copy.deepcopy(row)
    other.update(model='different', skill_mode='skillless', source_result='/other')
    assert queue.digest(identity) == queue.digest(queue.semantic_identity(other, tc, contract))
    other['input_hashes']['kernel.h'] = 'changed'
    assert queue.digest(identity) != queue.digest(queue.semantic_identity(other, tc, contract))
    assert queue.digest(identity) != queue.digest(queue.semantic_identity(row, {**tc, 'clock_ns': 5}, contract))
    assert queue.digest(identity) != queue.digest(queue.semantic_identity(row, tc, {**contract, 'sha256': 'changed'}))


@pytest.mark.parametrize('status', ['pass', 'fail', 'timeout', 'not_run'])
def test_every_previous_outcome_excludes_rerun(case, status):
    root, source, row, tc = case
    path = root / 'july_cosim_backfill_20260928/manifest.json'
    raw = root / 'old_raw'
    save(path, dict(rows=[row], raw_root=str(raw), toolchain=tc))
    measure = dict(validation_id='old-id', kernel_sha256=row['kernel_sha256'],
                   cosim=dict(status=status))
    save(raw / 'measurements/old-id.json', measure)
    covered, _, issues = queue.coverage_index([path], source)
    assert not issues
    got = queue.classify(row, covered, tc, source)
    assert got['queue_status'] == 'existing_evidence'
    assert got['existing_measurements'][0]['status'] == status


def test_registered_pending_other_queue_is_reserved(case):
    root, source, row, tc = case
    path = root / 'replay/manifest.json'
    save(path, dict(rows=[row], raw_root=str(root / 'raw'), toolchain=tc))
    covered, _, issues = queue.coverage_index([path], source)
    assert not issues
    assert queue.classify(row, covered, tc, source)['queue_status'] == 'existing_evidence'


def test_misattributed_prior_measurement_fails_coverage_audit(case):
    root, source, row, tc = case
    path = root / 'july_cosim_backfill_20260928/manifest.json'
    raw = root / 'old_raw'
    save(path, dict(rows=[row], raw_root=str(raw), toolchain=tc))
    save(raw / 'measurements/old-id.json', dict(validation_id='old-id',
         kernel_sha256='different-kernel', cosim=dict(status='pass')))
    covered, _, issues = queue.coverage_index([path], source)
    assert not covered
    assert len(issues) == 1
    assert issues[0]['error'] == 'prior measurement identity mismatch'


def test_changed_source_result_blocks(case):
    _, source, row, tc = case
    Path(row['source_result']).write_text('{}')
    got = queue.classify(row, {}, tc, source)
    assert got['queue_status'] == 'blocked'
    assert 'source result missing or changed' in got['queue_reason']


def test_changed_harness_blocks_even_with_same_benchmark_name(case):
    _, source, row, tc = case
    p = Path(row['staged_benchmark']) / 'testbench.cpp'
    p.write_text('a different public harness')
    row['input_hashes']['testbench.cpp'] = queue.file_sha(p)
    got = queue.classify(row, {}, tc, source)
    assert got['queue_status'] == 'blocked'
    assert 'audited registry' in got['queue_reason']


def test_blocked_old_rows_are_not_repaired_or_reselected(case):
    _, source, row, tc = case
    row.update(inventory_status='blocked', reason='selected_code_report_source_mismatch')
    got = queue.classify(row, {}, tc, source)
    assert got['queue_status'] == 'blocked'
    assert got['queue_reason'] == row['reason']


def test_only_historical_csim_pass_is_eligible(case):
    _, source, row, tc = case
    row['historical_csim_passed'] = False
    assert queue.classify(row, {}, tc, source)['queue_status'] == 'excluded'


def test_staging_deduplicates_but_keeps_policy_rows_and_new_ids(case):
    root, source, row, tc = case
    first = queue.classify(row, {}, tc, source)
    second = copy.deepcopy(first)
    second.update(record_id='other-policy', skill_mode='skillless')
    raw = root / 'new_raw'
    rows = queue.stage_rows([first, second], raw, tc, {'worker': 'fingerprint'}, 'new_campaign')
    assert len(rows) == 2
    assert rows[0]['validation_id'] == rows[1]['validation_id']
    assert rows[0]['validation_id'] != row['validation_id']
    assert len(list((raw / 'inputs').iterdir())) == 1
    assert queue.file_sha(rows[0]['staged_kernel']) == row['kernel_sha256']
    assert Path(row['staged_kernel']).exists()


def test_prepare_verifies_frozen_source_and_preserves_originals(case):
    root, frozen, row, tc = case
    hashes = {str(p.relative_to(frozen)): queue.file_sha(p) for p in frozen.rglob('*') if p.is_file()}
    old = root / 'july_cosim_backfill_20260928/manifest.json'
    save(old, dict(rows=[row], toolchain=tc, raw_root=str(root / 'old_raw')))
    baseline = root / 'fingerprint.json'
    save(baseline, dict(code_hashes=hashes, toolchain=tc))
    before = queue.file_sha(old)
    summary = queue.prepare(old, root / 'new_output', root / 'new_raw', frozen, baseline, [old])
    assert summary['distinct_validations'] == 1
    assert summary['launches_performed'] == 0
    assert queue.file_sha(old) == before
    manifest = queue.read(root / 'new_output/manifest.json')
    assert manifest['code_hashes'] == hashes
    assert manifest['policy']['predicted_cycle_skips'] is False
    with pytest.raises(ValueError, match='never overwrite'):
        queue.prepare(old, root / 'new_output', root / 'new_raw', frozen, baseline, [old])
    (frozen / 'scripts/run_historical_cosim.py').write_text('# changed')
    with pytest.raises(ValueError, match='input hash mismatch'):
        queue.prepare(old, root / 'other_output', root / 'other_raw', frozen, baseline, [old])


def test_unsafe_input_path_rejected():
    for path in ('../kernel.cpp', '/tmp/elsewhere', ''):
        with pytest.raises(ValueError):
            queue.safe_relative(path)


def test_early_policy_and_dose_repeat_labels_are_explicit():
    x = queue.routing_labels(dict(skill_mode='skill_on', campaign='matrix_qwen_dynamic_selective_positive_20260708'))
    assert x['routing_policy'] == 'selective_positive'
    assert x['historical_sample_index'] is None
    x = queue.routing_labels(dict(skill_mode='dose_3_action_only_sample2', campaign='skill_dose_pilot_v1'))
    assert x['historical_sample_index'] == 2
