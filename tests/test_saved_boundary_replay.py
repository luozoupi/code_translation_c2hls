"""Integrity and controller-boundary tests; no compiler, model or Vitis work."""
import copy
import json
from pathlib import Path

import pytest

from scripts import build_saved_replay as builder
from scripts import compare_v6_replay as replay


ROOT = Path(__file__).resolve().parents[1]
SELECTION_FIXTURE = ROOT / 'tests/fixtures/v6_replay'
TOOLCHAIN = {'vitis_version': '2023.2', 'part': 'xcu280-fsvh2892-2L-e',
             'clock_ns': 3.33, 'flow_target': 'vitis'}


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


@pytest.fixture
def archived_selection(tmp_path):
    benchmark = tmp_path / 'benchmark'
    benchmark.mkdir()
    source = 'void workload(int *a) { a[0] += 1; }'
    (benchmark / 'plain.cpp').write_text(source)
    (benchmark / 'kernel.h').write_text('void workload(int *a);')
    (benchmark / 'testbench.cpp').write_text('int main() { return 0; }')
    save(benchmark / 'metadata.json', {'plain_c_file': 'plain.cpp', 'header_file': 'kernel.h',
                                      'testbench_file': 'testbench.cpp'})
    kernel = tmp_path / 'kernel.cpp'
    kernel.write_text(source)
    result = tmp_path / 'result.json'
    save(result, {'synth_report': {'latency_cycles': 100}})
    history = tmp_path / 'history.json'
    save(history, {'messages': [{'role': 'assistant', 'content': '```cpp\n' + source + '\n```'}]})
    registry = {'sha256': 'a' * 64, 'testbench_sha256': builder.sha(benchmark / 'testbench.cpp')}
    row = {'record_id': 'model/kernel/setup', 'validation_id': 'validation1',
           'model': 'model', 'benchmark': 'kernel', 'source_setup': 'setup',
           'source_result': str(result), 'source_result_sha256': builder.sha(result),
           'history_json': str(history), 'history_sha256': builder.sha(history),
           'staged_kernel': str(kernel), 'kernel_sha256': builder.sha(kernel),
           'staged_benchmark': str(benchmark),
           'input_hashes': {p.name: builder.sha(p) for p in benchmark.iterdir()},
           'golden_registry': registry, 'current_preflight_native_golden_output_sha256': 'b' * 64,
           'paper_slots': ['model/kernel/selected']}
    measurement = {'kernel_sha256': row['kernel_sha256'], 'validation_id': row['validation_id'],
                   'row_id': row['record_id'], 'toolchain': TOOLCHAIN,
                   'golden': {'output_sha256': 'b' * 64, 'shape_registry': registry},
                   'csim': {'passed': False, 'success': False, 'correctness': {'passed': False}},
                   'cosim': {'ran': False}}
    measurement_path = tmp_path / 'raw/measurements/validation1.json'
    save(measurement_path, measurement)
    manifest = tmp_path / 'manifest.json'
    save(manifest, {'rows': [row], 'raw_root': str(tmp_path / 'raw'), 'toolchain': TOOLCHAIN})
    return manifest, measurement_path


def test_generator_retains_exact_saved_records_and_authenticated_selection_probe(archived_selection, tmp_path):
    manifest, measurement_path = archived_selection
    output = tmp_path / 'fixture'
    coverage = builder.build(manifest, output, SELECTION_FIXTURE)
    fixture = replay.load_fixture(output)
    assert coverage['registered_designs'] == coverage['replayed_designs'] == coverage['source_report_designs'] == 1
    assert coverage['exclusions'] == []
    item = fixture['cases'][0]
    assert item['independent_measurement'] == json.loads(measurement_path.read_text())
    assert item['report'] == {'latency_cycles': 100}
    assert item['provenance']['independent_measurement_sha256'] == builder.sha(measurement_path)
    assert item['provenance']['independent_measurement_content_sha256'] == replay.digest(item['independent_measurement'])
    assert len(fixture['selection_candidates']) == 7


def test_unmatched_reply_is_explicitly_excluded_from_agent_replay_only(archived_selection, tmp_path):
    manifest, _ = archived_selection
    data = json.loads(manifest.read_text())
    row = data['rows'][0]
    history = Path(row['history_json'])
    save(history, {'messages': [{'role': 'assistant', 'content': 'No saved selected source here.'}]})
    row['history_sha256'] = builder.sha(history)
    save(manifest, data)
    coverage = builder.build(manifest, tmp_path / 'fixture', SELECTION_FIXTURE)
    fixture = replay.load_fixture(tmp_path / 'fixture')
    assert coverage['replayed_designs'] == 0
    assert coverage['source_report_designs'] == 1
    assert fixture['cases'] == [] and fixture['source_cases'][0]['reply'] is None
    assert len(coverage['exclusions']) == 1
    assert 'No saved assistant response' in coverage['exclusions'][0]['reason']


def test_multistep_final_report_is_preserved_without_replacing_missing_data(archived_selection, tmp_path):
    manifest, _ = archived_selection
    data = json.loads(manifest.read_text())
    row = data['rows'][0]
    result = Path(row['source_result'])
    save(result, {'final_report': {'latency_cycles': 321, 'lut': 42}})
    row['source_result_sha256'] = builder.sha(result)
    save(manifest, data)
    coverage = builder.build(manifest, tmp_path / 'fixture', SELECTION_FIXTURE)
    fixture = replay.load_fixture(tmp_path / 'fixture')
    assert coverage['replayed_designs'] == coverage['source_report_designs'] == 1
    assert fixture['cases'][0]['report'] == {'latency_cycles': 321, 'lut': 42}


@pytest.mark.parametrize('field,value', [
    ('row_id', 'another/selection'), ('validation_id', 'another-validation'),
    ('kernel_sha256', '0' * 64), ('toolchain', {**TOOLCHAIN, 'clock_ns': 4.0}),
])
def test_wrong_independent_measurement_association_is_rejected(archived_selection, tmp_path, field, value):
    manifest, measurement_path = archived_selection
    measurement = json.loads(measurement_path.read_text())
    measurement[field] = value
    save(measurement_path, measurement)
    with pytest.raises(ValueError, match='association differs|toolchain differs'):
        builder.build(manifest, tmp_path / 'fixture', SELECTION_FIXTURE)


@pytest.mark.parametrize('change', ['harness', 'golden', 'tool_source', 'tool_golden'])
def test_wrong_harness_or_tool_evidence_is_rejected(archived_selection, tmp_path, change):
    manifest, measurement_path = archived_selection
    measurement = json.loads(measurement_path.read_text())
    if change == 'harness':
        measurement['golden']['shape_registry']['testbench_sha256'] = '0' * 64
    elif change == 'golden':
        measurement['golden']['output_sha256'] = '0' * 64
    elif change == 'tool_source':
        measurement['cosim']['target_code_sha256'] = '0' * 64
    else:
        measurement['csim']['golden_output_sha256'] = '0' * 64
    save(measurement_path, measurement)
    with pytest.raises(ValueError, match='association differs'):
        builder.build(manifest, tmp_path / 'fixture', SELECTION_FIXTURE)


def test_changed_inputs_and_duplicate_identities_cannot_inflate_coverage(archived_selection, tmp_path):
    manifest, _ = archived_selection
    data = json.loads(manifest.read_text())
    data['rows'].append(copy.deepcopy(data['rows'][0]))
    save(manifest, data)
    with pytest.raises(ValueError, match='Duplicate registered replay record'):
        builder.build(manifest, tmp_path / 'duplicate', SELECTION_FIXTURE)
    data['rows'].pop()
    save(manifest, data)
    Path(data['rows'][0]['staged_benchmark'], 'plain.cpp').write_text('changed source')
    with pytest.raises(ValueError, match='Archived identity changed'):
        builder.build(manifest, tmp_path / 'changed', SELECTION_FIXTURE)


def test_embedded_measurement_digest_is_checked_even_if_snapshot_checksum_is_updated(archived_selection, tmp_path):
    manifest, _ = archived_selection
    output = tmp_path / 'fixture'
    builder.build(manifest, output, SELECTION_FIXTURE)
    snapshot = json.loads((output / 'snapshot.json').read_text())
    snapshot['source_cases'][0]['independent_measurement']['csim']['passed'] = True
    save(output / 'snapshot.json', snapshot)
    receipt = json.loads((output / 'manifest.json').read_text())
    receipt['files']['snapshot.json'] = builder.sha(output / 'snapshot.json')
    save(output / 'manifest.json', receipt)
    with pytest.raises(ValueError, match='Archived independent measurement changed'):
        replay.load_fixture(output)


def test_saved_source_replay_keeps_real_csim_failure_as_synthesis_and_rtl_gate(monkeypatch):
    import c2hls as runtime
    item = copy.deepcopy(replay.load_fixture(SELECTION_FIXTURE)['cases'][1])
    assert item['independent_measurement']['csim']['passed'] is False
    def blocked(*args, **kwargs):
        raise AssertionError('Replay must not execute real tools')
    monkeypatch.setattr(replay.subprocess, 'Popen', blocked)
    result = replay.source_report_replay(runtime, item)
    assert [call['tool'] for call in result['tool_calls']] == ['csim']
    assert all(call['source_sha256'] == replay.digest(item['kernel']) for call in result['tool_calls'])
    assert result['evaluation']['synth']['skipped']
    assert result['evaluation']['cosim'] is None
    assert not result['feasibility']['feasible']
    assert result['csim_status'] == 'failed'
