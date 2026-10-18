import pytest
import json
import hashlib

from scripts.audit_commercial_archive_inputs import recover_prompt_inputs, validate_prompt_matches


def history(header='HEADER', plain='PLAIN'):
    return {'messages': [{'role': 'user', 'content':
        f'Here is the header:\n```cpp\n{header}\n```\nHere is the plain C kernel:\n```cpp\n{plain}\n```'}]}


def test_recovers_original_input_without_reading_assistant_candidate():
    data = history()
    data['messages'].append({'role': 'assistant', 'content': history('WRONG', 'WRONG')['messages'][0]['content']})
    assert recover_prompt_inputs(data) == ('HEADER', 'PLAIN')


def test_outer_whitespace_only_is_declared_match():
    out = validate_prompt_matches(history(), '\nHEADER\n', '\nPLAIN\n')
    assert out['plain_matches_public_ignoring_outer_whitespace'] is True


def test_changed_problem_dimension_cannot_match():
    with pytest.raises(ValueError, match='differs'):
        validate_prompt_matches(history('#define N 40'), '#define N 8', 'PLAIN')


def test_multiple_conflicting_original_input_prompts_block():
    data = history()
    data['messages'].extend(history('OTHER')['messages'])
    with pytest.raises(ValueError, match='conflicting'):
        recover_prompt_inputs(data)


def test_missing_prompt_is_not_inferred_from_candidate():
    with pytest.raises(ValueError, match='missing'):
        recover_prompt_inputs({'messages': [{'role': 'assistant', 'content': '```cpp\nCODE\n```'}]})


def test_incomplete_prompt_blocks():
    data = history()
    data['messages'][0]['content'] = 'Here is the plain C kernel:\n```cpp\nPLAIN\n```'
    with pytest.raises(ValueError, match='incomplete'):
        recover_prompt_inputs(data)


def test_candidate_preflight_compares_full_output_and_retains_mismatch(tmp_path, monkeypatch):
    from scripts import preflight_commercial_candidates as candidate
    bench = tmp_path / 'bench'
    bench.mkdir()
    header = bench / 'kernel.h'
    header.write_text('header')
    code = tmp_path / 'candidate.cpp'
    code.write_text('candidate')
    gold = tmp_path / 'gold.txt'
    gold.write_text('==BEGIN DUMP_ARRAYS==\nbegin dump: A\n1.0\nend   dump: A\n==END   DUMP_ARRAYS==\n')
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    audit = tmp_path / 'audit.json'
    audit.write_text(json.dumps({'benchmarks': [{'benchmark': 'hlsfactory_test',
        'staged_benchmark': str(bench), 'input_hashes': {'kernel.h': digest(header)},
        'native_golden': {'success': True, 'output': str(gold), 'output_sha256': digest(gold)},
        'outputs': {'A': {'shape': [1], 'kind': 'float'}}}]}))
    inventory = tmp_path / 'selected.json'
    row = {'record_id': 'm/test/s', 'benchmark': 'hlsfactory_test',
        'kernel_source': str(code), 'kernel_sha256': digest(code), 'priority_paper_slots': ['slot'],
        'paper_source_join_status': 'exact_selected_source_linked'}
    inventory.write_text(json.dumps([row]))
    def mock_native(code, bench, output):
        output.mkdir()
        path = output / 'output.txt'
        path.write_text(gold.read_text().replace('1.0', '2.0'))
        return {'success': True, 'status': 'executed', 'output': str(path)}
    monkeypatch.setattr(candidate, 'native_run', mock_native)
    candidate.prepare(inventory, audit, tmp_path / 'out', 1)
    report = json.loads((tmp_path / 'out' / 'report.json').read_text())
    assert report['counts'] == {'output_mismatch': 1}
    assert report['rows'][0]['comparison']['passed'] is False
    assert report['source_modified'] is False


def test_changed_selected_source_blocks_before_native_execution(tmp_path):
    from scripts.preflight_commercial_candidates import verify_selected
    path = tmp_path / 'kernel.cpp'
    path.write_text('new source')
    row = {'paper_source_join_status': 'exact_selected_source_linked',
           'kernel_source': str(path), 'kernel_sha256': 'invalid'}
    with pytest.raises(ValueError, match='source changed'):
        verify_selected(row, {})
