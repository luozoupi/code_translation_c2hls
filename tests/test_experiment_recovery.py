"""Offline serializer-recovery proofs: no imports of numeric evaluators/tools."""
from copy import deepcopy
from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

SPEC = spec_from_file_location('standalone_usage_recovery', Path(__file__).parents[1]
                              / 'hlsmith/research/experiment_recovery.py')
recovery = module_from_spec(SPEC)
SPEC.loader.exec_module(recovery)


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


@pytest.fixture
def saved_cell(tmp_path):
    job = {'id': 'kernel__oneshot__r1__clean-researcher', 'benchmark': 'kernel',
           'arm': 'oneshot', 'repeat': 1, 'cohort': 'clean-researcher', 'maximum_job_seconds': 100}
    manifest = {'jobs': [job], 'cohorts': [{'id': job['cohort'], 'source': 'clean',
                                          'prompt_profile': 'researcher-v1'}],
                'arms': [{'id': 'oneshot', 'entrypoint': 'run_benchmark'}],
                'provider': {'provider': 'deepseek', 'model': 'deepseek-flash'},
                'target': {'part': 'u280', 'clock_ns': 3.33},
                'decoding': {'thinking': 'disabled', 'temperature': 0, 'top_p_requested': 1},
                'max_output_tokens_per_request': 8192}
    write(tmp_path / 'manifest.json', manifest)
    digest = recovery.read_json(tmp_path / 'manifest.json')[1]['sha256']
    state = {'manifest_sha256': digest, 'status': 'stopped_execution_error', 'job_seconds_used': 25,
             'jobs': [{'id': job['id'], 'status': 'execution_error', 'elapsed_seconds': 25}]}
    write(tmp_path / 'state.json', state)
    write(tmp_path / 'launch.json', {'manifest_sha256': digest,
          'limits': {'api_usd': 100, 'compute_seconds': 1000, 'max_input_tokens_per_cell': 1000000}})
    profile = {'name': 'researcher-v1', 'sha256': 'a' * 64}
    events, reservations = [], []
    for index, (inputs, cached, outputs) in enumerate([(100, 20, 50), (200, 120, 30)]):
        event = {'provider': 'deepseek', 'model': 'deepseek-flash', 'response_model': 'deepseek-flash',
                 'request_id': None, 'usage_available': True, 'prompt_sha256': str(index + 1) * 64,
                 'input_tokens': inputs, 'cached_tokens': cached, 'output_tokens': outputs,
                 'total_tokens': inputs + outputs, 'max_tokens': 8192, 'prompt_profile': profile,
                 'decoding': {'thinking': 'disabled', 'temperature': 0, 'requested_top_p': 1,
                              'seed': None, 'seed_supported': False}}
        events.append(event)
        reservations.append({'id': 'request-' + str(index), 'run_id': job['id'], 'model': 'deepseek-flash',
             'response_model': 'deepseek-flash', 'request_id': None, 'status': 'settled', 'overrun': False,
             'prompt_sha256': event['prompt_sha256'], 'input_tokens': inputs, 'cached_tokens': cached,
             'output_tokens': outputs, 'input_token_bound': 1000, 'output_token_bound': 8192,
             'charged_or_reserved_usd': ((inputs - cached) * .3 + cached * .006 + outputs * 1.2) / 1e6})
    usage = {'calls': 2, 'candidate_requests': 2, 'usage_missing_calls': 0, 'events': events,
             'input_tokens': 300, 'cached_tokens': 140, 'output_tokens': 80, 'total_tokens': 380}
    history = {'model': 'deepseek-flash', 'model_translator': 'deepseek-flash',
               'model_synthesis': 'deepseek-flash', 'model_quality_repair': 'deepseek-flash',
               'llm_usage': usage, 'messages': []}
    result = {'benchmark': 'kernel', 'success': False, 'phase': 'B', 'generated_status': 'failed',
              'error': 'HLS synthesis/correctness failed', 'turn_history': [],
              'run': {'model': 'deepseek-flash', 'part': 'u280', 'clock_ns': 3.33,
                      'reference_blind': True, 'ground_truth_control_enabled': False,
                      'prompt_profile': profile, 'llm_usage': usage}}
    folder = tmp_path / 'jobs' / job['id']
    write(folder / 'generation/kernel_history.json', history)
    write(folder / 'generation.json', result)
    write(folder / 'generation_process.json', {'status': 'execution_error', 'returncode': 1,
                                              'elapsed_seconds': 25})
    ledger = {'schema_version': 'c2hls.api-budget.v1', 'limit_usd': 100,
              'prices_per_million': {'deepseek-flash': {'input': .3, 'cached_input': .006, 'output': 1.2}},
              'reservations': reservations}
    write(tmp_path / 'api_budget.json', ledger)
    return SimpleNamespace(output=tmp_path, manifest=manifest, job=job, folder=folder,
                           history=history, result=result, ledger=ledger)


def inspect(cell):
    return recovery.inspect_early_nonpass(cell.output, cell.manifest, cell.job['id'])


def update_usage(cell, change):
    change(cell.history['llm_usage'])
    cell.result['run']['llm_usage'] = deepcopy(cell.history['llm_usage'])
    write(cell.folder / 'generation/kernel_history.json', cell.history)
    write(cell.folder / 'generation.json', cell.result)


def test_existing_error_has_pure_hash_bound_nonpass_proposal_without_refunds(saved_cell):
    cell = saved_cell
    before = {path: path.read_bytes() for path in cell.output.rglob('*') if path.is_file()}
    receipt = recovery.correction_proposal(cell.output, cell.manifest, cell.job['id'])
    assert receipt['measurement_proposal'] == {'job': cell.job, 'status': 'generation_nonpass', 'rtl': {'ran': False}}
    assert receipt['elapsed_seconds_retained'] == 25
    assert receipt['additional_provider_calls'] == 0 and receipt['applied'] is False
    assert receipt['ledger_reservation_ids'] == ['request-0', 'request-1']
    assert before == {path: path.read_bytes() for path in cell.output.rglob('*') if path.is_file()}


@pytest.mark.parametrize('corrupt', ['prompt', 'tokens', 'calls', 'model', 'profile', 'decoding'])
def test_rejects_mismatched_saved_usage_even_when_nested_copy_matches(saved_cell, corrupt):
    def change(usage):
        if corrupt == 'prompt': usage['events'][0]['prompt_sha256'] = 'f' * 64
        elif corrupt == 'tokens': usage['events'][0]['input_tokens'] += 1
        elif corrupt == 'calls': usage['calls'] += 1
        elif corrupt == 'model': usage['events'][0]['model'] = 'other-model'
        elif corrupt == 'profile': usage['events'][0]['prompt_profile'] = {'name': 'paper-sigmetrics-2026', 'sha256': 'a' * 64}
        else: usage['events'][0]['decoding']['thinking'] = 'enabled'
    update_usage(saved_cell, change)
    with pytest.raises(ValueError): inspect(saved_cell)


@pytest.mark.parametrize('corrupt', ['unknown_request', 'wrong_job', 'duplicate_id', 'charge', 'bound'])
def test_rejects_unsafe_or_differently_bound_ledger(saved_cell, corrupt):
    ledger = saved_cell.ledger
    if corrupt == 'unknown_request':
        ledger['reservations'].append({'id': 'other', 'run_id': 'other', 'status': 'reserved'})
    elif corrupt == 'wrong_job': ledger['reservations'][0]['run_id'] = 'other-job'
    elif corrupt == 'duplicate_id': ledger['reservations'][1]['id'] = ledger['reservations'][0]['id']
    elif corrupt == 'charge': ledger['reservations'][0]['charged_or_reserved_usd'] = 0
    else: ledger['reservations'][0]['input_token_bound'] = 99
    write(saved_cell.output / 'api_budget.json', ledger)
    with pytest.raises(ValueError): inspect(saved_cell)


def test_rejects_history_outside_exact_job_folder_and_existing_selection(saved_cell, tmp_path):
    cell = saved_cell
    path = cell.folder / 'generation/kernel_history.json'
    outside = tmp_path / 'different-history.json'
    path.rename(outside)
    path.symlink_to(outside)
    with pytest.raises((ValueError, OSError)): inspect(cell)
    path.unlink()
    outside.rename(path)
    (cell.folder / 'selected.cpp').write_text('void selected() {}')
    with pytest.raises(ValueError, match='selected design exists'): inspect(cell)


def test_no_fallback_over_existing_top_level_usage_or_wrong_benchmark(saved_cell):
    cell = saved_cell
    cell.result['llm_usage'] = {}
    write(cell.folder / 'generation.json', cell.result)
    with pytest.raises(ValueError, match='top-level usage is present'): inspect(cell)
    del cell.result['llm_usage']
    cell.result['benchmark'] = 'other'
    write(cell.folder / 'generation.json', cell.result)
    with pytest.raises(ValueError, match='early Phase-B'): inspect(cell)


def test_save_hook_changes_only_one_metadata_field_and_restores_original_writer(saved_cell):
    cell = saved_cell
    (cell.folder / 'generation.json').unlink()  # Simulate first save of a NEW future cell.
    original = deepcopy(cell.result)
    study = SimpleNamespace(save=write)
    with recovery.saved_usage_fallback(study, cell.output, cell.job['id']) as receipts:
        study.save(cell.folder / 'generation.json', cell.result)
        assert len(receipts) == 1
    assert study.save is write
    usage = cell.result.pop('llm_usage')
    assert cell.result == original
    assert usage == original['run']['llm_usage']
    assert json.loads((cell.folder / 'generation.returned-raw.json').read_text()) == original
    assert json.loads((cell.folder / 'generation.json').read_text())['llm_usage'] == usage
    assert (cell.folder / 'usage-recovery.json').is_file()


def test_save_hook_preserves_present_usage_and_zero_call_early_input_failure(saved_cell):
    cell = saved_cell
    value = {'llm_usage': {}, 'phase': 'B', 'success': False}
    (cell.folder / 'generation.json').unlink()  # First save of a new cell.
    study = SimpleNamespace(save=write)
    with recovery.saved_usage_fallback(study, cell.output, cell.job['id']) as receipts:
        study.save(cell.folder / 'generation.json', value)
        assert not receipts
    assert json.loads((cell.folder / 'generation.json').read_text()) == value
    (cell.folder / 'generation.json').unlink()  # Separate first-save scenario.
    with recovery.saved_usage_fallback(study, cell.output, cell.job['id']) as receipts:
        study.save(cell.folder / 'generation.json', {'phase': 'A', 'success': False})
        assert not receipts
    assert not (cell.folder / 'usage-recovery.json').exists()


def test_save_hook_refuses_existing_artifact_before_writing_sidecars(saved_cell):
    cell = saved_cell
    before = (cell.folder / 'generation.json').read_bytes()
    study = SimpleNamespace(save=write)
    with recovery.saved_usage_fallback(study, cell.output, cell.job['id']):
        with pytest.raises(ValueError, match='already exists'):
            study.save(cell.folder / 'generation.json', cell.result)
    assert study.save is write
    assert (cell.folder / 'generation.json').read_bytes() == before
    assert not (cell.folder / 'usage-recovery.json').exists()


def test_known_frozen_profile_omission_is_explicit_and_not_invented(saved_cell):
    cell = saved_cell
    cell.manifest['cohorts'][0].update(source='frozen', prompt_profile='paper-sigmetrics-2026')
    del cell.result['run']['prompt_profile']
    for event in cell.history['llm_usage']['events']:
        event.pop('prompt_profile')
        event.pop('response_model')
        event.pop('request_id')
    cell.result['run']['llm_usage'] = deepcopy(cell.history['llm_usage'])
    write(cell.output / 'manifest.json', cell.manifest)
    digest = recovery.read_json(cell.output / 'manifest.json')[1]['sha256']
    for filename in ('state.json', 'launch.json'):
        data = json.loads((cell.output / filename).read_text())
        data['manifest_sha256'] = digest
        write(cell.output / filename, data)
    write(cell.folder / 'generation/kernel_history.json', cell.history)
    write(cell.folder / 'generation.json', cell.result)
    usage, receipt = inspect(cell)
    assert receipt['legacy_prompt_profile_metadata_absent'] is True
    assert all('prompt_profile' not in event for event in usage['events'])


def test_multistep_entrypoint_passes_through_without_fallback(saved_cell):
    cell = saved_cell
    cell.manifest['arms'][0]['entrypoint'] = 'run_benchmark_multistep'
    write(cell.output / 'manifest.json', cell.manifest)
    digest = recovery.read_json(cell.output / 'manifest.json')[1]['sha256']
    for filename in ('state.json', 'launch.json'):
        data = json.loads((cell.output / filename).read_text())
        data['manifest_sha256'] = digest
        write(cell.output / filename, data)
    with pytest.raises(ValueError, match='single-shot'):
        inspect(cell)
    (cell.folder / 'generation.json').unlink()
    original = deepcopy(cell.result)
    study = SimpleNamespace(save=write)
    with recovery.saved_usage_fallback(study, cell.output, cell.job['id']) as receipts:
        study.save(cell.folder / 'generation.json', cell.result)
        assert not receipts
    assert cell.result == original
    assert not (cell.folder / 'usage-recovery.json').exists()


def test_invalid_fallback_still_preserves_original_return_before_stopping(saved_cell):
    cell = saved_cell
    cell.ledger['reservations'][0]['prompt_sha256'] = 'f' * 64
    write(cell.output / 'api_budget.json', cell.ledger)
    (cell.folder / 'generation.json').unlink()
    original = deepcopy(cell.result)
    study = SimpleNamespace(save=write)
    with recovery.saved_usage_fallback(study, cell.output, cell.job['id']):
        with pytest.raises(ValueError, match='prompt hashes'):
            study.save(cell.folder / 'generation.json', cell.result)
    assert cell.result == original
    assert json.loads((cell.folder / 'generation.json').read_text()) == original
    assert not (cell.folder / 'usage-recovery.json').exists()
    assert study.save is write
