"""Strict accounting repair at the frozen experiment worker's save boundary.

This module is standalone stdlib code: callers supply the pinned experiments
module. It never imports a numeric evaluator, requests a response, launches a
process, writes campaign state, or changes a selected design. Pure inspection
returns evidence; the context manager writes only new per-cell sidecars through
the original worker's save function when that worker first saves generation.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat

SCHEMA = 'hlsmith.saved-usage-recovery.v1'
_HEX = re.compile(r'[0-9a-f]{64}')
_TOKEN = re.compile(r'[A-Za-z0-9][A-Za-z0-9_-]*')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def content_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def integer(value, label):
    require(type(value) is int and value >= 0, 'invalid ' + label)
    return value


def number(value, label):
    require(not isinstance(value, bool) and isinstance(value, (int, float))
            and math.isfinite(value) and value >= 0, 'invalid ' + label)
    return float(value)


def read_json(path):
    """Reject symlinks in the exact artifact path and hash the bytes read."""
    path = Path(path)
    for parent in path.parents:
        require(stat.S_ISDIR(parent.lstat().st_mode), 'artifact parent is not a real directory')
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(descriptor, 'rb') as stream:
        before = os.fstat(stream.fileno())
        require(stat.S_ISREG(before.st_mode), 'artifact is not a regular file')
        require(before.st_size <= 64 * 1024 * 1024, 'artifact exceeds bounded JSON read')
        raw = stream.read(64 * 1024 * 1024 + 1)
        after = os.fstat(stream.fileno())
    signature = lambda value: (value.st_dev, value.st_ino, value.st_size,
                               value.st_mtime_ns, value.st_ctime_ns)
    require(len(raw) <= 64 * 1024 * 1024 and signature(before) == signature(after),
            'artifact changed during read')
    result = json.loads(raw)
    require(isinstance(result, dict), 'artifact must contain a JSON object')
    return result, {'path': str(path), 'sha256': hashlib.sha256(raw).hexdigest()}


def registered_job(output, manifest, job_id):
    output = Path(output).resolve()
    recorded, manifest_proof = read_json(output / 'manifest.json')
    require(recorded == manifest, 'supplied manifest differs from registered bytes')
    matches = [job for job in manifest['jobs'] if job['id'] == job_id]
    require(len(matches) == 1, 'job identity is not uniquely registered')
    job = matches[0]
    for name in ('benchmark', 'arm', 'cohort'):
        require(isinstance(job[name], str) and _TOKEN.fullmatch(job[name]), 'unsafe registered job name')
    repeat = integer(job['repeat'], 'repeat')
    require(repeat > 0 and job_id == '{}__{}__r{}__{}'.format(
        job['benchmark'], job['arm'], repeat, job['cohort']), 'registered job/folder identity differs')
    folder = output / 'jobs' / job_id
    require(stat.S_ISDIR(folder.lstat().st_mode), 'job folder is not a real directory')
    state, _ = read_json(output / 'state.json')
    launch, launch_proof = read_json(output / 'launch.json')
    require(state['manifest_sha256'] == manifest_proof['sha256']
            == launch['manifest_sha256'], 'registered manifest binding differs')
    rows = [row for row in state['jobs'] if row['id'] == job_id]
    require(len(rows) == 1, 'state job identity is not unique')
    cohorts = [cohort for cohort in manifest['cohorts'] if cohort['id'] == job['cohort']]
    arms = [arm for arm in manifest['arms'] if arm['id'] == job['arm']]
    require(len(cohorts) == len(arms) == 1
            and arms[0]['entrypoint'] in ('run_benchmark', 'run_benchmark_multistep'),
            'unsupported registered generation entrypoint')
    return output, folder, job, cohorts[0], rows[0], launch, manifest_proof, launch_proof


def inspect_early_nonpass(output, manifest, job_id, result=None):
    """Return verified usage and a proposal, without writing any artifact.

    Only the missing TOP-LEVEL field is repaired. The separately serialized
    run.llm_usage and exact-folder history must already agree completely.
    """
    output, folder, job, cohort, state_row, launch, manifest_proof, launch_proof = registered_job(
        output, manifest, job_id)
    arm = next(arm for arm in manifest['arms'] if arm['id'] == job['arm'])
    require(arm['entrypoint'] == 'run_benchmark',
            'recovery is limited to the known single-shot early-return defect')
    result_proof = None
    if result is None:
        result, result_proof = read_json(folder / 'generation.json')
    require('llm_usage' not in result, 'top-level usage is present; never replace existing accounting')
    require(result.get('success') is False and result.get('phase') == 'B'
            and result.get('benchmark') == job['benchmark']
            and result.get('generated_status') == 'failed', 'result is not the known early Phase-B nonpass')
    require(not os.path.lexists(folder / 'selected.cpp'), 'selected design exists; never reclassify selection')
    history, history_proof = read_json(folder / 'generation' / (job['benchmark'] + '_history.json'))
    usage = history.get('llm_usage')
    require(isinstance(usage, dict) and isinstance(result.get('run'), dict)
            and result['run'].get('llm_usage') == usage, 'nested run usage and saved history differ')
    run = result['run']
    model = manifest['provider']['model']
    require(run.get('model') == model and history.get('model') == model
            and run.get('part') == manifest['target']['part']
            and run.get('clock_ns') == manifest['target']['clock_ns']
            and run.get('reference_blind') is True
            and run.get('ground_truth_control_enabled') is False, 'run model, target or reference boundary differs')
    for field in ('model_translator', 'model_synthesis', 'model_quality_repair'):
        require(history.get(field) == model, 'history provider routing differs')
    events = usage.get('events')
    require(isinstance(events, list) and events, 'saved history has no usage events')
    count = len(events)
    require(integer(usage.get('calls'), 'usage calls') == count
            and integer(usage.get('candidate_requests'), 'candidate requests') == count
            and integer(usage.get('usage_missing_calls'), 'missing usage') == 0, 'usage call coverage differs')
    ledger, ledger_proof = read_json(output / 'api_budget.json')
    require(ledger.get('schema_version') == 'c2hls.api-budget.v1', 'unsupported ledger schema')
    cap = number(launch['limits']['api_usd'], 'API cap')
    require(cap > 0 and ledger['limit_usd'] == cap, 'ledger/launch API caps differ')
    reservations = ledger['reservations']
    require(isinstance(reservations, list)
            and all(row.get('status') == 'settled' and row.get('overrun') is False for row in reservations),
            'ledger has unsettled requests or overruns')
    identities = [row['id'] for row in reservations]
    require(len(identities) == len(set(identities)), 'duplicate ledger reservation identity')
    rows = [row for row in reservations if row['run_id'] == job_id]
    require(len(rows) == count, 'per-job ledger and history call counts differ')
    require(sum(number(row['charged_or_reserved_usd'], 'charged dollars') for row in reservations) <= cap + 1e-12,
            'campaign API cap already exceeded')
    totals = dict.fromkeys(('input_tokens', 'output_tokens', 'total_tokens', 'cached_tokens'), 0)
    profile = run.get('prompt_profile')
    legacy_profile = cohort['source'] == 'frozen' and cohort['prompt_profile'] == 'paper-sigmetrics-2026'
    if profile is None:
        require(legacy_profile, 'required prompt-profile identity missing')
    else:
        require(profile.get('name') == cohort['prompt_profile']
                and _HEX.fullmatch(profile.get('sha256', '')) is not None,
                'run prompt-profile identity differs')
    for event, row in zip(events, rows):
        require(event.get('usage_available') is True and event.get('provider') == manifest['provider']['provider']
                and event.get('model') == row.get('model') == model, 'usage provider identity differs')
        prompt = event.get('prompt_sha256')
        require(isinstance(prompt, str) and _HEX.fullmatch(prompt) and prompt == row.get('prompt_sha256'),
                'per-request prompt hashes differ')
        for key in ('input_tokens', 'output_tokens', 'cached_tokens'):
            value = integer(event.get(key), 'event ' + key)
            require(value == integer(row.get(key), 'ledger ' + key), 'per-request token usage differs')
            totals[key] += value
        require(event['cached_tokens'] <= event['input_tokens'], 'cached tokens exceed input')
        require(integer(event.get('total_tokens'), 'event total') == event['input_tokens'] + event['output_tokens'],
                'event token total differs')
        totals['total_tokens'] += event['total_tokens']
        require(row['input_tokens'] <= integer(row['input_token_bound'], 'input bound')
                and row['output_tokens'] <= integer(row['output_token_bound'], 'output bound')
                and event.get('max_tokens') == row['output_token_bound']
                == manifest['max_output_tokens_per_request'], 'request token bounds differ')
        require(row.get('response_model') == model
                and event.get('response_model', model) == row['response_model'], 'returned model identity differs')
        if 'request_id' in event:
            require(event['request_id'] == row.get('request_id'), 'request identities differ')
        if profile is not None:
            require(event.get('prompt_profile') == profile, 'per-request prompt profile differs')
        else:
            require('prompt_profile' not in event, 'legacy profile coverage is inconsistent')
        decoding = event.get('decoding') or {}
        require(decoding.get('thinking') == manifest['decoding']['thinking']
                and decoding.get('temperature') == manifest['decoding']['temperature']
                and decoding.get('requested_top_p') == manifest['decoding']['top_p_requested']
                and decoding.get('seed') is None and decoding.get('seed_supported') is False,
                'request decoding controls differ')
        price = ledger['prices_per_million'][model]
        expected = ((row['input_tokens'] - row['cached_tokens']) * number(price['input'], 'input price')
                    + row['cached_tokens'] * number(price['cached_input'], 'cache price')
                    + row['output_tokens'] * number(price['output'], 'output price')) / 1e6
        require(math.isclose(number(row['charged_or_reserved_usd'], 'request charge'), expected,
                             rel_tol=0, abs_tol=1e-12), 'settled charge differs from frozen rate accounting')
    require(all(integer(usage.get(key), 'summary ' + key) == value for key, value in totals.items()),
            'usage summary token totals differ')
    require(totals['input_tokens'] <= integer(launch['limits']['max_input_tokens_per_cell'], 'cell input cap'),
            'per-cell aggregate input cap exceeded')
    augmented = deepcopy(result)
    augmented['llm_usage'] = deepcopy(usage)
    receipt = {'schema': SCHEMA, 'job': deepcopy(job), 'original_state_row': deepcopy(state_row),
        'manifest': manifest_proof, 'launch': launch_proof, 'history': history_proof,
        'ledger': ledger_proof, 'ledger_job_rows_content_sha256': content_sha(rows),
        'ledger_reservation_ids': [row['id'] for row in rows],
        'returned_result_content_sha256': content_sha(result),
        'augmented_result_content_sha256': content_sha(augmented),
        'result_artifact': result_proof, 'usage_content_sha256': content_sha(usage),
        'mutation': 'add missing top-level llm_usage only', 'additional_provider_calls': 0,
        'legacy_prompt_profile_metadata_absent': profile is None,
        'qualification': 'Original early generation nonpass retained. History has no separate job ID; exact registered path, benchmark, nested run usage and one-to-one ledger identities bind it to this cell.'}
    return deepcopy(usage), receipt


def correction_proposal(output, manifest, job_id):
    """Describe an existing stopped cell's correction; never apply it."""
    output = Path(output).resolve()
    usage, receipt = inspect_early_nonpass(output, manifest, job_id)
    state, state_proof = read_json(output / 'state.json')
    row = receipt['original_state_row']
    require(state['status'] == 'stopped_execution_error' and row['status'] == 'execution_error'
            and not state.get('active_job') and not state.get('active_jobs'), 'campaign is not safely stopped')
    folder = output / 'jobs' / job_id
    require(not os.path.lexists(folder / 'measurement.json'), 'terminal measurement already exists')
    process, process_proof = read_json(folder / 'generation_process.json')
    require(process.get('status') == 'execution_error' and process.get('returncode') not in (None, 0),
            'saved generation did not end in the inspected execution error')
    receipt.update({'state': state_proof, 'generation_process': process_proof,
                    'original_execution_error_preserved': True,
                    'measurement_proposal': {'job': receipt['job'], 'status': 'generation_nonpass', 'rtl': {'ran': False}},
                    'elapsed_seconds_retained': number(row['elapsed_seconds'], 'elapsed seconds'),
                    'applied': False})
    return receipt


@contextmanager
def saved_usage_fallback(study, output, job_id):
    """Supply only the missing serializer field before original accounting.

    The caller invokes the pinned worker with stage='generate' (not 'all').
    New sidecars are written through its original save function; old artifacts
    and results with an existing usage field are never rewritten by this hook.
    """
    output = Path(output).resolve()
    manifest, _ = read_json(output / 'manifest.json')
    _, folder, job, _, _, _, _, _ = registered_job(output, manifest, job_id)
    arm = next(arm for arm in manifest['arms'] if arm['id'] == job['arm'])
    destination = folder / 'generation.json'
    original_save = study.save
    receipts = []

    def save_new(path, value):
        require(not os.path.lexists(path), 'recovery sidecar already exists')
        original_save(path, value)
        return read_json(path)[1]

    def intercepted(path, value):
        if Path(path).absolute() != destination:
            return original_save(path, value)
        require(not os.path.lexists(destination), 'generation artifact already exists; refuse overwrite')
        if (arm['entrypoint'] != 'run_benchmark' or 'llm_usage' in value or value.get('phase') != 'B'
                or value.get('success') is not False):
            return original_save(path, value)
        try:
            usage, receipt = inspect_early_nonpass(output, manifest, job_id, value)
            raw_path = folder / 'generation.returned-raw.json'
            receipt_path = folder / 'usage-recovery.json'
            require(not os.path.lexists(raw_path) and not os.path.lexists(receipt_path),
                    'recovery sidecar already exists')
            receipt['raw_return_artifact'] = save_new(raw_path, value)
            save_new(receipt_path, receipt)
        except Exception:
            # The frozen worker would save its returned result before rejecting
            # accounting. Preserve that evidence even when fallback validation
            # fails; no usage is added and the original error still propagates.
            if not os.path.lexists(destination):
                original_save(path, value)
            raise
        value['llm_usage'] = usage
        receipts.append(receipt)
        return original_save(path, value)

    study.save = intercepted
    try:
        yield receipts
    finally:
        study.save = original_save
