#!/usr/bin/env python3
"""Prospective, bounded fresh-agent attempts on all packaged PolyBench kernels.

K counts full agent invocations, not repair calls. This campaign reports observed
success within K attempts, not the IID combinatorial pass@k estimator. It is
separate from the frozen recipe study and never changes that study's outcomes.
"""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from contextlib import contextmanager

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SCHEMA = 'c2hls.polybench-agent-success-within-k.v1'
TOOLCHAIN = {'vitis_version': '2023.2', 'part': 'xcu280-fsvh2892-2L-e', 'clock_ns': 3.33, 'flow_target': 'vitis'}
DEFAULT_K = 5  # Same total-attempt convention as the sweep's five-attempt limit.
from cosim_walltime import TIMING_SCOPE


class AttemptAllowanceExhausted(RuntimeError):
    """A prespecified local allowance ended before another provider request."""


@contextmanager
def capped_agent(module, config, instances):
    """Bound the central API entry point for every agent role, not just repairs."""
    from sigmetrics_main_policy import check_request_allowance, SearchAllowanceExhausted
    original = module.C2HLSOrchestrator

    class CappedOrchestrator(original):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.passk_requests_started = 0
            instances.append(self)

        def _call_llm_with_model(self, messages, model=None, max_tokens=None, agent_name='orchestrator'):
            if self.passk_requests_started >= config['max_model_requests_per_attempt']:
                raise AttemptAllowanceExhausted('passk_budget_exhausted: maximum_model_requests')
            try:
                check_request_allowance(
                    self, messages,
                    request_cap=config['aggregate_input_token_cap_per_attempt'],
                    aggregate_cap=config['aggregate_input_token_cap_per_attempt'],
                )
            except SearchAllowanceExhausted as exc:
                raise AttemptAllowanceExhausted('passk_budget_exhausted: input_reservation_limit') from exc
            requested_tokens = max_tokens if max_tokens is not None else self.max_completion_tokens
            if requested_tokens > config['max_output_tokens_per_request']:
                raise ValueError('model request exceeded registered output limit')
            self.passk_requests_started += 1
            return super()._call_llm_with_model(
                messages, model=model, max_tokens=max_tokens, agent_name=agent_name,
            )

    module.C2HLSOrchestrator = CappedOrchestrator
    try:
        yield
    finally:
        module.C2HLSOrchestrator = original


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    temporary.replace(path)


def normalize(name):
    return name.removeprefix('hlsfactory_').replace('-', '_')


def phase_limit(reference=None, *, multiplier=2.0, floor=120, ceiling=10800, fallback=3600):
    """No cycles, censored durations, XSIM-only time, or whole-flow substitutes."""
    reference = reference or {}
    seconds = reference.get('cosim_design_wall_seconds')
    eligible = (reference.get('status') == 'pass' and reference.get('exact_input_match') is True
                and reference.get('toolchain_match') is True and reference.get('censored') is False
                and reference.get('timing_scope') == TIMING_SCOPE
                and isinstance(seconds, (int, float)) and not isinstance(seconds, bool)
                and math.isfinite(seconds) and seconds > 0)
    return {'seconds': min(ceiling, max(floor, math.ceil(multiplier * seconds))) if eligible else fallback,
            'basis': 'matched_successful_reference' if eligible else 'fixed_fallback_no_comparable_reference',
            'multiplier': multiplier, 'floor_seconds': floor, 'ceiling_seconds': ceiling,
            'fallback_seconds': fallback, 'reference': reference if eligible else None,
            'timing_scope': TIMING_SCOPE}


def public_metadata(original):
    keys = ('benchmark', 'source_repo', 'plain_c_file', 'header_file', 'testbench_file',
            'kernel_top', 'hls_top', 'translated_hls_top', 'support_files', 'include_dirs',
            'cosim_depths', 'part', 'clock_ns')
    result = {k: original[k] for k in keys if k in original}
    result.update(supports_csim=True, supports_cosim=True,
                  study_role='development_corpus_not_held_out',
                  validation_input_scope='unchanged_public_testbench_no_reduced_cosim_overrides')
    return result


def prepare(output, benchmark_root, modes, reference_report):
    if output.exists():
        raise FileExistsError(output)
    if not modes or len(modes) != len(set(modes)) or set(modes) - {'flash', 'dynamic'}:
        raise ValueError('Modes must be distinct registered framework strategies')
    suite = read(ROOT / 'configs/hlsfactory_development_suite.json')
    registry = read(ROOT / 'configs/hlsfactory_output_shapes_fc27133.json')
    names = {normalize(n) for n in suite['kernels']}
    available = {normalize(p.name): p for p in benchmark_root.glob('hlsfactory_*') if p.is_dir()}
    if len(names) != 28 or names - set(available):
        raise ValueError('Packaged 28-kernel suite incomplete; missing kernels must not be dropped')
    ledger_path = ROOT / 'artifacts/sigmetrics2027/api_budget.json'
    ledger = read(ledger_path)
    price = ledger['prices_per_million']['deepseek-flash']
    from experiment_budget import cost
    per_attempt_bound = cost(price, 100000, 5 * 8192)
    maximum_cost = 28 * len(modes) * DEFAULT_K * per_attempt_bound
    if sum(r['charged_or_reserved_usd'] for r in ledger['reservations']) + maximum_cost > ledger['limit_usd']:
        raise ValueError('Insufficient existing shared API budget headroom')
    output.mkdir(parents=True)
    source = output / 'source'
    source.mkdir()
    # Freeze implementation, prompts, shape registries, and the skill library.
    paths = list(ROOT.glob('*.py'))
    for folder in ('configs', 'skills', 'dataset_pipeline'):
        paths += [p for p in (ROOT / folder).rglob('*') if p.is_file() and p.suffix in ('.py', '.json', '.txt')]
    for name in ('run_polybench_passk.py', 'resume_sigmetrics_recipe_main.py'):
        paths.append(ROOT / 'scripts' / name)
    for path in paths:
        target = source / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    for required in ('cosim_walltime.py', 'hls_eval.py', 'scripts/run_polybench_passk.py'):
        if not (source / required).is_file():
            raise ValueError('New wall-time implementation is not ready: ' + required)
    reference_target = output / 'reference_walltime_audit.json'
    shutil.copy2(reference_report, reference_target)
    # Audit reports are deliberately not auto-promoted across incompatible timing scopes.
    # A future matched calibration is a versioned manifest amendment, never an in-run update.
    entries = []
    for name in sorted(names):
        original = available[name]
        meta = read(original / 'metadata.json')
        cleaned = public_metadata(meta)
        cleaned.update(benchmark=original.name, plain_c_file=meta.get('plain_c_file') or 'plain.cpp',
                       header_file=meta.get('header_file') or 'kernel.h',
                       testbench_file=meta.get('testbench_file') or 'testbench.cpp')
        bench = output / 'inputs' / original.name
        bench.mkdir(parents=True)
        files = {cleaned['plain_c_file'], cleaned['header_file'], cleaned['testbench_file']}
        files.update(meta.get('support_files') or [])
        files.update(p.name for p in original.glob('*.h'))
        if any(Path(n).is_absolute() or '..' in Path(n).parts or 'gold' in n.lower() for n in files):
            raise ValueError('Unsafe or expert support file in public inputs: ' + original.name)
        original_hashes = {'metadata.json': sha(original / 'metadata.json')}
        for filename in sorted(files):
            target = bench / filename
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(original / filename, target)
            original_hashes[filename] = sha(original / filename)
        save(bench / 'metadata.json', cleaned)
        regkey = next((k for k in registry['benchmarks'] if normalize(k) == name), None)
        if not regkey or registry['benchmarks'][regkey]['testbench_sha256'] != sha(bench / cleaned['testbench_file']):
            raise ValueError('Public testbench does not match authoritative registry: ' + original.name)
        entries.append({'benchmark': original.name, 'path': str(bench), 'original_path': str(original),
                        'original_hashes': original_hashes,
                        'staged_hashes': {str(p.relative_to(bench)): sha(p) for p in bench.rglob('*') if p.is_file()},
                        'metadata_amendment': 'Public-only metadata; no expert source, variants, reduced-size co-sim testbench or expert support.',
                        'wall_time_policy': phase_limit()})
    provider = read(Path('/home/luo00466/code_translation-c2hls-hpca2027/artifacts/experiment_matrix/deepseek_flash_reviewer_ab27_20260925.json'))
    groups = [{'id': e['benchmark'] + '__' + mode, 'benchmark': e['benchmark'], 'mode': mode,
               'status': 'pending', 'next_attempt': 1, 'attempts': []} for e in entries for mode in modes]
    config = {'schema_version': SCHEMA, 'prepared_at': now(), 'source': str(source),
              'python': str(Path(sys.executable).resolve()), 'model': 'deepseek-flash',
              'model_documented_version': 'DeepSeek-V4.1-Flash',
              'model_source_url': 'https://api-docs.deepseek.com/quick_start/pricing/',
              'model_alias_caveat': 'Provider alias may change weights; preserve actual response model and experiment dates.',
              'endpoint': provider['endpoint'], 'key_file': provider['key_file'],
              'api_budget': str(ledger_path), 'shared_limit_usd': ledger['limit_usd'], 'rates': ledger['prices_per_million'],
              'maximum_estimated_usd': maximum_cost, 'per_attempt_maximum_estimated_usd': per_attempt_bound,
              'k': DEFAULT_K, 'modes': modes, 'groups': groups, 'benchmark_entries': entries,
              'toolchain': TOOLCHAIN, 'skill_scope': 'smart_exhaustive_v2', 'skill_library': str(source / 'skills/skills.json'),
              'max_model_requests_per_attempt': 5, 'max_selection_syntheses_per_attempt': 5,
              'max_output_tokens_per_request': 8192, 'aggregate_input_token_cap_per_attempt': 100000,
              'internal_repair_turns': 3, 'temperature': 0.7,
              'model_revision': 'deepseek-flash@api.deepseek.com-20261006-alias',
              'timeouts': {'llm': 300, 'csim': 300, 'csynth': 1200, 'cosim_invocation': 4800,
                           'cosim_invocation_synthesis_allowance': 1200, 'attempt_outer': 28800},
              'reference_audit': str(reference_target), 'reference_audit_sha256': sha(reference_target),
              'reference_policy': 'No currently audited record qualifies for exact public-input cosim_design phase calibration. Freeze fallback before generation; never infer seconds from cycles, kernel_runtime_us, XSIM-only or whole-flow duration.',
              'retry_policy': 'Fresh agent context after each measured nonpass, including wall-time timeout; at most K total invocations; stop on first independently verified feasible RTL pass. Infrastructure/accounting errors stop this lane, not a free sample retry.',
              'metric': 'Observed success within k full agent invocations, k=1..5; not an IID pass@k estimate.',
              'dataset_role': 'All 28 packaged development/skill-mining kernels; not held-out generalization evidence.',
              'frozen_files': {str(p): sha(p) for p in source.rglob('*') if p.is_file()}}
    config['frozen_files'].update({str(p): sha(p) for p in (output / 'inputs').rglob('*') if p.is_file()})
    config['frozen_files'][str(reference_target)] = sha(reference_target)
    config['frozen_files'][config['python']] = sha(config['python'])
    save(output / 'manifest.json', config)
    save(output / 'state.json', {'status': 'prepared', 'updated_at': now(), 'manifest_sha256': sha(output / 'manifest.json'), 'groups': groups})
    print(json.dumps({'prepared': True, 'benchmarks': 28, 'groups': len(groups), 'max_attempts': len(groups)*DEFAULT_K,
                      'maximum_estimated_usd': maximum_cost, 'source_runner': str(source / 'scripts/run_polybench_passk.py')}))


def verify(output):
    config = read(output / 'manifest.json')
    if config['schema_version'] != SCHEMA:
        raise ValueError('Unexpected campaign')
    if read(output / 'state.json')['manifest_sha256'] != sha(output / 'manifest.json'):
        raise ValueError('Campaign manifest differs from prepared identity')
    for path, digest in config['frozen_files'].items():
        if sha(path) != digest:
            raise ValueError('Frozen input changed: ' + path)
    budget = read(config['api_budget'])
    if budget['limit_usd'] != config['shared_limit_usd'] or budget['prices_per_million'] != config['rates']:
        raise ValueError('API budget policy changed')
    return config


def environment(config, group, attempt, *, preflight=False):
    from evaluation_repro import REFERENCE_BLIND_OVERRIDES
    env = {k: v for k, v in os.environ.items() if not k.startswith('C2HLS_') and k not in ('OPENAI_API_KEY', 'OPENAI_BASE_URL')}
    env.update(REFERENCE_BLIND_OVERRIDES)
    entry = next(e for e in config['benchmark_entries'] if e['benchmark'] == group['benchmark'])
    identity = group['id'] + f'__attempt{attempt}'
    scratch = Path(config['source']).parent / 'scratch' / identity
    env.update({'C2HLS_REFERENCE_VALIDATE_MODE': 'public_only', 'C2HLS_COSIM_REQUIRED': '1',
                'C2HLS_HLSFACTORY_SHAPE_REGISTRY': str(Path(config['source']) / 'configs/hlsfactory_output_shapes_fc27133.json'),
                'C2HLS_VITIS_VERSION': '2023.2', 'C2HLS_PART': TOOLCHAIN['part'], 'C2HLS_CLOCK_NS': '3.33',
                'C2HLS_CSIM_TIMEOUT': '300', 'C2HLS_SYNTH_TIMEOUT': '1200',
                'C2HLS_COSIM_TIMEOUT': str(entry['wall_time_policy']['seconds'] + config['timeouts']['cosim_invocation_synthesis_allowance']),
                'C2HLS_COSIM_WALL_TIMEOUT_SECONDS': str(entry['wall_time_policy']['seconds']),
                'C2HLS_LLM_TIMEOUT': '300', 'C2HLS_HW_EMU_FINAL': '0', 'C2HLS_TMP_ROOT': str(scratch),
                'C2HLS_VITIS_USER_HOME': str(scratch / 'vitis_home'),
                'C2HLS_SAMPLE_ID': Path(config['source']).parent.name + '__' + identity,
                'C2HLS_API_BUDGET_LEDGER': config['api_budget'], 'C2HLS_AGGREGATE_INPUT_TOKEN_BUDGET': '100000',
                'C2HLS_MAX_COMPLETION_TOKENS': '8192', 'C2HLS_CANDIDATES_PER_STEP': '1',
                'C2HLS_ATTEMPTS_PER_CANDIDATE': str(config['internal_repair_turns']), 'C2HLS_EXHAUSTIVE_CANDIDATE_ATTEMPTS': '1',
                'C2HLS_TURNS': str(config['internal_repair_turns']),
                'C2HLS_LLM_CANDIDATE_BUDGET': str(config['max_model_requests_per_attempt']),
                'C2HLS_SYNTHESIS_EVAL_BUDGET': str(config['max_selection_syntheses_per_attempt']),
                'C2HLS_MODEL_REVISION': config['model_revision'], 'C2HLS_PROVIDER_MAX_RETRIES': '0',
                'C2HLS_STRATEGY': group['mode'], 'C2HLS_DYNAMIC_ROUTING': '1' if group['mode'] == 'dynamic' else '0',
                'C2HLS_SKILL_MODE': 'on', 'C2HLS_FORCE_SKILL_PROMPTS': '1', 'C2HLS_SKILL_PROMPT_SCOPE': config['skill_scope'],
                'C2HLS_SKILL_LIBRARY_PATH': config['skill_library'], 'C2HLS_QOR_DESIGN_SWEEP': '0',
                'C2HLS_QOR_PROFILE_REVIEW': '0', 'C2HLS_QUALITY_REPAIR_TURNS': '0', 'C2HLS_LLM_TEMPERATURE': str(config['temperature']),
                'C2HLS_DEEPSEEK_THINKING': 'disabled', 'C2HLS_DEEPSEEK_REASONING_EFFORT': 'high',
                'C2HLS_SKILL_SNAPSHOT_SHA256': sha(config['skill_library']),
                'C2HLS_VITIS_SETTINGS': '/mnt/data/luo00466/Xilinx/Vitis/2023.2/settings64.sh',
                'C2HLS_FLOW_TARGET': 'vitis', 'C2HLS_COSIM_TRACE_LEVEL': 'none', 'OMP_NUM_THREADS': '8'})
    if not preflight:
        env['OPENAI_API_KEY'] = Path(config['key_file']).read_text().strip()
        env['OPENAI_BASE_URL'] = config['endpoint']
    else:
        env['OPENAI_API_KEY'] = 'offline-no-provider-call'
        env['OPENAI_BASE_URL'] = 'http://127.0.0.1:9'
    return env


def classify(result):
    cosim = result.get('cosim') or {}
    code = result.get('hls_code') or ''
    selected = result.get('selected_code_sha256')
    c = cosim.get('kernel_runtime_cycles')
    rtl_evidence = (cosim.get('passed') is True and cosim.get('success') is True
             and cosim.get('ran') is True and isinstance(c, (float, int)) and not isinstance(c, bool)
             and math.isfinite(c) and c > 0 and (cosim.get('correctness') or {}).get('passed') is True
             and isinstance(code, str) and bool(code) and selected == hashlib.sha256(code.encode()).hexdigest()
             and selected == result.get('cosim_target_code_sha256') == cosim.get('target_code_sha256')
             and cosim.get('classification') == 'completed'
             and cosim.get('phase_markers_complete') is True
             and cosim.get('timing_scope') == TIMING_SCOPE
             and cosim.get('validation_log_scan', {}).get('complete') is True
             and not cosim.get('timed_out'))
    valid = (rtl_evidence and result.get('success') is True
             and result.get('rtl_executed_pass') is True
             and (result.get('csim') or {}).get('passed') is True
             and (result.get('csim') or {}).get('success') is True
             and ((result.get('csim') or {}).get('correctness') or {}).get('passed') is True
             and (result.get('candidate_feasibility') or {}).get('feasible') is True
             and (result.get('reference_isolation_audit') or {}).get('passed') is True)
    if valid:
        return 'pass'
    if cosim.get('passed') is True and not rtl_evidence:
        return 'evidence_error'
    if cosim.get('status') == 'timeout' or cosim.get('timed_out') is True:
        return 'timeout'
    return 'fail'


def accounting_issues(result, reservations, config, *, requests_started=None):
    """Reconcile every provider request; never settle unknown usage by inference."""
    usage = result.get('llm_usage') or {}
    events = usage.get('events') or []
    issues = []
    if result.get('exception_type') and result['exception_type'] != 'AttemptAllowanceExhausted':
        issues.append('execution_exception')
    for step in result.get('steps') or []:
        if step.get('exception_type') and step['exception_type'] != 'AttemptAllowanceExhausted':
            issues.append('execution_exception')
    if len(reservations) != len(events) or len(reservations) > config['max_model_requests_per_attempt']:
        issues.append('request_accounting_mismatch')
    if requests_started is not None and len(reservations) != requests_started:
        issues.append('request_dispatch_accounting_mismatch')
    if usage.get('calls', len(events)) != len(events):
        issues.append('usage_event_count_mismatch')
    for reservation, event in zip(reservations, events):
        if reservation['status'] != 'settled' or reservation.get('overrun'):
            issues.append('unknown_or_overrun_usage')
        if event.get('usage_available') is not True:
            issues.append('missing_provider_usage')
        if reservation.get('model') != config['model'] or event.get('model') != config['model']:
            issues.append('unexpected_provider_model')
        if any(reservation.get(k) != event.get(k) for k in ('input_tokens', 'output_tokens', 'cached_tokens', 'prompt_sha256')):
            issues.append('token_or_prompt_accounting_mismatch')
        if event.get('max_tokens') != config['max_output_tokens_per_request']:
            issues.append('output_token_limit_mismatch')
    # Check every reservation even if its response never produced a usage event.
    if any(row['status'] != 'settled' or row.get('overrun') for row in reservations):
        issues.append('unknown_or_overrun_usage')
    if sum(int(r.get('input_tokens') or 0) for r in reservations) > config['aggregate_input_token_cap_per_attempt']:
        issues.append('aggregate_input_limit_exceeded')
    if (result.get('tool_flows') or {}).get('counts', {}).get('csynth', 0) > config['max_selection_syntheses_per_attempt']:
        issues.append('synthesis_budget_exceeded')
    if reservations and not (result.get('reference_isolation_audit') or {}).get('passed'):
        issues.append('reference_isolation_audit_failed')
    return sorted(set(issues))


def require_attempt_ready(output, config, group, index):
    if not isinstance(index, int) or not 1 <= index <= config['k']:
        raise ValueError('Attempt limit exceeded')
    preflight = read(output / 'input_preflight.json')
    if not preflight.get('complete') or preflight.get('manifest_sha256') != sha(output / 'manifest.json'):
        raise ValueError('Complete matching offline input preflight is required')
    records = [r for r in preflight['records'] if r['benchmark'] == group['benchmark']]
    if len(records) != 1 or records[0].get('success') is not True:
        raise ValueError('Benchmark native input validation is blocked')
    for previous in range(1, index):
        record = read(output / 'attempts' / group['id'] / str(previous) / 'result.json')
        status = (record.get('passk_attempt') or {}).get('status')
        if status not in ('fail', 'timeout'):
            raise ValueError('Previous attempt passed or has unresolved evidence/accounting')


def attempt(output, group_id, index):
    config = verify(output)
    group = next(g for g in config['groups'] if g['id'] == group_id)
    require_attempt_ready(output, config, group, index)
    folder = output / 'attempts' / group_id / str(index)
    folder.mkdir(parents=True, exist_ok=False)
    env = environment(config, group, index)
    os.environ.clear(); os.environ.update(env)
    # Import after binding all experiment controls.
    import c2hls
    from evaluation_repro import build_run_fingerprint, attach_run_provenance
    from reference_isolation import audit_history_file
    from experiment_telemetry import capture_tool_calls, tool_summary
    entry = next(e for e in config['benchmark_entries'] if e['benchmark'] == group['benchmark'])
    profile = {'name': SCHEMA, 'reference_blind': True}
    fp = build_run_fingerprint(
        repo=Path(config['source']), benchmark_dir=Path(entry['path']),
        benchmark=group['benchmark'], model_id=config['model'], model_label='deepseek_flash',
        skill_mode=config['skill_scope'], steps=None, profile=profile, environ=env,
    )
    started = time.monotonic()
    calls, instances = [], []
    try:
        with capped_agent(c2hls, config, instances), capture_tool_calls() as calls:
            result = c2hls.run_benchmark_multistep(entry['path'], str(folder), gpt_model=config['model'],
                                                 turns_limitation=config['internal_repair_turns'], quality_repair_turns=0)
    except Exception as exc:
        # No credentials or arbitrary provider exception text in compact state.
        result = {'success': False, 'exception_type': type(exc).__name__}
        if isinstance(exc, AttemptAllowanceExhausted):
            result['budget_exhausted'] = True
        if instances:
            agent = instances[-1]
            agent._seal_candidate_event_stream(reason='attempt ended before candidate evaluation completed')
            result.update(llm_usage=agent._llm_usage_summary(),
                          synthesis_evaluations=agent._synthesis_evaluation_summary(),
                          hls_code=agent.hls_code, csim=agent.generated_csim, cosim=agent.generated_cosim,
                          selected_code_sha256=agent.selected_code_sha256,
                          cosim_target_code_sha256=agent.cosim_target_code_sha256)
            agent.save_multistep_results(str(folder), group['benchmark'], result)
    history = folder / (group['benchmark'] + '_history.json')
    audit = audit_history_file(history, benchmark_dir=Path(entry['path']),
                               reference_data=result.get('reference_validation'), controller_data=result)
    attach_run_provenance(result, fingerprint=fp, profile=profile,
                          elapsed_seconds=time.monotonic()-started, history_path=history, reference_audit=audit)
    result['tool_flows'] = tool_summary(calls)
    reservations = [r for r in read(config['api_budget'])['reservations'] if r['run_id'] == env['C2HLS_SAMPLE_ID']]
    issues = accounting_issues(result, reservations, config,
                               requests_started=instances[-1].passk_requests_started if instances else 0)
    status = 'execution_error' if issues else classify(result)
    result['passk_attempt'] = {'group_id': group_id, 'attempt': index, 'k': config['k'],
                              'status': status, 'issues': sorted(set(issues)), 'finished_at': now(),
                              'wall_time_policy': entry['wall_time_policy'], 'provider_requests': len(reservations),
                              'estimated_or_reserved_usd': sum(r['charged_or_reserved_usd'] for r in reservations)}
    save(folder / 'result.json', result)
    print(json.dumps(result['passk_attempt']))
    return 2 if status in ('execution_error', 'evidence_error') else 0


def preflight(output):
    config = verify(output)
    records = []
    for entry in config['benchmark_entries']:
        command = [config['python'], str(Path(config['source']) / 'scripts/run_polybench_passk.py'),
                   'check-input', '--output', str(output), '--group', entry['benchmark'] + '__' + config['modes'][0]]
        try:
            result = subprocess.run(command, env=environment(config, {'id': 'preflight_' + entry['benchmark'], 'benchmark': entry['benchmark'], 'mode': config['modes'][0]}, 0, preflight=True),
                                    cwd=config['source'], capture_output=True, text=True, timeout=240)
            record = json.loads(result.stdout.splitlines()[-1])
            if result.returncode != 0 or record.get('benchmark') != entry['benchmark']:
                record = {'benchmark': entry['benchmark'], 'success': False, 'error': 'preflight_worker_failed'}
        except (json.JSONDecodeError, IndexError, subprocess.TimeoutExpired):
            record = {'benchmark': entry['benchmark'], 'success': False, 'error': 'preflight_worker_failed'}
        records.append(record)
        save(output / 'input_preflight.json', {'records': records, 'complete': len(records) == len(config['benchmark_entries']),
                                             'manifest_sha256': sha(output / 'manifest.json'), 'updated_at': now(),
                                             'counts': dict(Counter('pass' if r['success'] else 'blocked_input' for r in records))})
        print(json.dumps(record), flush=True)
    return 0 if all(r['success'] for r in records) else 2


def check_input(output, group_id):
    config = read(output / 'manifest.json')
    group = next(g for g in config['groups'] if g['id'] == group_id)
    entry = next(e for e in config['benchmark_entries'] if e['benchmark'] == group['benchmark'])
    import c2hls
    try:
        inputs = c2hls._load_benchmark_inputs(entry['path'])
        if inputs['ground_truth_code'] or inputs['gold_hls_source_code'] or inputs['gt_variants']:
            raise ValueError('Expert material was staged')
        oracle = c2hls._prepare_independent_golden(inputs)
        record = {'benchmark': entry['benchmark'], 'success': oracle.get('success') is True,
                  'golden': oracle.get('provenance'), 'error': oracle.get('error'), 'provider_calls': 0, 'vitis_calls': 0}
    except Exception as exc:
        record = {'benchmark': entry['benchmark'], 'success': False, 'error': 'native_preflight_exception',
                  'exception_type': type(exc).__name__, 'provider_calls': 0, 'vitis_calls': 0}
    print(json.dumps(record))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['prepare', 'verify', 'attempt', 'preflight', 'check-input'])
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--benchmark-root', type=Path, default=Path('/home/luo00466/code_translation-c2hls-enhanced-fc27133/benchmarks'))
    p.add_argument('--modes', nargs='+', choices=['flash', 'dynamic'], default=['flash', 'dynamic'])
    p.add_argument('--reference-report', type=Path, default=ROOT/'artifacts/sigmetrics2027/reference_walltime_20261006/report.json')
    p.add_argument('--group')
    p.add_argument('--attempt', type=int)
    args = p.parse_args(); output = args.output.resolve()
    if args.action == 'prepare': prepare(output, args.benchmark_root, args.modes, args.reference_report)
    elif args.action == 'verify': print(json.dumps({'verified': True, 'groups': len(verify(output)['groups'])}))
    elif args.action == 'attempt': return attempt(output, args.group, args.attempt)
    elif args.action == 'preflight': return preflight(output)
    else: check_input(output, args.group)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
