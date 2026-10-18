#!/usr/bin/env python3
"""Frozen 216-cell recipe-representation study with prerequisite parent validation.

The source snapshot, complete denominator, policies and job order precede model
outcomes. Each kernel's recipes and first prompts are frozen after its unchanged
public-C parent passes five CSim and five RTL cases, before any cell on that
kernel. Failed parents block all 18 registered cells without replacements.
"""
from __future__ import annotations
import argparse
import collections
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SCHEMA = 'c2hls.sigmetrics-recipe-main.v1'
MODELS = ('deepseek-flash', 'gpt-5.6-luna')
ARMS = ('none', 'flat', 'structured')
SCRIPT = 'scripts/run_sigmetrics_recipe_main.py'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(data, indent=2) + '\n')
    temp.replace(path)


def record_files(paths):
    return [{'path': str(p.resolve()), 'sha256': sha(p)} for p in sorted(set(map(Path, paths)))]


def verify_files(records):
    for item in records:
        if sha(item['path']) != item['sha256']:
            raise ValueError('frozen input changed: ' + item['path'])


def make_cells(benchmarks):
    cells = []
    # Interleave kernels, rotate arm positions over repeats, and reverse provider
    # order on alternate kernels. The ready queue never sorts on measured QoR.
    for repeat in range(3):
        for position in range(3):
            for index, benchmark in enumerate(benchmarks):
                arm = ARMS[(position + repeat + index) % 3]
                for model in (MODELS if (index + repeat) % 2 == 0 else MODELS[::-1]):
                    identity = f'{benchmark}__{model}__{arm}__r{repeat + 1}'
                    cells.append(dict(cell_id=identity, benchmark=benchmark, model=model,
                                      arm=arm, repeat=repeat + 1, status='pending'))
    return cells


def patch_snapshot(source):
    """Versioned local corrections; never mutate the shared or pilot controller."""
    prompt = source / 'prompt_c2hls.py'
    text = prompt.read_text()
    changes = {
        'You apply HLS optimizations incrementally, one technique at a time, to systematically improve performance.':
        'You optimize a complete HLS implementation. Combine compatible loop, memory, and interface transformations when useful; compiler feedback may guide subsequent complete implementations.',
        '- Each step should build on the previous code, adding ONE optimization technique.':
        '- Each candidate must preserve program behavior and the public interface; it may combine multiple optimization techniques.',
        '- Do NOT add optimizations beyond the one requested.':
        '- Select transformations appropriate to the requested complete-design optimization and observed compiler constraints.',
    }
    for before, after in changes.items():
        if text.count(before) != 1:
            raise ValueError('system prompt amendment no longer matches source')
        text = text.replace(before, after)
    prompt.write_text(text)
    controller = source / 'c2hls.py'
    original = controller.read_text()
    start = original.index('    def _optimization_step_attempt_single(')
    end = original.index('\n    def ', start + 5)
    method = original[start:end]
    loop_start = method.index('        for turn in range(attempt_limit):')
    prefix, loop = method[:loop_start], method[loop_start:]
    prefix += '        from sigmetrics_main_policy import SearchAllowanceExhausted\n'
    output = []
    guarded = 0
    for line in loop.splitlines(keepends=True):
        if 'reply = self._call_llm(self.messages)' in line:
            indent = line[:len(line) - len(line.lstrip())]
            output.extend([indent + 'if turn >= attempt_limit - 1:\n', indent + '    break\n'])
            output.extend([indent + 'try:\n', '    ' + line,
                           indent + 'except SearchAllowanceExhausted as exc:\n',
                           indent + '    self._append_history("system", "[Main search allowance] " + str(exc))\n',
                           indent + '    break\n'])
            guarded += 1
        else:
            output.append(line)
    if guarded != 7:
        raise ValueError(f'expected seven bounded repair/improvement calls; found {guarded}')
    changed = original[:start] + prefix + ''.join(output) + original[end:]
    compile(changed, str(controller), 'exec')
    controller.write_text(changed)
    return {'version': 'main-v1-flash-scope-and-final-attempt',
            'system_prompt_replacements': changes,
            'repair_calls_with_last_attempt_guard': guarded,
            'reason': 'consistent complete-design scope; do not request an unevaluated sixth response or lose earlier feasible candidates at the budget boundary'}


def parent_gate(measurement):
    issues = []
    if measurement.get('origin') != 'unchanged_public_c' or measurement.get('model_calls') != 0:
        issues.append('parent_not_unchanged_public_c')
    if measurement.get('success') is not True:
        issues.append('parent_not_successful')
    for stage in ('csim', 'cosim'):
        result = measurement.get(stage) or {}
        suite = result.get('validation_suite') or {}
        if result.get('passed') is not True or suite.get('executed_cases') != 5 or suite.get('all_cases_passed') is not True:
            issues.append(stage + '_missing_five_passes')
    counts = (measurement.get('tool_flows') or {}).get('counts') or {}
    if any(counts.get(kind) != count for kind, count in [('csim', 5), ('csynth', 1), ('rtl_cosim', 5)]):
        issues.append('parent_tool_counts_mismatch')
    return issues


def prepare(output):
    if output.exists():
        raise ValueError('campaign exists; refusing overwrite')
    inputs = ROOT / 'artifacts/sigmetrics2027/core_inputs_v1/manifest.json'
    pilot = ROOT / 'artifacts/sigmetrics2027/pilot_reports/20260929T024333638806Z/report.json'
    if json.loads(pilot.read_text()).get('engineering_gate_passed') is not True:
        raise ValueError('engineering pilot did not pass')
    from validation_suite import load_suite
    data = json.loads(inputs.read_text())
    benchmarks = [e['benchmark'] for e in data['entries']]
    if len(benchmarks) != 12 or len(set(benchmarks)) != 12:
        raise ValueError('expected exactly twelve prespecified kernels')
    for benchmark in benchmarks:
        load_suite(inputs, benchmark)
    ledger = ROOT / 'artifacts/sigmetrics2027/api_budget.json'
    budget = json.loads(ledger.read_text())
    if budget['limit_usd'] != 300 or any(r.get('overrun') or r['status'] != 'settled' for r in budget['reservations']):
        raise ValueError('shared budget requires reconciliation')
    from experiment_budget import cost
    bound = sum(108 * cost(budget['prices_per_million'][model], 500000, 5 * 16384) for model in MODELS)
    used = sum(r['charged_or_reserved_usd'] for r in budget['reservations'])
    if used + bound > budget['limit_usd']:
        raise ValueError('insufficient shared ledger headroom for registered upper bound')
    output.mkdir(parents=True)
    source = output / 'source'
    source.mkdir()
    paths = list(ROOT.glob('*.py')) + list((ROOT / 'configs').glob('*.json'))
    paths += [ROOT / SCRIPT, ROOT / 'scripts/build_sigmetrics_parent.py']
    frozen = []
    for path in paths:
        target = source / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        frozen.append(target)
    amendment = patch_snapshot(source)
    save(output / 'source_amendment.json', amendment)
    skills = output / 'skills.json'
    shutil.copy2('/home/luo00466/code_translation-c2hls-hpca2027/skill_v3_no_rmw_fc27133/skills.json', skills)
    native_path = inputs.parent / 'native_validation.json'
    native = json.loads(native_path.read_text())
    frozen += [inputs, native_path, pilot, skills, output / 'source_amendment.json', ROOT / 'docs/sigmetrics_recipe_main_20260930.md', ROOT / 'tests/test_sigmetrics_recipe_main.py']
    for entry in data['entries']:
        frozen += [p for p in Path(entry['benchmark_dir']).iterdir() if p.is_file()]
        frozen += [Path(case['shape_registry_path']) for case in entry['cases']]
    frozen += [Path(case['golden_output_path']) for case in native['cases']]
    atax = ROOT / 'artifacts/sigmetrics2027/parent_pilot_20260928/atax'
    if parent_gate(json.loads((atax / 'measurement.json').read_text())):
        raise ValueError('reused ATAX parent no longer passes strict five-case gate')
    for name in ('measurement.json', 'parent.cpp', 'phase_b_manifest.json'):
        frozen.append(atax / name)
    cells = make_cells(benchmarks)
    config = dict(schema_version=SCHEMA, prepared_at=now(), cells=cells,
                  inputs=str(inputs), skills=str(skills), source=str(source), api_budget=str(ledger),
                  benchmark_entries=data['entries'], workers=2, llm_call_cap=5, selection_synthesis_cap=5,
                  attempts_per_candidate=5, candidates_per_step=1, max_response_tokens=16384,
                  aggregate_input_token_cap=500000, per_request_input_token_bound_cap=250000, strategy='flash', qor_reviewer=False,
                  resource_thresholds={'scratch_free_gib': 60, 'memory_available_gib': 64, 'load_per_cpu': 0.8},
                  source_amendment=amendment, rates_per_million=budget['prices_per_million'],
                  campaign_upper_bound_estimated_usd=bound, shared_limit_usd=300,
                  parent_job_timeout_seconds=24000, cell_job_timeout_seconds=43200,
                  parent_reuse={'hlsfactory_atax': str(atax)},
                  schedule_policy='one parent plus one ready cell while parents remain; then two cells; fixed ready order, no QoR sorting',
                  failure_policy='all 18 cells remain blocked on a failed parent or no frozen-router match; no replacement or reroll; accounting/infrastructure errors halt new work',
                  repeat_policy='three independent API runs; no supported seed control or immutable provider-weight claim',
                  search_policy='up to five complete-design proposals with compiler feedback; continue after feasible success; malformed responses and errors may stop early',
                  selection_policy='best feasible candidate or unchanged public parent, selected before RTL; no reselection after failed RTL',
                  knowledge_control='generic HLS prompt shared by every arm; identical complete recipe statements in flat and structured arms; no extra recipes in none arm',
                  timeouts={'llm': 300, 'csim': 300, 'csynth': 1200, 'cosim': 3600},
                  files=record_files(frozen))
    save(output / 'manifest.json', config)
    parents = [dict(benchmark=b, status='pending', path=config['parent_reuse'].get(b, str(output / 'parents' / b))) for b in benchmarks]
    save(output / 'state.json', dict(status='prepared', updated_at=now(), manifest_sha256=sha(output / 'manifest.json'), parents=parents, cells=cells))
    command = [sys.executable, str(source / SCRIPT), 'qualify', '--output', str(output), '--benchmark', 'hlsfactory_atax']
    result = execute(command, source, output / 'hlsfactory_atax.qualification.log', base_environment(config, 'hlsfactory_atax__qualification'), 600)
    if result['returncode'] != 0:
        raise ValueError('offline ATAX qualification failed; preserve preparation artifacts and inspect log')
    state = json.loads((output / 'state.json').read_text())
    reused = next(parent for parent in state['parents'] if parent['benchmark'] == 'hlsfactory_atax')
    reused.update(status='ready', qualified_at=now(), qualification_sha256=sha(qualification_path(output, 'hlsfactory_atax')))
    save(output / 'state.json', state)
    print(json.dumps({'cells': len(cells), 'source_files': len(paths), 'maximum_estimated_usd': bound, 'manifest_sha256': sha(output / 'manifest.json'), 'atax_qualified_and_prompts_frozen': True}))


def read_config(output):
    config = json.loads((output / 'manifest.json').read_text())
    if config['schema_version'] != SCHEMA:
        raise ValueError('unexpected campaign schema')
    verify_files(config['files'])
    budget = json.loads(Path(config['api_budget']).read_text())
    if budget['limit_usd'] != config['shared_limit_usd'] or budget['prices_per_million'] != config['rates_per_million']:
        raise ValueError('shared ledger cap/rates drifted from registration')
    return config


def base_environment(config, identity):
    from evaluation_repro import REFERENCE_BLIND_OVERRIDES
    env = {k: v for k, v in os.environ.items() if not k.startswith('C2HLS_')}
    env.update(REFERENCE_BLIND_OVERRIDES)
    scratch = Path('/tmp') / Path(config['source']).parent.name / identity
    env.update({'C2HLS_REFERENCE_VALIDATE_MODE': 'public_only', 'C2HLS_COSIM_REQUIRED': '1',
                'C2HLS_HLSFACTORY_SHAPE_REGISTRY': str(Path(config['inputs']).parent / 'shapes_case0.json'),
                'C2HLS_VITIS_VERSION': '2023.2', 'C2HLS_CSIM_TIMEOUT': '300', 'C2HLS_SYNTH_TIMEOUT': '1200',
                'C2HLS_COSIM_TIMEOUT': '3600', 'C2HLS_LLM_TIMEOUT': '300', 'C2HLS_HW_EMU_FINAL': '0',
                'C2HLS_TMP_ROOT': str(scratch), 'C2HLS_VITIS_USER_HOME': str(scratch / 'vitis_home'),
                'OMP_NUM_THREADS': '8'})
    return env


def setup_cell_environment(config, spec, qualification, preview=False):
    identity = spec['cell_id']
    os.environ.clear()
    # Caller supplies clean base environment before this function.
    os.environ.update(setup_cell_environment.base)
    os.environ.update({'C2HLS_PHASE_B_SEED_MANIFEST': qualification['parent_manifest'],
        'C2HLS_SAMPLE_ID': Path(config['source']).parent.name + '__' + identity,
        'C2HLS_API_BUDGET_LEDGER': config['api_budget'],
        'C2HLS_AGGREGATE_INPUT_TOKEN_BUDGET': str(config['aggregate_input_token_cap']),
        'C2HLS_MAX_COMPLETION_TOKENS': str(config['max_response_tokens']),
        'C2HLS_LLM_CANDIDATE_BUDGET': '5', 'C2HLS_SYNTHESIS_EVAL_BUDGET': '5',
        'C2HLS_CANDIDATES_PER_STEP': '1', 'C2HLS_ATTEMPTS_PER_CANDIDATE': '5',
        'C2HLS_EXHAUSTIVE_CANDIDATE_ATTEMPTS': '1', 'C2HLS_STRATEGY': 'flash',
        'C2HLS_DYNAMIC_ROUTING': '0', 'C2HLS_QOR_DESIGN_SWEEP': '0', 'C2HLS_QOR_PROFILE_REVIEW': '0',
        'C2HLS_QUALITY_REPAIR_TURNS': '0', 'C2HLS_SKILL_USAGE_DECLARATION': '0',
        'C2HLS_SKILL_LIBRARY_PATH': config['skills'], 'C2HLS_LLM_TEMPERATURE': '0', 'C2HLS_LLM_TOP_P': '1',
        'C2HLS_DEEPSEEK_THINKING': 'disabled', 'C2HLS_OPENAI_REASONING_EFFORT': 'none'})
    if spec['arm'] == 'none':
        os.environ.update(C2HLS_SKILL_MODE='off', C2HLS_FORCE_SKILL_PROMPTS='0', C2HLS_SKILL_PROMPT_SCOPE='skillless')
    else:
        os.environ.update(C2HLS_SKILL_MODE='on', C2HLS_FORCE_SKILL_PROMPTS='1', C2HLS_SKILL_PROMPT_SCOPE='explicit',
                          C2HLS_SKILL_EXPLICIT_IDS=','.join(qualification['recipe_ids']), C2HLS_SKILL_PROMPT_MODE='same_content_' + spec['arm'])
    if preview:
        os.environ['OPENAI_API_KEY'] = 'offline-preview-no-network'
        os.environ['OPENAI_BASE_URL'] = 'http://127.0.0.1:9'
    elif spec['model'] == 'deepseek-flash':
        old = json.loads(Path('/home/luo00466/code_translation-c2hls-hpca2027/artifacts/experiment_matrix/deepseek_flash_reviewer_ab27_20260925.json').read_text())
        os.environ['OPENAI_API_KEY'] = Path(old['key_file']).read_text().strip()
        os.environ['OPENAI_BASE_URL'] = old['endpoint']
    else:
        os.environ['OPENAI_API_KEY'] = Path('/home/luo00466/gpt-key.txt').read_text().strip()
        os.environ['OPENAI_BASE_URL'] = 'https://api.openai.com/v1'


def prompt_digest(messages):
    return hashlib.sha256(json.dumps(messages, ensure_ascii=True, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


class ProtocolBlocked(ValueError):
    pass


def qualification_path(output, benchmark):
    return output / 'qualification' / benchmark / 'qualification.json'


def qualify(output, benchmark):
    config = read_config(output)
    state = json.loads((output / 'state.json').read_text())
    parent = next(p for p in state['parents'] if p['benchmark'] == benchmark)
    directory = Path(parent['path'])
    measurement = json.loads((directory / 'measurement.json').read_text())
    issues = parent_gate(measurement)
    if issues:
        raise ValueError('parent qualification failed: ' + ', '.join(issues))
    from phase_b_manifest import load_phase_b_seed
    from skill_library import load_frozen_library, same_content_recipe_atoms, render_same_content_recipe
    from smart_skill_router import route_smart_skills
    import c2hls
    entry = next(e for e in config['benchmark_entries'] if e['benchmark'] == benchmark)
    public = c2hls._load_benchmark_inputs(entry['benchmark_dir'])
    seed = load_phase_b_seed(directory / 'phase_b_manifest.json', benchmark=benchmark,
                            input_c=public['c_code'], header_code=public['header_code'],
                            expected_part='xcu280-fsvh2892-2L-e', expected_clock_ns=3.33, expected_vitis_version='2023.2')
    if seed['entry'].get('source_artifacts', {}).get('measurement_sha256') != sha(directory / 'measurement.json'):
        raise ValueError('phase-B parent does not identify its actual measurement')
    parent_feasibility = c2hls._paper_candidate_feasibility(seed['csynth_report'], csim=seed['csim'], correctness_required=True, part='xcu280-fsvh2892-2L-e', clock_ns=3.33)
    if seed['code'] != public['c_code']:
        raise ValueError('parent source is not unchanged public C')
    if measurement.get('validation_manifest_sha256') != sha(config['inputs']):
        raise ValueError('parent measured with a different input suite')
    library = load_frozen_library(config['skills'])
    route = route_smart_skills(library, scope='smart_exhaustive_v2', step_name='flash', current_code=seed['code'],
                              synth_report=seed['csynth_report'], vitis_version='2023.2', fpga='xcu280-fsvh2892-2L-e', max_skills=3, min_score=4.0)
    if not route.selected:
        raise ProtocolBlocked('no recipes selected by the prospective frozen router; do not substitute by hand')
    dest = qualification_path(output, benchmark)
    if dest.exists():
        raise ValueError('qualification already exists; refusing re-route')
    records = record_files([directory / name for name in ['measurement.json', 'parent.cpp', 'phase_b_manifest.json']])
    recipes = [dict(id=sk.id, atoms=same_content_recipe_atoms(sk), flat=render_same_content_recipe(sk, structured=False),
                    structured=render_same_content_recipe(sk, structured=True)) for sk in route.selected]
    qualification = dict(benchmark=benchmark, qualified_at=now(), parent_manifest=str(directory / 'phase_b_manifest.json'),
                         parent_gate_issues=[], parent_feasibility=parent_feasibility, parent_code_sha256=seed['provenance']['code_sha256'],
                         recipe_ids=[sk.id for sk in route.selected], route=route.audit, recipes=recipes, files=records,
                         selection_source='unchanged public-C parent only, before any model call on this kernel',
                         campaign_manifest_sha256=sha(output / 'manifest.json'))
    save(dest, qualification)
    previews = {}
    for model in MODELS:
        for arm in ARMS:
            spec = next(c for c in config['cells'] if c['benchmark'] == benchmark and c['model'] == model and c['arm'] == arm)
            preview_path = dest.parent / f'{model}__{arm}.json'
            command = [sys.executable, str(Path(config['source']) / SCRIPT), 'preview', '--output', str(output), '--cell-id', spec['cell_id']]
            result = subprocess.run(command, cwd=config['source'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            (dest.parent / f'{model}__{arm}.preview.log').write_text(result.stdout)
            if result.returncode != 0 or not preview_path.exists():
                raise ValueError('offline first-call preview failed: ' + spec['cell_id'])
            payload = json.loads(preview_path.read_text())
            text = '\n'.join(message['content'] for message in payload['messages'])
            if arm != 'none':
                for recipe in recipes:
                    for _, atom in recipe['atoms']:
                        if atom not in text:
                            raise ValueError('recipe content absent from prompt')
            previews[model + '__' + arm] = {'path': str(preview_path), 'sha256': sha(preview_path), 'prompt_sha256': payload['prompt_sha256']}
    for arm in ARMS:
        if previews[MODELS[0] + '__' + arm]['prompt_sha256'] != previews[MODELS[1] + '__' + arm]['prompt_sha256']:
            raise ValueError('provider prompts differ within an arm')
    qualification.update(previews=previews, first_prompts_frozen_at=now(), provider_calls=0, hls_calls=0)
    qualification['files'] += record_files([Path(p['path']) for p in previews.values()])
    save(dest, qualification)
    return qualification


class PreviewComplete(BaseException):
    pass


def reconcile(result, rows, expected_prompt):
    events = (result.get('llm_usage') or {}).get('events') or []
    flows = result.get('tool_flows') or {}
    counts = flows.get('counts') or {}
    issues = []
    if len(events) != len(rows):
        issues.append('provider_reservation_count_mismatch')
    for event, row in zip(events, rows):
        if row['status'] != 'settled' or row.get('overrun'):
            issues.append('unsettled_or_overrun_provider_request')
        for key in ['input_tokens', 'output_tokens', 'cached_tokens', 'prompt_sha256']:
            if event.get(key, 0) != row.get(key, 0):
                issues.append('provider_' + key + '_mismatch')
    if rows and rows[0]['prompt_sha256'] != expected_prompt:
        issues.append('first_prompt_not_frozen_preview')
    if len(events) > 5 or counts.get('csynth', 0) > 5:
        issues.append('registered_search_cap_exceeded')
    if counts.get('csynth', 0) != (result.get('synthesis_evaluations') or {}).get('count', 0):
        issues.append('selection_synthesis_count_mismatch')
    if not (result.get('candidate_ledger') or {}).get('complete'):
        issues.append('candidate_ledger_incomplete')
    selected = result.get('cosim') or {}
    suite = selected.get('validation_suite') or {}
    if counts.get('rtl_cosim', 0) != suite.get('executed_cases', 0):
        issues.append('rtl_case_flow_count_mismatch')
    if selected.get('passed'):
        if suite.get('executed_cases') != 5 or not suite.get('all_cases_passed'):
            issues.append('passing_selection_missing_five_case_evidence')
        if not result.get('rtl_executed_pass') or result.get('selected_code_sha256') != result.get('cosim_target_code_sha256'):
            issues.append('selected_source_rtl_identity_mismatch')
    for event in flows.get('events', []):
        if event['kind'] == 'rtl_cosim' and event.get('code_sha256') != result.get('selected_code_sha256'):
            issues.append('rtl_flow_source_mismatch')
    if not result.get('reference_isolation_audit', {}).get('passed'):
        issues.append('reference_audit_failed')
    if result.get('exception_type') or any(step.get('exception_type') for step in result.get('steps', [])):
        issues.append('unhandled_execution_failure')
    return dict(complete=not issues, issues=issues, provider_requests=len(rows),
                estimated_usd=sum(r['charged_or_reserved_usd'] for r in rows),
                selection_syntheses=counts.get('csynth', 0), rtl_resyntheses=counts.get('rtl_cosim', 0),
                all_tool_flows=flows.get('total_flow_invocations', 0))


def cell(output, identity, preview=False):
    config = read_config(output)
    spec = next(c for c in config['cells'] if c['cell_id'] == identity)
    qpath = qualification_path(output, spec['benchmark'])
    qualification = json.loads(qpath.read_text())
    verify_files(qualification['files'])
    if qualification['campaign_manifest_sha256'] != sha(output / 'manifest.json'):
        raise ValueError('qualification belongs to a different campaign')
    if not preview:
        parent_state = next(p for p in json.loads((output / 'state.json').read_text())['parents'] if p['benchmark'] == spec['benchmark'])
        if parent_state.get('qualification_sha256') != sha(qpath):
            raise ValueError('qualification changed after controller registration')
        if not qualification.get('first_prompts_frozen_at'):
            raise ValueError('prompts have not passed the offline gate')
    directory = (qpath.parent / 'preview_runs' if preview else output / 'cells') / identity
    directory.mkdir(parents=True, exist_ok=False)
    setup_cell_environment.base = base_environment(config, identity + ('__preview' if preview else ''))
    setup_cell_environment(config, spec, qualification, preview=preview)
    import c2hls
    from evaluation_repro import build_run_fingerprint, attach_run_provenance
    from experiment_telemetry import capture_tool_calls, tool_summary, unified_candidate_ledger
    from validation_suite import use_validation_suite
    from reference_isolation import audit_history_file
    if preview:
        def forbidden(*args, **kwargs):
            raise AssertionError('Vitis is forbidden during prompt preview')
        c2hls.run_hls_synthesis = c2hls.run_csim = c2hls.run_cosim = forbidden
        def capture(self, messages, model=None, max_tokens=None, agent_name='orchestrator'):
            from sigmetrics_main_policy import check_request_allowance
            check_request_allowance(self, messages, request_cap=config['per_request_input_token_bound_cap'], aggregate_cap=config['aggregate_input_token_cap'])
            save(qpath.parent / f'{spec["model"]}__{spec["arm"]}.json', dict(messages=messages, prompt_sha256=prompt_digest(messages), provider_calls=0, hls_calls=0))
            raise PreviewComplete()
        c2hls.C2HLSOrchestrator._call_llm_with_model = capture
    else:
        expected_prompt = qualification['previews'][spec['model'] + '__' + spec['arm']]['prompt_sha256']
        actual = c2hls.C2HLSOrchestrator._call_llm_with_model
        sent = []
        def checked_send(self, messages, *args, **kwargs):
            from sigmetrics_main_policy import check_request_allowance, SearchAllowanceExhausted
            try:
                check_request_allowance(self, messages, request_cap=config['per_request_input_token_bound_cap'], aggregate_cap=config['aggregate_input_token_cap'])
            except SearchAllowanceExhausted as exc:
                save(directory / 'search_stop.json', {'reason': str(exc), 'stopped_at': now(), 'provider_requests_before_stop': len(getattr(self, 'llm_usage_events', [])), 'next_request_sent': False})
                raise
            digest = prompt_digest(messages)
            if not sent and digest != expected_prompt:
                raise ValueError('first prompt differs from frozen preview before network call')
            sent.append(digest)
            return actual(self, messages, *args, **kwargs)
        c2hls.C2HLSOrchestrator._call_llm_with_model = checked_send
    entry = next(e for e in config['benchmark_entries'] if e['benchmark'] == spec['benchmark'])
    profile = {'name': 'sigmetrics2027_public_recipe_main', 'reference_blind': True}
    fingerprint = build_run_fingerprint(repo=ROOT, benchmark_dir=Path(entry['benchmark_dir']), benchmark=spec['benchmark'],
        model_id=spec['model'], model_label=spec['model'], skill_mode=spec['arm'], steps=['flash'], profile=profile)
    save(directory / 'preflight.json', dict(fingerprint=fingerprint, manifest_sha256=sha(output / 'manifest.json'), qualification_sha256=sha(qpath), spec=spec))
    started = time.monotonic()
    with capture_tool_calls() as events:
        try:
            with use_validation_suite(config['inputs'], spec['benchmark']):
                result = c2hls.run_benchmark_multistep(entry['benchmark_dir'], str(directory), gpt_model=spec['model'], steps=['flash'], turns_limitation=5, quality_repair_turns=0)
        except PreviewComplete:
            return 0
        except Exception as exc:
            if preview:
                raise
            result = dict(success=False, exception_type=type(exc).__name__, phase='unhandled_engineering_failure')
    if preview:
        raise ValueError('preview did not reach the provider boundary')
    result['tool_flows'] = tool_summary(events)
    result['candidate_ledger'] = unified_candidate_ledger(result)
    history = directory / (spec['benchmark'] + '_history.json')
    audit = audit_history_file(history, benchmark_dir=Path(entry['benchmark_dir']), reference_data=result.get('reference_validation'), controller_data=result)
    attach_run_provenance(result, fingerprint=fingerprint, profile=profile, elapsed_seconds=time.monotonic() - started, history_path=history, reference_audit=audit)
    ledger = json.loads(Path(config['api_budget']).read_text())
    rows = [r for r in ledger['reservations'] if r['run_id'] == os.environ['C2HLS_SAMPLE_ID']]
    result['accounting_reconciliation'] = reconcile(result, rows, expected_prompt)
    if (directory / 'search_stop.json').exists():
        result['search_allowance_stop'] = json.loads((directory / 'search_stop.json').read_text())
    result['registration'] = dict(spec=spec, manifest_sha256=sha(output / 'manifest.json'), qualification_sha256=sha(qpath))
    save(directory / 'result.json', result)
    print(json.dumps({'cell': identity, 'success': result.get('success'), 'accounting': result['accounting_reconciliation']}), flush=True)
    return 0 if result['accounting_reconciliation']['complete'] else 2


def start_ticks(pid):
    return Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[19]


def execute(command, cwd, log, env, timeout):
    with Path(log).open('x') as stream:
        process = subprocess.Popen(command, cwd=cwd, stdout=stream, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        try:
            return {'returncode': process.wait(timeout=timeout), 'timed_out': False}
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            return {'returncode': 124, 'timed_out': True}


def resource_gate(config):
    gib = 1024 ** 3
    memory = {row.split(':')[0]: int(row.split()[1]) * 1024 for row in Path('/proc/meminfo').read_text().splitlines()}
    observed = {'scratch_free_gib': shutil.disk_usage('/tmp').free / gib, 'memory_available_gib': memory['MemAvailable'] / gib,
                'load_per_cpu': os.getloadavg()[0] / (os.cpu_count() or 1)}
    limits = config['resource_thresholds']
    reasons = [name for name in ('scratch_free_gib', 'memory_available_gib') if observed[name] < limits[name]]
    if observed['load_per_cpu'] > limits['load_per_cpu']:
        reasons.append('load_per_cpu')
    return {'allowed': not reasons, 'reasons': reasons, 'observed': observed, 'thresholds': limits}


def summarize(state):
    return {'status': state['status'], 'updated_at': state['updated_at'], 'registered_cells': len(state['cells']),
            'parents': dict(collections.Counter(p['status'] for p in state['parents'])),
            'cells': dict(collections.Counter(c['status'] for c in state['cells'])),
            'resource_gate': state.get('resource_gate'), 'stop_requested': state.get('stop_requested')}


def run(output):
    import fcntl
    lock = (output / 'controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    config = read_config(output)
    state = json.loads((output / 'state.json').read_text())
    if state['manifest_sha256'] != sha(output / 'manifest.json') or state['status'] != 'prepared':
        raise ValueError('campaign already started or registration changed; no implicit rerolls')
    save(output / 'controller_identity.json', dict(pid=os.getpid(), start_ticks=start_ticks(os.getpid()),
         manifest_sha256=sha(output / 'manifest.json'), source_runner_sha256=sha(Path(config['source']) / SCRIPT), registered_at=now()))
    stop_request = {}
    def request_stop(signum, frame):
        stop_request.update(signal=signum, received_at=now(), policy='stop submissions and drain active jobs')
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    state.update(status='running', pid=os.getpid(), started_at=now())
    save(output / 'state.json', state)
    source = Path(config['source'])
    def launch_parent(parent):
        benchmark = parent['benchmark']
        if benchmark not in config['parent_reuse']:
            command = [sys.executable, str(source / 'scripts/build_sigmetrics_parent.py'), '--inputs', config['inputs'], '--benchmark', benchmark, '--output', parent['path']]
            result = execute(command, source, output / (benchmark + '.parent.log'), base_environment(config, benchmark + '__parent'), config['parent_job_timeout_seconds'])
            if result['returncode']:
                return {**result, 'kind': 'parent_blocked' if result['timed_out'] or (Path(parent['path']) / 'measurement.json').exists() else 'engineering_failure'}
        command = [sys.executable, str(source / SCRIPT), 'qualify', '--output', str(output), '--benchmark', benchmark]
        result = execute(command, source, output / (benchmark + '.qualification.log'), base_environment(config, benchmark + '__qualification'), 600)
        return {**result, 'kind': 'ready' if result['returncode'] == 0 else 'parent_blocked' if result['returncode'] == 4 else 'engineering_failure'}
    def launch_cell(spec):
        verify_files(config['files'])
        command = [sys.executable, str(source / SCRIPT), 'cell', '--output', str(output), '--cell-id', spec['cell_id']]
        return execute(command, source, output / (spec['cell_id'] + '.log'), base_environment(config, spec['cell_id']), config['cell_job_timeout_seconds'])
    # Qualify the already measured ATAX parent first, without consuming a Vitis slot.
    reused = next(p for p in state['parents'] if p['benchmark'] == 'hlsfactory_atax')
    if reused.get('status') == 'ready':
        qpath = qualification_path(output, reused['benchmark'])
        if reused.get('qualification_sha256') != sha(qpath):
            raise ValueError('prepared ATAX qualification changed')
        verify_files(json.loads(qpath.read_text())['files'])
    else:
        raise ValueError('ATAX prerequisite was not qualified during preparation')
    active = {}
    stopped = False
    with ThreadPoolExecutor(max_workers=2) as pool:
        while True:
            if stop_request:
                stopped = True
                state['stop_requested'] = dict(stop_request)
            gate = resource_gate(config)
            state['resource_gate'] = gate
            can_submit = not stopped and gate['allowed']
            pending_parents = [p for p in state['parents'] if p['status'] == 'pending']
            ready = {p['benchmark'] for p in state['parents'] if p['status'] == 'ready'}
            pending_cells = [c for c in state['cells'] if c['status'] == 'pending' and c['benchmark'] in ready]
            active_kinds = [kind for kind, _ in active.values()]
            if can_submit and len(active) < 2 and pending_parents and 'parent' not in active_kinds:
                parent = pending_parents[0]
                parent.update(status='running', started_at=now())
                active[pool.submit(launch_parent, parent)] = ('parent', parent)
            while can_submit and len(active) < 2 and pending_cells:
                spec = pending_cells.pop(0)
                spec.update(status='running', started_at=now())
                active[pool.submit(launch_cell, spec)] = ('cell', spec)
            state.update(updated_at=now())
            save(output / 'state.json', state)
            save(output / 'summary.json', summarize(state))
            if not active:
                if not stopped and not gate['allowed'] and (pending_parents or pending_cells):
                    time.sleep(10)
                    continue
                break
            finished, _ = wait(active, timeout=10, return_when=FIRST_COMPLETED)
            for future in finished:
                kind, spec = active.pop(future)
                try:
                    answer = future.result()
                except Exception as exc:
                    answer = {'returncode': 3, 'kind': 'engineering_failure', 'exception_type': type(exc).__name__}
                spec.update(finished_at=now(), execution=answer)
                if kind == 'parent':
                    if answer['kind'] == 'ready':
                        spec.update(status='ready', qualification_sha256=sha(qualification_path(output, spec['benchmark'])))
                    elif answer['kind'] == 'parent_blocked':
                        spec['status'] = 'failed_validation'
                        for candidate in state['cells']:
                            if candidate['benchmark'] == spec['benchmark']:
                                candidate.update(status='blocked_parent', blocked_at=now(), reason='parent_validation_failed_or_frozen_router_has_no_match')
                    else:
                        spec['status'] = 'engineering_failure'
                        stopped = True
                else:
                    spec['status'] = 'complete' if answer['returncode'] == 0 else 'accounting_or_execution_failure'
                    if answer['returncode']:
                        stopped = True
            state.update(updated_at=now())
            save(output / 'state.json', state)
    state.update(status='drained_on_request' if stop_request else 'stopped_on_failure' if stopped else 'complete', finished_at=now(), updated_at=now())
    save(output / 'state.json', state)
    save(output / 'summary.json', summarize(state))
    return 2 if stopped else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'run', 'qualify', 'cell', 'preview', 'status'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cell-id')
    parser.add_argument('--benchmark')
    args = parser.parse_args()
    output = args.output.resolve()
    if args.action == 'prepare':
        prepare(output)
    elif args.action == 'run':
        return run(output)
    elif args.action == 'qualify':
        try:
            qualify(output, args.benchmark)
        except ProtocolBlocked as exc:
            save(output / 'qualification' / args.benchmark / 'blocked.json', {'reason': str(exc), 'recorded_at': now(), 'provider_calls': 0})
            return 4
    elif args.action in ('cell', 'preview'):
        return cell(output, args.cell_id, preview=args.action == 'preview')
    else:
        print(json.dumps(summarize(json.loads((output / 'state.json').read_text())), indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
