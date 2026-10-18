#!/usr/bin/env python3
"""Six-cell engineering pilot; one call/synthesis per cell, no efficacy claim.

prepare freezes code, inputs, parent and recipe selection before generation.
run launches at most two isolated workers. Completed or failed cells are never
rerolled. An interrupted cell requires a new explicitly identified experiment.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime, timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def now():return datetime.now(timezone.utc).isoformat()
def save(path,data):
    path=Path(path);temp=path.with_name(path.name+'.tmp');temp.write_text(json.dumps(data,indent=2)+'\n');temp.replace(path)


def reconcile(result,reservations):
    usage=result.get('llm_usage') or {};events=usage.get('events') or []
    tools=result.get('tool_flows') or {};counts=tools.get('counts') or {};issues=[]
    if len(events)!=len(reservations):issues.append('provider_reservation_count_mismatch')
    for event,row in zip(events,reservations):
        if row['status']!='settled':issues.append('unsettled_provider_request')
        for key in ['input_tokens','output_tokens','cached_tokens','prompt_sha256']:
            if event.get(key,0)!=row.get(key,0):issues.append('provider_'+key+'_mismatch')
    if counts.get('csynth',0)!=(result.get('synthesis_evaluations') or {}).get('count',0):
        issues.append('selection_synthesis_count_mismatch')
    if len(events)>1 or counts.get('csynth',0)>1:issues.append('engineering_pilot_cap_exceeded')
    if not (result.get('candidate_ledger') or {}).get('complete'):issues.append('candidate_ledger_incomplete')
    selected=result.get('cosim') or {}
    suite=selected.get('validation_suite') or {}
    if counts.get('rtl_cosim',0)!=suite.get('executed_cases',0):
        issues.append('rtl_case_flow_count_mismatch')
    if selected.get('passed'):
        if suite.get('executed_cases')!=5 or not suite.get('all_cases_passed'):
            issues.append('passing_selection_missing_five_case_evidence')
        if not result.get('rtl_executed_pass') or result.get('selected_code_sha256')!=result.get('cosim_target_code_sha256'):
            issues.append('selected_source_rtl_identity_mismatch')
    for event in tools.get('events',[]):
        if event['kind']=='rtl_cosim' and event.get('code_sha256')!=result.get('selected_code_sha256'):
            issues.append('rtl_flow_source_mismatch')
    if not result.get('reference_isolation_audit',{}).get('passed'):issues.append('reference_audit_failed')
    return {'complete':not issues,'issues':issues,'provider_requests':len(reservations),
            'estimated_usd':sum(r['charged_or_reserved_usd'] for r in reservations),
            'selection_syntheses':counts.get('csynth',0),'rtl_resyntheses':counts.get('rtl_cosim',0),
            'all_tool_flows':tools.get('total_flow_invocations',0)}


def prepare(output):
    import c2hls
    from phase_b_manifest import load_phase_b_seed
    from skill_library import load_frozen_library,same_content_recipe_atoms,render_same_content_recipe
    from smart_skill_router import route_smart_skills
    from validation_suite import load_suite
    if output.exists():raise ValueError('pilot already prepared; refusing overwrite')
    inputs=ROOT/'artifacts/sigmetrics2027/core_inputs_v1/manifest.json'
    parent=ROOT/'artifacts/sigmetrics2027/parent_pilot_20260928/atax/phase_b_manifest.json'
    benchmark='hlsfactory_atax';suite=load_suite(inputs,benchmark)
    entry=next(e for e in json.loads(inputs.read_text())['entries'] if e['benchmark']==benchmark)
    public=c2hls._load_benchmark_inputs(entry['benchmark_dir'])
    seed=load_phase_b_seed(parent,benchmark=benchmark,input_c=public['c_code'],header_code=public['header_code'],
                          expected_part='xcu280-fsvh2892-2L-e',expected_clock_ns=3.33,expected_vitis_version='2023.2')
    output.mkdir(parents=True);skills=output/'skills.json'
    shutil.copy2('/home/luo00466/code_translation-c2hls-hpca2027/skill_v3_no_rmw_fc27133/skills.json',skills)
    library=load_frozen_library(skills)
    route=route_smart_skills(library,scope='smart_exhaustive_v2',step_name='flash',current_code=seed['code'],
                            synth_report=seed['csynth_report'],vitis_version='2023.2',fpga='xcu280-fsvh2892-2L-e',max_skills=3,min_score=4.0)
    if not route.selected:raise ValueError('no recipe matches parent; preserve preflight, do not hand-pick after outcomes')
    recipes={'route':route.audit,'selection_source':'unchanged public-C parent only',
             'recipes':[{'id':sk.id,'atoms':same_content_recipe_atoms(sk),
                         'flat':render_same_content_recipe(sk,structured=False),
                         'structured':render_same_content_recipe(sk,structured=True)} for sk in route.selected]}
    save(output/'recipes.json',recipes)
    source=output/'source';source.mkdir()
    source_files=list(ROOT.glob('*.py'))+[Path(__file__).resolve()]+list((ROOT/'configs').glob('*.json'))
    frozen=[]
    for path in source_files:
        target=source/path.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target);frozen.append(target)
    frozen.extend([inputs,inputs.parent/'native_validation.json',parent,parent.parent/'parent.cpp',parent.parent/'measurement.json',skills,output/'recipes.json'])
    frozen.extend(p for p in Path(entry['benchmark_dir']).glob('*') if p.is_file())
    frozen.extend(Path(c['shape_registry_path']) for c in entry['cases'])
    cells=[]
    for model,arms in [('deepseek-flash',['none','flat','structured']),('gpt-5.6-luna',['structured','flat','none'])]:
        for arm in arms:cells.append({'cell_id':model+'_'+arm,'model':model,'arm':arm,'status':'pending'})
    config={'schema_version':'c2hls.sigmetrics-recipe-pilot.v1','prepared_at':now(),'purpose':'engineering and accounting only; not part of 216-cell efficacy matrix',
            'benchmark':benchmark,'benchmark_dir':entry['benchmark_dir'],'inputs':str(inputs),'parent_manifest':str(parent),
            'skills':str(skills),'recipe_ids':[s.id for s in route.selected],
            'api_budget':str(ROOT/'artifacts/sigmetrics2027/api_budget.json'),'source':str(source),
            'workers':2,'llm_call_cap':1,'selection_synthesis_cap':1,'candidate_attempts':1,
            'max_response_tokens':16384,'aggregate_input_token_cap':100000,
            'selection':'same feasible-parent retention policy; no reselection after failed RTL',
            'knowledge_control':'common generic HLS prompt in every arm; flat and structured contain identical full recipe statements',
            'files':[{'path':str(p),'sha256':sha(p)} for p in sorted(set(frozen))], 'cells':cells}
    save(output/'manifest.json',config);save(output/'state.json',{'status':'prepared','updated_at':now(),'manifest_sha256':sha(output/'manifest.json'),'cells':cells})
    print(json.dumps({'cells':len(cells),'recipes':config['recipe_ids'],'source_files':len(source_files)}))


def verify(config):
    for item in config['files']:
        if sha(item['path'])!=item['sha256']:raise ValueError('frozen input changed: '+item['path'])


def cell(output,identity):
    config=json.loads((output/'manifest.json').read_text());verify(config)
    spec=next(c for c in config['cells'] if c['cell_id']==identity);directory=output/'cells'/identity
    directory.mkdir(parents=True,exist_ok=False)
    for key in list(os.environ):
        if key.startswith('C2HLS_'):del os.environ[key]
    from evaluation_repro import REFERENCE_BLIND_OVERRIDES,build_run_fingerprint,attach_run_provenance
    os.environ.update(REFERENCE_BLIND_OVERRIDES)
    os.environ.update({'C2HLS_REFERENCE_VALIDATE_MODE':'public_only','C2HLS_COSIM_REQUIRED':'1',
        'C2HLS_PHASE_B_SEED_MANIFEST':config['parent_manifest'],'C2HLS_HLSFACTORY_SHAPE_REGISTRY':str(Path(config['inputs']).parent/'shapes_case0.json'),
        'C2HLS_SAMPLE_ID':'recipe_pilot_v1_'+identity,'C2HLS_API_BUDGET_LEDGER':config['api_budget'],
        'C2HLS_AGGREGATE_INPUT_TOKEN_BUDGET':'100000','C2HLS_MAX_COMPLETION_TOKENS':'16384',
        'C2HLS_LLM_CANDIDATE_BUDGET':'1','C2HLS_SYNTHESIS_EVAL_BUDGET':'1',
        'C2HLS_CANDIDATES_PER_STEP':'1','C2HLS_ATTEMPTS_PER_CANDIDATE':'1','C2HLS_EXHAUSTIVE_CANDIDATE_ATTEMPTS':'1',
        'C2HLS_STRATEGY':'flash','C2HLS_DYNAMIC_ROUTING':'0','C2HLS_QOR_DESIGN_SWEEP':'0','C2HLS_QOR_PROFILE_REVIEW':'0',
        'C2HLS_QUALITY_REPAIR_TURNS':'0','C2HLS_HW_EMU_FINAL':'0','C2HLS_SKILL_USAGE_DECLARATION':'0',
        'C2HLS_SKILL_LIBRARY_PATH':config['skills'],'C2HLS_LLM_TEMPERATURE':'0','C2HLS_LLM_TOP_P':'1',
        'C2HLS_LLM_SEED':'42','C2HLS_DEEPSEEK_THINKING':'disabled','C2HLS_OPENAI_REASONING_EFFORT':'none',
        'C2HLS_VITIS_VERSION':'2023.2','C2HLS_CSIM_TIMEOUT':'300','C2HLS_SYNTH_TIMEOUT':'1200','C2HLS_COSIM_TIMEOUT':'3600',
        'C2HLS_LLM_TIMEOUT':'300','C2HLS_TMP_ROOT':str(directory/'scratch'),'C2HLS_VITIS_USER_HOME':str(directory/'vitis_home'),
        'OMP_NUM_THREADS':'8'})
    if spec['arm']=='none':os.environ.update(C2HLS_SKILL_MODE='off',C2HLS_FORCE_SKILL_PROMPTS='0',C2HLS_SKILL_PROMPT_SCOPE='skillless')
    else:os.environ.update(C2HLS_SKILL_MODE='on',C2HLS_FORCE_SKILL_PROMPTS='1',C2HLS_SKILL_PROMPT_SCOPE='explicit',
            C2HLS_SKILL_EXPLICIT_IDS=','.join(config['recipe_ids']),C2HLS_SKILL_PROMPT_MODE='same_content_'+spec['arm'])
    if spec['model']=='deepseek-flash':
        old=json.loads(Path('/home/luo00466/code_translation-c2hls-hpca2027/artifacts/experiment_matrix/deepseek_flash_reviewer_ab27_20260925.json').read_text())
        os.environ['OPENAI_API_KEY']=Path(old['key_file']).read_text().strip();os.environ['OPENAI_BASE_URL']=old['endpoint']
    else:
        os.environ['OPENAI_API_KEY']=Path('/home/luo00466/gpt-key.txt').read_text().strip();os.environ['OPENAI_BASE_URL']='https://api.openai.com/v1'
    import c2hls
    from experiment_telemetry import capture_tool_calls,tool_summary,unified_candidate_ledger
    from validation_suite import use_validation_suite
    from reference_isolation import audit_history_file
    profile={'name':'sigmetrics2027_public_recipe_pilot','reference_blind':True}
    fingerprint=build_run_fingerprint(repo=ROOT,benchmark_dir=Path(config['benchmark_dir']),benchmark=config['benchmark'],
                                     model_id=spec['model'],model_label=spec['model'],skill_mode=spec['arm'],steps=['flash'],profile=profile)
    save(directory/'preflight.json',{'fingerprint':fingerprint,'manifest_sha256':sha(output/'manifest.json')})
    started=time.monotonic();result={}
    with capture_tool_calls() as events:
        try:
            with use_validation_suite(config['inputs'],config['benchmark']):
                result=c2hls.run_benchmark_multistep(config['benchmark_dir'],str(directory),gpt_model=spec['model'],steps=['flash'],turns_limitation=1,quality_repair_turns=0)
        except Exception as exc:
            result={'success':False,'exception_type':type(exc).__name__,'phase':'unhandled_engineering_failure'}
    result['tool_flows']=tool_summary(events);result['candidate_ledger']=unified_candidate_ledger(result)
    history=directory/(config['benchmark']+'_history.json')
    audit=audit_history_file(history,benchmark_dir=Path(config['benchmark_dir']),reference_data=result.get('reference_validation'),controller_data=result)
    attach_run_provenance(result,fingerprint=fingerprint,profile=profile,elapsed_seconds=time.monotonic()-started,history_path=history,reference_audit=audit)
    ledger=json.loads(Path(config['api_budget']).read_text());rows=[r for r in ledger['reservations'] if r['run_id']==os.environ['C2HLS_SAMPLE_ID']]
    result['accounting_reconciliation']=reconcile(result,rows)
    save(directory/'result.json',result)
    print(json.dumps({'cell':identity,'success':result.get('success'),'accounting':result['accounting_reconciliation']}),flush=True)
    return 0 if result['accounting_reconciliation']['complete'] else 2


def run(output):
    import fcntl
    lock=(output/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    config=json.loads((output/'manifest.json').read_text());verify(config)
    state=json.loads((output/'state.json').read_text())
    if state.get('manifest_sha256')!=sha(output/'manifest.json'):raise ValueError('pilot manifest changed after preparation')
    if state['status'] not in {'prepared'}:raise ValueError('pilot has already started; completed/failed/interrupted cells cannot be rerolled')
    state.update(status='running',pid=os.getpid(),started_at=now());save(output/'state.json',state)
    def launch(spec):
        verify(config);log=output/(spec['cell_id']+'.log')
        with log.open('x') as stream:
            return subprocess.run([sys.executable,str(Path(config['source'])/'scripts/run_sigmetrics_recipe_pilot.py'),'cell','--output',str(output),'--cell-id',spec['cell_id']],cwd=config['source'],stdout=stream,stderr=subprocess.STDOUT).returncode
    pending=list(state['cells']);active={};stopped=False
    with ThreadPoolExecutor(max_workers=2) as pool:
        while pending or active:
            while pending and len(active)<2 and not stopped:
                spec=pending.pop(0);spec.update(status='running',started_at=now());active[pool.submit(launch,spec)]=spec
            if not active:break
            finished,_=wait(active,timeout=10,return_when=FIRST_COMPLETED)
            for future in finished:
                spec=active.pop(future)
                try:code=future.result()
                except Exception:code=3
                spec.update(status='complete' if code==0 else 'accounting_or_execution_failure',returncode=code,finished_at=now())
                if code:stopped=True
            state.update(updated_at=now());save(output/'state.json',state)
    state.update(status='stopped_on_failure' if stopped else 'complete',finished_at=now());save(output/'state.json',state)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['prepare','run','cell']);parser.add_argument('--output',required=True,type=Path);parser.add_argument('--cell-id')
    args=parser.parse_args();output=args.output.resolve()
    if args.action=='prepare':prepare(output)
    elif args.action=='run':run(output)
    else:return cell(output,args.cell_id)
    return 0

if __name__=='__main__':raise SystemExit(main())
