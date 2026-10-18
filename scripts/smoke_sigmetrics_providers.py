#!/usr/bin/env python3
"""Tiny budgeted compatibility checks; never emit credentials or raw exceptions."""
import json
import os
import sys
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiment_budget import initialize


def main():
    out=ROOT/'artifacts/sigmetrics2027';ledger=out/'api_budget.json';result_path=out/'provider_generation_smoke.json'
    if result_path.exists():raise ValueError('provider smoke already recorded')
    if not ledger.exists():
        initialize(ledger,limit_usd=300.0,prices={'deepseek-flash':{'input':0.3,'cached_input':0.006,'output':1.2},
                   'gpt-5.6-luna':{'input':0.2,'cached_input':0.02,'output':1.2}})
        (out/'price_assumptions.json').write_text(json.dumps({
            'verified_at':datetime.now(timezone.utc).isoformat(),
            'sources':['https://api-docs.deepseek.com/quick_start/pricing','https://developers.openai.com/api/docs/models/gpt-5.6-luna'],
            'policy':'DeepSeek peak rate is an upper bound; GPT requests stay below the long-context threshold; no web/tools/cache-write requests.',
            'unit':'USD per million tokens; estimated cost, not invoice'},indent=2)+'\n')
    os.environ.update(C2HLS_API_BUDGET_LEDGER=str(ledger),C2HLS_AGGREGATE_INPUT_TOKEN_BUDGET='100000',
        C2HLS_LLM_TEMPERATURE='0',C2HLS_LLM_TOP_P='1',C2HLS_DEEPSEEK_THINKING='disabled',
        C2HLS_DEEPSEEK_REASONING_EFFORT='high',C2HLS_OPENAI_REASONING_EFFORT='none',C2HLS_LLM_TIMEOUT='120')
    config=json.loads(Path('/home/luo00466/code_translation-c2hls-hpca2027/artifacts/experiment_matrix/deepseek_flash_reviewer_ab27_20260925.json').read_text())
    # Load provider-specific files directly, so a inherited key cannot select the wrong account.
    keyfiles={'deepseek-flash':Path(config['key_file']),'gpt-5.6-luna':Path('/home/luo00466/gpt-key.txt')}
    import c2hls
    records=[]
    for model in keyfiles:
        os.environ['OPENAI_API_KEY']=keyfiles[model].read_text().strip()
        os.environ['OPENAI_BASE_URL']=config['endpoint'] if model=='deepseek-flash' else 'https://api.openai.com/v1'
        os.environ['C2HLS_SAMPLE_ID']='compatibility_'+model
        orch=c2hls.C2HLSOrchestrator(gpt_model=model,quality_repair_turns=0)
        row={'model':model,'purpose':'API compatibility, not experiment sample'}
        try:
            output=orch._call_llm_with_model([{'role':'user','content':'Reply with only the word READY.'}],max_tokens=32,agent_name='compatibility')
            row.update(success=bool(output),output=output,llm_usage=orch._llm_usage_summary())
        except Exception as exc:
            row.update(success=False,exception_type=type(exc).__name__)
        records.append(row)
        result_path.write_text(json.dumps({'records':records},indent=2)+'\n')
        print(json.dumps({'model':model,'success':row['success'],'exception_type':row.get('exception_type')}),flush=True)
    return 0 if all(x['success'] for x in records) else 1

if __name__=='__main__':raise SystemExit(main())
