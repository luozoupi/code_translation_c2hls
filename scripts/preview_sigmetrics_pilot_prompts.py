#!/usr/bin/env python3
"""Capture exact first-call pilot messages without calling any model or Vitis."""
import argparse,hashlib,importlib.util,json,sys
from pathlib import Path

class PreviewComplete(BaseException):pass

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--pilot',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args();args.pilot=args.pilot.resolve();args.output=args.output.resolve()
    config=json.loads((args.pilot/'manifest.json').read_text());source=Path(config['source']);sys.path.insert(0,str(source))
    spec=importlib.util.spec_from_file_location('frozen_pilot',source/'scripts/run_sigmetrics_recipe_pilot.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    import c2hls
    def forbidden(*a,**k):raise AssertionError('Vitis invocation forbidden in prompt preview')
    c2hls.run_hls_synthesis=c2hls.run_csim=c2hls.run_cosim=forbidden
    actual_run=c2hls.run_benchmark_multistep
    def diagnosed_run(*args,**kwargs):
        try:return actual_run(*args,**kwargs)
        except Exception:
            import traceback
            traceback.print_exc()
            raise
    c2hls.run_benchmark_multistep=diagnosed_run

    args.output.mkdir(parents=True,exist_ok=False);(args.output/'manifest.json').write_text(json.dumps(config,indent=2)+'\n')
    previews=[]
    for cell in config['cells']:
        destination=args.output/(cell['cell_id']+'_messages.json')
        def capture(self,messages,model=None,max_tokens=None,agent_name='orchestrator'):
            payload={'model':model or self.gpt_model,'destination':'https://api.deepseek.com/chat/completions' if cell['model']=='deepseek-flash' else 'https://api.openai.com/v1/chat/completions',
                     'max_tokens':max_tokens or self.max_completion_tokens,'agent':agent_name,'messages':messages,'provider_calls':0,'hls_calls':0}
            destination.write_text(json.dumps(payload,indent=2)+'\n');raise PreviewComplete()
        c2hls.C2HLSOrchestrator._call_llm_with_model=capture
        try:module.cell(args.output,cell['cell_id'])
        except PreviewComplete:pass
        else:raise ValueError('preview did not reach provider boundary')
        previews.append({'cell':cell['cell_id'],'path':str(destination),'sha256':hashlib.sha256(destination.read_bytes()).hexdigest()})
    (args.output/'index.json').write_text(json.dumps({'provider_calls':0,'hls_calls':0,'previews':previews},indent=2)+'\n')
    print(json.dumps({'previews':len(previews),'provider_calls':0,'hls_calls':0}))

if __name__=='__main__':main()
