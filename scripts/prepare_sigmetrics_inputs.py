#!/usr/bin/env python3
"""Freeze public-only inputs and five domain-valid test cases per core kernel."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
NAMES=['gemm','2mm','atax','bicg','jacobi-2d','seidel-2d','lu','trisolv','durbin','gramschmidt','nussinov','floyd-warshall']
SEEDS=[11,29,47,83]


def initialization_body(name,case):
    bodies={
        'gemm': '''*alpha=0.75+0.25*CASE; *beta=-0.5+0.25*CASE;
for(int i=0;i<ni;i++) for(int j=0;j<nj;j++) C[i][j]=sample_signed();
for(int i=0;i<ni;i++) for(int j=0;j<nk;j++) A[i][j]=sample_signed();
for(int i=0;i<nk;i++) for(int j=0;j<nj;j++) B[i][j]=sample_signed();''',
        '2mm': '''*alpha=0.75+0.25*CASE; *beta=-0.5+0.25*CASE;
for(int i=0;i<ni;i++) for(int j=0;j<nk;j++) A[i][j]=sample_signed();
for(int i=0;i<nk;i++) for(int j=0;j<nj;j++) B[i][j]=sample_signed();
for(int i=0;i<nj;i++) for(int j=0;j<nl;j++) C[i][j]=sample_signed();
for(int i=0;i<ni;i++) for(int j=0;j<nl;j++) D[i][j]=sample_signed();''',
        'atax': '''for(int i=0;i<n;i++) x[i]=sample_signed();
for(int i=0;i<m;i++) for(int j=0;j<n;j++) A[i][j]=sample_signed();''',
        'bicg': '''for(int i=0;i<m;i++) p[i]=sample_signed();
for(int i=0;i<n;i++) { r[i]=sample_signed(); for(int j=0;j<m;j++) A[i][j]=sample_signed(); }''',
        'jacobi-2d': '''for(int i=0;i<n;i++) for(int j=0;j<n;j++) { A[i][j]=sample_signed(); B[i][j]=sample_signed(); }''',
        'seidel-2d': '''for(int i=0;i<n;i++) for(int j=0;j<n;j++) A[i][j]=sample_signed();''',
        'lu': '''for(int i=0;i<n;i++) {
A[i][i]=n+1.0+0.1*sample_signed();
for(int j=0;j<i;j++) { double v=0.1*sample_signed(); A[i][j]=v; A[j][i]=v; }
}''',
        'trisolv': '''for(int i=0;i<n;i++) { x[i]=-999; b[i]=sample_signed();
for(int j=0;j<n;j++) L[i][j]=(j<i ? 0.1*sample_signed() : (j==i ? n+1.0 : 0.0)); }''',
        'durbin': '''double rho=(CASE%2 ? 1.0 : -1.0)*0.1*CASE; double value=rho;
for(int i=0;i<n;i++) { r[i]=value; value*=rho; }''',
        'gramschmidt': '''for(int i=0;i<m;i++) for(int j=0;j<n;j++) { A[i][j]=(i==j ? 2.0 : 0.0)+0.001*sample_signed(); Q[i][j]=0.0; }
for(int i=0;i<n;i++) for(int j=0;j<n;j++) R[i][j]=0.0;''',
        'nussinov': '''for(int i=0;i<n;i++) seq[i]=(char)(sample_uint()%4u);
for(int i=0;i<n;i++) for(int j=0;j<n;j++) table[i][j]=0;''',
        'floyd-warshall': '''for(int i=0;i<n;i++) for(int j=0;j<n;j++) {
unsigned v=sample_uint(); path[i][j]=(i==j ? 0 : (v%4u==0 ? 999 : (int)(1+v%20u))); }''',
    }
    return bodies[name].replace('CASE',str(case))


def replace_initializer(text,body,seed):
    match=re.search(r'\bvoid\s+init_array\s*\(',text)
    if not match:raise ValueError('no public init_array function')
    start=text.index('{',match.end());depth=1;end=start+1
    while depth:
        if text[end]=='{':depth+=1
        elif text[end]=='}':depth-=1
        end+=1
    helper=f'''// Prespecified SIGMETRICS input case; unsigned LCG is deterministic.
static unsigned input_state={seed}u;
static unsigned sample_uint() {{ input_state=1664525u*input_state+1013904223u; return input_state; }}
static double sample_signed() {{ return ((sample_uint()>>8)&65535u)/32767.5-1.0; }}
'''
    return text[:match.start()]+helper+text[match.start():start]+'{\n'+body+'\n}'+text[end:]


def prepare(source,output):
    if output.exists():raise ValueError('refusing to overwrite frozen inputs')
    registry=json.loads((ROOT/'configs/hlsfactory_output_shapes_fc27133.json').read_text())
    registries=[{**copy.deepcopy(registry),'benchmarks':{}} for _ in range(5)]
    entries=[]
    for name in NAMES:
        old_name='hlsfactory_'+name
        bench=source/old_name;meta=json.loads((bench/'metadata.json').read_text())
        new_name=old_name if name!='gramschmidt' else 'hlsfactory_gramschmidt_m60_n40'
        dst=output/new_name;dst.mkdir(parents=True)
        plain=(bench/meta.get('plain_c_file','plain.cpp')).read_text()
        header=(bench/meta['header_file']).read_text()
        testbench=(bench/meta['testbench_file']).read_text()
        contract=copy.deepcopy(registry['benchmarks'][old_name])
        if name=='gramschmidt':
            header,count=re.subn(r'(#define\s+N\s+)80\b',r'\g<1>40',header)
            if count!=1:raise ValueError('unexpected Gram-Schmidt shape')
            for spec in contract['outputs'].values():
                spec['shape']=[40 if n==80 else n for n in spec['shape']]
                spec['logical_shape']=[40 if n==80 else n for n in spec['logical_shape']]
            # The new variant uses a full-rank input even in its first case.
            testbench=replace_initializer(testbench,initialization_body(name,1),7)
        public_keys=['source_repo','plain_c_file','header_file','testbench_file','translated_hls_top','hls_top',
                     'supports_csim','supports_cosim','cosim_depths','part','clock_ns','support_files','include_dirs']
        public={k:meta[k] for k in public_keys if k in meta}
        public.update(benchmark=new_name,part='xcu280-fsvh2892-2L-e',clock_ns=3.33,
                      independent_golden_required=True,source_cohort='fc27133 public inputs',
                      validation_case_count=5)
        (dst/'metadata.json').write_text(json.dumps(public,indent=2)+'\n')
        (dst/'plain.cpp').write_text(plain)
        (dst/meta['header_file']).write_text(header)
        (dst/'testbench.cpp').write_text(testbench)
        cases=[]
        for case in range(5):
            # Start from the upstream TB for every replacement; helpers do not accumulate.
            tb=testbench if case==0 else replace_initializer((bench/meta['testbench_file']).read_text(),initialization_body(name,case),SEEDS[case-1])
            path=dst/f'testbench_case{case}.cpp';path.write_text(tb)
            bound=copy.deepcopy(contract);bound['testbench_sha256']=hashlib.sha256(tb.encode()).hexdigest()
            registries[case]['benchmarks'][new_name]=bound
            cases.append({'case_id':case,'seed':None if case==0 and name!='gramschmidt' else (7 if case==0 else SEEDS[case-1]),
                          'testbench_path':str(path),'testbench_sha256':bound['testbench_sha256'],
                          'shape_registry_path':str(output/f'shapes_case{case}.json')})
        entries.append({'benchmark':new_name,'original_benchmark':old_name,'benchmark_dir':str(dst),
                        'cases':cases,'domain_amendment':'M60_N40_full_column_rank' if name=='gramschmidt' else None,
                        'input_c_sha256':hashlib.sha256(plain.encode()).hexdigest(),
                        'header_sha256':hashlib.sha256(header.encode()).hexdigest()})
    for case,contract in enumerate(registries):
        (output/f'shapes_case{case}.json').write_text(json.dumps(contract,indent=2)+'\n')
    manifest={'schema_version':'c2hls.sigmetrics-inputs.v1','created_at':datetime.now(timezone.utc).isoformat(),
              'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'purpose':'prespecified domain-valid validation inputs, not proof of equivalence',
              'plain_source_changed':False,'reference_hls_files_copied':False,
              'gramschmidt_amendment':'Historical M=60,N=80 cannot have full column rank; new M=60,N=40 and well-conditioned inputs require a newly measured parent.',
              'entries':entries}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=Path('/home/luo00466/code_translation-c2hls-enhanced-fc27133/benchmarks'))
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();manifest=prepare(args.source.resolve(),args.output.resolve())
    print(json.dumps({'benchmarks':len(manifest['entries']),'native_cases':sum(len(x['cases']) for x in manifest['entries']),'output':str(args.output)}))


if __name__=='__main__':main()
