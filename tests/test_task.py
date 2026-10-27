"""Plain-source preparation uses compiler facts and native executions only."""
import json
from pathlib import Path

import pytest

from hlsmith_minimal import task
from hlsmith_minimal.tools import validate_native


def prepared(tmp_path, code, *, top='kernel', **kwargs):
    source = tmp_path/'original.c'
    source.write_text(code)
    path = task.prepare(source, tmp_path/'prepared', top=top, **kwargs)
    return source, json.loads(path.read_text())


def check(tmp_path, source, config, candidate):
    path = tmp_path/'candidate.cpp'; path.write_text(candidate)
    includes = {**config['headers'], **config['test_files']}
    headers = list(config['headers'])
    if 'hlsmith_task.h' in headers:
        headers.remove('hlsmith_task.h'); headers.insert(0,'hlsmith_task.h')
    return validate_native(source,path,config['testbench'],out=tmp_path/'check',
                           include_files=includes,interface_headers=headers)


def test_typedef_scalar_harness_executes_actual_kernel_and_rejects_wrong_result(tmp_path):
    source, config = prepared(tmp_path, '#include <stdint.h>\ntypedef uint32_t word; word kernel(word x){return x+3u;}')
    assert config['testbench_provenance']['cases'] == 3
    assert 'unsigned int kernel(unsigned int x)' in config['testbench_provenance']['prototype']
    candidate = '#include "hlsmith_task.h"\nunsigned int kernel(unsigned int x){return x+4u;}'
    result = check(tmp_path,source,config,candidate)
    assert not result['passed'] and result['comparison']['reason'] == 'output value differs'
    assert len(result['reference']['stdout'].splitlines()) == 3


def test_full_pointer_extent_including_last_element_is_observed(tmp_path):
    source,config = prepared(tmp_path,'void kernel(int *a){for(int i=0;i<8;++i)a[i]+=2;}',sizes=['a=8'])
    tb=Path(config['testbench']).read_text()
    assert 'm_param_alloc(0, sizeof(int[8]))' in tb
    result=check(tmp_path,source,config,'#include "hlsmith_task.h"\nvoid kernel(int *a){for(int i=0;i<7;++i)a[i]+=2;}')
    assert not result['passed'] and result['comparison']['reason']=='output value differs'
    assert len(result['reference']['stdout'].split())==24


def test_multidimensional_pointer_preserves_rows_and_compares_all_cells(tmp_path):
    source,config=prepared(tmp_path,'void kernel(double a[2][3]){for(int i=0;i<2;++i)for(int j=0;j<3;++j)a[i][j]+=0.5;}',sizes=['a=2x3'])
    assert 'double (*a)[3]' in config['testbench_provenance']['prototype']
    assert 'm_param_alloc(0, sizeof(double[2][3]))' in Path(config['testbench']).read_text()
    result=check(tmp_path,source,config,'#include "hlsmith_task.h"\nvoid kernel(double a[2][3]){for(int i=0;i<2;++i)for(int j=0;j<3;++j)a[i][j]+=0.5;}')
    assert result['passed'],result
    assert len(result['reference']['stdout'].split())==18


def test_fixed_scalar_loop_domain_and_array_typedef(tmp_path):
    source,config=prepared(tmp_path,'typedef int row[3]; void kernel(row *a,int n){for(int i=0;i<n;++i)for(int j=0;j<3;++j)a[i][j]+=1;}',sizes=['a=2x3'],values=['n=2'])
    result=check(tmp_path,source,config,'#include "hlsmith_task.h"\nvoid kernel(int (*a)[3],int n){for(int i=0;i<n;++i)for(int j=0;j<3;++j)a[i][j]+=1;}')
    assert result['passed'],result
    assert config['testbench_provenance']['fixed_values']=={'n':'2'}


def test_provided_testbench_needs_no_clang_and_keeps_exclusive_dependencies_private(tmp_path,monkeypatch):
    public=tmp_path/'api.h'; public.write_text('int kernel(int);\n')
    private=tmp_path/'vectors.h';private.write_text('#define PRIVATE_INPUT 13\n')
    tb=tmp_path/'tb.c';tb.write_text('#include "api.h"\n#include "vectors.h"\n#include <stdio.h>\nint main(void){printf("%d\\n",kernel(PRIVATE_INPUT));return 0;}')
    monkeypatch.setattr(task,'_signature',lambda *args:pytest.fail('provided TB must not invoke Clang'))
    source,config=prepared(tmp_path,'#include "api.h"\nint kernel(int x){return x+1;}',testbench=tb)
    assert Path(config['testbench']).read_bytes()==tb.read_bytes()
    assert set(config['headers'])=={'hlsmith_task.h','api.h'} and set(config['test_files'])=={'vectors.h'}
    assert config['testbench_provenance']['kind']=='provided'
    assert config['testbench_provenance']['version']=='provided-v1'
    assert 'compiler' not in config['testbench_provenance']
    result=check(tmp_path,source,config,'extern "C" {\n#include "api.h"\n}\nint kernel(int x){return x+1;}')
    assert result['passed'],result


def test_provided_extent_wrapper_preserves_tb_bytes_and_automatic_checking(tmp_path):
    tb=tmp_path/'tb.c';raw='#include <stdio.h>\nvoid kernel(int *a);\nint main(void){int a[4]={1,2,3,4};kernel(a);printf("%d\\n",a[3]);return 0;}'
    tb.write_text(raw)
    source,config=prepared(tmp_path,'void kernel(int *a){a[3]+=2;}',testbench=tb,sizes=['a=4'])
    assert tb.read_text()==raw and Path(config['test_files']['hlsmith_original_testbench.inc']).read_text()==raw
    wrapper=Path(config['testbench']).read_text()
    assert 'm_param_alloc(0, sizeof(int[4]))' in wrapper
    assert 'CUSTOM_VERIFICATION' not in wrapper and 'tb-param-size' not in wrapper
    assert config['testbench_provenance']['kind']=='provided_with_extents'
    result=check(tmp_path,source,config,'#include "hlsmith_task.h"\nvoid kernel(int *a){a[3]+=2;}')
    assert result['passed'],result


@pytest.mark.parametrize('code,options',[
    ('void kernel(int *a){}',{}),
    ('void kernel(int a[2][3]){}',{'sizes':['a=2x4']}),
    ('void kernel(int **a){}',{'sizes':['a=2x3']}),
    ('static int kernel(int x){return x;}',{}),
    ('int kernel(int x,...){return x;}',{}),
    ('struct S{int x;};struct S kernel(int x){struct S s={x};return s;}',{}),
    ('void kernel(const int *a){}',{'sizes':['a=4']}),
    ('void kernel(int n){}',{}),
    ('int state; int kernel(int x){state+=x;return state;}',{}),
    ('int kernel(int x){static int state;state+=x;return state;}',{}),
    ('int helper(void){static int state;return ++state;} int kernel(int x){return helper()+x;}',{}),
    ('int kernel(unsigned int n){return n;}',{'values':['n=-1']}),
    ('int kernel(int n){return n;}',{'sizes':['n=4']}),
    ('int kernel(int n){return n;}',{'values':['other=2']}),
    ('void kernel(int *a){}',{'sizes':['a=999999']}),
])
def test_unknown_or_unsupported_contracts_stop_before_run(tmp_path,code,options):
    with pytest.raises(ValueError):
        prepared(tmp_path,code,**options)


def test_config_paths_resolve_at_config_location_and_task_identity_cannot_be_overridden(tmp_path):
    source=tmp_path/'kernel.c';source.write_text('int kernel(int x){return x;}')
    settings=tmp_path/'settings';settings.mkdir()
    (settings/'private.dat').write_text('private')
    config=settings/'options.json'
    config.write_text(json.dumps({'bambu':{'executable':'../bin/bambu'},'lightningsim':{'executable':'tools/lightningsim','hls_root':'hls'},'test_files':{'input.dat':'private.dat'}}))
    path=task.prepare(source,tmp_path/'prepared',top='kernel',config=config)
    result=json.loads(path.read_text())
    assert result['bambu']['executable']==str(tmp_path/'bin/bambu')
    assert result['lightningsim']['hls_root']==str(settings/'hls')
    assert result['test_files']['input.dat']==str(settings/'private.dat')
    config.write_text(json.dumps({'source':'other.c'}))
    with pytest.raises(ValueError,match='settings only'):
        task.prepare(source,tmp_path/'rejected',top='kernel',config=config)


def test_generated_public_header_handles_ordinary_c_header_without_cpp_linkage(tmp_path):
    (tmp_path/'api.h').write_text('#ifndef API_H\n#define API_H\ntypedef unsigned int word;\nword kernel(word);\n#endif\n')
    source,config=prepared(tmp_path,'#include "api.h"\nword kernel(word x){return x+1u;}')
    assert 'api.h' in config['headers']
    result=check(tmp_path,source,config,'#include "hlsmith_task.h"\nunsigned int kernel(unsigned int x){return x+1u;}')
    assert result['passed'],result


def test_preparation_is_deterministic_and_refuses_old_output(tmp_path):
    source=tmp_path/'kernel.c';source.write_text('int kernel(int x){return x+1;}')
    paths=[task.prepare(source,tmp_path/name,top='kernel') for name in ('one','two')]
    configs=[json.loads(path.read_text()) for path in paths]
    assert Path(configs[0]['testbench']).read_bytes()==Path(configs[1]['testbench']).read_bytes()
    assert configs[0]['testbench_provenance']['testbench_sha256']==configs[1]['testbench_provenance']['testbench_sha256']
    with pytest.raises(ValueError,match='empty'):
        task.prepare(source,tmp_path/'one',top='kernel')


@pytest.mark.parametrize('provided',[False,True])
def test_mdpi_branch_compiles_and_registers_whole_array_without_disabling_comparison(tmp_path,provided):
    tb=None
    if provided:
        tb=tmp_path/'original_tb.c'
        tb.write_text('#include <stdio.h>\nvoid kernel(int *a);\nint main(void){int a[4]={1,2,3,4};kernel(a);printf("%d\\n",a[3]);return 0;}')
    source,config=prepared(tmp_path,'void kernel(int *a){a[3]+=2;}',testbench=tb,sizes=['a=4'])
    instrumented=tmp_path/'instrumented.c'
    instrumented.write_text('#define __BAMBU_SIM__ 1\n'+Path(config['testbench']).read_text())
    config['testbench']=str(instrumented)
    stub=tmp_path/'stub.h'
    stub.write_text('#include <stdint.h>\n#include <stddef.h>\n#include <stdlib.h>\nstatic void m_param_alloc(uint8_t i,size_t n){if(i!=0||n!=sizeof(int[4]))abort();}\n')
    config['test_files']['mdpi/mdpi_user.h']=str(stub)
    result=check(tmp_path,source,config,'#include "hlsmith_task.h"\nvoid kernel(int *a){a[3]+=2;}')
    assert result['passed'],result


def test_large_fixed_integer_does_not_round_through_a_floating_literal(tmp_path):
    source,config=prepared(tmp_path,'unsigned long long kernel(unsigned long long x){return x;}',values=['x=9007199254740993'])
    result=check(tmp_path,source,config,'#include "hlsmith_task.h"\nunsigned long long kernel(unsigned long long x){return x;}')
    assert result['passed']
    assert result['reference']['stdout'].split()==['9007199254740993']*3
