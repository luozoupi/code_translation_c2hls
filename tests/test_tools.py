"""Short local compiler/simulator tests; never require Vitis or hosted models."""
from pathlib import Path
import shutil
import sys

import pytest

from hlsmith_minimal.tools import compare_outputs, run_command, simulate_verilator, synthesize_vitis, validate_native


@pytest.fixture
def native_inputs(tmp_path):
    header = tmp_path / 'kernel.h'
    header.write_text('#ifdef __cplusplus\nextern "C" {\n#endif\nunsigned kernel(unsigned);\n#ifdef __cplusplus\n}\n#endif\n')
    reference = tmp_path / 'plain.c'
    candidate = tmp_path / 'generated.cpp'
    testbench = tmp_path / 'tb.c'
    reference.write_text('#include "kernel.h"\nunsigned kernel(unsigned x) {return x+1u;}\n')
    candidate.write_text(reference.read_text())
    testbench.write_text('#include <stdio.h>\n#include "kernel.h"\nint main(void) {printf("%u %u\\n", kernel(5), kernel(4294967295u)); return 0;}\n')
    return reference, candidate, testbench, {'kernel.h': header}


def test_original_c_and_cpp_candidate_use_same_testbench(native_inputs, tmp_path):
    reference, candidate, testbench, headers = native_inputs
    result = validate_native(reference, candidate, testbench, out=tmp_path/'run', include_files=headers)
    assert result['status'] == 'passed' and result['passed']
    assert result['reference']['stdout'] == result['candidate']['stdout'] == '6 0\n'
    assert result['input_hashes']['reference'] == result['input_hashes']['candidate']


def test_reference_preflight_accepts_c11_only_source_without_compiling_a_candidate(native_inputs, tmp_path, monkeypatch):
    from hlsmith_minimal import tools
    reference, _, testbench, headers = native_inputs
    reference.write_text('#include "kernel.h"\nunsigned kernel(unsigned x) {return x + _Generic(x, unsigned: 7u, default: 9u);}\n')
    actual_tool = tools._tool
    monkeypatch.setattr(tools, '_tool', lambda name: None if name == 'g++' else actual_tool(name))
    out = tmp_path/'preflight'
    result = validate_native(reference, testbench=testbench, out=out, include_files=headers,
                             interface_headers=['kernel.h'])
    assert result['passed'] and result['status'] == 'passed'
    assert result['reference']['stdout'] == '12 6\n'
    assert result['candidate'] == {} and result['comparison'] is None
    assert 'candidate' not in result['input_hashes']
    assert 'preflight' in result['qualification'] and 'no candidate comparison' in result['qualification']
    assert len(result['commands']) == 3
    assert '-std=c11' in result['commands'][0]['argv']
    assert '-std=c11' in result['commands'][1]['argv']
    assert not (out/'candidate').exists() and not (out/'testbench.o').exists()
    assert not Path(result['reference']['argv'][0]).exists()
    assert not list(out.rglob('reference.*'))


@pytest.mark.parametrize('failure', ['invalid_c', 'nonzero_exit', 'empty_output'])
def test_reference_preflight_rejects_invalid_original_or_unusable_testbench(native_inputs, tmp_path, failure):
    reference, _, testbench, headers = native_inputs
    if failure == 'invalid_c':
        reference.write_text('not valid C\n')
    else:
        testbench.write_text('int main(void) {return ' + ('1' if failure == 'nonzero_exit' else '0') + ';}\n')
    out = tmp_path/'preflight'
    result = validate_native(reference, None, testbench, out=out, include_files=headers)
    assert result['status'] == 'failed' and not result['passed']
    assert result['candidate'] == {} and result['comparison'] is None
    assert not (out/'candidate').exists() and not (out/'testbench.o').exists()
    if failure == 'empty_output':
        assert result['reference']['status'] == 'passed'
        assert result['error'] == 'reference output is empty'
    else:
        assert result['error'] in ('native compilation or linking failed', 'reference execution failed')


def test_wrong_candidate_output_is_rejected(native_inputs, tmp_path):
    reference, candidate, testbench, headers = native_inputs
    candidate.write_text('#include "kernel.h"\nunsigned kernel(unsigned x) {return x+2u;}\n')
    result = validate_native(reference, candidate, testbench, out=tmp_path/'run', include_files=headers)
    assert result['status'] == 'failed' and not result['passed']
    assert result['comparison']['reason'] == 'output value differs'


def test_compilation_failure_is_not_oracle_validation(native_inputs, tmp_path):
    reference, candidate, testbench, headers = native_inputs
    candidate.write_text('not valid C++')
    result = validate_native(reference, candidate, testbench, out=tmp_path/'run', include_files=headers)
    assert result['status'] == 'failed' and not result['passed']
    assert result['comparison'] is None


def test_large_integers_stay_exact_and_floats_need_explicit_tolerance():
    assert not compare_outputs('9007199254740992', '9007199254740993')['passed']
    assert not compare_outputs('4294967294', '4294967295', rtol=.1)['passed']
    assert not compare_outputs('1.0000', '1.0001')['passed']
    assert compare_outputs('1.0000', '1.0001', atol=.001)['passed']
    for value in ('NaN', 'Inf', '-Infinity'):
        assert not compare_outputs(value, value)['passed']
    assert not compare_outputs('', '')['passed']


def test_timeout_kills_owned_process_group(tmp_path):
    program = "import subprocess,sys,time;from pathlib import Path;p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)']);Path('child.pid').write_text(str(p.pid));time.sleep(60)"
    result = run_command([sys.executable, '-c', program], cwd=tmp_path, timeout=.5)
    assert result['status'] == 'timeout'
    child = int((tmp_path/'child.pid').read_text())
    path = Path('/proc') / str(child) / 'stat'
    try:
        body = path.read_text()
    except (FileNotFoundError, ProcessLookupError):
        return  # A child reaped between observation and read is terminated.
    assert body[body.rfind(')')+2:].split()[0] == 'Z'


def test_unavailable_tools_return_no_measurement(tmp_path):
    result = synthesize_vitis(tmp_path/'unused.cpp', out=tmp_path/'vitis', top='kernel',
                              part='part', clock_ns=3.33, executable='hlsmith-nonexistent-tool')
    assert result['status'] == 'unavailable' and result['metrics'] == {'hls_estimate': {}}
    result = simulate_verilator({'executable': 'hlsmith-nonexistent-tool'}, out=tmp_path/'rtl')
    assert result['status'] == 'unavailable' and result['metrics']['rtl_cycles'] is None


@pytest.mark.skipif(not shutil.which('verilator'), reason='optional Verilator not installed')
def test_real_verilator_matches_original_oracle_output(tmp_path):
    rtl = tmp_path/'add.sv'
    harness = tmp_path/'harness.cpp'
    rtl.write_text('module add(input logic clk, input logic [31:0] a, output logic [31:0] y); always_ff @(posedge clk) y<=a+32\'d1; endmodule\n')
    harness.write_text('#include "Vdut.h"\n#include <cstdio>\nint main(){Vdut d;int cycles=0;auto tick=[&](){d.clk=0;d.eval();d.clk=1;d.eval();++cycles;};d.a=5;tick();std::printf("%u ",d.y);d.a=0xffffffffu;tick();std::printf("%u\\nHLSMITH_CYCLES: %d\\n",d.y,cycles);d.final();}\n')
    result = simulate_verilator({'rtl_files': [rtl], 'harness': harness, 'top_module': 'add',
                                 'expected_output': '6 0\n', 'build_timeout': 60, 'run_timeout': 5},
                                out=tmp_path/'simulation')
    assert result['status'] == 'passed', result
    assert result['passed'] and result['metrics']['rtl_cycles'] == 2
    assert result['cycle_measurement_qualified']


def test_rtl_pass_requires_nonempty_oracle(tmp_path):
    result = simulate_verilator({'expected_output': '', 'top_module': 'add'}, out=tmp_path/'rtl')
    assert result['status'] in ('error', 'unavailable') and not result['passed']


def test_candidate_cannot_use_oracle_files_from_its_working_directory(native_inputs, tmp_path):
    reference, candidate, testbench, headers = native_inputs
    candidate.write_text('#include "kernel.h"\n#include <unistd.h>\nunsigned kernel(unsigned x){return access("reference.exe",F_OK)==0 || access("reference.c",F_OK)==0 ? 123u : x+1u;}\n')
    result = validate_native(reference, candidate, testbench, out=tmp_path/'run', include_files=headers)
    assert result['passed']
    assert not list((tmp_path/'run').rglob('reference.*'))
    assert not Path(result['reference']['argv'][0]).exists()  # Oracle removed before candidate runs.
    assert 'kernel.h' in result['input_hashes']['include_files']


def test_forced_public_header_rejects_wrong_signature_even_without_candidate_include(native_inputs, tmp_path):
    reference, candidate, testbench, headers = native_inputs
    candidate.write_text('extern "C" int kernel(unsigned x){return x+1u;}\n')
    result = validate_native(reference, candidate, testbench, out=tmp_path/'run', include_files=headers,
                              interface_headers=['kernel.h'])
    assert not result['passed'] and result['comparison'] is None
    assert result['status'] == 'failed'
    assert 'conflicting' in result['commands'][-1]['stderr']


def test_tool_output_reuse_is_rejected(tmp_path):
    out = tmp_path/'old'; out.mkdir(); (out/'old.exe').write_bytes(b'old')
    with pytest.raises(ValueError, match='empty'):
        simulate_verilator({}, out=out)
    assert (out/'old.exe').read_bytes() == b'old'


@pytest.mark.parametrize('clock,expected_status', [('3.33','passed'), ('5','error'), ('missing','error')])
def test_optional_vitis_binds_reported_clock_without_executing_tools(monkeypatch, tmp_path, clock, expected_status):
    from hlsmith_minimal import tools
    candidate = tmp_path/'candidate.cpp'; candidate.write_text('void kernel() {}')
    monkeypatch.setattr(tools, '_tool', lambda name: '/fake/vitis_hls')
    def fake_command(argv, *, cwd, timeout, env=None):
        if '-f' in argv:
            report = Path(cwd)/'hls_project/solution/syn/report/kernel_csynth.xml'
            report.parent.mkdir(parents=True)
            period = '<TargetClockPeriod>'+clock+'</TargetClockPeriod>' if clock != 'missing' else ''
            report.write_text('<Report><UserAssignments>'+period+'</UserAssignments><PerformanceEstimates><SummaryOfOverallLatency><Worst-caseLatency>12</Worst-caseLatency></SummaryOfOverallLatency></PerformanceEstimates><AreaEstimates><Resources><LUT>3.5</LUT><FF>-1</FF></Resources></AreaEstimates></Report>')
        return {'status':'passed','returncode':0,'stdout':'fixture tool version','stderr':'','argv':list(argv)}
    monkeypatch.setattr(tools, 'run_command', fake_command)
    result = synthesize_vitis(candidate, out=tmp_path/'synth', top='kernel', part='device', clock_ns=3.33)
    assert result['status'] == expected_status
    if expected_status == 'passed':
        assert result['metrics']['hls_estimate']['requested_clock_period_ns'] == 3.33
        assert result['metrics']['hls_estimate']['lut'] is None
        assert result['metrics']['hls_estimate']['ff'] is None


@pytest.mark.parametrize('with_trace_inputs', [False, True])
def test_vitis_registers_trace_testbench_only_when_requested(monkeypatch, tmp_path, with_trace_inputs):
    from hlsmith_minimal import tools
    candidate = tmp_path/'source.cpp'; candidate.write_text('void kernel() {}')
    testbench = tmp_path/'test.c'; testbench.write_text('int main(void) {return 0;}')
    vectors = tmp_path/'vectors.h'; vectors.write_text('#define PRIVATE_INPUT 7\n')
    monkeypatch.setattr(tools, '_tool', lambda name: '/fake/vitis_hls')
    def fake_command(argv, *, cwd, timeout, env=None):
        if '-f' in argv:
            report = Path(cwd)/'hls_project/solution/syn/report/kernel_csynth.xml'
            report.parent.mkdir(parents=True)
            report.write_text('<Report><UserAssignments><TargetClockPeriod>3.33</TargetClockPeriod>'
                              '</UserAssignments></Report>')
        return {'status':'passed','returncode':0,'stdout':'fixture','stderr':'','argv':list(argv)}
    monkeypatch.setattr(tools, 'run_command', fake_command)
    extra = {'testbench':testbench,'test_files':{'vectors.h':vectors}} if with_trace_inputs else {}
    out = tmp_path/'synth'
    result = synthesize_vitis(candidate,out=out,top='kernel',part='device',clock_ns=3.33,**extra)
    assert result['passed']
    assert ('add_files -tb' in (out/'synthesize.tcl').read_text()) == with_trace_inputs
    assert ('testbench' in result['input_hashes']) == with_trace_inputs
    assert (out/'vectors.h').exists() == with_trace_inputs
    assert result['solution_dir'] == str(out/'hls_project/solution')
    assert result['workspace'] == str(out)
