"""Official MDPI artifact contracts, with fake compiler responses only."""
from pathlib import Path

import pytest

from hlsmith_minimal import bambu


WRAPPER = '''void kernel() {
__m_sim_start();
#ifndef CUSTOM_VERIFICATION
__m_kernel(P0_gold);
#endif
__m_sim_end();
size_t mismatch_count = 0;
m_argcmp(0, _integer);
if(mismatch_count) { __m_abort(); }
}'''


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    candidate = tmp_path/'kernel.c'; candidate.write_text('int kernel(int a) {return a+1;}')
    tb = tmp_path/'tb.xml'; tb.write_text('<function><testbench a="1"/><testbench a="2"/></function>')
    tool = tmp_path/'fake-tool'; tool.write_text('fixture, never executed')
    monkeypatch.setattr(bambu.shutil, 'which', lambda executable: str(tool))
    return candidate, tb


def mock_command(monkeypatch, *, text='0|8,10|15,\n0\n', wrapper=WRAPPER, status='passed', code=0):
    calls = []
    def command(argv, *, cwd, timeout, env):
        calls.append((argv, Path(cwd), env))
        if argv[-1] == '--version':
            return {'status': 'passed', 'returncode': 0, 'stdout': 'Version: PandA 2024.10 - fixture', 'stderr': ''}
        out = Path(cwd)
        (out/'HLS_output/simulation').mkdir(parents=True)
        if wrapper is not None:
            (out/'HLS_output/simulation/mdpi_wrapper.cpp').write_text(wrapper)
        if text is not None:
            (out/'results.txt').write_text(text)
        (out/'kernel.v').write_text('module kernel; endmodule // fixture only')
        (out/'HLS_output/simulation/bambu_testbench.v').write_text('parameter CLOCK_PERIOD=2.0; // fixture only')
        (out/'simulate_kernel.sh').write_text('verilator --cc --exe fixture.v\n')
        return {'status': status, 'returncode': code, 'stdout': 'fixture tool output', 'stderr': ''}
    monkeypatch.setattr(bambu, 'run_command', command)
    return calls


def test_checked_result_uses_interval_cycles_and_registered_call_count(inputs, tmp_path, monkeypatch):
    calls = mock_command(monkeypatch)
    result = bambu.evaluate(*inputs, out=tmp_path/'out', top='kernel', clock_ns=3.33)
    assert result['passed'] and result['builtin_validation_enabled']
    assert result['metrics']['rtl_cycles'] == 7
    assert result['metrics']['rtl_cycles_per_execution'] == [4, 3]
    assert result['metrics']['num_executions'] == 2
    assert result['artifacts']['results.txt'] and result['input_hashes']['candidate.c']
    argv, out, env = calls[-1]
    assert '--simulator=VERILATOR' in argv and '--writer=V' in argv
    assert '--clock-period=3.33' in argv and result['requested_clock_ns'] == 3.33
    assert '--max-ulp=0' in argv and result['cosimulation_max_ulp'] == 0
    assert '-m64' in argv and result['abi_bits'] == 64 and result['build_threads'] == 1
    assert result['tools']['verilator']['identity_scope'] == 'outer_environment_candidate'
    assert result['simulator_version'] is None and result['artifacts']['simulate_kernel.sh']
    assert not any(arg.startswith(('--evaluation', '--tb-param-size')) for arg in argv)
    assert env['CCACHE_DISABLE'] == '1' and Path(env['TMPDIR']).is_relative_to(out)
    assert env['OMP_NUM_THREADS'] == '1'
    assert inputs[0].read_text() == 'int kernel(int a) {return a+1;}'


@pytest.mark.parametrize('text', [None, '', '1 20\n', '0|8\n0\n', '0|8,10|15\n1\n',
                                  '0|8,10|X\n0\n', '0|8,10|A\n0\n', '0|8,15|10\n0\n',
                                  '0|8,10|10\n0\n', '0|8,10|15,,\n0\n', '0|8,10|15\n0\ntrailing'])
def test_exit_zero_without_complete_matching_checked_evidence_is_not_a_pass(inputs, tmp_path, monkeypatch, text):
    mock_command(monkeypatch, text=text)
    result = bambu.evaluate(*inputs, out=tmp_path/'out', top='kernel')
    assert result['status'] == 'failed' and not result['passed']
    assert result['metrics']['rtl_cycles'] is None


@pytest.mark.parametrize('wrapper', [None, 'generic comparison macros only',
                                    WRAPPER.replace('__m_kernel(P0_gold);', ''),
                                    WRAPPER.replace('m_argcmp(0, _integer);', ''),
                                    '#define CUSTOM_VERIFICATION\n' + WRAPPER])
def test_missing_or_disabled_value_comparison_is_not_a_pass(inputs, tmp_path, monkeypatch, wrapper):
    mock_command(monkeypatch, wrapper=wrapper)
    result = bambu.evaluate(*inputs, out=tmp_path/'out', top='kernel')
    assert not result['passed'] and not result['builtin_validation_enabled']


@pytest.mark.parametrize('status,code', [('failed', 1), ('timeout', -9), ('passed', 1)])
def test_partial_artifacts_do_not_override_process_failure(inputs, tmp_path, monkeypatch, status, code):
    mock_command(monkeypatch, status=status, code=code)
    result = bambu.evaluate(*inputs, out=tmp_path/'out', top='kernel')
    assert not result['passed'] and result['metrics']['rtl_cycles'] is None


def test_main_custom_verification_and_empty_vectors_rejected_before_execution(inputs, tmp_path, monkeypatch):
    calls = mock_command(monkeypatch)
    assert bambu.evaluate(*inputs, out=tmp_path/'main', top='main')['status'] == 'unsupported'
    inputs[0].write_text('#define CUSTOM_VERIFICATION\nint kernel(int a){return a;}')
    assert bambu.evaluate(*inputs, out=tmp_path/'custom', top='kernel')['status'] == 'unsupported'
    inputs[0].write_text('int kernel(int a){return a;}')
    inputs[1].write_text('<function/>')
    assert bambu.evaluate(*inputs, out=tmp_path/'empty', top='kernel')['status'] == 'unsupported'
    assert not calls


def test_c_testbench_uses_executed_calls_and_copies_named_assets(inputs, tmp_path, monkeypatch):
    mock_command(monkeypatch)
    tb = tmp_path/'tb.c'; tb.write_text('int main(){return 0;}')
    header = tmp_path/'public.h'; header.write_text('int kernel(int);')
    result = bambu.evaluate(inputs[0], tb, include_files={'api/kernel.h': header}, out=tmp_path/'out', top='kernel')
    assert result['passed'] and result['metrics']['num_executions'] == 2
    assert result['input_hashes']['api/kernel.h'] == bambu._sha(header)
    assert (tmp_path/'out/api/kernel.h').read_bytes() == header.read_bytes()


def test_stale_output_and_unsafe_names_are_rejected(inputs, tmp_path, monkeypatch):
    calls = mock_command(monkeypatch)
    out = tmp_path/'out'; out.mkdir(); (out/'results.txt').write_text('old')
    with pytest.raises(ValueError, match='empty'):
        bambu.evaluate(*inputs, out=out, top='kernel')
    with pytest.raises(ValueError, match='relative'):
        bambu.evaluate(*inputs, include_files={'../bad.h': inputs[0]}, out=tmp_path/'new', top='kernel')
    result = bambu.evaluate(*inputs, out=tmp_path/'shell;injection', top='kernel')
    assert result['status'] == 'unsupported' and not calls


def test_missing_tool_is_explicit_and_creates_no_workspace(inputs, tmp_path, monkeypatch):
    monkeypatch.setattr(bambu.shutil, 'which', lambda executable: None)
    result = bambu.evaluate(*inputs, out=tmp_path/'out', top='kernel')
    assert result['status'] == 'unavailable' and not result['passed']
    assert not (tmp_path/'out').exists()


@pytest.mark.parametrize('script', [None, 'verilator --cc -DCUSTOM_VERIFICATION fixture.v\n'])
def test_missing_or_unchecked_simulation_script_cannot_qualify(inputs, tmp_path, monkeypatch, script):
    mock_command(monkeypatch)
    original = bambu.run_command
    def command(argv, **kwargs):
        result = original(argv, **kwargs)
        if argv[-1] != '--version':
            path = Path(kwargs['cwd'])/'simulate_kernel.sh'
            path.unlink() if script is None else path.write_text(script)
        return result
    monkeypatch.setattr(bambu, 'run_command', command)
    result = bambu.evaluate(*inputs, out=tmp_path/'out', top='kernel')
    assert not result['passed'] and result['metrics']['rtl_cycles'] is None


def test_changed_snapshot_cannot_be_attributed_to_original_input_hash(inputs, tmp_path, monkeypatch):
    calls = mock_command(monkeypatch)
    copy = bambu.shutil.copyfile
    def changed_copy(source, destination):
        copy(source, destination)
        Path(destination).write_text('bytes changed during capture')
    monkeypatch.setattr(bambu.shutil, 'copyfile', changed_copy)
    result = bambu.evaluate(*inputs, out=tmp_path/'out', top='kernel')
    assert result['status'] == 'error' and not result['passed'] and not calls


def test_inherited_make_or_compiler_overrides_are_removed_and_never_recorded(inputs, tmp_path, monkeypatch):
    for name in ('CFLAGS', 'CXXFLAGS', 'CPPFLAGS', 'MAKEFLAGS'):
        monkeypatch.setenv(name, '-DCUSTOM_VERIFICATION private-flag-value')
    calls = mock_command(monkeypatch)
    result = bambu.evaluate(*inputs, out=tmp_path/'out', top='kernel')
    assert result['passed']
    assert all(name not in calls[-1][2] for name in ('CFLAGS', 'CXXFLAGS', 'CPPFLAGS', 'MAKEFLAGS'))
    assert set(result['cleared_build_environment']) >= {'CFLAGS', 'CXXFLAGS', 'CPPFLAGS', 'MAKEFLAGS'}
    assert 'private-flag-value' not in str(result)


def test_different_tool_protocol_or_testbench_period_cannot_qualify_cycles(inputs, tmp_path, monkeypatch):
    calls = mock_command(monkeypatch)
    original = bambu.run_command
    def command(argv, **kwargs):
        result = original(argv, **kwargs)
        if argv[-1] != '--version':
            (Path(kwargs['cwd'])/'HLS_output/simulation/bambu_testbench.v').write_text('parameter CLOCK_PERIOD=4.0;')
        return result
    monkeypatch.setattr(bambu, 'run_command', command)
    assert not bambu.evaluate(*inputs, out=tmp_path/'period', top='kernel')['passed']
    def unknown(argv, **kwargs):
        return {'status': 'passed', 'returncode': 0, 'stdout': 'PandA 2099.1', 'stderr': ''}
    monkeypatch.setattr(bambu, 'run_command', unknown)
    result = bambu.evaluate(*inputs, out=tmp_path/'version', top='kernel')
    assert result['status'] == 'unsupported' and not result['passed']
