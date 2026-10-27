"""Optional adapter protocol tests with fake responses, never real LightningSim."""
from pathlib import Path
import xml.etree.ElementTree as ET

import pytest

from hlsmith_minimal import lightningsim as ls


@pytest.fixture
def solution(tmp_path, monkeypatch):
    workspace = tmp_path / 'original'
    sol = workspace / 'project/solution'
    database = sol / '.autopilot/db'
    database.mkdir(parents=True)
    (database / 'dut.hcp').write_bytes(b'fixture sentinel, not real synthesis')
    (database / 'a.o.3.bc').write_bytes(b'fixture bitcode, not a real program')
    source = workspace / 'candidate.cpp'; source.write_text('void kernel() {}')
    testbench = workspace / 'testbench.cpp'; testbench.write_text('int main() {return 0;}')
    app = sol.parent / 'hls.app'
    app.write_text('<project top="kernel"><files>'
                   f'<file name="{source}" tb="false" cflags="-std=c++17"/>'
                   f'<file name="{testbench}" tb="true" cflags=""/>'
                   '</files></project>')
    executable = tmp_path / 'fake-lightningsim'
    executable.write_text('fixture only; never executed')
    hls = tmp_path / 'hls'; hls.mkdir()
    monkeypatch.setattr(ls.shutil, 'which', lambda value: str(executable))
    return workspace, sol, hls


def mock_cli(monkeypatch, *, stdout='[ 7-23] kernel\n\t[ 8-10] child\n', status='passed', code=0):
    calls = []
    def command(argv, *, cwd, timeout, env):
        calls.append({'argv': list(argv), 'cwd': cwd, 'env': env})
        if argv[-1] == '--help':
            return {'status': 'passed', 'returncode': 0, 'stdout': 'solution_dir --cli --gui', 'stderr': ''}
        copied_solution = Path(argv[1])
        (copied_solution / '.autopilot/db/dut.hcp').write_text('changed only in copied workspace')
        return {'status': status, 'returncode': code, 'stdout': stdout, 'stderr': ''}
    monkeypatch.setattr(ls, 'run_command', command)
    return calls


def test_estimate_uses_top_duration_and_never_claims_correctness(solution, tmp_path, monkeypatch):
    workspace, sol, hls = solution
    before = ls._manifest(workspace)
    calls = mock_cli(monkeypatch)
    result = ls.estimate(sol, workspace=workspace, out=tmp_path/'estimate', hls_root=hls,
                         provenance={'candidate_sha256': 'declared-integration-identity'})
    assert result['status'] == 'estimated' and result['estimate_available']
    assert result['passed'] is False and result['correctness_checked'] is False
    assert result['metrics'] == {'trace_estimate_cycles': 16}
    assert result['version'] is None and result['provenance']['help_sha256']
    assert ls._manifest(workspace) == before
    assert calls[1]['argv'][-1] == '--cli'
    assert Path(calls[1]['argv'][1]) != sol
    assert calls[1]['env']['XILINX_HLS'] == str(hls.resolve())
    assert Path(calls[1]['env']['TMPDIR']).is_relative_to(tmp_path/'estimate')
    assert calls[1]['env']['CCACHE_DISABLE'] == '1'
    assert result['provenance']['effective_hls_root'] == str(hls.resolve())
    assert calls[1]['env']['PATH'].split(ls.os.pathsep)[0] == str(Path(calls[1]['argv'][0]).parent)
    copied_app = tmp_path/'estimate/workspace/project/hls.app'
    entries = ET.parse(copied_app).findall('.//file')
    assert all(Path(entry.attrib['name']).is_relative_to(tmp_path/'estimate/workspace') for entry in entries)


@pytest.mark.parametrize('status,code', [('failed', 1), ('timeout', -9), ('passed', 1)])
def test_failed_or_deadlocked_run_never_qualifies_partial_intervals(solution, tmp_path, monkeypatch, status, code):
    workspace, sol, hls = solution
    mock_cli(monkeypatch, status=status, code=code)
    result = ls.estimate(sol, workspace=workspace, out=tmp_path/'estimate', hls_root=hls)
    assert not result['estimate_available'] and not result['passed']
    assert result['metrics']['trace_estimate_cycles'] is None
    assert result['status'] in ('failed', 'timeout')


@pytest.mark.parametrize('text', ['\t[0-3] child', '[0-3] one\n[0-4] two', '[3-3] top',
                                  '[5-3] top', '[0-3.5] top', 'Simulation finished.', '[0-0] top'])
def test_missing_ambiguous_or_invalid_root_intervals_are_rejected(text):
    with pytest.raises(ValueError):
        ls.parse_interval(text)


def test_external_hls_app_dependency_is_rejected_without_running_estimator(solution, tmp_path, monkeypatch):
    workspace, sol, hls = solution
    outside = tmp_path/'external.cpp'; outside.write_text('external source')
    app = sol.parent/'hls.app'; tree = ET.parse(app)
    tree.findall('.//file')[0].set('name', str(outside)); tree.write(app)
    calls = mock_cli(monkeypatch)
    result = ls.estimate(sol, workspace=workspace, out=tmp_path/'estimate', hls_root=hls)
    assert result['status'] == 'unsupported' and 'external' in result['error']
    assert not calls
    assert outside.read_text() == 'external source'


def test_symlink_and_missing_synthesis_artifacts_never_run(solution, tmp_path, monkeypatch):
    workspace, sol, hls = solution
    calls = mock_cli(monkeypatch)
    (workspace/'link').symlink_to(workspace/'candidate.cpp')
    result = ls.estimate(sol, workspace=workspace, out=tmp_path/'symlink', hls_root=hls)
    assert result['status'] == 'unsupported' and not calls
    (workspace/'link').unlink()
    (sol/'.autopilot/db/dut.hcp').unlink()
    result = ls.estimate(sol, workspace=workspace, out=tmp_path/'missing', hls_root=hls)
    assert result['status'] == 'unsupported' and not calls


def test_no_implicit_unsafe_copy_scope_and_no_output_reuse(solution, tmp_path, monkeypatch):
    workspace, sol, hls = solution
    calls = mock_cli(monkeypatch)
    result = ls.estimate(sol, out=tmp_path/'narrow', hls_root=hls)
    assert result['status'] == 'unsupported' and not calls  # Explicit workspace required for siblings.
    existing = tmp_path/'existing'; existing.mkdir(); (existing/'old').write_text('keep')
    with pytest.raises(ValueError, match='empty'):
        ls.estimate(sol, workspace=workspace, out=existing, hls_root=hls)
    assert (existing/'old').read_text() == 'keep'
    with pytest.raises(ValueError, match='disjoint'):
        ls.estimate(sol, workspace=workspace, out=workspace/'inside', hls_root=hls)


def test_missing_executable_or_hls_environment_is_explicit(tmp_path, monkeypatch):
    monkeypatch.setattr(ls.shutil, 'which', lambda value: None)
    result = ls.estimate(tmp_path/'nonexistent', out=tmp_path/'out')
    assert result['status'] == 'unavailable' and result['metrics']['trace_estimate_cycles'] is None
    assert not (tmp_path/'out').exists()


def test_copy_budget_and_unknown_stdout_are_rejected(solution, tmp_path, monkeypatch):
    workspace, sol, hls = solution
    calls = mock_cli(monkeypatch)
    monkeypatch.setattr(ls, '_MAX_COPY_BYTES', 1)
    result = ls.estimate(sol, workspace=workspace, out=tmp_path/'bounded', hls_root=hls)
    assert result['status'] == 'unsupported' and 'copy limit' in result['error']
    assert not calls
    with pytest.raises(ValueError):
        ls.parse_interval('[0-10] kernel\n[bad-interval] other')


def test_manifest_stops_iteration_at_entry_bound(solution, monkeypatch):
    workspace, _, _ = solution
    def bounded_walk(root, pattern):
        yield workspace/'candidate.cpp'
        yield workspace/'testbench.cpp'
        raise AssertionError('traversal continued beyond entry budget')
    monkeypatch.setattr(Path, 'rglob', bounded_walk)
    monkeypatch.setattr(ls, '_MAX_COPY_ENTRIES', 1)
    with pytest.raises(ValueError, match='entry limit'):
        ls._manifest(workspace)


def test_external_conda_metadata_records_actual_package_identity(solution, tmp_path, monkeypatch):
    import json
    workspace, sol, hls = solution
    prefix = tmp_path/'isolated-environment'
    executable = prefix/'bin/lightningsim'
    executable.parent.mkdir(parents=True)
    executable.write_text('fixture launcher; never executed')
    metadata = prefix/'conda-meta/lightningsim-0.2.6-py310.json'
    metadata.parent.mkdir()
    metadata.write_text(json.dumps({'name':'lightningsim','version':'0.2.6','build':'py310_0',
                                   'sha256':'a'*64,'url':'not exported'}))
    monkeypatch.setattr(ls.shutil, 'which', lambda value: str(executable))
    mock_cli(monkeypatch)
    result = ls.estimate(sol, workspace=workspace, out=tmp_path/'estimate', hls_root=hls)
    assert result['version'] == '0.2.6'
    identity = result['provenance']['package_identity']
    assert identity['sha256'] == 'a'*64 and identity['metadata_sha256'] == ls._sha(metadata)
    assert 'url' not in identity


@pytest.mark.parametrize('flag', ['-I../../../../outside', '-isystem /outside', '-I..'])
def test_relative_or_absolute_compiler_escape_is_unsupported(solution, tmp_path, monkeypatch, flag):
    workspace, sol, hls = solution
    app = sol.parent/'hls.app'
    tree = ET.parse(app); tree.findall('.//file')[0].set('cflags', flag); tree.write(app)
    calls = mock_cli(monkeypatch)
    result = ls.estimate(sol, workspace=workspace, out=tmp_path/'estimate', hls_root=hls)
    assert result['status'] == 'unsupported' and 'compiler flags' in result['error']
    assert not calls
