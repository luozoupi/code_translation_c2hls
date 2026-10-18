"""CLI resume uses actual input/runtime identities without calling providers."""
import json
from pathlib import Path
from unittest.mock import Mock

import pytest

import c2hls as core
from hlsmith.cli import resume_identity


@pytest.fixture
def cli_case(tmp_path, monkeypatch):
    benchmarks = tmp_path / 'benchmarks'
    benchmark = benchmarks / 'toy'
    benchmark.mkdir(parents=True)
    (benchmark / 'plain.cpp').write_text('void workload() {}\n')
    (benchmark / 'kernel.h').write_text('void workload();\n')
    (benchmark / 'testbench.cpp').write_text('int main() { return 0; }\n')
    (benchmark / 'metadata.json').write_text(json.dumps({'benchmark': 'toy', 'plain_c_file': 'plain.cpp',
           'header_file': 'kernel.h', 'testbench_file': 'testbench.cpp'}))
    (benchmarks / 'index.json').write_text(json.dumps([{'benchmark': 'toy'}]))
    output = tmp_path / 'results'
    monkeypatch.setenv('C2HLS_BENCH_ROOT', str(benchmarks))
    monkeypatch.delenv('C2HLS_NO_RESUME', raising=False)
    run = Mock(return_value={'benchmark': 'toy', 'success': True})
    # Fresh results prevent helper metadata from being retained by the mock.
    run.side_effect = lambda *args, **kwargs: {'benchmark': 'toy', 'success': True}
    monkeypatch.setattr(core, 'run_benchmark', run)
    args = ['--all', '--output-dir', str(output), '--model', 'deepseek-flash']
    return benchmark, output, run, args


def test_all_writes_per_benchmark_output_and_identical_resume_uses_no_provider(cli_case):
    benchmark, output, run, args = cli_case
    core.main(args)
    run.assert_called_once()
    assert run.call_args.args[0] == str(benchmark)
    assert run.call_args.kwargs['output_dir'] == str(output / 'toy')
    result_file = output / 'toy' / 'toy_results.json'
    saved = json.loads(result_file.read_text())
    assert saved['cli_resume_identity']['schema_version'] == 'hlsmith.cli-resume.v1'
    assert (output / 'all_results.json').is_file()
    core.main(args)
    assert run.call_count == 1


@pytest.mark.parametrize('change', ['model', 'input', 'control', 'skill'])
def test_changed_model_input_control_or_skill_bytes_reruns(cli_case, monkeypatch, tmp_path, change):
    benchmark, output, run, args = cli_case
    skill = tmp_path / 'skills.json'
    skill.write_text('{"skills": []}')
    monkeypatch.setenv('C2HLS_SKILL_LIBRARY_PATH', str(skill))
    core.main(args)
    if change == 'model':
        args[-1] = 'different-model'
    elif change == 'input':
        (benchmark / 'plain.cpp').write_text('void workload() { /* changed */ }\n')
    elif change == 'control':
        monkeypatch.setenv('C2HLS_CANDIDATES_PER_STEP', '4')
    else:
        skill.write_text('{"skills": [{"id": "new"}]}')
    core.main(args)
    assert run.call_count == 2


def test_legacy_result_without_identity_is_not_reused(cli_case):
    benchmark, output, run, args = cli_case
    folder = output / 'toy'
    folder.mkdir(parents=True)
    (folder / 'toy_results.json').write_text(json.dumps({'benchmark': 'toy', 'success': True,
                         'resume_configuration': {'model': 'deepseek-flash'}}))
    core.main(args)
    assert run.call_count == 1


def test_unavailable_identity_never_reuses_unfingerprinted_results(cli_case, monkeypatch):
    import hlsmith.cli
    benchmark, output, run, args = cli_case
    def unavailable(*args, **kwargs):
        raise OSError('source cannot be read')
    monkeypatch.setattr(hlsmith.cli, 'resume_identity', unavailable)
    core.main(args)
    core.main(args)
    assert run.call_count == 2


def test_env_values_are_hashed_and_generated_caches_do_not_change_identity(cli_case, tmp_path):
    benchmark, output, run, args = cli_case
    ledger = tmp_path / 'ledger.json'
    ledger.write_text('{"mutable": 1}')
    environment = {'C2HLS_PRIVATE_TOKEN': 'private-value-never-store-this',
                   'C2HLS_API_BUDGET_LEDGER': str(ledger), 'OPENAI_BASE_URL': 'https://private-provider.example'}
    before = resume_identity(benchmark, {'model': 'test'}, environ=environment)
    persisted = json.dumps(before)
    assert 'private-value-never-store-this' not in persisted
    assert 'https://private-provider.example' not in persisted
    assert str(ledger) not in persisted
    ledger.write_text('{"mutable": 2}')
    (benchmark / 'toy_results.json').write_text('{"generated": true}')
    (benchmark / '__pycache__').mkdir()
    (benchmark / '__pycache__' / 'generated.py').write_text('generated cache')
    assert before == resume_identity(benchmark, {'model': 'test'}, environ=environment)


def test_installed_layout_hashes_code_outside_the_packaged_data_root(cli_case, tmp_path):
    benchmark = cli_case[0]
    package = tmp_path / 'site-packages' / 'hlsmith'
    data = package / 'data'
    data.mkdir(parents=True)
    (package / 'pipeline.py').write_text('# installed runtime v1\n')
    (package.parent / 'prompt_c2hls.py').write_text('instruction = "paper"\n')
    before = resume_identity(benchmark, {}, environ={}, package_root=package, data_root=data)
    (package / 'artifacts').mkdir()
    (package / 'artifacts' / 'generated.py').write_text('# generated tool helper\n')
    assert before == resume_identity(benchmark, {}, environ={}, package_root=package, data_root=data)
    (package / 'pipeline.py').write_text('# installed runtime v2\n')
    after = resume_identity(benchmark, {}, environ={}, package_root=package, data_root=data)
    assert before != after
    (package / 'pipeline.py').write_text('# installed runtime v1\n')
    assert before == resume_identity(benchmark, {}, environ={}, package_root=package, data_root=data)
    (package.parent / 'prompt_c2hls.py').write_text('instruction = "changed paper"\n')
    assert before != resume_identity(benchmark, {}, environ={}, package_root=package, data_root=data)
