"""Catch runtime files that work locally but disappear from Git exports."""
import json
from pathlib import Path
import shutil
import subprocess
import zipfile

import pytest

from scripts.check_release_completeness import audit


def git(root, *arguments):
    subprocess.run(['git', '-C', str(root), *arguments], check=True, capture_output=True)


@pytest.fixture
def release_tree(tmp_path):
    root = tmp_path / 'source'
    root.mkdir()
    git(root, 'init')
    (root / 'hlsmith/toolchain').mkdir(parents=True)
    (root / 'hlsmith/data').mkdir()
    (root / 'hlsmith/__init__.py').write_text('')
    (root / 'hlsmith/toolchain/__init__.py').write_text('')
    (root / 'hlsmith/toolchain/c2hls_temp.py').write_text('VALUE = 1\n')
    (root / 'c2hls_temp.py').write_text('from hlsmith.toolchain.c2hls_temp import VALUE\n')
    (root / 'hlsmith/data/required.dat').write_bytes(b'benchmark input')
    (root / 'hlsmith/legacy_modules.json').write_text(json.dumps({'c2hls_temp': 'hlsmith.toolchain.c2hls_temp'}))
    (root / 'pyproject.toml').write_text('[tool.setuptools]\npy-modules = ["c2hls_temp"]\n')
    (root / 'README.md').write_text('Release fixture\n')
    git(root, 'add', '.')
    return root


def test_nested_runtime_ignored_by_basename_is_rejected_even_when_root_shim_is_tracked(release_tree):
    root = release_tree
    # Reproduce the real blocker: a broad pattern plus an exception for only
    # the root alias lets a working-tree wheel conceal the missing target.
    git(root, 'rm', '--cached', 'hlsmith/toolchain/c2hls_temp.py')
    (root / '.gitignore').write_text('c2hls_temp.py\n!/c2hls_temp.py\n')
    report = audit(root)
    assert not report['passed']
    assert report['untracked_runtime_files'] == ['hlsmith/toolchain/c2hls_temp.py']
    assert report['ignored_runtime_files'] == ['hlsmith/toolchain/c2hls_temp.py']
    (root / '.gitignore').write_text('/c2hls_temp.py\n!/c2hls_temp.py\n')
    git(root, 'add', 'hlsmith/toolchain/c2hls_temp.py')
    assert audit(root)['passed']


def test_ignored_runtime_assets_are_checked_without_extension_whitelisting(release_tree):
    root = release_tree
    git(root, 'rm', '--cached', 'hlsmith/data/required.dat')
    (root / '.gitignore').write_text('*.dat\n')
    report = audit(root)
    assert not report['passed']
    assert report['ignored_runtime_files'] == ['hlsmith/data/required.dat']
    assert report['untracked_runtime_files'] == ['hlsmith/data/required.dat']


def test_export_must_contain_identical_runtime_sources_and_assets(release_tree, tmp_path):
    root = release_tree
    export = tmp_path / 'export'
    shutil.copytree(root, export, ignore=shutil.ignore_patterns('.git'))
    assert audit(root, export)['passed']
    (export / 'hlsmith/toolchain/c2hls_temp.py').unlink()
    (export / 'hlsmith/data/required.dat').write_bytes(b'changed benchmark')
    report = audit(root, export)
    assert not report['passed']
    assert report['export_missing_files'] == ['hlsmith/toolchain/c2hls_temp.py']
    assert report['export_changed_files'] == ['hlsmith/data/required.dat']


def test_missing_declared_root_module_or_alias_target_is_a_release_failure(release_tree):
    root = release_tree
    for name in ('hlsmith/toolchain/c2hls_temp.py', 'c2hls_temp.py'):
        git(root, 'rm', '--cached', name)
        (root / name).unlink()
    report = audit(root)
    assert not report['passed']
    assert report['missing_runtime_files'] == ['c2hls_temp.py', 'hlsmith/toolchain/c2hls_temp.py']


def test_wheel_must_include_profile_sources_and_every_asset_with_identical_bytes(release_tree, tmp_path):
    root = release_tree
    profile = root / 'hlsmith/prompts/researcher_v1.py'
    profile.parent.mkdir()
    profile.write_text('PROMPT = "readable profile"\n')
    (profile.parent / '__init__.py').write_text('')
    git(root, 'add', 'hlsmith/prompts')
    members = ('c2hls_temp.py', 'hlsmith/__init__.py', 'hlsmith/toolchain/__init__.py',
               'hlsmith/toolchain/c2hls_temp.py', 'hlsmith/legacy_modules.json', 'hlsmith/data/required.dat',
               'hlsmith/prompts/__init__.py')
    wheel = tmp_path / 'fixture.whl'
    with zipfile.ZipFile(wheel, 'w') as archive:
        for name in members:
            archive.write(root / name, name)
    report = audit(root, wheel=wheel)
    assert not report['passed']
    assert report['wheel_missing_files'] == ['hlsmith/prompts/researcher_v1.py']
    with zipfile.ZipFile(wheel, 'a') as archive:
        archive.write(profile, 'hlsmith/prompts/researcher_v1.py')
    assert audit(root, wheel=wheel)['passed']
    with zipfile.ZipFile(wheel, 'w') as archive:
        for name in (*members, 'hlsmith/prompts/researcher_v1.py'):
            archive.writestr(name, b'changed' if name.endswith('.dat') else (root / name).read_bytes())
    assert audit(root, wheel=wheel)['wheel_changed_files'] == ['hlsmith/data/required.dat']


def test_python_directories_require_initializers_for_explicit_package_discovery(release_tree):
    root = release_tree
    prompts = root / 'hlsmith/prompts'
    prompts.mkdir()
    (prompts / 'researcher_v1.py').write_text('PROMPT = "saved text"\n')
    git(root, 'add', 'hlsmith/prompts/researcher_v1.py')
    report = audit(root)
    assert not report['passed']
    assert report['missing_package_initializers'] == ['hlsmith/prompts/__init__.py']
