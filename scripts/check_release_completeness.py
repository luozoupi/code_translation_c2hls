#!/usr/bin/env python3
"""Check that Git and an optional clean export contain the runtime distribution.

Run before releasing, then repeat with --export-root pointing at a fresh
``git archive`` extraction. This reads files and Git metadata only.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess
import zipfile


PACKAGES = ('hlsmith', 'dataset_pipeline', 'paper_eval')


def _runtime_path(name):
    path = Path(name)
    if '__pycache__' in path.parts:
        return False
    return ((path.suffix == '.py' and (len(path.parts) == 1 or path.parts[0] in PACKAGES))
            or name.startswith('hlsmith/data/') or name == 'hlsmith/legacy_modules.json')


def _git(root, arguments, **kwargs):
    return subprocess.run(['git', '-C', str(root), *arguments], capture_output=True,
                          check=False, **kwargs)


def audit(root, export_root=None, wheel=None):
    root = Path(root).resolve()
    tracked_result = _git(root, ['ls-files', '-z'])
    if tracked_result.returncode:
        raise ValueError('release check requires the source Git checkout')
    tracked = set(tracked_result.stdout.decode().split('\0')) - {''}
    actual = {p.relative_to(root).as_posix() for p in root.glob('*.py') if p.is_file()}
    for package in PACKAGES:
        actual.update(p.relative_to(root).as_posix() for p in (root / package).rglob('*')
                      if p.is_file() and _runtime_path(p.relative_to(root).as_posix()))
    required = actual | {name for name in tracked if _runtime_path(name)}
    initializers = set()
    for name in actual:
        path = Path(name)
        if path.suffix == '.py' and len(path.parts) > 1 and path.parts[0] in PACKAGES and not name.startswith('hlsmith/data/'):
            initializers.update((parent / '__init__.py').as_posix() for parent in path.parents
                                if parent != Path('.'))
    required.update(initializers)
    required.update(('pyproject.toml', 'README.md', 'hlsmith/legacy_modules.json'))
    # This repository declares py-modules as a literal string array. Parse that
    # array without requiring a TOML package on the supported Python 3.10.
    declarations = re.search(r'(?ms)^py-modules\s*=\s*(\[.*?\])',
                             (root / 'pyproject.toml').read_text())
    if declarations is None:
        raise ValueError('pyproject.toml lacks its explicit runtime module list')
    modules = ast.literal_eval(declarations.group(1))
    if not isinstance(modules, list) or not all(isinstance(name, str) for name in modules):
        raise ValueError('py-modules must be a literal list of module names')
    required.update(name.replace('.', '/') + '.py' for name in modules)
    aliases = json.loads((root / 'hlsmith/legacy_modules.json').read_text())
    required.update(target.replace('.', '/') + '.py' for target in aliases.values())
    ignored_result = _git(root, ['check-ignore', '--no-index', '-z', '--stdin'],
                          input='\0'.join(sorted(required)).encode() + b'\0')
    if ignored_result.returncode not in (0, 1):
        raise ValueError('Git ignore check failed')
    ignored = sorted(set(ignored_result.stdout.decode().split('\0')) - {''})
    report = {'schema': 'hlsmith.release-completeness.v1', 'source': str(root),
              'runtime_files': len(required), 'untracked_runtime_files': sorted(required - tracked),
              'missing_runtime_files': sorted(name for name in required if not (root / name).is_file()),
              'missing_package_initializers': sorted(name for name in initializers if not (root / name).is_file()),
              'ignored_runtime_files': ignored, 'export_missing_files': [], 'export_changed_files': [],
              'wheel_missing_files': [], 'wheel_changed_files': [], 'wheel_unexpected_runtime_files': [],
              'provider_calls': 0, 'hls_jobs': 0}
    if export_root is not None:
        export = Path(export_root).resolve()
        report['export_root'] = str(export)
        for name in sorted(required):
            original, exported = root / name, export / name
            if not exported.is_file():
                report['export_missing_files'].append(name)
            elif original.is_file() and hashlib.sha256(original.read_bytes()).digest() != hashlib.sha256(exported.read_bytes()).digest():
                report['export_changed_files'].append(name)
    if wheel is not None:
        wheel = Path(wheel).resolve()
        report['wheel'] = str(wheel)
        report['wheel_sha256'] = hashlib.sha256(wheel.read_bytes()).hexdigest()
        expected_members = {name for name in required if _runtime_path(name)}
        with zipfile.ZipFile(wheel) as archive:
            members = {info.filename for info in archive.infolist() if not info.is_dir()}
            report['wheel_missing_files'] = sorted(expected_members - members)
            report['wheel_unexpected_runtime_files'] = sorted(name for name in members
                                                              if _runtime_path(name) and name not in expected_members)
            for name in sorted(expected_members & members):
                if (root / name).is_file() and archive.read(name) != (root / name).read_bytes():
                    report['wheel_changed_files'].append(name)
    report['passed'] = not any(report[key] for key in
                              ('untracked_runtime_files', 'missing_runtime_files', 'ignored_runtime_files',
                               'missing_package_initializers',
                               'export_missing_files', 'export_changed_files', 'wheel_missing_files',
                               'wheel_changed_files', 'wheel_unexpected_runtime_files'))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--export-root', type=Path)
    parser.add_argument('--wheel', type=Path)
    args = parser.parse_args(argv)
    report = audit(args.root, args.export_root, args.wheel)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
