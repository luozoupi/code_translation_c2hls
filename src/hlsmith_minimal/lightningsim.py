"""Optional LightningSim CLI adapter; estimates are never correctness checks.

Protocol: upstream v0.2.5 main.py run_simple prints one unindented
[start-end] module line and tab-indented children. The upstream CLI does not
expose JSON/version flags or verify the testbench return code. Only a copied,
self-contained Vitis workspace is passed to the external executable.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import stat
import xml.etree.ElementTree as ET

from .tools import run_command

_PROTOCOL = 'lightningsim-cli-module-tree-v0.2.5'
_SOURCE = 'https://github.com/sharc-lab/LightningSim/blob/91edbb2a35b3ef06ecf68e16bfcc94bebb1915fa/backend/lightningsim/main.py#L394-L435'
_MAX_COPY_BYTES = 512 * 1024 * 1024
_MAX_COPY_FILES = 10000
_MAX_COPY_ENTRIES = 20000
_ROOT = re.compile(r'\[\s*(\d+)-\s*(\d+)\] ([^\r\n]+)')


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()



def _package_identity(executable):
    found = []
    for path in (Path(executable).parent.parent / 'conda-meta').glob('lightningsim-*.json'):
        try:
            if path.stat().st_size > 2 * 1024 * 1024:
                continue
            value = json.loads(path.read_text())
            if isinstance(value, dict) and value.get('name') == 'lightningsim':
                found.append({key: value[key] for key in
                    ('name', 'version', 'build', 'build_number', 'sha256', 'md5', 'subdir') if key in value})
                found[-1].update(metadata_path=str(path), metadata_sha256=_sha(path))
        except (OSError, ValueError):
            continue
    return found[0] if len(found) == 1 else None

def _manifest(root):
    files, size = {}, 0
    for index, path in enumerate(root.rglob('*')):
        if index >= _MAX_COPY_ENTRIES:
            raise ValueError('workspace exceeds copy entry limit (20000 entries)')
        info = path.lstat()
        mode = info.st_mode
        if stat.S_ISREG(mode):
            size += info.st_size
            if len(files) >= _MAX_COPY_FILES or size > _MAX_COPY_BYTES:
                raise ValueError('workspace exceeds copy limit (10000 files / 512 MiB)')
            files[str(path.relative_to(root))] = _sha(path)
        elif not stat.S_ISDIR(mode):
            raise ValueError('workspace symlinks/special files are unsupported')
    return files


def _project_files(solution, workspace, xml):
    files = xml.find('{*}files')
    if files is None or not xml.get('top'):
        raise ValueError('hls.app lacks its top function or file list')
    resolved, testbenches = [], 0
    for entry in files.findall('{*}file'):
        tb = entry.get('tb') in ('1', 'true')
        base = solution if tb else solution.parent.parent
        path = (base / entry.attrib['name']).resolve()
        if not path.is_relative_to(workspace) or not path.exists():
            raise ValueError('hls.app has an external or missing file dependency')
        # Absolute flags could read the original tree even after cloning. A
        # future adapter may support rewriting them; this small one refuses.
        for flag in shlex.split(entry.get('cflags', '')):
            if '/' in flag or '..' in flag:
                raise ValueError('path-bearing compiler flags are unsupported')
        resolved.append((entry, path))
        testbenches += tb and path.suffix.lower() in ('.c', '.cpp', '.cc', '.cxx')
    if not testbenches:
        raise ValueError('hls.app does not declare a source testbench')
    simulation = xml.find('{*}Simulation')
    if simulation is not None:
        flow = simulation.find('{*}SimFlow')
        options = simulation.get('argv', '') + ' ' + (flow.get('ldflags', '') if flow is not None else '')
        if any('/' in token or '..' in token for token in shlex.split(options)):
            raise ValueError('path-bearing testbench arguments/linker flags are unsupported')
    return resolved


def parse_interval(stdout):
    """Never mistake an indented child or a partial/error log for the root."""
    lines = [line for line in stdout.splitlines() if line and not line[0].isspace()]
    roots = [_ROOT.fullmatch(line) for line in lines]
    if len(roots) != 1 or roots[0] is None:
        raise ValueError('expected exactly one unindented module interval')
    start, end = map(int, roots[0].group(1, 2))
    if end <= start:
        raise ValueError('top module interval must have positive integral duration')
    return {'module': roots[0].group(3), 'start': start, 'end': end, 'cycles': end - start}


def estimate(solution, *, out, workspace=None, executable='lightningsim', timeout=120,
             provenance=None, hls_root=None):
    """Estimate a copied solution; caller provenance is recorded, not invented.

    Supply workspace for projects with source/TB files outside the project
    directory. No RTL correctness, physical timing, or ranking qualification is
    implied, even when the external estimator exits successfully.
    """
    result = {'status': 'unavailable', 'passed': False, 'estimate_available': False,
        'backend': 'lightningsim', 'measurement_origin': 'trace_estimate', 'correctness_checked': False,
        'metrics': {'trace_estimate_cycles': None}, 'version': None,
        'version_note': 'Upstream CLI has no version flag; executable/help hashes identify this invocation.',
        'version_compatibility': 'uncalibrated',
        'interval_scope': 'top module interval from the registered trace; upstream may analyze only the first top-level call',
        'provenance': {'cli_profile': _PROTOCOL, 'protocol_source': _SOURCE,
                       'declared_inputs': dict(provenance or {})},
        'qualification': 'Trace-based diagnostic estimate only; upstream CLI does not establish testbench correctness or executed RTL cycles.'}
    executable = shutil.which(str(executable))
    if executable is None:
        result['error'] = 'LightningSim executable is unavailable; install/configure it separately'
        return result
    executable = str(Path(executable).resolve())
    result['provenance']['executable'] = {'path': executable, 'sha256': _sha(executable)}
    identity = _package_identity(executable)
    if identity is not None:
        result['version'] = identity.get('version')
        result['version_note'] = 'Version read from the external CLI environment conda package record.'
        result['provenance']['package_identity'] = identity
    configured_hls = hls_root or os.environ.get('XILINX_HLS')
    if not configured_hls or not Path(configured_hls).is_dir():
        result['error'] = 'A valid Vitis HLS root (hls_root or XILINX_HLS) is required'
        return result
    solution = Path(solution).resolve()
    workspace = Path(workspace).resolve() if workspace is not None else solution.parent
    out = Path(out).resolve()
    if out == workspace or out.is_relative_to(workspace) or workspace.is_relative_to(out):
        raise ValueError('estimator output and original workspace must be disjoint')
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise ValueError('estimator output directory must be empty')
    try:
        if not solution.is_dir() or not solution.is_relative_to(workspace):
            raise ValueError('solution must be a directory inside its workspace')
        for name in ('.autopilot/db/dut.hcp', '.autopilot/db/a.o.3.bc'):
            if not (solution / name).is_file():
                raise ValueError('solution is missing required synthesis bitcode/sentinel')
        app = solution.parent / 'hls.app'
        if not app.is_relative_to(workspace):
            raise ValueError('workspace must contain parent hls.app')
        xml = ET.parse(app)
        dependencies = _project_files(solution, workspace, xml.getroot())
        manifest = _manifest(workspace)
        copied = out / 'workspace'
        shutil.copytree(workspace, copied)
        if _manifest(copied) != manifest:
            raise ValueError('workspace changed while copied')
        for entry, path in dependencies:
            entry.set('name', str(copied / path.relative_to(workspace)))
        copied_app = copied / app.relative_to(workspace)
        xml.write(copied_app, encoding='utf-8', xml_declaration=True)
    except (ValueError, OSError, KeyError, ET.ParseError) as error:
        result.update(status='unsupported', error=str(error))
        return result
    result['provenance'].update({'original_workspace': str(workspace), 'original_solution': str(solution),
        'workspace_files': manifest, 'workspace_manifest_sha256': hashlib.sha256(
            json.dumps(manifest, sort_keys=True).encode()).hexdigest(), 'copied_hls_app_sha256': _sha(copied_app)})
    environment = os.environ.copy()
    environment['XILINX_HLS'] = str(Path(configured_hls).resolve())
    temporary = out / 'tmp'
    temporary.mkdir()
    environment.update(TMPDIR=str(temporary), CCACHE_DISABLE='1', CCACHE_TEMPDIR=str(temporary))
    environment['PATH'] = os.pathsep.join((str(Path(executable).parent),
        str(Path(configured_hls).resolve() / 'bin'), environment.get('PATH', '')))
    result['provenance'].update(effective_hls_root=environment['XILINX_HLS'],
        hls_root_source='argument' if hls_root is not None else 'environment', execution_tmpdir=str(temporary))
    help_result = run_command([executable, '--help'], cwd=out, timeout=min(10, timeout), env=environment)
    result['provenance']['help_sha256'] = hashlib.sha256(help_result['stdout'].encode()).hexdigest()
    if help_result['status'] != 'passed' or '--cli' not in help_result['stdout']:
        result.update(status='unsupported', error='executable does not expose the supported --cli interface')
        return result
    execution = run_command([executable, str(copied / solution.relative_to(workspace)), '--cli'],
                            cwd=out, timeout=timeout, env=environment)
    result['execution'] = execution
    if execution['status'] != 'passed' or execution['returncode'] != 0:
        result.update(status=execution['status'] if execution['status'] != 'passed' else 'failed',
                      error='LightningSim failed or timed out')
        return result
    try:
        interval = parse_interval(execution['stdout'])
    except ValueError as error:
        result.update(status='error', error=str(error))
        return result
    result.update(status='estimated', estimate_available=True, interval=interval,
                  metrics={'trace_estimate_cycles': interval['cycles']})
    return result
