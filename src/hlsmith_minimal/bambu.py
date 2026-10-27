"""Optional Bambu C/C++ to Verilog and checked Verilator co-simulation.

The result contract is pinned to the 2024.10 MDPI implementation, not the
older tutorial's ``1 cycles`` format. This compares RTL with *candidate* C;
the caller must independently compare that candidate with the original C.
"""
import hashlib
import math
import os
from pathlib import Path
import re
import shutil
import xml.etree.ElementTree as ET

from .tools import run_command

_SOURCE = 'https://github.com/ferrandi/PandA-bambu/blob/2024.10/src/wrapper/simulation/SimulationTool.cpp'
_MAX_ARTIFACT = 4 * 1024 * 1024
_UNSAFE_PATH = re.compile(r'[^A-Za-z0-9_./+-]')


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def parse_results(text, *, expected_executions=None):
    """Read MDPI time intervals and checked testbench exit status.

    VerilatorWrapper uses a clock period of two simulation units. Bambu's
    DetermineCycles rounds each interval up before accumulating cycles.
    """
    lines = text.splitlines()
    if len(lines) != 2 or lines[1].strip() != '0':
        raise ValueError('expected intervals and a zero co-simulation testbench return code')
    # TestbenchFSM appends a comma after every completed top invocation.
    intervals = lines[0].removesuffix(',').split(',')
    cycles = []
    for interval in intervals:
        match = re.fullmatch(r'(\d+)\|(\d+)', interval.strip())
        if not match:
            raise ValueError('missing, aborted, timed out or malformed co-simulation interval')
        start, end = map(int, match.groups())
        if end <= start:
            raise ValueError('co-simulation interval must advance time')
        cycles.append((end - start + 1) // 2)
    if expected_executions is not None and len(cycles) != expected_executions:
        raise ValueError('executed call count differs from supplied XML test vectors')
    return {'rtl_cycles': sum(cycles), 'rtl_cycles_per_execution': cycles,
            'num_executions': len(cycles), 'cycle_scope': 'sum of all registered testbench top calls'}


def _read(path):
    if path.is_symlink() or not path.is_file() or path.stat().st_size > _MAX_ARTIFACT:
        raise ValueError('required co-simulation artifact is absent, linked or too large: ' + str(path))
    return path.read_text()


def _verification_wrapper(text):
    # Check the actual function body, not the generic comparison macro
    # definitions earlier in this generated file.
    body = text.split('__m_sim_start();', 1)
    if len(body) != 2 or '__m_sim_end();' not in body[1]:
        raise ValueError('generated wrapper has no checked MDPI execution body')
    golden = body[1].split('__m_sim_end();', 1)[0]
    if not re.search(r'#ifndef\s+CUSTOM_VERIFICATION\s+[^#;]+\([^;]*\);\s*#endif', golden):
        raise ValueError('generated wrapper has no enabled golden C call')
    comparisons = body[1].split('__m_sim_end();', 1)[1]
    if not re.search(r'\bm_(?:arg|retval|channel)cmp\s*\(', comparisons) or 'if(mismatch_count)' not in comparisons:
        raise ValueError('generated wrapper has no observable C/RTL value comparison')
    if re.search(r'^\s*#\s*define\s+CUSTOM_VERIFICATION\b', text, re.M):
        raise ValueError('automatic C/RTL verification was disabled')


def evaluate(candidate, testbench, *, out, top, include_files=None,
             executable='bambu', timeout=300, clock_ns=None):
    """Synthesize and co-simulate in a fresh directory; invoke no vendor flow.

    ``testbench`` is XML vectors or a C/C++ main calling ``top``. Extra files
    use safe relative names. No unchecked compiler/flow flags are accepted.
    """
    out = Path(out).resolve()
    result = {'status': 'unavailable', 'passed': False, 'backend': 'bambu',
              'compiler_backend': 'bambu', 'simulator': 'verilator',
              'measurement_origin': 'executed_rtl', 'metrics': {'rtl_cycles': None},
              'builtin_validation_enabled': False, 'commands': [],
              'cosimulation_max_ulp': 0, 'simulator_version': None,
              'abi_bits': 64, 'build_threads': 1,
              'qualification': 'Candidate C/RTL agreement for these testbench calls; original-C agreement is separate.',
              'protocol_source': _SOURCE, 'protocol': 'bambu-2024.10-mdpi'}
    result['requested_clock_ns'] = clock_ns
    if clock_ns is not None and (not math.isfinite(float(clock_ns)) or float(clock_ns) <= 0):
        raise ValueError('clock_ns must be positive and finite')
    if out.exists() and any(out.iterdir()):
        raise ValueError('Bambu output directory must be empty; refusing stale artifacts')
    if not re.fullmatch(r'[A-Za-z_]\w*', top or '') or top == 'main':
        result.update(status='unsupported', error='a named top function other than main is required for automatic verification')
        return result
    candidate, testbench = Path(candidate).resolve(), Path(testbench).resolve()
    if candidate.suffix.lower() not in ('.c', '.cc', '.cpp', '.cxx') or testbench.suffix.lower() not in ('.xml', '.c', '.cc', '.cpp', '.cxx'):
        result.update(status='unsupported', error='candidate C/C++ and a C/C++ or XML testbench are required')
        return result
    # Bambu itself emits shell/make scripts. Keep generated path arguments
    # literal even though our own subprocess invocation never uses a shell.
    if _UNSAFE_PATH.search(str(out)):
        result.update(status='unsupported', error='Bambu workspace path must contain only letters, digits, _ . / + -')
        return result
    include_files = include_files or {}
    inputs = {'candidate' + candidate.suffix.lower(): candidate,
              'testbench' + testbench.suffix.lower(): testbench}
    for name, path in include_files.items():
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts or not relative.parts or _UNSAFE_PATH.search(str(relative)) or str(relative) in inputs or relative.parts[0] in ('HLS_output', 'tmp', 'results.txt'):
            raise ValueError('include files require safe, nonreserved relative names')
        inputs[str(relative)] = Path(path).resolve()
    if any(path == out or out in path.parents for path in inputs.values()):
        raise ValueError('Bambu inputs must be outside the fresh output directory')
    try:
        for path in inputs.values():
            if re.search(r'^\s*#\s*define\s+CUSTOM_VERIFICATION\b', _read(path), re.M):
                raise ValueError('input disables automatic C/RTL verification')
        expected = None
        if testbench.suffix.lower() == '.xml':
            expected = len(ET.fromstring(_read(testbench)).findall('testbench'))
            if not expected:
                raise ValueError('XML testbench has no explicit input vectors')
        hashes = {name: _sha(path) for name, path in inputs.items()}
    except (OSError, ValueError, UnicodeError, ET.ParseError) as error:
        result.update(status='unsupported', error=str(error))
        return result
    result['input_hashes'] = hashes
    bambu, verilator = shutil.which(str(executable)), shutil.which('verilator')
    if not bambu or not verilator:
        result['error'] = 'Bambu and Verilator must both be installed explicitly'
        return result
    out.mkdir(parents=True, exist_ok=True)
    for name, path in inputs.items():
        destination = out/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
        if _sha(destination) != hashes[name]:
            result.update(status='error', error='input changed while creating Bambu snapshot')
            return result
    env = os.environ.copy()
    # Make/compiler environment overrides must not disable the generated
    # checker or silently change the requested ABI and bounded build width.
    result['cleared_build_environment'] = [name for name in ('CFLAGS', 'CXXFLAGS', 'CPPFLAGS', 'MAKEFLAGS', 'MFLAGS', 'GNUMAKEFLAGS') if name in env]
    for name in result['cleared_build_environment']:
        env.pop(name)
    (out/'tmp').mkdir()
    env.update(TMPDIR=str(out/'tmp'), CCACHE_DISABLE='1', CCACHE_TEMPDIR=str(out/'tmp'), OMP_NUM_THREADS='1')
    result['tools'] = {}
    for name, tool in (('bambu', bambu), ('verilator', verilator)):
        identity = run_command([tool, '--version'], cwd=out, timeout=5, env=env)
        result['tools'][name] = {'executable': tool, 'executable_sha256': _sha(tool),
                                  'version_output': (identity['stdout'] or identity['stderr']).strip(),
                                  'version_status': identity['status'],
                                  'identity_scope': 'invoked_compiler' if name == 'bambu' else 'outer_environment_candidate'}
        if identity['status'] != 'passed':
            result.update(status=identity['status'], error=name + ' version probe failed')
            return result
    if not re.search(r'\bPandA\s+2024\.10\b', result['tools']['bambu']['version_output']):
        result.update(status='unsupported', error='only the verified Bambu 2024.10 MDPI protocol is supported')
        return result
    argv = [bambu, 'candidate' + candidate.suffix.lower(), '--top-fname=' + top,
            '--writer=V', '--no-mixed-design', '--simulate', '--simulator=VERILATOR',
            '--generate-tb=testbench' + testbench.suffix.lower(), '-I.', '--tb-extra-gcc-options=-I.', '--max-ulp=0', '-m64']
    if clock_ns is not None:
        argv.append('--clock-period=' + str(float(clock_ns)))
    execution = run_command(argv, cwd=out, timeout=timeout, env=env)
    result['commands'].append(execution)
    result['workspace'] = str(out)
    if execution['status'] != 'passed' or execution['returncode'] != 0:
        result.update(status=execution['status'] if execution['status'] != 'passed' else 'failed',
                      error='Bambu synthesis/co-simulation failed; inspect the command result')
        return result
    try:
        wrapper = out/'HLS_output/simulation/mdpi_wrapper.cpp'
        _verification_wrapper(_read(wrapper))
        report = out/'results.txt'
        metrics = parse_results(_read(report), expected_executions=expected)
        testbench_rtl = out/'HLS_output/simulation/bambu_testbench.v'
        periods = re.findall(r'\bCLOCK_PERIOD\s*=\s*([0-9.]+)', _read(testbench_rtl))
        if not periods or any(float(period) != 2 for period in periods):
            raise ValueError('generated testbench clock period does not match the verified cycle conversion')
        # We do not enable profiling. Accepting a surprise profiling file
        # would change the scope of the cycle measurement in DetermineCycles.
        if (out/'profiling_results.txt').exists():
            raise ValueError('unexpected profiling output changes the cycle measurement scope')
        scripts = list(out.glob('simulate_*.sh'))
        script_text = '\n'.join(_read(path) for path in scripts)
        if not re.search(r'(?:^|\n)\s*verilator(?:_bin)?\s+--cc\b', script_text):
            raise ValueError('missing generated Verilator simulation script')
        if re.search(r'-D[\'"\\]*CUSTOM_VERIFICATION\b', script_text):
            raise ValueError('simulation build disables automatic value checking')
        result['simulator_invocation'] = 'verilator resolved inside the Bambu-generated script environment'
        rtl = list(out.glob('*.v'))
        if not rtl:
            raise ValueError('Bambu produced no top-level Verilog artifact')
        result.update(status='passed', passed=True, metrics=metrics, builtin_validation_enabled=True,
                      artifacts={str(path.relative_to(out)): _sha(path) for path in [report, wrapper, testbench_rtl, *scripts, *rtl]})
    except (OSError, ValueError) as error:
        result.update(status='failed', error=str(error))
    return result
