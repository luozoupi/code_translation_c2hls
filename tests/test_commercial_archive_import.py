import hashlib
import importlib.util
import io
from pathlib import Path
import tarfile

import pytest


SPEC = importlib.util.spec_from_file_location('commercial_import', Path(__file__).resolve().parents[1] /
                                            'scripts/import_commercial_sweep_20261008.py')
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


@pytest.mark.parametrize('name', ['/absolute', '../escape', 'c2hls_newskills_commercial_sweep/../escape',
                                  'c2hls_newskills_commercial_sweep/a\\b',
                                  'c2hls_newskills_commercial_sweep/./a', 'wrong-root/a'])
def test_unsafe_paths_rejected(name):
    with pytest.raises(ValueError):
        MODULE.safe_name(name)


def write_tar(path, payload=b'kernel', expected=None, special=None):
    root = MODULE.ARCHIVE_ROOT
    with tarfile.open(path, 'w:gz') as stream:
        content = (expected or hashlib.sha256(payload).hexdigest()) + '  results/kernel.cpp\n'
        info = tarfile.TarInfo(root + '/MANIFEST.sha256')
        raw = content.encode()
        info.size = len(raw)
        stream.addfile(info, io.BytesIO(raw))
        info = tarfile.TarInfo(root + '/results/kernel.cpp')
        info.size = len(payload)
        stream.addfile(info, io.BytesIO(payload))
        if special is not None:
            stream.addfile(special)


def test_import_verifies_all_files_and_preserves_bytes(tmp_path):
    source, output = tmp_path / 'archive.tgz', tmp_path / 'import'
    write_tar(source)
    receipt = MODULE.import_archive(source, output)
    target = Path(receipt['payload_root']) / 'results/kernel.cpp'
    assert target.read_bytes() == b'kernel'
    assert target.stat().st_mode & 0o222 == 0
    assert receipt['manifest_verified_files'] == 1
    assert receipt['all_payload_checksums_verified']
    # Restore directory write permission for pytest cleanup only.
    for directory in [target.parent, target.parent.parent, target.parent.parent.parent]:
        directory.chmod(0o755)
    with pytest.raises(FileExistsError):
        MODULE.import_archive(source, output)


def test_tampered_payload_rejected(tmp_path):
    source = tmp_path / 'archive.tgz'
    write_tar(source, expected='0' * 64)
    with pytest.raises(ValueError, match='Checksum mismatch'):
        MODULE.import_archive(source, tmp_path / 'import')


@pytest.mark.parametrize('kind', [tarfile.SYMTYPE, tarfile.LNKTYPE, tarfile.FIFOTYPE])
def test_links_and_specials_rejected_before_extraction(tmp_path, kind):
    source, output = tmp_path / 'archive.tgz', tmp_path / 'import'
    member = tarfile.TarInfo(MODULE.ARCHIVE_ROOT + '/link')
    member.type = kind
    member.linkname = '/tmp/elsewhere'
    write_tar(source, special=member)
    with pytest.raises(ValueError, match='Link or special'):
        MODULE.import_archive(source, output)
    assert not output.exists()


def test_unmanifested_regular_file_rejected(tmp_path):
    source, output = tmp_path / 'archive.tgz', tmp_path / 'import'
    member = tarfile.TarInfo(MODULE.ARCHIVE_ROOT + '/extra')
    write_tar(source, special=member)
    with pytest.raises(ValueError, match='Manifest coverage mismatch'):
        MODULE.import_archive(source, output)
    assert not output.exists()


def test_cycle_equality_does_not_bind_a_different_source():
    report = {'latency_cycles': 100}
    result = {'final_report': report,
              'generated_step_history': [{'code': 'different', 'report': report, 'csim': {'passed': True}}]}
    assert MODULE.candidate_binding(result, 'saved', {})['status'] == 'source_report_unresolved'


def test_same_code_with_different_report_does_not_bind():
    result = {'final_report': {'latency_cycles': 100},
              'generated_step_history': [{'code': 'saved', 'report': {'latency_cycles': 101},
                                           'csim': {'passed': True}}]}
    assert MODULE.candidate_binding(result, 'saved', {})['status'] == 'source_report_unresolved'


def test_explicit_promoted_baseline_requires_assistant_source_and_report():
    report = {'latency_cycles': 100}
    result = {'final_report': report, 'baseline_report': report, 'baseline_csim': {'passed': True},
              'best_so_far_promotion': {'promoted': True, 'from_step_index': -1, 'from_step_name': 'baseline'}}
    history = {'messages': [{'role': 'assistant', 'content': '```cpp\nsaved\n```'}]}
    assert MODULE.candidate_binding(result, 'saved', history)['status'] == 'source_report_and_csim_linked'
    history['messages'][0]['role'] = 'user'
    assert MODULE.candidate_binding(result, 'saved', history)['status'] == 'source_report_unresolved'


def test_source_match_does_not_imply_csim_pass():
    result = {'hls_code': 'saved', 'synth_report': {'latency_cycles': 100}, 'csim': {'passed': False}}
    assert MODULE.candidate_binding(result, 'saved', {})['status'] == 'source_report_linked_csim_unconfirmed'
