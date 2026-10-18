"""Scheduling changes must not replace or reinterpret frozen measurements."""
import json
from pathlib import Path

import pytest

from scripts.resume_historical_cosim_parallel import sha, verify_campaign


@pytest.fixture
def frozen_campaign(tmp_path):
    output = tmp_path / 'campaign'
    source = tmp_path / 'frozen'
    raw = tmp_path / 'raw'
    output.mkdir()
    source.mkdir()
    (raw / 'measurements').mkdir(parents=True)
    evaluator = source / 'worker.py'
    evaluator.write_text('# frozen validator\n')
    measurement = raw / 'measurements' / 'saved.json'
    measurement.write_text(json.dumps({'cosim': {'status': 'timeout', 'kernel_runtime_cycles': None}}))
    manifest = output / 'manifest.json'
    manifest.write_text(json.dumps({'code_hashes': {'worker.py': sha(evaluator)}, 'raw_root': str(raw)}))
    (output / 'launch_receipt.json').write_text(json.dumps({'source_snapshot': str(source)}))
    request = {'manifest_sha256': sha(manifest), 'completed_measurement_hashes': {'saved.json': sha(measurement)}}
    return output, source, measurement, request


def test_parallel_preflight_preserves_failed_measurement(frozen_campaign):
    output, source, measurement, request = frozen_campaign
    paths = [p for p in output.parent.rglob('*') if p.is_file()]
    before = {p: p.read_bytes() for p in paths}
    _, actual_source = verify_campaign(output, request)
    assert actual_source == source
    assert {p: p.read_bytes() for p in paths} == before
    assert json.loads(measurement.read_text())['cosim']['status'] == 'timeout'


@pytest.mark.parametrize('target,expected', [
    ('manifest', 'campaign manifest changed'),
    ('validator', 'frozen validator changed'),
    ('measurement', 'completed measurement changed'),
])
def test_parallel_preflight_refuses_provenance_drift(frozen_campaign, target, expected):
    output, source, measurement, request = frozen_campaign
    path = {'manifest': output / 'manifest.json', 'validator': source / 'worker.py',
            'measurement': measurement}[target]
    path.write_text(path.read_text() + '\n')
    with pytest.raises(ValueError, match=expected):
        verify_campaign(output, request)
