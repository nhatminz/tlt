"""Training admission must not depend on archive copies or server launcher edits."""
import hashlib
import json
from pathlib import Path

import pytest

from scripts import check_source_manifest as checker


@pytest.fixture
def checkout(tmp_path, monkeypatch):
    root = tmp_path / 'TltReflex'
    files = []
    destinations = (
        'helper/opd_reflex.py', 'grpo_speculative.py', 'train_draft.py',
        'scripts/launch/train_model.sh', 'configs/b200.env',
    )
    for destination in destinations:
        data = ('reference ' + destination + '\n').encode()
        snapshot = 'sources/SpecNaacl/' + destination
        for relative in (destination, snapshot):
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        sha = hashlib.sha256(data).hexdigest()
        files.append(dict(destination=destination, snapshot=snapshot,
                          sha256=sha, port_sha256=sha, port_status='exact'))
    reference = root / 'sources/FastGRPO/reference.py'
    reference.parent.mkdir(parents=True, exist_ok=True)
    reference.write_bytes(b'archive only\n')
    files.append(dict(destination=None, snapshot=str(reference.relative_to(root)),
                      sha256=hashlib.sha256(reference.read_bytes()).hexdigest()))
    tlt = root / 'helper/tlt_generate.py'
    tlt.write_bytes(b'tlt algorithm\n')
    manifest = dict(files=files, tlt_algorithm_files={
        'helper/tlt_generate.py': hashlib.sha256(tlt.read_bytes()).hexdigest(),
    })
    (root / 'SOURCE_MANIFEST.json').write_text(json.dumps(manifest))
    monkeypatch.setattr(checker, 'ROOT', root)
    return root


def test_full_and_runtime_checks_clean_checkout(checkout):
    assert checker.check() == 6
    assert checker.check(runtime=True) == 4


@pytest.mark.parametrize('name', ['scripts/launch/train_model.sh', 'configs/b200.env'])
def test_server_operational_edits_do_not_block_runtime_but_remain_audited(checkout, name):
    manifest_before = (checkout / 'SOURCE_MANIFEST.json').read_bytes()
    (checkout / name).write_bytes(b'server-specific operational setting\n')
    assert checker.check(runtime=True) == 4
    with pytest.raises(ValueError, match='port differs from manifest: ' + name):
        checker.check()
    assert (checkout / 'SOURCE_MANIFEST.json').read_bytes() == manifest_before


def test_missing_archives_do_not_block_runtime_and_full_audit_lists_all(checkout):
    for snapshot in (checkout / 'sources').rglob('*'):
        if snapshot.is_file():
            snapshot.unlink()
    assert checker.check(runtime=True) == 4
    with pytest.raises(ValueError) as exc:
        checker.check()
    message = str(exc.value)
    assert message.count('source snapshot missing:') == 6
    assert 'sources/SpecNaacl/helper/opd_reflex.py' in message
    assert 'sources/SpecNaacl/scripts/launch/train_model.sh' in message
    assert '--runtime' in message


@pytest.mark.parametrize('name', [
    'helper/opd_reflex.py', 'grpo_speculative.py', 'train_draft.py',
    'helper/tlt_generate.py',
])
@pytest.mark.parametrize('missing', [False, True])
def test_runtime_rejects_missing_or_changed_algorithms_without_rewriting_hashes(
    checkout, name, missing,
):
    manifest_before = (checkout / 'SOURCE_MANIFEST.json').read_bytes()
    if missing:
        (checkout / name).unlink()
    else:
        (checkout / name).write_bytes(b'changed algorithm\n')
    with pytest.raises(ValueError, match='runtime integrity failed') as exc:
        checker.check(runtime=True)
    assert name in str(exc.value)
    assert (checkout / 'SOURCE_MANIFEST.json').read_bytes() == manifest_before


def test_runtime_cli_reports_all_core_problems_without_traceback(checkout, capsys):
    (checkout / 'helper/opd_reflex.py').unlink()
    (checkout / 'helper/tlt_generate.py').write_bytes(b'changed\n')
    assert checker.main(['--runtime']) == 1
    stderr = capsys.readouterr().err
    assert 'helper/opd_reflex.py' in stderr
    assert 'helper/tlt_generate.py' in stderr
    assert 'same TltReflex revision' in stderr
    assert 'Traceback' not in stderr


def test_actual_manifest_no_longer_requires_python_version_files():
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / 'SOURCE_MANIFEST.json').read_text())
    for entry in manifest['files']:
        assert not any(Path(entry.get(key) or '').name == '.python-version'
                       for key in ('destination', 'snapshot', 'source'))
