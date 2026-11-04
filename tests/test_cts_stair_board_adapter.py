"""Validate offline adapter without SSH, SDK or robot processes."""
import importlib.util
from pathlib import Path
import hashlib
import pytest
ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('adapter', ROOT/'scripts/adapt_cts_stair_release.py')
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)

def fixture_release(tmp_path, monkeypatch):
    previous = b"# simulated known board revision\n"
    monkeypatch.setattr(adapter, "OLD", {name: hashlib.sha256(previous).hexdigest() for name in ("cts.py", "validate_cts_bundle.py")})
    for directory in ('src/m20_heightmap/m20_heightmap', 'install/m20_heightmap/lib/python3.8/site-packages/m20_heightmap'):
        target = tmp_path/directory/'cts.py'
        target.parent.mkdir(parents=True)
        target.write_bytes(previous)
    target = tmp_path/'scripts/validate_cts_bundle.py'
    target.parent.mkdir()
    target.write_bytes(previous)
    (tmp_path/"STAGED_NOT_ACCEPTED.txt").write_text("offline candidate")
    return tmp_path

def test_plan_covers_installed_and_source_and_is_idempotent(tmp_path, monkeypatch):
    release = fixture_release(tmp_path, monkeypatch)
    changes = adapter.plan(release)
    assert len(changes) == 5
    for target, value in changes.items():
        target.write_bytes(value)
    assert adapter.plan(release) == {}

def test_unknown_remote_edit_is_rejected_before_writes(tmp_path, monkeypatch):
    release = fixture_release(tmp_path, monkeypatch)
    target=release/'src/m20_heightmap/m20_heightmap/cts.py'
    target.write_text(target.read_text()+'\n# newer on robot\n')
    with pytest.raises(ValueError,match='Unknown/newer'):
        adapter.plan(release)
    assert not target.with_name('cts_stair_finetune.py').exists()

def test_unbuilt_candidate_rejected(tmp_path):
    with pytest.raises(ValueError,match='built release'):
        adapter.plan(tmp_path)


def test_apply_obeys_operator_lock(tmp_path, monkeypatch):
    import fcntl
    release = fixture_release(tmp_path/'release', monkeypatch)
    state = tmp_path/'state'; state.mkdir()
    monkeypatch.setattr(adapter, 'STATE_HOME', state)
    monkeypatch.setattr(adapter, 'PROC_ROOT', tmp_path/'empty_proc')
    with (state/'sdk.lock').open('a+') as owner:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            adapter.main(['--release',str(release),'--apply'])
    assert len(adapter.plan(release)) == 5
    adapter.main(['--release',str(release),'--apply'])
    assert adapter.plan(release) == {}


def test_partial_replace_rolls_back(tmp_path, monkeypatch):
    release = fixture_release(tmp_path/'release', monkeypatch)
    monkeypatch.setattr(adapter, 'STATE_HOME', tmp_path/'state')
    monkeypatch.setattr(adapter, 'PROC_ROOT', tmp_path/'empty_proc')
    original = Path.replace
    def failing(self, target):
        if target.name == 'cts.py':
            raise OSError('injected replacement failure')
        return original(self,target)
    monkeypatch.setattr(Path, 'replace', failing)
    with pytest.raises(OSError,match='injected'):
        adapter.main(['--release',str(release),'--apply'])
    assert len(adapter.plan(release)) == 5
    assert not list(release.rglob('*.stair_v2.tmp'))


def test_active_release_and_escaping_installed_symlink_rejected(tmp_path, monkeypatch):
    release = fixture_release(tmp_path/'release', monkeypatch)
    monkeypatch.setattr(adapter, 'ACTIVE_RELEASE', release)
    with pytest.raises(ValueError,match='never current'):
        adapter.plan(release)
    monkeypatch.setattr(adapter, 'ACTIVE_RELEASE', tmp_path/'other')
    path=release/'install/m20_heightmap/lib/python3.8/site-packages/m20_heightmap/cts.py'
    external=tmp_path/'active_cts.py'; external.write_bytes(path.read_bytes())
    path.unlink(); path.symlink_to(external)
    with pytest.raises(ValueError,match='escapes'):
        adapter.plan(release)
