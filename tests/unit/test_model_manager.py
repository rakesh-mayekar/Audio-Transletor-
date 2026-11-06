import hashlib
import json

import pytest

from ai import model_manager, model_registry
from ai.model_manager import MANIFEST, ModelManager
from ai.model_registry import ModelFileSpec, ModelSpec
from core.errors import (
    ModelError,
    ModelIntegrityError,
    ModelMissingError,
    ModelNetworkNotAllowedError,
)

FILES = {
    "config.json": b'{"a":1}',
    "model.bin": b"\x01" * 4096,
    "tokenizer.json": b"tok",
    "vocabulary.txt": b"v",
}


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


@pytest.fixture
def spec(monkeypatch):
    s = ModelSpec(
        "tiny",
        "org/tiny",
        "rev1",
        tuple(
            ModelFileSpec(n, len(b), sha(b) if n == "model.bin" else None) for n, b in FILES.items()
        ),
    )
    monkeypatch.setitem(model_registry.REGISTRY, "tiny", s)
    monkeypatch.setattr(model_manager, "REGISTRY", model_registry.REGISTRY)
    return s


@pytest.fixture
def src(tmp_path):
    d = tmp_path / "src"
    d.mkdir()
    for n, b in FILES.items():
        (d / n).write_bytes(b)
    return d


def test_registry_is_pinned_multilingual_and_has_no_turbo():
    for name, s in model_registry.REGISTRY.items():
        assert len(s.revision) == 40 and s.license == "MIT"
        assert not name.endswith(".en") and "turbo" not in name
        mb = next(f for f in s.files if f.name == "model.bin")
        assert mb.sha256 and len(mb.sha256) == 64
    assert model_registry.REGISTRY["small"].parameters_m == 244


def test_unknown_model_rejected(tmp_path):
    with pytest.raises(ModelError):
        ModelManager(tmp_path).spec("nope")


def test_missing_model_gives_actionable_error_and_never_downloads(tmp_path, spec):
    m = ModelManager(tmp_path)
    with pytest.raises(ModelMissingError) as ei:
        m.resolve("tiny")
    assert "install_models" in ei.value.hint
    assert not m.status("tiny").installed


def test_download_requires_explicit_permission(tmp_path, spec):
    with pytest.raises(ModelNetworkNotAllowedError):
        ModelManager(tmp_path).install("tiny")
    assert not (tmp_path / "tiny").exists()


def test_install_from_directory_and_verify(tmp_path, spec, src):
    m = ModelManager(tmp_path / "models")
    msgs = []
    m.install_from_directory("tiny", src, progress=msgs.append)
    assert m.status("tiny", full=True).verified
    assert m.resolve("tiny") == tmp_path / "models" / "tiny"
    manifest = json.loads((tmp_path / "models" / "tiny" / MANIFEST).read_text())
    assert manifest["revision"] == "rev1" and manifest["files"]["model.bin"]["sha256"] == sha(
        FILES["model.bin"]
    )
    assert [p.name for p in (tmp_path / "models").iterdir()] == ["tiny"]  # no staging leftovers
    assert msgs


def test_corrupt_download_is_rejected_and_nothing_installed(tmp_path, spec, src):
    (src / "model.bin").write_bytes(b"\x02" * 4096)  # right size, wrong hash
    m = ModelManager(tmp_path / "models")
    with pytest.raises(ModelIntegrityError):
        m.install_from_directory("tiny", src)
    assert not (tmp_path / "models" / "tiny").exists()
    assert [p.name for p in (tmp_path / "models").iterdir()] == []


def test_wrong_size_rejected(tmp_path, spec, src):
    (src / "config.json").write_bytes(b"x")
    with pytest.raises(ModelIntegrityError):
        ModelManager(tmp_path / "m").install_from_directory("tiny", src)


def test_missing_source_file(tmp_path, spec, src):
    (src / "vocabulary.txt").unlink()
    with pytest.raises(ModelMissingError):
        ModelManager(tmp_path / "m").install_from_directory("tiny", src)


def test_tampering_detected_by_full_verify_only(tmp_path, spec, src):
    m = ModelManager(tmp_path / "m")
    m.install_from_directory("tiny", src)
    p = tmp_path / "m" / "tiny" / "model.bin"
    data = bytearray(p.read_bytes())
    data[10] ^= 0xFF  # same size, flipped bit
    p.write_bytes(bytes(data))
    m.verify("tiny", full=False)  # quick check cannot see it
    with pytest.raises(ModelIntegrityError):
        m.verify("tiny", full=True)
    with pytest.raises(ModelIntegrityError):
        m.resolve("tiny")  # loading always does the full check
    st = m.status("tiny", full=True)
    assert st.installed and not st.verified and st.error


def test_tampered_unpinned_file_detected_via_manifest(tmp_path, spec, src):
    m = ModelManager(tmp_path / "m")
    m.install_from_directory("tiny", src)
    (tmp_path / "m" / "tiny" / "tokenizer.json").write_bytes(b"xyz")  # same length as "tok"
    with pytest.raises(ModelIntegrityError):
        m.verify("tiny")


def test_revision_mismatch_detected(tmp_path, spec, src, monkeypatch):
    m = ModelManager(tmp_path / "m")
    m.install_from_directory("tiny", src)
    newer = ModelSpec(spec.name, spec.repo_id, "rev2", spec.files)
    monkeypatch.setitem(model_registry.REGISTRY, "tiny", newer)
    with pytest.raises(ModelIntegrityError):
        m.verify("tiny")


def test_corrupt_manifest(tmp_path, spec, src):
    m = ModelManager(tmp_path / "m")
    m.install_from_directory("tiny", src)
    (tmp_path / "m" / "tiny" / MANIFEST).write_text("{")
    with pytest.raises(ModelIntegrityError):
        m.verify("tiny")


def test_reinstall_replaces_and_remove(tmp_path, spec, src):
    m = ModelManager(tmp_path / "m")
    m.install_from_directory("tiny", src)
    m.install_from_directory("tiny", src)
    assert m.remove("tiny") is True and m.remove("tiny") is False
    assert not m.status("tiny").installed


def test_list_status_covers_registry(tmp_path):
    names = [s.name for s in ModelManager(tmp_path).list_status()]
    assert names == sorted(model_registry.REGISTRY)
