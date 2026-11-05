import json
from pathlib import Path

import pytest

from app import download_models, gguf, reuse, transcribe
from app.config import DEFAULTS

REPO = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"


def make_model(d: Path, shards=None, size=1024):
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text("{}")
    if shards:
        (d / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {f"w{i}": s for i, s in enumerate(shards)}}))
        for s in shards:
            (d / s).write_bytes(b"\1" * size)
    else:
        (d / "model.safetensors").write_bytes(b"\1" * size)
    (d / "speech_tokenizer").mkdir(exist_ok=True)
    (d / "speech_tokenizer" / "model.safetensors").write_bytes(b"\2" * 10)
    return d


@pytest.fixture
def pc(tmp_path, monkeypatch):
    """A PC with a home folder, an install folder and no other copies yet."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: home)
    for var in (reuse.USER_HF_HOME, reuse.USER_HF_HUB_CACHE, reuse.OTHER_INSTALLS):
        monkeypatch.delenv(var, raising=False)
    models = tmp_path / "install" / "models"
    monkeypatch.setattr(reuse, "MODELS_DIR", models)
    monkeypatch.setattr(reuse, "CACHE_DIR", tmp_path / "install" / ".cache")
    monkeypatch.setattr(download_models, "local_model_dir", lambda repo: models / repo.split("/")[-1])
    return tmp_path, home, models


def hf_snapshot(cache: Path, repo: str) -> Path:
    return cache / ("models--" + repo.replace("/", "--")) / "snapshots" / "abc123"


def test_model_complete(tmp_path):
    assert not reuse.model_complete(tmp_path / "nothing")
    d = make_model(tmp_path / "m")
    assert reuse.model_complete(d)
    sharded = make_model(tmp_path / "s", shards=["a.safetensors", "b.safetensors"])
    assert reuse.model_complete(sharded)
    (sharded / "b.safetensors").unlink()
    assert not reuse.model_complete(sharded)
    partial = d / ".cache" / "huggingface" / "download"
    partial.mkdir(parents=True)
    (partial / "model.safetensors.incomplete").write_bytes(b"")
    assert not reuse.model_complete(d)


def test_finds_a_model_in_the_shared_hugging_face_cache(pc):
    _, home, _ = pc
    assert reuse.find_model(REPO) is None
    snap = make_model(hf_snapshot(home / ".cache" / "huggingface" / "hub", REPO))
    assert reuse.find_model(REPO) == snap.resolve()


def test_finds_a_model_in_a_custom_cache_and_another_install(pc, monkeypatch):
    tmp, _, _ = pc
    custom = make_model(hf_snapshot(tmp / "hfhome" / "hub", REPO))
    monkeypatch.setenv(reuse.USER_HF_HOME, str(tmp / "hfhome"))
    assert reuse.find_model(REPO) == custom.resolve()
    other = make_model(tmp / "old-install" / "models" / REPO.split("/")[-1])
    monkeypatch.setenv(reuse.OTHER_INSTALLS, f"{tmp / 'missing'};{tmp / 'old-install'}")
    assert reuse.find_model(REPO) == other.resolve()  # other installs come first


def test_never_reuses_its_own_folders(pc, monkeypatch):
    tmp, _, models = pc
    make_model(models / REPO.split("/")[-1])
    monkeypatch.setenv(reuse.OTHER_INSTALLS, str(tmp / "install"))
    assert reuse.other_installs() == []


def test_link_or_copy_tree(tmp_path):
    src = make_model(tmp_path / "src")
    (src / ".cache").mkdir()
    (src / ".cache" / "junk").write_text("x")
    (src / ".download-complete").write_text("x")
    dst = tmp_path / "dst"
    size = reuse.link_or_copy_tree(src, dst)
    assert size == 1024 + 2 + 10
    assert reuse.model_complete(dst)
    assert not (dst / ".cache").exists() and not (dst / ".download-complete").exists()
    # Same drive: a hard link, so no extra space.
    assert (dst / "model.safetensors").stat().st_ino == (src / "model.safetensors").stat().st_ino
    reuse.link_or_copy_tree(src, dst)  # again: nothing to do


def test_link_falls_back_to_copying(tmp_path, monkeypatch):
    src = tmp_path / "a.bin"
    src.write_bytes(b"abc")

    def no_links(*a):
        raise OSError("cross-device link")

    monkeypatch.setattr(reuse.os, "link", no_links)
    reuse.link_or_copy(src, tmp_path / "sub" / "b.bin")
    assert (tmp_path / "sub" / "b.bin").read_bytes() == b"abc"


@pytest.fixture
def fake_hub(monkeypatch):
    """snapshot_download that fills in a model, and records each call."""
    import huggingface_hub

    calls = []

    def snapshot_download(repo_id, local_dir, **kw):
        calls.append(repo_id)
        if not reuse.model_complete(Path(local_dir)):
            make_model(Path(local_dir))
        return local_dir

    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot_download)
    return calls


def test_download_skips_a_finished_model(pc, fake_hub):
    _, _, models = pc
    download_models.download(REPO, log=lambda m: None)
    assert fake_hub == [REPO]
    assert (models / REPO.split("/")[-1] / download_models.COMPLETE_MARKER).exists()
    logged = []
    download_models.download(REPO, log=logged.append)
    assert fake_hub == [REPO]  # not checked online again
    assert "already downloaded" in logged[0]


def test_download_reuses_a_copy_on_this_pc(pc, fake_hub, monkeypatch):
    tmp, home, models = pc
    snap = make_model(hf_snapshot(home / ".cache" / "huggingface" / "hub", REPO), size=4096)
    logged = []
    download_models.download(REPO, log=logged.append)
    target = models / REPO.split("/")[-1]
    assert (target / "model.safetensors").read_bytes() == (snap / "model.safetensors").read_bytes()
    assert any("isn't downloaded again" in m for m in logged)
    assert fake_hub == [REPO]  # then checked against the Hub, which found nothing missing


def test_download_works_offline_when_the_model_is_here(pc, monkeypatch):
    import huggingface_hub

    _, home, models = pc

    def offline(**kw):
        raise OSError("no internet")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", offline)
    with pytest.raises(OSError):
        download_models.download(REPO, log=lambda m: None)  # nothing here and no internet
    make_model(hf_snapshot(home / ".cache" / "huggingface" / "hub", REPO))
    logged = []
    download_models.download(REPO, log=logged.append)  # reused, and kept although it couldn't be checked
    assert download_models.is_downloaded(REPO)
    assert "Couldn't check" in logged[-1]
    assert not (models / REPO.split("/")[-1] / download_models.COMPLETE_MARKER).exists()


def test_gguf_files_and_koboldcpp_come_from_another_install(pc, monkeypatch, tmp_path):
    tmp, _, _ = pc
    old = tmp / "old"
    (old / "models" / "gguf").mkdir(parents=True)
    (old / ".tools" / "koboldcpp").mkdir(parents=True)
    settings = {**DEFAULTS, "device": "cuda:0"}
    for name in (settings["gguf_model"], settings["gguf_tokenizer"]):
        (old / "models" / "gguf" / name).write_bytes(b"gguf")
    (old / ".tools" / "koboldcpp" / "koboldcpp.exe").write_bytes(b"MZ")
    monkeypatch.setenv(reuse.OTHER_INSTALLS, str(old))
    monkeypatch.setattr(gguf, "GGUF_DIR", tmp / "install" / "models" / "gguf")
    monkeypatch.setattr(gguf, "KOBOLDCPP_DIR", tmp / "install" / ".tools" / "koboldcpp")
    monkeypatch.setattr(gguf, "_download", lambda *a: pytest.fail("KoboldCpp was downloaded"))
    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda **kw: pytest.fail("a GGUF file was downloaded"))
    assert len(gguf.missing(settings)) == 3
    gguf.install(settings, log=lambda m: None)
    assert gguf.missing(settings) == []


def test_speech_to_text_model_from_the_shared_cache(pc, monkeypatch):
    _, home, models = pc
    repo = transcribe.STT_MODEL
    monkeypatch.setattr(transcribe, "local_model_dir", lambda r: models / r.split("/")[-1])
    make_model(hf_snapshot(home / ".cache" / "huggingface" / "hub", repo))
    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda **kw: pytest.fail("downloaded"))
    settings = {**DEFAULTS}
    assert not transcribe.is_downloaded(settings)
    transcribe.download(settings, log=lambda m: None)
    assert transcribe.is_downloaded(settings)
