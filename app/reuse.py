"""Reuse model files already on this PC instead of downloading them again.

Two places are searched before anything is downloaded:
  * the Hugging Face cache that other AI apps share (ComfyUI, text-generation-webui,
    Python scripts...), usually C:\\Users\\<you>\\.cache\\huggingface\\hub. The
    installer points HF_HOME at the install folder, so it passes the user's own
    settings on in QWEN3TTS_USER_HF_HOME and QWEN3TTS_USER_HF_HUB_CACHE.
  * other Qwen3-TTS Studio folders (QWEN3TTS_OTHER_INSTALLS, separated by ;): an
    earlier install, the copy the installer was started from, or -ModelsFrom.

A file found there is hard-linked into models\\ when it is on the same drive (it
then takes no extra space, and deleting either copy leaves the other intact), and
copied otherwise.
"""

from __future__ import annotations

import json
import os
import shutil
from collections.abc import Iterator
from pathlib import Path

from .config import CACHE_DIR, MODELS_DIR

USER_HF_HOME = "QWEN3TTS_USER_HF_HOME"
USER_HF_HUB_CACHE = "QWEN3TTS_USER_HF_HUB_CACHE"
OTHER_INSTALLS = "QWEN3TTS_OTHER_INSTALLS"


def _resolved(p: Path) -> Path | None:
    try:
        return p.expanduser().resolve()
    except OSError:
        return None


def _is_ours(p: Path) -> bool:
    """This install folder itself, or inside its own models or caches."""
    root = _resolved(MODELS_DIR.parent)
    if p == root:
        return True
    for own in (MODELS_DIR, CACHE_DIR):
        own = _resolved(own)
        if own and (p == own or own in p.parents):
            return True
    return False


def _unique_dirs(candidates) -> list[Path]:
    out: list[Path] = []
    for c in candidates:
        r = _resolved(Path(c)) if c else None
        if r and r.is_dir() and r not in out and not _is_ours(r):
            out.append(r)
    return out


def hub_caches() -> list[Path]:
    """Hugging Face caches on this PC other than the install folder's own."""
    env = os.environ
    return _unique_dirs([
        env.get(USER_HF_HUB_CACHE),
        env.get(USER_HF_HOME) and Path(env[USER_HF_HOME]) / "hub",
        Path.home() / ".cache" / "huggingface" / "hub",
    ])


def other_installs() -> list[Path]:
    return _unique_dirs(p.strip() for p in os.environ.get(OTHER_INSTALLS, "").split(";") if p.strip())


def _snapshots(repo_id: str) -> Iterator[Path]:
    for cache in hub_caches():
        snaps = cache / ("models--" + repo_id.replace("/", "--")) / "snapshots"
        if snaps.is_dir():
            yield from sorted((d for d in snaps.iterdir() if d.is_dir()), key=lambda d: d.stat().st_mtime, reverse=True)


def model_complete(d: Path) -> bool:
    """Whether a model folder has its config and every weights file, and no unfinished downloads."""
    if not (d / "config.json").exists():
        return False
    for folder in [d, *(s for s in d.iterdir() if s.is_dir() and not s.name.startswith("."))]:
        index = folder / "model.safetensors.index.json"
        if index.exists():
            try:
                shards = set(json.loads(index.read_text(encoding="utf-8"))["weight_map"].values())
            except (OSError, ValueError, KeyError, AttributeError):
                return False
            if not all((folder / s).exists() for s in shards):
                return False
    if not any(d.glob("*.safetensors")):
        return False
    partial = d / ".cache" / "huggingface" / "download"
    return not (partial.is_dir() and any(partial.rglob("*.incomplete")))


def find_model(repo_id: str) -> Path | None:
    """A complete copy of a Hugging Face model somewhere else on this PC."""
    name = repo_id.split("/")[-1]
    for root in other_installs():
        d = root / "models" / name
        if model_complete(d):
            return d
    for snap in _snapshots(repo_id):
        if model_complete(snap):
            return snap
    return None


def find_file(repo_id: str, filename: str, install_subdir: str = "models/gguf") -> Path | None:
    """One file of a Hugging Face repo (e.g. a GGUF), or of another install's install_subdir."""
    for root in other_installs():
        f = root / install_subdir / filename
        if f.is_file():
            return f
    for snap in _snapshots(repo_id) if repo_id else ():
        f = snap / filename
        if f.is_file():
            return f
    return None


def link_or_copy(src: Path, dst: Path) -> None:
    """Hard-link src to dst (no extra space), or copy it when that isn't possible."""
    src = src.resolve()  # the cache's snapshot files can be links to its blobs
    if dst.exists() and dst.stat().st_size == src.stat().st_size:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".reuse-tmp")
    tmp.unlink(missing_ok=True)
    try:
        os.link(src, tmp)
    except OSError:  # another drive, or a drive that can't link (FAT, exFAT)
        shutil.copy2(src, tmp)
    tmp.replace(dst)


def link_or_copy_tree(src: Path, dst: Path) -> int:
    """Bring every file of a model folder over (skipping download bookkeeping). Returns the bytes."""
    total = 0
    for f in sorted(src.rglob("*")):
        rel = f.relative_to(src)
        if any(part.startswith(".") for part in rel.parts) or not f.is_file():
            continue
        link_or_copy(f, dst / rel)
        total += f.stat().st_size
    return total


def reuse_model(repo_id: str, target: Path, log=print) -> bool:
    """Fill target from a copy elsewhere on this PC. True when something was reused."""
    src = find_model(repo_id)
    if not src:
        return False
    log(f"Found {repo_id} in {src}, so it isn't downloaded again.")
    size = link_or_copy_tree(src, target)
    log(f"Reused {size / 1024**3:.1f} GB.")
    return True


def reuse_file(repo_id: str, filename: str, target: Path, log=print, install_subdir: str = "models/gguf") -> bool:
    src = find_file(repo_id, filename, install_subdir)
    if not src:
        return False
    log(f"Found {filename} in {src.parent}, so it isn't downloaded again.")
    link_or_copy(src, target)
    return True
