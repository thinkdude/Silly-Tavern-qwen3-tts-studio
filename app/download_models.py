"""Download the model weights into models/ (about 4.5 GB per 1.7B model).

Usage:  python -m app.download_models            # the models settings.json uses
        python -m app.download_models --all      # also the other clone size
"""

from __future__ import annotations

import argparse
import shutil
import sys

from . import reuse
from .config import CLONE_MODEL_LARGE, CLONE_MODEL_SMALL, MODELS_DIR, load_settings, local_model_dir, setup_environment


# Written into a model's folder once a download has been checked against the Hub, so
# later installs skip it without going online.
COMPLETE_MARKER = ".download-complete"


def download(repo_id: str, log=print) -> None:
    """Get a model into models/, reusing what's already here or elsewhere on this PC."""
    target = local_model_dir(repo_id)
    if (target / COMPLETE_MARKER).exists() and is_downloaded(repo_id):
        log(f"{repo_id}: already downloaded")
        return
    log(f"\n--- {repo_id} -> {target}")
    if not is_downloaded(repo_id):
        reuse.reuse_model(repo_id, target, log)
    from huggingface_hub import snapshot_download

    try:
        # Downloads only what's missing; files already here are checked, not fetched again.
        snapshot_download(repo_id=repo_id, local_dir=str(target))
    except Exception as exc:
        if not is_downloaded(repo_id):
            raise
        log(f"Couldn't check {repo_id} online ({exc}). The copy here is complete, so it's kept.")
        return
    (target / COMPLETE_MARKER).write_text(repo_id + "\n", encoding="utf-8")


def is_downloaded(repo_id: str) -> bool:
    return reuse.model_complete(local_model_dir(repo_id))


def size_gb(repo_id: str) -> float:
    d = local_model_dir(repo_id)
    if not d.exists():
        return 0.0
    return round(sum(f.stat().st_size for f in d.rglob("*") if f.is_file()) / 1024**3, 1)


def remove(repo_id: str) -> float:
    """Delete a downloaded model. Returns the GB freed."""
    freed = size_gb(repo_id)
    d = local_model_dir(repo_id)
    if d.exists():
        shutil.rmtree(d)
    return freed


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--all", action="store_true", help="download both clone model sizes")
    args = ap.parse_args()

    setup_environment()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    settings = load_settings()
    repos = [settings["design_model"], settings["clone_model"]]
    if args.all:
        repos += [CLONE_MODEL_LARGE, CLONE_MODEL_SMALL]

    failed = []
    for repo in dict.fromkeys(repos):  # de-duplicate, keep order
        try:
            download(repo, log=lambda msg: print(msg, flush=True))
        except Exception as exc:  # keep going so one hiccup doesn't waste the other downloads
            print(f"!! Failed to download {repo}: {exc}")
            failed.append(repo)
    if failed:
        print("\nSome downloads failed. Check your internet connection and run install.bat again;")
        print("finished files are kept, so it picks up where it left off.")
        return 1
    print("\nAll models are here.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
