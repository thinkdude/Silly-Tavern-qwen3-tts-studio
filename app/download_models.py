"""Download the model weights into models/ (about 4.5 GB per 1.7B model).

Usage:  python -m app.download_models            # the models settings.json uses
        python -m app.download_models --all      # also the other clone size
"""

from __future__ import annotations

import argparse
import sys

from .config import CLONE_MODEL_LARGE, CLONE_MODEL_SMALL, MODELS_DIR, load_settings, local_model_dir, setup_environment


def download(repo_id: str) -> None:
    from huggingface_hub import snapshot_download

    target = local_model_dir(repo_id)
    print(f"\n--- {repo_id} -> {target}", flush=True)
    snapshot_download(repo_id=repo_id, local_dir=str(target))


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
            download(repo)
        except Exception as exc:  # keep going so one hiccup doesn't waste the other downloads
            print(f"!! Failed to download {repo}: {exc}")
            failed.append(repo)
    if failed:
        print("\nSome downloads failed. Check your internet connection and run install.bat again;")
        print("finished files are kept, so it picks up where it left off.")
        return 1
    print("\nAll models downloaded.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
