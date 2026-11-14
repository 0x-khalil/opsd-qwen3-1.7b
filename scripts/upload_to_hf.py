"""Upload OPSD artifacts to the Hugging Face Hub.

Uploads one or more local folders into a single repo (as sub-folders), creating the
repo if needed.  Used by scripts/bootstrap_rental.sh, but also handy on its own:

    python scripts/upload_to_hf.py --repo_id my-user/opsd-qwen3-1.7b \
        --paths ~/runs/qwen31b_5090_package ~/runs/merged-qwen31b_5090

Authentication: either pass --token, or export HF_TOKEN, or run `hf auth login`.
"""

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo_id", required=True, help="e.g. my-user/opsd-qwen3-1.7b")
    ap.add_argument("--repo_type", default="model", choices=["model", "dataset"])
    ap.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    ap.add_argument("--private", default="1", choices=["0", "1"])
    ap.add_argument("--paths", nargs="+", required=True, help="folders to upload")
    ap.add_argument("--commit_message", default="OPSD Qwen3-1.7B run")
    args = ap.parse_args()

    if not args.token:
        raise SystemExit(
            "No token: pass --token, export HF_TOKEN, or run `hf auth login` on this machine."
        )

    api = HfApi(token=args.token)
    private = args.private == "1"

    try:
        repo = api.create_repo(
            repo_id=args.repo_id, repo_type=args.repo_type, exist_ok=True, private=private
        )
        # create_repo returns a RepoUrl (it has .repo_id); older/other versions
        # return an object with .repo_url
        repo_id = getattr(repo, "repo_id", None) or str(repo)
    except Exception as e:
        print(f"[hf] create_repo failed ({e}); assuming the repo already exists")
        repo_id = args.repo_id
    print(f"[hf] repo ready: https://huggingface.co/{repo_id}")

    for p in args.paths:
        folder = Path(p).expanduser().resolve()
        if not folder.exists():
            print(f"[hf] skip (missing): {folder}")
            continue
        size = sum(f.stat().st_size for f in folder.rglob("*") if f.is_file()) / 1e9
        print(f"[hf] uploading {folder} ({size:.2f} GB) -> {args.repo_id}/{folder.name}")
        api.upload_folder(
            folder_path=str(folder),
            repo_id=args.repo_id,
            repo_type=args.repo_type,
            path_in_repo=folder.name,
            commit_message=args.commit_message,
        )
        print(f"[hf] done: {args.repo_id}/{folder.name}")

    print(f"[hf] all uploaded -> https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
