"""Download TRIBE v2 weights (config.yaml + best.ckpt) from HF into Conversion/cache/tribev2/.

No token is required (repo is public); HF_TOKEN is forwarded if set.
"""

import os
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO_ID = "facebook/tribev2"
FILES = ["config.yaml", "best.ckpt"]
CACHE_DIR = Path(__file__).resolve().parent / "cache" / "tribev2"


def main() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    token = os.environ.get("HF_TOKEN") or None
    for name in FILES:
        path = hf_hub_download(
            repo_id=REPO_ID,
            filename=name,
            local_dir=CACHE_DIR,
            token=token,
        )
        print(f"{name} -> {path}")


if __name__ == "__main__":
    main()
