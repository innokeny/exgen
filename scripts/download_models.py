"""Pre-download base models into the HF cache.

Run this once before the first container start if you want to avoid the long
"first request" delay caused by HF Hub downloads:

    docker compose run --rm generator python scripts/download_models.py
"""

from __future__ import annotations

import os
import sys

from huggingface_hub import snapshot_download

# Keep this list in sync with app.config.SUPPORTED_MODELS.
DEFAULT_MODELS = [
    "Qwen/Qwen2.5-3B-Instruct",
]


def main(argv: list[str]) -> int:
    models = argv[1:] if len(argv) > 1 else DEFAULT_MODELS
    cache_dir = os.environ.get("HF_HOME", "/app/models_cache")
    token = os.environ.get("HF_TOKEN")

    for repo_id in models:
        print(f"[download] {repo_id} → {cache_dir}", flush=True)
        snapshot_download(
            repo_id=repo_id,
            cache_dir=cache_dir,
            token=token,
            local_files_only=False,
        )
    print("[download] done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
