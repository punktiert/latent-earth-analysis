"""Fetch what the analyses need from the public release (about 7 GB; images are fetched on demand by the figures).

    python scripts/download_release.py
"""

from huggingface_hub import snapshot_download

from latent_earth.paths import data_root

snapshot_download("Punktiert/Latent-Earth", repo_type="dataset", local_dir=data_root() / "hf",
                  allow_patterns=["data/*", "data/tiles/*", "embeddings/*/*", "results/*"])
print("release snapshot in", data_root() / "hf")
