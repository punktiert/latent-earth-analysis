"""Where the data lives. Everything is read from one folder (default ./data):

  data/hf/        a snapshot of the public release (scripts/download_release.py)
  data/derived/   what this repository computes from it (the analysis features)
  data/layout/    optional and not public: the exhibited atlas layout

Set LATENT_EARTH_DATA to keep the data elsewhere.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def data_root() -> Path:
    return Path(os.environ.get("LATENT_EARTH_DATA", REPO / "data"))


def resolve(rel) -> Path:
    p = Path(rel)
    return p if p.is_absolute() else data_root() / p
