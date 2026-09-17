"""Analysis configuration and the channel-comparable feature recipe.

All five recorded channels enter at the same size and weight (128 whitened
principal components each), so no channel carries more label information
purely by capacity.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

# Every modality at the SAME dim + weight — the whole point.
ANALYSIS_COMPONENTS: dict[str, tuple[int, float]] = {
    "dino":          (128, 1.0),
    "siglip":        (128, 1.0),
    "flux_residual": (128, 1.0),
    "t5_prompt":     (128, 1.0),
    "vlm_image":     (128, 1.0),
}
ANALYSIS_DIM = sum(t for t, _ in ANALYSIS_COMPONENTS.values())   # 640

WALL_SAMPLES: tuple[int, ...] = (0, 1, 2, 3)
REPLICATE_SAMPLE: int = 4

# Statistical defaults (pre-specified; see PLAN.md)
N_PERMUTATIONS = 999
N_BOOTSTRAP = 1000
KNN_K = 10
DENSITY_K = 20
SEED = 20260713            # the plan-approval date; fixed forever


@dataclass
class AnalysisConfig:
    """Paths + knobs for the paper analyses."""

    # inputs (relative to the data folder unless absolute)
    features_npy: str = "derived/analysis_features.npy"
    features_meta: str = "derived/analysis_features_meta.json"
    tile_ids_parquet: str = "derived/analysis_feature_tile_ids.parquet"
    manifest_parquet: str = "hf/data/tiles/*.parquet"      # the release's tile table
    places_parquet: str = "hf/data/places.parquet"
    captions_dir: str = "hf/data/captions.parquet"

    # outputs — the results contract read by paper/figures/*
    out_dir: str = "results"        # relative to CWD (the repo checkout)

    seed: int = SEED
    n_permutations: int = N_PERMUTATIONS
    n_bootstrap: int = N_BOOTSTRAP

    @classmethod
    def from_env(cls, **overrides) -> "AnalysisConfig":
        kwargs: dict = {}
        if (v := os.environ.get("LATENT_EARTH_OUT")) is not None:
            kwargs["out_dir"] = v
        kwargs.update(overrides)
        return cls(**kwargs)

    def out_path(self, name: str) -> Path:
        p = Path(self.out_dir) / name
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def write_result(self, name: str, payload: dict) -> Path:
        """Write a result JSON with provenance stamped in."""
        p = self.out_path(name)
        payload = {"_provenance": {"seed": self.seed,
                                   "n_permutations": self.n_permutations,
                                   "n_bootstrap": self.n_bootstrap},
                   **payload}
        p.write_text(json.dumps(payload, indent=2, default=_json_default),
                     encoding="utf-8")
        return p


def _json_default(o):
    import numpy as np
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON-serializable: {type(o)}")
