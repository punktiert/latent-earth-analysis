"""Which places have four identical second-stage prompts, and what the framing phrases do.

For 30,366 places the four second-stage prompts are identical
("<features>. <name>.") and only the seed varies. For 9,634 places, assigned
at random, each attempt also carries a framing phrase (roof detail, domestic
interior, single house, multi-storey city building) with capped feature
counts. A place belongs to the second subset iff its four prompts differ.
Consistency, retrieval and spread statistics use the first subset only.

Writes:
  results/prompt_subsets.parquet   place_id, n_prompts, identical_prompts
  results/prompt_subsets.json      dispersion by subset; the 4 x 4 mean distance
      between attempts per subset (640-D analysis features, cosine distance,
      attempt means removed); baselines.

    python scripts/prompt_subsets.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from latent_earth.discover import find_data, find_layout  # noqa: E402

from latent_earth.paths import data_root  # noqa: E402
RESULTS = ROOT / "results"
FRAMES = ["roof detail", "interior", "house", "city building"]
SUBSETS = {"identical_prompts": True, "framed_prompts": False}


def tile_table() -> pl.DataFrame:
    cols = ["place_id", "sample_idx", "tile_id", "step2_prompt"]
    release = data_root() / "hf" / "data" / "tiles"
    if release.exists():
        return pl.read_parquet(release / "*.parquet", columns=cols)
    return pl.read_parquet(find_data("manifest.parquet"), columns=cols)


def main() -> int:
    m4 = tile_table().filter(pl.col("sample_idx") <= 3)
    sub = (m4.group_by("place_id").agg(pl.col("step2_prompt").n_unique().alias("n_prompts"))
             .with_columns((pl.col("n_prompts") == 1).alias("identical_prompts")))
    sub.write_parquet(RESULTS / "prompt_subsets.parquet")
    d = (pl.read_parquet(RESULTS / "dispersion_per_place.parquet").select(["place_id", "dispersion_concat"])
           .join(sub, on="place_id"))
    out = {"n_places": d.height}
    for name, flag in SUBSETS.items():
        x = d.filter(pl.col("identical_prompts") == flag)["dispersion_concat"].to_numpy()
        out[name] = {"n": int(x.size), "share": float(x.size / d.height), "dispersion_median": float(np.median(x)),
                     "p10": float(np.percentile(x, 10)), "p90": float(np.percentile(x, 90))}
        print(f"{name}: n={x.size} share={x.size / d.height:.3f} median={np.median(x):.3f}")

    # ---- attempt x attempt distances in the analysis features
    feats = find_layout("analysis_features.npy")
    ids = pl.read_parquet(feats.with_name("analysis_feature_tile_ids.parquet"))["tile_id"].to_list()
    F = np.asarray(np.load(feats), dtype=np.float32)
    F /= np.maximum(np.linalg.norm(F, axis=1, keepdims=True), 1e-8)
    pos = {t: i for i, t in enumerate(ids)}
    t4 = m4.join(sub, on="place_id").with_columns(
        pl.col("tile_id").map_elements(lambda t: pos.get(t, -1), return_dtype=pl.Int64).alias("row")).filter(pl.col("row") >= 0)
    full = t4.group_by("place_id").len().filter(pl.col("len") == 4)["place_id"]
    t4 = t4.filter(pl.col("place_id").is_in(full.to_list())).sort(["place_id", "sample_idx"])
    X = F[t4["row"].to_numpy()].reshape(-1, 4, F.shape[1])
    same = t4.filter(pl.col("sample_idx") == 0)["identical_prompts"].to_numpy()
    Xc = X - X.mean(axis=0, keepdims=True)                      # attempt means out, as in dispersion.json
    Xc /= np.maximum(np.linalg.norm(Xc, axis=2, keepdims=True), 1e-8)
    for name, flag in SUBSETS.items():
        sel = Xc[same == flag]
        M = np.array([[float(1 - (sel[:, i] * sel[:, j]).sum(1).mean()) for j in range(4)] for i in range(4)])
        out[name]["attempt_matrix"] = M.round(4).tolist()
        out[name]["attempt_labels"] = ["seed 1", "seed 2", "seed 3", "seed 4"] if flag else FRAMES
        out[name]["mean_offdiag"] = float(M[np.triu_indices(4, 1)].mean())
    out["baseline_nn20_local_density"] = float(pl.read_parquet(RESULTS / "poverty_per_place.parquet")["local_density"].mean())
    (RESULTS / "prompt_subsets.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("wrote prompt_subsets.parquet / .json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
