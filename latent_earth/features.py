"""Build the analysis features from the public release.

Every recorded channel is reduced to 128 whitened principal components (fitted
on a 30,000-image subsample with a fixed seed), L2-normalised and concatenated
to 640 numbers per image, so that channels enter every comparison with equal
capacity. Saves, under data/derived/:

  analysis_features.npy              (N, 640) float32
  analysis_features_meta.json        {"blocks": {channel: [start, end]}, ...}
  analysis_feature_tile_ids.parquet  row-aligned tile_id

The paper's numbers were computed from the same recordings in their production
order; the subsample that fits the PCA depends on row order, so features
rebuilt from the release agree with the released results to within that
sampling noise, not bit for bit.
"""

from __future__ import annotations

import json
import logging

import numpy as np
import polars as pl
import pyarrow.parquet as pq

from latent_earth import paths
from latent_earth.config import ANALYSIS_COMPONENTS, AnalysisConfig

log = logging.getLogger(__name__)


def _l2_normalize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + eps)


def pca_reduce(x: np.ndarray, target: int, *, subsample: int, seed: int = 0, whiten: bool = False) -> np.ndarray:
    """Mean-centre, PCA to `target` dims fitted on a subsample, return (N, target)."""
    from sklearn.decomposition import PCA

    n, d = x.shape
    mean = x.mean(axis=0, keepdims=True, dtype=np.float64).astype(np.float32)
    k = min(subsample, n)
    rows = np.random.default_rng(seed).choice(n, size=k, replace=False) if k < n else np.arange(n)
    fit = x[np.sort(rows)].astype(np.float32) - mean
    eff = max(1, min(target, d, k - 1 if whiten else k))
    pca = PCA(n_components=eff, svd_solver="randomized" if eff < min(fit.shape) else "full", random_state=seed, whiten=whiten).fit(fit)
    out = np.zeros((n, target), dtype=np.float32)
    for i in range(0, n, 20_000):                       # transform in chunks: the prompt channel is 15,360 wide
        out[i:i + 20_000, :eff] = pca.transform(x[i:i + 20_000].astype(np.float32) - mean)
    return out


def _load_channel(name: str, order: dict[str, int]) -> np.ndarray:
    out = None
    for f in sorted(paths.resolve(f"hf/embeddings/{name}").glob("*.parquet")):
        t = pq.read_table(f)
        col = t.column(name).combine_chunks()
        arr = np.asarray(col.flatten()).reshape(-1, col.type.list_size)
        if out is None:
            out = np.zeros((len(order), arr.shape[1]), np.float16)
        ids = t.column("tile_id").to_pylist()
        keep = [i for i, tid in enumerate(ids) if tid in order]
        out[[order[ids[i]] for i in keep]] = arr[keep]
    if out is None:
        raise FileNotFoundError(f"no embeddings for channel {name}: run scripts/download_release.py")
    return out


def build_analysis_features(config: AnalysisConfig, *, first_n: int | None = None, pca_fit_subsample: int = 30_000):
    keys = list(ANALYSIS_COMPONENTS)
    have = None                                          # tiles present in every channel (3 of 200,000 are not)
    for k in keys:
        ids = set()
        for f in sorted(paths.resolve(f"hf/embeddings/{k}").glob("*.parquet")):
            ids.update(pq.read_table(f, columns=["tile_id"]).column("tile_id").to_pylist())
        have = ids if have is None else have & ids
    tiles = pl.read_parquet(paths.resolve(config.manifest_parquet), columns=["tile_id"])["tile_id"].to_list()
    tile_ids = [t for t in tiles if t in have][:first_n]
    order = {t: i for i, t in enumerate(tile_ids)}
    parts, blocks, off = [], {}, 0
    for key in keys:
        target, weight = ANALYSIS_COMPONENTS[key]
        red = pca_reduce(_load_channel(key, order), target, subsample=pca_fit_subsample, seed=config.seed, whiten=True)
        parts.append((_l2_normalize(red) * float(weight)).astype(np.float32))
        blocks[key] = (off, off + target)
        off += target
        log.info("  %-14s -> PCA %d whitened", key, target)
    return np.concatenate(parts, axis=1), tile_ids, blocks


def save_analysis_features(config: AnalysisConfig, feats, tile_ids, blocks) -> None:
    fp = paths.resolve(config.features_npy)
    fp.parent.mkdir(parents=True, exist_ok=True)
    np.save(fp, feats)
    pl.DataFrame({"tile_id": tile_ids}).write_parquet(paths.resolve(config.tile_ids_parquet))
    paths.resolve(config.features_meta).write_text(json.dumps(
        {"blocks": {k: list(v) for k, v in blocks.items()}, "dim": int(feats.shape[1]), "n": int(feats.shape[0]),
         "whitened": True, "seed": config.seed}, indent=2), encoding="utf-8")
    log.info("Saved analysis features %s", feats.shape)


def load_analysis_features(config: AnalysisConfig):
    feats = np.load(paths.resolve(config.features_npy))
    tile_ids = pl.read_parquet(paths.resolve(config.tile_ids_parquet))["tile_id"].to_list()
    meta = json.loads(paths.resolve(config.features_meta).read_text(encoding="utf-8"))
    return feats, tile_ids, {k: tuple(v) for k, v in meta["blocks"].items()}
