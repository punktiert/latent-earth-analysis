"""Shared analysis machinery: place-level tables, kNN, permutation + bootstrap.

Statistical stance (pre-specified): the UNIT OF ANALYSIS IS THE PLACE — the
four wall samples share one Step-1 image and are not independent draws. Every
headline quantity is computed on place-level vectors (mean of samples 0-3,
re-normalized), uncertainty via bootstrap over places, and significance via
label-permutation nulls. No parametric tests.
"""

from __future__ import annotations

import logging

import numpy as np
import polars as pl

from latent_earth import paths
from latent_earth.config import WALL_SAMPLES, AnalysisConfig

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Assembly: tiles -> place-level matrix + covariates
# ---------------------------------------------------------------------------
def tile_table(config: AnalysisConfig, tile_ids: list[str]) -> pl.DataFrame:
    """(tile_id, place_id, sample_idx) + place covariates, row-aligned to feats."""
    ids = pl.DataFrame({"tile_id": tile_ids}).with_row_index("row_idx")
    parts = ids["tile_id"].str.split_exact("__", 1)
    ids = ids.with_columns([
        parts.struct.field("field_0").alias("place_id"),
        parts.struct.field("field_1").cast(pl.Int8).alias("sample_idx"),
    ])
    places = pl.read_parquet(paths.resolve(config.places_parquet))
    keep = [c for c in ("place_id", "lat", "lng", "population", "population_band",
                        "koppen_class", "un_region", "country", "iso2",
                        "wiki_visibility_score", "wiki_visibility_bin", "ipf_weight")
            if c in places.columns]
    return ids.join(places.select(keep), on="place_id", how="left")


def place_level(
    feats: np.ndarray,
    tiles: pl.DataFrame,
    *,
    samples: tuple[int, ...] = WALL_SAMPLES,
    block: tuple[int, int] | None = None,
) -> tuple[np.ndarray, pl.DataFrame]:
    """Mean over each place's given samples, re-L2-normalized.

    Returns (P, D) matrix + one-row-per-place covariate frame (aligned).
    """
    sub = tiles.filter(pl.col("sample_idx").is_in(list(samples)))
    x = feats[sub["row_idx"].to_numpy()]
    if block is not None:
        x = x[:, block[0]:block[1]]
    grp = sub.select(["place_id"]).with_row_index("i")
    order = sub["place_id"].to_numpy()
    uniq, inv = np.unique(order, return_inverse=True)
    acc = np.zeros((uniq.shape[0], x.shape[1]), dtype=np.float64)
    cnt = np.zeros(uniq.shape[0], dtype=np.int64)
    np.add.at(acc, inv, x.astype(np.float64))
    np.add.at(cnt, inv, 1)
    mean = (acc / np.maximum(cnt, 1)[:, None]).astype(np.float32)
    n = np.linalg.norm(mean, axis=1, keepdims=True)
    mean = mean / np.maximum(n, 1e-8)
    cov = (sub.unique(subset=["place_id"], keep="first", maintain_order=False)
              .drop(["row_idx", "tile_id", "sample_idx"], strict=False))
    cov = pl.DataFrame({"place_id": uniq}).join(cov, on="place_id", how="left")
    return mean, cov


# ---------------------------------------------------------------------------
# kNN
# ---------------------------------------------------------------------------
def knn_indices(x: np.ndarray, k: int) -> np.ndarray:
    """(N, k) nearest-neighbour indices by cosine (inputs L2-normed), excl self. Exact, in chunks."""
    x = np.ascontiguousarray(x, dtype=np.float32)
    k = max(1, min(k, x.shape[0] - 1))
    out = np.empty((x.shape[0], k), dtype=np.int64)
    for i in range(0, x.shape[0], 2000):
        sim = x[i:i + 2000] @ x.T
        sim[np.arange(sim.shape[0]), np.arange(i, i + sim.shape[0])] = -2.0
        idx = np.argpartition(-sim, k, axis=1)[:, :k]
        out[i:i + 2000] = np.take_along_axis(idx, np.argsort(-np.take_along_axis(sim, idx, 1), 1), 1)
    return out


# ---------------------------------------------------------------------------
# kNN label prediction + permutation null + bootstrap
# ---------------------------------------------------------------------------
def _majority(nn_lab: np.ndarray, n_classes: int) -> np.ndarray:
    """Vectorized row-wise majority vote over (N, k) int labels in [-1, C).

    Rows' labels are offset by row*C and pooled through one bincount, then
    argmax per row. -1 (missing) neighbours are ignored; a row with no valid
    neighbour predicts -1. Ties break toward the lowest class id
    (deterministic; documented in methods).
    """
    n, k = nn_lab.shape
    rows = np.repeat(np.arange(n, dtype=np.int64), k)
    flat = nn_lab.reshape(-1).astype(np.int64)
    ok = flat >= 0
    counts = np.bincount(rows[ok] * n_classes + flat[ok],
                         minlength=n * n_classes).reshape(n, n_classes)
    pred = counts.argmax(axis=1)
    pred[counts.sum(axis=1) == 0] = -1
    return pred


def knn_label_accuracy(
    nn_idx: np.ndarray,
    labels: np.ndarray,
    *,
    rng: np.random.Generator,
    n_permutations: int,
    n_bootstrap: int,
) -> dict:
    """Majority-vote kNN accuracy vs majority-class + label-permutation null.

    labels: integer-coded (N,), -1 = missing (excluded from truth, ignored in
    votes). Null: permute labels over the valid rows, recompute. Bootstrap:
    resample valid rows (places) with replacement.
    """
    valid = labels >= 0
    idx = np.where(valid)[0]
    lab = labels.astype(np.int64)
    n_classes = int(lab.max()) + 1

    def _acc(labs: np.ndarray, rows: np.ndarray) -> tuple[float, float]:
        pred = _majority(labs[nn_idx[rows]], n_classes)
        truth = labs[rows]
        ok = pred == truth
        acc = float(ok.mean())
        classes = np.unique(truth)
        recalls = [float(ok[truth == c].mean()) for c in classes]
        return acc, float(np.mean(recalls))

    acc, bal = _acc(lab, idx)
    majority = float(np.bincount(lab[idx]).max() / idx.shape[0])

    null = np.empty(n_permutations, dtype=np.float64)
    for i in range(n_permutations):
        perm = lab.copy()
        perm[idx] = rng.permutation(perm[idx])
        pred = _majority(perm[nn_idx[idx]], n_classes)
        null[i] = float((pred == perm[idx]).mean())
    p = float((np.sum(null >= acc) + 1) / (n_permutations + 1))

    boots = np.empty(n_bootstrap, dtype=np.float64)
    for i in range(n_bootstrap):
        rows = rng.choice(idx, size=idx.shape[0], replace=True)
        pred = _majority(lab[nn_idx[rows]], n_classes)
        boots[i] = float((pred == lab[rows]).mean())
    lo, hi = np.percentile(boots, [2.5, 97.5])

    return {"accuracy": acc, "balanced_accuracy": bal,
            "majority_baseline": majority,
            "null_mean": float(null.mean()), "null_p975": float(np.percentile(null, 97.5)),
            "p_value": p, "ci95": [float(lo), float(hi)],
            "n": int(idx.shape[0])}


def encode_labels(values: pl.Series) -> tuple[np.ndarray, list[str]]:
    """Categorical series -> int codes (-1 for null/unknown), plus vocab."""
    vals = values.cast(pl.Utf8).to_list()
    vocab = sorted({v for v in vals if v not in (None, "", "unknown")})
    index = {v: i for i, v in enumerate(vocab)}
    codes = np.array([index.get(v, -1) for v in vals], dtype=np.int64)
    return codes, vocab


# ---------------------------------------------------------------------------
# Rank statistics with permutation + cluster bootstrap
# ---------------------------------------------------------------------------
def spearman_perm(
    x: np.ndarray, y: np.ndarray, *,
    rng: np.random.Generator, n_permutations: int,
) -> dict:
    """Spearman rho with a permutation p-value (two-sided)."""
    from scipy.stats import rankdata
    m = np.isfinite(x) & np.isfinite(y)
    xr, yr = rankdata(x[m]), rankdata(y[m])
    rho = float(np.corrcoef(xr, yr)[0, 1])
    null = np.empty(n_permutations)
    for i in range(n_permutations):
        null[i] = np.corrcoef(rng.permutation(xr), yr)[0, 1]
    p = float((np.sum(np.abs(null) >= abs(rho)) + 1) / (n_permutations + 1))
    return {"rho": rho, "p_value": p, "n": int(m.sum())}


def within_group_rank_slope(
    x: np.ndarray, y: np.ndarray, groups: np.ndarray, *,
    rng: np.random.Generator, n_bootstrap: int, min_group: int = 5,
) -> dict:
    """Rank-OLS slope of y on x with group (country) fixed effects,
    cluster-bootstrapped over groups.

    Ranks computed within group, then demeaned within group (the FE), pooled
    OLS slope. The within-country gradient — defeats "rich vs poor countries".
    """
    from scipy.stats import rankdata
    m = np.isfinite(x) & np.isfinite(y)
    x, y, groups = x[m], y[m], groups[m]
    uniq = np.unique(groups)

    def _slope(sel_groups: np.ndarray) -> float:
        xs, ys = [], []
        for g in sel_groups:
            gm = groups == g
            if gm.sum() < min_group:
                continue
            gx, gy = rankdata(x[gm]), rankdata(y[gm])
            xs.append(gx - gx.mean())
            ys.append(gy - gy.mean())
        if not xs:
            return np.nan
        xa, ya = np.concatenate(xs), np.concatenate(ys)
        denom = float(np.dot(xa, xa))
        return float(np.dot(xa, ya) / denom) if denom > 0 else np.nan

    slope = _slope(uniq)
    boots = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        boots[i] = _slope(rng.choice(uniq, size=uniq.shape[0], replace=True))
    boots = boots[np.isfinite(boots)]
    lo, hi = (np.percentile(boots, [2.5, 97.5]) if boots.size else (np.nan, np.nan))
    return {"slope": slope, "ci95": [float(lo), float(hi)],
            "n_groups": int(uniq.shape[0]), "n": int(x.shape[0])}


def haversine_km(lat1, lng1, lat2, lng2) -> np.ndarray:
    r1, r2 = np.radians(lat1), np.radians(lat2)
    dlat = np.radians(lat2 - lat1)
    dlng = np.radians(lng2 - lng1)
    a = np.sin(dlat / 2) ** 2 + np.cos(r1) * np.cos(r2) * np.sin(dlng / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
