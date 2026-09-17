"""Per-place consistency.

How consistently does the model imagine one place? Per place: the mean
pairwise cosine distance among its four scale-lensed samples AFTER removing
each scale's global centroid (so "eaves images differ from interiors" doesn't
masquerade as place inconsistency). Distribution shape, a permutation contrast
for the heavy tail, covariate correlates, and the check against the fifth,
independent probe. Read the pooled distribution together with
results/prompt_subsets.json: a quarter of the places carry framing phrases.

Output: results/dispersion.json + dispersion_per_place.parquet.
"""

from __future__ import annotations

import logging

import numpy as np
import polars as pl

from latent_earth.config import WALL_SAMPLES, AnalysisConfig
from latent_earth import util

log = logging.getLogger(__name__)


def scale_centered(feats: np.ndarray, tiles: pl.DataFrame,
                   samples: tuple[int, ...] = WALL_SAMPLES) -> tuple[np.ndarray, pl.DataFrame]:
    """Subtract each scale's global centroid; return centered rows + sub-table."""
    sub = tiles.filter(pl.col("sample_idx").is_in(list(samples)))
    x = feats[sub["row_idx"].to_numpy()].astype(np.float64)
    s = sub["sample_idx"].to_numpy()
    xc = x.copy()
    for si in np.unique(s):
        m = s == si
        xc[m] -= x[m].mean(axis=0, keepdims=True)
    n = np.linalg.norm(xc, axis=1, keepdims=True)
    xc = xc / np.maximum(n, 1e-12)
    return xc.astype(np.float32), sub


def per_place_dispersion(xc: np.ndarray, sub: pl.DataFrame) -> pl.DataFrame:
    """Mean pairwise cosine distance among each place's (centered) samples."""
    pid = sub["place_id"].to_numpy()
    uniq, inv = np.unique(pid, return_inverse=True)
    # accumulate sum and sum of outer-products cheaply: for unit vectors,
    # mean pairwise cosine distance = 1 - (||sum||^2 - n) / (n (n-1))
    d = xc.shape[1]
    sums = np.zeros((uniq.shape[0], d), dtype=np.float64)
    cnts = np.zeros(uniq.shape[0], dtype=np.int64)
    np.add.at(sums, inv, xc.astype(np.float64))
    np.add.at(cnts, inv, 1)
    nrm2 = np.einsum("ij,ij->i", sums, sums)
    n = cnts.astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_cos = (nrm2 - n) / (n * (n - 1.0))
    disp = 1.0 - mean_cos
    ok = cnts >= 2
    return pl.DataFrame({"place_id": uniq[ok],
                         "dispersion": disp[ok].astype(np.float64),
                         "n_samples": cnts[ok]})


def run_dispersion(
    feats: np.ndarray,
    tiles: pl.DataFrame,
    blocks: dict[str, tuple[int, int]],
    config: AnalysisConfig,
) -> dict:
    rng = np.random.default_rng(config.seed + 2)
    out: dict = {}

    # ---- primary: concat; secondary: flux_residual ----
    tables: dict[str, pl.DataFrame] = {}
    for name, block in (("concat", None), ("flux_residual", blocks.get("flux_residual"))):
        f = feats if block is None else feats[:, block[0]:block[1]]
        xc, sub = scale_centered(f, tiles)
        tables[name] = per_place_dispersion(xc, sub)

    d = tables["concat"]["dispersion"].to_numpy()
    q = {f"p{p}": float(np.percentile(d, p)) for p in (5, 25, 50, 75, 90, 95, 99)}
    out["concat"] = {"n_places": int(d.shape[0]), "mean": float(d.mean()),
                     "quantiles": q, "p90_over_median": float(q["p90"] / max(q["p50"], 1e-12))}

    # heavy-tail permutation contrast: scramble place membership WITHIN scale
    xc, sub = scale_centered(feats, tiles)
    pid = sub["place_id"].to_numpy()
    s = sub["sample_idx"].to_numpy()
    obs_ratio = out["concat"]["p90_over_median"]
    null = np.empty(200)
    for i in range(200):
        pid_perm = pid.copy()
        for si in np.unique(s):
            m = s == si
            pid_perm[m] = rng.permutation(pid_perm[m])
        sub_perm = sub.with_columns(pl.Series("place_id", pid_perm))
        dp = per_place_dispersion(xc, sub_perm)["dispersion"].to_numpy()
        null[i] = float(np.percentile(dp, 90) / max(np.percentile(dp, 50), 1e-12))
    out["tail_test"] = {
        "obs_p90_over_median": obs_ratio,
        "null_mean": float(null.mean()),
        "null_p975": float(np.percentile(null, 97.5)),
        "p_value": float((np.sum(null >= obs_ratio) + 1) / (null.shape[0] + 1)),
        "note": "null = place labels permuted within scale (dispersion of pseudo-places)",
    }

    # dip test if the optional package is present
    try:
        import diptest  # noqa: PLC0415
        dip, dip_p = diptest.diptest(d)
        out["dip_test"] = {"dip": float(dip), "p_value": float(dip_p)}
    except ImportError:
        out["dip_test"] = None

    # ---- covariate correlates ----
    _mat, cov = util.place_level(feats, tiles)
    dj = tables["concat"].join(cov, on="place_id", how="left")
    correlates: dict = {}
    dd = dj["dispersion"].to_numpy()
    for col, transform in (("population", np.log1p),
                           ("wiki_visibility_score", None)):
        if col not in dj.columns or dj[col].null_count() == dj.height:
            correlates[col] = None
            continue
        x = dj[col].cast(pl.Float64).fill_null(np.nan).to_numpy()
        if transform is not None:
            x = transform(x)
        correlates[col] = util.spearman_perm(
            x, dd, rng=rng, n_permutations=config.n_permutations)
    out["correlates"] = correlates

    # ---- sample_4 replicate: independent Step-1 ----
    s4 = tiles.filter(pl.col("sample_idx") == 4)
    if s4.height > 0:
        wall_mat, wall_cov = util.place_level(feats, tiles, samples=WALL_SAMPLES)
        s4_rows = s4["row_idx"].to_numpy()
        x4 = feats[s4_rows]
        n4 = np.linalg.norm(x4, axis=1, keepdims=True)
        x4 = x4 / np.maximum(n4, 1e-8)
        pid_order = {p: i for i, p in enumerate(wall_cov["place_id"].to_list())}
        keep = [i for i, p in enumerate(s4["place_id"].to_list()) if p in pid_order]
        rows4 = np.array(keep, dtype=np.int64)
        wall_idx = np.array([pid_order[s4["place_id"][int(i)]] for i in rows4])
        rep_dist = 1.0 - np.einsum("ij,ij->i", x4[rows4], wall_mat[wall_idx])
        rep_df = pl.DataFrame({"place_id": [s4["place_id"][int(i)] for i in rows4],
                               "s4_dist": rep_dist.astype(np.float64)})
        dj2 = tables["concat"].join(rep_df, on="place_id", how="inner")
        out["sample4_replicate"] = util.spearman_perm(
            dj2["dispersion"].to_numpy(), dj2["s4_dist"].to_numpy(),
            rng=rng, n_permutations=config.n_permutations)
        out["sample4_replicate"]["n_places_with_s4"] = dj2.height
    else:
        out["sample4_replicate"] = None

    # persist per-place table for figures + release
    both = tables["concat"].rename({"dispersion": "dispersion_concat"}).join(
        tables["flux_residual"].select(["place_id", "dispersion"])
        .rename({"dispersion": "dispersion_flux"}), on="place_id", how="left")
    both.write_parquet(config.out_path("dispersion_per_place.parquet"))

    config.write_result("dispersion.json", out)
    log.info("A3: n=%d median=%.4f p90=%.4f tail-p=%.4f",
             out["concat"]["n_places"], q["p50"], q["p90"], out["tail_test"]["p_value"])
    return out
