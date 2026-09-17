"""Verification analyses (run after geo-info and dispersion):

1. Distance decay decomposed into same-country vs different-country pairs —
   verifies the mixture reading of Fig. 2b (categorical, not metric): within-
   country similarity should be near-flat with distance; the pooled decline
   should track the falling share of same-kind pairs.
2. Per-place identification correctness + its overlap with the
   low-dispersion places.

Outputs: results/s3_decay_decomposition.json, s3_overlap.json,
s3_place_correct.parquet.
"""

from __future__ import annotations

import logging

import numpy as np
import polars as pl

from latent_earth.config import KNN_K, AnalysisConfig
from latent_earth import util

log = logging.getLogger(__name__)


def decay_decomposition(feats, tiles, config, *, n_pairs: int = 2_000_000) -> dict:
    rng = np.random.default_rng(config.seed + 7)
    mat, cov = util.place_level(feats, tiles)
    lat = cov["lat"].to_numpy().astype(np.float64)
    lng = cov["lng"].to_numpy().astype(np.float64)
    cc = cov["country"].cast(pl.Utf8).fill_null("?").to_numpy()
    p = mat.shape[0]
    i = rng.integers(0, p, size=n_pairs)
    j = rng.integers(0, p, size=n_pairs)
    keep = i != j
    i, j = i[keep], j[keep]
    gd = util.haversine_km(lat[i], lng[i], lat[j], lng[j])
    sims = np.einsum("ij,ij->i", mat[i], mat[j]).astype(np.float64)
    same = cc[i] == cc[j]
    edges = np.geomspace(1.0, 20_000.0, 25)
    bins = np.digitize(gd, edges)
    out = {"bin_edges_km": edges.tolist(), "same_country": [],
           "different_country": [], "share_same_country": []}
    for b in range(1, len(edges)):
        m = bins == b
        for key, sel in (("same_country", m & same), ("different_country", m & ~same)):
            out[key].append(
                {"mean": float(sims[sel].mean()), "n": int(sel.sum())}
                if sel.sum() >= 50 else None)
        out["share_same_country"].append(float(same[m].mean()) if m.sum() else None)
    config.write_result("s3_decay_decomposition.json", out)
    return out


def identification_overlap(feats, tiles, config, *, k: int = KNN_K,
                           stable_threshold: float = 0.30) -> dict:
    """Per-place country-identification correctness x stable/unstable split.

    stable_threshold: the dispersion valley separating the two populations
    (0.30 on the real corpus; pass the corpus-appropriate value elsewhere).
    """
    mat, cov = util.place_level(feats, tiles)
    codes, _vocab = util.encode_labels(cov["country"])
    nn = util.knn_indices(mat, k)
    n_classes = int(codes.max()) + 1
    pred = util._majority(codes[nn], n_classes)
    correct = (pred == codes) & (codes >= 0)
    # graded knowledge-of-place: share of the k most-similar places from the
    # same country (0..1). Drives the census plate's colour (Fig. 8).
    purity = (codes[nn] == codes[:, None]).mean(axis=1)
    purity[codes < 0] = np.nan
    per_place = pl.DataFrame({"place_id": cov["place_id"],
                              "country_correct": correct,
                              "country_purity": purity.astype(np.float64)})
    disp = pl.read_parquet(config.out_path("dispersion_per_place.parquet"))
    j = per_place.join(disp.select(["place_id", "dispersion_concat"]),
                       on="place_id", how="inner")
    stable = j["dispersion_concat"].to_numpy() < stable_threshold
    corr = j["country_correct"].to_numpy()
    tab = {
        "n": int(j.height),
        "stable_threshold": float(stable_threshold),
        "acc_stable": float(corr[stable].mean()) if stable.any() else None,
        "acc_unstable": float(corr[~stable].mean()) if (~stable).any() else None,
        "share_stable": float(stable.mean()),
        "phi": (float(np.corrcoef(stable.astype(float), corr.astype(float))[0, 1])
                if stable.any() and (~stable).any() else None),
    }
    per_place.write_parquet(config.out_path("s3_place_correct.parquet"))
    config.write_result("s3_overlap.json", tab)
    log.info("S3 overlap: acc stable=%.3f vs unstable=%.3f (phi=%.3f)",
             tab["acc_stable"], tab["acc_unstable"], tab["phi"])
    return tab


def run_s3(feats, tiles, config, *, stable_threshold: float = 0.30) -> dict:
    return {"decay_decomposition": decay_decomposition(feats, tiles, config),
            "overlap": identification_overlap(
                feats, tiles, config, stable_threshold=stable_threshold)}
