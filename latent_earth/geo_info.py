"""A1 + A2: geographic information content of the generated corpus (P1).

A1  kNN label purity per modality: how well do a place's nearest neighbours in
    each embedding space share its country / UN region / Köppen class?
    Key contrast: t5_prompt literally contains the place-name string ("In
    <place>") — its purity is the text-encoder CEILING. The scientific
    quantity is image-side purity (dino, siglip) and above all flux_residual:
    does geography survive into what the model actually renders?

A2  Distance decay + embedding-space Mantel: mean cosine similarity between
    place pairs, binned by great-circle distance; the per-modality similarity
    FLOOR (how alike two arbitrary places' renderings are) is itself the
    homogenization baseline.

All quantities place-level (mean of samples 0-3, re-normalized). Outputs
results/geo_knn.json + geo_decay.json.
"""

from __future__ import annotations

import logging

import numpy as np
import polars as pl

from latent_earth.config import KNN_K, AnalysisConfig
from latent_earth import util

log = logging.getLogger(__name__)

TARGETS = ("country", "un_region", "koppen_class")


def run_knn_purity(
    feats: np.ndarray,
    tiles: pl.DataFrame,
    blocks: dict[str, tuple[int, int]],
    config: AnalysisConfig,
    *,
    k: int = KNN_K,
    per_scale: bool = True,
) -> dict:
    """A1: kNN purity per modality (+ concat), place-level and per-scale."""
    rng = np.random.default_rng(config.seed)
    spaces: dict[str, tuple[int, int] | None] = {**blocks, "concat": None}
    out: dict = {"k": k, "targets": {}, "spaces": list(spaces.keys())}

    # ---- place-level (primary) ----
    place_mats: dict[str, np.ndarray] = {}
    cov = None
    for name, block in spaces.items():
        mat, cov = util.place_level(feats, tiles, block=block)
        place_mats[name] = mat
    labels: dict[str, tuple[np.ndarray, list[str]]] = {}
    for t in TARGETS:
        if cov is not None and t in cov.columns:
            labels[t] = util.encode_labels(cov[t])

    for t, (codes, vocab) in labels.items():
        if codes.max() < 1:                      # degenerate (e.g. koppen unknown)
            log.warning("target %s has <2 classes; skipped", t)
            continue
        out["targets"][t] = {"n_classes": len(vocab), "place_level": {}}
        for name, mat in place_mats.items():
            nn = util.knn_indices(mat, k)
            res = util.knn_label_accuracy(
                nn, codes, rng=rng,
                n_permutations=config.n_permutations,
                n_bootstrap=config.n_bootstrap)
            out["targets"][t]["place_level"][name] = res
            log.info("A1 %s/%s: acc=%.3f (chance %.3f, p=%.4f)",
                     t, name, res["accuracy"], res["majority_baseline"],
                     res["p_value"])

    # ---- per-scale (secondary; fewer permutations to keep runtime sane) ----
    if per_scale:
        for t, (codes_all, vocab) in labels.items():
            if codes_all.max() < 1:
                continue
            per_scale_res: dict = {}
            for s in (0, 1, 2, 3, 4):
                sub = tiles.filter(pl.col("sample_idx") == s)
                if sub.height == 0:
                    continue
                rows = sub["row_idx"].to_numpy()
                codes_s, _ = util.encode_labels(sub[t]) if t in sub.columns else (None, None)
                if codes_s is None or codes_s.max() < 1:
                    continue
                per_scale_res[str(s)] = {}
                for name, block in spaces.items():
                    x = feats[rows] if block is None else feats[rows, block[0]:block[1]]
                    n = np.linalg.norm(x, axis=1, keepdims=True)
                    x = x / np.maximum(n, 1e-8)
                    nn = util.knn_indices(x, k)
                    res = util.knn_label_accuracy(
                        nn, codes_s, rng=rng, n_permutations=99, n_bootstrap=200)
                    per_scale_res[str(s)][name] = {
                        "accuracy": res["accuracy"],
                        "majority_baseline": res["majority_baseline"],
                        "ci95": res["ci95"]}
            out["targets"][t]["per_scale"] = per_scale_res

    config.write_result("geo_knn.json", out)
    return out


def run_distance_decay(
    feats: np.ndarray,
    tiles: pl.DataFrame,
    blocks: dict[str, tuple[int, int]],
    config: AnalysisConfig,
    *,
    n_pairs: int = 2_000_000,
    n_mantel_places: int = 2000,
    n_mantel_repeats: int = 20,
) -> dict:
    """A2: distance-decay curves + embedding-space Mantel, per modality."""
    rng = np.random.default_rng(config.seed + 1)
    spaces: dict[str, tuple[int, int] | None] = {**blocks, "concat": None}

    _mat0, cov = util.place_level(feats, tiles)
    lat = cov["lat"].to_numpy().astype(np.float64)
    lng = cov["lng"].to_numpy().astype(np.float64)
    p = lat.shape[0]

    # log-spaced distance bins, 1 km .. 20,000 km
    edges = np.geomspace(1.0, 20_000.0, 25)
    i_idx = rng.integers(0, p, size=n_pairs)
    j_idx = rng.integers(0, p, size=n_pairs)
    keep = i_idx != j_idx
    i_idx, j_idx = i_idx[keep], j_idx[keep]
    gd = util.haversine_km(lat[i_idx], lng[i_idx], lat[j_idx], lng[j_idx])
    bins = np.digitize(gd, edges)

    out: dict = {"bin_edges_km": edges.tolist(), "curves": {}, "mantel": {}}
    for name, block in spaces.items():
        mat, _ = util.place_level(feats, tiles, block=block)
        sims = np.einsum("ij,ij->i", mat[i_idx], mat[j_idx]).astype(np.float64)
        curve = []
        for b in range(1, len(edges)):
            m = bins == b
            if m.sum() < 50:
                curve.append(None)
                continue
            s = sims[m]
            boot = rng.choice(s, size=(200, min(s.shape[0], 20_000)), replace=True)
            bm = boot.mean(axis=1)
            curve.append({"mean": float(s.mean()),
                          "ci95": [float(np.percentile(bm, 2.5)),
                                   float(np.percentile(bm, 97.5))],
                          "n": int(m.sum())})
        floor = float(sims[gd > 5000].mean()) if (gd > 5000).any() else None
        out["curves"][name] = {"bins": curve, "similarity_floor_gt5000km": floor}

        # embedding-space Mantel on subsamples
        rs = []
        for _ in range(n_mantel_repeats):
            sel = rng.choice(p, size=min(n_mantel_places, p), replace=False)
            sm = mat[sel]
            ed = 1.0 - sm @ sm.T
            gdm = util.haversine_km(lat[sel][:, None], lng[sel][:, None],
                                    lat[sel][None, :], lng[sel][None, :])
            iu = np.triu_indices(sel.shape[0], k=1)
            rs.append(float(np.corrcoef(ed[iu], gdm[iu])[0, 1]))
        rs = np.array(rs)
        # permutation p on the last subsample (representative; documented)
        null = np.empty(config.n_permutations)
        e_flat, g_flat = ed[iu], gdm[iu]
        n_sel = sel.shape[0]
        for i in range(config.n_permutations):
            perm = rng.permutation(n_sel)
            null[i] = np.corrcoef(ed[np.ix_(perm, perm)][iu], g_flat)[0, 1]
        p_val = float((np.sum(null >= rs[-1]) + 1) / (config.n_permutations + 1))
        out["mantel"][name] = {"r_mean": float(rs.mean()),
                               "r_std": float(rs.std()),
                               "repeats": rs.tolist(), "p_value_last": p_val}
        log.info("A2 %s: mantel r=%.3f±%.3f, floor=%.3f", name,
                 rs.mean(), rs.std(), floor if floor is not None else float("nan"))

    config.write_result("geo_decay.json", out)
    return out
