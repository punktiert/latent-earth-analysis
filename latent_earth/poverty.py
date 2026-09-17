"""A4: documentation and differentiation — the confirmatory analysis.

Measures the ASSOCIATION between how documented a place is online and how
distinctively the model renders it. Estimation, not hypothesis confirmation:
the direction and size of every estimate is an empirical result (compression
of the undocumented, exoticization of the undocumented, or no within-country
association are all reportable outcomes — see paper/OSF_prereg_draft.md).
Four outcomes per place (pre-specified):

  1. local_density    — mean cosine distance to the 20 nearest OTHER places
                        (LOW = crowded, generic region of latent space)
  2. genericity       — caption TF-IDF genericity (embedding-free; captions.py)
  3. centroid_dist    — cosine distance to the place's UN-region centroid
                        (LOW = close to the regional average rendering)
  4. megacluster      — log2 size of the place's k=64 k-means cluster
                        (HIGH = lives in a mass-produced type)

Predictor: wiki_visibility_score (backfilled PC1). Robustness: log population.
Stats: Spearman + permutation p; rank-OLS with COUNTRY fixed effects,
cluster-bootstrapped by country (the within-country gradient defeats the
"you measured rich vs poor countries" objection).

Also the stereotype quantifications: church-token rates, the street-scale
mega-cluster's geographic entropy, and >2,000 km "interchangeability pairs".

Output: results/poverty.json + poverty_per_place.parquet.
"""

from __future__ import annotations

import logging

import numpy as np
import polars as pl

from latent_earth.config import DENSITY_K, AnalysisConfig
from latent_earth import captions as cap
from latent_earth import util

log = logging.getLogger(__name__)

CHURCH_TOKENS = ("church", "steeple", "bell tower", "belfry", "spire", "chapel")


# ---------------------------------------------------------------------------
# Outcomes
# ---------------------------------------------------------------------------
def local_density(mat: np.ndarray, k: int = DENSITY_K) -> np.ndarray:
    """Mean cosine distance to the k nearest neighbours (place-level matrix)."""
    nn = util.knn_indices(mat, k)
    sims = np.einsum("ijk,ik->ij", mat[nn], mat)          # (P, k) cosine sims
    return (1.0 - sims).mean(axis=1)


def centroid_distance(mat: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Cosine distance from each place to its group's centroid (nan if no group)."""
    out = np.full(mat.shape[0], np.nan)
    for g in np.unique(groups):
        if g == "" or g is None:
            continue
        m = groups == g
        if m.sum() < 5:
            continue
        c = mat[m].mean(axis=0)
        c = c / max(np.linalg.norm(c), 1e-8)
        out[m] = 1.0 - mat[m] @ c
    return out


def megacluster_logsize(mat: np.ndarray, *, k: int = 64, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """k-means over places; per-place log2 cluster size + labels."""
    from sklearn.cluster import MiniBatchKMeans
    labels = MiniBatchKMeans(n_clusters=k, random_state=seed, n_init=3, batch_size=4096, max_iter=200).fit_predict(
        np.ascontiguousarray(mat, dtype=np.float32))
    sizes = np.bincount(labels, minlength=labels.max() + 1)
    return np.log2(sizes[labels].astype(np.float64)), labels


# ---------------------------------------------------------------------------
# Main run
# ---------------------------------------------------------------------------
def run_poverty(
    feats: np.ndarray,
    tiles: pl.DataFrame,
    blocks: dict[str, tuple[int, int]],
    config: AnalysisConfig,
) -> dict:
    rng = np.random.default_rng(config.seed + 3)
    mat, cov = util.place_level(feats, tiles)
    out: dict = {}

    dens = local_density(mat)
    cdist = centroid_distance(mat, cov["un_region"].cast(pl.Utf8).fill_null("").to_numpy())
    mega, klabels = megacluster_logsize(mat, seed=config.seed)

    per_place = pl.DataFrame({
        "place_id": cov["place_id"],
        "local_density": dens.astype(np.float64),
        "centroid_dist": cdist,
        "megacluster_log2size": mega,
        "kmeans_label": klabels.astype(np.int32),
    })

    # caption genericity (embedding-free outcome)
    try:
        gen_tab = cap.run_captions(config)
        per_place = per_place.join(
            gen_tab.select(["place_id", "genericity", "top50_share", "n_features"]),
            on="place_id", how="left")
    except FileNotFoundError:
        log.warning("captions dir absent — genericity outcome skipped")

    joined = per_place.join(cov, on="place_id", how="left")
    joined.write_parquet(config.out_path("poverty_per_place.parquet"))

    # ---- predictor availability ----
    vis_ok = ("wiki_visibility_score" in joined.columns
              and joined["wiki_visibility_score"].null_count() < joined.height)
    predictors = {}
    if vis_ok:
        predictors["visibility"] = joined["wiki_visibility_score"].cast(pl.Float64) \
            .fill_null(np.nan).to_numpy()
    else:
        log.warning("wiki_visibility_score not populated — run "
                    "falling back to population only")
    predictors["log_population"] = np.log1p(
        joined["population"].cast(pl.Float64).fill_null(np.nan).to_numpy())

    outcomes = {
        "local_density": joined["local_density"].to_numpy(),
        "centroid_dist": joined["centroid_dist"].to_numpy(),
        "megacluster_log2size": joined["megacluster_log2size"].to_numpy(),
    }
    if "genericity" in joined.columns:
        outcomes["genericity"] = joined["genericity"].cast(pl.Float64) \
            .fill_null(np.nan).to_numpy()

    countries = joined["country"].cast(pl.Utf8).fill_null("?").to_numpy()

    results: dict = {}
    for pname, pvals in predictors.items():
        results[pname] = {}
        for oname, ovals in outcomes.items():
            res = {
                "spearman": util.spearman_perm(
                    pvals, ovals, rng=rng, n_permutations=config.n_permutations),
                "within_country": util.within_group_rank_slope(
                    pvals, ovals, countries, rng=rng, n_bootstrap=config.n_bootstrap),
            }
            results[pname][oname] = res
            log.info("A4 %s->%s: rho=%.3f (p=%.4f), within-country slope=%.3f %s",
                     pname, oname, res["spearman"]["rho"], res["spearman"]["p_value"],
                     res["within_country"]["slope"], res["within_country"]["ci95"])
    out["regressions"] = results
    out["visibility_available"] = vis_ok

    # ---- stereotype quantifications ----
    stereo: dict = {}
    if "genericity" in joined.columns:
        pf = cap.load_place_features(config)
        church = {pid: any(t in f for f in feats for t in CHURCH_TOKENS)
                  for pid, feats in pf.items()}
        cdf = pl.DataFrame({"place_id": list(church.keys()),
                            "has_church": list(church.values())})
        cj = joined.join(cdf, on="place_id", how="inner")
        eu = cj.filter(pl.col("un_region").cast(pl.Utf8).str.contains("Europe"))
        small_eu = eu.filter(pl.col("population") < 50_000)
        rest = cj.filter(~pl.col("un_region").cast(pl.Utf8).str.contains("Europe"))
        stereo["church_rate_small_european"] = float(small_eu["has_church"].mean()) \
            if small_eu.height else None
        stereo["church_rate_non_european"] = float(rest["has_church"].mean()) \
            if rest.height else None

    # interchangeability pairs: very similar renderings, very far apart
    lat = cov["lat"].to_numpy().astype(np.float64)
    lng = cov["lng"].to_numpy().astype(np.float64)
    nn = util.knn_indices(mat, 5)
    within_country_d = []
    cc = cov["country"].cast(pl.Utf8).fill_null("?").to_numpy()
    pairs = []
    for i in range(mat.shape[0]):
        for j in nn[i]:
            j = int(j)
            d = float(1.0 - mat[i] @ mat[j])
            if cc[i] == cc[j]:
                within_country_d.append(d)
            gd = float(util.haversine_km(lat[i], lng[i], lat[j], lng[j]))
            if gd > 2000:
                pairs.append((i, j, d, gd))
    thresh = float(np.percentile(within_country_d, 5)) if within_country_d else 0.0
    far_twins = [(cov["place_id"][i], cov["place_id"][j], d, gd)
                 for i, j, d, gd in pairs if d < thresh]
    stereo["interchangeability"] = {
        "threshold_cos_dist_p5_within_country": thresh,
        "n_far_twin_pairs_gt2000km": len(far_twins),
        "examples": [{"a": a, "b": b, "cos_dist": d, "km": gd}
                     for a, b, d, gd in sorted(far_twins, key=lambda t: t[2])[:50]],
    }
    out["stereotypes"] = stereo

    config.write_result("poverty.json", out)
    return out
