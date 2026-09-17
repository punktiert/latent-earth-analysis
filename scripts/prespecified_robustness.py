"""The two robustness analyses that the pre-specified plan lists as "reported
regardless" and that the July run did not include:

  (i)  the estimates weighted by the sampling weight. The ipf_weight column
       was never populated (it is zero for every place, also in the public
       release), so the weights are rebuilt here as post-stratification
       weights: places in the source tables divided by places in the sample,
       per population band and UN subregion. They exist for the 25,043 towns
       and cities; metropolitan districts are not part of the source tables
       and are left out of this analysis. And
  (ii) the repeat in the generator's own state only (the flux_residual block
       of the analysis features), for the three embedding-based outcomes.

Same estimators as the plan: Spearman correlation with a permutation p-value
(999 shuffles) and the within-country rank slope with a cluster bootstrap
over countries (1,000 draws); seed 20260713. The documentation index is the
one in poverty_per_place.parquet (the revised index; Methods).

    python scripts/prespecified_robustness.py
Writes results/poverty_robustness.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from latent_earth.paths import data_root  # noqa: E402
RESULTS = ROOT / "results"
SEED, N_PERM, N_BOOT, DENSITY_K = 20260713, 999, 1000, 20
OUTCOMES = ("local_density", "centroid_dist", "megacluster_log2size", "genericity")


def wcorr(x, y, w):
    mx, my = np.average(x, weights=w), np.average(y, weights=w)
    cov = np.average((x - mx) * (y - my), weights=w)
    return float(cov / np.sqrt(np.average((x - mx) ** 2, weights=w) * np.average((y - my) ** 2, weights=w)))


def estimates(x, y, groups, w, rng) -> dict:
    """Spearman (weighted) with permutation p, and the within-country rank slope with
    a cluster bootstrap. Ranks inside a country do not depend on which other countries
    are drawn, so each country's sums are computed once and resampled."""
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(w)
    x, y, groups, w = x[m], y[m], groups[m], w[m]
    xr, yr = rankdata(x), rankdata(y)
    rho = wcorr(xr, yr, w)
    null = np.array([wcorr(rng.permutation(xr), yr, w) for _ in range(N_PERM)])
    sxy, sxx = [], []
    for g in np.unique(groups):
        gm = groups == g
        if gm.sum() < 5:
            continue
        gx, gy, gw = rankdata(x[gm]), rankdata(y[gm]), w[gm]
        gx = gx - np.average(gx, weights=gw); gy = gy - np.average(gy, weights=gw)
        sxy.append(float((gw * gx * gy).sum())); sxx.append(float((gw * gx * gx).sum()))
    sxy, sxx = np.array(sxy), np.array(sxx)
    idx = rng.integers(0, sxy.size, size=(N_BOOT, sxy.size))
    boots = sxy[idx].sum(1) / sxx[idx].sum(1)
    return {"spearman": {"rho": rho, "p_value": float((np.sum(np.abs(null) >= abs(rho)) + 1) / (N_PERM + 1)), "n": int(m.sum())},
            "within_country": {"slope": float(sxy.sum() / sxx.sum()), "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
                               "n_groups": int(sxy.size)}}


def poststrat_weights(d: pl.DataFrame) -> np.ndarray:
    """N_frame / n_sample per (population band, UN subregion); NaN for metropolitan districts."""
    src = pl.read_parquet(data_root() / "hf" / "data" / "places.parquet").select(["place_id", "source", "iso2"])
    t = d.join(src, on="place_id", how="left", suffix="_p")
    edges, labels = [20_000, 100_000, 500_000, 2_000_000], ["10-20k", "20-100k", "100-500k", "0.5-2M", ">2M"]
    band = lambda c: pl.col(c).cut(edges, labels=labels, left_closed=True).cast(pl.Utf8)  # noqa: E731
    counts, tables = RESULTS / "frame_strata_counts.csv", ROOT / "1-City Data"
    if tables.exists():                                  # the source tables themselves are not redistributed; their stratum counts are
        frame = pl.concat([pl.read_excel(tables / f).select(["iso2", pl.col("population").cast(pl.Float64)])
                           for f in ("SimpleMaps_worldcities.xlsx", "SimpleMaps_worldcities_10-100,000.xlsx")])
        region_of = dict(t.filter(pl.col("un_region").is_not_null()).select(["iso2", pl.col("un_region").cast(pl.Utf8)]).unique().iter_rows())
        frame = frame.filter(pl.col("population") >= 10_000).with_columns(
            pl.col("iso2").replace_strict(region_of, default=None).alias("region"), band("population").alias("band"))
        print(f"  frame: {frame.height:,} places, {frame['region'].null_count()} without a region in the sample")
        N = frame.drop_nulls("region").group_by(["band", "region"]).len().rename({"len": "N"}).sort(["region", "band"])
        N.write_csv(counts)
    else:
        N = pl.read_csv(counts)
    towns = t.filter(pl.col("source") != "district").with_columns(
        band("population").alias("band"), pl.col("un_region").cast(pl.Utf8).alias("region"))
    n = towns.group_by(["band", "region"]).len().rename({"len": "n"})
    w = towns.join(N, on=["band", "region"], how="left").join(n, on=["band", "region"], how="left").with_columns(
        (pl.col("N") / pl.col("n")).alias("weight")).select(["place_id", "weight"])
    out = d.select("place_id").join(w, on="place_id", how="left")
    out.write_parquet(RESULTS / "poststrat_weights.parquet")
    x = out["weight"].to_numpy()
    print(f"  weights: {np.isfinite(x).sum():,} places, min {np.nanmin(x):.2f}, median {np.nanmedian(x):.2f}, max {np.nanmax(x):.2f}")
    return x


def generator_outcomes(d: pl.DataFrame) -> pl.DataFrame:
    """local_density, centroid_dist and cluster size recomputed in the generator's own state."""
    from sklearn.cluster import MiniBatchKMeans

    from latent_earth.discover import find_layout
    feats = find_layout("analysis_features.npy")
    ids = pl.read_parquet(feats.with_name("analysis_feature_tile_ids.parquet"))["tile_id"].to_list()
    blocks = json.loads(feats.with_name("analysis_features_meta.json").read_text(encoding="utf-8"))["blocks"]
    a, b = blocks["flux_residual"]
    F = np.load(feats, mmap_mode="r")
    pid = np.array([t.split("__")[0] for t in ids]); slot = np.array([int(t.split("__")[1]) for t in ids])
    order = {p: i for i, p in enumerate(d["place_id"].to_list())}
    keep = (slot < 4) & np.array([p in order for p in pid])
    P = np.zeros((d.height, b - a), np.float64)
    np.add.at(P, [order[p] for p in pid[keep]], np.asarray(F[keep][:, a:b], dtype=np.float64))   # mean of samples 0-3, as in the plan
    P = (P / np.maximum(np.linalg.norm(P, axis=1, keepdims=True), 1e-8)).astype(np.float32)
    dens = np.empty(d.height)
    for i in range(0, d.height, 2000):
        sim = P[i:i + 2000] @ P.T
        sim[np.arange(sim.shape[0]), np.arange(i, i + sim.shape[0])] = -2
        dens[i:i + 2000] = 1 - np.sort(sim, 1)[:, -DENSITY_K:].mean(1)
    reg = d["un_region"].cast(pl.Utf8).fill_null("").to_numpy().astype(str)
    cdist = np.full(d.height, np.nan)
    for g in np.unique(reg):
        mm = reg == g
        if g and mm.sum() >= 5:
            c = P[mm].mean(0); c /= max(np.linalg.norm(c), 1e-8)
            cdist[mm] = 1 - P[mm] @ c
    lab = MiniBatchKMeans(64, random_state=SEED, n_init=3, batch_size=4096, max_iter=200).fit_predict(P)
    mega = np.log2(np.bincount(lab, minlength=64)[lab].astype(float))
    return pl.DataFrame({"local_density": dens, "centroid_dist": cdist, "megacluster_log2size": mega})


def main() -> int:
    rng = np.random.default_rng(SEED + 7)
    d = pl.read_parquet(RESULTS / "poverty_per_place.parquet")
    groups = d["country"].cast(pl.Utf8).fill_null("?").to_numpy().astype(str)
    preds = {"documentation": d["wiki_visibility_score"].cast(pl.Float64).fill_null(np.nan).to_numpy(),
             "log_population": np.log1p(d["population"].cast(pl.Float64).to_numpy())}
    w_ipf = poststrat_weights(d)
    ones = np.ones(d.height)
    out: dict = {"_provenance": {"seed": SEED, "n_permutations": N_PERM, "n_bootstrap": N_BOOT,
                                 "note": "pre-specified robustness analyses, computed in September 2026"},
                 "weights": "post-stratification (source tables / sample, population band x UN subregion), towns and cities only",
                 "ipf_weighted": {}, "unweighted_towns_and_cities": {}, "generator_state_only": {}}
    for pn, x in preds.items():
        out["ipf_weighted"][pn] = {o: estimates(x, d[o].cast(pl.Float64).fill_null(np.nan).to_numpy(), groups, w_ipf, rng) for o in OUTCOMES}
        towns_only = np.where(np.isfinite(w_ipf), 1.0, np.nan)       # the same places, unweighted, for comparison
        out["unweighted_towns_and_cities"][pn] = {o: estimates(x, d[o].cast(pl.Float64).fill_null(np.nan).to_numpy(), groups, towns_only, rng) for o in OUTCOMES}
    g = generator_outcomes(d)
    for pn, x in preds.items():
        out["generator_state_only"][pn] = {o: estimates(x, g[o].to_numpy(), groups, ones, rng) for o in g.columns}
    (RESULTS / "poverty_robustness.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    for block, res in (("ipf_weighted", out["ipf_weighted"]), ("unweighted_towns", out["unweighted_towns_and_cities"]),
                       ("generator_state_only", out["generator_state_only"])):
        for pn, r in res.items():
            for o, e in r.items():
                wc, sp = e["within_country"], e["spearman"]
                print(f"{block:22s} {pn:15s} {o:22s} rho={sp['rho']:+.3f} (p={sp['p_value']:.3f})  slope={wc['slope']:+.3f} [{wc['ci95'][0]:+.3f}, {wc['ci95'][1]:+.3f}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
