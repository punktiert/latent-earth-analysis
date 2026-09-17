"""NOTE: this script needs the exhibited atlas layout (data/layout/), which is not public; its
results are in results/atlas_wall.json and results/atlas_kinds.parquet. It is included to document
how the atlas statistics of the paper were computed.

What the wall shows: grid-space findings from the sorted atlas.

The paper's other analyses live in embedding space and deliberately disclaim
the 2-D picture (S4: grid-space geography is projection-dependent). This
script asks the complementary question: which properties of the arrangement
are robust ACROSS sort recipes, and therefore properties of the type space
rather than of one projection? Every statistic is computed against a null on
the same grid, and every headline statistic is recomputed on the six recipe
studies (one shared 25,600-tile subsample) so the paper can report a range.

Findings computed (all in grid cells; diag = grid diagonal):

  A1 four-tile spread    mean pairwise grid distance among a place's tiles,
                         split stable / unstable (dispersion valley 0.30),
                         vs random four-tile sets. Tests "settled places sit
                         together, ambiguous places scatter".
  A2 kind territories    per k-means kind: compactness ratio (observed mean
                         pairwise distance / random same-size), contiguity
                         (share of a kind's tiles whose 4-neighbours share
                         the kind), and a name from its most over-represented
                         caption features.
  A3 legibility          join-count lift per covariate: how much more often
                         adjacent tiles share a label than chance. kind,
                         region, country, climate, documentation, population,
                         stability, same-place.
  A4 repetitiveness      per-tile homogeneity = mean cosine similarity of the
                         512-D sort feature to its 8 grid neighbours; vs
                         documentation (the wall text's densest-regions claim).
  A5 country territories compactness + contiguity per country (n >= 50).
  A6 kind adjacency      which kinds border which, as lift over expectation.

Usage:
    python scripts/atlas_findings.py                      # default wall + 6 recipes
    python scripts/atlas_findings.py --grid <exhibited>   # the s_highdim 5x3 wall
Writes results/atlas_wall.json, atlas_kinds.parquet, atlas_per_place.parquet.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from latent_earth.discover import find_data, find_layout  # noqa: E402

from latent_earth.paths import data_root  # noqa: E402
RESULTS = ROOT / "results"
# The grid and the features are looked up under data/ (see latent_earth/discover.py).
LAYOUT = (find_layout("features.npy") or ROOT / "layout" / "features.npy").parent
EXHIBITED_GRID = "grid_default_wall_s_highdim_5x3.parquet"   # the 5 x 3 m piece
SEED = 20260713
STABLE_CUT = 0.30          # fallback cut when results/prompt_subsets.parquet is absent


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------
def load_grid(path: Path) -> pl.DataFrame:
    g = pl.read_parquet(path).select(["tile_id", "row", "col"])
    g = g.with_columns(
        pl.col("tile_id").str.split("__").list.get(0).alias("place_id"),
        pl.col("tile_id").str.split("__").list.get(1).cast(pl.Int8).alias("sample_idx"),
    )
    return g


def load_places() -> pl.DataFrame:
    p = pl.read_parquet(RESULTS / "poverty_per_place.parquet").select([
        "place_id", "kmeans_label", "country", "iso2", "un_region", "koppen_class",
        "wiki_visibility_score", "wiki_visibility_bin", "population_band",
        "population", "lat", "lng"])
    d = pl.read_parquet(RESULTS / "dispersion_per_place.parquet").select(
        ["place_id", "dispersion_concat"])
    c = pl.read_parquet(RESULTS / "s3_place_correct.parquet").select(
        ["place_id", "country_purity"])
    out = p.join(d, on="place_id", how="left").join(c, on="place_id", how="left")
    # "stability": the four prompts of a place are identical (n_prompts == 1) or carry framing phrases
    # (A = four identical Step-2 prompts -> 'stable'; B = four scale registers
    # -> 'unstable'); the 0.30 dispersion cut is the fallback and agrees with
    # (this flag explains the pooled dispersion for 99.0% of places)
    rev = RESULTS / "prompt_subsets.parquet"
    if rev.exists():
        out = out.join(pl.read_parquet(rev).select(["place_id", "n_prompts"]), on="place_id", how="left")
        stab = pl.when(pl.col("n_prompts") == 1).then(pl.lit("stable")).otherwise(pl.lit("unstable"))   # identical prompts vs framed
    else:
        stab = pl.when(pl.col("dispersion_concat") < STABLE_CUT).then(pl.lit("stable")).otherwise(pl.lit("unstable"))
    return out.with_columns(
        stab.alias("stability"),
        (pl.col("wiki_visibility_bin").cast(pl.Utf8)).alias("wiki_visibility_bin"),
        pl.col("population").log10().alias("log_pop"),
    )


FEATURES_USED: dict = {}


def load_features(tile_ids: list[str]) -> np.ndarray:
    """Per-tile features aligned to tile_ids, L2-normalised, for the
    repetitiveness measure. Prefers the 640-D analysis features (the paper's
    canonical space; covers all 200,000 tiles) and falls back to the 512-D
    sort features. The sort-feature table on a workstation may belong to
    whichever subset was sorted last, so it can miss a grid's tiles; those
    would get zero vectors, which is reported rather than silently accepted."""
    feats = find_layout("analysis_features.npy") or (LAYOUT / "features.npy")
    ids_path = feats.with_name("analysis_feature_tile_ids.parquet"
                               if feats.name.startswith("analysis") else "feature_tile_ids.parquet")
    ids = pl.read_parquet(ids_path)["tile_id"].to_list()
    pos = {t: i for i, t in enumerate(ids)}
    idx = np.array([pos.get(t, -1) for t in tile_ids])
    have = idx >= 0
    FEATURES_USED.update({"path": str(feats), "dim": None, "n_missing": int((~have).sum())})
    if not have.all():
        print(f"  note: {(~have).sum()} of {idx.size} grid tiles have no feature row in {feats.name}; zeroed")
    F = np.load(feats)                      # full read: sequential is fast over SMB, fancy-indexing a mmap is not
    FEATURES_USED["dim"] = int(F.shape[1])
    f = np.zeros((idx.size, F.shape[1]), np.float32)
    f[have] = np.asarray(F[idx[have]], dtype=np.float32)
    f /= np.maximum(np.linalg.norm(f, axis=1, keepdims=True), 1e-8)
    return f


STOP = set("""a an and are as at by for from in into is it its of on or over that the their to under with within without
visible structure structures building buildings architectural architecture style design area areas element elements feature
features level levels large small multiple various mixed simple modern traditional distinct prominent surrounding context
use used setting layout pattern patterns detail details type types space spaces open""".split())


def kind_names(places: pl.DataFrame, k_top: int = 3, min_share: float = 0.2) -> dict[int, str]:
    """Name each type by features that are both common in it and rare outside it.

    Over-representation alone (lift) picks rare phrases: a type was called "open-air
    marketplace" although 9 in 10 of its places show none. Here a word scores by
    share-in-type minus share-elsewhere, must describe at least `min_share` of the
    type's places, and is shown as the most frequent phrase that carries it, with
    its share. Only the description that produced the renderings (sample_0) counts.
    """
    cap_path = None
    for cand in (data_root() / "hf" / "data" / "captions.parquet", ROOT / "HF" / "data" / "captions.parquet"):
        try:
            if cand.exists():
                cap_path = cand
                break
        except OSError:
            pass
    if cap_path is None:                      # the public release, 4.5 MB
        try:
            from huggingface_hub import hf_hub_download
            cap_path = Path(hf_hub_download("Punktiert/Latent-Earth", "data/captions.parquet",
                                            repo_type="dataset", local_dir=ROOT / "HF"))
        except Exception as e:  # noqa: BLE001
            print(f"  captions unavailable ({type(e).__name__}); kinds will be numbered, not named")
            return {}
    print(f"  kind names from {cap_path}")
    cap = pl.read_parquet(cap_path).filter(pl.col("caption_slot") == "sample_0").select(["place_id", "features"])
    cap = cap.join(places.select(["place_id", "kmeans_label"]), on="place_id", how="inner")

    def words(phrase: str) -> set[str]:
        out = set()
        for w in re.findall(r"[a-z]+", phrase.lower().replace("-", " ")):
            w = w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w
            if len(w) > 2 and w not in STOP:
                out.add(w)
        return out

    total, n_kind = Counter(), Counter()
    per_kind: dict[int, Counter] = {}
    phrases: dict[int, dict[str, Counter]] = {}
    for row in cap.iter_rows(named=True):
        k = row["kmeans_label"]
        n_kind[k] += 1
        seen = set()
        for f in (row["features"] or []):
            f = str(f).strip().lower()
            for w in words(f):
                phrases.setdefault(k, {}).setdefault(w, Counter())[f] += 1
                seen.add(w)
        total.update(seen)
        per_kind.setdefault(k, Counter()).update(seen)
    n_all = sum(n_kind.values())
    names = {}
    for k, pk in per_kind.items():
        scored = sorted(((c / n_kind[k] - (total[w] - c) / max(n_all - n_kind[k], 1), c / n_kind[k], w)
                         for w, c in pk.items()), reverse=True)
        chosen, used = [], set()
        for pool in ([t for t in scored if t[1] >= min_share], scored):      # fall back if too few common words
            for _score, share, w in pool:
                phrase = phrases[k][w].most_common(1)[0][0]
                if w in used or words(phrase) & used:
                    continue
                chosen.append(f"{phrase} ({share:.0%})")
                used |= words(phrase) | {w}
                if len(chosen) == k_top:
                    break
            if len(chosen) == k_top:
                break
        names[k] = ", ".join(chosen) or f"kind {k}"
    return names


# ---------------------------------------------------------------------------
# geometry helpers
# ---------------------------------------------------------------------------
def adjacency_pairs(rows: np.ndarray, cols: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Index pairs (i, j) of 4-neighbour tiles on the grid."""
    H, W = rows.max() + 1, cols.max() + 1
    lut = -np.ones((H, W), dtype=np.int64)
    lut[rows, cols] = np.arange(rows.size)
    a = lut[:, :-1].ravel(); b = lut[:, 1:].ravel()          # horizontal
    c = lut[:-1, :].ravel(); d = lut[1:, :].ravel()          # vertical
    i = np.concatenate([a, c]); j = np.concatenate([b, d])
    m = (i >= 0) & (j >= 0)
    return i[m], j[m]


def neighbour8(rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """(n, 8) indices of 8-neighbours, -1 where off-grid."""
    H, W = rows.max() + 1, cols.max() + 1
    lut = -np.ones((H + 2, W + 2), dtype=np.int64)
    lut[rows + 1, cols + 1] = np.arange(rows.size)
    out = np.empty((rows.size, 8), dtype=np.int64)
    k = 0
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            out[:, k] = lut[rows + 1 + dr, cols + 1 + dc]
            k += 1
    return out


def geo_spread(lat: np.ndarray, lng: np.ndarray) -> float:
    """Mean great-circle distance (km) of points to their spherical centroid."""
    la, lo = np.radians(lat), np.radians(lng)
    ok = np.isfinite(la) & np.isfinite(lo)
    la, lo = la[ok], lo[ok]
    if la.size == 0:
        return float("nan")
    x, y, z = (np.cos(la) * np.cos(lo)).mean(), (np.cos(la) * np.sin(lo)).mean(), np.sin(la).mean()
    clat, clng = np.arctan2(z, np.hypot(x, y)), np.arctan2(y, x)
    a = np.sin((la - clat) / 2) ** 2 + np.cos(la) * np.cos(clat) * np.sin((lo - clng) / 2) ** 2
    return float((2 * 6371 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))).mean())


def join_lift(labels: np.ndarray, i: np.ndarray, j: np.ndarray) -> dict:
    """Join-count agreement of adjacent tiles vs expectation under random
    labelling (sum p^2), expressed as a kappa-style lift in [0, 1]."""
    ok = (labels[i] != None) & (labels[j] != None)  # noqa: E711
    li, lj = labels[i][ok], labels[j][ok]
    obs = float((li == lj).mean())
    _v, cnt = np.unique(labels[labels != None], return_counts=True)  # noqa: E711
    p = cnt / cnt.sum()
    exp = float((p ** 2).sum())
    lift = (obs - exp) / (1 - exp) if exp < 1 else 0.0
    return {"observed_same": obs, "expected_same": exp, "lift": lift}


def compactness(pts: np.ndarray, rng: np.random.Generator, all_pts: np.ndarray,
                n_sub: int = 1000, n_null: int = 8) -> dict:
    """Mean pairwise distance of a set vs random same-size sets from the grid."""
    def mpd(x):
        if x.shape[0] > n_sub:
            x = x[rng.choice(x.shape[0], n_sub, replace=False)]
        d = np.sqrt(((x[:, None, :] - x[None, :, :]) ** 2).sum(-1))
        return float(d[np.triu_indices(x.shape[0], 1)].mean())
    obs = mpd(pts)
    null = [mpd(all_pts[rng.choice(all_pts.shape[0], min(pts.shape[0], n_sub), replace=False)])
            for _ in range(n_null)]
    return {"mean_pairwise": obs, "null_mean": float(np.mean(null)),
            "ratio": obs / float(np.mean(null))}


# ---------------------------------------------------------------------------
# the analyses, for one grid
# ---------------------------------------------------------------------------
def analyse(grid: pl.DataFrame, places: pl.DataFrame, names: dict[int, str],
            with_features: bool, tag: str) -> dict:
    rng = np.random.default_rng(SEED)
    t = grid.join(places, on="place_id", how="inner")
    rows, cols = t["row"].to_numpy(), t["col"].to_numpy()
    H, W = int(rows.max() + 1), int(cols.max() + 1)
    diag = float(np.hypot(H, W))
    pts = np.stack([rows, cols], 1).astype(np.float64)
    out = {"tag": tag, "n_tiles": t.height, "grid": [H, W], "diag_cells": diag}
    print(f"\n=== {tag}: {t.height} tiles on {H}x{W} ===")

    # ---- A1 four-tile spread --------------------------------------------
    pid = t["place_id"].to_numpy()
    order = np.argsort(pid, kind="stable")
    pid_s, pts_s = pid[order], pts[order]
    uniq, start, count = np.unique(pid_s, return_index=True, return_counts=True)
    spread = np.full(uniq.size, np.nan)
    for n, (s, c) in enumerate(zip(start, count)):
        if c < 2:
            continue
        x = pts_s[s:s + c]
        d = np.sqrt(((x[:, None] - x[None]) ** 2).sum(-1))
        spread[n] = d[np.triu_indices(c, 1)].mean()
    # null: same group sizes, random tiles
    perm = rng.permutation(pts.shape[0])
    null_spread = np.full(uniq.size, np.nan)
    for n, (s, c) in enumerate(zip(start, count)):
        if c < 2:
            continue
        x = pts[perm[s:s + c]]
        d = np.sqrt(((x[:, None] - x[None]) ** 2).sum(-1))
        null_spread[n] = d[np.triu_indices(c, 1)].mean()
    per_place = pl.DataFrame({"place_id": uniq, "wall_spread_cells": spread,
                              "wall_spread_frac_diag": spread / diag})
    pp = per_place.join(places.select(["place_id", "stability", "dispersion_concat",
                                       "wiki_visibility_score"]), on="place_id")
    st = pp.filter(pl.col("stability") == "stable")["wall_spread_cells"].drop_nulls().to_numpy()
    un = pp.filter(pl.col("stability") == "unstable")["wall_spread_cells"].drop_nulls().to_numpy()
    from scipy.stats import spearmanr, mannwhitneyu
    ok = np.isfinite(pp["wall_spread_cells"].to_numpy()) & np.isfinite(pp["dispersion_concat"].to_numpy())
    rho = spearmanr(pp["dispersion_concat"].to_numpy()[ok], pp["wall_spread_cells"].to_numpy()[ok])
    r_close = 0.05 * diag
    out["four_tile_spread"] = {
        "n_places": int(np.isfinite(spread).sum()),
        "median_cells_all": float(np.nanmedian(spread)),
        "median_cells_stable": float(np.median(st)),
        "median_cells_unstable": float(np.median(un)),
        "median_cells_random": float(np.nanmedian(null_spread)),
        "median_frac_diag_stable": float(np.median(st) / diag),
        "median_frac_diag_unstable": float(np.median(un) / diag),
        "median_frac_diag_random": float(np.nanmedian(null_spread) / diag),
        "share_within_5pct_diag_stable": float((st < r_close).mean()),
        "share_within_5pct_diag_unstable": float((un < r_close).mean()),
        "share_within_5pct_diag_random": float((null_spread[np.isfinite(null_spread)] < r_close).mean()),
        "spearman_dispersion_vs_spread": {"rho": float(rho.statistic), "p": float(rho.pvalue)},
        "mannwhitney_p": float(mannwhitneyu(st, un, alternative="less").pvalue),
    }
    print("A1 spread (cells): stable %.1f  unstable %.1f  random %.1f  rho=%.3f" % (
        np.median(st), np.median(un), np.nanmedian(null_spread), rho.statistic))

    # ---- A3 legibility (join-count lift) ---------------------------------
    i, j = adjacency_pairs(rows, cols)
    leg = {}
    for col, lab in (("kmeans_label", "kind"), ("un_region", "region"), ("country", "country"),
                     ("koppen_class", "climate"), ("wiki_visibility_bin", "documentation"),
                     ("population_band", "population"), ("stability", "stability"),
                     ("place_id", "same_place")):
        v = t[col].to_numpy().astype(object)
        leg[lab] = join_lift(v, i, j)
    # beyond-region null: shuffle the label within world region (5 permutations)
    reg = t["un_region"].to_numpy().astype(object)
    for col, lab in (("country", "country"), ("koppen_class", "climate"),
                     ("wiki_visibility_bin", "documentation"),
                     ("population_band", "population"), ("stability", "stability")):
        v = t[col].to_numpy().astype(object)
        nulls = []
        for _ in range(5):
            vv = v.copy()
            for g in np.unique(reg):
                idx = np.where(reg == g)[0]
                vv[idx] = v[rng.permutation(idx)]
            nulls.append(join_lift(vv, i, j)["lift"])
        leg[lab]["lift_within_region_null"] = float(np.mean(nulls))
        leg[lab]["lift_beyond_region"] = leg[lab]["lift"] - float(np.mean(nulls))
    out["legibility"] = leg
    print("A3 lift: " + "  ".join(f"{k}={v['lift']:.3f}" for k, v in leg.items()))
    print("A3 beyond-region: " + "  ".join(
        f"{k}={v['lift_beyond_region']:+.3f}" for k, v in leg.items() if "lift_beyond_region" in v))

    # ---- A2 kind territories + A6 adjacency -------------------------------
    kl = t["kmeans_label"].to_numpy()
    kinds = np.unique(kl)
    K = int(kl.max() + 1)
    same = np.zeros(K); tot = np.zeros(K)
    np.add.at(tot, kl[i], 1); np.add.at(tot, kl[j], 1)
    m_same = kl[i] == kl[j]
    np.add.at(same, kl[i][m_same], 2)
    contig = np.where(tot > 0, same / np.maximum(tot, 1), np.nan)
    p_k = np.bincount(kl, minlength=K) / kl.size
    plat, plng = places["lat"].to_numpy(), places["lng"].to_numpy()
    pkind = places["kmeans_label"].to_numpy()
    punst = places["stability"].to_numpy() == "unstable"
    rows_k = []
    for k in kinds:
        m = kl == k
        pm = pkind == k
        c = compactness(pts[m], rng, pts)
        rows_k.append({"kind": int(k), "name": names.get(int(k), f"kind {k}"),
                       "n_tiles": int(m.sum()),
                       "n_places": int(len(set(pid[m].tolist()))),
                       "compactness_ratio": c["ratio"],
                       "contiguity": float(contig[k]),
                       "contiguity_expected": float(p_k[k]),
                       "geo_km": geo_spread(plat[pm], plng[pm]),      # spread on Earth
                       "unstable_share": float(punst[pm].mean()) if pm.any() else float("nan")})
    kinds_df = pl.DataFrame(rows_k).sort("compactness_ratio")
    out["kinds_summary"] = {
        "n_kinds": int(kinds.size),
        "median_compactness_ratio": float(kinds_df["compactness_ratio"].median()),
        "n_territories_ratio_lt_0_5": int((kinds_df["compactness_ratio"] < 0.5).sum()),
        "n_territories_ratio_lt_0_35": int((kinds_df["compactness_ratio"] < 0.35).sum()),
        "median_contiguity": float(kinds_df["contiguity"].median()),
        "most_compact": kinds_df.head(8).select(["kind", "name", "n_places", "compactness_ratio", "contiguity"]).to_dicts(),
        "least_compact": kinds_df.tail(5).select(["kind", "name", "n_places", "compactness_ratio", "contiguity"]).to_dicts(),
    }
    # adjacency lift between kinds
    pair = Counter(zip(kl[i].tolist(), kl[j].tolist()))
    adj = np.zeros((K, K))
    for (a, b), c in pair.items():
        adj[a, b] += c; adj[b, a] += c
    exp = np.outer(p_k, p_k) * adj.sum()
    lift = np.where(exp > 0, adj / np.maximum(exp, 1e-9), 0)
    np.fill_diagonal(lift, 0)
    top = []
    iu = np.triu_indices(K, 1)
    order = np.argsort(-lift[iu])[:12]
    for o in order:
        a, b = int(iu[0][o]), int(iu[1][o])
        if adj[a, b] < 30:
            continue
        top.append({"a": a, "a_name": names.get(a, str(a)), "b": b, "b_name": names.get(b, str(b)),
                    "shared_edges": int(adj[a, b]), "lift": float(lift[a, b])})
    out["kind_adjacency_top"] = top
    print("A2 kinds: median compactness ratio %.2f; %d/%d with ratio<0.5" % (
        kinds_df["compactness_ratio"].median(), (kinds_df["compactness_ratio"] < 0.5).sum(), kinds.size))

    # ---- A5 country territories ------------------------------------------
    ct = t.group_by("country").agg(pl.len().alias("n"),
                                   pl.col("country_purity").mean().alias("purity"))
    rows_c = []
    cl = t["country"].to_numpy().astype(object)
    for row in ct.filter(pl.col("n") >= 200).iter_rows(named=True):
        m = cl == row["country"]
        c = compactness(pts[m], rng, pts)
        rows_c.append({"country": row["country"], "n_tiles": int(row["n"]),
                       "compactness_ratio": c["ratio"], "country_purity": float(row["purity"] or np.nan)})
    cdf = pl.DataFrame(rows_c).sort("compactness_ratio")
    from scipy.stats import spearmanr as _sp
    okc = cdf["country_purity"].is_not_nan()
    rc = _sp(cdf.filter(okc)["compactness_ratio"].to_numpy(), cdf.filter(okc)["country_purity"].to_numpy())
    out["countries"] = {
        "n_countries": cdf.height,
        "most_compact": cdf.head(10).to_dicts(),
        "least_compact": cdf.tail(5).to_dicts(),
        "spearman_compactness_vs_purity": {"rho": float(rc.statistic), "p": float(rc.pvalue)},
    }
    print("A5 countries: most compact:", [r["country"] for r in cdf.head(6).to_dicts()])

    # ---- A5b region and climate territories: the geographic reading --------
    geo = {}
    for col, key in (("un_region", "regions"), ("koppen_class", "climates")):
        lab = t[col].to_numpy().astype(object)
        rows_g = []
        for g, n in Counter(lab.tolist()).items():
            if g is None or n < 400:
                continue
            c = compactness(pts[lab == g], rng, pts)
            rows_g.append({"group": str(g), "n_tiles": int(n), "compactness_ratio": c["ratio"]})
        rows_g.sort(key=lambda r: r["compactness_ratio"])
        geo[key] = rows_g
        print(f"A5b {key}: " + "  ".join(f"{r['group'][:14]}={r['compactness_ratio']:.2f}" for r in rows_g[:4])
              + "  ...  " + "  ".join(f"{r['group'][:14]}={r['compactness_ratio']:.2f}" for r in rows_g[-3:]))
    out["geo_territories"] = geo
    # wall compactness vs spread on Earth, across kinds
    kk = kinds_df.filter(pl.col("geo_km").is_not_nan())
    rk = spearmanr(kk["compactness_ratio"].to_numpy(), kk["geo_km"].to_numpy())
    out["kinds_summary"]["spearman_wall_vs_geo_spread"] = {"rho": float(rk.statistic), "p": float(rk.pvalue)}

    # ---- A4 repetitiveness vs documentation ------------------------------
    if with_features:
        F = load_features(t["tile_id"].to_list())
        nb = neighbour8(rows, cols)
        sims = np.zeros(F.shape[0]); cnt = np.zeros(F.shape[0])
        for k in range(8):
            m = nb[:, k] >= 0
            sims[m] += (F[m] * F[nb[m, k]]).sum(1); cnt[m] += 1
        homog = sims / np.maximum(cnt, 1)
        hp = (pl.DataFrame({"place_id": pid, "homog": homog}).group_by("place_id")
              .agg(pl.col("homog").mean()).join(places, on="place_id"))
        okh = hp["wiki_visibility_score"].is_not_null() & hp["wiki_visibility_score"].is_not_nan()
        rh = spearmanr(hp.filter(okh)["homog"].to_numpy(), hp.filter(okh)["wiki_visibility_score"].to_numpy())
        by_bin = (hp.group_by("wiki_visibility_bin").agg(pl.col("homog").mean(), pl.len())
                  .sort("wiki_visibility_bin").to_dicts())
        rp = spearmanr(hp.filter(okh)["homog"].to_numpy(), hp.filter(okh)["log_pop"].to_numpy())
        # densest kinds
        hk = (pl.DataFrame({"kind": kl, "homog": homog}).group_by("kind").agg(pl.col("homog").mean(), pl.len())
              .sort("homog", descending=True))
        dens = [{"kind": int(r["kind"]), "name": names.get(int(r["kind"]), ""), "homog": float(r["homog"]),
                 "n_tiles": int(r["len"])} for r in hk.head(5).to_dicts()]
        sparse = [{"kind": int(r["kind"]), "name": names.get(int(r["kind"]), ""), "homog": float(r["homog"]),
                   "n_tiles": int(r["len"])} for r in hk.tail(3).to_dicts()]
        out["repetitiveness"] = {
            "homog_mean": float(homog.mean()), "homog_p10": float(np.percentile(homog, 10)),
            "homog_p90": float(np.percentile(homog, 90)),
            "spearman_vs_documentation": {"rho": float(rh.statistic), "p": float(rh.pvalue)},
            "spearman_vs_log_population": {"rho": float(rp.statistic), "p": float(rp.pvalue)},
            "by_documentation_bin": by_bin,
            "densest_kinds": dens, "sparsest_kinds": sparse,
        }
        per_place = per_place.join(hp.select(["place_id", pl.col("homog").alias("wall_homogeneity")]),
                                   on="place_id", how="left")
        print("A4 homogeneity vs documentation rho=%.3f; vs log pop rho=%.3f" % (rh.statistic, rp.statistic))
        print("   densest kinds:", [d["name"][:40] for d in dens[:3]])
    return out, kinds_df, per_place


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grid", default=None,
                    help=f"primary grid; default: {EXHIBITED_GRID} if found, else grid_default_wall.parquet")
    ap.add_argument("--recipes", default="balanced,color,global,highdim,local,semantic")
    ap.add_argument("--out", default=str(RESULTS))
    ap.add_argument("--rename-only", action="store_true", help="only refresh the type names in atlas_kinds.parquet")
    args = ap.parse_args()
    if args.rename_only:
        names = kind_names(load_places())
        kp, jp = Path(args.out) / "atlas_kinds.parquet", Path(args.out) / "atlas_wall.json"
        k = pl.read_parquet(kp)
        k.with_columns(pl.Series("name", [names.get(int(i), f"kind {i}") for i in k["kind"]])).write_parquet(kp)

        def walk(o):
            if isinstance(o, dict):
                for key, nm in (("kind", "name"), ("a", "a_name"), ("b", "b_name")):
                    if isinstance(o.get(key), int) and nm in o:
                        o[nm] = names.get(o[key], o[nm])
                for v in o.values():
                    walk(v)
            elif isinstance(o, list):
                for v in o:
                    walk(v)
        J = json.loads(jp.read_text(encoding="utf-8"))
        walk(J)
        jp.write_text(json.dumps(J, indent=1), encoding="utf-8")
        print(f"renamed {len(names)} types in {kp.name} and {jp.name}")
        return 0
    grid = Path(args.grid) if args.grid else (find_layout(EXHIBITED_GRID)
                                              or find_layout("grid_default_wall.parquet"))
    if grid is None or not grid.exists():
        raise SystemExit("no wall grid found in any layout dir; pass --grid <parquet>")
    print(f"primary grid: {grid}  ({'EXHIBITED 5x3' if grid.name == EXHIBITED_GRID else 'default wall'})")
    print(f"sort features: {LAYOUT / 'features.npy'}")
    places = load_places()
    names = kind_names(places)
    print(f"places {places.height}; kind names for {len(names)} kinds")

    primary, kinds_df, per_place = analyse(load_grid(grid), places, names,
                                           with_features=True, tag=grid.stem)
    recipes = {}
    for r in [x for x in args.recipes.split(",") if x]:
        p = find_layout(f"grid_default_s_{r}.parquet")
        if p is None:
            print(f"  recipe s_{r}: grid not found, skipped")
            continue
        res, _k, _pp = analyse(load_grid(p), places, names, with_features=True, tag=f"s_{r}")
        recipes[r] = res

    # robustness table: min/max across recipes for the headline numbers
    def rng_of(path):
        vals = []
        for r in recipes.values():
            v = r
            for k in path:
                v = v[k]
            vals.append(v)
        return {"min": float(min(vals)), "max": float(max(vals)), "n": len(vals)} if vals else None
    robust = {
        "spread_median_stable_frac": rng_of(["four_tile_spread", "median_frac_diag_stable"]),
        "spread_median_unstable_frac": rng_of(["four_tile_spread", "median_frac_diag_unstable"]),
        "spread_median_random_frac": rng_of(["four_tile_spread", "median_frac_diag_random"]),
        "spread_rho": rng_of(["four_tile_spread", "spearman_dispersion_vs_spread", "rho"]),
        "homog_vs_documentation_rho": rng_of(["repetitiveness", "spearman_vs_documentation", "rho"]),
        "homog_vs_log_population_rho": rng_of(["repetitiveness", "spearman_vs_log_population", "rho"]),
        "legibility": {lab: rng_of(["legibility", lab, "lift"]) for lab in primary["legibility"]},
        "legibility_beyond_region": {lab: rng_of(["legibility", lab, "lift_beyond_region"])
                                     for lab, v in primary["legibility"].items()
                                     if "lift_beyond_region" in v},
        "median_kind_compactness": rng_of(["kinds_summary", "median_compactness_ratio"]),
    }
    out = {"_provenance": {"seed": SEED, "primary_grid": str(grid), "stable_cut": STABLE_CUT,
                           "sort_features": str(LAYOUT / "features.npy"), "recipes": list(recipes)},
           "primary": primary, "recipes": recipes, "robustness": robust}
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / "atlas_wall.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    kinds_df.write_parquet(Path(args.out) / "atlas_kinds.parquet")
    per_place.write_parquet(Path(args.out) / "atlas_per_place.parquet")
    print("\nwrote atlas_wall.json, atlas_kinds.parquet, atlas_per_place.parquet")
    print("\nROBUSTNESS (min-max over %d recipes):" % len(recipes))
    for k, v in robust.items():
        if k.startswith("legibility"):
            for lab, r in v.items():
                if r:
                    print(f"  {k}.{lab:14s} {r['min']:.3f} .. {r['max']:.3f}")
        elif v:
            print(f"  {k:32s} {v['min']:.3f} .. {v['max']:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
