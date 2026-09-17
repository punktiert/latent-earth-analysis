"""How well is a place represented? Computed from the public release only
(https://doi.org/10.57967/HF/10090, downloaded to data/hf), so every number
is reproducible without the production archive.

 1. Architecture at all: share of places whose name-only image contains no
    architectural content (caption of the Step-1 image).
 2. Consistency, image channels only (DINOv2 + SigLIP 2; the prompt and
    Step-1 channels are identical across a place's four renderings by
    construction and are left out): four seeds of one description, the fifth
    independent probe, and other places, all as distances between single
    renderings.
 3. Individuation: does the fifth, independent probe retrieve its own place
    among all places? Against the same-description control (a fourth seed).
 4. Feature recurrence: which architectural features named in one caption
    of a place recur in its independent second caption, against another
    place of the same country.
 Each by record type (town/city vs metropolitan district) and population.

    python scripts/representation.py            # needs data/hf (see README in paper/results)
Writes results/representation.json and representation_per_place.parquet.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]
from latent_earth.paths import data_root  # noqa: E402
HF = data_root() / "hf"
RESULTS = ROOT / "results"
SEED = 20260613
POP_EDGES = [0, 15_000, 20_000, 30_000, 50_000, 100_000, 250_000, 500_000, 1_000_000, 2_000_000, 10**9]
POP_LABELS = ["10-15k", "15-20k", "20-30k", "30-50k", "50-100k", "100-250k", "250-500k", "0.5-1M", "1-2M", ">2M"]

# architectural vocabulary: category -> regex over a caption's feature phrases
LEXICON = {
    "material": {
        "brick": r"\bbrick", "stone": r"\bstone|sandstone|limestone|granite", "concrete": r"concrete",
        "timber": r"timber|wooden|\bwood\b|half-timber|clapboard|bamboo", "render": r"stucco|plaster|white-?wash|painted wall|white wall",
        "glass": r"\bglass", "metal": r"\bmetal|steel|\biron\b|corrugated", "earth": r"mud|adobe|earthen|\bclay\b|rammed",
    },
    "roof": {
        "flat roof": r"flat[- ]roof|flat roofs|roof terrace", "pitched roof": r"pitched|gabled|\bgable|steep", "tiled roof": r"til(?:e|ed) roof|terracotta|red[- ]tiled|roof tiles",
        "slate roof": r"slate", "thatch": r"thatch", "metal roof": r"corrugated|metal roof|tin roof", "dome": r"\bdome", "tiered roof": r"tiered|pagoda|upturned|curved eaves|pyramidal",
    },
    "element": {
        "arch": r"\barch", "column": r"column|pillar|colonnade|portico", "balcony": r"balcon", "shutter": r"shutter", "chimney": r"chimney",
        "tower": r"tower|spire|steeple|minaret|belfry", "storefront": r"storefront|shop|market|stall|awning|signage", "courtyard": r"courtyard|patio",
        "porch": r"porch|veranda|verandah", "stair": r"stair|steps\b|stepped", "fence": r"fence|railing|\bgate",
    },
    "street": {
        "cobblestone": r"cobble", "paved": r"\bpaved|paving|asphalt|sidewalk|crosswalk", "unpaved": r"dirt|unpaved|sandy|dusty|gravel",
        "alley": r"alley|narrow street|narrow lane", "wires": r"utility pole|power line|overhead wir|wiring|cables",
    },
    "setting": {
        "mountain": r"mountain|hill", "palm": r"palm", "trees": r"\btree|greenery|vegetation|lush|garden|hedge", "arid": r"arid|desert|\bdry\b|barren",
        "water": r"river|canal|\bsea\b|ocean|beach|harbou?r|waterfront|\blake|dock|pier", "snow": r"snow", "field": r"\bfield|rural|farmland|pasture",
    },
}
NOARCH = r"(?i)\bnone\b|no architectural|no building|no visible"


def load_channel(name: str, order: dict[str, int]) -> np.ndarray:
    files = sorted((HF / "embeddings" / name).glob("*.parquet"))
    out = None
    for f in files:
        t = pq.read_table(f)
        ids = t.column("tile_id").to_pylist()
        col = t.column(name).combine_chunks()
        dim = col.type.list_size
        arr = np.asarray(col.flatten()).reshape(-1, dim)
        if out is None:
            out = np.zeros((len(order), dim), np.float16)
        out[[order[i] for i in ids]] = arr
    return out


def whiten128(X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """The paper's channel reduction: 128 whitened PCs fitted on 30,000 images, L2-normalised."""
    from sklearn.decomposition import PCA
    sub = rng.choice(X.shape[0], 30_000, replace=False)
    pca = PCA(n_components=128, whiten=True, random_state=0).fit(X[sub].astype(np.float32))
    Z = np.concatenate([pca.transform(X[i:i + 20_000].astype(np.float32)) for i in range(0, X.shape[0], 20_000)])
    return (Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-8)).astype(np.float32)


def unit(X):
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-8)


def space_measures(Z: np.ndarray, slot: np.ndarray, pidx: np.ndarray, n_places: int, same: np.ndarray, rng) -> dict:
    """Z: (n_tiles, d) unit rows. Returns per-place arrays (NaN where undefined)."""
    Z = Z.copy()
    for s in range(5):                                   # slot means out, as everywhere in the paper
        Z[slot == s] -= Z[slot == s].mean(0)
    Z = unit(Z)
    S = np.zeros((5, n_places, Z.shape[1]), np.float32)
    for s in range(5):
        S[s, pidx[slot == s]] = Z[slot == s]
    pairs = [(a, b) for a in range(4) for b in range(a + 1, 4)]
    d_seed = np.mean([1 - (S[a] * S[b]).sum(1) for a, b in pairs], 0)
    d_indep = np.mean([1 - (S[4] * S[a]).sum(1) for a in range(4)], 0)
    use = np.flatnonzero(same)                           # queries and gallery: same-description places only
    G = unit(S[:3, use].mean(0))                         # gallery: mean of renderings 0-2
    out = {"d_seed": d_seed, "d_indep": d_indep}
    for nm, q in (("seed", S[3, use]), ("indep", S[4, use])):
        rank = np.empty(use.size, np.int32)
        nn20 = np.empty(use.size, np.float32)
        best = np.empty(use.size, np.int64)             # nearest OTHER place in the gallery
        for i in range(0, use.size, 2000):
            sim = q[i:i + 2000] @ G.T
            own = sim[np.arange(sim.shape[0]), np.arange(i, i + sim.shape[0])]
            rank[i:i + 2000] = (sim > own[:, None]).sum(1) + 1
            sim[np.arange(sim.shape[0]), np.arange(i, i + sim.shape[0])] = -2
            best[i:i + 2000] = use[sim.argmax(1)]
            if nm == "seed":                             # a single rendering to the 20 nearest OTHER places' single renderings
                s1 = S[3, use][i:i + 2000] @ S[0, use].T
                s1[np.arange(s1.shape[0]), np.arange(i, i + s1.shape[0])] = -1
                nn20[i:i + 2000] = 1 - np.sort(s1, 1)[:, -20:].mean(1)
        full = np.full(n_places, np.nan, np.float32)
        full[use] = rank
        out[f"rank_{nm}"] = full
        fb = np.full(n_places, -1, np.int64); fb[use] = best
        out[f"other_{nm}"] = fb
        if nm == "seed":
            f2 = np.full(n_places, np.nan, np.float32); f2[use] = nn20
            out["d_nn20"] = f2
    return out


def within_country_climate(Z, slot, pidx, per: pl.DataFrame, k: int = 10) -> dict:
    """Climate is the one covariate no prompt contains. Is it recoverable among places of the SAME country?"""
    Zc = Z.copy()
    for s in range(4):
        Zc[slot == s] -= Zc[slot == s].mean(0)
    P = np.zeros((per.height, Z.shape[1]), np.float32)
    np.add.at(P, pidx[slot < 4], unit(Zc[slot < 4]))
    P = unit(P)
    iso, clim = per["iso2"].to_numpy().astype(str), per["koppen_class"].to_numpy().astype(str)
    hit = base = n = 0
    for c in np.unique(iso):
        ix = np.flatnonzero((iso == c) & (clim != "None"))
        if ix.size < 50 or np.unique(clim[ix]).size < 2:
            continue
        sim = P[ix] @ P[ix].T
        np.fill_diagonal(sim, -2)
        nb = np.argsort(-sim, 1)[:, :k]
        for a, row in enumerate(nb):
            v, cnt = np.unique(clim[ix][row], return_counts=True)
            hit += v[cnt.argmax()] == clim[ix][a]
        v, cnt = np.unique(clim[ix], return_counts=True)
        base += cnt.max(); n += ix.size
    return {"n_places": int(n), "accuracy": float(hit / n), "country_majority_baseline": float(base / n)}


def population_by_record(rng) -> dict:
    """The registered size association, split by record type: the frozen plan's estimator
    (within-country rank slope, cluster bootstrap over countries), exploratory split."""
    sys.path.insert(0, str(ROOT))
    from latent_earth.util import within_group_rank_slope
    d = pl.read_parquet(RESULTS / "poverty_per_place.parquet").join(
        pl.read_parquet(HF / "data" / "places.parquet").select(["place_id", "source"]), on="place_id")
    out = {}
    ys = ("local_density", "centroid_dist", "megacluster_log2size", "genericity")
    for rec, g in (("town or city", d.filter(pl.col("source") != "district")), ("metropolitan district", d.filter(pl.col("source") == "district")),
                   ("town or city below 500k", d.filter((pl.col("source") != "district") & (pl.col("population") < 500_000)))):
        c = g["country"].cast(pl.Utf8).fill_null("?").to_numpy()
        out[rec] = {"n": g.height, "mean_centroid_dist": float(g["centroid_dist"].drop_nans().mean()), "mean_local_density": float(g["local_density"].drop_nans().mean())}
        for pred, x in (("log_population", np.log1p(g["population"].cast(pl.Float64).to_numpy())),
                        ("documentation", g["wiki_visibility_score"].cast(pl.Float64).fill_null(np.nan).to_numpy())):
            out[rec][pred] = {y: within_group_rank_slope(x, g[y].cast(pl.Float64).fill_null(np.nan).to_numpy(), c, rng=rng, n_bootstrap=1000)
                              for y in (ys if "below" not in rec else ys[:2])}
    return out


def summarise(df: pl.DataFrame, prefix: str) -> dict:
    r = {}
    for nm in ("seed", "indep"):
        x = df[f"{prefix}_rank_{nm}"].drop_nulls().drop_nans().to_numpy()
        r[nm] = {"n": int(x.size), "top1": float((x <= 1).mean()), "top10": float((x <= 10).mean()),
                 "top100": float((x <= 100).mean()), "median_rank": float(np.median(x))}
    for k in ("d_seed", "d_indep", "d_nn20"):
        x = df[f"{prefix}_{k}"].drop_nulls().drop_nans().to_numpy()
        r[k] = {"median": float(np.median(x)), "p10": float(np.quantile(x, .1)), "p90": float(np.quantile(x, .9))}
    a, b = df[f"{prefix}_d_indep"].to_numpy(), df[f"{prefix}_d_nn20"].to_numpy()
    ok = np.isfinite(a) & np.isfinite(b)
    r["share_indep_closer_than_nn20"] = float((a[ok] < b[ok]).mean())
    return r


def main() -> int:
    rng = np.random.default_rng(SEED)
    tiles = pl.read_parquet(HF / "data" / "tiles" / "*.parquet").select(["tile_id", "place_id", "sample_idx"])
    places = pl.read_parquet(HF / "data" / "places.parquet").select(
        ["place_id", "name", "source", "population", "iso2", "un_region", "koppen_class"])
    rev = pl.read_parquet(RESULTS / "prompt_subsets.parquet").select(["place_id", pl.col("identical_prompts").alias("same_prompt")])
    places = places.join(rev, on="place_id", how="left").with_columns(
        pl.col("population").cut(POP_EDGES[1:-1], labels=POP_LABELS, left_closed=True).cast(pl.Utf8).alias("pop_bin"),
        pl.when(pl.col("source") == "district").then(pl.lit("metropolitan district")).otherwise(pl.lit("town or city")).alias("record"))
    pid = places["place_id"].to_list()
    pmap = {p: i for i, p in enumerate(pid)}
    order = {t: i for i, t in enumerate(tiles["tile_id"].to_list())}
    slot = tiles["sample_idx"].to_numpy().astype(int)
    pidx = np.array([pmap[p] for p in tiles["place_id"].to_list()])
    same = places["same_prompt"].fill_null(False).to_numpy()
    kinds = pl.read_parquet(RESULTS / "poverty_per_place.parquet").select(["place_id", pl.col("kmeans_label").alias("kind")])
    per = places.join(kinds, on="place_id", how="left").with_columns(
        pl.col("name").str.split(", ").list.get(-2, null_on_oob=True).alias("parent"))
    dist = per.filter(pl.col("source") == "district")
    out: dict = {"n_places": places.height, "n_same_prompt": int(same.sum()), "n_district": dist.height,
                 "n_metropolises": dist.select(pl.concat_str(["parent", "iso2"], separator="|").n_unique()).item(),
                 "district_parent_population_min": int(dist["population"].min()),
                 "towns": {"n": per.height - dist.height, "population_median": float(per.filter(pl.col("source") != "district")["population"].median()),
                           "population_max": int(per.filter(pl.col("source") != "district")["population"].max())}}

    # ---- 1 + 4: captions
    cap = pl.read_parquet(HF / "data" / "captions.parquet").with_columns(
        pl.col("features").list.join("; ").str.to_lowercase().alias("txt"))
    c0 = cap.filter(pl.col("caption_slot") == "sample_0").select(["place_id", pl.col("txt").alias("t0")])
    c4 = cap.filter(pl.col("caption_slot") == "sample_4").select(["place_id", pl.col("txt").alias("t4")])
    per = per.join(c0, on="place_id", how="left").join(c4, on="place_id", how="left").with_columns(
        pl.col("t0").str.contains(NOARCH).alias("noarch_name_only"), pl.col("t4").str.contains(NOARCH).alias("noarch_architecture_prompt"))
    out["no_architecture"] = {"name_only": float(per["noarch_name_only"].mean()), "n_name_only": int(per["noarch_name_only"].sum()),
                              "architecture_prompt": float(per["noarch_architecture_prompt"].mean())}
    both = per.filter(~pl.col("noarch_name_only") & pl.col("t0").is_not_null() & pl.col("t4").is_not_null())
    t0, t4, iso = both["t0"].to_list(), both["t4"].to_list(), both["iso2"].to_numpy()
    perm_c = np.arange(len(t0))                          # another place of the same country
    for c in np.unique(iso.astype(str)):
        ix = np.flatnonzero(iso.astype(str) == c)
        perm_c[ix] = np.roll(rng.permutation(ix), 1) if ix.size > 1 else ix
    perm_w = rng.permutation(len(t0))
    reg = both["un_region"].cast(pl.Utf8).to_numpy().astype(str)
    perm_r = np.full(len(t0), -1)                        # same world region, another country
    for g in np.unique(reg):
        ix = np.flatnonzero(reg == g)
        for i in ix:
            cand = ix[iso[ix].astype(str) != str(iso[i])]
            if cand.size:
                perm_r[i] = rng.choice(cand)
    okr = perm_r >= 0
    rec, hits = {}, np.zeros((len(t0), 3)); denom = np.zeros(len(t0))
    four, hits_r = {}, np.zeros(len(t0))
    for cat, terms in LEXICON.items():
        for term, rx in terms.items():
            r = re.compile(rx)
            a = np.array([bool(r.search(s)) for s in t0]); b = np.array([bool(r.search(s)) for s in t4])
            if a.sum() < 300:
                continue
            rec[term] = {"category": cat, "n": int(a.sum()), "share_of_places": float(a.mean()),
                         "recurs_same_place": float(b[a].mean()), "recurs_same_country": float(b[perm_c][a].mean()),
                         "recurs_any_place": float(b[perm_w][a].mean())}
            denom += a; hits[:, 0] += a & b; hits[:, 1] += a & b[perm_c]; hits[:, 2] += a & b[perm_w]
            br = np.where(okr, b[np.maximum(perm_r, 0)], False)
            hits_r += a & br
            four[term] = {"category": cat, "same_place": rec[term]["recurs_same_place"], "same_country": rec[term]["recurs_same_country"],
                          "same_region_other_country": float(br[a & okr].mean()), "any_place": rec[term]["recurs_any_place"]}
    ok = denom > 0
    out["feature_recurrence"] = {"terms": rec, "mean_share_recurring": {
        "same_place": float((hits[ok, 0] / denom[ok]).mean()), "same_country": float((hits[ok, 1] / denom[ok]).mean()),
        "any_place": float((hits[ok, 2] / denom[ok]).mean())}}
    o4 = ok & okr
    out["feature_recurrence"]["four_controls"] = {"terms": four, "mean_share_recurring": {
        "same_place": float((hits[o4, 0] / denom[o4]).mean()), "same_country": float((hits[o4, 1] / denom[o4]).mean()),
        "same_region_other_country": float((hits_r[o4] / denom[o4]).mean()), "any_place": float((hits[o4, 2] / denom[o4]).mean())}}
    rec_pp = np.full(len(t0), np.nan); rec_pp[ok] = hits[ok, 0] / denom[ok]
    per = per.join(pl.DataFrame({"place_id": both["place_id"], "feature_recurrence": rec_pp}), on="place_id", how="left")

    # ---- 2 + 3: embedding spaces
    spaces = {"image": ["dino", "siglip"], "generator": ["flux_residual"]}
    for sp, chans in spaces.items():
        if not all((HF / "embeddings" / c).exists() for c in chans):
            print(f"skip {sp}: channel not downloaded"); continue
        Z = unit(np.concatenate([whiten128(load_channel(c, order), rng) for c in chans], 1))
        m = space_measures(Z, slot, pidx, places.height, same, rng)
        if sp == "image":
            out["within_country_climate"] = within_country_climate(Z, slot, pidx, per)
        for k in ("d_seed", "d_indep"):                  # consistency is defined for same-description places
            m[k] = np.where(same, m[k], np.nan)
        others = {k: m.pop(k) for k in ("other_seed", "other_indep")}
        per = per.with_columns([pl.Series(f"{sp}_{k}", v.astype(np.float32)) for k, v in m.items()])
        out[sp] = summarise(per, sp)
        # where does the independent probe land when it misses the place?
        j = others["other_indep"]; okj = j >= 0
        lab = {"country": per["iso2"].to_numpy().astype(str), "world_region": per["un_region"].to_numpy().astype(str),
               "climate": per["koppen_class"].to_numpy().astype(str), "kind": per["kind"].to_numpy().astype(str),
               "metropolis": per["parent"].to_numpy().astype(str)}
        out[sp]["nearest_other_place_shares"] = {k: float((v[okj] == v[j[okj]]).mean()) for k, v in lab.items() if k != "metropolis"}
        dist = okj & (per["source"].to_numpy() == "district")
        out[sp]["nearest_other_place_shares"]["metropolis_for_districts"] = float((lab["metropolis"][dist] == lab["metropolis"][j[dist]]).mean())
        print(sp, json.dumps(out[sp], indent=1))

    # ---- by record type and population
    def table(by: list[str]) -> list[dict]:
        rows = []
        for key, g in per.group_by(by, maintain_order=True):
            row = dict(zip(by, key)) | {"n": g.height, "no_architecture": float(g["noarch_name_only"].mean()),
                                        "feature_recurrence": float(g["feature_recurrence"].drop_nulls().drop_nans().mean())}
            for sp in spaces:
                if f"{sp}_rank_indep" in g.columns:
                    x = g[f"{sp}_rank_indep"].drop_nulls().drop_nans().to_numpy()
                    y = g[f"{sp}_rank_seed"].drop_nulls().drop_nans().to_numpy()
                    row |= {f"{sp}_n": int(x.size), f"{sp}_indep_top1": float((x <= 1).mean()), f"{sp}_indep_top10": float((x <= 10).mean()),
                            f"{sp}_indep_top100": float((x <= 100).mean()), f"{sp}_seed_top1": float((y <= 1).mean()),
                            f"{sp}_d_indep": float(np.nanmedian(g[f"{sp}_d_indep"].to_numpy())),
                            f"{sp}_d_seed": float(np.nanmedian(g[f"{sp}_d_seed"].to_numpy()))}
            rows.append(row)
        return rows
    def wilson(k, n, z=1.96):
        p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
        return [float((c - h) / d), float((c + h) / d)]

    def hinge(x, y, grid=np.arange(4.1, 6.81, 0.05)):
        best = (np.inf, None, None)
        for c in grid:
            h = np.maximum(0, x - c); A = np.stack([np.ones_like(h), h], 1)
            coef, res, *_ = np.linalg.lstsq(A, y, rcond=None)
            sse = float(((A @ coef - y) ** 2).sum())
            if sse < best[0]:
                best = (sse, float(c), float(coef[1]))
        return best[1], best[2]
    out["by_record"] = table(["record"])
    towns = per.filter(pl.col("record") == "town or city")
    per_all = per
    per = towns
    rows = table(["pop_bin"]); rows.sort(key=lambda r: POP_LABELS.index(r["pop_bin"]))
    for r in rows:
        r["image_indep_top10_ci95"] = wilson(round(r["image_indep_top10"] * r["image_n"]), r["image_n"])
        r["no_architecture_ci95"] = wilson(round(r["no_architecture"] * r["n"]), r["n"])
    out["towns_by_population"] = rows
    tw = towns.filter(pl.col("image_rank_indep").is_not_null() & pl.col("image_rank_indep").is_not_nan())
    x, y = np.log10(tw["population"].to_numpy()), np.log10(tw["image_rank_indep"].to_numpy())
    c, b = hinge(x, y)
    boot = [hinge(x[ix], y[ix])[0] for ix in (rng.integers(0, x.size, x.size) for _ in range(300))]
    from scipy.stats import spearmanr
    lo = tw.filter(pl.col("population") < 500_000)
    out["population_threshold"] = {
        "outcome": "log10 rank of own place for the independent probe, towns and cities, image space",
        "breakpoint_population": float(10 ** c), "breakpoint_ci95": [float(10 ** np.quantile(boot, .025)), float(10 ** np.quantile(boot, .975))],
        "slope_above_per_decade": b, "n": int(x.size),
        "spearman_below_500k": float(spearmanr(lo["population"].to_numpy(), lo["image_rank_indep"].to_numpy()).statistic),
        "spearman_all_towns": float(spearmanr(x, y).statistic)}
    print(json.dumps(out["population_threshold"], indent=1))
    per = per_all
    out["population_by_record"] = population_by_record(rng)
    per.drop(["t0", "t4"]).write_parquet(RESULTS / "representation_per_place.parquet")
    (RESULTS / "representation.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("no_architecture", "by_record")}, indent=1))
    for r in out["towns_by_population"]:
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})
    print("wrote results/representation.json, representation_per_place.parquet")
    return 0


if __name__ == "__main__":
    sys.exit(main())
