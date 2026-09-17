"""The topology of the model's types, measured in representation space and
checked against the atlas.

 1. Type graph. k-nearest-neighbour links between places (640-D analysis
    features, cosine) aggregated to the 64 types: closure (links that stay
    inside a type), affinity between types (links beyond chance), degree,
    betweenness, communities. Which types are closed ends, which are hubs?
 2. Does the atlas keep that topology? Type adjacency on the exhibited grid
    against type affinity in representation space.
 3. Redraws move along the graph. When the fifth, independent probe of a
    place lands in another type, is it a neighbouring type?
 4. How many types are there? Cluster quality and stability for k = 8..512,
    nesting between resolutions, and a density-based count (HDBSCAN).

    python scripts/topology.py
Writes results/topology.json and type_graph.parquet.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from latent_earth.discover import find_layout  # noqa: E402

RESULTS = ROOT / "results"
SEED = 20260613
K_NN = 10


def unit(X):
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-8)


def knn(Q: np.ndarray, G: np.ndarray, k: int, exclude_self: bool) -> np.ndarray:
    out = np.empty((Q.shape[0], k), np.int64)
    for i in range(0, Q.shape[0], 2000):
        sim = Q[i:i + 2000] @ G.T
        if exclude_self:
            sim[np.arange(sim.shape[0]), np.arange(i, i + sim.shape[0])] = -2
        idx = np.argpartition(-sim, k, axis=1)[:, :k]
        out[i:i + 2000] = np.take_along_axis(idx, np.argsort(-np.take_along_axis(sim, idx, 1), 1), 1)
    return out


def main() -> int:
    import networkx as nx
    from scipy.stats import spearmanr
    from sklearn.cluster import HDBSCAN, MiniBatchKMeans
    from sklearn.decomposition import PCA
    from sklearn.metrics import adjusted_rand_score, silhouette_score

    rng = np.random.default_rng(SEED)
    feats = find_layout("analysis_features.npy")
    ids = pl.read_parquet(feats.with_name("analysis_feature_tile_ids.parquet"))["tile_id"].to_list()
    F = np.load(feats, mmap_mode="r")
    pid_t = np.array([t.split("__")[0] for t in ids])
    slot = np.array([int(t.split("__")[1]) for t in ids])
    meta = pl.read_parquet(RESULTS / "poverty_per_place.parquet").select(
        ["place_id", pl.col("kmeans_label").alias("type"), "un_region", "lat", "lng"])
    same = dict(pl.read_parquet(RESULTS / "prompt_subsets.parquet").select(["place_id", "identical_prompts"]).iter_rows())
    places = meta["place_id"].to_list()
    pmap = {p: i for i, p in enumerate(places)}
    pidx = np.array([pmap.get(p, -1) for p in pid_t])
    ok = pidx >= 0
    X = np.array(F, dtype=np.float32)                    # a writable copy
    for s in range(5):                                   # attempt means out
        m = ok & (slot == s)
        X[m] -= X[m].mean(0)
    n = len(places)
    typ = meta["type"].to_numpy().astype(int)
    K = int(typ.max() + 1)
    size = np.bincount(typ, minlength=K)
    names = dict(pl.read_parquet(RESULTS / "atlas_kinds.parquet").select(["kind", "name"]).iter_rows())
    kinds = pl.read_parquet(RESULTS / "atlas_kinds.parquet")
    out: dict = {"features": feats.name, "n_places": n, "k_nn": K_NN}   # file name only: no machine paths in results

    # ------------------------------------------------------------ 1. type graph
    P = np.zeros((n, X.shape[1]), np.float32)
    m = ok & (slot < 4)
    np.add.at(P, pidx[m], unit(X[m]))
    P = unit(P)
    nb = knn(P, P, K_NN, exclude_self=True)
    C = np.zeros((K, K))
    np.add.at(C, (np.repeat(typ, K_NN), typ[nb].ravel()), 1)
    T = C / C.sum(1, keepdims=True)                      # where a type's neighbour links go
    share = size / size.sum()
    L = T / share[None, :]                               # beyond chance
    S = np.sqrt(L * L.T)                                 # symmetric affinity
    closure = np.diag(T)
    G = nx.Graph()
    G.add_nodes_from(range(K))
    for i in range(K):
        for j in range(i + 1, K):
            if S[i, j] >= 1.0:                           # linked more often than chance, both ways
                G.add_edge(i, j, weight=float(S[i, j]), dist=1.0 / float(S[i, j]))
    deg = np.array([G.degree(i) for i in range(K)])
    strength = np.array([sum(d["weight"] for _, _, d in G.edges(i, data=True)) for i in range(K)])
    btw = nx.betweenness_centrality(G, weight="dist")
    comms = nx.community.louvain_communities(G, weight="weight", seed=1, resolution=1.0)
    comm_of = np.zeros(K, int)
    for c, members in enumerate(sorted(comms, key=len, reverse=True)):
        for i in members:
            comm_of[i] = c
    modularity = nx.community.modularity(G, comms, weight="weight")
    reg = meta["un_region"].cast(pl.Utf8).to_numpy().astype(str)
    fam = []
    for c in range(comm_of.max() + 1):
        mem = np.flatnonzero(comm_of == c)
        pl_in = np.isin(typ, mem)
        r, cnt = np.unique(reg[pl_in], return_counts=True)
        top = sorted(zip(cnt / pl_in.sum(), r), reverse=True)[:3]
        fam.append({"family": c, "n_types": int(mem.size), "n_places": int(pl_in.sum()),
                    "top_regions": [[rr, float(s)] for s, rr in top],
                    "types": [names.get(int(i), str(i)) for i in mem]})
    geo = dict(kinds.select(["kind", "geo_km"]).iter_rows())
    comp = dict(kinds.select(["kind", "compactness_ratio"]).iter_rows())
    gk = np.array([geo[i] for i in range(K)]); ck = np.array([comp[i] for i in range(K)])
    out["type_graph"] = {
        "closure_median": float(np.median(closure)), "closure_min": float(closure.min()), "closure_max": float(closure.max()),
        "closure_of_places": float((typ[nb] == typ[:, None]).mean()),
        "n_edges": G.number_of_edges(), "degree_median": float(np.median(deg)), "degree_max": int(deg.max()),
        "connected": nx.is_connected(G), "n_components": nx.number_connected_components(G),
        "diameter_hops": (nx.diameter(G) if nx.is_connected(G) else None),
        "mean_path_hops": (float(nx.average_shortest_path_length(G)) if nx.is_connected(G) else None),
        "modularity": float(modularity), "n_families": int(comm_of.max() + 1), "families": fam,
        "spearman_degree_vs_geo_spread": float(spearmanr(deg, gk).statistic),
        "spearman_closure_vs_geo_spread": float(spearmanr(closure, gk).statistic),
        "spearman_closure_vs_atlas_compactness": float(spearmanr(closure, ck).statistic),
        "spearman_degree_vs_atlas_compactness": float(spearmanr(deg, ck).statistic),
        "most_closed": [{"type": names[int(i)], "closure": float(closure[i]), "degree": int(deg[i])} for i in np.argsort(-closure)[:8]],
        "most_open": [{"type": names[int(i)], "closure": float(closure[i]), "degree": int(deg[i])} for i in np.argsort(closure)[:8]],
        "hubs": [{"type": names[int(i)], "degree": int(deg[i]), "betweenness": float(btw[i]), "geo_km": float(gk[i])}
                 for i in sorted(range(K), key=lambda i: -btw[i])[:8]],
    }
    # links between families run through which types?
    inter = [(i, j) for i, j in G.edges() if comm_of[i] != comm_of[j]]
    bridge = sorted({i for e in inter for i in e})
    rest = [i for i in range(K) if i not in bridge]
    giant = G.subgraph(max(nx.connected_components(G), key=len))
    out["type_graph"].update({
        "n_links_between_families": len(inter), "n_bridge_types": len(bridge),
        "bridge_types_median_geo_km": float(np.median(gk[bridge])), "other_types_median_geo_km": float(np.median(gk[rest])),
        "component_sizes": sorted((len(c) for c in nx.connected_components(G)), reverse=True),
        "giant_diameter_hops": int(nx.diameter(giant)), "giant_mean_path_hops": float(nx.average_shortest_path_length(giant)),
        "spearman_betweenness_vs_geo_spread": float(spearmanr([btw[i] for i in range(K)], gk).statistic)})
    # how far each world region's places concentrate in one family
    conc = {}
    for r_ in np.unique(reg):
        mreg = reg == r_
        if mreg.sum() >= 300:
            f_ = np.bincount(comm_of[typ[mreg]], minlength=comm_of.max() + 1)
            conc[str(r_)] = {"n": int(mreg.sum()), "family": int(f_.argmax()), "share": float(f_.max() / mreg.sum())}
    out["type_graph"]["region_concentration"] = conc
    # same-family share of links and of regions
    out["type_graph"]["links_within_family"] = float(sum(C[i, j] for i in range(K) for j in range(K) if comm_of[i] == comm_of[j]) / C.sum())

    # -------------------------------------------------- 2. does the atlas keep it?
    grid_p = find_layout("grid_default_wall_s_highdim_5x3.parquet")
    if grid_p is not None:
        g = pl.read_parquet(grid_p).select(["tile_id", "row", "col"]).with_columns(
            pl.col("tile_id").str.split("__").list.get(0).alias("place_id")).join(
            meta.select(["place_id", "type"]), on="place_id")
        rows, cols, gt = g["row"].to_numpy(), g["col"].to_numpy(), g["type"].to_numpy().astype(int)
        H, W = rows.max() + 1, cols.max() + 1
        lut = -np.ones((H, W), int); lut[rows, cols] = gt
        A = np.zeros((K, K))
        for a, b in ((lut[:, :-1], lut[:, 1:]), (lut[:-1, :], lut[1:, :])):
            v = (a >= 0) & (b >= 0)
            np.add.at(A, (a[v], b[v]), 1); np.add.at(A, (b[v], a[v]), 1)
        tile_share = np.bincount(gt, minlength=K) / gt.size
        LA = (A / A.sum(1, keepdims=True)) / tile_share[None, :]
        SA = np.sqrt(LA * LA.T)
        iu = np.triu_indices(K, 1)
        edge_on_atlas = np.mean([SA[i, j] >= 1.0 for i, j in G.edges()])
        med = np.array([[np.median(rows[gt == i]), np.median(cols[gt == i])] for i in range(K)])
        out["atlas_vs_space"] = {
            "spearman_adjacency_vs_affinity": float(spearmanr(SA[iu], S[iu]).statistic),
            "share_of_graph_edges_that_touch_on_the_atlas": float(edge_on_atlas),
            "share_of_atlas_contacts_that_are_graph_edges": float(np.mean([G.has_edge(i, j) for i, j in zip(*iu) if SA[i, j] >= 1.0])),
            "same_type_neighbour_share_on_atlas": float(np.trace(A) / A.sum()),
        }
    else:
        med, SA = np.full((K, 2), np.nan), None
        print("exhibited grid not found: skipping the atlas comparison")

    # ------------------------------------------------ 3. redraws move along the graph
    use = np.array([same.get(p, False) for p in places])
    img = slice(0, 256)                                   # DINOv2 + SigLIP 2 blocks
    Sl = np.zeros((5, n, 256), np.float32)
    for s in range(5):
        mm = ok & (slot == s)
        Sl[s, pidx[mm]] = unit(X[mm][:, img])
    u = np.flatnonzero(use)
    Gal = unit(Sl[:3, u].mean(0))
    near = {}
    for nm, q in (("fourth_seed", Sl[3, u]), ("fifth_probe", Sl[4, u])):
        near[nm] = typ[u][knn(unit(q), Gal, 1, exclude_self=True)[:, 0]]
    own = typ[u]
    top3 = np.argsort(-np.where(np.eye(K, dtype=bool), -1, S), 1)[:, :3]
    res = {}
    for nm, land in near.items():
        stay = land == own
        nbr = np.array([land[i] in top3[own[i]] for i in range(own.size)])
        linked = np.array([(land[i] == own[i]) or G.has_edge(int(own[i]), int(land[i])) for i in range(own.size)])
        res[nm] = {"same_type": float(stay.mean()), "one_of_three_nearest_types": float((nbr & ~stay).mean()),
                   "linked_type_or_same": float(linked.mean()), "elsewhere": float((~linked).mean())}
    res["chance_three_nearest_types"] = float(np.mean([share[top3[t]].sum() for t in own]))
    res["chance_linked_or_same"] = float(np.mean([share[t] + sum(share[j] for j in G.neighbors(int(t))) for t in own]))
    out["redraws"] = res

    # --------------------------------------------------------- 4. how many types?
    sub = rng.choice(n, 8000, replace=False)
    half = rng.permutation(n)
    A_, B_ = half[: n // 2], half[n // 2:]
    sweep, prev = [], None
    for k in (8, 16, 32, 64, 128, 256, 512):
        t0 = time.time()
        lab = MiniBatchKMeans(k, random_state=0, batch_size=4096, n_init=3).fit(P)
        la = MiniBatchKMeans(k, random_state=1, batch_size=4096, n_init=3).fit(P[A_]).predict(P[sub])
        lb = MiniBatchKMeans(k, random_state=2, batch_size=4096, n_init=3).fit(P[B_]).predict(P[sub])
        row = {"k": k, "silhouette": float(silhouette_score(P[sub], lab.labels_[sub], metric="cosine")),
               "stability_ari": float(adjusted_rand_score(la, lb)),
               "within_cluster_share_of_variance": float(lab.inertia_ / ((P - P.mean(0)) ** 2).sum())}
        if prev is not None:                              # do finer clusters nest inside coarser ones?
            tab = np.zeros((k, prev.max() + 1)); np.add.at(tab, (lab.labels_, prev), 1)
            row["nesting_purity"] = float(tab.max(1).sum() / tab.sum())
        prev = lab.labels_
        sweep.append(row)
        print(row, f"({time.time() - t0:.0f}s)")
    Z = PCA(30, random_state=0).fit_transform(P)
    hd = HDBSCAN(min_cluster_size=40, min_samples=10).fit(Z)
    n_cl = int(hd.labels_.max() + 1)
    out["number_of_types"] = {"sweep": sweep, "hdbscan": {"min_cluster_size": 40, "n_clusters": n_cl,
                              "share_in_no_cluster": float((hd.labels_ < 0).mean()),
                              "largest_cluster_share": float(np.bincount(hd.labels_[hd.labels_ >= 0]).max() / n)}}
    print(out["number_of_types"]["hdbscan"])

    pl.DataFrame({"type": np.arange(K), "name": [names[i] for i in range(K)], "n_places": size, "closure": closure,
                  "degree": deg, "strength": strength, "betweenness": [btw[i] for i in range(K)], "family": comm_of,
                  "atlas_row": med[:, 0], "atlas_col": med[:, 1]}).write_parquet(RESULTS / "type_graph_nodes.parquet")
    e = [(i, j, d["weight"]) for i, j, d in G.edges(data=True)]
    pl.DataFrame({"a": [x[0] for x in e], "b": [x[1] for x in e], "affinity": [x[2] for x in e]}).write_parquet(RESULTS / "type_graph.parquet")
    (RESULTS / "topology.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k in ("atlas_vs_space", "redraws")}, indent=1))
    tg = out["type_graph"]
    print({k: v for k, v in tg.items() if not isinstance(v, list)})
    for f in tg["families"]:
        print(f["family"], f["n_types"], f["n_places"], f["top_regions"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
