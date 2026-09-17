"""Does the number of kinds matter? Re-cluster at k = 32, 64, 128 and recompute
the two type-level statistics the paper reports: each kind's geographic
spread (place / regional / continental split) and its compactness on the wall.

k = 64 was fixed in the frozen analysis plan as a display resolution (an 8 x 8
legend for the atlas) and reused for the pre-registered cluster-size outcome.
It is not a claim that the model has 64 types. This check shows which
statements survive a fourfold change in k.

    python scripts/kind_robustness.py                      # local: 512-D sort features
    python scripts/kind_robustness.py --features <root>/layout/analysis_features.npy \
        --ids <root>/layout/analysis_feature_tile_ids.parquet   # the 640-D analysis features
Writes results/kind_robustness.json.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from atlas_findings import LAYOUT, RESULTS, compactness, load_grid, load_places  # noqa: E402


def geo_spread(lat, lng):
    la, lo = np.radians(lat), np.radians(lng)
    ok = np.isfinite(la) & np.isfinite(lo)
    la, lo = la[ok], lo[ok]
    x, y, z = (np.cos(la) * np.cos(lo)).mean(), (np.cos(la) * np.sin(lo)).mean(), np.sin(la).mean()
    clat, clng = np.arctan2(z, np.hypot(x, y)), np.arctan2(y, x)
    a = np.sin((la - clat) / 2) ** 2 + np.cos(la) * np.cos(clat) * np.sin((lo - clng) / 2) ** 2
    return float((2 * 6371 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))).mean())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", default=None,
                    help="default: analysis_features.npy (640-D) if found, else the 512-D sort features")
    ap.add_argument("--ids", default=None)
    ap.add_argument("--grid", default=None, help="default: the exhibited wall grid, else grid_default_wall")
    ap.add_argument("--ks", default="32,64,128")
    args = ap.parse_args()
    from sklearn.cluster import MiniBatchKMeans
    from latent_earth.discover import find_layout

    feats = Path(args.features) if args.features else (find_layout("analysis_features.npy")
                                                       or find_layout("features.npy"))
    if feats is None:
        raise SystemExit("no feature file found in any layout dir; pass --features")
    ids_path = Path(args.ids) if args.ids else feats.with_name(
        "analysis_feature_tile_ids.parquet" if feats.name.startswith("analysis") else "feature_tile_ids.parquet")
    grid_path = Path(args.grid) if args.grid else (find_layout("grid_default_wall_s_highdim_5x3.parquet")
                                                   or find_layout("grid_default_wall.parquet"))
    print(f"features: {feats}\nids:      {ids_path}\ngrid:     {grid_path}")
    args.features, args.grid = str(feats), str(grid_path or "")

    ids = pl.read_parquet(ids_path)["tile_id"].to_list()
    F = np.load(args.features, mmap_mode="r")
    pid = np.array([t.split("__")[0] for t in ids])
    uniq, inv = np.unique(pid, return_inverse=True)
    P = np.zeros((uniq.size, F.shape[1]), np.float32)          # place = mean of its tiles
    np.add.at(P, inv, np.asarray(F, dtype=np.float32))
    P /= np.bincount(inv)[:, None]
    P /= np.maximum(np.linalg.norm(P, axis=1, keepdims=True), 1e-8)

    places = load_places()
    meta = pl.DataFrame({"place_id": uniq}).join(places.select(["place_id", "lat", "lng"]), on="place_id", how="left")
    lat, lng = meta["lat"].to_numpy(), meta["lng"].to_numpy()
    have_grid = grid_path is not None and Path(args.grid).exists()      # the exhibited layout is not public
    if have_grid:
        grid = load_grid(Path(args.grid)).join(pl.DataFrame({"place_id": uniq, "pidx": np.arange(uniq.size)}), on="place_id")
        pts = np.stack([grid["row"].to_numpy(), grid["col"].to_numpy()], 1).astype(float)
        pidx = grid["pidx"].to_numpy()
    rng = np.random.default_rng(2)

    out = {"features": Path(args.features).name, "dim": int(F.shape[1]), "n_places": int(uniq.size), "per_k": {}}
    for k in [int(x) for x in args.ks.split(",")]:
        t0 = time.time()
        lab = MiniBatchKMeans(n_clusters=k, random_state=0, batch_size=4096, n_init=3).fit_predict(P)
        sp = np.array([geo_spread(lat[lab == c], lng[lab == c]) for c in range(k)])
        ratios = np.array([np.nan])
        if have_grid:
            tl = lab[pidx]
            ratios = np.array([compactness(pts[tl == c], rng, pts, n_sub=600, n_null=4)["ratio"]
                               for c in range(k) if (tl == c).sum() >= 100])
        out["per_k"][k] = {
            "share_place_kinds_lt1000km": float(np.mean(sp < 1000)),
            "share_regional_1000_3000km": float(np.mean((sp >= 1000) & (sp < 3000))),
            "share_continental_ge3000km": float(np.mean(sp >= 3000)),
            "median_geo_spread_km": float(np.median(sp)),
            "median_wall_compactness": float(np.median(ratios)),
            "share_wall_compact_lt0_5": float(np.mean(ratios < 0.5)),
        }
        r = out["per_k"][k]
        print(f"k={k:3d}: places {r['share_place_kinds_lt1000km']:.0%} regional {r['share_regional_1000_3000km']:.0%} "
              f"continental {r['share_continental_ge3000km']:.0%} | wall median {r['median_wall_compactness']:.2f} "
              f"compact {r['share_wall_compact_lt0_5']:.0%}  ({time.time() - t0:.0f}s)")
    (RESULTS / "kind_robustness.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("wrote results/kind_robustness.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
