"""Figure 6 - The atlas and the topology of types.

The atlas is the projection of 159,960 renderings from the 640-dimensional
representation space onto a grid. The figure shows what that projection
keeps and what structure it reveals.

  (a) the atlas coloured by family: the 64 types group into nine families
      (communities of the type graph, scripts/topology.py); a family's types
      differ in lightness. Families hold territories.
  (b) the type graph drawn at atlas coordinates: nodes are types at the
      median position of their tiles, links are affinities measured in
      representation space (nearest-neighbour links beyond chance). Links
      inside a family in the family's colour, the few links between families
      in black: families touch through their least place-specific types.
  (c) what the atlas sorts by: join-count lift per label, with what remains
      beyond regional sorting and the range over six alternative layouts.
  (d) repetitiveness: where neighbouring tiles are most alike.

Reads: atlas_wall.json, atlas_kinds.parquet, atlas_per_place.parquet,
       type_graph_nodes.parquet, type_graph.parquet, topology.json,
       poverty_per_place.parquet, and the exhibited grid (not released).
"""

from __future__ import annotations

import colorsys
import sys
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.colors import ListedColormap

import style

style.apply_style()
sys.path.insert(0, str(style.RESULTS.parent))
from latent_earth.discover import find_layout  # noqa: E402

A = style.load_json("atlas_wall.json")
T = style.load_json("topology.json")
P = A["primary"]
nodes = pl.read_parquet(style.RESULTS / "type_graph_nodes.parquet")
edges = pl.read_parquet(style.RESULTS / "type_graph.parquet")
per_place = pl.read_parquet(style.RESULTS / "atlas_per_place.parquet")

grid_path = Path(A["_provenance"]["primary_grid"])
if not grid_path.exists():
    grid_path = find_layout(grid_path.name)
    if grid_path is None:
        raise SystemExit("exhibited grid not found in any layout dir (it is not part of the public release)")
grid = pl.read_parquet(grid_path).select(["tile_id", "row", "col"]).with_columns(
    pl.col("tile_id").str.split("__").list.get(0).alias("place_id"))
meta = pl.read_parquet(style.RESULTS / "poverty_per_place.parquet").select(["place_id", "kmeans_label"])
t = grid.join(meta, on="place_id").join(per_place, on="place_id", how="left")
H, W = int(t["row"].max() + 1), int(t["col"].max() + 1)
rows, cols, kl = t["row"].to_numpy(), t["col"].to_numpy(), t["kmeans_label"].to_numpy()

FAMILY = {0: ("South and South-east Asia, sub-Saharan Africa", "#009E73"), 1: ("North America", "#0072B2"),
          2: ("North Africa to South Asia", "#E69F00"), 3: ("Southern and Western Europe", "#CC79A7"),
          4: ("Central and Eastern Europe", "#56B4E9"), 5: ("Latin America", "#D55E00"),
          6: ("Japan and Korea", "#B8A800"), 7: ("China", "#7F3C8D"), 8: ("Britain and Ireland", "#3A3A3A"),
          9: ("no architecture", "#BDBDBD")}
fam = dict(zip(nodes["type"].to_list(), nodes["family"].to_list()))
K = nodes.height
# a type's colour: its family's hue, lightness stepped within the family
type_rgb = np.zeros((K, 3))
for f in FAMILY:
    members = [k for k in range(K) if fam[k] == f]
    base = np.array(mcolors.to_rgb(FAMILY[f][1]))
    for i, k in enumerate(members):
        mix = 0.0 if len(members) == 1 else 0.5 * i / (len(members) - 1)
        type_rgb[k] = base * (1 - mix) + np.ones(3) * mix


def raster(values, fill=np.nan):
    img = np.full((H, W), fill, dtype=float)
    img[rows, cols] = values
    return img


fig = plt.figure(figsize=(7.2, 9.0))
outer = fig.add_gridspec(3, 1, height_ratios=[1.0, 1.0, 0.70], hspace=0.20,
                         left=0.02, right=0.985, top=0.965, bottom=0.05)
top_a = outer[0].subgridspec(1, 2, width_ratios=[0.745, 0.255], wspace=0.02)
top_b = outer[1].subgridspec(1, 2, width_ratios=[0.745, 0.255], wspace=0.02)
gs_cd = outer[2].subgridspec(1, 2, width_ratios=[1.0, 0.9], wspace=0.30)

# ------------------------------------------------------------ (a) atlas by family
axa = fig.add_subplot(top_a[0])
img = np.ones((H, W, 3))
img[rows, cols] = type_rgb[kl]
axa.imshow(img, interpolation="nearest", aspect="equal")
axa.set_xticks([]); axa.set_yticks([]); axa.grid(False)
for s in axa.spines.values():
    s.set_visible(False)
tg = T["type_graph"]
axa.set_title(f"a   The atlas, coloured by family: the {K} types group into nine families",
              loc="left", fontweight="bold", pad=4, fontsize=8.2)
places_f = {f: int(nodes.filter(pl.col("family") == f)["n_places"].sum()) for f in FAMILY}
types_f = {f: int((nodes["family"] == f).sum()) for f in FAMILY}
leg = fig.add_subplot(top_a[1]); leg.axis("off")
leg.text(0.0, 1.0, f"{tg['links_within_family']:.0%} of nearest-neighbour links\nstay inside a family", transform=leg.transAxes,
         fontsize=6.2, va="top", color=style.INK, fontweight="bold", linespacing=1.2)
for i, f in enumerate(FAMILY):
    y = 0.86 - i * 0.088
    leg.text(0.0, y, "■", transform=leg.transAxes, fontsize=9, color=FAMILY[f][1], va="top")
    leg.text(0.085, y - 0.004, f"{FAMILY[f][0]}", transform=leg.transAxes, fontsize=5.7, color=style.INK, va="top")
    leg.text(0.085, y - 0.040, f"{types_f[f]} types, {places_f[f]:,} places", transform=leg.transAxes, fontsize=5.3,
             color=style.INK_MUTED, va="top")

# ------------------------------------------------------------ (b) the type graph
axb = fig.add_subplot(top_b[0])
pos = {r["type"]: (r["atlas_col"], r["atlas_row"]) for r in nodes.iter_rows(named=True)}
for a, b, w in sorted(edges.iter_rows(), key=lambda e: fam[e[0]] != fam[e[1]]):
    inter = fam[a] != fam[b]
    axb.plot([pos[a][0], pos[b][0]], [pos[a][1], pos[b][1]], color="#111111" if inter else FAMILY[fam[a]][1],
             lw=(0.9 + 0.5 * w) if inter else 0.4 + 0.12 * w, alpha=0.95 if inter else 0.55, zorder=3 if inter else 2,
             solid_capstyle="round")
for r in nodes.iter_rows(named=True):
    axb.scatter(*pos[r["type"]], s=6 + r["n_places"] / 9, color=FAMILY[r["family"]][1], edgecolor="white", linewidth=0.5, zorder=4)
axb.set_xlim(0, W); axb.set_ylim(H, 0); axb.set_aspect("equal")
axb.set_xticks([]); axb.set_yticks([]); axb.grid(False)
for s in axb.spines.values():
    s.set_visible(False)
bridge = {a for a, b, _ in edges.iter_rows() if fam[a] != fam[b]} | {b for a, b, _ in edges.iter_rows() if fam[a] != fam[b]}
LABELS = {"glass curtain wall": ("glass towers", (6, 6)), "open fields (66%)": ("open fields", (6, -11)),
          "thatched roofs (60%)": ("thatch and mud-brick", (6, 6)), "wooden piers": ("waterfronts", (6, 5)),
          "lush vegetation": ("low houses in greenery", (7, -3)), "brick walls (98%)": ("brick terraces\nwith chimneys", (-62, 6))}
for r in nodes.iter_rows(named=True):
    for key, (lab, off) in LABELS.items():
        if r["name"].startswith(key):
            axb.annotate(lab, pos[r["type"]], xytext=off, textcoords="offset points", fontsize=5.8, color=style.INK,
                         zorder=6, bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))
n_inter = sum(1 for a, b, _ in edges.iter_rows() if fam[a] != fam[b])
axb.set_title("b   The type graph, drawn at atlas coordinates", loc="left", fontweight="bold", pad=4, fontsize=8.2)
av = T.get("atlas_vs_space", {})
note = fig.add_subplot(top_b[1]); note.axis("off")
note.text(0.0, 1.0,
          f"Nodes: types, at the median position\nof their tiles, sized by places.\n\n"
          f"Links: affinity measured in the\n640-dimensional representation\nspace, not in the atlas. {edges.height} links,\n"
          f"{n_inter} of them between families\n(black).\n\n"
          f"Types with a link to another family\nare spread over {tg['bridge_types_median_geo_km']:,.0f} km (median);\n"
          f"types without one, {tg['other_types_median_geo_km']:,.0f} km.\nFamilies touch through their least\nplace-specific types.\n\n"
          f"The atlas keeps the graph: rank\ncorrelation {av.get('spearman_adjacency_vs_affinity', float('nan')):.2f} between adjacency\n"
          f"in the atlas and affinity in\nrepresentation space; {av.get('share_of_graph_edges_that_touch_on_the_atlas', float('nan')):.0%} of links\ntouch in the atlas.",
          transform=note.transAxes, fontsize=5.7, color=style.INK, va="top", linespacing=1.25)

# ------------------------------------------------------------ (c) legibility
axc = fig.add_subplot(gs_cd[0])
order = ["region", "kind", "country", "climate", "population", "documentation"]
labels = {"kind": "type (of 64)", "region": "world region", "climate": "climate zone",
          "country": "country", "population": "population band", "documentation": "online documentation"}
lifts = [P["legibility"][k]["lift"] for k in order]
beyond = [P["legibility"][k].get("lift_beyond_region") for k in order]
rb = A["robustness"]["legibility"]
y = np.arange(len(order))
axc.barh(y, lifts, color="#C9D3DA", height=0.62, label="in the atlas")
bv = [b if b is not None else v for b, v in zip(beyond, lifts)]
axc.barh(y, bv, color="#3D5A6E", height=0.62, label="beyond what regional sorting explains")
for yi, k in enumerate(order):
    if rb.get(k):
        axc.plot([rb[k]["min"], rb[k]["max"]], [yi + 0.42, yi + 0.42], color=style.INK, lw=1.0, solid_capstyle="butt")
axc.set_yticks(y); axc.set_yticklabels([labels[k] for k in order], fontsize=6.6)
axc.invert_yaxis()
axc.set_xlim(0, max(0.78, max(lifts) * 1.15))
axc.set_xlabel("neighbouring tiles share the label, beyond chance (0 = no pattern, 1 = sorted)", fontsize=6.2)
axc.set_title("c   What the atlas sorts by", loc="left", fontweight="bold", pad=4, fontsize=8.2)
axc.grid(axis="y", visible=False)
for yi, v in enumerate(lifts):
    axc.text(v + 0.012, yi, f"{v:.2f}", va="center", fontsize=6, color=style.INK)
axc.legend(loc="lower right", fontsize=5.4, frameon=False)

# ------------------------------------------------------------ (d) repetitiveness
axd = fig.add_subplot(gs_cd[1])
hv = t["wall_homogeneity"].to_numpy().astype(float)
axd.imshow(raster(hv), cmap="Greys", interpolation="nearest", aspect="equal",
           vmin=np.nanpercentile(hv, 2), vmax=np.nanpercentile(hv, 98))
rho = P.get("repetitiveness", {}).get("spearman_vs_documentation", {}).get("rho", float("nan"))
axd.set_title(f"d   Repetitiveness (darker = neighbours more alike):\n     unrelated to documentation (rho = {rho:+.2f})",
              loc="left", fontweight="bold", pad=4, fontsize=8.2)
axd.set_xticks([]); axd.set_yticks([]); axd.grid(False)
for s in axd.spines.values():
    s.set_visible(False)

style.save(fig, "fig06_wall")
