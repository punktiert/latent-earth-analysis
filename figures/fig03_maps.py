"""Figure 3 - The model's types, mapped onto the real earth (exhaustive).

All 64 kinds (k-means groups of similarly-rendered places, computed once for
the corpus) are accounted for:
  (a) every kind whose members lie on average within 1,000 km of the kind's
      geographic centre is drawn on the map, numbered, and listed with its
      dominant countries. These are the "types that are places".
  (b) the full inventory: all 64 kinds ranked by geographic spread, so the
      reader sees the split rather than a hand-picked pair.
  (c) the three most dispersed kinds: types that recur across continents.
Selection is by threshold and rank, never by hand.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import polars as pl
import matplotlib.pyplot as plt

import style

style.apply_style()

COMPACT_KM = 1000        # "a place": mean member distance to the kind's centre
MIN_PLACES = 150

d = pl.read_parquet(style.RESULTS / "poverty_per_place.parquet")
lat = d["lat"].to_numpy()
lng = d["lng"].to_numpy()
lab = d["kmeans_label"].to_numpy()
country = d["country"].to_numpy().astype(object)
sizes = np.bincount(lab)


def centre_and_spread(t: int):
    m = lab == t
    la, lo = np.radians(lat[m]), np.radians(lng[m])
    x, y, z = (np.cos(la) * np.cos(lo)).mean(), (np.cos(la) * np.sin(lo)).mean(), np.sin(la).mean()
    clat, clng = np.arctan2(z, np.hypot(x, y)), np.arctan2(y, x)
    a = np.sin((la - clat) / 2) ** 2 + np.cos(la) * np.cos(clat) * np.sin((lo - clng) / 2) ** 2
    km = float((2 * 6371 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))).mean())
    return np.degrees(clat), np.degrees(clng), km


info = {t: centre_and_spread(t) for t in range(sizes.size) if sizes[t] >= MIN_PLACES}
compact = sorted([t for t in info if info[t][2] < COMPACT_KM], key=lambda t: info[t][2])
dispersed = sorted(info, key=lambda t: -info[t][2])[:3]


def top_countries(t: int, k: int = 2) -> str:
    c = Counter(country[lab == t].tolist()).most_common(k)
    n = (lab == t).sum()
    return ", ".join(f"{name}" for name, cnt in c if cnt / n >= 0.12) or c[0][0]


# 23-way qualitative palette: Okabe-Ito hues first, then tab20 fills.
base = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00", "#56B4E9",
        "#F0E442", "#8B4513"]
extra = [c for c in plt.get_cmap("tab20").colors] + [c for c in plt.get_cmap("tab20b").colors]
palette = base + [plt.matplotlib.colors.to_hex(c) for c in extra]

fig = plt.figure(figsize=(7.2, 9.8))
gs = fig.add_gridspec(3, 1, height_ratios=[1.62, 0.55, 1.0], hspace=0.38,
                      left=0.04, right=0.98, top=0.96, bottom=0.03)


def basemap(ax):
    ax.scatter(lng, lat, s=0.7, c="#D9D9D5", linewidths=0, rasterized=True)
    ax.set_xlim(-180, 180); ax.set_ylim(-60, 78); ax.set_aspect(1.15)
    ax.grid(False); ax.set_xticks([]); ax.set_yticks([])
    for s in ("left", "bottom"):
        ax.spines[s].set_visible(False)


# ------------------------------------------------------ (a) types that are places
axa = fig.add_subplot(gs[0])
basemap(axa)
for k, t in enumerate(compact):
    m = lab == t
    axa.scatter(lng[m], lat[m], s=2.2, c=palette[k], linewidths=0, rasterized=True, zorder=2)
for k, t in enumerate(compact):
    clat, clng, km = info[t]
    axa.text(clng, clat, str(k + 1), fontsize=6.2, fontweight="bold", ha="center",
             va="center", color=style.INK, zorder=4,
             bbox=dict(boxstyle="circle,pad=0.15", fc="white", ec=palette[k], lw=0.9))
axa.set_title(f"a   {len(compact)} of the 64 types are places: their members lie on average "
              f"within {COMPACT_KM:,} km of one centre",
              loc="left", fontweight="bold", pad=4, fontsize=9)
# the list, two columns under the map
lines = [f"{k + 1:>2}  {top_countries(t)}  ({sizes[t]:,} places, {info[t][2]:,.0f} km)"
         for k, t in enumerate(compact)]
ncol = 3
per = -(-len(lines) // ncol)
for ci in range(ncol):
    chunk = lines[ci * per:(ci + 1) * per]
    axa.text(0.0 + ci * 0.34, -0.02, "\n".join(chunk), transform=axa.transAxes,
             fontsize=5.6, va="top", ha="left", color=style.INK, family="monospace")

# ------------------------------------------------------ (b) the full inventory
axb = fig.add_subplot(gs[1])
order = sorted(info, key=lambda t: info[t][2])
xs = np.array([info[t][2] for t in order])
ys = np.arange(len(order))
cols = ["#0072B2" if x < COMPACT_KM else ("#8C8C88" if x < 3000 else "#D55E00") for x in xs]
axb.scatter(xs, ys, s=12, c=cols, linewidths=0, zorder=3)
axb.axvline(COMPACT_KM, color=style.INK_MUTED, lw=0.7, ls=(0, (3, 2)))
axb.set_xscale("log")
axb.set_xlim(60, 12000)
axb.set_yticks([])
axb.set_xlabel("average distance of a type's members to the type's geographic centre (km, log)",
               fontsize=7.4)
axb.set_title(f"b   All 64 types ranked by geographic spread: {len(compact)} are places, "
              f"{sum(1 for x in xs if COMPACT_KM <= x < 3000)} are regional, "
              f"{sum(1 for x in xs if x >= 3000)} span continents",
              loc="left", fontweight="bold", pad=4, fontsize=9)
axb.grid(axis="y", visible=False)
axb.text(COMPACT_KM * 1.08, 2, f"{COMPACT_KM:,} km", ha="left", va="center",
         fontsize=6.4, color=style.INK_MUTED)
axb.set_ylim(-2, len(order) + 2)

# ------------------------------------------------------ (c) types that span continents
axc = fig.add_subplot(gs[2])
basemap(axc)
dcols = ["#009E73", "#CC79A7", "#D55E00"]
for t, c in zip(dispersed, dcols):
    m = lab == t
    axc.scatter(lng[m], lat[m], s=2.4, c=c, linewidths=0, rasterized=True,
                label=f"{top_countries(t, 3)} and others  ({sizes[t]:,} places, "
                      f"spread {info[t][2]:,.0f} km)")
axc.set_title("c   The three most dispersed types recur on every inhabited continent",
              loc="left", fontweight="bold", pad=4, fontsize=9)
axc.legend(loc="lower left", fontsize=6.4, markerscale=4, handletextpad=0.4,
           borderaxespad=0.1, labelcolor=style.INK)

fig.text(0.04, 0.005,
         "Every map shows all 40,000 places (grey dots trace the continents). Coloured dots: all "
         "members of the numbered type at their true locations. Types are k-means groups of "
         "similarly-rendered places, computed once for the corpus.",
         fontsize=6.4, color=style.INK_MUTED)
style.save(fig, "fig03_maps")
print(f"compact kinds: {len(compact)}; dispersed shown: {dispersed}")
