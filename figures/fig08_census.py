"""Extended Data Fig. 1 - The census plate: how well the model locates every place.

All 40,000 places, one dot each, at true coordinates. Colour is the MEASURED
knowledge of place: the share of a place's 10 most-similar renderings that
come from the same country (0 = none, 1 = all ten). Light = the model's
rendering of this place is geographically anonymous; dark = it is firmly
located. Sequential single-hue ramp (colour carries one number, position is
plain geography).

Default: the page figure (180 mm wide, no in-figure title; the legend is in
the manuscript), written by style.save at FIG_DPI. FIG_PLATE=poster renders
the exhibition plate instead: 24 x 12 in at 600 dpi with title and caption,
as fig08_census_poster.png/.pdf.

Needs paper/results/s3_place_correct.parquet with country_purity (produced by
`lca-analysis s3-checks`); falls back to the boolean correct/incorrect
two-tone if only the older file is present.
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize

import style

style.apply_style()
POSTER = os.environ.get("FIG_PLATE") == "poster"

d = pl.read_parquet(style.RESULTS / "poverty_per_place.parquet")
s3 = pl.read_parquet(style.RESULTS / "s3_place_correct.parquet")
d = d.join(s3, on="place_id", how="inner")
lat = d["lat"].to_numpy()
lng = d["lng"].to_numpy()

fig, ax = plt.subplots(figsize=(24, 12) if POSTER else (7.2, 3.9))
DOT, FS, FS_SMALL = (2.4, 11, 10) if POSTER else (0.35, 7, 6.5)
if "country_purity" in d.columns:
    val = d["country_purity"].to_numpy()
    ok = np.isfinite(val)
    cmap = LinearSegmentedColormap.from_list(
        "knowledge", ["#E8E4DC", "#B9CFDD", "#5E93B4", "#1F5D80", "#0B2E45"])
    order = np.argsort(val[ok])                    # draw darkest last
    ax.scatter(lng[ok][order], lat[ok][order], s=DOT,
               c=val[ok][order], cmap=cmap, vmin=0, vmax=1,
               linewidths=0, rasterized=True)
    cbar = fig.colorbar(ScalarMappable(norm=Normalize(0, 1), cmap=cmap),
                        ax=ax, fraction=0.024, pad=0.01, shrink=0.55)
    cbar.set_label("knowledge of place: share of the 10 most-similar renderings "
                   "from the same country", fontsize=FS)
    cbar.ax.tick_params(labelsize=FS_SMALL)
    mode = "graded"
else:
    corr = d["country_correct"].to_numpy()
    ax.scatter(lng[~corr], lat[~corr], s=DOT * 0.85, c="#E0DCD3", linewidths=0,
               rasterized=True, label="not located")
    ax.scatter(lng[corr], lat[corr], s=DOT, c="#1F5D80", linewidths=0,
               rasterized=True, label="located (country identified)")
    ax.legend(loc="lower left", fontsize=FS, markerscale=6)
    mode = "boolean"

ax.set_xlim(-180, 180)
ax.set_ylim(-60, 78)
ax.set_aspect(1.15)
ax.grid(False)
ax.set_xticks([])
ax.set_yticks([])
for s_ in ("left", "bottom", "top", "right"):
    ax.spines[s_].set_visible(False)

if POSTER:
    ax.set_title("The census: 40,000 places, one dot each - how firmly the model's "
                 "rendering locates each place in its country",
                 loc="left", fontweight="bold", fontsize=18, pad=12)
    ax.text(0.0, -0.02,
            "Position: true coordinates. Colour: measured geographic knowledge - dark "
            "places are rendered with an identifiable national building culture; pale "
            "places are rendered geographically anonymous.",
            transform=ax.transAxes, fontsize=11, color=style.INK_MUTED, va="top")
    fig.savefig(style.OUT / "fig08_census_poster.png", dpi=600)
    fig.savefig(style.OUT / "fig08_census_poster.pdf")
    print(f"wrote fig08_census_poster ({mode} mode, 600 dpi)")
else:
    style.save(fig, "fig08_census")
    print(f"fig08_census ({mode} mode)")
