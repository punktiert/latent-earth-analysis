"""Figure 4 - How well a place is represented.

(a) Distance between single renderings in image space: four seeds of one
    description, the fifth independent probe of the same name, and the 20
    nearest other places. One description is one image; one name is not.
(b) Where the fifth probe lands: what its nearest other place shares with
    it, and how often its own place is retrieved among 30,366.
(c) Which architecture recurs: share of places in which a feature named in
    the first description returns in the independent second one, against
    another place of the same country, a place of the same world region in
    another country, and any place.
(d) Size: retrieval of the own place and the share of names that yield no
    architecture, by population, for towns and cities; metropolitan
    districts alongside.

Reads: representation.json, representation_per_place.parquet.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import matplotlib.pyplot as plt

import style

style.apply_style()
R = style.load_json("representation.json")
per = pl.read_parquet(style.RESULTS / "representation_per_place.parquet")
SEED_C, INDEP_C, OTHER_C = "#3D5A6E", "#D55E00", "#9A9A96"

fig = plt.figure(figsize=(7.2, 7.4))
gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.25], hspace=0.42, wspace=0.34,
                      left=0.085, right=0.975, top=0.95, bottom=0.085)

# ------------------------------------------------------------------ (a) distances
ax = fig.add_subplot(gs[0, 0])
bins = np.linspace(0, 1.2, 61)
for col, c, lab in (("image_d_seed", SEED_C, "four seeds of one description"),
                    ("image_d_nn20", OTHER_C, "to the 20 nearest other places"),
                    ("image_d_indep", INDEP_C, "fifth, independent probe")):
    x = per[col].drop_nulls().drop_nans().to_numpy()
    ax.hist(x, bins=bins, color=c, alpha=0.85, edgecolor=style.SURFACE, linewidth=0.2, label=f"{lab} (median {np.median(x):.2f})")
ax.set_xlabel("distance between two renderings of one place (image space)")
ax.set_ylabel("places")
ax.set_xlim(0, 1.2)
ax.legend(loc="upper right", fontsize=5.9, handlelength=1.2)
ax.set_title("a   One description is one image; one name is not", loc="left", fontweight="bold", fontsize=8.4, pad=5)

# ------------------------------------------------------------ (b) where the probe lands
ax = fig.add_subplot(gs[0, 1])
sh, ret, ctl = R["image"]["nearest_other_place_shares"], R["image"]["indep"], R["image"]["seed"]
rows = [("world region", sh["world_region"]), ("country", sh["country"]), ("climate zone", sh["climate"]), ("type (of 64)", sh["kind"]),
        None, ("within top 100", ret["top100"]), ("within top 10", ret["top10"]), ("first", ret["top1"])]
y = 0
ticks, labels = [], []
for r in rows:
    if r is None:
        y -= 0.7; continue
    ax.barh(y, r[1] * 100, color=INDEP_C, height=0.62)
    ax.text(r[1] * 100 + 1.5, y, f"{r[1] * 100:.0f}%" if r[1] > 0.05 else f"{r[1] * 100:.1f}%", va="center", fontsize=6.4, color=style.INK)
    ticks.append(y); labels.append(r[0]); y -= 1
ax.plot([ctl["top1"] * 100] * 2, [ticks[-1] - 0.42, ticks[-1] + 0.42], color=SEED_C, lw=2)
ax.text(ctl["top1"] * 100 - 1.5, ticks[-1], f"a fourth seed of the\nsame description: {ctl['top1'] * 100:.0f}%", va="center", ha="right", fontsize=5.8, color=SEED_C)
ax.set_yticks(ticks); ax.set_yticklabels(labels, fontsize=6.6)
ax.set_xlim(0, 100); ax.set_xlabel("% of 30,366 places")
ax.set_ylim(ticks[-1] - 0.75, ticks[0] + 1.25)
ax.text(0.8, ticks[0] + 0.72, "its nearest other place shares its", fontsize=6.2, color=style.INK_MUTED, ha="left", va="center")
ax.text(0.8, ticks[4] + 0.72, "its own place is retrieved", fontsize=6.2, color=style.INK_MUTED, ha="left", va="center")
ax.grid(axis="y", visible=False)
ax.set_title("b   Where the fifth probe lands", loc="left", fontweight="bold", fontsize=8.4, pad=5)

# ------------------------------------------------------------ (c) which architecture recurs
ax = fig.add_subplot(gs[1, 0])
T = R["feature_recurrence"]["four_controls"]["terms"]
order = [("material", ["stone", "brick", "glass", "render", "timber", "concrete", "earth"]),
         ("roof", ["tiled roof", "pitched roof", "flat roof", "metal roof"]),
         ("element", ["arch", "column", "balcony", "chimney", "storefront"]),
         ("setting", ["palm", "arid", "mountain", "water"])]
y, ticks, labels = 0, [], []
for cat, terms in order:
    ax.text(-0.31, y + 0.75, cat, transform=ax.get_yaxis_transform(), fontsize=6.2, color=style.INK_MUTED, fontweight="bold", ha="left")
    for t in terms:
        v = T[t]
        ax.plot([v["any_place"] * 100, v["same_place"] * 100], [y, y], color=style.GRID, lw=1.6, zorder=1)
        ax.scatter(v["any_place"] * 100, y, s=11, color="#D4D4CF", zorder=3)
        ax.scatter(v["same_region_other_country"] * 100, y, s=11, color="#A9A9A4", zorder=3)
        ax.scatter(v["same_country"] * 100, y, s=12, color="#6E6E6E", zorder=3)
        ax.scatter(v["same_place"] * 100, y, s=16, color=INDEP_C, zorder=4)
        ticks.append(y); labels.append(t); y -= 1
    y -= 0.9
ax.set_yticks(ticks); ax.set_yticklabels(labels, fontsize=6.2)
ax.set_xlim(0, 80); ax.set_xlabel("% of places where the feature returns in the second description")
ax.grid(axis="y", visible=False)
for lab, c in (("any place", "#D4D4CF"), ("same region, other country", "#A9A9A4"), ("same country", "#6E6E6E"), ("same place", INDEP_C)):
    ax.scatter([], [], s=14, color=c, label=lab)
m = R["feature_recurrence"]["four_controls"]["mean_share_recurring"]
leg = ax.legend(loc="lower right", fontsize=6, handletextpad=0.2, borderpad=0.4, title_fontsize=5.8,
                title=(f"all features: {m['any_place']:.0%}, {m['same_region_other_country']:.0%}, "
                       f"{m['same_country']:.0%}, {m['same_place']:.0%}"))
ax.set_title("c   What two probes of one name share", loc="left", fontweight="bold", fontsize=8.4, pad=5)
# ------------------------------------------------------------------ (d) size
ax = fig.add_subplot(gs[1, 1])
rows = R["towns_by_population"]
x = np.arange(len(rows))
for key, ci, c, lab in (("image_indep_top10", "image_indep_top10_ci95", INDEP_C, "own place within top 10 of the fifth probe"),
                        ("no_architecture", "no_architecture_ci95", "#0072B2", "bare name yields no architecture")):
    v = np.array([r[key] for r in rows]) * 100
    lo = v - np.array([r[ci][0] for r in rows]) * 100
    hi = np.array([r[ci][1] for r in rows]) * 100 - v
    ax.errorbar(x, v, yerr=[lo, hi], color=c, marker="o", ms=3.2, lw=1.1, capsize=1.8, elinewidth=0.7, label=lab)
d = next(r for r in R["by_record"] if r["record"] == "metropolitan district")
xd = len(rows) + 0.6
ax.scatter([xd], [d["image_indep_top10"] * 100], color=INDEP_C, marker="s", s=16, zorder=4)
ax.scatter([xd], [d["no_architecture"] * 100], color="#0072B2", marker="s", s=16, zorder=4)
ax.axvline(len(rows) - 0.2, color=style.GRID, lw=0.8)
bp = R["population_threshold"]
edges = np.log10([10e3, 15e3, 20e3, 30e3, 50e3, 100e3, 250e3, 500e3, 1e6, 2e6, 40e6])
pos = lambda p: float(np.interp(np.log10(p), edges, np.arange(len(edges)) - 0.5))  # noqa: E731
ax.axvspan(pos(bp["breakpoint_ci95"][0]), pos(bp["breakpoint_ci95"][1]), color="#F1E4D8", zorder=0, lw=0)
ax.text(pos(bp["breakpoint_population"]), 17.4, f"breakpoint\n{bp['breakpoint_population'] / 1e3:.0f}k", ha="center", va="top", fontsize=5.8, color=style.INK_MUTED)
ax.set_xticks(list(x) + [xd]); ax.set_xticklabels([r["pop_bin"] for r in rows] + ["metro.\ndistricts"], fontsize=5.6, rotation=55, ha="right")
ax.set_ylim(0, 22); ax.set_ylabel("% of places")
ax.set_xlabel("inhabitants (towns and cities)")
ax.legend(loc="upper left", fontsize=5.9, handlelength=1.4)
ax.set_title("d   Size matters only above a threshold", loc="left", fontweight="bold", fontsize=8.4, pad=5)

style.save(fig, "fig04_representation")
