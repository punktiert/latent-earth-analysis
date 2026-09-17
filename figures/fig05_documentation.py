"""Figure 5 - Documentation predicts nothing; size matters only through metropolis names.

Four registered outcomes against three predictors on one y-axis per panel:
online documentation for all places (the hypothesised driver, flat);
population for towns and cities (flat or falling); parent-city population
for metropolitan districts. The registered population association is the
offset between the last two series, not a gradient within either.

Reads: poverty_per_place.parquet, poverty.json, representation.json,
       representation_per_place.parquet (record type).
"""

from __future__ import annotations

import numpy as np
import polars as pl
import matplotlib.pyplot as plt

import style

style.apply_style()

d = pl.read_parquet(style.RESULTS / "poverty_per_place.parquet").join(
    pl.read_parquet(style.RESULTS / "representation_per_place.parquet").select(["place_id", "record"]), on="place_id")
pv = style.load_json("poverty.json")
byrec = style.load_json("representation.json")["population_by_record"]
town = (d["record"] == "town or city").to_numpy()

OUTCOMES = [
    ("local_density", "distance to its most-similar\nother places",
     "a   Does it stand apart\nfrom its neighbours?"),
    ("centroid_dist", "distance to its region's\naverage rendering",
     "b   Does it stand apart\nwithin its region?"),
    ("megacluster_log2size", "size of its group of\nlook-alike places (log2)",
     "c   Is it in a mass-\nproduced type?"),
    ("genericity", "how generic its described\nfeatures are (s.d. units)",
     "d   Is it described in\nstock phrases?"),
]
PRED = [("visibility", "wiki_visibility_score", "online documentation, all places", "#0072B2", "-", None),
        ("pop_town", "population", "population, towns and cities", "#E69F00", "--", True),
        ("pop_dist", "population", "parent-city population, metropolitan districts", "#8C5A00", ":", False)]

rng = np.random.default_rng(0)
fig, axes = plt.subplots(1, 4, figsize=(7.2, 4.3), gridspec_kw={"wspace": 0.6})


def deciles(x):
    ok = np.isfinite(x)
    e = np.quantile(x[ok], np.linspace(0, 1, 11))
    return ok, np.clip(np.digitize(x, e[1:-1]), 0, 9)


series = {}
for key, col, _lab, _c, _ls, rec in PRED:
    x = d[col].cast(pl.Float64).fill_null(np.nan).to_numpy()
    if col == "population":
        x = np.log10(np.maximum(x, 1))
    if rec is not None:
        x = np.where(town == rec, x, np.nan)          # deciles within the record type
    series[key] = deciles(x)

for ax, (col, ylab, title) in zip(axes, OUTCOMES):
    y = d[col].cast(pl.Float64).fill_null(np.nan).to_numpy()
    allv = []
    for key, _col, lab, c, ls, _rec in PRED:
        ok, dec = series[key]
        xs, means, los, his = [], [], [], []
        for b in range(10):
            m = ok & (dec == b) & np.isfinite(y)
            vals = y[m]
            boot = rng.choice(vals, size=(400, min(vals.size, 4000)), replace=True).mean(axis=1)
            xs.append(b + 1); means.append(vals.mean())
            los.append(np.percentile(boot, 2.5)); his.append(np.percentile(boot, 97.5))
        ax.fill_between(xs, los, his, color=c, alpha=0.15, lw=0)
        ax.plot(xs, means, ls, color=c, lw=1.5, marker="o", ms=3, mec="white", mew=0.5, label=lab)
        allv += los + his
    lo, hi = min(allv), max(allv)
    pad = (hi - lo) * 0.25
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xticks([1, 5, 10])
    ax.set_xlabel("decile of the predictor\n(1 = lowest, 10 = highest)", fontsize=7)
    ax.set_ylabel(ylab, fontsize=7.2)
    ax.set_title(title, loc="left", fontweight="bold", fontsize=7.6, pad=6)
    # within-country slopes: the two registered ones (all places), and population among towns only
    notes = [("documentation", pv["regressions"]["visibility"][col]["within_country"], "#0072B2"),
             ("population, all (registered)", pv["regressions"]["log_population"][col]["within_country"], style.INK),
             ("population, towns and cities", byrec["town or city"]["log_population"].get(col), "#E69F00")]
    for k, (short, w, c) in enumerate(n for n in notes if n[1]):
        ci = w["ci95"]
        excl = ci[0] > 0 or ci[1] < 0
        ax.text(0.5, -0.50 - 0.215 * k, f"{short}:\n{w['slope']:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}]",
                transform=ax.transAxes, ha="center", va="top", fontsize=5.6,
                color=c if excl else style.INK_MUTED, linespacing=1.15)

h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="upper left", bbox_to_anchor=(0.06, 0.92), ncol=3, fontsize=6.4,
           handlelength=2.2, frameon=False, columnspacing=1.4)
fig.suptitle("Documentation does not shape how distinctively a place is rendered; size does only through metropolis names",
             x=0.01, ha="left", fontsize=8.8, fontweight="bold")
fig.subplots_adjust(top=0.76, bottom=0.47, left=0.08, right=0.99)
style.save(fig, "fig05_documentation")
