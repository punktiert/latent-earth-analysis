"""Figure 2 - Where the model's geography lives.

(a) three stacked panels, one per question (country / region / climate):
    kNN identification accuracy per channel with the majority baseline and
    the permutation-null band. Stacking replaces the rotated group labels
    that collided in the previous layout.
(b) distance decay: mean cosine similarity between place pairs vs
    great-circle distance, per channel. Legend sits in the empty upper-right
    of the panel (every curve is below 0.12 beyond 1,000 km), so no label
    touches a line.

Reads: geo_knn.json, geo_decay.json.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

import style
from style import COLORS, LABELS, MODALITIES

style.apply_style()

knn = style.load_json("geo_knn.json")
decay = style.load_json("geo_decay.json")

# geographic questions first; the political one last
TARGETS = [("koppen_class", "Which climate zone?  (17 options)"),
           ("un_region", "Which world region?  (22 options)"),
           ("country", "Which country?  (207 options)")]

fig = plt.figure(figsize=(7.2, 5.0))
gs = fig.add_gridspec(3, 2, width_ratios=[1.0, 1.05], wspace=0.34, hspace=0.62,
                      left=0.20, right=0.98, top=0.90, bottom=0.13)

# ------------------------------------------------- (a) three stacked panels
for ti, (t, title) in enumerate(TARGETS):
    ax = fig.add_subplot(gs[ti, 0])
    block = knn["targets"][t]["place_level"]
    chance = block["concat"]["majority_baseline"]
    null97 = block["concat"]["null_p975"]
    ax.axvspan(0, null97, color="#EDEDEA", zorder=0, lw=0)
    ax.axvline(chance, color=style.INK_MUTED, lw=0.8, ls=(0, (3, 2)), zorder=1)
    for mi, m in enumerate(MODALITIES):
        r = block[m]
        lo, hi = r["ci95"]
        ax.plot([lo, hi], [mi, mi], color=COLORS[m], lw=2.2, solid_capstyle="round")
        ax.plot(r["accuracy"], mi, "o", ms=4.6, color=COLORS[m], mec="white",
                mew=0.8, zorder=3)
    ax.set_yticks(range(len(MODALITIES)))
    ax.set_yticklabels([LABELS[m] for m in MODALITIES], fontsize=7)
    ax.set_ylim(len(MODALITIES) - 0.4, -0.6)
    ax.set_xlim(0, 1.0)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0", "25%", "50%", "75%", "100%"] if ti == 2 else [])
    ax.grid(axis="y", visible=False)
    ax.set_title(title, loc="left", fontsize=7.8, pad=2, color=style.INK)
    if ti == 0:
        ax.text(null97 / 2, (len(MODALITIES) - 1) / 2, "chance", fontsize=6.2,
                color=style.INK_MUTED, ha="center", va="center", rotation=90)
        ax.annotate("most common\nanswer", xy=(chance, len(MODALITIES) - 0.5),
                    xytext=(chance + 0.05, len(MODALITIES) + 0.35),
                    textcoords="data", fontsize=6.2, color=style.INK_MUTED,
                    ha="left", va="top", annotation_clip=False,
                    arrowprops=dict(arrowstyle="-", color=style.INK_MUTED, lw=0.6))
    if ti == 2:
        ax.set_xlabel("share of places identified from their 10 most similar places",
                      fontsize=7.6)
fig.text(0.02, 0.955, "a   Can a rendering's origin be identified?",
         fontsize=9.5, fontweight="bold", va="center")

# ------------------------------------------------------- (b) decay curves
axb = fig.add_subplot(gs[:, 1])
edges = np.array(decay["bin_edges_km"])
mids = np.sqrt(edges[:-1] * edges[1:])
for m in MODALITIES:
    bins = decay["curves"][m]["bins"]
    xs, ys, los, his = [], [], [], []
    for x, b in zip(mids, bins):
        if b is None:
            continue
        xs.append(x); ys.append(b["mean"]); los.append(b["ci95"][0]); his.append(b["ci95"][1])
    axb.fill_between(xs, los, his, color=COLORS[m], alpha=0.16, lw=0)
    axb.plot(xs, ys, color=COLORS[m], lw=1.7 if m == "concat" else 1.3,
             label=LABELS[m], zorder=3 if m == "concat" else 2)
axb.set_xscale("log")
axb.set_xlabel("physical distance between two places (km)", fontsize=7.6)
axb.set_ylabel("similarity of their renderings (0 = unrelated)", fontsize=7.6)
axb.set_title("b   Does nearness make renderings similar?", loc="left",
              fontweight="bold", pad=8, fontsize=9.5)
axb.legend(loc="upper right", fontsize=6.8, handlelength=1.6, labelspacing=0.35,
           borderaxespad=0.3)
floor = decay["curves"]["concat"]["similarity_floor_gt5000km"]
axb.axhline(floor, color=style.INK_MUTED, lw=0.7, ls=(0, (3, 2)))
axb.annotate(f"floor beyond 5,000 km ({floor:.3f}):\nas unrelated as any two places",
             xy=(mids[0] * 1.1, floor), xytext=(0, 5), textcoords="offset points",
             ha="left", va="bottom", fontsize=6.4, color=style.INK_MUTED)
r = decay["mantel"]["concat"]["r_mean"]
axb.text(0.98, 0.60, f"overall distance-similarity\ncorrelation r = {r:.2f} (weak)",
         transform=axb.transAxes, fontsize=7, color=style.INK, ha="right", va="top")
axb.axvspan(1, 100, color="#EDEDEA", zorder=0, lw=0)
axb.text(10, axb.get_ylim()[1] * 0.985 if axb.get_ylim()[1] > 0 else 0.34,
         "flat to ~100 km", ha="center", va="top", fontsize=6.4, color=style.INK_MUTED)

fig.text(0.02, 0.02,
         "Each dot: one way of describing the renderings (see Fig. 1), with its 95% interval. "
         "Grey band: what pure chance produces. Dashed line: always guessing the most common answer.",
         fontsize=6.4, color=style.INK_MUTED)
style.save(fig, "fig02_geography")
