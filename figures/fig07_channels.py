"""Figure 7 - Where does knowledge of place live?

One panel per location: the prompt text (the encoder that read the name), the
rendered image, and the generator's internal state while drawing. Each shows
how often the country / region / climate of a place can be identified from
that channel alone. Reads geo_knn.json.
"""

from __future__ import annotations

import matplotlib.pyplot as plt

import style

style.apply_style()

knn = style.load_json("geo_knn.json")

GROUPS = [
    ("In the prompt text", "the encoder that read the place name", "t5_prompt",
     style.COLORS["t5_prompt"]),
    ("In the rendered image", "independent embedding of the picture (SigLIP-2)",
     "siglip", style.COLORS["siglip"]),
    ("In the generator's own state", "its internal activations while drawing",
     "flux_residual", style.COLORS["flux_residual"]),
]
TARGETS = [("country", "country\n(207 options)"),
           ("un_region", "world region\n(22 options)"),
           ("koppen_class", "climate zone\n(17 options)")]

fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.9), sharey=True,
                         gridspec_kw={"wspace": 0.12})
for ax, (title, sub, ch, color) in zip(axes, GROUPS):
    accs, chances = [], []
    for t, _lab in TARGETS:
        r = knn["targets"][t]["place_level"][ch]
        accs.append(r["accuracy"])
        chances.append(r["majority_baseline"])
    xs = range(len(TARGETS))
    ax.bar(xs, accs, width=0.62, color=color, edgecolor=style.SURFACE)
    for x, (a, c) in enumerate(zip(accs, chances)):
        ax.plot([x - 0.31, x + 0.31], [c, c], color=style.INK, lw=1.0,
                ls=(0, (3, 2)))
        ax.text(x, a + 0.025, f"{a:.0%}", ha="center", fontsize=8,
                fontweight="bold", color=style.INK)
    ax.set_xticks(list(xs))
    ax.set_xticklabels([lab for _t, lab in TARGETS], fontsize=6.8)
    ax.set_ylim(0, 1.0)
    ax.set_title(f"{title}\n", loc="left", fontweight="bold", fontsize=8.4, pad=2)
    ax.text(0, 1.005, sub, transform=ax.transAxes, fontsize=6.4,
            color=style.INK_MUTED, va="bottom")
    ax.grid(axis="x", visible=False)
axes[0].set_ylabel("share of places identified")
fig.suptitle("Knowledge of place is present at every station of the pipeline "
             "(dashes: always guessing the most common label)",
             x=0.01, ha="left", fontsize=9.5, fontweight="bold")
fig.subplots_adjust(top=0.74)
style.save(fig, "fig07_channels")
