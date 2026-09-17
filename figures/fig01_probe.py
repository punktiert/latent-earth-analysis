"""Figure 1 - The probe, on five real places.

Top: the three steps. Middle: five places, one row each. Column 1 is the
model's answer to the bare name; the features a vision model reads from it
are listed at the left; columns 2-5 are four renderings of that one
description under different seeds; column 6 is the fifth probe, which has
its own first stage and its own description. The rows show what the numbers
in Fig. 4 measure: four seeds of one description are one image, the
independent probe is another building of the same type, and for one name in
nineteen the bare name yields no architecture at all (Pest, Budapest: the
English word). Bottom: the five recordings.

Images come from the archive when a data root is mounted, else from the
public release (style.release_image).
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

import style

style.apply_style()

PLACES = ["Ware, Hertfordshire, United Kingdom", "Ziniaré, Plateau-Central, Burkina Faso",
          "Kitatajima, Saitama, Japan", "Villa Corzo, Chiapas, Mexico", "Pest, Budapest, Hungary"]
rep = pl.read_parquet(style.RESULTS / "representation_per_place.parquet").select(["place_id", "name"])
hf = style.OUT.parent / "data" / "hf" / "data" / "captions.parquet"
caps = pl.read_parquet(hf).filter(pl.col("caption_slot") == "sample_0") if hf.exists() else None


def features(pid: str, n: int = 5) -> str:
    if caps is None:
        return ""
    f = caps.filter(pl.col("place_id") == pid)["features"]
    return ", ".join(list(f[0])[:n]) if f.len() else ""


fig, ax = plt.subplots(figsize=(7.2, 8.3))
ax.set_xlim(0, 100); ax.set_ylim(0, 115); ax.axis("off")


def box(x, y, w, h, title, body, fs=5.9):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.5", fc="#FFFFFF", ec=style.INK_MUTED, lw=0.7))
    ax.text(x + w / 2, y + h - 1.5, title, ha="center", va="top", fontsize=7.2, fontweight="bold", color=style.INK)
    ax.text(x + w / 2, y + h - 5.2, body, ha="center", va="top", fontsize=fs, color=style.INK, linespacing=1.25)


def arrow(x0, y0, x1, y1, label=None):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=9, lw=0.9, color=style.INK))
    if label:
        ax.text((x0 + x1) / 2, (y0 + y1) / 2 + 1.4, label, ha="center", fontsize=6, color=style.INK_MUTED)


def wrap(text: str, width: int) -> str:
    lines, cur = [], ""
    for w in text.split(" "):
        if len(cur) + len(w) + 1 > width:
            lines.append(cur); cur = w
        else:
            cur = (cur + " " + w).strip()
    return "\n".join(lines + [cur])


# ---- the probe (top band) ---------------------------------------------------
TOP, H = 99, 13.5
box(1, TOP, 27, H, "Step 1 - the bare name",
    "The generator receives a place name\nand nothing else, e.g. \"Ware,\nHertfordshire, United Kingdom\"")
arrow(29.2, TOP + 7, 34.8, TOP + 7, "image")
box(36, TOP, 28, H, "Step 2 - what did it build?",
    "A vision model lists the built features it\nsees: form, material, element, street.\n"
    "Style labels and place names are removed;\npeople, dress and food are not read.")
arrow(65.2, TOP + 7, 70.8, TOP + 7, "features")
box(72, TOP, 27, H, "Step 3 - amplified back",
    "The features return as the prompt, name\nlast, four times with different seeds.\n"
    "A fifth probe repeats all three steps\nfrom \"Architecture in <name>\".")

# ---- five places ------------------------------------------------------------
X0, CELL, GAP, ROW = 27.5, 11.2, 0.7, 13.4
Y0 = 80
heads = [(X0 + CELL / 2, "bare name"), (X0 + (CELL + GAP) * 1 + (4 * CELL + 3 * GAP) / 2, "one description, four seeds"),
         (X0 + (CELL + GAP) * 5 + CELL / 2 + 1.2, "fifth probe")]
for x, t in heads:
    ax.text(x, Y0 + CELL + 1.2, t, ha="center", va="bottom", fontsize=6.4, fontweight="bold", color=style.INK)
for r, name in enumerate(PLACES):
    pid = rep.filter(pl.col("name") == name)["place_id"][0]
    y = Y0 - r * ROW
    head, _, tail = name.partition(", ")
    ax.text(X0 - 1.6, y + CELL - 0.6, head, ha="right", va="top", fontsize=6.8, fontweight="bold", color=style.INK)
    ax.text(X0 - 1.6, y + CELL - 3.0, tail, ha="right", va="top", fontsize=5.6, color=style.INK_MUTED)
    ax.text(X0 - 1.6, y + CELL - 5.6, wrap(features(pid), 34), ha="right", va="top", fontsize=5.3, color=style.INK,
            style="italic", linespacing=1.2)
    cells = [(1, 0)] + [(2, i) for i in range(4)] + [(2, 4)]
    for c, (step, slot) in enumerate(cells):
        x = X0 + c * (CELL + GAP) + (1.2 if c == 5 else 0) + (0.0 if c == 0 else 0.0)
        im = style.release_image(pid, slot, step)
        if im is not None:
            ax.imshow(np.asarray(im.resize((384, 384))), extent=(x, x + CELL, y, y + CELL), aspect="auto", zorder=5)
        else:
            ax.add_patch(FancyBboxPatch((x, y), CELL, CELL, boxstyle="round,pad=0.1", fc="#E9E9E6", ec=style.INK_MUTED, lw=0.4))
ax.plot([X0 + CELL + GAP / 2] * 2, [Y0 - 4 * ROW, Y0 + CELL], color=style.GRID, lw=0.8)

# ---- capture band -----------------------------------------------------------
BT = 22
ax.add_patch(FancyBboxPatch((1, 1), 98, BT - 1, boxstyle="round,pad=0.5", fc="#F4F4F1", ec=style.INK_MUTED, lw=0.7))
ax.text(50, BT - 1.2, "Recorded during every generation: five descriptions of the same rendering  (5 x 40,000 places = 200,000 images)",
        ha="center", va="top", fontsize=6.8, fontweight="bold", color=style.INK)
CHANNELS = [
    ("Generator's own state", "activations of the image\nmodel while drawing", style.COLORS["flux_residual"]),
    ("Prompt encoding", "how the model reads\nthe request text", style.COLORS["t5_prompt"]),
    ("Image appearance", "shapes and textures\n(DINOv2)", style.COLORS["dino"]),
    ("Image content", "what the picture shows\n(SigLIP 2)", style.COLORS["siglip"]),
    ("Captioner's view", "what the describing\nmodel saw (Qwen2.5-VL)", style.COLORS["vlm_image"]),
]
for i, (t, b, c) in enumerate(CHANNELS):
    x = 3 + i * 19.3
    ax.add_patch(FancyBboxPatch((x, 3), 17.4, 12.5, boxstyle="round,pad=0.4", fc="#FFFFFF", ec=c, lw=1.4))
    ax.text(x + 8.7, 13.8, t, ha="center", va="top", fontsize=6.3, fontweight="bold", color=c)
    ax.text(x + 8.7, 9.6, b, ha="center", va="top", fontsize=5.7, color=style.INK)
arrow(60, Y0 - 4 * ROW - 0.8, 60, BT + 0.8)

style.save(fig, "fig01_probe")
