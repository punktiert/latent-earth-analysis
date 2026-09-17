"""Figure 6 - The atlas: the corpus as a museum object.

Main panel: the full 400 x 400 wall (160,000 renderings sorted by
similarity), from the committed 16-px-per-tile preview. Right column: three
detail crops (40 x 40 tiles each) at the preview's native resolution, showing
the local texture the sort produces. The layout statistic in the caption
(Moran's I 0.909 vs shuffled -0.005) is the manipulation check; the wall
contributes no other statistic to the paper.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import style

style.apply_style()

REPO = Path(__file__).resolve().parents[2]
WALL = REPO / "layout" / "previews" / "preview_default_wall_16px.png"

img = mpimg.imread(str(WALL))
H, W = img.shape[0], img.shape[1]
px_per_tile = W // 400 or 1

# three 40x40-tile crops: upper-left region, center, lower-right region
CROPS = [(60, 60), (180, 180), (300, 260)]           # (tile_row, tile_col)
SIZE = 40                                             # tiles per crop side

fig = plt.figure(figsize=(7.2, 5.4))
gs = fig.add_gridspec(3, 2, width_ratios=[2.9, 1.0], wspace=0.04, hspace=0.22)

axm = fig.add_subplot(gs[:, 0])
axm.imshow(img)
axm.set_xticks([])
axm.set_yticks([])
for s in axm.spines.values():
    s.set_visible(False)
axm.set_title("a   The wall: 160,000 renderings, arranged only by similarity",
              loc="left", fontweight="bold", pad=6, fontsize=9)

for i, (tr, tc) in enumerate(CROPS):
    ax = fig.add_subplot(gs[i, 1])
    y0, x0 = tr * px_per_tile, tc * px_per_tile
    crop = img[y0:y0 + SIZE * px_per_tile, x0:x0 + SIZE * px_per_tile]
    ax.imshow(crop)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_color(style.INK_MUTED)
        s.set_linewidth(0.6)
    ax.set_title(f"b{i + 1}   detail, 40 x 40 tiles", loc="left", fontsize=7,
                 fontweight="bold", pad=2)
    # locator rectangle on the main panel
    from matplotlib.patches import Rectangle
    axm.add_patch(Rectangle((x0, y0), SIZE * px_per_tile, SIZE * px_per_tile,
                            fill=False, ec="#D55E00", lw=1.0))

fig.text(0.01, 0.005,
         "Preview at 16 px per tile; the physical wall prints each tile at 768 px "
         "on a 10 x 10 m surface. Layout manipulation check: Moran's I 0.909 vs "
         "shuffled -0.005.",
         fontsize=6.8, color=style.INK_MUTED)
style.save(fig, "fig06_atlas")
