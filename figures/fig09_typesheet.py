"""Extended Data Fig. 4 - The type inventory: all 64 types, exhaustively.

One row per type, six renderings each, drawn at random (seeded) from the
type's members whose four prompts are identical, so every picture is the
direct rendering of the description the name is taken from. A type's name
lists the three features that are common in it and rare outside it, each
with the share of the type's places that carry it (scripts/atlas_findings.py
kind_names); the earlier lift-only names picked rare phrases and did not
match the pictures. Rows are ordered by compactness in the atlas:
territories first, the scattered types last; two columns of 32, read
downwards.

The plate is laid out in physical units on a 180 x 232 mm page (Nature's
full-page figure) and rasterised at FIG_DPI (300 by default, 600 for the
submission files); type is 5-6 pt at that size. Images come from the archive
when a data root is mounted, else from the public release
(style.release_image).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import polars as pl
from PIL import Image, ImageDraw, ImageFont

import style

DPI = int(os.environ.get("FIG_DPI", 300))
OUT = Path(os.environ.get("FIG_OUT", style.OUT))
px = lambda mm: round(mm / 25.4 * DPI)   # noqa: E731
pt = lambda p: round(p / 72 * DPI)       # noqa: E731

PER_ROW, COLS, ROWS = 6, 2, 32
PAGE_W, GUTTER = px(180), px(3.4)
CELL, GAP = px(7.0), px(0.25)
COL_W = (PAGE_W - GUTTER) // COLS
LABEL_W = COL_W - PER_ROW * (CELL + GAP)
ROW_H = CELL + GAP

kinds = pl.read_parquet(style.RESULTS / "atlas_kinds.parquet").sort("compactness_ratio")
assert kinds.height == COLS * ROWS, kinds.height
same = pl.read_parquet(style.RESULTS / "prompt_subsets.parquet").filter(pl.col("identical_prompts")).select("place_id")
places = pl.read_parquet(style.RESULTS / "poverty_per_place.parquet").select(
    ["place_id", "kmeans_label", "iso2"]).join(same, on="place_id", how="inner")

try:
    font = ImageFont.truetype("arial.ttf", pt(6))
    small = ImageFont.truetype("arial.ttf", pt(5))
except OSError:
    font = small = ImageFont.load_default()


def wrap_px(text: str, fnt, max_w: int, max_lines: int) -> list[str]:
    """Greedy word wrap by measured width; the last permitted line is cut with an ellipsis."""
    lines, cur = [], ""
    for w in text.split():
        cand = f"{cur} {w}".strip()
        if fnt.getlength(cand) <= max_w or not cur:
            cur = cand
        else:
            lines.append(cur); cur = w
    lines.append(cur)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        while fnt.getlength(lines[-1] + "…") > max_w and " " in lines[-1]:
            lines[-1] = lines[-1].rsplit(" ", 1)[0]
        lines[-1] += "…"
    return lines


rng = np.random.default_rng(0)
sheet = Image.new("RGB", (PAGE_W, ROWS * ROW_H), "#FCFCFB")
draw = ImageDraw.Draw(sheet)
LH = pt(6.5)                                     # name line height
NUM_W, PAD = px(2.6), px(0.4)
n_ok, n_cut = 0, 0
for r_i, r in enumerate(kinds.iter_rows(named=True)):
    col_x = (r_i // ROWS) * (COL_W + GUTTER)
    y = (r_i % ROWS) * ROW_H
    members = places.filter(pl.col("kmeans_label") == r["kind"])
    idx = rng.choice(members.height, size=min(PER_ROW, members.height), replace=False)
    x = col_x + LABEL_W
    for i in idx.tolist():
        row = members.row(i, named=True)
        img = style.release_image(row["place_id"], 0, 2)
        if img is None:
            continue
        sheet.paste(img.resize((CELL, CELL), Image.LANCZOS), (x, y))
        draw.text((x + px(0.3), y + CELL - pt(5) - px(0.4)), str(row["iso2"] or "")[:3], fill="white", font=small,
                  stroke_width=max(1, pt(0.6)), stroke_fill="#1A1A1A")
        x += CELL + GAP
        n_ok += 1
    name = wrap_px(r["name"], font, LABEL_W - NUM_W - px(0.8), 2)
    n_cut += name[-1].endswith("…")
    draw.text((col_x + px(0.3), y + PAD), f"{r_i + 1}", fill="#1A1A1A", font=font)
    for li, line in enumerate(name):
        draw.text((col_x + NUM_W, y + PAD + li * LH), line, fill="#1A1A1A", font=font)
    draw.text((col_x + NUM_W, y + PAD + 2 * LH + px(0.2)),
              f"{r['n_places']:,} places · spread {r['geo_km']:,.0f} km · atlas {r['compactness_ratio']:.2f}",
              fill="#6E6E6E", font=small)

OUT.mkdir(parents=True, exist_ok=True)
out = OUT / "fig09_typesheet.png"
sheet.save(out, dpi=(DPI, DPI))
print(f"wrote {out}: {kinds.height} types x up to {PER_ROW} renderings ({n_ok} images), "
      f"{sheet.size[0]}x{sheet.size[1]} px at {DPI} dpi = {sheet.size[0] / DPI * 25.4:.0f}x{sheet.size[1] / DPI * 25.4:.0f} mm, "
      f"{n_cut} names shortened")
