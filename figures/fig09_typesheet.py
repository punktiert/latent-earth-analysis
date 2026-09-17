"""Extended Data Fig. 4 - The type inventory: all 64 types, exhaustively.

One row per type, six renderings each, drawn at random (seeded) from the
type's members whose four prompts are identical, so every picture is the
direct rendering of the description the name is taken from. A type's name
lists the three features that are common in it and rare outside it, each
with the share of the type's places that carry it (scripts/atlas_findings.py
kind_names); the earlier lift-only names picked rare phrases and did not
match the pictures. Rows are ordered by compactness in the atlas:
territories first, the scattered types last.

Images come from the archive when a data root is mounted, else from the
public release (style.release_image).
"""

from __future__ import annotations

import textwrap

import numpy as np
import polars as pl
from PIL import Image, ImageDraw, ImageFont

import style

PER_ROW, CELL, LABEL_W, GAP = 6, 110, 360, 3
kinds = pl.read_parquet(style.RESULTS / "atlas_kinds.parquet").sort("compactness_ratio")
same = pl.read_parquet(style.RESULTS / "prompt_subsets.parquet").filter(pl.col("identical_prompts")).select("place_id")
places = pl.read_parquet(style.RESULTS / "poverty_per_place.parquet").select(
    ["place_id", "kmeans_label", "iso2"]).join(same, on="place_id", how="inner")

try:
    font = ImageFont.truetype("arial.ttf", 13)
    small = ImageFont.truetype("arial.ttf", 11)
except OSError:
    font = small = ImageFont.load_default()

rng = np.random.default_rng(0)
W = LABEL_W + PER_ROW * (CELL + GAP)
H = kinds.height * (CELL + GAP)
sheet = Image.new("RGB", (W, H), "#FCFCFB")
draw = ImageDraw.Draw(sheet)
n_ok = 0
for r_i, r in enumerate(kinds.iter_rows(named=True)):
    y = r_i * (CELL + GAP)
    members = places.filter(pl.col("kmeans_label") == r["kind"])
    idx = rng.choice(members.height, size=min(PER_ROW, members.height), replace=False)
    x = LABEL_W
    for i in idx.tolist():
        row = members.row(i, named=True)
        img = style.release_image(row["place_id"], 0, 2)
        if img is None:
            continue
        sheet.paste(img.resize((CELL, CELL), Image.LANCZOS), (x, y))
        draw.text((x + 3, y + CELL - 14), str(row["iso2"] or "")[:3], fill="white", font=small,
                  stroke_width=2, stroke_fill="#1A1A1A")
        x += CELL + GAP
        n_ok += 1
    name = textwrap.wrap(r["name"], 50)[:3]
    draw.text((6, y + 10), f"{r_i + 1}", fill="#1A1A1A", font=font)
    for li, line in enumerate(name):
        draw.text((30, y + 10 + li * 17), line, fill="#1A1A1A", font=font)
    draw.text((30, y + 14 + len(name) * 17 + 6),
              f"{r['n_places']:,} places · spread {r['geo_km']:,.0f} km · atlas {r['compactness_ratio']:.2f}",
              fill="#6E6E6E", font=small)

out = style.OUT / "fig09_typesheet.png"
sheet.save(out)
print(f"wrote {out}: {kinds.height} types x up to {PER_ROW} renderings ({n_ok} images)")
