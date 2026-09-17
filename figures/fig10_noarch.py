"""Extended Data Fig. 5 - What a bare name yields when it yields no architecture.

The first 48 of the 203 inspected first-stage images in which the vision
model found no architecture, in identifier order (no selection), each with
the name that produced it. Reads noarch_inspection.csv; images from the
archive or the public release (style.release_image).
"""

from __future__ import annotations

import polars as pl
from PIL import Image, ImageDraw, ImageFont

import style

COLS, ROWS, CELL, LAB = 8, 6, 220, 30
rows = pl.read_csv(style.RESULTS / "noarch_inspection.csv").head(COLS * ROWS)
try:
    font = ImageFont.truetype("arial.ttf", 12)
except OSError:
    font = ImageFont.load_default()

sheet = Image.new("RGB", (COLS * CELL, ROWS * (CELL + LAB)), "#FCFCFB")
draw = ImageDraw.Draw(sheet)
for i, r in enumerate(rows.iter_rows(named=True)):
    x, y = (i % COLS) * CELL, (i // COLS) * (CELL + LAB)
    im = style.release_image(r["place_id"], 0, step=1)
    if im is not None:
        sheet.paste(im.resize((CELL - 4, CELL - 4), Image.LANCZOS), (x + 2, y + 2))
    name = r["name"]
    draw.text((x + 3, y + CELL), name if len(name) <= 36 else name[:35] + "…", fill="#1A1A1A", font=font)
    draw.text((x + 3, y + CELL + 14), r["literal_reading"] or "", fill="#6E6E6E", font=font)
out = style.OUT / "fig10_noarch.png"
sheet.save(out)
print(f"wrote {out}: {rows.height} images")
