"""Extended Data Fig. 5 - What a bare name yields when it yields no architecture.

The first 48 of the 203 inspected first-stage images in which the vision
model found no architecture, in identifier order (no selection), each with
the name that produced it. Reads noarch_inspection.csv; images from the
archive or the public release (style.release_image).

Laid out in physical units on a 180 mm wide page and rasterised at FIG_DPI
(300 by default, 600 for the submission files); type is 5.5 pt.
"""

from __future__ import annotations

import os
from pathlib import Path

import polars as pl
from PIL import Image, ImageDraw, ImageFont

import style

DPI = int(os.environ.get("FIG_DPI", 300))
OUT = Path(os.environ.get("FIG_OUT", style.OUT))
px = lambda mm: round(mm / 25.4 * DPI)   # noqa: E731
pt = lambda p: round(p / 72 * DPI)       # noqa: E731

COLS, ROWS = 8, 6
CELL = px(180) // COLS
LH = pt(6)
LAB = px(0.5) + 3 * LH + px(0.4)          # two name lines and the literal reading
rows = pl.read_csv(style.RESULTS / "noarch_inspection.csv").head(COLS * ROWS)
try:
    font = ImageFont.truetype("arial.ttf", pt(5.5))
except OSError:
    font = ImageFont.load_default()


def wrap_px(text: str, fnt, max_w: int, max_lines: int) -> list[str]:
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


sheet = Image.new("RGB", (COLS * CELL, ROWS * (CELL + LAB)), "#FCFCFB")
draw = ImageDraw.Draw(sheet)
inset = px(0.35)
for i, r in enumerate(rows.iter_rows(named=True)):
    x, y = (i % COLS) * CELL, (i // COLS) * (CELL + LAB)
    im = style.release_image(r["place_id"], 0, step=1)
    if im is not None:
        sheet.paste(im.resize((CELL - 2 * inset, CELL - 2 * inset), Image.LANCZOS), (x + inset, y + inset))
    text_w = CELL - 2 * inset - px(0.3)
    for li, line in enumerate(wrap_px(r["name"], font, text_w, 2)):
        draw.text((x + inset, y + CELL + px(0.5) + li * LH), line, fill="#1A1A1A", font=font)
    reading = wrap_px(r["literal_reading"] or "", font, text_w, 1)[0] if r["literal_reading"] else ""
    draw.text((x + inset, y + CELL + px(0.5) + 2 * LH), reading, fill="#6E6E6E", font=font)

OUT.mkdir(parents=True, exist_ok=True)
out = OUT / "fig10_noarch.png"
sheet.save(out, dpi=(DPI, DPI))
print(f"wrote {out}: {rows.height} images, {sheet.size[0]}x{sheet.size[1]} px at {DPI} dpi = "
      f"{sheet.size[0] / DPI * 25.4:.0f}x{sheet.size[1] / DPI * 25.4:.0f} mm")
