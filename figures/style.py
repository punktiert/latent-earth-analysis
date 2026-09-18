"""Shared figure style: CVD-validated palette, print-grade matplotlib defaults.

Palette validated with the dataviz six-checks validator (light surface):
lightness band PASS, chroma floor PASS, worst adjacent-pair CVD dE 17.9 PASS.
The contrast WARN on orange/pink/sky is relieved by direct labels everywhere
(no color-only identity in any figure).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib as mpl

RESULTS = Path(__file__).resolve().parent.parent / "results"
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
OUT = Path(__file__).resolve().parent

# Fixed modality order + colors — never cycled, never reassigned.
MODALITIES = ["dino", "siglip", "flux_residual", "t5_prompt", "vlm_image", "concat"]
COLORS = {
    "dino":          "#0072B2",
    "siglip":        "#E69F00",
    "flux_residual": "#009E73",
    "t5_prompt":     "#CC79A7",
    "vlm_image":     "#56B4E9",
    "concat":        "#1A1A1A",
}
# Plain-language channel names (technical name in parentheses once, in Fig 1).
LABELS = {
    "dino":          "Image appearance",
    "siglip":        "Image content",
    "flux_residual": "Generator's internal state",
    "t5_prompt":     "Prompt encoding",
    "vlm_image":     "Captioner's view",
    "concat":        "All combined",
}

INK = "#1A1A1A"
INK_MUTED = "#6E6E6E"
GRID = "#E3E3E0"
SURFACE = "#FCFCFB"


def apply_style() -> None:
    mpl.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8.5,
        "axes.titlesize": 9.5,
        "axes.labelsize": 8.5,
        "axes.edgecolor": INK_MUTED,
        "axes.linewidth": 0.6,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.5,
        "axes.axisbelow": True,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "xtick.labelcolor": INK,
        "ytick.labelcolor": INK,
        "text.color": INK,
        "axes.labelcolor": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "pdf.fonttype": 42,        # embed TrueType (journal requirement)
        "ps.fonttype": 42,
        "savefig.dpi": int(os.environ.get("FIG_DPI", 300)),   # FIG_DPI=600 for submission files
        "savefig.bbox": "tight",
    })


def load_json(name: str) -> dict:
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def save(fig, stem: str) -> None:
    import os
    out = Path(os.environ.get("FIG_OUT", OUT))   # FIG_OUT: render elsewhere for checking
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{stem}.pdf")
    fig.savefig(out / f"{stem}.png")
    print(f"wrote {out / stem}.pdf/.png")


def release_image(place_id: str, slot: int, step: int = 2):
    """A rendering as a PIL image, from the archive if a data root is mounted,
    else from the public release's tar shards (https://doi.org/10.57967/HF/10090,
    fetched on demand into data/hf). step=1: the name-only image (slot 0 or 4)."""
    import io
    import tarfile

    import polars as pl
    from PIL import Image

    from latent_earth.discover import find_data
    p = find_data(f"gen{step}/{place_id}/sample_{slot}.jpg")
    if p:
        return Image.open(p).convert("RGB")
    from latent_earth.paths import data_root
    hf = data_root() / "hf"
    if step == 2:
        table, member, n = hf / "data" / "tiles" / "*.parquet", f"{place_id}__{slot}.jpg", 200
        idx = pl.read_parquet(table, columns=["tile_id"])["tile_id"].to_list().index(f"{place_id}__{slot}")
    else:
        table, member, n = hf / "data" / "step1.parquet", f"{place_id}__sample_{slot}.jpg", 80
        t = pl.read_parquet(table)
        idx = t.with_row_index("i").filter((pl.col("place_id") == place_id) & (pl.col("step1_slot") == f"sample_{slot}"))["i"][0]
    from huggingface_hub import hf_hub_download
    for shard in (idx // 1000, idx // 1000 - 1, idx // 1000 + 1):      # shards follow table order; 3 of 200,000 images are absent
        if not 0 <= shard < n:
            continue
        f = hf_hub_download("Punktiert/Latent-Earth", f"images/gen{step}/gen{step}-{shard:05d}-of-{n:05d}.tar",
                            repo_type="dataset", local_dir=hf)
        with tarfile.open(f) as tf:
            try:
                return Image.open(io.BytesIO(tf.extractfile(member).read())).convert("RGB")
            except KeyError:
                continue
    return None
