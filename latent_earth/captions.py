"""Caption text mining: feature-vocabulary genericity (embedding-free outcome).

The VLM's extracted feature phrases are the pipeline's only language, and they
are data: a place whose features are all corpus-common phrases ("red tile
roof", "arched windows") is being described generically; a place with rare
phrases is being differentiated. Genericity here is computed purely from text
— fully independent of any image embedding, which is exactly what makes it the
non-circular outcome for the data-poverty analysis (A4).

Definitions:
  - normalize phrase: lowercase, strip punctuation, collapse whitespace.
  - doc-frequency over PLACES (a phrase counts once per place).
  - place genericity = mean IDF of its phrases, NEGATED and z-scored so
    HIGHER = MORE GENERIC (reads naturally in the regression).
  - top50_share = fraction of the place's phrases in the corpus top-50.
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from pathlib import Path

import numpy as np
import polars as pl

from latent_earth import paths
from latent_earth.config import AnalysisConfig

log = logging.getLogger(__name__)

_norm_re = re.compile(r"[^a-z0-9 ]+")


def normalize_phrase(s: str) -> str:
    s = _norm_re.sub(" ", s.lower())
    return " ".join(s.split())


def load_place_features(config: AnalysisConfig, *, samples: tuple[int, ...] = (0,)) -> dict[str, list[str]]:
    """place_id -> normalized feature phrases (union over the given description slots).

    Renderings 0-3 of a place share ONE description (slot sample_0); the fifth
    probe has its own (sample_4).
    """
    cap = pl.read_parquet(paths.resolve(config.captions_dir))
    cap = cap.filter(pl.col("caption_slot").is_in([f"sample_{s}" for s in samples]))
    out: dict[str, list[str]] = {}
    for pid, feats in cap.select(["place_id", "features"]).iter_rows():
        out.setdefault(pid, []).extend(normalize_phrase(x) for x in (feats or []) if x)
    out = {p: sorted(set(f)) for p, f in out.items() if f}
    log.info("captions: %d places", len(out))
    return out


def genericity_table(place_features: dict[str, list[str]]) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Returns (per-place genericity table, corpus vocabulary table)."""
    n_places = len(place_features)
    df_counter: Counter = Counter()
    for feats in place_features.values():
        df_counter.update(set(feats))
    idf = {ph: float(np.log(n_places / c)) for ph, c in df_counter.items()}
    top50 = {ph for ph, _ in df_counter.most_common(50)}

    rows = []
    for pid, feats in place_features.items():
        if not feats:
            continue
        mean_idf = float(np.mean([idf[f] for f in feats]))
        rows.append({"place_id": pid,
                     "mean_idf": mean_idf,
                     "top50_share": float(np.mean([f in top50 for f in feats])),
                     "n_features": len(feats)})
    tab = pl.DataFrame(rows)
    g = -tab["mean_idf"].to_numpy()
    g = (g - g.mean()) / (g.std() + 1e-12)
    tab = tab.with_columns(pl.Series("genericity", g))

    vocab = pl.DataFrame({
        "phrase": list(df_counter.keys()),
        "doc_freq": list(df_counter.values()),
    }).sort("doc_freq", descending=True)
    return tab, vocab


def run_captions(config: AnalysisConfig) -> pl.DataFrame:
    pf = load_place_features(config)
    tab, vocab = genericity_table(pf)
    tab.write_parquet(config.out_path("caption_genericity.parquet"))
    vocab.head(2000).write_parquet(config.out_path("caption_vocab_top2000.parquet"))
    config.write_result("captions.json", {
        "n_places": tab.height,
        "vocab_size": vocab.height,
        "top_20_phrases": vocab.head(20).to_dicts(),
        "genericity_quantiles": {f"p{p}": float(np.percentile(tab["genericity"], p))
                                 for p in (5, 50, 95)},
    })
    log.info("captions: %d places scored, vocab %d", tab.height, vocab.height)
    return tab
