"""A5: probe validity — prompt token economy (Methods subsection + Supplement S1).

Quantifies the pipeline-characterization findings so reviewers can trust the
probe: (i) the scale-collapse curve — how inter-scale embedding distance
shrinks as feature tokens crowd the prompt (the v9 fix's rationale, now
measured on the full corpus); (ii) a stratified audit sample for the
interior-leakage recount (list only; the audit itself is human/VLM).

Output: results/probe_validity.json + audit_sample.parquet.
"""

from __future__ import annotations

import logging

import numpy as np
import polars as pl

from latent_earth import paths
from latent_earth.config import AnalysisConfig
from latent_earth.dispersion import scale_centered  # noqa: F401 (kept for S1 ext)

log = logging.getLogger(__name__)


def scale_collapse_curve(
    feats: np.ndarray,
    tiles: pl.DataFrame,
    config: AnalysisConfig,
) -> dict:
    """Mean inter-scale distance per place, binned by the place's n_features.

    Prediction (from the v5-v8 failures): with more feature tokens, the scale
    noun's influence shrinks -> the four scale renderings converge. n_features
    comes from the manifest.
    """
    man = pl.read_parquet(paths.resolve(config.manifest_parquet))
    if "n_features" not in man.columns:
        log.warning(
            "manifest has no n_features column (reconstructed manifest?). "
            "Run `python -m latent_earth.cli enrich-manifest` to join the node manifests' "
            "caption metadata first. Skipping the scale-collapse curve.")
        return None
    nf = man.filter(pl.col("sample_idx") == 0).select(
        ["place_id", "n_features"]).unique(subset=["place_id"])

    sub = tiles.filter(pl.col("sample_idx").is_in([0, 1, 2, 3]))
    x = feats[sub["row_idx"].to_numpy()]
    n = np.linalg.norm(x, axis=1, keepdims=True)
    x = x / np.maximum(n, 1e-8)
    pid = sub["place_id"].to_numpy()
    uniq, inv = np.unique(pid, return_inverse=True)
    d = x.shape[1]
    sums = np.zeros((uniq.shape[0], d), dtype=np.float64)
    cnts = np.zeros(uniq.shape[0], dtype=np.int64)
    np.add.at(sums, inv, x.astype(np.float64))
    np.add.at(cnts, inv, 1)
    nrm2 = np.einsum("ij,ij->i", sums, sums)
    nn = cnts.astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        inter = 1.0 - (nrm2 - nn) / (nn * (nn - 1.0))     # mean pairwise cos dist
    tab = pl.DataFrame({"place_id": uniq, "inter_scale_dist": inter,
                        "n_samples": cnts}).filter(pl.col("n_samples") >= 2)
    tab = tab.join(nf, on="place_id", how="left").drop_nulls(["n_features"])

    curve = (tab.group_by("n_features")
                .agg([pl.col("inter_scale_dist").mean().alias("mean_dist"),
                      pl.col("inter_scale_dist").std().alias("std"),
                      pl.len().alias("n")])
                .sort("n_features"))
    from latent_earth.util import spearman_perm
    rng = np.random.default_rng(config.seed + 4)
    rho = spearman_perm(tab["n_features"].to_numpy().astype(np.float64),
                        tab["inter_scale_dist"].to_numpy(),
                        rng=rng, n_permutations=config.n_permutations)
    return {"curve": curve.to_dicts(), "spearman_nfeat_vs_dist": rho}


def audit_sample(
    tiles: pl.DataFrame,
    config: AnalysisConfig,
    *,
    n: int = 1000,
) -> pl.DataFrame:
    """Stratified tile sample for the manual/VLM leakage audit (paths only)."""
    rng = np.random.default_rng(config.seed + 5)
    man = pl.read_parquet(paths.resolve(config.manifest_parquet))
    sub = man.filter(pl.col("sample_idx").is_in([0, 1, 2, 3]))
    per_scale = max(1, n // 4)
    picks = []
    for s in (0, 1, 2, 3):
        ss = sub.filter(pl.col("sample_idx") == s)
        take = min(per_scale, ss.height)
        idx = rng.choice(ss.height, size=take, replace=False)
        picks.append(ss[idx.tolist()])
    want = ["tile_id", "place_id", "sample_idx", "step2_image_path", "step2_prompt"]
    have = [c for c in want if c in man.columns]
    out = pl.concat(picks).select(have)
    out.write_parquet(config.out_path("audit_sample.parquet"))
    return out


def run_probe_validity(
    feats: np.ndarray,
    tiles: pl.DataFrame,
    config: AnalysisConfig,
) -> dict:
    out = {"scale_collapse": scale_collapse_curve(feats, tiles, config)}
    aud = audit_sample(tiles, config)
    out["audit_sample_n"] = aud.height
    config.write_result("probe_validity.json", out)
    return out
