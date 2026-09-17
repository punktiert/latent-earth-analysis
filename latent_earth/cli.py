"""Command line: python -m latent_earth.cli <command>

    build-features     640-D analysis features from the release embeddings
    geo-info           identification (kNN) and distance decay
    dispersion         per-place consistency
    poverty            documentation and size against the four pre-specified outcomes
    probe-validity     prompt checks and the audit sample
    s3-checks          verification analyses
    all                everything above, in order

Results land in results/, the folder the figure scripts read.
"""

from __future__ import annotations

import io
import logging
import sys

import click
import numpy as np

from latent_earth.config import AnalysisConfig
from latent_earth import features as feat_mod

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", line_buffering=True)
    except (AttributeError, ValueError):
        pass

log = logging.getLogger("latent_earth.cli")


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S", stream=sys.stderr)


def _load(config: AnalysisConfig):
    from latent_earth import util
    feats, tile_ids, blocks = feat_mod.load_analysis_features(config)
    tiles = util.tile_table(config, tile_ids)
    return feats, tiles, blocks


@click.group()
def main() -> None:
    """Analyses of the Latent Earth corpus."""


@main.command(name="build-features")
@click.option("--first-n", type=int, default=None, help="Cap tiles (smoke test).")
@click.option("--out", "out_dir", default=None, help="Results dir (default paper/results).")
@click.option("-v", "--verbose", is_flag=True)
def build_features(first_n, out_dir, verbose) -> None:
    """Build the modality-comparable analysis features (all samples 0-4)."""
    _setup_logging(verbose)
    config = AnalysisConfig.from_env(**({"out_dir": out_dir} if out_dir else {}))
    feats, tile_ids, blocks = feat_mod.build_analysis_features(config, first_n=first_n)
    feat_mod.save_analysis_features(config, feats, tile_ids, blocks)
    click.echo(f"analysis features: {feats.shape} blocks={list(blocks)}")


def _common(f):
    f = click.option("--out", "out_dir", default=None,
                     help="Results dir (default paper/results).")(f)
    f = click.option("-v", "--verbose", is_flag=True)(f)
    return f


@main.command(name="geo-info")
@_common
def geo_info_cmd(out_dir, verbose) -> None:
    """A1 kNN geographic purity + A2 distance decay / embedding Mantel."""
    _setup_logging(verbose)
    from latent_earth import geo_info
    config = AnalysisConfig.from_env(**({"out_dir": out_dir} if out_dir else {}))
    feats, tiles, blocks = _load(config)
    geo_info.run_knn_purity(feats, tiles, blocks, config)
    geo_info.run_distance_decay(feats, tiles, blocks, config)
    click.echo("geo_knn.json + geo_decay.json written")


@main.command(name="dispersion")
@_common
def dispersion_cmd(out_dir, verbose) -> None:
    """Per-place consistency."""
    _setup_logging(verbose)
    from latent_earth import dispersion
    config = AnalysisConfig.from_env(**({"out_dir": out_dir} if out_dir else {}))
    feats, tiles, blocks = _load(config)
    dispersion.run_dispersion(feats, tiles, blocks, config)
    click.echo("dispersion.json written")


@main.command(name="poverty")
@_common
def poverty_cmd(out_dir, verbose) -> None:
    """A4 data poverty -> homogenization (headline)."""
    _setup_logging(verbose)
    from latent_earth import poverty
    config = AnalysisConfig.from_env(**({"out_dir": out_dir} if out_dir else {}))
    feats, tiles, blocks = _load(config)
    poverty.run_poverty(feats, tiles, blocks, config)
    click.echo("poverty.json written")


@main.command(name="probe-validity")
@_common
def probe_validity_cmd(out_dir, verbose) -> None:
    """A5 scale-collapse curve + leakage-audit sample."""
    _setup_logging(verbose)
    from latent_earth import probe_validity
    config = AnalysisConfig.from_env(**({"out_dir": out_dir} if out_dir else {}))
    feats, tiles, _blocks = _load(config)
    probe_validity.run_probe_validity(feats, tiles, config)
    click.echo("probe_validity.json written")


@main.command(name="s3-checks")
@_common
def s3_checks_cmd(out_dir, verbose) -> None:
    """S3: decay decomposition (same/diff country) + stable-vs-identified overlap."""
    _setup_logging(verbose)
    from latent_earth import s3_checks
    config = AnalysisConfig.from_env(**({"out_dir": out_dir} if out_dir else {}))
    feats, tiles, _blocks = _load(config)
    s3_checks.run_s3(feats, tiles, config)
    click.echo("s3_decay_decomposition.json + s3_overlap.json written")


@main.command(name="all")
@_common
def run_all(out_dir, verbose) -> None:
    """A1-A5 in order (features must exist; build-features first)."""
    _setup_logging(verbose)
    from latent_earth import dispersion, geo_info, poverty, probe_validity
    config = AnalysisConfig.from_env(**({"out_dir": out_dir} if out_dir else {}))
    feats, tiles, blocks = _load(config)
    click.echo("[1/4] geo-info…")
    geo_info.run_knn_purity(feats, tiles, blocks, config)
    geo_info.run_distance_decay(feats, tiles, blocks, config)
    click.echo("[2/4] dispersion…")
    dispersion.run_dispersion(feats, tiles, blocks, config)
    click.echo("[3/4] poverty…")
    poverty.run_poverty(feats, tiles, blocks, config)
    click.echo("[4/4] probe-validity…")
    probe_validity.run_probe_validity(feats, tiles, config)
    click.echo(f"\nAll results -> {config.out_dir}")


if __name__ == "__main__":
    main()
