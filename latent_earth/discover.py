"""File lookup for the scripts and figures."""

from __future__ import annotations

from pathlib import Path

from latent_earth.paths import data_root


def find_layout(name: str) -> Path | None:
    """Derived feature files, or the (non-public) atlas layout if you have it."""
    for d in (data_root() / "derived", data_root() / "layout"):
        if (d / name).exists():
            return d / name
    return None


def find_data(rel: str) -> Path | None:
    """Loose image files of a local archive; absent in the public setting, where images come from the release shards."""
    p = data_root() / "archive" / rel
    return p if p.exists() else None
