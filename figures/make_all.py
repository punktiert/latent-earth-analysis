"""Regenerate every figure from paper/results (repo-only figures).

Figures with images (fig01, fig09, fig10) read the archive when a data root
is mounted, else the public release (style.release_image). fig06_wall needs
the exhibited grid, which is not part of the release: run it where the NAS
is reachable.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

for script in ("fig01_probe.py", "fig02_geography.py", "fig03_maps.py",
               "fig04_representation.py", "fig05_documentation.py",
               "fig06_wall.py", "fig06_atlas.py", "fig07_channels.py", "fig08_census.py",
               "fig09_typesheet.py", "fig10_noarch.py"):
    print(f"== {script}")
    runpy.run_path(str(HERE / script), run_name="__main__")
print("done.")
