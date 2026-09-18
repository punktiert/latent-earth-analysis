# Latent Earth: analysis code

Code for the paper *Types, not towns: the architecture a generative image
model imagines for real places* (Daniel Koehler, The University of Texas at
Austin). It turns the public dataset

> **Latent Earth: An Atlas of Architecture in Flux.2**,
> https://doi.org/10.57967/HF/10090

into every number and figure of the paper. The dataset holds 200,000 generated
images of 40,000 real places together with five representations recorded while
each image was generated; this repository holds what was done with them.

Archived at Zenodo: https://doi.org/10.5281/zenodo.22821323 (the release
`public`); https://doi.org/10.5281/zenodo.22821322 resolves to the latest
version.

## What is here, and what is not

| Folder | Content |
| --- | --- |
| `PLAN.md` | the pre-specified analysis plan for the documentation question, verbatim as committed on 14 July 2026 |
| `PROVENANCE.md` | commit record and server-side push log that date the plan before the results; hashes of the analysis inputs; deviations from the plan |
| `latent_earth/` | the analysis package: features, identification, distance decay, consistency, documentation and size, verification |
| `scripts/` | representation of places (`representation.py`), type graph and number of types (`topology.py`), resolution check (`kind_robustness.py`), prompt subsets, pre-specified robustness analyses, atlas statistics |
| `figures/` | one script per figure; each reads only `results/` (and images from the public release) |
| `results/` | the computed results the paper reports (JSON / Parquet) |

Not here: the software that produced the commissioned atlas, that is place
sampling, image generation and description, sorting, printing and the viewer.
The atlas is an artwork of the author, commissioned by the Deutsches
Architekturmuseum (DAM), Frankfurt. Its procedure is described in the paper's
Methods; its prompts, seeds and settings are part of the dataset. The
exhibited layout (which tile sits in which cell) is not published either;
`scripts/atlas_findings.py` and `figures/fig06_wall.py` need it and are
included to document how the atlas statistics were computed. Their results are
in `results/`.

## Run it

```bash
pip install -r requirements.txt
python scripts/download_release.py          # about 7 GB into ./data/hf (tables + embeddings)
python -m latent_earth.cli build-features   # 640-D analysis features -> ./data/derived
python -m latent_earth.cli all              # identification, decay, consistency, documentation, checks
python scripts/prompt_subsets.py
python scripts/representation.py            # how well a place is represented (Fig. 4)
python scripts/topology.py                  # type graph, families, number of types (Fig. 6a,b)
python scripts/kind_robustness.py           # 32 / 64 / 128 types
python scripts/prespecified_robustness.py   # weighted and generator-only repeats of the plan
python figures/make_all.py
```

Set `LATENT_EARTH_DATA` to keep the data outside the repository. 32 GB of
memory is enough; the prompt-encoder channel is the large one.

**Reproduction.** Checked on 17 September 2026 with the analysis features
rebuilt from the release by `build-features`: every identification rate of
`results/geo_knn.json` (three targets, five channels and their combination)
is reproduced to four decimals, and the shares of local, regional and global
types at 32, 64 and 128 types are identical. The numbers in `results/` are
the paper's. Should a future version of the release list the tiles in another
order, the PCA subsample would change with it and a rebuild would agree to the
second or third decimal rather than exactly.

## Licence and citation

Code: MIT (see `LICENSE`). Dataset: CC BY-NC 4.0 (see the dataset card).
Please cite the paper and the dataset, and this code where it is used;
`CITATION.cff` has all three.
