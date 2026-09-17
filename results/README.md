# results: the computed results the paper reports

Every figure script reads only from this folder. The analysis code that
produces these files from the dataset is in the analysis repository (see the
dataset card); the numbers here are the ones in the paper.

| File | Producer | Content |
|---|---|---|
| geo_knn.json | `latent_earth.cli geo-info` | identification: share of places whose 10 most similar places carry the same label, per channel and target (accuracy, balanced accuracy, majority baseline, permutation null, bootstrap interval) |
| geo_decay.json | `latent_earth.cli geo-info` | similarity against geographic distance per channel; corpus floor; distance-similarity correlation |
| dispersion.json, dispersion_per_place.parquet | `latent_earth.cli dispersion` | distance among a place's four renderings. Pooled over places whose four prompts are identical and places whose prompts carry framing phrases; read together with prompt_subsets |
| prompt_subsets.json, prompt_subsets.parquet | `scripts/prompt_subsets.py` | per place: number of distinct second-stage prompts and whether the four are identical (30,366 places) or carry framing phrases (9,634); dispersion and attempt-by-attempt distances per subset |
| poverty.json, poverty_per_place.parquet | `latent_earth.cli poverty` | the pre-specified documentation analysis: four outcomes against the documentation index and against population (Spearman with permutation p, within-country rank slope with cluster bootstrap); per-place outcomes and the type label (k = 64) |
| poverty_robustness.json | `scripts/prespecified_robustness.py` | the two robustness analyses named in the plan: estimates weighted by post-stratification weights (towns and cities), and the repeat in the generator's own state only |
| poststrat_weights.parquet, frame_strata_counts.csv | `scripts/prespecified_robustness.py` | post-stratification weights for the 25,043 towns and cities (places in the source tables divided by places in the sample, per population band and UN subregion) and the stratum counts they are built from. The `ipf_weight` column of the place table is zero throughout and must not be used |
| captions.json, caption_genericity.parquet, caption_vocab_top2000.parquet | `latent_earth.cli poverty` | vocabulary of the descriptions and the genericity outcome |
| probe_validity.json, audit_sample.parquet | `latent_earth.cli probe-validity` | prompt checks; a 1,000-image stratified audit sample |
| s2_flux_arms.json | replication arms | identification under perturbed seeds and at 1,024 px / 28 steps on a 1,000-place subsample |
| s2_lcagcs.json | cross-model replication | identification in a sample of the LCA-GCS corpus (Stable Diffusion 2.1) |
| s3_decay_decomposition.json, s3_overlap.json, s3_place_correct.parquet | `latent_earth.cli s3-checks` | distance decay split into same-country and different-country pairs; per-place identification |
| representation.json, representation_per_place.parquet | `scripts/representation.py` | how well a place is represented: share of bare names without architecture; distances and retrieval for a fourth seed and for the fifth, independent probe (image space and generator state); where the fifth probe lands; feature recurrence against four controls; retrieval by population; size breakpoint; slopes by record type |
| noarch_inspection.csv | manual inspection | 203 first-stage images in which the vision model found no architecture: class and, where noted, the literal reading of the name |
| topology.json, type_graph.parquet, type_graph_nodes.parquet | `scripts/topology.py` | the type graph in the 640-dimensional representation space: closure, the 111 links, nine families, links between families, redraw transitions, agreement between adjacency in the atlas and affinity in representation space, number-of-types diagnostics (k = 8 to 512, HDBSCAN). Positions of types in the atlas are omitted |
| kind_robustness.json | `scripts/kind_robustness.py` | geographic reach and atlas compactness of the types at k = 32, 64 and 128 |
| atlas_wall.json, atlas_kinds.parquet | `scripts/atlas_findings.py` | aggregate statistics of the atlas: spread of a place's four tiles, legibility of labels, compactness of types, regions, climates and countries, adjacency between types, repetitiveness; the 64 types with their names (features common in the type and rare outside it, with shares), sizes, geographic spread and compactness. The layout itself is not published |

`representation.*` are computed from the public release alone. The other
embedding-based results use the 640-dimensional analysis features of the
paper's Methods (128 whitened principal components per channel), which the
analysis repository rebuilds from the embeddings in this release; a rebuild
agrees with the numbers here to within the sampling noise of the PCA fit.
