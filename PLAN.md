# Pre-specification: documentation and differentiation (analysis A4)

_Freeze this document on OSF (osf.io) BEFORE running `lca-analysis poverty` on
the real corpus, and cite the OSF timestamp in the paper._

_This is an ESTIMATION pre-specification, not a directional hypothesis test.
It fixes WHAT will be measured and HOW, before the data are seen. It does not
predict the direction or size of any association; every pre-specified estimate
is reported regardless of sign or significance, and the paper's conclusions
are drawn from the estimates after they exist._

## Research question

Is the degree to which the model differentiates a place in its renderings
associated with how documented that place is online — and if so, in which
direction, at what magnitude, and does the association persist within
countries?

Three outcomes are conceivable and all are reportable findings:
  (a) less-documented places are rendered LESS distinctively (compression
      toward generic types);
  (b) less-documented places are rendered MORE distinctively (e.g. exoticized
      or unstable renderings);
  (c) no association beyond country-level differences (differentiation is
      geopolitical, not documentational).

## Data (frozen per paper/DATA_FREEZE.md before this analysis runs)

- Corpus: DAM Atlas, Flux.2 [dev], 38,514 places × 4 wall samples (sample_4
  excluded from these outcomes; it serves only the A3 replicate check).
- Features: analysis features (5 modalities × PCA-128, whitened, L2-normed);
  place-level = mean of samples 0-3, re-normalized.
- Predictor: wiki_visibility_score = PC1 of z-scored [geotagged en-wiki
  article within 2 km (0/1), log1p(article bytes), log1p(Wikidata sitelinks),
  log1p(Commons geotagged files within 1 km)], oriented so higher = more
  documented. Computed by lca/places/visibility.py before the outcomes are
  computed or inspected.

## Pre-specified outcome measures (place-level, concat features)

1. local_density: mean cosine distance to the 20 nearest other places
   (a DISTANCE: higher = more isolated/distinctive neighborhood).
2. genericity: z-scored negative mean-IDF of the place's VLM caption phrases
   (doc-frequency over places; higher = more corpus-common vocabulary).
   Text-only — independent of any image embedding.
3. centroid_dist: cosine distance to the place's UN-region centroid.
4. megacluster_log2size: log2 size of the place's k=64 MiniBatchKMeans
   cluster (fixed seed).

## Pre-specified estimates and uncertainty

- Spearman rho of visibility vs each outcome, two-sided permutation p
  (999 permutations, seed 20260713).
- Within-country association: rank-OLS slope (ranks computed and demeaned
  within country, pooled), 95% CI by cluster bootstrap over countries
  (1,000 draws). This is the estimate that separates a documentation effect
  from between-country differences.
- Robustness (reported regardless): log-population as alternative predictor;
  ipf_weight-weighted sensitivity; flux_residual-block-only repeat.

## Interpretation rules (fixed in advance)

- An association is reported as "consistent" only if >= 3 of the 4 outcomes
  agree in sign AND their within-country CIs exclude 0.
- If raw Spearman is non-null but within-country CIs include 0, the finding
  is reported as between-country only (pattern (c) above).
- Effect sizes are reported as rank correlations and slopes with CIs; no
  claim rests on p-values alone.
- No outcome, transformation, or subgroup beyond those listed here enters the
  main text; anything else is labeled exploratory in the supplement.

## Exclusions (fixed in advance)

Places with fetch_error in the visibility backfill after 3 retry passes
(count reported); the 3 tiles with incomplete embeddings; places with fewer
than 2 wall samples.
