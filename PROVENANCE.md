# Provenance

## The analysis plan was fixed before the results existed

The documentation question of the paper (does online documentation predict how
distinctively a place is rendered?) was analysed under a plan written before
any outcome was computed. `PLAN.md` is that plan, verbatim.

| When (UTC) | What | Evidence |
|---|---|---|
| 13 Jul 2026 11:42 | analysis package and first version of the plan pushed to the project repository | commit `d9262b4` |
| 14 Jul 2026 06:11 | final plan pushed; the file has not changed since | commit `b784d8bf8304867844cd028b67f17609cb2dc96e` |
| 14 Jul 2026 16:12 | analysis features built | file date |
| 15 Jul 2026 16:08 | documentation metrics fetched for all 40,000 places | file date |
| 16 Jul 2026 06:41 | first documentation results pushed | commit `f60df7304d765b4dedc126ce2ea80b08d1ab31db` |

The project repository is private because it also holds the software of the
commissioned artwork. Commit dates are set by the author's machine; the push
times above are GitHub's server-side record
(`GET /repos/punktiert/DAM-Atlas/activity`), reproduced for 12 to 18 July 2026
in `provenance/github_push_log_2026-07.json`. The repository history can be
shown to editors and referees on request. No registry deposit was made before
the analysis; this deposit was made afterwards, and it is the history above,
not the deposit date, that documents the order.

## Deviations from the plan

1. The plan names 38,514 places, the number complete when it was written; the
   analysis ran on all 40,000.
2. The documentation index scored the nearest geotagged article. An audit found
   that this rule returned minor landmarks for large cities; a revised index
   scores the most substantial of the ten nearest articles. Both are reported;
   the verdict is the same.
3. The two robustness analyses named in the plan (weighted estimates;
   generator-state-only repeat) were computed in September 2026, not with the
   main run. The sampling weights had not been stored (`ipf_weight` is zero
   throughout the release) and were rebuilt as post-stratification weights for
   the 25,043 towns and cities (`results/poststrat_weights.parquet`,
   `results/frame_strata_counts.csv`). Results: `results/poverty_robustness.json`.
4. The plan confines analyses beyond its list to the supplement. The split of
   the population association by record type and the size floor are exploratory
   and are reported in the main text, labelled as such, because they change the
   reading of a pre-specified estimate.
5. The plan foresaw a hash record of the inputs before the analysis. It was
   completed on 17 September 2026 (below).

## Analysis inputs

| Input | SHA-256 | Bytes | File date (UTC) | Rows / shape |
|---|---|---|---|---|
| `manifest.parquet` | `609704e66fff3e2107ff64668e8ffe7c1cfa31b40fe6e88c0886e3773fd83aa4` | 10,251,310 | 2026-07-15 16:08 | 200,000 rows |
| `places/places.parquet` | `f60c50ee552a938d19fb8e9e2ac0a1688768dabe1acf9f75d61f802469550476` | 3,263,273 | 2026-07-16 20:46 | 40,000 rows |
| `places/visibility_raw.parquet` | `74d53eba71c3ea464b8bc0f315e24c2278cf4b2abc86451743cdc72baa38f397` | 1,726,892 | 2026-07-15 16:08 | 40,000 rows |
| `layout/analysis_features.npy` | `cffa083b2e3abbfaaefde8315ba91261e83784b4467de221145e5872a1548764` | 511,992,448 | 2026-07-14 16:12 | (199997, 640) |
| `layout/analysis_feature_tile_ids.parquet` | `6f077da0bc4ccc98833e4772a51dc97f9ece966c6370e1ad38ca0c6987801cce` | 1,088,951 | 2026-07-14 16:12 | 199,997 rows |
| `layout/analysis_features_meta.json` | `c5b2246eef4aba0af0d2ed264ceac1929e499cc7a622afcce961b791caf41cbf` | 345 | 2026-07-14 16:12 |  |

These are the files of the author's archive. The public release
(https://doi.org/10.57967/HF/10090, revision `eb14ec2039ccfd708bcde99a03e1753cd6e92efb`) carries the same
recordings; `latent_earth/features.py` rebuilds the analysis features from it.
