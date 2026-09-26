# Main Clustering Study — Qur'an vs. Pre-Islamic Poetry

Extends the reproduced source-study pipeline (Arabic-SBERT-100K embeddings
→ UMAP → HDBSCAN) to compare the Qur'an against the 260-poet corpus at
many granularities, and runs a controlled study of three methodological
confounds: (a) poet sample-size imbalance, (b) vector-averaging producing
artificially "central" representations, (c) HDBSCAN's inability to form a
real cluster from a single new point. See the root `README.md` for the
full narrative and key findings.

**Numbering note:** experiment numbers were assigned in the order ideas
came up, not the order files are listed here, and several numbers were
reassigned mid-project to avoid colliding with earlier ones (documented in
`docs/KNOWN_ISSUES.md`, section 3). The table below is the authoritative
map from experiment number → canonical notebook. Where a number isn't
listed as its own row (3-6, 8-followup), it was a sub-analysis reported
inline rather than an independently numbered experiment.

## Experiment index

All silhouette/percentage/similarity figures below are read directly from
each notebook's own printed cell output (Experiments 1-2, 7-24) or, for
Experiments 25-30, from
`08_experiments13_30_merged.ipynb`'s printed
cell output / `output/reports/experiments_13_30_report.txt`.

| Exp. | Comparison | N | Result | Notebook | Status |
|---|---|---|---|---|---|
| 1 | 260 poets + whole Qur'an (1 entity) | 261 | nearest-5 similarity 0.971–0.976; **19 poets** shared Qur'an's cluster | `01_experiments01_02_combined_clustering.ipynb` | working |
| 2 | 260 poets + 114 surahs (separate) | 374 | **30/114 (26.3%)** surahs in mixed clusters; silhouette **0.469**; ayat-weighted 13.5% | `01_experiments01_02_combined_clustering.ipynb` | working |
| 3–6 | Sub-analyses of Exp. 1–2 (minor-poet subset, 19-poet exclusion, size-matching) | — | minor-poet: 45/114 (39.0%) mixed, silhouette 0.607; nearest-5 similarity after excluding 19 poets: 0.969–0.972 | `02_experiments01_02_followups_extended_analysis.ipynb` | working |
| 7 | 35 well-documented poets + whole Qur'an | 36 | **9 poets** shared Qur'an's cluster; similarity to those 9: **0.967** (range 0.965–0.967) | `04_experiment07_large_poets_only.ipynb` | working (superseded buggy run: `superseded_or_unexecuted/experiment07_large_poets_BUGGY_umap_error.ipynb`) |
| 8 | 35 well-documented poets + 114 surahs (separate) | 149 | **0/114 (0.0%)** mixed — complete separation; silhouette **0.679** (strongest confound-controlled result) | `05_experiment08_surahs_separate_well_documented.ipynb` | working |
| 8-followup | Full cluster membership + figures for Exp. 8 | — | (qualitative detail, no new headline numbers) | `06_experiment08_followup_cluster_membership.ipynb` | working |
| 9 | 35 poets + 60 hizb | 95 | **4/60 (6.7%)** mixed; silhouette **0.739** | `07_experiments09_12_hizb_and_random_sampling.ipynb` | working |
| 10 | 35 poets + 30 hizb-pairs | 65 | **5/30 (16.7%)** mixed; silhouette **0.892** | `07_experiments09_12_hizb_and_random_sampling.ipynb` | working |
| 11 | 35 poets + 1 random surah, ×10 trials | 36 | **9/10** trials shared a cluster with poets (1 outlier) | `07_experiments09_12_hizb_and_random_sampling.ipynb` | working |
| 12 | 35 poets + 1 random 10-surah bundle, ×10 trials | 36 | **10/10** trials shared a cluster with poets | `07_experiments09_12_hizb_and_random_sampling.ipynb` | working |
| 14 | 35 poets + 1 synthetic/randomized control text, ×10 trials | 36 | **10/10** trials shared a cluster with poets — the key control: incoherent text is absorbed just as reliably as real text | `08_experiments13_30_merged.ipynb` | working |
| 15 | 60 hizb alone (no poets) | 60 | silhouette **0.882** | `08_experiments13_30_merged.ipynb` | working |
| 16 | 30 hizb-pairs alone (no poets) | 30 | silhouette **0.543** | `08_experiments13_30_merged.ipynb` | working |
| 17 | 114 surahs alone (no poets) | 114 | silhouette **0.663** (9 clusters, 3 outliers) | `08_experiments13_30_merged.ipynb` | working (see also independent earlier exploratory rerun below) |
| 18 | `AllPoets` (260 avg.) vs. whole Qur'an | 2 | cosine similarity **0.954** | `08_experiments13_30_merged.ipynb` | working |
| 19 | `AllPoets` + 114 surahs | 115 | 4 clusters, silhouette 0.675, `AllPoets` in cluster 1 | `08_experiments13_30_merged.ipynb` | working |
| 20 | `AllPoets` + 60 hizb | 61 | 3 clusters, silhouette 0.870, `AllPoets` in cluster 0 | `08_experiments13_30_merged.ipynb` | working |
| 21 | `AllPoets` + 30 hizb-pairs | 31 | 4 clusters, silhouette 0.661, `AllPoets` in cluster 3 | `08_experiments13_30_merged.ipynb` | working |
| 22 | Whole Qur'an vs. 9 individual poets (unmerged) | 10 | individual similarity range **0.965–0.967**, mean 0.9663 | `08_experiments13_30_merged.ipynb` | working |
| 23 | `35Vector` (35-poet aggregate) vs. whole Qur'an | 2 | cosine similarity **0.962** (0.9615) | `08_experiments13_30_merged.ipynb` | working |
| 24 | `19Vector` (19-poet aggregate) vs. whole Qur'an | 2 | cosine similarity **0.972** (0.9718) | `08_experiments13_30_merged.ipynb` | working |
| 25 | 114 surahs + `35Vector` | 115 | 7 clusters, silhouette 0.631, `35Vector` in cluster 6 | `08_experiments13_30_merged.ipynb` | working |
| 26 | 114 surahs + `19Vector` | 115 | 3 clusters, silhouette 0.749, `19Vector` in cluster 2 | `08_experiments13_30_merged.ipynb` | working |
| 27 | 60 hizb + `35Vector` | 61 | 3 clusters, silhouette 0.826, `35Vector` in cluster 0 | `08_experiments13_30_merged.ipynb` | working |
| 28 | 60 hizb + `19Vector` | 61 | 3 clusters, silhouette 0.849, `19Vector` in cluster 0 | `08_experiments13_30_merged.ipynb` | working |
| 29 | 30 hizb-pairs + `35Vector` | 31 | 4 clusters, silhouette 0.678, `35Vector` in cluster 0 | `08_experiments13_30_merged.ipynb` | working (beyond the notebook's original scope, added and run separately) |
| 30 | 30 hizb-pairs + `19Vector` | 31 | 3 clusters, silhouette 0.626, `19Vector` in cluster 0 | `08_experiments13_30_merged.ipynb` | working (beyond the notebook's original scope, added and run separately) |
| 33 | Umayya ibn Abi al-Salt vs. 114 surahs | 115 | **not independently verified in this pass** | `09_experiments33_36_umayya_scraped_UNEXECUTED.ipynb` | unexecuted (see also `superseded_or_unexecuted/experiments33_36_umayya_NOTFOUND_halted.ipynb`) |
| 34 | Umayya vs. 60 hizb | 61 | **not independently verified in this pass** | `09_experiments33_36_umayya_scraped_UNEXECUTED.ipynb` | unexecuted |
| 35 | Umayya vs. 30 hizb-pairs | 31 | **not independently verified in this pass** | `09_experiments33_36_umayya_scraped_UNEXECUTED.ipynb` | unexecuted |
| 36 | Umayya vs. whole Qur'an | 2 | **not independently verified in this pass** | `09_experiments33_36_umayya_scraped_UNEXECUTED.ipynb` | unexecuted |
| 37 | 240 quarter-hizb + `35Vector` | — | **not independently verified in this pass** — halted by the sqlite empty-DB bug (see `docs/KNOWN_ISSUES.md` §1) | `10_experiments37_40_quarters_eighths_vs_vectors_DBBUG_halted.ipynb` | halted |
| 38 | 240 quarter-hizb + `19Vector` | — | **not independently verified in this pass** | `10_experiments37_40_quarters_eighths_vs_vectors_DBBUG_halted.ipynb` | halted |
| 39 | 480 eighth-hizb + `35Vector` | — | **not independently verified in this pass** | `10_experiments37_40_quarters_eighths_vs_vectors_DBBUG_halted.ipynb` | halted |
| 40 | 480 eighth-hizb + `19Vector` | — | **not independently verified in this pass** | `10_experiments37_40_quarters_eighths_vs_vectors_DBBUG_halted.ipynb` | halted |
| 41–44 | 240/480 quarters/eighths vs. individual poets (35, then 19) | 275/259/515/499 | **not independently verified in this pass** — notebook never executed (no cell outputs) | `11_experiments41_44_quarters_eighths_vs_individual_poets_UNEXECUTED.ipynb` | unexecuted |
| 45–46 | 240/480 quarters/eighths clustered alone | 240/480 | **not independently verified in this pass** | `12_experiments45_46_quarters_eighths_alone_UNEXECUTED.ipynb` | unexecuted |
| 47–51 | Ayah-level clustering (6,236 ayat alone / + poets at various granularity / + 2,328 poems) | up to 8,564 | **not independently verified in this pass** | `13_experiments47_51_ayah_level_clustering_UNEXECUTED.ipynb` | unexecuted |

## Supplementary / exploratory notebook

**`03_quran_alone_clustering_meccan_medinan_exploratory.ipynb`** is an
earlier, standalone exploratory analysis that clusters the 114 surahs by
themselves (same question as Experiment 17) and additionally checks
whether the discovered clusters track the known Meccan/Medinan
classification. Its own printed output: **8 clusters, 3 outliers,
silhouette 0.715** — a different run of a similar question with a
different exact result than Experiment 17's later silhouette of 0.663 (see
`docs/KNOWN_ISSUES.md` §3 on run-to-run variation being expected for this
pipeline). Treat this as a separate, earlier data point, not as
Experiment 17 itself.

## A note on the missing Experiment 13 and 31-32 numbers

This repository covers the clustering study only. Experiment numbers 13
and 31-32 in the project's original numbering belonged to a separate,
unrelated letter-counting sub-project (not a clustering experiment) and
are not part of this repository — the numbering here simply skips them,
which is why the table above jumps from 12 to 14 and from 30 to 33. One
inline artifact of that separate sub-project remains, as project history,
inside `08_experiments13_30_merged.ipynb` (an early, buggy letter-count
cell that predates the notebook's real Experiments 15-30 clustering
content) — see `docs/KNOWN_ISSUES.md` §2 for exactly what it is and why
it's harmless to leave in place.
