# Stylometric Analysis of Quranic Surahs — Summary Report

## Objective

Reproduce a published computational-stylometry pipeline (Othman & Qranfal,
*Hidden Stylistic Schools of Pre-Islamic Poetry*, WIT/IEEE ICAD 2026),
originally used to cluster 260 pre-Islamic poets by writing style, then
apply the same method to the Quran's 114 surahs to test whether they form
stylistic clusters and how those relate to the known Meccan/Medinan
classification.

## Method

1. **Reproduced the original paper exactly.** Same embedding model
   (Arabic-SBERT-100K), same UMAP + HDBSCAN clustering pipeline. Verified
   against the paper's published numbers before extending it (dataset
   size matched exactly: 2,328 poems / 260 poets).
2. **Applied the same method to the Quran.** Fetched all 114 surahs
   (6,236 ayat total) via a public Quran API. Embedded each ayah
   individually, averaged all ayat within a surah into one style vector
   per surah, then ran the identical clustering pipeline.
3. **Checked alignment with Meccan/Medinan classification** — the
   established scholarly distinction between surahs revealed before vs.
   after the Prophet's migration to Medina.
4. **Randomization control** — shuffled which ayah belongs to which
   surah and re-clustered, to check whether the real data produces more
   structure than random chance would.

## Results

| Metric | Quran (this analysis) | Poetry paper (reference point) |
|---|---|---|
| Units clustered | 114 surahs | 260 poets |
| Mean pairwise similarity | **0.93** | 0.80 |
| Clusters found | **8** | 12 |
| Outliers | 3 | 28 |
| Silhouette score | 0.72 | 0.47 |
| Alignment with Meccan/Medinan (ARI, 0–1) | **0.11 (weak)** | — |
| Randomization check | Real: 8 clusters vs. shuffled: 11.4 avg | Real data showed *more* structure than shuffled |

## Interpretation

- **High overall homogeneity.** At 0.93 mean similarity, the 114 surahs
  are far more stylistically uniform than the 260 poets in the reference
  corpus (0.80). The text reads as one continuous stylistic register,
  not a patchwork of clearly distinct voices.
- **Some substructure exists, but it isn't chronological.** The 8
  clusters found do not track the Meccan/Medinan split — alignment was
  weak (0.11). Inspection suggests the clustering is picking up on
  surface features like surah length and verse rhythm instead — one
  cluster, for instance, grouped almost entirely short, rhythmically
  distinct surahs.
- **The randomization result ran opposite to the poetry paper.** There,
  real data produced clearly *more* cluster structure than shuffled
  data — evidence the clusters reflected genuine stylistic signal
  distinguishing different poets. Here, shuffled data produced *more*
  clusters (11.4) than the real data (8) — consistent with the real
  corpus being unusually cohesive and harder to fragment than noise.

## What this does and does not show

This method measures **writing-style similarity**, not authorship.
Stylistic homogeneity or variation on its own cannot establish or rule
out single authorship — a single author's style can shift meaningfully
across a long body of work spanning years and contexts, and shared style
across a large body of text is equally consistent with several
different explanations. Clustering results describe measurable
stylistic patterns; they don't adjudicate historical or theological
questions about authorship.

What this analysis does show: by this specific measure, the Quran's 114
surahs form an unusually stylistically consistent body of text — more
so than a comparable corpus known to have many different human authors —
with modest internal variation that does not map cleanly onto the
already-known Meccan/Medinan distinction.

## Caveats

- Clustering parameters (neighbor count, minimum cluster size) were
  scaled down for a much smaller sample (114 vs. 260 units); results are
  somewhat sensitive to these settings.
- Averaging all ayat in a surah into one vector is one reasonable choice
  among several possible ways to represent a surah's style; a different
  aggregation method could shift the exact cluster count.
- Meccan/Medinan is an established scholarly classification, not a
  stylometric ground truth — weak alignment means style and revelation
  period aren't the same axis, not that the classification is incorrect.
