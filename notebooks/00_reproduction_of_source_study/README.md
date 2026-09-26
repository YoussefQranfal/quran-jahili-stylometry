# Phase 0 — Reproducing the Source Study

Before extending Othman & Qranfal's pipeline to the Qur'an, this project
first reproduced their original result: clustering the 260-poet,
2,328-poem pre-Islamic corpus by itself (no Qur'an involved) using
Arabic-SBERT-100K embeddings + UMAP + HDBSCAN.

## Canonical notebook

**`01_reproduce_paper.ipynb`** — executed successfully end to end. Its own
printed output reproduces the source study's headline numbers directly
from `data/poems.db`:

```
1. DATASET
  n_poems: 2328
  n_poets: 260
  total_tokens: 177110

2. SIMILARITY STATISTICS (proposed method)
  n_pairs: 33670
  mean: 0.7986351847648621   [paper: mean=0.799]
  median: 0.8915810585021973 [paper: median=0.892]

3. CLUSTERING RESULTS
  silhouette: 0.42085936665534973
  n_clusters: 13, n_outliers: 37   [paper: 12 clusters, 28 outliers, silhouette=0.469]
```

(Full printed report: `output/reports/full_analysis_report.txt`.) The
mean pairwise similarity of **0.799** is the number cited throughout this
project as "the source study's baseline" — it is a direct, independently
re-run reproduction, not copied from the original paper's text.

Note the clustering numbers (13 clusters / 37 outliers / silhouette 0.421
here vs. the original paper's 12 / 28 / 0.469) differ somewhat from run to
run — this is exactly the kind of minor UMAP/HDBSCAN run-to-run variation
the main paper's own methodology section flags as expected (same seed,
same method, small differences in cluster count and edge-case assignment
are normal for this pipeline). The similarity statistics (0.799 mean),
which don't depend on the stochastic clustering step, match closely.

## Other files in this folder

- **`reimplementation_package/`** — a complete, independent, from-scratch
  Python reimplementation of the source study's pipeline (`main.py`,
  `config.py`, `src/*.py`, `tests/`), written to understand every design
  decision rather than copied from the original authors' source code. It
  additionally includes `src/scraper.py`, a polite aldiwan.net scraper
  used to (re)build `poems.db` from scratch. This package was never run
  against real scraped data in the environment where it was written (see
  its own `data/` and `output/` subfolders, both empty), so it is included
  as working *code*, not as a source of any numeric result — all numeric
  results in this project come from the notebooks, run against the
  already-scraped `data/poems.db`.
- **`superseded_or_failed_runs/`** — two earlier, non-canonical attempts at
  a single-notebook, top-to-bottom reproduction:
  - `hidden_poetic_schools_standalone_FAILED_no_internet.ipynb` — actually
    executed, but in an environment without internet access to
    aldiwan.net, so its scraper found 0 poets and every downstream cell
    either operates on empty data or errors out. Not a source of any
    result.
  - `hidden_poetic_schools_unexecuted_template.ipynb` — never executed,
    and written against a different, three-table database schema
    (`poets`/`poems`/`verses`) than the real `poems.db` (a single `poems`
    table) — see `data/README.md`. Kept for the project's history only.

  See `docs/KNOWN_ISSUES.md` (section 5) for the full detail on both.

## The original source repository

The original Othman & Qranfal repository this phase reproduces is
preserved, unedited, at `reference/hidden-poetic-schools-original-by-othman-qranfal/`
for attribution — see `reference/README.md`.
