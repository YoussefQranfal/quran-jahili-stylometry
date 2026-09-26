# Computational Stylometry of the Qur'an and Pre-Islamic Arabic Poetry

A reproducible research repository containing every notebook and the
corpus database behind a multi-granularity computational stylometric
comparison of the Qur'anic text against a 260-poet, 2,328-poem corpus of
pre-Islamic (Jahili) Arabic poetry.

It embeds poets and Qur'anic text with an Arabic Sentence-BERT model,
clusters them with UMAP + HDBSCAN, and runs a controlled study of three
methodological confounds: poet sample-size imbalance, vector-averaging
producing artificially "central" representations, and HDBSCAN's inability
to form a real cluster from a single new point. Its conclusions are about
**embedding similarity under one specific computational pipeline** — not
about authorship, historical composition, or theological questions.

This project reproduces, then extends, a published pipeline:

> Salem Othman & Youssef Qranfal (Wentworth Institute of Technology),
> "Hidden Stylistic Schools of Pre-Islamic Poetry: A Computational Analysis
> Using Sentence Embeddings," submitted to IEEE ICAD 2026.
> Original repository (reference copy, attributed, not this project's own
> work): `reference/hidden-poetic-schools-original-by-othman-qranfal/`,
> upstream at <https://github.com/SalemOthman/hidden-poetic-schools>.

## Key Findings

- Reproducing the source study on the 260-poet corpus alone gives a mean
  pairwise poet similarity of **0.799** (median 0.892) — this project's
  own independent rerun, not copied from the original paper's text (see
  `notebooks/00_reproduction_of_source_study/`).
- Taken at face value, the whole Qur'an looks stylistically close to the
  poetry corpus: nearest-poet similarities of 0.97+ and 19/260 poets
  sharing its cluster (Experiment 1).
- That result is **not robust** once three confounds are controlled for.
  The strongest confound-controlled finding: comparing the Qur'an's 114
  chapters against only the 35 poets with substantial surviving output
  (>286 verses) gives **complete separation — 0/114 chapters share a
  cluster with any poet** (silhouette 0.679, Experiment 8).
- The key control result: **synthetic, meaningless text gets absorbed
  into a poet cluster just as reliably as real Qur'anic text** (10/10
  trials, Experiment 14, vs. 9/10 for real single surahs, Experiment 11)
  — direct evidence that HDBSCAN's "a lone new point joins a nearby
  cluster" behavior, not genuine stylistic overlap, drives much of the
  apparent similarity.
- Findings are **granularity- and corpus-selection-dependent**: hizb and
  hizb-pair granularities retain partial mixing (6.7% and 16.7%
  respectively, Experiments 9-10) even under the same confound controls
  that produce complete separation at the chapter level.

## Project Structure

```
quran-jahili-stylometry/
├── README.md                       # this file
├── LICENSE                         # MIT (code); see LICENSE for data caveats
├── requirements.txt
├── .gitignore
├── data/
│   ├── poems.db                    # 3,457,024 bytes, table `poems`, 2,361 rows
│   └── README.md                   # schema + Quran-fetch-at-runtime notes
├── notebooks/
│   ├── 00_reproduction_of_source_study/
│   │   ├── 01_reproduce_paper.ipynb          # canonical phase-0 reproduction
│   │   ├── reimplementation_package/         # independent src/*.py rebuild + scraper
│   │   ├── superseded_or_failed_runs/
│   │   └── README.md
│   └── 01_main_clustering_study/
│       ├── 01_...combined_clustering.ipynb   # Experiments 1-2
│       ├── 02_...extended_analysis.ipynb     # Experiments 3-6 (sub-analyses)
│       ├── 03_...meccan_medinan_exploratory.ipynb
│       ├── 04_...experiment07...ipynb        # Experiment 7
│       ├── 05_...experiment08...ipynb        # Experiment 8
│       ├── 06_...experiment08_followup...ipynb
│       ├── 07_...experiments09_12...ipynb    # Experiments 9-12
│       ├── 08_...experiments13_30_merged.ipynb   # Experiments 14-30
│       ├── 09-13_...                         # Experiments 33-51 (see status column)
│       ├── superseded_or_unexecuted/
│       └── README.md                         # full experiment index + numbering map
├── docs/
│   ├── KNOWN_ISSUES.md             # bugs, renumbering history, unexecuted notebooks
│   └── reports/                    # supporting narrative reports + index
├── output/
│   ├── figures/                    # 17 rendered figures from the experiments
│   ├── tables/                     # README pointing to notebook-generated CSVs
│   └── reports/                    # verbatim text reports, reproduced from notebook output
└── reference/
    └── hidden-poetic-schools-original-by-othman-qranfal/   # original repo, attribution only
```

## Quick Start

```bash
git clone <this-repo-url>
cd quran-jahili-stylometry
pip install -r requirements.txt
```

`data/poems.db` is committed to this repository (it's the whole point of
turnkey reproducibility here) — no separate download step. Sanity-check it:

```bash
python -c "import sqlite3; print(sqlite3.connect('data/poems.db').execute('SELECT COUNT(*) FROM poems').fetchone())"
# expected: (2361,)
```

Then run notebooks in numeric order within each `notebooks/` subfolder —
`00_reproduction_of_source_study/` first, then `01_main_clustering_study/`.
Each notebook expects `data/poems.db` to be reachable from its own working
directory — copy or symlink it next to whichever notebook you're running
(see `data/README.md` for the sqlite gotcha this avoids).

- **Qur'anic text:** fetched fresh from the free, public
  [Al Quran Cloud API](https://alquran.cloud/api) on first run of any
  notebook that needs it, and cached locally afterward (`quran_cache/*.json`,
  gitignored). No manual data step needed.
- **Arabic-SBERT-100K model:** downloads automatically via
  `sentence-transformers`/`transformers` on first embedding run. A GPU
  will speed this up substantially; on CPU alone, the full 260-poet
  embedding step can take significantly longer. (Notebook runtimes quoted
  in each notebook's own intro — e.g. "~30-40 min first run, mostly
  embedding" — reflect the kind of machine this project was developed on;
  your mileage will vary with available hardware.)

## Data Access

The poetry corpus (`data/poems.db`) was compiled from
[Aldiwan.net](https://www.aldiwan.net), a publicly accessible repository of
Arabic poetry — the same source the original Othman & Qranfal study used.
See `data/README.md` for the full schema and provenance notes, and
`notebooks/00_reproduction_of_source_study/reimplementation_package/src/scraper.py`
for the (re)scraping code, if you ever need to rebuild it from scratch.

## Key Parameters

| Parameter | Value |
|---|---|
| Embedding model | `akhooli/Arabic-SBERT-100K` (768-dim, L2-normalized) |
| Poem aggregation | verse → poem → poet hierarchical average; max 20 sampled segments per poem, seed 42 |
| UMAP `n_neighbors` | `min(15, n-1)` |
| UMAP `n_components` | `min(50, n-2)` |
| UMAP `min_dist` | 0.0 for clustering, 0.1 for 2D visualization |
| UMAP metric | cosine |
| HDBSCAN `min_cluster_size` | `min(5, floor(n/10))` |
| HDBSCAN `min_samples` | 3 |
| HDBSCAN cluster selection | excess of mass |
| HDBSCAN metric | Euclidean (applied after UMAP) |
| Random seed | 42, throughout |
| "Well-documented" poet rule | total surviving output > 286 verses (the largest surah's verse count) |
| "Minor" poet rule | total surviving output ≤ 286 verses |
| Synthetic control text | ~200 tokens, 6-10 token pseudo-verses, 10 trials, seeds 1001-1010 |

## Results Summary

See `notebooks/01_main_clustering_study/README.md` for the complete,
experiment-by-experiment index (all ~50 experiments, with status). Headline
rows:

| Experiment | Comparison | Result |
|---|---|---|
| Source study reproduction | 260 poets alone | mean pairwise similarity 0.799 |
| 1 | 260 poets + whole Qur'an | 19/260 poets share Qur'an's cluster; nearest-5 similarity 0.971-0.976 |
| 2 | 260 poets + 114 surahs (separate) | 30/114 (26.3%) mixed; silhouette 0.469 |
| 7 | 35 well-documented poets + whole Qur'an | 9 poets share cluster; similarity 0.965-0.967 |
| 8 | 35 well-documented poets + 114 surahs | **0/114 (0.0%) mixed** — silhouette 0.679 |
| 9 | 35 poets + 60 hizb | 4/60 (6.7%) mixed; silhouette 0.739 |
| 10 | 35 poets + 30 hizb-pairs | 5/30 (16.7%) mixed; silhouette 0.892 |
| 11 | 35 poets + 1 random surah ×10 | 9/10 trials joined a poet cluster |
| 12 | 35 poets + 1 random 10-surah bundle ×10 | 10/10 trials joined a poet cluster |
| 14 | 35 poets + synthetic control text ×10 | **10/10 trials joined a poet cluster** (key control) |
| 15-17 | Qur'anic units alone (no poets) | silhouette 0.882 (60 hizb), 0.543 (30 hizb-pairs), 0.663 (114 surahs) |
| 18 | `AllPoets` (260) vs. whole Qur'an | similarity 0.954 |
| 22 | Whole Qur'an vs. 9 individual poets | similarity range 0.965-0.967 |
| 23 / 24 | `35Vector` / `19Vector` vs. whole Qur'an | 0.962 / 0.972 |

## Experiment Index

The complete, notebook-by-notebook index — including every notebook's
execution status (working / superseded / unexecuted / halted by a known
bug) — lives in `notebooks/01_main_clustering_study/README.md` (Experiments
1-51; note that experiment numbers 13 and 31-32 belong to a separate,
out-of-scope sub-project and don't appear here — see
`docs/KNOWN_ISSUES.md`).

## Reproducibility Notes

One bug and one renumbering quirk recur throughout this project's history
and are documented in full, with the exact notebooks and error messages
involved, in **`docs/KNOWN_ISSUES.md`**:

1. `sqlite3.connect(path)` silently creates an empty database instead of
   erroring when `path` doesn't exist — causes `no such table: poems`.
2. Experiment numbers were reassigned several times mid-project to avoid
   colliding with earlier ones.

`docs/KNOWN_ISSUES.md` also lists every notebook that has **no executed
cell output** in the copies available for this repository (Experiments
33-51, in whole or in part) — their design is included for completeness,
but their numeric claims are explicitly marked "not independently verified
in this pass" rather than presented as results.

## Citation

This repository reproduces and extends the methodology of the following
published study; cite it when referring to the source pipeline:

```bibtex
@inproceedings{othman2026hidden,
  author    = {Othman, Salem and Qranfal, Yaser},
  title     = {Hidden Stylistic Schools of Pre-Islamic Poetry},
  booktitle = {Proceedings of the IEEE International Conference on Advanced Data Mining and Applications (ICAD)},
  year      = {2026},
  address   = {Wentworth Institute of Technology}
}
```

(Note: this bib entry itself flags "publication details to be verified
before submission" in the source study's own reference list; the original
repository's own README spells the second author's name "Youssef
Qranfal" — both spellings appear across the source materials and are
reported here exactly as found, without resolving the discrepancy.)

## License

Code and written analysis in this repository: **MIT** (see `LICENSE`),
matching the original Othman & Qranfal repository's license.

`data/poems.db`'s poem text was compiled from Aldiwan.net. As the original
authors' own README notes for their (not-included) database, this text
carries whatever copyright status publicly-posted classical Arabic poetry
on that site carries — treat redistribution of the raw poem text itself
with the same caution the original authors expressed, independent of this
repository's own MIT license on the code and analysis built on top of it.
The Qur'anic text used throughout is fetched at runtime from the public Al
Quran Cloud API and is not redistributed as a bundled file in this
repository.
