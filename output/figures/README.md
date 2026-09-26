# `output/figures/`

These 17 PNGs are the rendered figures for the main clustering study's
experiments, saved directly from the notebooks that produced them. They
are the canonical, already-rendered figures — the source of truth if
you're citing a figure.

| Figure file | Content |
|---|---|
| `fig_baseline_paper.png` | Reproduction of the source study's original 260-poet-only clustering (no Qur'an), Appendix C |
| `fig_exp1.png` | Experiment 1 — 260 poets + whole Qur'an |
| `fig_exp2.png` | Experiment 2 — 260 poets + 114 surahs (separate) |
| `fig_exp7.png` | Experiment 7 — 35 well-documented poets + whole Qur'an |
| `fig_exp8.png`, `fig_exp8_outliers.png` | Experiment 8 — 35 well-documented poets + 114 surahs (complete separation) |
| `fig_exp9.png` | Experiment 9 — 35 poets + 60 hizb |
| `fig_exp10.png` | Experiment 10 — 35 poets + 30 hizb-pairs |
| `fig_exp11_12.png` | Experiments 11-12 — random surah / 10-surah-bundle trials |
| `fig_exp14.png` | Experiment 14 — synthetic-text control |
| `fig_exp15.png`, `fig_exp16.png`, `fig_exp17.png` | Experiments 15-17 — Qur'anic units clustered alone (hizb, hizb-pairs, surahs) |
| `fig_exp18_23_24.png` | Experiments 18, 23, 24 — direct-similarity comparisons (AllPoets/35Vector/19Vector vs. whole Qur'an) |
| `fig_exp22.png` | Experiment 22 — Qur'an vs. 9 individual (unmerged) poets |
| `fig_exp25.png`, `fig_exp27.png` | Experiments 25 & 27 — surahs / hizb vs. `35Vector` |

Regenerating any of these from scratch: run the matching notebook in
`notebooks/01_main_clustering_study/` — each notebook's own "Figures"
section saves its plot(s) to `output/figures/` under the filename it
prints.

Notebooks in this project describe additional figures beyond the 17 above
(e.g. Experiment 8 follow-up's clearer cluster-membership charts,
Experiments 9-12's UMAP scatters) — those exist
only as code inside their notebooks in this pass; their rendered PNGs were
not separately delivered as standalone files, so re-running the notebook
is the only way to produce them right now.
