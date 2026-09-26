# `output/tables/`

No raw CSV/table files from the notebooks' own runs were available as
standalone files when this repository was assembled — every table quoted
in the reports and the root `README.md` was read directly out
of a notebook's own printed cell output (a `pandas` DataFrame's `repr()`,
in every case), not from a saved CSV. This directory is intentionally
empty of data files for that reason: nothing here would be more than a
retyped copy of numbers already visible in the notebooks themselves, and
retyping risks a transcription error the notebook's own output doesn't
have.

To regenerate the actual CSV files, run the corresponding notebook — each
one saves its own tables under paths it prints, e.g.:

| Notebook | Tables it writes |
|---|---|
| `00_reproduction_of_source_study/01_reproduce_paper.ipynb` | `poet_summary.csv`, plus in-notebook cluster/baseline/stability tables |
| `01_main_clustering_study/01_experiments01_02_combined_clustering.ipynb` | `experiment1_clusters.csv`, `experiment2_clusters.csv` |
| `01_main_clustering_study/02_experiments01_02_followups_extended_analysis.ipynb` | `full_cluster_membership.csv`, `question4_poets_without_quran.csv`, `experiment5_clusters_excl19.csv`, `experiment6_clusters_sizematched.csv` |
| `01_main_clustering_study/04_experiment07_large_poets_only.ipynb` | `experiment7_large_poets_only.csv`, `poet_pool_comparison.csv` |
| `01_main_clustering_study/05_experiment08_surahs_separate_well_documented.ipynb` | `experiment8_clusters.csv`, `experiment8_comparison.csv` |
| `01_main_clustering_study/07_experiments09_12_hizb_and_random_sampling.ipynb` | `experiment9_clusters.csv`, `experiment10_clusters.csv`, `experiment11_results.csv`, `experiment12_results.csv` |
| `01_main_clustering_study/08_experiments13_30_merged.ipynb` | full CSV/figure set for Experiments 14-30 |

All of the underlying numbers these CSVs would contain are already quoted,
verbatim, in `output/reports/*.txt` and the root `README.md`'s results
tables — nothing is missing, it just isn't duplicated here as a second,
hand-copied file.
