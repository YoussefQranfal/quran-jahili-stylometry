# `output/reports/`

Each `.txt` file here is the text that its corresponding notebook printed
to its own "Report written to `output/reports/...`" cell (reconstructed
from that cell's own stream output, saved in the notebook file). Two files
— `experiments_13_30_report.txt` and `structural_analysis_report.txt` —
have had one section each removed (a letter-counting check belonging to a
separate, out-of-scope sub-project); everything else in every file here is
unedited. Re-running the matching notebook end to end regenerates the
full, unedited version at the same path (`08_experiments13_30_merged.ipynb`
itself has also had that section removed from its source, so re-running it
now reproduces this trimmed report directly).

| Report file | Produced by |
|---|---|
| `full_analysis_report.txt` | `notebooks/00_reproduction_of_source_study/01_reproduce_paper.ipynb` |
| `experiment7_report.txt` | `notebooks/01_main_clustering_study/04_experiment07_large_poets_only.ipynb` |
| `experiment8_report.txt` | `notebooks/01_main_clustering_study/05_experiment08_surahs_separate_well_documented.ipynb` |
| `experiment8_followup_report.txt` | `notebooks/01_main_clustering_study/06_experiment08_followup_cluster_membership.ipynb` |
| `experiments_9_12_report.txt` | `notebooks/01_main_clustering_study/07_experiments09_12_hizb_and_random_sampling.ipynb` |
| `experiments_13_30_report.txt` | `notebooks/01_main_clustering_study/08_experiments13_30_merged.ipynb` |
| `combined_experiments_report.txt` | `notebooks/01_main_clustering_study/01_experiments01_02_combined_clustering.ipynb` |
| `extended_analysis_report.txt` | `notebooks/01_main_clustering_study/02_experiments01_02_followups_extended_analysis.ipynb` |
| `quran_analysis_report.txt` | `notebooks/01_main_clustering_study/03_quran_alone_clustering_meccan_medinan_exploratory.ipynb` |
| `structural_analysis_report.txt` | Produced by a notebook that belongs to this project's separate letter-counting sub-project (not included in this repository) — kept here anyway because its remaining sections (lexical richness, word-length distribution, verse-ending/fawasil consistency) are general Qur'an-vs-poetry text-structure comparisons unrelated to that sub-project; the one section that wasn't (a letter-count check) has been removed from this file. |

This repository omits several report files that exist elsewhere in the
project's history (`experiment31_and_membership_report.txt`,
`experiment32_report.txt`, `full_battery_report.txt`, and the `.json`/`.txt`
outputs some unexecuted notebooks describe writing) because they either
belong entirely to that separate sub-project, or their source notebook was
never executed to completion — see `docs/KNOWN_ISSUES.md` section 4.
