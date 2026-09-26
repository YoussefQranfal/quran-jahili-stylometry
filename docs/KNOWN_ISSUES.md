# Known Issues, Bugs, and Renumbering History

This project was built incrementally, notebook by notebook, over many
sessions. This document records every bug, dead end, and renumbering
collision found while assembling the repository, so nobody re-discovers
them by surprise. Nothing here was silently fixed in-place — buggy
notebooks are kept, clearly labeled, alongside their corrected reruns, per
the project's "keep everything, label clearly" policy.

**Scope note:** this repository covers the main clustering study only. A
separate, unrelated sub-project was developed alongside it but is
intentionally excluded here entirely — no code, data, figures, or text
from it are included — since it uses different data, different
statistical logic, and answers an unrelated question. One early section of
a clustering-study notebook originally referenced that sub-project; it has
been removed from both the notebook's source and its saved output (see
issue 2 below), so no trace of it remains in this repository.

## 1. `sqlite3.connect()` silently creates an empty database

**Symptom:** `sqlite3.OperationalError: no such table: poems`, even though
`poems.db` "should" be right there.

**Root cause:** Python's `sqlite3.connect(path)` does **not** raise an
error if `path` doesn't exist at the location you pass — it silently
creates a brand-new, empty SQLite file with no tables at that path. If a
notebook is run from a different working directory than the one containing
the real `poems.db` (3,457,024 bytes, one table `poems`, 2,361 rows), you
don't get a missing-file error — you get a *new, empty* database sitting
right there, and every query against it fails with "no such table."

**Where this actually happened in this project's history:**
`notebooks/01_main_clustering_study/10_experiments37_40_quarters_eighths_vs_vectors_DBBUG_halted.ipynb`
hits exactly this error on its very first query (Cell 3):

```
DatabaseError: Execution failed on sql 'SELECT poet_name, poem_title, poem_text, poem_type, poem_meter, verses_count FROM poems WHERE poet_name IS NOT NULL AND poem_text IS NOT NULL AND LENGTH(poem_text) >= 50': no such table: poems
```

Execution of that notebook halts there; Experiments 37-40 (240 quarters /
480 eighths vs. `35Vector`/`19Vector`) as described in that notebook's own
markdown were never actually completed in the copy available for this
repository. Several other notebooks' markdown cells also carry an explicit
"before you start" warning about this exact failure mode ("put `poems.db`
in the same folder as this notebook — the real one, several hundred KB or
more, not an empty auto-created stand-in").

**Fix / practice:** always run notebooks from a directory where the real
`data/poems.db` is reachable (copy or symlink it next to the notebook
before running), and if you ever see "no such table: poems," check the
file size of whatever `poems.db` sits next to the notebook — if it's near
0 bytes, delete it, it's a stray empty file created by this exact
behavior, not the real corpus.

## 2. A section was removed from `08_experiments13_30_merged.ipynb`

That notebook originally opened with a short early section belonging to
the project's separate letter-counting sub-project (out of scope for this
repository — see the scope note above), before its real Experiments 15-30
clustering content. That section, and the corresponding block in its
final-report cell, have been deleted from both the notebook's source and
its saved output in this repository (and from `output/reports/experiments_13_30_report.txt`,
which that cell writes) — the notebook now runs Experiments 14-30 only,
with no gap left by the removal (nothing later in the notebook depended on
the removed section's variables; this was verified before removing it).

For the record, since it's part of this project's history: that removed
section was itself a buggy first pass at an independent count of
"Allah"/"Ar-Rahman"/"Ar-Raheem" occurrences in the Qur'anic text, which
came back as **0** for all three because the normalization function used
at the time didn't merge the ALEF WASLA character (`U+0671`, `ٱ`) — the
form the Uthmani script actually uses to open those words — into plain
alef. This bug had no effect on Experiments 15-30, which are unrelated
clustering comparisons.

## 3. Experiment renumbering history

Experiments in this project are numbered sequentially in the order they
were conceived, not the order they ended up presented in the root
`README.md`'s summary table. Several numbers
collided with earlier ones as the project grew and were renumbered by the
person building each new notebook, with the collision explicitly called
out in that notebook's own intro markdown:

- "Experiments 30-33" (Umayya ibn Abi al-Salt vs. the Qur'an) collided
  with the separate letter-counting sub-project's own experiment numbers
  → renumbered to **Experiments 33-36**.
- "Experiments 29-32" (quarters/eighths vs. `35Vector`/`19Vector`) collided
  twice over (once with the hizb-pair experiments, once with the
  separate sub-project's numbers) → renumbered to **Experiments 37-40**.
- Within the merged `experiments_13_to_30.ipynb`/`experiments_22_to_30.ipynb`
  build history, "Experiment 23" and "Experiment 24" were each
  independently used twice for different comparisons before being sorted
  into their final, non-colliding positions.

This is a normal, already-understood quirk of building ~50 notebooks
incrementally over time, not a data-integrity bug — but it does mean an
experiment number alone is not a stable identifier across every draft
notebook that ever existed; the **notebook file itself**, read alongside
the experiment-index table in the root `README.md`, is the source of
truth for what a given "Experiment N" actually compared in its final,
canonical form. Note that experiment numbers 13 and 31-32 do not appear
anywhere in this repository's index at all — they belonged entirely to
the separate letter-counting sub-project and are out of scope here (see
the scope note above and `notebooks/01_main_clustering_study/README.md`).

Two early, superseded, whole-notebook builds of "Experiments 13-30" exist
in the project's file history — `experiments_13_to_21.ipynb` and
`experiments_22_to_30.ipynb` (both never independently executed; their
markdown headers show experiment numbers that were still being sorted
out). Their content was superseded by, and folded into, the single merged
`08_experiments13_30_merged.ipynb`, which *was*
executed successfully end-to-end and is the canonical source for
Experiments 14-30 (see issue 2 above regarding that notebook's unrelated
early section). The two intermediate builds are not separately included
in this repository since they add no content beyond what the merged
notebook already contains and were never run.

## 4. Notebooks with no executed output in this pass

The following notebooks describe fully specified experiments but contain
**no executed cell outputs** in any copy available when this repository
was assembled (checked directly: every code cell's `execution_count` is
`null` and `outputs` is empty). Their design and intended method are
included in this repository for completeness and traceability, but **none
of their numeric claims should be treated as verified results** until
someone runs them:

- `notebooks/01_main_clustering_study/11_experiments41_44_quarters_eighths_vs_individual_poets_UNEXECUTED.ipynb`
  (Experiments 41-44)
- `notebooks/01_main_clustering_study/12_experiments45_46_quarters_eighths_alone_UNEXECUTED.ipynb`
  (Experiments 45-46)
- `notebooks/01_main_clustering_study/13_experiments47_51_ayah_level_clustering_UNEXECUTED.ipynb`
  (Experiments 47-51)
- `notebooks/01_main_clustering_study/09_experiments33_36_umayya_scraped_UNEXECUTED.ipynb`
  (Experiments 33-36, scraped-Umayya version)

A near-duplicate, byte-identical copy of the last file above also exists
in the project's raw file history under the name
`experiments_33_to_36_umayya_scraped.ipynb` (no trailing `_2`) — the two
are the same unexecuted notebook saved under two filenames at different
points; only one copy is kept in this repository.

The non-scraped attempt at Experiments 33-36
(`notebooks/01_main_clustering_study/superseded_or_unexecuted/experiments33_36_umayya_NOTFOUND_halted.ipynb`)
did execute, but halted itself deliberately at Cell 8 via `SystemExit`
after searching the 260-poet corpus for "Umayya ibn Abi al-Salt" (and
common spelling variants) and finding **zero exact matches** — its own
printed output states this explicitly. **No experiment in this project
currently has a verified numeric result for Umayya ibn Abi al-Salt vs. the
Qur'an at any granularity** — the poet-not-found run stopped before
reaching the comparison, and the scraped-fresh version that was meant to
route around that (by scraping his poems directly instead of relying on
the 260-poet database) was never executed.

## 5. Two buggy/UMAP-error notebook runs, superseded by working reruns

- `notebooks/01_main_clustering_study/superseded_or_unexecuted/experiment07_large_poets_BUGGY_umap_error.ipynb`
  fails at its clustering step with
  `TypeError: Cannot use scipy.linalg.eigh for sparse A with k >= N. Use scipy.linalg.eigh(A.toarray()) or reduce k.`
  — a UMAP/spectral-initialization numerical error, unrelated to the bug
  above. The working rerun,
  `notebooks/01_main_clustering_study/04_experiment07_large_poets_only.ipynb`,
  completes successfully and is the canonical source for Experiment 7's
  results (35 well-documented poets + whole Qur'an: 9 poets share the
  Qur'an's cluster, similarity range 0.965-0.967).

- `notebooks/00_reproduction_of_source_study/superseded_or_failed_runs/hidden_poetic_schools_standalone_FAILED_no_internet.ipynb`
  is a "one notebook, run top to bottom" attempt at reproducing the source
  study end-to-end, including its own scraper. In the environment where it
  was actually run, the scraper found **0 poets** (no internet access to
  aldiwan.net from that environment), and every downstream cell either
  silently operates on empty data or fails outright (a `NameError` at the
  clustering step, since embedding never produced any poet vectors). The
  companion file
  `hidden_poetic_schools_unexecuted_template.ipynb` in the same folder is
  an even earlier, never-executed draft of the same idea, written against
  a different (and, per `data/README.md`, incorrect) three-table DB schema
  (`poets`/`poems`/`verses`) than the real `poems.db`'s single `poems`
  table — it was never reconciled with the real schema.

  The actually-working, executed phase-0 reproduction of the source study
  is `notebooks/00_reproduction_of_source_study/01_reproduce_paper.ipynb`,
  which reproduces the real corpus statistics and this project's own
  0.799 mean pairwise similarity result (see the root `README.md`).

## 6. A section was removed from `structural_analysis_report.txt`

`output/reports/structural_analysis_report.txt` was produced by a notebook
that belongs to the separate letter-counting sub-project (not included in
this repository). Its other three sections — lexical richness,
word-length distribution, and verse-ending (fawasil) consistency — are
general Qur'an-vs-poetry text-structure comparisons with nothing to do
with that sub-project's question, so the file is kept, but its one section
that was that sub-project's content (a single letter-count check) has been
deleted from this copy of the file. The equivalent PDF version of this
report is not included in this repository at all, since a PDF's internal
text can't be edited the same way a plain-text file can — see
`docs/reports/README.md`.
