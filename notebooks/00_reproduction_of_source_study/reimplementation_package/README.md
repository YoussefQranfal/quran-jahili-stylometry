# Hidden Stylistic Schools of Pre-Islamic Poetry — Our Reimplementation

A learning-oriented rebuild of the computational-poetics pipeline described in
Othman & Qranfal's paper (WIT, submitted IEEE ICAD 2026): embed pre-Islamic
Arabic poets with Sentence-BERT, cluster them with UMAP+HDBSCAN to find latent
"stylistic schools," and validate the result against baselines, parameter
stability, and a randomization test.

This is **our own from-scratch implementation**, written to understand every
design decision — it is not a copy of the original repo's source code.

## Project Structure

```
hidden-poetic-schools/
├── main.py                 # Entry point — runs the full pipeline
├── config.py                # All hyperparameters and paths
├── compare_models.py        # 4-model embedding comparison
├── requirements.txt
├── src/
│   ├── scraper.py            # Standalone: builds data/poems.db (run locally, needs internet)
│   ├── data_loader.py         # Corpus loading, cleaning, stats
│   ├── embeddings.py           # SBERT embedding + 4 pooling strategies
│   ├── similarity.py            # Pairwise similarity, centrality, nearest neighbors
│   ├── clustering.py             # UMAP + HDBSCAN, K-means comparison
│   ├── baselines.py               # TF-IDF, char n-grams, NMF baselines
│   ├── validation.py               # Stability, randomization, metadata enrichment
│   ├── visualization.py             # Figures (seaborn/matplotlib)
│   └── utils.py                      # Arabic text normalization, logging, seeding
├── data/
│   └── poems.db              # NOT included — build it yourself, see below
├── output/
│   ├── figures/
│   ├── tables/
│   └── reports/
└── tests/
    └── test_pipeline.py       # Synthetic-data sanity checks (no scraping needed)
```

## Setup

```bash
cd hidden-poetic-schools
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Step 1 — Build the corpus (run this locally, with real internet access)

```bash
python src/scraper.py --limit-poets 5   # quick test run first
python src/scraper.py                    # full scrape (hundreds of poets — slow, be patient)
```

This politely scrapes aldiwan.net's pre-Islamic poets category into
`data/poems.db`. It's resumable — if it's interrupted, just rerun it and
already-scraped poets are skipped. Respect the built-in delay between
requests (`SCRAPER_REQUEST_DELAY_SECONDS` in `config.py`) — don't lower it.

## Step 2 — Run tests (optional, no data required)

```bash
pytest tests/ -v
```

## Step 3 — Run the pipeline

```bash
python main.py                       # full pipeline, all 8 steps
python main.py --step embeddings     # or run one step at a time
python main.py --step clustering
python main.py --step baselines
python main.py --step validation
python main.py --step figures
```

## Step 4 — Model comparison (optional, slow — downloads 3 extra models)

```bash
python compare_models.py
```

## What each step produces

| Step | Output |
|---|---|
| `data` | `output/tables/poet_summary.csv`, `output/reports/corpus_stats.json` |
| `embeddings` | in-memory poet embeddings (4 strategies) |
| `similarity` | `poet_centrality.csv`, `nearest_neighbors.csv`, `strategy_correlation_matrix.csv` |
| `clustering` | `cluster_assignments.csv`, `cluster_profiles.csv` |
| `baselines` | `baseline_comparison.csv` |
| `validation` | `stability_analysis.csv`, `stability_summary.json`, `randomization_summary.json` |
| `figures` | 8 PNGs in `output/figures/` |
| `report` | `output/reports/full_analysis_report.txt` |

## Key Parameters (config.py)

| Parameter | Value |
|---|---|
| SBERT model | `akhooli/Arabic-SBERT-100K` |
| UMAP n_neighbors | 15 |
| UMAP min_dist | 0.1 |
| HDBSCAN min_cluster_size | 5 |
| Random seed | 42 |

## Notes on differences from the original repo

- `src/validation.py`'s randomization test uses bootstrap resampling of
  poet-level embeddings as a lightweight proxy; the original paper likely
  does a true verse-level permutation test (shuffle which verses go to
  which poet before re-embedding). If you want to match the paper exactly,
  that's the piece to extend next — happy to help build it out.
- `metadata_enrichment` (tribal affiliation) needs poet bios wired through
  from `data_loader.py` — currently `poems_by_poet` only carries verses.
  Small extension: also return a `poet_bios: dict[str, str]` from
  `load_corpus` and pass it into `full_validation_pipeline`.

## License

MIT (matching the original repo's license — this is a fresh implementation
for learning purposes).
