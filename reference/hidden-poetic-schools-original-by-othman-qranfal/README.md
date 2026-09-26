# Hidden Stylistic Schools of Pre-Islamic Poetry

A computational exploration of stylistic relationships among pre-Islamic Arabic poets using sentence embeddings, unsupervised clustering, and network analysis.

## Paper

**"Hidden Stylistic Schools of Pre-Islamic Poetry: A Computational Analysis Using Sentence Embeddings"**
Salem Othman & Youssef Qranfal — Wentworth Institute of Technology
*Submitted to IEEE ICAD 2026*

## Overview

This repository provides full reproducibility for our computational study of pre-Islamic (Jāhilī) Arabic poetry. We embed verses using an Arabic Sentence-BERT model, evaluate multiple pooling strategies, discover latent poetic schools via UMAP + HDBSCAN, and validate findings with baselines and stability analysis.

## Key Findings

- **12 latent stylistic clusters** discovered among 260 pre-Islamic poets (2,328 poems)
- **Arabic-SBERT outperforms multilingual models** — achieves lowest mean similarity (0.800) and highest discrimination among 4 tested embedding models
- **Hierarchical embedding strategy** (verse → poem → poet) captures distinct stylistic information compared to naive concatenation (cross-correlation 0.68–0.75)
- **Validated through** randomization testing (p < 0.01), parameter stability analysis (64 configurations), and comparison against TF-IDF, character n-gram, and NMF baselines
- **Productivity ≠ influence** — a poet's stylistic centrality is independent of their output volume (R² = 0.027)

## Project Structure

```
hidden-poetic-schools/
├── main.py                     # Entry point — runs full pipeline
├── config.py                   # All hyperparameters and paths
├── compare_models.py           # 4-model embedding comparison
├── requirements.txt            # Python dependencies
├── src/
│   ├── data_loader.py          # Corpus loading, cleaning, stats
│   ├── embeddings.py           # SBERT embedding + 4 pooling strategies
│   ├── similarity.py           # Pairwise similarity, network centrality
│   ├── clustering.py           # UMAP + HDBSCAN, K-means comparison
│   ├── baselines.py            # TF-IDF, char n-grams, topic model baselines
│   ├── validation.py           # Stability, randomization, metadata enrichment
│   ├── visualization.py        # Publication-quality figures (seaborn/matplotlib)
│   └── utils.py                # Arabic font setup, text normalization, helpers
├── data/
│   └── poems.db                # SQLite corpus (not included — see Data Access)
├── output/
│   ├── figures/                # Generated figures (.png, 300 DPI)
│   ├── tables/                 # CSV exports, cluster assignments
│   └── reports/                # Text reports, findings summaries
└── tests/
    └── test_pipeline.py        # Sanity checks
```

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/SalemOthman/hidden-poetic-schools.git
cd hidden-poetic-schools
pip install -r requirements.txt

# 2. Place poems.db in data/ directory

# 3. Run full pipeline
python main.py

# 4. Run individual steps
python main.py --step embeddings
python main.py --step baselines
python main.py --step validation
python main.py --step figures
```

## Data Access

The corpus was compiled from [Aldiwan.net](https://www.aldiwan.net), a publicly accessible repository of Arabic poetry. The SQLite database contains 2,328 cleaned poems from 260 pre-Islamic poets (~177,000 tokens). Due to copyright considerations, the database is not included in this repository. To reproduce the dataset, run the scraper (see `src/data_loader.py` for schema details).

## Key Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| SBERT Model | `akhooli/Arabic-SBERT-100K` | Arabic sentence transformer |
| Embedding Dim | 768 | Fixed-dimensional verse vectors |
| UMAP n_neighbors | 15 | Local neighborhood size |
| UMAP min_dist | 0.1 | Minimum distance in projection |
| HDBSCAN min_cluster_size | 5 | Minimum cluster members |
| Random Seed | 42 | Reproducibility seed |

## Results Summary

### Model Comparison

| Model | Dim | Sim μ | Sim σ | Clusters | Silhouette |
|-------|-----|-------|-------|----------|------------|
| **Arabic-SBERT-100K** | **768** | **0.800** | **0.236** | **10** | **0.471** |
| Multilingual-MiniLM | 384 | 0.870 | 0.087 | 7 | 0.378 |
| Multilingual-mpnet | 768 | 0.907 | 0.063 | 8 | 0.254 |
| Multilingual-E5-large | 1024 | 0.973 | 0.017 | 3 | 0.605 |

### Baseline Comparison

| Method | Clusters | Silhouette |
|--------|----------|------------|
| **SBERT (proposed)** | **12** | **0.469** |
| TF-IDF | 3 | 0.442 |
| Char n-grams | 3 | 0.704 |
| NMF Topics | 21 | 0.817 |

### Validation

- **Randomization test:** 12 actual clusters vs. 2–3 shuffled (p < 0.01)
- **Parameter stability:** ARI = 0.500 ± 0.249 across 64 configurations
- **Clustering metrics:** Silhouette = 0.469, Calinski-Harabasz = 478.9, Davies-Bouldin = 0.647

## Figures

| Figure | Description |
|--------|-------------|
| Fig. 1 | Embedding strategy correlation comparison |
| Fig. 2 | Model comparison (6-panel) |
| Fig. 3 | Baseline comparison (SBERT vs alternatives) |
| Fig. 4 | Poet similarity heatmap (top 20) |
| Fig. 5 | Hidden poetic schools — UMAP scatter (main result) |
| Fig. 6 | Randomization test (actual vs shuffled) |
| Fig. 7 | Productivity vs stylistic influence |

## Citation

```bibtex
@inproceedings{othman2026hidden,
  title={Hidden Stylistic Schools of Pre-Islamic Poetry: A Computational Analysis Using Sentence Embeddings},
  author={Othman, Salem and Qranfal, Youssef},
  booktitle={IEEE International Conference on Artificial Intelligence and Development (ICAD)},
  year={2026}
}
```

## License

MIT License
