#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Compares 4 sentence-embedding models on this corpus: Arabic-SBERT-100K vs.
3 multilingual models. For each, computes the poet similarity matrix (using
the proposed poem_verse_avg strategy), clusters with UMAP+HDBSCAN, and
reports similarity mean/std and clustering quality — this reproduces the
"Model Comparison" table in the README/paper.

Run standalone:
    python compare_models.py
"""
import json

import pandas as pd

import config as cfg
from src.utils import setup_logging, set_seed
from src.data_loader import load_corpus
from src.embeddings import load_model, embed_poem_verse_average
from src.similarity import compute_similarity_matrix, similarity_statistics
from src.clustering import full_clustering_pipeline

logger = setup_logging()


def compare_models():
    set_seed(cfg.RANDOM_SEED)
    df, poems_by_poet = load_corpus(cfg.DB_PATH, cfg.MIN_POEMS_PER_POET, cfg.MIN_VERSES_PER_POEM)

    rows = []
    for label, model_name in cfg.COMPARISON_MODELS.items():
        logger.info("Evaluating model: %s (%s)", label, model_name)
        try:
            model = load_model(model_name)
        except Exception as e:
            logger.warning("Could not load %s: %s — skipping", model_name, e)
            continue

        poet_embeddings = embed_poem_verse_average(model, poems_by_poet)
        poet_names, sim_matrix = compute_similarity_matrix(poet_embeddings)
        sim_stats = similarity_statistics(poet_names, sim_matrix)

        cluster_results = full_clustering_pipeline(poet_embeddings, sim_matrix, poet_names, poems_by_poet, cfg)
        metrics = cluster_results["hdbscan_metrics"]

        rows.append({
            "model": label,
            "dim": model.get_sentence_embedding_dimension(),
            "sim_mean": round(sim_stats["mean"], 3),
            "sim_std": round(sim_stats["std"], 3),
            "n_clusters": metrics["n_clusters"],
            "silhouette": round(metrics["silhouette"], 3) if metrics["silhouette"] else None,
        })

        # free GPU/CPU memory between models
        del model
        import gc
        gc.collect()

    comparison_df = pd.DataFrame(rows)
    comparison_df.to_csv(cfg.TABLES_DIR / "model_comparison.csv", index=False)
    with open(cfg.REPORTS_DIR / "model_comparison.json", "w") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)

    logger.info("\n%s", comparison_df.to_string(index=False))
    logger.info("Saved to %s / model_comparison.csv", cfg.TABLES_DIR)
    return comparison_df


if __name__ == "__main__":
    compare_models()
