#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Hidden Stylistic Schools of Pre-Islamic Poetry
===============================================
Full analysis pipeline. Run with:
    python main.py              # Full pipeline
    python main.py --step X     # Single step

Steps: data, embeddings, similarity, clustering, baselines, validation, figures, report
"""
import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import config as cfg
from src.utils import setup_logging, set_seed, setup_plot_style
from src.data_loader import load_corpus, corpus_statistics
from src.embeddings import load_model, compute_all_strategies, embed_poem_verse_average
from src.similarity import (
    compute_similarity_matrix, similarity_statistics,
    compare_strategy_matrices, compute_centrality, nearest_neighbors,
)
from src.clustering import full_clustering_pipeline, cluster_validation_metrics
from src.baselines import run_all_baselines
from src.validation import full_validation_pipeline
from src import visualization as viz

logger = setup_logging()


def step_data():
    """Step 1: Load and characterize dataset."""
    logger.info("=" * 60)
    logger.info("STEP 1: DATA LOADING AND CHARACTERIZATION")
    logger.info("=" * 60)

    df, poems_by_poet = load_corpus(cfg.DB_PATH)
    stats, poet_summary = corpus_statistics(df, poems_by_poet)

    # Save
    poet_summary.to_csv(cfg.TABLES_DIR / "poet_summary.csv", index=False)
    with open(cfg.REPORTS_DIR / "corpus_stats.json", "w") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    logger.info("Dataset stats saved to %s", cfg.TABLES_DIR)
    return df, poems_by_poet, stats, poet_summary


def step_embeddings(poems_by_poet):
    """Step 2: Compute embeddings with all 4 strategies."""
    logger.info("=" * 60)
    logger.info("STEP 2: EMBEDDING COMPUTATION (4 STRATEGIES)")
    logger.info("=" * 60)

    model = load_model(cfg.SBERT_MODEL_NAME)
    strategies = compute_all_strategies(model, poems_by_poet)

    # Save embedding counts per strategy
    summary = {name: len(embs) for name, embs in strategies.items()}
    logger.info("Embedding summary: %s", summary)

    return model, strategies


def step_similarity(strategies, poems_by_poet):
    """Step 3: Similarity analysis and strategy comparison."""
    logger.info("=" * 60)
    logger.info("STEP 3: SIMILARITY ANALYSIS")
    logger.info("=" * 60)

    # Compare strategies (Table I)
    corr_df = compare_strategy_matrices(strategies)
    corr_df.to_csv(cfg.TABLES_DIR / "strategy_correlation_matrix.csv")

    # Use proposed method for main analysis
    proposed = strategies["poem_verse_avg"]
    poet_names, sim_matrix = compute_similarity_matrix(proposed)

    # Similarity statistics (FIX for Reviewer 3: accurate reporting)
    sim_stats = similarity_statistics(poet_names, sim_matrix)
    with open(cfg.REPORTS_DIR / "similarity_statistics.json", "w") as f:
        json.dump(sim_stats, f, indent=2)

    # Centrality
    centrality_df = compute_centrality(poet_names, sim_matrix)
    centrality_df.to_csv(cfg.TABLES_DIR / "poet_centrality.csv", index=False)

    # Nearest neighbors
    nn = nearest_neighbors(poet_names, sim_matrix, top_k=5)
    nn_rows = []
    for poet, neighbors in nn.items():
        for rank, (neighbor, sim) in enumerate(neighbors, 1):
            nn_rows.append({"poet": poet, "rank": rank,
                            "neighbor": neighbor, "similarity": sim})
    pd.DataFrame(nn_rows).to_csv(
        cfg.TABLES_DIR / "nearest_neighbors.csv", index=False
    )

    return poet_names, sim_matrix, sim_stats, corr_df, centrality_df


def step_clustering(strategies, poet_names, sim_matrix, poems_by_poet):
    """Step 4: UMAP + HDBSCAN clustering."""
    logger.info("=" * 60)
    logger.info("STEP 4: CLUSTERING (UMAP + HDBSCAN)")
    logger.info("=" * 60)

    proposed = strategies["poem_verse_avg"]
    results = full_clustering_pipeline(
        proposed, sim_matrix, poet_names, poems_by_poet, cfg
    )

    # Save cluster assignments
    cluster_df = pd.DataFrame({
        "poet_name": results["poet_names"],
        "cluster": results["hdbscan_labels"],
        "umap_x": results["umap_2d"][:, 0],
        "umap_y": results["umap_2d"][:, 1],
    })
    cluster_df.to_csv(cfg.TABLES_DIR / "cluster_assignments.csv", index=False)

    # Save cluster profiles
    profile_rows = []
    for p in results["cluster_profiles"]:
        profile_rows.append({
            "cluster": p["cluster_id"],
            "label": p["cluster_label"],
            "size": p["size"],
            "within_similarity": p["within_similarity"],
            "mean_productivity": p["mean_productivity"],
            "median_productivity": p["median_productivity"],
            "total_poems": p["total_poems"],
            "mean_centrality": p["mean_centrality"],
            "top_poets": "; ".join(["%s (%d)" % (name, cnt)
                                     for name, cnt in p["top_poets"]]),
        })
    pd.DataFrame(profile_rows).to_csv(
        cfg.TABLES_DIR / "cluster_profiles.csv", index=False
    )

    # Save metrics
    with open(cfg.REPORTS_DIR / "clustering_metrics.json", "w") as f:
        json.dump(results["hdbscan_metrics"], f, indent=2)

    logger.info("K-means silhouette (k=2): %.4f", results["kmeans_silhouette"])

    return results


def step_baselines(poems_by_poet):
    """Step 5: Run baseline comparisons (Reviewer 3)."""
    logger.info("=" * 60)
    logger.info("STEP 5: BASELINE COMPARISONS")
    logger.info("=" * 60)

    baselines, comparison_df = run_all_baselines(poems_by_poet, cfg)
    comparison_df.to_csv(cfg.TABLES_DIR / "baseline_comparison.csv", index=False)

    return baselines, comparison_df


def step_validation(cluster_results, poet_names, poems_by_poet, model):
    """Step 6: Stability, randomization, metadata enrichment (Reviewer 3)."""
    logger.info("=" * 60)
    logger.info("STEP 6: VALIDATION")
    logger.info("=" * 60)

    validation = full_validation_pipeline(
        cluster_results["emb_matrix"],
        cluster_results["hdbscan_labels"],
        poet_names, poems_by_poet, model, cfg,
    )

    # Save stability
    validation["stability_df"].to_csv(
        cfg.TABLES_DIR / "stability_analysis.csv", index=False
    )
    with open(cfg.REPORTS_DIR / "stability_summary.json", "w") as f:
        json.dump(validation["stability_summary"], f, indent=2)

    # Save randomization
    with open(cfg.REPORTS_DIR / "randomization_summary.json", "w") as f:
        json.dump(validation["randomization_summary"], f, indent=2)

    # Save metadata
    validation["enrichment_df"].to_csv(
        cfg.TABLES_DIR / "metadata_enrichment.csv"
    )
    with open(cfg.REPORTS_DIR / "metadata_report.txt", "w", encoding="utf-8") as f:
        for line in validation["metadata_report"]:
            f.write(line + "\n")

    return validation


def step_figures(poet_names, sim_matrix, poems_by_poet, corr_df,
                 cluster_results, baselines, validation):
    """Step 7: Generate all publication figures."""
    logger.info("=" * 60)
    logger.info("STEP 7: GENERATING FIGURES")
    logger.info("=" * 60)

    fig_dir = cfg.FIGURES_DIR

    # Fig 1: Similarity heatmap (top 20)
    viz.fig_similarity_heatmap(
        poet_names, sim_matrix, cfg.TOP_N_HEATMAP,
        fig_dir / "poet_similarity_network_heatmap_top_20.png",
    )

    # Fig 2: Productivity vs influence
    viz.fig_productivity_vs_influence(
        poet_names, sim_matrix, poems_by_poet,
        fig_dir / "poet_productivity_vs_stylistic_influence_scatter.png",
    )

    # Fig 3: UMAP + HDBSCAN scatter
    viz.fig_umap_hdbscan(
        cluster_results["umap_2d"],
        cluster_results["hdbscan_labels"],
        poet_names,
        fig_dir / "hidden_poetic_schools_umap_hdbscan_scatter.png",
    )

    # Fig 4: Strategy comparison
    viz.fig_strategy_comparison(
        corr_df,
        fig_dir / "embedding_strategy_correlation_comparison.png",
    )

    # Fig 5: Stability heatmap
    if validation and "stability_df" in validation:
        viz.fig_stability_heatmap(
            validation["stability_df"],
            fig_dir / "cluster_stability_across_parameters.png",
        )

    # Fig 6: Randomization test
    if validation and "randomization_results" in validation:
        actual_n = cluster_results["hdbscan_metrics"]["n_clusters"]
        actual_sil = cluster_results["hdbscan_metrics"]["silhouette"]
        viz.fig_randomization_comparison(
            actual_n, actual_sil,
            validation["randomization_results"],
            fig_dir / "randomization_test_actual_vs_shuffled.png",
        )

    # Fig 7: Baseline comparison
    if baselines:
        viz.fig_baseline_comparison(
            cluster_results["hdbscan_metrics"],
            baselines,
            fig_dir / "baseline_comparison_sbert_vs_alternatives.png",
        )

    # Fig 8: Metadata enrichment
    if validation and "enrichment_df" in validation:
        viz.fig_metadata_enrichment(
            validation["enrichment_df"],
            fig_dir / "cluster_tribal_enrichment_heatmap.png",
        )

    logger.info("All figures saved to %s", fig_dir)


def step_report(stats, sim_stats, cluster_results, baselines,
                validation, comparison_df):
    """Step 8: Generate comprehensive text report."""
    logger.info("=" * 60)
    logger.info("STEP 8: GENERATING REPORT")
    logger.info("=" * 60)

    report_path = cfg.REPORTS_DIR / "full_analysis_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("HIDDEN STYLISTIC SCHOOLS OF PRE-ISLAMIC POETRY\n")
        f.write("Full Analysis Report\n")
        f.write("Generated: %s\n" % datetime.now().strftime("%Y-%m-%d %H:%M"))
        f.write("=" * 70 + "\n\n")

        f.write("1. DATASET\n")
        f.write("-" * 40 + "\n")
        for k, v in stats.items():
            f.write("  %s: %s\n" % (k, v))

        f.write("\n2. SIMILARITY STATISTICS (Proposed Method)\n")
        f.write("-" * 40 + "\n")
        f.write("  NOTE: Computed on upper triangle of %d x %d matrix\n" %
                (stats["n_poets"], stats["n_poets"]))
        f.write("  Total pairs: %d\n" % sim_stats["n_pairs"])
        f.write("  Mean: %.4f\n" % sim_stats["mean"])
        f.write("  Std:  %.4f\n" % sim_stats["std"])
        f.write("  Min:  %.4f\n" % sim_stats["min"])
        f.write("  Max:  %.4f\n" % sim_stats["max"])
        f.write("  Range: [%.4f, %.4f]\n" % (sim_stats["min"], sim_stats["max"]))

        f.write("\n3. CLUSTERING RESULTS\n")
        f.write("-" * 40 + "\n")
        metrics = cluster_results["hdbscan_metrics"]
        for k, v in metrics.items():
            f.write("  %s: %s\n" % (k, v))

        f.write("\n4. CLUSTER PROFILES\n")
        f.write("-" * 40 + "\n")
        for p in cluster_results["cluster_profiles"]:
            f.write("\n  %s (n=%d)\n" % (p["cluster_label"], p["size"]))
            f.write("    Within similarity: %.4f\n" % p["within_similarity"])
            f.write("    Mean productivity: %.1f poems/poet\n" % p["mean_productivity"])
            f.write("    Mean centrality: %.4f\n" % p["mean_centrality"])
            f.write("    Top poets: %s\n" % ", ".join(
                ["%s (%d)" % (n, c) for n, c in p["top_poets"]]))

        if baselines:
            f.write("\n5. BASELINE COMPARISON\n")
            f.write("-" * 40 + "\n")
            f.write(comparison_df.to_string(index=False))
            f.write("\n")

        if validation:
            f.write("\n6. VALIDATION\n")
            f.write("-" * 40 + "\n")
            f.write("\n  Stability:\n")
            s = validation["stability_summary"]
            f.write("    Runs: %d\n" % s["n_runs"])
            f.write("    ARI: %.3f +/- %.3f\n" % (s["mean_ari"], s["std_ari"]))
            f.write("    NMI: %.3f +/- %.3f\n" % (s["mean_nmi"], s["std_nmi"]))
            f.write("    Clusters range: %d - %d\n" %
                    (s["min_clusters"], s["max_clusters"]))

            f.write("\n  Randomization:\n")
            r = validation["randomization_summary"]
            f.write("    Shuffles: %d\n" % r["n_shuffles"])
            f.write("    Shuffled avg clusters: %.1f +/- %.1f\n" %
                    (r["mean_clusters_shuffled"], r["std_clusters_shuffled"]))
            if r.get("mean_silhouette_shuffled"):
                f.write("    Shuffled avg silhouette: %.4f\n" %
                        r["mean_silhouette_shuffled"])

            f.write("\n  Metadata Enrichment:\n")
            for line in validation["metadata_report"]:
                f.write("    %s\n" % line)

    logger.info("Report saved: %s", report_path)


# =========================================================================
# MAIN
# =========================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Hidden Stylistic Schools of Pre-Islamic Poetry"
    )
    parser.add_argument(
        "--step", type=str, default="all",
        choices=["all", "data", "embeddings", "similarity", "clustering",
                 "baselines", "validation", "figures", "report"],
        help="Run a specific step or 'all' (default: all)",
    )
    args = parser.parse_args()

    set_seed(cfg.RANDOM_SEED)
    setup_plot_style()

    logger.info("=" * 70)
    logger.info("HIDDEN STYLISTIC SCHOOLS OF PRE-ISLAMIC POETRY")
    logger.info("=" * 70)
    logger.info("Start time: %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    logger.info("Config: UMAP(nn=%d, md=%.2f), HDBSCAN(mcs=%d)",
                cfg.UMAP_N_NEIGHBORS, cfg.UMAP_MIN_DIST,
                cfg.HDBSCAN_MIN_CLUSTER_SIZE)

    # Run pipeline
    df, poems_by_poet, stats, poet_summary = step_data()
    model, strategies = step_embeddings(poems_by_poet)
    poet_names, sim_matrix, sim_stats, corr_df, centrality_df = \
        step_similarity(strategies, poems_by_poet)
    cluster_results = step_clustering(
        strategies, poet_names, sim_matrix, poems_by_poet
    )
    baselines, comparison_df = step_baselines(poems_by_poet)
    validation = step_validation(
        cluster_results, poet_names, poems_by_poet, model
    )
    step_figures(
        poet_names, sim_matrix, poems_by_poet, corr_df,
        cluster_results, baselines, validation,
    )
    step_report(
        stats, sim_stats, cluster_results, baselines,
        validation, comparison_df,
    )

    logger.info("=" * 70)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 70)
    logger.info("Figures: %s", cfg.FIGURES_DIR)
    logger.info("Tables:  %s", cfg.TABLES_DIR)
    logger.info("Reports: %s", cfg.REPORTS_DIR)

    # List outputs
    for subdir in [cfg.FIGURES_DIR, cfg.TABLES_DIR, cfg.REPORTS_DIR]:
        for f in sorted(subdir.glob("*")):
            logger.info("  %s", f.relative_to(cfg.PROJECT_ROOT))


if __name__ == "__main__":
    main()
