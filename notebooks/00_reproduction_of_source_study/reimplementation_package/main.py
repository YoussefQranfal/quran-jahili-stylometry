#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Hidden Stylistic Schools of Pre-Islamic Poetry — our reimplementation
=======================================================================
Full analysis pipeline. Run with:
    python main.py                 # full pipeline
    python main.py --step X        # single step

Steps: data, embeddings, similarity, clustering, baselines, validation, figures, report

Before running: `python src/scraper.py` to build data/poems.db (needs
normal internet access — run that part on your own machine, not a sandbox).
"""
import argparse
import json
from datetime import datetime

import pandas as pd

import config as cfg
from src.utils import setup_logging, set_seed, setup_plot_style
from src.data_loader import load_corpus, corpus_statistics
from src.embeddings import load_model, compute_all_strategies
from src.similarity import (
    compute_similarity_matrix, similarity_statistics,
    compare_strategy_matrices, compute_centrality, nearest_neighbors,
)
from src.clustering import full_clustering_pipeline
from src.baselines import run_all_baselines
from src.validation import full_validation_pipeline
from src import visualization as viz

logger = setup_logging()


def step_data():
    logger.info("=" * 60)
    logger.info("STEP 1: DATA LOADING AND CHARACTERIZATION")
    logger.info("=" * 60)
    df, poems_by_poet = load_corpus(cfg.DB_PATH, cfg.MIN_POEMS_PER_POET, cfg.MIN_VERSES_PER_POEM)
    stats, poet_summary = corpus_statistics(df, poems_by_poet)
    poet_summary.to_csv(cfg.TABLES_DIR / "poet_summary.csv", index=False)
    with open(cfg.REPORTS_DIR / "corpus_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    logger.info("Corpus: %d poets, %d poems, %d verses", stats["n_poets"], stats["n_poems"], stats["n_verses"])
    return df, poems_by_poet, stats, poet_summary


def step_embeddings(poems_by_poet):
    logger.info("=" * 60)
    logger.info("STEP 2: EMBEDDING COMPUTATION (4 STRATEGIES)")
    logger.info("=" * 60)
    model = load_model(cfg.SBERT_MODEL_NAME)
    strategies = compute_all_strategies(model, poems_by_poet)
    logger.info("Embedding summary: %s", {k: len(v) for k, v in strategies.items()})
    return model, strategies


def step_similarity(strategies, poems_by_poet):
    logger.info("=" * 60)
    logger.info("STEP 3: SIMILARITY ANALYSIS")
    logger.info("=" * 60)
    corr_df = compare_strategy_matrices(strategies)
    corr_df.to_csv(cfg.TABLES_DIR / "strategy_correlation_matrix.csv", index=False)

    proposed = strategies["poem_verse_avg"]
    poet_names, sim_matrix = compute_similarity_matrix(proposed)

    sim_stats = similarity_statistics(poet_names, sim_matrix)
    with open(cfg.REPORTS_DIR / "similarity_statistics.json", "w") as f:
        json.dump(sim_stats, f, indent=2)

    centrality_df = compute_centrality(poet_names, sim_matrix)
    centrality_df.to_csv(cfg.TABLES_DIR / "poet_centrality.csv", index=False)

    nn = nearest_neighbors(poet_names, sim_matrix, top_k=5)
    nn_rows = [
        {"poet": poet, "rank": rank, "neighbor": neighbor, "similarity": sim}
        for poet, neighbors in nn.items()
        for rank, (neighbor, sim) in enumerate(neighbors, 1)
    ]
    pd.DataFrame(nn_rows).to_csv(cfg.TABLES_DIR / "nearest_neighbors.csv", index=False)

    return poet_names, sim_matrix, sim_stats, corr_df, centrality_df


def step_clustering(strategies, poet_names, sim_matrix, poems_by_poet):
    logger.info("=" * 60)
    logger.info("STEP 4: CLUSTERING (UMAP + HDBSCAN)")
    logger.info("=" * 60)
    proposed = strategies["poem_verse_avg"]
    results = full_clustering_pipeline(proposed, sim_matrix, poet_names, poems_by_poet, cfg)

    cluster_df = pd.DataFrame({
        "poet_name": results["poet_names"],
        "cluster": results["hdbscan_labels"],
        "umap_x": results["umap_2d"][:, 0],
        "umap_y": results["umap_2d"][:, 1],
    })
    cluster_df.to_csv(cfg.TABLES_DIR / "cluster_assignments.csv", index=False)

    profile_rows = [{
        "cluster": p["cluster_id"], "label": p["cluster_label"], "size": p["size"],
        "within_similarity": p["within_similarity"],
        "mean_productivity": p["mean_productivity"],
        "median_productivity": p["median_productivity"],
        "total_poems": p["total_poems"], "mean_centrality": p["mean_centrality"],
        "top_poets": "; ".join(f"{n} ({c})" for n, c in p["top_poets"]),
    } for p in results["cluster_profiles"]]
    pd.DataFrame(profile_rows).to_csv(cfg.TABLES_DIR / "cluster_profiles.csv", index=False)

    with open(cfg.REPORTS_DIR / "clustering_metrics.json", "w") as f:
        json.dump(results["hdbscan_metrics"], f, indent=2)

    logger.info("HDBSCAN found %d clusters (silhouette=%s)",
                results["hdbscan_metrics"]["n_clusters"], results["hdbscan_metrics"]["silhouette"])
    return results


def step_baselines(poems_by_poet):
    logger.info("=" * 60)
    logger.info("STEP 5: BASELINE COMPARISONS")
    logger.info("=" * 60)
    baselines, comparison_df = run_all_baselines(poems_by_poet, cfg)
    comparison_df.to_csv(cfg.TABLES_DIR / "baseline_comparison.csv", index=False)
    return baselines, comparison_df


def step_validation(cluster_results, poet_names, poems_by_poet, model):
    logger.info("=" * 60)
    logger.info("STEP 6: VALIDATION")
    logger.info("=" * 60)
    validation = full_validation_pipeline(
        cluster_results["emb_matrix"], cluster_results["hdbscan_labels"],
        poet_names, poems_by_poet, model, cfg,
    )
    validation["stability_df"].to_csv(cfg.TABLES_DIR / "stability_analysis.csv", index=False)
    with open(cfg.REPORTS_DIR / "stability_summary.json", "w") as f:
        json.dump(validation["stability_summary"], f, indent=2)
    with open(cfg.REPORTS_DIR / "randomization_summary.json", "w") as f:
        json.dump(validation["randomization_summary"], f, indent=2)
    validation["enrichment_df"].to_csv(cfg.TABLES_DIR / "metadata_enrichment.csv")
    with open(cfg.REPORTS_DIR / "metadata_report.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(validation["metadata_report"]) + "\n")
    return validation


def step_figures(poet_names, sim_matrix, poems_by_poet, corr_df, cluster_results, baselines, validation):
    logger.info("=" * 60)
    logger.info("STEP 7: GENERATING FIGURES")
    logger.info("=" * 60)
    fig_dir = cfg.FIGURES_DIR

    viz.fig_similarity_heatmap(poet_names, sim_matrix, cfg.TOP_N_HEATMAP,
                                fig_dir / "poet_similarity_heatmap_top20.png")
    viz.fig_productivity_vs_influence(poet_names, sim_matrix, poems_by_poet,
                                       fig_dir / "productivity_vs_influence.png")
    viz.fig_umap_hdbscan(cluster_results["umap_2d"], cluster_results["hdbscan_labels"], poet_names,
                          fig_dir / "hidden_poetic_schools_umap.png")
    viz.fig_strategy_comparison(corr_df, fig_dir / "strategy_correlation.png")

    if validation and "stability_df" in validation:
        viz.fig_stability_heatmap(validation["stability_df"], fig_dir / "stability_heatmap.png")
    if validation and "randomization_results" in validation:
        viz.fig_randomization_comparison(
            cluster_results["hdbscan_metrics"]["n_clusters"],
            cluster_results["hdbscan_metrics"]["silhouette"],
            validation["randomization_results"], fig_dir / "randomization_test.png",
        )
    if baselines:
        viz.fig_baseline_comparison(cluster_results["hdbscan_metrics"], baselines,
                                     fig_dir / "baseline_comparison.png")
    if validation and "enrichment_df" in validation:
        viz.fig_metadata_enrichment(validation["enrichment_df"], fig_dir / "metadata_enrichment.png")

    logger.info("Figures saved to %s", fig_dir)


def step_report(stats, sim_stats, cluster_results, baselines, validation, comparison_df):
    logger.info("=" * 60)
    logger.info("STEP 8: GENERATING REPORT")
    logger.info("=" * 60)
    report_path = cfg.REPORTS_DIR / "full_analysis_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\nHIDDEN STYLISTIC SCHOOLS OF PRE-ISLAMIC POETRY\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n" + "=" * 70 + "\n\n")

        f.write("1. DATASET\n" + "-" * 40 + "\n")
        for k, v in stats.items():
            f.write(f"  {k}: {v}\n")

        f.write("\n2. SIMILARITY STATISTICS\n" + "-" * 40 + "\n")
        for k, v in sim_stats.items():
            f.write(f"  {k}: {v}\n")

        f.write("\n3. CLUSTERING RESULTS\n" + "-" * 40 + "\n")
        for k, v in cluster_results["hdbscan_metrics"].items():
            f.write(f"  {k}: {v}\n")

        f.write("\n4. CLUSTER PROFILES\n" + "-" * 40 + "\n")
        for p in cluster_results["cluster_profiles"]:
            f.write(f"\n  {p['cluster_label']} (n={p['size']})\n")
            f.write(f"    Within similarity: {p['within_similarity']:.4f}\n" if p['within_similarity'] == p['within_similarity'] else "    Within similarity: n/a\n")
            f.write(f"    Mean productivity: {p['mean_productivity']:.1f} poems/poet\n")
            f.write(f"    Mean centrality: {p['mean_centrality']:.4f}\n")
            f.write(f"    Top poets: {', '.join(f'{n} ({c})' for n, c in p['top_poets'])}\n")

        if baselines:
            f.write("\n5. BASELINE COMPARISON\n" + "-" * 40 + "\n")
            f.write(comparison_df.to_string(index=False) + "\n")

        if validation:
            f.write("\n6. VALIDATION\n" + "-" * 40 + "\n")
            s = validation["stability_summary"]
            f.write(f"\n  Stability: {s['n_runs']} runs, ARI={s['mean_ari']:.3f}+/-{s['std_ari']:.3f}, "
                    f"NMI={s['mean_nmi']:.3f}+/-{s['std_nmi']:.3f}\n")
            r = validation["randomization_summary"]
            f.write(f"\n  Randomization: {r['n_shuffles']} shuffles, "
                    f"mean_clusters={r['mean_clusters_shuffled']:.1f}+/-{r['std_clusters_shuffled']:.1f}\n")
            f.write("\n  Metadata Enrichment:\n")
            for line in validation["metadata_report"]:
                f.write(f"    {line}\n")

    logger.info("Report saved: %s", report_path)


def main():
    parser = argparse.ArgumentParser(description="Hidden Stylistic Schools of Pre-Islamic Poetry")
    parser.add_argument("--step", type=str, default="all",
                         choices=["all", "data", "embeddings", "similarity", "clustering",
                                  "baselines", "validation", "figures", "report"])
    args = parser.parse_args()

    set_seed(cfg.RANDOM_SEED)
    setup_plot_style()

    logger.info("=" * 70)
    logger.info("HIDDEN STYLISTIC SCHOOLS OF PRE-ISLAMIC POETRY")
    logger.info("=" * 70)

    df, poems_by_poet, stats, poet_summary = step_data()
    model, strategies = step_embeddings(poems_by_poet)
    poet_names, sim_matrix, sim_stats, corr_df, centrality_df = step_similarity(strategies, poems_by_poet)
    cluster_results = step_clustering(strategies, poet_names, sim_matrix, poems_by_poet)
    baselines, comparison_df = step_baselines(poems_by_poet)
    validation = step_validation(cluster_results, poet_names, poems_by_poet, model)
    step_figures(poet_names, sim_matrix, poems_by_poet, corr_df, cluster_results, baselines, validation)
    step_report(stats, sim_stats, cluster_results, baselines, validation, comparison_df)

    logger.info("=" * 70)
    logger.info("PIPELINE COMPLETE — figures: %s, tables: %s, reports: %s",
                cfg.FIGURES_DIR, cfg.TABLES_DIR, cfg.REPORTS_DIR)


if __name__ == "__main__":
    main()
