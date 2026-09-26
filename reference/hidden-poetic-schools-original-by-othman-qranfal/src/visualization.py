"""
Publication-quality figure generation.
No titles on plots (added in LaTeX). Descriptive filenames.
"""
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress

from .utils import arabic_display, setup_plot_style

logger = logging.getLogger("poetic_schools")


def fig_similarity_heatmap(poet_names, sim_matrix, top_n, output_path):
    """
    Figure: Poet similarity network heatmap (top N most central poets).
    """
    setup_plot_style()

    # Select top-N by average similarity
    avg_sims = np.mean(sim_matrix, axis=1)
    top_idx = np.argsort(avg_sims)[-top_n:][::-1]

    sub_matrix = sim_matrix[np.ix_(top_idx, top_idx)]
    sub_names = [arabic_display(poet_names[i]) for i in top_idx]

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        sub_matrix,
        xticklabels=sub_names,
        yticklabels=sub_names,
        cmap="YlOrRd",
        vmin=sub_matrix[np.triu_indices(top_n, k=1)].min(),
        vmax=1.0,
        annot=False,
        square=True,
        linewidths=0.5,
        cbar_kws={"label": "Cosine Similarity", "shrink": 0.8},
        ax=ax,
    )
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.tick_params(axis="y", rotation=0, labelsize=8)
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", output_path)


def fig_productivity_vs_influence(poet_names, sim_matrix, poems_by_poet,
                                   output_path):
    """
    Figure: Poet productivity vs stylistic influence scatter.
    """
    setup_plot_style()

    productivity = []
    influence = []
    for i, poet in enumerate(poet_names):
        n_poems = len(poems_by_poet.get(poet, []))
        sims = np.concatenate([sim_matrix[i, :i], sim_matrix[i, i+1:]])
        avg_sim = np.mean(sims)
        productivity.append(n_poems)
        influence.append(avg_sim)

    productivity = np.array(productivity)
    influence = np.array(influence)

    # Linear regression
    slope, intercept, r, p, se = linregress(productivity, influence)
    r_sq = r ** 2

    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(
        productivity, influence,
        c=productivity, cmap="plasma",
        s=80, alpha=0.7, edgecolors="black", linewidth=0.5,
    )

    # Regression line
    x_line = np.linspace(0, productivity.max() * 1.05, 100)
    ax.plot(x_line, slope * x_line + intercept, "r-", linewidth=2,
            label="R$^2$ = %.3f, p = %.4f" % (r_sq, p))

    ax.set_xlabel("Productivity (Number of Poems)")
    ax.set_ylabel("Stylistic Centrality (Average Similarity)")
    ax.legend(loc="lower right", fontsize=10)

    cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
    cbar.set_label("Poem Count")

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s (R2=%.4f, p=%.4f)", output_path, r_sq, p)


def fig_umap_hdbscan(umap_2d, labels, poet_names, output_path):
    """
    Figure: Hidden poetic schools UMAP + HDBSCAN scatter.
    """
    setup_plot_style()

    unique_labels = sorted(set(labels))
    n_clusters = len([l for l in unique_labels if l >= 0])

    # Color map
    cluster_colors = plt.cm.tab20(np.linspace(0, 1, max(n_clusters, 1)))

    fig, ax = plt.subplots(figsize=(14, 11))

    for label in unique_labels:
        mask = np.array(labels) == label
        if label == -1:
            ax.scatter(umap_2d[mask, 0], umap_2d[mask, 1],
                       c="gray", marker="x", s=60, alpha=0.5,
                       linewidths=1.5, label="Outliers", zorder=1)
        else:
            color = cluster_colors[label % len(cluster_colors)]
            count = int(mask.sum())
            ax.scatter(umap_2d[mask, 0], umap_2d[mask, 1],
                       c=[color], s=90, alpha=0.75, edgecolors="black",
                       linewidth=0.5,
                       label="Cluster %d (n=%d)" % (label, count),
                       zorder=2)

            # Annotate central poet in each cluster
            cluster_pts = umap_2d[mask]
            center = cluster_pts.mean(axis=0)
            dists = np.linalg.norm(cluster_pts - center, axis=1)
            closest = np.where(mask)[0][np.argmin(dists)]
            ax.annotate(
                arabic_display(poet_names[closest]),
                xy=(umap_2d[closest, 0], umap_2d[closest, 1]),
                fontsize=7, alpha=0.85,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.7),
            )

    ax.set_xlabel("UMAP Dimension 1")
    ax.set_ylabel("UMAP Dimension 2")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8,
              framealpha=0.9)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", output_path)


def fig_strategy_comparison(corr_df, output_path):
    """
    Figure: Pooling strategy correlation matrix (Table I as heatmap).
    """
    setup_plot_style()

    display_names = {
        "all_in_one": "All-in-one",
        "poem_average": "Poem avg",
        "global_verse_avg": "Global verse avg",
        "poem_verse_avg": "Proposed",
    }
    renamed = corr_df.rename(index=display_names, columns=display_names)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        renamed, annot=True, fmt=".2f",
        cmap="Blues", vmin=0.85, vmax=1.0,
        square=True, linewidths=1,
        cbar_kws={"label": "Pearson Correlation"},
        ax=ax,
    )
    ax.tick_params(axis="x", rotation=30)
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", output_path)


def fig_stability_heatmap(stability_df, output_path):
    """
    Figure: Cluster stability across parameter variations (ARI heatmap).
    """
    setup_plot_style()

    # Pivot: rows = n_neighbors, cols = min_cluster_size, values = mean ARI
    pivot = stability_df.groupby(
        ["umap_n_neighbors", "hdbscan_min_cluster_size"]
    )["ari"].mean().reset_index()
    pivot_table = pivot.pivot(
        index="umap_n_neighbors",
        columns="hdbscan_min_cluster_size",
        values="ari",
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        pivot_table, annot=True, fmt=".2f",
        cmap="RdYlGn", vmin=0, vmax=1,
        linewidths=1,
        cbar_kws={"label": "Adjusted Rand Index (ARI)"},
        ax=ax,
    )
    ax.set_xlabel("HDBSCAN min_cluster_size")
    ax.set_ylabel("UMAP n_neighbors")
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", output_path)


def fig_randomization_comparison(actual_n_clusters, actual_silhouette,
                                  random_results, output_path):
    """
    Figure: Actual vs shuffled cluster counts distribution.
    """
    setup_plot_style()

    rand_clusters = [r["n_clusters"] for r in random_results]
    rand_sils = [r["silhouette"] for r in random_results if r["silhouette"] is not None]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel A: cluster count distribution
    axes[0].hist(rand_clusters, bins=range(0, max(rand_clusters) + 2),
                 color="lightgray", edgecolor="black", alpha=0.8,
                 label="Shuffled")
    axes[0].axvline(actual_n_clusters, color="red", linewidth=2.5,
                     linestyle="--", label="Actual (%d)" % actual_n_clusters)
    axes[0].set_xlabel("Number of Clusters")
    axes[0].set_ylabel("Frequency")
    axes[0].legend()

    # Panel B: silhouette distribution
    if rand_sils:
        axes[1].hist(rand_sils, bins=20, color="lightgray", edgecolor="black",
                     alpha=0.8, label="Shuffled")
        if actual_silhouette is not None:
            axes[1].axvline(actual_silhouette, color="red", linewidth=2.5,
                             linestyle="--",
                             label="Actual (%.3f)" % actual_silhouette)
        axes[1].set_xlabel("Silhouette Score")
        axes[1].set_ylabel("Frequency")
        axes[1].legend()
    else:
        axes[1].text(0.5, 0.5, "No valid silhouette\nscores from shuffles",
                     ha="center", va="center", transform=axes[1].transAxes)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", output_path)


def fig_baseline_comparison(sbert_metrics, baselines, output_path):
    """
    Figure: Bar chart comparing SBERT vs baselines on cluster quality.
    """
    setup_plot_style()

    methods = ["SBERT (proposed)"]
    silhouettes = [sbert_metrics.get("silhouette", 0)]
    n_clusters = [sbert_metrics.get("n_clusters", 0)]

    for key, result in baselines.items():
        methods.append(result["name"])
        silhouettes.append(result["metrics"].get("silhouette") or 0)
        n_clusters.append(result["metrics"]["n_clusters"])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    colors = ["#e74c3c"] + ["#3498db"] * len(baselines)

    # Panel A: silhouette
    bars = axes[0].bar(methods, silhouettes, color=colors, edgecolor="black")
    axes[0].set_ylabel("Silhouette Score")
    axes[0].set_ylim(0, max(silhouettes) * 1.3 if max(silhouettes) > 0 else 1)
    axes[0].tick_params(axis="x", rotation=25)
    for bar, val in zip(bars, silhouettes):
        if val:
            axes[0].text(bar.get_x() + bar.get_width() / 2, val + 0.01,
                         "%.3f" % val, ha="center", va="bottom", fontsize=9)

    # Panel B: number of clusters
    bars2 = axes[1].bar(methods, n_clusters, color=colors, edgecolor="black")
    axes[1].set_ylabel("Number of Clusters")
    axes[1].tick_params(axis="x", rotation=25)
    for bar, val in zip(bars2, n_clusters):
        axes[1].text(bar.get_x() + bar.get_width() / 2, val + 0.2,
                     str(val), ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", output_path)


def fig_metadata_enrichment(enrichment_df, output_path):
    """
    Figure: Cluster x tribe enrichment heatmap.
    """
    setup_plot_style()

    # Drop 'Unknown' column if it dominates
    plot_df = enrichment_df.copy()
    if "Unknown" in plot_df.columns:
        unknown_frac = plot_df["Unknown"].sum() / plot_df.sum().sum()
        if unknown_frac > 0.7:
            plot_df = plot_df.drop(columns=["Unknown"])

    if plot_df.empty or plot_df.sum().sum() == 0:
        logger.warning("No tribal metadata to plot")
        return

    fig, ax = plt.subplots(figsize=(12, max(6, len(plot_df) * 0.5)))
    sns.heatmap(
        plot_df, annot=True, fmt="d",
        cmap="YlGnBu", linewidths=0.5,
        cbar_kws={"label": "Poet Count"},
        ax=ax,
    )
    ax.set_xlabel("Tribal Affiliation")
    ax.set_ylabel("Cluster")
    ax.tick_params(axis="x", rotation=30)
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", output_path)
