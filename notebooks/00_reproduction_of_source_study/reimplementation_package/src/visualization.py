"""
Generates the 7 (+1) publication figures from the paper. Each function
takes already-computed results and a save path, and writes a 300 DPI PNG.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.utils import reshape_arabic


def _finish(fig, save_path):
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig_similarity_heatmap(poet_names, sim_matrix, top_n, save_path):
    """Fig. 4: Poet similarity heatmap, restricted to the top-N most central poets."""
    n = len(poet_names)
    mean_sim = sim_matrix.mean(axis=1)
    top_idx = np.argsort(mean_sim)[::-1][:top_n]
    sub_matrix = sim_matrix[np.ix_(top_idx, top_idx)]
    sub_names = [reshape_arabic(poet_names[i]) for i in top_idx]

    fig, ax = plt.subplots(figsize=(10, 9))
    sns.heatmap(sub_matrix, xticklabels=sub_names, yticklabels=sub_names,
                cmap="viridis", square=True, ax=ax, cbar_kws={"label": "Cosine similarity"})
    ax.set_title(f"Poet Stylistic Similarity — Top {top_n} Most Central Poets")
    plt.setp(ax.get_xticklabels(), rotation=90)
    _finish(fig, save_path)


def fig_productivity_vs_influence(poet_names, sim_matrix, poems_by_poet, save_path):
    """Fig. 7: productivity (poem count) vs. stylistic centrality — tests independence."""
    n = len(poet_names)
    mask = ~np.eye(n, dtype=bool)
    centrality = np.array([sim_matrix[i][mask[i]].mean() for i in range(n)])
    productivity = np.array([len(poems_by_poet.get(name, [])) for name in poet_names])

    from scipy.stats import pearsonr
    r, p = pearsonr(productivity, centrality)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(productivity, centrality, alpha=0.6, edgecolor="k", linewidth=0.3)
    z = np.polyfit(productivity, centrality, 1)
    xs = np.linspace(productivity.min(), productivity.max(), 100)
    ax.plot(xs, np.polyval(z, xs), color="crimson", linestyle="--", linewidth=1.5)
    ax.set_xlabel("Productivity (number of poems)")
    ax.set_ylabel("Mean stylistic centrality (similarity)")
    ax.set_title(f"Productivity vs. Stylistic Influence  (r={r:.3f}, R²={r**2:.3f}, p={p:.3g})")
    _finish(fig, save_path)


def fig_umap_hdbscan(umap_2d, labels, poet_names, save_path):
    """Fig. 5: main result — UMAP scatter colored by HDBSCAN cluster."""
    fig, ax = plt.subplots(figsize=(9, 8))
    unique_labels = sorted(set(labels))
    palette = sns.color_palette("husl", len([l for l in unique_labels if l != -1]))
    color_map = {}
    color_idx = 0
    for l in unique_labels:
        if l == -1:
            color_map[l] = (0.7, 0.7, 0.7)
        else:
            color_map[l] = palette[color_idx]
            color_idx += 1

    for l in unique_labels:
        idx = labels == l
        label_str = "Noise" if l == -1 else f"School {l}"
        ax.scatter(umap_2d[idx, 0], umap_2d[idx, 1], s=40, alpha=0.8,
                   color=color_map[l], label=label_str, edgecolor="k", linewidth=0.3)

    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.set_title("Hidden Stylistic Schools of Pre-Islamic Poetry")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    _finish(fig, save_path)


def fig_strategy_comparison(corr_df: pd.DataFrame, save_path):
    """Fig. 1: correlation between embedding strategies' similarity matrices."""
    if corr_df.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = corr_df["strategy_a"] + " vs\n" + corr_df["strategy_b"]
    ax.bar(labels, corr_df["pearson_r"], color="steelblue", edgecolor="k")
    ax.set_ylabel("Pearson correlation (r)")
    ax.set_title("Embedding Strategy Correlation Comparison")
    ax.set_ylim(0, 1)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=8)
    _finish(fig, save_path)


def fig_stability_heatmap(stability_df: pd.DataFrame, save_path):
    """Fig. 6 (parameter stability): mean ARI across UMAP neighbors x HDBSCAN min_cluster_size."""
    pivot = stability_df.pivot_table(
        index="umap_n_neighbors", columns="hdbscan_min_cluster_size",
        values="ari", aggfunc="mean",
    )
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="magma", ax=ax,
                cbar_kws={"label": "Mean ARI vs. main clustering"})
    ax.set_title("Cluster Stability Across Parameters")
    ax.set_xlabel("HDBSCAN min_cluster_size")
    ax.set_ylabel("UMAP n_neighbors")
    _finish(fig, save_path)


def fig_randomization_comparison(actual_n_clusters, actual_silhouette,
                                  randomization_results, save_path):
    """Fig. 6: actual cluster count/silhouette vs. distribution from shuffled runs."""
    shuffled_clusters = [r["n_clusters"] for r in randomization_results]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(shuffled_clusters, bins=range(0, max(shuffled_clusters + [actual_n_clusters]) + 2),
            color="lightgray", edgecolor="k", label="Shuffled runs")
    ax.axvline(actual_n_clusters, color="crimson", linewidth=2, label=f"Actual ({actual_n_clusters})")
    ax.set_xlabel("Number of clusters")
    ax.set_ylabel("Count of runs")
    ax.set_title("Randomization Test: Actual vs. Shuffled Cluster Counts")
    ax.legend()
    _finish(fig, save_path)


def fig_baseline_comparison(hdbscan_metrics: dict, baselines: dict, save_path):
    """Fig. 3: SBERT (proposed) vs. TF-IDF / char n-gram / NMF baselines."""
    methods = ["SBERT (proposed)"] + [b["method"] for b in baselines.values()]
    silhouettes = [hdbscan_metrics.get("silhouette") or 0] + \
                  [b["silhouette"] or 0 for b in baselines.values()]

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ["crimson"] + ["steelblue"] * (len(methods) - 1)
    ax.bar(methods, silhouettes, color=colors, edgecolor="k")
    ax.set_ylabel("Silhouette score")
    ax.set_title("Baseline Comparison: SBERT vs. Alternatives")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    _finish(fig, save_path)


def fig_metadata_enrichment(enrichment_df: pd.DataFrame, save_path):
    """Fig. 8: cluster x tribal-mention heatmap."""
    if enrichment_df.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.heatmap(enrichment_df, annot=True, fmt="d", cmap="YlGnBu", ax=ax)
    ax.set_title("Cluster / Tribal Mention Enrichment")
    ax.set_xlabel("Tribal keyword mentioned in bio")
    ax.set_ylabel("Cluster")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    _finish(fig, save_path)
