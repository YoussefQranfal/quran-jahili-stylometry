"""
Dimensionality reduction (UMAP) + density-based clustering (HDBSCAN) to
discover latent stylistic "schools" among poets, plus a K-means comparison
and human-readable cluster profiles.
"""
import hdbscan
import numpy as np
import pandas as pd
import umap
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score, calinski_harabasz_score, davies_bouldin_score,
)


def reduce_dimensions(emb_matrix: np.ndarray, n_neighbors: int, min_dist: float,
                       n_components: int, metric: str, seed: int):
    reducer = umap.UMAP(
        n_neighbors=n_neighbors, min_dist=min_dist,
        n_components=n_components, metric=metric,
        random_state=seed,
    )
    return reducer.fit_transform(emb_matrix)


def run_hdbscan(embedding: np.ndarray, min_cluster_size: int,
                 min_samples=None, metric: str = "euclidean",
                 cluster_selection_method: str = "eom"):
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=metric,
        cluster_selection_method=cluster_selection_method,
    )
    labels = clusterer.fit_predict(embedding)
    return labels, clusterer


def cluster_validation_metrics(embedding: np.ndarray, labels: np.ndarray) -> dict:
    """
    Standard internal clustering metrics. Noise points (label == -1, from
    HDBSCAN) are excluded since silhouette/CH/DB are undefined for them.
    """
    mask = labels != -1
    n_clusters = len(set(labels[mask]))
    metrics = {"n_clusters": n_clusters, "n_noise": int((~mask).sum())}

    if n_clusters >= 2 and mask.sum() > n_clusters:
        X, y = embedding[mask], labels[mask]
        metrics["silhouette"] = float(silhouette_score(X, y))
        metrics["calinski_harabasz"] = float(calinski_harabasz_score(X, y))
        metrics["davies_bouldin"] = float(davies_bouldin_score(X, y))
    else:
        metrics.update({"silhouette": None, "calinski_harabasz": None, "davies_bouldin": None})
    return metrics


def run_kmeans_comparison(embedding: np.ndarray, k: int, seed: int):
    km = KMeans(n_clusters=k, random_state=seed, n_init=10)
    labels = km.fit_predict(embedding)
    sil = silhouette_score(embedding, labels) if k >= 2 else None
    return labels, sil


def build_cluster_profiles(poet_names: list[str], labels: np.ndarray,
                            sim_matrix: np.ndarray, poems_by_poet: dict,
                            centrality_df: pd.DataFrame) -> list[dict]:
    """
    Per-cluster summary: size, within-cluster mean similarity, productivity
    (poems/poet), mean centrality, and top representative poets (most poems).
    """
    profiles = []
    centrality_lookup = dict(zip(centrality_df["poet_name"],
                                  centrality_df["mean_similarity_centrality"]))
    unique_labels = sorted(set(labels))

    for cluster_id in unique_labels:
        idx = np.where(labels == cluster_id)[0]
        members = [poet_names[i] for i in idx]
        if len(members) >= 2:
            sub = sim_matrix[np.ix_(idx, idx)]
            iu = np.triu_indices(len(idx), k=1)
            within_sim = float(sub[iu].mean())
        else:
            within_sim = float("nan")

        poem_counts = {m: len(poems_by_poet.get(m, [])) for m in members}
        total_poems = sum(poem_counts.values())
        mean_productivity = total_poems / len(members) if members else 0
        median_productivity = float(np.median(list(poem_counts.values()))) if poem_counts else 0
        mean_centrality = float(np.mean([centrality_lookup.get(m, 0) for m in members]))
        top_poets = sorted(poem_counts.items(), key=lambda x: -x[1])[:5]

        label_str = "Noise" if cluster_id == -1 else f"School {cluster_id}"
        profiles.append({
            "cluster_id": int(cluster_id),
            "cluster_label": label_str,
            "size": len(members),
            "within_similarity": within_sim,
            "mean_productivity": mean_productivity,
            "median_productivity": median_productivity,
            "total_poems": total_poems,
            "mean_centrality": mean_centrality,
            "top_poets": top_poets,
        })
    return sorted(profiles, key=lambda p: p["cluster_id"])


def full_clustering_pipeline(poet_embeddings: dict[str, np.ndarray],
                              sim_matrix: np.ndarray, poet_names: list[str],
                              poems_by_poet: dict, cfg) -> dict:
    from src.similarity import compute_centrality

    emb_matrix = np.stack([poet_embeddings[name] for name in poet_names])

    # 2D projection for visualization
    umap_2d = reduce_dimensions(
        emb_matrix, cfg.UMAP_N_NEIGHBORS, cfg.UMAP_MIN_DIST,
        cfg.UMAP_N_COMPONENTS_2D, cfg.UMAP_METRIC, cfg.RANDOM_SEED,
    )
    # Higher-dim projection actually used for clustering (2D is too lossy
    # for HDBSCAN to find good density structure)
    umap_cluster = reduce_dimensions(
        emb_matrix, cfg.UMAP_N_NEIGHBORS, cfg.UMAP_MIN_DIST,
        cfg.UMAP_N_COMPONENTS_CLUSTER, cfg.UMAP_METRIC, cfg.RANDOM_SEED,
    )

    hdbscan_labels, clusterer = run_hdbscan(
        umap_cluster, cfg.HDBSCAN_MIN_CLUSTER_SIZE, cfg.HDBSCAN_MIN_SAMPLES,
        cfg.HDBSCAN_METRIC, cfg.HDBSCAN_CLUSTER_SELECTION_METHOD,
    )
    hdbscan_metrics = cluster_validation_metrics(umap_cluster, hdbscan_labels)

    # K-means baseline at k=2 (coarsest split) for comparison
    kmeans_labels, kmeans_silhouette = run_kmeans_comparison(umap_cluster, k=2, seed=cfg.RANDOM_SEED)

    centrality_df = compute_centrality(poet_names, sim_matrix)
    cluster_profiles = build_cluster_profiles(
        poet_names, hdbscan_labels, sim_matrix, poems_by_poet, centrality_df,
    )

    return {
        "poet_names": poet_names,
        "emb_matrix": emb_matrix,
        "umap_2d": umap_2d,
        "umap_cluster": umap_cluster,
        "hdbscan_labels": hdbscan_labels,
        "hdbscan_metrics": hdbscan_metrics,
        "kmeans_labels": kmeans_labels,
        "kmeans_silhouette": kmeans_silhouette if kmeans_silhouette else float("nan"),
        "cluster_profiles": cluster_profiles,
    }
