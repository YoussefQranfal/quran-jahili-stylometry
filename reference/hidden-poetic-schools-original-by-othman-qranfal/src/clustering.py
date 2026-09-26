"""
Clustering: UMAP dimensionality reduction + HDBSCAN density-based clustering,
with K-means comparison and silhouette scoring.
"""
import logging

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import umap
import hdbscan

logger = logging.getLogger("poetic_schools")


def run_umap(embeddings_matrix, n_components=2, n_neighbors=15,
             min_dist=0.1, metric="cosine", random_state=42):
    """
    Project embeddings using UMAP.
    
    Returns:
        umap_embedding: np.ndarray of shape (n_samples, n_components)
    """
    reducer = umap.UMAP(
        n_neighbors=min(n_neighbors, len(embeddings_matrix) - 1),
        n_components=n_components,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
    )
    projected = reducer.fit_transform(embeddings_matrix)
    logger.info("UMAP: %d -> %d dimensions", embeddings_matrix.shape[1], n_components)
    return projected


def run_hdbscan(embeddings, min_cluster_size=5, min_samples=3,
                cluster_selection_method="eom"):
    """
    Cluster using HDBSCAN.
    
    Returns:
        labels: np.ndarray of cluster labels (-1 = outlier)
        clusterer: fitted HDBSCAN object
    """
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_method=cluster_selection_method,
        metric="euclidean",
    )
    labels = clusterer.fit_predict(embeddings)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int(np.sum(labels == -1))
    logger.info("HDBSCAN: %d clusters, %d outliers", n_clusters, n_noise)
    return labels, clusterer


def run_kmeans(embeddings, n_clusters=2, random_state=42):
    """
    K-means clustering (for comparison with HDBSCAN).
    
    Returns:
        labels, silhouette, kmeans_model
    """
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = km.fit_predict(embeddings)
    sil = silhouette_score(embeddings, labels)
    logger.info("K-means (k=%d): silhouette=%.4f", n_clusters, sil)
    return labels, sil, km


def cluster_validation_metrics(embeddings, labels):
    """
    Compute quantitative cluster validation metrics.
    Only uses non-outlier points (labels >= 0).
    
    Returns:
        metrics: dict with silhouette, calinski_harabasz, davies_bouldin
    """
    mask = labels >= 0
    if mask.sum() < 2 or len(set(labels[mask])) < 2:
        logger.warning("Not enough clusters for validation metrics")
        return {"silhouette": None, "calinski_harabasz": None, "davies_bouldin": None}

    emb_clean = embeddings[mask]
    lab_clean = labels[mask]

    metrics = {
        "silhouette": float(silhouette_score(emb_clean, lab_clean)),
        "calinski_harabasz": float(calinski_harabasz_score(emb_clean, lab_clean)),
        "davies_bouldin": float(davies_bouldin_score(emb_clean, lab_clean)),
        "n_clusters": len(set(lab_clean)),
        "n_outliers": int(np.sum(labels == -1)),
        "n_clustered": int(mask.sum()),
    }
    logger.info("Validation: silhouette=%.4f, CH=%.1f, DB=%.4f",
                metrics["silhouette"], metrics["calinski_harabasz"],
                metrics["davies_bouldin"])
    return metrics


def build_cluster_profiles(poet_names, labels, sim_matrix, poems_by_poet):
    """
    Build a detailed profile for each cluster.
    
    Returns:
        profiles: list of dicts with cluster stats
    """
    profiles = []
    unique_labels = sorted(set(labels))

    for label in unique_labels:
        mask = np.array(labels) == label
        members = [poet_names[i] for i in range(len(poet_names)) if mask[i]]

        # Within-cluster similarity
        idx = np.where(mask)[0]
        if len(idx) > 1:
            sub_sim = sim_matrix[np.ix_(idx, idx)]
            triu = sub_sim[np.triu_indices(len(idx), k=1)]
            within_sim = float(np.mean(triu))
        else:
            within_sim = 1.0

        # Productivity
        prod = [len(poems_by_poet.get(p, [])) for p in members]
        
        # Centrality (avg similarity to ALL poets)
        centralities = []
        for i in idx:
            others = np.concatenate([sim_matrix[i, :i], sim_matrix[i, i+1:]])
            centralities.append(float(np.mean(others)))

        profile = {
            "cluster_id": label,
            "cluster_label": "Outliers" if label == -1 else "Cluster %d" % label,
            "size": len(members),
            "poets": members,
            "within_similarity": within_sim,
            "mean_productivity": float(np.mean(prod)),
            "median_productivity": float(np.median(prod)),
            "total_poems": int(np.sum(prod)),
            "mean_centrality": float(np.mean(centralities)),
            "top_poets": sorted(zip(members, prod), key=lambda x: -x[1])[:5],
        }
        profiles.append(profile)

    return profiles


def full_clustering_pipeline(poet_embeddings, sim_matrix, poet_names,
                             poems_by_poet, config):
    """
    Run the complete clustering pipeline:
    1. UMAP (high-dim for clustering)
    2. HDBSCAN
    3. K-means comparison
    4. Validation metrics
    5. Cluster profiles
    
    Returns:
        results: dict with all clustering outputs
    """
    # Build embedding matrix in poet_names order
    emb_matrix = np.array([poet_embeddings[p] for p in poet_names])

    # UMAP for clustering (high-dim intermediate)
    umap_high = run_umap(
        emb_matrix,
        n_components=min(config.UMAP_N_COMPONENTS_HIGH, len(poet_names) - 1),
        n_neighbors=config.UMAP_N_NEIGHBORS,
        min_dist=0.0,
        metric=config.UMAP_METRIC,
        random_state=config.RANDOM_SEED,
    )

    # HDBSCAN
    hdbscan_labels, clusterer = run_hdbscan(
        umap_high,
        min_cluster_size=config.HDBSCAN_MIN_CLUSTER_SIZE,
        min_samples=config.HDBSCAN_MIN_SAMPLES,
        cluster_selection_method=config.HDBSCAN_CLUSTER_SELECTION_METHOD,
    )

    # UMAP for visualization (2D)
    umap_2d = run_umap(
        emb_matrix,
        n_components=2,
        n_neighbors=config.UMAP_N_NEIGHBORS,
        min_dist=config.UMAP_MIN_DIST,
        metric=config.UMAP_METRIC,
        random_state=config.RANDOM_SEED,
    )

    # K-means comparison on PCA
    pca = PCA(n_components=2, random_state=config.RANDOM_SEED)
    pca_2d = pca.fit_transform(emb_matrix)
    km_labels, km_sil, km_model = run_kmeans(pca_2d, n_clusters=2,
                                              random_state=config.RANDOM_SEED)

    # Validation metrics for HDBSCAN
    hdbscan_metrics = cluster_validation_metrics(umap_high, hdbscan_labels)

    # Cluster profiles
    profiles = build_cluster_profiles(
        poet_names, hdbscan_labels, sim_matrix, poems_by_poet
    )

    results = {
        "poet_names": poet_names,
        "emb_matrix": emb_matrix,
        "umap_2d": umap_2d,
        "umap_high": umap_high,
        "pca_2d": pca_2d,
        "hdbscan_labels": hdbscan_labels,
        "hdbscan_clusterer": clusterer,
        "hdbscan_metrics": hdbscan_metrics,
        "kmeans_labels": km_labels,
        "kmeans_silhouette": km_sil,
        "cluster_profiles": profiles,
    }
    return results
