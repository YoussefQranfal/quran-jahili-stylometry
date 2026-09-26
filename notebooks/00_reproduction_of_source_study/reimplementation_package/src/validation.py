"""
Three validation analyses that guard against the clusters being an artifact
of a single arbitrary parameter choice or of chance:

1. Parameter stability: rerun UMAP+HDBSCAN across a grid of hyperparameters,
   measure agreement (ARI/NMI) against the "main" clustering.
2. Randomization test: shuffle which poet each embedding "belongs to" and
   recluster — if real clusters aren't much better than shuffled ones, the
   structure isn't meaningful.
3. Metadata enrichment: check whether clusters correlate with an external
   variable (e.g. tribal affiliation, if available in poet bios) more than
   chance would predict.
"""
import re

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from src.clustering import reduce_dimensions, run_hdbscan, cluster_validation_metrics


def stability_analysis(emb_matrix: np.ndarray, reference_labels: np.ndarray, cfg):
    """
    Reruns clustering across a grid of UMAP n_neighbors x min_dist x HDBSCAN
    min_cluster_size, comparing each run's labels to the reference (main
    pipeline) labels via ARI and NMI.
    """
    rows = []
    for nn in cfg.STABILITY_UMAP_NEIGHBORS:
        for md in cfg.STABILITY_UMAP_MIN_DIST:
            for mcs in cfg.STABILITY_HDBSCAN_MIN_CLUSTER_SIZE:
                reduced = reduce_dimensions(
                    emb_matrix, nn, md, cfg.UMAP_N_COMPONENTS_CLUSTER,
                    cfg.UMAP_METRIC, cfg.RANDOM_SEED,
                )
                labels, _ = run_hdbscan(
                    reduced, mcs, None, cfg.HDBSCAN_METRIC,
                    cfg.HDBSCAN_CLUSTER_SELECTION_METHOD,
                )
                ari = adjusted_rand_score(reference_labels, labels)
                nmi = normalized_mutual_info_score(reference_labels, labels)
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                rows.append({
                    "umap_n_neighbors": nn, "umap_min_dist": md,
                    "hdbscan_min_cluster_size": mcs,
                    "n_clusters": n_clusters, "ari": ari, "nmi": nmi,
                })
    df = pd.DataFrame(rows)
    summary = {
        "n_runs": len(df),
        "mean_ari": float(df["ari"].mean()), "std_ari": float(df["ari"].std()),
        "mean_nmi": float(df["nmi"].mean()), "std_nmi": float(df["nmi"].std()),
        "min_clusters": int(df["n_clusters"].min()),
        "max_clusters": int(df["n_clusters"].max()),
    }
    return df, summary


def randomization_test(emb_matrix: np.ndarray, cfg, n_shuffles: int = None):
    """
    Shuffle poet identity labels (i.e. permute which embedding row goes with
    which poet) and recluster — this destroys any real poet-to-poet
    structure while preserving the overall geometry of the embedding cloud.
    Comparing actual vs. shuffled cluster counts / silhouette scores tells
    us whether the actual clustering beats chance.

    Note: shuffling *rows* of an already-computed embedding matrix doesn't
    change clustering geometry (labels are just names) — the meaningful
    randomization here is reshuffling which VERSES got averaged into which
    poet's embedding upstream. For a lightweight approximation we instead
    add small random rotations/permutations of the poet-level vectors
    relative to each other via bootstrap resampling of poets with
    replacement, which tests whether cluster count is robust to sampling
    noise rather than true label permutation. See paper section on
    randomization testing for the full verse-level permutation procedure.
    """
    n_shuffles = n_shuffles or cfg.N_RANDOMIZATION_SHUFFLES
    n = emb_matrix.shape[0]
    results = []
    rng = np.random.RandomState(cfg.RANDOM_SEED)

    for _ in range(n_shuffles):
        idx = rng.choice(n, size=n, replace=True)
        boot_matrix = emb_matrix[idx]
        reduced = reduce_dimensions(
            boot_matrix, min(cfg.UMAP_N_NEIGHBORS, n - 1), cfg.UMAP_MIN_DIST,
            cfg.UMAP_N_COMPONENTS_CLUSTER, cfg.UMAP_METRIC, None,
        )
        labels, _ = run_hdbscan(
            reduced, cfg.HDBSCAN_MIN_CLUSTER_SIZE, None,
            cfg.HDBSCAN_METRIC, cfg.HDBSCAN_CLUSTER_SELECTION_METHOD,
        )
        metrics = cluster_validation_metrics(reduced, labels)
        results.append(metrics)

    clusters = [r["n_clusters"] for r in results]
    silhouettes = [r["silhouette"] for r in results if r["silhouette"] is not None]
    summary = {
        "n_shuffles": n_shuffles,
        "mean_clusters_shuffled": float(np.mean(clusters)),
        "std_clusters_shuffled": float(np.std(clusters)),
        "mean_silhouette_shuffled": float(np.mean(silhouettes)) if silhouettes else None,
    }
    return results, summary


_TRIBE_KEYWORDS = [
    "بني", "بنو", "قبيلة", "تغلب", "بكر", "طيء", "هذيل", "تميم", "قيس",
    "كندة", "أسد", "غطفان", "فزارة", "عبس", "ذبيان", "هوازن",
]


def extract_tribe_mentions(bio_text: str) -> list[str]:
    """Very lightweight keyword extraction of tribal affiliation from bio text."""
    if not bio_text:
        return []
    return [kw for kw in _TRIBE_KEYWORDS if kw in bio_text]


def metadata_enrichment(poet_names: list[str], labels: np.ndarray,
                         poet_bios: dict[str, str]) -> tuple[pd.DataFrame, list[str]]:
    """
    Cross-tabulates cluster assignment against tribal-affiliation keywords
    found in each poet's bio, as a lightweight external-validity check.
    """
    rows = []
    for name, label in zip(poet_names, labels):
        tribes = extract_tribe_mentions(poet_bios.get(name, ""))
        rows.append({"poet_name": name, "cluster": label, "tribes": ", ".join(tribes)})
    df = pd.DataFrame(rows)

    enrichment = pd.crosstab(df["cluster"], df["tribes"].replace("", "unknown"))

    report_lines = ["Metadata enrichment (tribal keyword mentions by cluster):"]
    for cluster_id in sorted(df["cluster"].unique()):
        sub = df[df["cluster"] == cluster_id]
        named = sub[sub["tribes"] != ""]
        if len(named) > 0:
            top = named["tribes"].value_counts().idxmax()
            report_lines.append(f"  Cluster {cluster_id}: most common tribe mention = {top} "
                                 f"({len(named)}/{len(sub)} poets with a bio mention)")
        else:
            report_lines.append(f"  Cluster {cluster_id}: no tribal keywords found in bios")
    return enrichment, report_lines


def full_validation_pipeline(emb_matrix, hdbscan_labels, poet_names,
                              poems_by_poet, model, cfg):
    stability_df, stability_summary = stability_analysis(emb_matrix, hdbscan_labels, cfg)
    randomization_results, randomization_summary = randomization_test(emb_matrix, cfg)

    # Bios aren't retained in poems_by_poet (verses only) — if you want real
    # metadata enrichment, load poet bios separately from the poets table
    # and pass them in here. Left as an empty dict by default.
    poet_bios = {}
    enrichment_df, metadata_report = metadata_enrichment(poet_names, hdbscan_labels, poet_bios)

    return {
        "stability_df": stability_df,
        "stability_summary": stability_summary,
        "randomization_results": randomization_results,
        "randomization_summary": randomization_summary,
        "enrichment_df": enrichment_df,
        "metadata_report": metadata_report,
    }
