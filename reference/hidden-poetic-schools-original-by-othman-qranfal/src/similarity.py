"""
Pairwise similarity computation, nearest-neighbor analysis,
and network centrality measures.
"""
import logging

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger("poetic_schools")


def compute_similarity_matrix(poet_embeddings):
    """
    Compute pairwise cosine similarity matrix.
    
    Args:
        poet_embeddings: dict {poet_name: embedding_vector}
    
    Returns:
        poet_names: list of poet names (ordered)
        sim_matrix: np.ndarray of shape (n_poets, n_poets)
    """
    poet_names = sorted(poet_embeddings.keys())
    emb_matrix = np.array([poet_embeddings[p] for p in poet_names])
    sim_matrix = cosine_similarity(emb_matrix)
    logger.info("Similarity matrix: %d x %d", *sim_matrix.shape)
    return poet_names, sim_matrix


def similarity_statistics(poet_names, sim_matrix):
    """
    Compute statistics on the similarity distribution.
    Reports on the UPPER TRIANGLE only (excluding self-similarity diagonal).
    
    Returns:
        stats: dict with mean, std, min, max, median, q25, q75
    """
    n = len(poet_names)
    # Extract upper triangle (excluding diagonal)
    triu_idx = np.triu_indices(n, k=1)
    pairwise = sim_matrix[triu_idx]

    stats = {
        "n_pairs": len(pairwise),
        "mean": float(np.mean(pairwise)),
        "std": float(np.std(pairwise)),
        "min": float(np.min(pairwise)),
        "max": float(np.max(pairwise)),
        "median": float(np.median(pairwise)),
        "q25": float(np.percentile(pairwise, 25)),
        "q75": float(np.percentile(pairwise, 75)),
    }

    logger.info("Pairwise similarity: mean=%.4f, std=%.4f, "
                "range=[%.4f, %.4f]",
                stats["mean"], stats["std"], stats["min"], stats["max"])
    return stats


def compare_strategy_matrices(strategies):
    """
    Compute Pearson correlation between similarity matrices
    from different pooling strategies (Table I in paper).
    
    Args:
        strategies: dict {name: {poet: embedding}}
    
    Returns:
        corr_df: DataFrame of pairwise correlations
    """
    names = list(strategies.keys())
    matrices = {}
    common_poets = None

    for name, embs in strategies.items():
        poets = set(embs.keys())
        if common_poets is None:
            common_poets = poets
        else:
            common_poets = common_poets & poets

    common_poets = sorted(common_poets)
    logger.info("Comparing strategies on %d common poets", len(common_poets))

    for name, embs in strategies.items():
        emb_mat = np.array([embs[p] for p in common_poets])
        sim = cosine_similarity(emb_mat)
        triu = sim[np.triu_indices(len(common_poets), k=1)]
        matrices[name] = triu

    # Correlation matrix
    corr = np.zeros((len(names), len(names)))
    for i, n1 in enumerate(names):
        for j, n2 in enumerate(names):
            corr[i, j] = np.corrcoef(matrices[n1], matrices[n2])[0, 1]

    corr_df = pd.DataFrame(corr, index=names, columns=names)
    logger.info("Strategy correlation matrix:\n%s", corr_df.to_string())
    return corr_df


def compute_centrality(poet_names, sim_matrix):
    """
    Compute centrality measures for each poet.
    - Average similarity (stylistic influence proxy)
    - Degree centrality (above-threshold connections)
    
    Returns:
        centrality_df: DataFrame with poet_name, avg_similarity, degree
    """
    n = len(poet_names)
    rows = []
    for i, poet in enumerate(poet_names):
        # Average similarity to all others
        sims = np.concatenate([sim_matrix[i, :i], sim_matrix[i, i+1:]])
        avg_sim = float(np.mean(sims))
        # Degree: count of poets with similarity > median
        median_sim = float(np.median(sim_matrix[np.triu_indices(n, k=1)]))
        degree = int(np.sum(sims > median_sim))
        rows.append({
            "poet_name": poet,
            "avg_similarity": avg_sim,
            "degree": degree,
        })

    centrality_df = pd.DataFrame(rows).sort_values(
        "avg_similarity", ascending=False
    )
    top = centrality_df.iloc[0]
    logger.info("Most central poet: %s (avg_sim=%.4f)",
                top["poet_name"], top["avg_similarity"])
    return centrality_df


def nearest_neighbors(poet_names, sim_matrix, top_k=5):
    """
    For each poet, find top-k nearest neighbors.
    
    Returns:
        nn_dict: {poet: [(neighbor, similarity), ...]}
    """
    nn_dict = {}
    for i, poet in enumerate(poet_names):
        sims = sim_matrix[i].copy()
        sims[i] = -1  # exclude self
        top_idx = np.argsort(sims)[-top_k:][::-1]
        nn_dict[poet] = [(poet_names[j], float(sims[j])) for j in top_idx]
    return nn_dict
