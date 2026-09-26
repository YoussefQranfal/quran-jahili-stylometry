"""
Builds poet-by-poet cosine similarity matrices, compares embedding
strategies against each other, and computes network-style centrality and
nearest-neighbor lists over the similarity graph.
"""
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics.pairwise import cosine_similarity


def compute_similarity_matrix(poet_embeddings: dict[str, np.ndarray]):
    """Returns (poet_names: list, sim_matrix: NxN ndarray)."""
    poet_names = sorted(poet_embeddings.keys())
    emb_matrix = np.stack([poet_embeddings[name] for name in poet_names])
    sim_matrix = cosine_similarity(emb_matrix)
    return poet_names, sim_matrix


def similarity_statistics(poet_names: list[str], sim_matrix: np.ndarray) -> dict:
    """
    Stats computed ONLY on the upper triangle (excluding diagonal), so we
    don't double count pairs or include self-similarity (=1.0) which would
    bias the mean upward.
    """
    n = len(poet_names)
    iu = np.triu_indices(n, k=1)
    values = sim_matrix[iu]
    return {
        "n_pairs": int(len(values)),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "max": float(values.max()),
        "median": float(np.median(values)),
    }


def compare_strategy_matrices(strategies: dict[str, dict]) -> pd.DataFrame:
    """
    For each pair of embedding strategies, compute the poet-poet similarity
    matrix for both (restricted to poets present in both) and report the
    Pearson correlation between their flattened upper triangles. This is
    Table I in the paper.
    """
    names = list(strategies.keys())
    rows = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            s1, s2 = strategies[names[i]], strategies[names[j]]
            common = sorted(set(s1) & set(s2))
            if len(common) < 3:
                continue
            m1 = np.stack([s1[p] for p in common])
            m2 = np.stack([s2[p] for p in common])
            sim1 = cosine_similarity(m1)
            sim2 = cosine_similarity(m2)
            iu = np.triu_indices(len(common), k=1)
            r, p = pearsonr(sim1[iu], sim2[iu])
            rows.append({
                "strategy_a": names[i], "strategy_b": names[j],
                "n_common_poets": len(common),
                "pearson_r": round(float(r), 4), "p_value": p,
            })
    return pd.DataFrame(rows)


def compute_centrality(poet_names: list[str], sim_matrix: np.ndarray) -> pd.DataFrame:
    """
    Network centrality of each poet within the similarity graph: mean
    similarity to all other poets (eigenvector-style closeness proxy) plus
    degree centrality after thresholding at the median similarity.
    """
    n = len(poet_names)
    mask = ~np.eye(n, dtype=bool)
    mean_sim = np.array([sim_matrix[i][mask[i]].mean() for i in range(n)])

    threshold = np.median(sim_matrix[np.triu_indices(n, k=1)])
    adj = (sim_matrix > threshold) & mask
    degree = adj.sum(axis=1)

    df = pd.DataFrame({
        "poet_name": poet_names,
        "mean_similarity_centrality": mean_sim,
        "degree_centrality": degree,
        "degree_centrality_normalized": degree / (n - 1),
    })
    return df.sort_values("mean_similarity_centrality", ascending=False).reset_index(drop=True)


def nearest_neighbors(poet_names: list[str], sim_matrix: np.ndarray,
                       top_k: int = 5) -> dict[str, list[tuple[str, float]]]:
    """For each poet, return their top_k most stylistically similar poets."""
    n = len(poet_names)
    result = {}
    for i in range(n):
        sims = sim_matrix[i].copy()
        sims[i] = -np.inf  # exclude self
        top_idx = np.argsort(sims)[::-1][:top_k]
        result[poet_names[i]] = [(poet_names[j], float(sim_matrix[i, j])) for j in top_idx]
    return result
