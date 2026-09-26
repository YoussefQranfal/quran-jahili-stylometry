"""
Baseline stylistic-similarity methods to compare against SBERT embeddings:
TF-IDF word vectors, character n-gram vectors (captures morphology/style
independent of vocabulary), and NMF topic vectors (captures topical rather
than stylistic similarity). Each gets the same UMAP+HDBSCAN treatment so
results are directly comparable to the proposed method (Table II).
"""
import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer

from src.clustering import reduce_dimensions, run_hdbscan, cluster_validation_metrics


def _poet_documents(poems_by_poet: dict) -> tuple[list[str], list[str]]:
    """Concatenate all verses per poet into one document per poet."""
    names, docs = [], []
    for poet, poems in poems_by_poet.items():
        verses = [v for poem in poems for v in poem["verses"]]
        if verses:
            names.append(poet)
            docs.append(" ".join(verses))
    return names, docs


def run_tfidf_baseline(poems_by_poet: dict, cfg) -> dict:
    names, docs = _poet_documents(poems_by_poet)
    vectorizer = TfidfVectorizer(max_features=cfg.TFIDF_MAX_FEATURES)
    X = vectorizer.fit_transform(docs).toarray()
    return _cluster_baseline(names, X, cfg, method_name="TF-IDF")


def run_char_ngram_baseline(poems_by_poet: dict, cfg) -> dict:
    names, docs = _poet_documents(poems_by_poet)
    vectorizer = TfidfVectorizer(
        analyzer="char", ngram_range=cfg.CHAR_NGRAM_RANGE,
        max_features=cfg.CHAR_NGRAM_MAX_FEATURES,
    )
    X = vectorizer.fit_transform(docs).toarray()
    return _cluster_baseline(names, X, cfg, method_name="Char n-grams")


def run_nmf_baseline(poems_by_poet: dict, cfg) -> dict:
    names, docs = _poet_documents(poems_by_poet)
    vectorizer = TfidfVectorizer(max_features=cfg.TFIDF_MAX_FEATURES)
    X_tfidf = vectorizer.fit_transform(docs)
    nmf = NMF(n_components=cfg.NMF_N_TOPICS, random_state=cfg.RANDOM_SEED, init="nndsvda", max_iter=500)
    X_topics = nmf.fit_transform(X_tfidf)
    return _cluster_baseline(names, X_topics, cfg, method_name="NMF Topics")


def _cluster_baseline(names: list[str], X: np.ndarray, cfg, method_name: str) -> dict:
    """Shared UMAP+HDBSCAN clustering + metrics for any baseline feature matrix."""
    n_components = min(cfg.UMAP_N_COMPONENTS_CLUSTER, X.shape[0] - 2, X.shape[1] - 1)
    n_components = max(n_components, 2)
    n_neighbors = min(cfg.UMAP_N_NEIGHBORS, X.shape[0] - 1)

    reduced = reduce_dimensions(
        X, n_neighbors, cfg.UMAP_MIN_DIST, n_components,
        metric="cosine", seed=cfg.RANDOM_SEED,
    )
    labels, _ = run_hdbscan(
        reduced, cfg.HDBSCAN_MIN_CLUSTER_SIZE, cfg.HDBSCAN_MIN_SAMPLES,
        cfg.HDBSCAN_METRIC, cfg.HDBSCAN_CLUSTER_SELECTION_METHOD,
    )
    metrics = cluster_validation_metrics(reduced, labels)
    metrics["method"] = method_name
    metrics["poet_names"] = names
    metrics["labels"] = labels
    return metrics


def run_all_baselines(poems_by_poet: dict, cfg) -> tuple[dict, pd.DataFrame]:
    baselines = {
        "tfidf": run_tfidf_baseline(poems_by_poet, cfg),
        "char_ngram": run_char_ngram_baseline(poems_by_poet, cfg),
        "nmf": run_nmf_baseline(poems_by_poet, cfg),
    }
    rows = []
    for key, result in baselines.items():
        rows.append({
            "method": result["method"],
            "n_clusters": result["n_clusters"],
            "n_noise": result["n_noise"],
            "silhouette": result["silhouette"],
            "calinski_harabasz": result["calinski_harabasz"],
            "davies_bouldin": result["davies_bouldin"],
        })
    comparison_df = pd.DataFrame(rows)
    return baselines, comparison_df
