"""
Baseline comparison methods (Reviewer 3 requirement).

Implements three alternative representation methods:
1. TF-IDF word features + clustering
2. Character n-gram features + clustering  
3. Simple topic model (NMF) + clustering

Each baseline produces poet-level representations that are clustered
with the same UMAP + HDBSCAN pipeline, enabling fair comparison
against the proposed SBERT approach.
"""
import logging
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import silhouette_score
import umap
import hdbscan

from .utils import split_verses

logger = logging.getLogger("poetic_schools")


def _poet_texts(poems_by_poet):
    """Concatenate all poems per poet into a single document."""
    poet_names = sorted(poems_by_poet.keys())
    texts = [" ".join(poems_by_poet[p]) for p in poet_names]
    return poet_names, texts


def _cluster_and_score(embeddings, min_cluster_size=5):
    """Run UMAP + HDBSCAN on any embedding matrix, return labels and metrics."""
    n = len(embeddings)
    # UMAP intermediate
    n_comp = min(50, n - 1, embeddings.shape[1])
    reducer = umap.UMAP(
        n_neighbors=min(15, n - 1),
        n_components=max(2, n_comp),
        min_dist=0.0,
        metric="cosine",
        random_state=42,
    )
    projected = reducer.fit_transform(embeddings)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min(min_cluster_size, n // 5),
        min_samples=3,
        cluster_selection_method="eom",
    )
    labels = clusterer.fit_predict(projected)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int(np.sum(labels == -1))

    # Silhouette on clustered points only
    mask = labels >= 0
    sil = None
    if mask.sum() > 1 and len(set(labels[mask])) > 1:
        sil = float(silhouette_score(projected[mask], labels[mask]))

    return labels, {
        "n_clusters": n_clusters,
        "n_outliers": n_noise,
        "silhouette": sil,
    }


# =========================================================================
# Baseline 1: TF-IDF word features
# =========================================================================
def tfidf_baseline(poems_by_poet, max_features=5000, ngram_range=(1, 2)):
    """
    Represent each poet as a TF-IDF vector over word unigrams/bigrams.
    """
    logger.info("Baseline: TF-IDF words (max_features=%d)", max_features)
    poet_names, texts = _poet_texts(poems_by_poet)

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        sublinear_tf=True,
    )
    tfidf_matrix = vectorizer.fit_transform(texts).toarray()

    labels, metrics = _cluster_and_score(tfidf_matrix)
    logger.info("  TF-IDF: %d clusters, silhouette=%s",
                metrics["n_clusters"], metrics["silhouette"])

    return {
        "name": "TF-IDF",
        "poet_names": poet_names,
        "embeddings": tfidf_matrix,
        "labels": labels,
        "metrics": metrics,
        "vectorizer": vectorizer,
    }


# =========================================================================
# Baseline 2: Character n-gram features
# =========================================================================
def char_ngram_baseline(poems_by_poet, ngram_range=(2, 5), max_features=5000):
    """
    Represent each poet as a TF-IDF vector over character n-grams.
    Captures morphological and phonological patterns.
    """
    logger.info("Baseline: char n-grams (%s)", ngram_range)
    poet_names, texts = _poet_texts(poems_by_poet)

    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=ngram_range,
        max_features=max_features,
        sublinear_tf=True,
    )
    char_matrix = vectorizer.fit_transform(texts).toarray()

    labels, metrics = _cluster_and_score(char_matrix)
    logger.info("  Char n-grams: %d clusters, silhouette=%s",
                metrics["n_clusters"], metrics["silhouette"])

    return {
        "name": "Char n-grams",
        "poet_names": poet_names,
        "embeddings": char_matrix,
        "labels": labels,
        "metrics": metrics,
    }


# =========================================================================
# Baseline 3: Topic model (NMF)
# =========================================================================
def topic_model_baseline(poems_by_poet, n_topics=20, max_features=5000):
    """
    Represent each poet as a topic distribution via NMF on TF-IDF.
    """
    logger.info("Baseline: NMF topic model (n_topics=%d)", n_topics)
    poet_names, texts = _poet_texts(poems_by_poet)

    vectorizer = TfidfVectorizer(max_features=max_features, sublinear_tf=True)
    tfidf = vectorizer.fit_transform(texts)

    nmf = NMF(n_components=n_topics, random_state=42, max_iter=500)
    topic_matrix = nmf.fit_transform(tfidf)

    labels, metrics = _cluster_and_score(topic_matrix)
    logger.info("  NMF topics: %d clusters, silhouette=%s",
                metrics["n_clusters"], metrics["silhouette"])

    return {
        "name": "NMF Topics",
        "poet_names": poet_names,
        "embeddings": topic_matrix,
        "labels": labels,
        "metrics": metrics,
        "nmf_model": nmf,
        "feature_names": vectorizer.get_feature_names_out(),
    }


# =========================================================================
# Run all baselines
# =========================================================================
def run_all_baselines(poems_by_poet, config):
    """
    Run all baseline methods and return comparison summary.
    
    Returns:
        baselines: dict {name: result_dict}
        comparison_df: DataFrame comparing all methods
    """
    baselines = {}
    baselines["tfidf"] = tfidf_baseline(
        poems_by_poet,
        max_features=config.TFIDF_MAX_FEATURES,
        ngram_range=config.TFIDF_NGRAM_RANGE,
    )
    baselines["char_ngram"] = char_ngram_baseline(
        poems_by_poet,
        ngram_range=config.CHAR_NGRAM_RANGE,
        max_features=config.CHAR_NGRAM_MAX_FEATURES,
    )
    baselines["topic_model"] = topic_model_baseline(poems_by_poet)

    # Comparison table
    rows = []
    for key, result in baselines.items():
        rows.append({
            "method": result["name"],
            "n_clusters": result["metrics"]["n_clusters"],
            "n_outliers": result["metrics"]["n_outliers"],
            "silhouette": result["metrics"]["silhouette"],
        })
    comparison_df = pd.DataFrame(rows)
    logger.info("Baseline comparison:\n%s", comparison_df.to_string(index=False))

    return baselines, comparison_df
