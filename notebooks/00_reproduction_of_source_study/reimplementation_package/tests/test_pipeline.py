"""
Sanity checks that don't require the scraped corpus — they use small
synthetic poet/poem/verse fixtures so you can verify each stage works
before running the full pipeline on real data.

Run with: pytest tests/test_pipeline.py -v
"""
import numpy as np
import pytest

from src.utils import normalize_arabic, is_valid_arabic_verse
from src.similarity import compute_similarity_matrix, similarity_statistics, nearest_neighbors
from src.clustering import cluster_validation_metrics


@pytest.fixture
def fake_poems_by_poet():
    return {
        "poet_a": [
            {"poem_id": 1, "title": "t1", "topic": "general", "meter": "طويل",
             "verses": ["بيت أول من الشعر", "بيت ثان من الشعر"]},
            {"poem_id": 2, "title": "t2", "topic": "general", "meter": "كامل",
             "verses": ["بيت ثالث مختلف", "بيت رابع أيضا"]},
        ],
        "poet_b": [
            {"poem_id": 3, "title": "t3", "topic": "general", "meter": "طويل",
             "verses": ["كلمات أخرى تماما", "جمل مغايرة كليا"]},
            {"poem_id": 4, "title": "t4", "topic": "general", "meter": "كامل",
             "verses": ["نص جديد هنا", "سطر إضافي آخر"]},
        ],
    }


def test_normalize_arabic_strips_diacritics():
    text = "بِسْمِ اللَّهِ"
    result = normalize_arabic(text)
    # no tashkeel marks should remain
    assert "\u064B" not in result and "\u0650" not in result


def test_is_valid_arabic_verse():
    assert is_valid_arabic_verse("بيت شعر حقيقي")
    assert not is_valid_arabic_verse("")
    assert not is_valid_arabic_verse("12")


def test_similarity_matrix_shape_and_diagonal():
    fake_embeddings = {
        "a": np.array([1.0, 0.0, 0.0]),
        "b": np.array([0.0, 1.0, 0.0]),
        "c": np.array([1.0, 0.0, 0.0]),
    }
    names, sim = compute_similarity_matrix(fake_embeddings)
    assert sim.shape == (3, 3)
    assert np.allclose(np.diag(sim), 1.0, atol=1e-6)
    # a and c are identical vectors -> similarity should be ~1
    ia, ic = names.index("a"), names.index("c")
    assert sim[ia, ic] > 0.99


def test_similarity_statistics_excludes_diagonal():
    sim = np.array([[1.0, 0.5, 0.2], [0.5, 1.0, 0.3], [0.2, 0.3, 1.0]])
    stats = similarity_statistics(["a", "b", "c"], sim)
    assert stats["n_pairs"] == 3  # upper triangle only, k=1
    assert stats["max"] < 1.0     # diagonal (1.0) excluded


def test_nearest_neighbors_excludes_self():
    sim = np.array([[1.0, 0.9, 0.1], [0.9, 1.0, 0.2], [0.1, 0.2, 1.0]])
    nn = nearest_neighbors(["a", "b", "c"], sim, top_k=2)
    neighbor_names = [n for n, _ in nn["a"]]
    assert "a" not in neighbor_names


def test_cluster_validation_metrics_handles_all_noise():
    embedding = np.random.RandomState(0).rand(10, 5)
    labels = np.full(10, -1)  # everything is noise
    metrics = cluster_validation_metrics(embedding, labels)
    assert metrics["n_clusters"] == 0
    assert metrics["silhouette"] is None


def test_cluster_validation_metrics_two_clean_clusters():
    rng = np.random.RandomState(0)
    cluster1 = rng.normal(loc=0, scale=0.1, size=(10, 5))
    cluster2 = rng.normal(loc=10, scale=0.1, size=(10, 5))
    embedding = np.vstack([cluster1, cluster2])
    labels = np.array([0] * 10 + [1] * 10)
    metrics = cluster_validation_metrics(embedding, labels)
    assert metrics["n_clusters"] == 2
    assert metrics["silhouette"] > 0.9  # well-separated clusters
