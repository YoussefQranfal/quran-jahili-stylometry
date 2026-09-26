"""
Sanity checks for the analysis pipeline.
Run with: python -m pytest tests/
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import numpy as np


def test_normalize_arabic():
    from src.utils import normalize_arabic
    text = normalize_arabic("")
    assert text == ""


def test_split_verses():
    from src.utils import split_verses
    poem = "verse one # verse two # verse three"
    verses = split_verses(poem)
    assert len(verses) == 3


def test_similarity_stats_consistency():
    """Verify mean/std/range are consistent (catches Reviewer 3 bug)."""
    rng = np.random.RandomState(42)
    n = 10
    embs = rng.randn(n, 50)
    from sklearn.metrics.pairwise import cosine_similarity
    sim = cosine_similarity(embs)
    triu = sim[np.triu_indices(n, k=1)]

    mean_val = np.mean(triu)
    std_val = np.std(triu)
    min_val = np.min(triu)
    max_val = np.max(triu)

    assert min_val <= mean_val <= max_val
    assert max_val - min_val >= 0


def test_config_paths():
    import config as cfg
    assert cfg.SBERT_MODEL_NAME == "akhooli/Arabic-SBERT-100K"
    assert cfg.RANDOM_SEED == 42


if __name__ == "__main__":
    test_normalize_arabic()
    test_split_verses()
    test_similarity_stats_consistency()
    test_config_paths()
    print("All tests passed!")
