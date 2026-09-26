"""
Computes poet-level embeddings with an Arabic Sentence-BERT model, using
four aggregation strategies, so we can compare which one best separates
stylistic signal (this is what Fig. 1 / Table I in the paper compares).

Strategies
----------
1. poem_verse_avg   (proposed): embed each verse -> average verses within a
                     poem -> average poems within a poet. Hierarchical mean.
2. poem_concat      : concatenate all verses of a poem into one string,
                       embed once per poem -> average poems per poet.
3. poet_concat      : concatenate the poet's ENTIRE output into one string,
                       embed once per poet (naive baseline).
4. verse_direct_avg : embed every verse independently, average ALL verses
                       belonging to a poet directly (skips the poem-level
                       averaging step, so long poems dominate more).
"""
import numpy as np
from sentence_transformers import SentenceTransformer


def load_model(model_name: str) -> SentenceTransformer:
    return SentenceTransformer(model_name)


def _embed_texts(model: SentenceTransformer, texts: list[str],
                  batch_size: int = 64) -> np.ndarray:
    if not texts:
        return np.zeros((0, model.get_sentence_embedding_dimension()))
    return model.encode(
        texts, batch_size=batch_size, show_progress_bar=False,
        convert_to_numpy=True, normalize_embeddings=True,
    )


def embed_poem_verse_average(model, poems_by_poet: dict) -> dict[str, np.ndarray]:
    """Strategy 1 (proposed): verse -> poem -> poet hierarchical mean."""
    poet_embeddings = {}
    for poet, poems in poems_by_poet.items():
        poem_vecs = []
        for poem in poems:
            verse_embs = _embed_texts(model, poem["verses"])
            if len(verse_embs) == 0:
                continue
            poem_vecs.append(verse_embs.mean(axis=0))
        if poem_vecs:
            poet_embeddings[poet] = np.mean(poem_vecs, axis=0)
    return poet_embeddings


def embed_poem_concat(model, poems_by_poet: dict) -> dict[str, np.ndarray]:
    """Strategy 2: concatenate verses per poem, embed poem, average poems."""
    poet_embeddings = {}
    for poet, poems in poems_by_poet.items():
        poem_texts = [" ".join(poem["verses"]) for poem in poems if poem["verses"]]
        poem_embs = _embed_texts(model, poem_texts)
        if len(poem_embs) > 0:
            poet_embeddings[poet] = poem_embs.mean(axis=0)
    return poet_embeddings


def embed_poet_concat(model, poems_by_poet: dict) -> dict[str, np.ndarray]:
    """Strategy 3: naive baseline — concatenate ALL of a poet's verses."""
    poet_texts, poet_names = [], []
    for poet, poems in poems_by_poet.items():
        all_verses = [v for poem in poems for v in poem["verses"]]
        if all_verses:
            poet_texts.append(" ".join(all_verses))
            poet_names.append(poet)
    embs = _embed_texts(model, poet_texts)
    return {name: emb for name, emb in zip(poet_names, embs)}


def embed_verse_direct_average(model, poems_by_poet: dict) -> dict[str, np.ndarray]:
    """Strategy 4: average every verse embedding directly (no poem grouping)."""
    poet_embeddings = {}
    for poet, poems in poems_by_poet.items():
        all_verses = [v for poem in poems for v in poem["verses"]]
        verse_embs = _embed_texts(model, all_verses)
        if len(verse_embs) > 0:
            poet_embeddings[poet] = verse_embs.mean(axis=0)
    return poet_embeddings


def compute_all_strategies(model, poems_by_poet: dict) -> dict[str, dict]:
    """Run all 4 strategies, return {strategy_name: {poet: vector}}."""
    return {
        "poem_verse_avg": embed_poem_verse_average(model, poems_by_poet),
        "poem_concat": embed_poem_concat(model, poems_by_poet),
        "poet_concat": embed_poet_concat(model, poems_by_poet),
        "verse_direct_avg": embed_verse_direct_average(model, poems_by_poet),
    }
