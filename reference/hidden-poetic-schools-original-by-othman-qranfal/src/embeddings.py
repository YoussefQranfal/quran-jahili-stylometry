"""
Sentence-BERT embedding with four pooling strategies.
"""
import logging
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from .utils import split_verses

logger = logging.getLogger("poetic_schools")
MAX_VERSES_PER_POEM = 20


def load_model(model_name):
    logger.info("Loading SBERT model: %s", model_name)
    model = SentenceTransformer(model_name)
    logger.info("Model loaded (dim=%d)", model.get_sentence_embedding_dimension())
    return model


def _encode_batch(model, texts, batch_size=64):
    if not texts:
        return np.array([])
    return model.encode(texts, batch_size=batch_size, show_progress_bar=False,
                        normalize_embeddings=True, convert_to_numpy=True)


def embed_all_in_one(model, poems_by_poet, max_tokens=512):
    logger.info("Strategy 1: all-in-one")
    poet_embeddings = {}
    for poet, poems in poems_by_poet.items():
        full_text = " ".join(poems)
        words = full_text.split()[:max_tokens]
        truncated = " ".join(words)
        if truncated.strip():
            emb = _encode_batch(model, [truncated])
            poet_embeddings[poet] = emb[0]
    logger.info("  Embedded %d poets", len(poet_embeddings))
    return poet_embeddings


def embed_poem_average(model, poems_by_poet, max_tokens=512):
    logger.info("Strategy 2: poem average")
    poet_embeddings = {}
    for poet, poems in poems_by_poet.items():
        poem_embs = []
        for poem in poems:
            words = poem.split()[:max_tokens]
            text = " ".join(words)
            if text.strip():
                emb = _encode_batch(model, [text])
                poem_embs.append(emb[0])
        if poem_embs:
            poet_embeddings[poet] = np.mean(poem_embs, axis=0)
    logger.info("  Embedded %d poets", len(poet_embeddings))
    return poet_embeddings


def embed_global_verse_average(model, poems_by_poet):
    logger.info("Strategy 3: global verse average")
    poet_embeddings = {}
    for poet, poems in poems_by_poet.items():
        all_verses = []
        for poem in poems:
            all_verses.extend(split_verses(poem))
        if all_verses:
            embs = _encode_batch(model, all_verses)
            poet_embeddings[poet] = np.mean(embs, axis=0)
    logger.info("  Embedded %d poets", len(poet_embeddings))
    return poet_embeddings


def embed_poem_verse_average(model, poems_by_poet):
    logger.info("Strategy 4: poem -> verse -> average (PROPOSED)")
    poet_embeddings = {}
    rng = np.random.RandomState(42)
    for poet, poems in poems_by_poet.items():
        poem_vectors = []
        for poem in poems:
            verses = split_verses(poem)
            if len(verses) > MAX_VERSES_PER_POEM:
                idx = rng.choice(len(verses), MAX_VERSES_PER_POEM, replace=False)
                verses = [verses[i] for i in sorted(idx)]
            if verses:
                verse_embs = _encode_batch(model, verses)
                poem_vec = np.mean(verse_embs, axis=0)
                poem_vectors.append(poem_vec)
        if poem_vectors:
            poet_embeddings[poet] = np.mean(poem_vectors, axis=0)
    logger.info("  Embedded %d poets", len(poet_embeddings))
    return poet_embeddings


def compute_all_strategies(model, poems_by_poet):
    strategies = {
        "all_in_one": embed_all_in_one(model, poems_by_poet),
        "poem_average": embed_poem_average(model, poems_by_poet),
        "global_verse_avg": embed_global_verse_average(model, poems_by_poet),
        "poem_verse_avg": embed_poem_verse_average(model, poems_by_poet),
    }
    return strategies
