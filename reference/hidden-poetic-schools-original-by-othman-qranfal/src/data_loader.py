"""
Data loading, cleaning, and corpus statistics.

Schema: id, poet_name, poem_title, poem_text, poem_type, poem_meter,
        verses_count, url, scraped_at
"""
import sqlite3
import re
import hashlib
import logging
from collections import defaultdict

import numpy as np
import pandas as pd

from .utils import split_verses

logger = logging.getLogger("poetic_schools")


def load_corpus(db_path, min_text_length=50):
    """
    Load and clean the poetry corpus from SQLite.
    Matches the original notebook approach:
      - Filter: poet_name and poem_text not null, text >= 50 chars
      - Word count from raw text
      - No aggressive normalization (SBERT handles raw Arabic fine)

    Returns:
        df: Cleaned dataframe with poem data
        poems_by_poet: dict mapping poet_name to list of poem texts
    """
    logger.info("Loading corpus from %s", db_path)
    conn = sqlite3.connect(str(db_path))
    df = pd.read_sql_query(
        "SELECT poet_name, poem_title, poem_text, poem_type, "
        "poem_meter, verses_count "
        "FROM poems "
        "WHERE poet_name IS NOT NULL AND poem_text IS NOT NULL "
        "AND LENGTH(poem_text) >= ?",
        conn,
        params=(min_text_length,),
    )
    conn.close()

    n_raw = len(df)
    logger.info("Raw records (after SQL filter): %d", n_raw)

    # Word count from raw text
    df["word_count"] = df["poem_text"].str.split().str.len()

    # Verse list (split on newlines and sentence-ending punctuation)
    df["verse_list"] = df["poem_text"].apply(split_verses)
    df["verses_count_actual"] = df["verse_list"].apply(len)

    # Deduplication by hashing text
    df["poem_hash"] = df["poem_text"].apply(
        lambda t: hashlib.md5(t.strip().encode("utf-8")).hexdigest()
    )
    n_before_dedup = len(df)
    df = df.drop_duplicates(subset=["poem_hash"]).copy()
    n_dupes = n_before_dedup - len(df)
    if n_dupes:
        logger.info("Removed %d exact duplicates", n_dupes)

    # Build poems_by_poet (use raw text - SBERT handles Arabic natively)
    poems_by_poet = defaultdict(list)
    for _, row in df.iterrows():
        poems_by_poet[row["poet_name"]].append(row["poem_text"])

    n_poets = len(poems_by_poet)
    n_poems = len(df)
    logger.info("Cleaned corpus: %d poems from %d poets", n_poems, n_poets)

    return df, dict(poems_by_poet)


def corpus_statistics(df, poems_by_poet):
    """
    Compute corpus-level statistics for the paper.

    Returns:
        stats: dict of statistics
        poet_summary: per-poet DataFrame
    """
    stats = {}
    stats["n_poems"] = len(df)
    stats["n_poets"] = len(poems_by_poet)
    stats["median_word_count"] = float(df["word_count"].median())
    stats["mean_word_count"] = float(df["word_count"].mean())
    stats["std_word_count"] = float(df["word_count"].std())
    stats["median_verse_count"] = float(df["verses_count_actual"].median())
    stats["mean_verse_count"] = float(df["verses_count_actual"].mean())
    stats["total_tokens"] = int(df["word_count"].sum())

    # Poet-level summary
    poet_rows = []
    for poet, poems in poems_by_poet.items():
        n = len(poems)
        words = sum(len(p.split()) for p in poems)
        verses = sum(len(split_verses(p)) for p in poems)
        poet_rows.append({
            "poet_name": poet,
            "poem_count": n,
            "total_words": words,
            "total_verses": verses,
            "avg_words_per_poem": words / n if n else 0,
            "avg_verses_per_poem": verses / n if n else 0,
        })
    poet_summary = pd.DataFrame(poet_rows).sort_values(
        "poem_count", ascending=False
    )

    # Distribution stats
    counts = poet_summary["poem_count"]
    stats["max_poems_per_poet"] = int(counts.max())
    stats["min_poems_per_poet"] = int(counts.min())
    stats["poets_with_one_poem"] = int((counts == 1).sum())
    stats["poets_with_10plus"] = int((counts >= 10).sum())

    logger.info("Corpus: %d poems, %d poets, ~%d tokens",
                stats["n_poems"], stats["n_poets"], stats["total_tokens"])

    return stats, poet_summary
