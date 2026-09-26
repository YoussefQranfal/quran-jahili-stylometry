"""
Loads the poems.db SQLite corpus into pandas structures, applies cleaning
and minimum-count filters, and computes corpus-level statistics.
"""
import sqlite3
from pathlib import Path

import pandas as pd

from src.utils import normalize_arabic, is_valid_arabic_verse


def load_corpus(db_path: Path, min_poems_per_poet: int = 2,
                 min_verses_per_poem: int = 2):
    """
    Load poets/poems/verses from SQLite into a single flat DataFrame, plus a
    convenience dict grouping verses by poet.

    Returns
    -------
    df : pd.DataFrame
        One row per verse, columns: poet_name, poet_id, poem_id, poem_title,
        topic, meter, verse_order, verse_text (normalized).
    poems_by_poet : dict[str, list[dict]]
        poet_name -> list of {"poem_id": ..., "title": ..., "verses": [str, ...]}
    """
    if not Path(db_path).exists():
        raise FileNotFoundError(
            f"{db_path} not found. Run `python src/scraper.py` first to build "
            f"the corpus (on a machine with normal internet access), or place "
            f"an existing poems.db in the data/ directory."
        )

    conn = sqlite3.connect(db_path)
    query = """
        SELECT
            p.name AS poet_name, p.poet_id AS poet_id,
            pm.poem_id AS poem_id, pm.title AS poem_title,
            pm.topic AS topic, pm.meter AS meter,
            v.verse_order AS verse_order, v.text AS verse_text
        FROM verses v
        JOIN poems pm ON v.poem_id = pm.poem_id
        JOIN poets p ON pm.poet_id = p.poet_id
        ORDER BY p.poet_id, pm.poem_id, v.verse_order
    """
    df = pd.read_sql_query(query, conn)
    conn.close()

    # Clean + filter verses
    df["verse_text"] = df["verse_text"].apply(normalize_arabic)
    df = df[df["verse_text"].apply(is_valid_arabic_verse)].copy()

    # Drop poems that are too short after cleaning
    poem_lengths = df.groupby("poem_id")["verse_text"].transform("count")
    df = df[poem_lengths >= min_verses_per_poem].copy()

    # Drop poets with too few poems
    poems_per_poet = df.groupby("poet_name")["poem_id"].transform("nunique")
    df = df[poems_per_poet >= min_poems_per_poet].copy()

    # Build poems_by_poet structure
    poems_by_poet = {}
    for poet_name, poet_df in df.groupby("poet_name"):
        poems = []
        for poem_id, poem_df in poet_df.groupby("poem_id"):
            poem_df = poem_df.sort_values("verse_order")
            poems.append({
                "poem_id": int(poem_id),
                "title": poem_df["poem_title"].iloc[0],
                "topic": poem_df["topic"].iloc[0],
                "meter": poem_df["meter"].iloc[0],
                "verses": poem_df["verse_text"].tolist(),
            })
        poems_by_poet[poet_name] = poems

    return df, poems_by_poet


def corpus_statistics(df: pd.DataFrame, poems_by_poet: dict):
    """Compute summary stats used in the paper's dataset description."""
    n_poets = len(poems_by_poet)
    n_poems = df["poem_id"].nunique()
    n_verses = len(df)
    n_tokens = df["verse_text"].str.split().apply(len).sum()

    poems_per_poet = {name: len(poems) for name, poems in poems_by_poet.items()}
    verses_per_poet = df.groupby("poet_name")["verse_text"].count().to_dict()

    stats = {
        "n_poets": n_poets,
        "n_poems": n_poems,
        "n_verses": n_verses,
        "n_tokens": int(n_tokens),
        "mean_poems_per_poet": round(sum(poems_per_poet.values()) / n_poets, 2),
        "median_poems_per_poet": int(pd.Series(poems_per_poet).median()),
        "max_poems_single_poet": max(poems_per_poet, key=poems_per_poet.get),
    }

    poet_summary = pd.DataFrame({
        "poet_name": list(poems_per_poet.keys()),
        "n_poems": list(poems_per_poet.values()),
    })
    poet_summary["n_verses"] = poet_summary["poet_name"].map(verses_per_poet)
    poet_summary = poet_summary.sort_values("n_poems", ascending=False).reset_index(drop=True)

    return stats, poet_summary
