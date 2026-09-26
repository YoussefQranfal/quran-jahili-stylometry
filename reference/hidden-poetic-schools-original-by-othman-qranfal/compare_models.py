#!/usr/bin/env python3
"""
Compare Arabic/multilingual embedding models for poet stylistic analysis.

Tests multiple models on the same corpus and compares:
  1. Similarity distribution (mean, std, range) - key for Reviewer 3
  2. Clustering quality (silhouette, CH, DB, n_clusters)
  3. Discrimination power (how well the model separates poets)

Usage:
    cd ~/projects/hidden-poetic-schools
    source venv/bin/activate
    python compare_models.py 2>&1 | tee output/reports/model_comparison_log.txt
"""

import os, sys, time, json, re, sqlite3, warnings
import numpy as np
import pandas as pd
from collections import defaultdict
from pathlib import Path

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------

DB_PATH = "data/poems.db"
OUTPUT_DIR = "output/reports"
FIGURES_DIR = "output/figures"
MAX_VERSES_PER_POEM = 20
SEED = 42
np.random.seed(SEED)

# Models to compare - ordered from smallest to largest
# (display_name, HF_model_id, prefix_or_None)
MODELS = [
    ("Arabic-SBERT-100K (current)",
     "akhooli/Arabic-SBERT-100K",
     None),

    ("Multilingual-MiniLM (384d)",
     "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
     None),

    ("Multilingual-mpnet (768d)",
     "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
     None),

    ("Multilingual-E5-large (1024d)",
     "intfloat/multilingual-e5-large",
     "query: "),

    ("Arabic-Triple-ST (768d)",
     "Omartificial-Intelligence-Space/Arabic-TripleTArabic-sentence-transformer",
     None),
]

# ---------------------------------------------------------------
# DATA LOADING (same logic as main pipeline)
# ---------------------------------------------------------------

def load_corpus(db_path):
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(
        "SELECT poet_name, poem_text, verses_count "
        "FROM poems "
        "WHERE poet_name IS NOT NULL "
        "AND poem_text IS NOT NULL "
        "AND LENGTH(poem_text) >= 50",
        conn
    )
    conn.close()
    df["word_count"] = df["poem_text"].str.split().str.len()

    poems_by_poet = defaultdict(list)
    for _, row in df.iterrows():
        poems_by_poet[row["poet_name"]].append(row["poem_text"])

    print(f"  Loaded {len(df)} poems from {len(poems_by_poet)} poets")
    return df, poems_by_poet


def split_verses(poem_text):
    verses = re.split(r'[\n\r]+|[.!\u061F?\u061B;]+', poem_text)
    return [v.strip() for v in verses if len(v.strip()) > 10]


# ---------------------------------------------------------------
# EMBEDDING (hierarchical verse -> poem -> poet)
# ---------------------------------------------------------------

def embed_poets_hierarchical(model, poems_by_poet, prefix=None):
    rng = np.random.default_rng(SEED)
    poet_names = sorted(poems_by_poet.keys())
    poet_embeddings = {}

    for poet in poet_names:
        poems = poems_by_poet[poet]
        poem_vecs = []

        for poem_text in poems:
            verses = split_verses(poem_text)
            if not verses:
                text = prefix + poem_text if prefix else poem_text
                vec = model.encode([text], show_progress_bar=False)
                poem_vecs.append(vec[0])
                continue

            if len(verses) > MAX_VERSES_PER_POEM:
                idx = rng.choice(len(verses), MAX_VERSES_PER_POEM, replace=False)
                verses = [verses[i] for i in sorted(idx)]

            if prefix:
                verses = [prefix + v for v in verses]

            vecs = model.encode(verses, show_progress_bar=False, batch_size=64)
            poem_vecs.append(np.mean(vecs, axis=0))

        poet_embeddings[poet] = np.mean(poem_vecs, axis=0)

    names = sorted(poet_embeddings.keys())
    matrix = np.array([poet_embeddings[n] for n in names])

    # L2 normalize
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1
    matrix = matrix / norms

    return names, matrix


# ---------------------------------------------------------------
# ANALYSIS
# ---------------------------------------------------------------

def compute_similarity_stats(matrix):
    sim = matrix @ matrix.T
    n = sim.shape[0]
    upper = sim[np.triu_indices(n, k=1)]
    return {
        "mean": float(np.mean(upper)),
        "std": float(np.std(upper)),
        "min": float(np.min(upper)),
        "max": float(np.max(upper)),
        "median": float(np.median(upper)),
        "q25": float(np.percentile(upper, 25)),
        "q75": float(np.percentile(upper, 75)),
        "iqr": float(np.percentile(upper, 75) - np.percentile(upper, 25)),
    }


def cluster_and_evaluate(matrix):
    import umap
    import hdbscan
    from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

    reducer = umap.UMAP(
        n_components=50, n_neighbors=15, min_dist=0.1,
        metric="cosine", random_state=SEED,
    )
    embedding_50d = reducer.fit_transform(matrix)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=5, metric="euclidean",
        cluster_selection_method="eom",
    )
    labels = clusterer.fit_predict(embedding_50d)

    n_clusters = len(set(labels) - {-1})
    n_outliers = int(np.sum(labels == -1))

    mask = labels != -1
    if n_clusters >= 2 and np.sum(mask) > n_clusters:
        sil = float(silhouette_score(embedding_50d[mask], labels[mask]))
        ch = float(calinski_harabasz_score(embedding_50d[mask], labels[mask]))
        db = float(davies_bouldin_score(embedding_50d[mask], labels[mask]))
    else:
        sil, ch, db = -1.0, 0.0, 999.0

    return {
        "n_clusters": n_clusters,
        "n_outliers": n_outliers,
        "silhouette": sil,
        "calinski_harabasz": ch,
        "davies_bouldin": db,
        "labels": labels,
    }


# ---------------------------------------------------------------
# VISUALIZATION
# ---------------------------------------------------------------

def plot_comparison(results, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [r["name"] for r in results]
    n = len(names)
    colors = plt.cm.Set2(np.linspace(0, 1, n))

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle("Embedding Model Comparison for Arabic Poetry Analysis",
                 fontsize=16, fontweight="bold", y=0.98)

    # 1. Mean similarity (lower = better)
    ax = axes[0, 0]
    means = [r["sim"]["mean"] for r in results]
    ax.bar(range(n), means, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xticks(range(n))
    ax.set_xticklabels(names, fontsize=7, rotation=15, ha="right")
    ax.set_ylabel("Mean Cosine Similarity")
    ax.set_title("Mean Pairwise Similarity\n(lower = better discrimination)")
    for i, v in enumerate(means):
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=9, fontweight="bold")
    ax.set_ylim(0, 1.05)

    # 2. Similarity std (higher = better)
    ax = axes[0, 1]
    stds = [r["sim"]["std"] for r in results]
    ax.bar(range(n), stds, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xticks(range(n))
    ax.set_xticklabels(names, fontsize=7, rotation=15, ha="right")
    ax.set_ylabel("Std of Cosine Similarity")
    ax.set_title("Similarity Spread\n(higher = more discriminating)")
    for i, v in enumerate(stds):
        ax.text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=9, fontweight="bold")

    # 3. Box plots of similarity distributions
    ax = axes[0, 2]
    sim_dists = []
    for r in results:
        mat = r["matrix"]
        sim = mat @ mat.T
        nn = sim.shape[0]
        sim_dists.append(sim[np.triu_indices(nn, k=1)])
    bp = ax.boxplot(sim_dists, labels=names, patch_artist=True,
                    showmeans=True, showfliers=False)
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
    ax.set_ylabel("Cosine Similarity")
    ax.set_title("Similarity Distribution\n(wider + lower = better)")
    ax.tick_params(axis='x', labelsize=7, rotation=15)

    # 4. Number of clusters
    ax = axes[1, 0]
    ncl = [r["cluster"]["n_clusters"] for r in results]
    ax.bar(range(n), ncl, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xticks(range(n))
    ax.set_xticklabels(names, fontsize=7, rotation=15, ha="right")
    ax.set_ylabel("Number of Clusters")
    ax.set_title("Clusters Found (HDBSCAN)")
    for i, v in enumerate(ncl):
        ax.text(i, v + 0.3, str(v), ha="center", fontsize=11, fontweight="bold")

    # 5. Silhouette score
    ax = axes[1, 1]
    sils = [r["cluster"]["silhouette"] for r in results]
    ax.bar(range(n), sils, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xticks(range(n))
    ax.set_xticklabels(names, fontsize=7, rotation=15, ha="right")
    ax.set_ylabel("Silhouette Score")
    ax.set_title("Clustering Quality\n(higher = better)")
    for i, v in enumerate(sils):
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=9, fontweight="bold")
    ax.set_ylim(0, 1.0)

    # 6. Combined score
    ax = axes[1, 2]
    def norm(vals, higher_better=True):
        mn, mx = min(vals), max(vals)
        if mx == mn: return [0.5] * len(vals)
        return [(v-mn)/(mx-mn) if higher_better else (mx-v)/(mx-mn) for v in vals]

    sc_m = norm(means, higher_better=False)
    sc_s = norm(stds, higher_better=True)
    sc_si = norm(sils, higher_better=True)
    combined = [0.35*m + 0.30*s + 0.35*si for m, s, si in zip(sc_m, sc_s, sc_si)]
    ax.bar(range(n), combined, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xticks(range(n))
    ax.set_xticklabels(names, fontsize=7, rotation=15, ha="right")
    ax.set_ylabel("Combined Score")
    ax.set_title("Overall Score\n(0.35*discrim + 0.30*spread + 0.35*silhouette)")
    for i, v in enumerate(combined):
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=9, fontweight="bold")
    ax.set_ylim(0, 1.15)

    # Highlight best
    best_i = np.argmax(combined)
    for row in axes:
        for ax in row:
            pass  # bars already colored

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    outpath = os.path.join(output_dir, "model_comparison.png")
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Figure saved: {outpath}")


# ---------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------

def main():
    from sentence_transformers import SentenceTransformer

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    print("=" * 70)
    print("EMBEDDING MODEL COMPARISON")
    print("=" * 70)

    print("\n[1] Loading corpus...")
    df, poems_by_poet = load_corpus(DB_PATH)

    results = []

    for i, (display_name, model_id, prefix) in enumerate(MODELS):
        print(f"\n{'=' * 60}")
        print(f"[MODEL {i+1}/{len(MODELS)}] {display_name}")
        print(f"  HF ID: {model_id}")
        print(f"{'=' * 60}")

        try:
            t0 = time.time()
            print(f"  Loading model...")
            model = SentenceTransformer(model_id)
            load_time = time.time() - t0
            dim = model.get_sentence_embedding_dimension()
            print(f"  Loaded in {load_time:.1f}s (dim={dim})")

            t0 = time.time()
            print(f"  Embedding 260 poets (hierarchical verse->poem->poet)...")
            names, matrix = embed_poets_hierarchical(model, poems_by_poet, prefix=prefix)
            embed_time = time.time() - t0
            print(f"  Embedded in {embed_time:.1f}s")

            sim_stats = compute_similarity_stats(matrix)
            print(f"  Similarity: mean={sim_stats['mean']:.4f}, "
                  f"std={sim_stats['std']:.4f}, "
                  f"range=[{sim_stats['min']:.4f}, {sim_stats['max']:.4f}]")

            t0 = time.time()
            print(f"  Clustering (UMAP+HDBSCAN)...")
            cluster_stats = cluster_and_evaluate(matrix)
            cluster_time = time.time() - t0
            print(f"  => {cluster_stats['n_clusters']} clusters, "
                  f"{cluster_stats['n_outliers']} outliers, "
                  f"silhouette={cluster_stats['silhouette']:.4f}")

            results.append({
                "name": display_name,
                "model_id": model_id,
                "dim": dim,
                "sim": sim_stats,
                "cluster": cluster_stats,
                "matrix": matrix,
                "load_time": load_time,
                "embed_time": embed_time,
                "cluster_time": cluster_time,
            })

            del model
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            continue

    if not results:
        print("\nNo models completed!")
        return

    # --- Summary Table ---
    print("\n" + "=" * 100)
    print("COMPARISON SUMMARY")
    print("=" * 100)
    header = (f"{'Model':<35} {'Dim':>4} {'SimMean':>8} {'SimStd':>8} "
              f"{'SimMin':>8} {'Clust':>6} {'Outlrs':>6} {'Silhou':>8} {'Time':>6}")
    print(header)
    print("-" * len(header))

    for r in results:
        print(f"{r['name']:<35} {r['dim']:>4} "
              f"{r['sim']['mean']:>8.4f} {r['sim']['std']:>8.4f} "
              f"{r['sim']['min']:>8.4f} "
              f"{r['cluster']['n_clusters']:>6} "
              f"{r['cluster']['n_outliers']:>6} "
              f"{r['cluster']['silhouette']:>8.4f} "
              f"{r['embed_time']:>5.0f}s")

    # --- Recommendation ---
    print("\n" + "=" * 60)
    print("RECOMMENDATION")
    print("=" * 60)

    def norm(vals, higher_better=True):
        mn, mx = min(vals), max(vals)
        if mx == mn: return [0.5] * len(vals)
        return [(v-mn)/(mx-mn) if higher_better else (mx-v)/(mx-mn) for v in vals]

    sim_means = [r["sim"]["mean"] for r in results]
    sim_stds = [r["sim"]["std"] for r in results]
    sils = [r["cluster"]["silhouette"] for r in results]

    sc_m = norm(sim_means, higher_better=False)
    sc_s = norm(sim_stds, higher_better=True)
    sc_si = norm(sils, higher_better=True)
    combined = [0.35*m + 0.30*s + 0.35*si for m, s, si in zip(sc_m, sc_s, sc_si)]

    for i, r in enumerate(results):
        print(f"  {r['name']}: score={combined[i]:.3f}")

    best_idx = np.argmax(combined)
    best = results[best_idx]
    print(f"\n  >>> BEST: {best['name']} ({best['model_id']})")
    print(f"      Mean sim: {best['sim']['mean']:.4f}, "
          f"Std: {best['sim']['std']:.4f}, "
          f"Clusters: {best['cluster']['n_clusters']}, "
          f"Silhouette: {best['cluster']['silhouette']:.4f}")

    # --- Plot ---
    print("\n  Generating comparison figure...")
    plot_comparison(results, FIGURES_DIR)

    # --- Save JSON ---
    json_out = []
    for r in results:
        json_out.append({
            "name": r["name"],
            "model_id": r["model_id"],
            "dim": r["dim"],
            "similarity": r["sim"],
            "clustering": {k: v for k, v in r["cluster"].items() if k != "labels"},
            "embed_time_s": round(r["embed_time"], 1),
        })
    json_path = os.path.join(OUTPUT_DIR, "model_comparison.json")
    with open(json_path, "w") as f:
        json.dump(json_out, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {json_path}")

    # --- Save CSV ---
    rows = []
    for r in results:
        rows.append({
            "model": r["name"], "model_id": r["model_id"], "dim": r["dim"],
            "sim_mean": round(r["sim"]["mean"], 4),
            "sim_std": round(r["sim"]["std"], 4),
            "sim_min": round(r["sim"]["min"], 4),
            "n_clusters": r["cluster"]["n_clusters"],
            "n_outliers": r["cluster"]["n_outliers"],
            "silhouette": round(r["cluster"]["silhouette"], 4),
            "calinski_harabasz": round(r["cluster"]["calinski_harabasz"], 1),
            "davies_bouldin": round(r["cluster"]["davies_bouldin"], 4),
            "embed_time_s": round(r["embed_time"], 1),
        })
    csv_path = os.path.join("output/tables", "model_comparison.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")

    print("\n" + "=" * 60)
    print("DONE - Total models tested:", len(results))
    print("=" * 60)


if __name__ == "__main__":
    main()
