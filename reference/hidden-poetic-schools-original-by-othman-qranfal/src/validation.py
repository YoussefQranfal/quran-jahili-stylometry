"""
Cluster validation (Reviewer 3 requirements):

1. Stability analysis - vary UMAP/HDBSCAN params, measure ARI/NMI consistency
2. Randomization test - shuffle verse assignments, show clusters disappear  
3. Metadata enrichment - check clusters against known tribe/region/genre
"""
import logging
from collections import Counter, defaultdict
import re

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics import silhouette_score
import umap
import hdbscan

from .utils import split_verses

logger = logging.getLogger("poetic_schools")


# =========================================================================
# 1. CLUSTER STABILITY ANALYSIS
# =========================================================================
def stability_analysis(emb_matrix, reference_labels, config):
    """
    Test how sensitive clusters are to UMAP/HDBSCAN hyperparameters.
    Varies n_neighbors, min_dist, and min_cluster_size independently.
    Reports ARI and NMI against the reference clustering.
    
    Returns:
        stability_df: DataFrame with parameter combinations and scores
        summary: dict with mean/std ARI and NMI
    """
    logger.info("Running stability analysis...")
    n = len(emb_matrix)
    results = []

    for nn in config.STABILITY_UMAP_NEIGHBORS_RANGE:
        for md in config.STABILITY_UMAP_MIN_DIST_RANGE:
            for mcs in config.STABILITY_HDBSCAN_MIN_CLUSTER_RANGE:
                try:
                    reducer = umap.UMAP(
                        n_neighbors=min(nn, n - 1),
                        n_components=min(50, n - 1),
                        min_dist=md,
                        metric="cosine",
                        random_state=config.RANDOM_SEED,
                    )
                    proj = reducer.fit_transform(emb_matrix)

                    clusterer = hdbscan.HDBSCAN(
                        min_cluster_size=min(mcs, n // 3),
                        min_samples=3,
                        cluster_selection_method="eom",
                    )
                    labels = clusterer.fit_predict(proj)
                    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

                    ari = adjusted_rand_score(reference_labels, labels)
                    nmi = normalized_mutual_info_score(reference_labels, labels)

                    results.append({
                        "umap_n_neighbors": nn,
                        "umap_min_dist": md,
                        "hdbscan_min_cluster_size": mcs,
                        "n_clusters": n_clusters,
                        "n_outliers": int(np.sum(labels == -1)),
                        "ari": ari,
                        "nmi": nmi,
                    })
                except Exception as e:
                    logger.warning("Stability run failed (nn=%d, md=%.2f, mcs=%d): %s",
                                   nn, md, mcs, e)

    stability_df = pd.DataFrame(results)

    summary = {
        "n_runs": len(results),
        "mean_ari": float(stability_df["ari"].mean()),
        "std_ari": float(stability_df["ari"].std()),
        "mean_nmi": float(stability_df["nmi"].mean()),
        "std_nmi": float(stability_df["nmi"].std()),
        "min_clusters": int(stability_df["n_clusters"].min()),
        "max_clusters": int(stability_df["n_clusters"].max()),
        "median_clusters": float(stability_df["n_clusters"].median()),
    }

    logger.info("Stability: %d runs, ARI=%.3f+/-%.3f, NMI=%.3f+/-%.3f, "
                "clusters=%d-%d",
                summary["n_runs"], summary["mean_ari"], summary["std_ari"],
                summary["mean_nmi"], summary["std_nmi"],
                summary["min_clusters"], summary["max_clusters"])

    return stability_df, summary


# =========================================================================
# 2. RANDOMIZATION TEST (NULL BASELINE)
# =========================================================================
def randomization_test(poems_by_poet, model, n_shuffles=100, config=None):
    """
    Shuffle verse assignments across poets and re-cluster.
    If clusters are meaningful, shuffled data should produce
    fewer/weaker clusters.
    
    This tests whether the discovered structure is an artifact of
    the corpus composition or reflects genuine stylistic patterns.
    
    Returns:
        random_results: list of dicts with shuffle metrics
        summary: dict with comparison statistics
    """
    logger.info("Running randomization test (%d shuffles)...", n_shuffles)

    # Collect all verses
    all_verses = []
    poet_verse_counts = {}
    poet_names = sorted(poems_by_poet.keys())

    for poet in poet_names:
        poet_verses = []
        for poem in poems_by_poet[poet]:
            poet_verses.extend(split_verses(poem))
        poet_verse_counts[poet] = len(poet_verses)
        all_verses.extend(poet_verses)

    total_verses = len(all_verses)
    logger.info("  Total verses to shuffle: %d across %d poets",
                total_verses, len(poet_names))

    # Encode all verses once (expensive but done only once)
    logger.info("  Encoding all verses...")
    verse_embeddings = model.encode(
        all_verses, batch_size=64, show_progress_bar=False,
        normalize_embeddings=True, convert_to_numpy=True,
    )

    random_results = []
    rng = np.random.RandomState(42)

    for i in range(n_shuffles):
        # Shuffle verse-to-poet assignment
        perm = rng.permutation(total_verses)
        shuffled_embs = verse_embeddings[perm]

        # Reassign to poets with original counts
        poet_embs = {}
        idx = 0
        for poet in poet_names:
            count = poet_verse_counts[poet]
            if count > 0:
                poet_embs[poet] = np.mean(shuffled_embs[idx:idx+count], axis=0)
            idx += count

        # Cluster shuffled embeddings
        emb_mat = np.array([poet_embs[p] for p in poet_names if p in poet_embs])
        try:
            reducer = umap.UMAP(
                n_neighbors=min(15, len(emb_mat) - 1),
                n_components=min(50, len(emb_mat) - 1),
                min_dist=0.0, metric="cosine", random_state=i,
            )
            proj = reducer.fit_transform(emb_mat)
            clusterer = hdbscan.HDBSCAN(min_cluster_size=5, min_samples=3)
            labels = clusterer.fit_predict(proj)

            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            mask = labels >= 0
            sil = None
            if mask.sum() > 1 and len(set(labels[mask])) > 1:
                sil = float(silhouette_score(proj[mask], labels[mask]))

            random_results.append({
                "shuffle": i,
                "n_clusters": n_clusters,
                "n_outliers": int(np.sum(labels == -1)),
                "silhouette": sil,
            })
        except Exception:
            pass

        if (i + 1) % 20 == 0:
            logger.info("  Shuffle %d/%d complete", i + 1, n_shuffles)

    rand_df = pd.DataFrame(random_results)
    summary = {
        "n_shuffles": len(random_results),
        "mean_clusters_shuffled": float(rand_df["n_clusters"].mean()),
        "std_clusters_shuffled": float(rand_df["n_clusters"].std()),
        "mean_silhouette_shuffled": float(rand_df["silhouette"].dropna().mean())
            if rand_df["silhouette"].dropna().any() else None,
    }

    logger.info("Randomization: shuffled avg clusters=%.1f+/-%.1f",
                summary["mean_clusters_shuffled"],
                summary["std_clusters_shuffled"])

    return random_results, summary


# =========================================================================
# 3. METADATA ENRICHMENT / PURITY
# =========================================================================

# Known tribal/regional markers in poet names
TRIBAL_MARKERS = {
    "Fazari": ["الفزاري"],
    "Tamimi": ["التميمي", "تميم"],
    "Taghlibi": ["التغلبي", "تغلب"],
    "Hamdani": ["الهمداني", "همدان"],
    "Hudhali": ["الهذلي", "هذيل"],
    "Azdi": ["الأزدي", "أزد"],
    "Absi": ["العبسي", "عبس"],
    "Tai": ["الطائي", "طيء"],
    "Kinani": ["الكناني", "كنانة"],
    "Murri": ["المري", "مرة"],
    "Asadi": ["الأسدي", "أسد"],
    "Dhubiyani": ["الذبياني", "ذبيان"],
    "Ghanawi": ["الغنوي", "غني"],
    "Bakri": ["البكري", "بكر"],
    "Qurashi": ["القرشي", "قريش"],
}

# Known genre associations (from classical scholarship)
KNOWN_ELEGISTS = ["الخنساء", "متمم بن نويرة"]
KNOWN_WARRIORS = ["عنترة بن شداد", "عمرو بن كلثوم"]
KNOWN_MUALLAQAT = [
    "امرؤ القيس", "طرفة بن العبد", "زهير بن أبي سلمى",
    "لبيد بن ربيعة", "عمرو بن كلثوم", "عنترة بن شداد",
    "الحارث بن حلزة",
]


def _detect_tribe(poet_name):
    """Detect tribal affiliation from poet name."""
    for tribe, markers in TRIBAL_MARKERS.items():
        for marker in markers:
            if marker in poet_name:
                return tribe
    return "Unknown"


def metadata_enrichment(poet_names, labels, poems_by_poet):
    """
    Check cluster composition against known metadata:
    - Tribal affiliations (from poet names)
    - Genre associations (known elegists, warriors, etc.)
    - Mu'allaqat poets
    
    Reports:
    - Per-cluster tribal distribution
    - Enrichment: are certain tribes over-represented in specific clusters?
    - Purity: dominant metadata category per cluster
    
    Returns:
        enrichment_df: DataFrame with cluster x tribe counts
        purity_scores: dict with purity per cluster
        report: list of textual findings
    """
    logger.info("Running metadata enrichment analysis...")

    # Detect tribes
    poet_tribes = {p: _detect_tribe(p) for p in poet_names}

    # Build cluster membership
    cluster_members = defaultdict(list)
    for i, poet in enumerate(poet_names):
        cluster_members[labels[i]].append(poet)

    # Enrichment matrix: cluster x tribe
    all_tribes = sorted(set(poet_tribes.values()))
    enrichment = {}
    for cl, members in cluster_members.items():
        cl_label = "Outliers" if cl == -1 else "Cluster %d" % cl
        tribe_counts = Counter(_detect_tribe(p) for p in members)
        enrichment[cl_label] = {t: tribe_counts.get(t, 0) for t in all_tribes}

    enrichment_df = pd.DataFrame(enrichment).T.fillna(0).astype(int)

    # Purity: fraction of dominant tribe in each cluster
    purity_scores = {}
    report = []
    for cl, members in cluster_members.items():
        if cl == -1:
            continue
        cl_label = "Cluster %d" % cl
        tribes = [_detect_tribe(p) for p in members]
        tribe_counts = Counter(tribes)
        dominant = tribe_counts.most_common(1)[0]
        purity = dominant[1] / len(members) if members else 0

        purity_scores[cl_label] = {
            "dominant_tribe": dominant[0],
            "purity": purity,
            "size": len(members),
        }

        # Check for Mu'allaqat poets
        muallaqat_in = [p for p in members if p in KNOWN_MUALLAQAT]
        elegists_in = [p for p in members if p in KNOWN_ELEGISTS]
        warriors_in = [p for p in members if p in KNOWN_WARRIORS]

        finding = "%s (n=%d): dominant tribe=%s (%.0f%%)" % (
            cl_label, len(members), dominant[0], purity * 100)
        if muallaqat_in:
            finding += ", Mu'allaqat: %s" % ", ".join(muallaqat_in)
        if elegists_in:
            finding += ", elegists: %s" % ", ".join(elegists_in)
        if warriors_in:
            finding += ", warriors: %s" % ", ".join(warriors_in)
        report.append(finding)

    for line in report:
        logger.info("  %s", line)

    return enrichment_df, purity_scores, report


def full_validation_pipeline(emb_matrix, reference_labels, poet_names,
                             poems_by_poet, model, config):
    """
    Run all validation analyses.
    
    Returns:
        validation: dict with all results
    """
    # 1. Stability
    stability_df, stability_summary = stability_analysis(
        emb_matrix, reference_labels, config
    )

    # 2. Randomization
    rand_results, rand_summary = randomization_test(
        poems_by_poet, model,
        n_shuffles=config.N_RANDOM_SHUFFLES,
        config=config,
    )

    # 3. Metadata enrichment
    enrichment_df, purity, metadata_report = metadata_enrichment(
        poet_names, reference_labels, poems_by_poet
    )

    validation = {
        "stability_df": stability_df,
        "stability_summary": stability_summary,
        "randomization_results": rand_results,
        "randomization_summary": rand_summary,
        "enrichment_df": enrichment_df,
        "purity_scores": purity,
        "metadata_report": metadata_report,
    }
    return validation
