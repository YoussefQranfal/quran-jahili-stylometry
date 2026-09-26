"""
Configuration for Hidden Poetic Schools Analysis
All hyperparameters, paths, and settings in one place.
"""
from pathlib import Path

# =============================================================================
# PATHS
# =============================================================================
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"
FIGURES_DIR = OUTPUT_DIR / "figures"
TABLES_DIR = OUTPUT_DIR / "tables"
REPORTS_DIR = OUTPUT_DIR / "reports"

DB_PATH = DATA_DIR / "poems.db"

# Create output directories
for d in [FIGURES_DIR, TABLES_DIR, REPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# =============================================================================
# MODEL
# =============================================================================
SBERT_MODEL_NAME = "akhooli/Arabic-SBERT-100K"
EMBEDDING_DIM = 768
MAX_TOKENS_PER_POEM = 512

# =============================================================================
# CLUSTERING - UMAP
# =============================================================================
UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST = 0.1
UMAP_N_COMPONENTS_HIGH = 50
UMAP_N_COMPONENTS_VIS = 2
UMAP_METRIC = "cosine"

# =============================================================================
# CLUSTERING - HDBSCAN
# =============================================================================
HDBSCAN_MIN_CLUSTER_SIZE = 5
HDBSCAN_MIN_SAMPLES = 3
HDBSCAN_CLUSTER_SELECTION_METHOD = "eom"

# =============================================================================
# BASELINES
# =============================================================================
TFIDF_MAX_FEATURES = 5000
TFIDF_NGRAM_RANGE = (1, 2)
CHAR_NGRAM_RANGE = (2, 5)
CHAR_NGRAM_MAX_FEATURES = 5000

# =============================================================================
# VALIDATION
# =============================================================================
STABILITY_N_RUNS = 20
STABILITY_UMAP_NEIGHBORS_RANGE = [10, 15, 20, 25]
STABILITY_UMAP_MIN_DIST_RANGE = [0.0, 0.05, 0.1, 0.2]
STABILITY_HDBSCAN_MIN_CLUSTER_RANGE = [3, 5, 7, 10]
N_RANDOM_SHUFFLES = 100

# =============================================================================
# VISUALIZATION
# =============================================================================
FIGURE_DPI = 300
FIGURE_FORMAT = "png"
TOP_N_HEATMAP = 20

# =============================================================================
# GENERAL
# =============================================================================
RANDOM_SEED = 42
MIN_VERSES_PER_POEM = 3
