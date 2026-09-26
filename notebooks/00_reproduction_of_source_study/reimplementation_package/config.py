"""
Central configuration for the Hidden Poetic Schools pipeline.
All paths and hyperparameters live here so every module stays in sync.
"""
from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
DB_PATH = DATA_DIR / "poems.db"

OUTPUT_DIR = PROJECT_ROOT / "output"
FIGURES_DIR = OUTPUT_DIR / "figures"
TABLES_DIR = OUTPUT_DIR / "tables"
REPORTS_DIR = OUTPUT_DIR / "reports"

for d in (DATA_DIR, FIGURES_DIR, TABLES_DIR, REPORTS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------
RANDOM_SEED = 42

# --------------------------------------------------------------------------
# Embedding model
# --------------------------------------------------------------------------
SBERT_MODEL_NAME = "akhooli/Arabic-SBERT-100K"
EMBEDDING_DIM = 768

# Models used in compare_models.py (4-model comparison)
COMPARISON_MODELS = {
    "Arabic-SBERT-100K": "akhooli/Arabic-SBERT-100K",
    "Multilingual-MiniLM": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "Multilingual-mpnet": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    "Multilingual-E5-large": "intfloat/multilingual-e5-large",
}

# --------------------------------------------------------------------------
# Corpus filters
# --------------------------------------------------------------------------
MIN_POEMS_PER_POET = 2      # poets with fewer poems are dropped
MIN_VERSES_PER_POEM = 2     # very short fragments dropped

# --------------------------------------------------------------------------
# Clustering: UMAP
# --------------------------------------------------------------------------
UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST = 0.1
UMAP_N_COMPONENTS_2D = 2        # for visualization
UMAP_N_COMPONENTS_CLUSTER = 10  # for HDBSCAN input (higher dim keeps more structure)
UMAP_METRIC = "cosine"

# --------------------------------------------------------------------------
# Clustering: HDBSCAN
# --------------------------------------------------------------------------
HDBSCAN_MIN_CLUSTER_SIZE = 5
HDBSCAN_MIN_SAMPLES = None       # defaults to min_cluster_size if None
HDBSCAN_METRIC = "euclidean"     # operates on UMAP-reduced space
HDBSCAN_CLUSTER_SELECTION_METHOD = "eom"

# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------
N_STABILITY_RUNS = 64            # parameter grid size for stability analysis
STABILITY_UMAP_NEIGHBORS = [5, 10, 15, 20]
STABILITY_UMAP_MIN_DIST = [0.0, 0.1, 0.25, 0.5]
STABILITY_HDBSCAN_MIN_CLUSTER_SIZE = [3, 5, 8, 10]
N_RANDOMIZATION_SHUFFLES = 100

# --------------------------------------------------------------------------
# Baselines
# --------------------------------------------------------------------------
TFIDF_MAX_FEATURES = 5000
CHAR_NGRAM_RANGE = (2, 4)
CHAR_NGRAM_MAX_FEATURES = 5000
NMF_N_TOPICS = 20

# --------------------------------------------------------------------------
# Visualization
# --------------------------------------------------------------------------
TOP_N_HEATMAP = 20
FIGURE_DPI = 300
ARABIC_FONT_CANDIDATES = [
    "Amiri", "Noto Naskh Arabic", "Noto Sans Arabic",
    "Arial", "Tahoma",
]

# --------------------------------------------------------------------------
# Scraper (used by src/scraper.py, run locally — not part of main.py)
# --------------------------------------------------------------------------
ALDIWAN_BASE_URL = "https://www.aldiwan.net"
ALDIWAN_PRE_ISLAMIC_CATEGORY_URL = f"{ALDIWAN_BASE_URL}/cat-poets-pre-islamic-period"
SCRAPER_REQUEST_DELAY_SECONDS = 1.5   # be polite; avoid hammering the server
SCRAPER_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)
