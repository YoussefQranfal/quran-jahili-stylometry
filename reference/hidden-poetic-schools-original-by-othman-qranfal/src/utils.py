"""
Utility functions: Arabic font setup, text normalization, logging.
"""
import re
import warnings
import logging
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import arabic_reshaper
from bidi.algorithm import get_display

warnings.filterwarnings("ignore")

logger = logging.getLogger("poetic_schools")


def setup_logging(level=logging.INFO):
    """Configure project-wide logging."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger("poetic_schools")


def setup_arabic_font():
    """Find and configure an Arabic-capable font for matplotlib."""
    preferred = ("amiri", "scheherazade", "noto naskh", "noto sans arabic",
                 "arial", "tahoma", "dejavu")
    for font_file in fm.findSystemFonts(fontext="ttf"):
        if any(name in font_file.lower() for name in preferred):
            prop = fm.FontProperties(fname=font_file)
            plt.rcParams["font.family"] = prop.get_name()
            plt.rcParams["axes.unicode_minus"] = False
            logger.info("Arabic font: %s", prop.get_name())
            return prop.get_name()
    logger.warning("No Arabic font found; Arabic labels may render incorrectly.")
    return None


def setup_plot_style():
    """Set publication-quality matplotlib defaults."""
    plt.style.use("default")
    plt.rcParams.update({
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.linewidth": 1.2,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })
    setup_arabic_font()


def arabic_display(text):
    """Reshape and reorder Arabic text for matplotlib rendering."""
    try:
        reshaped = arabic_reshaper.reshape(text)
        return get_display(reshaped)
    except Exception:
        return text


def normalize_arabic(text):
    """
    Light normalization for Arabic text (for baselines/TF-IDF only).
    SBERT embeddings use raw text directly.
    """
    if not text:
        return ""
    text = re.sub(r"[\u0617-\u061A\u064B-\u0652\u0670]", "", text)
    text = re.sub(r"[\u0622\u0623\u0625]", "\u0627", text)
    text = text.replace("\u0629", "\u0647")
    text = text.replace("\u0640", "")
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def split_verses(poem_text):
    """
    Split poem text into individual verses.
    Matches original notebook: re.split(r'[\\n\\r]+|[.!?!;]+', poem)
    with Arabic sentence-ending punctuation.
    Minimum 10 characters per verse (filters out fragments).
    """
    if not poem_text or not isinstance(poem_text, str):
        return []
    verses = re.split(r'[\n\r]+|[.!\u061F?\u061B;]+', poem_text)
    verses = [v.strip() for v in verses if len(v.strip()) > 10]
    return verses


def set_seed(seed):
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
