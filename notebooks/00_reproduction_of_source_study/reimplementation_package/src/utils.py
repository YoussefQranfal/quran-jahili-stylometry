"""
Shared utilities: logging setup, reproducibility seeding, Arabic text
normalization, and matplotlib/seaborn styling for Arabic-friendly figures.
"""
import logging
import random
import re
import sys

import numpy as np

# --------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------
def setup_logging(level=logging.INFO):
    logger = logging.getLogger("hidden_poetic_schools")
    if logger.handlers:
        return logger  # avoid duplicate handlers on repeated calls
    logger.setLevel(level)
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# --------------------------------------------------------------------------
# Arabic text normalization
# --------------------------------------------------------------------------
# Arabic diacritics (tashkeel) unicode range
_TASHKEEL_RE = re.compile(r"[\u0610-\u061A\u064B-\u065F\u06D6-\u06DC\u06DF-\u06E8\u06EA-\u06ED\u0670]")
_TATWEEL_RE = re.compile(r"\u0640")
_MULTISPACE_RE = re.compile(r"\s+")


def strip_diacritics(text: str) -> str:
    """Remove Arabic diacritical marks (tashkeel) from text."""
    return _TASHKEEL_RE.sub("", text)


def normalize_arabic(text: str, remove_diacritics: bool = True) -> str:
    """
    Normalize Arabic text: strip diacritics, tatweel, normalize letter
    variants (alef forms, ya/alef maksura, ta marbuta), collapse whitespace.
    """
    if not text:
        return ""
    if remove_diacritics:
        text = strip_diacritics(text)
    text = _TATWEEL_RE.sub("", text)

    # Normalize alef variants -> bare alef
    text = re.sub(r"[إأآا]", "ا", text)
    # Normalize ya variants
    text = re.sub(r"ى", "ي", text)
    # Normalize ta marbuta -> ha (common in stylometry preprocessing)
    text = re.sub(r"ة", "ه", text)
    # Normalize hamza carriers lightly (keep standalone hamza forms as-is)

    text = _MULTISPACE_RE.sub(" ", text).strip()
    return text


def is_valid_arabic_verse(text: str, min_chars: int = 3) -> bool:
    """Heuristic check that a line looks like a real verse, not junk/UI text."""
    if not text or len(text.strip()) < min_chars:
        return False
    arabic_chars = re.findall(r"[\u0600-\u06FF]", text)
    return len(arabic_chars) >= min_chars


# --------------------------------------------------------------------------
# Plot styling
# --------------------------------------------------------------------------
def setup_plot_style():
    import matplotlib
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams["figure.dpi"] = 150
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["savefig.bbox"] = "tight"
    plt.rcParams["font.size"] = 10

    # Try to find an Arabic-capable font for figures that render Arabic text
    from matplotlib import font_manager
    available = {f.name for f in font_manager.fontManager.ttflist}
    from config import ARABIC_FONT_CANDIDATES
    for name in ARABIC_FONT_CANDIDATES:
        if name in available:
            plt.rcParams["font.family"] = name
            break


def reshape_arabic(text: str) -> str:
    """
    Reshape + apply bidi algorithm so Arabic text renders correctly (joined
    letterforms, right-to-left order) inside matplotlib figures.
    """
    try:
        import arabic_reshaper
        from bidi.algorithm import get_display
        reshaped = arabic_reshaper.reshape(text)
        return get_display(reshaped)
    except ImportError:
        return text
