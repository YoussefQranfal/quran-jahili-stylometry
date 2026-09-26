"""
Scraper: builds data/poems.db from aldiwan.net's pre-Islamic poets category.

This is a STANDALONE script, run separately from main.py:

    python src/scraper.py

It is deliberately not wired into main.py's pipeline, because scraping is a
one-time data-acquisition step, not part of the reproducible analysis.

Schema (matches what src/data_loader.py expects):

    poets(
        poet_id INTEGER PRIMARY KEY,
        name TEXT,
        slug TEXT UNIQUE,
        bio TEXT,
        era TEXT
    )
    poems(
        poem_id INTEGER PRIMARY KEY,
        poet_id INTEGER REFERENCES poets(poet_id),
        title TEXT,
        source_url TEXT UNIQUE,
        topic TEXT,
        meter TEXT,
        n_verses INTEGER
    )
    verses(
        verse_id INTEGER PRIMARY KEY,
        poem_id INTEGER REFERENCES poems(poem_id),
        verse_order INTEGER,
        text TEXT
    )

Politeness: this scraper sleeps SCRAPER_REQUEST_DELAY_SECONDS between
requests, uses a single session, and only touches pages under the
pre-Islamic poets category. Please don't crank the delay down — the site is
a small independent project (see aldiwan.net/aboutus).
"""
import re
import sqlite3
import sys
import time
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as cfg
from src.utils import setup_logging, is_valid_arabic_verse

logger = setup_logging()

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": cfg.SCRAPER_USER_AGENT})


def _get(url: str) -> BeautifulSoup:
    """Fetch a URL politely and return parsed HTML."""
    resp = SESSION.get(url, timeout=20)
    resp.raise_for_status()
    time.sleep(cfg.SCRAPER_REQUEST_DELAY_SECONDS)
    return BeautifulSoup(resp.text, "lxml")


def get_poet_links(category_url: str) -> list[dict]:
    """
    Scrape the pre-Islamic poets category page for poet name + profile URL.
    Returns e.g. [{"name": "...", "url": "https://www.aldiwan.net/cat-poet-..."}]
    """
    soup = _get(category_url)
    poets = []
    seen = set()
    for a in soup.select("a[href*='/cat-poet-']"):
        href = a.get("href", "")
        if not href or "/cat-poet-" not in href:
            continue
        url = urljoin(cfg.ALDIWAN_BASE_URL, href)
        name = a.get_text(strip=True)
        # skip the little "follow" icon links etc. — keep ones with real text
        if not name or url in seen:
            continue
        seen.add(url)
        poets.append({"name": name, "url": url})
    logger.info("Found %d poet links on category page", len(poets))
    return poets


def get_poet_bio(soup: BeautifulSoup) -> str:
    """Extract the poet biography paragraph from a poet profile page."""
    bio_tag = soup.find("h4") or soup.select_one("div.poet-bio, div.bio")
    return bio_tag.get_text(strip=True) if bio_tag else ""


def get_poem_links(poet_soup: BeautifulSoup) -> list[dict]:
    """
    From a poet's profile page, extract links to each individual poem, plus
    the lightweight metadata shown inline (topic, meter, verse count).
    """
    poems = []
    for a in poet_soup.select("a[href*='/poem']"):
        href = a.get("href", "")
        m = re.search(r"/poem(\d+)\.html", href)
        if not m:
            continue
        url = urljoin(cfg.ALDIWAN_BASE_URL, href)
        title = a.get_text(strip=True)
        if not title:
            continue
        poems.append({"poem_id_hint": int(m.group(1)), "title": title, "url": url})
    # de-duplicate by url, preserve order
    dedup, seen = [], set()
    for p in poems:
        if p["url"] not in seen:
            seen.add(p["url"])
            dedup.append(p)
    return dedup


def scrape_poem(url: str) -> dict:
    """
    Fetch a single poem page. Verses are rendered as individual <h3> tags in
    document order (one per hemistich/line).
    """
    soup = _get(url)
    verse_tags = soup.select("h3")
    verses = [t.get_text(strip=True) for t in verse_tags]
    verses = [v for v in verses if is_valid_arabic_verse(v)]

    # topic / meter appear as links near "قصيدة ... من ..."
    topic, meter = None, None
    topic_link = soup.select_one("a[href*='/Poems-Topics-']")
    if topic_link:
        topic = topic_link.get_text(strip=True)
    meter_link = soup.select_one("a[href*='/sea-']")
    if meter_link:
        meter = meter_link.get_text(strip=True)

    return {"verses": verses, "topic": topic, "meter": meter}


def init_db(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS poets (
            poet_id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            slug TEXT UNIQUE NOT NULL,
            bio TEXT,
            era TEXT
        );
        CREATE TABLE IF NOT EXISTS poems (
            poem_id INTEGER PRIMARY KEY AUTOINCREMENT,
            poet_id INTEGER NOT NULL REFERENCES poets(poet_id),
            title TEXT,
            source_url TEXT UNIQUE NOT NULL,
            topic TEXT,
            meter TEXT,
            n_verses INTEGER
        );
        CREATE TABLE IF NOT EXISTS verses (
            verse_id INTEGER PRIMARY KEY AUTOINCREMENT,
            poem_id INTEGER NOT NULL REFERENCES poems(poem_id),
            verse_order INTEGER,
            text TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_poems_poet ON poems(poet_id);
        CREATE INDEX IF NOT EXISTS idx_verses_poem ON verses(poem_id);
        """
    )
    conn.commit()
    return conn


def poet_already_scraped(conn: sqlite3.Connection, slug: str) -> bool:
    cur = conn.execute("SELECT 1 FROM poets WHERE slug = ?", (slug,))
    return cur.fetchone() is not None


def run(resume: bool = True, limit_poets: int | None = None):
    """
    Full scrape: poet list -> per-poet poems -> per-poem verses -> SQLite.
    Set resume=True (default) to safely re-run after an interruption; already
    scraped poets (by slug) are skipped.
    """
    db_path = cfg.DB_PATH
    conn = init_db(db_path)

    poet_links = get_poet_links(cfg.ALDIWAN_PRE_ISLAMIC_CATEGORY_URL)
    if limit_poets:
        poet_links = poet_links[:limit_poets]

    for poet in tqdm(poet_links, desc="Poets"):
        slug = poet["url"].rstrip("/").split("/")[-1]
        if resume and poet_already_scraped(conn, slug):
            continue
        try:
            poet_soup = _get(poet["url"])
        except requests.RequestException as e:
            logger.warning("Failed to fetch poet %s: %s", poet["name"], e)
            continue

        bio = get_poet_bio(poet_soup)
        cur = conn.execute(
            "INSERT OR IGNORE INTO poets (name, slug, bio, era) VALUES (?, ?, ?, ?)",
            (poet["name"], slug, bio, "pre-islamic"),
        )
        conn.commit()
        poet_id_row = conn.execute(
            "SELECT poet_id FROM poets WHERE slug = ?", (slug,)
        ).fetchone()
        poet_id = poet_id_row[0]

        poem_links = get_poem_links(poet_soup)
        for poem in tqdm(poem_links, desc=f"  {poet['name']}", leave=False):
            existing = conn.execute(
                "SELECT 1 FROM poems WHERE source_url = ?", (poem["url"],)
            ).fetchone()
            if existing:
                continue
            try:
                data = scrape_poem(poem["url"])
            except requests.RequestException as e:
                logger.warning("Failed to fetch poem %s: %s", poem["url"], e)
                continue
            if len(data["verses"]) < cfg.MIN_VERSES_PER_POEM:
                continue

            cur = conn.execute(
                "INSERT INTO poems (poet_id, title, source_url, topic, meter, n_verses) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (poet_id, poem["title"], poem["url"], data["topic"], data["meter"],
                 len(data["verses"])),
            )
            poem_id = cur.lastrowid
            conn.executemany(
                "INSERT INTO verses (poem_id, verse_order, text) VALUES (?, ?, ?)",
                [(poem_id, i, v) for i, v in enumerate(data["verses"])],
            )
            conn.commit()

    n_poets = conn.execute("SELECT COUNT(*) FROM poets").fetchone()[0]
    n_poems = conn.execute("SELECT COUNT(*) FROM poems").fetchone()[0]
    n_verses = conn.execute("SELECT COUNT(*) FROM verses").fetchone()[0]
    logger.info("Done. poets=%d poems=%d verses=%d -> %s",
                n_poets, n_poems, n_verses, db_path)
    conn.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Scrape aldiwan.net pre-Islamic poets")
    parser.add_argument("--limit-poets", type=int, default=None,
                         help="Only scrape the first N poets (useful for testing)")
    parser.add_argument("--no-resume", action="store_true",
                         help="Re-scrape poets even if already in the DB")
    args = parser.parse_args()
    run(resume=not args.no_resume, limit_poets=args.limit_poets)
