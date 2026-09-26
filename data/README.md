# `data/poems.db`

SQLite database of the pre-Islamic (Jahili) poetry corpus. Compiled from
[Aldiwan.net](https://www.aldiwan.net), a publicly accessible repository of
Arabic poetry (same source the original Othman & Qranfal study used).

- **File size:** 3,457,024 bytes
- **Tables:** one table, `poems`
- **Row count:** 2,361 rows

Verified directly against this file:

```
$ python -c "import sqlite3; print(sqlite3.connect('data/poems.db').execute('SELECT COUNT(*) FROM poems').fetchone())"
(2361,)
```

Note: 2,361 raw rows in the table is not the same number as the "2,328
poems" figure quoted throughout the notebooks — the notebooks
apply a cleaning filter (`poem_text IS NOT NULL`, minimum length, etc.)
before counting poems for analysis, which is where 2,361 raw rows becomes
2,328 analyzed poems. Both numbers are real; they describe the table before
and after that filter.

## Schema (`poems` table)

Inspected directly with `PRAGMA table_info(poems)`:

| # | Column | Type | Notes |
|---|--------|------|-------|
| 0 | `id` | INTEGER | primary key |
| 1 | `poet_name` | TEXT | poet's Arabic name |
| 2 | `poem_title` | TEXT | first hemistich, used as a title |
| 3 | `poem_text` | TEXT | full poem text, verses newline-separated |
| 4 | `poem_type` | TEXT | e.g. `عموديه` (traditional monorhyme) |
| 5 | `poem_meter` | TEXT | classical Arabic meter (e.g. `الوافر`, `البسيط`) |
| 6 | `verses_count` | INTEGER | verse count as recorded by the source site |
| 7 | `url` | TEXT | source URL on aldiwan.net |
| 8 | `scraped_at` | TEXT | scrape timestamp |
| 9 | `word_count` | INTEGER | |
| 10 | `char_count` | INTEGER | |
| 11 | `estimated_verses_from_text` | INTEGER | verse count re-derived from the text itself |
| 12 | `digit_ratio` | REAL | data-quality signal used to flag scraping artifacts |
| 13 | `special_char_ratio` | REAL | data-quality signal |
| 14 | `repetition_ratio` | REAL | data-quality signal |

## Where this file came from

Every notebook in this project that needs the poetry corpus was written to
load it from a local `poems.db` sitting next to the notebook. Across the
project's history that produced many identical copies of the same file
(one per notebook's working folder). All copies found in this project's
files hash to the same content and the same size (3,457,024 bytes); this
repository ships exactly one copy, here, and every notebook should be run
with `poems.db` copied or symlinked next to it (or simply run from a
working directory where this `data/` folder is reachable — see the
per-notebook instructions).

**Known gotcha (see `docs/KNOWN_ISSUES.md`):** `sqlite3.connect(path)` does
not error if `path` does not exist — it silently creates a new, empty
0-byte (well, ~12KB after any write) database file with no tables. Several
notebooks in this project's history were run from the wrong working
directory and hit `sqlite3.OperationalError: no such table: poems` as a
result. If you see that error, it means a *new*, empty database was
created next to the notebook — delete that stray file and re-run from a
directory where the real `data/poems.db` (3,457,024 bytes) is visible.

## The Qur'anic text

No Qur'an text file is bundled in this repository. Every notebook that
needs it fetches the Uthmani Arabic text (114 surahs, 6,236 ayat, plus
hizb/quarter-hizb boundaries and Meccan/Medinan classification) fresh from
the free, public [Al Quran Cloud API](https://alquran.cloud/api) —
typically `https://api.alquran.cloud/v1/quran/quran-uthmani` — on first
run, and caches the response locally (`quran_cache/*.json`, gitignored) so
subsequent runs don't re-fetch it. No manual data-preparation step is
needed for the Qur'anic side of any experiment.
