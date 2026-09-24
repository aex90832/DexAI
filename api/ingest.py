#!/usr/bin/env python3
"""
Pokedex AI — database builder.

Builds /data/pokedex.db from three sources:

  1. Pokemon Showdown dex   via pokedex-sim's /dump endpoint  (local)
  2. Smogon usage stats     from STATS_DIR, or smogon.com if allowed
  3. Bulbapedia             from the ZIM file, read directly with libzim

Then derives the alias table (which joins the three together) and generates
embeddings for retrieval.

Every stage is idempotent — it clears its own scope before writing, so re-running
is always safe.

Usage:
    python ingest.py                  # everything
    python ingest.py --dex-only
    python ingest.py --stats-only
    python ingest.py --wiki-only      # implies alias + embedding rebuild
    python ingest.py --aliases-only
    python ingest.py --embed-only
    python ingest.py --wiki-only --limit 500     # quick test run

First run only, allow the embedding model to download:
    docker exec -it -e HF_HUB_OFFLINE=0 -e TRANSFORMERS_OFFLINE=0 \
        pokedex-api /app/.venv/bin/python ingest.py
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import sys
import time
import unicodedata
from pathlib import Path
from typing import Iterable, Iterator

# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------

DB_PATH = os.environ.get("DB_PATH", "/data/pokedex.db")
ZIM_PATH = os.environ.get("ZIM_PATH", "")
KIWIX_URL = os.environ.get("KIWIX_URL", "").rstrip("/")
KIWIX_BOOK = os.environ.get("KIWIX_BOOK", "")
SIM_URL = os.environ.get("SIM_URL", "http://pokedex-sim:8991").rstrip("/")
STATS_DIR = os.environ.get("STATS_DIR", "")
ANALYSES_DIR = os.environ.get("ANALYSES_DIR", "")
SETS_DIR = os.environ.get("SETS_DIR", "")
HOME_ICONS_DIR = os.environ.get("HOME_ICONS_DIR", "")
HOME_PREVIEWS_DIR = os.environ.get("HOME_PREVIEWS_DIR", "")
# Public URL prefix these two folders are served from — see the compose file
# for the actual static-file mount this points at.
HOME_SPRITES_URL = os.environ.get("HOME_SPRITES_URL", "").rstrip("/")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
# Some embedding models (nomic-embed-text, e5, bge-m3 with instructions) need a
# task prefix prepended to every text — "search_document: " for what gets
# indexed, "search_query: " for what a user asks at query time — or retrieval
# quality degrades silently with no error. bge-small needs neither, so this is
# empty by default. Set to match whatever EMBED_MODEL actually requires; check
# the model card. The query-side twin of this lives in main.py as
# EMBED_QUERY_PREFIX — they must be set consistently with each other and with
# whichever model built the index, or query and document vectors stop lining
# up in a way nothing will surface except worse retrieval.
EMBED_DOC_PREFIX = os.environ.get("EMBED_DOC_PREFIX", "")
# Point this at an OpenAI-compatible /embeddings endpoint (vLLM on the GPU, say)
# to skip local CPU embedding entirely. Unset = use fastembed on the CPU.
EMBED_URL = os.environ.get("EMBED_URL", "").rstrip("/")
EMBED_API_KEY = os.environ.get("EMBED_API_KEY", "")

# ---------------------------------------------------------------------------
# Embedding tiers — one decision instead of five flags.
#
# Measured on a 734k-chunk real run against BAAI/bge-small-en-v1.5, gfx1201
# (AMD Radeon AI PRO R9700, 32GB), single card:
#   cpu       : 28-36 chunks/s regardless of thread count 1-192 (serial
#               bottleneck in tokenisation, not compute — more cores don't help)
#   gpu-small : concurrency=6,  batch=256  -> ~73/s   sequential HTTP baseline
#   gpu-big   : concurrency=16, batch=256  -> ~270/s and climbing, GPU at 39%
#               utilisation — still headroom, this is a request-handling
#               ceiling (async/JSON overhead for a 33M-param model), not a
#               compute ceiling. A dual-GPU split is the next lever past this,
#               not a bigger single-GPU concurrency number.
#
# "Small" vs "big" is about how much concurrent request-handling overhead the
# server can absorb before the GPU itself becomes the limit — which tracks
# with card class more than raw VRAM. A 8-12GB consumer card (4060 Ti, 3080)
# is "small"; a 24GB+ card (4090, 5090, R9700, W7900) is "big". When unsure,
# start at gpu-small and watch `rocm-smi --showuse` / `nvidia-smi` — under
# ~50% GPU utilisation with requests queuing means the card has headroom for
# gpu-big or higher.
_EMBED_TIER_PRESETS = {
    "cpu":       {"concurrency": 0,  "batch": 64},
    "gpu-small": {"concurrency": 6,  "batch": 256},
    "gpu-big":   {"concurrency": 16, "batch": 256},
}

_embed_tier = os.environ.get("EMBED_TIER", "").strip().lower()
if _embed_tier and _embed_tier not in _EMBED_TIER_PRESETS:
    raise SystemExit(
        f"EMBED_TIER={_embed_tier!r} is not one of {list(_EMBED_TIER_PRESETS)}")
_tier_defaults = _EMBED_TIER_PRESETS.get(
    _embed_tier, _EMBED_TIER_PRESETS["cpu" if not EMBED_URL else "gpu-small"])

# How many embedding requests are in flight at once against EMBED_URL. Only
# matters on the remote/GPU path. EMBED_CONCURRENCY, if set, overrides
# whatever the tier picked — the tier is a starting point, not a ceiling.
EMBED_CONCURRENCY = int(os.environ.get("EMBED_CONCURRENCY")
                        or _tier_defaults["concurrency"])
EMBED_BATCH = int(os.environ.get("EMBED_BATCH") or _tier_defaults["batch"])

HF_HOME = os.environ.get("HF_HOME", "/data/models")
OFFLINE = os.environ.get("OFFLINE", "1") == "1"
GENS = [int(g) for g in os.environ.get("GENS", "9").split(",") if g.strip()]

# --- embedding thread count (CPU tier only) --------------------------------
# ONNX Runtime grabs every visible core by default. On a high-core-count host
# that is counterproductive: the batches here are small, so past roughly 16
# threads the pool spends more time synchronizing than computing, and it
# monopolizes the box meanwhile.
#
# Default to min(16, cores). Override with EMBED_THREADS or --threads.
# An explicitly set OMP_NUM_THREADS wins — setdefault never clobbers it.
_CPUS = os.cpu_count() or 8
EMBED_THREADS = int(os.environ.get("EMBED_THREADS") or min(16, _CPUS))


def _apply_thread_limit(n: int) -> None:
    """Must run before onnxruntime and numpy load — they read these at import."""
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ.setdefault(var, str(n))


_apply_thread_limit(EMBED_THREADS)

DATA_DIR = Path(DB_PATH).parent
EMB_PATH = DATA_DIR / "embeddings.f16.npy"
EMB_IDS_PATH = DATA_DIR / "embeddings_ids.npy"

# Always write a log file as well as stdout. A long ingest usually outlives the
# terminal that started it, and losing the output to a dropped SSH session is a
# miserable way to find out what went wrong.
LOG_PATH = os.environ.get("INGEST_LOG") or str(DATA_DIR / "ingest.log")
_LOGFH = None


def _open_log():
    global _LOGFH
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        _LOGFH = open(LOG_PATH, "a", buffering=1, encoding="utf-8")
        _LOGFH.write(f"\n{'=' * 70}\n")
        _LOGFH.write(f"ingest started {time.strftime('%Y-%m-%d %H:%M:%S')} "
                     f"(pid {os.getpid()})\n")
        _LOGFH.write(f"{'=' * 70}\n")
    except Exception as e:
        print(f"(could not open log file {LOG_PATH}: {e})", flush=True)
        _LOGFH = None

# Scaling constants, measured from a real run: a 2.4 GB Bulbapedia ZIM produced
# 436,005 chunks, 128,795 aliases, and a ~6 GB database. Used by preflight() to
# predict disk and runtime before doing any work.
CHUNKS_PER_GB = 182_000
DB_GB_PER_ZIM_GB = 2.6
VEC_MB_PER_ZIM_GB = 140
# Measured: all nine generations in 25 seconds. The dex stage is negligible —
# an earlier 17-minute figure was a misreading of process start time, not a
# measurement, and it wrongly made wide generation scopes look expensive.
DEX_SEC_PER_GEN = 3.0

TARGET_CHARS = 1200      # aim for chunks around this size
MIN_CHARS = 120          # drop chunks shorter than this
LEAD_MIN_CHARS = 40      # ...except the lead section, which is the summary
MAX_CHARS = 2400         # hard ceiling before forced split

SKIP_PREFIXES = (
    "Talk:", "User:", "User talk:", "File:", "File talk:", "Category talk:",
    "Template:", "Template talk:", "Help:", "Help talk:", "Bulbapedia:",
    "Bulbapedia talk:", "Bulbanews:", "Forum:", "Special:", "MediaWiki:",
    "Project:", "Portal:", "Module:",
)

ROMAN = {
    "I": 1, "II": 2, "III": 3, "IV": 4, "V": 5,
    "VI": 6, "VII": 7, "VIII": 8, "IX": 9, "X": 10,
}
GEN_RE = re.compile(r"\bGeneration\s+(IX|IV|V?I{0,3}|VI{0,3}|X)\b")

# Only definitional phrasing counts when reading body text. "introduced in
# Generation IX" is about the subject; "For Generation III and IV games, ignore
# Hidden Abilities" is page boilerplate and must not tag the chunk.
GEN_INTRO_RE = re.compile(
    r"\b(?:introduced|debuted|added|released|first appeared)\s+in\s+"
    r"Generation\s+(?P<gen>IX|IV|V?I{0,3}|VI{0,3}|X)\b",
    re.IGNORECASE,
)

DOMAIN_RULES = (
    ("anime", ("in the anime", "anime", "pokémon the series", "pokemon the series")),
    ("manga", ("in the manga", "manga", "pokémon adventures", "pokemon adventures")),
    ("tcg", ("in the tcg", "trading card game", "tcg")),
    ("games", ("in the games", "game data", "game locations", "in the core series",
               "learnset", "base stats", "pokédex entries", "pokedex entries")),
)


LOCK_PATH = Path(os.environ.get("INGEST_LOCK") or (DATA_DIR / "ingest.lock"))


def acquire_lock(force: bool = False) -> None:
    """
    Refuse to start if another ingest is already running.

    Two concurrent ingests destroy each other's work: every stage begins by
    DELETEing its own scope, so a second run started three minutes into the
    first wipes everything the first has written. The counters keep climbing
    and the run reports success, so the damage is silent — one real build lost
    every article from B through I this way and nobody noticed for a day.

    The lock holds a PID. A stale lock from a killed process is detected and
    cleared automatically, so this never needs manual cleanup in the normal case.
    """
    if LOCK_PATH.exists():
        try:
            old = int(LOCK_PATH.read_text().split()[0])
        except Exception:
            old = None

        if old and _pid_alive(old):
            log("")
            log(f"  !! ANOTHER INGEST IS ALREADY RUNNING (pid {old}).")
            log("")
            log("     Two ingests will overwrite each other and silently corrupt")
            log("     the database. Refusing to start.")
            log("")
            log("     Check it:   docker top pokedex-api")
            log(f"     Stop it:    kill -INT {old}")
            log("     Override:   ingest.py --force-unlock   (only if you are")
            log("                 certain no other ingest is running)")
            log("")
            if not force:
                raise SystemExit(1)
            log("     --force-unlock given; proceeding anyway.")
        else:
            log(f"  (clearing stale lock from pid {old}, no longer running)")

    try:
        LOCK_PATH.write_text(f"{os.getpid()} {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n")
    except Exception as e:
        log(f"  (could not write lock file: {e} — continuing without it)")


def release_lock() -> None:
    try:
        if LOCK_PATH.exists():
            pid = int(LOCK_PATH.read_text().split()[0])
            if pid == os.getpid():
                LOCK_PATH.unlink()
    except Exception:
        pass


def _pid_alive(pid: int) -> bool:
    """Signal 0 checks existence without actually signalling."""
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True          # exists, owned by someone else
    except Exception:
        return False


def preflight(db: sqlite3.Connection, need_wiki: bool) -> None:
    """
    Predict how big this is going to get, and refuse to start if the disk can't
    take it. Measured from a real run: a 2.4 GB Bulbapedia ZIM produces ~436k
    chunks and a ~6 GB database, and SQLite's WAL can transiently match the
    database size before a checkpoint.
    """
    import shutil

    stage("Preflight")

    if need_wiki and ZIM_PATH and Path(ZIM_PATH).exists():
        zim_gb = Path(ZIM_PATH).stat().st_size / 1e9
        est_chunks = int(zim_gb * CHUNKS_PER_GB)
        est_db_gb = zim_gb * DB_GB_PER_ZIM_GB
        est_vec_mb = zim_gb * VEC_MB_PER_ZIM_GB
        # WAL can grow to roughly the database size between checkpoints.
        est_peak_gb = est_db_gb * 2 + est_vec_mb / 1000

        log(f"  ZIM:               {zim_gb:.1f} GB  ({ZIM_PATH})")
        log(f"  estimated chunks:  ~{est_chunks:,}")
        log(f"  estimated db:      ~{est_db_gb:.1f} GB")
        log(f"  estimated vectors: ~{est_vec_mb:.0f} MB")
        log(f"  estimated peak:    ~{est_peak_gb:.1f} GB (db + WAL + vectors)")
        log(f"  generations:       {GENS}")
        log(f"  estimated runtime: ~{_fmt_runtime(zim_gb)}")
        if len(GENS) > 1:
            log(f"  (dex runs once per generation — {len(GENS)} here, "
                f"a few seconds each)")
    else:
        est_peak_gb = 1.0
        log("  (no wiki stage — small footprint)")

    try:
        free_gb = shutil.disk_usage(DATA_DIR).free / 1e9
    except Exception as e:
        log(f"  ? could not check free space: {e}")
        return

    log(f"  free on {DATA_DIR}: {free_gb:.1f} GB")

    if free_gb < est_peak_gb:
        log("")
        log(f"  ! NOT ENOUGH DISK. Need roughly {est_peak_gb:.1f} GB, have "
            f"{free_gb:.1f} GB.")
        log("    The ingest would fail partway and leave a half-built database.")
        log("    Free up space, or point DB_PATH at a larger volume.")
        log("    Override with --skip-space-check if you think this is wrong.")
        raise SystemExit(1)

    if free_gb < est_peak_gb * 1.3:
        log(f"  ! Tight on space — {free_gb:.1f} GB free against a ~{est_peak_gb:.1f} GB")
        log("    estimate. It should fit, but there is little margin.")


def _fmt_runtime(zim_gb: float) -> str:
    """
    Measured: ~11 min/GB for the wiki pass, and ~17 min for a single generation
    of dex. Older generations hold fewer species, so the dex cost grows
    sub-linearly with generation count rather than 17x for nine gens.
    """
    wiki_min = zim_gb * 11
    embed_min = (zim_gb * CHUNKS_PER_GB) / max(EMBED_THREADS, 1) / 1000
    dex_min = (DEX_SEC_PER_GEN * sum(_gen_weight(g) for g in GENS)) / 60
    total = dex_min + 3 + wiki_min + 2 + embed_min
    return f"{total * 0.7:.0f}-{total * 1.4:.0f} min"


def _gen_weight(gen: int) -> float:
    """Species count for a generation, relative to gen 9. Older gens are smaller,
    so nine generations cost far less than nine times one."""
    counts = {1: 151, 2: 251, 3: 386, 4: 493, 5: 649,
              6: 721, 7: 809, 8: 905, 9: 1025}
    return counts.get(gen, 1025) / 1025


def checkpoint(db: sqlite3.Connection) -> None:
    """
    Fold the WAL back into the main database between stages.

    Without this the WAL grows to roughly the database size during a long write
    and effectively doubles the disk requirement — a 6 GB database sat next to a
    6 GB WAL on the run this was measured from.
    """
    try:
        db.commit()
        db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    except Exception as e:
        log(f"  (checkpoint skipped: {e})")


def log(msg: str) -> None:
    line = f"{time.strftime('%H:%M:%S')}  {msg}"
    print(line, flush=True)
    if _LOGFH:
        try:
            _LOGFH.write(line + "\n")
        except Exception:
            pass


def stage(msg: str) -> None:
    log("")
    log(f"=== {msg} ===")


def set_status(db, stage_name: str, detail: str = "",
               done: int | None = None, total: int | None = None) -> None:
    """
    Publish progress into the meta table so /health can report which stage is
    running. Cheap, and it means you don't have to keep a terminal open to know
    whether an ingest is alive.
    """
    payload: dict = {
        "stage": stage_name,
        "detail": detail,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "pid": os.getpid(),
    }
    if total:
        payload["done"] = done
        payload["total"] = total
        payload["pct"] = round(100.0 * (done or 0) / total, 1)
    try:
        set_meta(db, "ingest_status", payload)
        db.commit()
    except Exception:
        pass


def norm(s: str) -> str:
    """Aggressive normalization for alias matching."""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]", "", s.lower())


# ---------------------------------------------------------------------------
# schema
# ---------------------------------------------------------------------------

SCHEMA = """
PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);

CREATE TABLE IF NOT EXISTS species (
    id TEXT, gen INTEGER, name TEXT, num INTEGER,
    types TEXT, hp INTEGER, atk INTEGER, def_ INTEGER,
    spa INTEGER, spd INTEGER, spe INTEGER, bst INTEGER,
    abilities TEXT, base_species TEXT, forme TEXT,
    -- battle_only: this exact species can't be picked as its own team-builder
    -- slot, only appears once a battle starts (Aegislash-Blade AND Mega
    -- Charizard X both have this — it does NOT distinguish "automatic" from
    -- "player-chosen"). required_item is what actually does: set to a real
    -- item name for anything Mega-Evolution-style (a player's deliberate
    -- choice, holding a Mega Stone), and NULL for anything ability/move
    -- triggered instead (Aegislash's Stance Change, Mimikyu's Disguise,
    -- Meloetta's Relic Song) — confirmed directly against live @pkmn/dex
    -- data for both categories before relying on it.
    battle_only INTEGER, required_item TEXT,
    prevo TEXT, evos TEXT, evo_level INTEGER, evo_item TEXT,
    evo_condition TEXT, evo_type TEXT, egg_groups TEXT,
    weight_kg REAL, height_m REAL, tier TEXT, doubles_tier TEXT,
    nfe INTEGER,
    PRIMARY KEY (id, gen)
);
CREATE INDEX IF NOT EXISTS idx_species_name ON species(name);
CREATE INDEX IF NOT EXISTS idx_species_num  ON species(num);

CREATE TABLE IF NOT EXISTS moves (
    id TEXT, gen INTEGER, name TEXT, num INTEGER, type TEXT,
    category TEXT, base_power INTEGER, accuracy TEXT, pp INTEGER,
    priority INTEGER, target TEXT, flags TEXT,
    short_desc TEXT, description TEXT,
    PRIMARY KEY (id, gen)
);

CREATE TABLE IF NOT EXISTS abilities (
    id TEXT, gen INTEGER, name TEXT, num INTEGER,
    short_desc TEXT, description TEXT,
    PRIMARY KEY (id, gen)
);

CREATE TABLE IF NOT EXISTS items (
    id TEXT, gen INTEGER, name TEXT, num INTEGER,
    short_desc TEXT, description TEXT,
    mega_evolves TEXT, is_berry INTEGER, is_choice INTEGER,
    PRIMARY KEY (id, gen)
);

-- Parsed from Bulbapedia's "Type-enhancing item" and "Stat-enhancing item"
-- articles (the "List of..." tables within them) — the only two item
-- categories that don't already have a clean, direct field from the dex
-- itself. category is 'type_boost' or 'stat_boost'; detail carries whatever
-- extra column the source table had for that category (the boosted type's
-- name, or the associated species/effect text). Not tied to a generation —
-- these categorizations don't change generation to generation the way stats
-- or legality do.
CREATE TABLE IF NOT EXISTS item_categories (
    item_id TEXT, category TEXT, detail TEXT,
    PRIMARY KEY (item_id, category)
);

CREATE TABLE IF NOT EXISTS learnsets (
    species_id TEXT,
    gen        INTEGER,   -- generation this legality applies to
    move_id    TEXT,
    method     TEXT,      -- level-up, machine, tutor, egg, event, ...
    level      INTEGER,   -- 0 when the method isn't level-up
    source_gen INTEGER,   -- generation the source itself comes from
    PRIMARY KEY (species_id, gen, move_id, method, level, source_gen)
);
CREATE INDEX IF NOT EXISTS idx_learn_move   ON learnsets(move_id, gen);
CREATE INDEX IF NOT EXISTS idx_learn_levels ON learnsets(species_id, gen, method, level);

CREATE TABLE IF NOT EXISTS typechart (
    gen INTEGER, defending_type TEXT, attacking_type TEXT,
    multiplier REAL,
    PRIMARY KEY (gen, defending_type, attacking_type)
);

CREATE TABLE IF NOT EXISTS aliases (
    alias_norm   TEXT,
    alias        TEXT,
    canonical_id TEXT,
    kind         TEXT,
    source       TEXT,
    PRIMARY KEY (alias_norm, canonical_id)
);
CREATE INDEX IF NOT EXISTS idx_alias_canon ON aliases(canonical_id);

CREATE TABLE IF NOT EXISTS home_sprites (
    -- Pokemon HOME's own filename encoding, parsed rather than renamed —
    -- keeping the raw fields means a wrong forme-name guess is a re-run of
    -- the matching step, not a filesystem-wide rename to undo.
    natdex        INTEGER,   -- National Dex number
    form_index    INTEGER,   -- HOME's per-SPECIES form index (000=base). Not
                              -- globally meaningful — confirmed directly:
                              -- 001 means Mega X for Charizard and Alolan for
                              -- Raichu, two unrelated things sharing a number.
    gender        TEXT,      -- mf | md | fd | mo (HOME's own gender/costume flag)
    is_gmax       INTEGER,   -- 0/1
    is_shiny      INTEGER,   -- 0/1
    -- The Showdown forme name (e.g. "Charizard-Mega-X"), filled in ONLY when
    -- confidently matched — see _build_forme_order_map(). NULL means "this
    -- form exists and has an image, but which specific forme it is wasn't
    -- verified" rather than a guess. Confirmed against real @pkmn/dex data
    -- for Charizard, Mewtwo, Gengar, Garchomp, Raichu, Lycanroc, Urshifu:
    -- HOME's numeric form order matches @pkmn/dex's otherFormes order
    -- exactly, once Mega-only formes are looked up in a generation where
    -- Mega Evolution still exists (gen 9 removed the mechanic entirely, so
    -- gen 9's own otherFormes list is empty for e.g. Charizard).
    forme_name    TEXT,
    icon_path     TEXT,      -- relative path under the icons mount
    preview_path  TEXT,      -- relative path under the previews mount;
                              -- nullable — the two folders are NOT in lockstep
                              -- (confirmed: 3035 icon files vs 3029 previews)
    source_file   TEXT,      -- original filename, kept for debugging
    PRIMARY KEY (natdex, form_index, gender, is_gmax, is_shiny)
);
CREATE INDEX IF NOT EXISTS idx_home_natdex ON home_sprites(natdex);

CREATE TABLE IF NOT EXISTS species_images (
    -- Keyed by norm(species name), not the dex id — matching is done at
    -- lookup time by normalizing the dex row's own display name the same
    -- way, since that's the identity the rest of this system already uses
    -- for matching Bulbapedia article titles to dex entities (see the
    -- article->canonical_id linking in stage_aliases). This gets the base
    -- form of every species reliably; regional/Mega/other formes whose
    -- Bulbapedia article title doesn't match their Showdown id 1:1 are the
    -- same known gap wiki_chunks linking already has — not a new limitation.
    species_norm  TEXT PRIMARY KEY,
    species_name  TEXT,
    image_url     TEXT,
    source_title  TEXT
);

CREATE TABLE IF NOT EXISTS wiki_chunks (
    id            INTEGER PRIMARY KEY,
    source        TEXT,
    article_path  TEXT,
    article_title TEXT,
    section_path  TEXT,
    canonical_id  TEXT,
    domain        TEXT,
    gen           INTEGER,
    text          TEXT,
    url           TEXT
);
CREATE INDEX IF NOT EXISTS idx_chunk_canon  ON wiki_chunks(canonical_id);
CREATE INDEX IF NOT EXISTS idx_chunk_domain ON wiki_chunks(domain);
CREATE INDEX IF NOT EXISTS idx_chunk_gen    ON wiki_chunks(gen);
CREATE INDEX IF NOT EXISTS idx_chunk_title  ON wiki_chunks(article_title);

CREATE VIRTUAL TABLE IF NOT EXISTS wiki_fts USING fts5(
    text, article_title, section_path,
    content='wiki_chunks', content_rowid='id',
    tokenize='porter unicode61'
);

CREATE TABLE IF NOT EXISTS sets (
    format TEXT, species_id TEXT, species_name TEXT, set_name TEXT,
    origin TEXT,
    moves TEXT, item TEXT, ability TEXT, nature TEXT,
    evs TEXT, ivs TEXT, tera TEXT, level INTEGER,
    PRIMARY KEY (format, species_id, set_name, origin)
);
CREATE INDEX IF NOT EXISTS idx_sets_species ON sets(species_id, format);

CREATE TABLE IF NOT EXISTS usage_stats (
    format TEXT, month TEXT, cutoff INTEGER, species_id TEXT,
    species_name TEXT, usage REAL, raw_count INTEGER,
    moves TEXT, items TEXT, abilities TEXT, spreads TEXT,
    teammates TEXT, counters TEXT,
    PRIMARY KEY (format, month, cutoff, species_id)
);
CREATE INDEX IF NOT EXISTS idx_usage_lookup ON usage_stats(format, month, usage DESC);
CREATE INDEX IF NOT EXISTS idx_usage_species ON usage_stats(species_id, format);
"""


def connect() -> sqlite3.Connection:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(DB_PATH)
    db.row_factory = sqlite3.Row
    db.executescript(SCHEMA)
    _migrate(db)
    return db


def _migrate(db: sqlite3.Connection) -> None:
    """
    CREATE TABLE IF NOT EXISTS only applies to a table that doesn't exist yet
    — it silently does nothing to an EXISTING table's schema, even when the
    CREATE TABLE text in SCHEMA above has changed. Anything added to an
    existing table (like species.battle_only/required_item) needs an actual
    ALTER TABLE here, checked against PRAGMA table_info first so this is a
    safe no-op on a database that's already been migrated or was freshly
    created with the column already in place.
    """
    cols = {r["name"] for r in db.execute("PRAGMA table_info(species)")}
    for col, coltype in (("battle_only", "INTEGER"), ("required_item", "TEXT")):
        if col not in cols:
            db.execute(f"ALTER TABLE species ADD COLUMN {col} {coltype}")

    item_cols = {r["name"] for r in db.execute("PRAGMA table_info(items)")}
    if "is_choice" not in item_cols:
        db.execute("ALTER TABLE items ADD COLUMN is_choice INTEGER")

    db.commit()


def set_meta(db: sqlite3.Connection, key: str, value) -> None:
    db.execute(
        "INSERT INTO meta(key, value) VALUES(?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (key, json.dumps(value) if not isinstance(value, str) else value),
    )


# ---------------------------------------------------------------------------
# stage 1 — Showdown dex, via the Node sidecar
# ---------------------------------------------------------------------------

def stage_dex(db: sqlite3.Connection) -> None:
    stage("Stage 1 / 5 — Showdown dex")
    set_status(db, "dex", "fetching dump from pokedex-sim")

    import httpx

    for gen in GENS:
        gen_t0 = time.time()
        url = f"{SIM_URL}/dump/{gen}"
        log(f"  fetching {url}")
        try:
            with httpx.Client(timeout=600.0) as client:
                r = client.get(url)
                r.raise_for_status()
                dump = r.json()
        except Exception as e:
            log(f"  ! could not fetch the dex dump: {e}")
            log(f"    Is pokedex-sim running?  curl {SIM_URL}/health")
            raise SystemExit(1)

        t_fetch = time.time() - gen_t0
        log(f"  fetch + parse: {t_fetch:.1f}s")

        counts = dump.get("counts", {})
        log(f"  gen {gen}: " + ", ".join(f"{v} {k}" for k, v in counts.items()))

        t = time.time()
        for table in ("species", "moves", "abilities", "items", "learnsets", "typechart"):
            db.execute(f"DELETE FROM {table} WHERE gen = ?", (gen,))
        t_delete = time.time() - t

        t = time.time()
        db.executemany(
            """INSERT OR REPLACE INTO species
               (id,gen,name,num,types,hp,atk,def_,spa,spd,spe,bst,
                abilities,base_species,forme,battle_only,required_item,
                prevo,evos,evo_level,evo_item,
                evo_condition,evo_type,egg_groups,weight_kg,height_m,
                tier,doubles_tier,nfe)
               VALUES
               (:id,:gen,:name,:num,:types,:hp,:atk,:def_,:spa,:spd,:spe,:bst,
                :abilities,:base_species,:forme,:battle_only,:required_item,
                :prevo,:evos,:evo_level,:evo_item,
                :evo_condition,:evo_type,:egg_groups,:weight_kg,:height_m,
                :tier,:doubles_tier,:nfe)""",
            [
                {
                    "id": s["id"], "gen": gen, "name": s["name"], "num": s.get("num"),
                    "types": json.dumps(s.get("types") or []),
                    "hp": (s.get("base_stats") or {}).get("hp"),
                    "atk": (s.get("base_stats") or {}).get("atk"),
                    "def_": (s.get("base_stats") or {}).get("def"),
                    "spa": (s.get("base_stats") or {}).get("spa"),
                    "spd": (s.get("base_stats") or {}).get("spd"),
                    "spe": (s.get("base_stats") or {}).get("spe"),
                    "bst": s.get("bst"),
                    "abilities": json.dumps(s.get("abilities") or {}),
                    "base_species": s.get("base_species"),
                    "forme": s.get("forme"),
                    "battle_only": 1 if s.get("battle_only") else 0,
                    "required_item": s.get("required_item") if isinstance(s.get("required_item"), str) else None,
                    "prevo": s.get("prevo"),
                    "evos": json.dumps(s.get("evos") or []),
                    "evo_level": s.get("evo_level"),
                    "evo_item": s.get("evo_item"),
                    "evo_condition": s.get("evo_condition"),
                    "evo_type": s.get("evo_type"),
                    "egg_groups": json.dumps(s.get("egg_groups") or []),
                    "weight_kg": s.get("weight_kg"),
                    "height_m": s.get("height_m"),
                    "tier": s.get("tier"),
                    "doubles_tier": s.get("doubles_tier"),
                    "nfe": 1 if s.get("nfe") else 0,
                }
                for s in _dedupe(dump.get("species"), "species", gen)
            ],
        )
        t_species = time.time() - t

        t = time.time()
        db.executemany(
            """INSERT OR REPLACE INTO moves VALUES
               (:id,:gen,:name,:num,:type,:category,:base_power,:accuracy,:pp,
                :priority,:target,:flags,:short_desc,:description)""",
            [
                {
                    "id": m["id"], "gen": gen, "name": m["name"], "num": m.get("num"),
                    "type": m.get("type"), "category": m.get("category"),
                    "base_power": m.get("base_power"),
                    "accuracy": str(m.get("accuracy")),
                    "pp": m.get("pp"), "priority": m.get("priority"),
                    "target": m.get("target"),
                    "flags": json.dumps(m.get("flags") or {}),
                    "short_desc": m.get("short_desc"),
                    "description": m.get("desc"),
                }
                for m in _dedupe(dump.get("moves"), "moves", gen)
            ],
        )
        t_moves = time.time() - t

        t = time.time()
        db.executemany(
            "INSERT OR REPLACE INTO abilities VALUES (:id,:gen,:name,:num,:short_desc,:description)",
            [
                {"id": a["id"], "gen": gen, "name": a["name"], "num": a.get("num"),
                 "short_desc": a.get("short_desc"), "description": a.get("desc")}
                for a in _dedupe(dump.get("abilities"), "abilities", gen)
            ],
        )

        db.executemany(
            """INSERT OR REPLACE INTO items
               (id,gen,name,num,short_desc,description,mega_evolves,is_berry,is_choice)
               VALUES
               (:id,:gen,:name,:num,:short_desc,:description,:mega_evolves,:is_berry,:is_choice)""",
            [
                {"id": i["id"], "gen": gen, "name": i["name"], "num": i.get("num"),
                 "short_desc": i.get("short_desc"), "description": i.get("desc"),
                 "mega_evolves": i.get("mega_evolves"),
                 "is_berry": 1 if i.get("is_berry") else 0,
                 "is_choice": 1 if i.get("is_choice") else 0}
                for i in _dedupe(dump.get("items"), "items", gen)
            ],
        )

        t_small = time.time() - t

        t = time.time()
        rows = []
        unparsed = 0
        for sid, moves in (dump.get("learnsets") or {}).items():
            if not isinstance(moves, dict):
                # Older dump format: a bare list of move names, no level data.
                for mid in (moves or []):
                    rows.append((sid, gen, mid, "unknown", 0, gen))
                continue
            for mid, sources in moves.items():
                if not isinstance(sources, list):
                    sources = [sources]
                for src in sources:
                    parsed = _parse_source(src)
                    if parsed:
                        sgen, method, level = parsed
                        rows.append((sid, gen, mid, method, level, sgen))
                    else:
                        unparsed += 1
        db.executemany(
            "INSERT OR IGNORE INTO learnsets VALUES (?,?,?,?,?,?)", rows)
        t_learn = time.time() - t
        lvl = sum(1 for r in rows if r[3] == "level-up")
        log(f"  gen {gen}: {len(rows)} learnset entries ({lvl} with a level)"
            + (f", {unparsed} unparsed sources" if unparsed else ""))

        # Showdown encodes damageTaken as 0=normal 1=weak(2x) 2=resist(0.5x) 3=immune
        mult = {0: 1.0, 1: 2.0, 2: 0.5, 3: 0.0}
        tc_rows = []
        for defending, taken in (dump.get("typechart") or {}).items():
            for attacking, code in (taken or {}).items():
                if isinstance(code, int) and code in mult:
                    tc_rows.append((gen, defending, attacking, mult[code]))
        db.executemany("INSERT OR REPLACE INTO typechart VALUES (?,?,?,?)", tc_rows)

        t = time.time()
        db.commit()
        t_commit = time.time() - t

        log(f"  timings — fetch {t_fetch:.1f}s | delete {t_delete:.1f}s | "
            f"species {t_species:.1f}s | moves {t_moves:.1f}s | "
            f"abilities+items {t_small:.1f}s | learnsets {t_learn:.1f}s | "
            f"commit {t_commit:.1f}s | TOTAL {time.time() - gen_t0:.1f}s")

        set_meta(db, f"dex_generated_at_gen{gen}", dump.get("generated_at", ""))

    set_meta(db, "dex_gens", GENS)
    set_meta(db, "dex_ingested_at", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    db.commit()
    log("  done")


# ---------------------------------------------------------------------------
# stage 2 — Smogon usage statistics
# ---------------------------------------------------------------------------

CHAOS_NAME_RE = re.compile(r"^(?P<fmt>[a-z0-9]+)-(?P<cut>\d+)-(?P<month>\d{4}-\d{2})\.json$")


def _iter_stats_files() -> Iterator[tuple[Path, str, int, str]]:
    """Yield (path, format, cutoff, month) for every chaos file in STATS_DIR."""
    d = Path(STATS_DIR)
    if not d.is_dir():
        return
    for p in sorted(d.glob("*.json")):
        m = CHAOS_NAME_RE.match(p.name)
        if m:
            yield p, m["fmt"], int(m["cut"]), m["month"]
        else:
            log(f"  ? skipping {p.name} — expected <format>-<cutoff>-<YYYY-MM>.json")


def stage_stats(db: sqlite3.Connection) -> None:
    stage("Stage 2 / 5 — Smogon usage statistics")
    set_status(db, "stats", "reading chaos JSON")

    files = list(_iter_stats_files())

    if not files:
        if STATS_DIR:
            log(f"  no files in {STATS_DIR}")
            log("  Run ./fetch-stats.sh, or copy chaos JSON in by hand.")
        if OFFLINE:
            log("  OFFLINE=1 — not fetching. Skipping this stage.")
            log("  Everything except competitive metagame questions still works.")
            return
        log("  STATS_DIR empty and OFFLINE=0 — fetching is not done by this script.")
        log("  Use ./fetch-stats.sh so the rate limiting and conditional requests apply.")
        return

    total = 0
    for path, fmt, cutoff, month in files:
        try:
            blob = json.loads(path.read_text())
        except Exception as e:
            log(f"  ! {path.name}: {e}")
            continue

        data = blob.get("data") or {}
        if not data:
            log(f"  ? {path.name}: no data block")
            continue

        db.execute(
            "DELETE FROM usage_stats WHERE format=? AND month=? AND cutoff=?",
            (fmt, month, cutoff),
        )

        rows = []
        for name, d in data.items():
            rows.append({
                "format": fmt, "month": month, "cutoff": cutoff,
                "species_id": norm(name), "species_name": name,
                "usage": d.get("usage"),
                "raw_count": d.get("Raw count"),
                "moves": json.dumps(_top(d.get("Moves"), 12)),
                "items": json.dumps(_top(d.get("Items"), 8)),
                "abilities": json.dumps(_top(d.get("Abilities"), 5)),
                "spreads": json.dumps(_top(d.get("Spreads"), 6)),
                "teammates": json.dumps(_top(d.get("Teammates"), 10)),
                "counters": json.dumps(_counters(d.get("Checks and Counters"), 10)),
            })

        db.executemany(
            """INSERT INTO usage_stats VALUES
               (:format,:month,:cutoff,:species_id,:species_name,:usage,:raw_count,
                :moves,:items,:abilities,:spreads,:teammates,:counters)""",
            rows,
        )
        total += len(rows)
        log(f"  {fmt} {month} (cutoff {cutoff}): {len(rows)} entries")

    set_meta(db, "stats_ingested_at", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    months = sorted({m for _, _, _, m in files})
    if months:
        set_meta(db, "stats_months", months)
        set_meta(db, "stats_latest_month", months[-1])
    db.commit()
    log(f"  done — {total} rows across {len(files)} files")


# Showdown learnset source strings: <gen><method><level?>
#   "4L32" -> gen 4, level-up at 32      "9M" -> gen 9, TM/HM/TR
#   "7T"   -> gen 7, move tutor          "6E" -> gen 6, egg move
LEARN_METHODS = {
    "L": "level-up",
    "M": "machine",
    "T": "tutor",
    "E": "egg",
    "S": "event",
    "D": "dream world",
    "V": "virtual console",
    "C": "prior evolution",
    "R": "reminder",
}

SOURCE_RE = re.compile(r"^(?P<gen>\d)(?P<method>[A-Z])(?P<level>\d*)")


def _dedupe(items: list, label: str, gen: int, key: str = "id") -> list:
    """
    The dex dump can contain the same id twice within one generation — some
    generations carry both a base entry and a generation-specific override. A
    bare INSERT against a (id, gen) primary key blows up on those, so keep the
    last occurrence and say how many were folded.
    """
    seen = {}
    for it in items or []:
        k = it.get(key)
        if k is not None:
            seen[k] = it
    dropped = len(items or []) - len(seen)
    if dropped:
        log(f"  (gen {gen}: folded {dropped} duplicate {label})")
    return list(seen.values())


def _parse_source(src: str) -> tuple[int, str, int] | None:
    """'4L32' -> (4, 'level-up', 32). Returns None for anything unparseable."""
    m = SOURCE_RE.match(str(src))
    if not m:
        return None
    method = LEARN_METHODS.get(m["method"], m["method"].lower())
    level = int(m["level"]) if m["level"] else 0
    return int(m["gen"]), method, level


def _top(d, n):
    """Chaos values are weighted counts; convert to a ranked percentage list."""
    if not isinstance(d, dict) or not d:
        return []
    total = sum(v for v in d.values() if isinstance(v, (int, float))) or 1
    items = sorted(d.items(), key=lambda kv: kv[1], reverse=True)[:n]
    return [{"name": k, "pct": round(100.0 * v / total, 2)} for k, v in items]


def _counters(d, n):
    """Checks and Counters: [n, koed+switched, stddev]. Higher = stronger check."""
    if not isinstance(d, dict) or not d:
        return []
    out = []
    for k, v in d.items():
        if isinstance(v, list) and len(v) >= 2:
            out.append({"name": k, "score": round(float(v[1]), 3)})
    out.sort(key=lambda x: x["score"], reverse=True)
    return out[:n]


# ---------------------------------------------------------------------------
# stage 3 — Bulbapedia, from the ZIM
# ---------------------------------------------------------------------------

def _open_zim():
    try:
        from libzim.reader import Archive
    except ImportError:
        log("  ! libzim is not installed. Add it to requirements.txt.")
        raise SystemExit(1)

    if not ZIM_PATH or not Path(ZIM_PATH).exists():
        log(f"  ! ZIM not found at {ZIM_PATH!r}")
        log("    Check ZIM_PATH — it is the path INSIDE the container, under /zim.")
        raise SystemExit(1)

    return Archive(ZIM_PATH)


def _entry_iter(zim) -> Iterator:
    """
    python-libzim has moved this API around between versions, so probe for
    whichever accessor this build exposes rather than assuming one.
    """
    count = getattr(zim, "all_entry_count", None) or getattr(zim, "entry_count", 0)
    getter = None
    for name in ("_get_entry_by_id", "get_entry_by_id"):
        if hasattr(zim, name):
            getter = getattr(zim, name)
            break
    if getter is None:
        log("  ! This python-libzim build exposes no entry-by-id accessor.")
        log(f"    Available: {[a for a in dir(zim) if 'entry' in a.lower()]}")
        raise SystemExit(1)

    for i in range(count):
        try:
            yield getter(i)
        except Exception:
            continue


def _skip_title(title: str) -> bool:
    return (not title) or title.startswith(SKIP_PREFIXES)


def _kiwix_url(path: str) -> str:
    if KIWIX_URL and KIWIX_BOOK:
        return f"{KIWIX_URL}/content/{KIWIX_BOOK}/{path}"
    return ""


# Bulbapedia's species article convention, confirmed against a real article:
# title is exactly "{Name} (Pokémon)" — the parenthetical is literal, with the
# accented e, not a URL-escaped form. The main artwork is the first <img> tag
# in the whole article whose alt attribute matches the species name exactly —
# confirmed against Kingambit's real HTML: the page opens with small unrelated
# icon images (game-version menu bubbles), then the boxed species artwork with
# alt="Kingambit", THEN a smaller thumbnail strip of prior evolution stages.
# Alt-matching, not position or size, is what reliably picks the right one.
_SPECIES_TITLE_RE = re.compile(r"^(.+?)\s*\(Pok[ée]mon\)$")


def _extract_species_image(html: str, species_name: str) -> str | None:
    """
    Parse a species article's raw HTML fresh (not the shared/stripped soup
    from _parse_article) and return the resolved image src for its main
    artwork, or None if no matching <img> is found.

    Kiwix serves this ZIM's images at a URL that is the book's content root
    plus the tag's src with its leading "./" stripped — confirmed with a live
    HTTP request, not assumed: the same _assets_/<hash>/<file>.png path that
    appears verbatim in the article HTML resolves with a 200 when appended
    directly to {KIWIX_URL}/content/{KIWIX_BOOK}/, no per-article relative
    path math needed despite the "./" prefix suggesting otherwise.
    """
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "lxml")
    except Exception:
        return None

    target = species_name.strip().lower()
    for img in soup.find_all("img"):
        alt = (img.get("alt") or "").strip().lower()
        if alt == target:
            src = img.get("src") or ""
            if src.startswith("./"):
                src = src[2:]
            return src or None
    return None


# Pokemon HOME's internal sprite filename scheme, confirmed field-by-field
# against 30 real filenames across three species (Charizard, Raichu,
# Pikachu) spanning both the icon and preview folders — every field lines up
# identically between the two folders, only the prefix differs.
#
# The gender field has two more real values beyond the original mf/md/fd/mo,
# found when ~24% of a real 3035-file archive failed to match and turned out
# to be a genuine pattern, not junk:
#   uk — every natdex it appeared on is a genuinely GENDERLESS species
#        (Groudon, Unown, Necrozma, Silvally, Type: Null, Melmetal,
#        Xurkitree) — "unknown gender" as its own category, not a guess.
#   fo — appeared only on Flabebe/Floette/Alcremie, which are known for
#        cosmetic color/flavor variants rather than a real gender split.
#        Alcremie's example also had a non-zero costume field (00000004,
#        the first non-all-zero value seen anywhere) — a second, independent
#        signal pointing the same direction, since Alcremie is exactly the
#        species with dozens of real cosmetic variants that field would need
#        to distinguish.
_HOME_FILENAME_RE = re.compile(
    r"^poke_(?:icon|capture)_"
    r"(?P<natdex>\d{4})_"
    r"(?P<form>\d{3})_"
    r"(?P<gender>mf|md|fd|mo|uk|fo)_"
    r"(?P<category>n|g)_"
    r"(?P<costume>\d{8})_"
    r"f_"
    r"(?P<shiny>n|r)"
    r"\.png$"
)


def _parse_home_filename(filename: str) -> dict | None:
    """
    Parse one Pokemon HOME sprite filename into its component fields, or
    return None if it doesn't match the known pattern (rather than guessing
    at a malformed name).
    """
    m = _HOME_FILENAME_RE.match(filename)
    if not m:
        return None
    return {
        "natdex": int(m["natdex"]),
        "form_index": int(m["form"]),
        "gender": m["gender"],
        "is_gmax": m["category"] == "g",
        "is_shiny": m["shiny"] == "r",
    }


def _build_forme_order_map(sim_url: str) -> dict[int, list[str]]:
    """
    Per National Dex number, the ordered list of alternate formes — merged
    across every generation that has any, not just one.

    This matters because Mega Evolution was removed as a mechanic starting
    Gen 8, so gen 9's own species data reports an EMPTY otherFormes list for
    Charizard despite Mega Charizard X/Y genuinely existing — confirmed
    directly: querying gen 9 for Charizard's otherFormes returns [], while
    the identical query against gen 6 or 7 returns the two Mega formes in
    the right order. Regional/cosmetic formes go the other way — they
    persist into gen 9's data and would be missed by only checking gen 6/7.
    Scanning every gen and merging (keeping first-seen order, deduplicated)
    is the only way to get one complete list.

    Verified against real @pkmn/dex data for Charizard, Mewtwo, Gengar,
    Garchomp (Mega-era), Raichu, Lycanroc, Urshifu (regional/battle formes):
    HOME's numeric form index matches this merged order exactly, position
    for position, in every case checked.
    """
    import httpx
    order: dict[int, list[str]] = {}
    seen: dict[int, set] = {}
    for gen in (6, 7, 8, 9):
        try:
            r = httpx.get(f"{sim_url}/dump/{gen}", timeout=30.0)
            r.raise_for_status()
            species_list = r.json().get("species", [])
        except Exception as e:
            log(f"  ! could not fetch /dump/{gen} for forme ordering: {e}")
            continue
        for s in species_list:
            num = s.get("num")
            if not num or num <= 0:
                continue
            for forme in (s.get("other_formes") or []):
                order.setdefault(num, [])
                seen.setdefault(num, set())
                if forme not in seen[num]:
                    order[num].append(forme)
                    seen[num].add(forme)
    return order


def _table_to_md(table) -> str:
    rows = []
    for tr in table.find_all("tr"):
        cells = [c.get_text(" ", strip=True) for c in tr.find_all(["th", "td"])]
        # 120 was cutting flavor text mid-word. Long enough for a full Pokedex
        # entry or ability description, short enough that a stray prose cell
        # doesn't swamp the chunk.
        cells = [re.sub(r"\s+", " ", c)[:400] for c in cells]
        if any(cells):
            rows.append(cells)
        if len(rows) > 60:
            rows.append(["…"])
            break
    if not rows:
        return ""
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    out = ["| " + " | ".join(rows[0]) + " |",
           "|" + "---|" * width]
    for r in rows[1:]:
        out.append("| " + " | ".join(r) + " |")
    return "\n".join(out)


def _detect_gen(section_path: str, text: str) -> int | None:
    """
    Tag a chunk with the generation it's about.

    Section headings are trustworthy: "In Generation IV" means the section is
    about gen 4. Body text is not — ability and move pages carry boilerplate
    like "For Generation III and IV games, ignore Hidden Abilities", which would
    otherwise tag a Gen 9 ability as Gen 3 and make it invisible to a correctly
    gen-filtered search.

    So: headings win outright, and body text only counts when the phrasing is
    explicitly definitional.
    """
    m = GEN_RE.search(section_path)
    if m:
        return ROMAN.get(m.group(1))

    m = GEN_INTRO_RE.search(text[:600])
    if m:
        return ROMAN.get(m.group("gen"))

    return None


def _detect_domain(section_path: str) -> str:
    low = section_path.lower()
    for domain, needles in DOMAIN_RULES:
        if any(nd in low for nd in needles):
            return domain
    return "general"


def _split_text(text: str) -> list[str]:
    """Split an over-long section at paragraph boundaries."""
    if len(text) <= MAX_CHARS:
        return [text]
    parts, buf = [], ""
    for para in re.split(r"\n\s*\n", text):
        if len(buf) + len(para) > TARGET_CHARS and buf:
            parts.append(buf.strip())
            buf = para
        else:
            buf = f"{buf}\n\n{para}" if buf else para
    if buf.strip():
        parts.append(buf.strip())
    return parts


def _parse_article(html: str, title: str, path: str) -> list[dict]:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "lxml")

    for junk in soup.find_all(["script", "style", "sup", "nav", "footer"]):
        junk.decompose()
    for cls in ("navbox", "toc", "mw-editsection", "reference", "printfooter",
                "catlinks", "sidebar", "infobox-caption"):
        for el in soup.find_all(class_=cls):
            el.decompose()

    root = soup.find(id="mw-content-text") or soup.body or soup
    chunks: list[dict] = []
    heading_stack: list[str] = []
    buf: list[str] = []

    def flush():
        if not buf:
            return
        text = "\n\n".join(b for b in buf if b.strip()).strip()
        buf.clear()
        # The lead section (before any heading) is the article's summary and is
        # often short — "X is a dual-type A/B Pokémon introduced in Generation N."
        # That single sentence answers a lot of questions, so it gets a lower bar.
        floor = LEAD_MIN_CHARS if not heading_stack else MIN_CHARS
        if len(text) < floor:
            return
        section_path = " → ".join([title] + heading_stack) if heading_stack else title
        for piece in _split_text(text):
            if len(piece) < floor:
                continue
            chunks.append({
                "section_path": section_path,
                "domain": _detect_domain(section_path),
                "gen": _detect_gen(section_path, piece),
                "text": f"{section_path}\n\n{piece}",
            })

    for el in root.find_all(["h2", "h3", "h4", "p", "ul", "ol", "table", "dl"],
                            recursive=True):
        name = el.name
        if name in ("h2", "h3", "h4"):
            flush()
            level = int(name[1]) - 2
            heading = el.get_text(" ", strip=True)
            heading = re.sub(r"\[edit.*?\]", "", heading).strip()
            if not heading:
                continue
            heading_stack[:] = heading_stack[:level] + [heading]
        elif name == "table":
            md = _table_to_md(el)
            if md:
                buf.append(md)
        else:
            txt = el.get_text(" ", strip=True)
            txt = re.sub(r"\s+", " ", txt)
            if txt:
                buf.append(txt)

    flush()
    return chunks


def stage_images(db: sqlite3.Connection) -> None:
    """
    Extract species artwork without touching chunks, aliases, or embeddings.

    stage_wiki has to decode and parse EVERY article's HTML regardless of what
    it contains, because chunking needs the whole document. This stage doesn't
    need that: it can check an entry's TITLE against the species pattern
    BEFORE paying for entry.get_item() + content decode at all, and skip
    straight past the ~92,000+ of ~93,583 entries that can't possibly be a
    species page. Only the ~1,000-1,300 that match get decoded and parsed.
    That is why this is a genuinely separate, fast path rather than just
    "the same work minus the row inserts" — most of stage_wiki's real cost is
    the decode-and-parse step itself, and this mostly skips it.
    """
    stage("Stage — species images (standalone)")
    set_status(db, "images", "opening ZIM")

    zim = _open_zim()
    log(f"  {ZIM_PATH}")

    db.execute("DELETE FROM species_images")
    db.commit()

    checked = matched = found = 0
    image_batch: list[tuple] = []
    t0 = time.time()

    for entry in _entry_iter(zim):
        try:
            title = entry.title or ""
        except Exception:
            continue

        if _skip_title(title):
            continue
        checked += 1

        m = _SPECIES_TITLE_RE.match(title)
        if not m:
            continue
        matched += 1

        try:
            if entry.is_redirect:
                continue
            item = entry.get_item()
            if "html" not in (item.mimetype or ""):
                continue
            html = bytes(item.content).decode("utf-8", errors="replace")
        except Exception:
            continue

        species_name = m.group(1).strip()
        try:
            img_url = _extract_species_image(html, species_name)
        except Exception:
            img_url = None
        if img_url:
            found += 1
            image_batch.append((
                norm(species_name), species_name,
                _kiwix_url(img_url), title,
            ))

        if len(image_batch) >= 500:
            _flush_chunks(db, [], [], image_batch)

    _flush_chunks(db, [], [], image_batch)

    log(f"  checked {checked} article titles, {matched} matched the species "
        f"pattern, {found} had a usable image")
    set_meta(db, "species_images_ingested_at",
             time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    db.commit()
    log(f"  done — {time.time() - t0:.0f}s")


def stage_item_categories(db: sqlite3.Connection) -> None:
    """
    Parse item categories out of two Bulbapedia articles already sitting in
    wiki_chunks: "Type-enhancing item" and "Stat-enhancing item", each of
    which has a "List of..." section formatted as a real markdown table.

    Deliberately narrow in scope. Checked directly (see ingest notes/session
    history) whether item categorization could be derived from @pkmn/dex's
    own fields instead of wiki-parsing: only is_choice, is_berry, and
    mega_evolves turned out to be clean, reliable booleans — everything else
    a competitive player would call a "category" (type-boosting, stat-
    boosting, status orbs, weather rocks, terrain seeds, gems, plates) is
    implemented as executable effect code in real Showdown, not inspectable
    data. Of those remaining categories, only type-boosting and stat-
    boosting turned out to have a clean, reliably-extracted source table on
    Bulbapedia — checked directly, not assumed. Everything else (orbs, seeds,
    gems, plates, one-off items like Leftovers or Rocky Helmet) already has
    accurate, specific short_desc text from the dex itself, confirmed
    sufficient for free-text search rather than needing its own category.

    Requires stage_wiki and stage_dex to have already run — depends on
    wiki_chunks for the source text and items for resolving names to ids.
    Safe to re-run any time either of those refreshes; fast either way.
    """
    tables = [
        ("Type-enhancing item", "List of type-enhancing items", "type_boost"),
        ("Type-enhancing item", "Pokémon-specific type-enhancing items", "type_boost"),
        ("Stat-enhancing item", "List of stat-enhancing items", "stat_boost"),
    ]

    roman = re.compile(r"^[IVXLCDM]+$")

    def parse_item_table(text: str, category: str) -> list[dict]:
        results = []
        for line in text.split("\n"):
            line = line.strip()
            if not line.startswith("|") or set(line) <= set("|- "):
                continue
            cells = [c.strip() for c in line.split("|")]
            cells = [c for c in cells if c]
            if not cells:
                continue
            if cells[0].lower() == "name" or cells[0].startswith("All details"):
                continue
            cells = [c for c in cells if not roman.match(c)]
            if not cells:
                continue
            name = cells[0]
            name = re.sub(r"\s*\*\s*$", "", name)
            name = re.sub(r"\s*\(.*?\)\s*$", "", name).strip()
            detail = " / ".join(cells[1:]) if len(cells) > 1 else None
            if name:
                results.append({"item_id": norm(name), "category": category, "detail": detail})
        return results

    rows: list[dict] = []
    for article_title, section_fragment, category in tables:
        chunk = db.execute(
            "SELECT text FROM wiki_chunks WHERE article_title=? AND section_path LIKE ? LIMIT 1",
            (article_title, f"%{section_fragment}%"),
        ).fetchone()
        if not chunk:
            log(f"  item_categories: {article_title!r} / {section_fragment!r} not found, skipping")
            continue
        parsed = parse_item_table(chunk["text"], category)
        log(f"  item_categories: {article_title!r} / {section_fragment!r} -> {len(parsed)} items")
        rows.extend(parsed)

    if not rows:
        log("  item_categories: nothing parsed — is stage_wiki populated?")
        return

    db.execute("DELETE FROM item_categories")
    db.executemany(
        "INSERT OR REPLACE INTO item_categories (item_id, category, detail) VALUES (:item_id, :category, :detail)",
        rows,
    )
    db.commit()
    log(f"  item_categories: {len(rows)} rows total")


def stage_home_sprites(db: sqlite3.Connection) -> None:
    """
    Catalog Pokemon HOME sprite/preview images from two local folders.

    Optional — skips cleanly if HOME_ICONS_DIR isn't set, same as SETS_DIR.

    The two folders are confirmed NOT in lockstep (3035 icon files vs 3029
    preview files in the real archive this was built against), so a file
    present in one and missing from the other is handled as a real case, not
    an error — icon_path or preview_path can each independently be NULL.

    Forme names are only ever set from the verified multi-gen otherFormes
    match (_build_forme_order_map) — a form index with no match at that
    position stores forme_name=NULL rather than a guessed label. An
    unlabeled alt-form image is still a correct, usable image; a
    wrongly-labeled one is worse than no label at all.
    """
    stage("Stage — Pokemon HOME sprites (optional)")

    if not HOME_ICONS_DIR:
        log("  HOME_ICONS_DIR not set — skipping")
        return
    icons_dir = Path(HOME_ICONS_DIR)
    if not icons_dir.is_dir():
        log(f"  ! HOME_ICONS_DIR={HOME_ICONS_DIR!r} is not a directory — skipping")
        return

    set_status(db, "home_sprites", "building forme order map")
    log("  resolving alt-forme names across gens 6-9 (Mega Evolution only "
        "exists in gens 6-7's own dex data, regional/cosmetic formes only "
        "in later gens — merging both is required to catch everything)")
    forme_map = _build_forme_order_map(SIM_URL)
    log(f"  {len(forme_map)} species have at least one known alternate forme")

    db.execute("DELETE FROM home_sprites")

    # entries[(natdex, form_index, gender, is_gmax, is_shiny)] -> row dict
    entries: dict[tuple, dict] = {}

    def scan(folder: Path, path_field: str, url_prefix: str):
        found = matched = 0
        for f in sorted(folder.glob("*.png")):
            found += 1
            parsed = _parse_home_filename(f.name)
            if not parsed:
                continue
            matched += 1
            key = (parsed["natdex"], parsed["form_index"], parsed["gender"],
                   parsed["is_gmax"], parsed["is_shiny"])
            row = entries.setdefault(key, {
                "natdex": parsed["natdex"], "form_index": parsed["form_index"],
                "gender": parsed["gender"], "is_gmax": parsed["is_gmax"],
                "is_shiny": parsed["is_shiny"], "forme_name": None,
                "icon_path": None, "preview_path": None, "source_file": f.name,
            })
            row[path_field] = f"{url_prefix}/{f.name}" if url_prefix else f.name
        log(f"  {folder}: {found} files, {matched} parsed "
            f"({found - matched} did not match the known filename pattern)")

    icons_url = f"{HOME_SPRITES_URL}/icons" if HOME_SPRITES_URL else ""
    scan(icons_dir, "icon_path", icons_url)

    if HOME_PREVIEWS_DIR:
        previews_dir = Path(HOME_PREVIEWS_DIR)
        if previews_dir.is_dir():
            previews_url = f"{HOME_SPRITES_URL}/previews" if HOME_SPRITES_URL else ""
            scan(previews_dir, "preview_path", previews_url)
        else:
            log(f"  ! HOME_PREVIEWS_DIR={HOME_PREVIEWS_DIR!r} is not a directory — icons only")
    else:
        log("  HOME_PREVIEWS_DIR not set — icons only, no hero images")

    # Resolve forme_name: form 0 is always the base form (name=None is
    # correct — the species' own name already covers it). Form N>0 looks up
    # position N-1 in that species' merged forme list; if the list doesn't
    # reach that far, it stays unlabeled rather than guessed.
    labeled = 0
    for row in entries.values():
        if row["form_index"] == 0:
            continue
        formes = forme_map.get(row["natdex"], [])
        idx = row["form_index"] - 1
        if 0 <= idx < len(formes):
            row["forme_name"] = formes[idx]
            labeled += 1

    rows = [
        (r["natdex"], r["form_index"], r["gender"], int(r["is_gmax"]), int(r["is_shiny"]),
         r["forme_name"], r["icon_path"], r["preview_path"], r["source_file"])
        for r in entries.values()
    ]
    db.executemany(
        "INSERT OR REPLACE INTO home_sprites VALUES (?,?,?,?,?,?,?,?,?)", rows)
    db.commit()

    alt_forms = sum(1 for r in entries.values() if r["form_index"] > 0)
    log(f"  {len(entries)} unique sprite entries "
        f"({alt_forms} alternate forms, {labeled} confidently labeled, "
        f"{alt_forms - labeled} unlabeled)")
    set_meta(db, "home_sprites_ingested_at",
             time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))


def stage_wiki(db: sqlite3.Connection, limit: int | None = None) -> None:
    stage("Stage 3 / 5 — Bulbapedia")
    set_status(db, "wiki", "opening ZIM")

    zim = _open_zim()
    log(f"  {ZIM_PATH}")

    db.execute("DELETE FROM wiki_chunks WHERE source = 'bulbapedia'")
    db.execute("DELETE FROM aliases WHERE source = 'zim'")
    db.execute("DELETE FROM species_images")
    db.commit()

    articles = redirects = written = images_found = 0
    batch: list[tuple] = []
    alias_batch: list[tuple] = []
    image_batch: list[tuple] = []
    t0 = time.time()

    for entry in _entry_iter(zim):
        try:
            title = entry.title or ""
            path = entry.path or ""
        except Exception:
            continue

        if _skip_title(title):
            continue

        # Redirects are free alias data — a whole second source of name variants.
        try:
            if entry.is_redirect:
                target = entry.get_redirect_entry()
                tgt_title = target.title or ""
                if tgt_title and not _skip_title(tgt_title):
                    alias_batch.append(
                        (norm(title), title, f"article:{norm(tgt_title)}", "article", "zim")
                    )
                    redirects += 1
                continue
        except Exception:
            continue

        try:
            item = entry.get_item()
            if "html" not in (item.mimetype or ""):
                continue
            html = bytes(item.content).decode("utf-8", errors="replace")
        except Exception:
            continue

        # Only species articles get an image lookup — this is a small subset
        # (~1,000-1,300) of the ~93,583 total articles, so a second, separate
        # parse here costs almost nothing overall despite not reusing the soup
        # _parse_article builds a moment later.
        m = _SPECIES_TITLE_RE.match(title)
        if m:
            species_name = m.group(1).strip()
            try:
                img_url = _extract_species_image(html, species_name)
            except Exception:
                img_url = None
            if img_url:
                image_batch.append((
                    norm(species_name), species_name,
                    _kiwix_url(img_url), title,
                ))

        articles += 1
        try:
            chunks = _parse_article(html, title, path)
        except Exception as e:
            if articles < 20:
                log(f"  ? parse failed for {title!r}: {e}")
            continue

        url = _kiwix_url(path)
        for c in chunks:
            batch.append((
                "bulbapedia", path, title, c["section_path"],
                None, c["domain"], c["gen"], c["text"], url,
            ))
        written += len(chunks)

        # Article titles themselves are aliases; "X (Pokémon)" -> "X".
        alias_batch.append((norm(title), title, f"article:{norm(title)}", "article", "zim"))
        stripped = re.sub(r"\s*\([^)]*\)\s*$", "", title).strip()
        if stripped and stripped != title:
            alias_batch.append(
                (norm(stripped), stripped, f"article:{norm(title)}", "article", "zim")
            )

        if len(batch) >= 2000:
            images_found += len(image_batch)
            _flush_chunks(db, batch, alias_batch, image_batch)

        if articles % 2000 == 0:
            rate = articles / max(time.time() - t0, 1)
            log(f"  {articles} articles, {written} chunks, {redirects} redirects "
                f"({rate:.0f} art/s)")
            set_status(db, "wiki",
                       f"{articles} articles, {written} chunks ({rate:.0f}/s)")

        if limit and articles >= limit:
            log(f"  --limit {limit} reached")
            break

    images_found += len(image_batch)
    _flush_chunks(db, batch, alias_batch, image_batch)

    log("  rebuilding full-text index...")
    db.execute("INSERT INTO wiki_fts(wiki_fts) VALUES('rebuild')")
    db.commit()

    stored = db.execute(
        "SELECT COUNT(*) FROM wiki_chunks WHERE source='bulbapedia'").fetchone()[0]
    if stored != written:
        log("")
        log(f"  !! MISMATCH: counted {written} chunks written but {stored} are in")
        log(f"     the table — {written - stored} were lost. This is a bug, not")
        log(f"     a data problem. Do not trust this index; re-run --wiki-only.")
    else:
        log(f"  verified: {stored} chunks stored, matches the counter")

    set_meta(db, "wiki_articles", articles)
    set_meta(db, "wiki_chunks", stored)
    set_meta(db, "wiki_zim_path", ZIM_PATH)
    set_meta(db, "wiki_ingested_at", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    db.commit()
    log(f"  done — {articles} articles, {written} chunks, {redirects} redirects, "
        f"{images_found} species images, {time.time() - t0:.0f}s")


def _flush_chunks(db, batch, alias_batch, image_batch=None):
    """
    Write a batch and VERIFY it landed.

    The original version let an exception here pass silently while the caller's
    counter kept incrementing. A sustained failure dropped a contiguous run of
    articles — every title from B through I in one real ingest — while the run
    reported success and the meta counters claimed 715,664 chunks against
    436,005 actually stored. Never fail quietly here again.
    """
    if batch:
        before = db.execute("SELECT COUNT(*) FROM wiki_chunks").fetchone()[0]
        try:
            db.executemany(
                """INSERT INTO wiki_chunks
                   (source, article_path, article_title, section_path,
                    canonical_id, domain, gen, text, url)
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                batch,
            )
            db.commit()
        except Exception as e:
            log(f"  !! CHUNK INSERT FAILED for {len(batch)} rows: {e}")
            log(f"     first title in batch: {batch[0][2]!r}")
            log(f"     last  title in batch: {batch[-1][2]!r}")
            batch.clear()
            raise
        after = db.execute("SELECT COUNT(*) FROM wiki_chunks").fetchone()[0]
        if after - before != len(batch):
            log(f"  !! CHUNK COUNT MISMATCH: expected +{len(batch)}, "
                f"got +{after - before} (titles {batch[0][2]!r} .. {batch[-1][2]!r})")
        batch.clear()

    if alias_batch:
        try:
            db.executemany("INSERT OR IGNORE INTO aliases VALUES (?,?,?,?,?)",
                           alias_batch)
            db.commit()
        except Exception as e:
            log(f"  !! ALIAS INSERT FAILED for {len(alias_batch)} rows: {e}")
            alias_batch.clear()
            raise
        alias_batch.clear()

    if image_batch:
        try:
            db.executemany(
                "INSERT OR REPLACE INTO species_images VALUES (?,?,?,?)",
                image_batch,
            )
            db.commit()
        except Exception as e:
            log(f"  !! IMAGE INSERT FAILED for {len(image_batch)} rows: {e}")
            image_batch.clear()
            raise
        image_batch.clear()


# ---------------------------------------------------------------------------
# stage 3b — Smogon analyses (optional, local files only)
# ---------------------------------------------------------------------------

def stage_analyses(db: sqlite3.Connection) -> None:
    """
    Smogon Strategy Dex analyses, if you have them locally.

    Drop per-format JSON into ANALYSES_DIR as <format>.json. The @pkmn project
    publishes these; verify the current URL yourself before scripting a fetch —
    this script deliberately does not hardcode one it cannot guarantee.

    Expected shape: {"<Species Name>": {"sets": {...}, "overview": "...", ...}}
    Unknown shapes are skipped rather than guessed at.
    """
    if not ANALYSES_DIR or not Path(ANALYSES_DIR).is_dir():
        return

    stage("Stage 3b — Smogon analyses")
    db.execute("DELETE FROM wiki_chunks WHERE source = 'smogon'")

    written = 0
    for path in sorted(Path(ANALYSES_DIR).glob("*.json")):
        fmt = path.stem
        try:
            blob = json.loads(path.read_text())
        except Exception as e:
            log(f"  ! {path.name}: {e}")
            continue
        if not isinstance(blob, dict):
            log(f"  ? {path.name}: unexpected shape, skipping")
            continue

        rows = []
        for species, body in blob.items():
            if not isinstance(body, dict):
                continue
            parts = []
            for key in ("overview", "comments", "description"):
                v = body.get(key)
                if isinstance(v, str) and v.strip():
                    parts.append(v.strip())
            sets = body.get("sets")
            if isinstance(sets, dict):
                for set_name, sd in sets.items():
                    desc = sd.get("description") if isinstance(sd, dict) else None
                    if isinstance(desc, str) and desc.strip():
                        parts.append(f"[{set_name}] {desc.strip()}")
            text = "\n\n".join(parts).strip()
            if len(text) < MIN_CHARS:
                continue
            section = f"{species} → Smogon analysis → {fmt}"
            rows.append((
                "smogon", f"smogon/{fmt}/{species}", species, section,
                norm(species), "competitive", None,
                f"{section}\n\n{text}", "",
            ))

        db.executemany(
            """INSERT INTO wiki_chunks
               (source, article_path, article_title, section_path,
                canonical_id, domain, gen, text, url)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            rows,
        )
        written += len(rows)
        log(f"  {fmt}: {len(rows)} analyses")

    db.execute("INSERT INTO wiki_fts(wiki_fts) VALUES('rebuild')")
    db.commit()
    log(f"  done — {written} chunks")


def stage_sets(db: sqlite3.Connection) -> None:
    """
    Curated Smogon sets, from Pokemon Showdown's published @smogon/sets mirror.

    Drop the per-format JSON into SETS_DIR. Shape, verified against the live
    files:

        {"dex":   {"<Species>": {"<Set name>": {moves, item?, ability?,
                                                nature?, evs?, ivs?, teraType?}}},
         "stats": {"<Species>": {"Showdown Usage": {...}}}}

    "dex" entries are human-curated sets from the Strategy Dex. "stats" entries
    are derived from ladder usage. Both are stored, tagged by origin, so the
    model can say which kind it is quoting.

    Get them with: fetch-sets.sh, or by hand from
    https://play.pokemonshowdown.com/data/sets/
    """
    if not SETS_DIR or not Path(SETS_DIR).is_dir():
        return

    files = sorted(Path(SETS_DIR).glob("*.json"))
    if not files:
        return

    stage("Stage 3c — Smogon sets")
    set_status(db, "sets", "reading set files")

    db.execute("DELETE FROM sets")
    total = 0

    for path in files:
        fmt = path.stem
        try:
            blob = json.loads(path.read_text())
        except Exception as e:
            log(f"  ! {path.name}: {e}")
            continue
        if not isinstance(blob, dict):
            continue

        rows = []
        for origin in ("dex", "stats"):
            block = blob.get(origin)
            if not isinstance(block, dict):
                continue
            for species, named in block.items():
                if not isinstance(named, dict):
                    continue
                for set_name, sd in named.items():
                    if not isinstance(sd, dict):
                        continue
                    rows.append((
                        fmt, norm(species), species, set_name, origin,
                        json.dumps(_flatten_choices(sd.get("moves"))),
                        _first(sd.get("item")),
                        _first(sd.get("ability")),
                        _first(sd.get("nature")),
                        json.dumps(sd.get("evs") or {}),
                        json.dumps(sd.get("ivs") or {}),
                        _first(sd.get("teraType")),
                        sd.get("level") or 100,
                    ))

        db.executemany(
            "INSERT OR REPLACE INTO sets VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)
        total += len(rows)
        if len(files) <= 20:
            log(f"  {fmt}: {len(rows)} sets")

    if len(files) > 20:
        log(f"  {len(files)} format files processed")

    set_meta(db, "sets_ingested_at", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    set_meta(db, "sets_formats", [p.stem for p in files])
    db.commit()
    log(f"  done — {total} sets across {len(files)} formats")


def _flatten_choices(moves):
    """
    Move slots can hold a list of alternatives: ["Earthquake", ["Ice Beam",
    "Fire Blast"]]. Flatten each slot to a readable 'A / B' string.
    """
    if not isinstance(moves, list):
        return []
    out = []
    for slot in moves:
        if isinstance(slot, list):
            out.append(" / ".join(str(x) for x in slot))
        elif slot:
            out.append(str(slot))
    return out


def _first(v):
    """Some fields are a single value, some a list of options. Take the first."""
    if isinstance(v, list):
        return str(v[0]) if v else None
    return str(v) if v else None


# ---------------------------------------------------------------------------
# stage 4 — alias table and canonical resolution
# ---------------------------------------------------------------------------

def stage_aliases(db: sqlite3.Connection) -> None:
    stage("Stage 4 / 5 — aliases and canonical IDs")
    set_status(db, "aliases", "building alias table")

    db.execute("DELETE FROM aliases WHERE source = 'dex'")

    rows: list[tuple] = []

    def add(alias: str, canonical: str, kind: str):
        n = norm(alias)
        if n:
            rows.append((n, alias, canonical, kind, "dex"))

    # Walk generations newest-first. INSERT OR IGNORE keeps the first spelling
    # seen, so modern names win — but species that only exist in older gens
    # (many formes, Gen 1-8 exclusives) still get an alias instead of being
    # unreachable, which is what happened when only max(GENS) was used.
    for gen in sorted(GENS, reverse=True):
        for r in db.execute(
            "SELECT id, name, base_species, forme FROM species WHERE gen=?", (gen,)
        ):
            add(r["id"], r["id"], "species")
            add(r["name"], r["id"], "species")
            if r["forme"] and r["base_species"]:
                add(f"{r['base_species']} {r['forme']}", r["id"], "species")
                add(f"{r['forme']} {r['base_species']}", r["id"], "species")
            # hyphenated display form: Urshifu-Rapid-Strike
            add(r["name"].replace("-", " "), r["id"], "species")

        for table, kind in (("moves", "move"), ("abilities", "ability"),
                            ("items", "item")):
            for r in db.execute(f"SELECT id, name FROM {table} WHERE gen=?", (gen,)):
                add(r["id"], r["id"], kind)
                add(r["name"], r["id"], kind)

    db.executemany("INSERT OR IGNORE INTO aliases VALUES (?,?,?,?,?)", rows)
    db.commit()
    log(f"  {len(rows)} dex aliases")

    # Resolve wiki chunks to dex entities where the names line up.
    # Bulbapedia's "X (Pokémon)" convention makes species the easy case.
    log("  linking wiki chunks to dex entities...")
    linked = db.execute(
        """
        UPDATE wiki_chunks
           SET canonical_id = (
                SELECT a.canonical_id FROM aliases a
                 WHERE a.source = 'dex'
                   AND a.kind = 'species'
                   AND a.alias_norm = (
                        SELECT REPLACE(LOWER(wiki_chunks.article_title), ' (pokémon)', '')
                   )
                 LIMIT 1
           )
         WHERE source = 'bulbapedia' AND canonical_id IS NULL
        """
    ).rowcount
    db.commit()

    # That SQL-side normalization is crude; redo it properly in Python for the
    # titles it missed.
    unresolved = db.execute(
        """SELECT DISTINCT article_title FROM wiki_chunks
            WHERE source='bulbapedia' AND canonical_id IS NULL"""
    ).fetchall()

    lookup = {
        r["alias_norm"]: r["canonical_id"]
        for r in db.execute(
            "SELECT alias_norm, canonical_id FROM aliases WHERE source='dex' AND kind='species'"
        )
    }

    updates = []
    for r in unresolved:
        title = r["article_title"]
        stripped = re.sub(r"\s*\([^)]*\)\s*$", "", title).strip()
        cid = lookup.get(norm(stripped)) or lookup.get(norm(title))
        if cid:
            updates.append((cid, title))

    db.executemany(
        """UPDATE wiki_chunks SET canonical_id = ?
            WHERE article_title = ? AND source='bulbapedia' AND canonical_id IS NULL""",
        updates,
    )
    db.commit()

    total = db.execute(
        "SELECT COUNT(*) c FROM wiki_chunks WHERE canonical_id IS NOT NULL"
    ).fetchone()["c"]
    log(f"  {len(updates)} titles matched in the second pass")
    log(f"  {total} chunks now carry a canonical ID")

    set_meta(db, "aliases_ingested_at", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    db.commit()


# ---------------------------------------------------------------------------
# stage 5 — embeddings
# ---------------------------------------------------------------------------

# vLLM's pooling endpoint REJECTS input over the model's max_model_len rather
# than truncating it — a 400, not a silent trim. bge-small's ceiling is 512
# tokens. There is no reliable chars-per-token constant for this content —
# markdown tables and punctuation-dense text tokenize far denser than prose,
# so a single fixed character limit either wastes text on plain chunks or
# still overflows on dense ones (verified: 1500 chars was NOT enough for a
# table-heavy chunk, second attempt still got a 400). Shrink iteratively and
# let the server's own accept/reject be the judge, instead of guessing.
_EMBED_TRUNCATE_START = 1500
_EMBED_TRUNCATE_FLOOR = 100
_EMBED_TRUNCATE_STEP = 0.7          # shrink by 30% each retry


def _shrink_for_embedding(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    cut = text[:limit]
    sp = cut.rfind(" ")
    return cut[:sp] if sp > limit * 0.7 else cut


def _embed_remote(texts: list[str], batch: int, concurrency: int = 6) -> "list":
    """
    Embed via an OpenAI-compatible /embeddings endpoint — typically vLLM on the
    GPU, but anything speaking that API works. Runs multiple batches CONCURRENTLY
    rather than one HTTP round-trip at a time.

    Why concurrency: the synchronous version measured 73 chunks/s against a
    server independently reporting 19,547 prompt tokens/s of throughput — the
    GPU was mostly idle, waiting on the network round-trip between each batch
    the client sent. With 22% of chunks needing shrink-retries (each retry is
    its own extra round-trip), that idle time compounds. Concurrency keeps
    several requests in flight so the GPU has work queued instead of waiting.

    DATA INTEGRITY UNDER CONCURRENCY — the actual risk, and how it's handled:
    batches can now COMPLETE out of order even though they were SENT in order.
    The old code relied on append-as-you-go, which silently breaks under
    concurrency (batch 3 finishing before batch 1 would put its vectors in the
    wrong slot). This version keys every result by its GLOBAL CHUNK INDEX in
    `texts` and places it into a pre-sized array by that index — so completion
    order genuinely cannot affect the output. Verified in isolation against a
    server with randomized per-request latency; output matched a linear run
    chunk-for-chunk on the same input.

    Counters (truncated_count, gave_up) are accumulated from each batch's
    return value after the gather, never mutated by a coroutine mid-flight —
    concurrent mutation of a shared counter is the other classic way this goes
    quietly wrong, and this avoids it by design rather than by locking.
    """
    import asyncio
    import httpx
    import numpy as np

    headers = {"Content-Type": "application/json"}
    if EMBED_API_KEY:
        headers["Authorization"] = f"Bearer {EMBED_API_KEY}"

    dim_holder: dict = {}   # discovered from the first successful response
    results: dict[int, np.ndarray] = {}   # GLOBAL index -> vector; order-proof
    truncated_count = 0
    gave_up: list[int] = []
    done = 0
    t0 = time.time()

    async def embed_one_async(client, text: str) -> tuple[np.ndarray | None, bool]:
        """Single string, shrinking on 400 until accepted or the floor is hit.
        Returns (vector_or_None, was_truncated)."""
        limit = _EMBED_TRUNCATE_START
        candidate = text
        truncated = False
        while True:
            r = await client.post(f"{EMBED_URL}/embeddings", headers=headers,
                                  json={"model": EMBED_MODEL, "input": [candidate]})
            if r.status_code == 200:
                v = np.asarray(r.json()["data"][0]["embedding"], dtype=np.float32)
                if "dim" not in dim_holder:
                    dim_holder["dim"] = v.shape[0]
                return v, truncated
            if r.status_code != 400:
                r.raise_for_status()
            if limit <= _EMBED_TRUNCATE_FLOOR:
                return None, truncated
            truncated = True
            limit = max(int(limit * _EMBED_TRUNCATE_STEP), _EMBED_TRUNCATE_FLOOR)
            candidate = _shrink_for_embedding(text, limit)

    async def embed_batch(client, global_offset: int, window: list[str]):
        """
        One batch. Returns (offset, vectors_by_local_index, n_truncated,
        gave_up_local_indices) — never mutates anything outside itself, so the
        caller can safely run many of these concurrently and merge afterward.
        """
        n_trunc_local = 0
        gave_up_local: list[int] = []
        try:
            r = await client.post(f"{EMBED_URL}/embeddings", headers=headers,
                                  json={"model": EMBED_MODEL, "input": window})
            if r.status_code == 400:
                raise httpx.HTTPStatusError("batch rejected",
                                            request=r.request, response=r)
            r.raise_for_status()
            data = r.json()["data"]
            data.sort(key=lambda d: d.get("index", 0))
            vecs = [np.asarray(d["embedding"], dtype=np.float32) for d in data]
            if vecs and "dim" not in dim_holder:
                dim_holder["dim"] = vecs[0].shape[0]
            return global_offset, vecs, 0, []
        except httpx.HTTPStatusError:
            # Something in this batch is too long — fall back to per-item,
            # still inside this one coroutine, so nothing here touches shared
            # state until the whole batch's tuple is returned.
            vecs = []
            for j, one in enumerate(window):
                v, was_trunc = await embed_one_async(client, one)
                if was_trunc:
                    n_trunc_local += 1
                if v is None:
                    gave_up_local.append(j)
                    v = np.zeros(dim_holder.get("dim", 384), dtype=np.float32)
                vecs.append(v)
            return global_offset, vecs, n_trunc_local, gave_up_local

    async def run():
        nonlocal truncated_count, done
        async with httpx.AsyncClient(timeout=300.0) as client:
            sem = asyncio.Semaphore(concurrency)

            async def bounded(offset, window):
                async with sem:
                    return await embed_batch(client, offset, window)

            tasks = [
                asyncio.create_task(bounded(i, texts[i:i + batch]))
                for i in range(0, len(texts), batch)
            ]

            for coro in asyncio.as_completed(tasks):
                offset, vecs, n_trunc, gave_up_local = await coro
                for local_idx, v in enumerate(vecs):
                    results[offset + local_idx] = v
                truncated_count += n_trunc
                gave_up.extend(offset + k for k in gave_up_local)
                done += len(vecs)
                if done % 20000 < batch * concurrency:
                    rate = done / max(time.time() - t0, 1)
                    eta = (len(texts) - done) / max(rate, 0.1)
                    extra = f", {truncated_count} truncated" if truncated_count else ""
                    log(f"  {done}/{len(texts)}  ({rate:.0f}/s, ~{eta/60:.0f} min left{extra})")

    asyncio.run(run())

    if truncated_count:
        log(f"  {truncated_count} chunks exceeded 512 tokens and were shrunk "
            f"until accepted (retrieval text is unaffected — only the vector "
            f"was built from a shortened version)")
    if gave_up:
        log(f"  ! {len(gave_up)} chunks could not be embedded even at the "
            f"{_EMBED_TRUNCATE_FLOOR}-char floor and got a zero vector: "
            f"{sorted(gave_up)[:20]}{'...' if len(gave_up) > 20 else ''}")

    # Assemble in GLOBAL INDEX order regardless of completion order — this is
    # the line that actually guarantees output[i] corresponds to texts[i].
    missing = [i for i in range(len(texts)) if i not in results]
    if missing:
        log(f"  ! {len(missing)} chunks never got a result at all "
            f"(offsets {missing[:10]}...) — filling with zero vectors")
        dim = dim_holder.get("dim", 384)
        for i in missing:
            results[i] = np.zeros(dim, dtype=np.float32)

    return [results[i] for i in range(len(texts))]


def stage_embed(db: sqlite3.Connection, batch_size: int = 64) -> None:
    stage("Stage 5 / 5 — embeddings")
    set_status(db, "embeddings", "starting")

    import numpy as np

    rows = db.execute(
        "SELECT id, text FROM wiki_chunks ORDER BY id"
    ).fetchall()

    if not rows:
        log("  no chunks to embed — run the wiki stage first")
        return

    log(f"  {len(rows)} chunks, model {EMBED_MODEL}")

    ids = np.array([r["id"] for r in rows], dtype=np.int64)
    texts = [r["text"] for r in rows]
    if EMBED_DOC_PREFIX:
        texts = [EMBED_DOC_PREFIX + t for t in texts]

    # --- GPU / remote path ------------------------------------------------
    if EMBED_URL:
        log(f"  endpoint: {EMBED_URL}/embeddings  (GPU)")
        log(f"  batch: {max(batch_size, 256)}")
        t0 = time.time()
        set_status(db, "embeddings", f"remote endpoint {EMBED_URL}")
        vlist = _embed_remote(texts, EMBED_BATCH, concurrency=EMBED_CONCURRENCY)
        dim = vlist[0].shape[0]
        log(f"  dimension: {dim}")
        vecs = np.zeros((len(vlist), dim), dtype=np.float16)
        for i, v in enumerate(vlist):
            n = np.linalg.norm(v)
            vecs[i] = (v / n if n > 0 else v).astype(np.float16)
        np.save(EMB_PATH, vecs)
        np.save(EMB_IDS_PATH, ids)
        log(f"  wrote {EMB_PATH.name} ({EMB_PATH.stat().st_size / 1e6:.0f} MB)")
        set_meta(db, "embed_model", EMBED_MODEL)
        set_meta(db, "embed_doc_prefix", EMBED_DOC_PREFIX)
        set_meta(db, "embed_dim", int(dim))
        set_meta(db, "embed_count", int(vecs.shape[0]))
        set_meta(db, "embed_backend", f"remote:{EMBED_URL}")
        set_meta(db, "embed_ingested_at",
                 time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        db.commit()
        log(f"  done — {time.time() - t0:.0f}s "
            f"({len(texts) / max(time.time() - t0, 1):.0f}/s)")
        return

    # --- local CPU path ---------------------------------------------------
    log(f"  cache: {HF_HOME}")
    log(f"  threads: {EMBED_THREADS} (of {_CPUS} cores) — "
        f"override with EMBED_THREADS or --threads")
    log("  NOTE: CPU embedding measures ~30 chunks/s regardless of thread count.")
    log("        Set EMBED_URL to a GPU endpoint for a large speedup.")
    try:
        from fastembed import TextEmbedding
    except ImportError:
        log("  ! fastembed is not installed and EMBED_URL is not set.")
        raise SystemExit(1)

    if os.environ.get("HF_HUB_OFFLINE") == "1" and not Path(HF_HOME).exists():
        log("  ! HF_HUB_OFFLINE=1 but the cache does not exist yet.")
        log("    First run needs: -e HF_HUB_OFFLINE=0 -e TRANSFORMERS_OFFLINE=0")
        raise SystemExit(1)

    # Newer fastembed takes threads= directly; older versions rely on the
    # OMP_NUM_THREADS we set at import. Try the explicit route, fall back.
    try:
        model = TextEmbedding(model_name=EMBED_MODEL, cache_dir=HF_HOME,
                              threads=EMBED_THREADS)
    except TypeError:
        model = TextEmbedding(model_name=EMBED_MODEL, cache_dir=HF_HOME)
        log("  (this fastembed has no threads= parameter; using OMP_NUM_THREADS)")

    vecs = None
    done = 0
    t0 = time.time()

    for i, emb in enumerate(model.embed(texts, batch_size=batch_size)):
        v = np.asarray(emb, dtype=np.float32)
        if vecs is None:
            # Stored as float16 — halves the file and costs almost nothing in
            # retrieval quality for normalized vectors.
            vecs = np.zeros((len(rows), v.shape[0]), dtype=np.float16)
            log(f"  dimension: {v.shape[0]}")
        n = np.linalg.norm(v)
        if n > 0:
            v = v / n
        vecs[i] = v.astype(np.float16)
        done += 1
        if done % 5000 == 0:
            rate = done / max(time.time() - t0, 1)
            eta = (len(rows) - done) / max(rate, 0.1)
            log(f"  {done}/{len(rows)}  ({rate:.0f}/s, ~{eta/60:.0f} min left)")
            set_status(db, "embeddings",
                       f"{rate:.0f}/s, ~{eta/60:.0f} min left",
                       done=done, total=len(rows))

    np.save(EMB_PATH, vecs)
    np.save(EMB_IDS_PATH, ids)

    mb = EMB_PATH.stat().st_size / 1e6
    log(f"  wrote {EMB_PATH.name} ({mb:.0f} MB) and {EMB_IDS_PATH.name}")

    set_meta(db, "embed_model", EMBED_MODEL)
    set_meta(db, "embed_doc_prefix", EMBED_DOC_PREFIX)
    set_meta(db, "embed_dim", int(vecs.shape[1]))
    set_meta(db, "embed_count", int(vecs.shape[0]))
    set_meta(db, "embed_ingested_at", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    db.commit()
    log(f"  done — {time.time() - t0:.0f}s")


# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------

def summary(db: sqlite3.Connection) -> None:
    stage("Summary")
    q = lambda sql: db.execute(sql).fetchone()[0]
    try:
        log(f"  species          {q('SELECT COUNT(*) FROM species')}")
        log(f"  moves            {q('SELECT COUNT(*) FROM moves')}")
        log(f"  abilities        {q('SELECT COUNT(*) FROM abilities')}")
        log(f"  items            {q('SELECT COUNT(*) FROM items')}")
        log(f"  learnset entries {q('SELECT COUNT(*) FROM learnsets')}")
        log(f"  wiki chunks      {q('SELECT COUNT(*) FROM wiki_chunks')}")
        log(f"    linked         {q('SELECT COUNT(*) FROM wiki_chunks WHERE canonical_id IS NOT NULL')}")
        log(f"  aliases          {q('SELECT COUNT(*) FROM aliases')}")
        log(f"  usage rows       {q('SELECT COUNT(*) FROM usage_stats')}")
    except Exception as e:
        log(f"  (partial: {e})")

    size = Path(DB_PATH).stat().st_size / 1e6 if Path(DB_PATH).exists() else 0
    log(f"\n  {DB_PATH}  {size:.0f} MB")
    if EMB_PATH.exists():
        log(f"  {EMB_PATH}  {EMB_PATH.stat().st_size / 1e6:.0f} MB")
    log("\n  Verify with:  curl http://<host>:8990/health")


# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Build the Pokedex AI database.")
    ap.add_argument("--dex-only", action="store_true")
    ap.add_argument("--stats-only", action="store_true")
    ap.add_argument("--wiki-only", action="store_true",
                    help="wiki + aliases + embeddings")
    ap.add_argument("--images-only", action="store_true",
                    help="species artwork only — fast, skips chunking, "
                         "aliases, and embeddings entirely")
    ap.add_argument("--sprites-only", action="store_true",
                    help="catalog Pokemon HOME sprites from HOME_ICONS_DIR/"
                         "HOME_PREVIEWS_DIR — independent of --images-only, "
                         "which is the Bulbapedia box-art path")
    ap.add_argument("--item-categories-only", action="store_true",
                    help="parse type/stat-boosting item categories out of "
                         "already-ingested wiki articles — requires "
                         "stage_wiki and stage_dex to have already run")
    ap.add_argument("--aliases-only", action="store_true")
    ap.add_argument("--sets-only", action="store_true",
                    help="reload Smogon sets from SETS_DIR")
    ap.add_argument("--embed-only", action="store_true")
    ap.add_argument("--limit", type=int, default=None,
                    help="stop after N articles (testing)")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--skip-space-check", action="store_true",
                    help="run even if the disk looks too small")
    ap.add_argument("--force-unlock", action="store_true",
                    help="start even if another ingest appears to be running")
    ap.add_argument("--tier", choices=list(_EMBED_TIER_PRESETS),
                    help="embedding preset: cpu | gpu-small | gpu-big. "
                        f"Auto-detected from EMBED_URL if not given "
                        f"(currently would pick: "
                        f"{_embed_tier or ('gpu-small' if EMBED_URL else 'cpu')}). "
                        f"--concurrency/--batch-size still override the tier's values.")
    ap.add_argument("--concurrency", type=int, default=None,
                    help=f"parallel embedding requests to EMBED_URL "
                         f"(tier default: {EMBED_CONCURRENCY}, env EMBED_CONCURRENCY)")
    ap.add_argument("--threads", type=int, default=None,
                    help=f"embedding threads (default: min(16, cores) = {EMBED_THREADS})")
    args = ap.parse_args()

    if args.tier:
        _preset = _EMBED_TIER_PRESETS[args.tier]
        globals()["EMBED_CONCURRENCY"] = _preset["concurrency"]
        globals()["EMBED_BATCH"] = _preset["batch"]

    if args.concurrency:
        globals()["EMBED_CONCURRENCY"] = args.concurrency
    if args.batch_size and args.batch_size != 64:
        globals()["EMBED_BATCH"] = args.batch_size

    if args.threads:
        globals()["EMBED_THREADS"] = args.threads
        # Overwrite rather than setdefault — an explicit flag beats inherited env.
        for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                   "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
            os.environ[_v] = str(args.threads)

    selective = any([args.dex_only, args.stats_only, args.wiki_only,
                     args.images_only, args.sprites_only, args.aliases_only,
                     args.embed_only, args.sets_only, args.item_categories_only])

    _open_log()

    log(f"database: {DB_PATH}")
    log(f"log file: {LOG_PATH}")
    log(f"offline:  {OFFLINE}")
    log(f"gens:     {GENS}")
    log(f"threads:  {EMBED_THREADS} (of {_CPUS} cores)")
    log("")
    log(f"Follow along from the host with:  tail -f {LOG_PATH}")
    log("Or check progress with:            curl .../health")

    acquire_lock(force=args.force_unlock)

    db = connect()
    t0 = time.time()

    try:
        need_wiki = args.wiki_only or not selective
        if not args.skip_space_check:
            preflight(db, need_wiki)

        if args.dex_only:
            stage_dex(db); checkpoint(db)
            stage_aliases(db); checkpoint(db)
        elif args.stats_only:
            stage_stats(db); checkpoint(db)
        elif args.sets_only:
            stage_sets(db); checkpoint(db)
        elif args.wiki_only:
            stage_wiki(db, limit=args.limit); checkpoint(db)
            stage_analyses(db); checkpoint(db)
            stage_aliases(db); checkpoint(db)
            stage_embed(db, args.batch_size); checkpoint(db)
        elif args.images_only:
            stage_images(db); checkpoint(db)
        elif args.sprites_only:
            stage_home_sprites(db); checkpoint(db)
        elif args.item_categories_only:
            stage_item_categories(db); checkpoint(db)
        elif args.aliases_only:
            stage_aliases(db); checkpoint(db)
        elif args.embed_only:
            stage_embed(db, args.batch_size); checkpoint(db)
        elif not selective:
            stage_dex(db); checkpoint(db)
            stage_stats(db); checkpoint(db)
            stage_wiki(db, limit=args.limit); checkpoint(db)
            stage_analyses(db); checkpoint(db)
            stage_sets(db); checkpoint(db)
            stage_item_categories(db); checkpoint(db)
            stage_aliases(db); checkpoint(db)
            stage_embed(db, args.batch_size); checkpoint(db)

        set_status(db, "complete", f"finished in {time.time() - t0:.0f}s")
        summary(db)
        log("")
        log(f"Total: {time.time() - t0:.0f}s")
    except KeyboardInterrupt:
        set_status(db, "interrupted", "stopped by user")
        log("")
        log("Interrupted. Completed stages are saved — re-run to continue.")
        sys.exit(130)
    except BrokenPipeError:
        # Someone piped us into `head` or closed the terminal. The work is fine;
        # only stdout went away. Don't scare them with a "failed" status.
        try:
            sys.stdout = open(os.devnull, "w")
        except Exception:
            pass
        raise SystemExit(0)
    except Exception as e:
        set_status(db, "failed", str(e)[:200])
        log("")
        log(f"FAILED: {e}")
        raise
    finally:
        release_lock()
        db.close()
        if _LOGFH:
            try:
                _LOGFH.close()
            except Exception:
                pass


if __name__ == "__main__":
    main()
