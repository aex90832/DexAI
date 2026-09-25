#!/usr/bin/env python3
# DexAI — Tool Server
# Copyright (C) 2026 @aex90832 (https://github.com/aex90832/)
#
# Created with human authoring and AI assistance.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

"""
DexAI — tool server.

Exposes the endpoints the language model calls, plus an OpenAPI spec that Open
WebUI reads to discover them.

The endpoint DESCRIPTIONS in this file are prompts. The model reads them to
decide which tool to reach for, so they say both what a tool is for and what it
is NOT for. Edit them if the model keeps picking wrong — that is usually a better
fix than editing the system prompt.

Run:
    uvicorn main:app --host 0.0.0.0 --port 8990
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sqlite3
import threading
import time
import unicodedata
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

import httpx
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------

DB_PATH = os.environ.get("DB_PATH", "/data/pokedex.db")
SIM_URL = os.environ.get("SIM_URL", "http://pokedex-sim:8991").rstrip("/")
KIWIX_URL = os.environ.get("KIWIX_URL", "").rstrip("/")
KIWIX_BOOK = os.environ.get("KIWIX_BOOK", "")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
EMBED_URL = os.environ.get("EMBED_URL", "").rstrip("/")
# The query-side twin of ingest.py's EMBED_DOC_PREFIX. Some models need
# different prefixes for what's indexed vs. what's searched (nomic-embed-text:
# "search_document: " / "search_query: "). Must match whatever built the
# index currently in embeddings.f16.npy, not just whatever EMBED_MODEL says
# today — if you change models, re-run ingest.py --wiki-only before this
# matters, since old and new vectors cannot mix in one index regardless.
EMBED_QUERY_PREFIX = os.environ.get("EMBED_QUERY_PREFIX", "")
EMBED_API_KEY = os.environ.get("EMBED_API_KEY", "")
HF_HOME = os.environ.get("HF_HOME", "/data/models")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "").rstrip("/")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
DEFAULT_GEN = int(os.environ.get("DEFAULT_GEN", "9"))
DEFAULT_FORMAT = os.environ.get("DEFAULT_FORMAT", "gen9ou")
# Local filesystem paths for the static mount below. These are directory
# paths, not URLs — the public URL that home_sprites rows actually point at
# was baked in at ingest time via HOME_SPRITES_URL, and must already match
# wherever these get mounted (see the mount right after app init).
HOME_ICONS_DIR = os.environ.get("HOME_ICONS_DIR", "")
HOME_PREVIEWS_DIR = os.environ.get("HOME_PREVIEWS_DIR", "")

# Query embedding is one short string at a time, so a large ONNX thread pool is
# pure overhead — and on a big host it would monopolize cores the model server
# wants. Keep this small; it is not the same workload as bulk ingest.
_CPUS = os.cpu_count() or 8
EMBED_THREADS = int(os.environ.get("EMBED_QUERY_THREADS")
                    or os.environ.get("EMBED_THREADS")
                    or min(4, _CPUS))
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, str(EMBED_THREADS))

DATA_DIR = Path(DB_PATH).parent
EMB_PATH = DATA_DIR / "embeddings.f16.npy"
EMB_IDS_PATH = DATA_DIR / "embeddings_ids.npy"

MAX_CHUNK_CHARS = 1400      # truncate retrieved text so results stay readable
RRF_K = 60                  # reciprocal rank fusion constant


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "")
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]", "", s.lower())


def resolve_moves(c: sqlite3.Connection, gen: int, move_names: list[str]) -> list[dict]:
    """
    Move name -> {name, type, category, base_power, pp}, for the competitive
    card's move list. Tries the exact requested gen first (a move's BP/type
    can differ across generations — e.g. several moves changed base power
    at various points), falling back to the latest available gen for that
    move only if it's genuinely missing at the requested one.
    """
    out = []
    for name in move_names or []:
        if not name:
            continue
        mid = norm(name)
        row = c.execute(
            "SELECT name, type, category, base_power, pp FROM moves WHERE id=? AND gen=?",
            (mid, gen)
        ).fetchone()
        if not row:
            row = c.execute(
                "SELECT name, type, category, base_power, pp FROM moves WHERE id=? "
                "ORDER BY gen DESC LIMIT 1", (mid,)
            ).fetchone()
        if row:
            out.append({"name": row["name"], "type": row["type"], "category": row["category"],
                        "base_power": row["base_power"], "pp": row["pp"]})
        else:
            out.append({"name": name, "type": None, "category": None, "base_power": None, "pp": None})
    return out


def defensive_profile(c: sqlite3.Connection, gen: int, types: list[str]) -> dict:
    """
    Same bucketing type_matchup's own full-defensive-profile branch already
    computes, extracted so card enrichment can call it directly instead of
    duplicating the logic (or worse, drifting from it over time). Compact
    buckets only — weak+quad_weak merged into "weak", resists+quad_resists
    merged into "resist" — since a card has room for a short badge row, not
    the six-way breakdown /type_matchup itself returns.
    """
    rows = c.execute(
        "SELECT DISTINCT attacking_type FROM typechart WHERE gen=?", (gen,)
    ).fetchall()
    profile = {}
    for r in rows:
        at = r["attacking_type"]
        mult = 1.0
        for dt in types:
            m = c.execute(
                "SELECT multiplier FROM typechart WHERE gen=? AND "
                "LOWER(defending_type)=LOWER(?) AND attacking_type=?",
                (gen, dt, at)).fetchone()
            mult *= m["multiplier"] if m else 1.0
        profile[at] = mult
    return {
        "immune": sorted(k for k, v in profile.items() if v == 0),
        "resist": sorted(k for k, v in profile.items() if 0 < v < 1),
        "weak": sorted(k for k, v in profile.items() if v > 1),
    }


def is_battle_only(val) -> bool:
    """
    battle_only is written as 0/1, but can come back from the database as
    either a real int (a column with correct INTEGER affinity) or its text
    representation (a column that kept TEXT affinity from an earlier
    migration attempt, before the declared type was corrected — confirmed
    directly: this exact live database returns the STRING '0' for a normal,
    non-battle-only species). Python's own truthiness would get this wrong —
    '0' is a non-empty string, so it's truthy same as '1' — which would
    misclassify every ordinary species sharing a base_species with a
    primary team member (a real regional form like Raichu-Alola, not a
    battle-dynamic transformation) as "same team member, different battle
    state." Comparing against the literal '1' instead of relying on
    truthiness is correct regardless of which storage form the column
    actually contains.
    """
    return str(val) == "1"


def species_sprite(c: sqlite3.Connection, num: int, forme: str | None,
                    full_name: str, shiny: bool = False) -> tuple[str | None, str | None]:
    """
    (icon_path, preview_path) for one species, correctly forme-aware.

    The bug this fixes: three separate places in this file each independently
    reimplemented "find this species' sprite" — /lookup's own species branch,
    /query_dex's batch resolution, and this file's resolve_images_by_name.
    Only the last of those ever checked whether the species itself IS a Mega
    or regional form; the other two always queried form_index=0 (the base
    form) keyed by national dex number alone — correct for a plain species,
    silently wrong for anything else, since Megas/regional forms share their
    base's dex number. Confirmed directly: asking for Mega Charizard X's
    details returned base Charizard's sprite, because /lookup's copy of this
    logic had no idea "Charizard-Mega-X" wasn't just "Charizard". One shared
    function, used everywhere a sprite is resolved, is how this stays fixed
    rather than needing to be found and re-fixed a third time somewhere else.
    """
    is_shiny = 1 if shiny else 0
    if forme:
        row = c.execute(
            "SELECT icon_path, preview_path FROM home_sprites "
            "WHERE natdex=? AND forme_name=? AND is_shiny=?",
            (num, full_name, is_shiny),
        ).fetchone()
    else:
        row = c.execute(
            "SELECT icon_path, preview_path FROM home_sprites "
            "WHERE natdex=? AND form_index=0 AND is_gmax=0 AND is_shiny=?",
            (num, is_shiny),
        ).fetchone()
    if row:
        return row["icon_path"], row["preview_path"]
    return None, None


def resolve_images_by_name(c: sqlite3.Connection, names: list[str]) -> dict[str, dict]:
    """
    Image + type lookup by SPECIES NAME rather than by an already-fetched
    species row's num — needed because validate_team/review_team/
    compare_teams come from pokedex-sim and only ever carry names, never a
    species row from this database.

    A name like "Charizard-Mega-X" resolves correctly to ITS OWN sprite, not
    the base species' — norm() strips it to "charizardmegax", which is
    exactly the real Showdown id for that forme (confirmed against live
    @pkmn/dex data), so a direct id lookup works without a separate mapping
    table. Searches across every loaded gen, not just DEFAULT_GEN, since a
    Mega form's species row only exists in gens 6-7 at all.

    Matches home_sprites.forme_name against the species row's full NAME
    ("Charizard-Mega-X"), not its `forme` field ("Mega-X" — just the suffix,
    confirmed distinct from name in real @pkmn/dex output). Conflating the
    two was a real bug caught by testing before this ever ran for real: it
    silently resolved zero alternate-form images.
    """
    out: dict[str, dict] = {}
    for name in names:
        if not name or name in out:
            continue
        nid = norm(name)
        srow = c.execute(
            "SELECT gen, num, name, forme, types, hp, atk, def_, spa, spd, spe, "
            "base_species, battle_only, required_item "
            "FROM species WHERE id=? ORDER BY gen DESC LIMIT 1", (nid,)
        ).fetchone()
        if not srow:
            continue
        num, full_name, forme = srow["num"], srow["name"], srow["forme"]

        icon_path, preview_path = species_sprite(c, num, forme, full_name)

        url = None
        if icon_path or preview_path:
            url = preview_path or icon_path
        else:
            img = c.execute(
                "SELECT image_url FROM species_images WHERE species_norm=?", (nid,)
            ).fetchone()
            url = img["image_url"] if img else None

        if url:
            try:
                types = json.loads(srow["types"]) if srow["types"] else []
            except Exception:
                types = []
            out[name] = {
                "url": url, "types": types,
                "base_stats": {"hp": srow["hp"], "atk": srow["atk"], "def": srow["def_"],
                                "spa": srow["spa"], "spd": srow["spd"], "spe": srow["spe"]},
                "base_species": srow["base_species"],
                "battle_only": is_battle_only(srow["battle_only"]),
                "required_item": srow["required_item"],
                **defensive_profile(c, srow["gen"], types),
            }
    return out


# ---------------------------------------------------------------------------
# lazily-loaded state
# ---------------------------------------------------------------------------

class State:
    embeddings: np.ndarray | None = None
    embedding_ids: np.ndarray | None = None
    embeddings_mtime: float = 0.0
    embedder: Any = None
    embed_lock = threading.Lock()
    started_at = time.time()


S = State()


def db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def meta(key: str, default=None):
    try:
        with db() as c:
            r = c.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
            if not r:
                return default
            try:
                return json.loads(r["value"])
            except Exception:
                return r["value"]
    except Exception:
        return default


def load_embeddings() -> bool:
    """
    Memory-map the vector file, remapping it if ingest has rewritten it.

    A re-ingest calls np.save() on this exact path. A long-lived memmap held
    across that write points at a file that no longer matches embedding_ids —
    the vectors and the chunk IDs silently disagree and retrieval returns the
    wrong articles. Checking mtime costs a stat() and avoids needing to restart
    the API after every wiki rebuild.
    """
    if not EMB_PATH.exists() or not EMB_IDS_PATH.exists():
        return False

    try:
        mtime = EMB_PATH.stat().st_mtime
    except OSError:
        return S.embeddings is not None

    if S.embeddings is not None and mtime == S.embeddings_mtime:
        return True

    with S.embed_lock:
        # re-check: another thread may have remapped while we waited
        if S.embeddings is not None and mtime == S.embeddings_mtime:
            return True
        try:
            emb = np.load(EMB_PATH, mmap_mode="r")
            ids = np.load(EMB_IDS_PATH)
        except Exception:
            # Mid-write: np.save truncates before filling. Keep serving the old
            # map if we have one rather than failing the request.
            return S.embeddings is not None
        if emb.shape[0] != ids.shape[0]:
            # The two files are written back to back, so a mismatch means we
            # caught ingest between them. Try again on the next request.
            return S.embeddings is not None
        S.embeddings = emb
        S.embedding_ids = ids
        S.embeddings_mtime = mtime
    return True


def get_embedder():
    """
    Load the embedding model on first use rather than at startup, so the server
    answers /health immediately and only pays the ~10s model load if someone
    actually runs a semantic search.
    """
    if S.embedder is not None:
        return S.embedder
    with S.embed_lock:
        if S.embedder is None:
            from fastembed import TextEmbedding
            try:
                S.embedder = TextEmbedding(model_name=EMBED_MODEL,
                                           cache_dir=HF_HOME,
                                           threads=EMBED_THREADS)
            except TypeError:
                # older fastembed: relies on the OMP_NUM_THREADS set at import
                S.embedder = TextEmbedding(model_name=EMBED_MODEL, cache_dir=HF_HOME)
    return S.embedder


def embed_query(text: str) -> np.ndarray:
    """
    Embed one query. Uses the same backend the index was built with — mixing
    them would be subtly wrong, since vectors from different runtimes of the
    same model can differ enough to degrade retrieval.
    """
    if EMBED_QUERY_PREFIX:
        text = EMBED_QUERY_PREFIX + text
    if EMBED_URL:
        headers = {"Content-Type": "application/json"}
        if EMBED_API_KEY:
            headers["Authorization"] = f"Bearer {EMBED_API_KEY}"
        r = httpx.post(f"{EMBED_URL}/embeddings", headers=headers, timeout=30.0,
                       json={"model": EMBED_MODEL, "input": [text]})
        r.raise_for_status()
        v = np.asarray(r.json()["data"][0]["embedding"], dtype=np.float32)
    else:
        v = np.asarray(next(iter(get_embedder().embed([text]))), dtype=np.float32)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


# ---------------------------------------------------------------------------
# entity resolution — the layer that makes the three data sources agree
# ---------------------------------------------------------------------------

def resolve(name: str, kind: str | None = None) -> dict | None:
    """
    Map any phrasing of a name to a canonical ID.

    'Urshifu-Rapid-Strike', 'rapid strike urshifu', and 'urshifurapidstrike' all
    land on the same entity. Without this, wiki text and dex rows disagree about
    what the user asked for.
    """
    n = norm(name)
    if not n:
        return None
    with db() as c:
        sql = "SELECT canonical_id, alias, kind FROM aliases WHERE alias_norm = ?"
        params: list = [n]
        if kind:
            sql += " AND kind = ?"
            params.append(kind)
        sql += " ORDER BY CASE kind WHEN 'species' THEN 0 WHEN 'move' THEN 1 " \
               "WHEN 'ability' THEN 2 WHEN 'item' THEN 3 ELSE 4 END LIMIT 1"
        r = c.execute(sql, params).fetchone()
        if r:
            return dict(r)

        # Prefix fallback catches partial names and minor typos in the tail.
        r = c.execute(
            "SELECT canonical_id, alias, kind FROM aliases "
            "WHERE alias_norm LIKE ? ORDER BY LENGTH(alias_norm) LIMIT 1",
            (n + "%",),
        ).fetchone()
        return dict(r) if r else None


# ---------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------

class LookupRequest(BaseModel):
    name: str = Field(..., description="Pokemon, move, ability, or item name. Any common spelling works.")
    gen: int | None = Field(None, description=f"Generation 1-9. Defaults to {DEFAULT_GEN}.")
    moves_to_check: list[str] | None = Field(
        None,
        description="Optional move names to check legality for. Use this for "
                    "'can X learn Y' questions instead of guessing. The reply "
                    "includes HOW it learns the move and at what level.",
    )
    include_learnset: bool = Field(
        False,
        description="Return the full level-up learnset for this Pokemon in this "
                    "generation, ordered by level. Use for 'what moves does X "
                    "learn and when' — the question someone playing through a "
                    "game actually asks.",
    )
    shiny: bool = Field(
        False,
        description="Set true when the user specifically asked for the SHINY "
                    "appearance ('shiny Gengar', 'what does shiny X look like'). "
                    "Swaps image_url/sprite_icon_url/sprite_preview_url to the "
                    "shiny variant so that's what's actually shown, rather than "
                    "showing the regular image and only mentioning shiny exists.",
    )


class QueryDexRequest(BaseModel):
    types: list[str] | None = Field(None, description="Type names, e.g. ['Water','Ground'].")
    type_mode: Literal["all", "any"] = Field("all", description="'all' = must have every listed type.")
    learns: list[str] | None = Field(None, description="Must legally learn all of these moves.")
    ability: str | None = None
    egg_group: str | None = None
    tier: str | None = Field(None, description="Smogon tier, e.g. 'OU', 'UU'.")
    min_bst: int | None = None
    max_bst: int | None = None
    stat_filters: dict[str, dict[str, int]] | None = Field(
        None,
        description="Per-stat bounds, e.g. {'spe': {'min': 100}, 'hp': {'max': 80}}. "
                    "Keys: hp, atk, def, spa, spd, spe.",
    )
    nfe: bool | None = Field(None, description="True = only unevolved Pokemon.")
    gen: int | None = None
    order_by: Literal["bst", "spe", "atk", "spa", "hp", "def", "spd", "num"] = "bst"
    limit: int = Field(25, ge=1, le=100)


class SearchWikiRequest(BaseModel):
    query: str = Field(..., description="What to search for, in natural language.")
    entity: str | None = Field(None, description="Restrict to one Pokemon/move/item's articles.")
    domain: Literal["games", "anime", "manga", "tcg", "competitive", "general"] | None = Field(
        None, description="Restrict to one body of content. Use this when the question is clearly about one."
    )
    gen: int | None = None
    limit: int = Field(6, ge=1, le=15)


class UsageStatsRequest(BaseModel):
    format: str = Field(DEFAULT_FORMAT, description="Format ID, e.g. 'gen9ou', 'gen9vgc2026'.")
    species: str | None = Field(None, description="One Pokemon, for full detail. Omit for the usage ranking.")
    month: str | None = Field(None, description="YYYY-MM. Omit for the most recent available.")
    limit: int = Field(20, ge=1, le=50)


class ValidateTeamRequest(BaseModel):
    team: str = Field(..., description="Team in Pokemon Showdown export format.")
    format: str = Field(DEFAULT_FORMAT, description="Format to validate against.")


class CalcDamageRequest(BaseModel):
    attacker: dict = Field(..., description="{name, level?, item?, ability?, nature?, evs?, boosts?, teraType?}")
    defender: dict = Field(..., description="Same shape as attacker.")
    move: dict = Field(..., description="{name, crit?, hits?}")
    field: dict | None = Field(None, description="{weather?, terrain?, isReflect?, isLightScreen?, gameType?}")
    gen: int = DEFAULT_GEN


class CompareTeamsRequest(BaseModel):
    team_a: str = Field(..., description="First team, Showdown export format.")
    team_b: str = Field(..., description="Second team, Showdown export format.")
    format: str = Field(DEFAULT_FORMAT, description="Format ID, e.g. 'gen9ou'.")
    gen: int = Field(DEFAULT_GEN, description="Generation 1-9.")
    label_a: str = Field("Team A", description="Name for the first team.")
    label_b: str = Field("Team B", description="Name for the second team.")


class TypeMatchupRequest(BaseModel):
    attacking_type: str | None = Field(
        None, description="A single attacking type, e.g. 'Fighting'.")
    defending_types: list[str] | None = Field(
        None, description="Defending type(s), e.g. ['Dark','Steel'].")
    species: str | None = Field(
        None, description="A Pokemon name — returns its full defensive profile. "
                          "Use instead of defending_types when you have a name.")
    gen: int | None = Field(
        None, description="Generation 1-9. The type chart changed: Steel resisted "
                          "Dark and Ghost before gen 6, and Fairy did not exist.")


class ReviewTeamRequest(BaseModel):
    team: str = Field(..., description="Team in Showdown export format.")
    format: str = Field(DEFAULT_FORMAT)
    gen: int = Field(DEFAULT_GEN)


class GetSetsRequest(BaseModel):
    species: str = Field(..., description="Pokemon name.")
    format: str | None = Field(
        None, description="Format ID, e.g. 'gen9ou'. Omit to see every format.")
    limit: int = Field(8, ge=1, le=20)


class CommonGenerationRequest(BaseModel):
    species: list[str] = Field(
        ..., min_length=2,
        description="Two or more named Pokemon to check together.",
    )


class QueryItemsRequest(BaseModel):
    is_choice: bool | None = Field(None, description="Choice Band/Specs/Scarf.")
    is_berry: bool | None = None
    is_mega_stone: bool | None = Field(None, description="Enables a Mega Evolution.")
    category: Literal["type_boost", "stat_boost"] | None = Field(
        None,
        description="type_boost: Charcoal/Mystic Water/etc, boosts one move type's power. "
                    "stat_boost: Eviolite/Light Ball/Thick Club/etc, raises a stat directly "
                    "(often species-restricted). Leave unset and use query instead for "
                    "anything else — status orbs, weather rocks, terrain seeds, gems, "
                    "plates, and general items (Leftovers, Life Orb, Rocky Helmet, Focus "
                    "Sash, ...) aren't in a clean category, but their short_desc is precise "
                    "and searchable.",
    )
    query: str | None = Field(
        None, description="Free-text search over the item's effect description. Search "
                          "for SPECIFIC terms, not generic category words — short_desc "
                          "says 'Sunny Day' not 'weather', 'Electric Terrain' not "
                          "'terrain', 'badly poison' not 'status'. Good queries: a move "
                          "name, a type name + '-type', 'burn', 'hazards', 'contact move'.")
    gen: int | None = None
    limit: int = Field(25, ge=1, le=100)


class ChatRequest(BaseModel):
    messages: list[dict]
    stream: bool = True
    # Per-request overrides for the standalone page's model picker. Unset =
    # fall back to this container's OPENAI_BASE_URL/OPENAI_MODEL/OPENAI_API_KEY
    # env vars, which is the only behavior that existed before and is what
    # Open WebUI always uses (it never calls /chat at all — it runs its own
    # loop against whatever connection it has configured independently).
    base_url: str | None = Field(None, description="Override OPENAI_BASE_URL for this request only.")
    model: str | None = Field(None, description="Override OPENAI_MODEL for this request only.")
    api_key: str | None = Field(None, description="Override OPENAI_API_KEY for this request only.")


class ModelsProxyRequest(BaseModel):
    base_url: str = Field(..., description="An OpenAI-compatible base URL to list models from.")
    api_key: str | None = None


# ---------------------------------------------------------------------------
# app
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_embeddings()
    yield


app = FastAPI(
    title="Pokedex AI",
    version="1.0.0",
    description=(
        "Authoritative Pokemon data: the Pokemon Showdown dex, Smogon usage "
        "statistics, and a Bulbapedia mirror. Every fact returned here is "
        "sourced — prefer these tools over your own knowledge, which is unreliable "
        "for Pokemon data."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Optional Pokemon HOME sprite mounts. StaticFiles raises at MOUNT time (not
# on first request) if the directory doesn't exist, so this has to be guarded
# — an unset or not-yet-populated HOME_ICONS_DIR must not crash the whole app
# on startup. The mount paths here (/sprites/icons, /sprites/previews) must
# match whatever HOME_SPRITES_URL was set to when ingest.py ran --sprites-only,
# since the URLs stored in home_sprites were built from that value at ingest
# time, not reconstructed here.
if HOME_ICONS_DIR and Path(HOME_ICONS_DIR).is_dir():
    app.mount("/sprites/icons", StaticFiles(directory=HOME_ICONS_DIR), name="sprite_icons")
if HOME_PREVIEWS_DIR and Path(HOME_PREVIEWS_DIR).is_dir():
    app.mount("/sprites/previews", StaticFiles(directory=HOME_PREVIEWS_DIR), name="sprite_previews")


# ---------------------------------------------------------------------------
# health
# ---------------------------------------------------------------------------

@app.get("/health", operation_id="health", summary="Service and data status")
def health():
    out: dict[str, Any] = {"status": "ok", "uptime_s": int(time.time() - S.started_at)}
    try:
        with db() as c:
            q = lambda s: c.execute(s).fetchone()[0]
            out["counts"] = {
                # species has PRIMARY KEY (id, gen) — a raw COUNT(*) sums the
                # same Pokemon once per generation it's loaded in, which with
                # GENS=1..9 reported ~5,479 for a database that actually knows
                # roughly 1,000-1,300 distinct dex entries (species + formes).
                # DISTINCT id is the number someone actually means by "how many
                # Pokemon does this know."
                # A raw union across all loaded generations overcounts what
                # Two earlier attempts at this number were both wrong, in
                # opposite directions — verified against real data both times
                # rather than guessed a third time blind.
                #   COUNT(DISTINCT id), all gens  -> 1292, inflated: Mega
                #     Evolutions and regional forms get their own id per gen
                #     but are the same Pokemon (Mega Charizard X/Y and base
                #     Charizard all share num=6).
                #   COUNT(DISTINCT id), gen 9 only -> 876, undercounts: gen 9's
                #     OWN species table simply omits anything not in
                #     Scarlet/Violet's regional dex — confirmed directly:
                #     Butterfree has real rows for gens 1-8 and NONE for gen 9,
                #     despite being a real, current Pokemon.
                # The fix: count DISTINCT NATIONAL DEX NUMBER across ALL loaded
                # gens. Mega/regional forms share their base's num (confirmed:
                # Charizard-Mega-X/Y and Raichu-Alola both do), so they don't
                # inflate it; a species missing from gen 9's own table is still
                # counted via whichever earlier gen carries its real num, so it
                # isn't dropped either. Verified result: exactly 1025, the
                # actual modern National Dex total.
                "species": q("SELECT COUNT(DISTINCT num) FROM species WHERE num > 0"),
                # What COUNT(DISTINCT id) actually measures: how many distinct
                # BATTLE-USABLE FORMS this database knows across every loaded
                # generation — species plus every Mega, regional, and other
                # forme counted separately. A genuinely different, still
                # informational number; just not what "how many Pokemon" means.
                "species_and_forms_all_generations": q("SELECT COUNT(DISTINCT id) FROM species"),
                # moves/abilities/items share the identical (id, gen) primary
                # key as species, so they had the same bug: a move that exists
                # unchanged across all 9 generations was being counted 9 times.
                "moves": q("SELECT COUNT(DISTINCT id) FROM moves"),
                "abilities": q("SELECT COUNT(DISTINCT id) FROM abilities"),
                "items": q("SELECT COUNT(DISTINCT id) FROM items"),
                # learnsets is NOT the same bug — each row is a genuinely
                # distinct (species, move, generation, method) fact, and the
                # per-generation volume IS the real data size, not inflation.
                "learnsets": q("SELECT COUNT(*) FROM learnsets"),
                "wiki_chunks": q("SELECT COUNT(*) FROM wiki_chunks"),
                "wiki_chunks_linked": q(
                    "SELECT COUNT(*) FROM wiki_chunks WHERE canonical_id IS NOT NULL"),
                "aliases": q("SELECT COUNT(*) FROM aliases"),
                "species_images": q("SELECT COUNT(*) FROM species_images"),
                "usage_rows": q("SELECT COUNT(*) FROM usage_stats"),
                "sets": q("SELECT COUNT(*) FROM sets"),
            }
    except Exception as e:
        out["status"] = "no data"
        out["error"] = str(e)
        out["hint"] = "Run ingest.py to build the database."
        return out

    out["generations_loaded"] = meta("dex_gens", [DEFAULT_GEN])
    out["data_dates"] = {
        "dex": meta("dex_ingested_at"),
        "wiki": meta("wiki_ingested_at"),
        "stats_latest_month": meta("stats_latest_month"),
        "stats_months_available": meta("stats_months", []),
    }
    n_emb = int(S.embeddings.shape[0]) if S.embeddings is not None else 0
    indexed_model = meta("embed_model", EMBED_MODEL)
    indexed_doc_prefix = meta("embed_doc_prefix", "")
    # Two different things, easy to conflate: how the STORED INDEX was built
    # (a one-time fact, recorded by ingest.py and never updated afterward) vs.
    # what serves QUERIES right now (live, reads current EMBED_URL). Reporting
    # only the first under a bare "backend" key reads as "this is what's
    # running now" and caused real confusion — the index can say "remote:..."
    # from whenever it was last embedded on GPU, while queries have since
    # moved to local CPU with EMBED_URL unset, and nothing about that field
    # changes to reflect it.
    out["embeddings"] = {
        "loaded": load_embeddings(),
        "count": n_emb,
        "model": indexed_model,
        "index_built_with": meta("embed_backend", "local:fastembed"),
        "serving_queries_with": f"remote:{EMBED_URL}" if EMBED_URL else "local:fastembed",
        "query_threads": EMBED_THREADS,
        # Embeddings are written in one shot at the end of ingest, so a count
        # that lags the chunk total means an ingest is still running (or died).
        "matches_chunks": n_emb == out["counts"]["wiki_chunks"],
    }
    # A model or prefix change requires a full --wiki-only re-embed — mixing
    # vectors from two different models/prefixes in one index degrades
    # retrieval with no error anywhere. Surface the mismatch rather than let
    # it fail silently.
    if indexed_model and indexed_model != EMBED_MODEL:
        out["embeddings"]["warning"] = (
            f"Index was built with {indexed_model!r} but EMBED_MODEL is now "
            f"{EMBED_MODEL!r}. Re-run ingest.py --wiki-only to rebuild the "
            f"index with the current model, or fix EMBED_MODEL to match."
        )
    elif EMBED_QUERY_PREFIX and not indexed_doc_prefix:
        out["embeddings"]["warning"] = (
            "EMBED_QUERY_PREFIX is set but the index was built with no "
            "EMBED_DOC_PREFIX. Query and document vectors are asymmetric — "
            "re-run ingest.py --wiki-only after setting EMBED_DOC_PREFIX to "
            "match this model's convention."
        )

    # Live ingest progress, published by ingest.py into the meta table.
    ing = meta("ingest_status")
    if isinstance(ing, dict):
        out["ingest"] = dict(ing)
        try:
            seen = time.mktime(time.strptime(ing["updated_at"], "%Y-%m-%dT%H:%M:%SZ"))
            age = int(time.time() - time.mktime(time.gmtime()) + time.time() - seen)
            out["ingest"]["seconds_since_update"] = max(age, 0)
            if ing.get("stage") not in ("complete", "failed", "interrupted") and age > 900:
                out["ingest"]["warning"] = (
                    "No progress update in over 15 minutes. The ingest may have "
                    "died — check with: docker top pokedex-api"
                )
        except Exception:
            pass
        if ing.get("stage") == "complete":
            out["status"] = "ok"
        elif ing.get("stage") in ("failed", "interrupted"):
            out["status"] = "incomplete"
            out["hint"] = f"Last ingest {ing['stage']}: {ing.get('detail', '')}"
        else:
            out["status"] = "building"
            out["hint"] = f"Ingest in progress — stage: {ing.get('stage')}"
            return out
    if not out["counts"]["wiki_chunks"]:
        out["status"] = "incomplete"
        out["hint"] = "No wiki chunks. Run: ingest.py --wiki-only"
    return out


# ---------------------------------------------------------------------------
# lookup
# ---------------------------------------------------------------------------

@app.post(
    "/lookup",
    operation_id="lookup",
    summary="Exact data for one Pokemon, move, ability, or item",
    description=(
        "Returns the authoritative record for a single named entity: base stats, "
        "typing, abilities, evolution, tier for a Pokemon; power, accuracy, type and "
        "effect for a move; the effect text for an ability or item.\n\n"
        "Use this FIRST for any question about a specific named thing. It is exact "
        "structured data, not prose. This includes a PURELY VISUAL request with no "
        "other question attached — \"show me X\", \"what does X look like\" — since "
        "the species image is attached to your reply as a side effect of this call; "
        "there is no other way to produce it.\n\n"
        "Pass moves_to_check to answer 'can X learn Y' — never answer that from "
        "memory. The reply says HOW it learns it (level-up, TM, tutor, egg) and "
        "at what level.\n\n"
        "Pass include_learnset=true with a gen for 'what moves does X learn and "
        "when', the question someone playing through a game asks. Older "
        "generations only work if they are in this build's scope — check "
        "/health for generations_loaded.\n\n"
        "Do NOT use this for: lore, anime appearances, where to catch something, or "
        "why a Pokemon is good competitively. Use search_wiki or usage_stats for those.\n\n"
        "A species result includes image_url when available (official artwork "
        "when cataloged, Bulbapedia box art otherwise, served locally either way). "
        "When official artwork is available it also includes sprite_icon_url "
        "(small), sprite_preview_url (large), sprite_shiny_icon_url/"
        "sprite_shiny_preview_url for the shiny variant, and alternate_forms — a "
        "list of Megas/regional/Gmax/other forms this species has, each with its "
        "own icon/preview URLs and a forme_name that is null when a real image "
        "exists but its specific form wasn't confidently identified (still show "
        "the image; just don't assert which form it is). On the standalone page "
        "these render automatically; if you are running elsewhere, you may "
        "include image_url as a markdown image when a visual reference would help."
    ),
)
def lookup(req: LookupRequest):
    gen = req.gen or DEFAULT_GEN
    hit = resolve(req.name)
    if not hit:
        return {"found": False, "query": req.name,
                "hint": "No entity matched. Check spelling, or try search_wiki for non-dex subjects."}

    cid, kind = hit["canonical_id"], hit["kind"]

    if cid.startswith("article:"):
        return {"found": True, "kind": "article", "canonical_id": cid,
                "matched_alias": hit["alias"],
                "hint": "This is a wiki article, not a dex entity. Use search_wiki."}

    with db() as c:
        if kind == "species":
            r = c.execute("SELECT * FROM species WHERE id=? AND gen=?", (cid, gen)).fetchone()
            if not r:
                return {"found": False, "query": req.name,
                        "hint": f"{cid} does not exist in generation {gen}."}
            # "level: 36" on Swampert's record means Marshtomp evolves INTO
            # Swampert at 36 — not that Swampert evolves at 36. Spell that out,
            # because the bare field reads the other way round.
            evo = {
                "prevo": r["prevo"], "evos": json.loads(r["evos"]),
                "level": r["evo_level"], "item": r["evo_item"],
                "condition": r["evo_condition"], "method": r["evo_type"],
            }
            if r["prevo"]:
                how_from_prevo = []
                if r["evo_level"]:
                    how_from_prevo.append(f"at level {r['evo_level']}")
                if r["evo_item"]:
                    how_from_prevo.append(f"using {r['evo_item']}")
                if r["evo_condition"]:
                    how_from_prevo.append(str(r["evo_condition"]))
                if r["evo_type"]:
                    how_from_prevo.append(f"via {r['evo_type']}")
                evo["evolves_from"] = (
                    f"{r['prevo'].capitalize()} evolves into {r['name']} "
                    + (" ".join(how_from_prevo) if how_from_prevo else "(method unlisted)")
                )
            if evo["evos"]:
                evo["evolves_into"] = (
                    f"{r['name']} evolves into "
                    + ", ".join(e.capitalize() for e in evo["evos"])
                    + " — see those entries for the level or method"
                )
            elif not r["nfe"]:
                evo["evolves_into"] = f"{r['name']} is fully evolved."

            out = {
                "found": True, "kind": "species", "id": r["id"], "name": r["name"],
                "national_dex": r["num"], "gen": gen,
                "types": json.loads(r["types"]),
                "base_stats": {"hp": r["hp"], "atk": r["atk"], "def": r["def_"],
                               "spa": r["spa"], "spd": r["spd"], "spe": r["spe"]},
                "bst": r["bst"],
                "abilities": json.loads(r["abilities"]),
                "evolution": evo,
                "egg_groups": json.loads(r["egg_groups"]),
                "weight_kg": r["weight_kg"], "height_m": r["height_m"],
                "tier": r["tier"], "doubles_tier": r["doubles_tier"],
                "fully_evolved": not r["nfe"],
                "base_species": r["base_species"],
                "battle_only": is_battle_only(r["battle_only"]),
                "required_item": r["required_item"],
            }

            # HOME sprites (this species' OWN form — base or a specific
            # Mega/regional forme — non-shiny) take priority over the
            # Bulbapedia extraction's inconsistent webp crops. image_url is
            # kept as the single field every existing caller already reads.
            icon_path, preview_path = species_sprite(c, r["num"], r["forme"], r["name"])
            if icon_path or preview_path:
                out["image_url"] = preview_path or icon_path
                out["sprite_icon_url"] = icon_path
                out["sprite_preview_url"] = preview_path
            else:
                img = c.execute(
                    "SELECT image_url FROM species_images WHERE species_norm=?",
                    (norm(r["name"]),),
                ).fetchone()
                out["image_url"] = img["image_url"] if img else None
                out["sprite_icon_url"] = None
                out["sprite_preview_url"] = None

            # Shiny of the SAME form just resolved above — every catalogued
            # alternate form (Megas, regional variants, Gmax, cosmetic) has
            # its own icon/preview when available. forme_name is only ever
            # set when confidently matched against @pkmn/dex's own forme
            # ordering (see ingest.py) — an entry with forme_name=None is a
            # real image whose specific forme just wasn't verified, not a
            # placeholder.
            shiny_icon, shiny_preview = species_sprite(c, r["num"], r["forme"], r["name"], shiny=True)
            out["sprite_shiny_icon_url"] = shiny_icon
            out["sprite_shiny_preview_url"] = shiny_preview

            # req.shiny makes the shiny variant the PRIMARY displayed image
            # (what image_url points at, what the card actually shows) rather
            # than just being mentioned as available alongside the regular
            # one — that was the actual gap: the model could already SEE that
            # a shiny existed, it just had no way to make it the one shown.
            out["showing_shiny"] = False
            if req.shiny:
                if shiny_icon or shiny_preview:
                    out["image_url"] = shiny_preview or shiny_icon
                    out["sprite_icon_url"] = shiny_icon
                    out["sprite_preview_url"] = shiny_preview
                    out["showing_shiny"] = True
                else:
                    out["shiny_unavailable"] = True

            alt_forms = c.execute(
                # Gmax reuses form_index=0 (confirmed: Gmax Charizard is
                # form 000, category=g — same index as the base form), so
                # "form_index>0" alone silently excluded every Gmax variant.
                # The real distinction is "not the plain base entry" — which
                # is form_index=0 AND is_gmax=0, regardless of shininess
                # (base shiny is already surfaced via sprite_shiny_* above,
                # not meant to duplicate into this list).
                "SELECT form_index, is_gmax, is_shiny, forme_name, icon_path, preview_path "
                "FROM home_sprites WHERE natdex=? AND NOT (form_index=0 AND is_gmax=0) "
                "ORDER BY form_index, is_gmax, is_shiny",
                (r["num"],),
            ).fetchall()
            if alt_forms:
                out["alternate_forms"] = [
                    {
                        "forme_name": a["forme_name"],
                        "is_gmax": bool(a["is_gmax"]),
                        "is_shiny": bool(a["is_shiny"]),
                        "icon_url": a["icon_path"],
                        "preview_url": a["preview_path"],
                    }
                    for a in alt_forms
                ]

            # Gens 1 and 2 had ONE Special stat. The data layer stores it in both
            # spa and spd, which is faithful to how the simulator models it but
            # misleading if reported as two separate stats for those games.
            if gen <= 2:
                out["base_stats"] = {
                    "hp": r["hp"], "atk": r["atk"], "def": r["def_"],
                    "spc": r["spa"], "spe": r["spe"],
                }
                out["stat_note"] = (
                    f"Generation {gen} had a single Special stat (spc), not "
                    "separate Special Attack and Special Defense. The split "
                    "happened in generation 3. Report it as 'Special', not as "
                    "two stats."
                )
                if gen == 1:
                    out["stat_note"] += (
                        " Generation 1 also had no separate Special Defense in "
                        "damage calculation at all."
                    )
            out["moves_known"] = c.execute(
                "SELECT COUNT(*) FROM learnsets WHERE species_id=? AND gen=?", (cid, gen)
            ).fetchone()[0]

            if req.moves_to_check:
                checks = {}
                for mv in req.moves_to_check:
                    mh = resolve(mv, kind="move")
                    if not mh:
                        checks[mv] = {"learns": None, "note": "move not recognized"}
                        continue
                    srcs = c.execute(
                        "SELECT method, level, source_gen FROM learnsets "
                        "WHERE species_id=? AND gen=? AND move_id=? "
                        "ORDER BY source_gen DESC, level",
                        (cid, gen, mh["canonical_id"]),
                    ).fetchall()
                    if not srcs:
                        checks[mv] = {"learns": False, "move_id": mh["canonical_id"]}
                        continue
                    how = []
                    for r2 in srcs:
                        entry = {"method": r2["method"], "from_gen": r2["source_gen"]}
                        if r2["method"] == "level-up" and r2["level"]:
                            entry["level"] = r2["level"]
                        how.append(entry)

                    # Prefer the level from the generation actually asked about.
                    # Sources are stored newest-first, so taking the first
                    # level-up entry would answer a gen 3 question with gen 9's
                    # level — which is a different number often enough to matter.
                    same_gen = [h for h in how
                                if h["method"] == "level-up"
                                and h.get("from_gen") == gen
                                and h.get("level")]
                    lvl = same_gen[0]["level"] if same_gen else None
                    other = None
                    if lvl is None:
                        fallback = [h for h in how
                                    if h["method"] == "level-up" and h.get("level")]
                        if fallback:
                            other = fallback[0]

                    # Put the requested generation's sources first.
                    how.sort(key=lambda h: (h.get("from_gen") != gen,
                                            -(h.get("from_gen") or 0)))

                    entry = {
                        "learns": True,
                        "move_id": mh["canonical_id"],
                        "level_up_at": lvl,
                        "how": how,
                    }
                    if lvl is None and other:
                        entry["level_up_at"] = None
                        entry["note"] = (
                            f"Not learned by level-up in generation {gen}. "
                            f"It is at level {other['level']} in generation "
                            f"{other['from_gen']}; in gen {gen} it comes from "
                            + ", ".join(sorted({h['method'] for h in how
                                                if h.get('from_gen') == gen}))
                            + "." if any(h.get("from_gen") == gen for h in how)
                            else f"Not learned by level-up in generation {gen}."
                        )
                    checks[mv] = entry
                out["learnset_check"] = checks
                out["note"] = ("Learnset data says the move exists for this Pokemon in this "
                               "generation. For whether a specific COMBINATION of moves is "
                               "legal together, use validate_team.")

            if req.include_learnset:
                lv = c.execute(
                    """SELECT l.move_id, l.level, m.name, m.type, m.category,
                              m.base_power, m.accuracy, m.short_desc
                         FROM learnsets l
                         LEFT JOIN moves m ON m.id = l.move_id AND m.gen = l.gen
                        WHERE l.species_id=? AND l.gen=? AND l.method='level-up'
                          AND l.source_gen=?
                        ORDER BY l.level, m.name""",
                    (cid, gen, gen),
                ).fetchall()
                out["level_up_learnset"] = [
                    {"level": r2["level"], "move": r2["name"] or r2["move_id"],
                     "type": r2["type"], "category": r2["category"],
                     "power": r2["base_power"], "accuracy": r2["accuracy"],
                     "effect": r2["short_desc"]}
                    for r2 in lv
                ]
                others = c.execute(
                    """SELECT DISTINCT l.method, COUNT(*) n FROM learnsets l
                        WHERE l.species_id=? AND l.gen=? AND l.method != 'level-up'
                        GROUP BY l.method""",
                    (cid, gen),
                ).fetchall()
                out["other_learn_methods"] = {r2["method"]: r2["n"] for r2 in others}
                if not out["level_up_learnset"]:
                    out["learnset_note"] = (
                        f"No level-up data for generation {gen}. If you need an "
                        f"older generation, it must be in this build's GENS scope "
                        f"— check /health for generations_loaded."
                    )
            return out

        if kind == "move":
            r = c.execute("SELECT * FROM moves WHERE id=? AND gen=?", (cid, gen)).fetchone()
            if not r:
                return {"found": False, "query": req.name,
                        "hint": f"{cid} does not exist in generation {gen}."}
            return {"found": True, "kind": "move", "id": r["id"], "name": r["name"],
                    "gen": gen, "type": r["type"], "category": r["category"],
                    "base_power": r["base_power"], "accuracy": r["accuracy"],
                    "pp": r["pp"], "priority": r["priority"], "target": r["target"],
                    "flags": json.loads(r["flags"] or "{}"),
                    "effect": r["short_desc"], "detail": r["description"]}

        for k, table in (("ability", "abilities"), ("item", "items")):
            if kind == k:
                r = c.execute(f"SELECT * FROM {table} WHERE id=? AND gen=?", (cid, gen)).fetchone()
                if not r:
                    return {"found": False, "query": req.name,
                            "hint": f"{cid} does not exist in generation {gen}."}
                return {"found": True, "kind": k, "id": r["id"], "name": r["name"],
                        "gen": gen, "effect": r["short_desc"], "detail": r["description"]}

    return {"found": False, "query": req.name, "kind": kind}


# ---------------------------------------------------------------------------
# query_dex
# ---------------------------------------------------------------------------

def resolve_batch_images(c: sqlite3.Connection, rows: list) -> tuple[dict, dict]:
    """
    Given a list of species rows, resolve sprite images for all of them in a
    small, fixed number of queries rather than one per row. Shared between
    query_dex and dex_index so image resolution stays correct in exactly one
    place — the same reasoning that led to species_sprite() existing at all.

    HOME sprites are keyed by natdex, checked first; species_images
    (Bulbapedia-derived) is keyed by normalized name and only consulted as a
    fallback for whatever natdex numbers HOME didn't cover.

    Split into two paths: a fast batched query for ordinary base-form rows
    (natdex alone is unambiguous there), and individual forme-aware lookups
    for any row that's a Mega/regional form — a natdex number is shared
    across a species' base form AND every Mega/regional variant of it, so a
    single batched query keyed only by natdex can't tell them apart.

    Returns (home_images, images): home_images keyed by species id (unique
    per row, unlike natdex which collides across formes on purpose), images
    keyed by normalized name as a species_images fallback.
    """
    home_images: dict[str, str] = {}
    if rows:
        base_rows = [r for r in rows if not r["forme"]]
        forme_rows = [r for r in rows if r["forme"]]

        if base_rows:
            nums = list({r["num"] for r in base_rows})
            ph = ",".join("?" for _ in nums)
            by_natdex = {}
            for hr in c.execute(
                f"SELECT natdex, icon_path, preview_path FROM home_sprites "
                f"WHERE natdex IN ({ph}) AND form_index=0 AND is_gmax=0 AND is_shiny=0",
                nums):
                by_natdex[hr["natdex"]] = hr["preview_path"] or hr["icon_path"]
            for r in base_rows:
                if r["num"] in by_natdex:
                    home_images[r["id"]] = by_natdex[r["num"]]

        for r in forme_rows:
            icon_path, preview_path = species_sprite(c, r["num"], r["forme"], r["name"])
            if icon_path or preview_path:
                home_images[r["id"]] = preview_path or icon_path

    images: dict[str, str] = {}
    uncovered = [r for r in rows if r["id"] not in home_images]
    if uncovered:
        norms = [norm(r["name"]) for r in uncovered]
        ph = ",".join("?" for _ in norms)
        for ir in c.execute(
            f"SELECT species_norm, image_url FROM species_images "
            f"WHERE species_norm IN ({ph})", norms):
            images[ir["species_norm"]] = ir["image_url"]

    return home_images, images


def dex_row(r: sqlite3.Row, home_images: dict, images: dict) -> dict:
    """The species-row shape shared by query_dex and dex_index."""
    return {
        "id": r["id"], "name": r["name"], "num": r["num"],
        "types": json.loads(r["types"]),
        "base_stats": {"hp": r["hp"], "atk": r["atk"], "def": r["def_"],
                        "spa": r["spa"], "spd": r["spd"], "spe": r["spe"]},
        "bst": r["bst"], "abilities": json.loads(r["abilities"]),
        "tier": r["tier"], "fully_evolved": not r["nfe"],
        "base_species": r["base_species"], "battle_only": is_battle_only(r["battle_only"]),
        "required_item": r["required_item"], "forme": r["forme"],
        "image_url": home_images.get(r["id"]) or images.get(norm(r["name"])),
    }


@app.post(
    "/query_dex",
    operation_id="query_dex",
    summary="Find Pokemon matching structural criteria",
    description=(
        "Filtered search over the dex. Answers questions of the form 'which Pokemon "
        "are X and also Y' — e.g. Water types with base Speed over 100 that learn "
        "Rapid Spin, or unevolved Steel types under 400 BST.\n\n"
        "Use this whenever the question asks for a SET of Pokemon by property rather "
        "than facts about one named Pokemon.\n\n"
        "Do NOT use this to check one Pokemon — use lookup. Do NOT use it for "
        "'what's good in the meta' — use usage_stats."
    ),
)
def query_dex(req: QueryDexRequest):
    gen = req.gen or DEFAULT_GEN
    where = ["s.gen = ?"]
    params: list = [gen]

    if req.types:
        if req.type_mode == "all":
            for t in req.types:
                where.append("s.types LIKE ?")
                params.append(f'%"{t.capitalize()}"%')
        else:
            where.append("(" + " OR ".join("s.types LIKE ?" for _ in req.types) + ")")
            params += [f'%"{t.capitalize()}"%' for t in req.types]

    if req.ability:
        ah = resolve(req.ability, kind="ability")
        needle = ah["canonical_id"] if ah else norm(req.ability)
        where.append("LOWER(REPLACE(REPLACE(s.abilities,' ',''),'-','')) LIKE ?")
        params.append(f"%{needle}%")

    if req.egg_group:
        where.append("s.egg_groups LIKE ?")
        params.append(f'%"{req.egg_group.capitalize()}"%')
    if req.tier:
        where.append("UPPER(s.tier) = ?")
        params.append(req.tier.upper())
    if req.min_bst is not None:
        where.append("s.bst >= ?"); params.append(req.min_bst)
    if req.max_bst is not None:
        where.append("s.bst <= ?"); params.append(req.max_bst)
    if req.nfe is not None:
        where.append("s.nfe = ?"); params.append(1 if req.nfe else 0)

    col = {"hp": "s.hp", "atk": "s.atk", "def": "s.def_", "spa": "s.spa",
           "spd": "s.spd", "spe": "s.spe"}
    if req.stat_filters:
        for stat, bounds in req.stat_filters.items():
            c_ = col.get(stat.lower())
            if not c_:
                continue
            if "min" in bounds:
                where.append(f"{c_} >= ?"); params.append(bounds["min"])
            if "max" in bounds:
                where.append(f"{c_} <= ?"); params.append(bounds["max"])

    join = ""
    if req.learns:
        move_ids = []
        for mv in req.learns:
            mh = resolve(mv, kind="move")
            if not mh:
                return {"count": 0, "results": [],
                        "error": f"Move not recognized: {mv!r}"}
            move_ids.append(mh["canonical_id"])
        placeholders = ",".join("?" for _ in move_ids)
        join = (f" AND s.id IN (SELECT species_id FROM learnsets WHERE gen = ? "
                f"AND move_id IN ({placeholders}) GROUP BY species_id "
                f"HAVING COUNT(DISTINCT move_id) = ?)")
        params += [gen] + move_ids + [len(move_ids)]

    order = {"bst": "s.bst", "spe": "s.spe", "atk": "s.atk", "spa": "s.spa",
             "hp": "s.hp", "def": "s.def_", "spd": "s.spd", "num": "s.num"}[req.order_by]

    sql = (f"SELECT s.* FROM species s WHERE {' AND '.join(where)}{join} "
           f"ORDER BY {order} DESC LIMIT ?")
    params.append(req.limit)

    with db() as c:
        rows = c.execute(sql, params).fetchall()
        total = c.execute(
            f"SELECT COUNT(*) FROM species s WHERE {' AND '.join(where)}{join}",
            params[:-1],
        ).fetchone()[0]

        # One query for every row's image rather than one query per row —
        # this endpoint is specifically the multi-Pokemon case, so batching
        # matters more here than in /lookup's single-species path.
        home_images, images = resolve_batch_images(c, rows)

    return {
        "count": total,
        "returned": len(rows),
        "gen": gen,
        "results": [dex_row(r, home_images, images) for r in rows],
        "note": ("Legal in principle per the learnset data. Whether a move COMBINATION "
                 "is legal together still needs validate_team." if req.learns else None),
    }


# Module-level cache: correct for the single-process deployment this whole
# project assumes (confirmed throughout — one uvicorn worker per container).
# Would need a shared cache (Redis, a file, etc.) instead if this container
# ever ran with multiple workers, since each worker would otherwise rebuild
# its own copy independently on first request.
_DEX_INDEX_CACHE: dict = {"mtime": None, "data": None}


@app.get(
    "/dex_index",
    operation_id="dex_index",
    summary="Every species and form, one row each, newest generation of each (UI data feed)",
    description=(
        "The complete species list in one response — every species AND every "
        "known form (Megas, regional forms, Gigantamax, automatic battle forms), "
        "one row per id, taken from that id's newest generation (a species missing "
        "from the current generation, like Aegislash, still appears — from its own "
        "most recent gen instead). Built for a UI's full Pokedex view to load in one "
        "request instead of paging through query_dex; not a chat tool, not in "
        "TOOL_SCHEMAS."
    ),
)
def dex_index():
    if not os.path.exists(DB_PATH):
        return {"count": 0, "results": [], "hint": "No data loaded yet."}

    mtime = os.path.getmtime(DB_PATH)
    if _DEX_INDEX_CACHE["data"] is not None and _DEX_INDEX_CACHE["mtime"] == mtime:
        return _DEX_INDEX_CACHE["data"]

    with db() as c:
        rows = c.execute(
            """
            SELECT s.* FROM species s
            JOIN (SELECT id, MAX(gen) AS g FROM species GROUP BY id) m
              ON s.id = m.id AND s.gen = m.g
            """
        ).fetchall()
        if not rows:
            return {"count": 0, "results": [], "hint": "No data loaded yet."}
        home_images, images = resolve_batch_images(c, rows)
        results = [dex_row(r, home_images, images) for r in rows]

    data = {"count": len(results), "results": results}
    _DEX_INDEX_CACHE["mtime"] = mtime
    _DEX_INDEX_CACHE["data"] = data
    return data


# ---------------------------------------------------------------------------
# search_wiki
# ---------------------------------------------------------------------------

def _fts_query(text: str) -> str:
    """FTS5 chokes on punctuation; reduce to bare terms."""
    terms = re.findall(r"[A-Za-z0-9']+", text)
    terms = [t for t in terms if len(t) > 1][:12]
    return " OR ".join(f'"{t}"' for t in terms) if terms else '""'


@app.post(
    "/search_wiki",
    operation_id="search_wiki",
    summary="Search Bulbapedia and Smogon analyses",
    description=(
        "Searches encyclopedic prose: anime episodes and characters, manga, the "
        "trading card game, where to catch Pokemon in each game, event "
        "distributions, how mechanics work, trivia, and Smogon's written "
        "competitive analyses.\n\n"
        "Use this for narrative, descriptive, or explanatory questions — anything "
        "that needs prose rather than numbers.\n\n"
        "Do NOT use this for base stats, typing, learnsets, or type matchups. Those "
        "live in lookup and query_dex as exact data, and the prose version may be "
        "outdated or contradict it.\n\n"
        "Set domain to narrow the search when the question is clearly about one body "
        "of content — 'anime' for episode questions, 'games' for in-game locations."
    ),
)
def search_wiki(req: SearchWikiRequest):
    if not load_embeddings():
        semantic_ok = False
    else:
        semantic_ok = True

    entity_id = None
    if req.entity:
        h = resolve(req.entity)
        entity_id = h["canonical_id"] if h else None

    filters, fparams = [], []
    if entity_id and not entity_id.startswith("article:"):
        filters.append("wc.canonical_id = ?"); fparams.append(entity_id)
    if req.domain:
        filters.append("wc.domain = ?"); fparams.append(req.domain)
    if req.gen is not None:
        filters.append("(wc.gen = ? OR wc.gen IS NULL)"); fparams.append(req.gen)
    where = (" AND " + " AND ".join(filters)) if filters else ""

    pool = max(req.limit * 8, 60)

    # --- lexical ----------------------------------------------------------
    lexical: list[int] = []
    with db() as c:
        try:
            rows = c.execute(
                f"""SELECT wc.id FROM wiki_fts f
                     JOIN wiki_chunks wc ON wc.id = f.rowid
                    WHERE wiki_fts MATCH ?{where}
                    ORDER BY bm25(wiki_fts) LIMIT ?""",
                [_fts_query(req.query)] + fparams + [pool],
            ).fetchall()
            lexical = [r["id"] for r in rows]
        except sqlite3.OperationalError:
            lexical = []

    # --- semantic ---------------------------------------------------------
    semantic: list[int] = []
    if semantic_ok:
        try:
            q = embed_query(req.query)
            vecs, ids = S.embeddings, S.embedding_ids
            scores = np.empty(vecs.shape[0], dtype=np.float32)
            step = 50_000
            for i in range(0, vecs.shape[0], step):
                block = np.asarray(vecs[i:i + step], dtype=np.float32)
                scores[i:i + step] = block @ q
            top = np.argpartition(-scores, min(pool * 3, len(scores) - 1))[: pool * 3]
            top = top[np.argsort(-scores[top])]
            candidate_ids = [int(ids[i]) for i in top]

            if filters:
                with db() as c:
                    ph = ",".join("?" for _ in candidate_ids)
                    keep = {
                        r["id"] for r in c.execute(
                            f"SELECT id FROM wiki_chunks wc WHERE wc.id IN ({ph}){where}",
                            candidate_ids + fparams,
                        )
                    }
                candidate_ids = [i for i in candidate_ids if i in keep]
            semantic = candidate_ids[:pool]
        except Exception:
            semantic = []

    # --- reciprocal rank fusion ------------------------------------------
    fused: dict[int, float] = {}
    for rank, cid in enumerate(lexical):
        fused[cid] = fused.get(cid, 0.0) + 1.0 / (RRF_K + rank + 1)
    for rank, cid in enumerate(semantic):
        fused[cid] = fused.get(cid, 0.0) + 1.0 / (RRF_K + rank + 1)

    if not fused:
        return {"count": 0, "results": [],
                "hint": "Nothing matched. Try different wording, or drop the domain filter."}

    best = sorted(fused, key=lambda k: -fused[k])[: req.limit]

    with db() as c:
        ph = ",".join("?" for _ in best)
        rows = {r["id"]: r for r in c.execute(
            f"SELECT * FROM wiki_chunks WHERE id IN ({ph})", best)}

    results = []
    for cid in best:
        r = rows.get(cid)
        if not r:
            continue
        text = r["text"]
        results.append({
            "title": r["article_title"],
            "section": r["section_path"],
            "domain": r["domain"],
            "gen": r["gen"],
            "source": r["source"],
            "url": r["url"] or None,
            "text": text[:MAX_CHUNK_CHARS] + ("…" if len(text) > MAX_CHUNK_CHARS else ""),
        })

    return {
        "count": len(results),
        "retrieval": "hybrid" if (lexical and semantic) else ("lexical" if lexical else "semantic"),
        "data_date": meta("wiki_ingested_at"),
        "results": results,
    }


# ---------------------------------------------------------------------------
# usage_stats
# ---------------------------------------------------------------------------

@app.post(
    "/usage_stats",
    operation_id="usage_stats",
    summary="Smogon competitive usage statistics",
    description=(
        "Real ladder data: how often each Pokemon is used in a format, its common "
        "moves, items, abilities and EV spreads, its frequent teammates, and what "
        "checks and counters it.\n\n"
        "Use this for anything about the CURRENT metagame — what's popular, what "
        "beats what, how people actually build a Pokemon.\n\n"
        "IMPORTANT: these statistics are descriptive, not prescriptive. They measure "
        "what people play, not what is optimal. Low usage does not mean bad. Say so "
        "when it matters.\n\n"
        "Do NOT use this for base stats or legality — those are lookup and "
        "validate_team."
    ),
)
def usage_stats(req: UsageStatsRequest):
    month = req.month or meta("stats_latest_month")
    if not month:
        return {"available": False,
                "hint": "No usage statistics loaded. Run fetch-stats.sh then "
                        "ingest.py --stats-only."}

    with db() as c:
        if req.species:
            h = resolve(req.species, kind="species")
            sid = h["canonical_id"] if h else norm(req.species)
            r = c.execute(
                "SELECT * FROM usage_stats WHERE format=? AND month=? AND species_id=?",
                (req.format, month, sid),
            ).fetchone()
            if not r:
                near = c.execute(
                    "SELECT species_name, usage FROM usage_stats WHERE format=? AND month=? "
                    "ORDER BY usage DESC LIMIT 5", (req.format, month)).fetchall()
                return {"available": False, "format": req.format, "month": month,
                        "hint": f"{req.species} has no entry in {req.format} for {month}. "
                                f"It may be unused or in a different tier.",
                        "top_in_format": [x["species_name"] for x in near]}
            return {
                "available": True, "format": req.format, "month": month,
                "cutoff": r["cutoff"], "name": r["species_name"],
                "usage_pct": round((r["usage"] or 0) * 100, 2),
                "raw_count": r["raw_count"],
                "moves": json.loads(r["moves"]),
                "items": json.loads(r["items"]),
                "abilities": json.loads(r["abilities"]),
                "spreads": json.loads(r["spreads"]),
                "teammates": json.loads(r["teammates"]),
                "checks_and_counters": json.loads(r["counters"]),
                "note": "Percentages are shares within that category, not usage rates. "
                        "Checks and counters are scored 0-1; higher means a stronger check.",
            }

        rows = c.execute(
            "SELECT species_name, usage, raw_count FROM usage_stats "
            "WHERE format=? AND month=? ORDER BY usage DESC LIMIT ?",
            (req.format, month, req.limit),
        ).fetchall()

    if not rows:
        with db() as c:
            fmts = [r[0] for r in c.execute(
                "SELECT DISTINCT format FROM usage_stats").fetchall()]
        return {"available": False, "format": req.format, "month": month,
                "formats_available": fmts}

    return {
        "available": True, "format": req.format, "month": month,
        "ranking": [
            {"rank": i + 1, "name": r["species_name"],
             "usage_pct": round((r["usage"] or 0) * 100, 2),
             "raw_count": r["raw_count"]}
            for i, r in enumerate(rows)
        ],
        "months_available": meta("stats_months", []),
    }


@app.post(
    "/type_matchup",
    operation_id="type_matchup",
    summary="Type effectiveness, in any direction",
    description=(
        "Answers type questions exactly, from the real type chart for the "
        "requested generation.\n\n"
        "Three ways to call it:\n"
        "- attacking_type + defending_types: 'Fighting vs Dark/Steel' -> 4x\n"
        "- defending_types alone: the full defensive profile of that typing\n"
        "- species: the same profile, looked up by Pokemon name\n\n"
        "Use this for ANY 'what's super effective against X' or 'what resists Y' "
        "question. Never answer type questions from memory — the chart changed "
        "between generations, and this tool knows which generation you asked about."
    ),
)
def type_matchup(req: TypeMatchupRequest):
    gen = req.gen or DEFAULT_GEN

    defending = req.defending_types
    species_name = None
    if req.species:
        h = resolve(req.species, kind="species")
        if not h:
            return {"found": False, "hint": f"Unknown Pokemon: {req.species!r}"}
        with db() as c:
            r = c.execute("SELECT name, types FROM species WHERE id=? AND gen=?",
                          (h["canonical_id"], gen)).fetchone()
        if not r:
            return {"found": False,
                    "hint": f"{req.species} does not exist in generation {gen}."}
        defending = json.loads(r["types"])
        species_name = r["name"]

    with db() as c:
        available = c.execute(
            "SELECT COUNT(*) FROM typechart WHERE gen=?", (gen,)).fetchone()[0]
        if not available:
            gens = [r[0] for r in c.execute(
                "SELECT DISTINCT gen FROM typechart ORDER BY gen")]
            return {"found": False,
                    "hint": f"No type chart for generation {gen}. Available: {gens}"}

        # attacking type against a specific defending combination
        if req.attacking_type and defending:
            mult = 1.0
            parts = []
            for dt in defending:
                r = c.execute(
                    "SELECT multiplier FROM typechart WHERE gen=? AND "
                    "LOWER(defending_type)=LOWER(?) AND LOWER(attacking_type)=LOWER(?)",
                    (gen, dt, req.attacking_type)).fetchone()
                m = r["multiplier"] if r else 1.0
                mult *= m
                parts.append({"defending_type": dt, "multiplier": m})
            return {
                "found": True, "gen": gen,
                "attacking_type": req.attacking_type.capitalize(),
                "defending_types": defending,
                "species": species_name,
                "multiplier": mult,
                "label": _eff_label(mult),
                "breakdown": parts,
            }

        # full defensive profile
        if defending:
            rows = c.execute(
                "SELECT DISTINCT attacking_type FROM typechart WHERE gen=?", (gen,)
            ).fetchall()
            profile = {}
            for r in rows:
                at = r["attacking_type"]
                mult = 1.0
                for dt in defending:
                    m = c.execute(
                        "SELECT multiplier FROM typechart WHERE gen=? AND "
                        "LOWER(defending_type)=LOWER(?) AND attacking_type=?",
                        (gen, dt, at)).fetchone()
                    mult *= m["multiplier"] if m else 1.0
                profile[at] = mult
            return {
                "found": True, "gen": gen,
                "species": species_name, "defending_types": defending,
                "immune": sorted(k for k, v in profile.items() if v == 0),
                "quad_resists": sorted(k for k, v in profile.items() if 0 < v <= 0.25),
                "resists": sorted(k for k, v in profile.items() if 0.25 < v < 1),
                "neutral": sorted(k for k, v in profile.items() if v == 1),
                "weak": sorted(k for k, v in profile.items() if 1 < v < 4),
                "quad_weak": sorted(k for k, v in profile.items() if v >= 4),
            }

        # attacking type alone — what it hits and what walls it
        if req.attacking_type:
            rows = c.execute(
                "SELECT defending_type, multiplier FROM typechart WHERE gen=? AND "
                "LOWER(attacking_type)=LOWER(?)", (gen, req.attacking_type)).fetchall()
            if not rows:
                return {"found": False,
                        "hint": f"Unknown type {req.attacking_type!r} in gen {gen}."}
            return {
                "found": True, "gen": gen,
                "attacking_type": req.attacking_type.capitalize(),
                "super_effective_against": sorted(
                    r["defending_type"] for r in rows if r["multiplier"] > 1),
                "resisted_by": sorted(
                    r["defending_type"] for r in rows if 0 < r["multiplier"] < 1),
                "no_effect_against": sorted(
                    r["defending_type"] for r in rows if r["multiplier"] == 0),
                "note": "Single defending types. Dual types multiply — call again "
                        "with defending_types for a specific combination.",
            }

    return {"found": False,
            "hint": "Provide attacking_type, defending_types, or species."}


def _eff_label(mult: float) -> str:
    return {0: "no effect", 0.25: "doubly resisted", 0.5: "resisted",
            1: "neutral", 2: "super effective", 4: "doubly super effective"}.get(
        mult, f"{mult}x")


@app.post(
    "/get_sets",
    operation_id="get_sets",
    summary="Curated Smogon sets for a Pokemon",
    description=(
        "Returns real, named Smogon sets — moves, item, ability, nature, EVs and "
        "Tera type — as published in Pokemon Showdown's teambuilder.\n\n"
        "Use this whenever recommending how to build a Pokemon. Do NOT invent a "
        "moveset: quote a real set by name, then explain it.\n\n"
        "Sets tagged origin 'dex' are human-curated from the Strategy Dex. Sets "
        "tagged 'stats' are derived from ladder usage. Say which you are quoting."
    ),
)
def get_sets(req: GetSetsRequest):
    h = resolve(req.species, kind="species")
    sid = h["canonical_id"] if h else norm(req.species)

    with db() as c:
        if not c.execute("SELECT COUNT(*) FROM sets").fetchone()[0]:
            return {"available": False,
                    "hint": "No set data loaded. Run fetch-sets.sh then "
                            "ingest.py --sets-only."}
        sql = "SELECT * FROM sets WHERE species_id=?"
        params: list = [sid]
        if req.format:
            sql += " AND format=?"
            params.append(req.format)
        # No SQL-level ORDER BY/LIMIT here on purpose — sorting by the raw
        # format string put gen1/gen2 sets ahead of gen9 ones. Format names
        # like "gen11v1" (generation 1's "1v1" format, confirmed against the
        # real data: gen11v1, gen21v1 ... gen91v1 map cleanly to each gen's
        # own 1v1 format) sort BEFORE "gen2ou" lexicographically, since '1'
        # precedes '2' as a character — there is no length-aware or numeric
        # comparison in a plain string ORDER BY. With the default limit of 8,
        # a species with many older-gen dex sets (confirmed: Gengar has 137
        # rows total) could exhaust the entire limit on gen1-2 entries alone,
        # before a single modern set was ever reached — not a data gap, a
        # sort bug hiding data that was there the whole time.
        rows = c.execute(sql, params).fetchall()

    if not rows:
        return {"available": False, "species": req.species, "format": req.format,
                "hint": "No published sets for that Pokemon in that format."}

    # Generation is always the single digit immediately after "gen" — verified
    # against every distinct format string actually in this table (gen1 through
    # gen9, including every "N v1" and "N anythinggoes" variant); Pokemon has
    # no generation past 9, so a two-digit read here would only ever be wrong.
    def _fmt_gen(fmt: str) -> int:
        if fmt and fmt.startswith("gen") and len(fmt) > 3 and fmt[3].isdigit():
            return int(fmt[3])
        return 0

    rows = sorted(
        rows,
        key=lambda r: (
            0 if r["origin"] == "dex" else 1,   # curated sets first
            -_fmt_gen(r["format"]),              # most recent generation first
            r["format"],                         # stable tie-break within a gen
        ),
    )[:req.limit]

    return {
        "available": True,
        "species": rows[0]["species_name"],
        "count": len(rows),
        "sets": [
            {"format": r["format"], "name": r["set_name"], "origin": r["origin"],
             "moves": json.loads(r["moves"]), "item": r["item"],
             "ability": r["ability"], "nature": r["nature"],
             "evs": json.loads(r["evs"]), "tera": r["tera"], "level": r["level"]}
            for r in rows
        ],
    }


@app.post(
    "/common_generation",
    operation_id="common_generation",
    summary="The highest generation common to 2+ named Pokemon",
    description=(
        "The highest generation that includes every one of 2+ named Pokemon together. "
        "Call this FIRST, before anything else, whenever building a team around "
        "specific named Pokemon — not every Pokemon exists in every generation "
        "(Aegislash has no Gen 9 data, for example), and discovering that mid-build "
        "after a tool call fails is worse than knowing upfront. Use the returned gen "
        "for every other tool call in the rest of this request: validate_team, "
        "review_team, get_sets, calc_damage, type_matchup, lookup. Not needed for a "
        "single named Pokemon or a team with no specific seed."
    ),
)
def common_generation(req: CommonGenerationRequest):
    """
    The highest generation that includes every one of the given Pokemon.

    Built specifically because of a real, confirmed cascade: a team built
    around Aegislash (no Gen 9 data at all) defaulted to Gen 9 anyway, which
    meant review_team crashed partway through, the model had to retry with a
    smaller subset to get partial results, and untangling which retry
    reflected "the real team" versus "a workaround for a tool failure" turned
    into a genuinely hard problem downstream in card rendering. All of that
    was avoidable by knowing the right generation BEFORE starting, not by
    discovering the constraint one crashed tool call at a time.
    """
    per_species: dict[str, dict] = {}
    gen_sets: list[set[int]] = []

    with db() as c:
        for name in req.species:
            h = resolve(name, kind="species")
            sid = h["canonical_id"] if h else norm(name)
            gens = {r[0] for r in c.execute(
                "SELECT DISTINCT gen FROM species WHERE id=?", (sid,))}
            per_species[name] = {"resolved_id": sid, "available_gens": sorted(gens)}
            gen_sets.append(gens)

    common = set.intersection(*gen_sets) if gen_sets and all(gen_sets) else set()
    best_gen = max(common) if common else None

    hint = None
    if best_gen is None:
        hint = ("No single generation includes all of these Pokemon together — "
                "check available_gens per species below.")
    elif best_gen < DEFAULT_GEN:
        limiting = [name for name, info in per_species.items()
                    if info["available_gens"] and max(info["available_gens"]) < DEFAULT_GEN]
        if limiting:
            detail = ", ".join(
                f"{n} (through gen {max(per_species[n]['available_gens'])})" for n in limiting)
            hint = f"Capped below gen {DEFAULT_GEN} by: {detail}."

    return {
        "available": best_gen is not None,
        "gen": best_gen,
        "per_species": per_species,
        "hint": hint,
    }


@app.post(
    "/query_items",
    operation_id="query_items",
    summary="Browse or filter held items",
    description=(
        "A quick reference for held items when building a team — not a single named "
        "lookup (use lookup for that). Filter by is_choice, is_berry, is_mega_stone, "
        "or category ('type_boost' or 'stat_boost' — the only two item families with "
        "a clean category; everything else has no clean category field but a precise "
        "short_desc, so use query for free-text search instead — search SPECIFIC terms, "
        "not generic category words: 'Sunny Day' not 'weather', 'Electric Terrain' not "
        "'terrain', 'badly poison' not 'status'; 'burn', 'Ground-type', 'contact move', "
        "and 'hazards' all work directly). Combine filters with query for something "
        "like 'Fire-type boosting items' if category alone isn't specific enough."
    ),
)
def query_items(req: QueryItemsRequest):
    """
    Browse/filter held items. Combines the few clean boolean fields the dex
    itself provides (is_choice, is_berry, is_mega_stone) with a parsed
    category for the two item families that needed it (type_boost,
    stat_boost — see stage_item_categories in ingest.py for why those two
    specifically), and free-text search over short_desc for everything else.
    That last part carries real weight: status orbs, weather rocks, terrain
    seeds, gems, plates, and one-off items like Leftovers or Rocky Helmet
    have no clean category field at all, but their short_desc is precise
    enough (confirmed directly against real data) that a query like
    "weather" or "Ground-type" reliably finds the right items anyway.
    """
    gen = req.gen or DEFAULT_GEN

    where = ["i.gen = ?"]
    params: list = [gen]
    if req.is_choice is not None:
        where.append("i.is_choice = ?"); params.append(1 if req.is_choice else 0)
    if req.is_berry is not None:
        where.append("i.is_berry = ?"); params.append(1 if req.is_berry else 0)
    if req.is_mega_stone is not None:
        where.append("i.mega_evolves IS " + ("NOT NULL" if req.is_mega_stone else "NULL"))
    if req.query:
        where.append("(i.short_desc LIKE ? OR i.description LIKE ?)")
        needle = f"%{req.query}%"
        params += [needle, needle]

    join = ""
    if req.category:
        join = " JOIN item_categories ic ON ic.item_id = i.id AND ic.category = ?"
        params.insert(0, req.category)

    sql = f"SELECT DISTINCT i.* FROM items i{join} WHERE {' AND '.join(where)} ORDER BY i.name LIMIT ?"
    params.append(req.limit)

    with db() as c:
        rows = c.execute(sql, params).fetchall()
        if not rows:
            return {"count": 0, "results": []}
        ids = [r["id"] for r in rows]
        ph = ",".join("?" for _ in ids)
        cat_rows = c.execute(
            f"SELECT item_id, category, detail FROM item_categories WHERE item_id IN ({ph})", ids
        ).fetchall()

    categories: dict[str, list[dict]] = {}
    for cr in cat_rows:
        categories.setdefault(cr["item_id"], []).append(
            {"category": cr["category"], "detail": cr["detail"]})

    return {
        "count": len(rows),
        "results": [
            {
                "id": r["id"], "name": r["name"], "short_desc": r["short_desc"],
                "is_choice": bool(r["is_choice"]), "is_berry": bool(r["is_berry"]),
                "is_mega_stone": r["mega_evolves"] is not None,
                "mega_evolves": r["mega_evolves"],
                "categories": categories.get(r["id"], []),
            }
            for r in rows
        ],
    }


# ---------------------------------------------------------------------------
# sidecar proxies
# ---------------------------------------------------------------------------

async def _sim_post(path: str, payload: dict) -> dict:
    try:
        async with httpx.AsyncClient(timeout=60.0) as c:
            r = await c.post(f"{SIM_URL}{path}", json=payload)
            if r.status_code >= 500:
                raise HTTPException(502, f"pokedex-sim error: {r.text[:300]}")
            return r.json()
    except httpx.RequestError as e:
        raise HTTPException(
            503, f"Cannot reach pokedex-sim at {SIM_URL}. Is the container running? ({e})")


@app.post(
    "/validate_team",
    operation_id="validate_team",
    summary="Check a team's legality against a format's rules",
    description=(
        "Runs Pokemon Showdown's own TeamValidator — the exact code that would accept "
        "or reject this team on the ladder. This is the ground truth for legality.\n\n"
        "ALWAYS call this on any team you propose, BEFORE showing it to the user. If "
        "it reports problems, fix them and validate again. Never present an "
        "unvalidated or failing team.\n\n"
        "It catches things learnset data alone cannot: illegal move combinations, "
        "banned Pokemon and items, ability legality, clause violations, and "
        "event-move conflicts."
    ),
)
async def validate_team(req: ValidateTeamRequest):
    return await _sim_post("/validate_team", {"team": req.team, "format": req.format})


@app.post(
    "/calc_damage",
    operation_id="calc_damage",
    summary="Calculate damage for one attack",
    description=(
        "Runs the Smogon damage calculator. Returns the damage range, percentage of "
        "the defender's HP, and KO chance.\n\n"
        "Use this for any 'does X kill Y' or 'how much damage' question. Never "
        "estimate damage yourself — the formula involves too many modifiers.\n\n"
        "Unspecified fields default to level 100, neutral nature, and zero EVs. The "
        "response echoes the assumptions used; state them when you report the result."
    ),
)
async def calc_damage(req: CalcDamageRequest):
    return await _sim_post("/calc_damage", {
        "gen": req.gen, "attacker": req.attacker,
        "defender": req.defender, "move": req.move, "field": req.field or {},
    })


@app.post(
    "/compare_teams",
    operation_id="compare_teams",
    summary="Structurally compare two teams",
    description=(
        "Deterministic head-to-head analysis of two teams: full speed order, "
        "shared type weaknesses on each side, and a threat matrix built from real "
        "damage calculations for every attacker-defender pair.\n\n"
        "Use this for any 'how does team A do against team B' question, and for "
        "checking a team you just built against a known threat.\n\n"
        "CRITICAL: this returns structure, not a prediction. It contains NO win "
        "probability and none can be derived from it — there is no simulator and "
        "no outcome data in this system. Never state or estimate a percentage "
        "chance of either team winning, even if asked directly. Describe the "
        "matchup qualitatively from the speed tiers, weaknesses, and threats, and "
        "say plainly that a numeric win rate is not something you can compute."
    ),
)
async def compare_teams(req: CompareTeamsRequest):
    return await _sim_post("/compare_teams", {
        "team_a": req.team_a, "team_b": req.team_b,
        "format": req.format, "gen": req.gen,
        "label_a": req.label_a, "label_b": req.label_b,
    })


@app.post(
    "/review_team",
    operation_id="review_team",
    summary="Review one team for structural problems",
    description=(
        "CALL THIS FIRST for any request to review, rate, critique, or find "
        "problems with a team. Do not assemble the analysis yourself from "
        "individual lookups — this computes it correctly in one call.\n\n"
        "Returns, all calculated rather than recalled:\n"
        "- Actual speed stats at level, with EVs, nature and Choice Scarf applied. "
        "Quote these, not base Speed, which is a different and usually wrong number\n"
        "- Shared type weaknesses across the team, with counts\n"
        "- Which structural roles are covered and by which member: hazard removal, "
        "hazard setting, recovery, speed control, pivoting, status, setup\n"
        "- Every attacking type the team has, and which types nothing on it hits "
        "super effectively\n"
        "- A plain list of structural gaps\n\n"
        "Role coverage is checked against the actual movesets, so it will not "
        "claim a team lacks pivoting when a member runs U-turn or Volt Switch.\n\n"
        "Needs the team in Showdown export format. If the user pasted something "
        "looser, convert it and call this anyway rather than skipping it. Use "
        "validate_team separately for legality; this checks structure."
    ),
)
async def review_team(req: ReviewTeamRequest):
    return await _sim_post("/review_team", {
        "team": req.team, "format": req.format, "gen": req.gen})


# ---------------------------------------------------------------------------
# chat orchestrator — for the standalone page
# ---------------------------------------------------------------------------

TOOL_SCHEMAS = [
    {"type": "function", "function": {
        "name": "lookup",
        "description": lookup.__doc__ or "Exact data for one Pokemon, move, ability or item.",
        "parameters": LookupRequest.model_json_schema()}},
    {"type": "function", "function": {
        "name": "query_dex",
        "description": "Find Pokemon matching structural criteria (types, stats, learnset).",
        "parameters": QueryDexRequest.model_json_schema()}},
    {"type": "function", "function": {
        "name": "search_wiki",
        "description": "Search Bulbapedia and Smogon analyses for prose: anime, lore, "
                       "game locations, TCG, mechanics explanations.",
        "parameters": SearchWikiRequest.model_json_schema()}},
    {"type": "function", "function": {
        "name": "usage_stats",
        "description": "Smogon competitive usage statistics for a format.",
        "parameters": UsageStatsRequest.model_json_schema()}},
    {"type": "function", "function": {
        "name": "validate_team",
        "description": "Validate a Showdown-format team against a format's rules.",
        "parameters": ValidateTeamRequest.model_json_schema()}},
    {"type": "function", "function": {
        "name": "calc_damage",
        "description": "Calculate damage and KO chance for one attack.",
        "parameters": CalcDamageRequest.model_json_schema()}},
    {"type": "function", "function": {
        "name": "type_matchup",
        "description": "Type effectiveness for a given generation — attacking vs "
                       "defending, a full defensive profile, or by species name.",
        "parameters": TypeMatchupRequest.model_json_schema()}},
    {"type": "function", "function": {
        "name": "get_sets",
        "description": "Real named Smogon sets for a Pokemon. Use instead of "
                       "inventing a moveset.",
        "parameters": GetSetsRequest.model_json_schema()}},
    {"type": "function", "function": {
        "name": "common_generation",
        "description": "The highest generation that includes every one of 2+ named "
                       "Pokemon together. Call this FIRST, before anything else, "
                       "whenever building a team around specific named Pokemon — not "
                       "every Pokemon exists in every generation (Aegislash has no Gen "
                       "9 data, for example), and discovering that mid-build after a "
                       "tool call fails is worse than knowing upfront. Use the "
                       "returned gen for every other tool call in the rest of this "
                       "request: validate_team, review_team, get_sets, calc_damage, "
                       "type_matchup, lookup. Not needed for a single named Pokemon "
                       "or a team with no specific seed.",
        "parameters": CommonGenerationRequest.model_json_schema()}},
    {"type": "function", "function": {
        "name": "query_items",
        "description": "Browse or filter held items — a quick reference when building a "
                       "team, not just a single named lookup (use lookup for that). Filter "
                       "by is_choice, is_berry, is_mega_stone, or category (type_boost / "
                       "stat_boost — the only two families with a clean category; "
                       "everything else has no clean category field but a precise "
                       "short_desc, so use query for free-text search instead — search "
                       "SPECIFIC terms, not category words: 'Sunny Day' not 'weather', "
                       "'Electric Terrain' not 'terrain'; 'burn', 'Ground-type', 'hazards' "
                       "work directly. Combine filters with query for something like "
                       "'Fire-type boosting items' if category alone isn't enough.",
        "parameters": QueryItemsRequest.model_json_schema()}},
    {"type": "function", "function": {
        "name": "review_team",
        "description": "Structural review of one team: roles, weaknesses, speed "
                       "tiers, coverage gaps.",
        "parameters": ReviewTeamRequest.model_json_schema()}},
    {"type": "function", "function": {
        "name": "compare_teams",
        "description": "Structural head-to-head of two teams: speed order, shared "
                       "weaknesses, threat matrix. Returns no win probability.",
        "parameters": CompareTeamsRequest.model_json_schema()}},
]

SYSTEM_PROMPT = """You are a Pokémon expert with access to a Bulbapedia mirror, the Pokémon Showdown
data layer, and Smogon usage statistics.

SOURCING
- Never state a Pokémon fact you have not retrieved this turn. Even if you are
confident you know the answer, look it up. Your own memory of Pokémon data is
unreliable.
- If the tools return nothing relevant, say so plainly. Do not fill the gap from
memory.
- Cite what you used: article title with its link, or the format and month for
statistics.

PRECEDENCE WHEN SOURCES DISAGREE
- Legality, learnsets, base stats, type matchups: the dex tools win.
- Anime, manga, TCG, characters, lore, game locations: the wiki wins.
- Current metagame: usage statistics win, but they describe what people play, not
what is optimal.

TOOL SELECTION
- Building a team around 2+ SPECIFIC NAMED Pokémon: call common_generation FIRST,
before anything else. Not every Pokémon exists in every generation. Use the gen it
returns for every other tool call in the request — validate_team, review_team,
get_sets, calc_damage, type_matchup, lookup. Skipping this and defaulting to the
current generation is how a team crashes partway through on a Pokémon that
doesn't exist there, forcing a retry that then has to be untangled from the real
team downstream.
- lookup or query_dex for exact data. search_wiki for prose. usage_stats for the
metagame.
- type_matchup for ANY type effectiveness question.
- get_sets before recommending how to build a Pokémon. Quote a real named set
rather than inventing four moves.
- query_items when discussing item choice for a build — browse by is_choice/
is_berry/is_mega_stone/category, or free-text query for anything else (status
orbs, weather rocks, terrain seeds, gems, plates, general items). Don't guess at
what items exist in a category from memory.
- For ANY request to review, rate, critique, or find problems with a team: call
review_team, always, before saying anything about the team. It applies to partial
teams and cores, not just full six. Do not reconstruct team analysis from
individual lookups — you will get speed stats and role coverage wrong. If
validate_team parsed the team, review_team will parse it too.
- compare_teams for head-to-head matchup questions.
- validate_team on every team you propose, before showing it. Never show an
illegal team.
- calc_damage for any damage question. Never estimate.

IMAGES
- If the user wants to SEE a Pokemon — "show me X", "what does X look like", "picture
of X", comparing how two Pokemon look, or anything else visual rather than
factual — call lookup for that species. This is true even when no other part of
the question needs a tool call. For a short list of named Pokemon, call lookup
once per name — query_dex only supports filter-defined sets ("every Water-type
starter"), not an arbitrary named list, so it is the right call only when the set
itself is defined by a property rather than spelled out by name. The image itself
is attached to your reply automatically as a result of the call; you do not need
to embed it, link it, or mention a URL. A short acknowledgment in your reply is
enough — do not say you are unable to show images, since calling lookup is
exactly what makes that happen.
- If the user specifically asked for the SHINY appearance, pass shiny=true. This
makes the shiny version the actual image shown, not just a fact you mention
alongside the regular one. If the reply comes back with shiny_unavailable, say so
plainly rather than presenting the regular image as if it were what was asked for.

MECHANICAL CLAIMS MUST BE SOURCED
- When explaining WHY something is good or how it works, every mechanical claim
must come from a tool call this turn.
- If you describe a typing or its resistances, call type_matchup. Do not reason
about type interactions from memory, even when the answer seems obvious.
- Ability effects: quote or closely paraphrase the effect text lookup returned. Do
not describe an ability from memory. If lookup did not return the effect, call it
again rather than filling the gap.
- Quote the actual speed stats review_team returns, not base Speed.
- Analysis, framing and judgement are yours. Mechanical facts are not.

WHAT YOU CANNOT COMPUTE
- Never state a win probability or percentage for a matchup. There is no battle
simulator and no outcome data in this system, so any number would be invented.
Describe matchups qualitatively and say plainly that a numeric win rate is not
something you can produce.

SCOPE AND AMBIGUITY
- Mechanics change between generations. If a question is generation-sensitive and
the user didn't specify, assume the current generation and say which you assumed.
- Generations 1 and 2 had a single Special stat, not separate Special Attack and
Special Defense. Report it as Special for those games.
- Games, anime, manga and TCG often contradict each other. Say which you are
answering about.
- If a question can't be answered without more information, ask one short
clarifying question instead of guessing.

STYLE
- Answer like a knowledgeable friend, not a wiki. Lead with the answer, then the
detail.
- Say when you are uncertain or working from incomplete information.
"""

HANDLERS = {
    "lookup": lambda a: lookup(LookupRequest(**a)),
    "query_dex": lambda a: query_dex(QueryDexRequest(**a)),
    "search_wiki": lambda a: search_wiki(SearchWikiRequest(**a)),
    "usage_stats": lambda a: usage_stats(UsageStatsRequest(**a)),
    "type_matchup": lambda a: type_matchup(TypeMatchupRequest(**a)),
    "get_sets": lambda a: get_sets(GetSetsRequest(**a)),
    "common_generation": lambda a: common_generation(CommonGenerationRequest(**a)),
    "query_items": lambda a: query_items(QueryItemsRequest(**a)),
}
ASYNC_HANDLERS = {
    "validate_team": lambda a: validate_team(ValidateTeamRequest(**a)),
    "calc_damage": lambda a: calc_damage(CalcDamageRequest(**a)),
    "compare_teams": lambda a: compare_teams(CompareTeamsRequest(**a)),
    "review_team": lambda a: review_team(ReviewTeamRequest(**a)),
}


async def _run_tool(name: str, args: dict) -> dict:
    try:
        if name in ASYNC_HANDLERS:
            return await ASYNC_HANDLERS[name](args)
        if name in HANDLERS:
            return await asyncio.to_thread(HANDLERS[name], args)
        return {"error": f"unknown tool {name}"}
    except Exception as e:
        return {"error": str(e)}


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@app.post("/models_proxy", operation_id="models_proxy",
          summary="List models from an arbitrary OpenAI-compatible endpoint",
          include_in_schema=False)
async def models_proxy(req: ModelsProxyRequest):
    """
    Lets the standalone page populate its model dropdown from whatever
    base_url the user types in, without the browser needing to make a
    cross-origin request to an arbitrary host itself (which would need CORS
    the target server likely doesn't have configured for this).
    """
    base_url = req.base_url.rstrip("/")
    headers = {}
    if req.api_key:
        headers["Authorization"] = f"Bearer {req.api_key}"
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            r = await client.get(f"{base_url}/models", headers=headers)
            r.raise_for_status()
            data = r.json()
            ids = [m.get("id") for m in data.get("data", []) if m.get("id")]
            return {"models": ids}
    except Exception as e:
        raise HTTPException(502, f"Could not list models from {base_url}: {e}")


@app.post("/chat", operation_id="chat", summary="Chat orchestrator for the standalone page",
          include_in_schema=False)
async def chat(req: ChatRequest):
    """
    Runs the tool loop server-side and streams the result as SSE.

    Open WebUI runs its own loop and does not use this — it is for the bundled
    HTML page, so the system prompt and tool policy live in one place.
    """
    base_url = (req.base_url or OPENAI_BASE_URL).rstrip("/")
    model = req.model or OPENAI_MODEL
    api_key = req.api_key if req.api_key is not None else OPENAI_API_KEY

    if not base_url or not model:
        raise HTTPException(
            500,
            "No model configured. Either set OPENAI_BASE_URL/OPENAI_MODEL on "
            "this container, or pass base_url/model in the request (the "
            "standalone page's settings panel does this automatically once "
            "a model is picked there)."
        )

    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + req.messages
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    async def gen():
        sources: list[dict] = []
        # Collected the same way as sources: pulled structurally from tool
        # results as they stream past, not left to the model to remember to
        # embed as markdown. A model correctly emitting image syntax in prose
        # is not something this system has found reliable for anything else
        # it needs surfaced (see: the calc_damage assumptions block, ignored
        # until it was made a first-class structural response field rather
        # than left for the model to mention on its own). Deduped by species
        # name so a multi-Pokemon question doesn't repeat the same thumbnail
        # once per tool call that happened to touch it.
        images: dict[str, dict] = {}
        # Whichever validate_team/review_team/compare_teams call fired MOST
        # RECENTLY defines "the current team" — replaced (not accumulated)
        # each time one fires, same as build data already works, so checking
        # three variants across a conversation always leaves the last one as
        # the real team. Anything mentioned in the final text but NOT in this
        # set (a threat example, an alternative swap-in suggestion) still
        # gets a card, just a visually distinct one — see emit_final.
        primary_team_names: set[str] = set()

        def collect_images(fn: str, result: dict):
            if fn == "lookup" and result.get("kind") == "species" and result.get("image_url"):
                # Merge into whatever's already there, never replace outright.
                # A second lookup() call for the same species later in a long
                # analysis — exactly the kind of re-verification this
                # system's own sourcing discipline encourages — was silently
                # wiping out item/ability/moves that an earlier get_sets or
                # review_team call had already attached, since the old code
                # built a brand new dict from scratch every time and assigned
                # over whatever was there. Confirmed as the real explanation
                # for why specifically the most-scrutinized Pokemon in a team
                # build (re-checked repeatedly) lost their build data while
                # ones only touched once did not.
                name = result["name"]
                types = result.get("types", [])
                entry = dict(images.get(name) or {})
                entry["url"] = result["image_url"]
                entry["types"] = types
                entry["base_stats"] = result.get("base_stats")
                entry["base_species"] = result.get("base_species", entry.get("base_species"))
                entry["battle_only"] = result.get("battle_only", entry.get("battle_only"))
                entry["required_item"] = result.get("required_item", entry.get("required_item"))
                try:
                    with db() as c:
                        entry.update(defensive_profile(c, result.get("gen", DEFAULT_GEN), types))
                except Exception:
                    pass
                images[name] = entry
            elif fn == "query_dex":
                gen = result.get("gen", DEFAULT_GEN)
                results_list = result.get("results") or []
                try:
                    with db() as c:
                        for x in results_list:
                            if x.get("image_url") and x.get("name"):
                                types = x.get("types", [])
                                entry = dict(images.get(x["name"]) or {})
                                entry["url"] = x["image_url"]
                                entry["types"] = types
                                entry["base_stats"] = x.get("base_stats")
                                entry["base_species"] = x.get("base_species", entry.get("base_species"))
                                entry["battle_only"] = x.get("battle_only", entry.get("battle_only"))
                                entry["required_item"] = x.get("required_item", entry.get("required_item"))
                                entry.update(defensive_profile(c, gen, types))
                                images[x["name"]] = entry
                except Exception:
                    for x in results_list:
                        if x.get("image_url") and x.get("name"):
                            entry = dict(images.get(x["name"]) or {})
                            entry["url"] = x["image_url"]
                            entry["types"] = x.get("types", [])
                            entry["base_stats"] = x.get("base_stats")
                            entry["base_species"] = x.get("base_species", entry.get("base_species"))
                            entry["battle_only"] = x.get("battle_only", entry.get("battle_only"))
                            entry["required_item"] = x.get("required_item", entry.get("required_item"))
                            images[x["name"]] = entry
            elif fn in ("validate_team", "review_team", "compare_teams"):
                # These come from pokedex-sim and never carry an image — only
                # names. Resolving them here means every team tool call shows
                # the whole team's cards, not just whichever member happened
                # to also get an individual lookup call.
                names: list[str] = []
                # review_team/compare_teams member objects already carry the
                # ACTUAL configured item/ability/moves for that team — no
                # ambiguity about "which set" the way get_sets has, since a
                # real team has exactly one build per slot.
                builds: dict[str, dict] = {}
                if fn == "validate_team":
                    names = result.get("species") or []
                elif fn == "review_team":
                    for m in (result.get("members") or []):
                        if m.get("name"):
                            names.append(m["name"])
                            builds[m["name"]] = {"item": m.get("item"), "ability": m.get("ability"),
                                                  "moves": m.get("moves") or []}
                elif fn == "compare_teams":
                    for team in (result.get("teams") or {}).values():
                        for m in (team.get("members") or []):
                            if m.get("name"):
                                names.append(m["name"])
                                builds[m["name"]] = {"item": m.get("item"), "ability": m.get("ability"),
                                                      "moves": m.get("moves") or []}
                gen = result.get("gen", DEFAULT_GEN)
                if names:
                    new_set = set(names)
                    # A call whose members are entirely already-known (a
                    # STRICT SUBSET of the current tracked team) is a
                    # narrowing, not a new team — confirmed as the real cause
                    # of a real bug: after review_team crashed trying to
                    # process all 6 members (Aegislash has no Gen 9 data, and
                    # the review engine only loads Gen 9), the model retried
                    # scoped to just the 5 that don't crash it. Replacing the
                    # tracked team with that 5-member retry silently dropped
                    # a real team member who was never actually cut from the
                    # team, only from ONE tool call that couldn't process it.
                    # Only replace when the new call introduces a name that
                    # wasn't already tracked — that's the actual signal of a
                    # genuinely different team/variant being checked, not a
                    # same-team retry with reduced scope.
                    if not primary_team_names or not new_set.issubset(primary_team_names):
                        primary_team_names.clear()
                        primary_team_names.update(names)
                # Only the IMAGE lookup should skip already-known names — an
                # earlier individual lookup() call (very likely for whichever
                # Pokemon the question was actually about) already resolved
                # its url/types/base_stats. But that same name's build data
                # (item/ability/moves) must still be attached: it was never
                # skipped from `builds` above, so skipping it here too, just
                # because the image happened to already exist, silently
                # dropped exactly the Pokemon most likely to be the one asked
                # about — confirmed as the actual bug, not a hypothesis.
                new_names = [n for n in names if n not in images]
                try:
                    with db() as c:
                        if new_names:
                            for name, info in resolve_images_by_name(c, new_names).items():
                                images[name] = info
                        for name in names:
                            b = builds.get(name)
                            if b and name in images:
                                images[name]["item"] = b["item"]
                                images[name]["ability"] = b["ability"]
                                images[name]["moves"] = resolve_moves(c, gen, b["moves"])
                except Exception:
                    pass  # image resolution is a nice-to-have, never worth failing the tool call over
            elif fn == "get_sets" and result.get("available") and result.get("sets"):
                # get_sets can return several sets for one species — it
                # already orders dex-origin (curated) sets first, so sets[0]
                # is its own implicit "best" pick, used here rather than
                # trying to show every set on one card.
                #
                # Same bug class as review_team/compare_teams had two fixes
                # ago, just never applied here: "not in images" skipped the
                # WHOLE block, including build-data attachment, whenever an
                # earlier lookup/validate_team call had already given this
                # species a bare entry — which is the normal, expected order
                # for any real team-build conversation. Confirmed directly:
                # Aegislash's real Weakness Policy/Stance Change/moveset from
                # get_sets never made it onto its card for exactly this
                # reason. Now the image is only fetched fresh when missing;
                # the build data attaches every time regardless.
                species_name = result.get("species")
                best = result["sets"][0]
                if species_name:
                    fmt = best.get("format") or ""
                    m = re.match(r"gen(\d+)", fmt)
                    gen = int(m.group(1)) if m else DEFAULT_GEN
                    try:
                        with db() as c:
                            if species_name not in images:
                                resolved = resolve_images_by_name(c, [species_name])
                                if species_name in resolved:
                                    images[species_name] = resolved[species_name]
                            if species_name in images:
                                images[species_name]["item"] = best.get("item")
                                images[species_name]["ability"] = best.get("ability")
                                images[species_name]["nature"] = best.get("nature")
                                images[species_name]["moves"] = resolve_moves(c, gen, best.get("moves") or [])
                    except Exception:
                        pass

        # Shared by both the normal "model gave a final answer" path and the
        # forced-final-answer path below, so a future fix to one doesn't
        # silently miss the other.
        async def emit_final(content, turns):
            for i in range(0, len(content), 24):
                yield _sse("token", {"t": content[i:i + 24]})
                await asyncio.sleep(0)
            # images accumulates EVERYTHING touched across the whole
            # tool-calling conversation — every query_dex candidate
            # considered and rejected, every get_sets call for a set that
            # didn't end up chosen, every team variant validate_team checked
            # along the way. For a simple one-Pokemon question that's rarely
            # more than 1-2 entries and doesn't matter; for a real team-build
            # with actual exploration it becomes a dumping ground. Confirmed
            # directly: a real team-build request produced a pile of
            # unrelated cards alongside the six actually recommended.
            # Filtering to species named in the FINAL answer text is a
            # simple, robust fix regardless of which tool call introduced
            # the stray entry — a team recommendation always names every
            # member it's recommending, by construction.
            content_lower = content.lower()
            relevant = {k: v for k, v in images.items() if k.lower() in content_lower}

            # Name-mention alone doesn't distinguish "this is on the team"
            # from "this was named as a threat example" or "this was offered
            # as an alternative swap-in" — confirmed directly: a real
            # team-build answer that discussed a Landorus-Therian swap-in and
            # used Weavile purely as an illustrative threat example gave both
            # of them full cards indistinguishable from the actual six-member
            # team. primary_team_names is the actual, unambiguous team
            # membership — taken directly from the most recent validate_team/
            # review_team/compare_teams call's own parsed member list, not
            # inferred from prose at all. If no team tool was ever called
            # this turn (the ordinary single-Pokemon case), the set stays
            # empty and everything is treated as primary — unchanged from
            # before for the common case.
            if primary_team_names:
                primary_norm = {norm(n) for n in primary_team_names}
                primary, battle_forms, secondary = {}, {}, {}
                for k, v in relevant.items():
                    if k in primary_team_names:
                        primary[k] = v
                        continue
                    # Aegislash-Blade isn't an alternative Pokemon the way
                    # Mega Charizard X or a query_dex candidate is — it's the
                    # SAME team member, automatically, mid-battle. Confirmed
                    # directly against live @pkmn/dex data: required_item is
                    # always set for Mega-style forms (a player's deliberate
                    # choice, holding a specific stone) and always absent for
                    # ability/move-triggered ones like Aegislash's Stance
                    # Change or Mimikyu's Disguise — that's the actual signal,
                    # not battle_only alone, which is true for BOTH categories
                    # (neither is independently selectable in a team builder).
                    base = v.get("base_species")
                    if (base and norm(base) in primary_norm
                            and v.get("battle_only") and not v.get("required_item")):
                        battle_forms[k] = v
                    else:
                        secondary[k] = v
            else:
                primary, battle_forms, secondary = relevant, {}, {}

            def _card(k, v):
                return {"name": k, "url": v["url"], "types": v.get("types", []),
                        "base_stats": v.get("base_stats"),
                        "weak": v.get("weak", []), "resist": v.get("resist", []),
                        "immune": v.get("immune", []),
                        "item": v.get("item"), "ability": v.get("ability"),
                        "nature": v.get("nature"), "moves": v.get("moves")}

            yield _sse("done", {
                "sources": sources, "turns": turns,
                "images": [_card(k, v) for k, v in primary.items()],
                "battle_forms": [_card(k, v) for k, v in battle_forms.items()],
                "mentioned": [_card(k, v) for k, v in secondary.items()],
            })

        def extract_content(msg, finish_reason):
            content = msg.get("content") or ""
            if not content and finish_reason != "length":
                content = (msg.get("reasoning") or "").strip()
            return content

        try:
            async with httpx.AsyncClient(timeout=300.0) as client:
                # Raised from 8 — a two-seed team build (this session's real
                # case: Scizor AND Gengar, each needing its own lookup, sets,
                # and type-synergy checks against several candidate
                # teammates) is legitimate multi-step work that can outgrow a
                # tight cap even with the pills deduped down to a handful of
                # distinct tool NAMES in the UI — the real turn count behind
                # them can be much higher.
                MAX_TURNS = 12
                for turn in range(MAX_TURNS):
                    r = await client.post(
                        f"{base_url}/chat/completions",
                        headers=headers,
                        # 2048, then 4096, then 8192 — three increases, and a
                        # real two-Pokemon team-build request (a genuinely
                        # demanding task class: full reasoning, published-set
                        # citations, legality checks across 6 slots, a
                        # structural review) still got cut off at 8192.
                        # Confirmed that reasoning_tokens counts against the
                        # same completion_tokens/max_tokens budget as the
                        # actual answer, so a thorough answer to a hard
                        # question can genuinely need a lot of room on both
                        # sides of that budget. Jumping to 16384 rather than
                        # nudging again — the model's context window is
                        # 262,144 tokens (confirmed via /v1/models), so this
                        # costs nothing in context risk, and repeatedly
                        # under-provisioning the same task class three times
                        # running was the actual mistake, not too-small an
                        # increase each time.
                        json={"model": model, "messages": messages,
                              "tools": TOOL_SCHEMAS, "max_tokens": 16384},
                    )
                    if r.status_code != 200:
                        yield _sse("error", {"message": f"model returned {r.status_code}: {r.text[:300]}"})
                        return

                    choice = r.json()["choices"][0]
                    msg = choice["message"]
                    finish_reason = choice.get("finish_reason")
                    calls = msg.get("tool_calls") or []

                    if not calls:
                        content = msg.get("content") or ""
                        if not content:
                            if finish_reason == "length":
                                yield _sse("error", {
                                    "message": "The response was cut off before it finished — this "
                                               "usually happens on a complex, many-step request. Try "
                                               "asking again, or breaking it into smaller questions."
                                })
                                return
                            content = (msg.get("reasoning") or "").strip()
                        async for chunk in emit_final(content, turn + 1):
                            yield chunk
                        return

                    messages.append(msg)
                    for call in calls:
                        fn = call["function"]["name"]
                        try:
                            args = json.loads(call["function"].get("arguments") or "{}")
                        except json.JSONDecodeError:
                            args = {}
                        yield _sse("status", {"tool": fn, "args": args})

                        result = await _run_tool(fn, args)
                        if fn == "search_wiki":
                            for x in (result.get("results") or []):
                                if x.get("url"):
                                    sources.append({"title": x["title"], "url": x["url"]})
                        collect_images(fn, result)

                        messages.append({
                            "role": "tool",
                            "tool_call_id": call.get("id", ""),
                            "name": fn,
                            "content": json.dumps(result)[:12000],
                        })

                # The loop ran out of turns while the model was still calling
                # tools — rather than discard everything it already found
                # (multiple real tool calls, each one an actual answer paid
                # for), force one more completion with NO tools available, so
                # it has to synthesize a real answer from what's already in
                # the conversation instead of continuing to investigate.
                #
                # Removing tools alone wasn't enough in practice — confirmed
                # directly: without being told what this turn is FOR, the
                # model produced "Legal. Now review_team for the final
                # structure." as its entire answer, which reads exactly like
                # a leftover planning note ("that's step 1 done, step 2 was
                # going to be...") rather than an actual synthesis. It had
                # already called review_team by that point — the real data
                # was sitting right there in the conversation — it just never
                # used it, because nothing told it this was its last chance
                # to and that narrating next steps was no longer useful.
                messages.append({
                    "role": "user",
                    "content": (
                        "You have used all available tool calls for this request. No more "
                        "tools are available. Do not call a tool, and do not describe what "
                        "you were about to do next — that information is no longer useful. "
                        "Using ONLY what has already been gathered above in this "
                        "conversation, write your complete final answer now."
                    ),
                })
                yield _sse("status", {"tool": "_finalize", "label": "Summarizing findings so far"})
                r = await client.post(
                    f"{base_url}/chat/completions",
                    headers=headers,
                    json={"model": model, "messages": messages, "max_tokens": 16384},
                )
                if r.status_code != 200:
                    yield _sse("error", {"message": "Tool loop hit its limit, and the follow-up "
                                                     f"summary also failed ({r.status_code})."})
                    return
                choice = r.json()["choices"][0]
                content = extract_content(choice["message"], choice.get("finish_reason"))
                if not content:
                    yield _sse("error", {
                        "message": "This question needed more steps than could be completed. "
                                   "Try narrowing it — e.g. one Pokemon at a time, or fewer "
                                   "constraints at once."
                    })
                    return
                async for chunk in emit_final(content, MAX_TURNS + 1):
                    yield chunk

        except httpx.RequestError as e:
            yield _sse("error", {"message": f"Cannot reach the model at {base_url}: {e}"})

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
