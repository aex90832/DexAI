# DexAI — API Reference

Every endpoint `pokedex-api` exposes, with request fields and real response
shapes. For deployment instructions, see `pokedex-ai-README.md`. This document
covers the interface only: what to send, what comes back, and what each field
means.

## Base URL and conventions

All endpoints below are on `pokedex-api`, default port `8990`. All tool
endpoints are `POST` with a JSON body, `Content-Type: application/json`, and
return JSON. `/health` is `GET`.

```
http://<HOST_IP>:8990/lookup
http://<HOST_IP>:8990/query_dex
http://<HOST_IP>:8990/health          (GET)
...
```

**Generation defaults.** Every endpoint that takes a `gen` field defaults to
the server's `DEFAULT_GEN` (9, unless configured otherwise) when omitted.
Passing an explicit `gen` is always safer than relying on the default,
especially for anything involving a Pokémon that might not exist in the
current generation — see `common_generation` below.

**Species name resolution.** Every endpoint that takes a species/move/
ability/item name runs it through the same alias resolver — common
misspellings, nicknames matched from the wiki's own redirects, and exact
Showdown IDs all work. A name that doesn't resolve returns a `found: false` /
`available: false` shape with a `hint`, never a 404 or an exception.

**Four response families, not one.** Endpoints don't share a single schema —
each returns whatever shape fits what it answers. What they do share: a
boolean indicating whether the query resolved to anything (`found`,
`available`, or implicit in a non-empty `results`/`count`), and a `hint`
string when it didn't, explaining why and what to try instead. Check that
boolean before reading anything else out of the response.

---

## `POST /health`

Service and data status. No request body.

```json
{
  "status": "ok",
  "uptime_s": 4213,
  "counts": {
    "species": 1025,
    "species_and_forms_all_generations": 1292,
    "moves": 919,
    "abilities": 310,
    "items": 490,
    "learnsets": 1696741,
    "wiki_chunks": 734541,
    "wiki_chunks_linked": 221697,
    "aliases": 233158,
    "species_images": 5479,
    "usage_rows": 7462,
    "sets": 23427
  },
  "generations_loaded": [1, 2, 3, 4, 5, 6, 7, 8, 9],
  "data_dates": {
    "dex": "2026-09-21T00:39:24Z",
    "wiki": "2026-08-30T02:11:00Z",
    "stats_latest_month": "2026-08",
    "stats_months_available": ["2026-06", "2026-07", "2026-08"]
  },
  "embeddings": {
    "loaded": true,
    "count": 734541,
    "model": "BAAI/bge-small-en-v1.5"
  }
}
```

`counts.species` is the real National Dex total — distinct national dex
*number*, not distinct database row. `species_and_forms_all_generations` is
the larger, different number: every battle-usable form counted separately
(base + every Mega, regional, and other forme), across every loaded
generation. Neither is wrong; they answer different questions — "how many
Pokémon" versus "how many distinct forms this database knows."

If the database hasn't been built yet, `status` is `"no data"` and a `hint`
points at running `ingest.py`.

---

## `POST /lookup`

Exact data for one named Pokémon, move, ability, or item. The single most
important endpoint — call this before answering any factual question about a
specific named thing.

**Request**

| Field | Type | Required | Notes |
|---|---|---|---|
| `name` | string | Yes | Any common spelling or nickname |
| `gen` | int | No | 1–9, defaults to current |
| `moves_to_check` | list[string] | No | Answers "can X learn Y" — see below |
| `include_learnset` | bool | No | Full level-up learnset, ordered by level |
| `shiny` | bool | No | Makes the shiny sprite the one returned in `image_url` |

**Response shape depends on what `name` resolved to** (`kind` tells you
which): `species`, `move`, `ability`, `item`, or `article` (a wiki-only
subject with no dex data — the response just points you at `search_wiki`
instead). Not found at all returns `{"found": false, "query": "...", "hint": "..."}`.

### Species

```json
{
  "found": true,
  "kind": "species",
  "id": "kingambit",
  "name": "Kingambit",
  "national_dex": 983,
  "gen": 9,
  "types": ["Dark", "Steel"],
  "base_stats": {"hp": 100, "atk": 135, "def": 120, "spa": 60, "spd": 85, "spe": 50},
  "bst": 550,
  "abilities": ["Defiant", "Supreme Overlord", "Pressure"],
  "evolution": {
    "prevo": "bisharp",
    "evos": [],
    "level": null,
    "item": null,
    "condition": "Ally is defeated",
    "method": null,
    "evolves_from": "Bisharp evolves into Kingambit (Ally is defeated)",
    "evolves_into": "Kingambit is fully evolved."
  },
  "egg_groups": ["Mineral", "Humanshape"],
  "weight_kg": 120.0,
  "height_m": 2.0,
  "tier": "OU",
  "doubles_tier": "DOU",
  "fully_evolved": true,
  "base_species": null,
  "battle_only": false,
  "required_item": null,
  "image_url": "http://192.168.1.50:8990/sprites/previews/poke_capture_0983_000_mf_n_00000000_f_n.png",
  "sprite_icon_url": "http://.../sprites/icons/poke_icon_0983_000_mf_n_00000000_f_n.png",
  "sprite_preview_url": "http://.../sprites/previews/poke_capture_0983_000_mf_n_00000000_f_n.png",
  "sprite_shiny_icon_url": "http://.../sprites/icons/poke_icon_0983_000_mf_n_00000000_f_r.png",
  "sprite_shiny_preview_url": "http://.../sprites/previews/poke_capture_0983_000_mf_n_00000000_f_r.png",
  "showing_shiny": false,
  "moves_known": 68
}
```

**Alternate-form fields** (`base_species`, `battle_only`, `required_item`) —
present on every species, `null`/`false` for an ordinary one:

- `base_species` — set when this entry is a form of something else (a Mega,
  regional variant, or automatic battle form). `null` for a base species.
- `battle_only` — `true` when this exact form can't be selected as its own
  slot in a team builder, only appears once a battle starts. **True for both
  Mega Evolutions and automatic transformations like Aegislash-Blade** — this
  field alone doesn't distinguish "a player's deliberate choice" from "an
  automatic in-battle state change."
- `required_item` — the actual distinguishing field. A real item name
  (`"Charizardite X"`) for anything Mega-Evolution-style — a player's
  deliberate choice, holding that item. `null` for anything ability/move
  triggered instead (Aegislash's Stance Change, Mimikyu's Disguise, Meloetta's
  Relic Song). **A card representing an alternate form is "the same team
  member, different battle state" when `battle_only` is true AND
  `required_item` is null — and "a genuinely different, optional Pokémon"
  when both are true.**

**Alternate forms list.** When the species has known Mega/regional/Gigantamax/
cosmetic variants cataloged, an `alternate_forms` array is included:

```json
"alternate_forms": [
  {
    "forme_name": "Charizard-Mega-X",
    "is_gmax": false,
    "is_shiny": false,
    "icon_url": "http://.../poke_icon_0006_001_mf_n_00000000_f_n.png",
    "preview_url": "http://.../poke_capture_0006_001_mf_n_00000000_f_n.png"
  }
]
```

`forme_name` is `null` when a real image exists but its specific form wasn't
confidently matched against the dex data — still a real image, just not
labeled with which alternate form it specifically is.

**Shiny requests.** Pass `shiny: true` to make the shiny sprite the one
`image_url`/`sprite_icon_url`/`sprite_preview_url` actually point at, rather
than just being mentioned in the separate `sprite_shiny_*` fields. If no
shiny sprite is cataloged for this species, the response includes
`"shiny_unavailable": true` and `image_url` stays the regular sprite —
check for this key before assuming the shiny request succeeded.

**Generation 1–2 stats.** These generations had a single Special stat, not
separate Special Attack/Defense. For `gen <= 2`, `base_stats` is shaped
differently: `{"hp", "atk", "def", "spc", "spe"}` (five keys, not six — `spc`
replaces `spa`/`spd`), and a `stat_note` field explains why.

**`moves_to_check`** — pass move names to check learnability instead of
guessing:

```json
"learnset_check": {
  "Sucker Punch": {
    "learns": true,
    "move_id": "suckerpunch",
    "level_up_at": 1,
    "how": [{"method": "level-up", "from_gen": 9, "level": 1}]
  },
  "Fire Blast": {
    "learns": false,
    "move_id": "fireblast"
  }
}
```

**`include_learnset`** — the full level-up learnset, ordered by level:

```json
"level_up_learnset": [
  {"level": 1, "move": "Iron Head", "type": "Steel", "category": "Physical",
   "power": 80, "accuracy": 100, "effect": "30% chance to flinch."}
],
"other_learn_methods": {"tm": 42, "egg": 3, "tutor": 8}
```

### Move

```json
{
  "found": true, "kind": "move", "id": "suckerpunch", "name": "Sucker Punch",
  "gen": 9, "type": "Dark", "category": "Physical",
  "base_power": 70, "accuracy": 100, "pp": 5, "priority": 1, "target": "normal",
  "flags": {"contact": 1, "protect": 1, "mirror": 1},
  "effect": "Fails if the target is not attacking this turn.",
  "detail": "..."
}
```

### Ability / Item

Same shape for both (`kind` is `"ability"` or `"item"`):

```json
{
  "found": true, "kind": "ability", "id": "supremeoverlord", "name": "Supreme Overlord",
  "gen": 9, "effect": "This Pokemon's Attack and Special Attack are boosted...",
  "detail": "..."
}
```

---

## `POST /query_dex`

Filtered search — "which Pokémon are X and also Y." Never use this for one
named Pokémon; that's `lookup`.

**Request**

| Field | Type | Notes |
|---|---|---|
| `types` | list[string] | e.g. `["Water", "Ground"]` |
| `type_mode` | `"all"` \| `"any"` | Default `"all"` |
| `learns` | list[string] | Must legally learn every move listed |
| `ability` | string | |
| `egg_group` | string | |
| `tier` | string | e.g. `"OU"` |
| `min_bst` / `max_bst` | int | |
| `stat_filters` | dict | Per-stat bounds — see below |
| `nfe` | bool | `true` = unevolved only |
| `gen` | int | |
| `order_by` | string | `bst`\|`spe`\|`atk`\|`spa`\|`hp`\|`def`\|`spd`\|`num` |
| `limit` | int | 1–100, default 25 |

`stat_filters` example: `{"spe": {"min": 100}, "hp": {"max": 80}}` — keys are
`hp`, `atk`, `def`, `spa`, `spd`, `spe`.

**Response**

```json
{
  "count": 47,
  "returned": 25,
  "gen": 9,
  "results": [
    {
      "id": "kingambit", "name": "Kingambit", "num": 983,
      "types": ["Dark", "Steel"],
      "base_stats": {"hp": 100, "atk": 135, "def": 120, "spa": 60, "spd": 85, "spe": 50},
      "bst": 550, "abilities": ["Defiant", "Supreme Overlord", "Pressure"],
      "tier": "OU", "fully_evolved": true,
      "base_species": null, "battle_only": false, "required_item": null, "forme": null,
      "image_url": "http://.../poke_capture_0983_000_mf_n_00000000_f_n.png"
    }
  ],
  "note": null
}
```

`count` is the total matches; `returned` is how many came back under
`limit`. Same `base_species`/`battle_only`/`required_item` trio as `/lookup`,
same meaning. This row shape is shared verbatim with `dex_index` below —
same fields, same semantics, one function builds both.

---

## `GET /dex_index`

Every species and known form in one response — the complete list, not a
filtered search. Built for a UI's full Pokédex view to load in a single
request instead of paging through `query_dex` a few hundred rows at a time.
Not a chat tool: `GET`, no request body, and deliberately outside
`TOOL_SCHEMAS` — nothing for the model to call here, this is a UI data feed.

**Request:** none.

**Response**

```json
{
  "count": 1312,
  "results": [
    {
      "id": "aegislash", "name": "Aegislash", "num": 681,
      "types": ["Steel", "Ghost"],
      "base_stats": {"hp": 60, "atk": 50, "def": 150, "spa": 50, "spd": 150, "spe": 60},
      "bst": 520, "abilities": ["Stance Change"],
      "tier": "OU", "fully_evolved": true,
      "base_species": null, "battle_only": false, "required_item": null, "forme": null,
      "image_url": "http://.../poke_capture_0681_000_mf_n_00000000_f_n.png"
    },
    {
      "id": "aegislashblade", "name": "Aegislash-Blade", "num": 681,
      "types": ["Steel", "Ghost"],
      "base_stats": {"hp": 60, "atk": 150, "def": 50, "spa": 150, "spd": 50, "spe": 60},
      "bst": 520, "abilities": ["Stance Change"],
      "tier": "OU", "fully_evolved": true,
      "base_species": "Aegislash", "battle_only": true, "required_item": null, "forme": "Blade",
      "image_url": "http://.../poke_capture_0681_001_mf_n_00000000_f_n.png"
    }
  ]
}
```

One row per species **id** — not per national dex number — so a species and
every one of its known forms (Mega, regional, Gigantamax, automatic battle
forms like Aegislash-Blade) each get their own row. Each row comes from that
specific id's newest generation, independently: Aegislash has no Gen 9 data
at all, so its row (and Aegislash-Blade's) comes from Gen 8, while an
ordinary species with full Gen 9 data comes from there instead. This means
`dex_index` is not filtered to the current generation the way most other
endpoints default — it deliberately mixes generations, one per id, to
include everything.

Row shape is identical to `query_dex`'s `results` rows (same function builds
both) — see `base_species`/`battle_only`/`required_item`'s meaning under
`/lookup` above.

**Cached in memory, invalidated by the database file's own modification
time** — cheap to call repeatedly; a re-ingest is picked up automatically on
the next request with no restart needed. The cache is per-process: a
deployment running multiple `pokedex-api` workers would rebuild it
independently in each one, since nothing shares it between them.

---

## `POST /search_wiki`

Hybrid lexical + semantic search over the Bulbapedia snapshot. Prose, not
structured data — anime, manga, TCG, lore, locations, mechanics explained in
words.

**Request**

| Field | Type | Notes |
|---|---|---|
| `query` | string | Required, natural language |
| `entity` | string | Restrict to one Pokémon/move/item's articles |
| `domain` | string | `games`\|`anime`\|`manga`\|`tcg`\|`competitive`\|`general` |
| `gen` | int | |
| `limit` | int | 1–15, default 6 |

**Response**

```json
{
  "count": 6,
  "retrieval": "hybrid",
  "data_date": "2026-08-30T02:11:00Z",
  "results": [
    {
      "title": "Kingambit (Pokémon)",
      "section": "Biology",
      "domain": "games",
      "gen": 9,
      "source": "bulbapedia",
      "url": "http://192.168.1.50:30236/content/bulbagarden_en_all_maxi_.../Kingambit_(Pok%C3%A9mon)",
      "text": "Kingambit is a bipedal, humanoid Pokémon..."
    }
  ]
}
```

`retrieval` reports which strategies contributed: `"hybrid"` (both lexical and
semantic agreed on candidates), `"lexical"`, or `"semantic"` alone. An empty
result set returns `{"count": 0, "results": [], "hint": "..."}` rather than an
error.

---

## `POST /usage_stats`

Smogon ladder usage. Descriptive (what people play), not prescriptive (what's
optimal) — say so when it matters.

**Request**

| Field | Type | Notes |
|---|---|---|
| `format` | string | e.g. `"gen9ou"`, defaults to the server's default format |
| `species` | string | Omit for the overall usage ranking instead of one Pokémon's detail |
| `month` | string | `YYYY-MM`, omit for most recent available |
| `limit` | int | 1–50, default 20 (only applies to the ranking form) |

**With `species`** — full detail for one Pokémon:

```json
{
  "available": true, "format": "gen9ou", "month": "2026-08", "cutoff": 1630,
  "name": "Kingambit", "usage_pct": 14.32, "raw_count": 48213,
  "moves": {"Swords Dance": 61.2, "Sucker Punch": 58.9, "Kowtow Cleave": 71.4},
  "items": {"Leftovers": 34.1, "Black Glasses": 22.7},
  "abilities": {"Supreme Overlord": 98.2, "Defiant": 1.8},
  "spreads": {"Adamant:252/0/4/0/0/252": 18.3},
  "teammates": {"Great Tusk": 24.1, "Gholdengo": 19.8},
  "checks_and_counters": {"Great Tusk": 0.71, "Zamazenta": 0.68},
  "note": "Percentages are shares within that category, not usage rates. ..."
}
```

**Without `species`** — the ranking:

```json
{
  "available": true, "format": "gen9ou", "month": "2026-08",
  "ranking": [
    {"rank": 1, "name": "Great Tusk", "usage_pct": 28.4, "raw_count": 95201}
  ],
  "months_available": ["2026-06", "2026-07", "2026-08"]
}
```

Not found returns `available: false` with either `top_in_format` (species
lookup — a hint at what *is* used in that format) or `formats_available`
(format lookup — every format that has data at all).

---

## `POST /type_matchup`

Type effectiveness. Three distinct modes depending on which fields you pass —
the response shape differs by mode, not just the content.

**Request**

| Field | Type | Notes |
|---|---|---|
| `attacking_type` | string | e.g. `"Fighting"` |
| `defending_types` | list[string] | e.g. `["Dark", "Steel"]` |
| `species` | string | Alternative to `defending_types` — looked up by name |
| `gen` | int | The chart changed across generations (Fairy didn't exist before gen 6) |

**Mode 1 — attacking type vs. a specific defending combination** (pass both
`attacking_type` and `defending_types`, or `species`):

```json
{
  "found": true, "gen": 9, "attacking_type": "Fighting",
  "defending_types": ["Dark", "Steel"], "species": null,
  "multiplier": 2.0, "label": "super effective",
  "breakdown": [
    {"defending_type": "Dark", "multiplier": 2.0},
    {"defending_type": "Steel", "multiplier": 1.0}
  ]
}
```

**Mode 2 — full defensive profile** (pass only `defending_types` or `species`,
no `attacking_type`):

```json
{
  "found": true, "gen": 9, "species": "Kingambit",
  "defending_types": ["Dark", "Steel"],
  "immune": ["Psychic"],
  "quad_resists": [],
  "resists": ["Normal", "Flying", "Rock", "Bug", "Steel", "Fairy", "Ghost"],
  "neutral": ["Water", "Electric", "Grass", "Ice", "Dragon", "Dark"],
  "weak": ["Fighting", "Ground", "Fire"],
  "quad_weak": []
}
```

**Mode 3 — attacking type alone** (what it hits and what walls it):

```json
{
  "found": true, "gen": 9, "attacking_type": "Fighting",
  "super_effective_against": ["Normal", "Rock", "Steel", "Ice", "Dark"],
  "resisted_by": ["Flying", "Poison", "Bug", "Psychic", "Fairy"],
  "no_effect_against": ["Ghost"],
  "note": "Single defending types. Dual types multiply — call again with defending_types for a specific combination."
}
```

---

## `POST /get_sets`

Real, published Smogon sets — never invent a moveset when this can be called
instead.

**Request**

| Field | Type | Notes |
|---|---|---|
| `species` | string | Required |
| `format` | string | Optional filter, e.g. `"gen9ou"`. Omit to see every format |
| `limit` | int | 1–20, default 8 |

**Response**

```json
{
  "available": true, "species": "Kingambit", "count": 2,
  "sets": [
    {
      "format": "gen9ou", "name": "Swords Dance", "origin": "dex",
      "moves": ["Swords Dance", "Sucker Punch", "Kowtow Cleave", "Iron Head"],
      "item": "Leftovers", "ability": "Supreme Overlord", "nature": "Adamant",
      "evs": {"hp": 0, "atk": 252, "def": 0, "spa": 0, "spd": 4, "spe": 252},
      "tera": "Dark", "level": 100
    }
  ]
}
```

Results are ordered curated (`origin: "dex"`, human-written by Smogon)
before usage-derived (`origin: "stats"`), and within each, most recent
generation first. Say which origin you're quoting.

Not found (no species match, or no sets in that specific format) returns
`available: false` with a `hint`.

---

## `POST /common_generation`

The highest generation that includes every one of 2+ named Pokémon together.
Call this first, before any other tool, when building a team around specific
named Pokémon — not every Pokémon exists in every generation.

**Request**

| Field | Type | Notes |
|---|---|---|
| `species` | list[string] | Required, minimum 2 names |

**Response**

```json
{
  "available": true,
  "gen": 8,
  "per_species": {
    "Aegislash": {"resolved_id": "aegislash", "available_gens": [6, 7, 8]},
    "Hydreigon": {"resolved_id": "hydreigon", "available_gens": [5, 6, 7, 8, 9]}
  },
  "hint": "Capped below gen 9 by: Aegislash (through gen 8)."
}
```

`hint` is `null` when there's no real constraint (every named Pokémon reaches
the current generation). `gen` is `null` and `available` is `false` only if no
single generation includes all of them — genuinely rare, since generation
ranges are contiguous.

---

## `POST /query_items`

A quick reference for held items when building a team — filter or browse,
rather than looking up one item by name (`lookup` handles that). Combines
three clean boolean flags the dex provides directly with two wiki-derived
categories and free-text search for everything else.

**Request**

| Field | Type | Notes |
|---|---|---|
| `is_choice` | bool | Choice Band/Specs/Scarf |
| `is_berry` | bool | |
| `is_mega_stone` | bool | Enables a Mega Evolution |
| `category` | `"type_boost"` \| `"stat_boost"` | See below — only these two families have a clean category |
| `query` | string | Free-text search over the item's effect description |
| `gen` | int | Defaults to current |
| `limit` | int | 1–100, default 25 |

**Why only two categories.** Most of what makes an item competitively
meaningful — Choice's 1.5x, type-boosting, Assault Vest's Special Defense
boost — is implemented as executable effect code in real Showdown, not an
inspectable data field. `is_choice`, `is_berry`, and Mega Stone status
(derived from `mega_evolves`) are the only genuinely clean booleans
available. Of the remaining categories a competitive player would name, only
type-boosting and stat-boosting have a reliably-parseable source table (two
Bulbapedia articles, each with a real "List of..." table) — status orbs,
weather rocks, terrain seeds, gems, plates, and one-off items like Leftovers
or Rocky Helmet have no clean category field at all. Their `short_desc` is
precise enough that free-text search covers them well regardless — search
specific terms, not category words (`"Sunny Day"` not `"weather"`,
`"Electric Terrain"` not `"terrain"`).

**Response**

```json
{
  "count": 1,
  "results": [
    {
      "id": "charcoal", "name": "Charcoal",
      "short_desc": "Holder's Fire-type attacks have 1.2x power.",
      "is_choice": false, "is_berry": false,
      "is_mega_stone": false, "mega_evolves": null,
      "categories": [{"category": "type_boost", "detail": "Fire"}]
    }
  ]
}
```

`categories` is a list, not a single value — an item can belong to more than
one (Soul Dew is both `type_boost`, boosting Latios/Latias's Psychic- and
Dragon-type moves, and `stat_boost`, historically raising both Special
stats). Most items have an empty `categories` list; that doesn't mean
nothing is known about them, just that they're in the free-text-search-only
long tail — their `short_desc` is still there to search.

**Example — everything that boosts Fire-type moves specifically:**

```json
{"category": "type_boost", "query": "Fire"}
```

Combining a category filter with `query` narrows within that category rather
than replacing it — this example returns just Charcoal, not every
type-boosting item.

---

## `POST /validate_team`

Runs Pokémon Showdown's real `TeamValidator` — the same code that would
accept or reject the team on the actual ladder. Ground truth for legality.

**Request**

| Field | Type | Notes |
|---|---|---|
| `team` | string | Required — Showdown export format |
| `format` | string | Defaults to the server's default format |

**Response**

```json
{
  "valid": true,
  "problems": [],
  "format": "gen9ou",
  "species": ["Kingambit", "Great Tusk", "Gholdengo", "Toxapex", "Landorus-Therian", "Dragapult"],
  "export": "Kingambit @ Leftovers\nAbility: Supreme Overlord\n..."
}
```

`valid: false` with a populated `problems` array (plain-English strings from
the validator itself) on any legality failure — banned Pokémon/items, illegal
move combinations, clause violations, unparseable export text. `export` is the
team re-serialized by Showdown's own exporter, useful for confirming exactly
what was parsed.

---

## `POST /calc_damage`

Runs `@smogon/calc` — the real Smogon damage calculator, not an estimate.

**Request**

| Field | Type | Notes |
|---|---|---|
| `attacker` | object | `{name, level?, item?, ability?, nature?, evs?, ivs?, boosts?, status?, teraType?, alliesFainted?}` |
| `defender` | object | Same shape |
| `move` | object | `{name, crit?, hits?}` |
| `field` | object | `{weather?, terrain?, isReflect?, isLightScreen?, gameType?}` |
| `gen` | int | Defaults to current |

Only `attacker.name`, `defender.name`, and `move.name` are required.
Everything else defaults: level 100, neutral nature, zero EVs, no item, no
boosts, `alliesFainted: 0`.

**Response**

```json
{
  "description": "252+ Atk Supreme Overlord Kingambit Kowtow Cleave vs. 252 HP / 0 Def Toxapex: 138-163 (45.3 - 53.5%)",
  "damage": [138, 141, 144, 147, 150, 153, 156, 159, 163],
  "range": [138, 163],
  "percent_range": [45.3, 53.5],
  "ko_chance": "guaranteed 2HKO",
  "defender_max_hp": 304,
  "speed_order": "Kingambit moves first (50 vs 106 Speed)",
  "assumptions": {
    "gen": 9,
    "attacker_level": 100,
    "defender_level": 100,
    "attacker_evs": "none specified (0)",
    "defender_evs": {"hp": 252, "def": 0},
    "attacker_item": "none specified",
    "defender_item": "none specified",
    "attacker_allies_fainted": 0,
    "defender_allies_fainted": 0,
    "single_hit_no_sequencing": true,
    "note": "This is one hit, one turn, in isolation. It does not know who moves first (see speed_order above), does not apply recoil, recovery, weather, or multi-turn effects..."
  }
}
```

**`speed_order` matters.** Two matchups can both show "guaranteed OHKO" —
which one actually happens depends entirely on who moves first, which
`percent_range` alone doesn't tell you. Read `speed_order` before concluding
who wins the exchange.

**`alliesFainted` matters for Supreme Overlord, Beast Boost, and similar
abilities.** It defaults to 0 — a calculation for a Pokémon deep into a
sweep, with several allies already down, needs this passed explicitly or the
result reflects the ability's base state, not its realistic late-game value.

This is a single hit in isolation — no recoil, no recovery, no weather or
terrain ticking, no multi-turn sequencing. `assumptions.note` restates this
every time; read it rather than assume the number accounts for more than it
does.

---

## `POST /review_team`

Structural analysis of one team — call this before saying anything about a
team's quality, rather than reconstructing the analysis from individual
`lookup` calls (which will get speed stats and role coverage wrong).

**Request**

| Field | Type | Notes |
|---|---|---|
| `team` | string | Required — Showdown export format |
| `format` | string | Defaults to the server's default format |
| `gen` | int | Defaults to current |

**Response**

```json
{
  "format": "gen9ou", "gen": 9, "size": 6,
  "members": [
    {
      "name": "Kingambit", "types": ["Dark", "Steel"], "ability": "Supreme Overlord",
      "item": "Leftovers", "tera": "Dark", "speed": 50,
      "moves": ["Swords Dance", "Sucker Punch", "Kowtow Cleave", "Iron Head"]
    }
  ],
  "speed_tiers": [
    {"name": "Dragapult", "speed": 289, "note": "Choice Scarf (1.5x applied)"},
    {"name": "Kingambit", "speed": 50}
  ],
  "shared_weaknesses": [
    {"type": "Fighting", "count": 3, "members": ["Kingambit", "Toxapex", "Great Tusk"]}
  ],
  "roles": {
    "hazard removal": {"covered": true, "by": ["Great Tusk"]},
    "speed control": {"covered": false, "by": []}
  },
  "offensive_gaps": {
    "attacking_types": ["Dark", "Steel", "Ground", "Ghost", "Water"],
    "types_not_hit_super_effectively": ["Fairy"]
  },
  "structural_gaps": ["No speed control.", "3 members weak to Fighting: Kingambit, Toxapex, Great Tusk."],
  "note": "Structural review only. It does not judge whether individual sets are good..."
}
```

`speed` is the actual effective speed at level, with EVs/nature/Choice Scarf
applied — not base Speed, which is a different and usually much smaller
number. `roles` covers hazard removal, entry hazards, recovery, speed
control, pivoting, status, and setup, each checked against the team's actual
movesets. `structural_gaps` is a plain-English summary of what's missing,
generated from the same data — not a separate judgment call.

**Known limitation:** the underlying damage/stat engine only loads current-
generation dex data. A team containing a Pokémon that doesn't exist in the
current generation (Aegislash in Gen 9, for example) will error on that
member specifically. Call `common_generation` first and pass its answer as
`gen` to avoid this.

---

## `POST /compare_teams`

Structural head-to-head between two teams — speed order across both, shared
weaknesses on each side, and a full threat matrix from real damage
calculations for every attacker/defender pair.

**Request**

| Field | Type | Notes |
|---|---|---|
| `team_a` / `team_b` | string | Required — Showdown export format |
| `format` | string | Defaults to the server's default format |
| `gen` | int | Defaults to current |
| `label_a` / `label_b` | string | Default `"Team A"` / `"Team B"` |

**Response**

```json
{
  "format": "gen9ou", "gen": 9,
  "teams": {
    "Team A": {"size": 6, "members": [{"name": "Kingambit", "types": ["Dark","Steel"], "ability": "Supreme Overlord", "item": "Leftovers", "tera": "Dark", "speed": 50, "moves": ["..."]}]},
    "Team B": {"size": 6, "members": ["..."]}
  },
  "speed_order": [
    {"name": "Dragapult", "team": "Team A", "speed": 289, "note": "Choice Scarf (1.5x applied)"}
  ],
  "shared_weaknesses": {
    "Team A": [{"type": "Fighting", "count": 3, "members": ["..."]}],
    "Team B": [{"type": "Ice", "count": 2, "members": ["..."]}]
  },
  "threats": {
    "Team A -> Team B": [{"attacker": "Kingambit", "defender": "Toxapex", "max_pct": 53.5}],
    "Team B -> Team A": ["..."]
  },
  "summary": {
    "Team A can OHKO": "4 of 6",
    "Team B can OHKO": "2 of 6",
    "fastest": {"name": "Dragapult", "team": "Team A", "speed": 289}
  },
  "assumptions": {"note": "Damage uses the sets as written. Unspecified EVs default to 0..."},
  "disclaimer": "STRUCTURAL COMPARISON ONLY. This is not a prediction and contains no win probability. No battle was simulated and there is no outcome data behind these numbers..."
}
```

**No win probability, ever, by design.** There is no battle simulator and no
outcome data anywhere in this response — `disclaimer` says so explicitly,
every time. Nothing here supports stating or estimating a percentage chance
of either team winning. Describe the matchup from `speed_order`,
`shared_weaknesses`, and `threats` instead.

`threats` entries are capped at 24 per direction, sorted by `max_pct`
descending — the most dangerous matchups first, not every possible pairing.

---

## `POST /chat`

The tool-calling orchestrator behind the standalone page (`index.html`).
Open WebUI does not use this — it runs its own independent tool loop against
whichever connection it has configured. This endpoint exists specifically for
building an alternative chat frontend against this backend.

**Request**

| Field | Type | Notes |
|---|---|---|
| `messages` | list[dict] | OpenAI-style chat messages |
| `stream` | bool | Default `true` |
| `base_url` / `model` / `api_key` | string | Per-request overrides for this container's `OPENAI_*` env vars |

**Response: Server-Sent Events.** `Content-Type: text/event-stream`. Each
event is `event: <type>\ndata: <json>\n\n`. Four event types:

### `status`

One per tool call, as it happens — drive a "thinking" indicator or a list of
tool-call pills from this.

```
event: status
data: {"tool": "lookup", "args": {"name": "Kingambit"}}
```

A special tool name, `_finalize`, appears if the model exhausts its tool-call
budget without answering — the backend forces one last completion with no
tools available, synthesizing from whatever was already gathered rather than
erroring out:

```
event: status
data: {"tool": "_finalize", "label": "Summarizing findings so far"}
```

### `token`

Streamed answer text, in small chunks. Concatenate in order to reconstruct
the full answer.

```
event: token
data: {"t": "Kingambit is a Dark/Steel"}
```

### `done`

Sent once, at the end. Carries sources, and up to three groups of species
cards.

```json
{
  "sources": [{"title": "Kingambit (Pokémon)", "url": "http://.../Kingambit_..."}],
  "turns": 3,
  "images": [
    {
      "name": "Kingambit", "url": "http://.../poke_capture_0983...png",
      "types": ["Dark", "Steel"],
      "base_stats": {"hp": 100, "atk": 135, "def": 120, "spa": 60, "spd": 85, "spe": 50},
      "weak": ["Fighting", "Ground", "Fire"], "resist": ["Normal", "Flying", "..."],
      "immune": ["Psychic"],
      "item": "Leftovers", "ability": "Supreme Overlord", "nature": "Adamant",
      "moves": [{"name": "Sucker Punch", "type": "Dark", "category": "Physical", "base_power": 70, "pp": 5}]
    }
  ],
  "battle_forms": [],
  "mentioned": []
}
```

Three card groups, in order of how directly they're "the team":

- **`images`** — the actual species this answer is about. For a team-build
  answer, exactly the team's real members (taken from the most recent
  `validate_team`/`review_team`/`compare_teams` call's own parsed member
  list, not inferred from prose). For a single-Pokémon question, just that
  one species.
- **`battle_forms`** — an automatic in-battle form of something already in
  `images` (Aegislash-Blade, for example) — same team member, not a separate
  one, not optional. Distinguish from `images` in the UI, but don't count it
  as an additional team member or treat it as unrelated.
- **`mentioned`** — named in the answer but not part of the team: a threat
  example, an alternative the model considered but didn't recommend, a
  candidate a search turned up and rejected. Real data, worth showing, but
  should read as clearly separate from the team.

`item`/`ability`/`nature`/`moves` are populated only when the conversation
actually established a specific build for that Pokémon (a team context, or a
`get_sets` call) — `null`/absent for a plain species lookup with no build
context.

### `error`

```
event: error
data: {"message": "The response was cut off before it finished — this usually happens on a complex, many-step request. Try asking again, or breaking it into smaller questions."}
```

Ends the stream. No further events follow an `error`.

---

## `POST /models_proxy`

Utility endpoint for the standalone page's model picker — lists models from
an arbitrary OpenAI-compatible endpoint. Not part of the Pokémon data surface;
`{"base_url": "http://..."}` in, whatever that endpoint's `/v1/models` returns,
out.

---

## `GET /health` (on `pokedex-sim`, port 8991)

A separate, simpler health check on the Node sidecar directly — useful when
diagnosing whether `pokedex-sim` itself is up, independent of `pokedex-api`.

```json
{
  "status": "ok", "service": "pokedex-sim", "node": "v20.11.0",
  "packages": {"@pkmn/dex": "0.9.9", "@pkmn/sim": "0.9.9", "@smogon/calc": "0.10.0"},
  "app_version": "1.0.0"
}
```
