# DexAI

<div align="center">
<img width="1024" height="765" alt="DexAI Logo" src="https://github.com/user-attachments/assets/a170f33b-5f41-4b44-a2b6-48d0c1534917" />
</div>


A self-hosted Pokémon knowledge base you can talk to. Ask anything (games, anime,
TCG, competitive VGC) and it uses authoritative sources instead of
whatever the model hallucinates. Supports desktop and mobile.


**Note:** DexAI is an unofficial, non-commercial fan project. Not affiliated with Nintendo or The Pokémon Company. See [Disclaimer & License](#license--disclaimer) below.

<img width="1539" height="916" alt="Dex1" src="https://github.com/user-attachments/assets/bd174669-99a2-4091-ad28-f862d15cbc4a" />
<img width="1546" height="919" alt="Dex2" src="https://github.com/user-attachments/assets/c94f3968-8938-4195-a606-c3164da4b4c6" />
<img width="1540" height="921" alt="Dex5" src="https://github.com/user-attachments/assets/2130d700-445e-4589-97e9-8fffabd75c30" />
<img width="1539" height="916" alt="Dex6" src="https://github.com/user-attachments/assets/9b4b59f9-28a3-4635-8b70-9816736fb70a" />
<div align="center">
<img width="773" height="461" alt="Dex3" src="https://github.com/user-attachments/assets/79d0386f-4c6e-43ba-ac7e-d244bcb0b323" />
</div>
<img width="1533" height="919" alt="Dex4" src="https://github.com/user-attachments/assets/4a3092bf-8c41-475c-98aa-f801cd6336ec" />

This started from having my son and nephew asking me tons of questions as they have gotten into competitive Pokemon battles together. It was also an chance to get better ad coding and see what can be done with self hosted AI.

This guide assumes **nothing is set up**. Every step has sub-steps depending on what
you already run. Skip the ones that don't apply

---

## Table of contents

1. [What this is and how it works](#1-what-this-is-and-how-it-works)
2. [What it can and cannot answer](#2-what-it-can-and-cannot-answer)
3. [Data sources](#3-data-sources)
4. [Build scope — mini, standard, full](#4-build-scope--mini-standard-full)  
    4b. [**Embedding tiers — CPU, GPU, and long-context**](#4b-embedding-tiers--cpu-gpu-and-long-context)
5. [Network behavior and offline operation](#5-network-behavior-and-offline-operation)
6. [Pick your path](#6-pick-your-path)
7. [Host prerequisites](#7-host-prerequisites)
8. [**Step 0 — Quick start (the wizard)**](#step-0--quick-start)
9. [Step 1 — Directory structure](#step-1--directory-structure)
10. [Step 2 — The Bulbapedia ZIM](#step-2--the-bulbapedia-zim)
11. [Step 3 — A chat model endpoint](#step-3--a-chat-model-endpoint)
12. [Step 4 — A chat front end](#step-4--a-chat-front-end)
13. [Step 5 — The compose file (manual alternative)](#step-5--the-compose-file-manual-alternative)
14. [Step 6 — Deploy](#step-6--deploy)
15. [Step 7 — Run the ingest](#step-7--run-the-ingest)
16. [Step 8 — Verify the backend](#step-8--verify-the-backend)
17. [Step 9 — Connect Open WebUI](#step-9--connect-open-webui)
18. [Step 10 — The standalone page](#step-10--the-standalone-page-optional)
19. [Environment variable reference](#environment-variable-reference)
20. [Troubleshooting](#troubleshooting)
21. [Maintenance and upgrading](#maintenance-and-upgrading)
22. [Caveats for shared or public deployments](#caveats-for-shared-or-public-deployments)
23. [File manifest](#file-manifest)

---

## 1. What this is and how it works

### The core idea

**The language model does not know any Pokémon facts. It knows how to look them up
and how to explain them.**

This is the most important thing to understand. A model fine-tuned on Pokémon data
will confidently tell you Garchomp learns moves it doesn't, because that knowledge
is baked into weights you can't inspect or correct.

Instead, this system gives the model a set of tools. When you ask a question, it
decides what to look up, calls the tools, reads the results, and writes an answer
from what came back. If the answer is wrong, you fix the data — not the model.

### The request loop

```
  You ask a question
        │
        ▼
  Chat model  ──────────────┐   "What do I need to look up?"
        │                   │
        ▼                   │
  pokedex-api               │   Wiki search, dex lookup,
  (the tool server)         │   usage stats, team validation
        │                   │
        └───────────────────┘   loops until it has enough
        │
        ▼
  Chat model                    Writes the answer from what
        │                       the tools returned
        ▼
  Grounded answer with citations
```

### Worked example

You ask: *"Why is Kingambit so good in OU?"*

1. Model calls `lookup("Kingambit")` → typing, stats, abilities
2. Model calls `search_wiki("Supreme Overlord")` → how the ability works
3. Model calls `usage_stats("gen9ou")` → usage %, common sets, checks and counters
4. Model writes a paragraph connecting all three, linking to the Bulbapedia article
   on your own Kiwix server

You never picked a tool. You just talked.

### Components

| Component | What it is | Provided by this stack? |
|---|---|---|
| `pokedex-api` | Python/FastAPI. Owns the database, retrieval, and all tool endpoints. Exposes an OpenAPI spec. | **Yes — required** |
| `pokedex-sim` | Node sidecar wrapping `@pkmn/sim` (team validator) and `@smogon/calc`, which are JavaScript-only. Also serves team review and comparison. | **Yes — required** |
| Kiwix | Serves the Bulbapedia ZIM for full-text search and citation links. | Optional service in the YAML |
| Chat model | Any OpenAI-compatible endpoint **with tool calling**. | No — you supply this |
| Open WebUI | The chat interface. | Optional service in the YAML |
| `pokedex-ui` | Single-page chat UI, if you don't want Open WebUI. | Optional service in the YAML |

---

## 2. What it can and cannot answer

Read this before deciding whether to build it. The system is deliberately
narrow in some places, and knowing where saves disappointment later.

### The tools

| Tool | Answers | Backed by |
|---|---|---|
| `lookup` | Stats, typing, abilities, evolution, "can X learn Y", **level-up learnsets with levels**, species images (including shiny and animated). A purely visual form (Pikachu's caps, Antique Sinistea...) answers with its base species' data plus a note saying so | Showdown dex, HOME sprites |
| `query_dex` | "Which Pokémon are X and also Y" | Showdown dex |
| `type_matchup` | "What's super effective against Steel/Fairy" | Type chart, per generation |
| `search_wiki` | Anime, manga, TCG, events, mechanics, lore (game locations: see `get_locations`) | Bulbapedia |
| `usage_stats` | What's used in a metagame, with what, countered by what; for Pokémon Champions also win rate and how often a Pokémon is actually brought | Smogon ladder data; Pikalytics for the Champions formats |
| `get_sets` | Real named Smogon sets | Showdown teambuilder |
| `validate_team` | Is this team legal in this format | Showdown TeamValidator |
| `calc_damage` | Damage range and KO chance for one attack | `@smogon/calc` |
| `review_team` | Structural problems with one team | Derived |
| `compare_teams` | Head-to-head structure of two teams | Derived |
| `common_generation` | The highest generation that includes every one of 2+ named Pokémon together — call this first for any team built around specific named Pokémon, since not every Pokémon exists in every generation | Showdown dex |
| `get_locations` | Where a Pokémon can be found, per generation and game — filter by `gen` or `game`; a form returns its base species' locations. Main-series games only | Bulbapedia location tables, parsed from raw article HTML |
| `query_items` | Held item reference for team-building — filter by Choice/berry/Mega Stone/type-boost/stat-boost, or free-text search for anything else (orbs, weather rocks, terrain seeds, gems, plates, general items) | Showdown dex + Bulbapedia |

`pokedex-api` also serves a few REST routes that aren't in this list because
they're not something the model calls — `GET /dex_index` is a full
species-and-forms feed built for a UI's Pokédex browse view, and `GET /formats`
lists every competitive format that has sets or usage data (Gen 1 through
Gen 9) to drive a UI's format selector — neither is a chat answer. See `pokedex-ai-API.md` for the complete REST surface,
tool-callable or not.

### Answers well

- **Factual recall of anything in one place.** Base stats, learnsets, type
  matchups, where to catch something, episode summaries, card text, how a
  mechanic works.
- **"What level does X learn Y, in which game."** Learnset sources carry the
  method (level-up, TM, tutor, egg, event) and the level, per generation. Ask
  for a full level-up learnset and you get it ordered by level, with each move's
  type, power and effect — the actual question someone playing through a game
  has. Requires the generation to be in your build scope (section 4).
- **Legality.** It runs Showdown's real validator, so "is this team legal" has a
  correct answer, not an opinion.
- **Damage.** Real calculations, not estimates.
- **Metagame description.** What's popular, what checks what, how usage moved
  over the months you've fetched.
- **Team structure.** Missing roles, shared weaknesses, speed tiers, coverage
  gaps — the mechanical analysis that's tedious by hand.
- **Follow-ups.** Tool results stay in context, so "okay, what beats it?" works
  without re-establishing who "it" is.
- **Species images, including shiny and known alternate forms.** "Show me
  shiny Mega Charizard X" is directly answerable — if the sprite pack is set up
  (Step 2b). Without it, species images still work, just as whatever Bulbapedia's
  article artwork happened to be, rather than consistent official art. Either
  way, this is species artwork only — TCG card art, anime screenshots, and
  other imagery embedded in wiki articles is in the ZIM but not indexed.

### Cannot answer, by design

**Win probability.** There is no battle simulator and no outcome data. Asked
"what percent does team A win," a language model will invent a confident number.
Both the tool descriptions and the system prompt forbid it, and `compare_teams`
attaches a disclaimer to every response. You get the matchup described
qualitatively from measured facts instead.

**Anything outside your build scope.** A mini build knows generation 9 only. Ask
about Gen 3 base stats and it falls back to wiki prose, which may be vaguer or
absent. See section 4.

**Anything after your ZIM's build date.** The system never reads the live
Bulbapedia. A ZIM from 2024 does not know about 2026 releases.

**Live ladder data.** Usage statistics are monthly snapshots, not real time.

### Struggles with

**Cross-article aggregation.** "Which Pokémon appeared in the most episodes"
needs counting across hundreds of articles. Retrieval returns chunks, not
aggregates, and the model may answer confidently from a partial sample. The
system prompt tells it to flag incomplete evidence; it won't always.

**Ambiguity it doesn't notice.** "How much damage does Earthquake do" has no
answer without an attacker, a defender and a generation. It's told to ask one
clarifying question rather than guess.

**Set quality judgement.** `review_team` checks structure, not whether an
individual set is well built. It won't tell you your EV spread is wrong.

### The design principle behind all of this

The model supplies no Pokémon facts. It routes questions to tools and explains
what comes back. When it's wrong, you fix the data — and the wrongness is
usually visible, because answers cite where they came from.

---

## 3. Data sources

Three sources feed this system. Each is authoritative for different things, and
knowing which is which is the whole reason the answers can be trusted.

### At a glance

| Source | Authoritative for | How it's captured | Where it's stored | Which tool reads it |
|---|---|---|---|---|
| **Bulbapedia** (ZIM) | Lore, anime, manga, TCG, game locations, event distributions, mechanic explanations, trivia | Offline read of the ZIM during ingest | `wiki_chunks` + vector index | `search_wiki` |
| **Pokémon Showdown** (`@pkmn`) | Base stats, typing, abilities, learnsets, moves, items, type chart, format rules and banlists | `/dump/:gen` from the sidecar during ingest, **plus live calls at query time** | `species`, `moves`, `abilities`, `items`, `learnsets`, `typechart` | `lookup`, `query_dex`, `validate_team`, `calc_damage` |
| **Smogon usage stats** | What people actually play, by format and month | HTTP fetch during ingest | `usage_stats` | `usage_stats` |
| **Smogon analyses** | Why sets work — roles, spreads, checks and counters | HTTP fetch during ingest | `wiki_chunks`, tagged `source='smogon'` | `search_wiki` |
| **Smogon sets** | Curated named sets: moves, item, ability, nature, EVs, Tera | `fetch-sets.sh` from Showdown's mirror | `sets` | `get_sets` |
| **Pikalytics** | Pokémon Champions ladder usage: moves, items, abilities, teammates, win rate | `fetch-pikalytics.py` on a machine with internet, then read from disk | `usage_stats` + `usage_extra` | `usage_stats` |

Everything lands in one SQLite file at `/data/pokedex.db`. No Postgres, no vector
database, no extra containers.

### 3.1 — Bulbapedia (the ZIM)

**What it gives you.** The encyclopedia. Everything narrative or descriptive:
episode summaries, character histories, where to catch things in each game, event
distribution details, TCG set information, manga continuity, and long-form
explanations of how mechanics work.

**How it's captured.** During ingest, `ingest.py` opens the ZIM file directly with
`libzim` and iterates every entry. For each article it:

1. Skips non-article namespaces (Talk, User, File, Category)
2. Parses the MediaWiki HTML and splits it at heading boundaries
3. Converts tables to markdown rather than stripping them — Bulbapedia's game
   locations, TM availability, and event data all live in tables, and flattening
   them to prose destroys them
4. Prepends the article title and full section path to each chunk, so
   `Pikachu → In the anime → Ash's Pikachu` is embedded along with the text
5. Records ZIM redirects into the alias table (see 2.5)

**Why read the file rather than crawl Kiwix.** Kiwix's HTTP search returns ranked
whole articles. You need chunks, section paths, tables, and redirect data — all of
which require reading the archive directly. Kiwix stays in the picture for live
search and citation links, not for ingest.

**How it's referenced.** `search_wiki` runs hybrid retrieval — FTS5 keyword search
plus vector similarity, merged. Hybrid matters here because Pokémon names need exact
lexical matching; pure semantic search confuses Iron Valiant with Iron Hands. Each
result carries its article title, section path, and a Kiwix URL the model cites.

**Caveats.** It's a fan-maintained wiki and a dated snapshot. It occasionally
contradicts the game data, and it will not know about anything released after the
ZIM was built. Treat it as authoritative for narrative, never for battle numbers.

### 3.2 — Pokémon Showdown data layer

**What it gives you.** The hard numbers. Every species' base stats, types,
abilities, and evolution data; every move's power, accuracy, priority, and flags;
every item; complete learnsets; the type chart; and the rules and banlists for every
competitive format.

**Why it's the authority.** This is the data layer that Pokémon Showdown itself runs
on — the same code the competitive community plays on every day. It's maintained
continuously, versioned, and executable. When it says a Pokémon can't learn a move
in a given format, that's not an opinion; it's the rule that would reject your team
on the ladder.

**How it's captured.** Two different ways, and the distinction matters:

- **Snapshot, at ingest.** `ingest.py` calls `GET http://pokedex-sim:8991/dump/9`.
  The Node sidecar reads `@pkmn/dex` and `@pkmn/data` and returns the full
  generation as JSON, which gets written into SQLite tables. This is what powers
  fast lookups.
- **Live, at query time.** `validate_team` and `calc_damage` are *not* stored data —
  they're executed. The model's request goes through `pokedex-api` to the sidecar,
  which runs Showdown's actual `TeamValidator` and `@smogon/calc` and returns the
  result. You can't precompute team legality; there are too many combinations.

**How it's referenced.**

- `lookup` — canonical record for one entity, straight SQL
- `query_dex` — filtered search ("Water types, base Spe ≥ 100, learns Rapid Spin")
- `validate_team` — live call to the sidecar; the model must run this on every team
  it proposes
- `calc_damage` — live call to the sidecar
- `compare_teams` — live call to the sidecar; head-to-head structural analysis

**Keeping it current.** `docker exec pokedex-sim npm update`, restart the sidecar,
then `ingest.py --dex-only`. Do this after a DLC, a new generation, or a tier shift.

### 3.3 — Smogon usage statistics

**What it gives you.** Monthly aggregates from millions of ladder games: how often
each Pokémon is used in each format, what moves and items and EV spreads it runs,
what it's commonly paired with, and what checks and counters it.

**How it's captured.** `ingest.py` fetches the "chaos" JSON files published at:

```
https://www.smogon.com/stats/YYYY-MM/chaos/FORMAT-CUTOFF.json
```

`CUTOFF` is a ladder rating threshold — `0`, `1500`, `1630`, `1695`, `1825`
depending on the format. `1695` is the cutoff Smogon uses for OU tiering decisions,
so that's the sensible default. The ingest pulls the last twelve months so the model
can talk about trends, not just the current snapshot.

**How it's stored.** A `usage_stats` table keyed by format, month, and cutoff. The
history is kept deliberately — "Kingambit's usage climbed from 12% to 24% over six
months" is a question this data can answer and a single snapshot can't.

**How it's referenced.** The `usage_stats` tool, which the model calls for anything
about the current metagame.

**Caveats.** Usage statistics are *descriptive, not prescriptive*. They tell you
what people play, not what's optimal. Popular is not the same as good, and the
bottom of the usage table is full of perfectly viable Pokémon nobody bothers with.
The system prompt should keep the model from confusing the two.

### 3.3b — Pikalytics (Pokémon Champions usage)

**What it gives you.** Ladder usage for Pokémon Champions, which Smogon doesn't
publish: per Pokémon, its usage, win rate and record, how often it is actually
brought to a battle, its moves, items, abilities and teammates, and how often it
is used as a Mega. Two formats are fetched by default, both verified to return real data, but the data is of two kinds:

- `gen9championsvgc2026regmc` (the current VGC ladder) publishes a **usage share**, win rate,
  how often a Pokémon is brought, leads and Mega share — but no spreads or natures.
- `battledataregmbs3` (the M-B season 3 battle data) is the reverse: **no usage share** at
  all, but games played, wins and losses, a Pikalytics rank, and real builds — spreads and
  natures. Its `usage_pct` is `null` and the API says so; it is ranked by Pikalytics' own
  rank, which is *not* ordered by games played (a rank-3 Pokémon has more games than rank 1),
  so no usage number is ever derived from the games count. Its teammate lists carry ranks
  without percentages and are returned in rank order.

**Its spreads are Stat Points, not EVs.** Pokémon Champions replaced EVs with Stat Points,
up to 32 in each stat and 66 in all, and the spreads (`2/32/0/0/0/32`, in HP/Atk/Def/SpA/
SpD/Spe order) sum to that. The response says so in `spread_unit`. Do not compare them
with the Smogon EV spreads (252/0/4/...), or convert between the two.

**How it's captured.** `fetch-pikalytics.py`, run on any machine with internet,
writes one JSON file per format into `pikalytics/`; `ingest.py --pikalytics-only`
then loads them into the same `usage_stats` table the Smogon data lives in, so
the `usage_stats` tool works on Champions formats unchanged. See Step 2e.

**Caveats — read these before trusting a number.**

- **The endpoints are undocumented.** Pikalytics' documented AI interface only
  exposes each format's top 50. The full roster comes from the same `/api/`
  endpoints the site's own pages call, which are not a published interface and
  can change without notice. The fetcher is deliberately gentle (an honest
  User-Agent, two-second gaps, about two requests per format) and fails loudly —
  and writes nothing — if the answer looks wrong. No terms-of-use or rate-limit
  statement was found either way.
- **Credit Pikalytics.** It asks to be credited. Every Champions `usage_stats`
  response carries `source: "Pikalytics"` and the system prompt tells the model
  to name it.
- **Not every percentage is the same kind of number.** Ability and item
  percentages are shares that sum to about 100. Move and teammate percentages
  do not (one Pokémon's top ten moves added to 166%). The response's `note` says
  so, and the ingest log prints the check on real data.
- **What isn't there.** The VGC data has no spreads or natures (the fields are empty in the
  file; the battle-data format does have them). Neither format's "counters" list is ingested: it gives
  a win percentage and a game count but never says whose win rate it is, and the
  samples are tiny (19 games for the top entry on one Pokémon).
- **It can be old.** Each format's own page states a "Data Date". At the time of
  writing (September 2026) the Champions data was dated 2026-05. `usage_stats`
  reports the month it is actually using.
- **Legends Z-A has no usage data anywhere.** Pikalytics lists a `za` format, but
  every value in it is a `-1%` placeholder, so it is not ingested.
- **Not for the battle tools.** Like the Z-A Megas, Champions formats aren't in
  the simulator, so `validate_team` and `calc_damage` don't work with them.

### 3.4 — Smogon strategy analyses

**What it gives you.** The written analyses from Smogon's Strategy Dex — why a set
is built the way it is, what role a Pokémon plays, what it beats and loses to. This
is the reasoning layer that turns "here are six legal Pokémon" into "here is a team
with a coherent plan."

**How it's captured and stored.** Fetched during ingest and written into the same
`wiki_chunks` table as Bulbapedia, tagged `source='smogon'` so retrieval can filter
by source when a question is clearly competitive or clearly not.

**How it's referenced.** `search_wiki`, same as Bulbapedia content.

### 3.5 — The alias table (how the sources get joined)

This is the least glamorous part of the system and the most load-bearing.

The same Pokémon has a different name in every source:

| Source | Identifier |
|---|---|
| Bulbapedia article title | `Urshifu (Pokémon)` |
| Showdown ID | `urshifurapidstrike` |
| Showdown display name | `Urshifu-Rapid-Strike` |
| Common usage | `Rapid Strike Urshifu`, `RS Urshifu` |

Without a mapping, a question about Urshifu retrieves wiki text about one thing and
dex data about another. Regional forms, Megas, Paradox Pokémon, Ogerpon masks, and
Rotom appliances all make this worse.

**How it's built.** During ingest, from three inputs merged into one `aliases`
table:

1. Every ZIM redirect (Bulbapedia has thousands — they're free alias data)
2. Every Showdown ID and display name from the dex dump
3. Normalization rules: lowercase, strip punctuation and spaces, handle the
   `Species-Forme` convention

**How it's used.** Every tool resolves the user's phrasing to a canonical ID
*before* doing anything else. `lookup`, `query_dex`, and `search_wiki` all filter on
the canonical ID, which is what makes the wiki text and the dex row line up.

### 3.6 — Generation scoping

Every dex row and every wiki chunk carries a generation tag.

This is not optional detail. "Does Blissey learn Seismic Toss" has different answers
in Gen 4 and Gen 9. Steel resisted Dark until Gen 6. Without generation tags, the
model will cheerfully mix Gen 5 mechanics into a Gen 9 answer and sound completely
confident doing it.

Retrieval filters on generation. When the user doesn't specify one, the tools assume
the current generation and return that assumption in the response, so the model can
state it rather than hide it.

### 3.7 — Precedence: what wins when sources disagree

They will disagree. The rules, which are encoded in the system prompt:

| Question type | Winner | Why |
|---|---|---|
| Legality, learnsets, base stats, type matchups | **Showdown** | Executable code, continuously maintained |
| Anime, manga, TCG, characters, lore | **Bulbapedia** | Showdown has no concept of these |
| Where a Pokémon can be found, per generation and game | **`get_locations`** | Bulbapedia's own location tables, parsed into structured data |
| Encounter rates, event distributions | **Bulbapedia** (`search_wiki`) | Showdown only models battles |
| "What's good right now", tier placement | **Smogon usage** | It's literally a measurement |
| How a mechanic works | **Showdown**, explained by **Bulbapedia** | Bulbapedia's prose is clearer; Showdown's numbers are correct |

The practical version: if a Bulbapedia article and the Showdown data layer disagree
about a battle mechanic, Showdown is right. Bulbapedia is written by people;
Showdown is run by a simulator that would break if it were wrong.

### 3.8 — What is deliberately not captured

- **Images embedded in wiki articles.** TCG card art, anime screenshots, and
  other imagery inside the ZIM are not indexed or extracted. This is separate
  from species artwork — see Step 2b, which is a different, optional data
  source entirely, not derived from the ZIM at all.
- **Talk and user pages.** Filtered out during ingest as noise.
- **Pokémon GO and mobile spinoff data.** Only what Bulbapedia happens to cover.
- **Live ladder data.** Usage statistics are monthly snapshots, not real time.
- **Legends Z-A ranked usage.** No source publishes it (Pikalytics' `za` entry is placeholder data).
- **Champions counters, and spreads for the VGC format.** Counters have no defined meaning and tiny samples; only the battle-data format publishes spreads — see 3.3b.
- **Sprites for Totem Pokémon.** Pokémon HOME has none, so the 12 Totem forms have no image.
- **Anything after your ZIM's build date.** The system does not reach out to the
  live Bulbapedia.

---

## 4. Build scope — mini, standard, full

The wizard asks how many Pokémon generations to index. This is the single
biggest decision about what the system can answer, and the easiest one to
regret, so it's worth understanding before you pick.

### Why generations matter

Pokémon is not one game with one rule set. Across generations:

- **Base stats changed.** Several Pokémon were buffed in gens 6 and 7.
- **Move power changed.** Many moves were rebalanced in gen 6.
- **The type chart changed.** Steel resisted Dark and Ghost until gen 6. Fairy
  did not exist before gen 6.
- **Abilities did not exist** before gen 3.
- **Special was a single stat** in gen 1, split into Special Attack and Special
  Defense in gen 2.
- **Learnsets differ** in every generation.

If you index generation 9 only and then ask about Emerald or HeartGold, the
structured data has nothing and the answer falls back to whatever Bulbapedia
prose happens to be retrieved. That's often fine for narrative questions and
poor for exact numbers.

### What a wider scope actually unlocks

Concretely, with generation 3 indexed you can ask:

- *"What level does Swampert learn Muddy Water in Emerald?"* — exact level
- *"Give me Gardevoir's full level-up learnset in gen 3"* — every move, ordered
  by level, with type and power as they were then
- *"Was Steel resistant to Ghost back then?"* — yes, and the type chart says so
- *"What were Alakazam's base stats in gen 1?"* — one Special stat, not two

Without that generation indexed, each of those falls back to wiki prose or
simply isn't answerable. Learnset entries store the method too — level-up, `machine` (TM/HM), tutor, egg, event, `dream world` — so "how do I get this move" is answered as precisely as "can I get it".

**How learnsets are stored, and the rule for reading them.** Every generation's rows carry the species' *complete* source list: Gardevoir's Gen 3 and Gen 9 rows are identical, with sources from Gen 3 through Gen 9 in both. The `gen` column alone therefore filters nothing; `source_gen` says which generation a source belongs to. The API applies two rules:

- **What a generation's games offered** (`POST /learnset`, and the level-up list and method counts in `lookup`): `source_gen = gen`.
- **Whether a Pokémon can learn a move by a generation** (`moves_to_check`, `query_dex` with `learns`, `moves_known`): `source_gen <= gen` — a source from a later generation can never apply.

Reading `gen` without `source_gen` gives wrong answers, such as a Gen 3 Gardevoir with 320 TM entries and a Gen 5 Dream World move.

### The three scopes

| Scope | Generations | Dex stage | Sets | Best for |
|---|---|---|---|---|
| **Mini** | 9 | ~2 sec | No | Current-gen play only |
| **Standard** | 7, 8, 9 | ~8 sec | Yes | Sun/Moon onward — most modern play |
| **Full** | 1–9 | ~25 sec | Yes | Replaying older games, cross-gen history |

**Those are seconds, measured.** All nine generations — 5,479 species, 4,473
moves, 1.64 million learnset entries — ingest in 25 seconds and add under 200 MB.
The dex stage is not a meaningful cost, which makes scope essentially a free
choice.

**So pick Full unless you have a reason not to.** The only argument for a
narrower scope is a slightly smaller database.

Scope does **not** affect the wiki, which is always indexed in full. Every
scope answers anime, TCG, lore, and location questions identically.

### Upgrading later

Scope is not a one-way door. Adding generations does **not** require rebuilding
the wiki index or regenerating embeddings — that's the expensive part, and it's
untouched.

**Mini → Full**, or any other change:

```bash
# 1. edit .env
GENS="1,2,3,4,5,6,7,8,9"

# 2. restart so the container picks up the new value
sudo docker compose up -d          # or restart the app in TrueNAS

# 3. rebuild only the dex and aliases
sudo docker exec -d pokedex-api /app/.venv/bin/python ingest.py --dex-only
```

`--dex-only` refetches the dex for every generation in `GENS` and rebuilds the
alias table. It leaves `wiki_chunks`, `usage_stats`, and the embedding file
alone. Watch it with `tail -f /mnt/Apps/pokedex/data/ingest.log`.

Confirm it took:

```bash
curl -s http://<HOST_IP>:8990/health | python3 -m json.tool | grep -A2 generations
```

`generations_loaded` should list every generation you asked for.

**Downgrading** works the same way — set a shorter `GENS` and re-run
`--dex-only`. The dex stage deletes per-generation before writing, so removed
generations disappear cleanly.

### Adding Smogon sets to an existing build

If you chose mini and later want curated sets:

```bash
./fetch-sets.sh                    # ~8 MB, needs internet once
sudo docker exec pokedex-api /app/.venv/bin/python ingest.py --sets-only
```

About a minute. Nothing else is touched.

### Disk cost of a wider scope

Modest. Learnsets are the bulk of it — about 1.7 million rows for all nine generations at the time of writing (`/health` reports the live figure), because each generation stores the species' full source list. That's a few hundred megabytes on top of a
database already measured in gigabytes because of the wiki. Scope is a time
decision far more than a space decision.

---

## 4b. Embedding tiers — CPU, GPU, and long-context

Turning every wiki chunk into a search vector is the slowest single stage of
the build, and it's the one place where your hardware choice changes build
time by an order of magnitude rather than a percentage.

### The three tiers

| Tier | What it needs | Concurrency | Measured throughput | A ~700k-chunk build |
|---|---|---|---|---|
| **cpu** | Nothing extra | — | 28–36 chunks/s, flat across 1–192 threads | 5–7 hours |
| **gpu-small** | Any GPU, 8–16GB | 6 requests in flight | ~73 chunks/s sequential baseline | ~2.5 hours |
| **gpu-big** | 24GB+ card | 16 requests in flight | ~270 chunks/s and climbing, GPU at 39% utilization | ~45 min |

These are measured, not estimated — on real Bulbapedia content, a single AMD
Radeon AI PRO R9700 (32GB, gfx1201).

**CPU throughput is flat regardless of thread count.** 16 threads, 64, and 192
all land in the same 28–36/s range. The bottleneck is serial — Python-side
tokenization and per-item overhead, not compute — so a bigger CPU host doesn't
help here. If you have any spare GPU, even a modest one, it's almost always
worth using.

**`gpu-small` vs `gpu-big` is about how much concurrent request-handling
overhead a card can absorb, not raw VRAM.** For a 33M-parameter model like the
default, the GPU forward pass is nearly instant — the real cost is the HTTP
round-trip and JSON handling around it. A bigger card handles more requests
in flight before that overhead becomes the bottleneck. Pushing `gpu-big`'s
concurrency past 16 kept helping in testing (GPU utilization was still only
39%), so if you have real headroom, `EMBED_CONCURRENCY` is worth raising
further — see the environment variable reference.

### Picking one

The wizard asks this directly (Step "How should embeddings be generated?").
If you're unsure: pick CPU now, add a GPU later. Nothing about picking CPU
first locks you out of switching — a re-embed is `ingest.py --wiki-only` or
`--embed-only`, not a rebuild.

### The long-context option

The default model, `BAAI/bge-small-en-v1.5`, has a 512-token window. Roughly
**15–25% of real Bulbapedia chunks** — table-heavy sections especially —
exceed that. `ingest.py` handles this automatically: it shrinks and retries
the specific chunks that overflow, so nothing is lost, but each retry costs
extra round-trips and the resulting vector only represents part of the chunk.

`nomic-ai/nomic-embed-text-v1.5` supports **8192 tokens** — those same chunks
embed whole, on the first request. It's a larger model (137M vs 33M
parameters) but still light enough that a 24GB+ card has room to spare. This
is squarely "let a big GPU stretch its legs" territory: more context, not
more raw horsepower, is what actually helps here.

**It needs a prompt prefix, or retrieval quality degrades silently.** Nomic's
convention is `"search_document: "` prepended to everything indexed and
`"search_query: "` prepended to everything searched. Get this wrong — set one
without the other, or leave both unset with this model — and nothing errors;
search results just get quietly worse. The wizard sets both together when you
choose the long-context option. If you configure this by hand, set
`EMBED_DOC_PREFIX` in `ingest.py`'s environment and the matching
`EMBED_QUERY_PREFIX` in `main.py`'s — see the environment variable reference.

**Switching models always requires a full re-embed.** Vectors from two
different models aren't comparable and can't share an index. `/health` warns
if `EMBED_MODEL` doesn't match what actually built the current index, so a
mismatch is visible rather than a silent quality regression:

```bash
curl -s http://<HOST_IP>:8990/health | python3 -m json.tool | grep -A3 embeddings
```

### Running your own embedding server

If you chose a GPU tier and asked the wizard to bundle one, it generated a
`pokedex-embed` service using `rocm/vllm:latest` with a placeholder comment
telling you to swap the image if it doesn't start. Vanilla vLLM images can
fail outright on some AMD cards — `KeyError: 'gfx1201'` from an AITER import
that runs unconditionally at startup, before vLLM even reaches its CLI, is a
known failure on RDNA4. If you hit that, a tuned fork (this system's own
development used `stilldeadcode/vllm-radiance`) or any other image known to
work on your card is the fix — the compose service accepts any image serving
an OpenAI-compatible `/embeddings` endpoint.

**Test the endpoint before running the ingest, not after it fails partway
through:**

```bash
curl -s http://<HOST_IP>:<EMBED_HOST_PORT>/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"model":"<EMBED_MODEL>","input":["test"]}'
```

Expect a 384-dimension vector back for the default model, 768 for
nomic-embed-text. A `400 Bad Request` on a short test string like `"test"`
usually means a CLI flag mismatch — check your vLLM version's flag for
pooling/embedding mode (`--runner pooling` on 0.27+, `--task embed` on
older releases) and whether it needs an explicit `entrypoint:` override or
already provides one.

### Pointing at an endpoint you already run

If you already run vLLM or lemonade-server for your chat model, you may be
able to reuse it — or a sibling instance of it — for embeddings instead of
standing up something new. Choose "I already run a compatible endpoint
elsewhere" in the wizard, or set `EMBED_URL` and `EMBED_MODEL` by hand. Both
cases below plug into the exact same `EMBED_URL`/`EMBED_MODEL` mechanism; only
the value on the right differs.

**vLLM.** A running vLLM instance serves one model, loaded at startup —
it cannot also serve a second, different model for embeddings on the same
process. If your existing vLLM is already busy with a chat model, you need a
**second instance** pointed at the embedding model, the same way `pokedex-embed`
works: launched with `--runner pooling` (vLLM 0.27+) or `--task embed`
(older releases). That second instance is what the wizard's bundled
`pokedex-embed` service already gives you — if you'd rather run it yourself
outside this stack, the flags are identical, just point `EMBED_URL` at wherever
you put it:

```bash
EMBED_URL=http://<your-vllm-host>:<port>/v1
EMBED_MODEL=BAAI/bge-small-en-v1.5    # must match what that instance actually loaded
```

**lemonade-server.** Genuinely simpler here: lemonade already exposes
`/v1/embeddings` on the **same server** your chat model runs on. No second
instance — just point at a different model name and lemonade loads it on
demand:

```bash
EMBED_URL=http://<your-lemonade-host>:13305/v1
EMBED_MODEL=nomic-embed-text-v1-GGUF    # or whatever embedding model you've pulled
```

**One real constraint:** lemonade's embeddings endpoint only works for models
using the `llamacpp` or `flm` recipe — **not** ONNX/OGA models. If your
lemonade deployment runs an ONNX build for NPU acceleration, embeddings are
not available through that model, and you'll need a GGUF-based embedding
model pulled separately (`lemonade-server pull nomic-embed-text-v1-GGUF` or
similar) rather than trying to reuse your existing chat model's recipe.

**Base path varies by lemonade version**, same as it does for chat (Step 3.4)
— some builds use `/v1`, others `/api/v1`. Test both if the first 404s:

```bash
curl -s http://<host>:13305/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"model":"nomic-embed-text-v1-GGUF","input":["test"]}'
# if that 404s:
curl -s http://<host>:13305/api/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"model":"nomic-embed-text-v1-GGUF","input":["test"]}'
```

**The prompt-prefix requirement is a property of the model, not the server.**
If you point either vLLM or lemonade at `nomic-embed-text`, you still need
both `EMBED_DOC_PREFIX` and `EMBED_QUERY_PREFIX` set correctly — which server
is hosting the model doesn't change what the model itself expects.

Whichever path you use, confirm it before running the ingest — the same test
as above, with your real `EMBED_URL` and `EMBED_MODEL`, expecting a 384- or
768-dimension vector back depending on the model.

---

## 5. Network behavior and offline operation

**Once set up, this system makes no internet requests at query time.** Every
question is answered from your ZIM, your database, your sidecar, and your model —
all on your own hardware.

What needs the internet is setup and periodic refresh. This section lists exactly
what, when, and how to eliminate each one.

### 3.1 — What reaches the internet, and when

| Call | When | Needed at query time? | How to eliminate |
|---|---|---|---|
| `pip install` | Container start | No | Sentinel file — 3.2 |
| `npm install` | Container start | No | Already skipped when `node_modules` exists — 3.2 |
| Embedding model download | First ingest only | No | Cached to `/data/models`, then forced offline — 3.3 |
| Smogon chaos JSON | Ingest and monthly refresh | No | Pre-download to a local directory — 3.4 |
| Open WebUI telemetry / update check | Startup, periodically | No | Environment variables — 3.5 |
| Your chat model | Every query | **Only if hosted** | Self-host it (Step 3.1, 3.3, or 3.4) |

Kiwix, `pokedex-sim`, SQLite, retrieval, and embedding inference are already local
with no configuration needed.

### 3.2 — Make package installs happen only once

The compose commands as written re-run `pip install` on every container start. With
exact pins pip usually short-circuits, but without internet it can hang or fail the
start entirely. Use a sentinel file instead, so the install runs once and never
again:

```yaml
  pokedex-api:
    command: >
      sh -c "if [ ! -f /app/.venv/.installed ]; then
               python -m venv /app/.venv &&
               /app/.venv/bin/pip install -r requirements.txt &&
               touch /app/.venv/.installed;
             fi;
             exec /app/.venv/bin/uvicorn main:app --host 0.0.0.0 --port 8990"
```

```yaml
  pokedex-sim:
    command: >
      sh -c "if [ ! -f node_modules/.installed ]; then
               npm install --no-audit --no-fund &&
               touch node_modules/.installed;
             fi;
             exec node server.js"
```

After first run, both containers start with zero network activity. To force a
reinstall later, delete the sentinel and restart.

Also pin exact versions in `requirements.txt` (`fastapi==0.115.6`, not
`fastapi>=0.115`). Unpinned requirements make pip query PyPI even when everything is
already installed.

### 3.3 — Pin the embedding model locally

This is the one that bites people. `fastembed` and `sentence-transformers` contact
HuggingFace on startup to check the model, and with no internet they hang rather
than fail fast.

Add to `pokedex-api`'s environment:

```yaml
      HF_HOME: /data/models          # cache on the persistent volume
      HF_HUB_OFFLINE: "1"            # never contact HuggingFace
      TRANSFORMERS_OFFLINE: "1"
```

The model still has to be downloaded once. Do it during the first ingest by
overriding the offline flag for that single command:

```bash
docker exec -e HF_HUB_OFFLINE=0 -e TRANSFORMERS_OFFLINE=0 \
  pokedex-api /app/.venv/bin/python ingest.py
```

From then on the cached model in `/data/models` is used and the offline flags apply
normally. Back that directory up along with the database.

### 3.4 — Smogon statistics without live fetching

Usage statistics are the only data source that lives on the internet rather than on
your disk. They're also the only one that changes monthly.

**Pre-download them.** On any machine with internet:

```bash
mkdir -p /mnt/Apps/pokedex/stats

for MONTH in 2026-04 2026-05 2026-06 2026-07 2026-08; do
  for FMT in gen9ou gen9vgc2026 gen9ubers gen9uu; do
    curl -fsS -o "/mnt/Apps/pokedex/stats/${FMT}-1695-${MONTH}.json" \
      "https://www.smogon.com/stats/${MONTH}/chaos/${FMT}-1695.json" \
      && echo "got ${FMT} ${MONTH}"
  done
done
```

Mount the directory and point ingest at it:

```yaml
    volumes:
      - /mnt/Apps/pokedex/stats:/stats:ro
    environment:
      STATS_DIR: /stats
      OFFLINE: "1"                   # app never fetches at runtime
```

With `STATS_DIR` set, `ingest.py --stats-only` reads from disk instead of
smogon.com. Refreshing monthly becomes "download four files, re-run one command" —
and you can do the download from a laptop and copy the files over.

Formats with no cutoff at 1695 use a different threshold; check what exists at
`https://www.smogon.com/stats/YYYY-MM/chaos/` before scripting a format you're
unsure about.

### 3.5 — Open WebUI telemetry

Open WebUI performs version checks and analytics calls by default. Add to its
environment:

```yaml
      WEBUI_AUTH: "True"
      ANONYMIZED_TELEMETRY: "False"
      DO_NOT_TRACK: "true"
      SCARF_NO_ANALYTICS: "true"
      ENABLE_OPENAI_API: "True"
      ENABLE_OLLAMA_API: "False"     # unless you actually use Ollama
```

These are the documented switches, but Open WebUI changes quickly and new phone-home
features do appear. If total isolation matters, verify with 3.6 after any upgrade
rather than trusting the flags.

### 3.6 — Verify it's actually offline

The honest test is to block egress and use it.

Temporarily deny the containers outbound internet — a firewall rule on the host, or
pulling the WAN cable — then run the Step 9.3 test prompts. If all five answer
normally, you're offline-clean.

To see what a container is *attempting* to reach:

```bash
docker exec pokedex-api sh -c \
  "timeout 5 curl -sI https://huggingface.co >/dev/null && echo 'has egress' || echo 'no egress'"
```

That tells you whether egress exists, not whether the app needs it. Only the
block-and-use test answers the real question.

### 3.7 — Fully air-gapped installation

For a host that will never have internet, stage everything on a machine that does
and copy the whole tree over:

1. Run the full deployment on the staging machine through Step 7 (ingest complete)
2. Stop the containers
3. Copy `/mnt/Apps/pokedex/` entirely — including `api/.venv/`, `sim/node_modules/`,
   `data/pokedex.db`, `data/models/`, `zim/`, and `stats/`
4. Pre-pull the base images:
   ```bash
   docker save python:3.12-slim node:22-slim nginx:alpine \
     ghcr.io/kiwix/kiwix-serve:latest ghcr.io/open-webui/open-webui:main \
     -o pokedex-images.tar
   # on the air-gapped host:
   docker load -i pokedex-images.tar
   ```
5. Deploy the compose file on the target host

The venv and `node_modules` are architecture-specific — stage on the same platform
(x86-64 Linux to x86-64 Linux). The sentinel files from 3.2 travel with them, so
nothing tries to reinstall.

### 3.8 — What you lose by going fully offline

Being honest about the trade:

- **Usage statistics go stale** between manual refreshes. A three-month-old
  metagame picture is usually fine; during a tier shift it isn't.
- **The dex doesn't update** until you run `npm update` with internet. New DLC
  Pokémon and rule changes won't exist until then.
- **Bulbapedia stays frozen** at your ZIM's build date. This is true regardless of
  offline mode — the system never reads the live wiki — but it's more noticeable
  when you're not refreshing anything else either.

None of this affects correctness for the overwhelming majority of questions.
Mechanics, learnsets, locations, and lore don't change. Only the competitive
metagame layer decays, and it decays gracefully as long as the model reports the
date of the statistics it used.

---

## 6. Pick your path

Answer these four questions. Each points you at the sub-steps you need.

| Question | If no | If yes |
|---|---|---|
| Do you have a Bulbapedia ZIM file? | → Step 2.1 | → Step 2.2 or 2.3 |
| Do you already run Kiwix? | → Step 2.1 / 2.2 | → Step 2.3 |
| Do you already run a chat model server? | → Step 3.1 or 3.2 | → Step 3.3, 3.4 or 3.5 |
| Do you already run Open WebUI? | → Step 4.1 | → Step 4.2 |

Sub-step numbers prefixed with "Step" refer to the deployment steps below, not to
the data-source subsections in section 3.

If you answered *no* to everything, this deployment is nearly self-contained — the
compose file in Step 5 includes Kiwix and Open WebUI, and you only need to supply a
chat model.

---

## 7. Host prerequisites

| Requirement | Notes |
|---|---|
| Docker + Compose v2 | TrueNAS SCALE 24.10+ (Electric Eel) uses Docker natively |
| Disk | Scales with your ZIM — see the table below. Budget **6× the ZIM size.** |
| 8 GB RAM minimum | 16 GB comfortable. The embedding stage holds all chunk text in memory. |
| Internet on first run | Downloads packages, embedding model, Smogon data |
| A GPU | **Not required by this stack.** Only your chat model needs one, and that can be a hosted API. |

### Sizing, measured from a real run

A 2.4 GB Bulbapedia ZIM produced 436,005 chunks and a ~6 GB database. Everything
scales close to linearly with ZIM size:

| ZIM size | Chunks | Database | Vectors | **Free disk needed** | Runtime |
|---|---|---|---|---|---|
| 1 GB | ~180,000 | ~2.6 GB | ~140 MB | **~6 GB** | ~35–70 min |
| 2.4 GB | ~440,000 | ~6.2 GB | ~340 MB | **~13 GB** | ~55–105 min |
| 4 GB | ~730,000 | ~10.4 GB | ~560 MB | **~22 GB** | ~80–160 min |
| 8 GB | ~1,450,000 | ~20.8 GB | ~1.1 GB | **~43 GB** | ~150–300 min |

The free-disk figure is roughly double the database size, because SQLite's WAL
can grow to match the database during a long write before it gets checkpointed.
The ingest checkpoints between stages to keep this bounded, but the peak still
lands inside the wiki stage.

**`ingest.py` checks this for you.** It measures your ZIM, predicts the
footprint, compares it to free space, and refuses to start rather than dying
half-built. Override with `--skip-space-check` if you think it's wrong.

Runtime is dominated by two stages and depends heavily on your hardware:

| Stage | Cost | Notes |
|---|---|---|
| Showdown dex | 15–20 min | One async learnset call per species; slower than it looks |
| Smogon stats | 1–3 min | Scales with how many format-months you fetched |
| Bulbapedia | ~11 min per GB of ZIM | Single-threaded HTML parsing; no parallelism to exploit |
| Aliases | 1–2 min | |
| Embeddings | Depends on cores | ONNX parallelizes well; capped at 16 threads by default |

Have your host's LAN IP ready — you'll need it several times. `ip addr` or
`hostname -I`. This guide writes it as `<HOST_IP>`.

---

## Step 0 — Quick start

**Most people should stop here and run the wizard.**

`deploy.sh` asks what you already have, tests what needs testing, and writes your
configuration. It replaces Steps 1 through 6 below.

```bash
mkdir -p /mnt/Apps/pokedex && cd /mnt/Apps/pokedex
# place deploy.sh, README, and the source files here
chmod +x deploy.sh
./deploy.sh
```

It will ask you:

- Where to install, and this host's LAN IP
- Which network profile you want — fully offline, manual refresh, or scheduled
- Whether you already run Kiwix (and if so, its URL, which it then tests)
- Your chat model's base URL and model name — **and it runs the tool-calling test
  for you**, refusing to continue quietly if that fails
- Whether you already run Open WebUI
- Which ports to use, checking each for conflicts
- Whether you want `docker-compose.yml` + `.env`, a resolved YAML for the TrueNAS
  paste-in editor, or both

It writes:

| File | What it's for |
|---|---|
| `.env` | Your configuration. Edit this later, not the YAML. |
| `docker-compose.yml` | For `docker compose up -d` |
| `docker-compose.truenas.yml` | Values inline, for TrueNAS "Install via YAML" |
| `fetch-stats.sh` | Polite Smogon downloader, if you chose a refresh profile |

Nothing is deployed and nothing large is downloaded. Review the files, then follow
the numbered next steps it prints. Re-run it any time to regenerate.

### When to read the rest of this document

| You want to... | Go to |
|---|---|
| Understand a question the wizard asked | Steps 2, 3, or 4 — they explain the choices |
| Skip the wizard and configure by hand | Step 5 |
| Build the database | Step 7 |
| Connect Open WebUI | Step 9 |
| Fix something | Troubleshooting |

Steps 1 through 6 are the manual path. They remain fully accurate and are the
reference for anything the wizard doesn't cover — but you don't need them if the
wizard worked.

---

## Step 1 — Directory structure

> Skip if you ran the wizard — it offers to create these for you.

```bash
mkdir -p /mnt/Apps/pokedex/{api,sim,data,ui,stats,pikalytics,zim,sprites/icons,sprites/previews,sprites/animated,openwebui}
```

Adjust `/mnt/Apps` to wherever you keep app data. If you change it, change it
everywhere in the compose file too.

```
/mnt/Apps/pokedex/
├── api/          main.py, ingest.py, requirements.txt      (you place these)
├── sim/          server.js, package.json                   (you place these)
├── data/         pokedex.db                                (created by ingest)
├── ui/           index.html                                (optional)
├── zim/          bulbapedia_*.zim                          (Step 2)
├── pikalytics/   Champions usage JSON from fetch-pikalytics.py  (Step 2e, optional)
├── sprites/
│   ├── icons/    Pokemon HOME icon PNGs                     (Step 2b, optional)
│   ├── previews/ Pokemon HOME preview PNGs                  (Step 2b, optional)
│   └── animated/ Animated sprite GIFs                       (Step 2b, further optional)
└── openwebui/    Open WebUI's data                         (only if using bundled)
```

Copy `main.py`, `ingest.py`, `requirements.txt` into `api/`, and `server.js`,
`package.json` into `sim/`. These are bind-mounted into the containers — you can
edit them on the host and just restart the app. No image rebuilds, ever.

---

## Step 2 — The Bulbapedia ZIM

> The wizard asks these questions for you. Read this section if you're unsure how
> to answer one of them, or if you're configuring by hand.

A ZIM is a compressed offline snapshot of a website. This one is your encyclopedia.

### 3.1 — You have no ZIM file

Check the Kiwix library first:

- Browse https://library.kiwix.org and search for "Bulbapedia"
- Or the raw file listing at https://download.kiwix.org/zim/

**Caveat:** the publicly distributed Bulbapedia ZIM has historically lagged behind
the live wiki, sometimes considerably. It's perfectly usable — mechanics and older
content don't change — but it may not have the newest games or episodes. Check the
date in the filename before committing to it.

If it's too old for you, build your own with
[`mwoffliner`](https://github.com/openzim/mwoffliner). That's a multi-hour scrape
and a separate project; do it later if the packaged ZIM disappoints.

Put the file in `/mnt/Apps/pokedex/zim/`, then continue to 2.2.

### 3.2 — You have a ZIM file but don't run Kiwix

**Keep** the `kiwix` service in the compose file (Step 5). It will serve your ZIM at
`http://<HOST_IP>:8993`.

Set in the compose file:

```yaml
    command: ["bulbapedia_en_all_maxi.zim"]      # your exact filename
```

```yaml
      KIWIX_URL: http://kiwix:8080               # container name — same project
      KIWIX_BOOK: bulbapedia_en_all_maxi         # filename WITHOUT .zim
      ZIM_PATH: /zim/bulbapedia_en_all_maxi.zim
```

`KIWIX_URL` uses the container name here because Kiwix is in the same compose
project. That's the easy case.

### 3.3 — You already run Kiwix

**Delete** the `kiwix` service from the compose file.

Find where your Kiwix app stores its ZIMs:

```bash
docker inspect <your-kiwix-container> --format '{{json .Mounts}}' | python3 -m json.tool
```

Then in the compose file:

- Change `pokedex-api`'s ZIM volume to point at **your** Kiwix directory
- Set `KIWIX_URL` to your Kiwix host IP and port — **not** a container name, since
  it's a different compose project on a different Docker network

```yaml
    volumes:
      - /mnt/Apps/kiwix/library:/zim:ro          # your Kiwix ZIM directory
```

```yaml
      KIWIX_URL: http://<HOST_IP>:8080           # your existing Kiwix
      KIWIX_BOOK: bulbapedia_en_all_maxi
      ZIM_PATH: /zim/bulbapedia_en_all_maxi.zim
```

Verify Kiwix answers before moving on:

```bash
curl "http://<HOST_IP>:8080/search?books.name=bulbapedia_en_all_maxi&pattern=pikachu&format=xml" | head -20
```

If that returns results, `KIWIX_URL` and `KIWIX_BOOK` are correct.

---

## Step 2b — Pokemon HOME sprites (optional)

Official artwork and icons for every species, including shiny variants and known
alternate forms (Mega Evolutions, regional forms, Gigantamax). Entirely optional —
without this, species images fall back to whatever Bulbapedia's own article
artwork the ZIM extraction found, which is present but visually inconsistent
(different crops, sizes, and file formats depending on what that specific wiki
article happened to embed).

### Where the files come from

This isn't fetched or built by anything in this project — you need an existing
archive of Pokemon HOME's own sprite assets, organized into two folders:

- **icons** — small sprite images (the ones Pokemon HOME itself uses for team
  lists, box view, etc.)
- **previews** — larger official artwork, one per Pokemon/form

Both folders must use Pokemon HOME's own internal filename scheme:

```
poke_icon_0006_000_mf_n_00000000_f_n.png     (icons folder)
poke_capture_0006_000_mf_n_00000000_f_n.png  (previews folder)
```

The fields, left to right: National Dex number (4 digits) — form index, `000` for
the base form, `001`+ for a species' own alternate forms in order — gender/costume
flag (`mf`/`md`/`fd`/`mo`/`uk`/`fo`) — category (`n` normal, `g` Gigantamax) — an
8-digit costume slot (usually all zeros; non-zero for species with many cosmetic
variants, like Alcremie's flavors) — a literal `f` — shiny flag (`n` normal, `r`
shiny). The ingest stage (2c below) parses this directly; if your archive uses a
different naming convention, it won't match anything and every file will be
skipped harmlessly rather than crash.

### Placing the files

```bash
# icons and previews are two SEPARATE folders of PNGs, matching the naming
# convention above
cp /path/to/your/icons/*.png     /mnt/Apps/pokedex/sprites/icons/
cp /path/to/your/previews/*.png  /mnt/Apps/pokedex/sprites/previews/
```

If either folder came from a Windows machine, delete any `Thumbs.db` file first —
harmless either way (the parser just skips anything that doesn't match the naming
pattern), but no reason to carry it along:

```bash
find /mnt/Apps/pokedex/sprites -iname "Thumbs.db" -delete
```

### Animated sprites (further optional)

A third, independently-sourced folder of animated GIFs, using a different and
much simpler naming convention than the icons/previews above:

```
0006_charizard_base_normal.gif
0006_charizard_mega-x_shiny.gif
0026_raichu_alola_normal.gif
0003_venusaur_female_shiny.gif
```

`{dex:04d}_{species-slug}_{form-slug}_{variant}.gif` — National Dex number,
species name as a lowercase hyphenated slug, a form slug (`base` for the
default form, otherwise a Mega/regional/etc. form name — `mega`, `mega-x`,
`mega-y`, `alola`, and so on), and `normal`/`shiny`.

One naming quirk worth understanding rather than working around: `female` is
**not** a form slug the way `mega`/`alola` are, even though it looks like one.
A hyphenated form slug (Mega, Alola, Galar...) names a genuinely different
battle entity with its own stats/typing/ability. `female` names a purely
cosmetic difference — same species, same stats, just a different in-game
sprite for female individuals (Venusaur, Butterfree, and a handful of others
have a visibly different female appearance with zero mechanical difference).
The ingest stage treats these as two different things internally; you don't
need to do anything differently when placing the files; it's mentioned here
so a `female` file showing up where you might expect a form name doesn't look
like a mistake.

This source has its own coverage gap: no Gigantamax animations exist in it at
all. That's a property of the source archive, not something ingest can fix —
species with only a Gigantamax form (or only a Gigantamax-and-base pairing)
simply won't have an animated option, and the frontend's static/animated
toggle correctly doesn't appear for them.

```bash
cp /path/to/your/animated/*.gif /mnt/Apps/pokedex/sprites/animated/
```

### Compose file additions

Add to `pokedex-api`'s `environment` and `volumes` (Step 5):

```yaml
    environment:
      HOME_ICONS_DIR: /sprites/icons
      HOME_PREVIEWS_DIR: /sprites/previews
      # Animated sprites (optional, independent of the above — see "Animated
      # sprites" just above). Omit this line entirely if you don't have one.
      HOME_ANIMATED_DIR: /sprites/animated
      # The public URL these get served at — this container mounts them as
      # static routes at /sprites/icons, /sprites/previews, and (if configured)
      # /sprites/animated (see main.py), and this MUST match those mount paths
      # exactly, or the URLs written into the database at ingest time will
      # point at a route that doesn't exist.
      HOME_SPRITES_URL: http://<HOST_IP>:8990/sprites     # <<< your host IP
    volumes:
      - /mnt/Apps/pokedex/sprites/icons:/sprites/icons:ro
      - /mnt/Apps/pokedex/sprites/previews:/sprites/previews:ro
      - /mnt/Apps/pokedex/sprites/animated:/sprites/animated:ro
```

### Running the ingest for it

A dedicated, fast stage — separate from the full dex rebuild, and safe to
re-run any time you update the sprite archive:

```bash
sudo docker exec pokedex-api /app/.venv/bin/python /app/ingest.py --sprites-only
```

This does two things: parses every filename in all three folders (skipping
anything that doesn't match, reporting the count so you can sanity-check
coverage), and resolves each species' alternate forms against `pokedex-sim`'s
own dex data to label them correctly where possible (e.g. recognizing that a
given Mega Evolution's sprite is specifically "Charizard-Mega-X" and not just
"some Charizard variant"). That labeling isn't always resolvable — cosmetic-only
variants in particular sometimes have real images but no confirmed form name —
the summary output reports how many were confidently labeled versus not, and
for the animated folder specifically, how many filenames parsed but named a
form that couldn't be matched to anything in the current dex data (this
happens for content newer than what `pokedex-sim` knows about — see Step 2d).

Verify it worked:

```bash
curl -s -X POST http://<HOST_IP>:8990/lookup -H 'Content-Type: application/json' \
  -d '{"name":"Charizard"}' | python3 -m json.tool | grep -A2 sprite
```

Want real URLs for `sprite_icon_url`/`sprite_preview_url`, not `null`. If you
configured `HOME_ANIMATED_DIR`, the response also carries `animated_url` (and
`sprite_shiny_animated_url` for the shiny variant) — `null` is the correct,
expected answer for anything with no animated file, not a sign something's
broken.

### How forms get their names — and how to fix one

A static sprite's filename says which form it is only as a number:

```
poke_capture_0019_000_md_n_00000000_f_n.png
             │dex│form│gen│ │costume│  │shiny (n normal / r shiny)
```

`form` is Pokémon HOME's own form number. `gen` is a gender code: `mf` (same for
both sexes), `md` / `fd` (the male / female look of a species whose sexes differ),
`mo` / `fo` (male-only / female-only species), `uk` (genderless).

**The labelling rule.** Form N gets the Nth name in the species' merged
`otherFormes` list from the simulator's dex (position N is labelled form_index
N+1). That is a guess about HOME's numbering, and it holds for most species. It
fails where HOME numbers differently from Showdown — confirmed for Greninja
(HOME has one Ash/Bond form, Showdown two names), Floette, Zygarde, Magearna and
Tatsugiri, and Pikachu, where HOME numbers the caps 001–007 and 009 while the simulator lists the Cosplay forms first, so the rule labelled the caps as Cosplay Pikachu; and Minior, where HOME's form 0 is the Meteor Form (the in-game default) but the simulator's plain "Minior" is the Core, so the same picture must serve both `Minior` and `Minior-Meteor` — and it can't name anything the simulator doesn't know about (the Legends Z-A Megas). A form the rule can't name stays *unlabeled*, never guessed. **A wrong label is harder to spot than a missing one**: it never appears in the list of Pokémon without an image, so if a form's name looks odd, check it with `sprite_dump.py`.

**Which row an animation joins.** An animated filename has no gender. The ingest
files it under the gender of the static row that already exists for that form,
preferring `mf`, `md`, `mo`, `fo`, `uk`, then `fd` last. So Rattata's base
animation lands on its male (`md`) row and its `female` animation on the `fd` row,
and a female-only species like Nidorina gets its animation on the `fo` row. Every
lookup picks "the" sprite by that same order: the male look is the default of a
species whose sexes differ; the female look is a variant.

**Z-A Megas.** These have no `otherFormes` entry, so a Mega's row is *created* from
its animated file (named from the species table), and its static picture is
merged in afterwards. The merge is deliberately cautious: it only pairs a Mega
with a static sprite when the number of candidate *forms* equals the number of
Mega names, and it never guesses between siblings. Dimorphic Megas that ship a
male and a female file (Staraptor) count as one form and keep both. Everything
ambiguous is resolved by hand, below.

**Fixing one by hand.** Confirmed answers live in a handful of small tables in `ingest.py`. Only add an entry after looking at the images;
several first guesses in the sprite-naming plan were wrong.

| Table | What it says | Current entries |
|---|---|---|
| `CONFIRMED_FORM_INDEX_OVERRIDES` | This `(dex, form_index)` static sprite is this named form. Merges into the animated row of that name, or labels the sprite in place if there isn't one. | Zygarde 004 Complete / 005 Mega; Greninja 001 Mega; Magearna 002 Mega / 003 Original Color Mega; Floette 005 Mega / 006 Eternal; Tatsugiri 003 / 004 / 005 Curly / Droopy / Stretchy Mega; Minior 007–013 Red / Orange / Yellow / Green / Blue / Indigo / Violet |
| `HOME_FORME_ORDER_OVERRIDES` | HOME's own form order for a species, replacing the simulator's. Position N is HOME form N+1; `None` marks a number HOME skips. Both the still labels and the animation matching follow it. | Pikachu: 001 Original, 002 Hoenn, 003 Sinnoh, 004 Unova, 005 Kalos, 006 Alola, 007 Partner, (008 skipped), 009 World |
| `HOME_FORM_REMAP` | HOME's form 0 isn't the look the dex should default to, so renumber its forms before anything is labeled. | None at present (Minior was tried and reverted: its form 0, the Meteor Form, is the in-game default) |
| `SHARED_SHINY_STILL` | These forms have no shiny still of their own and share another named form's. | Minior's colours share the one "Shiny Core" (Red's shiny) |
| `BASE_FORM_ALSO_NAMED` | The default form is also a named form the simulator treats separately; its rows are copied under that name. | Minior: form 0 (Meteor Form) is also `Minior-Meteor` |
| `CONFIRMED_SAME_FORM` | Two Showdown names are one HOME form; the second gets a copy of the first's sprite. | Greninja-Ash / Greninja-Bond |
| `ZA_MEGA_FORM_VARIANTS` | One parsed Mega has several forms sharing its stats; each gets its own named row. | Tatsugiri-Mega → Droopy and Stretchy |
| `ANIMATED_SLUG_SYNONYMS` | An animated slug names a form the species table has under a plainer name. | Tatsugiri `mega-curly` → the plain Mega; Pikachu `<cap>-cap` → the cap's own name |

The workflow:

```bash
docker cp sprite_dump.py pokedex-api:/tmp/sprite_dump.py
docker exec pokedex-api /app/.venv/bin/python /tmp/sprite_dump.py 978      # the dex number
```

`sprite_dump.py` prints the simulator's forme order, every `home_sprites` row, the
static and animated files on disk, and any files carrying a non-zero costume code.
Open the images to decide which file is which, add one line to the right table,
then `--za-megas-only` (if a species row is involved) followed by `--sprites-only`.
That order matters: the sprite stage reads the Mega species rows.

Things learned the hard way, worth knowing before you go looking:

- A Z-A Mega's static sprite ships as a **preview with no icon**. A leftover icon
  under the same number usually belongs to an older form and will be inherited by
  the Mega row, so rename it.
- DLC Megas have **no shiny** static or animated sprites.
- If a form number on disk collides with an existing form, the sprite-naming tool
  (per its own notes) keeps one file and lists the other in a duplicates report.
  Check there first when a form seems to be missing.
- The static row key ignores the **costume code**, so two files that differ only by costume would share a row and one would silently win. No species has been found to do this (Pikachu's caps, the suspected case, turned out to be ordinary form numbers); `sprite_dump.py` flags any that do.

---

## Step 2c — Legends Z-A / Champions Mega Evolutions (optional)

New Mega Evolutions introduced in Pokemon Legends: Z-A (including its Mega
Dimension DLC, shared with Pokemon Champions) — around 45 new Megas that
don't exist in `@pkmn/dex`/`pokedex-sim` at all yet, since Showdown's own
data only covers Gens 1-9. This stage parses them directly out of Bulbapedia,
which already has real, structured tables for both the base stats and the
species/type/ability/Mega Stone details — no external fetch needed beyond the
ZIM you already have.

### Requirements

`stage_wiki` must have already run (`--wiki-only` or a full build) — this
stage reads from `wiki_chunks`, it doesn't touch the ZIM directly.

### Running it

```bash
sudo docker exec pokedex-api /app/.venv/bin/python /app/ingest.py --za-megas-only
```

Fast — a couple of seconds, since it's pure SQL/regex work against
already-ingested text, no external fetching. Deletes and re-inserts its own
batch each run, so it's safe to re-run any time (for instance, after a wiki
re-ingest picks up content that wasn't there before — see the note below).

The log reports how many parsed successfully versus were skipped, and why:
a species whose Bulbapedia page only has its own regular stat block (the
Mega hasn't had its stats published there yet) is skipped rather than
inserted with wrong data, same for a page that can't be found at all under
the expected title.

### Species with more than one Mega

Bulbapedia's table gives a species a second row with no dex number when it has a
second Mega, and the same shape is used for something that isn't a Mega at all,
so the parser tells them apart by the row's first cell:

- **Empty first cell** (Raichu's Mega Y): a real second Mega. It gets its own row.
  Its stats are the *last* stat block on the species page and the first Mega's
  are the one before it — Raichu's page has five blocks, and reading "the last"
  for both would have given Mega X Mega Y's stats.
- **A name in the first cell** (Meowstic Female): a gender-specific ability note,
  skipped.

One parsed Mega can also have several forms with identical stats. Tatsugiri's
Curly, Droopy and Stretchy Megas all share 68/65/90/135/125/92 (every source
checked agrees), so the stage writes the parsed row as `Tatsugiri-Mega` and copies
it to `Tatsugiri-Droopy-Mega` and `Tatsugiri-Stretchy-Mega`, each findable by
name. New entries go in `ZA_MEGA_FORM_VARIANTS`.

Each Mega gets its own self-alias, so it's findable without re-running the alias
stage, and is tagged with its base species' newest generation. That matters for
species removed from later games: Zygarde has no Gen 9 data, so its Mega is
tagged Gen 8 — `/lookup` matches the generation exactly, and one tagged 9 would
be unreachable alongside Zygarde's own forms.

### Real limitation, not a bug: these don't work with the battle-simulator tools

These new Mega rows live entirely in this project's own `species` table,
sourced from Bulbapedia — `pokedex-sim`'s real `@pkmn/sim` battle engine has
no knowledge of them at all, since nothing here touches its own data.
`lookup`/`query_dex` work fine; `calc_damage`, `validate_team`, `review_team`,
and `compare_teams` will all fail on any of them, since the simulator's own
dex has no such species. `/lookup`'s response carries a `sim_note` field for
exactly these species, saying so plainly — the system prompt tells the model
to surface it upfront rather than let one of those tools fail with a generic
error.

### A note on data freshness

This stage is only as good as what's already in `wiki_chunks`. If Bulbapedia's
own article for a species gets updated with new content (a newly-published
Mega stat chart, for instance) and you haven't re-run `--wiki-only` since,
this stage will see the old, incomplete version even though the ZIM file
itself is unchanged — the ZIM parses correctly every time it's read fresh,
but `wiki_chunks` is a stored snapshot from whenever `stage_wiki` last ran,
not the ZIM itself. If a species you'd expect to work is being skipped,
re-running `--wiki-only` before `--za-megas-only` is worth trying before
assuming it's a real gap in the source.

---

## Step 2d — Per-generation game locations (optional)

Where each Pokémon can be found in the main-series games, by generation and game,
behind the `get_locations` tool. It's parsed from the **raw HTML** of each
Bulbapedia species page, not from `wiki_chunks`: the flattened chunk text loses
the table structure (which games share a location) that this needs.

```bash
sudo docker exec pokedex-api /app/.venv/bin/python /app/ingest.py --locations-only
```

About two minutes. It needs the `species` table (for matching article titles and
for each species' debut generation, so nothing is recorded for a generation
before a species existed), deletes and re-inserts its own rows, and is safe to
re-run. The summary reports rows per generation and lists any article that
matched no species — anime characters and category pages are expected there.
Side games, events and promotions are not included.

Note that `get_locations` (like `lookup`) takes a `gen`, and a species that isn't
in that generation's data is reported as not found: Rattata is in Gens 1–7 only.

---

## Step 2e — Champions usage from Pikalytics (optional)

Usage, win rates and builds for the Pokémon Champions ladder. Read section 3.3b
first — the endpoints involved are undocumented and the data has real limits.

Like the Smogon statistics, fetching and ingesting are separate: the ingest never
touches the network. Run the fetcher anywhere with internet and Python 3 (it uses
only the standard library), pointing at the folder the container mounts:

```bash
python3 fetch-pikalytics.py --out /mnt/Apps/pokedex/pikalytics
```

Per format it reads the data month from Pikalytics' own page (never guessed), then
makes one request for the whole roster — each entry already carries that Pokémon's
full record — and falls back to a per-Pokémon request only for an entry that
arrives without one. Files are written atomically as `<format>-<cutoff>-<YYYY-MM>.json`.
It skips a format it already has for the current month, and treats an empty answer
as a failure that writes nothing (this API answers `200 []` for a wrong key), so a
good file is never overwritten by a bad fetch.

| Option | |
|---|---|
| `--formats a,b` | Format codes. Default: the two verified ones. |
| `--cutoff 1760` | Rating cutoff(s). 1760 is the value verified to return data. |
| `--delay 2` | Seconds between requests. |
| `--force` | Refetch even if the month's file exists. |
| `--probe` | Test which format codes respond, and write nothing. |

Then load it (run `--aliases-only` first if you haven't, so names resolve to species
ids):

```bash
sudo docker exec pokedex-api /app/.venv/bin/python /app/ingest.py --pikalytics-only
```

The container needs `PIKALYTICS_DIR: /pikalytics` and the mount
`/mnt/Apps/pokedex/pikalytics:/pikalytics:ro` (the wizard adds both; see Step 5).
The log reports entries per format, any duplicate spellings collapsed (the roster
lists Sirfetch'd three times), names that match no species row (cosmetic forms
mostly; kept anyway), and a sanity line showing that item shares sum to about
100% while move percentages don't.

Verify:

```bash
curl -s -X POST http://<HOST_IP>:8990/usage_stats -H 'Content-Type: application/json' \
  -d '{"format":"gen9championsvgc2026regmc","species":"Rillaboom"}' | python3 -m json.tool | head -30
curl -s http://<HOST_IP>:8990/formats | python3 -m json.tool | grep -B2 -A6 Pikalytics
```

Want `source: "Pikalytics"`, a `win_rate_pct`, and a `month` that is the format's
own (not necessarily the Smogon month). It runs monthly at most; there is nothing
to gain from fetching more often.

---

## Step 2f — Extra species and visual-only forms (optional)

Two small curated tables in `ingest.py`, applied by
`ingest.py --extra-forms-only` (run it after `--aliases-only`; it's also part of a
full build).

**`EXTRA_SPECIES`** adds species the simulator's dex dump lacks but that have their
own data. Currently Floette-Eternal (AZ's Floette): 74/65/67/125/128/92, Fairy,
Flower Veil, verified against several independent sources. It is added for every
generation Floette exists in.

**`VISUAL_ONLY_FORMES`** lists forms that differ from their base species only in
appearance — Pikachu's caps, Pichu-Spiky-eared, Antique Sinistea and Polteageist,
Artisan Poltchageist, Masterpiece Sinistcha, and Minior's seven core colours, Red included (the Meteor Form is not one: it has different stats, and it is the dex default). A visual form:

- is **clickable**: `/lookup` on it succeeds even when no species row exists;
- answers with the **base species' data** — stats, typing, abilities, learnset,
  evolution — while keeping its own name and its own picture if it has one;
- says so in the response: `visual_only: true`, `visual_form_of`, and a short
  `visual_form_note`;
- borrows the base species' still, with `image_from_base: true`, when it has none of its own — keeping an animation of its own if it has one — and never as a stand-in for a shiny;
- is listed, flagged `visual_only`, in its base species' `alternate_forms`, and
  tagged `visual_form_of` in `/dex_index` and `/query_dex`.

The list is curated on purpose. Identical stats, typing and abilities are necessary
but not sufficient: the Cosplay Pikachus match Pikachu's stats but each has a
unique move, and Rockruff-Dusk is ability-related, not visual. The ingest log
therefore prints the forms whose data *is* identical to their base as candidates to
review. To add one, confirm it really is only cosmetic, add `"Form-Name": "Base"`,
run `--extra-forms-only`, and check `/lookup` on it.

Sprites for these come from the same pipeline as everything else. Where a form has a still of its own (all of Pikachu's caps do) it is used; where it doesn't, the base picture is shown instead.

---

> The wizard asks for your base URL and model name, lists what your server reports,
> and runs the tool-calling test in 3.6 automatically. Read this section to
> understand what it's asking, or to fix a failing test.

You need an OpenAI-compatible `/v1/chat/completions` endpoint that **supports tool
calling**. This stack does not provide one.

### The thing that matters

Your model server must **parse tool calls out of the model's output and return them
as structured JSON**. A model that "supports function calling" in principle will
still fail if the server wasn't launched with the right parser.

The failure mode is quiet: the model answers your Pokémon question from its own
memory, never calls a tool, and sounds perfectly confident. Section 3.6 catches this
in thirty seconds.

### 3.1 — Nothing yet, and you have a GPU

Easiest self-hosted option is llama.cpp's server, which handles tool calling through
the model's Jinja chat template:

```bash
docker run -d --name llamacpp --gpus all -p 8080:8080 \
  -v /path/to/models:/models \
  ghcr.io/ggml-org/llama.cpp:server-cuda \
  -m /models/your-model.gguf \
  --host 0.0.0.0 --port 8080 \
  --jinja \
  -ngl 99
```

**`--jinja` is mandatory.** Without it, tool calls are not parsed.

Pick a model known to be good at tool calling — Qwen 2.5/3 Instruct, Hermes, or
Llama 3.1/3.3 Instruct. Model size matters less than tool-call reliability here,
since the model isn't supplying the knowledge.

```yaml
      OPENAI_BASE_URL: http://<HOST_IP>:8080/v1
      OPENAI_MODEL: your-model.gguf
```

### 3.2 — Nothing yet, and you have no GPU

Use a hosted API. Anything OpenAI-compatible works:

```yaml
      OPENAI_BASE_URL: https://api.openai.com/v1
      OPENAI_MODEL: gpt-4o-mini
      OPENAI_API_KEY: sk-...
```

**Caveat:** every question sends your query plus tool results to a third party. Fine
for personal use; think about it before sharing the deployment. Your ZIM and
database stay local either way — only the conversation text leaves.

### 3.3 — You already run vLLM

vLLM needs two flags at launch, or it will not emit `tool_calls`:

```
--enable-auto-tool-choice
--tool-call-parser hermes
```

Parser depends on the model family:

| Model family | `--tool-call-parser` |
|---|---|
| Qwen 2.5 / Qwen 3 | `hermes` |
| Llama 3.1 / 3.2 / 3.3 | `llama3_json` |
| Mistral | `mistral` |
| Hermes / NousResearch | `hermes` |

Find your exact model name:

```bash
curl http://<HOST_IP>:8000/v1/models
```

If you launched with `--served-model-name`, use that value — not the HuggingFace
repo path.

```yaml
      OPENAI_BASE_URL: http://<HOST_IP>:8000/v1
      OPENAI_MODEL: <name from /v1/models>
```

### 3.4 — You already run lemonade-server

Two things to confirm.

**Find the base URL path.** lemonade has used both `/api/v1` and `/v1` depending on
version:

```bash
curl http://<HOST_IP>:13305/api/v1/models
curl http://<HOST_IP>:13305/v1/models
```

Whichever returns a model list is yours.

**Confirm `--jinja` is set.** lemonade runs llama.cpp underneath, which needs that
flag for tool calling. Add it to the model profile's arguments in the lemonade GUI
and restart the container.

```yaml
      OPENAI_BASE_URL: http://<HOST_IP>:13305/api/v1
      OPENAI_MODEL: <your profile name>
```

### 3.5 — You already run Ollama

```yaml
      OPENAI_BASE_URL: http://<HOST_IP>:11434/v1
      OPENAI_MODEL: qwen2.5:14b
```

Ollama handles tool calling natively for models whose templates declare it. Not
every model in the library does — run the test in 3.6 before assuming.

### 3.6 — Everyone: verify tool calling

Do not skip this. Substitute your base URL and model name:

```bash
curl http://<HOST_IP>:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "YOUR_MODEL_NAME",
    "messages": [{"role":"user","content":"What is the weather in Tokyo?"}],
    "tools": [{
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get current weather for a city",
        "parameters": {
          "type": "object",
          "properties": {"city": {"type":"string"}},
          "required": ["city"]
        }
      }
    }]
  }'
```

**Pass** — the response contains a `tool_calls` array and `finish_reason` is
`"tool_calls"`:

```json
"message": {
  "role": "assistant",
  "content": null,
  "tool_calls": [{
    "type": "function",
    "function": {"name": "get_weather", "arguments": "{\"city\": \"Tokyo\"}"}
  }]
},
"finish_reason": "tool_calls"
```

**Fail** — the response is prose about weather, or contains raw `<tool_call>` tags
inside `content`. The model tried; the server didn't parse it. Go back to your
launch flags.

Nothing downstream works until this passes.

---

## Step 4 — A chat front end

> The wizard asks which of these you want.

### 4.1 — You don't have Open WebUI

**Keep** the `open-webui` service in the compose file. It'll be at
`http://<HOST_IP>:8994`. First visit creates an admin account.

Because it's in the same compose project, it can reach `pokedex-api` by container
name — one less IP to get wrong.

### 4.2 — You already have Open WebUI

**Delete** the `open-webui` service from the compose file. You'll register the tool
server manually in Step 9. Requires Open WebUI **v0.6 or later** for OpenAPI tool
server support — check the version in the UI footer.

### 4.3 — You want neither

Use the bundled single-page UI (`pokedex-ui`, Step 10). Less capable than Open WebUI
— no conversation history, no model switching — but it's one HTML file you can
restyle freely.

---

## Step 5 — The compose file (manual alternative)

> **You do not need this section if you ran `deploy.sh`** — it generated your
> compose file already. This is the hand-editing path, kept for people who'd rather
> not run a script, or who need to customize something the wizard doesn't ask about.

Every service is marked REQUIRED or OPTIONAL. Delete what you don't need, then fix
the values marked `<<<`.

```yaml
# =============================================================================
# Pokedex AI
#
# REQUIRED services: pokedex-sim, pokedex-api
# OPTIONAL services: kiwix, open-webui, pokedex-ui
#
# Delete any OPTIONAL service you already run elsewhere, then update the
# matching URL in pokedex-api's environment to point at your existing instance.
#
# RULE OF THUMB FOR URLS:
#   Service is IN this file    -> use the container name  (http://kiwix:8080)
#   Service is somewhere else  -> use the host IP + port  (http://192.168.1.50:8080)
# Container names only resolve within the same compose project.
# =============================================================================

services:

  # ===========================================================================
  # REQUIRED
  # Node sidecar. Wraps @pkmn/sim (team legality) and @smogon/calc (damage).
  # These packages are JavaScript-only, which is why this container exists.
  # Also serves /dump, which the ingest reads to build the dex tables.
  # The model never calls this directly - pokedex-api proxies it.
  # ===========================================================================
  pokedex-sim:
    image: node:22-slim
    container_name: pokedex-sim
    working_dir: /app
    volumes:
      - /mnt/Apps/pokedex/sim:/app          # <<< server.js + package.json here
      - /mnt/Apps/pokedex/data:/data
    environment:
      PORT: "8991"
    command: >
      sh -c "if [ ! -f node_modules/.installed ]; then
               npm install --no-audit --no-fund &&
               touch node_modules/.installed;
             fi;
             exec node server.js"
    ports:
      - "8991:8991"                          # only needed for manual curl tests
    restart: unless-stopped

  # ===========================================================================
  # REQUIRED
  # The brain. Owns the SQLite database, retrieval, and every tool endpoint.
  # This is the URL you register in Open WebUI.
  # ===========================================================================
  pokedex-api:
    image: python:3.12-slim
    container_name: pokedex-api
    working_dir: /app
    environment:
      # --- storage ---------------------------------------------------------
      DB_PATH: /data/pokedex.db

      # --- offline operation (section 3) -----------------------------------
      OFFLINE: "1"                   # never fetch anything at query time
      STATS_DIR: /stats              # read Smogon JSON from disk, not the web
      PIKALYTICS_DIR: /pikalytics    # Champions usage JSON from fetch-pikalytics.py (Step 2e, optional)
      HF_HOME: /data/models          # embedding model cached on the volume
      HF_HUB_OFFLINE: "1"            # override to 0 for the FIRST ingest only
      TRANSFORMERS_OFFLINE: "1"

      # --- Bulbapedia ------------------------------------------------------
      # ZIM_PATH is the file as seen INSIDE the container (under /zim).
      ZIM_PATH: /zim/bulbapedia_en_all_maxi.zim        # <<< your filename

      # KIWIX_URL: bundled kiwix service -> http://kiwix:8080
      #            your own kiwix        -> http://<HOST_IP>:<port>
      KIWIX_URL: http://kiwix:8080                     # <<<
      KIWIX_BOOK: bulbapedia_en_all_maxi               # <<< filename, no .zim

      # --- Pokemon HOME sprites (Step 2b, optional) -------------------------
      # Species images fall back to Bulbapedia's article artwork without
      # these — delete lines you're not using.
      HOME_ICONS_DIR: /sprites/icons
      HOME_PREVIEWS_DIR: /sprites/previews
      # HOME_ANIMATED_DIR: /sprites/animated             # <<< further optional,
                                                           #     independent of
                                                           #     the two above
      HOME_SPRITES_URL: http://<HOST_IP>:8990/sprites  # <<< must match this
                                                         #     container's own
                                                         #     host:port + the
                                                         #     /sprites mounts
                                                         #     below, exactly

      # --- chat model (Step 3) ---------------------------------------------
      OPENAI_BASE_URL: http://192.168.1.50:8000/v1     # <<< YOUR model endpoint
      OPENAI_MODEL: qwen2.5-14b-instruct               # <<< exact name from /v1/models
      # OPENAI_API_KEY: sk-...                         # only for hosted APIs

      # --- internal --------------------------------------------------------
      SIM_URL: http://pokedex-sim:8991       # same project, container name is fine
      EMBED_MODEL: BAAI/bge-small-en-v1.5    # runs on CPU, no GPU needed
    volumes:
      - /mnt/Apps/pokedex/api:/app           # <<< main.py, ingest.py, requirements.txt
      - /mnt/Apps/pokedex/data:/data         #     database lives here
      - /mnt/Apps/pokedex/zim:/zim:ro        # <<< IF USING THE BUNDLED KIWIX
      # - /mnt/Apps/kiwix/library:/zim:ro    # <<< IF USING YOUR OWN KIWIX (swap these)
      - /mnt/Apps/pokedex/stats:/stats:ro    #     pre-downloaded Smogon JSON (3.4)
      - /mnt/Apps/pokedex/pikalytics:/pikalytics:ro   # Champions usage JSON (Step 2e, optional)
      # --- Step 2b, optional: delete lines you're not using ---
      - /mnt/Apps/pokedex/sprites/icons:/sprites/icons:ro
      - /mnt/Apps/pokedex/sprites/previews:/sprites/previews:ro
      # - /mnt/Apps/pokedex/sprites/animated:/sprites/animated:ro   # <<< further optional
    command: >
      sh -c "if [ ! -f /app/.venv/.installed ]; then
               python -m venv /app/.venv &&
               /app/.venv/bin/pip install -r requirements.txt &&
               touch /app/.venv/.installed;
             fi;
             exec /app/.venv/bin/uvicorn main:app --host 0.0.0.0 --port 8990"
    ports:
      - "8990:8990"
    depends_on:
      - pokedex-sim
    restart: unless-stopped

  # ===========================================================================
  # OPTIONAL - DELETE THIS SERVICE IF YOU ALREADY RUN KIWIX
  # Serves the Bulbapedia ZIM for full-text search and citation links.
  # If you delete it: set KIWIX_URL above to your existing instance, and point
  # pokedex-api's /zim mount at your existing ZIM directory.
  # ===========================================================================
  kiwix:
    image: ghcr.io/kiwix/kiwix-serve:latest
    container_name: pokedex-kiwix
    volumes:
      - /mnt/Apps/pokedex/zim:/data:ro       # <<< directory holding the .zim
    command: ["bulbapedia_en_all_maxi.zim"]  # <<< exact filename, WITH .zim
    ports:
      - "8993:8080"                          # browse at http://<HOST_IP>:8993
    restart: unless-stopped

  # ===========================================================================
  # OPTIONAL - DELETE THIS SERVICE IF YOU ALREADY RUN OPEN WEBUI
  # The chat interface. Requires v0.6+ for OpenAPI tool servers.
  # First visit creates the admin account.
  # ===========================================================================
  open-webui:
    image: ghcr.io/open-webui/open-webui:main
    container_name: pokedex-openwebui
    volumes:
      - /mnt/Apps/pokedex/openwebui:/app/backend/data
    environment:
      # Same endpoint you set as OPENAI_BASE_URL above.
      OPENAI_API_BASE_URL: http://192.168.1.50:8000/v1   # <<< YOUR model endpoint
      OPENAI_API_KEY: none                               # <<< real key if hosted
      WEBUI_AUTH: "True"                                 # leave this on
      # --- telemetry off (section 3.5) ---
      ANONYMIZED_TELEMETRY: "False"
      DO_NOT_TRACK: "true"
      SCARF_NO_ANALYTICS: "true"
      ENABLE_OLLAMA_API: "False"                         # unless you use Ollama
    ports:
      - "8994:8080"                          # UI at http://<HOST_IP>:8994
    depends_on:
      - pokedex-api
    restart: unless-stopped

  # ===========================================================================
  # OPTIONAL - DELETE UNLESS YOU WANT THE STANDALONE PAGE
  # Single-file chat UI. Use instead of Open WebUI, or alongside it.
  # If you already run nginx, delete this and drop index.html in your webroot.
  # ===========================================================================
  pokedex-ui:
    image: nginx:alpine
    container_name: pokedex-ui
    volumes:
      - /mnt/Apps/pokedex/ui:/usr/share/nginx/html:ro   # <<< index.html here
    ports:
      - "8992:80"
    depends_on:
      - pokedex-api
    restart: unless-stopped
```

### Port summary

| Port | Service | Needed? |
|---|---|---|
| 8990 | `pokedex-api` | Yes — Open WebUI reads this |
| 8991 | `pokedex-sim` | Only for manual testing; safe to remove |
| 8992 | `pokedex-ui` | Only if using the standalone page |
| 8993 | `kiwix` | Only if using bundled Kiwix |
| 8994 | `open-webui` | Only if using bundled Open WebUI |

Change any of these if they collide with something you run. Only 8990 is referenced
elsewhere in this guide.

---

## Step 6 — Deploy

### 6.1 — TrueNAS SCALE

1. **Apps → Discover Apps → Custom App**
2. Choose **Install via YAML**
3. Paste `docker-compose.truenas.yml` (from the wizard) or your hand-edited file

   TrueNAS cannot read a `.env` file, which is why the wizard produces a second
   YAML with every value written inline. Use that one here.
4. Name it `pokedex`
5. Install

Watch the logs at **Apps → pokedex → Workloads**. First start takes 2–5 minutes
while `npm install` and `pip install` run. Later restarts take seconds, because
`node_modules` and `.venv` persist on the bind mounts.

**Note on TrueNAS and `build:`** — TrueNAS custom apps can't build from a
Dockerfile. That's why this stack uses stock base images with bind-mounted source
and install-on-start. It also means you edit code on the NAS and restart, with no
rebuild cycle.

### 6.2 — Plain Docker

```bash
cd /mnt/Apps/pokedex
docker compose up -d
docker compose logs -f
```

Compose reads `.env` from the same directory automatically. To change a setting
later, edit `.env` and `docker compose up -d` again — no YAML edits.

### 6.3 — Confirm containers are healthy

```bash
docker ps --filter name=pokedex
curl http://<HOST_IP>:8991/health     # sidecar
curl http://<HOST_IP>:8990/health     # api — will report "no data yet"
```

Both must respond before ingest.

---

## Step 7 — Run the ingest

One-time job that builds the database.

**First run only**, override the offline flags so the embedding model can download:

```bash
docker exec -it -e HF_HUB_OFFLINE=0 -e TRANSFORMERS_OFFLINE=0 \
  pokedex-api /app/.venv/bin/python ingest.py
```

Every run after that is plain:

```bash
docker exec -it pokedex-api /app/.venv/bin/python ingest.py
```

| Stage | Source | Network? | Roughly |
|---|---|---|---|
| 1. Dex tables | `pokedex-sim` `/dump` | No | 2–25 sec, by scope |
| 2. Usage stats | `STATS_DIR`, or smogon.com if unset | Only if `STATS_DIR` unset | 1–3 min |
| 3. Wiki chunks | Your ZIM, via libzim | No | ~11 min per GB of ZIM |
| 4. Alias table | ZIM redirects + Showdown IDs | No | 1–2 min |
| 5. Embeddings | Local model in `/data/models` | First run only | The long pole — see below |

**Stage 3 and stage 5 are the whole cost.** A 2.4 GB ZIM takes about 27 minutes
to chunk, and embedding the resulting 436,000 chunks is the longest single stage.
`ingest.py` prints its own estimate during preflight, before doing any work.

**A warning about embedding thread count.** On a 192-core host with ONNX left
unbounded, stage 5 took **3 hours 44 minutes** — roughly 33 chunks/second while
pinning every core. ONNX's thread pool spin-waits, so oversubscription burns
enormous CPU for little throughput. `EMBED_THREADS` now defaults to
`min(16, cores)` for this reason. If you raise it, measure rather than assume
more is better.

It writes `/data/ingest.log` automatically and publishes progress to `/health`,
so you can close the terminal and check in later:

```bash
tail -f /mnt/Apps/pokedex/data/ingest.log
curl -s http://<HOST_IP>:8990/health | python3 -m json.tool
```

While it runs, `/health` reports `"status": "building"` with the current stage,
a percentage for the embedding stage, and a warning if no progress update has
landed for 15 minutes.

### What healthy output looks like

For a 2.4 GB ZIM, the finished database should show roughly:

| Count | Expected | Why |
|---|---|---|
| `species` | 800–1,000 | Gen 9 only by default, not the full National Dex |
| `moves` | ~685 | Gen 9 legal moves |
| `abilities` | ~310 | |
| `learnsets` | ~1.7 million (all nine generations) | One row per species, generation, move, method, level and source generation. Every generation stores the species' *full* source list, so the same fact repeats once per generation and `source_gen` is what places a source in time (see below). |
| `wiki_chunks` | ~440,000 | ~180,000 per GB of ZIM |
| `wiki_chunks_linked` | **~20%** | See below |
| `aliases` | ~130,000 | ZIM redirects plus dex names |

**`wiki_chunks_linked` being around 20% is correct, not a failure.** Only
Pokémon, move, ability, and item articles have a dex entity to link to. Episode
summaries, character pages, locations, and TCG articles have no counterpart and
stay unlinked — they remain fully searchable, they just can't be filtered by
entity. If this number is near zero after a full run, that *is* a problem: the
alias stage didn't run or the dex tables are empty.

Partial re-runs:

```bash
docker exec pokedex-api /app/.venv/bin/python ingest.py --dex-only
docker exec pokedex-api /app/.venv/bin/python ingest.py --stats-only
docker exec pokedex-api /app/.venv/bin/python ingest.py --wiki-only
docker exec pokedex-api /app/.venv/bin/python ingest.py --sprites-only   # Step 2b
docker exec pokedex-api /app/.venv/bin/python ingest.py --item-categories-only
docker exec pokedex-api /app/.venv/bin/python ingest.py --za-megas-only  # Step 2c
docker exec pokedex-api /app/.venv/bin/python ingest.py --locations-only # Step 2d
```

`--sprites-only` is fast (seconds, not minutes) and safe to re-run any time the
sprite archive changes — it doesn't touch the dex, wiki, or usage tables.

`--item-categories-only` parses two Bulbapedia articles already sitting in
`wiki_chunks` into the `item_categories` table (used by `query_items`) — fast,
and safe to re-run any time. It requires `--dex-only` and `--wiki-only` to have
already been run at least once (it reads from both `items` and `wiki_chunks`);
running it against an empty database just logs that nothing was found rather
than erroring.

`--za-megas-only` parses Bulbapedia's Legends Z-A / Champions Mega Evolution
tables (already sitting in `wiki_chunks`) into new rows in the `species`
table — fast, and safe to re-run any time; it deletes and re-inserts its own
batch each run. Requires `--wiki-only` to have already run. Worth re-running
after any `--wiki-only` re-ingest, since a species that was previously skipped
for missing stats may now resolve if Bulbapedia's article for it changed.

`--locations-only` builds the `species_locations` table behind `get_locations`. Unlike the
other Bulbapedia stages it does **not** read `wiki_chunks`: it reads each species
page's raw HTML straight from the ZIM, because the flattened chunk text loses
the table structure this needs. Each generation is a nested table whose rows are
one or more game-name cells followed by a location cell, so games in the same row
share a location. It takes about 2 minutes, deletes and re-inserts its own rows
each run, and is safe to re-run any time. Requires the `species` table
(`--dex-only`) for matching article titles to species IDs and for each species'
debut generation — rows for a generation before a species existed are dropped.
Only the main-series section is parsed; side games and events are excluded. The
summary reports rows per generation and lists any article titles that matched
no species (anime characters and category pages are expected there). It also
runs as part of a full build, after the alias stage.

**Schema migrations run automatically, every time.** Any run of `ingest.py` —
full or partial — checks the database's actual column set against what the
current code expects and adds anything missing via `ALTER TABLE ADD COLUMN`
before doing any other work. This means upgrading to a newer version of
`ingest.py`/`main.py` that adds a new column never requires deleting and
rebuilding the database from scratch — just copy in the new files and run
`--dex-only` (or whichever partial stage touches the changed table). One
real wrinkle worth knowing: `ALTER TABLE ADD COLUMN` always appends the new
column at the table's *physical* end, regardless of where it's written in the
`CREATE TABLE` text — so any code that writes to that table with a bare
`INSERT ... VALUES (...)` (positional, no column list) will silently write
values into the wrong columns on an already-migrated database, even though the
exact same code works fine on a brand new one. Every `INSERT` in `ingest.py`
uses an explicit column list for exactly this reason — if you add a new
`INSERT` yourself later, do the same.

---

## Step 8 — Verify the backend

### 8.0 — Untested paths, in the order they'll bite you

Three things in this stack could not be verified before you deploy, because they
depend on your specific ZIM, your installed package versions, and a running
container. Test them in this order — each takes a minute and saves you from
discovering a problem 30 minutes into a full ingest.

**1. The libzim entry iterator.** `python-libzim` has moved its entry-by-id
accessor between versions. `ingest.py` probes for whichever method your build
exposes and prints what it found if none match.

```bash
docker exec -it pokedex-api /app/.venv/bin/python ingest.py --wiki-only --limit 100
```

Expect ~100 articles and a few hundred chunks in well under a minute. If it exits
with "exposes no entry-by-id accessor", it prints the available attribute names —
send those along and it's a one-line fix.

**2. The `/dump` endpoint's field names.** The Showdown data layer occasionally
renames fields between `@pkmn` releases.

```bash
curl -s http://<HOST_IP>:8991/dump/9 | head -c 400
```

Expect a `counts` block with non-zero species, moves, abilities, items, and
learnsets. A 500 response includes the stack trace — that names the field that
moved.

**3. Your ZIM's article structure.** mwoffliner's HTML layout has changed over the
years, and an older Bulbapedia ZIM may not use `<div id="mw-content-text">`. The
parser falls back to `<body>`, but the chunks come out worse.

After the `--limit 100` run above:

```bash
docker exec pokedex-api /app/.venv/bin/python -c "
import sqlite3; d=sqlite3.connect('/data/pokedex.db'); d.row_factory=sqlite3.Row
print('chunks:', d.execute('SELECT COUNT(*) FROM wiki_chunks').fetchone()[0])
for r in d.execute('SELECT section_path, length(text) n FROM wiki_chunks LIMIT 8'):
    print(f'  {r[\"n\"]:5}  {r[\"section_path\"][:80]}')
print('domains:', dict(d.execute('SELECT domain, COUNT(*) FROM wiki_chunks GROUP BY domain')))
"
```

Healthy output has section paths with `→` separators showing real heading
hierarchy, chunk lengths mostly between 200 and 2400, and a domain spread across
`general`, `games`, and `anime`. If every chunk is `domain=general` with no `→` in
the path, headings aren't being found and the parser needs a different content
selector for your ZIM.

Only after all three pass should you run the full ingest.

### 8.1 — Endpoint checks

All five should succeed:

```bash
# 1. Health, now with data counts
curl http://<HOST_IP>:8990/health

# 2. OpenAPI spec — this is what Open WebUI reads
curl http://<HOST_IP>:8990/openapi.json | head -40

# 3. Structured lookup
curl -X POST http://<HOST_IP>:8990/lookup \
  -H 'Content-Type: application/json' \
  -d '{"name": "Kingambit"}'

# 4. Wiki retrieval
curl -X POST http://<HOST_IP>:8990/search_wiki \
  -H 'Content-Type: application/json' \
  -d '{"query": "Supreme Overlord ability"}'

# 5. Type chart — generation-aware
curl -X POST http://<HOST_IP>:8990/type_matchup \
  -H 'Content-Type: application/json' \
  -d '{"attacking_type":"Fighting","defending_types":["Dark","Steel"]}'
# expect multiplier 4

# 6. Same question in gen 5, where Steel still resisted Dark
curl -X POST http://<HOST_IP>:8990/type_matchup \
  -H 'Content-Type: application/json' \
  -d '{"species":"Kingambit","gen":9}'

# 7. Curated sets (only if you fetched them)
curl -X POST http://<HOST_IP>:8990/get_sets \
  -H 'Content-Type: application/json' \
  -d '{"species":"Kingambit","format":"gen9ou"}'

# 8. Legality — should REJECT (Gholdengo cannot have Levitate)
curl -X POST http://<HOST_IP>:8991/validate_team \
  -H 'Content-Type: application/json' \
  -d '{"format":"gen9ou","team":"Gholdengo @ Air Balloon\nAbility: Levitate\nTera Type: Flying\nEVs: 252 SpA / 252 Spe\nTimid Nature\n- Make It Rain\n- Shadow Ball\n- Nasty Plot\n- Recover"}'

# 9. HOME sprites (only if you set them up in Step 2b) — sprite_icon_url and
#    sprite_preview_url should be real URLs, not null
curl -s -X POST http://<HOST_IP>:8990/lookup \
  -H 'Content-Type: application/json' \
  -d '{"name":"Charizard"}' | python3 -m json.tool | grep sprite

# 10. common_generation — should return gen 8 here, since Aegislash has no
#     Gen 9 data at all. If this returns 9, something's wrong with generation
#     resolution and every downstream team-build tool will hit the same wall.
curl -s -X POST http://<HOST_IP>:8990/common_generation \
  -H 'Content-Type: application/json' \
  -d '{"species":["Aegislash","Hydreigon"]}'

# 11. query_items — should return exactly Charcoal. Confirms both is_choice
#     (Step 2b-independent, from the dex directly) and item_categories (the
#     Bulbapedia-parsed table) are populated and joined correctly.
curl -s -X POST http://<HOST_IP>:8990/query_items \
  -H 'Content-Type: application/json' \
  -d '{"category":"type_boost","query":"Fire"}'
```

If test 5 returns `"valid": true`, something is wrong — the validator isn't loading
format rules.

Backend is done. Everything after this is front-end configuration.

---

## Step 9 — Connect Open WebUI

### 9.1 — Register the tool server

1. **Settings → Admin → Tool Servers**
   (older versions: **Settings → Connections** or **Settings → Integrations**)
2. Add a server, type **OpenAPI**
3. URL:
   - Bundled Open WebUI (same compose project): `http://pokedex-api:8990`
   - Your own Open WebUI (different project): `http://<HOST_IP>:8990`
4. Spec path: `/openapi.json`
5. ID: `pokedex` — lowercase letters, digits, underscores only. It becomes the tool
   name prefix the model sees.
6. Save, and confirm it reports a successful connection

**Register it as an admin/global tool server, not a user one.** User-level tool
servers are called from your browser, which means the URL must resolve from wherever
you're sitting and CORS has to be configured. Admin-level servers are called
server-side from the Open WebUI container, which just works.

### 9.2 — Create the model preset

**Workspace → Models → Create**

- Base model: the one you configured in Step 3
- Name: `Pokédex`
- Tools: enable `pokedex`
- System prompt: paste the block below
- Save

```
You are a Pokémon expert with access to a Bulbapedia mirror, the Pokémon Showdown
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
- lookup or query_dex for exact data. search_wiki for prose. usage_stats for the
metagame.
- type_matchup for ANY type effectiveness question.
- get_sets before recommending how to build a Pokémon. Quote a real named set
rather than inventing four moves.
- For ANY request to review, rate, critique, or find problems with a team: call
review_team, always, before saying anything about the team. It applies to partial
teams and cores, not just full six. Do not reconstruct team analysis from
individual lookups — you will get speed stats and role coverage wrong. If
validate_team parsed the team, review_team will parse it too.
- compare_teams for head-to-head matchup questions.
- validate_team on every team you propose, before showing it. Never show an
illegal team.
- calc_damage for any damage question. Never estimate.

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
```

Three parts of this earned their place through testing, not theory:

**"Even if you are confident, look it up."** Without it the model answers easy
questions from its weights, and its weights contain a lot of plausible-sounding
Pokémon misinformation.

**The `review_team` instruction.** Without an explicit, emphatic rule the model
reconstructs team analysis from individual lookups and gets speed stats and role
coverage wrong — it quoted base Speed instead of actual Speed, and twice claimed
a generation lacked Volt Switch. A strongly-worded tool description alone was not
enough; this line was.

**The ability-effect rule.** The single most persistent error was describing
abilities from memory while the correct effect text sat unread in the tool
results — Weak Armor as raising Attack, Competitive as affecting the opponent,
an ability the Pokémon does not even have. This line fixed it.

### 9.3 — Test it

Ask these in order. Each exercises a different path:

| Prompt | Should trigger |
|---|---|
| "What's Charizard's hidden ability?" | `lookup` — one call, fast |
| "What level does Garchomp learn Dragon Claw in gen 4?" | `lookup` with `moves_to_check` (needs gen 4 in scope) |
| "Give me Gardevoir's full gen 3 level-up learnset" | `lookup` with `include_learnset` (needs gen 3 in scope) |
| "What's super effective against Steel/Fairy?" | `type_matchup` |
| "Where do I catch Feebas in Emerald?" | `search_wiki` |
| "Which episode did Ash's Charizard finally obey him?" | `search_wiki`, anime-scoped |
| "Give me a standard Kingambit set" | `get_sets` |
| "Is Kingambit better than Bisharp in OU?" | `lookup` ×2 + `usage_stats` |
| "Here's my team, what's wrong with it?" | `review_team` |
| "How does my team do against this one?" | `compare_teams` |
| "Build me a rain team for gen9ou without Pelipper" | multi-step + `validate_team` |

Two that should be **refused or reframed**, and are worth checking:

| Prompt | Correct behaviour |
|---|---|
| "What percent chance does team A have of winning?" | Explains it can't compute a win rate, then describes the matchup qualitatively |
| "What were Alakazam's base stats in Gen 1?" (mini build) | Says the structured data only covers gen 9, or answers from wiki prose and labels it as such |

If question 1 answers instantly with **no visible tool call**, go back to Step 3.6.

---

## Step 10 — The standalone page (optional)

A single self-contained HTML file (plain JavaScript, no framework) that talks to
`pokedex-api`'s `/chat` endpoint, which runs the tool loop server-side and streams
results back over SSE.

- Via the bundled service: `http://<HOST_IP>:8992`
- Via your own nginx: drop `index.html` in your webroot, delete `pokedex-ui` from
  the compose file, and point the page's API base URL at `http://<HOST_IP>:8990`

Use it when you want a purpose-built Pokémon interface. Open WebUI is more capable
as a general chat client; this is simpler, fully skinnable, and has two things
Open WebUI doesn't: rendered species cards and a direct-query Tools tab.

### Chat and species cards

The chat side works like any tool-calling chat interface, with one addition:
whenever a tool call resolves a real species (a direct lookup, a team review, a
dex search), the page renders it as a card — image, types, base-stat bars,
weakness/resistance badges, and, for a team context, the actual configured item,
ability, and moves with type-colored move names.

For anything beyond a single species — a team-build answer especially — cards
split into up to three groups, since not everything a team-build answer mentions
is equally "part of the team":

- **The team itself.** Whatever the most recent `validate_team`/`review_team`/
  `compare_teams` call actually parsed as the team's members — full weight, no
  distinction from a plain single-species card.
- **Same team member, different battle state.** Species like Aegislash that
  automatically switch forms mid-battle (Stance Change, Disguise, and similar
  abilities) aren't a different Pokémon and aren't optional the way a Mega
  Evolution is — so a card for the alternate form gets its own honestly-labeled
  section instead of being folded into the team count or treated as an
  unrelated mention.
- **Also mentioned, not part of the team.** A threat example ("Weavile
  outspeeds your whole team"), a suggested swap-in the model considered but
  didn't ultimately recommend, or a query_dex candidate that got looked at and
  rejected. Shown, but visually muted and under its own label, so it never
  reads as if it were a seventh member of a six-member team.

Ask for a shiny variant by name ("show me shiny Gengar") and the card shows the
shiny sprite specifically, not the regular one with a note that shiny exists.

### The Tools tab

Seven sub-panels, each a direct form over one backend tool — useful when you
want a specific structured answer without going through chat, or when you're
verifying a tool's raw output while developing against this stack.

| Tab | Calls | Fields |
|---|---|---|
| **Dex** | `query_dex` | Types (with all/any mode), tier, ability, min/max BST, generation, sort order, result limit |
| **Types** | `type_matchup` | Attacking type, generation, and either two defending types or a species name instead |
| **Damage** | `calc_damage` | Full attacker and defender builds (name, item, ability, nature, relevant EVs, allies fainted for the attacker), a move, and generation |
| **Team** | `validate_team` / `review_team` | A pasted Showdown team export, format, generation |
| **Compare** | `compare_teams` | Two pasted team exports, format, generation |
| **Sets** | `get_sets` | Species, and an optional format filter |
| **Usage** | `usage_stats` | Format, an optional species (omit for the overall ranking), optional month, result limit |

Each tab's result renders as a small styled panel — the same visual language as
chat's species cards, but scoped to just what that one tool returned.

---

## Environment variable reference

All on `pokedex-api`.

| Variable | Required | What it does | Example |
|---|---|---|---|
| `DB_PATH` | Yes | SQLite file location | `/data/pokedex.db` |
| `ZIM_PATH` | Yes | ZIM file as seen **inside** the container | `/zim/bulbapedia_en_all_maxi.zim` |
| `KIWIX_URL` | Yes | Kiwix base URL. Container name if bundled, host IP if external. | `http://kiwix:8080` |
| `KIWIX_BOOK` | Yes | ZIM book name — filename **without** `.zim` | `bulbapedia_en_all_maxi` |
| `HOME_ICONS_DIR` | No | Pokemon HOME icon PNGs, as seen **inside** the container (Step 2b) | `/sprites/icons` |
| `HOME_PREVIEWS_DIR` | No | Pokemon HOME preview PNGs, as seen **inside** the container (Step 2b) | `/sprites/previews` |
| `HOME_ANIMATED_DIR` | No | Animated sprite GIFs, as seen **inside** the container (Step 2b, "Animated sprites") — independent of the two above, no Gigantamax coverage in this source | `/sprites/animated` |
| `HOME_SPRITES_URL` | No | Public URL prefix for all of the above, written into the database at ingest time. Must exactly match where this container actually serves `/sprites/icons`, `/sprites/previews`, and (if configured) `/sprites/animated`. | `http://192.168.1.50:8990/sprites` |
| `SIM_URL` | Yes | Node sidecar. Always the container name. | `http://pokedex-sim:8991` |
| `OPENAI_BASE_URL` | Yes | Chat model endpoint, **including** `/v1` | `http://192.168.1.50:8000/v1` |
| `OPENAI_MODEL` | Yes | Exact name from `/v1/models` | `qwen2.5-14b-instruct` |
| `OPENAI_API_KEY` | No | Only for hosted APIs | `sk-...` |
| `GENS` | No | Generations to index, comma separated. This is the build scope. | `1,2,3,4,5,6,7,8,9` |
| `SETS_DIR` | No | Curated Smogon sets, from `fetch-sets.sh` | `/sets` |
| `BUILD_SCOPE` | No | Label only — records which tier the wizard set up | `full` |
| `EMBED_MODEL` | No | Embedding model name — must match what `EMBED_URL` actually serves | `BAAI/bge-small-en-v1.5` |
| `EMBED_TIER` | No | `cpu` \| `gpu-small` \| `gpu-big` — sets concurrency and batch defaults together. Auto-detects `cpu`/`gpu-small` from whether `EMBED_URL` is set if omitted. | `gpu-big` |
| `EMBED_URL` | No | GPU/remote embedding endpoint (OpenAI-compatible `/embeddings`). Unset = CPU. | `http://pokedex-embed:8993/v1` |
| `EMBED_API_KEY` | No | Auth for `EMBED_URL`, if the endpoint needs one | — |
| `EMBED_CONCURRENCY` | No | Parallel requests to `EMBED_URL`. Overrides whatever `EMBED_TIER` picked. | `16` |
| `EMBED_BATCH` | No | Texts per request to `EMBED_URL`. Overrides the tier default. | `256` |
| `EMBED_DOC_PREFIX` | No | Prepended to every chunk before indexing — only some models need this | `search_document: ` |
| `EMBED_QUERY_PREFIX` | No | Prepended to every search query — must pair with `EMBED_DOC_PREFIX` on the same model | `search_query: ` |
| `EMBED_THREADS` | No | ONNX threads during ingest — **CPU tier only**. Default min(16, cores). | `16` |
| `EMBED_QUERY_THREADS` | No | ONNX threads per query at serve time — **CPU tier only** | `4` |
| `INGEST_LOG` | No | Where `ingest.py` writes its log | `/data/ingest.log` |
| `OFFLINE` | No | Never fetch anything at query time | `1` |
| `STATS_DIR` | No | Read Smogon JSON from disk instead of the web | `/stats` |
| `PIKALYTICS_DIR` | No | Champions usage JSON written by `fetch-pikalytics.py`, as seen inside the container (Step 2e). Unset or empty just skips that ingest stage. | `/pikalytics` |
| `HF_HOME` | No | Embedding model cache location | `/data/models` |
| `HF_HUB_OFFLINE` | No | Block HuggingFace calls. Set `0` for the first ingest. | `1` |
| `TRANSFORMERS_OFFLINE` | No | Same, for the transformers library | `1` |

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `/lookup` says "X does not exist in generation 9" for a Pokémon that clearly exists (Rattata, Nidorina, Zygarde) | `/lookup` matches the generation exactly and the default is 9; the species isn't in that generation's data. `/dex_index` rows carry no generation, so a grid can't tell. | Pass the species' own `gen` (Rattata 7, Nidorina 8, Zygarde 8), or call `common_generation` first |
| A gendered species (Rattata, Nidorina, Staraptor) has no image, or defaults to its female sprite | Old ingest filed its animation on a separate image-less `mf` row, and lookups picked between the male and female rows arbitrarily | Current `ingest.py` and `main.py`, then `ingest.py --sprites-only` |
| A Z-A Mega has an animation but no picture | Its static sprite couldn't be paired safely (several unlabeled candidates, or no animated file to merge into) | Step 2b, "Fixing one by hand": `sprite_dump.py`, then one line in `CONFIRMED_FORM_INDEX_OVERRIDES` |
| A Z-A Mega's animation is filed under the wrong Mega, or is missing | Its animated slug carries a token the species-table name lacks (`mega-curly` vs `Tatsugiri-Mega`) | Add the slug to `ANIMATED_SLUG_SYNONYMS` |
| `fetch-pikalytics.py` prints "FAILED — nothing written" | The API answers `200 []` for a wrong format code, cutoff or date, and the fetcher treats that as failure | Check the code with `--probe`; try `--cutoff`; the site may have changed |
| `usage_stats` for a Champions format reports no entry for a month | Older versions used one global "latest month" | Current `main.py` defaults to the latest month *for that format* |
| Pikalytics ingest logs names that "match no species row" | Cosmetic forms (Alcremie flavours, Vivillon patterns, Furfrou trims) and the like | Expected; the rows are kept. Add an alias only if it's a real species |
| A cosmetic form isn't clickable, or `/lookup` says it isn't found | It isn't in `VISUAL_ONLY_FORMES`, or `--extra-forms-only` hasn't run | Step 2f |
| An older-generation lookup says a Pokémon learns a move that didn't exist then, or its method counts are huge | Older `main.py` read the learnsets table without `source_gen` (every generation stores the full source list) | Current `main.py`; no re-ingest needed |
| A script that restarts a container and immediately runs a command against it fails with "Connection refused" or "Connection reset by peer" | `docker restart` returns as soon as the restart is *initiated*, not once the process inside has actually finished starting and bound its port — a command that follows immediately can beat it | Add a short sleep, poll the container's `/health` until it responds, or just retry the failed command once the container's had a few seconds |
| Model answers Pokémon questions without calling tools | Tool parsing not enabled on the model server | Step 3.6. vLLM needs `--enable-auto-tool-choice --tool-call-parser`; llama.cpp and lemonade need `--jinja` |
| Raw `<tool_call>` tags appear in the answer | Server received the call but didn't parse it | Wrong `--tool-call-parser` for the model family, or missing chat template |
| Open WebUI: "Connection failed" on the tool server | Registered as a user tool server, or wrong host | Use Admin → Tool Servers, and the host IP rather than `localhost` |
| Open WebUI has no Tool Servers section | Version below 0.6 | Upgrade |
| `search_wiki` returns nothing | Ingest hasn't run, or wrong `KIWIX_BOOK` | `curl .../health` for chunk count; book name must have no `.zim` extension |
| Kiwix container exits immediately | Filename in `command:` doesn't match the file | `ls /mnt/Apps/pokedex/zim/` and copy the name exactly |
| `/dump/9` returns 500 | `@pkmn` package field names shifted between versions | Error response includes the stack; usually a one-line fix in `server.js` |
| Ingest killed partway | Host memory pressure | On TrueNAS, cap `zfs_arc_max`. Re-run — it resumes from the last completed stage |
| Ingest dies when the terminal closes | Started with `docker exec -it` | Use `-d`. It logs to `/data/ingest.log` on its own |
| "ANOTHER INGEST IS ALREADY RUNNING" | A second ingest was started while one was live | Correct behaviour — it refused rather than corrupting the database. `docker top pokedex-api` to see the running one |
| Chunk counts lower than expected, whole letters of the alphabet missing | Two ingests ran concurrently; each stage deletes its own scope, so the second wiped the first's work | Prevented now by a PID lock. Re-run `--wiki-only` once, and confirm with `docker top pokedex-api \| grep -c ingest.py` (want 1) |
| Ingest refuses to start, "NOT ENOUGH DISK" | Preflight estimate exceeds free space | Free space, move `DB_PATH`, or `--skip-space-check` if you disagree |
| `pokedex.db-wal` is enormous | Normal mid-write; it checkpoints between stages | It collapses into the database when writes stop |
| `wiki_chunks_linked` near zero after a full run | Alias stage didn't run, or dex tables empty | `ingest.py --dex-only` then `--aliases-only` |
| `embeddings.count` doesn't match `wiki_chunks` | Ingest still running, or died during stage 5 | Check `/health` → `ingest.stage`; resume with `--embed-only` |
| Embedding stage pins every core on a big host | ONNX defaults to all cores and spin-waits | Capped at 16 by default. On 192 cores unbounded this took 3h44m |
| `UNIQUE constraint failed: moves.id, moves.gen` | A generation's dump contains duplicate move IDs | Fixed — inserts are now idempotent and duplicates are folded with a log line |
| `sqlite3.OperationalError` on `--dex-only` after an update | The learnsets schema gained columns | `DROP TABLE learnsets` then re-run `--dex-only` |
| `sqlite3.ProgrammingError: Error binding parameter N: type 'list' is not supported` during `--dex-only` | A dex field that's usually a single string can come back as an array for some species (`battleOnly` is an array for Zygarde-Complete, reachable from either Zygarde-10% or Zygarde-50%) | Don't assume a field's type from checking a handful of species — normalize defensively at the source (`server.js`) and again on read (`ingest.py`), the way `battle_only`/`required_item` do |
| A value written as a clean `0`/`1` reads back as the *string* `'0'`/`'1'` instead of an int, and truthiness checks on it misbehave | The column kept `TEXT` affinity from an earlier migration attempt, before the declared type was corrected to `INTEGER` — `ALTER TABLE ADD COLUMN` doesn't retroactively fix an already-added column's affinity | Never rely on Python truthiness for a value that might be a string; compare explicitly (`str(val) == "1"`), which is correct regardless of which storage form the column actually has |
| A Mega Evolution, regional form, or Gigantamax's `lookup()` result shows the *base* species' image, not its own | Whatever resolved the sprite queried `home_sprites` by national dex number alone (`form_index=0`), which is correct for a plain species but silently wrong for anything else — a Mega/regional form shares its base's dex number | Use `species_sprite()`, the one shared, forme-aware resolver — check `main.py` for every place a sprite gets looked up and confirm none of them query `home_sprites` directly with their own copy of this logic |
| `get_sets` only returns very old generations' sets for a species with a long history (e.g. gen 1–2 sets for a Pokémon that's had modern competitive sets for years) | Sets were sorted by the raw format string (`ORDER BY format`), which is lexicographic, not numeric — `"gen11v1"` (gen 1's 1v1 format) sorts before `"gen2ou"` because `'1' < '2'` as a character. With the default result limit, a species with many older-gen sets can exhaust it before a single modern set is ever reached | Sort by the generation number parsed out of the format string, not the raw string itself |
| A complex multi-step request (e.g. a team built around 2+ named Pokémon) ends in "response was cut off" or hits a tool-call turn limit | The model's own internal reasoning counts against the same token budget as its final answer — a genuinely demanding question can need a lot of room on both sides of that budget, and enough tool calls to exceed a conservative turn cap | Both are generous by default now (large `max_tokens`, a forced final-synthesis pass if the turn limit is reached) — if it still happens, the request is likely genuinely more complex than either ceiling anticipates |
| Building a team around a specific named Pokémon crashes partway through (`review_team`'s damage/stat engine errors on one member) or produces confusing/incomplete cards | The named Pokémon doesn't exist in the current generation (e.g. Aegislash has no Gen 9 data at all) — tools defaulted to the current gen and discovered the mismatch mid-process | Call `common_generation` first for any team build with 2+ named seed Pokémon, and use its answer for every other tool call in the request |
| Model reviews a team without calling `review_team` | It prefers reconstructing from lookups | The system prompt in Step 9.2 has an explicit rule for this; a tool description alone was not enough |
| `embeddings.count` stale after a re-ingest | Older builds mapped the file once at startup | Fixed — the API remaps on mtime change. Restart if running an older `main.py` |
| `libzim` import error on start | Missing build tools for the wheel | Add `apt-get update && apt-get install -y build-essential &&` to the start of `pokedex-api`'s command |
| First response after restart is slow | Embedding model loading into memory | Normal, ~10 s, once per container start |
| Answers are right but cite nothing | System prompt not applied | Check the **model preset**, not the base model — Open WebUI applies preset prompts only to presets |
| `npm install` runs on every restart | `node_modules` not persisting | Confirm the `sim` bind mount is read-write, not `:ro` |
| Container hangs on start with no logs | `HF_HUB_OFFLINE` set but the model was never downloaded | Run the first ingest with `-e HF_HUB_OFFLINE=0` (Step 7) |
| Ingest fails at the stats stage | `STATS_DIR` set but empty, or no internet and `STATS_DIR` unset | Pre-download per section 3.4, or unset `STATS_DIR` and allow egress once |
| `animated_url`/`sprite_shiny_animated_url` in a response return a real-looking URL, but fetching it 404s even though the file genuinely exists on disk | `main.py` was reading `HOME_ICONS_DIR`/`HOME_PREVIEWS_DIR` and mounting those two as static routes, but never read `HOME_ANIMATED_DIR` or mounted a third route for it — the database had the right URL, the file was really there, there was just no HTTP route registered to serve it | Fixed — confirm `main.py` mounts all three (`/sprites/icons`, `/sprites/previews`, `/sprites/animated`); if only two, it's running an older build |
| Everything works, then breaks after a restart with no internet | `pip` or `npm` re-running on start | Apply the sentinel-file commands from section 3.2 |
| `usage_stats` returns nothing for a format | That format has no file at your chosen rating cutoff | Check what exists at `smogon.com/stats/YYYY-MM/chaos/` — cutoffs vary by format |

---

## Maintenance and upgrading

### What to re-run when

Every data source is independent. Each stage deletes only its own scope before
rewriting, so refreshing one never forces a rebuild of the others.

| What changed | Command | Roughly | Rebuilds embeddings? |
|---|---|---|---|
| New Smogon month | `ingest.py --stats-only` | 2 min | No |
| New Smogon sets (quarterly) | `ingest.py --sets-only` | ~1 min | No |
| DLC, new generation, tier shift | `ingest.py --dex-only` | 15–90 min by scope | No |
| **Changing build scope** | `ingest.py --dex-only` | 15–90 min | No |
| Updated/expanded sprite archive | `ingest.py --sprites-only` | Seconds | No |
| New Bulbapedia ZIM changes item category articles | `ingest.py --item-categories-only` | Seconds | No |
| New Legends Z-A / Champions Mega revealed on Bulbapedia | `ingest.py --za-megas-only` | Seconds | No |
| New Bulbapedia ZIM, or new species added to the dex | `ingest.py --locations-only` | ~2 min | No |
| New Pikalytics month (Champions) | `fetch-pikalytics.py`, then `ingest.py --pikalytics-only` | ~10 s + seconds | No |
| Edited `EXTRA_SPECIES` / `VISUAL_ONLY_FORMES` | `ingest.py --extra-forms-only` | Seconds | No |
| Edited a sprite override table | `ingest.py --za-megas-only` (if a species row is involved), then `--sprites-only` | Seconds | No |
| New Bulbapedia ZIM | `ingest.py --wiki-only` | ~1 hour+ | Yes, all of them |
| Everything | `ingest.py` | Full build | Yes |

Only a ZIM refresh is expensive, and that's a once-or-twice-a-year event. Both
the monthly stats refresh and a scope change leave your wiki chunks and
embeddings untouched.

### Typical rhythm

| Frequency | Action |
|---|---|
| Monthly | `./fetch-stats.sh` then `ingest.py --stats-only` |
| Quarterly | `./fetch-sets.sh` then `ingest.py --sets-only` |
| On a new generation or DLC | `docker exec pokedex-sim npm update`, restart it, then `ingest.py --dex-only` |
| When you find a newer ZIM | Swap the file, update `.env`, then `ingest.py --wiki-only` |
| When you want more generations | Edit `GENS` in `.env`, restart, then `ingest.py --dex-only` |

Why the wiki case costs so much: chunk IDs are reassigned on rebuild, so the
vector file has to be regenerated wholesale. There's no incremental path, and
given how rarely ZIMs are republished, that's the right trade.

The API detects a rewritten embedding file by mtime and remaps it, so no restart
is needed after a re-ingest. It also rejects a half-written pair — if it catches
the ingest between writing the vectors and writing the IDs, it keeps serving the
previous map rather than returning mismatched results.

**Monthly** — refresh Smogon usage stats.

Offline (recommended): download the new month's files to `/mnt/Apps/pokedex/stats/`
using the script in section 3.4, from any machine with internet, then:

```bash
docker exec pokedex-api /app/.venv/bin/python ingest.py --stats-only
```

Online: unset `STATS_DIR` and run the same command — it will fetch from smogon.com
directly.

**When you update the ZIM** — replace the file, update `ZIM_PATH`, `KIWIX_BOOK`, and
Kiwix's `command:` if the filename changed, restart, then:

```bash
docker exec pokedex-api /app/.venv/bin/python ingest.py --wiki-only
```

**When a new generation or DLC ships** — refresh the dex:

```bash
docker exec pokedex-sim npm update        # pull newer @pkmn packages
docker restart pokedex-sim
docker exec pokedex-api /app/.venv/bin/python ingest.py --dex-only
```

**Backup** — back up these:

| Path | Why |
|---|---|
| `data/pokedex.db` | The whole knowledge base. Rebuildable but slow. |
| `data/models/` | Cached embedding model. Without it, a restore needs internet. |
| `api/`, `sim/` | Your source files |
| `stats/` | Pre-downloaded Smogon JSON |

The `.venv` and `node_modules` directories are disposable if you have internet to
rebuild them — but keep them if you're air-gapped, since they can't be recreated
offline. The ZIM is large and re-downloadable; back it up only if bandwidth is
scarce.

---

## Caveats for shared or public deployments

### Licensing and attribution

- **Bulbapedia** content is CC BY-NC-SA 2.5 — attribution, non-commercial,
  share-alike. Personal and community use is fine. Do not build a commercial product
  on it. `search_wiki` responses cite article titles by design, which is what
  attribution looks like in practice.
- **Smogon** usage statistics and analyses are Smogon's. Credit them visibly if you
  publish anything built on this.
- **Pokémon** itself is Nintendo / Creatures / GAME FREAK intellectual property. This
  is a fan tool. Don't sell it, don't imply endorsement.

### Security

- **`pokedex-api` has no authentication.** It's designed to sit on a trusted LAN
  behind Open WebUI. Anyone who can reach port 8990 can query it.
- **Do not expose 8990 to the internet.** If you need remote access, put it behind
  Tailscale, a VPN, or an authenticating reverse proxy. Open WebUI has real auth; the
  tool server does not.
- If you use a hosted model API, the text of every conversation — including tool
  results — goes to that provider. Your ZIM and database stay local.
- Leave `WEBUI_AUTH: "True"` on. Turning it off makes Open WebUI open to anyone on
  the network.

### Redistribution

You can share the compose file, `server.js`, `ingest.py`, and `main.py` freely —
they're your code and configuration. **Don't redistribute the ZIM or the built
`pokedex.db`**, which contain Bulbapedia's content. Point people at
library.kiwix.org and let them build their own database.

---

## File manifest

| File | Goes in | Purpose |
|---|---|---|
| `deploy.sh` | install root | Configuration wizard — start here |
| `.env` | install root | Generated configuration; edit this to change settings |
| `docker-compose.yml` | install root | Generated; for `docker compose up -d` |
| `docker-compose.truenas.yml` | paste into TrueNAS UI | Generated; values inline |
| `fetch-stats.sh` | install root | Generated; polite Smogon usage-stats downloader |
| `fetch-sets.sh` | install root | Generated; curated Smogon sets downloader |
| `NEXT-STEPS.txt` | install root | Generated; step-by-step, formatted for a terminal |
| `NEXT-STEPS.md` | install root | Generated; the same, for an editor |
| `server.js` | `/mnt/Apps/pokedex/sim/` | Node sidecar |
| `package.json` | `/mnt/Apps/pokedex/sim/` | Sidecar dependencies |
| `main.py` | `/mnt/Apps/pokedex/api/` | FastAPI tool server |
| `ingest.py` | `/mnt/Apps/pokedex/api/` | Database builder |
| `requirements.txt` | `/mnt/Apps/pokedex/api/` | Python dependencies |
| `index.html` | `/mnt/Apps/pokedex/ui/` | Standalone page (optional) |
| `*.zim` | `/mnt/Apps/pokedex/zim/` | Bulbapedia snapshot |
| `poke_icon_*.png` | `/mnt/Apps/pokedex/sprites/icons/` | Pokemon HOME icons (Step 2b, optional) |
| `poke_capture_*.png` | `/mnt/Apps/pokedex/sprites/previews/` | Pokemon HOME preview art (Step 2b, optional) |
| `{dex}_{species}_{form}_{variant}.gif` | `/mnt/Apps/pokedex/sprites/animated/` | Animated sprites (Step 2b, further optional, no Gigantamax) |
| `fetch-pikalytics.py` | anywhere with internet | Pikalytics downloader (Step 2e, optional) |
| `<format>-<cutoff>-<YYYY-MM>.json` | `/mnt/Apps/pokedex/pikalytics/` | Champions usage snapshots written by the fetcher |
| `sprite_dump.py` | copy into the container when needed | Read-only diagnostic for the sprite pipeline (Step 2b) |

---

---

### On win probabilities

The system deliberately cannot produce one, and both the tool description and the
system prompt say so explicitly.

Win probability depends on many turns of decision-making — switches, prediction,
sacrifices. None of that is in the data, there is no simulator in the loop, and
there is no corpus of battle outcomes. A model asked "what percent does team A
win" will happily produce a confident number from nothing, which is worse than
refusing.

`compare_teams` gives the model the facts a good player would reason from — who
outspeeds whom, what the shared weaknesses are, which attacks OHKO what — and the
response carries a disclaimer the model sees on every call. Matchup questions get
a qualitative answer built on measured data, not a fabricated statistic.



---

## License Disclaimer

### License & Attribution
DexAI is open-source software released under the **GNU Affero General Public License v3.0 (AGPLv3)**. 

Pursuant to Section 7 of the AGPLv3, any modified or network-hosted versions of DexAI must preserve original copyright notices and author attributions:
* **Project:** DexAI
* **Original Author:** @aex90832
* **Source Code:** `https://github.com/aex90832/DexAI`

### Non-Affiliation Disclaimer
DexAI is an unofficial, non-commercial fan-made project. It is not affiliated with, endorsed, sponsored, or approved by Nintendo, Game Freak, or The Pokémon Company. Pokémon and Pokémon character names are registered trademarks of Nintendo.
