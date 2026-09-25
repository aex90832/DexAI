#!/usr/bin/env bash
# DexAI — Tool Server
# Copyright (C) 2026 @aex90832 (https://github.com/aex90832)
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
#
# Asks what you already run, tests what needs testing, and writes:
#   .env                          your configuration
#   docker-compose.yml            for `docker compose up -d`
#   docker-compose.truenas.yml    values resolved inline, for the TrueNAS UI
#   fetch-stats.sh                polite Smogon downloader (if you chose online)
#
# Nothing is deployed. Review the output before you run it.
#
# Usage:  ./deploy.sh [--output-dir DIR]

set -euo pipefail

OUT_DIR="$(pwd)"
while [ $# -gt 0 ]; do
  case "$1" in
    --output-dir) OUT_DIR="$2"; shift 2 ;;
    -h|--help) sed -n '2,16p' "$0"; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

if [ -t 1 ]; then
  B=$'\033[1m'; DIM=$'\033[2m'; GRN=$'\033[32m'; YEL=$'\033[33m'; RED=$'\033[31m'; N=$'\033[0m'
else
  B=""; DIM=""; GRN=""; YEL=""; RED=""; N=""
fi

hr()   { printf '%s\n' "────────────────────────────────────────────────────────────"; }
head1(){ echo; hr; echo "${B}$*${N}"; hr; }
note() { echo "${DIM}$*${N}"; }
ok()   { echo "${GRN}✓${N} $*"; }
warn() { echo "${YEL}!${N} $*"; }
err()  { echo "${RED}✗${N} $*" >&2; }

# ask VAR "prompt" "default"
ask() {
  local __var="$1" __prompt="$2" __default="${3:-}" __reply
  if [ -n "$__default" ]; then
    read -r -p "$__prompt [$__default]: " __reply || true
    __reply="${__reply:-$__default}"
  else
    while :; do
      read -r -p "$__prompt: " __reply || true
      [ -n "$__reply" ] && break
      echo "  (required)"
    done
  fi
  printf -v "$__var" '%s' "$__reply"
}

# choose VAR "prompt" "opt1" "opt2" ...
choose() {
  local __var="$1"; shift
  local __prompt="$1"; shift
  local __opts=("$@") __i __reply
  echo
  echo "$__prompt"
  for __i in "${!__opts[@]}"; do
    printf '  %d) %s\n' "$((__i+1))" "${__opts[$__i]}"
  done
  while :; do
    read -r -p "Choice [1]: " __reply || true
    __reply="${__reply:-1}"
    if [[ "$__reply" =~ ^[0-9]+$ ]] && [ "$__reply" -ge 1 ] && [ "$__reply" -le "${#__opts[@]}" ]; then
      printf -v "$__var" '%s' "$__reply"
      return 0
    fi
    echo "  (enter 1-${#__opts[@]})"
  done
}

confirm() {
  local __reply
  read -r -p "$1 [y/N]: " __reply || true
  [[ "$__reply" =~ ^[Yy] ]]
}

confirm_yes() {
  local __reply
  read -r -p "$1 [Y/n]: " __reply || true
  [[ ! "$__reply" =~ ^[Nn] ]]
}

port_in_use() {
  (exec 3<>"/dev/tcp/127.0.0.1/$1") 2>/dev/null && { exec 3<&- ; return 0; }
  return 1
}

# A bare number is almost always someone typing a port into the host prompt.
# Catch it here rather than letting it end up in every URL in the config.
valid_host() {
  local h="$1"
  [ -n "$h" ] || return 1
  case "$h" in
    *[!0-9.]*)
      # contains something other than digits and dots -> treat as a hostname
      case "$h" in
        *://*|*/*|*:*) return 1 ;;   # not a bare host
        *) return 0 ;;
      esac ;;
    *)
      # digits and dots only -> must be four dotted octets
      printf '%s' "$h" | grep -Eq '^([0-9]{1,3}\.){3}[0-9]{1,3}$' || return 1
      return 0 ;;
  esac
}

ask_host() {
  local __var="$1" __prompt="$2" __default="$3" __h
  while :; do
    ask __h "$__prompt" "$__default"
    if valid_host "$__h"; then
      printf -v "$__var" '%s' "$__h"
      return 0
    fi
    case "$__h" in
      *[!0-9]*) err "\"$__h\" doesn't look like an address." ;;
      *)        err "\"$__h\" looks like a port number, not an address." ;;
    esac
    note "  Enter the IP or hostname this machine is reachable at on your"
    note "  network — for example 192.168.1.50 or nas.local."
    note "  No http://, no port number, no trailing slash."
    if command -v hostname >/dev/null 2>&1; then
      local __found
      __found="$(hostname -I 2>/dev/null | tr ' ' '\n' | grep -E '^[0-9.]+$' | head -3 | tr '\n' ' ')"
      [ -n "$__found" ] && note "  Addresses detected on this host: $__found"
    fi
  done
}

ask_port() {
  local __var="$1" __prompt="$2" __default="$3" __p
  while :; do
    ask __p "$__prompt" "$__default"
    if port_in_use "$__p"; then
      warn "Port $__p is already in use on this host."
      confirm "  Use it anyway?" && { printf -v "$__var" '%s' "$__p"; return 0; }
    else
      printf -v "$__var" '%s' "$__p"; return 0
    fi
  done
}

json_has() {  # json_has <file> <key>
  if command -v python3 >/dev/null 2>&1; then
    python3 -c "import json,sys
try:
    d=json.load(open(sys.argv[1]))
except Exception:
    sys.exit(1)
sys.exit(0 if sys.argv[2] in json.dumps(d) else 1)" "$1" "$2"
  else
    grep -q "\"$2\"" "$1"
  fi
}

# ---------------------------------------------------------------------------
# preflight
# ---------------------------------------------------------------------------

head1 "Pokedex AI — deployment wizard"

cat <<'INTRO'
This writes configuration files. It does not deploy anything, does not
download anything large, and does not modify existing containers.

You can re-run it any time to regenerate configuration.
INTRO

MISSING=""
for c in curl docker; do
  command -v "$c" >/dev/null 2>&1 || MISSING="$MISSING $c"
done
if [ -n "$MISSING" ]; then
  warn "Not found on PATH:$MISSING"
  note "  The wizard still works, but you'll need these to deploy."
fi

# Does this shell have permission to talk to the Docker socket? On TrueNAS and
# most distro defaults it does not, and every generated command needs sudo.
DOCKER_CMD="docker"
if command -v docker >/dev/null 2>&1; then
  if docker info >/dev/null 2>&1; then
    ok "Docker is reachable without sudo."
  else
    DOCKER_CMD="sudo docker"
    warn "This shell can't reach the Docker socket without elevation."
    note "  All generated commands will be prefixed with sudo."
    note "  To avoid sudo permanently, add yourself to the docker group:"
    note "    sudo usermod -aG docker \$USER    (then log out and back in)"
    note "  Note that docker group membership is equivalent to root access."
  fi
fi

# ---------------------------------------------------------------------------
# 1. install root
# ---------------------------------------------------------------------------

head1 "1 / 8  Where should this live?"
note "All persistent data — database, source, ZIM — goes under one directory."
ask INSTALL_ROOT "Install root" "/mnt/Apps/pokedex"
INSTALL_ROOT="${INSTALL_ROOT%/}"

HOST_IP_GUESS="$(hostname -I 2>/dev/null | awk '{print $1}' || true)"
[ -z "$HOST_IP_GUESS" ] && HOST_IP_GUESS="192.168.1.50" || true
echo
note "The address other machines use to reach this host. Needed whenever a"
note "container talks to something outside this compose project."
note "Examples: 192.168.1.50  or  nas.local"
note "Not a port number, no http://, no trailing slash."
ask_host HOST_IP "This host's LAN IP or hostname" "$HOST_IP_GUESS"

# ---------------------------------------------------------------------------
# 2. network profile
# ---------------------------------------------------------------------------

head1 "2 / 8  Network profile"

cat <<'PROFILES'
How much should this system reach the internet after setup?

  Fully offline      Nothing leaves your network, ever. Usage statistics
                     are pre-downloaded by you, by hand, whenever you feel
                     like it. Maximum privacy, zero external load.

  Manual refresh     Offline by default. A helper script fetches Smogon
                     statistics when YOU run it. Recommended.

  Scheduled refresh  Same helper, run automatically about once a month on a
                     randomized day, with rate limiting and conditional
                     requests. Convenient, still very light on Smogon.

In all three, question answering is 100% local. The only difference is how
monthly competitive statistics get updated.
PROFILES

choose NET_CHOICE "Pick a profile:" \
  "Manual refresh — offline, you run the fetch script (recommended)" \
  "Fully offline — no fetch script at all" \
  "Scheduled refresh — automatic monthly, rate limited"

case "$NET_CHOICE" in
  1) NET_PROFILE="manual";    OFFLINE=1; WANT_CRON=0 ;;
  2) NET_PROFILE="airgapped"; OFFLINE=1; WANT_CRON=0 ;;
  3) NET_PROFILE="scheduled"; OFFLINE=1; WANT_CRON=1 ;;
esac
ok "Profile: $NET_PROFILE"

# The fetch script is generated for EVERY profile. In airgapped mode it is meant
# to be carried to a machine that has internet, not run here — but you still need
# it, otherwise "copy the stats in by hand" is advice with no tool attached.
WANT_FETCH=1

echo
if [ "$NET_PROFILE" = "airgapped" ]; then
  cat <<'AIRGAP'
This host will never contact Smogon. You still get fetch-stats.sh — copy it to
any machine that has internet, run it there, and bring the resulting stats
folder back on a USB stick or over your LAN.

You can also skip statistics entirely. Everything except competitive metagame
questions works without them.
AIRGAP
else
  note "Smogon publishes these for free and asks nothing in return, so the"
  note "fetch script goes out of its way not to be a nuisance:"
  note "  · one request at a time, with a delay between them"
  note "  · conditional requests, so unchanged files aren't re-downloaded"
  note "  · a refusal to run more than once every 28 days"
  note "  · a User-Agent identifying the tool, so they can contact you"
fi

echo
ask STATS_CONTACT "Contact string for the User-Agent (email or URL, or 'none')" "none"
echo
note "Formats: 'all' discovers everything Smogon published that month and takes"
note "one ladder cutoff each — roughly 150-250 files, a few hundred MB. Or list"
note "just the metagames you care about, e.g.  gen9ou gen9vgc2026"
ask STATS_FORMATS "Formats to track" "all"
echo
note "History: 1 = the most recent published month only. More months multiply"
note "the transfer, but past months are downloaded once and never refetched."
ask STATS_MONTHS  "Months of history" "1"

# ---------------------------------------------------------------------------
# 2b. build scope
# ---------------------------------------------------------------------------

head1 "How thorough should the build be?"

cat <<'SCOPES'
This controls how many Pokemon generations get indexed, and whether curated
Smogon sets are included. It does not affect the Bulbapedia wiki, which is
always indexed in full.

Why it matters: base stats, move power, abilities and the type chart all
CHANGED between generations. Steel resisted Dark and Ghost before gen 6.
Fairy did not exist. Alakazam had one Special stat in gen 1. If you only
index gen 9, questions about older games fall back to wiki prose instead of
exact data.

  Full      All nine generations, plus sets. About 25 SECONDS and
            under 200 MB. Every mainline game answerable from
            structured data, including level-up learnsets.

  Standard  Generations 7, 8 and 9, plus sets. About 8 seconds.

  Mini      Generation 9 only. About 2 seconds. Smallest database.

Those are measured, not estimated. The dex stage is cheap enough that there
is little reason to narrow it — pick Full unless you specifically want a
smaller database.

You can change this later and re-run just the dex stage — it does not require
rebuilding the wiki index or the embeddings.
SCOPES

choose SCOPE_CHOICE "Pick a scope:" \
  "Full — all nine generations plus sets (recommended)" \
  "Standard — gens 7-9 plus sets" \
  "Mini — gen 9 only"

case "$SCOPE_CHOICE" in
  1) SCOPE="full";     GENS="1,2,3,4,5,6,7,8,9"; WANT_SETS=1; DEX_SEC=25 ;;
  2) SCOPE="standard"; GENS="7,8,9"; WANT_SETS=1; DEX_SEC=8 ;;
  3) SCOPE="mini";     GENS="9";     WANT_SETS=0; DEX_SEC=2 ;;
esac

ok "Scope: $SCOPE  (generations $GENS)"
echo
note "The dex stage takes about ${DEX_SEC} SECONDS for this scope — it is not a"
note "meaningful cost. Bulbapedia indexing and embeddings dominate the build,"
note "and those are identical at every scope."
[ "$WANT_SETS" = "1" ] && note "Smogon sets add ~8 MB and about a minute." || true

# ---------------------------------------------------------------------------
# 2c. embedding tier
# ---------------------------------------------------------------------------

head1 "How should embeddings be generated?"

cat <<'EMBEDINFO'
This is the slowest single stage of the build — it turns every wiki chunk into
a search vector. It runs on CPU by default, which works everywhere but is
genuinely slow: measured at 28-36 chunks/second regardless of thread count, on
a real ~700k-chunk Bulbapedia index that's 5-7 hours.

A GPU changes this by an order of magnitude, and even a modest one helps —
this does NOT need a card dedicated to it. If you have a GPU sitting mostly
idle, it is almost always worth using here.
EMBEDINFO

choose EMBED_CHOICE "Pick an option:" \
  "CPU only — no GPU needed, works everywhere, slowest" \
  "GPU, consumer card (8-16GB) — a few times faster" \
  "GPU, high-end card (24GB+) — fastest, most headroom" \
  "Not sure — use CPU now, revisit later"

EMBED_TIER=""; EMBED_MODEL="BAAI/bge-small-en-v1.5"
EMBED_URL=""; EMBED_DOC_PREFIX=""; EMBED_QUERY_PREFIX=""
BUNDLE_EMBED=0; EMBED_HOST_PORT=""; EMBED_LONGCTX=0

case "$EMBED_CHOICE" in
  1|4)
    EMBED_TIER="cpu"
    ok "CPU embedding. No extra service, no extra config."
    if [ "$EMBED_CHOICE" = "4" ]; then
      note "You can switch to GPU later — see README 'Embedding tiers' — by"
      note "adding a pokedex-embed service and setting EMBED_URL, then"
      note "re-running: ingest.py --wiki-only  (vectors must be rebuilt, they"
      note "cannot be mixed between CPU and GPU runs of the same model)."
    fi
    ;;
  2|3)
    if [ "$EMBED_CHOICE" = "2" ]; then
      EMBED_TIER="gpu-small"
      note "gpu-small: 6 requests in flight at once. Conservative default for a"
      note "card with less VRAM and fewer compute units — higher concurrency"
      note "just queues on a card this size without helping."
    else
      EMBED_TIER="gpu-big"
      note "gpu-big: 16 requests in flight at once. Measured on a 32GB card:"
      note "roughly 3-4x the throughput of gpu-small on the same workload,"
      note "with the GPU still under 50% utilised — there's room to push"
      note "higher later with EMBED_CONCURRENCY if you want to experiment."
    fi

    echo
    if confirm "Also try the long-context embedding model (nomic-embed-text, 8192 tokens vs 512)?"; then
      EMBED_LONGCTX=1
      EMBED_MODEL="nomic-ai/nomic-embed-text-v1.5"
      EMBED_DOC_PREFIX="search_document: "
      EMBED_QUERY_PREFIX="search_query: "
      note ""
      note "Why this matters: roughly 15-25% of real Bulbapedia chunks (table-"
      note "heavy sections especially) are longer than 512 tokens. The default"
      note "model truncates and retries those — it still works, but costs extra"
      note "round trips and the vector only covers part of the chunk. This"
      note "model's 8192-token window embeds those chunks whole on the first try."
      note "It's a bigger model (137M vs 33M) but still light enough that a"
      note "24GB+ card has room to spare — this is squarely 'use the headroom'"
      note "territory, not a workload that needs a big card to begin with."
      if [ "$EMBED_CHOICE" = "2" ]; then
        warn "You picked a consumer-card tier — the long-context model asks more"
        warn "of a smaller card. Fine to try; drop back to the default model if"
        warn "it struggles."
      fi
    else
      note "Using BAAI/bge-small-en-v1.5 (512-token window, 33M params)."
    fi

    echo
    choose EMBED_HOST_CHOICE "Where does the embedding model run?" \
      "Include a pokedex-embed container in this deployment" \
      "I already run a compatible endpoint (vLLM, etc.) elsewhere"

    if [ "$EMBED_HOST_CHOICE" = "1" ]; then
      BUNDLE_EMBED=1
      ask_port EMBED_HOST_PORT "Port for pokedex-embed" "8993"
      EMBED_URL="http://pokedex-embed:8993/v1"
      note ""
      note "This deployment will run its own embedding server. It needs an"
      note "image that can serve $EMBED_MODEL on your GPU. If you already run"
      note "vLLM elsewhere for a chat model, the same image usually works —"
      note "see the generated compose file's pokedex-embed service, and"
      note "README 'Embedding tiers' for AMD/gfx1201-specific notes if that's"
      note "your hardware (vanilla vLLM images can fail to start on some AMD"
      note "cards; a tuned fork may be needed)."
    else
      ask EMBED_URL "Existing endpoint base URL (OpenAI-compatible /embeddings)" \
        "http://$HOST_IP:8993/v1"
      note ""
      note "Confirm it actually serves $EMBED_MODEL before running the ingest:"
      note "  curl -s $EMBED_URL/embeddings -H 'Content-Type: application/json' \\"
      note "    -d '{\"model\":\"$EMBED_MODEL\",\"input\":[\"test\"]}'"
    fi
    ;;
esac

echo
LONGCTX_LABEL=""
[ "$EMBED_LONGCTX" = "1" ] && LONGCTX_LABEL=" (long-context model)"
ok "Embedding: ${EMBED_TIER:-cpu}${LONGCTX_LABEL}"

# ---------------------------------------------------------------------------
# 3. Bulbapedia / Kiwix
# ---------------------------------------------------------------------------

head1 "3 / 8  Bulbapedia"

choose KIWIX_CHOICE "Do you already run Kiwix?" \
  "No — include a Kiwix container in the deployment" \
  "Yes — point at my existing Kiwix"

if [ "$KIWIX_CHOICE" = "1" ]; then
  BUNDLE_KIWIX=1
  ZIM_HOST_DIR="$INSTALL_ROOT/zim"
  echo
  note "Put your .zim file in: $ZIM_HOST_DIR"
  note "No ZIM yet? Check https://library.kiwix.org — and please use the"
  note "BitTorrent link if one is offered. These files are multi-gigabyte and"
  note "Kiwix runs on donations."
  echo
  ask ZIM_FILENAME "ZIM filename (with .zim)" "bulbapedia_en_all_maxi.zim"
  ask_port KIWIX_PORT "Port to serve Kiwix on" "8993"
  KIWIX_URL="http://kiwix:8080"
else
  BUNDLE_KIWIX=0
  KIWIX_PORT=""
  echo
  ask ZIM_HOST_DIR   "Directory your Kiwix reads ZIMs from" "/mnt/Apps/kiwix/library"
  ask ZIM_FILENAME   "ZIM filename (with .zim)" "bulbapedia_en_all_maxi.zim"
  ask KIWIX_EXISTING "Your Kiwix base URL" "http://$HOST_IP:8080"
  KIWIX_URL="$KIWIX_EXISTING"

  if command -v curl >/dev/null 2>&1; then
    echo
    echo -n "Testing Kiwix search... "
    BOOK="${ZIM_FILENAME%.zim}"
    if curl -fsS --max-time 10 \
        "$KIWIX_URL/search?books.name=$BOOK&pattern=pikachu&format=xml" \
        >/tmp/.pokedex_kiwix_test 2>/dev/null; then
      ok "Kiwix responded."
    else
      warn "No usable response."
      note "  Check the URL and that the book name is '$BOOK' (filename without .zim)."
      confirm "  Continue anyway?" || exit 1
    fi
    rm -f /tmp/.pokedex_kiwix_test
  fi
fi

ZIM_BOOK="${ZIM_FILENAME%.zim}"

# ---------------------------------------------------------------------------
# 4. chat model
# ---------------------------------------------------------------------------

head1 "4 / 8  Chat model"

cat <<'MODELNOTE'
This stack does not provide a model. You need an OpenAI-compatible endpoint
that SUPPORTS TOOL CALLING — that last part is where most setups fail, so
we'll test it before continuing.
MODELNOTE

choose MODEL_CHOICE "What are you using?" \
  "vLLM" \
  "llama.cpp server" \
  "lemonade-server" \
  "Ollama"

case "$MODEL_CHOICE" in
  1) DEFAULT_URL="http://$HOST_IP:8000/v1"
     note "Reminder: vLLM needs --enable-auto-tool-choice --tool-call-parser <parser>" ;;
  2) DEFAULT_URL="http://$HOST_IP:8080/v1"
     note "Reminder: llama.cpp needs --jinja for tool calling" ;;
  3) DEFAULT_URL="http://$HOST_IP:13305/api/v1"
     note "Reminder: lemonade runs llama.cpp underneath and needs --jinja" ;;
  4) DEFAULT_URL="http://$HOST_IP:11434/v1" ;;
esac

echo
ask OPENAI_BASE_URL "Base URL (including /v1)" "$DEFAULT_URL"

if command -v curl >/dev/null 2>&1; then
  echo -n "Listing models... "
  if curl -fsS --max-time 10 "$OPENAI_BASE_URL/models" -o /tmp/.pokedex_models 2>/dev/null; then
    ok "reachable"
    if command -v python3 >/dev/null 2>&1; then
      python3 -c "
import json
try:
    d=json.load(open('/tmp/.pokedex_models'))
    for m in d.get('data',[])[:15]:
        print('    -', m.get('id'))
except Exception:
    pass" || true
    fi
  else
    warn "Could not reach $OPENAI_BASE_URL/models"
    note "  Check the URL. lemonade in particular uses /api/v1 on some versions and /v1 on others."
  fi
  rm -f /tmp/.pokedex_models
fi

echo
ask OPENAI_MODEL "Model name (exactly as the server reports it)"

OPENAI_API_KEY=""
if confirm "Does this endpoint need an API key?"; then
  ask OPENAI_API_KEY "API key"
  warn "The key will be written to $OUT_DIR/.env — chmod 600 it."
fi

# --- the test that matters -------------------------------------------------

echo
head1 "Tool-calling check"
note "Asking your model to call a fake weather function."

TOOLTEST_OK=0
if command -v curl >/dev/null 2>&1; then
  AUTH_HEADER=()
  [ -n "$OPENAI_API_KEY" ] && AUTH_HEADER=(-H "Authorization: Bearer $OPENAI_API_KEY") || true

  cat > /tmp/.pokedex_tooltest_req <<EOF
{
  "model": "$OPENAI_MODEL",
  "messages": [{"role":"user","content":"What is the weather in Tokyo?"}],
  "tools": [{
    "type": "function",
    "function": {
      "name": "get_weather",
      "description": "Get current weather for a city",
      "parameters": {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"]
      }
    }
  }]
}
EOF

  if curl -fsS --max-time 90 "$OPENAI_BASE_URL/chat/completions" \
      -H 'Content-Type: application/json' "${AUTH_HEADER[@]}" \
      -d @/tmp/.pokedex_tooltest_req \
      -o /tmp/.pokedex_tooltest 2>/dev/null; then

    if json_has /tmp/.pokedex_tooltest "tool_calls"; then
      ok "Tool calling works. The model returned a structured tool_calls array."
      TOOLTEST_OK=1
    elif grep -q '<tool_call>' /tmp/.pokedex_tooltest 2>/dev/null; then
      err "The model TRIED to call the tool but the server did not parse it."
      note "  This is a server flag problem, not a model problem:"
      note "    vLLM      --enable-auto-tool-choice --tool-call-parser hermes"
      note "    llama.cpp --jinja"
      note "    lemonade  add --jinja to the model profile"
    else
      err "The model answered in prose instead of calling the tool."
      note "  Either tool parsing is off, or this model does not do tool calling well."
      note "  Try Qwen 2.5/3 Instruct, Hermes, or Llama 3.1+ Instruct."
    fi
  else
    err "Request failed."
  fi
  rm -f /tmp/.pokedex_tooltest /tmp/.pokedex_tooltest_req
else
  warn "curl not available — skipping the test. Run it by hand before deploying."
fi

if [ "$TOOLTEST_OK" != "1" ]; then
  echo
  warn "Without tool calling, the model will answer Pokemon questions from its own"
  warn "memory and sound completely confident while doing it. Nothing else works."
  confirm "Continue and fix it later?" || exit 1
fi

# ---------------------------------------------------------------------------
# 5. front end
# ---------------------------------------------------------------------------

head1 "5 / 8  Chat interface"

choose UI_CHOICE "How do you want to talk to it?" \
  "Include Open WebUI in this deployment" \
  "I already run Open WebUI — I'll register the tool server myself" \
  "Neither — just the standalone HTML page" \
  "Neither — API only, I'll wire up my own client"

BUNDLE_OPENWEBUI=0; BUNDLE_UI=0; OPENWEBUI_PORT=""; UI_PORT=""; OPENWEBUI_URL=""
case "$UI_CHOICE" in
  1) BUNDLE_OPENWEBUI=1; ask_port OPENWEBUI_PORT "Port for Open WebUI" "8994" ;;
  2) echo
     note "pokedex-api doesn't need to know about Open WebUI — the connection goes"
     note "the other way, when you register the tool server. This is only so the"
     note "summary below can print you a working link."
     ask OPENWEBUI_URL "Your Open WebUI URL" "http://$HOST_IP:8080" ;;
  3) BUNDLE_UI=1; ask_port UI_PORT "Port for the standalone page" "8992" ;;
  4) : ;;
esac

if [ "$UI_CHOICE" != "3" ] && [ "$UI_CHOICE" != "4" ]; then
  if confirm "Also include the standalone HTML page?"; then
    BUNDLE_UI=1; ask_port UI_PORT "Port for the standalone page" "8992"
  fi
fi

# ---------------------------------------------------------------------------
# 6. ports
# ---------------------------------------------------------------------------

head1 "6 / 8  Ports"
ask_port API_PORT "pokedex-api (the tool server)" "8990"
echo
note "pokedex-api reaches the sidecar over the internal Docker network, so"
note "publishing this port is not required for the stack to work. But the"
note "verification steps use it to check the sidecar directly, so say yes"
note "unless the port is already taken."
if confirm_yes "Publish pokedex-sim's port to the host?"; then
  ask_port SIM_PORT "pokedex-sim" "8991"
  EXPOSE_SIM=1
else
  SIM_PORT="8991"; EXPOSE_SIM=0
  warn "Not published. Verification will use 'docker exec' instead of curl."
fi

# ---------------------------------------------------------------------------
# 7. output format
# ---------------------------------------------------------------------------

head1 "7 / 8  Output format"

cat <<'OUTNOTE'
  compose + .env    Standard. `docker compose up -d` reads .env automatically.
                    Change settings by editing .env, no YAML edits.

  resolved YAML     Every value written inline, no variable references.
                    TrueNAS "Install via YAML" cannot read a .env file, so
                    this is what you paste there.

  both              Recommended — .env stays your source of truth, and you
                    regenerate the resolved YAML by re-running this script.
OUTNOTE

choose OUT_CHOICE "Which do you want?" \
  "Both" \
  "compose + .env only" \
  "Resolved YAML only (TrueNAS)"

# ---------------------------------------------------------------------------
# 8. write everything
# ---------------------------------------------------------------------------

head1 "8 / 8  Writing files"

mkdir -p "$OUT_DIR"

# --- .env ------------------------------------------------------------------

ENV_FILE="$OUT_DIR/.env"

# Values are quoted so multi-word settings survive `set -a; . ./.env`.
# Docker Compose strips surrounding quotes, so both readers are happy.
ev() { printf '%s="%s"\n' "$1" "$2"; }

{
  echo "# Pokedex AI configuration — generated $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "# Regenerate with ./deploy.sh, or edit by hand and restart."
  echo
  ev INSTALL_ROOT "$INSTALL_ROOT"
  ev HOST_IP      "$HOST_IP"
  ev ZIM_HOST_DIR "$ZIM_HOST_DIR"
  echo
  echo "# --- Bulbapedia ---"
  ev ZIM_FILENAME "$ZIM_FILENAME"
  ev ZIM_BOOK     "$ZIM_BOOK"
  ev KIWIX_URL    "$KIWIX_URL"
  [ -n "$KIWIX_PORT" ] && ev KIWIX_PORT "$KIWIX_PORT" || true
  echo
  echo "# --- chat model ---"
  ev OPENAI_BASE_URL "$OPENAI_BASE_URL"
  ev OPENAI_MODEL    "$OPENAI_MODEL"
  ev OPENAI_API_KEY  "$OPENAI_API_KEY"
  echo
  echo "# --- ports ---"
  ev API_PORT "$API_PORT"
  ev SIM_PORT "$SIM_PORT"
  [ -n "$UI_PORT" ]        && ev UI_PORT        "$UI_PORT" || true
  [ -n "$OPENWEBUI_PORT" ] && ev OPENWEBUI_PORT "$OPENWEBUI_PORT" || true
  echo
  echo "# --- offline behavior ---"
  ev NET_PROFILE          "$NET_PROFILE"
  ev OFFLINE              "$OFFLINE"
  ev STATS_DIR            "/stats"
  ev HF_HOME              "/data/models"
  ev HF_HUB_OFFLINE       "1"
  ev TRANSFORMERS_OFFLINE "1"
  ev GENS                 "$GENS"
  ev SETS_DIR             "/sets"
  ev BUILD_SCOPE          "$SCOPE"
  ev EMBED_MODEL          "$EMBED_MODEL"
  ev EMBED_THREADS        "16"    # bulk ingest; 0 or unset = min(16, cores); CPU tier only
  ev EMBED_TIER           "${EMBED_TIER:-cpu}"
  ev EMBED_URL            "$EMBED_URL"
  ev EMBED_API_KEY        ""
  ev EMBED_DOC_PREFIX     "$EMBED_DOC_PREFIX"
  ev EMBED_QUERY_PREFIX   "$EMBED_QUERY_PREFIX"
  ev EMBED_QUERY_THREADS  "4"     # per-query at serve time
  [ -n "$EMBED_HOST_PORT" ] && ev EMBED_HOST_PORT "$EMBED_HOST_PORT" || true
  if [ "$WANT_FETCH" = "1" ]; then
    echo
    echo "# --- Smogon fetch politeness ---"
    ev STATS_FORMATS           "$STATS_FORMATS"
    ev STATS_MONTHS            "$STATS_MONTHS"
    ev STATS_CUTOFF            "auto"
    ev STATS_CONTACT           "$STATS_CONTACT"
    ev STATS_MIN_INTERVAL_DAYS "28"
    ev STATS_REQUEST_DELAY     "3"
  fi
} > "$ENV_FILE"
chmod 600 "$ENV_FILE"
ok "$ENV_FILE"

# --- compose ---------------------------------------------------------------

COMPOSE_FILE="$OUT_DIR/docker-compose.yml"
{
cat <<'YAML_HEAD'
# Generated by deploy.sh — edit .env rather than this file where possible.

services:

  pokedex-sim:
    image: node:22-slim
    container_name: pokedex-sim
    working_dir: /app
    volumes:
      - ${INSTALL_ROOT}/sim:/app
      - ${INSTALL_ROOT}/data:/data
    environment:
      PORT: "8991"
    command: >
      sh -c "if [ ! -f node_modules/.installed ]; then
               npm install --no-audit --no-fund &&
               touch node_modules/.installed;
             fi;
             exec node server.js"
YAML_HEAD

if [ "$EXPOSE_SIM" = "1" ]; then
cat <<'YAML'
    ports:
      - "${SIM_PORT}:8991"
YAML
fi

cat <<'YAML'
    restart: unless-stopped

  pokedex-api:
    image: python:3.12-slim
    container_name: pokedex-api
    working_dir: /app
    volumes:
      - ${INSTALL_ROOT}/api:/app
      - ${INSTALL_ROOT}/data:/data
      - ${INSTALL_ROOT}/stats:/stats:ro
      - ${ZIM_HOST_DIR}:/zim:ro
    environment:
      DB_PATH: /data/pokedex.db
      ZIM_PATH: /zim/${ZIM_FILENAME}
      KIWIX_URL: ${KIWIX_URL}
      KIWIX_BOOK: ${ZIM_BOOK}
      OPENAI_BASE_URL: ${OPENAI_BASE_URL}
      OPENAI_MODEL: ${OPENAI_MODEL}
      OPENAI_API_KEY: ${OPENAI_API_KEY}
      SIM_URL: http://pokedex-sim:8991
      GENS: ${GENS}                                # generations to index
      SETS_DIR: ${SETS_DIR}                        # curated Smogon sets
      EMBED_MODEL: ${EMBED_MODEL}
      EMBED_THREADS: ${EMBED_THREADS}              # ONNX threads during ingest — CPU tier only
      EMBED_QUERY_THREADS: ${EMBED_QUERY_THREADS}  # ONNX threads per query — CPU tier only
      EMBED_TIER: ${EMBED_TIER}                    # cpu | gpu-small | gpu-big
      EMBED_URL: ${EMBED_URL}                      # set = use the GPU/remote path; unset = CPU
      EMBED_API_KEY: ${EMBED_API_KEY}
      EMBED_DOC_PREFIX: ${EMBED_DOC_PREFIX}         # only some models need this — see README
      EMBED_QUERY_PREFIX: ${EMBED_QUERY_PREFIX}
      OFFLINE: "${OFFLINE}"
      STATS_DIR: ${STATS_DIR}
      HF_HOME: ${HF_HOME}
      HF_HUB_OFFLINE: "${HF_HUB_OFFLINE}"
      TRANSFORMERS_OFFLINE: "${TRANSFORMERS_OFFLINE}"
    command: >
      sh -c "if [ ! -f /app/.venv/.installed ]; then
               python -m venv /app/.venv &&
               /app/.venv/bin/pip install -r requirements.txt &&
               touch /app/.venv/.installed;
             fi;
             exec /app/.venv/bin/uvicorn main:app --host 0.0.0.0 --port 8990"
    ports:
      - "${API_PORT}:8990"
    depends_on:
      - pokedex-sim
    restart: unless-stopped
YAML

if [ "$BUNDLE_KIWIX" = "1" ]; then
cat <<'YAML'

  kiwix:
    image: ghcr.io/kiwix/kiwix-serve:latest
    container_name: pokedex-kiwix
    volumes:
      - ${ZIM_HOST_DIR}:/data:ro
    command: ["${ZIM_FILENAME}"]
    ports:
      - "${KIWIX_PORT}:8080"
    restart: unless-stopped
YAML
fi

if [ "$BUNDLE_EMBED" = "1" ]; then
  EMBED_MAX_LEN=512
  [ "$EMBED_LONGCTX" = "1" ] && EMBED_MAX_LEN=8192
cat <<YAML

  # GPU embedding server. See README "Embedding tiers" before deploying —
  # vanilla vLLM images can fail to start on some AMD cards (KeyError on the
  # gfx arch string); a tuned fork may be required. Test with:
  #   curl \$EMBED_URL/embeddings -d '{"model":"$EMBED_MODEL","input":["test"]}'
  # before running the ingest, not after it fails partway through.
  pokedex-embed:
    image: rocm/vllm:latest   # <<< swap for a known-working image on your GPU if this fails
    container_name: pokedex-embed
    command: >
      $EMBED_MODEL
      --runner pooling
      --port 8993
      --gpu-memory-utilization 0.10
      --max-model-len $EMBED_MAX_LEN$([ "$EMBED_LONGCTX" = "1" ] && echo "
      --trust-remote-code")
    environment:
      HF_HOME: /data/models
      HIP_VISIBLE_DEVICES: '0'   # <<< the GPU index this server should use
    devices:
      - /dev/kfd
      - /dev/dri
    group_add:
      - video
    ipc: host
    security_opt:
      - seccomp:unconfined
    volumes:
      - \${INSTALL_ROOT}/data:/data
    ports:
      - "\${EMBED_HOST_PORT}:8993"
    restart: unless-stopped
YAML
fi

if [ "$BUNDLE_OPENWEBUI" = "1" ]; then
cat <<'YAML'

  open-webui:
    image: ghcr.io/open-webui/open-webui:main
    container_name: pokedex-openwebui
    volumes:
      - ${INSTALL_ROOT}/openwebui:/app/backend/data
    environment:
      OPENAI_API_BASE_URL: ${OPENAI_BASE_URL}
      OPENAI_API_KEY: ${OPENAI_API_KEY}
      WEBUI_AUTH: "True"
      ANONYMIZED_TELEMETRY: "False"
      DO_NOT_TRACK: "true"
      SCARF_NO_ANALYTICS: "true"
      ENABLE_OLLAMA_API: "False"
    ports:
      - "${OPENWEBUI_PORT}:8080"
    depends_on:
      - pokedex-api
    restart: unless-stopped
YAML
fi

if [ "$BUNDLE_UI" = "1" ]; then
cat <<'YAML'

  pokedex-ui:
    image: nginx:alpine
    container_name: pokedex-ui
    volumes:
      - ${INSTALL_ROOT}/ui:/usr/share/nginx/html:ro
    ports:
      - "${UI_PORT}:80"
    depends_on:
      - pokedex-api
    restart: unless-stopped
YAML
fi
} > "$COMPOSE_FILE"

[ "$OUT_CHOICE" = "3" ] || ok "$COMPOSE_FILE"

# --- resolved YAML for TrueNAS --------------------------------------------

if [ "$OUT_CHOICE" = "1" ] || [ "$OUT_CHOICE" = "3" ]; then
  RESOLVED="$OUT_DIR/docker-compose.truenas.yml"
  cp "$COMPOSE_FILE" "$RESOLVED"
  while IFS='=' read -r k v; do
    case "$k" in ''|\#*) continue ;; esac
    v="${v%\"}"; v="${v#\"}"          # strip the quotes ev() added
    esc=$(printf '%s' "$v" | sed -e 's/[\/&]/\\&/g')
    sed -i "s/\${$k}/$esc/g" "$RESOLVED"
  done < "$ENV_FILE"
  sed -i '1s|.*|# Generated by deploy.sh — values resolved inline. Paste into the TrueNAS custom app YAML editor.|' "$RESOLVED"
  ok "$RESOLVED"
  if grep -q '\${' "$RESOLVED"; then
    warn "Some \${...} placeholders were left unresolved in $RESOLVED:"
    grep -o '\${[A-Z_]*}' "$RESOLVED" | sort -u | sed 's/^/    /'
  fi
fi

[ "$OUT_CHOICE" = "3" ] && rm -f "$COMPOSE_FILE" || true

# --- fetch-stats.sh --------------------------------------------------------

if [ "$WANT_FETCH" = "1" ]; then
  FETCH="$OUT_DIR/fetch-stats.sh"
  cat > "$FETCH" <<'FETCHEOF'
#!/usr/bin/env bash
#
# Polite Smogon usage-statistics downloader.
#
# DEFAULTS: every format Smogon published, most recent month only.
#
# PORTABLE ON PURPOSE. It reads .env if one sits next to it, but everything can
# be passed as a flag instead — so you can copy this single file to a laptop,
# run it there, and carry the results back to an air-gapped server.
#
# Smogon publishes these for free and asks nothing in return, so this script
# goes out of its way not to be a nuisance:
#
#   · one request at a time, with a delay between each
#   · one cutoff per format, not all four (see PICKING A CUTOFF below)
#   · conditional requests (If-Modified-Since) so unchanged files aren't refetched
#   · refuses to run more than once every 28 days
#   · no retry storms — two attempts, then it gives up until next time
#   · asks before pulling a large number of files
#   · an identifying User-Agent so Smogon can reach you if something goes wrong
#
# PICKING A CUTOFF
#   Smogon publishes each format at several ladder-rating cutoffs (0, 1500,
#   1630, 1695, 1760, 1825). They describe the same metagame at different skill
#   levels. Downloading all of them multiplies the transfer for very little
#   extra insight, so by default this takes the highest cutoff each format
#   actually offers. Override with --cutoff.
#
# Usage:
#   ./fetch-stats.sh                      # all formats, latest month
#   ./fetch-stats.sh --dry-run            # list what it would fetch, download nothing
#   ./fetch-stats.sh --formats "gen9ou gen9vgc2026"
#   ./fetch-stats.sh --months 6           # more history (multiplies the transfer)
#   ./fetch-stats.sh --cutoff 1500        # pin one cutoff
#   ./fetch-stats.sh --cutoff all         # every cutoff (big)
#   ./fetch-stats.sh --out ./stats        # write somewhere specific
#   ./fetch-stats.sh --yes                # skip the "that's a lot of files" prompt
#   ./fetch-stats.sh --force              # ignore the 28-day interval
#
# Air-gapped workflow:
#   1. copy this file to a machine with internet
#   2. ./fetch-stats.sh --out ./stats
#   3. copy ./stats/*.json to <install root>/stats/ on the server
#   4. docker exec pokedex-api /app/.venv/bin/python ingest.py --stats-only
#
# No script at all? These are plain files:
#   curl -A "PokedexAI-selfhosted/1.0" -o gen9ou-1695-2026-08.json \
#     https://www.smogon.com/stats/2026-08/chaos/gen9ou-1695.json
#   Name them <format>-<cutoff>-<YYYY-MM>.json or ingest.py will skip them.

set -euo pipefail
cd "$(dirname "$0")"

# Parse .env rather than sourcing it. Sourcing executes the file, so a
# hand-edited line like  STATS_FORMATS=gen9ou gen9vgc2026  (no quotes) would try
# to run gen9vgc2026 as a command. This reads it as data instead.
if [ -f .env ]; then
  while IFS='=' read -r _k _v || [ -n "$_k" ]; do
    case "$_k" in ''|\#*|*[!A-Za-z0-9_]*) continue ;; esac
    _v="${_v%\"}"; _v="${_v#\"}"
    _v="${_v%\'}"; _v="${_v#\'}"
    export "$_k=$_v"
  done < .env
fi

OUT_OVERRIDE=""
FORCE=0; DRY=0; ASSUME_YES=0
while [ $# -gt 0 ]; do
  case "$1" in
    --out)     OUT_OVERRIDE="$2"; shift 2 ;;
    --formats) STATS_FORMATS="$2"; shift 2 ;;
    --months)  STATS_MONTHS="$2";  shift 2 ;;
    --cutoff)  STATS_CUTOFF="$2";  shift 2 ;;
    --yes|-y)  ASSUME_YES=1; shift ;;
    --force)   FORCE=1; shift ;;
    --dry-run) DRY=1; shift ;;
    -h|--help) sed -n '2,50p' "$0"; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; echo "Try --help" >&2; exit 1 ;;
  esac
done

if [ -n "$OUT_OVERRIDE" ]; then
  STATS_HOST_DIR="$OUT_OVERRIDE"
elif [ -n "${INSTALL_ROOT:-}" ]; then
  STATS_HOST_DIR="$INSTALL_ROOT/stats"
else
  STATS_HOST_DIR="./stats"
fi

FORMATS="${STATS_FORMATS:-all}"
MONTHS="${STATS_MONTHS:-1}"
CONTACT="${STATS_CONTACT:-none}"
MIN_DAYS="${STATS_MIN_INTERVAL_DAYS:-28}"
DELAY="${STATS_REQUEST_DELAY:-3}"
CUTOFF="${STATS_CUTOFF:-auto}"
BASE="https://www.smogon.com/stats"
STAMP="$STATS_HOST_DIR/.last-fetch"
CONFIRM_ABOVE=60

mkdir -p "$STATS_HOST_DIR"

UA="PokedexAI-selfhosted/1.0"
[ "$CONTACT" != "none" ] && UA="$UA (+$CONTACT)" || true

# --- rate limit ------------------------------------------------------------
if [ -f "$STAMP" ] && [ "$FORCE" != "1" ] && [ "$DRY" != "1" ]; then
  LAST=$(cat "$STAMP" 2>/dev/null || echo 0)
  NOW=$(date +%s)
  ELAPSED_DAYS=$(( (NOW - LAST) / 86400 ))
  if [ "$ELAPSED_DAYS" -lt "$MIN_DAYS" ]; then
    echo "Last fetch was $ELAPSED_DAYS days ago; minimum interval is $MIN_DAYS."
    echo "Smogon publishes monthly — there is nothing new yet."
    echo "Use --force only if you know a file changed."
    exit 0
  fi
fi

month_offset() {   # month_offset N -> YYYY-MM, N months ago
  if date -v-1m >/dev/null 2>&1; then
    date -v-"$1"m +%Y-%m           # BSD / macOS
  else
    date -d "$1 months ago" +%Y-%m # GNU
  fi
}

# --- find the most recent month that actually exists -----------------------
# Smogon publishes a month's statistics early in the FOLLOWING month, so the
# current month is usually not there yet. Walk backwards until one responds.
LATEST=""
for back in 0 1 2 3; do
  M=$(month_offset "$back")
  CODE=$(curl -sS -o /dev/null -w '%{http_code}' -A "$UA" --max-time 30 \
         "$BASE/$M/chaos/" 2>/dev/null || echo "000")
  if [ "$CODE" = "200" ]; then LATEST="$M"; break; fi
done

if [ -z "$LATEST" ]; then
  echo "Could not reach $BASE — no network, or Smogon is down." >&2
  exit 1
fi
echo "Most recent published month: $LATEST"

# --- work out which files to take -----------------------------------------
INDEX_RAW=$(curl -sS -A "$UA" --max-time 90 "$BASE/$LATEST/chaos/" 2>/dev/null \
            | grep -oE '[a-z0-9-]+-[0-9]+\.json' | sort -u || true)

if [ -z "$INDEX_RAW" ]; then
  echo "Could not read the file listing for $LATEST." >&2
  echo "Pass explicit formats instead, e.g. --formats \"gen9ou\"" >&2
  exit 1
fi

TOTAL_PUBLISHED=$(printf '%s\n' "$INDEX_RAW" | wc -l | tr -d ' ')

# Reduce to one cutoff per format unless told otherwise.
pick_cutoffs() {
  awk -F'\n' -v want="$CUTOFF" '
    {
      file=$0; sub(/\.json$/,"",file);
      n=split(file, part, "-");
      cut=part[n];
      fmt="";
      for (i=1; i<n; i++) fmt = fmt (i>1 ? "-" : "") part[i];
      if (want != "auto" && want != "all") { if (cut == want) print fmt "|" cut; next }
      if (want == "all") { print fmt "|" cut; next }
      rank = 1;
      if (cut == 1500) rank = 2;
      else if (cut == 1630) rank = 3;
      else if (cut == 1695) rank = 4;
      else if (cut == 1760) rank = 5;
      else if (cut == 1825) rank = 6;
      if (rank > best[fmt]) { best[fmt] = rank; pick[fmt] = cut }
    }
    END { if (want == "auto") for (f in pick) print f "|" pick[f] }
  ' | sort
}

if [ "$FORMATS" = "all" ]; then
  SELECTED=$(printf '%s\n' "$INDEX_RAW" | pick_cutoffs)
else
  SELECTED=""
  for F in $FORMATS; do
    MATCH=$(printf '%s\n' "$INDEX_RAW" | grep -E "^${F}-[0-9]+\.json$" | pick_cutoffs || true)
    if [ -z "$MATCH" ]; then
      echo "  ? no published file for format '$F' in $LATEST — skipping"
    else
      SELECTED="$SELECTED$MATCH"$'\n'
    fi
  done
  SELECTED=$(printf '%s' "$SELECTED" | grep -v '^$' || true)
fi

if [ -z "$SELECTED" ]; then
  echo "Nothing to fetch." >&2
  exit 1
fi

N_FORMATS=$(printf '%s\n' "$SELECTED" | wc -l | tr -d ' ')
N_FILES=$(( N_FORMATS * MONTHS ))

echo "Formats published this month: $TOTAL_PUBLISHED files across all cutoffs"
echo "Selected:                     $N_FORMATS (cutoff: $CUTOFF)"
echo "Months of history:            $MONTHS"
echo "Files to consider:            $N_FILES"
echo "Destination:                  $STATS_HOST_DIR"
echo "Delay between requests:       ${DELAY}s"
echo "User-Agent:                   $UA"
echo

# --- confirm before a big pull --------------------------------------------
if [ "$N_FILES" -gt "$CONFIRM_ABOVE" ] && [ "$DRY" != "1" ] && [ "$ASSUME_YES" != "1" ]; then
  MINS=$(( (N_FILES * (DELAY + 2)) / 60 ))
  echo "That is a lot of files. At ${DELAY}s apart this takes roughly ${MINS} minutes"
  echo "and may transfer several hundred MB to a few GB."
  echo
  echo "Narrow it with --formats \"gen9ou gen9vgc2026\" if you only care about a"
  echo "few metagames. --dry-run lists what would be fetched."
  echo
  if [ -t 0 ]; then
    printf 'Continue? [y/N]: '
    read -r REPLY || true
    case "$REPLY" in [Yy]*) ;; *) echo "Aborted."; exit 0 ;; esac
  else
    echo "Not a terminal and --yes not given. Aborting."
    exit 0
  fi
  echo
fi

# --- fetch -----------------------------------------------------------------
FETCHED=0; SKIPPED=0; MISSING=0; FIRST=1

for i in $(seq 0 $((MONTHS - 1))); do
  if [ "$i" -eq 0 ]; then
    MONTH="$LATEST"
  else
    # step back from the latest published month, not from today
    Y=${LATEST%-*}; MM=${LATEST#*-}
    IDX=$(( 10#$MM - i ))
    while [ "$IDX" -le 0 ]; do IDX=$(( IDX + 12 )); Y=$(( Y - 1 )); done
    MONTH=$(printf '%04d-%02d' "$Y" "$IDX")
  fi

  while IFS='|' read -r FMT CUT; do
    [ -z "$FMT" ] && continue
    FILE="$STATS_HOST_DIR/${FMT}-${CUT}-${MONTH}.json"
    URL="$BASE/${MONTH}/chaos/${FMT}-${CUT}.json"

    # A completed past month never changes — never refetch it.
    if [ -s "$FILE" ] && [ "$i" -gt 0 ]; then
      SKIPPED=$(( SKIPPED + 1 )); continue
    fi

    if [ "$DRY" = "1" ]; then
      echo "  would fetch $URL"
      continue
    fi

    [ "$FIRST" = "1" ] && FIRST=0 || sleep "$DELAY"

    TIMECOND=()
    [ -s "$FILE" ] && TIMECOND=(-z "$FILE") || true

    HTTP=$(curl -sS -w '%{http_code}' -o "$FILE.tmp" \
            -A "$UA" --max-time 180 --retry 1 --retry-delay 30 \
            "${TIMECOND[@]}" "$URL" 2>/dev/null || echo "000")

    case "$HTTP" in
      200) mv "$FILE.tmp" "$FILE"
           echo "  ✓ ${FMT}-${CUT} ${MONTH}"; FETCHED=$(( FETCHED + 1 )) ;;
      304) rm -f "$FILE.tmp"
           echo "  = ${FMT}-${CUT} ${MONTH} (unchanged)"; SKIPPED=$(( SKIPPED + 1 )) ;;
      404) rm -f "$FILE.tmp"
           echo "  - ${FMT}-${CUT} ${MONTH} (not published)"; MISSING=$(( MISSING + 1 )) ;;
      *)   rm -f "$FILE.tmp"
           echo "  ! ${FMT}-${CUT} ${MONTH} (HTTP $HTTP) — skipping until next run"
           MISSING=$(( MISSING + 1 )) ;;
    esac
  done <<< "$SELECTED"
done

[ "$DRY" = "1" ] && exit 0 || true

date +%s > "$STAMP"

echo
echo "Fetched $FETCHED, skipped $SKIPPED, unavailable $MISSING"
DU=$(du -sh "$STATS_HOST_DIR" 2>/dev/null | cut -f1 || echo "?")
echo "Total in $STATS_HOST_DIR: $DU"
echo
echo "Load it into the database with:"
echo "  docker exec pokedex-api /app/.venv/bin/python ingest.py --stats-only"
FETCHEOF
  chmod +x "$FETCH"
  ok "$FETCH"
fi

# --- fetch-sets.sh ---------------------------------------------------------

if [ "$WANT_SETS" = "1" ]; then
  SETSFETCH="$OUT_DIR/fetch-sets.sh"
  cat > "$SETSFETCH" <<'SETSEOF'
#!/usr/bin/env bash
#
# Download curated Smogon sets from Pokemon Showdown's published mirror.
#
# These are the named sets you see in the Showdown teambuilder — moves, item,
# ability, nature, EVs, Tera type — for every format across all nine
# generations. About 8 MB in total, and they change roughly quarterly.
#
# Unlike usage statistics these are not monthly, so there is no rate-limit
# guard here. Still one request at a time with a delay, because it is a
# community server and there is no hurry.
#
# Usage:
#   ./fetch-sets.sh                 # all formats
#   ./fetch-sets.sh --gen 9         # one generation
#   ./fetch-sets.sh --out ./sets
#   ./fetch-sets.sh --dry-run

set -euo pipefail
cd "$(dirname "$0")"

if [ -f .env ]; then
  while IFS='=' read -r _k _v || [ -n "$_k" ]; do
    case "$_k" in ''|\#*|*[!A-Za-z0-9_]*) continue ;; esac
    _v="${_v%\"}"; _v="${_v#\"}"; _v="${_v%\'}"; _v="${_v#\'}"
    export "$_k=$_v"
  done < .env
fi

BASE="https://play.pokemonshowdown.com/data/sets"
OUT=""; ONLY_GEN=""; DRY=0; DELAY="${STATS_REQUEST_DELAY:-2}"
while [ $# -gt 0 ]; do
  case "$1" in
    --out)     OUT="$2"; shift 2 ;;
    --gen)     ONLY_GEN="$2"; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

if [ -n "$OUT" ]; then SETS_HOST_DIR="$OUT"
elif [ -n "${INSTALL_ROOT:-}" ]; then SETS_HOST_DIR="$INSTALL_ROOT/sets"
else SETS_HOST_DIR="./sets"; fi
mkdir -p "$SETS_HOST_DIR"

UA="PokedexAI-selfhosted/1.0"
[ "${STATS_CONTACT:-none}" != "none" ] && UA="$UA (+$STATS_CONTACT)" || true

echo "Listing $BASE/"
INDEX=$(curl -sS -A "$UA" --max-time 60 "$BASE/" 2>/dev/null \
        | grep -oE 'gen[0-9][a-z0-9]*\.json' | sort -u || true)

if [ -z "$INDEX" ]; then
  echo "Could not read the listing — no network, or the layout changed." >&2
  exit 1
fi

if [ -n "$ONLY_GEN" ]; then
  INDEX=$(printf '%s\n' "$INDEX" | grep -E "^gen${ONLY_GEN}" || true)
fi

N=$(printf '%s\n' "$INDEX" | grep -c . || echo 0)
echo "Files to fetch: $N"
echo "Destination:    $SETS_HOST_DIR"
echo

OK=0; SKIP=0; FAIL=0; FIRST=1
while read -r f; do
  [ -z "$f" ] && continue
  DEST="$SETS_HOST_DIR/$f"
  if [ "$DRY" = "1" ]; then echo "  would fetch $BASE/$f"; continue; fi
  [ "$FIRST" = "1" ] && FIRST=0 || sleep "$DELAY"
  TC=(); [ -s "$DEST" ] && TC=(-z "$DEST") || true
  CODE=$(curl -sS -w '%{http_code}' -o "$DEST.tmp" -A "$UA" \
          --max-time 120 --retry 1 --retry-delay 20 "${TC[@]}" \
          "$BASE/$f" 2>/dev/null || echo "000")
  case "$CODE" in
    200) mv "$DEST.tmp" "$DEST"; echo "  ✓ $f"; OK=$((OK+1)) ;;
    304) rm -f "$DEST.tmp"; echo "  = $f (unchanged)"; SKIP=$((SKIP+1)) ;;
    *)   rm -f "$DEST.tmp"; echo "  ! $f (HTTP $CODE)"; FAIL=$((FAIL+1)) ;;
  esac
done <<< "$INDEX"

[ "$DRY" = "1" ] && exit 0 || true

echo
echo "Fetched $OK, unchanged $SKIP, failed $FAIL"
echo "Total: $(du -sh "$SETS_HOST_DIR" 2>/dev/null | cut -f1 || echo '?')"
echo
echo "Load them with:"
echo "  docker exec pokedex-api /app/.venv/bin/python ingest.py --sets-only"
SETSEOF
  chmod +x "$SETSFETCH"
  ok "$SETSFETCH"
fi

# --- directories -----------------------------------------------------------

echo
if confirm "Create the directory structure under $INSTALL_ROOT now?"; then
  mkdir -p "$INSTALL_ROOT"/{api,sim,data,ui,stats,sets,openwebui}
  [ "$BUNDLE_KIWIX" = "1" ] && mkdir -p "$INSTALL_ROOT/zim" || true
  ok "Directories created."
fi

# ---------------------------------------------------------------------------
# next steps
# ---------------------------------------------------------------------------

# next steps — written to NEXT-STEPS.md and summarised on screen
# ---------------------------------------------------------------------------

STEPS_FILE="$OUT_DIR/NEXT-STEPS.md"

CONTAINER_ZIM="/zim/$ZIM_FILENAME"
SIM_URL="http://pokedex-sim:8991"

# The sidecar is only reachable from the host if its port was published.
# Emit verification commands that actually work either way.
if [ "$EXPOSE_SIM" = "1" ]; then
  SIM_HEALTH_CMD="curl http://$HOST_IP:$SIM_PORT/health"
  SIM_DUMP_CMD="curl http://$HOST_IP:$SIM_PORT/dump/9 | head -c 200"
  SIM_NOTE=""
else
  SIM_HEALTH_CMD="$DOCKER_CMD exec pokedex-sim node -e \"fetch('http://127.0.0.1:8991/health').then(r=>r.text()).then(console.log)\""
  SIM_DUMP_CMD="$DOCKER_CMD exec pokedex-sim node -e \"fetch('http://127.0.0.1:8991/dump/9').then(r=>r.json()).then(d=>console.log(d.counts))\""
  SIM_NOTE="NOTE: pokedex-sim's port is not published, so these run inside the
container. That is fine — pokedex-api reaches it over the internal Docker
network regardless. Publish it later by adding a ports: entry if you want to
curl it from the host."
fi

DEPLOY_CMD="$DOCKER_CMD compose up -d"
[ "$OUT_CHOICE" = "3" ] && DEPLOY_CMD="paste docker-compose.truenas.yml into TrueNAS → Apps → Custom App → Install via YAML" || true

{
cat <<EOF
# Pokedex AI — next steps

Generated $(date -u +%Y-%m-%dT%H:%M:%SZ) for install root \`$INSTALL_ROOT\`.

Work through these in order. Each step says what it needs, what it does, and how
to tell it worked.

---

## 1. Place the source files

These are bind-mounted into the containers, so you edit them on the host and
restart — there is never an image to rebuild.

| File | Goes in |
|---|---|
| \`main.py\`, \`ingest.py\`, \`requirements.txt\` | \`$INSTALL_ROOT/api/\` |
| \`server.js\`, \`package.json\` | \`$INSTALL_ROOT/sim/\` |
EOF
[ "$BUNDLE_UI" = "1" ] && echo "| \`index.html\` | \`$INSTALL_ROOT/ui/\` |" || true

cat <<EOF

**Confirm:**

\`\`\`bash
ls -l $INSTALL_ROOT/api/main.py $INSTALL_ROOT/api/ingest.py \\
      $INSTALL_ROOT/api/requirements.txt $INSTALL_ROOT/sim/server.js \\
      $INSTALL_ROOT/sim/package.json
\`\`\`

All five must exist before you deploy.

---

## 2. Put the Bulbapedia ZIM in place

**What this file is.** A ZIM is a single compressed archive containing an entire
website offline — every article, with images. You need the **Bulbapedia** one.
It is not a database dump, not a torrent of ROMs, and not something this script
can generate. It is one file, typically 3–8 GB, named something like
\`bulbapedia_en_all_maxi_2024-01.zim\`.

**Where to get it.** Browse <https://library.kiwix.org> and search for
Bulbapedia, or the raw listing at <https://download.kiwix.org/zim/>. Use the
BitTorrent link if one is offered — these are large files and Kiwix runs on
donations.

**Heads up:** the published Bulbapedia ZIM has historically lagged the live
wiki. Mechanics and older content are unaffected; very recent games or episodes
may be missing. Check the date in the filename.

**Where it must go.**

\`\`\`
$ZIM_HOST_DIR/$ZIM_FILENAME
\`\`\`

**Confirm:**

\`\`\`bash
ls -lh $ZIM_HOST_DIR/$ZIM_FILENAME
\`\`\`

If your file has a **different name**, you must update it in two places or
nothing will find it:

- \`.env\` → \`ZIM_FILENAME\` and \`ZIM_BOOK\` (the same name without \`.zim\`)
EOF
[ "$BUNDLE_KIWIX" = "1" ] && echo "- the \`kiwix\` service's \`command:\` in your compose file" || true
cat <<EOF

Easiest option: re-run \`./deploy.sh\` and give it the real filename.

---

## 3. Competitive statistics (optional)

EOF

if [ "$NET_PROFILE" = "airgapped" ]; then
cat <<EOF
You chose the fully-offline profile, so this host will never contact Smogon.

**What these files are.** Smogon publishes monthly usage statistics as JSON —
which Pokémon get used, with what moves, items and spreads, and what checks
them. They are the only data in this system that lives on the internet rather
than on your disk.

**Skipping is fine.** Everything except current-metagame questions works
without them.

**To get them anyway**, on a machine that has internet:

\`\`\`bash
# copy fetch-stats.sh there, then:
./fetch-stats.sh --out ./stats --dry-run    # see the list first
./fetch-stats.sh --out ./stats              # all formats, latest month
\`\`\`

Then copy the resulting \`*.json\` files into \`$INSTALL_ROOT/stats/\`.

**Or download them directly** — they are plain files:

\`\`\`bash
EOF
if [ "$STATS_FORMATS" = "all" ]; then
  echo "# browse https://www.smogon.com/stats/<YYYY-MM>/chaos/ for the full list,"
  echo "# then for each one you want:"
  echo "curl -A 'PokedexAI-selfhosted/1.0' \\"
  echo "  -o gen9ou-1695-$(date +%Y-%m).json \\"
  echo "  https://www.smogon.com/stats/$(date +%Y-%m)/chaos/gen9ou-1695.json"
else
  for F in $STATS_FORMATS; do
    echo "curl -A 'PokedexAI-selfhosted/1.0' \\"
    echo "  -o ${F}-1695-$(date +%Y-%m).json \\"
    echo "  https://www.smogon.com/stats/$(date +%Y-%m)/chaos/${F}-1695.json"
  done
fi
cat <<EOF
\`\`\`

The filename **must** be \`<format>-<cutoff>-<YYYY-MM>.json\` or the ingest
skips it. \`1695\` is the ladder rating cutoff Smogon uses for OU tiering; some
formats only publish at \`0\` or \`1500\`. Browse what exists at
<https://www.smogon.com/stats/>.
EOF
else
cat <<EOF
**What these files are.** Smogon publishes monthly usage statistics as JSON —
which Pokémon get used, with what moves, items and spreads, and what checks
them. Needs internet; everything else in this system is local.

\`\`\`bash
./fetch-stats.sh --dry-run    # see what it would fetch
./fetch-stats.sh              # actually fetch
\`\`\`

Files land in \`$INSTALL_ROOT/stats/\`. The script refuses to run more than once
every 28 days, because Smogon publishes monthly and there is nothing new to get.
EOF
fi

cat <<EOF

**Confirm:**

\`\`\`bash
ls -l $INSTALL_ROOT/stats/
\`\`\`

An empty directory is fine — the ingest will just skip that stage.

---

## 4. Deploy the containers

\`\`\`bash
$DEPLOY_CMD
\`\`\`

First start takes 2–5 minutes while \`npm install\` and \`pip install\` run. Later
restarts take seconds, because both install once and leave a marker file.

**Confirm — both must respond:**

\`\`\`bash
curl http://$HOST_IP:$API_PORT/health     # expect "no data" at this point
$SIM_HEALTH_CMD
\`\`\`

$SIM_NOTE

\`/health\` reporting **"no data"** here is correct. The database does not exist
yet — that is the next step.

---

## 5. Build the database

This is the step that turns your ZIM and dex into something searchable.

### What it reads

| Input | Where it comes from | Required? |
|---|---|---|
| Pokémon Showdown dex | \`$SIM_URL/dump/9\` — the \`pokedex-sim\` container | Yes |
| Bulbapedia articles | \`$CONTAINER_ZIM\` inside the container<br>= \`$ZIM_HOST_DIR/$ZIM_FILENAME\` on the host | Yes |
| Smogon usage stats | \`/stats/*.json\` = \`$INSTALL_ROOT/stats/\` | No — skipped if empty |
| Embedding model | Downloaded once to \`$INSTALL_ROOT/data/models\` | Yes, first run only |

### What it writes

| Output | What it is |
|---|---|
| \`$INSTALL_ROOT/data/pokedex.db\` | The knowledge base — dex tables, wiki chunks, aliases, statistics |
| \`$INSTALL_ROOT/data/embeddings.f16.npy\` | Vector index for semantic search |

### First, confirm the container can actually see its inputs

A path that is right on the host can still be wrong inside the container. Check
from the inside:

\`\`\`bash
$DOCKER_CMD exec pokedex-api ls -lh $CONTAINER_ZIM
$DOCKER_CMD exec pokedex-api ls /stats/
$SIM_DUMP_CMD
\`\`\`

The first command must show your ZIM. If it says "No such file", the
\`ZIM_PATH\` or the volume mount is wrong — fix that before going further.

### Then smoke-test on 100 articles (about a minute)

Do not skip this. It proves the ZIM parses the way the script expects, before
you commit to a 40-minute run.

\`\`\`bash
$DOCKER_CMD exec -it -e HF_HUB_OFFLINE=0 -e TRANSFORMERS_OFFLINE=0 \\
  pokedex-api /app/.venv/bin/python ingest.py --wiki-only --limit 100
\`\`\`

Expect roughly 100 articles and a few hundred chunks. If it fails, see
"Untested paths" in the README — that section covers the three known ways this
can go wrong and what each looks like.

### Then the full run (20–40 minutes)

**Run it detached.** \`-it\` ties the process to your terminal, so if the shell
drops — SSH timeout, closed laptop — the ingest dies with it. \`-d\` survives that.

\`\`\`bash
$DOCKER_CMD exec -d -e HF_HUB_OFFLINE=0 -e TRANSFORMERS_OFFLINE=0 \\
  pokedex-api /app/.venv/bin/python ingest.py
\`\`\`

Watch it from the host — \`/data\` is bind-mounted, and you can stop tailing at
any time without affecting the run:

\`\`\`bash
tail -f $INSTALL_ROOT/data/ingest.log
\`\`\`

\`HF_HUB_OFFLINE=0\` is needed **only on the first run**, so the embedding model
can download. Every run after that is plain \`ingest.py\` with no flags.

**If it gets interrupted**, nothing is corrupted. Each stage clears its own
scope before rewriting, and completed stages stay committed. Resume with:

| Interrupted during | Resume with |
|---|---|
| Bulbapedia, aliases, or embeddings | \`ingest.py --wiki-only\` |
| Smogon statistics | \`ingest.py --stats-only\` |
| Showdown dex | \`ingest.py --dex-only\` |

Check whether one is still running with \`$DOCKER_CMD top pokedex-api\` — look for
a \`python ingest.py\` process. Never start a second ingest while one is running;
they would write to the same database.

---

## 6. Verify

\`\`\`bash
curl http://$HOST_IP:$API_PORT/health
\`\`\`

\`status\` should now be \`ok\`, with non-zero counts for species, moves, and
wiki_chunks. Then try a real lookup:

\`\`\`bash
curl -X POST http://$HOST_IP:$API_PORT/lookup \\
  -H 'Content-Type: application/json' \\
  -d '{"name": "Kingambit"}'

curl -X POST http://$HOST_IP:$API_PORT/search_wiki \\
  -H 'Content-Type: application/json' \\
  -d '{"query": "Supreme Overlord ability"}'
\`\`\`

---

## 7. Connect your interface

EOF

if [ "$BUNDLE_OPENWEBUI" = "1" ]; then
cat <<EOF
Open <http://$HOST_IP:$OPENWEBUI_PORT> and create your admin account.

1. **Settings → Admin → Tool Servers**, add one:
   - Type: **OpenAPI**
   - URL: \`http://pokedex-api:8990\` (same compose project, container name works)
   - Spec path: \`/openapi.json\`
   - ID: \`pokedex\`
2. **Workspace → Models → Create**: pick your base model, enable the \`pokedex\`
   tool, and paste the system prompt from README Step 9.2.
EOF
elif [ "$UI_CHOICE" = "2" ]; then
cat <<EOF
In your existing Open WebUI at <$OPENWEBUI_URL>:

1. **Settings → Admin → Tool Servers**, add one:
   - Type: **OpenAPI**
   - URL: \`http://$HOST_IP:$API_PORT\` (different compose project — host IP, not a container name)
   - Spec path: \`/openapi.json\`
   - ID: \`pokedex\`
2. **Workspace → Models → Create**: pick your base model, enable the \`pokedex\`
   tool, and paste the system prompt from README Step 9.2.

Register it as an **admin** tool server, not a user one. User tool servers are
called from your browser and need CORS plus a URL that resolves from wherever
you are sitting.
EOF
else
cat <<EOF
No Open WebUI configured. The API is at \`http://$HOST_IP:$API_PORT\` with its
OpenAPI spec at \`/openapi.json\`.
EOF
fi

if [ "$BUNDLE_UI" = "1" ]; then
cat <<EOF

The standalone page is at <http://$HOST_IP:$UI_PORT>. It auto-detects the API on
port 8990 of the same hostname; the gear icon overrides that.
EOF
fi

cat <<EOF

---

## Test prompts

Ask these in order — each exercises a different part of the stack.

| Prompt | Should use |
|---|---|
| What's Charizard's hidden ability? | \`lookup\` — one call, fast |
| Where do I catch Feebas in Emerald? | \`search_wiki\` |
| Which episode did Ash's Charizard finally obey him? | \`search_wiki\`, anime |
| Is Kingambit better than Bisharp in OU? | \`lookup\` ×2 + \`usage_stats\` |
| Build me a rain team for gen9ou without Pelipper | multi-step + \`validate_team\` |

If the first one answers instantly with **no visible tool call**, tool calling is
not working on your model server. See README Step 3.6.

---

## Your configuration

| Setting | Value |
|---|---|
| Install root | \`$INSTALL_ROOT\` |
| Host address | \`$HOST_IP\` |
| Network profile | \`$NET_PROFILE\` |
| ZIM file | \`$ZIM_HOST_DIR/$ZIM_FILENAME\` |
| Kiwix | \`$KIWIX_URL\` |
| Model endpoint | \`$OPENAI_BASE_URL\` |
| Model name | \`$OPENAI_MODEL\` |
| API port | \`$API_PORT\` |

Change any of these by editing \`.env\` and restarting, or by re-running
\`./deploy.sh\`.
EOF
} > "$STEPS_FILE"

ok "$STEPS_FILE"

# --- plain-text version, for reading in a terminal -------------------------
# Same content, no markdown syntax: aligned columns instead of pipe tables,
# "$" prefixes instead of fenced blocks, hard-wrapped to 72 columns.

STEPS_TXT="$OUT_DIR/NEXT-STEPS.txt"
TRULE="──────────────────────────────────────────────────────────────────────"

{
printf '%s\n' "$TRULE"
printf '  POKEDEX AI  —  NEXT STEPS\n'
printf '  generated %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
printf '  install root: %s\n' "$INSTALL_ROOT"
printf '%s\n' "$TRULE"
cat <<EOF

Work through these in order. Each step says what it needs, what it does,
and how to check that it worked.

EOF

printf '%s\n STEP 1  —  Place the source files\n%s\n\n' "$TRULE" "$TRULE"
cat <<EOF
These are bind-mounted into the containers. You edit them on the host and
restart; there is never an image to rebuild.

  main.py, ingest.py, requirements.txt
      -> $INSTALL_ROOT/api/

  server.js, package.json
      -> $INSTALL_ROOT/sim/
EOF
[ "$BUNDLE_UI" = "1" ] && printf '\n  index.html\n      -> %s/ui/\n' "$INSTALL_ROOT" || true
cat <<EOF

  CONFIRM
    \$ ls -l $INSTALL_ROOT/api/main.py \\
             $INSTALL_ROOT/api/ingest.py \\
             $INSTALL_ROOT/api/requirements.txt \\
             $INSTALL_ROOT/sim/server.js \\
             $INSTALL_ROOT/sim/package.json

  All five must exist before you deploy.

EOF

printf '%s\n STEP 2  —  Put the Bulbapedia ZIM in place\n%s\n\n' "$TRULE" "$TRULE"
cat <<EOF
WHAT THIS FILE IS
  A ZIM is one compressed archive containing an entire website offline —
  every article, with images. You need the Bulbapedia one. It is not a
  database dump and not something this script can generate. It is a
  single file, typically 3-8 GB, named something like
  bulbapedia_en_all_maxi_2024-01.zim

WHERE TO GET IT
  https://library.kiwix.org          (search for "Bulbapedia")
  https://download.kiwix.org/zim/    (raw file listing)

  Use the BitTorrent link if one is offered. These are large files and
  Kiwix runs on donations.

  Heads up: the published Bulbapedia ZIM has historically lagged behind
  the live wiki. Mechanics and older content are unaffected; very recent
  games or episodes may be missing. Check the date in the filename.

WHERE IT MUST GO
  $ZIM_HOST_DIR/$ZIM_FILENAME

  CONFIRM
    \$ ls -lh $ZIM_HOST_DIR/$ZIM_FILENAME

IF YOUR FILE HAS A DIFFERENT NAME
  Update it in two places or nothing will find it:
    - .env  ->  ZIM_FILENAME, and ZIM_BOOK (same name without .zim)
EOF
[ "$BUNDLE_KIWIX" = "1" ] && printf '    - the kiwix service'"'"'s  command:  line in your compose file\n' || true
cat <<EOF

  Easiest option: re-run ./deploy.sh and give it the real filename.

EOF

printf '%s\n STEP 3  —  Competitive statistics  (optional)\n%s\n\n' "$TRULE" "$TRULE"
cat <<EOF
WHAT THESE FILES ARE
  Smogon publishes monthly usage statistics as JSON: which Pokemon get
  used, with what moves, items and spreads, and what checks them. They
  are the only data in this system that lives on the internet rather
  than on your disk.

  Skipping is fine. Everything except current-metagame questions works
  without them.

EOF
if [ "$NET_PROFILE" = "airgapped" ]; then
cat <<EOF
YOU CHOSE FULLY OFFLINE
  This host will never contact Smogon. To get the files anyway, on a
  machine that does have internet:

    \$ ./fetch-stats.sh --out ./stats --dry-run    (see the list first)
    \$ ./fetch-stats.sh --out ./stats              (all formats, latest month)

  Then copy the resulting *.json into:
    $INSTALL_ROOT/stats/

  OR DOWNLOAD DIRECTLY — they are plain files.
  Browse https://www.smogon.com/stats/<YYYY-MM>/chaos/ for the list,
  then for each one you want:

    \$ curl -A 'PokedexAI-selfhosted/1.0' \\
        -o gen9ou-1695-$(date +%Y-%m).json \\
        https://www.smogon.com/stats/$(date +%Y-%m)/chaos/gen9ou-1695.json

  The filename MUST be  <format>-<cutoff>-<YYYY-MM>.json  or the ingest
  skips it. 1695 is the ladder cutoff Smogon uses for OU tiering; some
  formats only publish at 0 or 1500.
EOF
else
cat <<EOF
  \$ ./fetch-stats.sh --dry-run    (list what it would fetch)
  \$ ./fetch-stats.sh              (all formats, latest month)

  Files land in $INSTALL_ROOT/stats/
  The script refuses to run more than once every 28 days, because Smogon
  publishes monthly and there is nothing new to get.
EOF
fi
cat <<EOF

  CONFIRM
    \$ ls -l $INSTALL_ROOT/stats/

  An empty directory is fine — the ingest just skips that stage.

EOF

printf '%s\n STEP 4  —  Deploy the containers\n%s\n\n' "$TRULE" "$TRULE"
cat <<EOF
  \$ $DEPLOY_CMD

First start takes 2-5 minutes while npm install and pip install run.
Later restarts take seconds, because both install once and leave a
marker file.

  CONFIRM — both must respond
    \$ curl http://$HOST_IP:$API_PORT/health
    \$ $SIM_HEALTH_CMD

  /health saying "no data" here is CORRECT. The database does not exist
  yet. That is the next step.

EOF

printf '%s\n STEP 5  —  Build the database\n%s\n\n' "$TRULE" "$TRULE"
cat <<EOF
This is the step that turns your ZIM and dex into something searchable.

WHAT IT READS
  Showdown dex        $SIM_URL/dump/9
                      (the pokedex-sim container)          REQUIRED

  Bulbapedia          $CONTAINER_ZIM
                      inside the container, which is
                      $ZIM_HOST_DIR/$ZIM_FILENAME
                      on the host                          REQUIRED

  Smogon stats        /stats/*.json  =
                      $INSTALL_ROOT/stats/
                                                           optional

  Embedding model     downloaded once to
                      $INSTALL_ROOT/data/models
                                                           first run only

WHAT IT WRITES
  $INSTALL_ROOT/data/pokedex.db
      the knowledge base: dex tables, wiki chunks, aliases, statistics

  $INSTALL_ROOT/data/embeddings.f16.npy
      vector index for semantic search

FIRST, CONFIRM THE CONTAINER CAN SEE ITS INPUTS
  A path that is right on the host can still be wrong inside the
  container. Check from the inside:

    \$ $DOCKER_CMD exec pokedex-api ls -lh $CONTAINER_ZIM
    \$ $DOCKER_CMD exec pokedex-api ls /stats/
    \$ $SIM_DUMP_CMD

  The first command must show your ZIM. "No such file" means ZIM_PATH or
  the volume mount is wrong. Fix that before going further.

THEN SMOKE-TEST ON 100 ARTICLES  (about a minute)
  Do not skip this. It proves the ZIM parses the way the script expects,
  before you commit to a 40-minute run.

    \$ $DOCKER_CMD exec -it -e HF_HUB_OFFLINE=0 -e TRANSFORMERS_OFFLINE=0 \\
        pokedex-api /app/.venv/bin/python ingest.py --wiki-only --limit 100

  Expect roughly 100 articles and a few hundred chunks. If it fails, see
  "Untested paths" in the README.

THEN THE FULL RUN  (20-40 minutes)
  RUN IT DETACHED. "-it" ties the process to your terminal, so if the
  shell drops — SSH timeout, closed laptop — the ingest dies with it.

    \$ $DOCKER_CMD exec -d -e HF_HUB_OFFLINE=0 -e TRANSFORMERS_OFFLINE=0 \\
        pokedex-api /app/.venv/bin/python ingest.py

  It writes /data/ingest.log on its own — no shell redirect needed.
  Watch it from the host, and stop tailing any time:

    \$ tail -f $INSTALL_ROOT/data/ingest.log

  HF_HUB_OFFLINE=0 is needed ONLY on this first run, so the embedding
  model can download. Every run after that is plain ingest.py.

IF IT GETS INTERRUPTED
  Nothing is corrupted. Each stage clears its own scope before
  rewriting, and completed stages stay committed. Resume with:

    interrupted during          resume with
    ----------------------------------------------------
    Bulbapedia / aliases /      ingest.py --wiki-only
      embeddings
    Smogon statistics           ingest.py --stats-only
    Showdown dex                ingest.py --dex-only

  Check whether one is still running:
    \$ $DOCKER_CMD top pokedex-api      (look for "python ingest.py")

  Never start a second ingest while one is running — they would write
  to the same database.

EOF

printf '%s\n STEP 6  —  Verify\n%s\n\n' "$TRULE" "$TRULE"
cat <<EOF
  \$ curl http://$HOST_IP:$API_PORT/health

  status should now be "ok", with non-zero counts for species, moves and
  wiki_chunks. Then try a real lookup:

  \$ curl -X POST http://$HOST_IP:$API_PORT/lookup \\
      -H 'Content-Type: application/json' \\
      -d '{"name": "Kingambit"}'

  \$ curl -X POST http://$HOST_IP:$API_PORT/search_wiki \\
      -H 'Content-Type: application/json' \\
      -d '{"query": "Supreme Overlord ability"}'

EOF

printf '%s\n STEP 7  —  Connect your interface\n%s\n\n' "$TRULE" "$TRULE"
if [ "$BUNDLE_OPENWEBUI" = "1" ]; then
cat <<EOF
Open  http://$HOST_IP:$OPENWEBUI_PORT  and create your admin account.

  1. Settings -> Admin -> Tool Servers, add one:
       Type       OpenAPI
       URL        http://pokedex-api:8990
                  (same compose project, container name works)
       Spec path  /openapi.json
       ID         pokedex

  2. Workspace -> Models -> Create:
       pick your base model, enable the "pokedex" tool, and paste the
       system prompt from README Step 9.2
EOF
elif [ "$UI_CHOICE" = "2" ]; then
cat <<EOF
In your existing Open WebUI at  $OPENWEBUI_URL

  1. Settings -> Admin -> Tool Servers, add one:
       Type       OpenAPI
       URL        http://$HOST_IP:$API_PORT
                  (different compose project — host IP, NOT a container name)
       Spec path  /openapi.json
       ID         pokedex

  2. Workspace -> Models -> Create:
       pick your base model, enable the "pokedex" tool, and paste the
       system prompt from README Step 9.2

  Register it as an ADMIN tool server, not a user one. User tool servers
  are called from your browser and need CORS plus a URL that resolves
  from wherever you are sitting.
EOF
else
cat <<EOF
No Open WebUI configured. The API is at:
  http://$HOST_IP:$API_PORT
with its OpenAPI spec at /openapi.json
EOF
fi
[ "$BUNDLE_UI" = "1" ] && cat <<EOF

The standalone page is at  http://$HOST_IP:$UI_PORT
It auto-detects the API on port 8990 of the same hostname; the gear icon
overrides that.
EOF
echo

printf '%s\n TEST PROMPTS\n%s\n\n' "$TRULE" "$TRULE"
cat <<'EOF'
Ask these in order. Each exercises a different part of the stack.

  What's Charizard's hidden ability?
      -> lookup, one call, fast

  Where do I catch Feebas in Emerald?
      -> search_wiki

  Which episode did Ash's Charizard finally obey him?
      -> search_wiki, anime

  Is Kingambit better than Bisharp in OU?
      -> lookup x2 + usage_stats

  Build me a rain team for gen9ou without Pelipper
      -> multi-step + validate_team

If the first one answers instantly with NO visible tool call, tool
calling is not working on your model server. See README Step 3.6.

EOF

printf '%s\n YOUR CONFIGURATION\n%s\n\n' "$TRULE" "$TRULE"
printf '  %-18s %s\n' "install root"    "$INSTALL_ROOT"
printf '  %-18s %s\n' "host address"    "$HOST_IP"
printf '  %-18s %s\n' "network profile" "$NET_PROFILE"
printf '  %-18s %s\n' "ZIM file"        "$ZIM_HOST_DIR/$ZIM_FILENAME"
printf '  %-18s %s\n' "kiwix"           "$KIWIX_URL"
printf '  %-18s %s\n' "model endpoint"  "$OPENAI_BASE_URL"
printf '  %-18s %s\n' "model name"      "$OPENAI_MODEL"
printf '  %-18s %s\n' "api port"        "$API_PORT"
printf '  %-18s %s\n' "stats formats"   "$STATS_FORMATS"
printf '  %-18s %s\n' "stats months"    "$STATS_MONTHS"
cat <<EOF

Change any of these by editing .env and restarting, or by re-running
./deploy.sh

EOF
printf '%s\n' "$TRULE"
} > "$STEPS_TXT"

ok "$STEPS_TXT"

# --- directories -----------------------------------------------------------

echo
if confirm "Create the directory structure under $INSTALL_ROOT now?"; then
  mkdir -p "$INSTALL_ROOT"/{api,sim,data,ui,stats,sets,openwebui}
  [ "$BUNDLE_KIWIX" = "1" ] && mkdir -p "$INSTALL_ROOT/zim" || true
  ok "Directories created."
fi

# --- on-screen summary -----------------------------------------------------

head1 "Next steps"

cat <<EOS
Full instructions — with what each step needs, what it produces, and how to
check it — were written to:

  ${B}$STEPS_TXT${N}   ${DIM}(plain text, for the terminal)${N}
  ${B}$STEPS_FILE${N}   ${DIM}(markdown, for an editor or browser)${N}

Read them right now with:

  ${B}less $STEPS_TXT${N}

The short version:

  1. Copy main.py, ingest.py, requirements.txt  ->  $INSTALL_ROOT/api/
     Copy server.js, package.json               ->  $INSTALL_ROOT/sim/

  2. Put the Bulbapedia ZIM here:
       $ZIM_HOST_DIR/$ZIM_FILENAME
     That is an offline archive of the whole wiki, roughly 3-8 GB.
     Get it from https://library.kiwix.org (search "Bulbapedia").
     Different filename? Re-run ./deploy.sh with the real one.

  3. Competitive stats (optional, skippable):
EOS

if [ "$NET_PROFILE" = "airgapped" ]; then
  echo "       run fetch-stats.sh on a machine with internet, copy the JSON"
  echo "       into $INSTALL_ROOT/stats/   — see NEXT-STEPS.txt for direct curls"
else
  echo "       ./fetch-stats.sh"
fi

cat <<EOS

  4. Deploy:
       $DEPLOY_CMD
     Then check both:
       curl http://$HOST_IP:$API_PORT/health   (will say "no data" — correct)
       $SIM_HEALTH_CMD

  5. Confirm the container can see the ZIM:
       $DOCKER_CMD exec pokedex-api ls -lh $CONTAINER_ZIM

  6. Smoke-test the ingest on 100 articles (~1 min) BEFORE the full run:
       $DOCKER_CMD exec -it -e HF_HUB_OFFLINE=0 -e TRANSFORMERS_OFFLINE=0 \\
         pokedex-api /app/.venv/bin/python ingest.py --wiki-only --limit 100

  7. Full ingest (20-40 min). The HF_HUB_OFFLINE=0 flags are needed only on
     this first run, so the embedding model can download:
       $DOCKER_CMD exec -it -e HF_HUB_OFFLINE=0 -e TRANSFORMERS_OFFLINE=0 \\
         pokedex-api /app/.venv/bin/python ingest.py

  8. Verify:
       curl http://$HOST_IP:$API_PORT/health     (status should be "ok" now)

  9. Connect your interface — see step 7 in NEXT-STEPS.txt.
EOS

if [ "$WANT_CRON" = "1" ]; then
  echo
  hr
  echo "${B}Scheduled refresh${N}"
  hr
  DAY=$(( (RANDOM % 27) + 1 ))
  HOUR=$(( RANDOM % 24 ))
  MIN=$(( RANDOM % 60 ))
  echo "Add to crontab. Day and time are randomized on purpose, so deployments"
  echo "of this tool don't all hit Smogon at midnight on the 1st:"
  echo
  echo "  $MIN $HOUR $DAY * * cd $OUT_DIR && ./fetch-stats.sh >> $INSTALL_ROOT/data/fetch.log 2>&1"
  echo "  $MIN $((HOUR + 1)) $DAY * * $DOCKER_CMD exec pokedex-api /app/.venv/bin/python ingest.py --stats-only"
  echo
  note "fetch-stats.sh also refuses to run more than once every 28 days, so a"
  note "misconfigured cron can't turn into a hammer."
fi

echo
hr
ok "Done. Nothing has been deployed — review the files first."
hr
