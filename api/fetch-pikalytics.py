#!/usr/bin/env python3
# DexAI — fetch-pikalytics.py — polite downloader for Pikalytics usage data.
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
fetch-pikalytics.py — polite downloader for Pikalytics usage data.

Run this on a machine with internet access (the NAS host is fine), then let
ingest.py read the files it writes:

    ./fetch-pikalytics.py --out /mnt/Apps/pokedex/pikalytics
    docker exec pokedex-api /app/.venv/bin/python /app/ingest.py --pikalytics-only

What it fetches, per format:
  1. /ai/pokedex/<format>            -> the "Data Date" (YYYY-MM). Never guessed.
  2. /api/l/<date>/<format>-<cutoff> -> the whole roster. Each entry already
     carries that Pokemon's full record (moves, items, abilities, teammates...),
     so this is ONE request per format, not one per Pokemon. A per-Pokemon
     request (/api/p/...) is made only for an entry that arrives without its
     record, and only up to --max-detail of them.

The /api/ endpoints are the ones Pikalytics' own website calls; they are not
part of its documented AI-agent interface. This script is deliberately gentle:
an honest User-Agent, a delay between requests, a handful of requests per run,
and it skips a format it already has for the current data month.

IMPORTANT behaviour: these endpoints answer HTTP 200 with `[]` or `false`
when a key is wrong or data is missing. An empty or implausibly small roster is
therefore treated as a FAILURE — nothing is written, and any good file from a
previous run is left untouched.

Pikalytics asks to be credited when its data is used; the files record
"source": "Pikalytics" and ingest carries that through to the API.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

BASE = "https://www.pikalytics.com"
UA = "PokedexAI-personal-fetcher/1.0 (self-hosted, low volume; ~2 requests per format)"

# Verified to return real data (cutoff 1760): the current Champions ladder
# format and the M-B season-3 battle data.
DEFAULT_FORMATS = ["gen9championsvgc2026regmc", "battledataregmbs3"]

# Format codes that MIGHT exist. These are guesses used only by --probe, which
# reports which ones respond and writes nothing.
PROBE_CANDIDATES = [
    "gen9championsvgc2026regma", "gen9championsvgc2026regmb",
    "gen9championsvgc2026regmabo3", "gen9championsvgc2026bo3regma",
    "gen9championsbssregma", "gen9championsou",
    "battledataregmas2", "battledataregmbs1", "battledataregmbs2",
]

# A real roster has hundreds of entries. Anything below this is an API
# hiccup or a wrong key, not data.
MIN_ROSTER = 10

# Fields kept from each roster entry. The rest (search strings, FAQ text,
# type-matchup tables, tournament team lists) is bulk ingest doesn't use.
KEEP = (
    "name", "rank", "percent", "raw", "raw_count", "games", "wins", "losses", "ties",
    "winPercent", "winRate", "types", "stats", "abilities", "items", "natures",
    "moves", "spreads", "team", "leads", "counters", "megas", "mega_percent",
    "brought_count", "brought_percent", "id",
)

_last_request = 0.0


class FetchError(Exception):
    pass


def http_get(url: str, delay: float, retries: int = 3) -> str:
    """GET with a politeness delay, honest UA, and bounded retries."""
    global _last_request
    for attempt in range(1, retries + 1):
        wait = delay - (time.time() - _last_request)
        if wait > 0:
            time.sleep(wait)
        _last_request = time.time()
        req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "*/*"})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < retries:
                print(f"    429 rate limited — waiting 30s (attempt {attempt}/{retries})")
                time.sleep(30)
                continue
            if 500 <= e.code < 600 and attempt < retries:
                time.sleep(5 * attempt)
                continue
            raise FetchError(f"HTTP {e.code} for {url}")
        except (urllib.error.URLError, TimeoutError) as e:
            if attempt < retries:
                time.sleep(5 * attempt)
                continue
            raise FetchError(f"{e} for {url}")
    raise FetchError(f"gave up on {url}")


def discover_date(base: str, fmt: str, delay: float) -> str:
    """The data month, read from Pikalytics' own page. Never guessed."""
    text = http_get(f"{base}/ai/pokedex/{urllib.parse.quote(fmt)}", delay)
    m = re.search(r"\*\*Data Date\*\*:\s*(\d{4}-\d{2})", text)
    if not m:
        raise FetchError(f"no 'Data Date' on /ai/pokedex/{fmt} — is the format code right?")
    return m.group(1)


def fetch_roster(base: str, fmt: str, cutoff: int, date: str, delay: float) -> list:
    url = f"{base}/api/l/{date}/{urllib.parse.quote(fmt)}-{cutoff}"
    raw = http_get(url, delay)
    try:
        data = json.loads(raw)
    except ValueError:
        raise FetchError(f"roster for {fmt}-{cutoff} was not JSON: {raw[:80]!r}")
    if not isinstance(data, list) or len(data) < MIN_ROSTER:
        n = len(data) if isinstance(data, list) else type(data).__name__
        raise FetchError(
            f"roster for {fmt}-{cutoff} on {date} came back as {n} — this API answers "
            f"200 with an empty result for a wrong key, so this is a failure, not data")
    good = [e for e in data if isinstance(e, dict) and e.get("name")]
    if len(good) < MIN_ROSTER:
        raise FetchError(f"roster for {fmt}-{cutoff} had too few usable entries ({len(good)})")
    return good


def fill_missing_details(base: str, fmt: str, cutoff: int, date: str, roster: list,
                         delay: float, max_detail: int) -> tuple[int, int]:
    """Per-Pokemon fetch, only for entries that arrived without their record."""
    missing = [e for e in roster if "moves" not in e or "items" not in e]
    filled = failed = 0
    for e in missing[:max_detail]:
        url = (f"{base}/api/p/{date}/{urllib.parse.quote(fmt)}-{cutoff}/"
               f"{urllib.parse.quote(e['name'])}")
        try:
            d = json.loads(http_get(url, delay))
        except (FetchError, ValueError):
            failed += 1
            continue
        if isinstance(d, dict) and "moves" in d:
            e.update({k: v for k, v in d.items() if k not in e})
            filled += 1
        else:
            failed += 1
    return filled, failed + max(0, len(missing) - max_detail)


def trim(e: dict) -> dict:
    return {k: e[k] for k in KEEP if k in e}


def write_atomic(path: str, payload: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    os.replace(tmp, path)


def fetch_format(base: str, fmt: str, cutoff: int, out_dir: str, delay: float,
                 max_detail: int, force: bool) -> str:
    date = discover_date(base, fmt, delay)
    path = os.path.join(out_dir, f"{fmt}-{cutoff}-{date}.json")
    if os.path.exists(path) and not force:
        return f"already have {os.path.basename(path)} — skipped (use --force to refetch)"
    roster = fetch_roster(base, fmt, cutoff, date, delay)
    filled, unfilled = fill_missing_details(base, fmt, cutoff, date, roster, delay, max_detail)
    write_atomic(path, {
        "source": "Pikalytics", "format": fmt, "cutoff": cutoff, "date": date,
        "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "roster": [trim(e) for e in roster],
    })
    detail = f", {filled} filled via per-Pokemon requests" if filled else ""
    warn = f" ({unfilled} entries left without a full record)" if unfilled else ""
    # Not every format publishes a usage share. The battle-data format carries games, a win
    # rate and a rank instead; the ingest loads it with usage left empty. Say so here, so the
    # difference is visible when the file is written rather than only when it is read.
    if roster and not any(isinstance(e, dict) and e.get("percent") not in (None, "") for e in roster):
        warn += " — NOTE: this format publishes no usage share (games, win rate and rank instead)"
    return f"wrote {os.path.basename(path)}: {len(roster)} Pokemon{detail}{warn}"


def probe(base: str, formats: list[str], cutoffs: list[int], delay: float) -> None:
    print("Probing (reads only, writes nothing). Format codes below are guesses.\n")
    for fmt in formats:
        try:
            date = discover_date(base, fmt, delay)
        except FetchError as e:
            print(f"  {fmt:34} no such format page ({e})")
            continue
        found = False
        for cut in cutoffs:
            try:
                n = len(fetch_roster(base, fmt, cut, date, delay))
                print(f"  {fmt:34} OK  date={date} cutoff={cut} roster={n}")
                found = True
                break
            except FetchError:
                continue
        if not found:
            print(f"  {fmt:34} page exists (date={date}) but no cutoff in {cutoffs} returned data")


def main() -> int:
    ap = argparse.ArgumentParser(description="Polite Pikalytics usage downloader.")
    ap.add_argument("--out", default=os.environ.get("PIKALYTICS_DIR", "./pikalytics"),
                    help="directory to write into (default: $PIKALYTICS_DIR or ./pikalytics)")
    ap.add_argument("--formats", default=",".join(DEFAULT_FORMATS),
                    help="comma-separated format codes (default: %(default)s)")
    ap.add_argument("--cutoff", default="1760",
                    help="rating cutoff(s), comma-separated (default 1760, the value verified to work)")
    ap.add_argument("--delay", type=float, default=2.0, help="seconds between requests (default 2)")
    ap.add_argument("--max-detail", type=int, default=100,
                    help="cap on per-Pokemon fallback requests per format (default 100)")
    ap.add_argument("--force", action="store_true", help="refetch even if the file exists")
    ap.add_argument("--probe", action="store_true",
                    help="test which format codes respond; write nothing")
    ap.add_argument("--base", default=BASE, help=argparse.SUPPRESS)
    args = ap.parse_args()

    formats = [f.strip() for f in args.formats.split(",") if f.strip()]
    cutoffs = [int(c) for c in args.cutoff.split(",") if c.strip()]

    if args.probe:
        cands = formats if args.formats != ",".join(DEFAULT_FORMATS) else PROBE_CANDIDATES
        probe(args.base, cands, cutoffs, args.delay)
        return 0

    os.makedirs(args.out, exist_ok=True)
    failures = 0
    for fmt in formats:
        for cut in cutoffs:
            print(f"{fmt} (cutoff {cut}):")
            try:
                print("  " + fetch_format(args.base, fmt, cut, args.out, args.delay,
                                          args.max_detail, args.force))
            except FetchError as e:
                failures += 1
                print(f"  FAILED — nothing written, existing files untouched: {e}")
    if failures:
        print(f"\n{failures} fetch(es) failed.")
        return 1
    print("\nDone. Now: docker exec pokedex-api /app/.venv/bin/python /app/ingest.py --pikalytics-only")
    return 0


if __name__ == "__main__":
    sys.exit(main())
