"""OpenAlex dental research scraper for CPT data.

OpenAlex (https://openalex.org) is a free, open catalog of the world's scholarly works.
API is free, no auth required, 100k requests/day limit (very generous).

Strategy:
  - Filter by dental/oral health concepts (OpenAlex concept IDs)
  - Fetch works (title + abstract) in batches via cursor pagination
  - Output as CPT chunks

Rate limit: 10 RPS without polite pool, 100 RPS with polite pool (mailto param).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "cpt_prepared" / "openalex_dental"
for d in (OUT_DIR,):
    d.mkdir(parents=True, exist_ok=True)

BASE_URL = "https://api.openalex.org"
EMAIL = os.environ.get("NCBI_EMAIL", "docsmile@example.com")

# OpenAlex concept IDs for dental topics (full URL format required)
DENTAL_CONCEPTS = [
    "https://openalex.org/C199343813",   # Dentistry (1.9M works)
    "https://openalex.org/C144374066",   # Periodontology (12k)
    "https://openalex.org/C2778400979",  # Crown (dentistry) (133k)
    "https://openalex.org/C2776268601",  # Occlusion (179k)
]

# Use the broadest concept (Dentistry) which covers subconcepts
DENTAL_FILTER = "concepts.id:https://openalex.org/C199343813"

# Post-filter: require at least 2 dental terms in title+abstract
_DENTAL_RE = re.compile(
    r"\b(dent(?:al|ist|in|ition|ure)|tooth|teeth|molar|premolar|incisor|canine teeth|"
    r"oral (?:health|cavity|mucosa|cancer|surgery|hygiene|pathol)|"
    r"gingiv|periodon|endodon|orthodon|prosthodon|pulp(?:itis|ectomy|otomy)|"
    r"caries|dental crown|dental bridge|dental implant|extraction|root canal|"
    r"mandib|maxill|alveol|occlus|malocclu|bruxism|"
    r"fluoride|plaque|tartar|calculus|scaling|"
    r"amalgam|composite resin|ceramic|zirconia|"
    r"tmj|temporomandibular|stomatitis|glossitis|"
    r"leukoplakia|erythroplakia|oral lichen|"
    r"salivary gland|parotid|cleft (?:lip|palate))\b",
    re.IGNORECASE,
)


def _http_get(url: str, max_retries: int = 4) -> dict:
    backoff = 1.0
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(url, headers={
                "User-Agent": "DocSmile/1.0 (mailto:docsmile@example.com)",
            })
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            time.sleep(backoff)
            backoff = min(backoff * 2, 30.0)
    raise RuntimeError("unreachable")


def _extract_abstract(inverted_index: dict | None) -> str:
    """OpenAlex stores abstracts as inverted indexes. Reconstruct plain text."""
    if not inverted_index:
        return ""
    # inverted_index: {word: [positions]}
    positions = {}
    for word, pos_list in inverted_index.items():
        for pos in pos_list:
            positions[pos] = word
    if not positions:
        return ""
    max_pos = max(positions.keys())
    words = [positions.get(i, "") for i in range(max_pos + 1)]
    return " ".join(w for w in words if w)


def fetch_works(cursor: str | None = None, per_page: int = 200,
                pub_year_min: int = 2000) -> tuple[list[dict], str | None]:
    """Fetch a page of dental works from OpenAlex."""
    params = {
        "filter": f"{DENTAL_FILTER},from_publication_date:{pub_year_min}-01-01,has_abstract:true",
        "per_page": str(per_page),
        "mailto": EMAIL,
        "select": "id,title,publication_year,abstract_inverted_index,primary_location,authorships",
    }
    if cursor:
        params["cursor"] = cursor
    else:
        params["cursor"] = "*"

    url = f"{BASE_URL}/works?{urllib.parse.urlencode(params)}"
    data = _http_get(url)

    results = data.get("results", [])
    meta = data.get("meta", {})
    next_cursor = meta.get("next_cursor")

    works = []
    for r in results:
        title = (r.get("title") or "").strip()
        abstract = _extract_abstract(r.get("abstract_inverted_index"))
        year = r.get("publication_year", "")
        oa_id = (r.get("id") or "").replace("https://openalex.org/", "")

        # Get journal/source name
        loc = r.get("primary_location") or {}
        source = loc.get("source", {}) or {}
        journal = (source.get("display_name") or "").strip()

        if not title or not abstract or len(abstract) < 100:
            continue

        text = f"{title}\n\n{abstract}"
        # Post-filter: must have at least 2 dental terms
        if len(_DENTAL_RE.findall(text)) < 2:
            continue
        works.append({
            "text": text,
            "source": f"openalex:{oa_id}",
            "journal": journal,
            "year": year,
        })

    return works, next_cursor


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, default=200000,
                        help="Target number of CPT chunks")
    parser.add_argument("--pub-year-min", type=int, default=2000)
    args = parser.parse_args(argv)

    state_path = OUT_DIR / "state.json"
    out_path = OUT_DIR / "openalex_dental_cpt.jsonl"

    # Resume state
    cursor = None
    n_chunks = 0
    seen_ids: set[str] = set()
    if state_path.exists():
        try:
            state = json.loads(state_path.read_text())
            cursor = state.get("cursor")
            n_chunks = state.get("n_chunks", 0)
            seen_ids = set(state.get("seen_ids", []))
            print(f"[openalex] resume: chunks={n_chunks} seen={len(seen_ids)} cursor={'yes' if cursor else 'start'}")
        except Exception:
            pass

    started = time.time()
    last_log = time.time()
    n_pages = 0

    try:
        with out_path.open("a", encoding="utf-8") as f_out:
            while n_chunks < args.target:
                works, next_cursor = fetch_works(
                    cursor=cursor, pub_year_min=args.pub_year_min
                )
                n_pages += 1

                if not works and not next_cursor:
                    print("[openalex] no more results, stopping")
                    break

                for w in works:
                    oa_id = w["source"]
                    if oa_id in seen_ids:
                        continue
                    seen_ids.add(oa_id)
                    row = {"text": w["text"], "source": w["source"]}
                    f_out.write(json.dumps(row, ensure_ascii=False) + "\n")
                    n_chunks += 1

                cursor = next_cursor
                time.sleep(0.12)  # ~8 RPS

                if time.time() - last_log > 20:
                    elapsed = time.time() - started
                    rate = n_chunks / max(1.0, elapsed)
                    print(f"[openalex] pages={n_pages} chunks={n_chunks}/{args.target} "
                          f"rate={rate:.0f}/s")
                    # Save state
                    state_path.write_text(json.dumps({
                        "cursor": cursor,
                        "n_chunks": n_chunks,
                        "seen_ids": list(seen_ids)[-100000:],  # keep last 100k to limit file size
                    }))
                    last_log = time.time()

                if not next_cursor:
                    print("[openalex] cursor exhausted")
                    break
    finally:
        state_path.write_text(json.dumps({
            "cursor": cursor,
            "n_chunks": n_chunks,
            "seen_ids": list(seen_ids)[-100000:],
        }))

    print(f"\n[openalex] DONE: chunks={n_chunks} pages={n_pages}")
    print(f"  -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
