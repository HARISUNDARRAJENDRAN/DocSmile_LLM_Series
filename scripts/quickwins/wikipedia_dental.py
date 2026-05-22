"""Wikipedia dental article scraper via MediaWiki API.

Strategy:
  1. Enumerate all pages in Category:Dentistry and subcategories (recursively)
  2. Fetch plain-text extracts via the TextExtracts API
  3. Chunk into CPT-ready segments

Uses only the public MediaWiki API, no special auth needed.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "cpt_prepared" / "wikipedia_dental"
for d in (OUT_DIR,):
    d.mkdir(parents=True, exist_ok=True)

API_URL = "https://en.wikipedia.org/w/api.php"

# Seed categories — we'll recurse into subcategories
SEED_CATEGORIES = [
    "Category:Dentistry",
    "Category:Oral and maxillofacial surgery",
    "Category:Teeth",
    "Category:Periodontology",
    "Category:Orthodontics",
    "Category:Endodontics",
    "Category:Prosthodontology",
    "Category:Dental materials",
    "Category:Dental anatomy",
    "Category:Oral hygiene",
    "Category:Oral pathology",
    "Category:Tooth diseases",
    "Category:Mouth diseases",
    "Category:Dental procedures",
    "Category:Oral cancer",
    "Category:Dentistry branches",
    "Category:Dental restorations",
    "Category:Dental implants",
    "Category:Dentistry procedures",
    "Category:History of dentistry",
    "Category:Pediatric dentistry",
    "Category:Dental equipment",
    "Category:Types of dentistry",
]

MIN_CHUNK_CHARS = 300
MAX_CHUNK_CHARS = 3000


def _api_get(params: dict) -> dict:
    params.setdefault("format", "json")
    url = f"{API_URL}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": "DocSmile/1.0 (dental LLM training)"})
    backoff = 1.0
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception:
            if attempt == 3:
                raise
            time.sleep(backoff)
            backoff *= 2
    raise RuntimeError("unreachable")


def get_category_members(category: str, seen_cats: set, depth: int = 0, max_depth: int = 4) -> list[str]:
    """Recursively get all page titles under a category."""
    if depth > max_depth or category in seen_cats:
        return []
    seen_cats.add(category)
    pages = []
    cmcontinue = None

    while True:
        params = {
            "action": "query", "list": "categorymembers",
            "cmtitle": category, "cmlimit": "500",
        }
        if cmcontinue:
            params["cmcontinue"] = cmcontinue

        data = _api_get(params)
        members = data.get("query", {}).get("categorymembers", [])

        for m in members:
            ns = m.get("ns", 0)
            title = m.get("title", "")
            if ns == 0:  # Article
                pages.append(title)
            elif ns == 14:  # Subcategory
                pages.extend(get_category_members(title, seen_cats, depth + 1, max_depth))

        cont = data.get("continue", {})
        cmcontinue = cont.get("cmcontinue")
        if not cmcontinue:
            break
        time.sleep(0.1)

    return pages


def get_page_text(title: str) -> str | None:
    """Fetch plain-text extract of a Wikipedia article."""
    params = {
        "action": "query", "prop": "extracts",
        "titles": title, "explaintext": "1",
        "exsectionformat": "plain",
    }
    data = _api_get(params)
    pages = data.get("query", {}).get("pages", {})
    for pid, page in pages.items():
        if pid == "-1":
            return None
        text = (page.get("extract") or "").strip()
        return text if len(text) > 200 else None
    return None


def chunk_text(text: str, source: str) -> list[dict]:
    """Split article text into CPT chunks."""
    # Split on section headers (== ... ==) or double newlines
    sections = re.split(r"\n\n+", text)
    chunks = []
    current = ""
    for section in sections:
        section = section.strip()
        if not section:
            continue
        if len(current) + len(section) + 2 > MAX_CHUNK_CHARS and len(current) >= MIN_CHUNK_CHARS:
            chunks.append({"text": current.strip(), "source": source})
            current = section
        else:
            current = current + "\n\n" + section if current else section
    if len(current) >= MIN_CHUNK_CHARS:
        chunks.append({"text": current.strip(), "source": source})
    return chunks


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-articles", type=int, default=5000)
    args = parser.parse_args(argv)

    out_path = OUT_DIR / "wikipedia_dental_cpt.jsonl"
    state_path = OUT_DIR / "state.json"

    # Resume state
    done_titles: set[str] = set()
    n_chunks = 0
    if state_path.exists():
        try:
            state = json.loads(state_path.read_text())
            done_titles = set(state.get("done_titles", []))
            n_chunks = state.get("n_chunks", 0)
            print(f"[wiki] resume: done={len(done_titles)} chunks={n_chunks}")
        except Exception:
            pass

    # Phase 1: Enumerate pages
    titles_cache = OUT_DIR / "titles.json"
    if titles_cache.exists():
        all_titles = json.loads(titles_cache.read_text())
        print(f"[wiki] loaded {len(all_titles)} cached titles")
    else:
        print("[wiki] enumerating dental categories...")
        seen_cats: set[str] = set()
        all_titles_set: set[str] = set()
        for cat in SEED_CATEGORIES:
            pages = get_category_members(cat, seen_cats)
            all_titles_set.update(pages)
            print(f"[wiki]   {cat}: {len(pages)} pages (total unique: {len(all_titles_set)})")
            time.sleep(0.2)
        all_titles = sorted(all_titles_set)
        titles_cache.write_text(json.dumps(all_titles))
        print(f"[wiki] found {len(all_titles)} unique dental articles")

    # Phase 2: Fetch and chunk
    to_fetch = [t for t in all_titles if t not in done_titles][:args.max_articles]
    print(f"[wiki] articles to fetch: {len(to_fetch)} (skipping {len(done_titles)} done)")

    started = time.time()
    last_log = time.time()
    n_fetched = 0
    n_empty = 0

    try:
        with out_path.open("a", encoding="utf-8") as f_out:
            for title in to_fetch:
                text = get_page_text(title)
                done_titles.add(title)
                n_fetched += 1

                if not text:
                    n_empty += 1
                    time.sleep(0.15)
                    continue

                article_chunks = chunk_text(text, f"wikipedia:{title}")
                for chunk in article_chunks:
                    f_out.write(json.dumps(chunk, ensure_ascii=False) + "\n")
                    n_chunks += 1

                time.sleep(0.15)  # Respectful rate limit

                if time.time() - last_log > 20:
                    elapsed = time.time() - started
                    rate = n_fetched / max(1.0, elapsed)
                    eta = (len(to_fetch) - n_fetched) / max(1.0, rate)
                    print(f"[wiki] fetched={n_fetched}/{len(to_fetch)} "
                          f"chunks={n_chunks} empty={n_empty} "
                          f"rate={rate:.1f} art/s eta={eta:.0f}s")
                    state_path.write_text(json.dumps({
                        "done_titles": list(done_titles),
                        "n_chunks": n_chunks,
                    }))
                    last_log = time.time()
    finally:
        state_path.write_text(json.dumps({
            "done_titles": list(done_titles),
            "n_chunks": n_chunks,
        }))

    print(f"\n[wiki] DONE: fetched={n_fetched} empty={n_empty} chunks={n_chunks}")
    print(f"  -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
