"""PMC Open Access full-text dental article scraper via BioC API.

Strategy:
  1. Use E-utilities esearch to find PMC IDs for dental articles in PMC OA subset
  2. Fetch full-text via BioC REST API (JSON format)
  3. Extract plain text passages (paragraphs), chunk to 1500-3000 chars
  4. Output as CPT chunks: {text, source}

PMC BioC API: https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/BioC_json/{PMCID}/unicode
No documented rate limit, but we respect 3 RPS as a courtesy.
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
OUT_DIR = ROOT / "cpt_prepared" / "pmc_dental"
LOG_DIR = ROOT / "logs" / "pmc"
for d in (OUT_DIR, LOG_DIR):
    d.mkdir(parents=True, exist_ok=True)

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
BIOC_API = "https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/BioC_json"
TOOL = "docsmile-pmc"
EMAIL = os.environ.get("NCBI_EMAIL", "docsmile@example.com")
API_KEY = os.environ.get("NCBI_API_KEY", "")

# Dental MeSH query for PMC
QUERY = (
    '("Dentistry"[MeSH] OR "Tooth Diseases"[MeSH] OR "Mouth Diseases"[MeSH] '
    'OR "Periodontal Diseases"[MeSH] OR "Stomatognathic Diseases"[MeSH] '
    'OR "Endodontics"[MeSH] OR "Orthodontics"[MeSH] OR "Prosthodontics"[MeSH] '
    'OR "Oral Surgical Procedures"[MeSH] OR "Dental Implants"[MeSH] '
    'OR "Dental Caries"[MeSH] OR "Periodontitis"[MeSH] OR "Gingivitis"[MeSH] '
    'OR "Pulpitis"[MeSH] OR "Malocclusion"[MeSH] OR "Tooth Loss"[MeSH] '
    'OR "Temporomandibular Joint Disorders"[MeSH] '
    'OR "Mouth Neoplasms"[MeSH]) AND open access[filter]'
)

MIN_CHUNK_CHARS = 300
MAX_CHUNK_CHARS = 3000
TARGET_CHUNK_CHARS = 1800


def _qs(params: dict) -> str:
    base = {"tool": TOOL, "email": EMAIL}
    if API_KEY:
        base["api_key"] = API_KEY
    base.update(params)
    return urllib.parse.urlencode(base)


def _http_get(url: str, max_retries: int = 4) -> bytes:
    backoff = 1.0
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "DocSmile/1.0"})
            with urllib.request.urlopen(req, timeout=120) as resp:
                return resp.read()
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            time.sleep(backoff)
            backoff = min(backoff * 2, 30.0)
    raise RuntimeError("unreachable")


def _rate_sleep() -> None:
    time.sleep(0.4 if not API_KEY else 0.15)


def _safe_json_loads(raw: bytes) -> dict:
    """Parse JSON from NCBI, stripping control characters that sometimes appear."""
    text = raw.decode("utf-8", errors="replace")
    # Strip control characters except \n, \r, \t
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', text)
    return json.loads(text)


def search_pmc_ids(start_year: int, end_year: int, retmax: int = 9999) -> list[str]:
    """Search PMC for dental OA articles and return PMCIDs.

    Uses date-range chunking to work around retstart limits.
    """
    all_ids = []

    for year in range(end_year, start_year - 1, -1):
        term = f'{QUERY} AND ("{year}"[PDAT] : "{year}"[PDAT])'
        params = {
            "db": "pmc", "term": term, "retmode": "json",
            "retmax": str(retmax),
        }
        url = f"{EUTILS}/esearch.fcgi?{_qs(params)}"
        data = _safe_json_loads(_http_get(url))
        res = data.get("esearchresult", {})
        count = int(res.get("count", 0))
        ids = res.get("idlist", [])
        all_ids.extend(ids)
        print(f"[pmc] year={year}: count={count} got={len(ids)} total={len(all_ids)}")
        _rate_sleep()

    # PMC IDs from esearch are numeric; prepend "PMC"
    return [f"PMC{pid}" for pid in all_ids]


def fetch_fulltext(pmc_id: str) -> str | None:
    """Fetch full text of a PMC article via BioC API. Returns plain text or None."""
    url = f"{BIOC_API}/{pmc_id}/unicode"
    try:
        raw = _http_get(url)
    except Exception:
        return None
    try:
        data = _safe_json_loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None

    # BioC JSON structure: documents[0].passages[] each with .text
    passages = []
    docs = data if isinstance(data, list) else [data]
    for doc in docs:
        if not isinstance(doc, dict):
            continue
        for passage in doc.get("documents", [{}])[0].get("passages", []):
            text = (passage.get("text") or "").strip()
            section = (passage.get("infons", {}).get("section_type") or "").lower()
            # Skip references, acknowledgments, supplementary
            if section in ("ref", "references", "ack", "acknowledgments",
                           "supplementary-material", "back", "fn"):
                continue
            if len(text) > 50:
                passages.append(text)
    return "\n\n".join(passages) if passages else None


def chunk_text(text: str, source: str) -> list[dict]:
    """Split full text into CPT chunks of TARGET_CHUNK_CHARS."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current = ""
    for para in paragraphs:
        if len(current) + len(para) + 2 > MAX_CHUNK_CHARS and len(current) >= MIN_CHUNK_CHARS:
            chunks.append({"text": current.strip(), "source": source})
            current = para
        else:
            current = current + "\n\n" + para if current else para
    if len(current) >= MIN_CHUNK_CHARS:
        chunks.append({"text": current.strip(), "source": source})
    return chunks


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, default=50000,
                        help="Target number of CPT chunks")
    parser.add_argument("--start-year", type=int, default=2026)
    parser.add_argument("--end-year", type=int, default=2000)
    parser.add_argument("--max-articles", type=int, default=25000,
                        help="Max articles to fetch full text for")
    args = parser.parse_args(argv)

    state_path = OUT_DIR / "state.json"
    out_path = OUT_DIR / "pmc_dental_cpt.jsonl"

    # Resume state
    done_ids: set[str] = set()
    n_chunks = 0
    if state_path.exists():
        try:
            state = json.loads(state_path.read_text())
            done_ids = set(state.get("done_ids", []))
            n_chunks = state.get("n_chunks", 0)
            print(f"[pmc] resume: done_ids={len(done_ids)} chunks={n_chunks}")
        except Exception:
            pass

    # Phase 1: Get PMC IDs
    ids_cache = OUT_DIR / "pmc_ids.json"
    if ids_cache.exists():
        pmc_ids = json.loads(ids_cache.read_text())
        print(f"[pmc] loaded {len(pmc_ids)} cached PMCIDs")
    else:
        print("[pmc] searching for dental PMC articles...")
        pmc_ids = search_pmc_ids(args.end_year, args.start_year)
        ids_cache.write_text(json.dumps(pmc_ids))
        print(f"[pmc] found {len(pmc_ids)} PMCIDs, cached to {ids_cache}")

    # Phase 2: Fetch full text and chunk
    to_fetch = [pid for pid in pmc_ids if pid not in done_ids][:args.max_articles]
    print(f"[pmc] articles to fetch: {len(to_fetch)} (skipping {len(done_ids)} already done)")

    started = time.time()
    last_log = time.time()
    n_fetched = 0
    n_empty = 0

    try:
        with out_path.open("a", encoding="utf-8") as f_out:
            for i, pmc_id in enumerate(to_fetch):
                if n_chunks >= args.target:
                    break
                text = fetch_fulltext(pmc_id)
                done_ids.add(pmc_id)
                n_fetched += 1

                if not text or len(text) < 500:
                    n_empty += 1
                    _rate_sleep()
                    continue

                article_chunks = chunk_text(text, f"pmc:{pmc_id}")
                for chunk in article_chunks:
                    f_out.write(json.dumps(chunk, ensure_ascii=False) + "\n")
                    n_chunks += 1

                _rate_sleep()

                if time.time() - last_log > 20:
                    elapsed = time.time() - started
                    rate = n_fetched / max(1.0, elapsed)
                    eta = (len(to_fetch) - n_fetched) / max(1.0, rate)
                    print(f"[pmc] fetched={n_fetched}/{len(to_fetch)} "
                          f"chunks={n_chunks} empty={n_empty} "
                          f"rate={rate:.1f} art/s eta={eta:.0f}s")
                    # Save state periodically
                    state_path.write_text(json.dumps({
                        "done_ids": list(done_ids),
                        "n_chunks": n_chunks,
                    }))
                    last_log = time.time()
    finally:
        state_path.write_text(json.dumps({
            "done_ids": list(done_ids),
            "n_chunks": n_chunks,
        }))

    print(f"\n[pmc] DONE: fetched={n_fetched} empty={n_empty} chunks={n_chunks}")
    print(f"  -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
