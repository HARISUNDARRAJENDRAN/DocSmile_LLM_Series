"""PubMed dental abstract scraper via NCBI E-utilities, with year-range chunking
to avoid the 10k retstart cap per WebEnv.

Strategy:
  - Walk year-ranges from 2026 down to 1900.
  - For each range, esearch with usehistory=y; if count > 9000, split year range in half.
  - efetch in batches of 200 within each safe chunk.
  - Persist a global checkpoint in state.json.

Rate limit:
  3 RPS without NCBI_API_KEY; 10 RPS with one.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "cpt_prepared" / "pubmed_dental"
LOG_DIR = ROOT / "logs" / "pubmed"
for d in (OUT_DIR, LOG_DIR):
    d.mkdir(parents=True, exist_ok=True)

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
TOOL = "docsmile-quickwins"
EMAIL = os.environ.get("NCBI_EMAIL", "docsmile@example.com")
API_KEY = os.environ.get("NCBI_API_KEY", "")
MAX_PER_WEBENV = 9000  # stay safely under NCBI's retstart limit per session

QUERY = (
    '("Dentistry"[MeSH] OR "Tooth Diseases"[MeSH] OR "Mouth Diseases"[MeSH] '
    'OR "Periodontal Diseases"[MeSH] OR "Stomatognathic Diseases"[MeSH] '
    'OR "Stomatognathic System"[MeSH] OR "Tooth"[MeSH] OR "Mouth"[MeSH] '
    'OR "Endodontics"[MeSH] OR "Orthodontics"[MeSH] OR "Prosthodontics"[MeSH] '
    'OR "Oral Surgical Procedures"[MeSH] OR "Pediatric Dentistry"[MeSH] '
    'OR "Dental Implants"[MeSH] OR "Dental Caries"[MeSH] '
    'OR "Periodontitis"[MeSH] OR "Gingivitis"[MeSH] OR "Pulpitis"[MeSH] '
    'OR "Malocclusion"[MeSH] OR "Tooth Loss"[MeSH] '
    'OR "Temporomandibular Joint Disorders"[MeSH] '
    'OR "Mouth Neoplasms"[MeSH] OR "Tongue Neoplasms"[MeSH])'
)


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
            with urllib.request.urlopen(url, timeout=90) as resp:
                return resp.read()
        except Exception:
            if attempt == max_retries - 1:
                raise
            time.sleep(backoff)
            backoff = min(backoff * 2, 30.0)
    raise RuntimeError("unreachable")


def _rate_sleep() -> None:
    time.sleep(0.35 if not API_KEY else 0.12)


def esearch_dated(min_date: str, max_date: str) -> tuple[str, str, int]:
    """Search dental MeSH restricted to a date range. Dates as YYYY/MM/DD."""
    term = f'{QUERY} AND ("{min_date}"[PDAT] : "{max_date}"[PDAT])'
    params = {
        "db": "pubmed", "term": term, "usehistory": "y",
        "retmode": "json", "retmax": 0,
    }
    url = f"{EUTILS}/esearch.fcgi?{_qs(params)}"
    data = json.loads(_http_get(url).decode("utf-8"))
    res = data.get("esearchresult", {})
    count = int(res.get("count", 0))
    webenv = res.get("webenv") or res.get("WebEnv")
    qkey = res.get("querykey") or res.get("QueryKey")
    return webenv, qkey, count


def efetch_window(webenv: str, qkey: str, retstart: int, retmax: int) -> bytes:
    params = {
        "db": "pubmed", "WebEnv": webenv, "query_key": qkey,
        "retstart": str(retstart), "retmax": str(retmax),
        "retmode": "xml",
    }
    url = f"{EUTILS}/efetch.fcgi?{_qs(params)}"
    return _http_get(url)


def _parse_articles(xml_bytes: bytes) -> list[dict]:
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError:
        return []
    out = []
    for art in root.findall(".//PubmedArticle"):
        try:
            pmid = (art.findtext(".//PMID") or "").strip()
            title = " ".join((art.findtext(".//ArticleTitle") or "").split()).strip()
            abs_parts = []
            for ab in art.findall(".//Abstract/AbstractText"):
                label = ab.get("Label") or ""
                txt = "".join(ab.itertext()).strip()
                if not txt:
                    continue
                abs_parts.append(f"{label}: {txt}" if label else txt)
            abstract = "\n".join(abs_parts).strip()
            journal = (art.findtext(".//Journal/Title") or "").strip()
            year = (art.findtext(".//PubDate/Year") or "").strip()
            if not year:
                medline_date = (art.findtext(".//PubDate/MedlineDate") or "").strip()
                year = medline_date[:4] if medline_date else ""
            mesh = [(m.text or "").strip()
                    for m in art.findall(".//MeshHeading/DescriptorName")]
            pub_types = [(pt.text or "").strip()
                         for pt in art.findall(".//PublicationType")]
            if not pmid:
                continue
            out.append({
                "pmid": pmid, "title": title, "abstract": abstract,
                "journal": journal, "year": year, "mesh": mesh,
                "pub_types": pub_types,
            })
        except Exception:
            continue
    return out


def _date_chunks(start_year: int, end_year: int):
    """Yield (min_date, max_date, webenv, qkey, count) chunks <= MAX_PER_WEBENV.

    Recursively splits date ranges: year -> half-year -> quarter -> month -> half-month.
    """
    def _recurse(min_date: str, max_date: str, depth: int = 0):
        if depth > 6:
            # Shouldn't happen; safety cap. Yield as-is and let the caller cope.
            webenv, qkey, count = esearch_dated(min_date, max_date)
            _rate_sleep()
            if count > 0:
                yield (min_date, max_date, webenv, qkey, min(count, MAX_PER_WEBENV))
            return
        webenv, qkey, count = esearch_dated(min_date, max_date)
        _rate_sleep()
        if count == 0:
            return
        if count <= MAX_PER_WEBENV:
            yield (min_date, max_date, webenv, qkey, count)
            return
        # Split the range
        from datetime import date, timedelta
        y0, m0, d0 = (int(x) for x in min_date.split("/"))
        y1, m1, d1 = (int(x) for x in max_date.split("/"))
        d_start = date(y0, m0, d0)
        d_end = date(y1, m1, d1)
        mid_ord = (d_start.toordinal() + d_end.toordinal()) // 2
        d_mid = date.fromordinal(mid_ord)
        d_mid_next = date.fromordinal(mid_ord + 1)
        left_max = f"{d_mid.year}/{d_mid.month:02d}/{d_mid.day:02d}"
        right_min = f"{d_mid_next.year}/{d_mid_next.month:02d}/{d_mid_next.day:02d}"
        yield from _recurse(min_date, left_max, depth + 1)
        yield from _recurse(right_min, max_date, depth + 1)

    for y in range(start_year, end_year - 1, -1):
        yield from _recurse(f"{y}/01/01", f"{y}/12/31")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, default=120_000)
    parser.add_argument("--batch", type=int, default=200)
    parser.add_argument("--start-year", type=int, default=2026)
    parser.add_argument("--end-year", type=int, default=1990)
    parser.add_argument("--out", default="pubmed_dental_abstracts.jsonl")
    args = parser.parse_args(argv)

    state_path = OUT_DIR / "state.json"
    out_path = OUT_DIR / args.out

    if state_path.exists():
        try:
            state = json.loads(state_path.read_text())
            seen = set(state.get("seen_pmids", []))
            parsed = state.get("n_parsed", 0)
            done_chunks = set(tuple(c) for c in state.get("done_chunks", []))
            print(f"[pubmed] resume parsed={parsed} done_chunks={len(done_chunks)} seen={len(seen)}")
        except Exception:
            state = None
            seen, parsed, done_chunks = set(), 0, set()
    else:
        state, seen, parsed, done_chunks = None, set(), 0, set()

    if state is None:
        state = {"n_parsed": 0, "done_chunks": [], "seen_pmids": []}

    started = time.time()
    last_log = time.time()

    try:
        with out_path.open("a", encoding="utf-8") as f_out:
            for chunk in _date_chunks(args.start_year, args.end_year):
                if parsed >= args.target:
                    break
                d0, d1, webenv, qkey, count = chunk
                chunk_key = (d0, d1)
                if chunk_key in done_chunks:
                    continue
                print(f"[pubmed] chunk {d0}..{d1}: count={count}")
                retstart = 0
                while retstart < count and parsed < args.target:
                    retmax = min(args.batch, count - retstart, args.target - parsed)
                    try:
                        xml_bytes = efetch_window(webenv, qkey, retstart, retmax)
                    except Exception as e:
                        print(f"[pubmed] efetch err at chunk={d0}..{d1} retstart={retstart}: {e} sleep 15s")
                        time.sleep(15)
                        continue
                    arts = _parse_articles(xml_bytes)
                    for a in arts:
                        pmid = a.get("pmid")
                        if not pmid or pmid in seen:
                            continue
                        seen.add(pmid)
                        f_out.write(json.dumps(a, ensure_ascii=False) + "\n")
                        parsed += 1
                    retstart += retmax
                    if time.time() - last_log > 15:
                        elapsed = time.time() - started
                        rate = parsed / max(1.0, elapsed)
                        eta = (args.target - parsed) / max(1.0, rate)
                        print(f"[pubmed] parsed={parsed}/{args.target} chunk={d0}..{d1} "
                              f"retstart={retstart}/{count} rate={rate:.0f}/s eta={eta:.0f}s")
                        state["n_parsed"] = parsed
                        state["seen_pmids"] = list(seen) if len(seen) < 500_000 else []
                        state_path.write_text(json.dumps(state))
                        last_log = time.time()
                    _rate_sleep()
                done_chunks.add(chunk_key)
                state["done_chunks"] = [list(c) for c in done_chunks]
                state["n_parsed"] = parsed
                state["seen_pmids"] = list(seen) if len(seen) < 500_000 else []
                state_path.write_text(json.dumps(state))
    finally:
        state["n_parsed"] = parsed
        state["done_chunks"] = [list(c) for c in done_chunks]
        state["seen_pmids"] = list(seen) if len(seen) < 500_000 else []
        state_path.write_text(json.dumps(state))
    print(f"[pubmed] DONE parsed={parsed} -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
