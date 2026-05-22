"""StatPearls dental chapter scraper from NCBI Bookshelf.

StatPearls is the open-access medical reference on Bookshelf. Each chapter is at
  https://www.ncbi.nlm.nih.gov/books/{NBK_ID}/
and the OAI-PMH endpoint at /entrez/eutils/efetch.fcgi?db=books&id={NBK_ID}
returns BookDoc XML. The cleaner path is the HTML page itself.

We use the existing dental-keyword search via Bookshelf search to enumerate chapters,
then download each chapter as plain text.

License: CC BY-NC-ND 4.0 (research training OK; no redistribution).
"""
from __future__ import annotations

import json
import re
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "cpt_prepared" / "statpearls_dental"
RAW_DIR = OUT_DIR / "_raw"
OUT_DIR.mkdir(parents=True, exist_ok=True)
RAW_DIR.mkdir(parents=True, exist_ok=True)

ESEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
BOOK_URL = "https://www.ncbi.nlm.nih.gov/books/{}/"


# Dental keywords for StatPearls Bookshelf search
KEYWORDS = [
    "dental", "dentistry", "tooth", "teeth", "periodontal", "periodontitis",
    "gingivitis", "endodontic", "orthodontic", "prosthodontic", "oral surgery",
    "dental caries", "dental implant", "oral cancer", "oral mucosa",
    "tongue", "salivary", "temporomandibular", "TMJ", "occlusion",
    "malocclusion", "wisdom tooth", "dental anesthesia", "dental trauma",
    "dental abscess", "oral pathology", "oral lichen planus", "oral candidiasis",
    "leukoplakia", "ludwig angina", "fluorosis", "cleft palate",
]


class _TextExtractor(HTMLParser):
    SKIP = {"script", "style", "noscript", "header", "footer", "nav", "aside"}

    def __init__(self):
        super().__init__()
        self.parts: list[str] = []
        self._skipping = 0
        self._in_body = False

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self._skipping += 1

    def handle_endtag(self, tag):
        if tag in self.SKIP and self._skipping > 0:
            self._skipping -= 1

    def handle_data(self, data):
        if self._skipping:
            return
        if data:
            self.parts.append(data)


def _http_get(url: str, max_retries: int = 4) -> bytes:
    backoff = 1.0
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "docsmile-scraper/1.0"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                return resp.read()
        except Exception:
            if attempt == max_retries - 1:
                raise
            time.sleep(backoff)
            backoff *= 2
    raise RuntimeError("unreachable")


def _search_book_ids(keyword: str, retmax: int = 200) -> list[str]:
    term = f"({keyword}) AND StatPearls[BOOK]"
    params = {"db": "books", "term": term, "retmode": "json", "retmax": str(retmax)}
    url = ESEARCH + "?" + urllib.parse.urlencode(params)
    try:
        data = json.loads(_http_get(url).decode("utf-8"))
    except Exception as e:
        print(f"[statpearls] esearch err for {keyword}: {e}")
        return []
    return data.get("esearchresult", {}).get("idlist", [])


def _fetch_chapter_text(nbk_id: str) -> str:
    """Download chapter as HTML and extract plain text."""
    url = BOOK_URL.format(nbk_id)
    html = _http_get(url).decode("utf-8", errors="replace")
    parser = _TextExtractor()
    parser.feed(html)
    text = "\n".join(parser.parts)
    # Normalise whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return "\n".join(lines)


def main() -> int:
    state_path = OUT_DIR / "state.json"
    out_path = OUT_DIR / "statpearls_dental_chapters.jsonl"

    # Collect chapter IDs across all keywords (dedup)
    if state_path.exists():
        state = json.loads(state_path.read_text())
    else:
        state = {"ids": [], "fetched": []}

    if not state.get("ids"):
        all_ids: set[str] = set()
        for kw in KEYWORDS:
            ids = _search_book_ids(kw)
            print(f"[statpearls] '{kw}': {len(ids)} hits")
            all_ids.update(ids)
            time.sleep(0.4)
        state["ids"] = sorted(all_ids)
        state_path.write_text(json.dumps(state))
        print(f"[statpearls] unique dental chapter IDs: {len(state['ids'])}")

    fetched = set(state.get("fetched", []))
    with out_path.open("a", encoding="utf-8") as f_out:
        for i, nbk in enumerate(state["ids"]):
            if nbk in fetched:
                continue
            try:
                txt = _fetch_chapter_text(nbk)
            except Exception as e:
                print(f"[statpearls] {nbk} err: {e}")
                continue
            if len(txt) < 500:
                fetched.add(nbk)
                continue
            rec = {"source": f"statpearls:NBK{nbk}", "text": txt}
            f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fetched.add(nbk)
            if (i + 1) % 10 == 0:
                state["fetched"] = sorted(fetched)
                state_path.write_text(json.dumps(state))
                print(f"[statpearls] {len(fetched)}/{len(state['ids'])} done")
            time.sleep(0.5)
    state["fetched"] = sorted(fetched)
    state_path.write_text(json.dumps(state))
    print(f"[statpearls] DONE {len(fetched)} chapters -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
