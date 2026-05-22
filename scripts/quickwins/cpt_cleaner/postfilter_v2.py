"""Tighten the post-filter on the textbook CPT corpus.

Reads `cpt_prepared/dental_cpt_books.jsonl` and produces
`cpt_prepared/dental_cpt_books_v2.jsonl` with stricter cuts:
  - drop chunks with quality < 0.75 (when present in per-book det_chunks)
  - drop chunks whose page-reference density is high (likely index leakage)
  - drop chunks whose token diversity is very low

Most chunks in the original aggregate strip the `quality` field during
aggregation, so we re-scan from per-book det_chunks.jsonl and rebuild.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BOOKS_DIR = ROOT / "cpt_prepared" / "books_cleaned"
OUT_PATH = ROOT / "cpt_prepared" / "dental_cpt_books_v2.jsonl"

PAGE_REF_RE = re.compile(r"\b\d{2,4}\b")
WORD_RE = re.compile(r"[A-Za-zÀ-ÿ]+")


def page_ref_density(text: str) -> float:
    n_words = len(WORD_RE.findall(text))
    n_pages = len(PAGE_REF_RE.findall(text))
    if n_words < 20:
        return 1.0
    return n_pages / n_words


def vocab_diversity(text: str) -> float:
    words = [w.lower() for w in WORD_RE.findall(text)]
    if len(words) < 30:
        return 0.0
    return len(set(words)) / len(words)


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())[:500]


def main() -> int:
    seen: set[str] = set()
    kept = 0
    dropped_q = dropped_index = dropped_div = dropped_dup = dropped_short = 0
    with OUT_PATH.open("w", encoding="utf-8") as out_f:
        for book_dir in sorted(BOOKS_DIR.iterdir()):
            chunks_p = book_dir / "det_chunks.jsonl"
            if not chunks_p.exists():
                continue
            for line in chunks_p.open(encoding="utf-8"):
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                text = r.get("text", "")
                q = r.get("quality", 1.0)
                if not text or len(text) < 400:
                    dropped_short += 1
                    continue
                if q < 0.75:
                    dropped_q += 1
                    continue
                # Index leakage: lots of page-number-like tokens mixed with short words
                if page_ref_density(text) > 0.10 and len(text) < 3000:
                    dropped_index += 1
                    continue
                if vocab_diversity(text) < 0.30:
                    dropped_div += 1
                    continue
                key = hashlib.sha1(normalize(text).encode()).hexdigest()
                if key in seen:
                    dropped_dup += 1
                    continue
                seen.add(key)
                out_f.write(json.dumps({
                    "text": text,
                    "source": r.get("source", book_dir.name),
                    "book": r.get("book", book_dir.name),
                    "chunk_idx": r.get("chunk_idx"),
                    "quality": q,
                }, ensure_ascii=False) + "\n")
                kept += 1
    print(f"[v2] kept={kept}")
    print(f"     dropped_short={dropped_short}")
    print(f"     dropped_low_quality={dropped_q}")
    print(f"     dropped_index_pattern={dropped_index}")
    print(f"     dropped_low_diversity={dropped_div}")
    print(f"     dropped_duplicates={dropped_dup}")
    print(f"     -> {OUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
