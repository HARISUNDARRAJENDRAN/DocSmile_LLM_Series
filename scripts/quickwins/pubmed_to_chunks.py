"""Convert PubMed dental raw dump -> CPT chunks + SFT abstract Q&A pairs."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Iterator

ROOT = Path(__file__).resolve().parents[2]
PM_DIR = ROOT / "cpt_prepared" / "pubmed_dental"
RAW = PM_DIR / "pubmed_dental_abstracts.jsonl"

OUT_CPT = PM_DIR / "pubmed_dental_cpt.jsonl"
OUT_SFT = PM_DIR / "pubmed_dental_sft.jsonl"


def article_to_cpt(art: dict) -> dict | None:
    title = (art.get("title") or "").strip()
    abstract = (art.get("abstract") or "").strip()
    if not abstract or len(abstract) < 150:
        return None
    journal = (art.get("journal") or "").strip()
    year = (art.get("year") or "").strip()
    header = title
    if journal or year:
        header += f" ({journal} {year})".rstrip()
    text = header + "\n\n" + abstract
    pmid = art.get("pmid") or ""
    return {"text": text, "source": f"pubmed:{pmid}"}


def article_to_sft(art: dict) -> Iterator[dict]:
    title = (art.get("title") or "").strip()
    abstract = (art.get("abstract") or "").strip()
    if not title or not abstract or len(abstract) < 200:
        return
    pmid = art.get("pmid") or ""
    source = f"pubmed:{pmid}"
    topic = "Dental Research Literature"

    # Q: summarise abstract; A: abstract
    yield {
        "question": f"Summarise the key findings of the dental research paper titled '{title}'.",
        "answer": abstract,
        "source": source,
        "topic": topic,
    }


def main() -> int:
    if not RAW.exists():
        print(f"[pubmed->cpt] missing input: {RAW}")
        return 1
    n_in = n_cpt = n_sft = 0
    with RAW.open(encoding="utf-8") as f_in, \
            OUT_CPT.open("w", encoding="utf-8") as f_cpt, \
            OUT_SFT.open("w", encoding="utf-8") as f_sft:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            try:
                art = json.loads(line)
            except Exception:
                continue
            n_in += 1
            cpt = article_to_cpt(art)
            if cpt:
                f_cpt.write(json.dumps(cpt, ensure_ascii=False) + "\n")
                n_cpt += 1
            for sft in article_to_sft(art):
                f_sft.write(json.dumps(sft, ensure_ascii=False) + "\n")
                n_sft += 1
    print(f"[pubmed->cpt] read {n_in} -> CPT={n_cpt} SFT={n_sft}")
    print(f"  -> {OUT_CPT}")
    print(f"  -> {OUT_SFT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
