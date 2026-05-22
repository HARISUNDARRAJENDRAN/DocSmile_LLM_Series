"""Mega-merge: combine quick_wins + CT.gov + PubMed (when available) into
final dental SFT and CPT corpora with dedup + train/val split.

Outputs (in rl_prepared/):
  dental_sft_mega.jsonl       (all SFT, deduped)
  dental_sft_mega_train.jsonl (95% shuffled split)
  dental_sft_mega_val.jsonl   (5% shuffled split)

Outputs (in cpt_prepared/):
  dental_cpt_mega.jsonl       (all CPT chunks, deduped)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

SFT_SOURCES = [
    # quick-wins (HF datasets)
    ROOT / "cpt_prepared" / "quick_wins" / "medmcqa_dental.jsonl",
    ROOT / "cpt_prepared" / "quick_wins" / "headqa_dental.jsonl",
    ROOT / "cpt_prepared" / "quick_wins" / "periodontal_reasoning_40k.jsonl",
    ROOT / "cpt_prepared" / "quick_wins" / "chatdoctor_dental.jsonl",
    ROOT / "cpt_prepared" / "quick_wins" / "lexicon_shift_qna.jsonl",
    ROOT / "cpt_prepared" / "quick_wins" / "birdiebyte_dental_implants.jsonl",
    ROOT / "cpt_prepared" / "quick_wins" / "jonathankang_dental_qa.jsonl",
    ROOT / "cpt_prepared" / "quick_wins" / "vuha2003_medmcqa_dental.jsonl",
    ROOT / "cpt_prepared" / "quick_wins" / "pubmedqa_dental.jsonl",
    # CT.gov SFT
    ROOT / "cpt_prepared" / "clinicaltrials_dental" / "ctgov_dental_sft.jsonl",
    # PubMed SFT (only present once converted)
    ROOT / "cpt_prepared" / "pubmed_dental" / "pubmed_dental_sft.jsonl",
    # Synthetic from MeSH descriptors
    ROOT / "cpt_prepared" / "synthetic_mesh_dental" / "synthetic_mesh_dental_sft.jsonl",
]

CPT_SOURCES = [
    ROOT / "cpt_prepared" / "quick_wins_cpt" / "lexicon_shift_fb_articles.jsonl",
    ROOT / "cpt_prepared" / "quick_wins_cpt" / "lexicon_shift_sinhala_cpt.jsonl",
    ROOT / "cpt_prepared" / "clinicaltrials_dental" / "ctgov_dental_cpt.jsonl",
    ROOT / "cpt_prepared" / "pubmed_dental" / "pubmed_dental_cpt.jsonl",
    ROOT / "cpt_prepared" / "statpearls_dental" / "statpearls_dental_chapters.jsonl",
    # 113 textbooks cleaned via scripts.quickwins.cpt_cleaner
    ROOT / "cpt_prepared" / "dental_cpt_books.jsonl",
]

EXISTING_SFT = ROOT / "rl_prepared" / "rl_sft.jsonl"

OUT_SFT = ROOT / "rl_prepared" / "dental_sft_mega.jsonl"
OUT_TRAIN = ROOT / "rl_prepared" / "dental_sft_mega_train.jsonl"
OUT_VAL = ROOT / "rl_prepared" / "dental_sft_mega_val.jsonl"
OUT_CPT = ROOT / "cpt_prepared" / "dental_cpt_mega.jsonl"
REPORT = ROOT / "rl_prepared" / "dental_sft_mega_report.json"


def _norm(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9À-ɏ一-鿿\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _hash(s: str) -> str:
    return hashlib.sha1(_norm(s).encode("utf-8")).hexdigest()


def _iter_jsonl(path: Path):
    if not path.exists():
        return
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--val-frac", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skip-existing-dedup", action="store_true")
    args = p.parse_args(argv)

    # Seed dedup with existing SFT
    seen: set[str] = set()
    if not args.skip_existing_dedup and EXISTING_SFT.exists():
        for r in _iter_jsonl(EXISTING_SFT):
            q = r.get("question") or r.get("prompt") or ""
            if q:
                seen.add(_hash(q))
        print(f"[mega] seeded dedup with {len(seen)} existing SFT questions")

    # --- SFT ---
    sft_rows: list[dict] = []
    src_counter: Counter = Counter()
    per_file: dict[str, int] = {}

    for path in SFT_SOURCES:
        added = 0
        for r in _iter_jsonl(path):
            q = r.get("question") or ""
            a = r.get("answer") or ""
            if not q or not a:
                continue
            h = _hash(q)
            if h in seen:
                continue
            seen.add(h)
            sft_rows.append(r)
            src_counter[r.get("source", "?")] += 1
            added += 1
        per_file[path.name] = added
        print(f"[mega] SFT {path.name}: +{added}")

    random.seed(args.seed)
    random.shuffle(sft_rows)
    n_val = max(1, int(len(sft_rows) * args.val_frac))
    val = sft_rows[:n_val]
    train = sft_rows[n_val:]

    with OUT_SFT.open("w", encoding="utf-8") as f:
        for r in sft_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with OUT_TRAIN.open("w", encoding="utf-8") as f:
        for r in train:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with OUT_VAL.open("w", encoding="utf-8") as f:
        for r in val:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[mega] SFT total: {len(sft_rows)}  train={len(train)}  val={len(val)}")
    print(f"  -> {OUT_SFT}")
    print(f"  -> {OUT_TRAIN}")
    print(f"  -> {OUT_VAL}")

    # --- CPT ---
    cpt_seen: set[str] = set()
    cpt_rows: list[dict] = []
    cpt_per_file: dict[str, int] = {}
    for path in CPT_SOURCES:
        added = 0
        for r in _iter_jsonl(path):
            text = r.get("text") or ""
            if not text:
                continue
            h = _hash(text[:500])
            if h in cpt_seen:
                continue
            cpt_seen.add(h)
            cpt_rows.append(r)
            added += 1
        cpt_per_file[path.name] = added
        print(f"[mega] CPT {path.name}: +{added}")
    with OUT_CPT.open("w", encoding="utf-8") as f:
        for r in cpt_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[mega] CPT total: {len(cpt_rows)}  -> {OUT_CPT}")

    report = {
        "sft_total": len(sft_rows),
        "sft_train": len(train),
        "sft_val": len(val),
        "cpt_total": len(cpt_rows),
        "sft_per_file": per_file,
        "cpt_per_file": cpt_per_file,
        "sft_top_sources": src_counter.most_common(30),
    }
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[mega] report -> {REPORT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
