"""FINAL merge: combine ALL sources into definitive CPT / SFT / DPO datasets.

Outputs to FINAL/:
  FINAL/dental_cpt_final.jsonl       — all CPT chunks, deduped, schema: {text, source}
  FINAL/dental_sft_final.jsonl       — all SFT rows, deduped, schema: {question, answer, source, topic}
  FINAL/dental_sft_final_train.jsonl — 95% train split
  FINAL/dental_sft_final_val.jsonl   — 5% val split
  FINAL/dental_dpo_final.jsonl       — all DPO rows, deduped, schema: {prompt, chosen, rejected, source, topic}
  FINAL/dental_dpo_final_train.jsonl — 95% train split
  FINAL/dental_dpo_final_val.jsonl   — 5% val split
  FINAL/report.json                  — counts per source + dedup stats
"""
from __future__ import annotations

import hashlib
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FINAL = ROOT / "FINAL"
FINAL.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Source manifests
# ---------------------------------------------------------------------------

CPT_SOURCES = [
    # Existing mega (already contains: lexicon_shift, ctgov, pubmed, statpearls, dental_cpt_books)
    ROOT / "cpt_prepared" / "dental_cpt_mega.jsonl",
    # New scrapers (this session)
    ROOT / "cpt_prepared" / "openalex_dental" / "openalex_dental_cpt.jsonl",
    ROOT / "cpt_prepared" / "pmc_dental" / "pmc_dental_cpt.jsonl",
    ROOT / "cpt_prepared" / "wikipedia_dental" / "wikipedia_dental_cpt.jsonl",
    ROOT / "cpt_prepared" / "quick_wins" / "hf_dental_extra_cpt.jsonl",
]

SFT_SOURCES = [
    # Existing mega (already contains: medmcqa, headqa, periodontal, chatdoctor, lexicon_shift,
    # birdiebyte, jonathankang, vuha2003, pubmedqa, ctgov, pubmed, synthetic_mesh)
    ROOT / "rl_prepared" / "dental_sft_mega.jsonl",
    # New SFT sources (this session)
    ROOT / "cpt_prepared" / "quick_wins" / "hf_dental_extra_sft.jsonl",
    ROOT / "cpt_prepared" / "quick_wins" / "medqa_dental_sft.jsonl",
    ROOT / "cpt_prepared" / "quick_wins" / "medical_instruction_dental_sft.jsonl",
    # Textbook-generated SFT (Gemini)
    ROOT / "cpt_prepared" / "textbook_sft_combined.jsonl",
]

DPO_SOURCES = [
    # Original baseline DPO
    ROOT / "rl_prepared" / "rl_dpo.jsonl",
    # Textbook-generated DPO (Gemini)
    ROOT / "cpt_prepared" / "textbook_dpo_combined.jsonl",
]

# Outputs
OUT_CPT = FINAL / "dental_cpt_final.jsonl"
OUT_SFT = FINAL / "dental_sft_final.jsonl"
OUT_SFT_TRAIN = FINAL / "dental_sft_final_train.jsonl"
OUT_SFT_VAL = FINAL / "dental_sft_final_val.jsonl"
OUT_DPO = FINAL / "dental_dpo_final.jsonl"
OUT_DPO_TRAIN = FINAL / "dental_dpo_final_train.jsonl"
OUT_DPO_VAL = FINAL / "dental_dpo_final_val.jsonl"
REPORT_PATH = FINAL / "report.json"

VAL_FRAC = 0.05
SEED = 42

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _norm(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9À-ɏ一-鿿\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _hash(s: str) -> str:
    return hashlib.sha1(_norm(s).encode("utf-8")).hexdigest()


def _iter_jsonl(path: Path):
    if not path.exists():
        print(f"  [skip] {path} not found")
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


def _validate_sft(r: dict) -> dict | None:
    """Coerce to {question, answer, source, topic} schema."""
    q = (r.get("question") or r.get("prompt") or "").strip()
    a = (r.get("answer") or r.get("response") or "").strip()
    src = (r.get("source") or "unknown").strip()
    topic = (r.get("topic") or "Dentistry").strip()
    if len(q) < 10 or len(a) < 20:
        return None
    return {"question": q, "answer": a, "source": src, "topic": topic}


def _validate_dpo(r: dict) -> dict | None:
    """Coerce to {prompt, chosen, rejected, source, topic} schema."""
    p = (r.get("prompt") or r.get("question") or "").strip()
    c = (r.get("chosen") or "").strip()
    rj = (r.get("rejected") or "").strip()
    src = (r.get("source") or "unknown").strip()
    topic = (r.get("topic") or "Dentistry").strip()
    if len(p) < 10 or len(c) < 20 or len(rj) < 20:
        return None
    return {"prompt": p, "chosen": c, "rejected": rj, "source": src, "topic": topic}


def _validate_cpt(r: dict) -> dict | None:
    """Coerce to {text, source} schema."""
    t = (r.get("text") or "").strip()
    src = (r.get("source") or "unknown").strip()
    if len(t) < 50:
        return None
    return {"text": t, "source": src}


# ---------------------------------------------------------------------------
# Merge phases
# ---------------------------------------------------------------------------

def merge_cpt() -> tuple[int, dict]:
    print("\n=== MERGING CPT ===")
    seen: set[str] = set()
    per_source: dict[str, int] = {}
    per_file: dict[str, int] = {}
    total_in = 0
    total_out = 0

    with OUT_CPT.open("w", encoding="utf-8") as f_out:
        for path in CPT_SOURCES:
            added = 0
            file_in = 0
            for r in _iter_jsonl(path):
                file_in += 1
                v = _validate_cpt(r)
                if not v:
                    continue
                h = _hash(v["text"][:500])
                if h in seen:
                    continue
                seen.add(h)
                f_out.write(json.dumps(v, ensure_ascii=False) + "\n")
                added += 1
                src_prefix = v["source"].split(":")[0] if ":" in v["source"] else v["source"]
                per_source[src_prefix] = per_source.get(src_prefix, 0) + 1
            per_file[path.name] = added
            total_in += file_in
            total_out += added
            print(f"  {path.name}: in={file_in:>8}  added={added:>8}")

    print(f"  TOTAL: in={total_in} -> kept={total_out} (dups removed: {total_in - total_out})")
    return total_out, {"per_file": per_file, "per_source_prefix": per_source}


def merge_sft() -> tuple[int, int, int, dict]:
    print("\n=== MERGING SFT ===")
    seen: set[str] = set()
    rows: list[dict] = []
    per_file: dict[str, int] = {}
    per_source: Counter = Counter()
    total_in = 0

    for path in SFT_SOURCES:
        added = 0
        file_in = 0
        for r in _iter_jsonl(path):
            file_in += 1
            v = _validate_sft(r)
            if not v:
                continue
            h = _hash(v["question"])
            if h in seen:
                continue
            seen.add(h)
            rows.append(v)
            per_source[v["source"]] += 1
            added += 1
        per_file[path.name] = added
        total_in += file_in
        print(f"  {path.name}: in={file_in:>8}  added={added:>8}")

    # Shuffle + split
    random.seed(SEED)
    random.shuffle(rows)
    n_val = max(1, int(len(rows) * VAL_FRAC))
    val = rows[:n_val]
    train = rows[n_val:]

    with OUT_SFT.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with OUT_SFT_TRAIN.open("w", encoding="utf-8") as f:
        for r in train:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with OUT_SFT_VAL.open("w", encoding="utf-8") as f:
        for r in val:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"  TOTAL: in={total_in} -> kept={len(rows)} (dups removed: {total_in - len(rows)})")
    print(f"  split: train={len(train)}  val={len(val)}")
    return len(rows), len(train), len(val), {
        "per_file": per_file,
        "top_sources": per_source.most_common(30),
    }


def merge_dpo() -> tuple[int, int, int, dict]:
    print("\n=== MERGING DPO ===")
    seen: set[str] = set()
    rows: list[dict] = []
    per_file: dict[str, int] = {}
    per_source: Counter = Counter()
    total_in = 0

    for path in DPO_SOURCES:
        added = 0
        file_in = 0
        for r in _iter_jsonl(path):
            file_in += 1
            v = _validate_dpo(r)
            if not v:
                continue
            h = _hash(v["prompt"])
            if h in seen:
                continue
            seen.add(h)
            rows.append(v)
            per_source[v["source"]] += 1
            added += 1
        per_file[path.name] = added
        total_in += file_in
        print(f"  {path.name}: in={file_in:>8}  added={added:>8}")

    random.seed(SEED)
    random.shuffle(rows)
    n_val = max(1, int(len(rows) * VAL_FRAC))
    val = rows[:n_val]
    train = rows[n_val:]

    with OUT_DPO.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with OUT_DPO_TRAIN.open("w", encoding="utf-8") as f:
        for r in train:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with OUT_DPO_VAL.open("w", encoding="utf-8") as f:
        for r in val:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"  TOTAL: in={total_in} -> kept={len(rows)} (dups removed: {total_in - len(rows)})")
    print(f"  split: train={len(train)}  val={len(val)}")
    return len(rows), len(train), len(val), {
        "per_file": per_file,
        "top_sources": per_source.most_common(30),
    }


def main() -> int:
    print(f"FINAL merge -> {FINAL}\n")

    cpt_n, cpt_info = merge_cpt()
    sft_n, sft_train, sft_val, sft_info = merge_sft()
    dpo_n, dpo_train, dpo_val, dpo_info = merge_dpo()

    report = {
        "summary": {
            "cpt_total": cpt_n,
            "sft_total": sft_n,
            "sft_train": sft_train,
            "sft_val": sft_val,
            "dpo_total": dpo_n,
            "dpo_train": dpo_train,
            "dpo_val": dpo_val,
        },
        "cpt": cpt_info,
        "sft": sft_info,
        "dpo": dpo_info,
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n" + "=" * 60)
    print("FINAL DATASETS")
    print("=" * 60)
    print(f"  CPT: {cpt_n:>8,}  -> {OUT_CPT}")
    print(f"  SFT: {sft_n:>8,}  -> {OUT_SFT}")
    print(f"         train={sft_train:,}  val={sft_val:,}")
    print(f"  DPO: {dpo_n:>8,}  -> {OUT_DPO}")
    print(f"         train={dpo_train:,}  val={dpo_val:,}")
    print(f"\n  report -> {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
