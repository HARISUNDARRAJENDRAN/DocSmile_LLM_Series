"""Orchestrator for the CPT book-cleaning pipeline.

Usage:
  # Deterministic pass over all books (no Gemini), produce per-book chunks + audit:
  python -m scripts.quickwins.cpt_cleaner.run --stage deterministic

  # Single book test:
  python -m scripts.quickwins.cpt_cleaner.run --stage deterministic \\
      --book "Color Atlas of Biochemistry"

  # Full pipeline (deterministic + Gemini) over the first 5 books:
  python -m scripts.quickwins.cpt_cleaner.run --stage full --limit 5

Outputs:
  cpt_prepared/books_cleaned/{book_stem}/det_chunks.jsonl   (post-deterministic)
  cpt_prepared/books_cleaned/{book_stem}/det_audit.jsonl    (blocks dropped/kept)
  cpt_prepared/books_cleaned/{book_stem}/det_stats.json
  cpt_prepared/books_cleaned/{book_stem}/final_chunks.jsonl (post-Gemini)
  cpt_prepared/books_cleaned/{book_stem}/final_stats.json
  cpt_prepared/dental_cpt_books.jsonl                       (aggregated, deduped)
  logs/cleaner/run_{ts}.log
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from . import deterministic as det

ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = ROOT / "cpt_prepared" / "core_cpt_text_cleaned"
OUT_DIR = ROOT / "cpt_prepared" / "books_cleaned"
LOG_DIR = ROOT / "logs" / "cleaner"
AGG_PATH = ROOT / "cpt_prepared" / "dental_cpt_books.jsonl"
AGG_GEMINI_PATH = ROOT / "cpt_prepared" / "dental_cpt_books_gemini.jsonl"
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)


def _safe_stem(name: str) -> str:
    return name.replace("/", "_").replace("\\", "_")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _list_books(filter_substr: str | None = None) -> list[Path]:
    paths = sorted(SRC_DIR.glob("*.txt"))
    if filter_substr:
        sub = filter_substr.lower()
        paths = [p for p in paths if sub in p.stem.lower()]
    return paths


def run_deterministic_one(path: Path, target_chars: int, min_chunk_chars: int,
                          min_prose_quality: float) -> dict:
    """Process one book through the deterministic pipeline."""
    stem = _safe_stem(path.stem)
    book_out = OUT_DIR / stem
    book_out.mkdir(parents=True, exist_ok=True)
    chunks, audit, stats = det.process_file(
        path, target_chars=target_chars, min_chunk_chars=min_chunk_chars,
        min_prose_quality=min_prose_quality,
    )
    _write_jsonl(book_out / "det_chunks.jsonl", chunks)
    _write_jsonl(book_out / "det_audit.jsonl", audit)
    (book_out / "det_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return {
        "book": stem,
        "raw_chars": stats["raw_chars"],
        "kept_chars": stats["chunk_chars"],
        "retention": stats["retention_ratio"],
        "n_chunks": stats["n_chunks"],
        "tag_block_counts": stats["tag_block_counts"],
        "tag_char_totals": stats["tag_char_totals"],
    }


def run_gemini_one(path: Path) -> dict:
    """Run Gemini cleanup on the post-deterministic chunks for a single book."""
    from .gemini_pass import clean_chunks
    from ..common.gemini import GeminiPool
    stem = _safe_stem(path.stem)
    book_out = OUT_DIR / stem
    det_path = book_out / "det_chunks.jsonl"
    if not det_path.exists():
        return {"book": stem, "error": "no det_chunks"}
    chunks = [json.loads(l) for l in det_path.open(encoding="utf-8") if l.strip()]
    pool = GeminiPool()
    cleaned, stats = clean_chunks(chunks, pool=pool)
    _write_jsonl(book_out / "final_chunks.jsonl", cleaned)
    (book_out / "final_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return {"book": stem, **stats}


def aggregate(stage: str = "deterministic") -> dict:
    """Merge per-book outputs into a single aggregate JSONL with dedup."""
    if stage == "gemini":
        in_name = "final_chunks.jsonl"
        out_path = AGG_GEMINI_PATH
    else:
        in_name = "det_chunks.jsonl"
        out_path = AGG_PATH
    seen: set[str] = set()
    rows: list[dict] = []
    n_dup = 0
    for book_dir in sorted(OUT_DIR.iterdir()):
        p = book_dir / in_name
        if not p.exists():
            continue
        for line in p.open(encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            txt = r.get("text", "")
            if not txt:
                continue
            key = txt[:500].lower()
            if key in seen:
                n_dup += 1
                continue
            seen.add(key)
            rows.append({"text": txt, "source": r.get("source", book_dir.name)})
    _write_jsonl(out_path, rows)
    return {"rows": len(rows), "dups_dropped": n_dup, "out": str(out_path)}


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=["deterministic", "gemini", "full", "aggregate"],
                   default="deterministic")
    p.add_argument("--book", help="Substring filter to select a single book by name")
    p.add_argument("--limit", type=int, default=0, help="Process only first N books")
    p.add_argument("--target-chars", type=int, default=6000)
    p.add_argument("--min-chunk-chars", type=int, default=800)
    p.add_argument("--min-prose-quality", type=float, default=0.35)
    p.add_argument("--force", action="store_true", help="Re-run even if outputs exist")
    args = p.parse_args(argv)

    log = LOG_DIR / f"run_{time.strftime('%Y%m%d_%H%M%S')}.log"
    log_f = log.open("w", encoding="utf-8")
    def L(msg: str) -> None:
        print(msg, flush=True)
        log_f.write(msg + "\n")
        log_f.flush()

    if args.stage == "aggregate":
        for s in ("deterministic", "gemini"):
            res = aggregate(s)
            L(f"[agg] stage={s} -> {res}")
        return 0

    books = _list_books(args.book)
    if args.limit > 0:
        books = books[:args.limit]
    L(f"[run] stage={args.stage}  books={len(books)}")

    det_summaries: list[dict] = []
    gem_summaries: list[dict] = []

    for path in books:
        stem = _safe_stem(path.stem)
        det_path = OUT_DIR / stem / "det_chunks.jsonl"
        if args.stage in ("deterministic", "full"):
            if det_path.exists() and not args.force:
                L(f"[det] cached: {stem}")
                stats_p = OUT_DIR / stem / "det_stats.json"
                if stats_p.exists():
                    s = json.loads(stats_p.read_text(encoding="utf-8"))
                    det_summaries.append({"book": stem, "raw_chars": s.get("raw_chars"),
                                          "kept_chars": s.get("chunk_chars"),
                                          "retention": s.get("retention_ratio"),
                                          "n_chunks": s.get("n_chunks")})
                continue
            try:
                summary = run_deterministic_one(
                    path, args.target_chars, args.min_chunk_chars, args.min_prose_quality)
                det_summaries.append(summary)
                L(f"[det] {stem}: kept_chars={summary['kept_chars']:>8}/"
                  f"{summary['raw_chars']:>8} ({summary['retention']*100:.1f}%)  "
                  f"chunks={summary['n_chunks']}")
            except Exception as e:
                L(f"[det] {stem} FAILED: {e}")

    if args.stage in ("gemini", "full"):
        for path in books:
            stem = _safe_stem(path.stem)
            final_path = OUT_DIR / stem / "final_chunks.jsonl"
            if final_path.exists() and not args.force:
                L(f"[gem] cached: {stem}")
                continue
            try:
                summary = run_gemini_one(path)
                gem_summaries.append(summary)
                L(f"[gem] {stem}: {summary}")
            except Exception as e:
                L(f"[gem] {stem} FAILED: {e}")

    # Always aggregate at the end
    res_det = aggregate("deterministic")
    L(f"[agg] deterministic -> {res_det}")
    if args.stage in ("gemini", "full"):
        res_gem = aggregate("gemini")
        L(f"[agg] gemini -> {res_gem}")

    # ----- final summary -----
    L("")
    L("=" * 80)
    L("SUMMARY")
    L("=" * 80)
    if det_summaries:
        from collections import Counter
        all_tags = Counter()
        all_chars = Counter()
        total_raw = total_kept = 0
        for s in det_summaries:
            full = OUT_DIR / s["book"] / "det_stats.json"
            if full.exists():
                d = json.loads(full.read_text(encoding="utf-8"))
                for t, n in d.get("tag_block_counts", {}).items():
                    all_tags[t] += n
                for t, n in d.get("tag_char_totals", {}).items():
                    all_chars[t] += n
                total_raw += d.get("raw_chars", 0)
                total_kept += d.get("chunk_chars", 0)
        L(f"books processed: {len(det_summaries)}")
        L(f"total raw chars : {total_raw:,}")
        L(f"total kept chars: {total_kept:,}  ({100*total_kept/max(1,total_raw):.1f}%)")
        L(f"block tag counts (top 10):")
        for t, n in all_tags.most_common(10):
            L(f"  {t:<22} blocks={n:>7}  chars={all_chars[t]:>10,}")

    log_f.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
