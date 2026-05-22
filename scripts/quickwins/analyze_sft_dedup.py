"""Analyze duplicate + near-duplicate + contradiction state of SFT corpora.

Reports:
  - exact duplicate (question, answer) pairs
  - exact duplicate questions with differing answers (potential contradictions)
  - near-duplicate questions (same after normalization)
  - distribution of duplicate-multiplicity (how many copies of each unique Q)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def norm(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9À-ɏ一-鿿\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def qhash(s: str) -> str:
    return hashlib.sha1(norm(s).encode()).hexdigest()


def analyze(path: Path) -> dict:
    if not path.exists():
        return {"error": f"missing: {path}"}
    n_rows = 0
    q_norm_counts: Counter = Counter()           # how many times each normalized Q appears
    qa_counts: Counter = Counter()               # exact (Q, A) duplicates
    q_to_answers: dict[str, set[str]] = defaultdict(set)  # for contradiction detection
    q_to_sources: dict[str, set[str]] = defaultdict(set)
    src_counts: Counter = Counter()

    with path.open(encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            q = r.get("question") or r.get("prompt") or ""
            a = r.get("answer") or r.get("chosen") or ""
            src = r.get("source") or "?"
            if not q or not a:
                continue
            n_rows += 1
            qh = qhash(q)
            ah_short = hashlib.sha1(norm(a)[:300].encode()).hexdigest()
            q_norm_counts[qh] += 1
            qa_counts[(qh, ah_short)] += 1
            q_to_answers[qh].add(ah_short)
            q_to_sources[qh].add(src)
            src_counts[src] += 1

    # exact (Q, A) duplicates
    exact_qa_dups = sum(v - 1 for v in qa_counts.values() if v > 1)
    distinct_qa = len(qa_counts)

    # near-duplicate questions (same normalized question, ANY answer)
    repeated_q = sum(v - 1 for v in q_norm_counts.values() if v > 1)
    distinct_q = len(q_norm_counts)

    # contradictions: same Q, multiple distinct A hashes
    contradiction_qhashes = [qh for qh, answers in q_to_answers.items() if len(answers) > 1]
    n_contradictions = len(contradiction_qhashes)

    # distribution of question multiplicity
    mult_buckets = Counter()
    for v in q_norm_counts.values():
        if v == 1:
            mult_buckets["1 (unique)"] += 1
        elif v == 2:
            mult_buckets["2"] += 1
        elif v <= 5:
            mult_buckets["3-5"] += 1
        elif v <= 10:
            mult_buckets["6-10"] += 1
        else:
            mult_buckets[">10"] += 1

    return {
        "path": str(path),
        "n_rows": n_rows,
        "distinct_questions": distinct_q,
        "repeated_question_extras": repeated_q,
        "exact_qa_pairs_distinct": distinct_qa,
        "exact_qa_duplicate_extras": exact_qa_dups,
        "contradictions_same_q_diff_a": n_contradictions,
        "duplication_rate_pct": round(100 * repeated_q / max(1, n_rows), 2),
        "exact_qa_dup_rate_pct": round(100 * exact_qa_dups / max(1, n_rows), 2),
        "question_multiplicity": dict(mult_buckets),
        "top_repeated_questions": _samples(q_norm_counts, q_to_sources, 8),
        "n_sources": len(src_counts),
    }


def _samples(q_counts: Counter, q_to_sources: dict, k: int) -> list[dict]:
    top = q_counts.most_common(k)
    out = []
    for qh, cnt in top:
        if cnt <= 1:
            break
        out.append({"qhash": qh[:12], "count": cnt,
                    "sources": sorted(q_to_sources[qh])[:6]})
    return out


def show_contradiction_samples(path: Path, k: int = 5) -> None:
    """Print actual sample rows where same Q has different A."""
    if not path.exists():
        return
    q_to_rows: dict[str, list[dict]] = defaultdict(list)
    for line in path.open(encoding="utf-8"):
        try:
            r = json.loads(line)
        except Exception:
            continue
        q = r.get("question") or r.get("prompt") or ""
        a = r.get("answer") or r.get("chosen") or ""
        if not q or not a:
            continue
        qh = qhash(q)
        q_to_rows[qh].append(r)
    # find Qs with multiple distinct A's
    candidates = []
    for qh, rows in q_to_rows.items():
        if len(rows) <= 1:
            continue
        a_set = {hashlib.sha1(norm(r.get("answer") or r.get("chosen") or "")[:300].encode()).hexdigest()
                 for r in rows}
        if len(a_set) > 1:
            candidates.append((qh, rows))
    print(f"\n--- {len(candidates)} questions with multiple distinct answers ---")
    for qh, rows in candidates[:k]:
        print(f"\nQ: {(rows[0].get('question') or rows[0].get('prompt'))[:300]}")
        seen_a = set()
        for r in rows[:3]:
            a = r.get("answer") or r.get("chosen") or ""
            ah = hashlib.sha1(norm(a)[:300].encode()).hexdigest()
            if ah in seen_a:
                continue
            seen_a.add(ah)
            print(f"  [source: {r.get('source', '?')}] A: {a[:280]}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("files", nargs="*",
                   default=["rl_prepared/rl_sft.jsonl",
                            "rl_prepared/dental_sft_mega.jsonl"])
    p.add_argument("--show-contradictions", action="store_true")
    args = p.parse_args()

    for f in args.files:
        path = ROOT / f
        report = analyze(path)
        print(f"\n=== {path.name} ===")
        for k, v in report.items():
            if k in ("top_repeated_questions",):
                continue
            print(f"  {k}: {v}")
        if report.get("top_repeated_questions"):
            print(f"  top_repeated_questions:")
            for s in report["top_repeated_questions"]:
                print(f"    cnt={s['count']:>5}  sources={s['sources']}")
        if args.show_contradictions:
            show_contradiction_samples(path, k=4)
    return 0


if __name__ == "__main__":
    sys.exit(main())
