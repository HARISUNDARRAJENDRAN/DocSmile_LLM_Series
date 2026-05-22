"""Combined SFT + DPO generator from cleaned textbook chunks.

For each high-quality chunk, one Gemini call yields 2-3 items each of the form:
  {
    "question":         exam-style clinical question grounded in the passage,
    "chosen":           factually correct, source-supported answer,
    "rejected":         same-length, same-topic answer with a SUBTLE clinical error,
    "topic":            dental specialty label,
  }

Outputs:
  cpt_prepared/textbook_sft/<book>.jsonl                # SFT rows: {question, answer, source, topic}
  cpt_prepared/textbook_dpo/<book>.jsonl                # DPO rows: {prompt, chosen, rejected, source, topic}
  cpt_prepared/textbook_sft_combined.jsonl              # aggregated SFT
  cpt_prepared/textbook_dpo_combined.jsonl              # aggregated DPO

Validation:
  - chosen answer must share >= 40% non-stopword tokens with source chunk (grounding)
  - rejected answer must NOT be near-identical to chosen (subtle differences required)
  - drop any item where Q or A is too short / too long
  - dedup generated Q against existing rl_prepared/dental_sft_mega.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

from ..common.gemini import GeminiPool
from ..common.dedup import qhash

ROOT = Path(__file__).resolve().parents[3]
IN_PATH = ROOT / "cpt_prepared" / "dental_cpt_books_v2.jsonl"
OUT_SFT_DIR = ROOT / "cpt_prepared" / "textbook_sft"
OUT_DPO_DIR = ROOT / "cpt_prepared" / "textbook_dpo"
AGG_SFT = ROOT / "cpt_prepared" / "textbook_sft_combined.jsonl"
AGG_DPO = ROOT / "cpt_prepared" / "textbook_dpo_combined.jsonl"
LOG_DIR = ROOT / "logs" / "textbook_sft"
EXISTING_SFT = ROOT / "rl_prepared" / "dental_sft_mega.jsonl"

for d in (OUT_SFT_DIR, OUT_DPO_DIR, LOG_DIR):
    d.mkdir(parents=True, exist_ok=True)


TOPIC_TAXONOMY = [
    "Operative Dentistry", "Endodontics", "Periodontics", "Prosthodontics",
    "Orthodontics", "Oral Surgery", "Oral Medicine", "Oral Pathology",
    "Oral Radiology", "Pediatric Dentistry", "Preventive Dentistry",
    "Dental Public Health", "Dental Anatomy", "Dental Materials",
    "Dental Pharmacology", "Implantology", "Dental Hygiene", "Oral Cancer",
    "TMJ Disorders", "Dental Caries", "Dental Emergencies",
    "Oral Microbiology", "Dental Anesthesia", "Forensic Dentistry",
    "Geriatric Dentistry", "Dental Embryology", "Dental Histology",
    "Biochemistry", "Microbiology", "Pharmacology", "Genetics",
    "Human Anatomy", "Pathology", "Physiology", "Immunology",
    "Dentistry",
]


SYSTEM = (
    "You generate high-quality dental and medical Q&A items from textbook passages "
    "for training a clinical LLM. Be precise, exam-style, and grounded in the passage. "
    "Do NOT invent facts beyond the passage. For each item you also produce a 'rejected' "
    "answer that is the SAME LENGTH and SAME FORMAT as the chosen answer, but contains "
    "a SUBTLE clinical error (wrong drug class, wrong nerve, wrong tooth number, off-by-one "
    "stage, inverted cause/effect, outdated convention, etc.) — never obviously absurd. "
    "Return strict JSON only."
)


PROMPT_TEMPLATE = """Read the passage below and generate {n} dental/medical Q&A items.

REQUIREMENTS for each item:
- "question": a focused, clinically meaningful, exam-style question that is fully
  answerable from the passage. Phrase as a real student would ask, not as
  "what does the passage say". Vary phrasing across the {n} items: mix of
  definitions, mechanisms, comparisons, indications, contraindications,
  differential diagnoses, procedural steps, classifications.

  ABSOLUTELY DO NOT generate questions of these forbidden meta-types:
  - "According to the text/passage/textbook..."
  - "What does the textbook/author/section state about..."
  - "How are X organized within the section/chapter/atlas..."
  - "What specific information is provided in..."
  - Any question that references the textbook's structure, layout, charts,
    page numbers, sections, chapters, figures, or its own organization.
  Ask about the underlying clinical/scientific content as if the textbook
  didn't exist — pretend you're writing an exam question for a dental student.
  If the passage is metatextual (preface, instructions for the reader,
  publisher boilerplate, table of contents commentary), return "items": [].

- "chosen": 2-5 sentence factual answer. ALL claims must be supported by the
  passage. Preserve technical precision (drug names, doses, numbers, names of
  classifications). No hedging.
- "rejected": same length and topic as chosen, but contains a SUBTLE clinical
  error that a confused student might make. Examples of subtle errors:
  wrong drug class, wrong nerve, off-by-one tooth number, inverted cause/effect,
  swapped Stage I/II, wrong direction of effect, outdated terminology, wrong
  artery, wrong layer of tissue. Never absurd; must read as plausibly correct
  at first glance.
- "topic": one label from this list (pick the most specific): {topics}

Return STRICT JSON of this exact shape, no commentary:
{{
  "items": [
    {{"question": "...", "chosen": "...", "rejected": "...", "topic": "..."}},
    ...
  ]
}}

If the passage is unsuitable (too short, too noisy, no extractable clinical
content, or only describes the textbook itself), return: {{"items": []}}

PASSAGE:
---
{chunk}
---
"""


# ---------------------------------------------------------------------------
# Grounding / validation
# ---------------------------------------------------------------------------

_STOPWORDS = set("""
a an the and or but if then so as of in on at to from by with for is are was
were be been being have has had do does did this that these those it its
their there which who whom what when where why how about into over under
between among also can could should would may might must shall will not no
yes more less very much many such all any each every some both either neither
"""
.split())

WORD_RE = re.compile(r"[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ\-]+")

# Boilerplate / front-matter / disclaimer terms — high density of these in a chunk
# signals it is publisher/legal/copyright content, not training-worthy prose.
_BOILERPLATE_TERMS = re.compile(
    r"\b(isbn|copyright|publisher|reproduced|reproduction|leaflet|manufacturer|"
    r"acknowledg(?:ement|ments?)|dedication|disclaimer|cataloging|preface|"
    r"foreword|trademark|imprint|edited by|edition|reprint|distributors?|"
    r"library of congress|all rights reserved|illustrator|translator|"
    r"contributors?|about the authors?|biographical|registered)\b",
    re.IGNORECASE,
)

# Skip the first N chunks of each book — empirically front matter escapes classification
SKIP_FIRST_CHUNKS = 5

# Reject questions that ask about the textbook itself rather than the content
META_Q_PATTERNS = re.compile(
    r"\b(according to (the )?(text|passage|textbook|atlas|author|chapter|section)|"
    r"what does the (text|passage|textbook|author|atlas|chapter|section)|"
    r"how (is|are) .{0,40} (organized|presented|structured|arranged) (in|within|throughout)|"
    r"what (specific )?information is provided in|"
    r"what is described in this (section|chapter|passage|text)|"
    r"page numbers? pp?\.|"
    r"the (text|passage|atlas|textbook|author) (states|describes|covers|outlines|explains))\b",
    re.IGNORECASE,
)


def _is_meta_question(q: str) -> bool:
    return bool(META_Q_PATTERNS.search(q))


def _boilerplate_density(text: str) -> float:
    n_words = max(1, len(WORD_RE.findall(text)))
    n_hits = len(_BOILERPLATE_TERMS.findall(text))
    return n_hits / n_words


def _content_tokens(text: str) -> set[str]:
    return {w.lower() for w in WORD_RE.findall(text or "")
            if w.lower() not in _STOPWORDS and len(w) > 2}


def grounded_overlap(answer: str, source: str) -> float:
    a = _content_tokens(answer)
    s = _content_tokens(source)
    if not a:
        return 0.0
    return len(a & s) / len(a)


# ---------------------------------------------------------------------------
# Output validation
# ---------------------------------------------------------------------------

MIN_Q_CHARS = 25
MAX_Q_CHARS = 800
MIN_A_CHARS = 80
MAX_A_CHARS = 3000
MIN_GROUNDING = 0.40         # chosen answer must overlap >= 40% with source
MIN_DIVERGENCE = 0.30        # rejected must differ from chosen by >= 30% of tokens


def _diverges(chosen: str, rejected: str) -> bool:
    a = _content_tokens(chosen)
    b = _content_tokens(rejected)
    if not a or not b:
        return False
    inter = len(a & b)
    union = max(1, len(a | b))
    jaccard = inter / union
    return (1 - jaccard) >= MIN_DIVERGENCE


def validate_item(item: dict, source_chunk: str) -> tuple[bool, str]:
    q = (item.get("question") or "").strip()
    chosen = (item.get("chosen") or "").strip()
    rejected = (item.get("rejected") or "").strip()
    topic = (item.get("topic") or "Dentistry").strip()
    if not (MIN_Q_CHARS <= len(q) <= MAX_Q_CHARS):
        return False, "q_length"
    if _is_meta_question(q):
        return False, "meta_question"
    if not (MIN_A_CHARS <= len(chosen) <= MAX_A_CHARS):
        return False, "chosen_length"
    if not (MIN_A_CHARS <= len(rejected) <= MAX_A_CHARS):
        return False, "rejected_length"
    if topic not in TOPIC_TAXONOMY:
        item["topic"] = "Dentistry"
    if grounded_overlap(chosen, source_chunk) < MIN_GROUNDING:
        return False, "grounding"
    if not _diverges(chosen, rejected):
        return False, "rejected_too_similar"
    if chosen.lower() == rejected.lower():
        return False, "identical"
    return True, "ok"


# ---------------------------------------------------------------------------
# JSON-parsing of Gemini output
# ---------------------------------------------------------------------------

_JSON_RE = re.compile(r"\{[\s\S]*\}")


def _parse(out: str) -> list[dict]:
    if not out:
        return []
    m = _JSON_RE.search(out)
    if not m:
        return []
    try:
        obj = json.loads(m.group(0))
    except Exception:
        return []
    items = obj.get("items") or []
    if not isinstance(items, list):
        return []
    return items


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def _build_prompt(chunk_text: str, n_items: int = 3) -> str:
    return PROMPT_TEMPLATE.format(
        n=n_items, topics=", ".join(TOPIC_TAXONOMY),
        chunk=chunk_text[:5500],
    )


def gen_for_chunk(pool: GeminiPool, chunk: dict, n_items: int = 3) -> tuple[list[dict], list[dict]]:
    """One Gemini call per chunk. Returns (sft_rows, dpo_rows)."""
    prompt = _build_prompt(chunk["text"], n_items=n_items)
    est_tokens = len(chunk["text"]) // 3 + 1200
    raw = pool.generate(prompt, system=SYSTEM, est_tokens=min(est_tokens, 12000))
    items = _parse(raw)
    src = chunk.get("source") or f"book:{chunk.get('book', '?')}"
    sft, dpo = [], []
    for it in items:
        ok, reason = validate_item(it, chunk["text"])
        if not ok:
            continue
        q = it["question"].strip()
        chosen = it["chosen"].strip()
        rejected = it["rejected"].strip()
        topic = (it.get("topic") or "Dentistry").strip()
        sft.append({"question": q, "answer": chosen, "source": src, "topic": topic})
        dpo.append({"prompt": q, "chosen": chosen, "rejected": rejected,
                    "source": src, "topic": topic})
    return sft, dpo


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def _safe_stem(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", s)[:120]


def _load_existing_qhashes() -> set[str]:
    """Hashes of all questions in the current SFT mega corpus, to dedup against."""
    hashes: set[str] = set()
    if not EXISTING_SFT.exists():
        return hashes
    for line in EXISTING_SFT.open(encoding="utf-8"):
        try:
            r = json.loads(line)
        except Exception:
            continue
        q = r.get("question") or r.get("prompt")
        if q:
            hashes.add(qhash(q))
    return hashes


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=0, help="Process only first N chunks")
    p.add_argument("--n-items", type=int, default=4, help="Q&A items per chunk")
    p.add_argument("--books", help="Substring filter on source book name")
    p.add_argument("--dry-run", action="store_true", help="Print only, don't write")
    p.add_argument("--skip-existing-dedup", action="store_true")
    p.add_argument("--workers", type=int, default=12,
                   help="Concurrent Gemini calls in flight (6 keys × 2 = 12 default)")
    p.add_argument("--keys-files", nargs="+", default=["1.env"],
                   help="Env files to load Gemini keys from (default: 1.env only)")
    args = p.parse_args(argv)

    if not IN_PATH.exists():
        print(f"missing {IN_PATH} — run postfilter_v2 first", file=sys.stderr)
        return 1

    chunks: list[dict] = []
    for line in IN_PATH.open(encoding="utf-8"):
        try:
            r = json.loads(line)
        except Exception:
            continue
        if args.books and args.books.lower() not in (r.get("book") or r.get("source") or "").lower():
            continue
        # Skip the first few chunks of each book (front-matter that escaped classification)
        if (r.get("chunk_idx") or 0) < SKIP_FIRST_CHUNKS:
            continue
        # Skip chunks dominated by publisher/legal boilerplate
        if _boilerplate_density(r.get("text", "")) > 0.012:
            continue
        chunks.append(r)
    if args.limit > 0:
        chunks = chunks[:args.limit]
    print(f"[gen] chunks to process: {len(chunks)}")

    seen_q: set[str] = set()
    if not args.skip_existing_dedup:
        seen_q = _load_existing_qhashes()
        print(f"[gen] seeded dedup with {len(seen_q)} existing questions")

    pool = GeminiPool(files=args.keys_files)
    print(f"[gen] gemini key pool size: {len(pool.states)} (from {args.keys_files})")

    # Per-book output file handles. Open lazily; protected by a lock per-file.
    file_locks: dict[Path, threading.Lock] = {}
    agg_lock = threading.Lock()
    seen_lock = threading.Lock()
    stats_lock = threading.Lock()

    AGG_SFT.parent.mkdir(parents=True, exist_ok=True)
    AGG_DPO.parent.mkdir(parents=True, exist_ok=True)
    OUT_SFT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DPO_DIR.mkdir(parents=True, exist_ok=True)
    agg_sft_f = AGG_SFT.open("a", encoding="utf-8")
    agg_dpo_f = AGG_DPO.open("a", encoding="utf-8")

    drop_reasons: Counter = Counter()
    counters = {"n_sft": 0, "n_dpo": 0, "n_dup": 0, "n_err": 0, "n_done": 0}
    started = time.time()
    last_log_t = [time.time()]

    def _write_pair(book: str, sft_row: dict, dpo_row: dict):
        book_stem = _safe_stem(book)
        sft_path = OUT_SFT_DIR / f"{book_stem}.jsonl"
        dpo_path = OUT_DPO_DIR / f"{book_stem}.jsonl"
        sft_line = json.dumps(sft_row, ensure_ascii=False) + "\n"
        dpo_line = json.dumps(dpo_row, ensure_ascii=False) + "\n"
        lock = file_locks.setdefault(sft_path, threading.Lock())
        with lock:
            with sft_path.open("a", encoding="utf-8") as f:
                f.write(sft_line)
        lock = file_locks.setdefault(dpo_path, threading.Lock())
        with lock:
            with dpo_path.open("a", encoding="utf-8") as f:
                f.write(dpo_line)
        with agg_lock:
            agg_sft_f.write(sft_line)
            agg_dpo_f.write(dpo_line)

    def _process_one(chunk: dict, idx: int):
        book = chunk.get("book") or chunk.get("source") or "unknown"
        try:
            sft_rows, dpo_rows = gen_for_chunk(pool, chunk, n_items=args.n_items)
        except Exception as e:
            with stats_lock:
                counters["n_err"] += 1
                counters["n_done"] += 1
            print(f"[gen] err on chunk {idx}: {str(e)[:120]}")
            return
        for s, d in zip(sft_rows, dpo_rows):
            h = qhash(s["question"])
            with seen_lock:
                if h in seen_q:
                    with stats_lock:
                        counters["n_dup"] += 1
                    continue
                seen_q.add(h)
            if not args.dry_run:
                _write_pair(book, s, d)
            with stats_lock:
                counters["n_sft"] += 1
                counters["n_dpo"] += 1
        with stats_lock:
            counters["n_done"] += 1
            now = time.time()
            if now - last_log_t[0] > 20:
                elapsed = now - started
                rate = counters["n_done"] / max(1.0, elapsed)
                eta = (len(chunks) - counters["n_done"]) / max(1.0, rate)
                live = sum(1 for s in pool.states if not s.suspended)
                print(f"[gen] {counters['n_done']}/{len(chunks)} "
                      f"sft={counters['n_sft']} dpo={counters['n_dpo']} "
                      f"dup={counters['n_dup']} err={counters['n_err']} "
                      f"rate={rate:.2f}/s eta={eta:.0f}s live_keys={live}")
                last_log_t[0] = now

    try:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = [ex.submit(_process_one, c, i) for i, c in enumerate(chunks)]
            for fut in as_completed(futures):
                # exceptions are swallowed inside _process_one; just drain
                fut.result()
    finally:
        agg_sft_f.close()
        agg_dpo_f.close()

    print(f"\n[gen] DONE: sft={counters['n_sft']} dpo={counters['n_dpo']} "
          f"dup={counters['n_dup']} err={counters['n_err']}")
    print(f"  -> {AGG_SFT}")
    print(f"  -> {AGG_DPO}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
