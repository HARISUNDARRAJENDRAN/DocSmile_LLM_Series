"""HEAD-QA dental subset — HEAD-QA has no native dentistry category, so we
keyword-filter dental-relevant items across all categories (mostly drawn from
medicine, pharmacology, and biology exams).
"""
from __future__ import annotations

from typing import Iterator

from datasets import load_dataset

from ..common.dental_filter import is_dental
from ..common.schema import make_sft

LETTERS = ["A", "B", "C", "D", "E"]
SOURCE = "head_qa_dental_keyword"


def _format_mcq(q: str, options: list[str], correct_idx: int) -> tuple[str, str] | None:
    if not q or not options or correct_idx is None:
        return None
    if not (0 <= correct_idx < len(options)):
        return None
    correct_text = (options[correct_idx] or "").strip()
    if not correct_text:
        return None
    prompt = q.strip() + "\n\n" + "\n".join(
        f"{LETTERS[i]}. {(o or '').strip()}" for i, o in enumerate(options) if i < len(LETTERS)
    )
    answer = f"{LETTERS[correct_idx]}. {correct_text}"
    return prompt, answer


def _iter_v2(cfg: str) -> Iterator[dict]:
    try:
        dsd = load_dataset("alesi12/head_qa_v2", cfg)
    except Exception as e:
        print(f"[headqa] v2 {cfg} load failed: {e}")
        return
    for split, ds in dsd.items():
        for row in ds:
            q = (row.get("qtext") or "").strip()
            if not q:
                continue
            answers = row.get("answers") or []
            ra = row.get("ra")
            try:
                ra = int(ra)
            except (TypeError, ValueError):
                continue
            options: list[str] = []
            correct_idx: int | None = None
            for i, a in enumerate(answers):
                if not isinstance(a, dict):
                    continue
                txt = (a.get("atext") or "").strip()
                options.append(txt)
                try:
                    aid = int(a.get("aid", -1))
                except (TypeError, ValueError):
                    aid = -1
                if aid == ra:
                    correct_idx = i
            if correct_idx is None or not options:
                continue
            # Keyword-filter to dental relevance
            if not is_dental(q, *options):
                continue
            fmt = _format_mcq(q, options, correct_idx)
            if not fmt:
                continue
            prompt, answer = fmt
            cat = (row.get("category") or "").strip().capitalize()
            topic_tag = f"Dentistry ({cfg.upper()}/{cat})" if cat else f"Dentistry ({cfg.upper()})"
            rec = make_sft(prompt, answer, SOURCE, topic_tag)
            if rec:
                yield rec


def load() -> Iterator[dict]:
    yield from _iter_v2("en")
    yield from _iter_v2("es")
