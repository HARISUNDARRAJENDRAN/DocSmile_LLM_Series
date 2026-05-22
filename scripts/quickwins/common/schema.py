"""Target SFT/CPT schema + normalisation helpers."""
from __future__ import annotations

import re
import unicodedata

REQUIRED_SFT_FIELDS = ("question", "answer", "source", "topic")
REQUIRED_CPT_FIELDS = ("text", "source")

MIN_Q_CHARS = 5
MIN_A_CHARS = 5
MAX_Q_CHARS = 6000
MAX_A_CHARS = 30000


def clean_text(s: str) -> str:
    if s is None:
        return ""
    s = unicodedata.normalize("NFKC", str(s))
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def make_sft(question: str, answer: str, source: str, topic: str) -> dict | None:
    q = clean_text(question)
    a = clean_text(answer)
    s = clean_text(source) or "unknown"
    t = clean_text(topic) or "Dentistry"
    if not (MIN_Q_CHARS <= len(q) <= MAX_Q_CHARS):
        return None
    if not (MIN_A_CHARS <= len(a) <= MAX_A_CHARS):
        return None
    return {"question": q, "answer": a, "source": s, "topic": t}


def make_cpt(text: str, source: str) -> dict | None:
    t = clean_text(text)
    s = clean_text(source) or "unknown"
    if len(t) < 200:
        return None
    return {"text": t, "source": s}


def is_valid_sft(row: dict) -> bool:
    if not isinstance(row, dict):
        return False
    if not all(k in row for k in REQUIRED_SFT_FIELDS):
        return False
    q = row.get("question", "")
    a = row.get("answer", "")
    if not isinstance(q, str) or not isinstance(a, str):
        return False
    if len(q) < MIN_Q_CHARS or len(a) < MIN_A_CHARS:
        return False
    return True
