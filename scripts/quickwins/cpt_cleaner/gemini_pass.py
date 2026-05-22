"""Gemini cleanup pass for surviving CPT chunks.

Reuses the 41-key pool from scripts.quickwins.common.gemini.GeminiPool.

The prompt is adapted from scripts/clean_cpt_production_gemini.py — it asks the
model to clean OCR/extraction noise WITHOUT summarising or rewording. Outputs
that the model judges as pure noise come back as <SKIP> and are dropped here.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Iterable

from ..common.gemini import GeminiPool

CLEAN_SYSTEM = (
    "You are a careful pre-processor for continued pretraining of a medical/dental LLM. "
    "Given a chunk of textbook prose extracted from a PDF, you remove layout/OCR noise "
    "while PRESERVING ALL substantive scientific content verbatim. Do not summarise, "
    "paraphrase, simplify, add commentary, add headings, or change technical claims. "
    "If the whole chunk is unsalvageable noise (figure-label fragments, broken tables, "
    "boilerplate), respond with exactly <SKIP>."
)

CLEAN_PROMPT_TEMPLATE = """Clean the following textbook chunk for continued pretraining.

REMOVE:
- repeated page headers, footers, running titles, isolated page numbers
- broken figure or table captions whose figure is absent
- "see page X" cross-references, navigation breadcrumbs
- isolated OCR garbage, mojibake clusters, broken bullet lists with no context
- duplicated paragraphs or sentence fragments

FIX:
- hyphenated line-break words ("infor- mation" -> "information")
- broken sentence wrapping where it is unambiguous from context
- single-token-per-line fragments where they form one paragraph
- OCR confusions only when unambiguously implied by the surrounding sentence

PRESERVE every substantive dental/medical/scientific fact, definition, mechanism,
classification, diagnostic criterion, contraindication, complication, treatment
principle, anatomical or pathological detail. Keep numbered/lettered lists and
technical lists.

Do NOT add commentary or headings. Output only the cleaned prose.

If after cleaning fewer than 200 characters of substantive prose remain, output
exactly the token: <SKIP>

CHUNK:
---
{text}
---
"""


_SKIP_RE = re.compile(r"^\s*<\s*SKIP\s*>\s*$", re.IGNORECASE)


def _post_filter_quality(text: str) -> float:
    """Light quality score on the Gemini output. 0..1.

    Drops fragments, mojibake-heavy outputs, ultra-short responses.
    """
    if not text or len(text) < 200:
        return 0.0
    words = text.split()
    if len(words) < 30:
        return 0.0
    n_chars = len(text)
    n_alpha = sum(c.isalpha() for c in text)
    if n_alpha / n_chars < 0.55:
        return 0.0
    # punctuation density
    n_punct = sum(1 for c in text if not c.isalnum() and not c.isspace())
    if n_punct / n_chars > 0.25:
        return 0.0
    avg_word_len = sum(len(w) for w in words) / len(words)
    if not (3.2 <= avg_word_len <= 9.0):
        return 0.3
    return 0.8


def clean_chunk(pool: GeminiPool, text: str) -> dict:
    """Clean a single chunk via Gemini. Returns {status, text, quality}."""
    prompt = CLEAN_PROMPT_TEMPLATE.format(text=text)
    est_tokens = len(text) // 3 + len(CLEAN_SYSTEM) // 3 + 600
    raw = pool.generate(prompt, system=CLEAN_SYSTEM,
                        est_tokens=min(est_tokens, 15000))
    raw = (raw or "").strip()
    if not raw or _SKIP_RE.match(raw):
        return {"status": "skip", "text": "", "quality": 0.0}
    # strip any remaining "<SKIP>" tokens that leaked into prose
    cleaned = re.sub(r"<\s*SKIP\s*>", "", raw).strip()
    q = _post_filter_quality(cleaned)
    if q < 0.4:
        return {"status": "low_quality", "text": cleaned, "quality": q}
    return {"status": "ok", "text": cleaned, "quality": q}


def clean_chunks(chunks: Iterable[dict], pool: GeminiPool | None = None,
                 progress_every: int = 25) -> tuple[list[dict], dict]:
    """Run Gemini cleanup over a list of chunks. Returns (cleaned, stats)."""
    if pool is None:
        pool = GeminiPool()
    cleaned: list[dict] = []
    n_skip = n_low = n_ok = n_err = 0
    chunks_list = list(chunks)
    n_total = len(chunks_list)
    started = time.time()
    for i, c in enumerate(chunks_list):
        try:
            result = clean_chunk(pool, c["text"])
        except Exception as e:
            n_err += 1
            continue
        if result["status"] == "skip":
            n_skip += 1
            continue
        if result["status"] == "low_quality":
            n_low += 1
            continue
        new_c = dict(c)
        new_c["text"] = result["text"]
        new_c["gemini_quality"] = result["quality"]
        new_c["cleaned"] = True
        cleaned.append(new_c)
        n_ok += 1
        if (i + 1) % progress_every == 0:
            elapsed = time.time() - started
            rate = (i + 1) / max(1.0, elapsed)
            eta = (n_total - (i + 1)) / max(1.0, rate)
            print(f"[gemini] {i+1}/{n_total} ok={n_ok} skip={n_skip} low={n_low} "
                  f"err={n_err} rate={rate:.1f}/s eta={eta:.0f}s")
    stats = {"n_total": n_total, "n_ok": n_ok, "n_skip": n_skip,
             "n_low_quality": n_low, "n_err": n_err}
    return cleaned, stats
