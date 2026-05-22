"""LexiconShift dental trio (all Sinhala).

  - Dental_QnA_Instruct: column 'text' formatted as
        'Instruction: / <Q> / Response: / <A><eos>'  -> SFT
  - FB_Articles_Dental:   column 'Text' (sentence-level chunks) -> concatenated CPT
  - Continued_Pretrained_Dataset_Dental_Sinhala: column 'Text' -> concatenated CPT
"""
from __future__ import annotations

import re
from typing import Iterator

from datasets import load_dataset

from ..common.schema import make_cpt, make_sft

QNA_REPO = "LexiconShiftInnovations/Dental_QnA_Instruct"
FB_REPO = "LexiconShiftInnovations/FB_Articles_Dental"
SINHALA_REPO = "LexiconShiftInnovations/Continued_Pretrained_Dataset_Dental_Sinhala"

_QNA_RE = re.compile(
    r"^\s*Instruction:\s*/\s*(?P<q>.*?)\s*/\s*Response:\s*/\s*(?P<a>.*?)\s*(?:<eos>)?\s*$",
    re.DOTALL,
)


def load_qna_sft() -> Iterator[dict]:
    try:
        ds = load_dataset(QNA_REPO, split="train")
    except Exception as e:
        print(f"[lexicon_shift qna] load failed: {e}")
        return
    for row in ds:
        text = (row.get("text") or row.get("Text") or "").strip()
        if not text:
            continue
        m = _QNA_RE.match(text)
        if not m:
            # fallback: try to split on 'Response:' once
            if "Response:" in text:
                parts = text.split("Response:", 1)
                q = parts[0].replace("Instruction:", "").strip(" /:\n")
                a = parts[1].strip(" /:\n").removesuffix("<eos>").strip()
            else:
                continue
        else:
            q = m.group("q").strip()
            a = m.group("a").strip().removesuffix("<eos>").strip()
        if not q or not a:
            continue
        rec = make_sft(q, a, "lexicon_shift_dental_qna_sinhala", "Dentistry (SI)")
        if rec:
            yield rec


def _stream_concat_chunks(repo: str, source: str, target_chars: int = 1200) -> Iterator[dict]:
    """Sinhala paragraphs are sentence-level; concatenate until we hit ~target_chars."""
    try:
        ds = load_dataset(repo, split="train")
    except Exception as e:
        print(f"[lexicon_shift cpt {repo}] load failed: {e}")
        return
    buf: list[str] = []
    buf_len = 0
    for row in ds:
        text = (row.get("Text") or row.get("text") or "").strip()
        if not text:
            continue
        buf.append(text)
        buf_len += len(text) + 1
        if buf_len >= target_chars:
            chunk = " ".join(buf)
            buf = []
            buf_len = 0
            rec = make_cpt(chunk, source)
            if rec:
                yield rec
    if buf:
        rec = make_cpt(" ".join(buf), source)
        if rec:
            yield rec


def load_fb_sft() -> Iterator[dict]:
    """FB_Articles_Dental is Sinhala sentence fragments; route to CPT-style chunks."""
    yield from _stream_concat_chunks(FB_REPO, "lexicon_shift_fb_articles_sinhala")


def load_sinhala_cpt() -> Iterator[dict]:
    yield from _stream_concat_chunks(SINHALA_REPO, "lexicon_shift_sinhala_cpt")
