"""Deterministic CPT-book cleaning: normalise, segment, classify, chunk.

Pipeline:
    raw .txt
        -> normalise()         line-level: mojibake, ligatures, dehyphenate, drop URL/page lines
        -> segment()           split into blocks on blank-line groups
        -> classify_block()    tag each block: prose/toc/index/refs/form/chemfrag/front_matter/junk
        -> score_quality()     per-prose-block quality 0..1
        -> chunk()             group consecutive prose blocks into ~6000-char chunks

The output of process_file() is:
  - chunks: [{text, source, book, chunk_idx, lines, quality}]
  - audit:  [{book, line_start, line_end, tag, reason, sample, kept}]
"""
from __future__ import annotations

import re
import statistics
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# normalisation
# ---------------------------------------------------------------------------

# Curly quotes / dashes / common Unicode noise
_UNICODE_FIX = {
    "‘": "'", "’": "'", "‚": "'", "‛": "'",
    "“": '"', "”": '"', "„": '"', "‟": '"',
    "–": "-", "—": "-", "−": "-",
    "…": "...",
    " ": " ", "​": "", "‌": "", "‍": "", "﻿": "",
    "­": "",  # soft hyphen
}

# Chinese page-number markers and other locale artifacts
_LINE_NUKE_PATTERNS = [
    re.compile(r"^\s*页码[，,。:]\s*$"),
    re.compile(r"^\s*Bookmark URL:.*$"),
    re.compile(r"^\s*\d+/\d+\s*$"),                 # bare "12/345" page indicators
    re.compile(r"^\s*Page\s+\d+(\s+of\s+\d+)?\s*$", re.I),
    re.compile(r"^\s*(http|https|www)\S+\s*$"),
    re.compile(r"^\s*©.*$"),                        # copyright lines
    re.compile(r"^\s*ISBN[\s-]*\d.*$"),
]

# OCR ligature gaps: "dif cult" -> "difficult", "ef cient" -> "efficient"
_LIGATURE_FIXES = [
    (re.compile(r"\bdif culty?\b"), "difficult"),
    (re.compile(r"\bdif cult\b"), "difficult"),
    (re.compile(r"\bef cien(t|cy|tly)\b"), r"efficien\1"),
    (re.compile(r"\bef cacy\b"), "efficacy"),
    (re.compile(r"\bof cer\b"), "officer"),
    (re.compile(r"\bof ce\b"), "office"),
    (re.compile(r"\bof cial\b"), "official"),
    (re.compile(r"\bsuf cient\b"), "sufficient"),
    (re.compile(r"\bclassi cation\b"), "classification"),
    (re.compile(r"\bidenti cation\b"), "identification"),
    (re.compile(r"\bspeci c\b"), "specific"),
    (re.compile(r"\bsigni cant\b"), "significant"),
    (re.compile(r"\bbene t\b"), "benefit"),
    (re.compile(r"\bcon rm\b"), "confirm"),
    (re.compile(r"\bde nition\b"), "definition"),
    (re.compile(r"\bde ned\b"), "defined"),
    (re.compile(r"\bin uence\b"), "influence"),
    (re.compile(r"\bin ammation\b"), "inflammation"),
    (re.compile(r"\bcoef cient\b"), "coefficient"),
    (re.compile(r"\bmagni cation\b"), "magnification"),
    (re.compile(r"\bjusti cation\b"), "justification"),
]

# Citation brackets that pollute prose: drop "[12]", "[12,34]", "[12][34][56]"
_CITATION_RE = re.compile(r"\[\d{1,4}(?:[,\s-]*\d{1,4})*\](?:\[\d{1,4}(?:[,\s-]*\d{1,4})*\])*")

# Hyphenated line break: word- \n word  →  word\n word  AND join into single word
_DEHYPHEN_RE = re.compile(r"(\w)-\n([a-z])")


def _replace_unicode(s: str) -> str:
    for bad, good in _UNICODE_FIX.items():
        if bad in s:
            s = s.replace(bad, good)
    return s


def normalise(text: str) -> str:
    """Line-level cleanup: encoding, ligatures, citations, page markers."""
    text = unicodedata.normalize("NFKC", text)
    text = _replace_unicode(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Dehyphenate line-break hyphens
    text = _DEHYPHEN_RE.sub(r"\1\2", text)
    out_lines: list[str] = []
    for raw_line in text.split("\n"):
        line = raw_line.rstrip()
        if not line.strip():
            out_lines.append("")
            continue
        if any(p.match(line) for p in _LINE_NUKE_PATTERNS):
            continue
        # ligature fixes
        for rgx, repl in _LIGATURE_FIXES:
            line = rgx.sub(repl, line)
        # citation brackets - drop them inline
        line = _CITATION_RE.sub("", line)
        # collapse internal whitespace
        line = re.sub(r"[ \t]+", " ", line)
        out_lines.append(line)
    # collapse multiple blank lines to a single blank
    text = "\n".join(out_lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ---------------------------------------------------------------------------
# block segmentation
# ---------------------------------------------------------------------------

@dataclass
class Block:
    line_start: int                   # 1-indexed line in normalised text
    line_end: int
    lines: list[str] = field(default_factory=list)
    tag: str = "unknown"
    reason: str = ""
    quality: float = 0.0

    @property
    def text(self) -> str:
        return "\n".join(self.lines)

    @property
    def n_lines(self) -> int:
        return len(self.lines)

    @property
    def n_chars(self) -> int:
        return sum(len(l) for l in self.lines)


def segment(text: str) -> list[Block]:
    """Split normalised text into blocks on blank-line groups."""
    blocks: list[Block] = []
    cur_lines: list[str] = []
    cur_start = 1
    for i, line in enumerate(text.split("\n"), start=1):
        if not line.strip():
            if cur_lines:
                blocks.append(Block(line_start=cur_start, line_end=i - 1, lines=cur_lines))
                cur_lines = []
            cur_start = i + 1
        else:
            if not cur_lines:
                cur_start = i
            cur_lines.append(line)
    if cur_lines:
        blocks.append(Block(line_start=cur_start, line_end=cur_start + len(cur_lines) - 1,
                            lines=cur_lines))
    return blocks


# ---------------------------------------------------------------------------
# block classification
# ---------------------------------------------------------------------------

_DOTTED_LEADER_RE = re.compile(r"\.{3,}\s*\d+\s*$")
_PAGE_REF_TAIL_RE = re.compile(r",\s*\d+([\s,\-–]+\d+)*\s*$")
_REFERENCE_AUTHOR_RE = re.compile(r"^\s*\d+\.?\s+[A-Z][A-Za-z'\-]+,\s+[A-Z]\.")
_CHEM_FORMULA_RE = re.compile(r"^[A-Z]?[a-z]?\d?\+?-?$|^[A-Z][A-Z]?\d?\+?-?$|^[HCONSP][a-z0-9+\-]{0,3}$")
_ALPHA_SECTION_RE = re.compile(r"^\s*[A-Z]\s*$")     # single-letter section marker
_ALL_DIGITS_RE = re.compile(r"^\s*\d+(\s+\d+)*\s*$")


def _line_features(line: str) -> dict:
    s = line.strip()
    words = s.split()
    n_words = len(words)
    n_chars = len(s)
    n_digits = sum(c.isdigit() for c in s)
    n_alpha = sum(c.isalpha() for c in s)
    n_upper = sum(c.isupper() for c in s)
    return {
        "len": n_chars,
        "words": n_words,
        "digit_ratio": n_digits / max(1, n_chars),
        "alpha_ratio": n_alpha / max(1, n_chars),
        "upper_ratio": n_upper / max(1, n_alpha),
        "ends_digit": bool(s and s[-1].isdigit()),
        "dotted_leader": bool(_DOTTED_LEADER_RE.search(s)),
        "page_ref_tail": bool(_PAGE_REF_TAIL_RE.search(s)),
        "chem_only": bool(_CHEM_FORMULA_RE.fullmatch(s)) if s else False,
        "alpha_section": bool(_ALPHA_SECTION_RE.match(s)),
        "all_digits": bool(_ALL_DIGITS_RE.match(s)),
    }


def _block_features(block: Block) -> dict:
    feats = [_line_features(l) for l in block.lines]
    if not feats:
        return {}
    n = len(feats)
    return {
        "n": n,
        "median_len": statistics.median(f["len"] for f in feats),
        "mean_words": statistics.fmean(f["words"] for f in feats),
        "short_ratio": sum(1 for f in feats if f["len"] < 30) / n,
        "very_short_ratio": sum(1 for f in feats if f["words"] <= 2) / n,
        "ends_digit_ratio": sum(1 for f in feats if f["ends_digit"]) / n,
        "dotted_ratio": sum(1 for f in feats if f["dotted_leader"]) / n,
        "page_ref_tail_ratio": sum(1 for f in feats if f["page_ref_tail"]) / n,
        "chem_only_ratio": sum(1 for f in feats if f["chem_only"]) / n,
        "alpha_section_ratio": sum(1 for f in feats if f["alpha_section"]) / n,
        "mean_digit_ratio": statistics.fmean(f["digit_ratio"] for f in feats),
        "mean_alpha_ratio": statistics.fmean(f["alpha_ratio"] for f in feats),
        "mean_upper_ratio": statistics.fmean(f["upper_ratio"] for f in feats),
    }


# Front-matter cues found in the first ~15% of a book
_FRONT_MATTER_CUES = re.compile(
    r"\b(ISBN|Library of Congress|All rights reserved|Copyright|©|Verlag|"
    r"Publisher|Cover design|Typesetting by|Printed in|First published|"
    r"DOI:|Catalog(uing)?-in-Publication)\b",
    re.IGNORECASE,
)

_TOC_CUES = re.compile(r"\b(Contents|Table of Contents)\b", re.IGNORECASE)
_INDEX_CUES = re.compile(r"^\s*Index\s*$", re.IGNORECASE)


def classify_block(block: Block, position_pct: float, in_index: bool, in_refs: bool) -> str:
    """Return the block's tag. `position_pct` is 0..1 (start..end of file)."""
    feats = _block_features(block)
    if not feats:
        block.tag = "junk"
        block.reason = "empty"
        return "junk"

    n = feats["n"]
    line0 = block.lines[0].strip()

    # Hard signals first ------------------------------------------------------

    # Single-letter alphabetical section divider (index/reference letter headers)
    if n == 1 and feats["alpha_section_ratio"] == 1.0:
        block.tag = "alpha_section"
        block.reason = "single-letter divider"
        return "alpha_section"

    # Front matter — heuristic only valid in first 15% of file
    if position_pct < 0.15:
        block_text = block.text
        if _FRONT_MATTER_CUES.search(block_text):
            block.tag = "front_matter"
            block.reason = "ISBN/copyright/publisher boilerplate"
            return "front_matter"

    # TOC: explicit "Contents" header OR many dotted-leader lines
    if _TOC_CUES.search(line0) and n <= 3:
        block.tag = "toc_marker"
        block.reason = "Contents heading"
        return "toc_marker"
    if feats["dotted_ratio"] > 0.4 and n >= 4:
        block.tag = "toc"
        block.reason = f"dotted-leader ratio={feats['dotted_ratio']:.2f}"
        return "toc"

    # Index: explicit "Index" header in last 30% OR high page-ref-tail ratio
    if _INDEX_CUES.match(line0):
        block.tag = "index_marker"
        block.reason = "Index heading"
        return "index_marker"
    if position_pct > 0.65 and feats["page_ref_tail_ratio"] > 0.5 and feats["mean_words"] < 8:
        block.tag = "index"
        block.reason = f"page-ref tail ratio={feats['page_ref_tail_ratio']:.2f}, position={position_pct:.2f}"
        return "index"

    # References: numbered-author lines
    n_ref_lines = sum(1 for l in block.lines if _REFERENCE_AUTHOR_RE.match(l))
    if n >= 3 and n_ref_lines / n > 0.5:
        block.tag = "refs"
        block.reason = f"numbered-author refs={n_ref_lines}/{n}"
        return "refs"

    # In-index continuation: if previous block was index, lines have similar shape
    if in_index and feats["page_ref_tail_ratio"] > 0.35 and feats["mean_words"] < 12:
        block.tag = "index"
        block.reason = "index continuation"
        return "index"
    if in_refs and n_ref_lines / n > 0.3:
        block.tag = "refs"
        block.reason = "refs continuation"
        return "refs"

    # Fragmented chemistry / diagram labels: cluster of very-short, mostly-chem lines
    if feats["chem_only_ratio"] > 0.4 and feats["mean_words"] < 2.5:
        block.tag = "chemfrag"
        block.reason = f"chem-only lines ratio={feats['chem_only_ratio']:.2f}"
        return "chemfrag"
    if feats["very_short_ratio"] > 0.8 and feats["mean_words"] < 3 and n >= 4:
        block.tag = "fragments"
        block.reason = f"very-short-line cluster ({feats['mean_words']:.1f} w/line)"
        return "fragments"

    # Scanned-form / OCR-form fragments: lots of short ALL-CAPS lines
    if feats["mean_upper_ratio"] > 0.75 and feats["mean_words"] < 4 and n >= 3:
        block.tag = "form"
        block.reason = f"all-caps fragment cluster ({feats['mean_upper_ratio']:.2f} upper)"
        return "form"

    # Mostly-digits block (price tables, lab values)
    if feats["mean_digit_ratio"] > 0.5 and feats["mean_words"] < 4:
        block.tag = "numeric"
        block.reason = f"digit ratio={feats['mean_digit_ratio']:.2f}"
        return "numeric"

    # Default: prose (let quality scorer decide what to keep)
    block.tag = "prose"
    block.reason = ""
    return "prose"


# ---------------------------------------------------------------------------
# quality scoring (for prose blocks)
# ---------------------------------------------------------------------------

_SENT_END_RE = re.compile(r"[.!?][\s\n]")
_NON_WORD_RE = re.compile(r"[^\w\s]")


def score_quality(block: Block) -> float:
    """0..1 score for prose blocks. Used to drop low-quality survivors."""
    text = block.text
    if not text:
        return 0.0
    n_chars = len(text)
    words = text.split()
    n_words = len(words)
    if n_words < 8:
        return 0.0
    n_sentences = max(1, len(_SENT_END_RE.findall(text)))
    avg_word_len = sum(len(w) for w in words) / n_words
    avg_sent_len = n_words / n_sentences
    n_alpha = sum(c.isalpha() for c in text)
    alpha_ratio = n_alpha / n_chars
    # Punctuation density: too high suggests citation-clutter or tabular residue
    n_punct = sum(1 for c in text if not c.isalnum() and not c.isspace())
    punct_ratio = n_punct / n_chars
    # Vocabulary diversity (cap to avoid favouring short blocks)
    unique = len({w.lower() for w in words})
    vocab_diversity = unique / n_words

    score = 0.0
    # Reward sentences of reasonable length
    if 6 <= avg_sent_len <= 60:
        score += 0.30
    elif 3 <= avg_sent_len <= 90:
        score += 0.15
    # Reward typical word lengths
    if 3.5 <= avg_word_len <= 7.5:
        score += 0.20
    elif 3.0 <= avg_word_len <= 9.0:
        score += 0.10
    # Reward high alpha ratio
    if alpha_ratio > 0.6:
        score += 0.20
    elif alpha_ratio > 0.45:
        score += 0.10
    # Penalise punctuation-heavy blocks
    if punct_ratio < 0.10:
        score += 0.15
    elif punct_ratio < 0.18:
        score += 0.05
    # Reward vocab diversity
    if vocab_diversity > 0.50:
        score += 0.15
    elif vocab_diversity > 0.35:
        score += 0.08

    block.quality = round(score, 3)
    return block.quality


# ---------------------------------------------------------------------------
# pipeline
# ---------------------------------------------------------------------------

# Tags that count as "kept" (potentially become CPT chunks)
KEEP_TAGS = {"prose"}
# Tags worth keeping a tiny note for in the audit log
AUDIT_TAGS = {
    "front_matter", "toc", "toc_marker", "index", "index_marker", "alpha_section",
    "refs", "chemfrag", "fragments", "form", "numeric", "junk",
}


def classify_all(blocks: list[Block], total_lines: int) -> None:
    """Classify each block in order, propagating in-index / in-refs state forward."""
    in_index = False
    in_refs = False
    for blk in blocks:
        pos = blk.line_start / max(1, total_lines)
        tag = classify_block(blk, pos, in_index, in_refs)
        if tag in ("index_marker", "index"):
            in_index = True
        elif tag in ("refs",):
            in_refs = True
        elif tag in ("prose",) and (in_index or in_refs):
            # transient prose between index/refs sections may still belong;
            # require a *strong* prose signal to break out of these modes
            if blk.n_lines >= 3 and (_block_features(blk)["mean_words"] >= 12):
                in_index = False
                in_refs = False


def chunk_prose(blocks: list[Block], target_chars: int = 6000, min_chars: int = 800) -> list[dict]:
    """Group consecutive prose blocks into roughly target_chars-size chunks."""
    chunks: list[dict] = []
    buf: list[Block] = []
    buf_len = 0
    last_idx = -2

    def flush(reason: str = "size"):
        nonlocal buf, buf_len
        if not buf:
            return
        text = "\n\n".join(b.text for b in buf)
        if len(text) >= min_chars:
            qualities = [b.quality for b in buf if b.quality > 0]
            chunks.append({
                "text": text,
                "line_start": buf[0].line_start,
                "line_end": buf[-1].line_end,
                "n_blocks": len(buf),
                "quality": round(statistics.fmean(qualities), 3) if qualities else 0.0,
                "flush_reason": reason,
            })
        buf = []
        buf_len = 0

    for i, blk in enumerate(blocks):
        if blk.tag != "prose":
            # boundary — flush current buffer
            flush("boundary")
            last_idx = -2
            continue
        # If a gap (skipped non-prose) sits between us and the last prose block, flush.
        if last_idx >= 0 and i - last_idx > 1:
            flush("gap")
        last_idx = i
        buf.append(blk)
        buf_len += blk.n_chars + 2
        if buf_len >= target_chars:
            flush("size")
    flush("eof")
    return chunks


def process_file(path: Path, target_chars: int = 6000, min_chunk_chars: int = 800,
                 min_prose_quality: float = 0.35) -> tuple[list[dict], list[dict], dict]:
    """Run the full deterministic pipeline on a single book.

    Returns:
        chunks: list of {text, source, book, chunk_idx, line_start, line_end, quality}
        audit:  list of dropped blocks {line_start, line_end, tag, reason, sample}
        stats:  dict of counts and char totals per tag
    """
    raw = path.read_text(encoding="utf-8", errors="replace")
    raw_chars = len(raw)
    norm = normalise(raw)
    total_lines = norm.count("\n") + 1
    blocks = segment(norm)
    classify_all(blocks, total_lines)
    # quality score prose only
    for blk in blocks:
        if blk.tag == "prose":
            score_quality(blk)
            if blk.quality < min_prose_quality:
                blk.tag = "low_quality_prose"
                blk.reason = f"quality={blk.quality:.2f} below {min_prose_quality}"
    chunks_raw = chunk_prose(blocks, target_chars=target_chars, min_chars=min_chunk_chars)
    chunks = []
    for i, c in enumerate(chunks_raw):
        chunks.append({
            "text": c["text"],
            "source": f"book:{path.stem}",
            "book": path.stem,
            "chunk_idx": i,
            "line_start": c["line_start"],
            "line_end": c["line_end"],
            "quality": c["quality"],
        })

    audit = []
    from collections import Counter
    tag_counts = Counter(b.tag for b in blocks)
    tag_chars = Counter()
    for b in blocks:
        tag_chars[b.tag] += b.n_chars
        if b.tag in AUDIT_TAGS or b.tag == "low_quality_prose":
            sample = b.text[:240].replace("\n", " / ")
            audit.append({
                "line_start": b.line_start, "line_end": b.line_end,
                "tag": b.tag, "reason": b.reason,
                "n_lines": b.n_lines, "n_chars": b.n_chars,
                "sample": sample,
            })

    stats = {
        "raw_chars": raw_chars,
        "normalised_chars": len(norm),
        "total_blocks": len(blocks),
        "tag_block_counts": dict(tag_counts),
        "tag_char_totals": dict(tag_chars),
        "n_chunks": len(chunks),
        "chunk_chars": sum(len(c["text"]) for c in chunks),
        "retention_ratio": round(sum(len(c["text"]) for c in chunks) / max(1, raw_chars), 3),
    }
    return chunks, audit, stats
