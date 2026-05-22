#!/usr/bin/env python3
"""
High-throughput parallel CPT text cleaner using Gemini Flash Lite.

Cleans noisy OCR textbook output for use in continued pretraining.
Uses all available GEMINI_API_KEY{N} keys from .env with concurrent requests
to maximize throughput while respecting per-key rate limits.

Key design choices:
- Async + aiohttp: ~100 concurrent in-flight requests
- Round-robin key rotation with per-key rate-limit cooldown
- Paragraph-boundary chunking (~8K chars / chunk)
- Atomic writes (.tmp + rename) for crash safety
- Per-file checkpointing for resumability
- Order-preserving chunk reassembly
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import aiohttp
from dotenv import load_dotenv

try:
    from tqdm.asyncio import tqdm
except ImportError:
    from tqdm import tqdm

load_dotenv()

# ---------- Defaults ----------
DEFAULT_MODEL = "gemini-3.1-flash-lite-preview"
DEFAULT_CHUNK_SIZE = 8000           # chars/chunk (~2K tokens)
DEFAULT_CONCURRENT_PER_KEY = 2       # parallel requests per API key (conservative)
DEFAULT_MAX_RETRIES = 6
DEFAULT_TIMEOUT_SEC = 240
DEFAULT_OUTPUT_TOKENS = 8192

# ---------- Cleaning prompt ----------
CLEANING_PROMPT = """You are cleaning OCR text from medical/dental textbooks for use in language model continued pretraining.

CLEAN THE TEXT BY:
1. REMOVE entirely: page numbers, page headers/footers, ISBNs, copyright/publisher notices, table-of-contents lines, index entries, "see page X" cross references, figure/table captions that reference missing images, citation lists, reference list entries (e.g., "Smith J. et al., 2003, J Clin Dent..."), bare URLs, author affiliations, dedications, acknowledgments.
2. FIX: OCR errors (broken words from line breaks like "treat- ment", confusions like 0/O, 1/l, rn/m), incorrect spacing, garbled punctuation, weird unicode artifacts.
3. PRESERVE: ALL substantive medical/dental/anatomical/clinical content. Definitions, procedures, mechanisms, classifications, descriptions of disease, anatomy, pharmacology, materials, techniques.
4. FORMAT: Flowing paragraphs with natural sentence boundaries. No bullet-list fragments from broken layouts.

DO NOT:
- Summarize, rewrite, paraphrase, or shorten content
- Add commentary, section headings, or explanations
- Change technical terminology or clinical wording
- Translate or simplify medical language

OUTPUT ONLY the cleaned text. No preamble, no markdown fencing, no explanation.

If the chunk contains ONLY noise (front matter, references, indices, copyright pages), output exactly: <SKIP>

TEXT TO CLEAN:
---
{text}
---"""


# ---------- Data structures ----------
@dataclass
class JobStats:
    files_total: int = 0
    files_done: int = 0
    files_failed: int = 0
    chunks_total: int = 0
    chunks_done: int = 0
    chunks_failed: int = 0
    chunks_skipped: int = 0
    start_time: float = field(default_factory=time.time)
    api_errors: dict = field(default_factory=lambda: defaultdict(int))


# ---------- Helpers ----------
def load_api_keys() -> list[str]:
    keys = []
    for i in range(1, 50):
        k = os.getenv(f"GEMINI_API_KEY{i}")
        if k and k.strip() and not k.startswith("***"):
            keys.append(k.strip())
    return keys


def light_clean_fallback(text: str) -> str:
    """
    Regex-based minimal cleanup used when the Gemini API fails on a chunk.
    Removes the most obvious OCR noise so we never drop content entirely.

    Strips: bare page numbers, ISBNs, lone copyright lines, repeated headers,
    citation-style lines, very short fragment lines, URL-only lines.
    Preserves everything else.
    """
    if not text:
        return ""

    lines = text.split("\n")
    cleaned: list[str] = []
    isbn_re = re.compile(r"\bISBN[\s\-:]*[\d\-X]+", re.I)
    page_num_re = re.compile(r"^\s*\d{1,4}\s*$")
    url_only_re = re.compile(r"^\s*https?://\S+\s*$")
    citation_re = re.compile(
        r"^\s*[A-Z][a-zA-Z'\-]+(\s+[A-Z]\.?){0,3}.*\b(19|20)\d{2}\b.*[;:].*\d+[-\d]*\.?\s*$"
    )
    copyright_re = re.compile(r"^\s*(©|\(c\)|Copyright)\s+\d{4}", re.I)

    for ln in lines:
        s = ln.strip()
        if not s:
            cleaned.append("")
            continue
        if page_num_re.match(s):
            continue
        if url_only_re.match(s):
            continue
        if isbn_re.search(s) and len(s) < 80:
            continue
        if copyright_re.match(s):
            continue
        if citation_re.match(s) and len(s) < 200:
            continue
        cleaned.append(ln)

    # Collapse runs of blank lines
    out: list[str] = []
    prev_blank = False
    for ln in cleaned:
        is_blank = not ln.strip()
        if is_blank and prev_blank:
            continue
        out.append(ln)
        prev_blank = is_blank
    return "\n".join(out).strip()


def chunk_text(text: str, max_chars: int) -> list[str]:
    """Split at paragraph boundaries, packing up to max_chars per chunk."""
    text = text.strip()
    if not text:
        return []
    paragraphs = re.split(r"\n\s*\n", text)
    chunks: list[str] = []
    cur: list[str] = []
    cur_len = 0
    for p in paragraphs:
        p = p.strip()
        if not p:
            continue
        plen = len(p) + 2
        if cur_len + plen > max_chars and cur:
            chunks.append("\n\n".join(cur))
            cur = [p]
            cur_len = plen
        else:
            cur.append(p)
            cur_len += plen
        # Hard-split paragraphs that themselves exceed max_chars
        if len(p) > max_chars * 1.5:
            chunks.append(p[: max_chars])
            remainder = p[max_chars:]
            while len(remainder) > max_chars:
                chunks.append(remainder[: max_chars])
                remainder = remainder[max_chars:]
            cur = [remainder] if remainder else []
            cur_len = len(remainder)
    if cur:
        chunks.append("\n\n".join(cur))
    return chunks


class KeyRotator:
    """Round-robin API key selector with per-key semaphores and cooldowns."""

    def __init__(self, keys: list[str], concurrent_per_key: int):
        if not keys:
            raise ValueError("No API keys provided")
        self.keys = keys
        self.semaphores = {k: asyncio.Semaphore(concurrent_per_key) for k in keys}
        self.cooldowns: dict[str, float] = {}
        self.failures: dict[str, int] = defaultdict(int)
        self._idx = 0
        self._lock = asyncio.Lock()

    async def acquire(self) -> str:
        """Pick the next non-cooled-down key, round-robin."""
        async with self._lock:
            now = time.time()
            n = len(self.keys)
            for _ in range(n):
                k = self.keys[self._idx]
                self._idx = (self._idx + 1) % n
                if self.cooldowns.get(k, 0.0) <= now:
                    return k
            # All in cooldown — pick the one with the earliest cooldown
            k = min(self.keys, key=lambda x: self.cooldowns.get(x, 0.0))
            return k

    def cool_down(self, key: str, seconds: float):
        self.cooldowns[key] = time.time() + seconds
        self.failures[key] += 1


# ---------- Gemini API call ----------
async def call_gemini(
    session: aiohttp.ClientSession,
    chunk: str,
    rotator: KeyRotator,
    model: str,
    stats: JobStats,
    max_retries: int = DEFAULT_MAX_RETRIES,
    timeout_sec: int = DEFAULT_TIMEOUT_SEC,
    output_tokens: int = DEFAULT_OUTPUT_TOKENS,
) -> str | None:
    """Returns cleaned text, '' if skipped, None on hard failure."""
    prompt = CLEANING_PROMPT.format(text=chunk)
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.0,
            "topP": 0.95,
            "maxOutputTokens": output_tokens,
            # Flash-Lite default is "minimal"; set explicitly for predictability
            # and to make sure we don't pay for thinking tokens on cleanup work.
            "thinkingConfig": {"thinkingLevel": "minimal"},
        },
        "safetySettings": [
            {"category": cat, "threshold": "BLOCK_NONE"}
            for cat in (
                "HARM_CATEGORY_HARASSMENT",
                "HARM_CATEGORY_HATE_SPEECH",
                "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "HARM_CATEGORY_DANGEROUS_CONTENT",
            )
        ],
    }

    backoff = 1.0
    for attempt in range(max_retries):
        key = await rotator.acquire()
        sem = rotator.semaphores[key]
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{model}:generateContent?key={key}"
        )

        async with sem:
            try:
                async with session.post(
                    url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=timeout_sec),
                    headers={"Content-Type": "application/json"},
                ) as resp:
                    if resp.status == 429:
                        rotator.cool_down(key, 30.0)
                        stats.api_errors["429"] += 1
                        await asyncio.sleep(min(backoff, 5))
                        backoff *= 1.5
                        continue
                    if resp.status in (401, 403):
                        rotator.cool_down(key, 3600.0)
                        stats.api_errors[str(resp.status)] += 1
                        continue
                    if resp.status >= 500:
                        stats.api_errors[str(resp.status)] += 1
                        await asyncio.sleep(min(backoff, 8))
                        backoff *= 2
                        continue
                    if resp.status != 200:
                        stats.api_errors[str(resp.status)] += 1
                        try:
                            err_body = await resp.text()
                        except Exception:
                            err_body = ""
                        if attempt == max_retries - 1:
                            return None
                        await asyncio.sleep(backoff)
                        backoff *= 2
                        continue

                    data = await resp.json()
            except asyncio.TimeoutError:
                stats.api_errors["timeout"] += 1
                if attempt == max_retries - 1:
                    return None
                await asyncio.sleep(backoff)
                backoff *= 2
                continue
            except aiohttp.ClientError as e:
                stats.api_errors[f"client:{type(e).__name__}"] += 1
                if attempt == max_retries - 1:
                    return None
                await asyncio.sleep(backoff)
                backoff *= 2
                continue
            except Exception as e:
                stats.api_errors[f"other:{type(e).__name__}"] += 1
                if attempt == max_retries - 1:
                    return None
                await asyncio.sleep(backoff)
                backoff *= 2
                continue

        # Parse response
        try:
            cands = data.get("candidates", [])
            if not cands:
                # Probably blocked by safety filter; try once more then skip
                stats.api_errors["no_candidates"] += 1
                if attempt == max_retries - 1:
                    return ""
                continue
            parts = cands[0].get("content", {}).get("parts", [])
            text_out = "".join(p.get("text", "") for p in parts).strip()
            if not text_out:
                stats.api_errors["empty"] += 1
                if attempt == max_retries - 1:
                    return ""
                continue
            if text_out == "<SKIP>" or text_out.startswith("<SKIP>"):
                return ""
            # Strip any model preamble (rare with this prompt but safe)
            text_out = re.sub(r"^```[a-zA-Z]*\n?", "", text_out)
            text_out = re.sub(r"\n?```$", "", text_out)
            return text_out.strip()
        except (KeyError, IndexError, TypeError):
            stats.api_errors["parse"] += 1
            if attempt == max_retries - 1:
                return None
            continue

    return None


# ---------- File-level cleaning ----------
async def clean_file(
    session: aiohttp.ClientSession,
    src_path: Path,
    dst_path: Path,
    rotator: KeyRotator,
    model: str,
    chunk_size: int,
    stats: JobStats,
    file_pbar: tqdm,
) -> tuple[int, int]:
    """Clean one file. Returns (cleaned_chunk_count, total_chunk_count)."""
    text = src_path.read_text(encoding="utf-8", errors="ignore")
    chunks = chunk_text(text, chunk_size)
    if not chunks:
        # nothing to clean, write empty
        tmp = dst_path.with_suffix(dst_path.suffix + ".tmp")
        tmp.write_text("", encoding="utf-8")
        tmp.replace(dst_path)
        return 0, 0

    stats.chunks_total += len(chunks)

    # Submit all chunks in parallel, gather in order
    tasks = [call_gemini(session, c, rotator, model, stats) for c in chunks]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # CRITICAL: never drop content. On API failure, fall back to regex-cleaned
    # original chunk so we preserve all valuable text.
    cleaned_parts: list[str] = []
    for original_chunk, r in zip(chunks, results):
        if isinstance(r, Exception) or r is None:
            # API failed - keep original with light regex cleanup
            stats.chunks_failed += 1
            fallback = light_clean_fallback(original_chunk)
            if fallback:
                cleaned_parts.append(fallback)
        elif r == "":
            # Model explicitly said <SKIP> - actually noise, drop it
            stats.chunks_skipped += 1
        else:
            cleaned_parts.append(r)
            stats.chunks_done += 1

    if not cleaned_parts:
        return 0, len(chunks)

    cleaned = "\n\n".join(cleaned_parts)
    tmp = dst_path.with_suffix(dst_path.suffix + ".tmp")
    tmp.write_text(cleaned, encoding="utf-8")
    tmp.replace(dst_path)
    file_pbar.update(1)
    return len(cleaned_parts), len(chunks)


# ---------- Checkpointing ----------
def load_checkpoint(path: Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_checkpoint(path: Path, ckpt: dict):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(ckpt, indent=2), encoding="utf-8")
    tmp.replace(path)


# ---------- Main orchestration ----------
async def run(args):
    keys = load_api_keys()
    print(f"[init] Loaded {len(keys)} API keys")
    if not keys:
        print("[error] No GEMINI_API_KEY{N} found in environment", file=sys.stderr)
        return 2

    src_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir or args.input_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = out_dir / "cleaning_progress.json"
    ckpt = load_checkpoint(ckpt_path) if not args.no_resume else {}
    done_files = set(ckpt.get("done_files", []))

    files = sorted(p for p in src_dir.glob("*.txt") if not p.name.startswith("."))
    files = [f for f in files if args.no_resume or f.name not in done_files]
    if args.match:
        files = [f for f in files if args.match in f.name]
    if args.limit:
        files = files[: args.limit]

    stats = JobStats(files_total=len(files))
    print(f"[init] Cleaning {len(files)} files (resume skipped {len(done_files)})")
    print(f"[init] Model: {args.model}")
    print(f"[init] Concurrency: {len(keys)} keys x {args.concurrent_per_key} = "
          f"{len(keys) * args.concurrent_per_key} max in-flight")
    print(f"[init] Chunk size: {args.chunk_size} chars")

    rotator = KeyRotator(keys, args.concurrent_per_key)

    # Pre-flight: validate model name with 1 small request before launching the swarm.
    # Distinguish hard failures (404 = bad model name) from transient ones (429 = busy).
    async with aiohttp.ClientSession() as test_sess:
        test_stats = JobStats()
        test_result = await call_gemini(
            test_sess,
            "The patient presented with mild dental caries.",
            rotator, args.model, test_stats, max_retries=3, timeout_sec=30,
        )
        errs = dict(test_stats.api_errors)
        if test_result is not None:
            print(f"[pre-flight] Model '{args.model}' reachable. Starting full run.")
        elif errs and any(c in errs for c in ("404", "400")):
            print(f"[pre-flight] HARD FAIL. API errors: {errs}")
            print(f"[pre-flight] Model '{args.model}' likely doesn't exist.")
            print(f"[pre-flight] Try:  --model gemini-2.5-flash-lite")
            return 3
        elif errs and "429" in errs:
            print(f"[pre-flight] Rate-limited ({errs}). Keys still cooling down from prior run.")
            print(f"[pre-flight] Model name OK. Proceeding — chunk-level retries will handle 429s.")
        else:
            print(f"[pre-flight] WARNING: no successful response. Errors: {errs}")
            print(f"[pre-flight] Proceeding anyway since model name is documented to exist.")

    # File-level parallelism: process N files at a time, but chunks within
    # each file fan out fully across all keys.
    file_sem = asyncio.Semaphore(args.parallel_files)
    file_pbar = tqdm(total=len(files), desc="Files", unit="file", position=0)

    connector = aiohttp.TCPConnector(limit=len(keys) * args.concurrent_per_key * 2)
    async with aiohttp.ClientSession(connector=connector) as session:

        async def process_one(src: Path):
            async with file_sem:
                dst = out_dir / src.name
                try:
                    ok, total = await clean_file(
                        session, src, dst, rotator, args.model,
                        args.chunk_size, stats, file_pbar,
                    )
                    if ok > 0:
                        stats.files_done += 1
                        done_files.add(src.name)
                        ckpt["done_files"] = sorted(done_files)
                        ckpt["last_update"] = time.strftime("%Y-%m-%dT%H:%M:%S")
                        save_checkpoint(ckpt_path, ckpt)
                    else:
                        stats.files_failed += 1
                except Exception as e:
                    stats.files_failed += 1
                    print(f"[error] {src.name}: {e}", file=sys.stderr)

        await asyncio.gather(*(process_one(f) for f in files))

    file_pbar.close()

    elapsed = time.time() - stats.start_time
    print("\n" + "=" * 70)
    print("CLEANING COMPLETE")
    print("=" * 70)
    print(f"Files:  {stats.files_done} ok / {stats.files_failed} failed / {len(files)} total")
    print(f"Chunks: {stats.chunks_done} ok / {stats.chunks_skipped} skipped / "
          f"{stats.chunks_failed} failed / {stats.chunks_total} total")
    print(f"Elapsed: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    if stats.chunks_done:
        print(f"Throughput: {stats.chunks_done / elapsed:.2f} chunks/sec")
    if stats.api_errors:
        print(f"API errors: {dict(stats.api_errors)}")
    print(f"Checkpoint: {ckpt_path}")
    return 0


def parse_args():
    p = argparse.ArgumentParser(description="Async Gemini-based OCR text cleaner")
    p.add_argument("--input-dir", default="cpt_prepared/core_cpt_text_cleaned",
                   help="Directory of .txt files to clean")
    p.add_argument("--output-dir", default=None,
                   help="Output dir (defaults to --input-dir for in-place rewrite)")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    p.add_argument("--concurrent-per-key", type=int, default=DEFAULT_CONCURRENT_PER_KEY,
                   help="Max in-flight requests per API key")
    p.add_argument("--parallel-files", type=int, default=8,
                   help="Number of files processed concurrently (each fans out chunks)")
    p.add_argument("--limit", type=int, default=None, help="Cap on number of files")
    p.add_argument("--match", default=None, help="Only process files containing this substring")
    p.add_argument("--no-resume", action="store_true",
                   help="Reprocess everything, ignore existing checkpoint")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    try:
        rc = asyncio.run(run(args))
    except KeyboardInterrupt:
        print("\n[interrupted]", file=sys.stderr)
        rc = 130
    sys.exit(rc)
