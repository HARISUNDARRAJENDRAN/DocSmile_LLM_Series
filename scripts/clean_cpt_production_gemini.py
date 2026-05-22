#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import aiohttp
from dotenv import load_dotenv
from tqdm import tqdm


DEFAULT_MODEL = "gemini-3.1-flash-lite"
DEFAULT_CHUNK_CHARS = 7000
DEFAULT_OUTPUT_TOKENS = 8192


PROMPT = """You are cleaning OCR/extracted text from medical and dental textbooks for continued pretraining.

Goal: preserve high-value textbook knowledge while removing layout noise and extraction artifacts.

Remove completely:
- repeated page headers, footers, running titles, page numbers
- table of contents pages, index entries, reference/bibliography lists, copyright/publisher pages
- ISBNs, author affiliations, acknowledgments, dedications, bare URLs
- figure/table captions that are useless without the missing image
- repeated paragraphs, repeated question-answer templates, duplicated fragments
- isolated OCR garbage, broken page navigation, "see page X" cross references

Fix carefully:
- hyphenated line-break words, broken sentence wrapping, spacing, OCR punctuation
- obvious OCR confusions only when context is clear

Preserve:
- all substantive dental, medical, anatomical, pathological, pharmacological, materials, procedural, clinical, and scientific content
- meaningful definitions, mechanisms, classifications, diagnostic reasoning, contraindications, complications, and treatment principles
- useful figure/table captions only if they read as standalone educational text

Do not summarize, paraphrase, simplify, add commentary, add headings, or convert this into instructions.
Output only cleaned textbook prose. If the chunk is only removable noise, output exactly <SKIP>.

TEXT:
---
{text}
---"""


@dataclass
class Stats:
    files_total: int = 0
    files_done: int = 0
    files_failed: int = 0
    chunks_total: int = 0
    chunks_done: int = 0
    chunks_skipped: int = 0
    chunks_failed: int = 0
    api_errors: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    started: float = field(default_factory=time.time)


class KeyPool:
    def __init__(
        self,
        keys: list[tuple[str, str]],
        concurrent_per_key: int,
        per_key_rpm: float,
        max_requests_per_key: int,
    ) -> None:
        self.keys = keys
        self.semaphores = {name: asyncio.Semaphore(concurrent_per_key) for name, _ in keys}
        self.cooldowns: dict[str, float] = {}
        self.last_request_at: dict[str, float] = {}
        self.request_counts: dict[str, int] = defaultdict(int)
        self.spacing_locks = {name: asyncio.Lock() for name, _ in keys}
        self.per_key_min_interval = 60.0 / max(per_key_rpm, 0.1)
        self.extra_spacing_sec = 0.0
        self.max_requests_per_key = max_requests_per_key
        self.index = 0
        self.lock = asyncio.Lock()

    async def acquire(self) -> tuple[str, str]:
        async with self.lock:
            now = time.time()
            usable = [
                (name, value)
                for name, value in self.keys
                if not self.max_requests_per_key or self.request_counts[name] < self.max_requests_per_key
            ]
            if not usable:
                raise RuntimeError("All Gemini keys reached --requests-per-key for this run.")
            for _ in range(len(self.keys)):
                name, value = self.keys[self.index]
                self.index = (self.index + 1) % len(self.keys)
                if self.max_requests_per_key and self.request_counts[name] >= self.max_requests_per_key:
                    continue
                if self.cooldowns.get(name, 0.0) <= now:
                    return name, value
            return min(usable, key=lambda kv: self.cooldowns.get(kv[0], 0.0))

    def cool_down(self, name: str, seconds: float) -> None:
        self.cooldowns[name] = time.time() + seconds

    async def throttle(self, key_name: str, spacing_sec: float) -> None:
        async with self.spacing_locks[key_name]:
            now = time.time()
            effective_spacing = max(self.per_key_min_interval, spacing_sec)
            wait = self.last_request_at.get(key_name, 0.0) + effective_spacing - now
            if wait > 0:
                await asyncio.sleep(wait)
            self.last_request_at[key_name] = time.time()
            self.request_counts[key_name] += 1


def load_keys(env_path: Path) -> list[tuple[str, str]]:
    load_dotenv(env_path)
    keys: list[tuple[str, str]] = []
    seen: set[str] = set()
    for name in ["GEMINI_API_KEY", "GOOGLE_API_KEY", *[f"GEMINI_API_KEY{i}" for i in range(1, 80)]]:
        value = os.environ.get(name, "").strip()
        if value and not value.startswith("***") and value not in seen:
            keys.append((name, value))
            seen.add(value)
    return keys


def normalize_text(text: str) -> str:
    text = text.replace("\ufeff", "").replace("\x00", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"(?<=\w)-\n(?=\w)", "", text)
    text = re.sub(r"(?<=\w)\n(?=[a-z])", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text


def drop_repeated_short_lines(text: str, min_repeats: int = 4) -> str:
    lines = text.splitlines()
    norms = [re.sub(r"\s+", " ", line.strip()).lower() for line in lines]
    counts = Counter(n for n in norms if 3 <= len(n) <= 100)
    repeated = {n for n, c in counts.items() if c >= min_repeats}
    kept = [line for line, norm in zip(lines, norms) if norm not in repeated]
    return "\n".join(kept)


def deterministic_prefilter(text: str) -> str:
    text = drop_repeated_short_lines(normalize_text(text))
    out: list[str] = []
    page_num = re.compile(r"^\s*(?:page\s*)?\d{1,4}\s*$", re.I)
    isbn = re.compile(r"\bISBN(?:-1[03])?[\s:]*[\d\-Xx]+")
    copyright_line = re.compile(r"^\s*(copyright|\(c\)|©)\b", re.I)
    url_only = re.compile(r"^\s*(https?://|www\.)\S+\s*$", re.I)
    for line in text.splitlines():
        s = line.strip()
        if page_num.match(s):
            continue
        if url_only.match(s):
            continue
        if copyright_line.match(s):
            continue
        if isbn.search(s) and len(s) < 120:
            continue
        out.append(line)
    text = "\n".join(out)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text(text: str, max_chars: int) -> list[str]:
    text = deterministic_prefilter(text)
    if not text:
        return []
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    cur: list[str] = []
    cur_len = 0
    for paragraph in paragraphs:
        if len(paragraph) > max_chars:
            if cur:
                chunks.append("\n\n".join(cur))
                cur, cur_len = [], 0
            start = 0
            while start < len(paragraph):
                chunks.append(paragraph[start : start + max_chars])
                start += max_chars
            continue
        needed = len(paragraph) + 2
        if cur and cur_len + needed > max_chars:
            chunks.append("\n\n".join(cur))
            cur, cur_len = [paragraph], needed
        else:
            cur.append(paragraph)
            cur_len += needed
    if cur:
        chunks.append("\n\n".join(cur))
    return chunks


def clean_model_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    if text.startswith("<SKIP>"):
        return ""
    return deterministic_prefilter(text)


async def call_gemini(
    session: aiohttp.ClientSession,
    pool: KeyPool,
    model: str,
    chunk: str,
    stats: Stats,
    max_retries: int,
    timeout_sec: int,
    output_tokens: int,
    request_spacing_sec: float,
) -> str | None:
    payload = {
        "contents": [{"parts": [{"text": PROMPT.format(text=chunk)}]}],
        "generationConfig": {
            "temperature": 0.0,
            "topP": 0.95,
            "maxOutputTokens": output_tokens,
        },
        "safetySettings": [
            {"category": cat, "threshold": "BLOCK_NONE"}
            for cat in [
                "HARM_CATEGORY_HARASSMENT",
                "HARM_CATEGORY_HATE_SPEECH",
                "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "HARM_CATEGORY_DANGEROUS_CONTENT",
            ]
        ],
    }
    backoff = 1.0
    for attempt in range(max_retries):
        key_name, key = await pool.acquire()
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
        async with pool.semaphores[key_name]:
            try:
                await pool.throttle(key_name, request_spacing_sec)
                async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=timeout_sec)) as resp:
                    if resp.status == 429:
                        stats.api_errors["429"] += 1
                        pool.cool_down(key_name, 35)
                        await asyncio.sleep(min(backoff, 8))
                        backoff *= 1.7
                        continue
                    if resp.status in (401, 403):
                        stats.api_errors[f"{resp.status}:{key_name}"] += 1
                        pool.cool_down(key_name, 3600)
                        continue
                    if resp.status >= 500:
                        stats.api_errors[str(resp.status)] += 1
                        await asyncio.sleep(min(backoff, 10))
                        backoff *= 2
                        continue
                    if resp.status != 200:
                        stats.api_errors[str(resp.status)] += 1
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
            except aiohttp.ClientError as exc:
                stats.api_errors[type(exc).__name__] += 1
                if attempt == max_retries - 1:
                    return None
                await asyncio.sleep(backoff)
                backoff *= 2
                continue

        try:
            parts = data["candidates"][0]["content"]["parts"]
            text = "".join(part.get("text", "") for part in parts)
            return clean_model_text(text)
        except Exception:
            stats.api_errors["parse"] += 1
            if attempt == max_retries - 1:
                return None
    return None


def write_json_atomic(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


def write_text_atomic(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def load_done(progress_path: Path, no_resume: bool) -> set[str]:
    if no_resume or not progress_path.exists():
        return set()
    try:
        payload = json.loads(progress_path.read_text(encoding="utf-8"))
    except Exception:
        return set()
    return set(payload.get("completed_files", []))


def progress_payload(args: argparse.Namespace, stats: Stats, completed: set[str], status: str, **extra) -> dict:
    payload = {
        "status": status,
        "total_files": stats.files_total,
        "done_files": stats.files_done,
        "skipped_files": len(completed),
        "error_files": stats.files_failed,
        "chunks_total": stats.chunks_total,
        "chunks_done": stats.chunks_done,
        "chunks_skipped": stats.chunks_skipped,
        "chunks_failed": stats.chunks_failed,
        "api_errors": dict(stats.api_errors),
        "model": args.model,
        "input_dir": args.input_dir,
        "output_dir": args.output_dir,
        "last_update": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "completed_files": sorted(completed),
    }
    payload.update(extra)
    return payload


async def clean_file(
    src: Path,
    dst: Path,
    session: aiohttp.ClientSession,
    pool: KeyPool,
    args: argparse.Namespace,
    stats: Stats,
    progress_path: Path,
    completed: set[str],
) -> tuple[bool, dict]:
    raw = src.read_text(encoding="utf-8", errors="ignore")
    chunks = chunk_text(raw, args.chunk_chars)
    if args.max_chunks_per_file:
        chunks = chunks[: args.max_chunks_per_file]
    stats.chunks_total += len(chunks)
    if not chunks:
        write_text_atomic(dst, "")
        return True, {"input_chars": len(raw), "output_chars": 0, "chunks": 0, "retention": 0.0}

    async def run_chunk(index: int, chunk: str) -> tuple[int, str | None]:
        result = await call_gemini(
            session,
            pool,
            args.model,
            chunk,
            stats,
            args.max_retries,
            args.timeout_sec,
            args.output_tokens,
            args.request_spacing_sec,
        )
        return index, result

    results: list[str | None] = [None] * len(chunks)
    pending = [asyncio.create_task(run_chunk(index, chunk)) for index, chunk in enumerate(chunks)]
    completed_chunks = 0
    for task in asyncio.as_completed(pending):
        index, result = await task
        results[index] = result
        completed_chunks += 1
        if completed_chunks == 1 or completed_chunks % args.progress_every_chunks == 0 or completed_chunks == len(chunks):
            write_json_atomic(
                progress_path,
                progress_payload(
                    args,
                    stats,
                    completed,
                    "running",
                    current_file=src.name,
                    current_chunk=completed_chunks,
                    current_chunk_total=len(chunks),
                    current_file_progress=round(100 * completed_chunks / max(1, len(chunks)), 2),
                ),
            )
    cleaned_parts: list[str] = []
    failed = 0
    skipped = 0
    for result in results:
        if result is None:
            failed += 1
            stats.chunks_failed += 1
        elif not result:
            skipped += 1
            stats.chunks_skipped += 1
        else:
            cleaned_parts.append(result)
            stats.chunks_done += 1

    if failed and args.fail_on_chunk_error:
        return False, {"input_chars": len(raw), "chunks": len(chunks), "failed_chunks": failed}

    cleaned = "\n\n".join(cleaned_parts).strip()
    cleaned = deterministic_prefilter(cleaned)
    if not cleaned:
        return False, {"input_chars": len(raw), "chunks": len(chunks), "failed_chunks": failed, "skipped_chunks": skipped}

    write_text_atomic(dst, cleaned + "\n")
    return True, {
        "input_chars": len(raw),
        "output_chars": len(cleaned),
        "chunks": len(chunks),
        "failed_chunks": failed,
        "skipped_chunks": skipped,
        "retention": round(len(cleaned) / max(1, len(raw)), 4),
    }


async def run(args: argparse.Namespace) -> int:
    root = Path(__file__).resolve().parents[1]
    env_path = root / ".env"
    keys = load_keys(env_path)
    if not keys:
        print("No Gemini API keys found in .env/environment.", file=sys.stderr)
        return 2

    src_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    report_dir = out_dir / "_reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    progress_path = out_dir / "progress.json"
    completed = load_done(progress_path, args.no_resume)

    files = sorted(path for path in src_dir.glob("*.txt") if path.name not in completed)
    if args.match:
        files = [path for path in files if args.match.lower() in path.name.lower()]
    if args.limit:
        files = files[: args.limit]

    stats = Stats(files_total=len(files))
    print(f"[init] input={src_dir}")
    print(f"[init] output={out_dir}")
    print(f"[init] files={len(files)} resume_completed={len(completed)}")
    print(f"[init] model={args.model}")
    print(f"[init] keys={len(keys)} ({', '.join(name for name, _ in keys)})")
    print(f"[init] concurrency={len(keys) * args.concurrent_per_key} requests max")
    print(f"[init] per_key_rpm={args.per_key_rpm}")
    print(f"[init] request_spacing_sec={args.request_spacing_sec} per key")
    print(f"[init] requests_per_key={args.requests_per_key or 'unlimited'}")
    print(f"[init] fail_on_chunk_error={args.fail_on_chunk_error}")

    if not files:
        print(f"[error] No .txt files found in {src_dir}. Build cpt_raw_text_v2 first.", file=sys.stderr)
        return 2

    write_json_atomic(progress_path, progress_payload(args, stats, completed, "running"))
    pool = KeyPool(keys, args.concurrent_per_key, args.per_key_rpm, args.requests_per_key)

    connector = aiohttp.TCPConnector(limit=max(8, len(keys) * args.concurrent_per_key * 2))
    timeout = aiohttp.ClientTimeout(total=args.timeout_sec)
    pbar = tqdm(total=len(files), unit="file", desc="CPT clean")
    file_sem = asyncio.Semaphore(args.parallel_files)
    file_reports: dict[str, dict] = {}

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        async def worker(src: Path) -> None:
            async with file_sem:
                dst = out_dir / src.name
                write_json_atomic(
                    progress_path,
                    progress_payload(args, stats, completed, "running", current_file=src.name),
                )
                ok, report = await clean_file(src, dst, session, pool, args, stats, progress_path, completed)
                report["status"] = "done" if ok else "failed"
                file_reports[src.name] = report
                if ok:
                    stats.files_done += 1
                    completed.add(src.name)
                    pbar.update(1)
                else:
                    stats.files_failed += 1
                write_json_atomic(report_dir / "file_reports.json", file_reports)
                write_json_atomic(
                    progress_path,
                    progress_payload(
                        args,
                        stats,
                        completed,
                        "running",
                        last_file=src.name,
                        last_status=report["status"],
                    ),
                )

        await asyncio.gather(*(worker(path) for path in files))

    pbar.close()
    elapsed = time.time() - stats.started
    final_status = "completed" if stats.files_failed == 0 else "completed_with_errors"
    write_json_atomic(
        progress_path,
        progress_payload(args, stats, completed, final_status, elapsed_sec=round(elapsed, 1)),
    )
    print(f"[done] status={final_status} files_done={stats.files_done} files_failed={stats.files_failed}")
    print(f"[done] chunks_done={stats.chunks_done} skipped={stats.chunks_skipped} failed={stats.chunks_failed}")
    print(f"[done] progress={progress_path}")
    print(f"[done] reports={report_dir / 'file_reports.json'}")
    return 0 if stats.files_failed == 0 else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Production Gemini cleaner for CPT textbook text.")
    parser.add_argument("--input-dir", default="cpt_prepared/core_cpt_text")
    parser.add_argument("--output-dir", default="cpt_prepared/core_cpt_text_gemini_clean_v2")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--chunk-chars", type=int, default=DEFAULT_CHUNK_CHARS)
    parser.add_argument("--output-tokens", type=int, default=DEFAULT_OUTPUT_TOKENS)
    parser.add_argument("--concurrent-per-key", type=int, default=1)
    parser.add_argument("--parallel-files", type=int, default=8)
    parser.add_argument("--max-retries", type=int, default=6)
    parser.add_argument("--timeout-sec", type=int, default=240)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--match", default="")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--fail-on-chunk-error", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--request-spacing-sec",
        type=float,
        default=0.0,
        help="Extra per-key delay between Gemini requests. The per-key RPM limit is always enforced.",
    )
    parser.add_argument("--per-key-rpm", type=float, default=15.0, help="Per-key request rate. Use <=15 for Gemini free-tier keys.")
    parser.add_argument("--requests-per-key", type=int, default=0, help="Optional per-run request cap per key. Use 20 to respect a 20 RPD limit.")
    parser.add_argument("--max-chunks-per-file", type=int, default=0, help="Debug/smoke only; do not use for full CPT.")
    parser.add_argument("--progress-every-chunks", type=int, default=5)
    return parser.parse_args()


if __name__ == "__main__":
    sys.exit(asyncio.run(run(parse_args())))
