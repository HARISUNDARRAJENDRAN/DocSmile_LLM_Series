from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import random
import re
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path


API_KEY_NAME_RE = re.compile(r"^(?:GEMINI_API_KEY|GOOGLE_API_KEY)(?:\d+)?$")
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
IMAGE_PLACEHOLDER_RE = re.compile(r"(<image>|!\[[^\]]*]\([^)]+\)|<img\b[^>]*>)", re.IGNORECASE)
FIGURE_LINE_RE = re.compile(r"\b(fig(?:ure)?\.?|table)\s*[-.:]?\s*[0-9ivxlcdm]+", re.IGNORECASE)


@dataclass(frozen=True)
class ImageRecord:
    image_id: str
    image_path: str
    book_stem: str
    book_md_path: str
    image_index: int
    image_count: int


def natural_key(value: str) -> list[object]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", value)]


def load_api_keys(env_path: Path) -> list[tuple[str, str]]:
    candidates: list[tuple[str, str]] = []

    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export ") :].lstrip()
            if "=" not in line:
                continue
            name, value = line.split("=", 1)
            name = name.strip()
            value = value.strip().strip('"').strip("'")
            if API_KEY_NAME_RE.match(name) and value:
                candidates.append((name, value))

    for name, value in os.environ.items():
        if API_KEY_NAME_RE.match(name) and value:
            candidates.append((name, value))

    candidates.sort(key=lambda item: natural_key(item[0]))

    seen = set()
    keys: list[tuple[str, str]] = []
    for name, value in candidates:
        if value in seen:
            continue
        seen.add(value)
        keys.append((name, value))
    return keys


def load_books(book_dirs: list[Path]) -> dict[str, Path]:
    books: dict[str, Path] = {}
    for book_dir in book_dirs:
        if not book_dir.exists():
            continue
        for path in sorted(book_dir.glob("*.md")):
            # Earlier dirs win. Put core_cpt before rl when invoking if you want
            # core books to be preferred for duplicate stems.
            books.setdefault(path.stem, path)
    return books


def collect_image_records(image_root: Path, books: dict[str, Path]) -> list[ImageRecord]:
    records: list[ImageRecord] = []
    if not image_root.exists():
        raise FileNotFoundError(f"Image root not found: {image_root}")

    for book_dir in sorted([p for p in image_root.iterdir() if p.is_dir()], key=lambda p: natural_key(p.name)):
        book_path = books.get(book_dir.name)
        if not book_path:
            continue
        images = sorted(
            [p for p in book_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS],
            key=lambda p: natural_key(p.name),
        )
        for idx, image_path in enumerate(images):
            records.append(
                ImageRecord(
                    image_id=f"{book_dir.name}/{image_path.name}",
                    image_path=str(image_path.resolve()),
                    book_stem=book_dir.name,
                    book_md_path=str(book_path.resolve()),
                    image_index=idx,
                    image_count=len(images),
                )
            )
    return records


def anchor_positions(markdown: str) -> list[int]:
    anchors = [match.start() for match in IMAGE_PLACEHOLDER_RE.finditer(markdown)]
    if anchors:
        return anchors

    line_offsets = []
    offset = 0
    for line in markdown.splitlines(keepends=True):
        if FIGURE_LINE_RE.search(line):
            line_offsets.append(offset)
        offset += len(line)
    return line_offsets


def context_for_image(markdown: str, image_index: int, image_count: int, context_chars: int) -> tuple[str, dict]:
    anchors = anchor_positions(markdown)
    if anchors:
        if image_count > 1 and len(anchors) != image_count:
            anchor_idx = round(image_index * (len(anchors) - 1) / (image_count - 1))
            strategy = "scaled_placeholder_or_figure_anchor"
        else:
            anchor_idx = min(image_index, len(anchors) - 1)
            strategy = "placeholder_or_figure_anchor"
        center = anchors[anchor_idx]
    else:
        ratio = (image_index + 0.5) / max(1, image_count)
        center = int(len(markdown) * ratio)
        anchor_idx = -1
        strategy = "proportional_position_fallback"

    start = max(0, center - context_chars)
    end = min(len(markdown), center + context_chars)
    context = markdown[start:end].strip()

    # Trim to paragraph-ish boundaries when possible.
    if start > 0:
        first_break = context.find("\n\n")
        if 0 <= first_break < min(1000, len(context)):
            context = context[first_break + 2 :]
    if end < len(markdown):
        last_break = context.rfind("\n\n")
        if last_break > max(0, len(context) - 1000):
            context = context[:last_break]

    meta = {
        "context_strategy": strategy,
        "anchor_count": len(anchors),
        "image_count": image_count,
        "anchor_index": anchor_idx,
        "context_start": start,
        "context_end": end,
        "context_chars": len(context),
    }
    return context, meta


def compact_text(text: str, max_chars: int) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0].strip()


def build_prompt(record: ImageRecord, context: str) -> str:
    return (
        "You are annotating dental textbook images for a vision-language training dataset.\n"
        "Use BOTH the image and the provided nearby book context. The context may be imperfectly aligned, "
        "so never invent details that are not visible or text-supported.\n\n"
        "Return ONLY valid JSON with this schema:\n"
        "{\n"
        '  "image_type": "radiograph|clinical_photo|histology|diagram|chart_table|instrument_material|anatomy_illustration|other",\n'
        '  "subject_area": "short dental subject area",\n'
        '  "title": "short descriptive title",\n'
        '  "caption": "1-3 sentence grounded caption",\n'
        '  "visible_findings": ["visible image facts only"],\n'
        '  "dental_entities": ["anatomical structures, materials, instruments, diseases, procedures"],\n'
        '  "text_visible_in_image": ["OCR-like labels or text visible in the image, if any"],\n'
        '  "educational_use": "how this image helps dental learning",\n'
        '  "qa_pairs": [{"question": "image-grounded question", "answer": "concise answer"}],\n'
        '  "safety_notes": ["diagnostic caveats if applicable"],\n'
        '  "context_relevance": "high|medium|low",\n'
        '  "confidence": 0.0\n'
        "}\n\n"
        "Rules:\n"
        "- First decide whether the context actually describes the image. If not, set context_relevance to low.\n"
        "- If context_relevance is low, rely on visible image evidence and do not copy unrelated context into title/caption.\n"
        "- Do not identify patients or imply a definitive diagnosis from an image alone.\n"
        "- If the image is a radiograph or clinical photo, use educational wording, not treatment advice.\n"
        "- Keep qa_pairs to 3-5 useful examples.\n"
        "- confidence must be a number from 0 to 1.\n\n"
        f"BOOK: {record.book_stem}\n"
        f"IMAGE FILE: {Path(record.image_path).name}\n"
        f"IMAGE POSITION IN BOOK FOLDER: {record.image_index + 1}/{record.image_count}\n\n"
        "NEARBY BOOK CONTEXT:\n"
        f"{context}"
    )


def image_part(path: Path) -> dict:
    mime_type = mimetypes.guess_type(path.name)[0] or "image/png"
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return {"inline_data": {"mime_type": mime_type, "data": data}}


def parse_json_response(text: str) -> tuple[dict | None, str]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end > start:
        cleaned = cleaned[start : end + 1]
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        return None, text
    return parsed if isinstance(parsed, dict) else None, text


def annotation_quality_flags(annotation: dict, context_meta: dict) -> list[str]:
    flags: list[str] = []
    relevance = str(annotation.get("context_relevance", "")).strip().lower()
    try:
        confidence = float(annotation.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0

    if relevance == "low":
        flags.append("low_context_relevance")
    if confidence < 0.45:
        flags.append("low_confidence")
    if context_meta.get("context_strategy") in {"proportional_position_fallback", "scaled_placeholder_or_figure_anchor"}:
        flags.append(f"context_{context_meta.get('context_strategy')}")
    if context_meta.get("anchor_count") and context_meta.get("image_count"):
        anchor_count = int(context_meta["anchor_count"])
        image_count = int(context_meta["image_count"])
        if image_count and abs(anchor_count - image_count) / max(image_count, 1) > 0.2:
            flags.append("anchor_image_count_mismatch")
    return flags


def _parse_retry_after(headers) -> float:
    if not headers:
        return 0.0
    try:
        value = headers.get("Retry-After")
    except AttributeError:
        value = None
    if not value:
        return 0.0
    try:
        return float(value)
    except ValueError:
        return 0.0


def call_gemini(
    api_key: str,
    model: str,
    record: ImageRecord,
    context: str,
    max_retries: int,
    sleep_sec: float,
    retry_max_sleep: float,
) -> str:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    image_part(Path(record.image_path)),
                    {"text": build_prompt(record, context)},
                ],
            }
        ],
        "generationConfig": {
            "temperature": 0.15,
            "topP": 0.9,
            "responseMimeType": "application/json",
        },
    }

    data = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}

    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(url, data=data, headers=headers)
            with urllib.request.urlopen(req, timeout=180) as resp:
                raw = resp.read().decode("utf-8", errors="ignore")
                parsed = json.loads(raw)
                parts = parsed.get("candidates", [{}])[0].get("content", {}).get("parts", [])
                if not parts:
                    raise RuntimeError("Empty Gemini response")
                return "".join(part.get("text", "") for part in parts).strip()
        except (urllib.error.HTTPError, urllib.error.URLError, RuntimeError) as exc:
            if attempt == max_retries - 1:
                raise
            if isinstance(exc, urllib.error.HTTPError) and exc.code in {429, 500, 503}:
                retry_after = _parse_retry_after(getattr(exc, "headers", None))
                backoff = max(retry_after, sleep_sec * (2**attempt))
            else:
                backoff = sleep_sec * (2**attempt)
            backoff = min(retry_max_sleep, backoff)
            jitter = random.uniform(0.0, min(5.0, backoff * 0.25))
            time.sleep(backoff + jitter)

    raise RuntimeError("Gemini API call failed")


def call_with_key_fallback(
    api_keys: list[tuple[str, str]],
    start_index: int,
    model: str,
    record: ImageRecord,
    context: str,
    max_retries: int,
    sleep_sec: float,
    retry_max_sleep: float,
) -> tuple[str, int, str]:
    errors = []
    for offset in range(len(api_keys)):
        index = (start_index + offset) % len(api_keys)
        key_name, api_key = api_keys[index]
        try:
            text = call_gemini(api_key, model, record, context, max_retries, sleep_sec, retry_max_sleep)
            return text, index, key_name
        except urllib.error.HTTPError as exc:
            errors.append(f"{key_name}: HTTP {exc.code} {exc.reason}")
            if exc.code not in {400, 401, 403, 429, 500, 503}:
                raise
        except (urllib.error.URLError, RuntimeError) as exc:
            errors.append(f"{key_name}: {exc}")
    raise RuntimeError("All Gemini keys failed. " + " | ".join(errors[-len(api_keys) :]))


def load_done_ids(output_jsonl: Path) -> set[str]:
    done = set()
    if not output_jsonl.exists():
        return done
    with output_jsonl.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            image_id = row.get("image_id")
            if image_id:
                done.add(str(image_id))
    return done


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Annotate textbook images with Gemini using matching Markdown book context.")
    parser.add_argument("--image-root", default="IMAGES/textbooks")
    parser.add_argument("--book-dirs", nargs="+", default=["core_cpt", "rl"], help="Markdown dirs searched in order.")
    parser.add_argument("--output-jsonl", default="vlm_prepared/textbook_image_annotations.jsonl")
    parser.add_argument("--progress-json", default="vlm_prepared/textbook_image_annotation_progress.json")
    parser.add_argument("--model", default="gemini-3.1-flash-lite-preview")
    parser.add_argument("--context-chars", type=int, default=5000)
    parser.add_argument("--max-context-chars", type=int, default=9000)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--sleep-sec", type=float, default=0.5)
    parser.add_argument("--retry-max-sleep", type=float, default=60.0)
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root_dir = Path(__file__).resolve().parents[1]
    api_keys = load_api_keys(root_dir / ".env")
    if not api_keys:
        raise RuntimeError("Missing GEMINI_API_KEY or GOOGLE_API_KEY in .env or environment.")

    books = load_books([Path(path) for path in args.book_dirs])
    records = collect_image_records(Path(args.image_root), books)
    records = records[args.start_index :]
    if args.max_images > 0:
        records = records[: args.max_images]
    if not records:
        raise RuntimeError("No image/book pairs found. Check --image-root and --book-dirs.")

    output_jsonl = Path(args.output_jsonl)
    done_ids = load_done_ids(output_jsonl) if args.resume else set()
    progress_path = Path(args.progress_json)

    key_index = 0
    started = time.time()
    total = len(records)
    completed = 0
    skipped = 0
    errors = 0

    for ordinal, record in enumerate(records, start=1):
        if record.image_id in done_ids:
            skipped += 1
            continue

        progress = {
            "status": "running",
            "current": ordinal,
            "total": total,
            "completed": completed,
            "skipped": skipped,
            "errors": errors,
            "current_image_id": record.image_id,
            "elapsed_minutes": round((time.time() - started) / 60, 2),
            "last_update": datetime.now().isoformat(timespec="seconds"),
        }
        write_json(progress_path, progress)

        try:
            markdown = Path(record.book_md_path).read_text(encoding="utf-8", errors="ignore")
            context, context_meta = context_for_image(markdown, record.image_index, record.image_count, args.context_chars)
            context = compact_text(context, args.max_context_chars)
            response_text, key_index, key_name = call_with_key_fallback(
                api_keys=api_keys,
                start_index=key_index,
                model=args.model,
                record=record,
                context=context,
                max_retries=args.max_retries,
                sleep_sec=args.sleep_sec,
                retry_max_sleep=args.retry_max_sleep,
            )
            annotation, raw_text = parse_json_response(response_text)
            if annotation is None:
                raise RuntimeError("Gemini response was not valid JSON")
            quality_flags = annotation_quality_flags(annotation, context_meta)

            row = {
                "image_id": record.image_id,
                "image_path": record.image_path,
                "book_stem": record.book_stem,
                "book_md_path": record.book_md_path,
                "image_index": record.image_index,
                "image_count": record.image_count,
                "annotation_model": args.model,
                "api_key_name": key_name,
                "context_meta": context_meta,
                "context_excerpt": context,
                "annotation": annotation,
                "quality_flags": quality_flags,
            }
            append_jsonl(output_jsonl, row)
            completed += 1
            print(f"[{ordinal}/{total}] annotated {record.image_id}")
            time.sleep(args.sleep_sec)
        except Exception as exc:
            errors += 1
            error_row = {
                "image_id": record.image_id,
                "image_path": record.image_path,
                "book_stem": record.book_stem,
                "book_md_path": record.book_md_path,
                "error": str(exc),
                "timestamp": datetime.now().isoformat(timespec="seconds"),
            }
            append_jsonl(output_jsonl.with_suffix(".errors.jsonl"), error_row)
            print(f"[{ordinal}/{total}] ERROR {record.image_id}: {exc}")
            if not args.continue_on_error:
                raise

    write_json(
        progress_path,
        {
            "status": "completed",
            "total": total,
            "completed": completed,
            "skipped": skipped,
            "errors": errors,
            "output_jsonl": str(output_jsonl),
            "elapsed_minutes": round((time.time() - started) / 60, 2),
            "last_update": datetime.now().isoformat(timespec="seconds"),
        },
    )
    print(json.dumps({"output_jsonl": str(output_jsonl), "completed": completed, "skipped": skipped, "errors": errors}))


if __name__ == "__main__":
    main()
