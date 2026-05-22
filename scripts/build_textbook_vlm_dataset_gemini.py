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
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


API_KEY_NAME_RE = re.compile(r"^(?:GEMINI_API_KEY|GOOGLE_API_KEY)(?:\d+)?$")
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
IMAGE_ANCHOR_RE = re.compile(r"(<!--\s*image\s*-->|<image>|!\[[^\]]*]\([^)]+\)|<img\b[^>]*>)", re.IGNORECASE)
FIGURE_LINE_RE = re.compile(r"\b(fig(?:ure)?\.?|table)\s*[-.:]?\s*[0-9ivxlcdm]+", re.IGNORECASE)


@dataclass(frozen=True)
class ImageRecord:
    image_id: str
    image_path: Path
    book_stem: str
    book_md_path: Path
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
        for path in sorted(book_dir.glob("*.md"), key=lambda p: natural_key(p.name)):
            books.setdefault(path.stem, path)
    return books


def collect_records(image_root: Path, books: dict[str, Path]) -> list[ImageRecord]:
    records: list[ImageRecord] = []
    for book_dir in sorted([p for p in image_root.iterdir() if p.is_dir()], key=lambda p: natural_key(p.name)):
        book_path = books.get(book_dir.name)
        if not book_path:
            continue
        images = sorted(
            [p for p in book_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS],
            key=lambda p: natural_key(p.name),
        )
        for index, image_path in enumerate(images):
            records.append(
                ImageRecord(
                    image_id=f"{book_dir.name}/{image_path.name}",
                    image_path=image_path.resolve(),
                    book_stem=book_dir.name,
                    book_md_path=book_path.resolve(),
                    image_index=index,
                    image_count=len(images),
                )
            )
    return records


def image_to_data_uri(path: Path) -> tuple[str, str]:
    mime_type = mimetypes.guess_type(path.name)[0] or "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}", mime_type


def image_part(path: Path) -> dict:
    data_uri, mime_type = image_to_data_uri(path)
    encoded = data_uri.split(",", 1)[1]
    return {"inline_data": {"mime_type": mime_type, "data": encoded}}


def find_anchor_positions(markdown: str) -> list[tuple[str, int]]:
    anchors = [(match.group(0), match.start()) for match in IMAGE_ANCHOR_RE.finditer(markdown)]
    if anchors:
        return anchors
    out = []
    offset = 0
    for line in markdown.splitlines(keepends=True):
        if FIGURE_LINE_RE.search(line):
            out.append((line.strip(), offset))
        offset += len(line)
    return out


def paragraph_window(markdown: str, center: int, chars: int) -> str:
    start = max(0, center - chars)
    end = min(len(markdown), center + chars)
    text = markdown[start:end].strip()
    if start > 0:
        first_break = text.find("\n\n")
        if 0 <= first_break < 800:
            text = text[first_break + 2 :]
    if end < len(markdown):
        last_break = text.rfind("\n\n")
        if last_break > len(text) - 800:
            text = text[:last_break]
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def candidate_contexts(markdown: str, record: ImageRecord, context_chars: int, max_candidates: int) -> list[dict]:
    anchors = find_anchor_positions(markdown)
    candidates: list[dict] = []
    used_positions: set[int] = set()

    if anchors:
        direct_idx = min(record.image_index, len(anchors) - 1)
        scaled_idx = round(record.image_index * (len(anchors) - 1) / max(1, record.image_count - 1))
        seed_indices = [direct_idx, scaled_idx, scaled_idx - 1, scaled_idx + 1, direct_idx - 1, direct_idx + 1]
        for idx in seed_indices:
            if idx < 0 or idx >= len(anchors):
                continue
            anchor_text, pos = anchors[idx]
            if pos in used_positions:
                continue
            used_positions.add(pos)
            candidates.append(
                {
                    "candidate_id": f"anchor_{idx}",
                    "strategy": "anchor",
                    "anchor_index": idx,
                    "anchor_text": anchor_text[:160],
                    "context": paragraph_window(markdown, pos, context_chars),
                }
            )
            if len(candidates) >= max_candidates:
                break

    if not candidates:
        ratio = (record.image_index + 0.5) / max(1, record.image_count)
        pos = int(len(markdown) * ratio)
        candidates.append(
            {
                "candidate_id": "proportional_0",
                "strategy": "proportional",
                "anchor_index": -1,
                "anchor_text": "",
                "context": paragraph_window(markdown, pos, context_chars),
            }
        )

    return candidates


def build_prompt(record: ImageRecord, candidates: list[dict]) -> str:
    compact_candidates = [
        {
            "candidate_id": item["candidate_id"],
            "strategy": item["strategy"],
            "anchor_index": item["anchor_index"],
            "anchor_text": item["anchor_text"],
            "context": item["context"][:3000],
        }
        for item in candidates
    ]
    return (
        "You are building a high-quality dental/medical vision-language dataset.\n"
        "Follow this exact process internally:\n"
        "1. Inspect the image first and list only visible facts.\n"
        "2. Compare the image with each candidate textbook context.\n"
        "3. Pick the best matching context only if it genuinely describes or supports the image.\n"
        "4. If no candidate context matches, mark the row as image_only or reject.\n"
        "5. Create a clean one-turn VLM training example. Do not include long context in the answer.\n\n"
        "Return ONLY valid JSON with this schema:\n"
        "{\n"
        '  "visual_observation": {\n'
        '    "image_type": "radiograph|clinical_photo|histology|diagram|chart_table|instrument_material|anatomy_illustration|chemical_structure|other",\n'
        '    "visible_summary": "one sentence",\n'
        '    "visible_findings": ["visible facts"],\n'
        '    "visible_text": ["text visible in image"],\n'
        '    "likely_subject": "short subject",\n'
        '    "confidence": 0.0\n'
        "  },\n"
        '  "context_alignment": {\n'
        '    "best_candidate_id": "candidate id or none",\n'
        '    "alignment": "high|medium|low|none",\n'
        '    "matches_context": true,\n'
        '    "supporting_phrases": ["short phrases from context"],\n'
        '    "conflicting_phrases": ["short phrases if any"],\n'
        '    "reason": "brief reason"\n'
        "  },\n"
        '  "final_dataset_row": {\n'
        '    "use_for_training": true,\n'
        '    "training_mode": "book_grounded|image_only|reject",\n'
        '    "user_prompt": "short prompt to pair with <image>",\n'
        '    "assistant_response": "clean, concise, grounded answer",\n'
        '    "qa_pairs": [{"question": "optional image-grounded question", "answer": "answer"}]\n'
        "  },\n"
        '  "quality": {\n'
        '    "overall_score": 0.0,\n'
        '    "major_issues": ["issues"],\n'
        '    "safety_notes": ["notes"]\n'
        "  }\n"
        "}\n\n"
        "Hard rules:\n"
        "- Do not copy unrelated textbook context into the final answer.\n"
        "- If context is unrelated but image is clear, use training_mode image_only.\n"
        "- If the image is unclear, too cropped, or not educational, use training_mode reject.\n"
        "- Do not give clinical diagnosis or treatment instructions from an image alone.\n\n"
        f"BOOK: {record.book_stem}\n"
        f"IMAGE FILE: {record.image_path.name}\n"
        f"IMAGE POSITION: {record.image_index + 1}/{record.image_count}\n\n"
        "CANDIDATE_CONTEXTS:\n"
        f"{json.dumps(compact_candidates, ensure_ascii=False)}"
    )


def parse_json(text: str) -> dict:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end > start:
        cleaned = cleaned[start : end + 1]
    parsed = json.loads(cleaned)
    if not isinstance(parsed, dict):
        raise ValueError("Gemini response is not a JSON object")
    return parsed


def call_gemini(
    api_key: str,
    model: str,
    record: ImageRecord,
    candidates: list[dict],
    max_retries: int,
    sleep_sec: float,
) -> dict:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [image_part(record.image_path), {"text": build_prompt(record, candidates)}],
            }
        ],
        "generationConfig": {
            "temperature": 0.1,
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
                return parse_json("".join(part.get("text", "") for part in parts))
        except (urllib.error.HTTPError, urllib.error.URLError, RuntimeError, json.JSONDecodeError, ValueError):
            if attempt == max_retries - 1:
                raise
            time.sleep(sleep_sec * (2**attempt) + random.uniform(0, sleep_sec))
    raise RuntimeError("Gemini call failed")


def load_done_ids(path: Path) -> set[str]:
    done = set()
    if not path.exists():
        return done
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            image_id = row.get("metadata", {}).get("image_id") or row.get("image_id")
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


def make_sft_row(record: ImageRecord, result: dict, candidates: list[dict], embed_image: bool) -> dict:
    final = result.get("final_dataset_row", {}) if isinstance(result.get("final_dataset_row"), dict) else {}
    alignment = result.get("context_alignment", {}) if isinstance(result.get("context_alignment"), dict) else {}
    quality = result.get("quality", {}) if isinstance(result.get("quality"), dict) else {}
    observation = result.get("visual_observation", {}) if isinstance(result.get("visual_observation"), dict) else {}
    prompt = final.get("user_prompt") or "Describe this textbook image accurately."
    response = final.get("assistant_response") or observation.get("visible_summary") or ""
    data_uri, mime_type = image_to_data_uri(record.image_path)
    best_context = ""
    best_id = alignment.get("best_candidate_id")
    for candidate in candidates:
        if candidate["candidate_id"] == best_id:
            best_context = candidate["context"][:1000]
            break

    image_payload = data_uri if embed_image else str(record.image_path)
    return {
        "image": image_payload,
        "image_mime_type": mime_type,
        "messages": [
            {"role": "user", "content": f"<image>\n{prompt}"},
            {"role": "assistant", "content": response},
        ],
        "metadata": {
            "image_id": record.image_id,
            "book": record.book_stem,
            "source_md": str(record.book_md_path),
            "training_mode": final.get("training_mode", "reject"),
            "use_for_training": bool(final.get("use_for_training", False)),
            "context_alignment": alignment.get("alignment", "none"),
            "best_candidate_id": best_id,
            "context_snippet": best_context,
            "image_type": observation.get("image_type", "other"),
            "overall_score": quality.get("overall_score", 0.0),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build clean VLM SFT rows from textbook images using Gemini visual/context judging.")
    parser.add_argument("--image-root", default="IMAGES/textbooks")
    parser.add_argument("--book-dirs", nargs="+", default=["core_cpt", "rl"])
    parser.add_argument("--audit-jsonl", default="vlm_prepared/textbook_vlm_audit.jsonl")
    parser.add_argument("--sft-jsonl", default="vlm_prepared/textbook_vlm_sft.jsonl")
    parser.add_argument("--errors-jsonl", default="vlm_prepared/textbook_vlm_errors.jsonl")
    parser.add_argument("--progress-json", default="vlm_prepared/textbook_vlm_progress.json")
    parser.add_argument("--model", default="gemini-3.1-flash-lite-preview")
    parser.add_argument("--context-chars", type=int, default=1800)
    parser.add_argument("--max-candidates", type=int, default=5)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--sleep-sec", type=float, default=0.5)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--embed-images", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root_dir = Path(__file__).resolve().parents[1]
    api_keys = load_api_keys(root_dir / ".env")
    if not api_keys:
        raise RuntimeError("Missing GEMINI_API_KEY or GOOGLE_API_KEY.")

    books = load_books([Path(path) for path in args.book_dirs])
    records = collect_records(Path(args.image_root), books)
    records = records[args.start_index :]
    if args.max_images > 0:
        records = records[: args.max_images]
    if not records:
        raise RuntimeError("No image records matched to markdown books.")

    audit_path = Path(args.audit_jsonl)
    sft_path = Path(args.sft_jsonl)
    done_ids = load_done_ids(audit_path) if args.resume else set()

    completed = 0
    skipped = 0
    errors = 0
    key_index = 0
    started = time.time()

    for ordinal, record in enumerate(records, start=1):
        if record.image_id in done_ids:
            skipped += 1
            continue
        write_json(
            Path(args.progress_json),
            {
                "status": "running",
                "current": ordinal,
                "total": len(records),
                "completed": completed,
                "skipped": skipped,
                "errors": errors,
                "current_image_id": record.image_id,
                "elapsed_minutes": round((time.time() - started) / 60, 2),
                "last_update": datetime.now().isoformat(timespec="seconds"),
            },
        )
        try:
            markdown = record.book_md_path.read_text(encoding="utf-8", errors="ignore")
            candidates = candidate_contexts(markdown, record, args.context_chars, args.max_candidates)
            key_name, api_key = api_keys[key_index % len(api_keys)]
            key_index += 1
            result = call_gemini(api_key, args.model, record, candidates, args.max_retries, args.sleep_sec)
            audit_row = {
                "image_id": record.image_id,
                "image_path": str(record.image_path),
                "book_stem": record.book_stem,
                "book_md_path": str(record.book_md_path),
                "image_index": record.image_index,
                "image_count": record.image_count,
                "model": args.model,
                "api_key_name": key_name,
                "candidate_contexts": candidates,
                "result": result,
            }
            sft_row = make_sft_row(record, result, candidates, args.embed_images)
            append_jsonl(audit_path, audit_row)
            append_jsonl(sft_path, sft_row)
            completed += 1
            mode = sft_row["metadata"]["training_mode"]
            align = sft_row["metadata"]["context_alignment"]
            print(f"[{ordinal}/{len(records)}] {record.image_id} mode={mode} alignment={align}")
            time.sleep(args.sleep_sec)
        except Exception as exc:
            errors += 1
            append_jsonl(
                Path(args.errors_jsonl),
                {
                    "image_id": record.image_id,
                    "image_path": str(record.image_path),
                    "book_stem": record.book_stem,
                    "error": str(exc),
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                },
            )
            print(f"[{ordinal}/{len(records)}] ERROR {record.image_id}: {exc}")
            if not args.continue_on_error:
                raise

    write_json(
        Path(args.progress_json),
        {
            "status": "completed",
            "total": len(records),
            "completed": completed,
            "skipped": skipped,
            "errors": errors,
            "audit_jsonl": str(audit_path),
            "sft_jsonl": str(sft_path),
            "elapsed_minutes": round((time.time() - started) / 60, 2),
            "last_update": datetime.now().isoformat(timespec="seconds"),
        },
    )
    print(json.dumps({"completed": completed, "skipped": skipped, "errors": errors, "sft_jsonl": str(sft_path)}))


if __name__ == "__main__":
    main()
