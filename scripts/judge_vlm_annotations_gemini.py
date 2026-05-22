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
from pathlib import Path


API_KEY_NAME_RE = re.compile(r"^(?:GEMINI_API_KEY|GOOGLE_API_KEY)(?:\d+)?$")


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
    out = []
    for name, value in candidates:
        if value in seen:
            continue
        seen.add(value)
        out.append((name, value))
    return out


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield line_no, json.loads(line)
            except json.JSONDecodeError:
                continue


def image_part(path: Path) -> dict:
    mime_type = mimetypes.guess_type(path.name)[0] or "image/png"
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return {"inline_data": {"mime_type": mime_type, "data": data}}


def build_prompt(row: dict) -> str:
    compact = {
        "image_id": row.get("image_id"),
        "book_stem": row.get("book_stem"),
        "context_meta": row.get("context_meta"),
        "context_excerpt": row.get("context_excerpt", "")[:5000],
        "annotation": row.get("annotation"),
        "quality_flags": row.get("quality_flags", []),
    }
    return (
        "You are judging a dental VLM dataset annotation. Inspect the image and compare it to the annotation and book context.\n"
        "Return ONLY valid JSON with this schema:\n"
        "{\n"
        '  "visual_correctness": 0.0,\n'
        '  "context_alignment": 0.0,\n'
        '  "caption_quality": 0.0,\n'
        '  "qa_quality": 0.0,\n'
        '  "safety_quality": 0.0,\n'
        '  "overall_score": 0.0,\n'
        '  "recommended_action": "keep_book_grounded|keep_image_only|revise|reject",\n'
        '  "major_issues": ["short issue strings"],\n'
        '  "reason": "brief explanation"\n'
        "}\n\n"
        "Scoring rules:\n"
        "- visual_correctness: does the annotation describe what is actually visible?\n"
        "- context_alignment: does the book context genuinely support the annotation?\n"
        "- caption_quality: is the caption useful, specific, and not hallucinated?\n"
        "- qa_quality: are QA pairs image-grounded and educational?\n"
        "- safety_quality: no definitive diagnosis/treatment from image alone.\n"
        "- recommended_action should be reject for hallucinated or wrong annotations.\n"
        "- use keep_image_only if image annotation is good but context is irrelevant.\n\n"
        "ANNOTATION ROW:\n"
        f"{json.dumps(compact, ensure_ascii=False)}"
    )


def call_gemini(api_key: str, model: str, row: dict, max_retries: int, sleep_sec: float) -> str:
    image_path = Path(row["image_path"])
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    image_part(image_path),
                    {"text": build_prompt(row)},
                ],
            }
        ],
        "generationConfig": {
            "temperature": 0.05,
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
                parsed = json.loads(resp.read().decode("utf-8", errors="ignore"))
                parts = parsed.get("candidates", [{}])[0].get("content", {}).get("parts", [])
                if not parts:
                    raise RuntimeError("empty Gemini judge response")
                return "".join(part.get("text", "") for part in parts).strip()
        except (urllib.error.HTTPError, urllib.error.URLError, RuntimeError):
            if attempt == max_retries - 1:
                raise
            time.sleep(sleep_sec * (2**attempt) + random.uniform(0, sleep_sec))
    raise RuntimeError("Gemini judge call failed")


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
        raise ValueError("judge response is not an object")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description="Use Gemini to judge sampled textbook image annotations.")
    parser.add_argument("--input-jsonl", default="vlm_prepared/textbook_image_annotations.jsonl")
    parser.add_argument("--output-jsonl", default="vlm_prepared/textbook_image_annotation_judgments.jsonl")
    parser.add_argument("--report-json", default="vlm_prepared/textbook_image_annotation_judge_report.json")
    parser.add_argument("--model", default="gemini-3.1-flash-lite-preview")
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--min-line", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sleep-sec", type=float, default=0.5)
    parser.add_argument("--max-retries", type=int, default=3)
    args = parser.parse_args()

    input_path = Path(args.input_jsonl)
    if not input_path.exists():
        raise FileNotFoundError(f"Missing input JSONL: {input_path}")
    keys = load_api_keys(Path(__file__).resolve().parents[1] / ".env")
    if not keys:
        raise RuntimeError("Missing GEMINI_API_KEY or GOOGLE_API_KEY.")

    rows = [(line_no, row) for line_no, row in iter_jsonl(input_path) if line_no >= args.min_line]
    random.Random(args.seed).shuffle(rows)
    rows = rows[: args.sample_size]

    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scores = []
    actions = {}
    key_index = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for idx, (line_no, row) in enumerate(rows, start=1):
            key_name, api_key = keys[key_index % len(keys)]
            key_index += 1
            raw = call_gemini(api_key, args.model, row, args.max_retries, args.sleep_sec)
            judgment = parse_json(raw)
            score = float(judgment.get("overall_score", 0.0))
            action = str(judgment.get("recommended_action", "unknown"))
            scores.append(score)
            actions[action] = actions.get(action, 0) + 1
            out = {
                "line_no": line_no,
                "image_id": row.get("image_id"),
                "judge_model": args.model,
                "api_key_name": key_name,
                "judgment": judgment,
            }
            handle.write(json.dumps(out, ensure_ascii=False) + "\n")
            print(f"[{idx}/{len(rows)}] {row.get('image_id')} score={score:.2f} action={action}")
            time.sleep(args.sleep_sec)

    report = {
        "input_jsonl": str(input_path),
        "output_jsonl": str(output_path),
        "sample_size": len(rows),
        "avg_overall_score": round(sum(scores) / max(1, len(scores)), 4),
        "min_overall_score": min(scores) if scores else None,
        "max_overall_score": max(scores) if scores else None,
        "recommended_actions": actions,
    }
    Path(args.report_json).write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
