from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield line_no, json.loads(line), None
            except json.JSONDecodeError as exc:
                yield line_no, None, str(exc)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Gemini textbook-image annotation JSONL quality.")
    parser.add_argument("--input-jsonl", default="vlm_prepared/textbook_image_annotations.jsonl")
    parser.add_argument("--errors-jsonl", default="")
    parser.add_argument("--report-json", default="vlm_prepared/textbook_image_annotation_report.json")
    parser.add_argument("--min-confidence", type=float, default=0.6)
    args = parser.parse_args()

    input_path = Path(args.input_jsonl)
    if not input_path.exists():
        raise FileNotFoundError(f"Annotation JSONL not found: {input_path}")

    total = 0
    invalid_json = 0
    usable_book_grounded = 0
    usable_image_only = 0
    reject = 0
    missing_required = 0
    low_confidence = 0
    context_relevance = Counter()
    image_types = Counter()
    quality_flags = Counter()
    books = Counter()
    examples = {
        "invalid_json": [],
        "missing_required": [],
        "low_confidence": [],
        "low_context": [],
    }

    required_annotation_keys = {
        "image_type",
        "subject_area",
        "title",
        "caption",
        "visible_findings",
        "dental_entities",
        "qa_pairs",
        "context_relevance",
        "confidence",
    }

    for line_no, row, error in iter_jsonl(input_path):
        total += 1
        if error:
            invalid_json += 1
            if len(examples["invalid_json"]) < 5:
                examples["invalid_json"].append({"line": line_no, "error": error})
            continue

        annotation = row.get("annotation") if isinstance(row, dict) else None
        if not isinstance(annotation, dict):
            missing_required += 1
            if len(examples["missing_required"]) < 5:
                examples["missing_required"].append({"line": line_no, "image_id": row.get("image_id")})
            continue

        missing = sorted(required_annotation_keys - set(annotation))
        if missing:
            missing_required += 1
            if len(examples["missing_required"]) < 5:
                examples["missing_required"].append(
                    {"line": line_no, "image_id": row.get("image_id"), "missing": missing}
                )

        relevance = str(annotation.get("context_relevance", "")).strip().lower() or "unknown"
        context_relevance[relevance] += 1
        image_types[str(annotation.get("image_type", "unknown")).strip().lower() or "unknown"] += 1
        books[str(row.get("book_stem", "unknown"))] += 1
        for flag in row.get("quality_flags", []) or []:
            quality_flags[str(flag)] += 1

        try:
            confidence = float(annotation.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        if confidence < args.min_confidence:
            low_confidence += 1
            if len(examples["low_confidence"]) < 5:
                examples["low_confidence"].append(
                    {"line": line_no, "image_id": row.get("image_id"), "confidence": confidence}
                )

        flags = set(row.get("quality_flags", []) or [])
        if relevance == "low" or "low_context_relevance" in flags:
            if len(examples["low_context"]) < 5:
                examples["low_context"].append({"line": line_no, "image_id": row.get("image_id")})

        if missing:
            reject += 1
        elif confidence < args.min_confidence:
            reject += 1
        elif relevance in {"high", "medium"} and "anchor_image_count_mismatch" not in flags:
            usable_book_grounded += 1
        elif relevance in {"low", "none"}:
            usable_image_only += 1
        else:
            reject += 1

    errors_path = Path(args.errors_jsonl) if args.errors_jsonl else input_path.with_suffix(".errors.jsonl")
    api_errors = 0
    if errors_path.exists():
        api_errors = sum(1 for _ in errors_path.open("r", encoding="utf-8", errors="ignore") if _.strip())

    report = {
        "input_jsonl": str(input_path),
        "total_rows": total,
        "invalid_json": invalid_json,
        "api_error_rows": api_errors,
        "missing_required": missing_required,
        "low_confidence": low_confidence,
        "usable_book_grounded": usable_book_grounded,
        "usable_image_only": usable_image_only,
        "reject": reject,
        "context_relevance": dict(context_relevance.most_common()),
        "image_types": dict(image_types.most_common()),
        "quality_flags": dict(quality_flags.most_common()),
        "top_books": dict(books.most_common(20)),
        "examples": examples,
    }

    report_path = Path(args.report_json)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
