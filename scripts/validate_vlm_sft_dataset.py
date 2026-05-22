from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate compact VLM SFT JSONL rows.")
    parser.add_argument("--input-jsonl", default="vlm_prepared/textbook_vlm_sft.jsonl")
    parser.add_argument("--report-json", default="vlm_prepared/textbook_vlm_sft_report.json")
    args = parser.parse_args()

    path = Path(args.input_jsonl)
    if not path.exists():
        raise FileNotFoundError(path)

    total = 0
    invalid_json = 0
    invalid_schema = 0
    embedded_images = 0
    use_for_training = 0
    modes = Counter()
    alignments = Counter()
    image_types = Counter()
    scores = []
    examples = {"invalid_schema": [], "reject": [], "image_only": [], "book_grounded": []}

    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            total += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                invalid_json += 1
                continue
            messages = row.get("messages")
            metadata = row.get("metadata", {})
            if not isinstance(messages, list) or len(messages) != 2 or not isinstance(metadata, dict):
                invalid_schema += 1
                if len(examples["invalid_schema"]) < 5:
                    examples["invalid_schema"].append({"line": line_no})
                continue
            if str(row.get("image", "")).startswith("data:image/"):
                embedded_images += 1
            mode = str(metadata.get("training_mode", "unknown"))
            alignment = str(metadata.get("context_alignment", "unknown"))
            image_type = str(metadata.get("image_type", "unknown"))
            modes[mode] += 1
            alignments[alignment] += 1
            image_types[image_type] += 1
            if metadata.get("use_for_training"):
                use_for_training += 1
            try:
                scores.append(float(metadata.get("overall_score", 0.0)))
            except (TypeError, ValueError):
                pass
            bucket = mode if mode in examples else "reject"
            if len(examples[bucket]) < 3:
                examples[bucket].append(
                    {
                        "line": line_no,
                        "image_id": metadata.get("image_id"),
                        "prompt": messages[0].get("content", "")[:160],
                        "answer": messages[1].get("content", "")[:240],
                    }
                )

    report = {
        "input_jsonl": str(path),
        "total": total,
        "invalid_json": invalid_json,
        "invalid_schema": invalid_schema,
        "embedded_images": embedded_images,
        "use_for_training": use_for_training,
        "training_modes": dict(modes.most_common()),
        "context_alignments": dict(alignments.most_common()),
        "image_types": dict(image_types.most_common()),
        "avg_overall_score": round(sum(scores) / max(1, len(scores)), 4),
        "examples": examples,
    }
    report_path = Path(args.report_json)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
