from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

import pandas as pd


DR_M_RE = re.compile(r"(?P<question>.*?)(?:^|\s)Dr M:\s*(?P<answer>.*)", re.DOTALL)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def clean_text(text: object) -> str:
    value = str(text or "")
    value = re.sub(r"\s+", " ", value).strip()
    value = value.replace("Click to expand...", "").strip()
    return value


def build_medmcqa(root: Path, limit: int, seed: int) -> list[dict]:
    candidates = [
        root / "medmcqa-Dental-responses" / "qwen-3b" / "test-00000-of-00001.parquet",
        root / "medmcqa-Dental-responses" / "qwen-1.5b" / "test-00000-of-00001.parquet",
        root / "medmcqa-Dental-responses" / "qwen-0.5b" / "test-00000-of-00001.parquet",
    ]
    source = next((path for path in candidates if path.exists()), None)
    if source is None:
        return []

    df = pd.read_parquet(source)
    rows = []
    for item in df.to_dict(orient="records"):
        options = [clean_text(item.get(key)) for key in ("opa", "opb", "opc", "opd")]
        if not clean_text(item.get("question")) or any(not option for option in options):
            continue
        answer_index = int(item.get("cop"))
        rows.append(
            {
                "id": f"medmcqa_dental_{item.get('id')}",
                "task": "mcq",
                "source": "medmcqa-Dental-responses",
                "question": clean_text(item.get("question")),
                "options": options,
                "answer_index": answer_index,
                "answer_label": "ABCD"[answer_index],
                "explanation": clean_text(item.get("exp")),
            }
        )

    random.Random(seed).shuffle(rows)
    return rows[:limit] if limit > 0 else rows


def build_oral_disease(root: Path, limit: int, seed: int) -> list[dict]:
    source = root / "Open-Domain-Oral-Disease-QA-Dataset" / "extracted_all.jsonl"
    if not source.exists():
        return []

    rows = []
    for line in source.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if str(item.get("validity", "")).lower() != "correct":
            continue
        question = clean_text(item.get("question"))
        answer = clean_text(item.get("Answer"))
        disease = clean_text(item.get("disease"))
        if len(question) < 20 or len(answer) < 80:
            continue
        rows.append(
            {
                "id": f"oral_disease_{len(rows) + 1}",
                "task": "open_qa",
                "source": "Open-Domain-Oral-Disease-QA-Dataset",
                "question": question,
                "reference_answer": answer,
                "topic": disease,
            }
        )

    random.Random(seed).shuffle(rows)
    return rows[:limit] if limit > 0 else rows


def build_forum_qa(root: Path, limit: int, seed: int) -> list[dict]:
    rows = []
    for source in sorted((root / "dental_QA").glob("*.json")):
        try:
            payload = json.loads(source.read_text(encoding="utf-8", errors="ignore"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, list):
            continue

        for item in payload:
            dialogue = str(item.get("dialogue", ""))
            match = DR_M_RE.match(dialogue)
            if not match:
                continue
            question = clean_text(match.group("question"))
            answer = clean_text(match.group("answer"))
            if len(question) < 40 or len(answer) < 40:
                continue
            rows.append(
                {
                    "id": f"dental_forum_{source.stem}_{item.get('id')}",
                    "task": "open_qa",
                    "source": f"dental_QA/{source.name}",
                    "question": question,
                    "reference_answer": answer,
                    "topic": clean_text(item.get("title")),
                }
            )

    random.Random(seed).shuffle(rows)
    return rows[:limit] if limit > 0 else rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Build frozen text-only dental eval JSONL files.")
    parser.add_argument("--root", default=".", help="Repo root")
    parser.add_argument("--output-dir", default="evals/text_only", help="Output directory")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mcq-limit", type=int, default=250)
    parser.add_argument("--oral-disease-limit", type=int, default=250)
    parser.add_argument("--forum-limit", type=int, default=500)
    args = parser.parse_args()

    root = Path(args.root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    datasets = {
        "medmcqa_dental_mcq.jsonl": build_medmcqa(root, args.mcq_limit, args.seed),
        "oral_disease_open_qa.jsonl": build_oral_disease(root, args.oral_disease_limit, args.seed),
        "dental_forum_open_qa.jsonl": build_forum_qa(root, args.forum_limit, args.seed),
    }

    manifest = {
        "seed": args.seed,
        "datasets": {},
        "notes": [
            "Frozen pre-CPT baseline eval pack. Do not train on these rows.",
            "MCQ can be scored automatically. Open QA should be judged manually or with a separate judge model.",
        ],
    }
    for name, rows in datasets.items():
        write_jsonl(out_dir / name, rows)
        manifest["datasets"][name] = len(rows)

    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
