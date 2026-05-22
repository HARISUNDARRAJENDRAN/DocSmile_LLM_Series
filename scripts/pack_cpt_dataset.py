from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import torch
from transformers import AutoTokenizer


MULTI_PART_SUBJECTS = (
    "Dental Anatomy _ Oral Histology",
    "Oral Medicine _ Radiology",
)


@dataclass(frozen=True)
class TextRecord:
    source: str
    group: str
    subject: str
    path: str
    text: str | None = None


def infer_subject(source: str) -> str:
    stem = Path(source).stem
    prefix = "__lib_book_"
    if not stem.startswith(prefix):
        return "Unknown"

    for subject in sorted(MULTI_PART_SUBJECTS, key=len, reverse=True):
        marker = f"{prefix}{subject}_ "
        if stem.startswith(marker):
            return " ".join(subject.split())

    remainder = stem[len(prefix) :]
    return " ".join(remainder.split("_ ", 1)[0].split())


def read_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def scan_txt_dir(path: Path, group: str) -> list[TextRecord]:
    records = []
    for item in sorted(path.glob("*.txt")):
        if item.is_file():
            records.append(
                TextRecord(
                    source=item.name,
                    group=group,
                    subject=infer_subject(item.name),
                    path=str(item.resolve()),
                )
            )
    return records


def collect_from_layout(input_dir: Path) -> list[TextRecord]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    direct_txt = scan_txt_dir(input_dir, "mixed")
    if direct_txt:
        return direct_txt

    core_dir = input_dir / "core_cpt_text"
    cleaned_core_dir = input_dir / "core_cpt_text_cleaned"
    selective_dir = input_dir / "selective_cpt_text"

    records: list[TextRecord] = []
    seen: set[str] = set()

    core_paths = {record.source: record for record in scan_txt_dir(core_dir, "core")}
    cleaned_paths = {record.source: record for record in scan_txt_dir(cleaned_core_dir, "core")}
    selective_paths = {record.source: record for record in scan_txt_dir(selective_dir, "selective")}

    for source, record in sorted(core_paths.items()):
        chosen = cleaned_paths.get(source, record)
        records.append(chosen)
        seen.add(source)

    for source, record in sorted(selective_paths.items()):
        if source not in seen:
            records.append(record)
            seen.add(source)

    if records:
        return records

    # Last-resort recursive scan for custom layouts. Avoid using this for the
    # standard cpt_prepared tree because it can duplicate raw and cleaned files.
    for item in sorted(input_dir.rglob("*.txt")):
        if item.is_file():
            records.append(
                TextRecord(
                    source=item.name,
                    group=item.parent.name,
                    subject=infer_subject(item.name),
                    path=str(item.resolve()),
                )
            )
    return records


def collect_jsonl(path: Path, text_field: str, source_field: str, group: str) -> list[TextRecord]:
    records = []
    for index, row in enumerate(read_jsonl(path)):
        text = row.get(text_field)
        if not isinstance(text, str) or not text.strip():
            continue
        source = str(row.get(source_field) or row.get("id") or f"{path.stem}_{index:08d}")
        records.append(
            TextRecord(
                source=source,
                group=str(row.get("group") or group),
                subject=str(row.get("subject") or infer_subject(source)),
                path=str(path.resolve()),
                text=text,
            )
        )
    return records


def collect_records(args: argparse.Namespace, validation: bool = False) -> list[TextRecord]:
    dirs = args.validation_input_dirs if validation else args.input_dirs
    jsonl_files = args.validation_jsonl_files if validation else args.jsonl_files

    records: list[TextRecord] = []
    for raw in dirs:
        records.extend(collect_from_layout(Path(raw)))
    for raw in jsonl_files:
        path = Path(raw)
        default_group = "validation" if validation else path.stem
        records.extend(collect_jsonl(path, args.text_field, args.source_field, default_group))
    return records


def split_by_source(
    records: list[TextRecord],
    validation_ratio: float,
    seed: int,
) -> tuple[list[TextRecord], list[TextRecord]]:
    if validation_ratio <= 0.0:
        return records, []

    grouped: dict[str, list[TextRecord]] = {}
    for record in records:
        grouped.setdefault(record.source, []).append(record)

    sources = sorted(grouped)
    if len(sources) < 2:
        return records, []

    rng = random.Random(seed)
    rng.shuffle(sources)
    val_count = max(1, int(round(len(sources) * validation_ratio)))
    val_count = min(val_count, len(sources) - 1)
    val_sources = set(sources[:val_count])

    train_records: list[TextRecord] = []
    val_records: list[TextRecord] = []
    for source in sorted(grouped):
        if source in val_sources:
            val_records.extend(grouped[source])
        else:
            train_records.extend(grouped[source])
    return train_records, val_records


def expand_core_records(records: list[TextRecord], core_weight: float, seed: int) -> list[TextRecord]:
    if core_weight <= 1.0:
        return records

    rng = random.Random(seed)
    expanded: list[TextRecord] = []
    integral = int(core_weight)
    fractional = core_weight - integral
    for record in records:
        copies = 1
        if record.group == "core":
            copies = max(1, integral)
            if fractional > 0.0 and rng.random() < fractional:
                copies += 1
        expanded.extend([record] * copies)
    return expanded


def record_text(record: TextRecord) -> str:
    if record.text is not None:
        return record.text
    return Path(record.path).read_text(encoding="utf-8", errors="ignore")


def pack_records(
    records: list[TextRecord],
    tokenizer,
    sequence_length: int,
    max_sequences: int,
    seed: int,
) -> tuple[torch.Tensor, dict]:
    if sequence_length <= 0:
        raise ValueError("--sequence-length must be positive.")
    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer must define eos_token_id for CPT packing.")

    ordered = records[:]
    random.Random(seed).shuffle(ordered)

    buffer: list[int] = []
    blocks: list[list[int]] = []
    total_tokens = 0
    skipped_empty = 0

    for record in ordered:
        text = record_text(record).strip()
        if not text:
            skipped_empty += 1
            continue
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        if not token_ids:
            skipped_empty += 1
            continue
        token_ids.append(tokenizer.eos_token_id)
        total_tokens += len(token_ids)
        buffer.extend(token_ids)

        while len(buffer) >= sequence_length:
            blocks.append(buffer[:sequence_length])
            buffer = buffer[sequence_length:]
            if max_sequences > 0 and len(blocks) >= max_sequences:
                tensor = torch.tensor(blocks, dtype=torch.long)
                return tensor, {
                    "documents_or_rows": len(records),
                    "used_records": len(ordered) - skipped_empty,
                    "skipped_empty": skipped_empty,
                    "tokens_before_packing": total_tokens,
                    "packed_sequences": int(tensor.shape[0]),
                    "sequence_length": sequence_length,
                    "dropped_tail_tokens": len(buffer),
                    "limited_by_max_sequences": True,
                }

    if not blocks:
        raise RuntimeError("No packed sequences were produced. Lower sequence length or check input text.")

    tensor = torch.tensor(blocks, dtype=torch.long)
    return tensor, {
        "documents_or_rows": len(records),
        "used_records": len(records) - skipped_empty,
        "skipped_empty": skipped_empty,
        "tokens_before_packing": total_tokens,
        "packed_sequences": int(tensor.shape[0]),
        "sequence_length": sequence_length,
        "dropped_tail_tokens": len(buffer),
        "limited_by_max_sequences": False,
    }


def write_dataset(path: Path, input_ids: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"input_ids": input_ids}, path)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pack DocSmile CPT text into fixed-length causal-LM tensors.")
    parser.add_argument("--model-name-or-path", required=True, help="Tokenizer/model id used for packing.")
    parser.add_argument("--output-dir", required=True, help="Directory for train.pt, validation.pt, and manifest.json.")
    parser.add_argument("--input-dirs", nargs="*", default=["cpt_prepared"], help="Directories with .txt files or cpt_prepared layout.")
    parser.add_argument("--jsonl-files", nargs="*", default=[], help="JSONL files containing a text field.")
    parser.add_argument("--validation-input-dirs", nargs="*", default=[], help="Optional explicit validation text dirs.")
    parser.add_argument("--validation-jsonl-files", nargs="*", default=[], help="Optional explicit validation JSONL files.")
    parser.add_argument("--text-field", default="text", help="Text field for JSONL rows.")
    parser.add_argument("--source-field", default="source", help="Source/document field for JSONL rows.")
    parser.add_argument("--sequence-length", type=int, default=2048)
    parser.add_argument("--validation-ratio", type=float, default=0.03)
    parser.add_argument("--core-weight", type=float, default=1.25, help="Oversampling weight for records tagged group=core.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-sequences", type=int, default=0, help="Limit for smoke tests; 0 means no limit.")
    parser.add_argument("--max-validation-sequences", type=int, default=512, help="Validation cap; 0 means no limit.")
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    train_records = collect_records(args, validation=False)
    explicit_val_records = collect_records(args, validation=True)
    if not train_records:
        raise RuntimeError("No CPT training records found.")

    if explicit_val_records:
        val_records = explicit_val_records
    else:
        train_records, val_records = split_by_source(train_records, args.validation_ratio, args.seed)

    train_records = expand_core_records(train_records, args.core_weight, args.seed)

    train_tensor, train_stats = pack_records(
        train_records,
        tokenizer,
        args.sequence_length,
        args.max_train_sequences,
        args.seed,
    )
    write_dataset(output_dir / "train.pt", train_tensor)

    validation_stats = {}
    if val_records:
        val_tensor, validation_stats = pack_records(
            val_records,
            tokenizer,
            args.sequence_length,
            args.max_validation_sequences,
            args.seed + 17,
        )
        write_dataset(output_dir / "validation.pt", val_tensor)

    sources = sorted({record.source for record in train_records})
    val_sources = sorted({record.source for record in val_records})
    manifest = {
        "model_name_or_path": args.model_name_or_path,
        "sequence_length": args.sequence_length,
        "seed": args.seed,
        "core_weight": args.core_weight,
        "train_records": len(train_records),
        "validation_records": len(val_records),
        "train_sources": len(sources),
        "validation_sources": len(val_sources),
        "train_stats": train_stats,
        "validation_stats": validation_stats,
        "input_dirs": [str(Path(path).resolve()) for path in args.input_dirs],
        "jsonl_files": [str(Path(path).resolve()) for path in args.jsonl_files],
        "validation_input_dirs": [str(Path(path).resolve()) for path in args.validation_input_dirs],
        "validation_jsonl_files": [str(Path(path).resolve()) for path in args.validation_jsonl_files],
        "sample_train_manifest": [asdict(record) for record in train_records[:50]],
        "sample_validation_manifest": [asdict(record) for record in val_records[:50]],
    }
    write_json(output_dir / "manifest.json", manifest)

    print(json.dumps({"output_dir": str(output_dir), "train": train_stats, "validation": validation_stats}, indent=2))


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
