from __future__ import annotations

import argparse
import json
import math
import os
import random
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, set_seed


MULTI_PART_SUBJECTS = (
    "Dental Anatomy _ Oral Histology",
    "Oral Medicine _ Radiology",
)


@dataclass
class DocRecord:
    name: str
    path: str
    group: str
    subject: str
    word_count: int


@dataclass
class BookUnit:
    schedule_index: int
    stage: str
    name: str
    path: str
    subject: str
    group: str
    word_count: int


class PackedTokenDataset(Dataset):
    def __init__(self, sequences: list[list[int]]) -> None:
        if not sequences:
            raise ValueError("PackedTokenDataset requires at least one sequence.")
        self.input_ids = torch.tensor(sequences, dtype=torch.long)

    def __len__(self) -> int:
        return self.input_ids.shape[0]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.input_ids[index]
        return {
            "input_ids": row,
            "labels": row.clone(),
        }


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    lowered = value.strip().lower()
    if lowered in {"1", "true", "t", "yes", "y"}:
        return True
    if lowered in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Cannot parse boolean value from: {value}")


def env_default(name: str, fallback: str) -> str:
    return os.environ.get(name, fallback)


def normalize_subject(text: str) -> str:
    return " ".join(text.replace("  ", " ").strip().split())


def infer_subject(filename: str) -> str:
    stem = Path(filename).stem
    prefix = "__lib_book_"
    if not stem.startswith(prefix):
        return "Unknown"

    for subject in sorted(MULTI_PART_SUBJECTS, key=len, reverse=True):
        marker = f"{prefix}{subject}_ "
        if stem.startswith(marker):
            return normalize_subject(subject)

    remainder = stem[len(prefix) :]
    subject = remainder.split("_ ", 1)[0]
    return normalize_subject(subject)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def scan_named_files(path: Path) -> dict[str, Path]:
    if not path.exists():
        return {}
    return {item.name: item for item in sorted(path.glob("*.txt")) if item.is_file()}


def build_records_from_layout(train_dir: Path) -> list[DocRecord]:
    direct_txt = scan_named_files(train_dir)
    if direct_txt:
        records = []
        for path in direct_txt.values():
            text = read_text(path)
            records.append(
                DocRecord(
                    name=path.name,
                    path=str(path.resolve()),
                    group="mixed",
                    subject=infer_subject(path.name),
                    word_count=len(text.split()),
                )
            )
        return records

    core_dir = train_dir / "core_cpt_text"
    selective_dir = train_dir / "selective_cpt_text"
    cleaned_dir = train_dir / "core_cpt_text_cleaned"

    core_map = scan_named_files(core_dir)
    selective_map = scan_named_files(selective_dir)
    cleaned_map = scan_named_files(cleaned_dir)

    if not core_map and not selective_map and not cleaned_map:
        raise FileNotFoundError(
            "No .txt files found. Expected either direct .txt files under the train dir "
            "or a SageMaker channel folder containing core_cpt_text/selective_cpt_text."
        )

    records: list[DocRecord] = []
    seen: set[str] = set()

    def add_group(group_name: str, source_map: dict[str, Path]) -> None:
        for name, raw_path in sorted(source_map.items()):
            chosen_path = cleaned_map.get(name, raw_path)
            if name in seen:
                continue
            seen.add(name)
            text = read_text(chosen_path)
            records.append(
                DocRecord(
                    name=name,
                    path=str(chosen_path.resolve()),
                    group=group_name,
                    subject=infer_subject(name),
                    word_count=len(text.split()),
                )
            )

    add_group("core", core_map)
    add_group("selective", selective_map)

    if not records:
        for name, path in sorted(cleaned_map.items()):
            text = read_text(path)
            records.append(
                DocRecord(
                    name=name,
                    path=str(path.resolve()),
                    group="mixed",
                    subject=infer_subject(name),
                    word_count=len(text.split()),
                )
            )

    return records


def split_train_validation(
    records: list[DocRecord],
    seed: int,
    val_ratio: float,
    max_val_docs_per_subject: int,
    min_docs_for_subject_holdout: int,
) -> tuple[list[DocRecord], list[DocRecord]]:
    rng = random.Random(seed)
    by_subject: dict[str, list[DocRecord]] = defaultdict(list)
    for record in records:
        by_subject[record.subject].append(record)

    train_records: list[DocRecord] = []
    val_records: list[DocRecord] = []

    for subject, items in sorted(by_subject.items()):
        bucket = items[:]
        rng.shuffle(bucket)
        if len(bucket) < min_docs_for_subject_holdout or val_ratio <= 0.0:
            train_records.extend(bucket)
            continue

        val_count = max(1, int(round(len(bucket) * val_ratio)))
        if max_val_docs_per_subject > 0:
            val_count = min(val_count, max_val_docs_per_subject)
        val_count = min(val_count, len(bucket) - 1)
        if val_count <= 0:
            train_records.extend(bucket)
            continue

        val_records.extend(bucket[:val_count])
        train_records.extend(bucket[val_count:])

    return train_records, val_records


def round_robin_by_subject(records: list[DocRecord], seed: int) -> list[DocRecord]:
    rng = random.Random(seed)
    buckets: dict[str, list[DocRecord]] = defaultdict(list)
    for record in records:
        buckets[record.subject].append(record)

    subjects = list(sorted(buckets))
    for items in buckets.values():
        rng.shuffle(items)

    ordered: list[DocRecord] = []
    while True:
        made_progress = False
        rng.shuffle(subjects)
        for subject in subjects:
            if not buckets[subject]:
                continue
            ordered.append(buckets[subject].pop())
            made_progress = True
        if not made_progress:
            break
    return ordered


def expand_core_weight(records: list[DocRecord], core_weight_in_mixed: float, seed: int) -> list[DocRecord]:
    if core_weight_in_mixed <= 1.0:
        return records

    rng = random.Random(seed)
    expanded: list[DocRecord] = []
    integral = int(core_weight_in_mixed)
    fractional = core_weight_in_mixed - integral

    for record in records:
        copies = 1
        if record.group == "core":
            copies = max(1, integral)
            if fractional > 0.0 and rng.random() < fractional:
                copies += 1
        for _ in range(copies):
            expanded.append(record)
    return expanded


def build_book_schedule(
    train_records: list[DocRecord],
    stage1_epochs: float,
    stage2_epochs: float,
    core_weight_in_mixed: float,
    seed: int,
) -> list[BookUnit]:
    schedule: list[BookUnit] = []
    schedule_index = 0

    stage1_records = [record for record in train_records if record.group == "core"]
    if stage1_epochs > 0 and stage1_records:
        repeats = max(1, int(round(stage1_epochs)))
        ordered = round_robin_by_subject(stage1_records, seed)
        for epoch_idx in range(repeats):
            for record in ordered:
                schedule.append(
                    BookUnit(
                        schedule_index=schedule_index,
                        stage=f"stage1_core_epoch{epoch_idx + 1}",
                        name=record.name,
                        path=record.path,
                        subject=record.subject,
                        group=record.group,
                        word_count=record.word_count,
                    )
                )
                schedule_index += 1

    if stage2_epochs > 0:
        repeats = max(1, int(round(stage2_epochs)))
        mixed_records = expand_core_weight(train_records, core_weight_in_mixed, seed)
        ordered = round_robin_by_subject(mixed_records, seed + 7)
        for epoch_idx in range(repeats):
            for record in ordered:
                schedule.append(
                    BookUnit(
                        schedule_index=schedule_index,
                        stage=f"stage2_mixed_epoch{epoch_idx + 1}",
                        name=record.name,
                        path=record.path,
                        subject=record.subject,
                        group=record.group,
                        word_count=record.word_count,
                    )
                )
                schedule_index += 1

    return schedule


def pack_documents(
    records: list[DocRecord],
    tokenizer,
    sequence_length: int,
    seed: int,
) -> tuple[list[list[int]], dict]:
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        raise ValueError("Tokenizer must have an eos_token_id for causal language modeling.")

    ordered_records = round_robin_by_subject(records, seed)
    buffer: list[int] = []
    sequences: list[list[int]] = []
    total_tokens = 0
    used_docs = 0

    for record in ordered_records:
        text = read_text(Path(record.path)).strip()
        if not text:
            continue
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        if not token_ids:
            continue
        token_ids.append(eos_token_id)
        total_tokens += len(token_ids)
        used_docs += 1
        buffer.extend(token_ids)

        while len(buffer) >= sequence_length:
            sequences.append(buffer[:sequence_length])
            buffer = buffer[sequence_length:]

    stats = {
        "documents_used": used_docs,
        "packed_sequences": len(sequences),
        "sequence_length": sequence_length,
        "tokens_before_packing": total_tokens,
        "tokens_after_packing": len(sequences) * sequence_length,
        "dropped_tail_tokens": len(buffer),
    }
    return sequences, stats


def resolve_precision(args) -> tuple[torch.dtype | None, bool, bool]:
    if args.precision == "fp32":
        return torch.float32, False, False
    if args.precision == "fp16":
        return torch.float16, False, True
    if args.precision == "bf16":
        return torch.bfloat16, True, False

    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16, True, False
    if torch.cuda.is_available():
        return torch.float16, False, True
    return torch.float32, False, False


def maybe_apply_lora(model, args):
    if not args.use_lora:
        return model, {"enabled": False}

    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except ImportError as exc:
        raise RuntimeError("LoRA requested, but `peft` is not installed in the training image.") from exc

    target_modules = [item.strip() for item in args.lora_target_modules.split(",") if item.strip()]
    if not target_modules:
        raise ValueError("When --use-lora=True, --lora-target-modules must contain at least one module name.")

    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
    )
    model = get_peft_model(model, config)
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    trainable = 0
    total = 0
    for param in model.parameters():
        total += param.numel()
        if param.requires_grad:
            trainable += param.numel()

    return model, {
        "enabled": True,
        "r": args.lora_r,
        "alpha": args.lora_alpha,
        "dropout": args.lora_dropout,
        "target_modules": target_modules,
        "trainable_parameters": trainable,
        "total_parameters": total,
        "trainable_pct": round((100.0 * trainable / max(total, 1)), 4),
    }


def build_model_and_tokenizer(args):
    torch_dtype, bf16, fp16 = resolve_precision(args)
    tokenizer_name = args.tokenizer_name or args.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch_dtype,
        trust_remote_code=args.trust_remote_code,
    )

    if args.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    model, lora_info = maybe_apply_lora(model, args)
    return model, tokenizer, bf16, fp16, lora_info


def make_training_args(
    output_dir: Path,
    args,
    stage_epochs: float,
    bf16: bool,
    fp16: bool,
    eval_enabled: bool,
    run_name: str,
) -> TrainingArguments:
    return TrainingArguments(
        output_dir=str(output_dir),
        overwrite_output_dir=False,
        num_train_epochs=stage_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        max_grad_norm=args.max_grad_norm,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        eval_steps=args.eval_steps,
        eval_strategy="epoch" if eval_enabled else "no",
        save_strategy="epoch",
        bf16=bf16,
        fp16=fp16,
        dataloader_num_workers=args.dataloader_num_workers,
        report_to=[],
        remove_unused_columns=False,
        gradient_checkpointing=args.gradient_checkpointing,
        run_name=run_name,
        optim=args.optim,
    )


def trainer_metrics_to_report(metrics: dict | None) -> dict:
    if not metrics:
        return {}
    report = dict(metrics)
    loss_key = "eval_loss" if "eval_loss" in report else "train_loss" if "train_loss" in report else ""
    if loss_key:
        try:
            report[f"{loss_key}_perplexity"] = round(math.exp(report[loss_key]), 4)
        except OverflowError:
            report[f"{loss_key}_perplexity"] = float("inf")
    return report


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def safe_slug(text: str, max_len: int = 120) -> str:
    cleaned = "".join(char if char.isalnum() else "_" for char in text)
    cleaned = "_".join(part for part in cleaned.split("_") if part)
    return cleaned[:max_len] or "book"


def latest_checkpoint_path(output_dir: Path) -> str | None:
    if not output_dir.exists():
        return None
    checkpoints = []
    for path in output_dir.glob("checkpoint-*"):
        try:
            step = int(path.name.split("-")[-1])
        except ValueError:
            continue
        checkpoints.append((step, path))
    if not checkpoints:
        return None
    checkpoints.sort(key=lambda item: item[0])
    return str(checkpoints[-1][1])


def load_progress(progress_path: Path) -> dict:
    if not progress_path.exists():
        return {
            "status": "not_started",
            "completed_schedule_indices": [],
            "current_schedule_index": -1,
            "current_book": "",
            "current_stage": "",
            "last_completed_book": "",
            "last_completed_stage": "",
            "last_completed_snapshot": "",
        }
    try:
        payload = json.loads(progress_path.read_text(encoding="utf-8", errors="ignore"))
    except json.JSONDecodeError:
        payload = {}
    payload.setdefault("status", "not_started")
    payload.setdefault("completed_schedule_indices", [])
    payload.setdefault("current_schedule_index", -1)
    payload.setdefault("current_book", "")
    payload.setdefault("current_stage", "")
    payload.setdefault("last_completed_book", "")
    payload.setdefault("last_completed_stage", "")
    payload.setdefault("last_completed_snapshot", "")
    return payload


def save_progress(progress_path: Path, payload: dict) -> None:
    write_json(progress_path, payload)


def snapshot_dir_for_unit(base_dir: Path, unit: BookUnit) -> Path:
    return base_dir / f"{unit.schedule_index:05d}_{safe_slug(unit.stage)}_{safe_slug(unit.name, 80)}"


def load_snapshot_if_available(model, snapshot_path: str, args):
    if not snapshot_path:
        return model
    path = Path(snapshot_path)
    if not path.exists():
        return model

    adapter_config = path / "adapter_config.json"
    if adapter_config.exists():
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise RuntimeError("Found a PEFT snapshot for resume, but `peft` is not installed.") from exc
        return PeftModel.from_pretrained(model, str(path), is_trainable=True)

    return AutoModelForCausalLM.from_pretrained(
        str(path),
        torch_dtype=model.dtype,
        trust_remote_code=args.trust_remote_code,
    )


def maybe_load_validation_records(validation_dir: str) -> list[DocRecord]:
    if not validation_dir:
        return []
    path = Path(validation_dir)
    if not path.exists():
        return []
    return build_records_from_layout(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="SageMaker-friendly continued pretraining script for DocSmile CPT.")

    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen2.5-7B")
    parser.add_argument("--tokenizer_name", type=str, default="")
    parser.add_argument("--train_dir", type=str, default=env_default("SM_CHANNEL_TRAIN", "cpt_prepared"))
    parser.add_argument("--validation_dir", type=str, default=os.environ.get("SM_CHANNEL_VALIDATION", ""))
    parser.add_argument("--model_dir", type=str, default=env_default("SM_MODEL_DIR", "/opt/ml/model"))
    parser.add_argument("--checkpoint_dir", type=str, default="/opt/ml/checkpoints/cpt")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sequence_length", type=int, default=2048)
    parser.add_argument("--precision", choices=["auto", "bf16", "fp16", "fp32"], default="auto")
    parser.add_argument("--trust_remote_code", type=str2bool, default=True)
    parser.add_argument("--gradient_checkpointing", type=str2bool, default=True)

    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--optim", type=str, default="adamw_torch")
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--eval_steps", type=int, default=200)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--dataloader_num_workers", type=int, default=2)

    parser.add_argument("--stage1_epochs", type=float, default=1.0)
    parser.add_argument("--stage2_epochs", type=float, default=1.0)
    parser.add_argument("--core_weight_in_mixed", type=float, default=1.25)
    parser.add_argument("--max_books_per_run", type=int, default=0)

    parser.add_argument("--val_ratio", type=float, default=0.08)
    parser.add_argument("--max_val_docs_per_subject", type=int, default=1)
    parser.add_argument("--min_docs_for_subject_holdout", type=int, default=3)

    parser.add_argument("--use_lora", type=str2bool, default=False)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )

    args = parser.parse_args()
    set_seed(args.seed)

    train_dir = Path(args.train_dir)
    model_dir = Path(args.model_dir)
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    all_train_records = build_records_from_layout(train_dir)
    explicit_val_records = maybe_load_validation_records(args.validation_dir)
    if explicit_val_records:
        train_records = all_train_records
        val_records = explicit_val_records
    else:
        train_records, val_records = split_train_validation(
            records=all_train_records,
            seed=args.seed,
            val_ratio=args.val_ratio,
            max_val_docs_per_subject=args.max_val_docs_per_subject,
            min_docs_for_subject_holdout=args.min_docs_for_subject_holdout,
        )

    model, tokenizer, bf16, fp16, lora_info = build_model_and_tokenizer(args)
    model.config.use_cache = False

    if tokenizer.model_max_length and tokenizer.model_max_length < 10**12:
        if args.sequence_length > tokenizer.model_max_length:
            raise ValueError(
                f"sequence_length={args.sequence_length} exceeds tokenizer/model_max_length={tokenizer.model_max_length}."
            )

    val_sequences: list[list[int]] = []
    val_stats: dict = {}
    if val_records:
        val_sequences, val_stats = pack_documents(
            records=val_records,
            tokenizer=tokenizer,
            sequence_length=args.sequence_length,
            seed=args.seed + 99,
        )
    eval_dataset = PackedTokenDataset(val_sequences) if val_sequences else None

    schedule = build_book_schedule(
        train_records=train_records,
        stage1_epochs=args.stage1_epochs,
        stage2_epochs=args.stage2_epochs,
        core_weight_in_mixed=args.core_weight_in_mixed,
        seed=args.seed,
    )
    if not schedule:
        raise RuntimeError("No training schedule was created. Check the input corpus and stage epoch settings.")

    progress_path = checkpoint_dir / "progress.json"
    progress = load_progress(progress_path)
    progress["status"] = "running"
    progress["total_schedule_items"] = len(schedule)
    save_progress(progress_path, progress)

    completed_indices = set(int(value) for value in progress.get("completed_schedule_indices", []))
    last_snapshot = progress.get("last_completed_snapshot", "")
    if last_snapshot:
        model = load_snapshot_if_available(model, last_snapshot, args)
        model.config.use_cache = False

    stage_reports: list[dict] = []
    processed_this_run = 0

    for unit in schedule:
        if unit.schedule_index in completed_indices:
            continue
        if args.max_books_per_run > 0 and processed_this_run >= args.max_books_per_run:
            progress["status"] = "paused"
            save_progress(progress_path, progress)
            break

        progress["status"] = "running"
        progress["current_schedule_index"] = unit.schedule_index
        progress["current_book"] = unit.name
        progress["current_stage"] = unit.stage
        save_progress(progress_path, progress)

        train_sequences, train_stats = pack_documents(
            records=[
                DocRecord(
                    name=unit.name,
                    path=unit.path,
                    group=unit.group,
                    subject=unit.subject,
                    word_count=unit.word_count,
                )
            ],
            tokenizer=tokenizer,
            sequence_length=args.sequence_length,
            seed=args.seed + unit.schedule_index,
        )
        if not train_sequences:
            raise RuntimeError(f"{unit.name} produced zero training sequences.")

        train_dataset = PackedTokenDataset(train_sequences)
        unit_checkpoint_dir = snapshot_dir_for_unit(checkpoint_dir / "unit_checkpoints", unit)
        training_args = make_training_args(
            output_dir=unit_checkpoint_dir,
            args=args,
            stage_epochs=1.0,
            bf16=bf16,
            fp16=fp16,
            eval_enabled=eval_dataset is not None,
            run_name=unit.stage,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
        )

        train_result = trainer.train()
        eval_metrics = trainer.evaluate() if eval_dataset is not None else {}

        unit_snapshot_dir = snapshot_dir_for_unit(checkpoint_dir / "book_snapshots", unit)
        trainer.save_model(unit_snapshot_dir)
        tokenizer.save_pretrained(unit_snapshot_dir)

        progress["completed_schedule_indices"] = sorted(completed_indices | {unit.schedule_index})
        progress["last_completed_book"] = unit.name
        progress["last_completed_stage"] = unit.stage
        progress["last_completed_snapshot"] = str(unit_snapshot_dir)
        progress["current_schedule_index"] = unit.schedule_index
        progress["current_book"] = unit.name
        progress["current_stage"] = unit.stage
        save_progress(progress_path, progress)
        completed_indices.add(unit.schedule_index)

        stage_report = {
            "schedule_index": unit.schedule_index,
            "stage": unit.stage,
            "book": unit.name,
            "group": unit.group,
            "subject": unit.subject,
            "resume_checkpoint": "",
            "unit_checkpoint_dir": str(unit_checkpoint_dir),
            "snapshot_dir": str(unit_snapshot_dir),
            "train_stats": train_stats,
            "train_metrics": trainer_metrics_to_report(train_result.metrics),
            "eval_metrics": trainer_metrics_to_report(eval_metrics),
        }
        stage_reports.append(stage_report)
        print(json.dumps(stage_report, ensure_ascii=True), flush=True)
        model = trainer.model
        processed_this_run += 1

    if len(completed_indices) >= len(schedule):
        progress["status"] = "completed"
        progress["current_schedule_index"] = -1
        progress["current_book"] = ""
        progress["current_stage"] = ""
        save_progress(progress_path, progress)

    tokenizer.save_pretrained(model_dir)
    if hasattr(model, "save_pretrained"):
        model.save_pretrained(model_dir)

    existing_summary = read_json(model_dir / "cpt_training_summary.json")
    existing_reports = existing_summary.get("book_reports", [])
    merged_reports_by_index: dict[int, dict] = {}
    for row in existing_reports:
        if isinstance(row, dict) and "schedule_index" in row:
            merged_reports_by_index[int(row["schedule_index"])] = row
    for row in stage_reports:
        merged_reports_by_index[int(row["schedule_index"])] = row

    summary = {
        "model_name_or_path": args.model_name_or_path,
        "train_dir": str(train_dir.resolve()),
        "validation_dir": str(Path(args.validation_dir).resolve()) if args.validation_dir else "",
        "precision": args.precision,
        "bf16": bf16,
        "fp16": fp16,
        "sequence_length": args.sequence_length,
        "train_documents": len(train_records),
        "validation_documents": len(val_records),
        "validation_stats": val_stats,
        "lora": lora_info,
        "schedule_length": len(schedule),
        "book_reports": [merged_reports_by_index[key] for key in sorted(merged_reports_by_index)],
        "train_manifest": [asdict(record) for record in train_records],
        "validation_manifest": [asdict(record) for record in val_records],
        "schedule_manifest": [asdict(unit) for unit in schedule],
        "progress_path": str(progress_path),
        "run_status": progress.get("status", ""),
    }

    write_json(model_dir / "cpt_training_summary.json", summary)
    print(json.dumps({"final_model_dir": str(model_dir), "summary_path": str(model_dir / "cpt_training_summary.json")}))


if __name__ == "__main__":
    main()
