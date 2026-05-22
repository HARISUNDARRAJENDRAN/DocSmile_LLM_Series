from __future__ import annotations

import argparse
import inspect
import json
import math
import os
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, set_seed


class PackedCausalDataset(Dataset):
    def __init__(self, path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(f"Packed dataset not found: {path}")
        self.input_ids = self._load(path)
        if self.input_ids.ndim != 2:
            raise ValueError(f"Expected 2D packed tensor in {path}, got shape {tuple(self.input_ids.shape)}")

    @staticmethod
    def _load(path: Path) -> torch.Tensor:
        try:
            payload = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
        except TypeError:
            payload = torch.load(path, map_location="cpu")
        input_ids = payload["input_ids"] if isinstance(payload, dict) else payload
        return input_ids.to(dtype=torch.long)

    def __len__(self) -> int:
        return int(self.input_ids.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.input_ids[index]
        return {"input_ids": row, "labels": row.clone()}


def resolve_dtype(precision: str) -> tuple[torch.dtype, bool, bool]:
    if precision == "fp32":
        return torch.float32, False, False
    if precision == "fp16":
        return torch.float16, False, True
    if precision == "bf16":
        return torch.bfloat16, True, False
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16, True, False
    if torch.cuda.is_available():
        return torch.float16, False, True
    return torch.float32, False, False


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def auto_lora_targets(model) -> list[str]:
    model_type = getattr(model.config, "model_type", "").lower()
    if model_type in {"llama", "mistral", "mixtral", "qwen2", "qwen3", "gemma", "gemma2"}:
        return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    if model_type in {"gpt2"}:
        return ["c_attn", "c_proj", "c_fc"]
    # Common modern decoder names. PEFT will fail clearly if a model uses
    # unusual module names, and the user can pass --lora-target-modules.
    return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def maybe_liger_model_class(use_liger: bool):
    if not use_liger:
        return AutoModelForCausalLM
    try:
        from liger_kernel.transformers import AutoLigerKernelForCausalLM
    except ImportError as exc:
        raise RuntimeError(
            "--use-liger requires `liger-kernel` and a supported decoder architecture."
        ) from exc
    return AutoLigerKernelForCausalLM


def build_model_and_tokenizer(args: argparse.Namespace):
    dtype, bf16, fp16 = resolve_dtype(args.precision)
    tokenizer_name = args.tokenizer_name or args.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict = {
        "trust_remote_code": args.trust_remote_code,
    }
    if args.attn_implementation != "auto":
        model_kwargs["attn_implementation"] = args.attn_implementation

    quantized = args.training_mode == "qlora"
    if quantized:
        if not torch.cuda.is_available():
            raise RuntimeError("QLoRA needs CUDA plus bitsandbytes. Use --training-mode full for CPU smoke tests.")
        try:
            from transformers import BitsAndBytesConfig
        except ImportError as exc:
            raise RuntimeError("QLoRA requires a transformers build with BitsAndBytesConfig.") from exc
        compute_dtype = torch.bfloat16 if bf16 else torch.float16
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=args.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )
        local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
        if local_rank >= 0:
            model_kwargs["device_map"] = {"": local_rank}
    else:
        model_kwargs["torch_dtype"] = dtype

    model_class = maybe_liger_model_class(args.use_liger)
    model = model_class.from_pretrained(args.model_name_or_path, **model_kwargs)
    model.config.use_cache = False

    if args.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        try:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            model.gradient_checkpointing_enable()

    if args.training_mode in {"lora", "qlora"}:
        try:
            from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
        except ImportError as exc:
            raise RuntimeError("LoRA/QLoRA requires `peft`. Install requirements-cpt.txt on the GPU host.") from exc

        if quantized:
            model = prepare_model_for_kbit_training(
                model,
                use_gradient_checkpointing=args.gradient_checkpointing,
            )

        target_modules = auto_lora_targets(model) if args.lora_target_modules == "auto" else parse_csv(args.lora_target_modules)
        modules_to_save = parse_csv(args.lora_modules_to_save)
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            target_modules=target_modules,
            modules_to_save=modules_to_save or None,
        )
        model = get_peft_model(model, lora_config)
        if hasattr(model, "print_trainable_parameters"):
            model.print_trainable_parameters()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    return model, tokenizer, bf16, fp16


def load_manifest(dataset_dir: Path) -> dict:
    path = dataset_dir / "manifest.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    except json.JSONDecodeError:
        return {}


def latest_checkpoint(output_dir: Path) -> str | None:
    checkpoints = []
    for item in output_dir.glob("checkpoint-*"):
        if not item.is_dir():
            continue
        try:
            step = int(item.name.rsplit("-", 1)[-1])
        except ValueError:
            continue
        checkpoints.append((step, item))
    if not checkpoints:
        return None
    checkpoints.sort(key=lambda row: row[0])
    return str(checkpoints[-1][1])


def make_training_args(args: argparse.Namespace, bf16: bool, fp16: bool, has_eval: bool) -> TrainingArguments:
    sig = inspect.signature(TrainingArguments.__init__)
    kwargs = {
        "output_dir": args.output_dir,
        "overwrite_output_dir": False,
        "max_steps": args.max_steps,
        "num_train_epochs": args.num_train_epochs,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "lr_scheduler_type": args.lr_scheduler_type,
        "max_grad_norm": args.max_grad_norm,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "save_total_limit": args.save_total_limit,
        "save_strategy": "steps",
        "eval_steps": args.eval_steps,
        "bf16": bf16,
        "fp16": fp16,
        "optim": args.optim,
        "dataloader_num_workers": args.dataloader_num_workers,
        "remove_unused_columns": False,
        "gradient_checkpointing": args.gradient_checkpointing,
        "report_to": [] if args.report_to == "none" else parse_csv(args.report_to),
        "logging_dir": str(Path(args.output_dir) / "logs"),
        "run_name": args.run_name,
        "save_safetensors": True,
        "deepspeed": args.deepspeed or None,
        "torch_compile": args.torch_compile,
        "ddp_find_unused_parameters": False if int(os.environ.get("WORLD_SIZE", "1")) > 1 else None,
        "include_tokens_per_second": True,
        "include_num_input_tokens_seen": True,
    }

    if "eval_strategy" in sig.parameters:
        kwargs["eval_strategy"] = "steps" if has_eval else "no"
    else:
        kwargs["evaluation_strategy"] = "steps" if has_eval else "no"

    if args.fsdp:
        kwargs["fsdp"] = args.fsdp
    if args.fsdp_transformer_layer_cls_to_wrap:
        kwargs["fsdp_config"] = {"transformer_layer_cls_to_wrap": args.fsdp_transformer_layer_cls_to_wrap}

    filtered = {key: value for key, value in kwargs.items() if key in sig.parameters and value is not None}
    return TrainingArguments(**filtered)


def add_perplexity(metrics: dict) -> dict:
    out = dict(metrics)
    for key in ("eval_loss", "train_loss"):
        if key in out:
            try:
                out[f"{key}_perplexity"] = math.exp(float(out[key]))
            except OverflowError:
                out[f"{key}_perplexity"] = float("inf")
    return out


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Production CPT trainer for packed causal-LM tensors.")
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--tokenizer-name", default="")
    parser.add_argument("--dataset-dir", required=True, help="Directory created by scripts/pack_cpt_dataset.py.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-name", default="docsmile-cpt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trust-remote-code", action="store_true")

    parser.add_argument("--training-mode", choices=["full", "lora", "qlora"], default="qlora")
    parser.add_argument("--precision", choices=["auto", "bf16", "fp16", "fp32"], default="auto")
    parser.add_argument("--attn-implementation", choices=["auto", "flash_attention_2", "sdpa", "eager"], default="auto")
    parser.add_argument("--use-liger", action="store_true")
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--torch-compile", action="store_true")
    parser.add_argument("--bnb-4bit-quant-type", choices=["nf4", "fp4"], default="nf4")

    parser.add_argument("--lora-r", type=int, default=64)
    parser.add_argument("--lora-alpha", type=int, default=128)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-target-modules", default="auto")
    parser.add_argument("--lora-modules-to-save", default="", help="Optional CSV, e.g. embed_tokens,lm_head.")

    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--lr-scheduler-type", default="cosine")
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--optim", default="adamw_torch")

    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--eval-steps", type=int, default=200)
    parser.add_argument("--save-total-limit", type=int, default=3)
    parser.add_argument("--dataloader-num-workers", type=int, default=2)
    parser.add_argument("--report-to", default="tensorboard", help="none,tensorboard,wandb, or CSV.")
    parser.add_argument("--resume-from-checkpoint", default="")
    parser.add_argument("--auto-resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--deepspeed", default="")
    parser.add_argument("--fsdp", default="", help='Example: "full_shard auto_wrap". Prefer DeepSpeed for full CPT.')
    parser.add_argument("--fsdp-transformer-layer-cls-to-wrap", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    dataset_dir = Path(args.dataset_dir)
    train_dataset = PackedCausalDataset(dataset_dir / "train.pt")
    validation_path = dataset_dir / "validation.pt"
    eval_dataset = PackedCausalDataset(validation_path) if validation_path.exists() else None

    model, tokenizer, bf16, fp16 = build_model_and_tokenizer(args)
    training_args = make_training_args(args, bf16=bf16, fp16=fp16, has_eval=eval_dataset is not None)

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
    }
    trainer_sig = inspect.signature(Trainer.__init__)
    if "processing_class" in trainer_sig.parameters:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_sig.parameters:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = Trainer(**trainer_kwargs)

    resume = args.resume_from_checkpoint or (latest_checkpoint(Path(args.output_dir)) if args.auto_resume else None)
    train_result = trainer.train(resume_from_checkpoint=resume)
    eval_metrics = trainer.evaluate() if eval_dataset is not None else {}

    final_dir = Path(args.output_dir) / "final"
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)

    summary = {
        "model_name_or_path": args.model_name_or_path,
        "training_mode": args.training_mode,
        "precision": args.precision,
        "bf16": bf16,
        "fp16": fp16,
        "attn_implementation": args.attn_implementation,
        "use_liger": args.use_liger,
        "dataset_dir": str(dataset_dir.resolve()),
        "dataset_manifest": load_manifest(dataset_dir),
        "train_rows": len(train_dataset),
        "validation_rows": len(eval_dataset) if eval_dataset is not None else 0,
        "resume_from_checkpoint": resume or "",
        "train_metrics": add_perplexity(train_result.metrics),
        "eval_metrics": add_perplexity(eval_metrics),
        "final_dir": str(final_dir.resolve()),
        "torch_version": torch.__version__,
    }
    write_json(Path(args.output_dir) / "cpt_run_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
