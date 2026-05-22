"""DocSmile Continued Pretraining (CPT) script.

Llama 3.1 8B + Unsloth + QLoRA (rank 64, all-linear) + FlashAttention 2 +
sequence packing. Single A100 40GB.

Usage (on the Vast.ai instance):
    python training/cpt_train.py --config training/cpt_config.yaml

Resuming after spot-instance interruption:
    python training/cpt_train.py --config training/cpt_config.yaml --resume

TensorBoard:
    tensorboard --logdir /workspace/runs --port 6006 --bind_all
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import yaml

# IMPORTANT: unsloth must be imported BEFORE transformers / trl. Patches them.
from unsloth import FastLanguageModel, is_bfloat16_supported

import torch
from datasets import load_dataset
from transformers import TrainingArguments, set_seed
from trl import SFTTrainer

from callbacks import (
    GpuMemoryCallback,
    GradientNormCallback,
    MCQEvalCallback,
    ThroughputCallback,
)


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_env_dotenv() -> None:
    """Lightweight .env loader so HF_TOKEN / HF_HOME / etc. work without pip-installing python-dotenv."""
    env_path = Path(__file__).parent / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        if k and v and k not in os.environ:
            os.environ[k] = v


def banner(title: str) -> None:
    line = "=" * 70
    print(f"\n{line}\n  {title}\n{line}")


def build_model_and_tokenizer(cfg: dict):
    m = cfg["model"]
    banner(f"Loading {m['name']} (4-bit={m['load_in_4bit']}, ctx={m['max_seq_length']})")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=m["name"],
        max_seq_length=m["max_seq_length"],
        dtype=m["dtype"],
        load_in_4bit=m["load_in_4bit"],
    )

    l = cfg["lora"]
    banner(
        f"Attaching LoRA r={l['r']} alpha={l['alpha']} "
        f"drop={l['dropout']} targets={len(l['target_modules'])}"
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=l["r"],
        target_modules=l["target_modules"],
        lora_alpha=l["alpha"],
        lora_dropout=l["dropout"],
        bias=l["bias"],
        use_gradient_checkpointing=l["use_gradient_checkpointing"],
        use_rslora=l["use_rslora"],
        random_state=l["random_state"],
    )
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"  trainable params: {n_train / 1e6:.1f}M  /  {n_total / 1e9:.2f}B  "
          f"({100 * n_train / n_total:.3f}%)")

    return model, tokenizer


def build_datasets(cfg: dict):
    d = cfg["data"]
    banner(f"Loading dataset {d['repo']} (config={d['config']})")
    train_ds = load_dataset(
        d["repo"], d["config"], split=d["train_split"], token=os.environ.get("HF_TOKEN"),
    )
    val_ds = load_dataset(
        d["repo"], d["config"], split=d["val_split"], token=os.environ.get("HF_TOKEN"),
    )
    print(f"  train rows: {len(train_ds):,}")
    print(f"  val   rows: {len(val_ds):,}")
    return train_ds, val_ds


def build_training_args(cfg: dict) -> TrainingArguments:
    t = cfg["training"]
    s = cfg["schedule"]
    h = cfg.get("hub", {})
    tb = cfg.get("tensorboard", {})

    bf16 = t.get("bf16", False) and is_bfloat16_supported()
    fp16 = t.get("fp16", False) and (not bf16)

    report_to = []
    if tb.get("enabled", True):
        report_to.append("tensorboard")

    args = TrainingArguments(
        output_dir=t["output_dir"],
        overwrite_output_dir=False,
        num_train_epochs=t["num_train_epochs"],

        per_device_train_batch_size=t["per_device_train_batch_size"],
        per_device_eval_batch_size=t["per_device_eval_batch_size"],
        gradient_accumulation_steps=t["gradient_accumulation_steps"],

        learning_rate=t["learning_rate"],
        warmup_ratio=t["warmup_ratio"],
        lr_scheduler_type=t["lr_scheduler_type"],
        weight_decay=t["weight_decay"],
        max_grad_norm=t["max_grad_norm"],

        optim=t["optim"],
        bf16=bf16,
        fp16=fp16,

        logging_steps=s["logging_steps"],
        eval_strategy="steps",
        eval_steps=s["eval_steps"],
        save_strategy="steps",
        save_steps=s["save_steps"],
        save_total_limit=s["save_total_limit"],

        logging_dir=tb.get("logdir", "./runs/cpt"),
        report_to=report_to,

        push_to_hub=h.get("push_to_hub", False),
        hub_model_id=h.get("hub_model_id"),
        hub_strategy=h.get("hub_strategy", "every_save"),
        hub_private_repo=h.get("hub_private_repo", True),
        hub_token=os.environ.get("HF_TOKEN"),

        dataloader_num_workers=t["dataloader_num_workers"],
        seed=t["seed"],

        load_best_model_at_end=False,
        remove_unused_columns=False,
        group_by_length=t["group_by_length"],
    )
    return args


def build_trainer(cfg: dict, model, tokenizer, train_ds, val_ds, args):
    t = cfg["training"]
    d = cfg["data"]
    callbacks = [
        GpuMemoryCallback(),
        ThroughputCallback(max_seq_length=cfg["model"]["max_seq_length"]),
        GradientNormCallback(),
    ]
    mcq = cfg.get("mcq_eval", {})
    if mcq.get("enabled", False):
        eval_path = Path(__file__).parent / mcq["path"]
        callbacks.append(MCQEvalCallback(
            eval_path=eval_path.resolve(),
            tokenizer=tokenizer,
            every_n_steps=mcq.get("every_n_steps", 1000),
            max_samples=mcq.get("max_samples", 100),
        ))

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        dataset_text_field=d["text_field"],
        max_seq_length=cfg["model"]["max_seq_length"],
        dataset_num_proc=t["dataset_num_proc"],
        packing=t["packing"],
        args=args,
        callbacks=callbacks,
    )
    return trainer


def find_resume_checkpoint(output_dir: Path) -> Path | None:
    """Pick the latest valid `checkpoint-N` under output_dir."""
    if not output_dir.exists():
        return None
    candidates = []
    for p in output_dir.iterdir():
        if p.is_dir() and p.name.startswith("checkpoint-"):
            try:
                step = int(p.name.split("-", 1)[1])
                if (p / "adapter_config.json").exists() or (p / "adapter_model.safetensors").exists():
                    candidates.append((step, p))
            except ValueError:
                continue
    if not candidates:
        return None
    candidates.sort()
    return candidates[-1][1]


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="training/cpt_config.yaml")
    p.add_argument("--resume", action="store_true",
                   help="Resume from the latest checkpoint in output_dir if present")
    args = p.parse_args(argv)

    load_env_dotenv()

    cfg = load_config(Path(args.config))
    set_seed(cfg["training"]["seed"])

    # --- HW sanity ---
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available", file=sys.stderr)
        return 1
    banner("Hardware")
    print(f"  device : {torch.cuda.get_device_name(0)}")
    print(f"  vram   : {torch.cuda.get_device_properties(0).total_memory / 1024**3:.0f} GB")
    print(f"  bf16   : {is_bfloat16_supported()}")
    print(f"  flash_attn : ", end="")
    try:
        import flash_attn  # noqa: F401
        print("yes")
    except ImportError:
        print("no")

    # --- Model + tokenizer ---
    model, tokenizer = build_model_and_tokenizer(cfg)

    # --- Datasets ---
    train_ds, val_ds = build_datasets(cfg)

    # --- TrainingArguments + Trainer ---
    targs = build_training_args(cfg)
    trainer = build_trainer(cfg, model, tokenizer, train_ds, val_ds, targs)

    # --- Resume? ---
    resume_from = None
    if args.resume:
        resume_from = find_resume_checkpoint(Path(targs.output_dir))
        if resume_from is None:
            print(f"[resume] no checkpoint under {targs.output_dir} — starting from scratch")
        else:
            print(f"[resume] resuming from {resume_from}")

    # --- Effective batch summary ---
    effective_batch = (
        targs.per_device_train_batch_size * targs.gradient_accumulation_steps
    )
    tokens_per_step = effective_batch * cfg["model"]["max_seq_length"]
    banner("Effective batch + token math")
    print(f"  per_device_batch    : {targs.per_device_train_batch_size}")
    print(f"  grad_accum_steps    : {targs.gradient_accumulation_steps}")
    print(f"  effective batch     : {effective_batch}")
    print(f"  max_seq_length      : {cfg['model']['max_seq_length']}")
    print(f"  tokens per opt step : {tokens_per_step:,}")
    print(f"  total epochs        : {targs.num_train_epochs}")

    banner("Starting training")
    train_result = trainer.train(resume_from_checkpoint=resume_from)
    print(train_result)

    # Save final adapter
    final_dir = Path(targs.output_dir) / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"\n[done] final adapter saved to {final_dir}")

    # Final eval
    banner("Final eval (validation perplexity)")
    metrics = trainer.evaluate()
    loss = metrics.get("eval_loss")
    if loss is not None:
        import math as _math
        metrics["eval_perplexity"] = _math.exp(min(loss, 20.0))
        print(f"  eval_loss        = {loss:.4f}")
        print(f"  eval_perplexity  = {metrics['eval_perplexity']:.2f}")
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)

    # Final hub push (already pushed during training, but make sure the last
    # checkpoint is there)
    if targs.push_to_hub:
        print("[hub] final push to HF Hub")
        trainer.push_to_hub(commit_message="final CPT adapter")
    return 0


if __name__ == "__main__":
    sys.exit(main())
