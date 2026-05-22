from __future__ import annotations

import argparse
import json
import os
import re
import time
import urllib.request
from pathlib import Path


os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

CHOICE_RE = re.compile(r"\b([ABCD])\b", re.IGNORECASE)
EXPECTED_EVAL_STEMS = [
    "medmcqa_dental_mcq",
    "oral_disease_open_qa",
    "dental_forum_open_qa",
]


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def mcq_prompt(row: dict) -> str:
    options = row["options"]
    return (
        "Answer the dental multiple-choice question. "
        "Return only the letter A, B, C, or D.\n\n"
        f"Question: {row['question']}\n"
        f"A. {options[0]}\n"
        f"B. {options[1]}\n"
        f"C. {options[2]}\n"
        f"D. {options[3]}\n"
        "Answer:"
    )


def open_prompt(row: dict) -> str:
    return (
        "Answer the dental question accurately and safely. "
        "Be concise, grounded, and mention seeing a dentist when diagnosis or urgent care is needed.\n\n"
        f"Question: {row['question']}\n"
        "Answer:"
    )


def extract_choice(text: str) -> str:
    boxed = re.search(r"\\boxed\{([ABCD])\}", text, re.IGNORECASE)
    if boxed:
        return boxed.group(1).upper()
    first_line = text.strip().splitlines()[0] if text.strip() else ""
    match = CHOICE_RE.search(first_line) or CHOICE_RE.search(text)
    return match.group(1).upper() if match else ""


class OpenAICompatClient:
    def __init__(self, base_url: str, api_key: str, model: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model

    def generate(self, prompt: str, max_new_tokens: int) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": max_new_tokens,
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )
        with urllib.request.urlopen(req, timeout=180) as resp:
            parsed = json.loads(resp.read().decode("utf-8", errors="ignore"))
        return parsed["choices"][0]["message"]["content"].strip()


class LocalHFClient:
    def __init__(self, model: str, dtype: str = "auto", load_in_4bit: bool = False, adapter_path: str = "") -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is not available. A 7B/8B Qwen baseline is not practical on this CPU-only machine."
            )

        torch_dtype = "auto"
        if dtype == "bf16":
            torch_dtype = torch.bfloat16
        elif dtype == "fp16":
            torch_dtype = torch.float16

        model_kwargs = {
            "device_map": "auto",
            "trust_remote_code": True,
        }
        if load_in_4bit:
            try:
                from transformers import BitsAndBytesConfig
            except ImportError as exc:
                raise RuntimeError("4-bit loading requires bitsandbytes and a recent transformers version.") from exc
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        else:
            model_kwargs["torch_dtype"] = torch_dtype

        tokenizer_source = adapter_path or model
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)
        if self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model, **model_kwargs)
        if adapter_path:
            try:
                from peft import PeftModel
            except ImportError as exc:
                raise RuntimeError("Loading --adapter-path requires peft.") from exc
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self.model.eval()

    def generate(self, prompt: str, max_new_tokens: int) -> str:
        import torch

        messages = [{"role": "user", "content": prompt}]
        if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template:
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            text = prompt
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        new_tokens = out[0][inputs["input_ids"].shape[-1] :]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def build_client(args):
    if args.backend == "openai-compatible":
        api_key = args.api_key or os.environ.get("OPENAI_API_KEY", "")
        base_url = args.base_url or os.environ.get("OPENAI_BASE_URL", "")
        if not api_key or not base_url:
            raise RuntimeError("OpenAI-compatible backend requires --api-key/--base-url or env vars.")
        return OpenAICompatClient(base_url, api_key, args.model)
    return LocalHFClient(args.model, args.dtype, args.load_in_4bit, args.adapter_path)


def collect_eval_files(eval_dir: Path) -> tuple[list[Path], list[str]]:
    available = {}
    for path in sorted([*eval_dir.glob("*.jsonl"), *eval_dir.glob("*.json")]):
        if path.name == "manifest.json":
            continue
        available.setdefault(path.stem, path)

    missing = [stem for stem in EXPECTED_EVAL_STEMS if stem not in available]
    ordered = [available[stem] for stem in EXPECTED_EVAL_STEMS if stem in available]
    extras = sorted(path for stem, path in available.items() if stem not in EXPECTED_EVAL_STEMS)
    return ordered + extras, missing


def evaluate_file(client, path: Path, output_dir: Path, max_rows: int) -> dict:
    rows = read_jsonl(path)
    if max_rows > 0:
        rows = rows[:max_rows]

    predictions = []
    correct = 0
    scored = 0
    for index, row in enumerate(rows, start=1):
        task = row.get("task")
        prompt = mcq_prompt(row) if task == "mcq" else open_prompt(row)
        max_new_tokens = 8 if task == "mcq" else 256
        started = time.time()
        text = client.generate(prompt, max_new_tokens=max_new_tokens)
        elapsed = round(time.time() - started, 3)

        pred = {
            "id": row.get("id"),
            "task": task,
            "source": row.get("source"),
            "prediction": text,
            "elapsed_sec": elapsed,
        }
        if task == "mcq":
            choice = extract_choice(text)
            pred["predicted_label"] = choice
            pred["answer_label"] = row.get("answer_label")
            pred["correct"] = choice == row.get("answer_label")
            correct += int(pred["correct"])
            scored += 1
        predictions.append(pred)
        print(f"{path.name} [{index}/{len(rows)}] done", flush=True)

    out_path = output_dir / f"{path.stem}_predictions.jsonl"
    write_jsonl(out_path, predictions)
    return {
        "dataset": path.name,
        "rows": len(rows),
        "predictions": str(out_path),
        "mcq_accuracy": round(correct / scored, 4) if scored else None,
        "mcq_correct": correct if scored else None,
        "mcq_scored": scored if scored else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run text-only dental evals on a Qwen/HF or API model.")
    parser.add_argument("--eval-dir", default="evals/text_only")
    parser.add_argument("--output-dir", default="evals/results/qwen2_5_7b_base")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B")
    parser.add_argument("--backend", choices=["local-hf", "openai-compatible"], default="local-hf")
    parser.add_argument("--dtype", choices=["auto", "bf16", "fp16"], default="auto")
    parser.add_argument("--load-in-4bit", action="store_true", help="Load local HF model in 4-bit NF4; recommended for T4.")
    parser.add_argument("--adapter-path", default="", help="Optional PEFT adapter path to load on top of --model.")
    parser.add_argument("--base-url", default="")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--allow-missing", action="store_true", help="Run even if expected eval files are missing.")
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    client = build_client(args)
    summaries = []
    eval_files, missing = collect_eval_files(eval_dir)
    print(f"Eval directory: {eval_dir.resolve()}", flush=True)
    print(
        "Eval files found: "
        + ", ".join(
            path.name
            for path in sorted([*eval_dir.glob("*.jsonl"), *eval_dir.glob("*.json")])
            if path.name != "manifest.json"
        ),
        flush=True,
    )
    if missing:
        print(
            "WARNING: Missing expected eval files: " + ", ".join(missing),
            flush=True,
        )
        if not args.allow_missing:
            raise RuntimeError(
                "Missing expected eval files. Fix evals/text_only or rerun with --allow-missing."
            )
    if not eval_files:
        raise RuntimeError(f"No .jsonl eval files found in {eval_dir}")

    for path in eval_files:
        summaries.append(evaluate_file(client, path, output_dir, args.max_rows))

    summary = {
        "model": args.model,
        "backend": args.backend,
        "eval_dir": str(eval_dir),
        "eval_files": [str(path) for path in eval_files],
        "missing_expected_files": missing,
        "results": summaries,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
