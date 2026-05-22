#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python -m pip install -U pip
python -m pip uninstall -y torchvision || true
python -m pip install -U "transformers==4.49.0" "huggingface-hub<1.0" accelerate datasets sentencepiece safetensors tqdm bitsandbytes

export TRANSFORMERS_NO_TORCHVISION=1
export HF_HUB_DISABLE_TELEMETRY=1

python scripts/run_text_eval.py \
  --eval-dir evals/text_only \
  --output-dir evals/results/qwen2_5_7b_base \
  --model Qwen/Qwen2.5-7B \
  --backend local-hf \
  --load-in-4bit \
  --dtype fp16
