#!/usr/bin/env bash
# DocSmile Vast.ai bootstrap.
# Run ONCE on a fresh Vast.ai instance. ~5-7 minutes total on a good image.
#
# Pick an instance with image: pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime
# (or any PyTorch 2.4 + CUDA 12.1 image — same wheel set works).

set -euo pipefail

echo "============================================================"
echo "DocSmile Vast.ai setup"
echo "============================================================"

# --- 1. Persistent workspace + cache ---
mkdir -p /workspace/.hf_cache
mkdir -p /workspace/.hf_cache/datasets
mkdir -p /workspace/checkpoints
mkdir -p /workspace/runs

export HF_HOME=/workspace/.hf_cache
export TRANSFORMERS_CACHE=/workspace/.hf_cache
export HF_DATASETS_CACHE=/workspace/.hf_cache/datasets

# --- 2. System packages ---
echo "[1/5] Installing system packages..."
apt-get update -qq
apt-get install -y -qq git tmux htop nano curl wget tree >/dev/null

# --- 3. Python deps from requirements.txt ---
echo "[2/5] Installing Python deps (this is the slow part)..."
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
pip install --no-cache-dir --upgrade pip wheel setuptools >/dev/null
pip install --no-cache-dir -r "${SCRIPT_DIR}/requirements.txt"

# --- 4. FlashAttention 2 (prebuilt wheel for cu121 + torch 2.4) ---
echo "[3/5] Installing FlashAttention 2..."
# Use the prebuilt wheel — skips the 20-30 min source compile.
pip install --no-cache-dir --no-build-isolation flash-attn==2.6.3

# --- 5. Verify CUDA + Unsloth + flash-attn ---
echo "[4/5] Verifying install..."
python - <<'PY'
import torch
print(f"  torch          : {torch.__version__}")
print(f"  cuda available : {torch.cuda.is_available()}")
print(f"  cuda device    : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE'}")
print(f"  bf16 supported : {torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False}")
import unsloth
print(f"  unsloth        : {unsloth.__version__}")
try:
    import flash_attn
    print(f"  flash_attn     : {flash_attn.__version__}")
except ImportError:
    print("  flash_attn     : NOT INSTALLED")
PY

# --- 6. Env vars (persist across shells in same instance) ---
echo "[5/5] Writing env defaults to ~/.bashrc ..."
{
  echo ""
  echo "# DocSmile env"
  echo "export HF_HOME=/workspace/.hf_cache"
  echo "export TRANSFORMERS_CACHE=/workspace/.hf_cache"
  echo "export HF_DATASETS_CACHE=/workspace/.hf_cache/datasets"
  echo "export HF_HUB_DISABLE_TELEMETRY=1"
  echo "export TOKENIZERS_PARALLELISM=true"
} >> ~/.bashrc

echo ""
echo "============================================================"
echo "  Setup complete."
echo "  Next:"
echo "    1. cp training/.env.example training/.env  (and fill in HF_TOKEN)"
echo "    2. source training/.env && export \$(grep -v ^# training/.env | xargs)"
echo "    3. huggingface-cli login --token \$HF_TOKEN"
echo "    4. python training/cpt_train.py --config training/cpt_config.yaml"
echo "       (TensorBoard: in another tmux pane,"
echo "        tensorboard --logdir /workspace/runs --port 6006 --bind_all)"
echo "============================================================"
