#!/usr/bin/env bash
# DocSmile Vast.ai bootstrap.
# Run ONCE on a fresh Vast.ai instance. ~5-7 minutes total on a good image.
#
# Recommended instance image:
#   pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime
#   (or any PyTorch >= 2.2 + CUDA 12.1 image — wheels are CUDA-tagged)
#
# This script INTENTIONALLY does NOT touch the system torch install.
# The Vast.ai image ships a CUDA-matched torch build; reinstalling it can
# silently switch to CPU-only and break training.

set -euo pipefail

echo "============================================================"
echo "DocSmile Vast.ai setup"
echo "============================================================"

# --- 0. Sanity: GPU + CUDA must already work ---
echo "[0/6] Pre-flight: checking GPU + torch + CUDA on the image..."
python - <<'PY'
import sys
try:
    import torch
except ImportError:
    sys.exit("ERROR: torch is not installed on this image. Pick a pytorch:* image.")

if not torch.cuda.is_available():
    sys.exit("ERROR: CUDA is not available on this instance.")

ver = torch.__version__
cuda_ver = torch.version.cuda
print(f"  torch          : {ver}")
print(f"  torch.cuda     : {cuda_ver}")
print(f"  device         : {torch.cuda.get_device_name(0)}")
print(f"  bf16 supported : {torch.cuda.is_bf16_supported()}")

# Warn if we're outside the wheel's sweet spot
major, minor = (int(x) for x in ver.split('+')[0].split('.')[:2])
if (major, minor) < (2, 2):
    sys.exit(f"ERROR: torch {ver} is older than 2.2 — flash-attn wheels won't match.")
if cuda_ver and not cuda_ver.startswith("12."):
    print(f"  WARN: cuda {cuda_ver} is not 12.x — flash-attn wheel may not match.")
PY

# --- 1. Persistent workspace + cache ---
mkdir -p /workspace/.hf_cache
mkdir -p /workspace/.hf_cache/datasets
mkdir -p /workspace/checkpoints
mkdir -p /workspace/runs

export HF_HOME=/workspace/.hf_cache
export TRANSFORMERS_CACHE=/workspace/.hf_cache
export HF_DATASETS_CACHE=/workspace/.hf_cache/datasets

# --- 2. System packages ---
echo "[1/6] Installing system packages..."
apt-get update -qq
apt-get install -y -qq git tmux htop nano curl wget tree >/dev/null

# --- 3. Python deps from requirements.txt ---
# IMPORTANT FLAGS:
#   --no-deps-on-torch via constraints below would be cleaner; instead we
#   skip torch by pinning it to whatever's already installed (a constraint
#   file referring to the live torch version).
echo "[2/6] Installing Python deps (this is the slow part)..."
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
pip install --no-cache-dir --upgrade pip wheel setuptools >/dev/null

# Generate a constraints file that pins torch to its currently-installed
# version so the upcoming `pip install -r requirements.txt` never replaces it.
INSTALLED_TORCH="$(python -c 'import torch; print(torch.__version__)')"
TORCH_PIN="${INSTALLED_TORCH%%+*}"   # strip +cu121 suffix if present
CONSTRAINTS="$(mktemp)"
echo "torch==${INSTALLED_TORCH}" > "${CONSTRAINTS}"
echo "  (torch pinned to ${INSTALLED_TORCH} via temporary constraint file)"

pip install --no-cache-dir -c "${CONSTRAINTS}" -r "${SCRIPT_DIR}/requirements.txt"
rm -f "${CONSTRAINTS}"

# --- 4. FlashAttention 2 (prebuilt wheel) ---
echo "[3/6] Installing FlashAttention 2..."
# Try the prebuilt wheel first. If it doesn't exist for this torch+CUDA combo,
# fall back to plain pip (which may compile from source — slow but works).
if ! pip install --no-cache-dir --no-build-isolation \
       "flash-attn>=2.6.3,<3" 2>/dev/null; then
    echo "  WARN: prebuilt wheel didn't match; falling back to source build."
    echo "        This may take ~20-30 min on first install."
    MAX_JOBS=4 pip install --no-cache-dir --no-build-isolation \
        "flash-attn>=2.6.3,<3"
fi

# --- 5. Verify final install ---
echo "[4/6] Verifying final install..."
python - <<'PY'
import torch
print(f"  torch          : {torch.__version__}")
print(f"  cuda available : {torch.cuda.is_available()}")
print(f"  cuda device    : {torch.cuda.get_device_name(0)}")
print(f"  bf16 supported : {torch.cuda.is_bf16_supported()}")
import unsloth
print(f"  unsloth        : {unsloth.__version__}")
try:
    import unsloth_zoo
    print(f"  unsloth_zoo    : {unsloth_zoo.__version__}")
except (ImportError, AttributeError):
    print("  unsloth_zoo    : present (version unknown)")
try:
    import flash_attn
    print(f"  flash_attn     : {flash_attn.__version__}")
except ImportError:
    print("  flash_attn     : NOT INSTALLED  (training will fall back to xformers/sdpa)")
import transformers, trl, peft, datasets, bitsandbytes
print(f"  transformers   : {transformers.__version__}")
print(f"  trl            : {trl.__version__}")
print(f"  peft           : {peft.__version__}")
print(f"  datasets       : {datasets.__version__}")
print(f"  bitsandbytes   : {bitsandbytes.__version__}")
PY

# --- 6. Env vars persisted to bashrc ---
echo "[5/6] Writing env defaults to ~/.bashrc ..."
if ! grep -q "# DocSmile env" ~/.bashrc 2>/dev/null; then
    {
      echo ""
      echo "# DocSmile env"
      echo "export HF_HOME=/workspace/.hf_cache"
      echo "export TRANSFORMERS_CACHE=/workspace/.hf_cache"
      echo "export HF_DATASETS_CACHE=/workspace/.hf_cache/datasets"
      echo "export HF_HUB_DISABLE_TELEMETRY=1"
      echo "export TOKENIZERS_PARALLELISM=true"
    } >> ~/.bashrc
    echo "  (added to ~/.bashrc)"
else
    echo "  (already in ~/.bashrc, skipping)"
fi

# --- 7. Final hints ---
echo "[6/6] Done."
echo ""
echo "============================================================"
echo "  Setup complete."
echo "  Next:"
echo "    1. cp training/.env.example training/.env"
echo "       nano training/.env             # paste your HF_TOKEN"
echo "    2. huggingface-cli login --token \$(grep ^HF_TOKEN training/.env | cut -d= -f2)"
echo "    3. tmux new -s pall"
echo "    4. python training/cpt_train.py --config training/cpt_config.yaml"
echo "       (Ctrl-b d  to detach;  tmux attach -t pall  to reconnect)"
echo "    5. In another tmux pane:"
echo "       tensorboard --logdir /workspace/runs --port 6006 --bind_all"
echo "============================================================"
