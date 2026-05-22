#!/usr/bin/env bash
# Resume training after a Vast.ai spot-instance interruption.
# Re-runs setup + pulls latest checkpoint from HF Hub if needed.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ ! -f "${SCRIPT_DIR}/.env" ]]; then
    echo "ERROR: ${SCRIPT_DIR}/.env not found. Copy .env.example and fill it in." >&2
    exit 1
fi

# Load env
set -a
source "${SCRIPT_DIR}/.env"
set +a

# Pull latest checkpoint from HF Hub (in case persistent disk got wiped)
if [[ -n "${HF_MODEL_REPO:-}" ]]; then
    echo "[resume] pulling latest checkpoint from ${HF_MODEL_REPO} ..."
    huggingface-cli download "${HF_MODEL_REPO}" \
        --local-dir "${SCRIPT_DIR}/checkpoints/cpt" \
        --token "${HF_TOKEN}" || echo "[resume] (no prior checkpoint on hub yet — that's fine)"
fi

# Resume
python "${SCRIPT_DIR}/cpt_train.py" --config "${SCRIPT_DIR}/cpt_config.yaml" --resume
