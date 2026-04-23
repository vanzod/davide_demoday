#!/bin/bash
###############################################################################
# Data prep wrapper — downloads xLAM + Glaive, cleans, tokenizes, packs.
#
# Idempotent: skips steps whose outputs already exist. Delete the outputs
# manually to force re-prep.
#
# Prerequisites:
#   - setup_environment.sh has been run
#   - HF_TOKEN exported or `huggingface-cli login` run previously
#   - Glaive terms accepted on the HF website (gated dataset)
#
# Usage:
#   ./prepare_data.sh                # default paths
#   WORK_DIR_ROOT=/custom ./prepare_data.sh
###############################################################################

set -euo pipefail

# ---------- Paths ----------
VENV_DIR="${VENV_DIR:-${HOME}/llm-finetune}"
WORK_DIR_ROOT="${WORK_DIR_ROOT:-/mnt/data/qwen_finetune}"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DATA_DIR="${WORK_DIR_ROOT}/datasets"
HF_CACHE="${WORK_DIR_ROOT}/hf_cache"

SRC_JSONL="${DATA_DIR}/function_calling_train.jsonl"
TOKENIZED_DIR="${DATA_DIR}/tokenized"

MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"

# ---------- Env ----------
if [ ! -f "${VENV_DIR}/bin/activate" ]; then
    echo "ERROR: venv not found at ${VENV_DIR}. Run ./setup_environment.sh first." >&2
    exit 1
fi
source "${VENV_DIR}/bin/activate"
export HF_HOME="${HF_CACHE}"

mkdir -p "${DATA_DIR}"

echo "============================================="
echo " Data preparation"
echo "============================================="
echo "Project:     ${PROJECT_DIR}"
echo "Work dir:    ${WORK_DIR_ROOT}"
echo "Data out:    ${DATA_DIR}"
echo "Model:       ${MODEL_NAME}"
echo

# ---------- Step 1: Clean + merge xLAM + Glaive ----------
if [ -f "${SRC_JSONL}" ]; then
    ROWS=$(wc -l < "${SRC_JSONL}")
    SIZE=$(du -h "${SRC_JSONL}" | cut -f1)
    echo "[1/3] Cleaned JSONL exists (${ROWS} rows, ${SIZE}). Skipping."
    echo "      Delete ${SRC_JSONL} to re-prepare."
else
    echo "[1/3] Downloading + cleaning xLAM and Glaive ..."
    python "${PROJECT_DIR}/scripts/prepare_fc_datasets.py" \
        --output-dir "${DATA_DIR}"
    if [ ! -f "${SRC_JSONL}" ]; then
        echo "ERROR: prepare_fc_datasets.py did not produce ${SRC_JSONL}" >&2
        exit 1
    fi
fi
echo

# ---------- Step 2: Pre-fetch tokenizer (small, ~10 MB) ----------
# snapshot_download is idempotent — it skips files that already match locally.
# We do this separately from step 3 so tokenize_pack.py runs against a local
# path (no Hub latency on workers). The 144 GB weights are NOT downloaded
# here — only tokenizer JSON/vocab files via allow_patterns.
echo "[2/3] Ensuring tokenizer files are cached (weights skipped) ..."
TOK_SNAPSHOT=$(python -c "
from huggingface_hub import snapshot_download
print(snapshot_download(
    '${MODEL_NAME}',
    cache_dir='${HF_CACHE}/hub',
    allow_patterns=['*.json', '*.txt', 'tokenizer*', 'special_tokens_map.json'],
))
")
echo "      tokenizer at: ${TOK_SNAPSHOT}"
echo

# ---------- Step 3: Tokenize + pack ----------
if [ -f "${TOKENIZED_DIR}/train/dataset_info.json" ] \
   && [ -f "${TOKENIZED_DIR}/eval/dataset_info.json" ]; then
    echo "[3/3] Tokenized dataset exists at ${TOKENIZED_DIR}. Skipping."
    echo "      Delete the directory to re-tokenize."
else
    echo "[3/3] Tokenizing + packing ..."
    python "${PROJECT_DIR}/scripts/tokenize_pack.py" \
        --jsonl     "${SRC_JSONL}" \
        --out-dir   "${TOKENIZED_DIR}" \
        --tokenizer "${TOK_SNAPSHOT}" \
        --seq-len   2048 \
        --eval-size 500 \
        --workers   8
fi
echo

# ---------- Summary ----------
echo "============================================="
echo " Data preparation complete"
echo "============================================="
echo "Cleaned JSONL:    ${SRC_JSONL}"
echo "Packed train:     ${TOKENIZED_DIR}/train"
echo "Unpacked eval:    ${TOKENIZED_DIR}/eval"
echo
echo "Next step:"
echo "  sbatch launch.sbatch"
