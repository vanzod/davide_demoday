#!/bin/bash
###############################################################################
# Environment Setup — Qwen2.5-72B Function-Calling Fine-Tune
#
# Creates a Python venv and installs the minimum packages needed for full FT
# of Qwen2.5-72B on 2x8 H200 with FSDP2 + FlashAttention-2.
#
# GPU verification requires CUDA at install time. This script runs fine on
# any node with internet access; if no GPU is visible it installs everything
# and warns at the end. For full end-to-end verification, submit via the
# SLURM wrapper:
#
#   sbatch setup_environment.sbatch
#
# Or run interactively on a GPU node:
#
#   ./setup_environment.sh
#
# Flags:
#   --force      Recreate venv without prompting
#   --keep       Keep existing venv, skip install
###############################################################################

set -euo pipefail

VENV_DIR="${VENV_DIR:-${HOME}/llm-finetune}"
PYTHON_BIN="${PYTHON_BIN:-python3.12}"
FORCE_RECREATE=false
KEEP_EXISTING=false

# ---------- Arg parsing ----------
while [[ $# -gt 0 ]]; do
    case $1 in
        --force) FORCE_RECREATE=true; shift ;;
        --keep)  KEEP_EXISTING=true;  shift ;;
        -h|--help)
            grep '^#' "$0" | head -30
            exit 0
            ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

echo "============================================="
echo " Environment Setup"
echo "============================================="
echo "Venv target: ${VENV_DIR}"
echo "Python:      ${PYTHON_BIN} ($(${PYTHON_BIN} --version 2>&1))"
echo
echo "Note: CUDA + flash-attn runtime verification requires a visible GPU."
echo "      If this runs on a login/CPU node, install completes but the"
echo "      runtime check is skipped. Use setup_environment.sbatch to"
echo "      run on a GPU node for full verification."
echo

# ---------- Venv create/recreate ----------
if [ -d "${VENV_DIR}" ]; then
    if [ "$FORCE_RECREATE" = true ]; then
        echo "Removing existing venv (--force)"
        rm -rf "${VENV_DIR}"
    elif [ "$KEEP_EXISTING" = true ]; then
        echo "Keeping existing venv (--keep); skipping package install."
        exit 0
    else
        read -p "Venv exists at ${VENV_DIR}. Delete and recreate? [y/N] " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rm -rf "${VENV_DIR}"
        else
            echo "Aborted. Use --keep to skip install, --force to recreate."
            exit 0
        fi
    fi
fi

if [ ! -d "${VENV_DIR}" ]; then
    echo "Creating venv ..."
    ${PYTHON_BIN} -m venv "${VENV_DIR}"
fi

source "${VENV_DIR}/bin/activate"
pip install --upgrade pip wheel setuptools

# ---------- PyTorch + numpy together ----------
# Install numpy at the same time as torch — otherwise torch's import path
# logs a noisy "Failed to initialize NumPy" UserWarning on first import.
# cu128 index is required because our flash-attn wheel is built for cu12.8 +
# torch 2.8.0.
echo
echo "============================================="
echo " Installing PyTorch 2.8.0 + CUDA 12.8 + numpy"
echo "============================================="
pip install "torch==2.8.0" "numpy<3.0" \
    --index-url https://download.pytorch.org/whl/cu128 \
    --extra-index-url https://pypi.org/simple

# Verify torch import. CUDA availability is only asserted on GPU nodes; on
# login nodes torch will still import but cuda.is_available() == False.
python - <<'EOF'
import torch, numpy
assert torch.version.cuda and torch.version.cuda.startswith('12.8'), \
    f"Expected torch built against CUDA 12.8, got {torch.version.cuda!r}"
print(f"  torch:   {torch.__version__}")
print(f"  numpy:   {numpy.__version__}")
print(f"  CUDA build: {torch.version.cuda}")
print(f"  CUDA runtime available: {torch.cuda.is_available()}")
EOF

# ---------- Core libs ----------
# Pinned to the combo verified on 2×8 H200 for this pipeline.
# transformers     — model loading + lr scheduler
# datasets         — Arrow dataset IO
# safetensors      — used explicitly in the HF-format exporter
# huggingface_hub  — snapshot_download for the 144 GB model + Glaive auth
# accelerate       — transformers internals may touch it on some paths
# sentencepiece + protobuf — Qwen2.5 tokenizer backend
# einops           — flash-attn runtime dep
echo
echo "============================================="
echo " Installing ML libs"
echo "============================================="
pip install \
    "transformers==5.5.0" \
    "datasets>=4.0.0,<5.0.0" \
    "safetensors>=0.7.0" \
    "huggingface_hub>=1.0.0" \
    "accelerate>=1.0.0" \
    "sentencepiece>=0.2.0" \
    "protobuf>=5.0" \
    "einops>=0.8.0"

# ---------- Flash Attention 2 ----------
# Prebuilt wheel for torch 2.8.0 + cu12.8 + Python 3.12.
# If this URL 404s, find the equivalent at:
#   https://github.com/mjun0812/flash-attention-prebuild-wheels/releases
echo
echo "============================================="
echo " Installing FlashAttention 2 (prebuilt wheel)"
echo "============================================="
FA_WHEEL="https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.16/flash_attn-2.7.4+cu128torch2.8-cp312-cp312-linux_x86_64.whl"
pip install "$FA_WHEEL"

# ---------- Runtime verification (GPU-gated) ----------
# Use torch.cuda.is_available() as the authoritative check. nvidia-smi can
# return exit 0 even without GPUs, and /dev/nvidia* devices may exist on
# nodes that can't actually run CUDA workloads.
HAS_GPU=$(python -c "import torch; print('yes' if torch.cuda.is_available() else 'no')")

if [ "$HAS_GPU" = "yes" ]; then
    echo
    echo "============================================="
    echo " GPU runtime verification"
    echo "============================================="
    python - <<'EOF'
import torch
from flash_attn import flash_attn_func
q = torch.randn(1, 16, 8, 64, device='cuda', dtype=torch.bfloat16)
out = flash_attn_func(q, q, q, causal=True)
assert out.shape == q.shape
print(f"  Device:                 {torch.cuda.get_device_name(0)}")
print(f"  flash_attn.__version__: {__import__('flash_attn').__version__}")
print(f"  flash_attn forward:     OK (shape {tuple(out.shape)})")
EOF
else
    echo
    echo "  Skipping runtime verification — torch.cuda.is_available() is False."
    echo "  To verify end-to-end, submit: sbatch setup_environment.sbatch"
fi

# ---------- Workspace scaffolding ----------
WORK_DIR_ROOT="${WORK_DIR_ROOT:-/mnt/data/qwen_finetune}"
mkdir -p \
    "${WORK_DIR_ROOT}/logs" \
    "${WORK_DIR_ROOT}/datasets" \
    "${WORK_DIR_ROOT}/checkpoints" \
    "${WORK_DIR_ROOT}/hf_cache"

# ---------- Summary ----------
echo
echo "============================================="
echo " Setup complete"
echo "============================================="
echo "Venv:      ${VENV_DIR}"
echo "Workspace: ${WORK_DIR_ROOT}"
echo
echo "Activate:  source ${VENV_DIR}/bin/activate"
echo
echo "Installed packages:"
pip list 2>/dev/null | grep -iE "^(torch|numpy|transformers|datasets|safetensors|huggingface|accelerate|flash|sentencepiece|einops)"
echo
if [ "$HAS_GPU" = "no" ]; then
    echo "Note: runtime verification was SKIPPED (torch.cuda.is_available()==False)."
    echo "      Run 'sbatch setup_environment.sbatch' to verify on a GPU node."
fi
echo
echo "Next step:"
echo "  export HF_TOKEN=hf_xxx   # or: huggingface-cli login"
echo "  ./prepare_data.sh"
