#!/bin/bash
#SBATCH --job-name=qwen72b-fc
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=96
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --output=/mnt/data/qwen_finetune/logs/%j/slurm.out
#SBATCH --error=/mnt/data/qwen_finetune/logs/%j/slurm.err

# ============================================================================
# Qwen2.5-72B-Instruct full fine-tune, 2 nodes × 8 H200, FSDP2 HSDP.
#
# Prerequisites:
#   - setup_environment.sh has been run
#   - prepare_data.sh has been run (creates packed Arrow dataset)
#
# Launch:
#   sbatch launch.sbatch
#
# Resume from last rolling checkpoint:
#   sbatch launch.sbatch --resume
# ============================================================================

set -euo pipefail

# ---------- Paths ----------
# SLURM copies the sbatch script to its own spool dir and runs it from there,
# so ${BASH_SOURCE[0]} points to /var/spool/slurmd/... rather than our project.
# SLURM_SUBMIT_DIR is the directory from which `sbatch` was invoked — exactly
# what we want. The fallback handles direct shell execution (no sbatch).
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
export PROJECT_DIR
export VENV="${VENV_DIR:-${HOME}/llm-finetune}"
export WORK_DIR_ROOT="${WORK_DIR_ROOT:-/mnt/data/qwen_finetune}"

export DATA_DIR="${WORK_DIR_ROOT}/datasets/tokenized"
export CKPT_DIR="${WORK_DIR_ROOT}/checkpoints"
export LOG_ROOT="${WORK_DIR_ROOT}/logs"
export LOG_DIR="${LOG_ROOT}/${SLURM_JOB_ID}"
export HF_HOME="${WORK_DIR_ROOT}/hf_cache"

# Defensive mkdir (SLURM may not auto-create the job subdir for %j paths).
mkdir -p "$CKPT_DIR" "$LOG_DIR" "$HF_HOME"

# ---------- Activate venv ----------
if [ ! -f "$VENV/bin/activate" ]; then
    echo "FATAL: venv not found at $VENV. Run setup_environment.sh first." >&2
    exit 1
fi
source "$VENV/bin/activate"

# ---------- Input sanity check ----------
if [ ! -f "$DATA_DIR/train/dataset_info.json" ]; then
    echo "FATAL: tokenized dataset not found at $DATA_DIR/train." >&2
    echo "       Run prepare_data.sh first." >&2
    exit 1
fi

# ---------- NCCL / comm tuning for H200 + InfiniBand ----------
export NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
# Let NCCL auto-select IB interfaces; override if your cluster needs it:
# export NCCL_IB_HCA=mlx5
# export NCCL_SOCKET_IFNAME=bond0
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800
export CUDA_DEVICE_MAX_CONNECTIONS=1
export OMP_NUM_THREADS=8

# ---------- Rendezvous setup ----------
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500
export WORLD_SIZE=$(( SLURM_NNODES * SLURM_GPUS_PER_NODE ))

echo "=== Job $SLURM_JOB_ID on $(date) ==="
echo "Nodes:       $SLURM_JOB_NODELIST"
echo "NNodes:      $SLURM_NNODES × GPUs/node: $SLURM_GPUS_PER_NODE = $WORLD_SIZE total"
echo "Master:      $MASTER_ADDR:$MASTER_PORT"
echo "Project dir: $PROJECT_DIR"
echo "Work dir:    $WORK_DIR_ROOT"
echo "Data:        $DATA_DIR"
echo "Ckpt:        $CKPT_DIR"
echo "Logs:        $LOG_DIR"
echo

# ---------- Pre-download model weights to shared cache (one-shot) ----------
# Avoid 16 parallel downloads of 145 GB on first run. Verify by looking at the
# weight index + shard count; an interrupted download leaves a directory but
# missing weights, which silently breaks training.
MODEL_DIR=$HF_HOME/hub/models--Qwen--Qwen2.5-72B-Instruct

verify_cache() {
    local index_file
    index_file=$(find "$MODEL_DIR/snapshots" -name "model.safetensors.index.json" 2>/dev/null | head -1)
    if [ -z "$index_file" ] || [ ! -f "$index_file" ]; then
        return 1
    fi
    local expected actual
    expected=$(python -c "import json,sys; d=json.load(open(sys.argv[1])); print(len(set(d['weight_map'].values())))" "$index_file")
    actual=$(ls "$(dirname "$index_file")"/model-*-of-*.safetensors 2>/dev/null | wc -l)
    if [ "$expected" != "$actual" ]; then
        echo "Cache incomplete: expected $expected shards, found $actual"
        return 1
    fi
    echo "Cache complete: $actual safetensors shards + index"
    return 0
}

if ! verify_cache; then
    echo "Downloading Qwen2.5-72B-Instruct to $HF_HOME (this can take 15-30 min) ..."
    python -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen2.5-72B-Instruct', cache_dir='$HF_HOME/hub')
"
    if ! verify_cache; then
        echo "FATAL: cache still incomplete after download. Aborting."
        exit 1
    fi
fi

# Resolve the concrete local snapshot path. Passing this to the training script
# (instead of a Hub repo id) means worker ranks never need network access.
MODEL_LOCAL_PATH=$(find "$MODEL_DIR/snapshots" -mindepth 1 -maxdepth 1 -type d | head -1)
if [ -z "$MODEL_LOCAL_PATH" ]; then
    echo "FATAL: could not locate snapshot dir under $MODEL_DIR/snapshots" >&2
    exit 1
fi
echo "Model local path: $MODEL_LOCAL_PATH"

# ---------- Launch via srun + torchrun on each node ----------
# One srun task per node; torchrun spawns 8 worker processes per node.
srun --label --kill-on-bad-exit=1 bash -c '
set -euo pipefail
source "'"$VENV"'/bin/activate"

# After the pre-download above, the model is fully cached in $HF_HOME. Force
# offline mode so ranks do not issue HEAD requests to the hub.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

export NODE_RANK=$SLURM_NODEID
echo "[node $NODE_RANK / host $(hostname)] launching torchrun ..."

exec torchrun \
    --nnodes=$SLURM_NNODES \
    --nproc-per-node=$SLURM_GPUS_PER_NODE \
    --node-rank=$NODE_RANK \
    --master-addr=$MASTER_ADDR \
    --master-port=$MASTER_PORT \
    '"$PROJECT_DIR"'/scripts/train_fsdp2.py \
        --model-name '"$MODEL_LOCAL_PATH"' \
        --data-dir  '"$DATA_DIR"' \
        --ckpt-dir  '"$CKPT_DIR"' \
        --log-dir   '"$LOG_DIR"' \
        --epochs    2 \
        --global-batch-size 128 \
        --micro-batch-size  1 \
        --lr        5e-6 \
        --save-every 100 \
        --eval-every 50 \
        '"$@"'
'

echo "=== Job $SLURM_JOB_ID done on $(date) ==="
