# Qwen2.5-72B Function-Calling Fine-Tune

Full fine-tuning of `Qwen/Qwen2.5-72B-Instruct` for function calling on
**2 gpu-h200-sxm GPU VMs**, using FSDP2 HSDP with FlashAttention-2.
Dataset: blended + deduped xLAM-60k and Glaive-v2 (~114k examples).

## Directory layout

```
qwen72b_finetuning/
├── README.md                    this file
├── setup_environment.sh         one-time venv + dependency install
├── prepare_data.sh              downloads + cleans + tokenizes dataset
├── launch.sbatch                SLURM training submission
└── scripts/                     internal Python implementation
    ├── prepare_fc_datasets.py
    ├── tokenize_pack.py
    └── train_fsdp2.py
```

Top-level shell/sbatch files are invoked directly. Everything under `scripts/`
is implementation detail.

## Prerequisites

- SLURM cluster with 2 nodes × 8 H200 (or similar HBM-rich GPUs, ≥140 GiB each)
- Python 3.12
- CUDA 12.8 drivers on the compute nodes
- 1 TB free on the shared filesystem mounted at `/mnt/data`
- Hugging Face account with:
  - access to `Salesforce/xlam-function-calling-60k` (gated — accept terms on HF website)
  - access to `glaiveai/glaive-function-calling-v2` (public)

## Three-step run

### 1. Prepare environment

Run the envitonment setup script on the login (no GPU-side verification) or compute node:

```bash
./setup_environment.sh
```

The script creates a Python venv at `~/llm-finetune` containing the core packages:
* torch 2.8.0+cu128
* transformers 5.5.0
* FlashAttention-2 2.7.4

### 2. Prepare data

```bash
export HF_TOKEN=hf_xxx               # or: huggingface-cli login
./prepare_data.sh
```

Downloads xLAM + Glaive, cleans + deduplicates, applies Qwen2.5's chat template
with per-turn loss masking, and packs into 2048-token sequences with FFD bin
packing. Outputs:

- `/mnt/data/qwen_finetune/datasets/function_calling_train.jsonl` (raw cleaned)
- `/mnt/data/qwen_finetune/datasets/tokenized/train/` (packed Arrow)
- `/mnt/data/qwen_finetune/datasets/tokenized/eval/` (500 unpacked holdouts)

Takes ~5 minutes on first run. Idempotent: skips any step whose output
already exists. Delete the output to force re-prep.

### 3. Submit training

```bash
sbatch launch.sbatch
```

First run pre-downloads the 144 GB model to `/mnt/data/qwen_finetune/hf_cache/`,
then trains for 2 epochs.

During training:

- `/mnt/data/qwen_finetune/logs/<jobid>/slurm.out` — stdout
- `/mnt/data/qwen_finetune/logs/<jobid>/slurm.err` — stderr
- `/mnt/data/qwen_finetune/logs/<jobid>/train.log` — structured training log
- `/mnt/data/qwen_finetune/logs/<jobid>/metrics.jsonl` — per-step metrics
- `/mnt/data/qwen_finetune/checkpoints/rolling/` — weights-only DCP shards,
  overwritten every `--save-every` steps (fits 1 TB budget)

At end:

- `/mnt/data/qwen_finetune/checkpoints/final/` — HF-format safetensors +
  config + tokenizer (drop-in `from_pretrained`-loadable)

### Resume from the rolling checkpoint

```bash
sbatch launch.sbatch --resume
```

## Expected profile

| Metric                    | Value                |
|---------------------------|----------------------|
| Total training steps      | 410 (2 × 205)        |
| Global batch size         | 128 sequences        |
| Wall clock (training)     | ~90 min              |
| Wall clock (first run)    | ~130 min (includes model download) |
| Peak GPU memory           | 85-95 GB/GPU         |
| Starting eval loss        | ~0.76 (base model)   |
| Final eval loss           | ~0.45-0.55 (typical) |
| Total GPU-hours           | 24 H200-hours        |

## Customization

Override any default in `launch.sbatch`:

```bash
sbatch launch.sbatch --epochs 3 --lr 3e-6
```

Or set environment variables before submitting:

```bash
VENV_DIR=/opt/venvs/qwen WORK_DIR_ROOT=/scratch/qwen sbatch launch.sbatch
```

## Troubleshooting

### Venv missing packages after setup 

Re-run `./setup_environment.sh --force`
to recreate cleanly. Verify with `pip list | grep -E "torch|flash|transformers"`.

### `flash_attn` ImportError on training start 

The prebuilt wheel URL in
`setup_environment.sh` is pinned to torch 2.8.0 + cu12.8 + Python 3.12. If your
env differs, find the matching wheel at
https://github.com/mjun0812/flash-attention-prebuild-wheels/releases and update
the `FA_WHEEL` variable.

### OOM at first training step

It should not happen with H200 (140 GiB) given our settings. If it does,
reduce `--global-batch-size` to 64 in `launch.sbatch` or drop `--micro-batch-size`
(already at 1).

### Checkpoints fill disk

The rolling ckpt dir overwrites in place (`--save-every 100` keeps only 
the latest). If you see multiple `rolling*` directories accumulating,
a previous run crashed mid-save; delete the stale ones manually.
