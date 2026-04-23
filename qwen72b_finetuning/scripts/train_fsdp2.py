#!/usr/bin/env python3
"""
Full fine-tuning of Qwen2.5-72B-Instruct with FSDP2 HSDP on 2 nodes × 8 H200.

Expects to be launched via torchrun (see launch.sbatch). On each rank:
  * Builds a 2D device mesh: shard within node (8), replicate across nodes (2).
  * Loads Qwen2.5-72B-Instruct in bf16 via HF transformers.
  * Applies FSDP2 fully_shard to each Qwen2 decoder layer, then the whole model.
  * Wraps each decoder layer with non-reentrant activation checkpointing.
  * Uses FlashAttention-2 varlen via position_ids resets (no custom kernels needed).
  * AdamW, cosine schedule with 3% warmup, bf16 throughout, fp32 grad reduction.
  * Saves weights-only sharded DCP checkpoints (overwrites one rolling slot).
  * At end, gathers the full model to rank 0 and writes HF-format safetensors.

Input: packed Arrow dataset produced by tokenize_pack.py, with fields
  input_ids, labels, position_ids, seq_lens (all length = seq_len).
The seq_lens field is metadata only; position_id resets drive varlen attention.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Optional

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    CheckpointImpl,
)
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    set_model_state_dict,
    StateDictOptions,
)
from torch.optim import AdamW
from torch.utils.data import DataLoader, DistributedSampler

from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    get_cosine_schedule_with_warmup,
)


# ============================================================================
# Distributed + logging setup
# ============================================================================

def setup_distributed() -> tuple[int, int, int, torch.device]:
    """Init process group from torchrun env vars. Returns (rank, world, local, device)."""
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    dist.init_process_group(backend="nccl", device_id=device)
    return rank, world_size, local_rank, device


def setup_logging(log_dir: Path, rank: int) -> logging.Logger:
    """Rank 0 logs to stdout + file; other ranks stay silent."""
    log = logging.getLogger("train")
    log.setLevel(logging.INFO if rank == 0 else logging.ERROR)
    if rank == 0:
        log_dir.mkdir(parents=True, exist_ok=True)
        fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                                datefmt="%H:%M:%S")
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        log.addHandler(sh)
        fh = logging.FileHandler(log_dir / "train.log")
        fh.setFormatter(fmt)
        log.addHandler(fh)
    return log


def log0(log, msg, *args):
    """Log only from rank 0."""
    if dist.get_rank() == 0:
        log.info(msg, *args)


# ============================================================================
# Model construction + FSDP2 sharding
# ============================================================================

def build_and_shard_model(
    model_name_or_path: str,
    mesh,
    device: torch.device,
    log,
) -> torch.nn.Module:
    """Load Qwen2.5-72B in bf16 and apply FSDP2 HSDP + activation checkpointing.

    model_name_or_path is expected to be a local filesystem path (resolved by
    the sbatch launcher). We pass it directly to from_pretrained so worker
    ranks never call into huggingface_hub resolution logic — on compute nodes
    without internet the Hub calls hang or fail even with HF_HUB_OFFLINE set.
    """
    log0(log, "Loading from %s in bf16 ...", model_name_or_path)
    t0 = time.time()

    # Each rank loads the full model into CPU RAM (bf16 -> ~144 GB/rank).
    # With 8 ranks per node, that's ~1.15 TB per node — assumed to fit given
    # typical H200 node config. If OOM, reduce to rank-0-loads pattern.
    # Note: transformers 5.x uses `dtype` (was `torch_dtype`).
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        low_cpu_mem_usage=True,
        use_cache=False,
    )
    model.gradient_checkpointing_disable()  # we'll apply it manually, layerwise
    log0(log, "Loaded in %.1fs", time.time() - t0)

    # Disable HF's KV cache for training (saves memory + avoids FSDP interactions)
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    # Mixed precision policy: params+compute in bf16, grad reduction in fp32.
    mp_policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
    )

    # Activation checkpointing on each decoder layer. Non-reentrant is faster
    # and composes with FSDP2.
    decoder_layers = model.model.layers  # Qwen2ModelfordCausalLM.model.layers
    for i, layer in enumerate(decoder_layers):
        wrapped = checkpoint_wrapper(layer, checkpoint_impl=CheckpointImpl.NO_REENTRANT)
        decoder_layers[i] = wrapped
    log0(log, "Applied activation checkpointing to %d decoder layers", len(decoder_layers))

    # FSDP2 fully_shard per layer, then wrap the top-level model.
    # The 2D mesh does HSDP: replicate across the first dim, shard across the second.
    for layer in decoder_layers:
        fully_shard(layer, mesh=mesh, mp_policy=mp_policy)
    fully_shard(model, mesh=mesh, mp_policy=mp_policy)
    log0(log, "Applied FSDP2 HSDP (mesh shape: %s)", tuple(mesh.shape))

    # Move to GPU — FSDP2 materializes sharded params on device here.
    model.to(device)
    return model


# ============================================================================
# Data
# ============================================================================

def collate_packed(batch: list[dict]) -> dict[str, torch.Tensor]:
    """
    Convert a list of packed records into batched tensors. We drop seq_lens
    because FlashAttention-2 derives cu_seqlens from position_id resets.
    """
    return {
        "input_ids":    torch.tensor([b["input_ids"]    for b in batch], dtype=torch.long),
        "labels":       torch.tensor([b["labels"]       for b in batch], dtype=torch.long),
        "position_ids": torch.tensor([b["position_ids"] for b in batch], dtype=torch.long),
    }


def collate_eval(batch: list[dict]) -> dict[str, torch.Tensor]:
    """Eval batches use attention_mask (right-padded, not packed)."""
    return {
        "input_ids":      torch.tensor([b["input_ids"]      for b in batch], dtype=torch.long),
        "labels":         torch.tensor([b["labels"]         for b in batch], dtype=torch.long),
        "attention_mask": torch.tensor([b["attention_mask"] for b in batch], dtype=torch.long),
    }


# ============================================================================
# Checkpointing (DCP sharded, weights-only to fit 1 TB budget)
# ============================================================================

def save_sharded_ckpt(model, ckpt_dir: Path, step: int, log) -> None:
    """
    Save a weights-only sharded checkpoint. Each rank writes its own shard
    in parallel via torch.distributed.checkpoint. Overwrites in place.
    """
    tmp_dir = ckpt_dir.parent / f"{ckpt_dir.name}.tmp"
    if dist.get_rank() == 0:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        tmp_dir.mkdir(parents=True)
    dist.barrier()

    t0 = time.time()
    msd = get_model_state_dict(
        model,
        options=StateDictOptions(full_state_dict=False, cpu_offload=False),
    )
    dcp.save({"model": msd, "step": step}, checkpoint_id=str(tmp_dir))
    dist.barrier()

    if dist.get_rank() == 0:
        if ckpt_dir.exists():
            shutil.rmtree(ckpt_dir)
        tmp_dir.rename(ckpt_dir)
        log.info("Checkpoint saved to %s in %.1fs", ckpt_dir, time.time() - t0)
    dist.barrier()


def load_sharded_ckpt(model, ckpt_dir: Path, log) -> int:
    """Load a weights-only sharded checkpoint. Returns the saved step."""
    log0(log, "Resuming from %s ...", ckpt_dir)
    msd = get_model_state_dict(
        model,
        options=StateDictOptions(full_state_dict=False, cpu_offload=False),
    )
    state = {"model": msd, "step": 0}
    dcp.load(state, checkpoint_id=str(ckpt_dir))
    set_model_state_dict(
        model, state["model"],
        options=StateDictOptions(full_state_dict=False, cpu_offload=False),
    )
    return int(state["step"])


def export_final_hf(model, model_name_or_path: str, out_dir: Path, log) -> None:
    """
    Gather the full model to rank 0, save as sharded safetensors, and copy
    all non-weight files (config, tokenizer, chat template, generation config)
    from the local snapshot so the output is a drop-in from_pretrained-loadable
    checkpoint.

    model_name_or_path is a local filesystem path (resolved by the sbatch),
    not a Hub repo id — we never touch the network here.
    """
    log0(log, "Gathering full model to rank 0 for HF export ...")
    t0 = time.time()
    full_sd = get_model_state_dict(
        model,
        options=StateDictOptions(full_state_dict=True, cpu_offload=True,
                                 broadcast_from_rank0=False),
    )
    if dist.get_rank() == 0:
        out_dir.mkdir(parents=True, exist_ok=True)

        # --- Sharded safetensors for weights ---
        from safetensors.torch import save_file
        SHARD_BYTES = 5 * 1024**3  # 5 GiB per shard
        shards: list[dict[str, torch.Tensor]] = [{}]
        shard_bytes = 0
        index = {"metadata": {"total_size": 0}, "weight_map": {}}
        for name, tensor in full_sd.items():
            nbytes = tensor.numel() * tensor.element_size()
            if shard_bytes + nbytes > SHARD_BYTES and shards[-1]:
                shards.append({})
                shard_bytes = 0
            shards[-1][name] = tensor.contiguous()
            shard_bytes += nbytes
            index["metadata"]["total_size"] += nbytes
        for i, shard in enumerate(shards):
            fname = f"model-{i+1:05d}-of-{len(shards):05d}.safetensors"
            save_file(shard, str(out_dir / fname), metadata={"format": "pt"})
            for name in shard:
                index["weight_map"][name] = fname
        with open(out_dir / "model.safetensors.index.json", "w") as f:
            json.dump(index, f, indent=2)

        # --- Copy config + tokenizer + chat template from the local snapshot ---
        # model_name_or_path is already a resolved local path (the sbatch
        # resolved it), so we iterate it directly instead of calling back into
        # huggingface_hub.snapshot_download (which would try to reach the Hub).
        snapshot_path = Path(model_name_or_path)
        skip_suffixes = (".safetensors", ".bin", ".pt", ".pth", ".msgpack",
                         ".h5", ".ot", ".onnx")
        copied = []
        for p in snapshot_path.iterdir():
            if not p.is_file():
                continue
            if p.name.endswith(skip_suffixes):
                continue
            if p.name.startswith("."):
                continue
            # shutil.copy follows symlinks (HF cache uses symlinks into blobs)
            shutil.copy(p, out_dir / p.name)
            copied.append(p.name)
        log.info("Copied from cache: %s", sorted(copied))
        log.info("Final model exported to %s in %.1fs", out_dir, time.time() - t0)
    dist.barrier()


# ============================================================================
# Eval loop
# ============================================================================

@torch.no_grad()
def run_eval(model, eval_loader, device, log) -> float:
    """Average masked-token cross-entropy over the eval set."""
    model.eval()
    total_loss = torch.zeros(1, device=device)
    total_tokens = torch.zeros(1, device=device)
    for batch in eval_loader:
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        out = model(**batch)
        # HF returns mean loss over non-ignored tokens in the batch;
        # re-weight by token count for a correct global average.
        n = (batch["labels"] != -100).sum()
        total_loss += out.loss * n
        total_tokens += n
    dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_tokens, op=dist.ReduceOp.SUM)
    model.train()
    return (total_loss / total_tokens.clamp(min=1)).item()


# ============================================================================
# Main
# ============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-name", default="Qwen/Qwen2.5-72B-Instruct")
    ap.add_argument("--data-dir", type=Path, required=True,
                    help="Contains train/ and eval/ subdirs from tokenize_pack.py.")
    ap.add_argument("--ckpt-dir", type=Path, required=True)
    ap.add_argument("--log-dir", type=Path, required=True)
    ap.add_argument("--final-dir", type=Path, default=None,
                    help="Where to save the final HF-format model (defaults to ckpt-dir/final).")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--global-batch-size", type=int, default=128,
                    help="Total sequences per optimizer step across all ranks.")
    ap.add_argument("--micro-batch-size", type=int, default=1,
                    help="Sequences per GPU per forward.")
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--warmup-ratio", type=float, default=0.03)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--eval-every", type=int, default=50, help="Steps between eval runs.")
    ap.add_argument("--save-every", type=int, default=100, help="Steps between ckpt saves.")
    ap.add_argument("--log-every", type=int, default=5)
    ap.add_argument("--resume", action="store_true",
                    help="Resume from ckpt-dir/rolling if it exists.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # ----- Distributed init -----
    rank, world_size, local_rank, device = setup_distributed()
    log = setup_logging(args.log_dir, rank)
    torch.manual_seed(args.seed + rank)

    if world_size != 16:
        log0(log, "WARNING: script assumed 2×8=16 ranks, got world_size=%d", world_size)

    # 2D mesh: (replicate=num_nodes, shard=gpus_per_node)
    gpus_per_node = int(os.environ.get("LOCAL_WORLD_SIZE", 8))
    num_nodes = world_size // gpus_per_node
    mesh = init_device_mesh(
        "cuda", (num_nodes, gpus_per_node),
        mesh_dim_names=("replicate", "shard"),
    )
    log0(log, "World=%d, nodes=%d, gpus/node=%d", world_size, num_nodes, gpus_per_node)

    # ----- Tokenizer files will be copied from HF cache at export time -----
    # We intentionally do NOT instantiate a tokenizer here: the training data
    # is already tokenized, and the slow Qwen2 tokenizer has a BPE init bug in
    # transformers 5.x that trips even with use_fast=True. See export_final_hf.

    # ----- Model -----
    model = build_and_shard_model(args.model_name, mesh, device, log)
    model.train()

    # ----- Data -----
    log0(log, "Loading packed dataset from %s", args.data_dir)
    train_ds = load_from_disk(str(args.data_dir / "train"))
    eval_ds = load_from_disk(str(args.data_dir / "eval"))
    train_ds = train_ds.with_format("python")
    eval_ds = eval_ds.with_format("python")
    log0(log, "Train: %d packed seqs, Eval: %d unpacked seqs", len(train_ds), len(eval_ds))

    grad_accum_steps = args.global_batch_size // (args.micro_batch_size * world_size)
    assert args.global_batch_size == args.micro_batch_size * world_size * grad_accum_steps, \
        "global_batch_size must be divisible by micro_batch_size * world_size"
    steps_per_epoch = len(train_ds) // args.global_batch_size
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = max(1, int(args.warmup_ratio * total_steps))
    log0(log, "Schedule: %d steps/epoch × %d epochs = %d total steps (warmup %d, grad_accum %d)",
         steps_per_epoch, args.epochs, total_steps, warmup_steps, grad_accum_steps)

    train_sampler = DistributedSampler(
        train_ds, num_replicas=world_size, rank=rank,
        shuffle=True, seed=args.seed, drop_last=True,
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.micro_batch_size, sampler=train_sampler,
        collate_fn=collate_packed, num_workers=2, pin_memory=True, drop_last=True,
    )
    eval_sampler = DistributedSampler(
        eval_ds, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False,
    )
    eval_loader = DataLoader(
        eval_ds, batch_size=args.micro_batch_size, sampler=eval_sampler,
        collate_fn=collate_eval, num_workers=2, pin_memory=True, drop_last=False,
    )

    # ----- Optimizer + Scheduler -----
    optimizer = AdamW(
        model.parameters(), lr=args.lr,
        betas=(0.9, 0.95), weight_decay=args.weight_decay,
        fused=True,
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps,
    )

    # ----- Resume -----
    start_step = 0
    rolling_ckpt = args.ckpt_dir / "rolling"
    if args.resume and rolling_ckpt.exists():
        start_step = load_sharded_ckpt(model, rolling_ckpt, log)
        log0(log, "Resumed at step %d", start_step)
        # Advance scheduler; optimizer state starts fresh (we didn't save it).
        for _ in range(start_step):
            scheduler.step()

    # ----- Train loop -----
    log0(log, "Starting training ...")
    metrics_path = args.log_dir / "metrics.jsonl"
    metrics_f = open(metrics_path, "a") if rank == 0 else None
    t_start = time.time()
    step = start_step
    tokens_since_log = 0
    loss_since_log = torch.zeros(1, device=device)
    n_since_log = torch.zeros(1, device=device)

    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)
        accum_counter = 0
        for batch in train_loader:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss / grad_accum_steps
            loss.backward()
            # Track trained tokens for throughput metrics
            loss_since_log += out.loss.detach() * (batch["labels"] != -100).sum()
            n_since_log += (batch["labels"] != -100).sum()
            tokens_since_log += batch["input_ids"].numel()
            accum_counter += 1

            if accum_counter % grad_accum_steps == 0:
                # Grad clip — FSDP2 handles cross-shard reduction inside clip_grad_norm_
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1

                if step % args.log_every == 0:
                    dist.all_reduce(loss_since_log, op=dist.ReduceOp.SUM)
                    dist.all_reduce(n_since_log, op=dist.ReduceOp.SUM)
                    avg_loss = (loss_since_log / n_since_log.clamp(min=1)).item()
                    elapsed = time.time() - t_start
                    toks_per_sec = tokens_since_log * world_size / max(1e-3, elapsed) \
                                   if step == start_step + args.log_every else 0
                    # Reset accumulators after first log-window
                    if rank == 0:
                        lr_now = scheduler.get_last_lr()[0]
                        log.info("step %5d/%d  loss=%.4f  lr=%.2e  elapsed=%.0fs",
                                 step, total_steps, avg_loss, lr_now, elapsed)
                        json.dump({"step": step, "loss": avg_loss, "lr": lr_now,
                                   "elapsed_s": elapsed}, metrics_f)
                        metrics_f.write("\n"); metrics_f.flush()
                    loss_since_log.zero_(); n_since_log.zero_(); tokens_since_log = 0

                if step % args.eval_every == 0:
                    eval_loss = run_eval(model, eval_loader, device, log)
                    if rank == 0:
                        log.info("  eval step %d: loss=%.4f", step, eval_loss)
                        json.dump({"step": step, "eval_loss": eval_loss}, metrics_f)
                        metrics_f.write("\n"); metrics_f.flush()

                if step % args.save_every == 0:
                    save_sharded_ckpt(model, rolling_ckpt, step, log)

                if step >= total_steps:
                    break
        if step >= total_steps:
            break

    # ----- Final eval + export -----
    final_eval = run_eval(model, eval_loader, device, log)
    log0(log, "Final eval loss: %.4f", final_eval)

    final_dir = args.final_dir or (args.ckpt_dir / "final")
    export_final_hf(model, args.model_name, final_dir, log)

    if rank == 0 and metrics_f is not None:
        metrics_f.close()

    dist.barrier()
    dist.destroy_process_group()
    log0(log, "Done.")


if __name__ == "__main__":
    main()
