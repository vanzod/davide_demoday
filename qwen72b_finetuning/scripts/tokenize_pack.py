#!/usr/bin/env python3
"""
Tokenize + pack the cleaned JSONL into Arrow datasets ready for FSDP2 training.

Outputs to <out-dir>/{train, eval}/ as HuggingFace Dataset directories.

Key behaviors:
  - Applies Qwen2.5's native chat template (no custom formatting).
  - Masks loss on system/user/tool tokens; trains only on assistant turns
    (including <tool_call> emissions and final-text responses).
  - Drops examples longer than --seq-len after tokenization (negligible
    loss at seq_len=2048 per length_stats output).
  - Packs train into fixed-length sequences with first-fit-decreasing.
    Records per-document boundaries (position_ids reset, seq_lens) so the
    training loop can use FlashAttention-2 variable-length attention to
    prevent cross-document attention within a pack.
  - Eval left unpacked (one conversation per record, padded) for clean
    per-example loss monitoring.

Output schema:
  train/
    input_ids    : List[int], length = seq_len
    labels       : List[int], length = seq_len (-100 where masked)
    position_ids : List[int], length = seq_len (resets per doc, zeros in pad)
    seq_lens     : List[int], per-document lengths within the pack (sum <= seq_len)
  eval/
    input_ids      : List[int], length = seq_len (right-padded)
    labels         : List[int], length = seq_len (-100 where masked / padded)
    attention_mask : List[int], length = seq_len (1 for real, 0 for pad)
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase


logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("tokenize_pack")


# ============================================================================
# Per-example tokenization with loss masking
# ============================================================================

IGNORE_INDEX = -100  # standard HF convention for "don't contribute to loss"


def tokenize_example(
    tok: PreTrainedTokenizerBase,
    messages: list[dict],
    tools: Optional[list[dict]],
) -> Optional[tuple[list[int], list[int]]]:
    """
    Tokenize one conversation with per-role loss masking.

    Strategy: incrementally render the conversation prefix message-by-message.
    The tokens added by each new message get labels based on that message's
    role: assistant -> train (label=token_id), everything else -> mask (-100).

    This is O(K^2) in message count per conversation but K is small (<=40),
    and it's robust to the template inserting wrapper tokens (e.g. the
    auto-injected system prompt with tool descriptions) without us having
    to model the template internals.

    Returns (input_ids, labels) or None on render failure.
    """
    try:
        # Tokenize the empty-prefix baseline to subtract any unconditional
        # template boilerplate; with Qwen2.5 this is usually just the system
        # prompt opener, but we want an exact diff.
        # Actually, for Qwen2.5 the template produces nothing meaningful
        # without messages — we just skip the baseline and start from msg 0.
        pass
    except Exception:
        return None

    input_ids: list[int] = []
    labels: list[int] = []
    prev_len = 0

    for i in range(len(messages)):
        try:
            rendered = tok.apply_chat_template(
                messages[: i + 1],
                tools=tools or None,
                tokenize=False,
                add_generation_prompt=False,
            )
        except Exception as e:
            log.debug("template render failed at msg %d: %s", i, e)
            return None

        # add_special_tokens=False because the chat template already wraps with
        # <|im_start|>/<|im_end|> and Qwen2.5 does not use a BOS.
        ids = tok(rendered, add_special_tokens=False).input_ids
        if len(ids) < prev_len:
            # Shouldn't happen, but guard against pathological template behavior
            # where adding a message somehow shortens the tokenization.
            log.warning("non-monotonic tokenization at msg %d; dropping example", i)
            return None
        new_tokens = ids[prev_len:]
        role = messages[i]["role"]
        if role == "assistant":
            # Train on assistant tokens. This includes the <|im_start|>assistant
            # opener and <|im_end|> closer, so the model learns to open and
            # terminate assistant turns correctly.
            labels.extend(new_tokens)
        else:
            # Mask user/system/tool. The tool-response rendering is wrapped as
            # a user turn by Qwen2.5's template (role=tool -> <tool_response>
            # inside <|im_start|>user...<|im_end|>), which gets masked here
            # because our source role is "tool".
            labels.extend([IGNORE_INDEX] * len(new_tokens))

        input_ids.extend(new_tokens)
        prev_len = len(ids)

    if not input_ids:
        return None
    if all(l == IGNORE_INDEX for l in labels):
        # Nothing to train on (no assistant tokens). Drop.
        return None
    return input_ids, labels


# ============================================================================
# First-fit-decreasing packing
# ============================================================================

def pack_ffd(
    items: list[tuple[list[int], list[int]]],
    seq_len: int,
) -> list[list[tuple[list[int], list[int]]]]:
    """
    First-fit-decreasing bin packing. Each bin holds one or more
    (input_ids, labels) pairs whose total length <= seq_len.

    Items have already been filtered for length <= seq_len by the caller.
    """
    # Sort descending by length to reduce fragmentation.
    indexed = sorted(range(len(items)),
                     key=lambda j: -len(items[j][0]))

    bins: list[list[int]] = []           # bins[i] = list of item indices
    bin_remaining: list[int] = []        # bins[i] remaining capacity

    for j in indexed:
        L = len(items[j][0])
        placed = False
        # Scan bins; the natural ordering puts the largest items first, so
        # later smaller items tend to find a fit quickly. O(N * bins) worst
        # case; in practice fast enough for ~100k items.
        for b in range(len(bins)):
            if bin_remaining[b] >= L:
                bins[b].append(j)
                bin_remaining[b] -= L
                placed = True
                break
        if not placed:
            bins.append([j])
            bin_remaining.append(seq_len - L)

    return [[items[j] for j in bin_] for bin_ in bins]


def build_packed_record(
    bin_items: list[tuple[list[int], list[int]]],
    seq_len: int,
    pad_token_id: int,
) -> dict:
    """
    Concatenate items in a bin into a single record of length seq_len.

    position_ids reset to 0 at each document boundary, enabling the training
    loop to emit correct RoPE positions per document.
    seq_lens records per-document lengths (for FlashAttention varlen).
    Padding uses pad_token_id, labels=-100, position_ids=0.
    """
    input_ids: list[int] = []
    labels: list[int] = []
    position_ids: list[int] = []
    seq_lens: list[int] = []

    for ids, lbls in bin_items:
        input_ids.extend(ids)
        labels.extend(lbls)
        position_ids.extend(range(len(ids)))
        seq_lens.append(len(ids))

    # Right-pad to seq_len
    pad = seq_len - len(input_ids)
    assert pad >= 0, f"pack overflow: {len(input_ids)} > {seq_len}"
    if pad:
        input_ids.extend([pad_token_id] * pad)
        labels.extend([IGNORE_INDEX] * pad)
        position_ids.extend([0] * pad)
        # Record padding as a trailing "document" only if nonzero, so the
        # attention impl can ignore it. Some impls prefer not to include it;
        # we include it so sum(seq_lens) == seq_len always.
        seq_lens.append(pad)

    return {
        "input_ids": input_ids,
        "labels": labels,
        "position_ids": position_ids,
        "seq_lens": seq_lens,
    }


def build_eval_record(
    ids: list[int],
    lbls: list[int],
    seq_len: int,
    pad_token_id: int,
) -> dict:
    """One conversation -> one eval record, right-padded with attention_mask."""
    assert len(ids) <= seq_len
    attn = [1] * len(ids) + [0] * (seq_len - len(ids))
    ids = ids + [pad_token_id] * (seq_len - len(ids))
    lbls = lbls + [IGNORE_INDEX] * (seq_len - len(lbls))
    return {"input_ids": ids, "labels": lbls, "attention_mask": attn}


# ============================================================================
# Main
# ============================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jsonl", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--tokenizer", default="Qwen/Qwen2.5-72B-Instruct",
                    help="HF id or local path. Tokenizer is identical across Qwen2.5 sizes.")
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--eval-size", type=int, default=500,
                    help="Examples held out for eval monitoring.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=8,
                    help="num_proc for parallel tokenization.")
    args = ap.parse_args()

    random.seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # --- Load tokenizer ---
    log.info("Loading tokenizer: %s", args.tokenizer)
    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
        log.info("pad_token_id unset; using eos_token_id=%d", tok.pad_token_id)

    # --- Load JSONL ---
    log.info("Loading %s", args.jsonl)
    with open(args.jsonl) as f:
        rows = [json.loads(line) for line in f]
    log.info("Loaded %d rows", len(rows))
    random.shuffle(rows)

    # --- Train / eval split ---
    eval_rows = rows[: args.eval_size]
    train_rows = rows[args.eval_size :]
    log.info("Split: %d train, %d eval", len(train_rows), len(eval_rows))

    # --- Tokenize (parallel via Dataset.map) ---
    # PyArrow can't infer a consistent schema over messages/tools because their
    # nested dicts (tool parameters, tool_call arguments) vary row-to-row.
    # Workaround: pass them through as opaque JSON strings, parse inside the mapper.
    def _tok_map(batch):
        out_ids, out_lbls = [], []
        for msgs_json, tools_json in zip(batch["_messages_json"], batch["_tools_json"]):
            messages = json.loads(msgs_json)
            tools = json.loads(tools_json) if tools_json else None
            r = tokenize_example(tok, messages, tools)
            if r is None:
                out_ids.append(None); out_lbls.append(None)
            else:
                out_ids.append(r[0]); out_lbls.append(r[1])
        return {"input_ids": out_ids, "labels": out_lbls}

    def _tokenize_split(split_rows, name):
        # Serialize nested fields to strings; PyArrow handles strings trivially.
        serialized = [
            {
                "_messages_json": json.dumps(r["messages"], ensure_ascii=False),
                "_tools_json": json.dumps(r.get("tools") or [], ensure_ascii=False),
            }
            for r in split_rows
        ]
        ds = Dataset.from_list(serialized)
        log.info("[%s] tokenizing %d examples with %d workers ...",
                 name, len(ds), args.workers)
        ds = ds.map(_tok_map, batched=True, batch_size=64,
                    num_proc=args.workers,
                    remove_columns=ds.column_names)
        # Drop render failures
        before = len(ds)
        ds = ds.filter(lambda r: r["input_ids"] is not None,
                       num_proc=args.workers)
        dropped_failed = before - len(ds)
        # Drop over-length
        before = len(ds)
        ds = ds.filter(lambda r: len(r["input_ids"]) <= args.seq_len,
                       num_proc=args.workers)
        dropped_long = before - len(ds)
        log.info("[%s] dropped %d render failures, %d over-length (>%d)",
                 name, dropped_failed, dropped_long, args.seq_len)
        return ds

    train_tok = _tokenize_split(train_rows, "train")
    eval_tok = _tokenize_split(eval_rows, "eval")

    # --- Pack train ---
    log.info("Packing train with first-fit-decreasing into seq_len=%d ...",
             args.seq_len)
    train_items = [(r["input_ids"], r["labels"]) for r in train_tok]
    train_lengths = np.array([len(x[0]) for x in train_items])
    log.info("  input token count: %s", f"{int(train_lengths.sum()):,}")
    log.info("  per-example p50/p95/max: %d / %d / %d",
             int(np.percentile(train_lengths, 50)),
             int(np.percentile(train_lengths, 95)),
             int(train_lengths.max()))

    bins = pack_ffd(train_items, args.seq_len)
    used = sum(sum(len(ids) for ids, _ in b) for b in bins)
    total_slots = len(bins) * args.seq_len
    log.info("  packed into %d sequences; density %.1f%% (%.2f docs/seq avg)",
             len(bins), 100 * used / total_slots,
             sum(len(b) for b in bins) / len(bins))

    train_packed = [build_packed_record(b, args.seq_len, tok.pad_token_id)
                    for b in bins]
    train_ds = Dataset.from_list(train_packed)

    # --- Build eval ---
    eval_built = [
        build_eval_record(r["input_ids"], r["labels"],
                          args.seq_len, tok.pad_token_id)
        for r in eval_tok
    ]
    eval_ds = Dataset.from_list(eval_built)

    # --- Save ---
    train_path = args.out_dir / "train"
    eval_path = args.out_dir / "eval"
    log.info("Saving train -> %s", train_path)
    train_ds.save_to_disk(str(train_path))
    log.info("Saving eval -> %s", eval_path)
    eval_ds.save_to_disk(str(eval_path))

    # --- Summary ---
    log.info("=" * 60)
    log.info("DONE")
    log.info("  train packed sequences: %d (%d tokens total)",
             len(train_ds), len(train_ds) * args.seq_len)
    log.info("  eval unpacked records:  %d", len(eval_ds))
    log.info("  seq_len:                %d", args.seq_len)
    log.info("  pad_token_id:           %d", tok.pad_token_id)


if __name__ == "__main__":
    main()
