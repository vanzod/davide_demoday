#!/usr/bin/env python3
"""
Dataset preparation for fine-tuning Qwen2.5-72B-Instruct on function calling.

Downloads Salesforce/xlam-function-calling-60k and glaiveai/glaive-function-calling-v2
from Hugging Face, cleans them, normalizes to OpenAI-style messages format (native
to Qwen2.5's chat template), deduplicates, and writes a single JSONL.

Output schema (one JSON object per line):
    {
        "source": "xlam" | "glaive",
        "tools": [
            {
                "type": "function",
                "function": {"name": str, "description": str, "parameters": dict}
            },
            ...
        ],
        "messages": [
            {"role": "system"|"user"|"assistant"|"tool", "content": str,
             "tool_calls": [{"type": "function", "function": {"name": str, "arguments": str}}, ...]  # assistant only, optional
            },
            ...
        ]
    }

Usage:
    pip install "datasets>=2.18"    # huggingface_hub is a transitive dep

    # Authentication (any one of the following; CLI not required):
    #   a) export HF_TOKEN=hf_xxx           (auto-detected by load_dataset)
    #   b) python -c "from huggingface_hub import login; login(token='hf_xxx')"
    #   c) pass --hf-token hf_xxx or --hf-token-file /path/to/token.txt
    # Gated datasets (Glaive) also require accepting terms on the HF website once.

    python prepare_fc_datasets.py --output-dir /path/to/data

    # Quick smoke test with small samples:
    python prepare_fc_datasets.py --output-dir ./data --max-xlam 500 --max-glaive 500
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import logging
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Optional

try:
    from datasets import load_dataset
except ImportError:
    sys.exit("ERROR: pip install 'datasets>=2.18'")


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("prepare_fc")


# Filtering thresholds — drop obvious garbage but keep the pipeline liberal.
MIN_USER_CHARS = 5
MAX_USER_CHARS = 8000
MIN_TOOLS = 1
MAX_TOOLS = 32
MAX_MESSAGES = 40


# ============================================================================
# xLAM processing
# ============================================================================

def _parse_permissive_dict(text: str) -> Optional[dict]:
    """
    Parse a dict that might be JSON or a Python-literal dict.

    Glaive serializes function calls as Python-literal dicts, often with
    single-quoted string values (e.g. the `arguments` field is single-quoted
    because its contents contain double quotes). `json.loads` rejects those;
    `ast.literal_eval` accepts them safely (literals only, no code execution).
    """
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass
    try:
        obj = ast.literal_eval(text)
        return obj if isinstance(obj, dict) else None
    except (ValueError, SyntaxError, MemoryError, TypeError):
        return None


def _normalize_tool_spec(raw: dict) -> Optional[dict]:
    """Coerce a tool dict to OpenAI tool-spec format."""
    if not isinstance(raw, dict):
        return None
    # xLAM tools look like {"name", "description", "parameters"} at the top level.
    # Glaive tools have the same shape. Already-wrapped {"type":"function","function":{...}}
    # is passed through unchanged.
    if raw.get("type") == "function" and isinstance(raw.get("function"), dict):
        fn = raw["function"]
    else:
        fn = raw
    if "name" not in fn or not isinstance(fn["name"], str):
        return None
    return {
        "type": "function",
        "function": {
            "name": fn["name"],
            "description": fn.get("description", "") or "",
            "parameters": fn.get("parameters", {}) or {},
        },
    }


def _normalize_tool_call(raw: dict) -> Optional[dict]:
    """Coerce a tool-call dict to OpenAI chat-completions format."""
    if not isinstance(raw, dict) or "name" not in raw:
        return None
    args = raw.get("arguments", {})
    # Arguments can be a dict (xLAM) or a JSON string (Glaive). Normalize to JSON string.
    if isinstance(args, dict):
        args_str = json.dumps(args, ensure_ascii=False)
    elif isinstance(args, str):
        # Validate it parses, but keep the string form.
        try:
            json.loads(args)
            args_str = args
        except json.JSONDecodeError:
            return None
    else:
        return None
    return {
        "type": "function",
        "function": {"name": raw["name"], "arguments": args_str},
    }


def process_xlam_row(row: dict) -> Optional[dict]:
    """
    xLAM schema:
        query:   str  - the user question
        tools:   str  - JSON list of tool specs
        answers: str  - JSON list of tool calls [{name, arguments}, ...]
    """
    query = row.get("query")
    if not query or not (MIN_USER_CHARS <= len(query) <= MAX_USER_CHARS):
        return None
    try:
        tools_raw = json.loads(row["tools"])
        answers_raw = json.loads(row["answers"])
    except (json.JSONDecodeError, KeyError, TypeError):
        return None
    if not isinstance(tools_raw, list) or not isinstance(answers_raw, list):
        return None
    if not (MIN_TOOLS <= len(tools_raw) <= MAX_TOOLS) or not answers_raw:
        return None

    tools = [t for t in (_normalize_tool_spec(x) for x in tools_raw) if t]
    if len(tools) != len(tools_raw):  # any tool failed to normalize
        return None

    tool_calls = [c for c in (_normalize_tool_call(x) for x in answers_raw) if c]
    if len(tool_calls) != len(answers_raw):
        return None

    return {
        "source": "xlam",
        "tools": tools,
        "messages": [
            {"role": "user", "content": query},
            {"role": "assistant", "content": "", "tool_calls": tool_calls},
        ],
    }


# ============================================================================
# Glaive processing (messier — needs real cleaning)
# ============================================================================

# Glaive's system prompt embeds function definitions as one or more JSON objects.
# Example:
#   "You are a helpful assistant, with access to the following functions. Use them
#    if required - {\n    \"name\": \"get_weather\", ... } {\n   \"name\": ..."
#
# We extract all top-level JSON objects by brace-matching (regex can't handle
# nested braces reliably).

def _extract_json_objects(text: str) -> list[dict]:
    """Return all top-level JSON objects found in `text`."""
    objects: list[dict] = []
    depth = 0
    start = -1
    in_string = False
    escape = False
    for i, ch in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    candidate = text[start : i + 1]
                    obj = _parse_permissive_dict(candidate)
                    if obj is not None:
                        objects.append(obj)
                    start = -1
    return objects


def extract_glaive_tools(system_text: str) -> list[dict]:
    """Pull function definitions out of a Glaive system prompt."""
    if not system_text:
        return []
    tools = []
    for obj in _extract_json_objects(system_text):
        if "name" in obj:
            spec = _normalize_tool_spec(obj)
            if spec:
                tools.append(spec)
    return tools


# Role markers in Glaive's `chat` string. The canonical format is:
#   USER: <text>
#
#
#   ASSISTANT: <text or functioncall> <|endoftext|>
#
#
#   FUNCTION RESPONSE: <json>
#
#
#   ASSISTANT: <text> <|endoftext|>
#
# Role boundaries are prefixed by either start-of-string or newline(s).
_ROLE_SPLIT_RE = re.compile(
    r"(?:^|\n)\s*(USER|ASSISTANT|FUNCTION RESPONSE):\s*",
    re.MULTILINE,
)

# Marker for function calls inside an ASSISTANT turn. The JSON body after the
# marker can contain nested braces (e.g. `"arguments": "{\"x\": 1}"`) so a
# non-greedy regex won't work — we reuse the brace-matching helper below.
_FUNCCALL_MARKER = "<functioncall>"


def _find_funccall(content: str) -> Optional[tuple[int, int, dict]]:
    """
    Locate a <functioncall> in an assistant turn and return
    (marker_start, json_end_exclusive, parsed_json) or None.
    """
    idx = content.find(_FUNCCALL_MARKER)
    if idx < 0:
        return None
    # Find the first `{` after the marker and match braces (string-aware).
    scan_start = idx + len(_FUNCCALL_MARKER)
    brace_start = content.find("{", scan_start)
    if brace_start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(brace_start, len(content)):
        ch = content[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                obj = _parse_permissive_dict(content[brace_start : i + 1])
                if obj is None:
                    return None
                return (idx, i + 1, obj)
    return None


def _split_glaive_chat(chat: str) -> list[tuple[str, str]]:
    """Split Glaive's chat blob into [(role, content), ...]."""
    chat = chat.replace("<|endoftext|>", "").strip()
    if not chat:
        return []
    # re.split with a capturing group returns [pre, role1, content1, role2, content2, ...]
    parts = _ROLE_SPLIT_RE.split(chat)
    # Drop any leading preamble before the first role.
    if parts and not parts[0].strip().startswith(("USER", "ASSISTANT", "FUNCTION RESPONSE")):
        parts = parts[1:]
    turns: list[tuple[str, str]] = []
    for i in range(0, len(parts) - 1, 2):
        role = parts[i].strip()
        content = parts[i + 1].strip()
        if role and content:
            turns.append((role, content))
    return turns


def parse_glaive_chat(chat: str) -> Optional[list[dict]]:
    """Parse Glaive's chat string into OpenAI-style messages."""
    turns = _split_glaive_chat(chat)
    if not turns:
        return None

    messages: list[dict] = []
    for role_raw, content in turns:
        if role_raw == "USER":
            if not (MIN_USER_CHARS <= len(content) <= MAX_USER_CHARS):
                return None
            messages.append({"role": "user", "content": content})

        elif role_raw == "ASSISTANT":
            fc = _find_funccall(content)
            if fc is not None:
                marker_start, _, call_obj = fc
                text_before = content[:marker_start].strip()
                tc = _normalize_tool_call(call_obj)
                if tc is None:
                    return None
                messages.append({
                    "role": "assistant",
                    "content": text_before,
                    "tool_calls": [tc],
                })
            elif _FUNCCALL_MARKER in content:
                # Marker present but couldn't parse => corrupt row, drop it.
                return None
            else:
                messages.append({"role": "assistant", "content": content})

        elif role_raw == "FUNCTION RESPONSE":
            messages.append({"role": "tool", "content": content})

    return messages or None


def process_glaive_row(row: dict) -> Optional[dict]:
    system = row.get("system", "") or ""
    chat = row.get("chat", "") or ""

    tools = extract_glaive_tools(system)
    if not (MIN_TOOLS <= len(tools) <= MAX_TOOLS):
        return None

    messages = parse_glaive_chat(chat)
    if not messages or len(messages) > MAX_MESSAGES:
        return None

    # Must contain at least one user turn and one assistant turn. Rows where
    # the assistant correctly REFUSES to call a tool (because none applies) are
    # valuable training signal for relevance/irrelevance detection and are kept.
    roles = [m["role"] for m in messages]
    if "user" not in roles or "assistant" not in roles:
        return None

    # Validate that every referenced tool was declared in the system prompt.
    declared_names = {t["function"]["name"] for t in tools}
    for m in messages:
        for tc in m.get("tool_calls") or []:
            if tc["function"]["name"] not in declared_names:
                return None

    return {"source": "glaive", "tools": tools, "messages": messages}


# ============================================================================
# Deduplication
# ============================================================================

def dedup_key(example: dict) -> str:
    """
    Hash the full conversation trajectory + declared tools.

    The earlier version hashed only the first user message + tool names, which
    collapsed genuinely distinct Glaive conversations that shared a common
    opening turn but diverged in clarification flow, argument values, or
    assistant summaries. That threw away ~72% of Glaive.

    This version hashes every message's role, content, and any tool calls
    (name + arguments), so only byte-identical conversations collide.
    """
    parts: list[str] = []
    for m in example["messages"]:
        role = m["role"]
        content = (m.get("content") or "").strip().lower()
        call_sigs: list[str] = []
        for tc in m.get("tool_calls") or []:
            fn = tc["function"]
            call_sigs.append(f"{fn['name']}({fn['arguments']})")
        parts.append(f"{role}::{content}::{'|'.join(call_sigs)}")
    tool_names = sorted(t["function"]["name"] for t in example.get("tools", []))
    payload = "\n".join(parts) + "##" + ",".join(tool_names)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


# ============================================================================
# Main
# ============================================================================

def _process_dataset(
    name: str,
    hf_id: str,
    processor,
    cache_dir: Path,
    stats: Counter,
    max_rows: Optional[int],
) -> list[dict]:
    log.info("Downloading %s ...", hf_id)
    ds = load_dataset(hf_id, cache_dir=str(cache_dir), split="train")
    if max_rows is not None:
        ds = ds.select(range(min(max_rows, len(ds))))
    log.info("  %s raw rows: %d", name, len(ds))

    kept: list[dict] = []
    for row in ds:
        stats[f"{name}_total"] += 1
        result = processor(row)
        if result is None:
            stats[f"{name}_dropped"] += 1
            continue
        kept.append(result)
        stats[f"{name}_kept"] += 1
    log.info("  %s kept: %d / %d (%.1f%%)",
             name, len(kept), stats[f"{name}_total"],
             100 * len(kept) / max(stats[f"{name}_total"], 1))
    return kept


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Directory for HF cache + cleaned JSONL + stats.")
    parser.add_argument("--xlam-only", action="store_true")
    parser.add_argument("--glaive-only", action="store_true")
    parser.add_argument("--max-xlam", type=int, default=None,
                        help="Cap xLAM rows for smoke testing.")
    parser.add_argument("--max-glaive", type=int, default=None,
                        help="Cap Glaive rows for smoke testing.")
    parser.add_argument("--output-name", type=str, default="function_calling_train.jsonl")
    parser.add_argument(
        "--hf-token", type=str, default=None,
        help="HF token literal. Overrides HF_TOKEN env var. "
             "Prefer --hf-token-file or HF_TOKEN for security.",
    )
    parser.add_argument(
        "--hf-token-file", type=Path, default=None,
        help="Path to a file containing an HF token on a single line.",
    )
    args = parser.parse_args()

    if args.xlam_only and args.glaive_only:
        sys.exit("ERROR: --xlam-only and --glaive-only are mutually exclusive")

    # Resolve HF token. load_dataset will auto-detect HF_TOKEN from env, but
    # if the caller passed a token explicitly we surface it via the env var
    # (which huggingface_hub reads on every request).
    token = args.hf_token
    if token is None and args.hf_token_file is not None:
        token = args.hf_token_file.read_text().strip()
    if token:
        import os
        os.environ["HF_TOKEN"] = token
        log.info("HF token set from %s",
                 "--hf-token" if args.hf_token else "--hf-token-file")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.output_dir / "hf_cache"
    cache_dir.mkdir(exist_ok=True)

    stats: Counter = Counter()
    all_examples: list[dict] = []

    if not args.glaive_only:
        all_examples.extend(_process_dataset(
            "xlam", "Salesforce/xlam-function-calling-60k",
            process_xlam_row, cache_dir, stats, args.max_xlam,
        ))

    if not args.xlam_only:
        all_examples.extend(_process_dataset(
            "glaive", "glaiveai/glaive-function-calling-v2",
            process_glaive_row, cache_dir, stats, args.max_glaive,
        ))

    # Deduplicate
    log.info("Deduplicating %d examples ...", len(all_examples))
    seen: set[str] = set()
    deduped: list[dict] = []
    for ex in all_examples:
        k = dedup_key(ex)
        if k in seen:
            stats["duplicates_dropped"] += 1
            continue
        seen.add(k)
        deduped.append(ex)
    log.info("  After dedup: %d (removed %d duplicates)",
             len(deduped), stats["duplicates_dropped"])

    # Write JSONL
    out_path = args.output_dir / args.output_name
    log.info("Writing %s ...", out_path)
    with open(out_path, "w", encoding="utf-8") as f:
        for ex in deduped:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    # Stats
    stats["final_examples"] = len(deduped)
    stats["by_source_xlam"] = sum(1 for e in deduped if e["source"] == "xlam")
    stats["by_source_glaive"] = sum(1 for e in deduped if e["source"] == "glaive")
    stats_path = args.output_dir / "dataset_stats.json"
    with open(stats_path, "w") as f:
        json.dump(dict(stats), f, indent=2, sort_keys=True)

    log.info("=" * 60)
    log.info("DONE")
    for k, v in sorted(stats.items()):
        log.info("  %-28s %d", k, v)
    log.info("Output: %s  (%.1f MB)",
             out_path, out_path.stat().st_size / 1e6)


if __name__ == "__main__":
    main()
