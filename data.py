"""
Dataset for finetuning GreesyGPT-Ultra on GreesyGuard-style moderation data.

Expects records shaped like the Hugging Face dataset used for Mini,
`OnlyCheeini/greesyguard-3-mini-claude-4.6-sonnet-2000x`, i.e. each
example has (system prompt, user message, <think> reasoning, verdict).
Records can come from a local JSONL file or be pulled from the Hub.

Each record is rendered into the training format from the model card:

    <|system|>
    {system}
    </|system|>

    <|user|>
    {message}
    </|user|>

    <|assistant|>
    <think>
    {reasoning}
    </think>

    {final}<|endoftext|>

`{final}` is whichever of these the record has (checked in order):
`detailed_answer` (reasoning-then-answer data, no moderation label),
`verdict` (Mini-style moderation output), or `label` (bare label
fallback) — so this module works for both the moderation-verdict
format and the label-free reasoning+answer format produced by
`make_dataset.py`.

Only tokens after `<|assistant|>` contribute to the loss — everything
before that (system + user turns, and the `<|assistant|>` tag itself)
is masked out with label id -100.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional

import torch
from torch.utils.data import Dataset

DEFAULT_SYSTEM_PROMPT = (
    "You are GreesyGuard, a content moderation reasoning model. "
    "Analyze the message, reason step by step inside a <think> block, "
    "then give a final moderation verdict."
)

# System prompt to use for the label-free reasoning+answer format
# (records that have `detailed_answer` instead of `verdict`/`label`).
ANSWER_SYSTEM_PROMPT = (
    "You are a careful reasoning assistant. Think step by step inside a "
    "<think> block, then give a detailed, direct answer."
)


def render_example(record: Dict, system_prompt: Optional[str] = None) -> str:
    is_answer_format = "detailed_answer" in record
    default_system = ANSWER_SYSTEM_PROMPT if is_answer_format else DEFAULT_SYSTEM_PROMPT
    system = record.get("system", system_prompt or default_system)

    user_message = (
        record.get("prompt") or record.get("message") or record.get("instruction")
    )
    if user_message is None:
        raise KeyError("record has none of: prompt, message, instruction")

    reasoning = record.get("think", record.get("reasoning", "")).strip()
    final = (
        record.get("detailed_answer")
        or record.get("verdict")
        or record.get("label", "")
    ).strip()

    return (
        f"<|system|>\n{system}\n</|system|>\n\n"
        f"<|user|>\n{user_message}\n</|user|>\n\n"
        f"<|assistant|>\n"
        f"<think>\n{reasoning}\n</think>\n\n"
        f"{final}<|endoftext|>"
    )


def load_jsonl(path: str) -> List[Dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_hf_dataset(name: str, split: str = "train") -> List[Dict]:
    from datasets import load_dataset  # local import so `datasets` is optional
    ds = load_dataset(name, split=split)
    return list(ds)


class ModerationDataset(Dataset):
    """
    Tokenizes each rendered example once and masks every token before the
    `<|assistant|>\\n` marker with -100 so the loss only sees assistant output
    (the <think> block + the verdict), exactly as the Mini card specifies:
    "Only assistant tokens contribute to the training loss."
    """

    def __init__(
        self,
        records: List[Dict],
        tokenizer,
        max_len: int = 32768,
        system_prompt: Optional[str] = None,
    ):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.examples = []

        assistant_marker_ids = tokenizer.encode("<|assistant|>\n")

        for record in records:
            text = render_example(record, system_prompt)
            ids = tokenizer.encode(text)
            if len(ids) > max_len:
                ids = ids[:max_len]
            if len(ids) < 2:
                continue

            split_idx = self._find_subsequence(ids, assistant_marker_ids)
            if split_idx is None:
                # couldn't locate the marker (e.g. truncation) — skip a
                # malformed example rather than silently training on it
                continue
            prompt_len = split_idx + len(assistant_marker_ids)

            labels = list(ids)
            for i in range(min(prompt_len, len(labels))):
                labels[i] = -100

            self.examples.append((ids, labels))

    @staticmethod
    def _find_subsequence(haystack: List[int], needle: List[int]) -> Optional[int]:
        n, m = len(haystack), len(needle)
        for i in range(n - m + 1):
            if haystack[i:i + m] == needle:
                return i
        return None

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ids, labels = self.examples[idx]
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


@dataclass
class PadCollator:
    pad_token_id: int

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        max_len = max(item["input_ids"].shape[0] for item in batch)
        B = len(batch)

        input_ids = torch.full((B, max_len), self.pad_token_id, dtype=torch.long)
        labels = torch.full((B, max_len), -100, dtype=torch.long)
        attn_mask = torch.zeros((B, max_len), dtype=torch.long)

        for i, item in enumerate(batch):
            L = item["input_ids"].shape[0]
            input_ids[i, :L] = item["input_ids"]
            labels[i, :L] = item["labels"]
            attn_mask[i, :L] = 1

        return {"input_ids": input_ids, "labels": labels, "attention_mask": attn_mask}


def build_dataset(
    tokenizer,
    jsonl_path: Optional[str] = None,
    hf_dataset: Optional[str] = None,
    hf_split: str = "train",
    max_len: int = 32768,
    val_fraction: float = 0.02,
    seed: int = 42,
):
    if jsonl_path:
        records = load_jsonl(jsonl_path)
    elif hf_dataset:
        records = load_hf_dataset(hf_dataset, hf_split)
    else:
        raise ValueError("Provide either jsonl_path or hf_dataset")

    rng = random.Random(seed)
    rng.shuffle(records)
    n_val = max(1, int(len(records) * val_fraction))
    val_records, train_records = records[:n_val], records[n_val:]

    train_ds = ModerationDataset(train_records, tokenizer, max_len)
    val_ds = ModerationDataset(val_records, tokenizer, max_len)
    return train_ds, val_ds
