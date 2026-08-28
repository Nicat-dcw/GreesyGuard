"""
Thin tokenizer wrapper for GreesyGPT-Ultra.

The Mini card lists the tokenizer as "o200k_base (extended)" with a
vocab size of 8192 — i.e. a custom BPE trained/derived from OpenAI's
o200k_base merges but restricted to a small 8192-token vocabulary, plus
the GreesyGuard special tokens. That means you need the *actual*
tokenizer.json shipped alongside the checkpoint (from the GreesyGuard
repo/HF repo) — you can't just load raw o200k_base, since its vocab is
~200k tokens, not 8192.

This wrapper is deliberately backend-agnostic:
  - Preferred: a HuggingFace `tokenizers` Tokenizer loaded from a
    `tokenizer.json` file (this is almost certainly what GreesyGuard
    ships, given "Structured chat-template training" + extended
    o200k_base vocab).
  - Fallback: raw `tiktoken` o200k_base, ONLY useful for smoke-testing
    this code end-to-end before you have the real tokenizer file —
    it will NOT match the model's actual vocab_size=8192 and should
    not be used for real training.
"""

from __future__ import annotations

from typing import List, Optional

from model import SPECIAL_TOKENS


class GreesyTokenizer:
    def __init__(self, tokenizer_json_path: Optional[str] = None, use_tiktoken_fallback: bool = False):
        self.backend = None

        if tokenizer_json_path:
            from tokenizers import Tokenizer  # pip install tokenizers
            self._tok = Tokenizer.from_file(tokenizer_json_path)
            self.backend = "hf_tokenizers"
            existing = set(self._tok.get_vocab().keys())
            missing = [t for t in SPECIAL_TOKENS if t not in existing]
            if missing:
                raise ValueError(
                    f"tokenizer.json is missing expected special tokens: {missing}. "
                    "Use the tokenizer.json shipped with the GreesyGuard checkpoint, "
                    "not a generic one."
                )
        elif use_tiktoken_fallback:
            import tiktoken  # pip install tiktoken
            base = tiktoken.get_encoding("o200k_base")
            self._tok = tiktoken.Encoding(
                name="o200k_base_greesy_fallback",
                pat_str=base._pat_str,
                mergeable_ranks=base._mergeable_ranks,
                special_tokens={
                    **base._special_tokens,
                    **{t: base.n_vocab + i for i, t in enumerate(SPECIAL_TOKENS)},
                },
            )
            self.backend = "tiktoken_fallback"
            print(
                "[GreesyTokenizer] WARNING: using the raw o200k_base fallback tokenizer. "
                "This does NOT match the model's real vocab_size=8192 and is only "
                "meant for smoke-testing this code, not for real finetuning runs."
            )
        else:
            raise ValueError(
                "Pass tokenizer_json_path=... to load the real GreesyGuard tokenizer, "
                "or use_tiktoken_fallback=True for a smoke test only."
            )

    def encode(self, text: str, return_tensors: bool = False):
        if self.backend == "hf_tokenizers":
            ids = self._tok.encode(text).ids
        else:
            ids = self._tok.encode(text, allowed_special="all")
        if return_tensors:
            import torch
            return torch.tensor([ids], dtype=torch.long)
        return ids

    def decode(self, ids: List[int]) -> str:
        if self.backend == "hf_tokenizers":
            return self._tok.decode(ids, skip_special_tokens=False)
        return self._tok.decode(ids)

    def token_to_id(self, token: str) -> int:
        if self.backend == "hf_tokenizers":
            tid = self._tok.token_to_id(token)
            if tid is None:
                raise KeyError(f"Unknown special token: {token}")
            return tid
        return self._tok.encode(token, allowed_special="all")[0]

    @property
    def vocab_size(self) -> int:
        if self.backend == "hf_tokenizers":
            return self._tok.get_vocab_size()
        return self._tok.n_vocab
