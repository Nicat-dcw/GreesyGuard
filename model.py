"""
GreesyGPT-Ultra: a heavier decoder-only transformer for the GreesyGuard
content-moderation line, built to be finetuned on the same chat format
as greesyguard-3-mini-thinking:

    <|system|>
    moderation instructions
    </|system|>

    <|user|>
    message to review
    </|user|>

    <|assistant|>
    <think>
    step-by-step reasoning
    </think>

    verdict<|endoftext|>

Architecture deltas vs. Mini (see config.py for exact numbers):
  - Grouped-query attention (GQA) instead of full MHA, to keep KV-cache
    memory sane at Ultra's width/context.
  - SwiGLU MLP instead of a plain GELU MLP (standard "make it heavier and
    it still trains efficiently" upgrade).
  - RMSNorm instead of LayerNorm.
  - RoPE with a higher theta to support the longer context window.
Everything else (causal decoder, KV-cache-friendly forward pass,
<think>-block reasoning, verdict labels) mirrors the Mini model card.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Optional, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import GreesyGPTConfig, ULTRA_CONFIG


# --------------------------------------------------------------------------
# Reasoning modes / output formats (same public surface as the Mini model)
# --------------------------------------------------------------------------

class ReasoningMode(str, Enum):
    NONE = "NONE"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    EXTREME = "EXTREME"   # new tier — only meaningful with Ultra's larger context


class OutputFormat(str, Enum):
    JSON = "json"
    MARKDOWN = "markdown"
    PLAIN = "plain"


MODERATION_LABELS = [
    "SAFE", "SPAM", "MISINFORMATION", "HARASSMENT",
    "HATE_SPEECH", "CRISIS_REFERRAL", "UNSAFE",
]

SPECIAL_TOKENS = [
    "<|system|>", "</|system|>",
    "<|user|>", "</|user|>",
    "<|assistant|>",
    "<think>", "</think>",
    "<|endoftext|>",
]


# --------------------------------------------------------------------------
# RoPE
# --------------------------------------------------------------------------

def precompute_rope(head_dim: int, max_positions: int, theta: float, device, dtype):
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(max_positions, device=device).float()
    freqs = torch.outer(t, inv_freq)                      # [T, head_dim/2]
    cos, sin = freqs.cos(), freqs.sin()
    return cos.to(dtype), sin.to(dtype)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: [B, H, T, D]
    x1, x2 = x[..., ::2], x[..., 1::2]
    cos = cos[None, None, :x.shape[2], :]
    sin = sin[None, None, :x.shape[2], :]
    rotated = torch.empty_like(x)
    rotated[..., ::2] = x1 * cos - x2 * sin
    rotated[..., 1::2] = x1 * sin + x2 * cos
    return rotated


# --------------------------------------------------------------------------
# Building blocks
# --------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm * self.weight


class GQAttention(nn.Module):
    """Grouped-query causal self-attention with RoPE and an optional KV-cache."""

    def __init__(self, cfg: GreesyGPTConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.head_dim
        self.n_rep = cfg.n_heads // cfg.n_kv_heads

        self.q_proj = nn.Linear(cfg.d_model, cfg.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, cfg.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, cfg.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(cfg.n_heads * self.head_dim, cfg.d_model, bias=False)
        self.dropout = cfg.dropout

    @staticmethod
    def _repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
        if n_rep == 1:
            return x
        b, h, t, d = x.shape
        return x[:, :, None, :, :].expand(b, h, n_rep, t, d).reshape(b, h * n_rep, t, d)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ):
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # RoPE is applied using absolute position, so slice cos/sin by cache offset
        offset = past_kv[0].shape[2] if past_kv is not None else 0
        q = apply_rope(q, cos[offset:offset + T], sin[offset:offset + T])
        k = apply_rope(k, cos[offset:offset + T], sin[offset:offset + T])

        if past_kv is not None:
            k = torch.cat([past_kv[0], k], dim=2)
            v = torch.cat([past_kv[1], v], dim=2)
        new_kv = (k, v) if use_cache else None

        k = self._repeat_kv(k, self.n_rep)
        v = self._repeat_kv(v, self.n_rep)

        is_causal = past_kv is None  # single-token decode steps don't need a causal mask
        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal,
        )
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.o_proj(out), new_kv


class SwiGLU(nn.Module):
    def __init__(self, cfg: GreesyGPTConfig):
        super().__init__()
        hidden = int(cfg.ffn_mult * cfg.d_model)
        self.gate_proj = nn.Linear(cfg.d_model, hidden, bias=False)
        self.up_proj = nn.Linear(cfg.d_model, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Block(nn.Module):
    def __init__(self, cfg: GreesyGPTConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = GQAttention(cfg)
        self.mlp_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.mlp = SwiGLU(cfg)

    def forward(self, x, cos, sin, past_kv=None, use_cache=False):
        attn_out, new_kv = self.attn(self.attn_norm(x), cos, sin, past_kv, use_cache)
        x = x + attn_out
        x = x + self.mlp(self.mlp_norm(x))
        return x, new_kv


# --------------------------------------------------------------------------
# Full model
# --------------------------------------------------------------------------

class GreesyGPTUltra(nn.Module):
    def __init__(self, cfg: GreesyGPTConfig = ULTRA_CONFIG):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.final_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight

        self._rope_cache = {}
        self.apply(self._init_weights)
        # scaled residual init, GPT-2 / LLaMA style, keeps deep (32-layer) nets stable
        for name, p in self.named_parameters():
            if name.endswith("o_proj.weight") or name.endswith("down_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layers))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _get_rope(self, device, dtype):
        key = (device, dtype)
        if key not in self._rope_cache:
            self._rope_cache[key] = precompute_rope(
                self.cfg.head_dim, self.cfg.context_len, self.cfg.rope_theta, device, dtype
            )
        return self._rope_cache[key]

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        past_kvs: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
    ):
        B, T = input_ids.shape
        cos, sin = self._get_rope(input_ids.device, self.tok_emb.weight.dtype)

        x = self.tok_emb(input_ids)
        new_kvs = [] if use_cache else None
        for i, block in enumerate(self.blocks):
            past = past_kvs[i] if past_kvs is not None else None
            x, kv = block(x, cos, sin, past, use_cache)
            if use_cache:
                new_kvs.append(kv)
        x = self.final_norm(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
            )
        return logits, loss, new_kvs

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        eos_token_id: int,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> torch.Tensor:
        self.eval()
        past_kvs = None
        generated = input_ids
        step_input = input_ids
        for _ in range(max_new_tokens):
            logits, _, past_kvs = self.forward(step_input, past_kvs=past_kvs, use_cache=True)
            next_logits = logits[:, -1, :] / max(temperature, 1e-5)
            probs = F.softmax(next_logits, dim=-1)

            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cum_probs = torch.cumsum(sorted_probs, dim=-1)
            cutoff = (cum_probs > top_p).float().argmax(dim=-1, keepdim=True)
            mask = torch.arange(sorted_probs.size(-1), device=probs.device)[None, :] > cutoff
            sorted_probs = sorted_probs.masked_fill(mask, 0.0)
            sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
            next_in_sorted = torch.multinomial(sorted_probs, 1)
            next_token = sorted_idx.gather(-1, next_in_sorted)

            generated = torch.cat([generated, next_token], dim=1)
            step_input = next_token
            if (next_token == eos_token_id).all():
                break
        return generated


# --------------------------------------------------------------------------
# High-level moderation API (mirrors the Mini model's `generate_moderation`)
# --------------------------------------------------------------------------

def build_prompt(system: str, user_message: str) -> str:
    return (
        f"<|system|>\n{system}\n</|system|>\n\n"
        f"<|user|>\n{user_message}\n</|user|>\n\n"
        f"<|assistant|>\n"
    )


def generate_moderation(
    model: GreesyGPTUltra,
    tokenizer,
    prompt: str,
    mode: ReasoningMode = ReasoningMode.MEDIUM,
    output_format: OutputFormat = OutputFormat.JSON,
    system: str = "You are GreesyGuard, a content moderation reasoning model.",
) -> dict:
    budget = model.cfg.reasoning_budgets[mode.value]
    full_prompt = build_prompt(system, prompt)
    input_ids = tokenizer.encode(full_prompt, return_tensors=True)

    eos_id = tokenizer.token_to_id("<|endoftext|>")
    out_ids = model.generate(input_ids, max_new_tokens=budget, eos_token_id=eos_id)
    text = tokenizer.decode(out_ids[0, input_ids.shape[1]:].tolist())

    think, verdict = "", text
    if "<think>" in text and "</think>" in text:
        think = text.split("<think>", 1)[1].split("</think>", 1)[0].strip()
        verdict = text.split("</think>", 1)[1].strip()
    verdict = verdict.replace("<|endoftext|>", "").strip()

    label = next((l for l in MODERATION_LABELS if l in verdict), "UNSAFE")

    if output_format == OutputFormat.JSON:
        verdict_fmt = {"verdict": label, "reasoning": think}
    elif output_format == OutputFormat.MARKDOWN:
        verdict_fmt = f"## Verdict\n**{label}**\n\n<details><summary>Reasoning</summary>\n\n{think}\n</details>"
    else:
        verdict_fmt = f"{label}\n\n{think}"

    return {"label": label, "reasoning": think, "raw": text, "verdict_fmt": verdict_fmt}


if __name__ == "__main__":
    cfg = ULTRA_CONFIG
    model = GreesyGPTUltra(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"{cfg.name}: {n_params/1e6:.1f}M parameters")
