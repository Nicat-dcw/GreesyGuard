"""
Model configs for GreesyGuard (GreesyGPT).

The published `greesyguard-3-mini-thinking` checkpoint is:
    layers=12, heads=12, d_model=768, context=12_000, vocab=8192

`ULTRA_CONFIG` below scales every dimension of that up substantially
(roughly ~14x parameters) while keeping the same tokenizer/vocab and
chat format, so Ultra can be finetuned on the exact same dataset
format (`<|system|>/<|user|>/<|assistant|><think>...</think>verdict`).

Feel free to dial these down if you don't have the GPU memory for
the full Ultra config — `ULTRA_SMALL_CONFIG` is a more modest middle
ground between Mini and Ultra.
"""

from dataclasses import dataclass


@dataclass
class GreesyGPTConfig:
    name: str = "greesyguard-ultra"

    vocab_size: int = 8192          # o200k_base (extended) — same tokenizer as Mini
    d_model: int = 2560
    n_layers: int = 32
    n_heads: int = 20               # head_dim = d_model / n_heads = 128
    n_kv_heads: int = 5              # grouped-query attention (4:1 GQA ratio)
    ffn_mult: float = 3.5            # SwiGLU hidden size = ffn_mult * d_model
    context_len: int = 32768         # ~2.7x Mini's 12,000 token window
    rope_theta: float = 1_000_000.0  # bumped up to support the longer context
    dropout: float = 0.0
    tie_embeddings: bool = True
    norm_eps: float = 1e-5
    use_gqa: bool = True

    # reasoning-mode token budgets (see model.ReasoningMode) — bigger than
    # Mini's because Ultra has a much larger context window to spend on
    # deliberation before the verdict.
    reasoning_budgets: dict = None

    def __post_init__(self):
        if self.reasoning_budgets is None:
            self.reasoning_budgets = {
                "NONE": 300,
                "LOW": 1024,
                "MEDIUM": 3072,
                "HIGH": 6144,
                "EXTREME": 12288,
            }
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
        assert self.n_heads % self.n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    def approx_param_count(self) -> int:
        """Rough non-embedding + embedding parameter estimate, for sanity checks."""
        d, L = self.d_model, self.n_layers
        ffn_hidden = int(self.ffn_mult * d)
        attn_params = d * d + 2 * (d * (self.n_kv_heads * self.head_dim)) + d * d
        mlp_params = 3 * d * ffn_hidden  # SwiGLU: gate, up, down projections
        per_layer = attn_params + mlp_params + 2 * d  # + two RMSNorm weight vectors
        embed_params = self.vocab_size * d
        total = L * per_layer + embed_params
        if not self.tie_embeddings:
            total += embed_params
        return total


# Reference: the published Mini checkpoint (not for training here, just for comparison/logging)
MINI_CONFIG = GreesyGPTConfig(
    name="greesyguard-3-mini-thinking",
    vocab_size=8192,
    d_model=768,
    n_layers=12,
    n_heads=12,
    n_kv_heads=12,   # Mini card doesn't mention GQA, assume full MHA
    ffn_mult=4.0,
    context_len=12_000,
    use_gqa=False,
    reasoning_budgets={"NONE": 200, "LOW": 512, "MEDIUM": 1536, "HIGH": 3072},
)

# The full heavy config, ~14x Mini's parameter count.
ULTRA_CONFIG = GreesyGPTConfig()

# A lighter "ultra" for single-GPU experimentation (~4x Mini instead of ~14x).
ULTRA_SMALL_CONFIG = GreesyGPTConfig(
    name="greesyguard-ultra-small",
    d_model=1536,
    n_layers=24,
    n_heads=12,
    n_kv_heads=4,
    ffn_mult=3.5,
    context_len=24_576,
)


if __name__ == "__main__":
    for cfg in (MINI_CONFIG, ULTRA_SMALL_CONFIG, ULTRA_CONFIG):
        n = cfg.approx_param_count()
        print(f"{cfg.name:28s} ~{n/1e6:8.1f}M params  "
              f"(d={cfg.d_model}, L={cfg.n_layers}, H={cfg.n_heads}/{cfg.n_kv_heads} kv, "
              f"ctx={cfg.context_len})")
