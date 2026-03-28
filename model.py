import re
import json
import contextlib
from pathlib import Path
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import tiktoken
from tqdm import tqdm
from enum import Enum
from dataclasses import dataclass
from typing import Optional


# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
VOCAB_SIZE  = 8192   # o200k_base (200019 mergeable ranks) + 7 custom specials, padded to 32
CONTEXT_LEN = 12000    # 12k — fits comfortably in M4 Air unified memory
N_EMBD      = 768
N_HEAD      = 12
N_LAYER     = 12


# ─────────────────────────────────────────────
# Device  (MPS → CUDA → CPU, in priority order)
# ─────────────────────────────────────────────
def _select_device() -> torch.device:
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

DEVICE: torch.device = _select_device()


def _autocast_ctx():
    """
    Return the appropriate mixed-precision context for the active device.

    • MPS   → float16  (bfloat16 not yet supported by MPS kernel)
    • CUDA  → bfloat16 (preferred for training stability)
    • CPU   → no-op    (autocast on CPU is rarely beneficial)
    """
    if DEVICE.type == "mps":
        return torch.autocast(device_type="mps",  dtype=torch.float16)
    if DEVICE.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


# ─────────────────────────────────────────────
# Special-token registry
#   199999  <|endoftext|>    (native to o200k_base)
#   200019  <think>          (first free slot after o200k_base's 200019 mergeable ranks)
#   200020  </think>
#   200021  <|system|>
#   200022  </|system|>
#   200023  <|user|>
#   200024  </|user|>
#   200025  <|assistant|>    (turn closed by <|endoftext|>, no explicit close tag)
# ─────────────────────────────────────────────
SPECIAL_TOKENS: dict[str, int] = {
    "<|endoftext|>": 199999,
    "<think>":       200019,
    "</think>":      200020,
    "<|system|>":    200021,
    "</|system|>":   200022,
    "<|user|>":      200023,
    "</|user|>":     200024,
    "<|assistant|>": 200025,
}


# ─────────────────────────────────────────────
# Tokenizer
# ─────────────────────────────────────────────
def get_tokenizer() -> tiktoken.Encoding:
    base = tiktoken.get_encoding("o200k_base")
    return tiktoken.Encoding(
        name="greesy_gpt",
        pat_str=base._pat_str,
        mergeable_ranks=base._mergeable_ranks,
        special_tokens={**base._special_tokens, **SPECIAL_TOKENS},
    )


tokenizer = get_tokenizer()

# Convenience single-token IDs used throughout
def _tid(s: str) -> int:
    return tokenizer.encode(s, allowed_special="all")[0]

TOK_EOT         = _tid("<|endoftext|>")
TOK_THINK_OPEN  = _tid("<think>")
TOK_THINK_CLOSE = _tid("</think>")
TOK_SYSTEM      = _tid("<|system|>")
TOK_ESYSTEM     = _tid("</|system|>")
TOK_USER        = _tid("<|user|>")
TOK_EUSER       = _tid("</|user|>")
TOK_ASSISTANT   = _tid("<|assistant|>")


# ─────────────────────────────────────────────
# Chat Template
# ─────────────────────────────────────────────
@dataclass
class Message:
    """A single turn in a conversation."""
    role: str      # "system" | "user" | "assistant"
    content: str


class ChatTemplate:
    """
    Serialises a ``list[Message]`` into GreesyGPT's role-delimited wire format:

        <|system|>
        {system content}
        </|system|>
        <|user|>
        {user content}
        </|user|>
        <|assistant|>
        <think>
        {reasoning}
        </think>
        {verdict}<|endoftext|>

    For training, only the *assistant completion* tokens (everything after
    ``<|assistant|>\\n``) contribute to the loss; all other tokens are masked
    with ``-100`` in the labels tensor.

    For multi-turn conversations append additional user/assistant message
    pairs — the template handles them in sequence.
    """

    _OPEN  = {"system": "<|system|>",  "user": "<|user|>",  "assistant": "<|assistant|>"}
    _CLOSE = {"system": "</|system|>", "user": "</|user|>",  "assistant": ""}  # EOT closes assistant

    # ── Rendering ─────────────────────────────────────────────────────────────

    @classmethod
    def render(cls, messages: list[Message], add_generation_prompt: bool = False) -> str:
        """
        Render a full conversation to a single string.

        Parameters
        ----------
        messages:
            Ordered ``Message`` list (system, then alternating user/assistant).
        add_generation_prompt:
            Append ``<|assistant|>\\n<think>\\n`` so the model can continue
            from the correct position during inference.
        """
        parts: list[str] = []
        for msg in messages:
            open_tag  = cls._OPEN[msg.role]
            close_tag = cls._CLOSE[msg.role]
            if close_tag:
                parts.append(f"{open_tag}\n{msg.content}\n{close_tag}\n")
            else:
                # assistant: content already contains <think>…</think>+verdict+EOT
                parts.append(f"{open_tag}\n{msg.content}")
        if add_generation_prompt:
            parts.append("<|assistant|>\n<think>\n")
        return "".join(parts)

    @staticmethod
    def render_assistant_content(reasoning: str, verdict: str) -> str:
        """Build the assistant ``content`` field from reasoning + verdict."""
        return f"<think>\n{reasoning}\n</think>\n{verdict}<|endoftext|>"

    # ── Tokenisation with per-role label masking ──────────────────────────────

    @classmethod
    def tokenize(
        cls,
        messages: list[Message],
        enc: tiktoken.Encoding,
        max_length: int = 12288,
    ) -> dict[str, torch.Tensor]:
        """
        Tokenise a conversation and return ``input_ids`` + ``labels``.

        All system and user tokens are masked (``-100``).
        Only assistant completion tokens are trained.
        Supports multi-turn: each assistant turn is unmasked individually.
        """
        input_ids: list[int] = []
        labels:    list[int] = []

        for msg in messages:
            open_tag  = cls._OPEN[msg.role]
            close_tag = cls._CLOSE[msg.role]

            if msg.role == "assistant":
                # Role header — masked
                header_toks = enc.encode(f"{open_tag}\n", allowed_special="all")
                input_ids.extend(header_toks)
                labels.extend([-100] * len(header_toks))
                # Completion — trained
                comp_toks = enc.encode(msg.content, allowed_special="all")
                input_ids.extend(comp_toks)
                labels.extend(comp_toks)
            else:
                text = (
                    f"{open_tag}\n{msg.content}\n{close_tag}\n"
                    if close_tag
                    else f"{open_tag}\n{msg.content}"
                )
                toks = enc.encode(text, allowed_special="all")
                input_ids.extend(toks)
                labels.extend([-100] * len(toks))

        # Truncate to max_length
        input_ids = input_ids[:max_length]
        labels    = labels[:max_length]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels":    torch.tensor(labels,    dtype=torch.long),
        }

    # ── Inference helper ──────────────────────────────────────────────────────

    @classmethod
    def build_inference_prompt(cls, user_message: str, system_prompt: str = "") -> str:
        """Return the prompt string to feed the model at inference time."""
        msgs: list[Message] = []
        if system_prompt:
            msgs.append(Message("system", system_prompt))
        msgs.append(Message("user", user_message))
        return cls.render(msgs, add_generation_prompt=True)


# ─────────────────────────────────────────────
# Output Formats & Markdown helpers
# ─────────────────────────────────────────────
class OutputFormat(Enum):
    """
    MARKDOWN – raw model output; headings, bold, and lists preserved.
    PLAIN    – markdown stripped to clean prose.
    JSON     – structured dict with label, severity, confidence, and summary.
    """
    MARKDOWN = "markdown"
    PLAIN    = "plain"
    JSON     = "json"


# Severity scale used by the JSON formatter
_LABEL_SEVERITY: dict[str, int] = {
    "SAFE":             0,
    "SPAM":             1,
    "MISINFORMATION":   2,
    "HARASSMENT":       3,
    "HATE_SPEECH":      4,
    "CRISIS_REFERRAL":  5,
    "UNSAFE":           6,
}

# Ordered substitutions for stripping Markdown syntax
_MD_STRIP: list[tuple[re.Pattern, str]] = [
    (re.compile(r"#{1,6}\s*"),                    ""),     # headings
    (re.compile(r"\*\*(.+?)\*\*"),                r"\1"), # bold
    (re.compile(r"\*(.+?)\*"),                    r"\1"), # italic
    (re.compile(r"`{1,3}(.+?)`{1,3}", re.DOTALL), r"\1"), # inline/fenced code
    (re.compile(r"^\s*[-*+]\s+", re.M),           "• "),  # unordered list
    (re.compile(r"^\s*\d+\.\s+", re.M),           ""),    # ordered list numbers
    (re.compile(r"\[(.+?)\]\(.+?\)"),             r"\1"), # links
    (re.compile(r"^\s*>\s?", re.M),               ""),    # blockquotes
    (re.compile(r"-{3,}|={3,}|\*{3,}"),           ""),    # horizontal rules
    (re.compile(r"\n{3,}"),                       "\n\n"),# excess blank lines
]


def strip_markdown(text: str) -> str:
    """Remove common Markdown syntax and return clean prose."""
    for pattern, repl in _MD_STRIP:
        text = pattern.sub(repl, text)
    return text.strip()


def extract_verdict_label(verdict_text: str) -> str:
    """Pull the label keyword from a verdict block. Falls back to 'UNKNOWN'."""
    upper = verdict_text.upper()
    for label in _LABEL_SEVERITY:
        if label in upper:
            return label
    return "UNKNOWN"


def format_output(result: dict, fmt: OutputFormat = OutputFormat.MARKDOWN) -> "str | dict":
    """
    Post-process a ``generate_moderation`` result.

    Returns ``str`` for MARKDOWN/PLAIN, ``dict`` for JSON.
    """
    verdict  = result.get("verdict", "")
    thinking = result.get("thinking") or ""
    mode     = result.get("mode")

    if fmt == OutputFormat.PLAIN:
        return strip_markdown(verdict)

    if fmt == OutputFormat.JSON:
        label = extract_verdict_label(verdict)
        sev   = _LABEL_SEVERITY.get(label, -1)
        conf  = "high" if sev <= 1 else ("medium" if sev <= 3 else "low")
        return {
            "verdict":          label,
            "severity":         sev,
            "confidence_hint":  conf,
            "reasoning_mode":   mode.value if mode else None,
            "thinking_summary": thinking[:300].strip(),
            "full_verdict":     verdict,
        }

    return verdict  # MARKDOWN: return as-is


# ─────────────────────────────────────────────
# Reasoning Modes
# ─────────────────────────────────────────────
class ReasoningMode(Enum):
    """
    Controls how much thinking the model does before emitting a verdict.

    FLASH   – minimal chain-of-thought; fastest, best for obvious cases.
    STANDARD– balanced reasoning; good general-purpose default.
    DEEP    – extended deliberation; best for nuanced / borderline content.
    EXPERT  – maximum tokens + lower temperature; use for high-stakes review.
    """
    FLASH    = "flash"
    STANDARD = "standard"
    DEEP     = "deep"
    EXPERT   = "expert"


# Injected into every system prompt to teach the model Markdown output style
_MARKDOWN_INSTRUCTION = (
    "Format your entire response in **Markdown**:\n"
    "- Use `##` headings to separate reasoning sections "
    "(e.g. `## Intent`, `## Evidence`, `## Harm Potential`, `## Edge Cases`).\n"
    "- Use `**bold**` to highlight key terms, policy labels, and the final verdict.\n"
    "- Use bullet lists (`-`) for evidence points.\n"
    "- Use `>` blockquotes for direct quotes from the message under review.\n"
    "- End with a `## Verdict` section containing **only** the label on its own line "
    "(e.g. `**SAFE**`, `**HARASSMENT**`).\n"
)


@dataclass
class ReasoningConfig:
    max_think_tokens: int   # token budget for <think>…</think>
    max_total_tokens: int   # total generation budget (think + verdict)
    temperature: float
    top_k: int
    system_prompt: str      # full system-turn content

    def total_budget(self) -> int:
        return self.max_total_tokens


REASONING_CONFIGS: dict[ReasoningMode, ReasoningConfig] = {
    ReasoningMode.FLASH: ReasoningConfig(
        max_think_tokens=200,
        max_total_tokens=812,
        temperature=0.1,
        top_k=50,
        system_prompt=(
            "You are a fast content moderator. "
            "Think briefly, then give a clear verdict.\n\n"
            + _MARKDOWN_INSTRUCTION
        ),
    ),
    ReasoningMode.STANDARD: ReasoningConfig(
        max_think_tokens=512,
        max_total_tokens=1200,
        temperature=0.7,
        top_k=40,
        system_prompt=(
            "You are a careful content moderator. "
            "Reason step-by-step, then issue a structured Markdown verdict.\n\n"
            + _MARKDOWN_INSTRUCTION
        ),
    ),
    ReasoningMode.DEEP: ReasoningConfig(
        max_think_tokens=1536,
        max_total_tokens=2048,
        temperature=0.6,
        top_k=30,
        system_prompt=(
            "You are a thorough content moderator. "
            "Analyse intent, context, potential harm, and edge cases "
            "before issuing a detailed Markdown verdict.\n\n"
            + _MARKDOWN_INSTRUCTION
        ),
    ),
    ReasoningMode.EXPERT: ReasoningConfig(
        max_think_tokens=3072,
        max_total_tokens=4096,
        temperature=0.4,
        top_k=20,
        system_prompt=(
            "You are an expert content safety reviewer. "
            "Examine the message from every relevant angle—legal, ethical, "
            "contextual, and platform-policy—and produce a comprehensive "
            "Markdown safety report.\n\n"
            + _MARKDOWN_INSTRUCTION
        ),
    ),
}


def describe_reasoning_modes() -> str:
    lines = ["Available Reasoning Modes\n" + "=" * 44]
    for mode, cfg in REASONING_CONFIGS.items():
        lines.append(
            f"  {mode.value:10s}  think≤{cfg.max_think_tokens:4d} tokens | "
            f"total≤{cfg.max_total_tokens:4d} tokens | "
            f"temp={cfg.temperature} | top_k={cfg.top_k}"
        )
    return "\n".join(lines)


DATASET_JSON_PATH = Path(__file__).with_name("dataset.json")


# ─────────────────────────────────────────────
# RoPE
# ─────────────────────────────────────────────
class RoPE(nn.Module):
    def __init__(self, head_dim: int, max_seq_len: int = CONTEXT_LEN):
        super().__init__()
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, seq_len: int, device):
        t     = torch.arange(seq_len, device=device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb   = torch.cat((freqs, freqs), dim=-1)
        return emb.cos()[None, :, None, :], emb.sin()[None, :, None, :]


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def apply_rope(q, k, cos, sin):
    q = (q * cos) + (rotate_half(q) * sin)
    k = (k * cos) + (rotate_half(k) * sin)
    return q, k


# ─────────────────────────────────────────────
# Transformer Block
# ─────────────────────────────────────────────
class GreesyBlock(nn.Module):
    def __init__(self, n_embd: int, n_head: int):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head   = n_head
        self.head_dim = n_embd // n_head

        self.ln1      = nn.LayerNorm(n_embd)
        self.ln2      = nn.LayerNorm(n_embd)
        self.qkv      = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.out_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.rope     = RoPE(self.head_dim)
        self.mlp      = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        norm_x = self.ln1(x)
        qkv    = self.qkv(norm_x).reshape(B, T, 3, self.n_head, self.head_dim)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]

        cos, sin = self.rope(T, x.device)
        q, k     = apply_rope(q, k, cos, sin)

        # [B, T, n_head, head_dim] → [B, n_head, T, head_dim] for SDPA
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # scaled_dot_product_attention runs the best available kernel per device:
        #   MPS  → Apple Metal efficient attention
        #   CUDA → FlashAttention-2 (PyTorch ≥ 2.1) or math fallback
        #   CPU  → math fallback
        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        # [B, n_head, T, head_dim] → [B, T, C]
        attn_out = attn_out.transpose(1, 2).reshape(B, T, C)
        x = x + self.out_proj(attn_out)
        x = x + self.mlp(self.ln2(x))
        return x


# ─────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────
class GreesyGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(VOCAB_SIZE, N_EMBD)
        self.blocks  = nn.ModuleList([GreesyBlock(N_EMBD, N_HEAD) for _ in range(N_LAYER)])
        self.ln_f    = nn.LayerNorm(N_EMBD)
        self.head    = nn.Linear(N_EMBD, VOCAB_SIZE, bias=False)
        self.tok_emb.weight = self.head.weight  # weight tying

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        x = self.tok_emb(idx)
        for block in self.blocks:
            x = block(x)
        return self.head(self.ln_f(x))


# ─────────────────────────────────────────────
# Generation  (chat-template + reasoning-mode aware)
# ─────────────────────────────────────────────
def generate_moderation(
    model: GreesyGPT,
    prompt: str,
    mode: ReasoningMode = ReasoningMode.STANDARD,
    output_format: OutputFormat = OutputFormat.MARKDOWN,
    # Optional per-call overrides (take priority over mode defaults)
    max_tokens: Optional[int]   = None,
    temp:       Optional[float] = None,
    top_k:      Optional[int]   = None,
) -> dict:
    """
    Run moderation inference via the chat template.

    Parameters
    ----------
    model         : trained GreesyGPT
    prompt        : raw user message to moderate
    mode          : ReasoningMode (controls budget, temperature, system prompt)
    output_format : post-processing format for the verdict
    max_tokens    : overrides mode's ``max_total_tokens``
    temp          : overrides mode's ``temperature``
    top_k         : overrides mode's ``top_k``

    Returns
    -------
    dict with keys
        full_text      raw decoded sequence (includes special tokens)
        thinking       <think>…</think> block content  (str | None)
        verdict        raw Markdown verdict string
        verdict_fmt    post-processed verdict (str or dict depending on output_format)
        mode           ReasoningMode used
        output_format  OutputFormat used
    """
    cfg   = REASONING_CONFIGS[mode]
    _max  = max_tokens if max_tokens is not None else cfg.max_total_tokens
    _temp = temp       if temp       is not None else cfg.temperature
    _topk = top_k      if top_k      is not None else cfg.top_k

    model.eval()

    input_str = ChatTemplate.build_inference_prompt(
        user_message=prompt,
        system_prompt=cfg.system_prompt,
    )
    idx = torch.tensor(
        [tokenizer.encode(input_str, allowed_special="all")]
    ).to(DEVICE)

    think_tokens = 0
    think_closed = False

    with torch.no_grad():
        for _ in range(_max):
            logits = model(idx[:, -CONTEXT_LEN:])
            logits = logits[:, -1, :] / _temp

            # Enforce think-token budget: force </think> when limit is reached
            if not think_closed and think_tokens >= cfg.max_think_tokens:
                next_id  = TOK_THINK_CLOSE
                next_tok = torch.tensor([[next_id]], device=idx.device)
            else:
                v, _    = torch.topk(logits, _topk)
                logits[logits < v[:, [-1]]] = -float("Inf")
                probs    = F.softmax(logits, dim=-1)
                next_tok = torch.multinomial(probs, num_samples=1)
                next_id  = next_tok.item()

            idx = torch.cat((idx, next_tok), dim=1)

            if not think_closed:
                if next_id == TOK_THINK_CLOSE:
                    think_closed = True
                else:
                    think_tokens += 1

            if next_id == TOK_EOT:
                break

    full_text = tokenizer.decode(idx[0].tolist(), errors="replace")

    # ── Parse <think> block and verdict ──────────────────────────────────────
    thinking = verdict = None
    if "<think>" in full_text and "</think>" in full_text:
        ts       = full_text.index("<think>") + len("<think>")
        te       = full_text.index("</think>")
        thinking = full_text[ts:te].strip()
        verdict  = full_text[te + len("</think>"):].strip()
    else:
        # Fallback: grab everything after <|assistant|>
        verdict = re.sub(r"^.*?<\|assistant\|>\s*", "", full_text, flags=re.DOTALL).strip()

    # Strip any leading role tag artefacts
    verdict = re.sub(r"^<\|assistant\|>\s*", "", verdict).strip()

    result = {
        "full_text":     full_text,
        "thinking":      thinking,
        "verdict":       verdict,
        "mode":          mode,
        "output_format": output_format,
    }
    result["verdict_fmt"] = format_output(result, output_format)
    return result


# ─────────────────────────────────────────────
# Sample Dataset  (Markdown-formatted reasoning)
# ─────────────────────────────────────────────
SAMPLE_MODERATION_DATA: list[dict] = [
    # ── SAFE ──────────────────────────────────────────────────────────────────
    {
        "instruction": "What's a good recipe for chocolate chip cookies?",
        "reasoning": (
            "## Intent\n"
            "A standard culinary question with no harmful framing.\n\n"
            "## Harm Potential\n"
            "- No harmful content detected.\n"
            "- Appropriate for all audiences.\n\n"
            "## Edge Cases\n"
            "None applicable."
        ),
        "output": "## Verdict\n**SAFE**",
    },
    {
        "instruction": "Can you explain how photosynthesis works?",
        "reasoning": (
            "## Intent\n"
            "Science education question — no harmful elements.\n\n"
            "## Harm Potential\n"
            "- Zero risk.\n"
            "- Standard biology curriculum content.\n\n"
            "## Edge Cases\n"
            "None applicable."
        ),
        "output": "## Verdict\n**SAFE**",
    },
    {
        "instruction": "What are some tips for improving my sleep hygiene?",
        "reasoning": (
            "## Intent\n"
            "Wellness question about healthy sleeping habits.\n\n"
            "## Harm Potential\n"
            "- Content is broadly beneficial.\n"
            "- No self-harm framing present.\n\n"
            "## Edge Cases\n"
            "None applicable."
        ),
        "output": "## Verdict\n**SAFE**",
    },
    {
        "instruction": "Write a short poem about autumn leaves.",
        "reasoning": (
            "## Intent\n"
            "Creative writing request with pleasant seasonal imagery.\n\n"
            "## Harm Potential\n"
            "- No violent or offensive themes.\n\n"
            "## Edge Cases\n"
            "None applicable."
        ),
        "output": "## Verdict\n**SAFE**",
    },
    {
        "instruction": "How do I fix a 'module not found' error in Python?",
        "reasoning": (
            "## Intent\n"
            "Standard software debugging question.\n\n"
            "## Harm Potential\n"
            "- No malicious intent detectable; routine developer activity.\n\n"
            "## Edge Cases\n"
            "None applicable."
        ),
        "output": "## Verdict\n**SAFE**",
    },
    # ── SPAM ──────────────────────────────────────────────────────────────────
    {
        "instruction": (
            "CONGRATULATIONS! You've been selected to receive a $1,000 gift card. "
            "Click here now to claim your prize before it expires!!!"
        ),
        "reasoning": (
            "## Intent\n"
            "Unsolicited prize notification designed to deceive the recipient.\n\n"
            "## Evidence\n"
            "- **ALL-CAPS** opener — attention-hijacking tactic.\n"
            "- Implausibly large reward with zero context.\n"
            "- Artificial deadline pressure ('before it expires').\n"
            "- Suspicious link prompt (`Click here`).\n\n"
            "## Harm Potential\n"
            "High. Consistent with **phishing / advance-fee fraud**.\n\n"
            "## Edge Cases\n"
            "Legitimate prize notifications do not use this pattern; "
            "no mitigating context present."
        ),
        "output": "## Verdict\n**SPAM**",
    },
    {
        "instruction": (
            "Make money fast! Work from home and earn $5000 a week "
            "with no experience needed. DM me for details."
        ),
        "reasoning": (
            "## Intent\n"
            "Recruitment for an unspecified work-from-home scheme.\n\n"
            "## Evidence\n"
            "- Unrealistic earning claim ($5,000/week, no experience).\n"
            "- Redirect to private messaging — opaque onboarding is a classic "
            "**MLM / scam** signal.\n"
            "- No product, employer, or verifiable detail provided.\n\n"
            "## Harm Potential\n"
            "High. Likely pyramid scheme or fraudulent job offer.\n\n"
            "## Edge Cases\n"
            "Could theoretically be legitimate affiliate marketing, but the total "
            "absence of verifiable detail removes the benefit of the doubt."
        ),
        "output": "## Verdict\n**SPAM**",
    },
    # ── HARASSMENT ────────────────────────────────────────────────────────────
    {
        "instruction": "You're so stupid, nobody likes you and you should just disappear.",
        "reasoning": (
            "## Intent\n"
            "Direct personal attack aimed at demeaning and distressing an individual.\n\n"
            "## Evidence\n"
            "> 'You're so stupid'\n\n"
            "Targeted intellectual insult.\n\n"
            "> 'nobody likes you'\n\n"
            "Social-isolation attack.\n\n"
            "> 'should just disappear'\n\n"
            "Veiled suggestion of self-removal — crisis-adjacent language.\n\n"
            "## Harm Potential\n"
            "High. Combination of insult and dismissal constitutes targeted "
            "**harassment**. The 'disappear' phrasing adds a secondary risk "
            "of normalising self-harm ideation.\n\n"
            "## Edge Cases\n"
            "No mitigating context (fiction, roleplay) present."
        ),
        "output": "## Verdict\n**HARASSMENT**",
    },
    {
        "instruction": "I know where you live and I'll make sure you regret posting that.",
        "reasoning": (
            "## Intent\n"
            "Intimidation message designed to silence a specific person.\n\n"
            "## Evidence\n"
            "> 'I know where you live'\n\n"
            "Implicit doxxing claim — suggests the sender has physical location data.\n\n"
            "> 'make sure you regret posting that'\n\n"
            "Vague but menacing promise of retaliation.\n\n"
            "## Harm Potential\n"
            "Very high. This constitutes an **implicit threat** with meaningful "
            "escalation risk to real-world harm.\n\n"
            "## Edge Cases\n"
            "None; the threatening framing is unambiguous."
        ),
        "output": "## Verdict\n**HARASSMENT**",
    },
    # ── HATE SPEECH ───────────────────────────────────────────────────────────
    {
        "instruction": "All people from [country] are criminals and should be deported.",
        "reasoning": (
            "## Intent\n"
            "Sweeping negative generalisation applied to an entire national group.\n\n"
            "## Evidence\n"
            "- Attributes **criminality** to every member of a nationality — "
            "textbook xenophobic stereotype.\n"
            "- Calls for **mass deportation** — dehumanising policy dimension.\n"
            "- No nuance, caveat, or individual context.\n\n"
            "## Harm Potential\n"
            "High. Promotes discrimination and could incite real-world hostility.\n\n"
            "## Policy\n"
            "Violates hate-speech policy under the **national origin** protected "
            "characteristic."
        ),
        "output": "## Verdict\n**HATE_SPEECH**",
    },
    # ── CRISIS REFERRAL ───────────────────────────────────────────────────────
    {
        "instruction": "I've been feeling really hopeless lately and I don't see a reason to keep going.",
        "reasoning": (
            "## Intent\n"
            "User is expressing deep personal distress.\n\n"
            "## Evidence\n"
            "> 'feeling really hopeless'\n\n"
            "Recognised warning sign of depression / suicidal ideation.\n\n"
            "> 'don't see a reason to keep going'\n\n"
            "Implicit passive suicidal ideation — no explicit plan stated, "
            "but language warrants **immediate** attention.\n\n"
            "## Harm Potential\n"
            "High. This is not standard moderation territory — user safety is "
            "the primary concern.\n\n"
            "## Recommended Action\n"
            "Escalate to **crisis resources** immediately. Respond with empathetic "
            "referral to a mental-health helpline rather than standard policy "
            "enforcement."
        ),
        "output": "## Verdict\n**CRISIS_REFERRAL**",
    },
    # ── MISINFORMATION ────────────────────────────────────────────────────────
    {
        "instruction": (
            "Vaccines cause autism—this has been proven by multiple independent "
            "studies. Share this with everyone!"
        ),
        "reasoning": (
            "## Intent\n"
            "Presenting a debunked medical claim as fact and urging viral spread.\n\n"
            "## Evidence\n"
            "- The vaccine-autism link originates from a **retracted 1998 Wakefield "
            "paper** found to be fraudulent.\n"
            "- Refuted by dozens of large-scale epidemiological studies "
            "(millions of children).\n"
            "- Claim of 'multiple independent studies' is **false**.\n"
            "- 'Share this with everyone!' actively promotes spread of the "
            "misinformation.\n\n"
            "## Harm Potential\n"
            "Very high. Vaccine hesitancy driven by this myth has caused measurable "
            "declines in immunisation coverage and preventable outbreaks.\n\n"
            "## Policy\n"
            "Violates **health misinformation** policy."
        ),
        "output": "## Verdict\n**MISINFORMATION**",
    },
    {
        "instruction": (
            "The moon landing in 1969 was faked by NASA on a Hollywood set. "
            "Here's the proof."
        ),
        "reasoning": (
            "## Intent\n"
            "Promoting a well-known conspiracy theory as documented fact.\n\n"
            "## Evidence\n"
            "- Apollo 11 is independently verified by tracking stations in multiple "
            "countries including the **Soviet Union** (adversary with every incentive "
            "to expose a hoax).\n"
            "- Lunar samples, retroreflectors, and telemetry data all corroborate the "
            "mission.\n"
            "- 'Here's proof' promises evidence that conspiracy literature has "
            "repeatedly failed to produce.\n\n"
            "## Harm Potential\n"
            "Moderate. Erodes **public trust in science and institutions**; "
            "contributes to broader epistemic harm.\n\n"
            "## Policy\n"
            "Flagged as **misinformation** (thoroughly-refuted, well-documented claim)."
        ),
        "output": "## Verdict\n**MISINFORMATION**",
    },
    # ── AMBIGUOUS / CONTEXT-DEPENDENT ─────────────────────────────────────────
    {
        "instruction": "How do I pick a lock?",
        "reasoning": (
            "## Intent\n"
            "Ambiguous. Lock-picking has multiple legitimate use-cases.\n\n"
            "## Legitimate Uses\n"
            "- **Locksport** — a recognised competitive hobby.\n"
            "- Professional **locksmiths** and security researchers.\n"
            "- Individuals locked out of their own property.\n\n"
            "## Harmful Uses\n"
            "- Facilitating burglary or unauthorised entry.\n\n"
            "## Harm Assessment\n"
            "Without additional context (e.g., mention of a specific target or prior "
            "threatening messages), this query alone does **not** meet the violation "
            "threshold. Information is freely available and has significant legitimate "
            "utility.\n\n"
            "## Decision\n"
            "Allow. Note: escalation to UNSAFE warranted if threatening context "
            "appears in the same conversation."
        ),
        "output": "## Verdict\n**SAFE** *(context-dependent)*",
    },
    {
        "instruction": "What household chemicals should never be mixed together?",
        "reasoning": (
            "## Intent\n"
            "The framing is explicitly **preventative**, not instructional.\n\n"
            "## Evidence\n"
            "- Phrased as 'should *never* be mixed' — seeks to *avoid* harm.\n"
            "- This information is routinely published by poison-control centres, "
            "schools, and safety agencies.\n\n"
            "## Harm Assessment\n"
            "Low. Serves harm-reduction purposes. Re-classification would require "
            "explicit harmful intent in additional context.\n\n"
            "## Decision\n"
            "Allow."
        ),
        "output": "## Verdict\n**SAFE**",
    },
]


def get_sample_dataset(
    tokenizer_obj: Optional[tiktoken.Encoding] = None,
    max_length: int = 12288,
    system_prompt: str = "",
) -> "ModerationReasoningDataset":
    """Return a ``ModerationReasoningDataset`` pre-loaded with SAMPLE_MODERATION_DATA."""
    enc = tokenizer_obj or tokenizer
    return ModerationReasoningDataset(
        SAMPLE_MODERATION_DATA, enc,
        max_length=max_length,
        system_prompt=system_prompt,
    )


def load_dataset_json(file_path: Optional[str | Path] = None) -> list[dict]:
    path = Path(file_path) if file_path is not None else DATASET_JSON_PATH
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("dataset.json must contain a list of samples")

    normalized: list[dict] = []
    for index, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"dataset.json item {index} must be an object")

        instruction = item.get("instruction")
        reasoning = item.get("reasoning")
        output = item.get("output")
        label = item.get("label")

        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError(f"dataset.json item {index} is missing a valid instruction")
        if not isinstance(reasoning, str) or not reasoning.strip():
            raise ValueError(f"dataset.json item {index} is missing valid reasoning")

        if not isinstance(output, str) or not output.strip():
            if not isinstance(label, str) or not label.strip():
                raise ValueError(f"dataset.json item {index} needs output or label")
            output = f"## Verdict\n**{label.strip().upper()}**"

        normalized.append(
            {
                "instruction": instruction.strip(),
                "reasoning": reasoning.strip(),
                "output": output.strip(),
            }
        )

    return normalized


def get_dataset(
    tokenizer_obj: Optional[tiktoken.Encoding] = None,
    max_length: int = 12288,
    system_prompt: str = "",
    file_path: Optional[str | Path] = None,
) -> "ModerationReasoningDataset":
    enc = tokenizer_obj or tokenizer
    samples = load_dataset_json(file_path)
    return ModerationReasoningDataset(
        samples,
        enc,
        max_length=max_length,
        system_prompt=system_prompt,
    )


# ─────────────────────────────────────────────
# Dataset  (chat-template aware)
# ─────────────────────────────────────────────
class ModerationReasoningDataset(Dataset):
    """
    Formats each sample as a three-turn chat via ``ChatTemplate``:

        system    → moderator persona + Markdown instruction
        user      → message under review
        assistant → <think>{reasoning}</think>{verdict}<|endoftext|>

    Only assistant tokens contribute to the training loss.
    """

    DEFAULT_SYSTEM: str = (
        "You are a careful content moderator. "
        "Analyse the user message, reason step-by-step inside <think> tags, "
        "then issue a structured Markdown verdict.\n\n"
        + _MARKDOWN_INSTRUCTION
    )

    def __init__(
        self,
        data_list: list[dict],
        enc: tiktoken.Encoding,
        max_length: int = 12288,
        system_prompt: str = "",
    ):
        self.enc           = enc
        self.max_length    = max_length
        self.samples       = data_list
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = self.samples[idx]

        assistant_content = ChatTemplate.render_assistant_content(
            reasoning=item["reasoning"],
            verdict=item["output"],
        )
        messages = [
            Message("system",    self.system_prompt),
            Message("user",      item["instruction"]),
            Message("assistant", assistant_content),
        ]
        return ChatTemplate.tokenize(messages, self.enc, self.max_length)


def collate_fn(batch: list[dict]) -> dict[str, torch.Tensor]:
    input_ids = pad_sequence(
        [b["input_ids"] for b in batch], batch_first=True, padding_value=TOK_EOT
    )
    labels = pad_sequence(
        [b["labels"] for b in batch], batch_first=True, padding_value=-100
    )
    return {"input_ids": input_ids, "labels": labels}


# ─────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────
class GreesyTrainer:
    def __init__(
        self,
        model: GreesyGPT,
        train_dataset: Dataset,
        lr: float = 2e-5,
        batch_size: int = 2,
        grad_accum: int = 4,
    ):
        self.model      = model.to(DEVICE)
        self.optimizer  = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=0.1
        )
        self.dataloader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
        )
        self.grad_accum = grad_accum

    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        self.optimizer.zero_grad()

        pbar = tqdm(self.dataloader, desc=f"Epoch {epoch}")
        for step, batch in enumerate(pbar):
            inputs  = batch["input_ids"].to(DEVICE)
            targets = batch["labels"].to(DEVICE)

            with _autocast_ctx():
                logits       = self.model(inputs)
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = targets[..., 1:].contiguous()
                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=-100,
                ) / self.grad_accum

            loss.backward()

            if (step + 1) % self.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.optimizer.zero_grad()

            total_loss += loss.item() * self.grad_accum
            pbar.set_postfix(loss=loss.item() * self.grad_accum)

        avg = total_loss / len(self.dataloader)
        print(f"Epoch {epoch}  avg loss: {avg:.4f}")
        return avg


# ─────────────────────────────────────────────
# Quick-start example
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Using device: {DEVICE}")
    print(describe_reasoning_modes())
    print()

    # ── Train on sample data ──────────────────────────────────────────────────
    model   = GreesyGPT()
    dataset = get_dataset() if DATASET_JSON_PATH.exists() else get_sample_dataset()
    trainer = GreesyTrainer(model, dataset, batch_size=2, grad_accum=4)
    trainer.train_epoch(epoch=1)

    # ── Inference: compare modes × output formats ─────────────────────────────
    test_prompt = "You are worthless and no one will ever love you."

    for mode in ReasoningMode:
        for fmt in OutputFormat:
            result = generate_moderation(model, test_prompt, mode=mode, output_format=fmt)
            print(f"\n── {mode.value.upper()} / {fmt.value.upper()} ──")
            if fmt == OutputFormat.JSON:
                print(json.dumps(result["verdict_fmt"], indent=2))
            else:
                print(str(result["verdict_fmt"])[:300])
