# GreesyGuard Ultra — finetuning package

A heavier sibling of `greesyguard-3-mini-thinking`, trained on the same
`<|system|>/<|user|>/<|assistant|><think>...</think>verdict` chat format
and reasoning-mode idea, scaled up for more headroom.

| | Mini (published) | Ultra-small | Ultra (default) |
|---|---|---|---|
| layers | 12 | 24 | 32 |
| heads (q / kv) | 12 / 12 | 12 / 4 (GQA) | 20 / 5 (GQA) |
| d_model | 768 | 1536 | 2560 |
| context | 12,000 | 24,576 | 32,768 |
| params (approx) | ~120M | ~760M | ~2.7B |

Run `python config.py` to print exact approximate parameter counts for
any config you tweak.

## What's different from Mini, architecturally

- **Grouped-query attention** instead of full MHA — keeps the KV-cache
  affordable at this width and context length.
- **SwiGLU MLP** instead of a plain GELU MLP.
- **RMSNorm** instead of LayerNorm.
- **RoPE** with a higher `theta` to support the longer context window.
- An extra **EXTREME** reasoning-mode tier (12k think-tokens), since
  Ultra's context window can actually afford it.

Everything else — the special tokens, the moderation label set
(`SAFE / SPAM / MISINFORMATION / HARASSMENT / HATE_SPEECH /
CRISIS_REFERRAL / UNSAFE`), and the assistant-only loss masking — is
unchanged from Mini's card.

## Files

- `config.py` — `GreesyGPTConfig` dataclass + `MINI_CONFIG` /
  `ULTRA_SMALL_CONFIG` / `ULTRA_CONFIG`.
- `model.py` — `GreesyGPTUltra` (RoPE + GQA + SwiGLU decoder, KV-cache
  generation), `ReasoningMode`, `OutputFormat`, `generate_moderation()`.
- `tokenizer.py` — wrapper that loads the real GreesyGuard
  `tokenizer.json` (required for real training — see below), with a
  `tiktoken` o200k_base fallback for smoke-testing the pipeline only.
- `data.py` — renders records into the training-format string and
  masks everything except the assistant span with `label = -100`.
  Supports two record shapes: the moderation-verdict format
  (`verdict`/`label`) and the label-free reasoning+answer format
  (`detailed_answer`) produced by `make_dataset.py` — whichever field
  is present is used as the assistant's final output.
- `make_dataset.py` — converts a moderation-style dataset
  (`id`/`label`/`instruction`/`reasoning`/`output`/`complexity`/`source`)
  into a label-free dataset with only `{id, instruction, reasoning,
  detailed_answer}` — no verdict, label, output, complexity, or source
  fields at all. `detailed_answer` is synthesized from the existing
  `reasoning` text's Harm Potential / Edge Cases content, rephrased as
  a natural-language answer instead of a `## Verdict` stamp. It's a
  deterministic text transform, not a model call — swap
  `build_detailed_answer()` for an LLM call if you want richer,
  non-derivative answers.
- `inference.py` — CLI for running a finetuned checkpoint on the
  label-free format: prints `<think>` reasoning + a detailed answer
  (no verdict parsing). Supports a single `--prompt` or batch
  `--input_file`/`--output_file` over JSON/JSONL.
- `finetune.py` — the training loop: AMP (bf16/fp16), gradient
  checkpointing, gradient accumulation, cosine LR schedule, DDP for
  multi-GPU, periodic eval + checkpointing, resume support.

## Before you run this for real

You need the **actual GreesyGuard `tokenizer.json`** (the "o200k_base
extended", 8192-vocab tokenizer that Mini uses) — grab it from the
GreesyGuard repo/HF repo. This code can't guess that vocabulary for
you; the `tiktoken` fallback in `tokenizer.py` is explicitly not
usable for real training (wrong vocab size, so it can't produce a
model compatible with the published checkpoints).

## Usage

Smoke-test the pipeline end-to-end (tiny fallback tokenizer, tiny model, CPU-friendly):

```bash
python finetune.py --smoke_test --config ultra_small \
  --hf_dataset OnlyCheeini/greesyguard-3-mini-claude-4.6-sonnet-2000x \
  --output_dir ./ckpt-smoke --epochs 1 --batch_size 1 --grad_accum_steps 2 \
  --eval_every 5 --save_every 5 --log_every 1
```

Real single-GPU finetune:

```bash
python finetune.py \
  --tokenizer_json /path/to/tokenizer.json \
  --hf_dataset OnlyCheeini/greesyguard-3-mini-claude-4.6-sonnet-2000x \
  --output_dir ./ckpt-ultra \
  --batch_size 1 --grad_accum_steps 32 --lr 1e-5 --epochs 2
```

Multi-GPU (DDP), e.g. 4 GPUs on one node:

```bash
torchrun --nproc_per_node=4 finetune.py \
  --tokenizer_json /path/to/tokenizer.json \
  --hf_dataset OnlyCheeini/greesyguard-3-mini-claude-4.6-sonnet-2000x \
  --output_dir ./ckpt-ultra \
  --batch_size 2 --grad_accum_steps 16
```

If the full `ultra` config doesn't fit your GPU(s), use `--config
ultra_small` — same code path, ~4x Mini instead of ~23x.

### From your own data instead of the HF dataset

Point at a local JSONL file. Two record shapes are supported:

- moderation format: `prompt`/`message`/`instruction`, `think`/`reasoning`,
  `verdict`/`label`
- label-free reasoning+answer format: `instruction`, `reasoning`,
  `detailed_answer` (this is what `make_dataset.py` produces)

```bash
python finetune.py --tokenizer_json /path/to/tokenizer.json \
  --jsonl_path ./my_dataset.jsonl --output_dir ./ckpt-ultra
```

### Building a label-free reasoning+answer dataset

To turn a moderation-style dataset (like `moderation_dataset.json`)
into one with only `{id, instruction, reasoning, detailed_answer}` —
no label, output, complexity, or source fields:

```bash
python make_dataset.py --input moderation_dataset.json \
  --output reasoning_answer_dataset.jsonl --also_json
```

Then finetune directly on it with `--jsonl_path
reasoning_answer_dataset.jsonl` above.

## Inference after training

CLI, single prompt:

```bash
python inference.py --checkpoint ckpt-ultra/final.pt \
  --tokenizer_json /path/to/tokenizer.json \
  --prompt "Please help me with gardening advice for apartment herbs." \
  --mode MEDIUM
```

prints the `<think>` reasoning followed by the detailed answer. Batch
mode reads instructions from a JSON/JSONL file and writes
`{id, instruction, reasoning, detailed_answer}` results:

```bash
python inference.py --checkpoint ckpt-ultra/final.pt \
  --tokenizer_json /path/to/tokenizer.json \
  --input_file reasoning_answer_dataset.jsonl \
  --output_file predictions.jsonl --mode MEDIUM
```

Or from Python directly:

```python
from model import GreesyGPTUltra
from tokenizer import GreesyTokenizer
from config import ULTRA_CONFIG
from inference import run_inference
from model import ReasoningMode
import torch

tok = GreesyTokenizer(tokenizer_json_path="/path/to/tokenizer.json")
model = GreesyGPTUltra(ULTRA_CONFIG)
model.load_state_dict(torch.load("ckpt-ultra/final.pt")["model"])
model.eval()

result = run_inference(
    model, tok, "Please help me with gardening advice for apartment herbs.",
    system="You are a careful reasoning assistant. Think step by step inside "
           "a <think> block, then give a detailed, direct answer.",
    mode=ReasoningMode.MEDIUM, max_new_tokens=None,
    temperature=0.7, top_p=0.9, device="cuda",
)
print(result["reasoning"])
print(result["detailed_answer"])
```

(The original moderation-verdict `generate_moderation()` in
`model.py` still works too, if you're training on verdict-format
data instead.)

## Notes

- Ultra is not weight-compatible with Mini — you can't warm-start Ultra
  from the Mini checkpoint (`--init_checkpoint` expects an Ultra-shaped
  checkpoint, e.g. from a prior Ultra pretraining/finetuning run).
- A moderation model's job is to route human review, not to replace it
  — keep human oversight in the loop for high-impact actions, same
  guidance as on the Mini model card.
