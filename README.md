
# GreesyGPT: Reasoning-Based Content Moderation

GreesyGPT is a lightweight, specialized Transformer model designed for high-precision content moderation. Unlike traditional "black-box" classifiers, GreesyGPT utilizes **Chain-of-Thought (CoT)** reasoning to analyze intent, context, and policy violations before issuing a final verdict.

## 🚀 Key Features

*   **Native Reasoning:** Uses `<think>` blocks to deliberate on nuances, edge cases, and harm potential before committing to a label.
*   **Tiered Reasoning Modes:** Four distinct modes (`FLASH`, `STANDARD`, `DEEP`, `EXPERT`) allow you to trade off latency for depth of analysis.
*   **Flexible Output Formats:** Supports structured `JSON` for API integration, clean `PLAIN` text for logs, or `MARKDOWN` for human-readable reports.
*   **Modern Architecture:** 
    *   12-layer, 12-head Transformer (768 hidden dim).
    *   **RoPE** (Rotary Positional Embeddings) for better long-context handling.
    *   **12k Context Window** (optimized for Apple M-series and modern GPUs).
    *   **o200k_base Tokenizer** (the same vocabulary base as GPT-4o).

---

## 🛠️ Architecture Overview

| Parameter | Value |
| :--- | :--- |
| **Layers** | 12 |
| **Heads** | 12 |
| **Embedding Dim** | 768 |
| **Context Length** | 12,000 tokens |
| **Vocabulary Size** | 8,192 (Padded, based on o200k) |
| **Precision** | Autocast (BFloat16 on CUDA / Float16 on MPS) |

---

## 🚦 Quick Start

### 1. Requirements
```bash
pip install torch tiktoken tqdm
```

### 2. Basic Inference
GreesyGPT uses a specific chat template to trigger reasoning.

```python
from model import GreesyGPT, generate_moderation, ReasoningMode, OutputFormat

# Initialize model (ensure you have trained weights or initialize fresh)
model = GreesyGPT()

# Run a 'Deep' moderation check
result = generate_moderation(
    model, 
    prompt="You're so stupid, nobody likes you.",
    mode=ReasoningMode.DEEP,
    output_format=OutputFormat.JSON
)

# Access structured data
print(result["verdict_fmt"]["verdict"])  # e.g., "HARASSMENT"
print(result["thinking"])                # e.g., "The user is using targeted insults..."
```

---

## 🧠 Reasoning Modes

You can adjust the "thinking budget" based on the complexity of the content:

*   **FLASH**: Minimal CoT (128 tokens). High speed. Best for obvious spam.
*   **STANDARD**: Balanced reasoning (512 tokens). The default for general moderation.
*   **DEEP**: Extended deliberation (1.5k tokens). Best for nuanced, borderline cases.
*   **EXPERT**: Maximum token budget (3k tokens) and lower temperature. Used for high-stakes reviews.

---

## 📁 Data & Training

The model is trained on a role-delimited wire format:

```text
<|system|>
{Moderator Persona + Markdown Instructions}
</|system|>
<|user|>
{Message to review}
</|user|>
<|assistant|>
<think>
{Step-by-step analysis}
</think>
{Verdict (SAFE, SPAM, HATE_SPEECH, etc.)}<|endoftext|>
```

### Training your own
If you have a `dataset.json` following the schema `{"instruction": "...", "reasoning": "...", "output": "..."}`, you can start training immediately:

```python
from model import GreesyGPT, get_dataset, GreesyTrainer

model = GreesyGPT()
dataset = get_dataset(file_path="your_data.json")
trainer = GreesyTrainer(model, dataset, batch_size=2, grad_accum=4)

trainer.train_epoch(epoch=1)
```

---

## 📝 Output Schema (JSON Mode)

When using `OutputFormat.JSON`, the model post-processes the Markdown verdict into a structured dictionary:

```json
{
  "verdict": "MISINFORMATION",
  "severity": 2,
  "confidence_hint": "medium",
  "reasoning_mode": "standard",
  "thinking_summary": "The user is claiming vaccines cause autism, which is a debunked...",
  "full_verdict": "## Verdict\n**MISINFORMATION**"
}
```

## ⚖️ License
This project is provided for educational and research purposes in AI safety and content moderation. Always include a human-in-the-loop for high-severity enforcement actions.
