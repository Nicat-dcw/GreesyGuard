
# GreesyGuard: Reasoning-Based Content Moderation

GreesyGuard is a lightweight, specialized Transformer model designed for high-precision content moderation. Unlike traditional "black-box" classifiers, GreesyGPT utilizes **Chain-of-Thought (CoT)** reasoning to analyze intent, context, and policy violations before issuing a final verdict.

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
## 🤴 leaderboard

| Model           | Overall   | Flames                   | Safety                     | llm-trustworthy-leaderboard          | Legality                   | Data protection             |
|-----------------|-----------|----------------------------|----------------------------|----------------------------|----------------------------|-----------------------------|
| 3-mini-deep        | 77.91%    | 45.38% / 79.8              | 45.45% / 74.1              | 42.79% / 76.8              | 45.65% / 63.8              | 55.26% / 70.2               |
| 3-mini-expert           | 70.01%    | 41.37% / 78.2              | 27.51% / 67.7              | 50.75% / 80.6              | 30.43% / 53.6              | 50.0% / 66.7                |
| 3-mini-standart          | **63.77%**| **53.41%** / 83.4          | 28.44% / 65.5              | **77.11%** / **91.5**      | 71.74% / 81.2              | **88.16%** / **92.1**       |
| 3-mini-flash         | 23.66%    | 24.5% / 69.9               | 18.41% / 59.6              | 27.86% / 70.5              | 30.43% / 53.6              | 17.11% / 44.7               |




## 🚦 Quick Start

### 1. Requirements
```bash
pip install torch tiktoken tqdm
```

### 2. Basic Inference
GreesyGuard uses a specific chat template to trigger reasoning.

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
