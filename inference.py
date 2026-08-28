"""
Inference for GreesyGPT-Ultra finetuned on the label-free reasoning+answer
format (see make_dataset.py) — i.e. no moderation verdict/label, just a
<think> reasoning block followed by a detailed answer.

Single prompt:
    python inference.py --checkpoint ckpt-ultra/final.pt \
        --tokenizer_json /path/to/tokenizer.json \
        --prompt "Please help me with gardening advice for apartment herbs."

Batch over a JSONL/JSON file of {"instruction": ...} records:
    python inference.py --checkpoint ckpt-ultra/final.pt \
        --tokenizer_json /path/to/tokenizer.json \
        --input_file prompts.jsonl --output_file results.jsonl

Smoke test without a real checkpoint or tokenizer (random weights,
tiktoken fallback — just exercises the code path):
    python inference.py --smoke_test --prompt "hello"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterator, Optional

import torch

from config import ULTRA_CONFIG, ULTRA_SMALL_CONFIG, MINI_CONFIG
from model import GreesyGPTUltra, ReasoningMode
from tokenizer import GreesyTokenizer
from data import ANSWER_SYSTEM_PROMPT

CONFIGS = {"ultra": ULTRA_CONFIG, "ultra_small": ULTRA_SMALL_CONFIG, "mini": MINI_CONFIG}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", choices=list(CONFIGS.keys()), default="ultra")
    p.add_argument("--checkpoint", type=str, default=None,
                    help="Path to a finetuned .pt checkpoint (from finetune.py).")
    p.add_argument("--tokenizer_json", type=str, default=None,
                    help="Path to the GreesyGuard tokenizer.json.")
    p.add_argument("--smoke_test", action="store_true",
                    help="Skip loading a real checkpoint/tokenizer — random weights + "
                         "tiktoken fallback, to sanity-check the generation code path only.")

    p.add_argument("--prompt", type=str, default=None, help="A single instruction to run.")
    p.add_argument("--input_file", type=str, default=None,
                    help="JSON or JSONL file of records with an 'instruction' field.")
    p.add_argument("--output_file", type=str, default=None,
                    help="Where to write results (JSONL). Required with --input_file.")

    p.add_argument("--system", type=str, default=ANSWER_SYSTEM_PROMPT)
    p.add_argument("--mode", choices=[m.value for m in ReasoningMode], default="MEDIUM",
                    help="Reasoning budget tier — controls max think+answer tokens generated.")
    p.add_argument("--max_new_tokens", type=int, default=None,
                    help="Overrides the --mode token budget if set.")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--show_reasoning", action="store_true", default=True)
    p.add_argument("--hide_reasoning", dest="show_reasoning", action="store_false")
    return p.parse_args()


def load_model(config_name: str, checkpoint: Optional[str], device: str) -> GreesyGPTUltra:
    cfg = CONFIGS[config_name]
    model = GreesyGPTUltra(cfg).to(device)
    if checkpoint:
        state = torch.load(checkpoint, map_location=device)
        model.load_state_dict(state["model"] if "model" in state else state)
        print(f"[load] weights from {checkpoint}", file=sys.stderr)
    else:
        print("[load] WARNING: no --checkpoint given, using randomly initialized weights.",
              file=sys.stderr)
    model.eval()
    return model


def build_prompt(system: str, instruction: str) -> str:
    return (
        f"<|system|>\n{system}\n</|system|>\n\n"
        f"<|user|>\n{instruction}\n</|user|>\n\n"
        f"<|assistant|>\n"
    )


def run_inference(
    model: GreesyGPTUltra,
    tok: GreesyTokenizer,
    instruction: str,
    system: str,
    mode: ReasoningMode,
    max_new_tokens: Optional[int],
    temperature: float,
    top_p: float,
    device: str,
) -> Dict[str, str]:
    budget = max_new_tokens or model.cfg.reasoning_budgets[mode.value]
    prompt = build_prompt(system, instruction)
    input_ids = tok.encode(prompt, return_tensors=True).to(device)

    eos_id = tok.token_to_id("<|endoftext|>")
    out_ids = model.generate(
        input_ids, max_new_tokens=budget, eos_token_id=eos_id,
        temperature=temperature, top_p=top_p,
    )
    text = tok.decode(out_ids[0, input_ids.shape[1]:].tolist())
    text = text.replace("<|endoftext|>", "").strip()

    reasoning, answer = "", text
    if "<think>" in text and "</think>" in text:
        reasoning = text.split("<think>", 1)[1].split("</think>", 1)[0].strip()
        answer = text.split("</think>", 1)[1].strip()

    return {"instruction": instruction, "reasoning": reasoning, "detailed_answer": answer}


def iter_input_records(path: str) -> Iterator[Dict]:
    p = Path(path)
    if p.suffix == ".jsonl":
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)
    else:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        for rec in data:
            yield rec


def main():
    args = parse_args()

    if args.smoke_test:
        tok = GreesyTokenizer(use_tiktoken_fallback=True)
        cfg_name = "ultra_small" if args.config == "ultra" else args.config
        model = load_model(cfg_name, None, args.device)
    else:
        if not args.tokenizer_json:
            raise ValueError("--tokenizer_json is required (or pass --smoke_test).")
        tok = GreesyTokenizer(tokenizer_json_path=args.tokenizer_json)
        model = load_model(args.config, args.checkpoint, args.device)

    mode = ReasoningMode(args.mode)

    if args.prompt:
        result = run_inference(
            model, tok, args.prompt, args.system, mode,
            args.max_new_tokens, args.temperature, args.top_p, args.device,
        )
        if args.show_reasoning:
            print("=== Reasoning ===")
            print(result["reasoning"] or "(none)")
            print()
        print("=== Detailed Answer ===")
        print(result["detailed_answer"])
        return

    if args.input_file:
        if not args.output_file:
            raise ValueError("--output_file is required with --input_file")
        out_path = Path(args.output_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        n = 0
        with open(out_path, "w", encoding="utf-8") as f:
            for rec in iter_input_records(args.input_file):
                instruction = rec.get("instruction") or rec.get("prompt") or rec.get("message")
                if not instruction:
                    continue
                result = run_inference(
                    model, tok, instruction, args.system, mode,
                    args.max_new_tokens, args.temperature, args.top_p, args.device,
                )
                if "id" in rec:
                    result = {"id": rec["id"], **result}
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                n += 1
                if n % 20 == 0:
                    print(f"[inference] {n} done", file=sys.stderr)
        print(f"[inference] wrote {n} results to {out_path}")
        return

    raise ValueError("Provide either --prompt or --input_file")


if __name__ == "__main__":
    main()
