"""
Convert a moderation-style dataset (id/label/instruction/reasoning/output/
complexity/source — like `moderation_dataset.json`) into a label-free
reasoning+answer dataset: only {id, instruction, reasoning, detailed_answer}.

No `label`, `output`, `complexity`, or `source` field survives — nothing
verdict-shaped ends up in the output at all. `reasoning` is carried over
as the model's step-by-step <think> content. `detailed_answer` is a new
field: a flowing, natural-language explanation synthesized from the
existing reasoning's content (its harm-potential and edge-case
observations), so it reads like an actual answer rather than a
`## Verdict\n**LABEL**` stamp.

This is a heuristic, deterministic text transform — it does not call
any model and does not invent new facts, it just rephrases what the
existing `reasoning` field already says. If you want richer, genuinely
distinct answers, swap `build_detailed_answer()` below for a call to an
LLM of your choice.

Usage:
    python make_dataset.py --input moderation_dataset.json \
        --output reasoning_answer_dataset.jsonl

    # also emit a pretty-printed .json alongside the .jsonl
    python make_dataset.py --input moderation_dataset.json \
        --output reasoning_answer_dataset.jsonl --also_json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List

SECTION_RE = re.compile(r"##\s*([^\n]+)\n(.*?)(?=\n##\s|\Z)", re.S)


def extract_sections(reasoning_md: str) -> Dict[str, str]:
    """Splits a '## Intent\\n...\\n\\n## Evidence\\n...' style markdown
    string into {section_name: body}."""
    return {name.strip(): body.strip() for name, body in SECTION_RE.findall(reasoning_md)}


def first_line(text: str) -> str:
    """Returns the first non-empty, non-heading line/bullet of a section,
    with any leading '- ' or '**' bullet/markdown stripped."""
    for raw in text.splitlines():
        line = raw.strip()
        line = re.sub(r"^-\s*", "", line)
        line = line.strip("*").strip()
        if line:
            return line
    return ""


def build_detailed_answer(label: str, sections: Dict[str, str]) -> str:
    """Synthesizes a short, natural-language answer paragraph from the
    reasoning's Harm Potential / Edge Cases observations — no
    '## Verdict' formatting, no bare SAFE/UNSAFE tag."""
    harm_line = first_line(sections.get("Harm Potential", ""))
    edge_line = first_line(sections.get("Edge Cases", ""))

    if label == "SAFE":
        lead = "This is fine to help with directly."
    else:
        lead = "I'm not able to help with this request."

    sentences = [lead]
    if harm_line:
        sentences.append(harm_line)
    if edge_line:
        sentences.append(edge_line)
    return " ".join(sentences)


def convert_record(record: Dict) -> Dict:
    reasoning = record.get("reasoning", "").strip()
    sections = extract_sections(reasoning)
    detailed_answer = build_detailed_answer(record.get("label", ""), sections)

    return {
        "id": record.get("id"),
        "instruction": record.get("instruction", "").strip(),
        "reasoning": reasoning,
        "detailed_answer": detailed_answer,
    }


def convert_dataset(records: List[Dict]) -> List[Dict]:
    out = []
    for r in records:
        try:
            out.append(convert_record(r))
        except Exception as e:
            print(f"[skip] {r.get('id', '?')}: {e}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to the source JSON array file")
    ap.add_argument("--output", required=True, help="Output .jsonl path (one record per line)")
    ap.add_argument("--also_json", action="store_true",
                     help="Also write a pretty-printed .json array next to the .jsonl")
    args = ap.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        records = json.load(f)
    print(f"[load] {len(records)} records from {args.input}")

    converted = convert_dataset(records)
    print(f"[convert] {len(converted)} records converted")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in converted:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[write] {out_path}")

    if args.also_json:
        json_path = out_path.with_suffix(".json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(converted, f, ensure_ascii=False, indent=2)
        print(f"[write] {json_path}")

    if converted:
        print("\n[sample]")
        print(json.dumps(converted[0], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
