#!/usr/bin/env python3
"""Format filtered samples into chat training format for LoRA fine-tuning.

Output: training_data/train.jsonl and training_data/valid.jsonl
Format: {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}

This format is compatible with mlx_lm.lora, axolotl, and most LoRA training
frameworks that accept chat-style JSONL.
"""

import argparse
import json
import random
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Format filtered samples for LoRA fine-tuning"
    )
    parser.add_argument(
        "--input", type=str, default="filtered_samples.jsonl",
        help="Input filtered samples file (default: filtered_samples.jsonl)"
    )
    parser.add_argument(
        "--output-dir", type=str, default="training_data",
        help="Output directory (default: training_data)"
    )
    parser.add_argument(
        "--train-split", type=float, default=0.9,
        help="Fraction for training set (default: 0.9)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for split (default: 42)"
    )
    args = parser.parse_args()

    # -----------------------------------------------------------------------
    # Load filtered samples
    # -----------------------------------------------------------------------
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: {input_path} not found. Run filter_samples.py first.")
        sys.exit(1)

    samples = []
    with open(input_path) as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))

    print(f"Loaded {len(samples)} filtered samples from {input_path}")

    if not samples:
        print("ERROR: No samples to format.")
        sys.exit(1)

    # -----------------------------------------------------------------------
    # Format as chat JSONL
    # -----------------------------------------------------------------------
    formatted = []
    total_prompt_tokens = 0
    total_response_tokens = 0

    for sample in samples:
        prompt = sample["prompt"]
        response = sample["response"]

        record = {
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response},
            ]
        }
        formatted.append(record)

        # Rough token estimate (words / 0.75)
        total_prompt_tokens += len(prompt.split())
        total_response_tokens += len(response.split())

    # -----------------------------------------------------------------------
    # Shuffle and split
    # -----------------------------------------------------------------------
    random.seed(args.seed)
    random.shuffle(formatted)

    split_idx = int(len(formatted) * args.train_split)
    train_data = formatted[:split_idx]
    valid_data = formatted[split_idx:]

    # -----------------------------------------------------------------------
    # Write output
    # -----------------------------------------------------------------------
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = output_dir / "train.jsonl"
    valid_path = output_dir / "valid.jsonl"

    with open(train_path, "w") as f:
        for rec in train_data:
            f.write(json.dumps(rec) + "\n")

    with open(valid_path, "w") as f:
        for rec in valid_data:
            f.write(json.dumps(rec) + "\n")

    # -----------------------------------------------------------------------
    # Stats
    # -----------------------------------------------------------------------
    avg_total = (total_prompt_tokens + total_response_tokens) / len(formatted)

    print(f"\n--- Stats ---")
    print(f"Total examples:        {len(formatted)}")
    print(f"Train set:             {len(train_data)}")
    print(f"Valid set:             {len(valid_data)}")
    print(f"Avg tokens/example:    ~{avg_total:.0f} (rough word-based estimate)")
    print(f"Avg prompt tokens:     ~{total_prompt_tokens / len(formatted):.0f}")
    print(f"Avg response tokens:   ~{total_response_tokens / len(formatted):.0f}")
    print(f"Train file:            {train_path}")
    print(f"Valid file:            {valid_path}")


if __name__ == "__main__":
    main()
