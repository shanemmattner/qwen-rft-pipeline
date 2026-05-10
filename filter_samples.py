#!/usr/bin/env python3
"""Filter generated samples using the SSD paper's minimal filtering.

From Apple's SSD paper (arxiv 2604.01193):
  1. Remove empty or stub responses (no meaningful code)
  2. Remove the bottom N% shortest responses by character length

That's it. No LLM judge. No correctness signal. The paper's core insight
is that high-temperature diversity is sufficient -- quality filtering
beyond removing junk actually hurts.

Usage:
    python3 filter_samples.py --input samples.jsonl --output filtered_samples.jsonl
    python3 filter_samples.py --input samples.jsonl --remove-shortest-pct 15
    python3 filter_samples.py --input samples.jsonl --min-length 100
"""

import argparse
import json
import re
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Stub detection patterns
# ---------------------------------------------------------------------------
STUB_PATTERNS = [
    r"^\s*pass\s*$",
    r"^\s*\.\.\.\s*$",
    r"^\s*raise\s+NotImplementedError",
    r"^\s*#\s*TODO",
    r"^\s*#\s*your\s+code\s+here",
    r"^\s*#\s*write\s+your",
    r"^\s*#\s*implement",
]

STUB_RE = re.compile("|".join(STUB_PATTERNS), re.IGNORECASE | re.MULTILINE)


def is_empty_or_stub(response: str) -> bool:
    """Check if a response is empty, whitespace-only, or a stub."""
    text = response.strip()
    if not text:
        return True
    # Very short responses (less than 20 chars) are almost certainly stubs
    if len(text) < 20:
        return True
    # Check if the only code content is a stub pattern
    # Strip markdown code fences if present
    code = re.sub(r"```\w*\n?", "", text)
    code = code.strip()
    if not code:
        return True
    # Check if all non-empty, non-comment lines are stub patterns
    lines = [l for l in code.splitlines() if l.strip() and not l.strip().startswith("#")]
    if not lines:
        return True
    # If there's only one meaningful line and it matches a stub pattern
    if len(lines) <= 2 and STUB_RE.search(code):
        return True
    return False


def main():
    parser = argparse.ArgumentParser(
        description="Filter samples using SSD paper's minimal filtering (no LLM judge)"
    )
    parser.add_argument(
        "--input", type=str, default="samples.jsonl",
        help="Input samples file (default: samples.jsonl)"
    )
    parser.add_argument(
        "--output", type=str, default="filtered_samples.jsonl",
        help="Output filtered samples file (default: filtered_samples.jsonl)"
    )
    parser.add_argument(
        "--min-length", type=int, default=0,
        help="Minimum response length in characters (default: 0, use --remove-shortest-pct instead)"
    )
    parser.add_argument(
        "--remove-shortest-pct", type=float, default=10.0,
        help="Remove bottom N%% shortest responses (default: 10.0, per SSD paper)"
    )
    args = parser.parse_args()

    # -----------------------------------------------------------------------
    # Load samples
    # -----------------------------------------------------------------------
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: {input_path} not found. Run generate_samples.py first.")
        sys.exit(1)

    samples = []
    with open(input_path) as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))

    total_input = len(samples)
    print(f"Loaded {total_input} samples from {input_path}")

    if not samples:
        print("ERROR: No samples to filter.")
        sys.exit(1)

    # -----------------------------------------------------------------------
    # Step 1: Remove empty and stub responses
    # -----------------------------------------------------------------------
    empty_count = 0
    after_empty = []
    for s in samples:
        response = s.get("response", "")
        if is_empty_or_stub(response):
            empty_count += 1
        else:
            after_empty.append(s)

    print(f"Removed {empty_count} empty/stub responses ({100*empty_count/total_input:.1f}%)")

    # -----------------------------------------------------------------------
    # Step 2: Apply minimum length filter (if specified)
    # -----------------------------------------------------------------------
    after_minlen = after_empty
    minlen_removed = 0
    if args.min_length > 0:
        after_minlen = [s for s in after_empty if len(s.get("response", "")) >= args.min_length]
        minlen_removed = len(after_empty) - len(after_minlen)
        print(f"Removed {minlen_removed} below --min-length {args.min_length} chars")

    # -----------------------------------------------------------------------
    # Step 3: Remove bottom N% shortest responses
    # -----------------------------------------------------------------------
    after_pct = after_minlen
    pct_removed = 0
    if args.remove_shortest_pct > 0 and after_minlen:
        response_lengths = sorted(len(s.get("response", "")) for s in after_minlen)
        cutoff_idx = int(len(response_lengths) * args.remove_shortest_pct / 100)
        length_cutoff = response_lengths[cutoff_idx] if cutoff_idx < len(response_lengths) else 0

        after_pct = [s for s in after_minlen if len(s.get("response", "")) >= length_cutoff]
        pct_removed = len(after_minlen) - len(after_pct)
        print(f"Removed {pct_removed} in bottom {args.remove_shortest_pct}% shortest "
              f"(cutoff: {length_cutoff} chars)")

    # -----------------------------------------------------------------------
    # Save filtered samples
    # -----------------------------------------------------------------------
    output_path = Path(args.output)
    with open(output_path, "w") as f:
        for sample in after_pct:
            f.write(json.dumps(sample) + "\n")

    # -----------------------------------------------------------------------
    # Stats
    # -----------------------------------------------------------------------
    kept = len(after_pct)
    removed = total_input - kept
    avg_len = sum(len(s.get("response", "")) for s in after_pct) / kept if kept else 0

    print(f"\n--- Filter Summary ---")
    print(f"Input:               {total_input}")
    print(f"Empty/stub removed:  {empty_count}")
    if minlen_removed:
        print(f"Below min-length:    {minlen_removed}")
    print(f"Bottom {args.remove_shortest_pct}% removed: {pct_removed}")
    print(f"Kept:                {kept} ({100*kept/total_input:.1f}%)")
    print(f"Total removed:       {removed}")
    print(f"Avg response length: {avg_len:.0f} chars")
    print(f"Output:              {output_path}")


if __name__ == "__main__":
    main()
