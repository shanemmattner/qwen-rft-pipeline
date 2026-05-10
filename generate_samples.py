#!/usr/bin/env python3
"""Generate code samples via self-distillation.

Reads prompts.jsonl, calls a model via OpenAI-compatible API at high
temperature, saves responses to samples.jsonl with resume support.

Based on Apple's SSD paper (arxiv 2604.01193):
  - High temperature sampling for diversity
  - NO judge/verifier -- paper explicitly avoids correctness signal
  - Filter: remove empty/stubs + bottom 10% shortest (done separately)

Works with ANY OpenAI-compatible server: MLX, llama.cpp, vLLM, Ollama,
text-generation-inference, etc.
"""

import argparse
import asyncio
import datetime
import json
import signal
import sys
import time
from pathlib import Path

try:
    import httpx
except ImportError:
    print("ERROR: 'httpx' package not installed. Run: pip install httpx")
    sys.exit(1)

try:
    from tqdm import tqdm
except ImportError:
    print("ERROR: 'tqdm' package not installed. Run: pip install tqdm")
    sys.exit(1)

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


# ---------------------------------------------------------------------------
# Defaults from SSD paper (Table for Qwen3-30B-A3B-Instruct)
# ---------------------------------------------------------------------------
DEFAULT_ENDPOINT = "http://localhost:8807/v1/chat/completions"
DEFAULT_TEMPERATURE = 1.6
DEFAULT_TOP_K = 20
DEFAULT_TOP_P = 0.8
DEFAULT_MAX_TOKENS = 8192
DEFAULT_WORKERS = 5


def load_config(path: str) -> dict:
    """Load a YAML or JSON config file and return a flat dict of params."""
    p = Path(path)
    if not p.exists():
        print(f"ERROR: Config file not found: {path}")
        sys.exit(1)

    with open(p) as f:
        if p.suffix in (".yaml", ".yml"):
            if not HAS_YAML:
                print("ERROR: 'pyyaml' package required for YAML configs. Run: pip install pyyaml")
                sys.exit(1)
            data = yaml.safe_load(f)
        else:
            data = json.load(f)

    # Flatten nested sampling_params into top level
    if "sampling_params" in data:
        for k, v in data["sampling_params"].items():
            if k not in data:
                data[k] = v
        del data["sampling_params"]

    return data


async def generate_one(
    client: httpx.AsyncClient,
    endpoint: str,
    prompt: str,
    model: str,
    temperature: float,
    top_k: int,
    top_p: float,
    max_tokens: int,
    repetition_penalty: float = 1.0,
    disable_thinking: bool = False,
    retries: int = 3,
) -> tuple[str, int, float]:
    """Generate a single sample. Returns (response_text, token_count, time_seconds)."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "stream": False,
    }
    # top_k is not standard OpenAI API but MLX/vLLM support it
    if top_k > 0:
        payload["top_k"] = top_k
    if repetition_penalty != 1.0:
        payload["repetition_penalty"] = repetition_penalty
    # Disable thinking mode for Qwen3.x models -- prevents model from
    # emitting reasoning traces as regular output tokens, which massively
    # inflates token counts and generation time.
    if disable_thinking:
        payload["chat_template_kwargs"] = {"enable_thinking": False}

    last_err = None
    for attempt in range(retries):
        t0 = time.monotonic()
        try:
            resp = await client.post(endpoint, json=payload, timeout=600.0)
            resp.raise_for_status()
            data = resp.json()
            elapsed = time.monotonic() - t0

            choice = data["choices"][0]
            text = choice["message"]["content"]
            tokens = data.get("usage", {}).get("completion_tokens", len(text.split()))

            return text, tokens, elapsed

        except (httpx.HTTPStatusError, httpx.RequestError, KeyError) as e:
            last_err = e
            wait = 2 ** attempt
            if attempt < retries - 1:
                await asyncio.sleep(wait)
            continue

    raise RuntimeError(f"Failed after {retries} retries: {last_err}")


STOP_FLAG = False

def handle_signal(sig, frame):
    global STOP_FLAG
    print(f"\n[{datetime.datetime.now().isoformat()}] Received signal {sig}, finishing current samples...")
    STOP_FLAG = True

signal.signal(signal.SIGTERM, handle_signal)
signal.signal(signal.SIGINT, handle_signal)


async def run(args):
    global STOP_FLAG
    run_start = time.monotonic()
    start_ts = datetime.datetime.now().isoformat()

    # -----------------------------------------------------------------------
    # Load prompts
    # -----------------------------------------------------------------------
    prompts_path = Path(args.input)
    if not prompts_path.exists():
        print(f"ERROR: {prompts_path} not found. Run download_prompts.py first.")
        sys.exit(1)

    prompts = []
    with open(prompts_path) as f:
        for line in f:
            prompts.append(json.loads(line))

    print(f"Loaded {len(prompts)} prompts from {prompts_path}")

    # -----------------------------------------------------------------------
    # Load existing samples for resume support
    # -----------------------------------------------------------------------
    output_path = Path(args.output)
    completed_ids: set[str] = set()
    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    completed_ids.add(rec["id"])
                except (json.JSONDecodeError, KeyError):
                    continue
        print(f"Resuming: {len(completed_ids)} already completed")

    resumed_count = len(completed_ids)
    remaining = [p for p in prompts if p["id"] not in completed_ids]
    print(f"Remaining: {len(remaining)} prompts to generate")

    if not remaining:
        print("All prompts already completed!")
        return

    # -----------------------------------------------------------------------
    # Deadline
    # -----------------------------------------------------------------------
    deadline_ts = None
    if args.deadline:
        deadline_ts = float(args.deadline)
        remaining_s = deadline_ts - time.time()
        print(f"Deadline: {remaining_s/3600:.1f}h remaining")

    # -----------------------------------------------------------------------
    # Discover model name from endpoint
    # -----------------------------------------------------------------------
    if args.model:
        model_name = args.model
    else:
        base_url = args.endpoint.split("/v1/")[0] + "/v1"
        models_url = base_url + "/models"
        model_name = "default"
        try:
            async with httpx.AsyncClient() as c:
                r = await c.get(models_url, timeout=5.0)
                if r.status_code == 200:
                    models = r.json().get("data", [])
                    if models:
                        model_name = models[0].get("id", "default")
        except Exception:
            pass
    print(f"Using model: {model_name}")

    # -----------------------------------------------------------------------
    # Write run metadata
    # -----------------------------------------------------------------------
    meta_path = output_path.parent / "run_metadata.json"
    metadata = {
        "start_time": start_ts,
        "model": model_name,
        "endpoint": args.endpoint,
        "sampling_params": {
            "temperature": args.temperature,
            "top_k": args.top_k,
            "top_p": args.top_p,
            "repetition_penalty": args.repetition_penalty,
            "max_tokens": args.max_tokens,
            "disable_thinking": args.disable_thinking,
        },
        "dataset_source": "microsoft/rStar-Coder seed_sft",
        "prompts_file": str(prompts_path),
        "total_prompts": len(prompts),
        "resumed_from": resumed_count,
        "remaining_at_start": len(remaining),
        "workers": args.workers,
        "method": "SSD (arxiv 2604.01193) -- high-temp self-distillation, no correctness signal",
    }
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata: {meta_path}")

    # -----------------------------------------------------------------------
    # Generate samples
    # -----------------------------------------------------------------------
    semaphore = asyncio.Semaphore(args.workers)
    failures: list[dict] = []
    generated_count = 0
    total_tokens = 0
    total_gen_time = 0.0

    async def process_one(prompt_rec: dict, pbar: tqdm, out_file):
        nonlocal generated_count, total_tokens, total_gen_time
        if STOP_FLAG:
            return
        if deadline_ts and time.time() >= deadline_ts:
            return

        async with semaphore:
            if STOP_FLAG or (deadline_ts and time.time() >= deadline_ts):
                return
            pid = prompt_rec["id"]
            try:
                text, tokens, elapsed = await generate_one(
                    client=client,
                    endpoint=args.endpoint,
                    prompt=prompt_rec["prompt"],
                    model=model_name,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    max_tokens=args.max_tokens,
                    repetition_penalty=args.repetition_penalty,
                    disable_thinking=args.disable_thinking,
                )
                record = {
                    "id": pid,
                    "prompt": prompt_rec["prompt"],
                    "response": text,
                    "tokens": tokens,
                    "time_s": round(elapsed, 2),
                    "timestamp": datetime.datetime.now().isoformat(),
                    "model": model_name,
                    "sampling": {
                        "temperature": args.temperature,
                        "top_k": args.top_k,
                        "top_p": args.top_p,
                    },
                }
                line = json.dumps(record) + "\n"
                out_file.write(line)
                out_file.flush()
                generated_count += 1
                total_tokens += tokens
                total_gen_time += elapsed
                pbar.update(1)
                pbar.set_postfix(
                    tok_s=f"{total_tokens/total_gen_time:.0f}" if total_gen_time > 0 else "...",
                    avg=f"{total_gen_time/generated_count:.1f}s" if generated_count > 0 else "...",
                )

            except Exception as e:
                failures.append({"id": pid, "error": str(e), "timestamp": datetime.datetime.now().isoformat()})
                pbar.update(1)

    async with httpx.AsyncClient() as client:
        with open(output_path, "a") as out_file:
            with tqdm(total=len(remaining), desc="Generating", unit="sample") as pbar:
                tasks = [
                    process_one(p, pbar, out_file) for p in remaining
                ]
                await asyncio.gather(*tasks)

    # -----------------------------------------------------------------------
    # Write summary
    # -----------------------------------------------------------------------
    wall_time = time.monotonic() - run_start
    summary = {
        "start_time": start_ts,
        "end_time": datetime.datetime.now().isoformat(),
        "wall_time_seconds": round(wall_time, 1),
        "wall_time_human": f"{wall_time/3600:.1f}h",
        "samples_resumed": resumed_count,
        "samples_generated": generated_count,
        "samples_failed": len(failures),
        "total_samples": resumed_count + generated_count,
        "total_tokens": total_tokens,
        "total_inference_time_s": round(total_gen_time, 1),
        "avg_time_per_sample_s": round(total_gen_time / generated_count, 1) if generated_count else 0,
        "throughput_tok_per_s": round(total_tokens / total_gen_time, 1) if total_gen_time > 0 else 0,
        "stopped_by": "deadline" if (deadline_ts and time.time() >= deadline_ts) else ("signal" if STOP_FLAG else "completed"),
    }
    summary_path = output_path.parent / "run_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n--- Results ---")
    print(f"Generated:  {generated_count}")
    print(f"Failed:     {len(failures)}")
    print(f"Total:      {resumed_count + generated_count}")
    print(f"Throughput: {summary['throughput_tok_per_s']} tok/s")
    print(f"Avg time:   {summary['avg_time_per_sample_s']}s/sample")
    print(f"Output:     {output_path}")
    print(f"Summary:    {summary_path}")

    if failures:
        fail_path = output_path.parent / "generation_failures.jsonl"
        with open(fail_path, "w") as f:
            for rec in failures:
                f.write(json.dumps(rec) + "\n")
        print(f"Failures:   {fail_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate code samples for self-distillation (SSD paper, arxiv 2604.01193)"
    )
    parser.add_argument(
        "--input", type=str, default="prompts.jsonl",
        help="Input prompts file (default: prompts.jsonl)"
    )
    parser.add_argument(
        "--output", type=str, default="samples.jsonl",
        help="Output samples file (default: samples.jsonl)"
    )
    parser.add_argument(
        "--endpoint", type=str, default=DEFAULT_ENDPOINT,
        help=f"OpenAI-compatible chat completions endpoint (default: {DEFAULT_ENDPOINT})"
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Model name to use (default: auto-detect from /v1/models)"
    )
    parser.add_argument(
        "--temperature", type=float, default=DEFAULT_TEMPERATURE,
        help=f"Sampling temperature (default: {DEFAULT_TEMPERATURE})"
    )
    parser.add_argument(
        "--top-k", type=int, default=DEFAULT_TOP_K,
        help=f"Top-k sampling (default: {DEFAULT_TOP_K})"
    )
    parser.add_argument(
        "--top-p", type=float, default=DEFAULT_TOP_P,
        help=f"Top-p sampling (default: {DEFAULT_TOP_P})"
    )
    parser.add_argument(
        "--repetition-penalty", type=float, default=1.0,
        help="Repetition penalty (default: 1.0, i.e. none)"
    )
    parser.add_argument(
        "--max-tokens", type=int, default=DEFAULT_MAX_TOKENS,
        help=f"Max tokens per response (default: {DEFAULT_MAX_TOKENS})"
    )
    parser.add_argument(
        "--workers", type=int, default=DEFAULT_WORKERS,
        help=f"Concurrent requests (default: {DEFAULT_WORKERS})"
    )
    parser.add_argument(
        "--disable-thinking", action="store_true", default=False,
        help="Disable thinking mode for Qwen3.x models (adds chat_template_kwargs)"
    )
    parser.add_argument(
        "--deadline", type=str, default=None,
        help="Unix timestamp to stop generation (for overnight runs with hard stop)"
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to YAML/JSON config file (overrides CLI defaults, CLI flags override config)"
    )
    args = parser.parse_args()

    # Apply config file (config < CLI flags)
    if args.config:
        cfg = load_config(args.config)
        for key, val in cfg.items():
            attr = key.replace("-", "_")
            # Only apply config values for args that are at their default
            if hasattr(args, attr):
                cli_default = parser.get_default(attr)
                current = getattr(args, attr)
                if current == cli_default and val is not None:
                    setattr(args, attr, val)

    asyncio.run(run(args))


if __name__ == "__main__":
    main()
