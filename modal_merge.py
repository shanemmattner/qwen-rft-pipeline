"""
Merge PEFT LoRA adapter into base model weights on Modal (CPU, no GPU needed).

Loads the base model and LoRA adapter, merges weights via PEFT's
merge_and_unload(), and saves the result as a full HuggingFace model.
Includes a manual fallback merge path if PeftModel fails.

Optionally converts the merged model to GGUF format for llama.cpp / Ollama.

Usage:
    # Merge adapter into base model (bf16 HuggingFace output)
    modal run modal_merge.py --experiment <experiment-name>

    # Merge + convert to GGUF
    modal run modal_merge.py --experiment <experiment-name> --output-format gguf

    # Merge + GGUF with specific quantization
    modal run modal_merge.py --experiment <experiment-name> --output-format gguf --gguf-quant Q8_0

Download results:
    ./download_merged.sh <experiment-name>

Convert bf16 to MLX on Apple Silicon:
    python3 -m mlx_lm.convert --hf-path ./merged-bf16 --mlx-path ./merged-mlx-4bit --quantize --q-bits 4
"""

from __future__ import annotations

import pathlib

import modal

app = modal.App("rft-merge-adapter")

merge_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .uv_pip_install(
        "accelerate",
        "hf-transfer",
        "huggingface_hub",
        "peft @ git+https://github.com/huggingface/peft.git@main",
        "safetensors",
        "torch",
        "transformers>=5.5.0",
    )
    .env({"HF_HOME": "/model_cache"})
)

# Extended image with llama.cpp for GGUF conversion.
gguf_image = (
    merge_image
    .apt_install("cmake", "build-essential")
    .uv_pip_install("gguf", "numpy", "sentencepiece")
    .run_commands(
        "git clone --depth 1 https://github.com/ggml-org/llama.cpp.git /opt/llama-cpp",
        "cd /opt/llama-cpp && cmake -B build -DGGML_CUDA=OFF -DBUILD_SHARED_LIBS=OFF && "
        "cmake --build build --target llama-quantize -j$(nproc)",
        "ln -s /opt/llama-cpp/build/bin/llama-quantize /usr/local/bin/llama-quantize",
    )
)

with gguf_image.imports():
    import json
    import re
    import time
    import traceback

    import torch
    from peft import PeftModel
    from safetensors.torch import load_file as load_safetensors
    from transformers import AutoModelForCausalLM, AutoTokenizer

model_cache_volume = modal.Volume.from_name(
    "rft-model-cache", create_if_missing=True
)
checkpoint_volume = modal.Volume.from_name(
    "rft-checkpoints", create_if_missing=True
)


def _log_package_versions():
    """Print versions of key packages for troubleshooting."""
    import importlib.metadata

    packages = ["peft", "transformers", "torch", "accelerate", "safetensors"]
    print("\n--- Package Versions ---")
    for pkg in packages:
        try:
            ver = importlib.metadata.version(pkg)
            print(f"  {pkg}: {ver}")
        except importlib.metadata.PackageNotFoundError:
            print(f"  {pkg}: NOT INSTALLED")
    print("------------------------\n")


def _log_adapter_info(adapter_path: pathlib.Path):
    """Print adapter config and file listing for troubleshooting."""
    print(f"\n--- Adapter Info: {adapter_path} ---")

    config_path = adapter_path / "adapter_config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        print(f"  adapter_config.json:")
        for k, v in sorted(config.items()):
            print(f"    {k}: {v}")
    else:
        print(f"  WARNING: adapter_config.json not found at {config_path}")

    print(f"  Files:")
    for fpath in sorted(adapter_path.iterdir()):
        if fpath.is_file():
            size_mb = fpath.stat().st_size / (1024 * 1024)
            print(f"    {fpath.name}: {size_mb:.1f} MB")
    print("-----------------------------------\n")


def _manual_lora_merge(model, adapter_path: pathlib.Path):
    """Fallback: manually merge LoRA adapter into base model weights.

    Loads adapter_config.json for alpha/r, then for each lora_A/lora_B pair,
    computes delta = (B @ A) * (alpha / r) and adds it to the base weight.
    """
    print("FALLBACK: Using manual LoRA merge (PeftModel failed)")

    config_path = adapter_path / "adapter_config.json"
    with open(config_path) as f:
        config = json.load(f)

    lora_alpha = config["lora_alpha"]
    r = config["r"]
    scaling = lora_alpha / r
    print(f"  LoRA config: r={r}, alpha={lora_alpha}, scaling={scaling}")

    adapter_file = adapter_path / "adapter_model.safetensors"
    if not adapter_file.exists():
        raise FileNotFoundError(f"Adapter weights not found: {adapter_file}")

    adapter_weights = load_safetensors(str(adapter_file), device="cpu")
    print(f"  Loaded {len(adapter_weights)} adapter tensors")

    lora_a_keys = {k for k in adapter_weights if "lora_A" in k}
    merged_count = 0
    state_dict = model.state_dict()

    for a_key in sorted(lora_a_keys):
        b_key = a_key.replace("lora_A", "lora_B")
        if b_key not in adapter_weights:
            print(f"  WARNING: No matching lora_B for {a_key}, skipping")
            continue

        base_key = a_key
        base_key = re.sub(r"^base_model\.model\.", "", base_key)
        base_key = re.sub(r"\.lora_A\.default\.weight$", ".weight", base_key)

        if base_key not in state_dict:
            print(f"  WARNING: Base key '{base_key}' not found in model, skipping")
            continue

        lora_a = adapter_weights[a_key]
        lora_b = adapter_weights[b_key]
        delta = (lora_b @ lora_a) * scaling

        base_weight = state_dict[base_key]
        delta_norm = torch.norm(delta).item()
        base_norm = torch.norm(base_weight).item()
        print(f"  Merging {base_key}: delta_norm={delta_norm:.4f}, base_norm={base_norm:.4f}")

        state_dict[base_key] = base_weight + delta.to(base_weight.dtype)
        merged_count += 1

    print(f"  Merged {merged_count} LoRA modules into base weights")
    model.load_state_dict(state_dict)
    return model


def _convert_to_gguf(merged_path: pathlib.Path, gguf_path: pathlib.Path, gguf_quant: str):
    """Convert a merged HuggingFace model to GGUF format.

    Two-step process: convert_hf_to_gguf.py (HF -> F16 GGUF) then
    llama-quantize (F16 GGUF -> target quantization).
    """
    import subprocess

    gguf_path.mkdir(parents=True, exist_ok=True)

    f16_gguf = gguf_path / "model-F16.gguf"
    print(f"\n--- GGUF Conversion: {gguf_quant} ---")
    print(f"  Step 1: HF -> F16 GGUF")

    t0 = time.time()
    convert_cmd = [
        "python3", "/opt/llama-cpp/convert_hf_to_gguf.py",
        str(merged_path),
        "--outfile", str(f16_gguf),
        "--outtype", "f16",
    ]
    result = subprocess.run(convert_cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"  STDOUT: {result.stdout[-2000:]}" if result.stdout else "  STDOUT: (empty)")
        print(f"  STDERR: {result.stderr[-2000:]}" if result.stderr else "  STDERR: (empty)")
        raise RuntimeError(f"convert_hf_to_gguf.py failed (exit {result.returncode})")

    convert_sec = round(time.time() - t0, 2)
    f16_size_gb = f16_gguf.stat().st_size / (1024**3)
    print(f"  F16 GGUF size: {f16_size_gb:.2f} GB")
    print(f"[TIMING] GGUF convert (HF -> F16): {convert_sec}s")

    quantize_sec = 0.0
    if gguf_quant.upper() == "F16":
        final_gguf = f16_gguf
    else:
        final_gguf = gguf_path / f"model-{gguf_quant}.gguf"
        print(f"\n  Step 2: F16 GGUF -> {gguf_quant}")

        t0 = time.time()
        quantize_cmd = [
            "llama-quantize",
            str(f16_gguf),
            str(final_gguf),
            gguf_quant,
        ]
        result = subprocess.run(quantize_cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"  STDOUT: {result.stdout[-2000:]}" if result.stdout else "  STDOUT: (empty)")
            print(f"  STDERR: {result.stderr[-2000:]}" if result.stderr else "  STDERR: (empty)")
            raise RuntimeError(f"llama-quantize failed (exit {result.returncode})")

        quantize_sec = round(time.time() - t0, 2)
        final_size_gb = final_gguf.stat().st_size / (1024**3)
        print(f"  {gguf_quant} GGUF size: {final_size_gb:.2f} GB")
        print(f"[TIMING] GGUF quantize (F16 -> {gguf_quant}): {quantize_sec}s")

        print(f"  Removing intermediate F16 GGUF ({f16_size_gb:.2f} GB)")
        f16_gguf.unlink()

    print(f"  Final GGUF: {final_gguf}")
    return convert_sec, quantize_sec, final_gguf


def _verify_saved_model(merged_path: pathlib.Path):
    """Verify saved model files exist and print sizes."""
    print(f"\n--- Verifying saved model at {merged_path} ---")
    total_size = 0
    total_params = 0

    for fpath in sorted(merged_path.iterdir()):
        if fpath.is_file():
            size_mb = fpath.stat().st_size / (1024 * 1024)
            total_size += fpath.stat().st_size
            print(f"  {fpath.name}: {size_mb:.1f} MB")

            if fpath.suffix == ".safetensors":
                try:
                    tensors = load_safetensors(str(fpath), device="cpu")
                    n_params = sum(t.numel() for t in tensors.values())
                    total_params += n_params
                    print(f"    -> {n_params:,} parameters in {len(tensors)} tensors")
                except Exception as e:
                    print(f"    -> Could not inspect: {e}")

    print(f"  TOTAL size: {total_size / (1024**3):.2f} GB")
    if total_params > 0:
        print(f"  TOTAL parameters: {total_params:,}")
    print("-------------------------------------------\n")


@app.function(
    image=gguf_image,
    cpu=8,
    memory=65536,  # 64GB for 35B bf16 model
    volumes={
        "/model_cache": model_cache_volume,
        "/checkpoints": checkpoint_volume,
    },
    timeout=3600,
    single_use_containers=True,
)
def merge_and_convert(
    experiment: str,
    model_name: str = "Qwen/Qwen3.6-35B-A3B",
    output_format: str = "bf16",
    gguf_quant: str = "Q4_K_M",
):
    """Merge LoRA adapter into base model, optionally convert to GGUF."""
    t_total = time.time()
    adapter_path = pathlib.Path("/checkpoints") / "experiments" / experiment / "final_adapter"
    merged_path = pathlib.Path("/checkpoints") / "merged" / experiment / "bf16"

    _log_package_versions()

    if not adapter_path.exists():
        print(f"ERROR: Adapter not found at {adapter_path}")
        print("Available experiments:")
        exp_dir = pathlib.Path("/checkpoints") / "experiments"
        if exp_dir.exists():
            for d in sorted(exp_dir.iterdir()):
                print(f"  {d.name}")
        return

    _log_adapter_info(adapter_path)

    # Phase 1: Load base model
    t0 = time.time()
    print(f"Loading base model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=False,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    load_sec = round(time.time() - t0, 2)
    print(f"[TIMING] Model load: {load_sec}s")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Phase 2: Load adapter + merge
    merge_method = "unknown"
    try:
        t0 = time.time()
        print(f"\nLoading adapter via PeftModel from: {adapter_path}")
        model = PeftModel.from_pretrained(model, str(adapter_path))
        adapter_sec = round(time.time() - t0, 2)
        print(f"[TIMING] Adapter load (PeftModel): {adapter_sec}s")

        t0 = time.time()
        print("Merging adapter into base weights via merge_and_unload()...")
        model = model.merge_and_unload()
        merge_sec = round(time.time() - t0, 2)
        print(f"[TIMING] Merge (PeftModel): {merge_sec}s")
        merge_method = "PeftModel.merge_and_unload()"

    except Exception as e:
        print(f"\nERROR: PeftModel merge failed: {type(e).__name__}: {e}")
        traceback.print_exc()
        print("\nFalling back to manual LoRA merge...")

        t0 = time.time()
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="cpu",
            trust_remote_code=False,
        )
        reload_sec = round(time.time() - t0, 2)
        print(f"[TIMING] Model re-load: {reload_sec}s")

        t0 = time.time()
        model = _manual_lora_merge(model, adapter_path)
        adapter_sec = reload_sec
        merge_sec = round(time.time() - t0, 2)
        print(f"[TIMING] Manual merge: {merge_sec}s")
        merge_method = "manual LoRA delta (B @ A) * (alpha/r)"

    # Phase 3: Save merged model
    t0 = time.time()
    merged_path.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving merged model to: {merged_path}")
    model.save_pretrained(merged_path, safe_serialization=True)
    tokenizer.save_pretrained(merged_path)

    metadata = {
        "experiment": experiment,
        "base_model": model_name,
        "adapter_path": str(adapter_path),
        "merged_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "dtype": "bfloat16",
        "merge_method": merge_method,
    }

    adapter_meta_path = adapter_path / "metadata.json"
    if adapter_meta_path.exists():
        with open(adapter_meta_path) as f:
            metadata["training_metadata"] = json.load(f)

    with open(merged_path / "merge_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    save_sec = round(time.time() - t0, 2)
    print(f"[TIMING] Save merged: {save_sec}s")
    checkpoint_volume.commit()

    _verify_saved_model(merged_path)

    # Phase 4 (optional): GGUF conversion
    gguf_convert_sec = 0.0
    gguf_quantize_sec = 0.0
    final_gguf_file = None

    if output_format == "gguf":
        gguf_path = pathlib.Path("/checkpoints") / "merged" / experiment / "gguf"
        gguf_convert_sec, gguf_quantize_sec, final_gguf_file = _convert_to_gguf(
            merged_path, gguf_path, gguf_quant,
        )
        checkpoint_volume.commit()

    # Timing report
    total_sec = round(time.time() - t_total, 2)
    cpu_per_sec = 0.192 / 3600
    est_cost = total_sec * cpu_per_sec

    print(f"\n{'='*60}")
    print(f"MERGE REPORT -- {experiment}")
    print(f"{'='*60}")
    print(f"  Merge method:   {merge_method}")
    print(f"  Output format:  {output_format}")
    if output_format == "gguf":
        print(f"  GGUF quant:     {gguf_quant}")
    print(f"  Model load:     {load_sec:>8.1f}s")
    print(f"  Adapter load:   {adapter_sec:>8.1f}s")
    print(f"  Merge:          {merge_sec:>8.1f}s")
    print(f"  Save merged:    {save_sec:>8.1f}s")
    if output_format == "gguf":
        print(f"  GGUF convert:   {gguf_convert_sec:>8.1f}s")
        if gguf_quant.upper() != "F16":
            print(f"  GGUF quantize:  {gguf_quantize_sec:>8.1f}s")
    print(f"  ----------------------------------------")
    print(f"  TOTAL:          {total_sec:>8.1f}s ({total_sec/60:.1f} min)")
    print(f"  Estimated cost: ${est_cost:.2f}")
    print(f"{'='*60}")

    print(f"\nDownload merged bf16:")
    print(f"  ./download_merged.sh {experiment}")

    if output_format == "gguf" and final_gguf_file is not None:
        gguf_filename = final_gguf_file.name
        print(f"\nDownload GGUF ({gguf_quant}):")
        print(f"  modal volume get rft-checkpoints merged/{experiment}/gguf/{gguf_filename} ./{gguf_filename}")

    print(f"\nConvert bf16 to MLX on Apple Silicon:")
    print(f"  python3 -m mlx_lm.convert --hf-path ./merged-bf16 --mlx-path ./merged-mlx-4bit --quantize --q-bits 4")

    return experiment


@app.local_entrypoint()
def main(
    experiment: str = "",
    model_name: str = "Qwen/Qwen3.6-35B-A3B",
    output_format: str = "bf16",
    gguf_quant: str = "Q4_K_M",
):
    """Merge LoRA adapter into base model, optionally convert to GGUF.

    Args:
        experiment: Name of the training experiment (required).
        model_name: HuggingFace model ID for the base model.
        output_format: "bf16" (HuggingFace model) or "gguf" (GGUF file).
        gguf_quant: GGUF quantization type (Q4_K_M, Q6_K, Q8_0, F16).
    """
    if not experiment:
        print("ERROR: --experiment is required")
        print("Usage: modal run modal_merge.py --experiment <experiment-name>")
        return

    print(f"Merging adapter: {experiment}")
    print(f"Base model: {model_name}")
    print(f"Output format: {output_format}")
    if output_format == "gguf":
        print(f"GGUF quantization: {gguf_quant}")

    result = merge_and_convert.remote(
        experiment=experiment,
        model_name=model_name,
        output_format=output_format,
        gguf_quant=gguf_quant,
    )
    print(f"\nDone: {result}")
