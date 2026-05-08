"""
Modal SFT LoRA fine-tuning for MoE language models.

Trains a LoRA adapter on self-generated (or any) chat-format JSONL data using
Modal cloud GPUs. Designed for models with non-standard architectures like
Qwen3.6's Gated DeltaNet, where standard LoRA targets are insufficient.

Adapted from:
  1. Modal's official unsloth_finetune.py (infrastructure, volumes, retries)
  2. kreuzhofer's DGX Spark Qwen3.5-35B-A3B script (MoE workarounds, bf16 LoRA)

Note: Unsloth is NOT used — its Triton grouped GEMM kernel crashes on
Qwen3.6 MoE. Uses raw HF transformers + PEFT + TRL instead.

Usage:
    modal run modal_train.py --upload-data /path/to/training_data
    modal run modal_train.py --smoke-test
    modal run modal_train.py
    modal run modal_train.py --experiment-name my-experiment --max-steps 200

Download adapter after training:
    modal volume get rft-checkpoints experiments/<name>/final_adapter/ ./out/
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import modal

# ---------------------------------------------------------------------------
# Modal App
# ---------------------------------------------------------------------------

app = modal.App("rft-lora-train")

# ---------------------------------------------------------------------------
# Container Image
# ---------------------------------------------------------------------------

train_image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(
        "accelerate",
        "bitsandbytes",
        "datasets",
        "hf-transfer",
        "huggingface_hub",
        "peft",
        "torch",
        "transformers>=5.5.0",
        "trl",
    )
    .env({"HF_HOME": "/model_cache", "HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

with train_image.imports():
    import hashlib
    import importlib.metadata as _importlib_metadata
    import json
    import subprocess
    import sys
    import time

    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainerCallback,
    )
    from trl import SFTTrainer, SFTConfig

# ---------------------------------------------------------------------------
# Volume Configuration
# ---------------------------------------------------------------------------

model_cache_volume = modal.Volume.from_name(
    "rft-model-cache", create_if_missing=True
)
dataset_cache_volume = modal.Volume.from_name(
    "rft-training-data", create_if_missing=True
)
checkpoint_volume = modal.Volume.from_name(
    "rft-checkpoints", create_if_missing=True
)

# ---------------------------------------------------------------------------
# GPU Configuration
# ---------------------------------------------------------------------------

GPU_TYPE = "H200"
TIMEOUT_HOURS = 6

# ---------------------------------------------------------------------------
# Data constants
# ---------------------------------------------------------------------------

DATA_DIR = "/training_data"
TEXT_COLUMN = "text"
PREPROCESSING_WORKERS = 2

# ---------------------------------------------------------------------------
# LoRA target modules
# ---------------------------------------------------------------------------
# For Qwen3.6 DeltaNet architecture, you MUST include the DeltaNet-specific
# targets. Standard q_proj/k_proj/v_proj/o_proj alone -> 0.02% trainable -> NaN.
# See configs/qwen36-35b-a3b.yaml for the full reference.

LORA_TARGET_MODULES = [
    # Standard attention projections
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    # MLP projections
    "gate_proj",
    "up_proj",
    "down_proj",
    # DeltaNet linear attention projections
    "in_proj_qkv",
    "in_proj_z",
    "out_proj",
]


# ---------------------------------------------------------------------------
# Training Config
# ---------------------------------------------------------------------------

@dataclass
class TrainingConfig:
    model_name: str = "Qwen/Qwen3.6-35B-A3B"
    max_seq_length: int = 2048
    load_in_4bit: bool = False
    load_in_8bit: bool = False

    # LoRA hyperparameters
    lora_r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    lora_bias: str = "none"
    use_rslora: bool = False

    # Training hyperparameters
    optim: str = "adamw_8bit"
    batch_size: int = 4
    gradient_accumulation_steps: int = 8
    packing: bool = False
    learning_rate: float = 2e-4
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.06
    weight_decay: float = 0.01
    max_steps: int = 150
    save_steps: int = 50
    eval_steps: int = 25
    logging_steps: int = 5

    # Experiment
    seed: int = 42
    experiment_name: Optional[str] = None
    skip_eval: bool = False
    smoke_test: bool = False
    lora_targets: Optional[str] = None  # comma-separated override

    def __post_init__(self):
        if self.smoke_test:
            self.max_steps = 20
            self.save_steps = 10
            self.eval_steps = 10
            self.logging_steps = 2
        if self.experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            model_short = self.model_name.split("/")[-1]
            suffix = "smoke" if self.smoke_test else "full"
            self.experiment_name = f"rft-{model_short}-r{self.lora_r}-{suffix}-{timestamp}"


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

def get_text_tokenizer(tokenizer):
    """Workaround: Qwen3.5/3.6 may be a vision-language model whose tokenizer
    is a Qwen3VLProcessor wrapping a text tokenizer. Extract the inner one."""
    return getattr(tokenizer, "tokenizer", tokenizer)


def format_chat_template(examples, tokenizer):
    """Apply chat template to messages in role/content format."""
    text_tok = get_text_tokenizer(tokenizer)
    texts = []
    for messages in examples["messages"]:
        formatted_text = text_tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        texts.append(formatted_text)
    return {TEXT_COLUMN: texts}


def load_jsonl_datasets(config, tokenizer):
    """Load JSONL files from Modal volume and apply chat template.

    Expects train.jsonl (required) and valid.jsonl (optional) in DATA_DIR.
    Each line must have a "messages" field with chat-format messages.
    """
    from pathlib import Path

    train_path = Path(DATA_DIR) / "train.jsonl"
    valid_path = Path(DATA_DIR) / "valid.jsonl"

    if not train_path.exists():
        raise FileNotFoundError(
            f"No training data at {train_path}. Run with --upload-data first."
        )

    def read_jsonl(p):
        with open(p) as f:
            return [json.loads(line) for line in f if line.strip()]

    train_records = read_jsonl(train_path)
    valid_records = read_jsonl(valid_path) if valid_path.exists() else []

    print(f"Loaded {len(train_records)} train, {len(valid_records)} valid examples")

    train_dataset = Dataset.from_list(train_records)
    eval_dataset = Dataset.from_list(valid_records) if valid_records else None

    print("Formatting datasets with chat template...")
    train_dataset = train_dataset.map(
        lambda examples: format_chat_template(examples, tokenizer),
        batched=True,
        num_proc=PREPROCESSING_WORKERS,
        remove_columns=train_dataset.column_names,
    )

    if eval_dataset is not None:
        eval_dataset = eval_dataset.map(
            lambda examples: format_chat_template(examples, tokenizer),
            batched=True,
            num_proc=PREPROCESSING_WORKERS,
            remove_columns=eval_dataset.column_names,
        )

    return train_dataset, eval_dataset


# ---------------------------------------------------------------------------
# Model Loading
# ---------------------------------------------------------------------------

def load_model(config):
    """Load pretrained model in bf16 (not QLoRA — unreliable for MoE)."""
    print(f"Loading model: {config.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
    )
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    return model, tokenizer


# ---------------------------------------------------------------------------
# LoRA Setup
# ---------------------------------------------------------------------------

def setup_model_for_training(model, config, target_modules=None):
    """Configure LoRA adapters via PEFT."""
    targets = target_modules or LORA_TARGET_MODULES
    print(f"Configuring LoRA for training with {len(targets)} targets: {targets}")
    lora_config = LoraConfig(
        r=config.lora_r,
        target_modules=targets,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias=config.lora_bias,
        use_rslora=config.use_rslora,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    return model


# ---------------------------------------------------------------------------
# Training Config (SFTConfig)
# ---------------------------------------------------------------------------

def create_training_config(config, output_dir, skip_eval):
    """Create SFTConfig for TRL SFTTrainer."""
    return SFTConfig(
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        max_steps=config.max_steps,
        warmup_ratio=config.warmup_ratio,
        eval_steps=config.eval_steps,
        save_steps=config.save_steps,
        eval_strategy="no" if skip_eval else "steps",
        save_strategy="steps",
        do_eval=not skip_eval,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        optim=config.optim,
        weight_decay=config.weight_decay,
        lr_scheduler_type=config.lr_scheduler_type,
        logging_steps=config.logging_steps,
        output_dir=output_dir,
        report_to="none",
        seed=config.seed,
        save_total_limit=3,
        max_length=config.max_seq_length,
        packing=config.packing,
        dataset_text_field=TEXT_COLUMN,
    )


# ---------------------------------------------------------------------------
# Checkpoint Resume
# ---------------------------------------------------------------------------

def check_for_existing_checkpoint(checkpoint_dir):
    """Check for existing checkpoint to resume from."""
    import pathlib as _pl
    d = _pl.Path(checkpoint_dir)
    if not d.exists():
        return None
    checkpoints = list(d.glob("checkpoint-*"))
    if checkpoints:
        latest = max(checkpoints, key=lambda p: int(p.name.split("-")[1]))
        print(f"Found existing checkpoint: {latest}")
        return str(latest)
    return None


# ---------------------------------------------------------------------------
# GPU Memory Helper
# ---------------------------------------------------------------------------

def gpu_mem_gb():
    if not torch.cuda.is_available():
        return 0.0, 0.0
    alloc = torch.cuda.memory_allocated() / (1024**3)
    reserved = torch.cuda.memory_reserved() / (1024**3)
    return alloc, reserved


# ---------------------------------------------------------------------------
# Timing Callback
# ---------------------------------------------------------------------------

class TimingCallback:
    """Per-step timing, GPU memory tracking, eval/save timing."""

    def __init__(self, logging_steps):
        self.logging_steps = logging_steps
        self.step_times = []
        self.eval_times = []
        self.save_times = []
        self._step_start = None
        self._eval_start = None
        self._save_start = None
        self.peak_gpu_gb = 0.0

    def on_step_begin(self, args, state, control, **kwargs):
        self._step_start = time.time()

    def on_step_end(self, args, state, control, **kwargs):
        if self._step_start:
            elapsed = time.time() - self._step_start
            self.step_times.append(elapsed)
            alloc, _ = gpu_mem_gb()
            self.peak_gpu_gb = max(self.peak_gpu_gb, alloc)
            if state.global_step % self.logging_steps == 0:
                avg = sum(self.step_times[-self.logging_steps:]) / min(
                    len(self.step_times), self.logging_steps
                )
                print(
                    f"[TIMING] Step {state.global_step}: {elapsed:.2f}s "
                    f"(avg {avg:.2f}s) | GPU: {alloc:.1f} GB"
                )

    def on_evaluate(self, args, state, control, **kwargs):
        self._eval_start = time.time()

    def on_save(self, args, state, control, **kwargs):
        self._save_start = time.time()
        checkpoint_volume.commit()
        save_elapsed = time.time() - self._save_start
        self.save_times.append(save_elapsed)
        print(
            f"[TIMING] Checkpoint save + volume commit at step "
            f"{state.global_step}: {save_elapsed:.2f}s"
        )

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "eval_loss" in logs and self._eval_start:
            eval_elapsed = time.time() - self._eval_start
            self.eval_times.append(eval_elapsed)
            print(
                f"[TIMING] Eval at step {state.global_step}: "
                f"{eval_elapsed:.2f}s, eval_loss={logs['eval_loss']:.4f}"
            )
            self._eval_start = None

    def summary(self):
        return {
            "total_steps": len(self.step_times),
            "avg_step_sec": round(
                sum(self.step_times) / max(len(self.step_times), 1), 3
            ),
            "min_step_sec": round(min(self.step_times), 3) if self.step_times else 0,
            "max_step_sec": round(max(self.step_times), 3) if self.step_times else 0,
            "total_step_sec": round(sum(self.step_times), 2),
            "eval_count": len(self.eval_times),
            "total_eval_sec": round(sum(self.eval_times), 2),
            "save_count": len(self.save_times),
            "total_save_sec": round(sum(self.save_times), 2),
            "peak_gpu_gb": round(self.peak_gpu_gb, 1),
        }


# ---------------------------------------------------------------------------
# Main Training Function
# ---------------------------------------------------------------------------

@app.function(
    image=train_image,
    gpu=GPU_TYPE,
    volumes={
        "/model_cache": model_cache_volume,
        "/training_data": dataset_cache_volume,
        "/checkpoints": checkpoint_volume,
    },
    timeout=TIMEOUT_HOURS * 60 * 60,
    retries=0,
    single_use_containers=True,
)
def finetune(config: TrainingConfig):
    # Enable TF32 for faster matmul on Ampere+ GPUs (H200)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    timings = {}
    t_total = time.time()

    print(f"\n{'='*60}")
    print(f"LoRA Training: {config.experiment_name}")
    print(f"  Model: {config.model_name}")
    print(f"  Steps: {config.max_steps}, LR: {config.learning_rate}")
    print(f"  LoRA r={config.lora_r}, alpha={config.lora_alpha}")
    eff_batch = config.batch_size * config.gradient_accumulation_steps
    print(f"  Batch: {config.batch_size} x {config.gradient_accumulation_steps} = {eff_batch}")
    print(f"  load_in_4bit={config.load_in_4bit} (bf16 recommended for MoE)")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"  GPU: {torch.cuda.get_device_name(0)} ({props.total_memory / (1024**3):.0f} GB)")
    print(f"{'='*60}\n")

    # --- Capture environment for provenance ---
    env_info = {
        "python_version": sys.version.split()[0],
        "packages": {},
        "gpu": {},
        "cuda": {},
    }

    for pkg in ["torch", "transformers", "peft", "trl", "datasets", "accelerate", "bitsandbytes"]:
        try:
            env_info["packages"][pkg] = _importlib_metadata.version(pkg)
        except _importlib_metadata.PackageNotFoundError:
            pass

    if torch.cuda.is_available():
        env_info["gpu"]["name"] = torch.cuda.get_device_name(0)
        env_info["gpu"]["vram_gb"] = round(torch.cuda.get_device_properties(0).total_memory / (1024**3))
        env_info["cuda"]["version"] = torch.version.cuda or "unknown"
        env_info["cuda"]["cudnn"] = str(torch.backends.cudnn.version()) if torch.backends.cudnn.is_available() else "unknown"

    # Full pip freeze for artifact storage
    try:
        pip_freeze = subprocess.run(["pip", "freeze"], capture_output=True, text=True).stdout
        pip_freeze_sha256 = hashlib.sha256(pip_freeze.encode()).hexdigest()
    except Exception:
        pip_freeze = ""
        pip_freeze_sha256 = ""

    # Phase 1: Model load
    t0 = time.time()
    model, tokenizer = load_model(config)
    timings["model_load_sec"] = round(time.time() - t0, 2)
    alloc, res = gpu_mem_gb()
    print(f"[TIMING] Model load: {timings['model_load_sec']}s")
    print(f"[GPU] After model load: {alloc:.1f} GB allocated, {res:.1f} GB reserved")

    # Phase 2: Data load
    t0 = time.time()
    train_dataset, eval_dataset = load_jsonl_datasets(config, tokenizer)
    skip_eval = config.skip_eval or eval_dataset is None
    timings["data_load_sec"] = round(time.time() - t0, 2)
    print(f"[TIMING] Data load: {timings['data_load_sec']}s -- "
          f"{len(train_dataset)} train"
          f"{', ' + str(len(eval_dataset)) + ' valid' if eval_dataset else ''}")

    # Hash dataset files for provenance
    def sha256_file(path):
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    from pathlib import Path as _Path
    train_sha256 = sha256_file(_Path(DATA_DIR) / "train.jsonl")
    valid_sha256 = sha256_file(_Path(DATA_DIR) / "valid.jsonl") if (_Path(DATA_DIR) / "valid.jsonl").exists() else ""

    # Phase 3: LoRA setup
    t0 = time.time()
    custom_targets = config.lora_targets.split(",") if config.lora_targets else None
    model = setup_model_for_training(model, config, target_modules=custom_targets)
    timings["lora_setup_sec"] = round(time.time() - t0, 2)
    alloc, res = gpu_mem_gb()
    print(f"[TIMING] LoRA setup: {timings['lora_setup_sec']}s")
    print(f"[GPU] After LoRA: {alloc:.1f} GB allocated, {res:.1f} GB reserved")

    # Verify trainable params
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[PARAMS] {trainable_params:,} trainable / {total_params:,} total "
          f"({trainable_params / total_params * 100:.4f}%)")

    has_deltanet = any(
        "linear_attn" in n
        for n, p in model.named_parameters()
        if p.requires_grad
    )
    print(f"[CHECK] DeltaNet layers targeted: {'YES' if has_deltanet else 'MISSING'}")
    if not has_deltanet:
        print("[WARN] DeltaNet linear_attn layers NOT targeted by LoRA. "
              "Check if model architecture includes DeltaNet.")

    # Prepare checkpoint directory
    checkpoint_path = pathlib.Path("/checkpoints") / "experiments" / config.experiment_name
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    resume_from_checkpoint = check_for_existing_checkpoint(str(checkpoint_path))

    training_args = create_training_config(config, str(checkpoint_path), skip_eval)

    # Initialize timing callback
    timing_cb = TimingCallback(config.logging_steps)

    class _TimingCB(TrainerCallback):
        def on_step_begin(self, *a, **kw): timing_cb.on_step_begin(*a, **kw)
        def on_step_end(self, *a, **kw): timing_cb.on_step_end(*a, **kw)
        def on_evaluate(self, *a, **kw): timing_cb.on_evaluate(*a, **kw)
        def on_save(self, *a, **kw): timing_cb.on_save(*a, **kw)
        def on_log(self, *a, **kw): timing_cb.on_log(*a, **kw)

    callbacks = [_TimingCB()]

    print("Initializing SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
        callbacks=callbacks,
    )

    # Phase 4: Training
    t0 = time.time()
    print(f"\n[TRAINING] Starting -- {config.max_steps} steps...\n")
    if resume_from_checkpoint:
        print(f"Resuming training from {resume_from_checkpoint}")
        result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    else:
        print("Starting training from scratch...")
        result = trainer.train()
    timings["training_sec"] = round(time.time() - t0, 2)
    timings["training_loss"] = round(result.training_loss, 4)
    timings["training_steps"] = result.global_step
    alloc, res = gpu_mem_gb()
    print(f"[TIMING] Training: {timings['training_sec']}s ({timings['training_sec']/60:.1f} min)")
    print(f"[GPU] After training: {alloc:.1f} GB allocated, {res:.1f} GB reserved")

    # Phase 5: Save adapter
    t0 = time.time()
    final_model_path = checkpoint_path / "final_adapter"
    model.save_pretrained(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    timings["save_sec"] = round(time.time() - t0, 2)
    print(f"[TIMING] Save adapter: {timings['save_sec']}s")

    # Phase 6: Volume commit + metadata
    t0 = time.time()
    step_summary = timing_cb.summary()
    timings["step_timing"] = step_summary
    timings["total_sec"] = round(time.time() - t_total, 2)
    timings["total_min"] = round(timings["total_sec"] / 60, 2)
    timings["total_params"] = total_params
    timings["trainable_params"] = trainable_params
    timings["trainable_pct"] = round(trainable_params / total_params * 100, 4)

    # Compute adapter hash for provenance
    adapter_files = sorted(final_model_path.glob("*.safetensors"))
    adapter_sha256 = ""
    if adapter_files:
        h = hashlib.sha256()
        for af in adapter_files:
            h.update(af.read_bytes())
        adapter_sha256 = h.hexdigest()

    provenance = {
        "schema_version": "1.0.0",
        "model_id": config.experiment_name,
        "created_at": datetime.now().isoformat() + "Z",
        "description": f"LoRA adapter trained on {config.model_name}",

        "base_model": {
            "huggingface_id": config.model_name,
            "commit_sha": "",  # TODO: capture from HF hub API
        },

        "training": {
            "method": {
                "type": "LoRA-RFT",
                "lora_rank": config.lora_r,
                "lora_alpha": config.lora_alpha,
                "lora_target_modules": custom_targets or LORA_TARGET_MODULES,
                "lora_dropout": config.lora_dropout,
                "learning_rate": config.learning_rate,
                "lr_scheduler": config.lr_scheduler_type,
                "warmup_ratio": config.warmup_ratio,
                "max_steps": config.max_steps,
                "batch_size": config.batch_size,
                "gradient_accumulation_steps": config.gradient_accumulation_steps,
                "max_seq_length": config.max_seq_length,
                "optimizer": config.optim,
                "weight_decay": config.weight_decay,
                "bf16": torch.cuda.is_bf16_supported(),
            },
            "data": {
                "dataset_files": {
                    "train_file": "train.jsonl",
                    "train_sha256": train_sha256,
                    "train_examples": len(train_dataset),
                    "valid_file": "valid.jsonl",
                    "valid_sha256": valid_sha256,
                    "valid_examples": len(eval_dataset) if eval_dataset else 0,
                }
            },
            "infrastructure": {
                "provider": "Modal",
                "gpu": env_info["gpu"],
                "python_version": env_info["python_version"],
                "packages": env_info["packages"],
                "cuda": env_info["cuda"],
                "pip_freeze_sha256": pip_freeze_sha256,
                "random_seeds": {
                    "seed": config.seed,
                }
            },
            "run": {
                "training_script_repo": "https://gitlab.com/shanemmattner/llm-toolkit",
                "start_time": datetime.fromtimestamp(t_total).isoformat() + "Z",
                "end_time": datetime.now().isoformat() + "Z",
                "gpu_hours": round(timings["total_sec"] / 3600, 3),
                "final_train_loss": round(result.training_loss, 4),
                "total_steps": result.global_step,
            }
        },

        "artifacts": {
            "adapter_weights": str(final_model_path),
            "adapter_sha256": adapter_sha256,
            "pip_freeze": "pip_freeze.txt",
        },

        "notes": "",
    }

    # Save pip freeze as artifact
    if pip_freeze:
        with open(final_model_path / "pip_freeze.txt", "w") as f:
            f.write(pip_freeze)

    # Save provenance
    with open(final_model_path / "model_provenance.json", "w") as f:
        json.dump(provenance, f, indent=2, default=str)

    # Keep the existing metadata.json for backward compatibility
    metadata = {
        "experiment": config.experiment_name,
        "model": config.model_name,
        "config": config.__dict__,
        "results": {
            "train_loss": result.training_loss,
            "steps": result.global_step,
            "epoch": getattr(result, "epoch", None),
        },
        "timings": timings,
        "log_history": trainer.state.log_history,
    }
    with open(final_model_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    checkpoint_volume.commit()
    timings["commit_sec"] = round(time.time() - t0, 2)
    print(f"[TIMING] Final save + commit: {timings['commit_sec']}s")

    # Timing report
    print(f"\n{'='*60}")
    print(f"TIMING REPORT -- {config.experiment_name}")
    print(f"{'='*60}")
    print(f"  Model load:           {timings['model_load_sec']:>8.1f}s")
    print(f"  Data load:            {timings['data_load_sec']:>8.1f}s")
    print(f"  LoRA setup:           {timings['lora_setup_sec']:>8.1f}s")
    print(f"  Training ({timings['training_steps']} steps):  {timings['training_sec']:>8.1f}s")
    print(f"    avg step:           {step_summary['avg_step_sec']:>8.3f}s")
    print(f"    min/max step:       {step_summary['min_step_sec']:.3f}s / {step_summary['max_step_sec']:.3f}s")
    print(f"    evals ({step_summary['eval_count']}x):        {step_summary['total_eval_sec']:>8.1f}s")
    print(f"    saves ({step_summary['save_count']}x):        {step_summary['total_save_sec']:>8.1f}s")
    print(f"  Save adapter:         {timings['save_sec']:>8.1f}s")
    print(f"  Final commit:         {timings['commit_sec']:>8.1f}s")
    print(f"  ----------------------------------------")
    print(f"  TOTAL:                {timings['total_sec']:>8.1f}s ({timings['total_min']:.1f} min)")
    print(f"  Peak GPU:             {step_summary['peak_gpu_gb']:>8.1f} GB")
    print(f"  Final loss:           {timings['training_loss']}")
    print(f"  Trainable params:     {timings['trainable_params']:,} ({timings['trainable_pct']}%)")
    print(f"{'='*60}")

    # Cost estimate (H200 ~$4.76/hr on Modal)
    h200_per_sec = 4.76 / 3600
    est_cost = timings["total_sec"] * h200_per_sec
    print(f"\n  Estimated cost: ${est_cost:.2f} (H200 @ $4.76/hr)")

    print(f"\nDownload: modal volume get rft-checkpoints "
          f"experiments/{config.experiment_name}/final_adapter/ ./out/")
    return config.experiment_name


# ---------------------------------------------------------------------------
# Data Upload
# ---------------------------------------------------------------------------

@app.function(
    image=modal.Image.debian_slim(python_version="3.11"),
    volumes={DATA_DIR: dataset_cache_volume},
    timeout=300,
)
def _write_data(filename: str, content: bytes):
    p = pathlib.Path(DATA_DIR) / filename
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(content)
    print(f"  Wrote {p} ({len(content):,} bytes)")
    dataset_cache_volume.commit()


def _do_upload(local_dir: str):
    local_path = pathlib.Path(local_dir)
    if not local_path.exists():
        print(f"ERROR: {local_dir} not found")
        return
    uploaded = 0
    for fname in ["train.jsonl", "valid.jsonl"]:
        fpath = local_path / fname
        if fpath.exists():
            print(f"Uploading {fname} ({fpath.stat().st_size:,} bytes)...")
            _write_data.remote(fname, fpath.read_bytes())
            uploaded += 1
    print(
        f"\nUploaded {uploaded} file(s) to rft-training-data"
        if uploaded
        else "ERROR: No files found"
    )


# ---------------------------------------------------------------------------
# CLI Entrypoint
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(
    # Model
    model_name: str = "Qwen/Qwen3.6-35B-A3B",
    max_seq_length: int = 2048,
    load_in_4bit: bool = False,
    load_in_8bit: bool = False,
    # LoRA
    lora_r: int = 16,
    lora_alpha: int = 16,
    lora_dropout: float = 0.0,
    lora_bias: str = "none",
    use_rslora: bool = False,
    # Training
    optim: str = "adamw_8bit",
    batch_size: int = 4,
    gradient_accumulation_steps: int = 8,
    packing: bool = False,
    learning_rate: float = 2e-4,
    lr_scheduler_type: str = "cosine",
    warmup_ratio: float = 0.06,
    weight_decay: float = 0.01,
    max_steps: int = 150,
    save_steps: int = 50,
    eval_steps: int = 25,
    logging_steps: int = 5,
    # Experiment
    seed: int = 42,
    experiment_name: Optional[str] = None,
    skip_eval: bool = False,
    # Flags
    smoke_test: bool = False,
    upload_data: str = "",
    lora_targets: str = "",
):
    if upload_data:
        _do_upload(upload_data)
        return

    config = TrainingConfig(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
        load_in_8bit=load_in_8bit,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        lora_bias=lora_bias,
        use_rslora=use_rslora,
        optim=optim,
        batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        packing=packing,
        learning_rate=learning_rate,
        lr_scheduler_type=lr_scheduler_type,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        max_steps=max_steps,
        save_steps=save_steps,
        eval_steps=eval_steps,
        logging_steps=logging_steps,
        seed=seed,
        experiment_name=experiment_name,
        skip_eval=skip_eval,
        smoke_test=smoke_test,
        lora_targets=lora_targets or None,
    )

    print(f"Launching: {config.experiment_name} ({config.max_steps} steps, GPU: {GPU_TYPE})")
    print(f"Model: {config.model_name}")
    print(f"LoRA: rank={config.lora_r}, alpha={config.lora_alpha}")
    print(f"Effective batch size: {config.batch_size * config.gradient_accumulation_steps}")

    experiment_name = finetune.remote(config)
    print(f"\nDone: {experiment_name}")
    print(f"Download: modal volume get rft-checkpoints experiments/{experiment_name}/final_adapter/ ./out/")
