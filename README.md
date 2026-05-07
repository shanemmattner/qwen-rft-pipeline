# Rejection Fine-Tuning Pipeline for MoE Models

A practical pipeline for fine-tuning Mixture-of-Experts language models on their own filtered outputs. Tested on Qwen3.6-35B-A3B (35B total params, 3B active) using Apple Silicon for generation and Modal cloud GPUs for training. Total cloud cost: ~$7.

**Important note on terminology**: This pipeline is *inspired by* the Self-play Self-Distillation (SSD) paper ([arXiv 2604.01193](https://arxiv.org/abs/2604.01193)), but deviates from it in a key way. Pure SSD explicitly uses **no filtering or verification** of generated outputs. We filter for correctness (execution + test pass), reducing 2,000 generated samples to 1,796. This makes our approach **Rejection Fine-Tuning (RFT) on self-generated data** rather than true SSD.

## Key Discovery: DeltaNet LoRA Targets

Qwen3.6 uses a hybrid **Gated DeltaNet** architecture (30 linear attention layers + 10 standard attention layers + MoE). Standard LoRA target modules (`q_proj`, `k_proj`, `v_proj`, `o_proj`) only reach the 10 standard attention layers, resulting in ~0.02% trainable parameters and NaN training loss.

You must target the DeltaNet-specific projection layers. See [configs/qwen36-35b-a3b.yaml](configs/qwen36-35b-a3b.yaml) for the full list of working LoRA targets.

## Requirements

- **Generation**: Apple Silicon Mac with 32GB+ unified memory (M4 Max 128GB used here)
- **Training**: [Modal](https://modal.com) account (H200 GPU, ~$4.76/hr)
- **Software**: Python 3.11+, `modal` CLI, `mlx_lm` (for local generation/serving)
- **Accounts**: HuggingFace (model download), Modal (cloud training)
- **Cost**: ~$6 training + ~$1 merge = **~$7 total cloud spend** (generation is free on local hardware)

## Pipeline

### Step 1: Generate Samples Locally

Generate coding solutions from the base model using MLX on Apple Silicon. This step runs locally and costs nothing.

```bash
# Serve the base model locally
mlx_lm.server --model mlx-community/Qwen3.6-35B-A3B-6bit --port 8800

# Generate solutions (your own script — not included here)
# Output: raw JSONL with model-generated code solutions
# We generated 2,000 samples at temp=1.6, top_k=20, top_p=0.8
# Took ~10.6 hours on M4 Max 128GB
```

### Step 2: Filter and Format Training Data

Filter generated solutions for correctness (execution + test pass) and format as chat-template JSONL.

```
# Input:  raw generated solutions
# Filter: execute each solution, run tests, keep only passing ones
# Output: train.jsonl + valid.jsonl in chat message format
#
# Our results: 2,000 -> 1,796 survivors (89.8% pass rate)
#   Split: 1,616 train / 180 validation
#
# Each line in the JSONL must have a "messages" field:
# {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

### Step 3: Train LoRA on Modal (~$6)

Upload training data and run LoRA fine-tuning on an H200 GPU.

```bash
# Upload training data to Modal volume
modal run modal_train.py --upload-data /path/to/training_data/

# Run training (smoke test first)
modal run modal_train.py --smoke-test

# Run full training
modal run modal_train.py

# Training takes ~78 minutes on H200
# Final loss: ~0.46
```

### Step 4: Merge Adapter into Base Weights (~$1)

Merge the LoRA adapter back into the full base model weights.

```bash
# Merge adapter (runs on CPU, no GPU needed)
modal run modal_merge.py --experiment <experiment-name>

# Download merged model (use the script, not raw `modal volume get`)
./download_merged.sh <experiment-name>
./download_merged.sh <experiment-name> ./my-merged-model/  # custom destination
```

### Step 5: Convert to MLX and Test

Convert the merged bf16 model to MLX format for local inference, then test.

```bash
# Convert to MLX 4-bit on Apple Silicon
python3 -m mlx_lm.convert \
    --hf-path ./merged-bf16 \
    --mlx-path ./merged-mlx-4bit \
    --quantize --q-bits 4

# Serve the merged model
mlx_lm.server --model ./merged-mlx-4bit --port 8803

# Run the test suite (speed benchmark + quality pass-rate)
./test_merged_model.sh http://localhost:8803

# Save results, swap to base model, run again, then compare
mv results_*.json results_merged.json
# (restart server with base model on same port)
./test_merged_model.sh http://localhost:8803
mv results_*.json results_base.json

# Compare
./test_merged_model.sh --compare results_base.json results_merged.json
```

## DeltaNet LoRA Target Reference

These are the parameter name patterns that produce a working LoRA configuration for Qwen3.6-35B-A3B (0.45% trainable parameters, stable training loss):

| Component | Target Modules | Layer Count |
|-----------|---------------|-------------|
| DeltaNet linear attention | `linear_attn.in_proj_qkv`, `linear_attn.in_proj_z`, `linear_attn.out_proj` | 30 layers |
| Standard self-attention | `self_attn.q_proj`, `self_attn.k_proj`, `self_attn.v_proj`, `self_attn.o_proj` | 10 layers |
| MoE switch experts | `mlp.switch_mlp.gate_proj`, `mlp.switch_mlp.up_proj`, `mlp.switch_mlp.down_proj` | 256 experts x 40 layers |
| MoE shared expert | `mlp.shared_expert.gate_proj`, `mlp.shared_expert.up_proj`, `mlp.shared_expert.down_proj` | 40 layers |

**What does NOT work**: Targeting only `q_proj`, `k_proj`, `v_proj`, `o_proj` (standard LoRA defaults). These only match the 10 standard attention layers, giving ~0.02% trainable parameters and NaN loss during training.

**What we skipped**: `linear_attn.in_proj_a` and `linear_attn.in_proj_b` (DeltaNet gating projections). The model has 12 possible LoRA targets per layer — we used 10 of 12. We haven't tested whether adding these two would help.

### How We Found These Keys

This wasn't planned. We assumed Qwen3.6 was a standard transformer and started with the LoRA targets every tutorial uses. Here's what actually happened:

1. **Attempt 1**: Standard targets (`q_proj`, `k_proj`, `v_proj`, `o_proj`). Result: 0.02% trainable params, NaN loss on the first step. These keys only exist in 10 of the 40 layers — the other 30 are DeltaNet and use completely different parameter names.

2. **Discovery**: Printed the model's full parameter tree (`model.named_parameters()`) and saw names like `linear_attn.in_proj_qkv` where we expected `self_attn.q_proj`. That's when we realized 75% of the layers aren't standard transformers at all.

3. **~15 smoke test runs on Modal**: Tried different combinations of DeltaNet keys, with and without MLP targets, different ranks. Each smoke test (30 iterations) takes a few minutes and costs pennies. We were debugging, not doing controlled ablations — just trying to get loss to stop being NaN.

4. **Working config**: Added `in_proj_qkv`, `in_proj_z`, `out_proj` for DeltaNet layers + standard attention keys + MLP expert keys. Trainable params jumped to 0.055%, loss dropped immediately on the first step. Shipped it.

We don't know which individual keys contribute most. A proper ablation study (testing subsets of targets) would be a good follow-up — smoke tests are cheap enough to do it.

## Cost Breakdown

| Step | Resource | Time | Cost |
|------|----------|------|------|
| Generation | M4 Max 128GB (local) | 10.6 hours | $0 |
| Training | Modal H200 | 78 min | ~$6.20 |
| Merge | Modal CPU (64GB) | ~15 min | ~$0.80 |
| Convert to MLX | Apple Silicon (local) | ~10 min | $0 |
| **Total** | | | **~$7** |

## Results

On a 13-problem coding benchmark (10 samples each, temp=0.7):

| Model | Pass Rate |
|-------|-----------|
| Base (Qwen3.6-35B-A3B) | 126/130 (96.9%) |
| Merged (RFT fine-tuned) | 128/130 (98.5%) |

**Assessment**: The +1.5% delta is **not statistically significant** with this sample size. The pipeline works end-to-end and produces a functional merged model, but we cannot claim quality improvement from this experiment. A larger eval set and/or multiple training runs would be needed to measure a real effect.

## Model Architecture Reference

Qwen3.6-35B-A3B:
- **Total parameters**: 35B (3B active per token due to MoE)
- **Architecture**: Hybrid Gated DeltaNet
  - 30 layers with DeltaNet linear attention (subquadratic)
  - 10 layers with standard grouped-query attention
  - All 40 layers have MoE FFN (256 experts, 8 active per token + 1 shared expert)
- **Training dtype**: bfloat16 (QLoRA/4-bit not recommended for MoE models)

## Training Hyperparameters

| Parameter | Value |
|-----------|-------|
| LoRA rank (r) | 16 |
| LoRA alpha | 16 |
| Max steps | 150 |
| Batch size | 4 |
| Gradient accumulation | 8 |
| Effective batch size | 32 |
| Learning rate | 2e-4 |
| LR scheduler | cosine |
| Warmup | 6% of steps |
| Optimizer | AdamW 8-bit |
| Max sequence length | 2048 |

## Credits

- **SSD paper**: "Embarrassingly Simple Self-Distillation" ([arXiv 2604.01193](https://arxiv.org/abs/2604.01193)) for the core idea of training on self-generated outputs. Our pipeline deviates by filtering for correctness (making it RFT rather than pure SSD).
- **Modal**: Cloud GPU infrastructure for training and merging.
- **MLX**: Apple's framework for efficient local inference on Apple Silicon.
- **kreuzhofer's DGX Spark script**: Reference for MoE-specific training workarounds (bf16 over QLoRA, vision-language tokenizer handling).

## License

Apache 2.0 — see [LICENSE](LICENSE).
