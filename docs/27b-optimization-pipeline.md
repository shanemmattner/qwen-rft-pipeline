# Qwen3.6-27B Optimization Pipeline
> Base model to production-optimized coding assistant on Mac Studio M4 Max 128GB

**Goal**: Beat our A3B baseline (0.620 mean score / 81% pass rate / ~52 tok/s) with the 27B dense model's superior reasoning quality, while matching or exceeding A3B inference speed.

**Target end state**: ~50-55 tok/s, ~15-18 GB model size, domain-tuned for TunedVoice coding workflows.

**Total estimated timeline**: 6-7 days of compute time across all waves.

---

## Architecture Overview

```
Qwen3.6-27B BF16 (64 layers, 5120 hidden dim, ~54 GB)
    |
    v
[Wave 0] Validation Gate --- FAIL ---> AEON fallback (skip pruning)
    |
    v (PASS)
[Wave 1] E3-Pruner: remove 6-8 layers (12-17%)
    |
    v
[Wave 2] LoRA fine-tuning on pruned trunk (RFT + Q&A domain)
    |
    v
[Wave 3] Merge LoRA adapters into trunk
    |
    v
[Wave 4] Quantize to 6-bit MLX
    |
    v
[Wave 5] Graft MTP sidecar from Youssofal model
    |
    v
[Wave 6] FastMTP self-distillation recalibration
    |
    v
[Deploy] Rapid-MLX on Mac Studio via launchd
```

---

## Wave 0: Validation Gate

**Purpose**: Confirm that pruning + MTP is viable before committing to the full pipeline. No one has tested MTPLX with a layer-pruned model -- this is uncharted territory.

**Duration**: 1 day
**Cost**: $0 (all local on Mac Studio)

| Step | Action | Time | Details |
|------|--------|------|---------|
| 0.1 | Download `Qwen/Qwen3.6-27B` BF16 base | 30 min | ~54 GB. Source of truth for pruning analysis and MTP heads. |
| 0.2 | Download `Youssofal/Qwen3.6-27B-MTPLX-Optimized` | 15 min | ~20 GB. Reference for MTP sidecar format and `mtplx_runtime.json` config. |
| 0.3 | Run E3-Pruner BI (Block Influence) analysis on base model | 2-4 hrs | Use coding-relevant calibration data, NOT generic C4. Identifies layer redundancy scores. |
| 0.4 | Prune 2-3 layers as a minimal test | 30 min | Conservative test -- just enough to validate the pipeline works mechanically. |
| 0.5 | Quantize pruned test model to 6-bit MLX | 30 min | `mlx_lm.convert` with `--q-bits 6`. |
| 0.6 | Graft MTP sidecar onto pruned model | 30 min | Copy `mtp.safetensors` + `mtplx_runtime.json` from Youssofal model. Update `config.json` `num_hidden_layers` to reflect pruned layer count. |
| 0.7 | Test MTPLX with pruned model | 1 hr | Measure MTP acceptance rate and tok/s. |
| 0.8 | Eval coding quality on pruned model | 1 hr | Run against our 49 TunedVoice eval fixtures. |

### Decision Gate

| Metric | Threshold | Action if FAIL |
|--------|-----------|----------------|
| MTP acceptance rate | >= 50% | If < 50%, skip pruning entirely. Follow AEON pipeline: fine-tune -> merge -> quantize -> graft MTP (proven to work). |
| Coding quality | >= 90% of base score | If < 90% on just 2-3 pruned layers, pruning is too destructive for this model. Fall back to AEON pipeline. |
| MTPLX crash/incompatibility | Must not crash | If MTPLX cannot parse pruned model config, check if source modifications are needed. If unfixable, skip pruning. |

### Risk Assessment

| Risk | Severity | Mitigation |
|------|----------|------------|
| MTPLX crashes with pruned model config | MEDIUM | MTPLX reads `num_hidden_layers` from config. Must update config after pruning. May need MTPLX source patches. |
| E3-Pruner doesn't support Qwen3.6-27B architecture | MEDIUM | Verify architecture compatibility first. Qwen3.6-27B has 64 layers (not 48 as some docs state -- verify from model card). May need minor config changes. |
| MTP acceptance too low on pruned trunk | HIGH | This is the whole reason Wave 0 exists. The MTP heads were trained for 64 layers; removing layers shifts the final hidden state distribution. |

---

## Wave 1: Pruning

**Purpose**: Remove redundant layers to reduce model size and increase inference speed.

**Duration**: 1-2 days
**Cost**: $0 (local compute)
**Prerequisite**: Wave 0 passes decision gate.

| Step | Action | Time | Details |
|------|--------|------|---------|
| 1.1 | Full E3-Pruner analysis | 4-6 hrs | Identify 6-8 redundant layers (12-17% of 64). Use E3-Pruner's differentiable Gumbel-TopK searching stage with coding calibration data. |
| 1.2 | Select pruning ratio based on Wave 0 + full analysis | 1 hr | Conservative: 6-8 layers (12-17%). Aggressive: 10-12 layers (up to 19%). Do NOT go to 25% on first attempt. |
| 1.3 | Prune layers | 30 min | Remove identified layers. Update all config files (num_hidden_layers, etc). |
| 1.4 | Quick eval on pruned base (no fine-tuning) | 2 hrs | Run against 49 TunedVoice fixtures. This establishes the pruning quality baseline before fine-tuning recovers it. |
| 1.5 | Verify quality retention | 30 min | Target: 96%+ of base model score. If below 93%, reduce pruning ratio and repeat. |

### Expected Outcomes

| Pruning Level | Layers Removed | Params | BF16 Size | 6-bit Size | Quality Impact |
|--------------|----------------|--------|-----------|------------|----------------|
| 12% | 8 of 64 | ~23.8B | ~47.6 GB | ~17.8 GB | < 2% drop (E3-Pruner data) |
| 17% | 11 of 64 | ~22.2B | ~44.4 GB | ~16.6 GB | ~2-3% drop |
| 25% | 16 of 64 | ~20.3B | ~40.6 GB | ~15.2 GB | ~3-5% drop |

### Which Layers Get Removed

- **Middle-to-later layers are most redundant** -- high cosine similarity between layer input/output (ShortGPT finding).
- Protect early layers (L0-L3) and final layers -- these carry the most information.
- Per Gabe Ortiz: some layers actively *interfere* with coding performance. Layers 28-29 in Qwen3-Coder showed 123% improvement when removed. E3-Pruner with coding calibration data will find the equivalent layers in Qwen3.6-27B.

### Calibration Warning

Qwen3 models are sensitive to pruning search algorithm choice. CBO and fast-block-select produce "reconstruction error explosion" on Qwen3. **E3-Pruner's approach is the safest** -- do not substitute other pruning frameworks.

### Risk Assessment

| Risk | Severity | Mitigation |
|------|----------|------------|
| Remove a load-bearing layer for coding | HIGH | Use coding-relevant calibration data in E3-Pruner search. Start conservative (12%). Eval on coding benchmarks after each ratio. |
| Quality drop exceeds 5% | MEDIUM | Reduce pruning ratio. At worst, fall back to AEON pipeline (no pruning). |

---

## Wave 2: Fine-Tuning

**Purpose**: Recover quality lost to pruning AND add domain-specific knowledge (TunedVoice codebase, coding workflow tasks).

**Duration**: 2-3 days
**Cost**: ~$18-30 on Modal A100 (down from $125-172 before optimizations)
**Prerequisite**: Wave 1 complete with acceptable quality retention.

### Adapter Strategy

Single mixed-data run combining all categories (eliminates second training job):

| Data Category | % of Dataset | Source |
|---------------|-------------|--------|
| Q&A | 40% | 521 Q&A pairs (468 train / 53 valid) from bug-qa, crossfile-qa, deepseek-v4-qa |
| FIM (fill-in-middle) | 30% | Code completion examples |
| Commit-to-diff | 15% | Commit message to diff pairs |
| Bug detection | 10% | Bug identification examples |
| RFT-filtered | 5% | Generated samples filtered for correctness (temp=1.6, top_k=20, top_p=0.8) |

### LoRA Configuration (27B Dense)

```yaml
# 7 targets -- simpler than A3B's 10 (no DeltaNet/MoE)
target_modules:
  - "q_proj"
  - "k_proj"
  - "v_proj"
  - "o_proj"
  - "gate_proj"
  - "up_proj"
  - "down_proj"

# Optimized config (r=64 is sufficient for ~2K examples per LoRA Without Regret capacity analysis)
rank: 64
alpha: 128
dropout: 0.0
bias: none
base_precision: QLoRA 4-bit  # enables A100 80GB ($2.50/hr) instead of H200 ($4.76/hr)
```

### Training Pipeline

| Step | Action | Time | Cost | Details |
|------|--------|------|------|---------|
| 2.1 | 30-step ablation (3 configs) | 45 min | ~$3 | Config A: r=32/alpha=64/lr=2e-4. Config B: r=64/alpha=128/lr=2e-4. Config C: r=64/alpha=128/lr=1e-4. All QLoRA 4-bit on A100. |
| 2.2 | Select winning config | 15 min | $0 | Pick lowest eval_loss. |
| 2.3 | Single mixed-data training run | 3-4 hrs | ~$12-20 | ~120 steps with early stopping, eval every 10 steps. QLoRA 4-bit on A100 ($2.50/hr). |
| 2.4 | Evaluate adapter | 1 hr | $0 | Run against 49 fixtures with adapter loaded. |
| 2.5 | Merge + validation | 1-2 hrs | ~$3-5 | Merge adapter, verify quality matches adapter-loaded model. |

### Training Hyperparameters (optimized from A3B config + cost research)

```yaml
max_steps: 120          # ~2 epochs for 2K examples, with early stopping
batch_size: 4
gradient_accumulation_steps: 8  # effective batch: 32
learning_rate: 2.0e-4   # sweep 1e-4 vs 2e-4 in ablation
lr_scheduler_type: cosine
warmup_steps: 0.06      # fraction
weight_decay: 0.01
optimizer: adamw_8bit
max_seq_length: 2048
packing: false
seed: 42
eval_steps: 10          # early stopping when eval_loss plateaus for 20+ steps
```

### Key Differences from A3B Fine-Tuning

| Aspect | A3B | 27B Dense |
|--------|-----|-----------|
| Architecture | Hybrid (DeltaNet + MoE) | Standard transformer |
| LoRA targets | 10 (DeltaNet + attention + MLP) | 7 (attention + MLP) |
| QLoRA viable? | No (unreliable for MoE) | Yes -- cuts memory from ~54 GB to ~15 GB |
| Base precision | bf16 only (~67 GB on H200) | bf16 (~54 GB) or QLoRA (~15 GB) |
| Training complexity | High (NaN loss with wrong targets) | Low (standard transformer) |

### Decision: QLoRA 4-bit on A100

QLoRA is the right choice for 27B dense (unlike A3B where MoE routing made QLoRA unreliable):
- <2% quality loss vs full-precision LoRA (Dettmers et al. 2023, multiple 2025 benchmarks).
- Reduces base model memory from ~54 GB to ~15 GB, enabling A100 80GB ($2.50/hr) instead of H200 ($4.76/hr).
- 39% slower per step due to quant/dequant overhead, but ~55% lower total cost per run.
- Double-quantization concern (QLoRA + later 6-bit deployment) is mitigated: fine-tuning operates on the adapter in full precision, and the base model is re-quantized cleanly for deployment.

### Risk Assessment

| Risk | Severity | Mitigation |
|------|----------|------------|
| LoRA doesn't recover pruning quality loss | MEDIUM | Compare pruned+fine-tuned vs unpruned+fine-tuned. If gap > 5%, reduce pruning ratio. |
| 521 Q&A pairs insufficient for 27B dense | LOW | 27B base is already stronger than A3B -- needs less data to steer. Can expand to 1000+ pairs if needed (Claude generates questions, local model answers, Claude verifies). |
| Training data leaks to public repo | HIGH | qwen-rft-pipeline is public. Training data with TunedVoice code must NEVER be committed to this repo. Use Modal volumes or private storage. |

---

## Wave 3: Merge

**Purpose**: Merge LoRA adapter(s) into pruned trunk to produce a standalone model.

**Duration**: 2-4 hours
**Cost**: $0

| Step | Action | Time | Details |
|------|--------|------|---------|
| 3.1 | Merge LoRA adapter into pruned trunk | 30 min | Linear merge (base + alpha * adapter). Use `modal_merge.py` or `peft.merge_and_unload()`. |
| 3.2 | Verify merged model quality | 1-2 hrs | Run 49 fixtures against merged model. Compare to adapter-loaded model -- scores should be identical or near-identical. |
| 3.3 | Save merged safetensors | 30 min | Produces standalone checkpoint (~44-48 GB bf16 depending on pruning ratio). |

### Merge Method Decision

| Method | Use When | Notes |
|--------|----------|-------|
| **Linear merge** | Single adapter or sequential stack | Default. Simple, reliable. |
| TIES | Multiple adapters with conflicting updates | Trims + elects signs. More complex. |
| DARE | Multiple adapters, want diversity | Randomly drops adapter deltas. |

For the initial pipeline: **linear merge**. If running multi-adapter (RFT + Q&A stacked), merge sequentially or use mlx-optiq adapter stacking at inference (no merge needed).

### Risk Assessment

| Risk | Severity | Mitigation |
|------|----------|------------|
| Merged model quality differs from adapter model | LOW | Standard operation, well-tested. Verify with eval after merge. |
| MTP heads dropped during merge | EXPECTED | PEFT merge strips MTP weights via `_keys_to_ignore_on_load_unexpected`. This is expected -- MTP heads are grafted back in Wave 5. |

---

## Wave 4: Quantize

**Purpose**: Compress merged model to 6-bit for efficient inference on Apple Silicon.

**Duration**: 2-4 hours
**Cost**: $0

| Step | Action | Time | Details |
|------|--------|------|---------|
| 4.1 | Convert to 6-bit MLX | 30 min | `mlx_lm.convert --hf-path <merged-model> --q-bits 6`. This strips MTP weights (expected). |
| 4.2 | Verify quantized quality | 1-2 hrs | Run 49 fixtures. 6-bit typically shows < 1% quality loss vs bf16. |
| 4.3 | Verify model size | 5 min | Target: ~15-18 GB depending on pruning ratio. |

### Size Projections

| Pruning Level | bf16 Size | 6-bit Size | 8-bit Size |
|--------------|-----------|------------|------------|
| 12% pruned | ~47.6 GB | ~17.8 GB | ~23.8 GB |
| 17% pruned | ~44.4 GB | ~16.6 GB | ~22.2 GB |
| 25% pruned | ~40.6 GB | ~15.2 GB | ~20.3 GB |

### Why 6-bit, Not 4-bit

- 4-bit shows 2-5% quality degradation on coding tasks.
- 6-bit typically < 1% degradation.
- Memory budget is not tight (see below) -- no need to sacrifice quality for size.
- 8-bit is also viable if speed is acceptable, for even better quality.

### Memory Budget (Mac Studio M4 Max 128GB)

| Component | 6-bit, 12% pruned | 6-bit, 25% pruned |
|-----------|-------------------|-------------------|
| Model weights | ~17.8 GB | ~15.2 GB |
| MTP heads (BF16) | ~0.5 GB | ~0.5 GB |
| KV cache (32K ctx) | ~8 GB | ~7 GB |
| MTPLX overhead | ~2 GB | ~2 GB |
| OS + apps | ~15-20 GB | ~15-20 GB |
| **Total** | **~43-48 GB** | **~40-45 GB** |
| **Headroom** | **~80-85 GB** | **~83-88 GB** |

Plenty of room for additional services, embedding models, or larger KV caches.

### Risk Assessment

| Risk | Severity | Mitigation |
|------|----------|------------|
| 6-bit degrades coding quality | LOW | 6-bit is conservative. If quality drops, try 8-bit (still fits easily). |
| Quantization interacts poorly with pruning | LOW | Systematic study (arxiv 2511.19495) shows quantize-last is optimal. Pruning + fine-tuning have already stabilized weights before quantization. |

---

## Wave 5: MTP Graft

**Purpose**: Attach MTP speculative decoding heads to enable 2x+ decode speedup via MTPLX.

**Duration**: 2-4 hours
**Cost**: $0

| Step | Action | Time | Details |
|------|--------|------|---------|
| 5.1 | Extract MTP sidecar from Youssofal model | 15 min | Copy `mtp.safetensors` + `mtplx_runtime.json` from `Youssofal/Qwen3.6-27B-MTPLX-Optimized`. |
| 5.2 | Graft onto fine-tuned trunk | 30 min | Place sidecar files alongside quantized trunk weights. Update config if needed. |
| 5.3 | Test MTPLX acceptance rate | 1 hr | `mtplx start` and measure acceptance rate. Baseline expectation for pruned+fine-tuned+quantized trunk: 40-60% (degraded from 97.6% base). |
| 5.4 | Benchmark tok/s | 1 hr | Target: 50+ tok/s. Compare to 20 tok/s baseline (mlx_lm.server, 6-bit, no MTP). |

### The 19 MTP Tensors

```
model.mtp_block.0.mtp_proj.weight              # projects prior-layer hidden states
model.mtp_block.0.block.self_attn.q_proj.weight # attention Q projection
model.mtp_block.0.block.self_attn.k_proj.weight # attention K projection
model.mtp_block.0.block.self_attn.v_proj.weight # attention V projection
model.mtp_block.0.block.self_attn.o_proj.weight # attention O projection
model.mtp_block.0.block.self_attn.q_proj.bias   # (if present)
model.mtp_block.0.block.self_attn.k_proj.bias   # (if present)
model.mtp_block.0.block.self_attn.v_proj.bias   # (if present)
model.mtp_block.0.block.self_attn.o_proj.bias   # (if present)
model.mtp_block.0.block.mlp.gate_proj.weight    # MLP gate
model.mtp_block.0.block.mlp.up_proj.weight      # MLP up
model.mtp_block.0.block.mlp.down_proj.weight    # MLP down
model.mtp_block.0.block.input_layernorm.weight  # pre-attention norm
model.mtp_block.0.block.post_attention_layernorm.weight  # post-attention norm
model.mtp_block.0.enorm.weight                  # embedding norm
model.mtp_block.0.hnorm.weight                  # hidden state norm
lm_head.weight                                  # shared with MTP output (tied)
```

### MTP Acceptance Rate Expectations

| Trunk State | MTP Source | Expected Acceptance | Tested By |
|-------------|-----------|---------------------|-----------|
| Base (unmodified) | Native (built-in) | 85-89% | MTPLX, vLLM |
| Fine-tuned only | Base grafted | 60-75% | AEON-7, Reddit |
| Fine-tuned + quantized | Base grafted (BF16) | 55-70% | AEON-7 (indirect) |
| **Pruned + FT + quantized** | **Base grafted** | **40-60% (unknown)** | **No one** |
| **Pruned + FT + quantized** | **Recalibrated (FastMTP)** | **65-80% (hoped)** | **No one** |

### Risk Assessment

| Risk | Severity | Mitigation |
|------|----------|------------|
| Base MTP heads incompatible with pruned trunk | HIGH | This is the highest-risk step. Layer pruning changes final hidden state distribution. If acceptance < 30%, MTP provides no benefit and should be skipped. |
| MTPLX crashes with pruned model config | MEDIUM | Must update `num_hidden_layers` in config. May need MTPLX source patches. |
| MTP head precision mismatch with trunk | LOW | Youssofal's MTP sidecar is CyanKiwi-calibrated INT4. Our trunk is 6-bit. Mismatch may cause issues -- may need to re-quantize MTP heads to 6-bit or keep in BF16. |

---

## Wave 6: MTP Recalibration

**Purpose**: Improve MTP acceptance rate by recalibrating heads against the fine-tuned trunk via self-distillation.

**Duration**: 1-2 days
**Cost**: $0 (self-distillation uses the trunk itself, no external data)

| Step | Action | Time | Details |
|------|--------|------|---------|
| 6.1 | Set up FastMTP self-distillation | 2 hrs | FastMTP (arxiv 2509.18362) trains a single shared-weight MTP head against the trunk's own output distribution. No external dataset needed. |
| 6.2 | Run self-distillation | 4-8 hrs | Generates training signal by running the trunk, trains MTP head to predict the trunk's next-token distribution. |
| 6.3 | Replace grafted MTP heads with recalibrated ones | 30 min | Package as new `mtp.safetensors`. |
| 6.4 | Re-measure acceptance rate | 1 hr | Should be higher than Wave 5.3 (grafted). Target: 65%+ for pruned model, 75%+ for unpruned. |
| 6.5 | Final benchmark | 2 hrs | Full performance sweep: tok/s, TTFT, quality on 49 fixtures. |

### Why Recalibration Matters

Every modification to the trunk degrades MTP acceptance:
- **Pruning** is the most destructive (changes number of transformer blocks the hidden state passes through)
- **Fine-tuning** shifts the output distribution the MTP heads were trained on
- **Quantization** introduces noise in the hidden states

Without recalibration, compound degradation could push acceptance below useful thresholds. FastMTP's self-distillation is the lowest-effort way to recover.

### Fallback

If FastMTP self-distillation fails on pruned architecture:
1. Use base MTP heads without recalibration (AEON approach). Accept lower acceptance rates (50-70%).
2. Train MTP head from scratch (more compute, but guaranteed compatibility).
3. Ship without MTP entirely. Still get pruning + fine-tuning + quantization benefits.

### Risk Assessment

| Risk | Severity | Mitigation |
|------|----------|------------|
| FastMTP designed for standard architectures, may not work on pruned | MEDIUM | FastMTP only needs the trunk's forward pass. Pruned model still has a valid forward pass -- should work. |
| Self-distillation compute too expensive | LOW | FastMTP trains ONE head, not a full model. Should be feasible on Mac Studio. |

---

## Post-Pipeline: Production Deployment

**Duration**: 1 day
**Cost**: $0

| Step | Action | Time | Details |
|------|--------|------|---------|
| P.1 | Package final model | 1 hr | Trunk weights + MTP sidecar + config files. Upload to HuggingFace as `shaneMattner/qwen36-27b-tuned-v1`. |
| P.2 | Deploy on Mac Studio via Rapid-MLX | 1 hr | Update launchd plist to point to new model. Rapid-MLX handles KV caching and prompt cache. |
| P.3 | Final eval against A3B baseline | 2 hrs | Side-by-side comparison on 49 TunedVoice fixtures. |
| P.4 | Monitor production performance | Ongoing | Track acceptance rates, tok/s, quality regressions. |

### Success Criteria

| Metric | A3B Baseline | 27B Target | Verdict |
|--------|-------------|------------|---------|
| Mean eval score | 0.620 | > 0.620 | 27B dense should be stronger on coding benchmarks |
| Pass rate | 81% | > 81% | Denser model = better reasoning |
| Decode speed | ~52 tok/s | > 50 tok/s | MTPLX + pruning should close the gap |
| Model size | ~7 GB (4-bit A3B) | ~15-18 GB (6-bit pruned 27B) | Larger but within budget |
| TTFT (cached) | -- | < 0.5s | Rapid-MLX prompt cache |
| Memory usage | ~12 GB | ~28-45 GB | Within 128 GB budget |

---

## Full Pipeline Summary

| Wave | Duration | Cost | Key Output | Decision Gate |
|------|----------|------|------------|---------------|
| 0: Validation | 1 day | $0 | GO/NO-GO on pruning + MTP | Acceptance >= 50%, quality >= 90% of base |
| 1: Pruning | 1-2 days | $0 | Pruned model (6-8 layers removed) | Quality >= 96% of base |
| 2: Fine-tuning | 2-3 days | ~$18-30 | Single mixed-data adapter (QLoRA 4-bit, r=64, A100) | Eval loss converging, fixture pass rate improving |
| 3: Merge | 2-4 hrs | $0 | Standalone fine-tuned pruned model | Merged == adapter-loaded quality |
| 4: Quantize | 2-4 hrs | $0 | 6-bit MLX model (~15-18 GB) | < 1% quality loss from quantization |
| 5: MTP Graft | 2-4 hrs | $0 | Model with MTP speculative decoding | Acceptance >= 40% |
| 6: Recalibration | 1-2 days | $0 | Recalibrated MTP heads | Acceptance improved over Wave 5 |
| Deploy | 1 day | $0 | Production model on Mac Studio | Beats A3B baseline |
| **Total** | **~6-7 days** | **~$18-30** | | |

### Fallback Chain

If the pipeline fails at any point, fall back to the nearest working configuration:

1. **Wave 0 fails (pruning + MTP incompatible)** -> Skip pruning. Follow AEON pipeline: fine-tune base 27B -> merge -> quantize -> graft MTP. Proven to work.
2. **Wave 1 fails (pruning too destructive)** -> Use unpruned 27B. Larger model but still benefits from fine-tuning + MTP.
3. **Wave 2 fails (fine-tuning doesn't help)** -> Use pruned-only model. 27B base is already strong.
4. **Wave 5 fails (MTP graft unusable)** -> Ship without MTP. Lose 2x decode speedup but keep pruning + fine-tuning benefits. ~27 tok/s instead of 50+.
5. **Wave 6 fails (recalibration doesn't improve)** -> Use grafted base MTP heads without recalibration (AEON approach). Accept lower acceptance rates.

---

## Stacking Techniques Beyond the Core Pipeline

The core pipeline (prune -> fine-tune -> merge -> quantize -> MTP) handles model optimization. These additional techniques stack on top for further quality and speed gains in production.

### RAG (Already Built)

- **LanceDB** vector store with tree-sitter code chunking
- **Hybrid search**: vector similarity + keyword matching
- **Purpose**: Inject relevant code context into prompts without relying solely on model memorization
- **Status**: Production-ready, used in TunedVoice coding workflows

### Code Indexing with Embeddings

- **Model**: Qwen3-Embedding-0.6B-8bit on Mac Studio port 8001
- **Purpose**: Semantic code search across TunedVoice codebase
- **Complements RAG**: Embedding-based retrieval finds semantically similar code even when keywords don't match

### KV Cache Prefix Sharing

- **Structured prompts for prefix reuse**: Design system prompt as a stable prefix across all workflow phases. Only user query changes per request.
- **oMLX SSD cache**: Block-based KV cache with prefix sharing and copy-on-write. Persists across requests.
- **Rapid-MLX prompt cache**: Sub-100ms cached TTFT on repeated prefixes.
- **Impact**: In a 6-call workflow, eliminates re-encoding 2K-12K tokens per call. Total savings: ~30K tokens of redundant prefill.

### Multi-Adapter Hot-Swap

- **mlx-optiq** (v0.0.8+): Mount N adapters on one base model, switch per-request via ContextVar. No model reload.
- **Phase-specific adapters**: Different adapter for code generation vs code review vs dictation correction.
- **Swap latency**: Near-zero (pointer reassignment, not model reload). Only adapter weights (~50-200 MB) change.

### SwiftSyntax Pre-Validation

- **swift-syntax-check CLI**: Validates generated Swift code before presenting to user.
- **Latency**: 5-16ms per check.
- **Purpose**: Catch syntax errors immediately. Reject and regenerate if invalid.
- **Integration point**: After each code generation response, before presenting to user.

### Structured Output / Constrained Decoding

- **Outlines**: Grammar-constrained generation for JSON, structured responses.
- **XGrammar**: Alternative constrained decoding framework.
- **Purpose**: Ensure model output matches expected schema (JSON for tool calls, structured code blocks for generation tasks).

### Context Engineering

- **CONTEXT.md glossary**: Standardized terminology for TunedVoice codebase. Reduces ambiguity in prompts.
- **Doubt-driven development**: Structured doubt loop for reviewing generated code before accepting.
- **Prompt structure**: System prompt carries code context (file contents), user message carries the question. Assistant response is the answer. This is the training data format and should match inference format.

---

## The Full Stack for Coding Ability

Each layer of the stack contributes independently. Together they compound:

| Layer | What | How It Helps | Status |
|-------|------|-------------|--------|
| 1. Base model quality | 27B dense > A3B for coding benchmarks | Stronger reasoning, better code generation | Available (Qwen3.6-27B) |
| 2. Domain knowledge | Q&A fine-tuning on TunedVoice codebase | Model knows the codebase architecture, patterns, common bugs | 521 pairs ready, pipeline built |
| 3. Task specialization | RFT on coding workflow tasks | Model outputs match expected coding workflow format | Pipeline built, ablation needed for 27B |
| 4. Speed | MTPLX + pruning + KV caching | 50+ tok/s effective throughput | MTPLX available, pruning to validate |
| 5. Context quality | RAG + code index + prefix sharing | Right code context in every prompt | RAG built, embeddings deployed |
| 6. Output validation | SwiftSyntax + swift build | Catch errors before user sees them | swift-syntax-check built (5-16ms) |
| 7. Multi-adapter routing | Right adapter for right task phase | Specialized quality per workflow phase | mlx-optiq available, adapters to train |

### Quality vs Speed Tradeoff

```
                    Quality
                      ^
                      |
    27B+FT+RAG  *     |
                      |
    27B base    *     |    <-- Our target zone:
                      |        high quality + high speed
    A3B+FT      *     |
                      |
    A3B base    *     |
                      |
                      +----------------------------> Speed
                     20    30    40    50    60
                                          tok/s

    27B current:  ~20 tok/s, higher quality than A3B
    A3B current:  ~52 tok/s, 0.620 mean score
    27B target:   ~50 tok/s, > 0.620 mean score (pruned + MTPLX + fine-tuned)
```

---

## Key References

| Source | Relevance |
|--------|-----------|
| [Systematic Compression Ordering (arxiv 2511.19495)](https://arxiv.org/abs/2511.19495) | Confirms prune -> fine-tune -> quantize is optimal |
| [E3-Pruner (arxiv 2511.17205)](https://arxiv.org/abs/2511.17205) | Pruning framework, tested on Qwen3-32B |
| [FastMTP (arxiv 2509.18362)](https://arxiv.org/abs/2509.18362) | MTP recalibration via self-distillation |
| [AEON-7 Qwen3.6-27B](https://github.com/AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-DFlash) | Reference for MTP grafting on fine-tuned+quantized models |
| [MTPLX (GitHub)](https://github.com/youssofal/MTPLX) | MTP speculative decoding for Apple Silicon |
| [Gabe Ortiz Layer Surgery](https://gabeortiz.net/posts/2026-3-21-llm-layer-surgery) | Coding-harmful layers in Qwen3 architecture |
| [P-pruning (LREC-COLING 2024)](https://aclanthology.org/2024.lrec-main.1162) | Pruning before fine-tuning framework |
| [Youssofal/Qwen3.6-27B-MTPLX-Optimized](https://huggingface.co/Youssofal/Qwen3.6-27B-MTPLX-Optimized) | MTP sidecar reference model |
| [mlx-optiq](https://mlx-optiq.com/) | Multi-adapter hot-swap for MLX |

---

## Open Questions

1. **E3-Pruner on Qwen3.6-27B**: Does E3-Pruner work out of the box, or does it need architecture adaptation? The paper tested on Qwen3-32B (64 layers). Qwen3.6-27B layer count needs verification from model card.
2. **MTPLX + pruned models**: No one has tested this. Wave 0 exists to answer this question.
3. **MTP sidecar precision matching**: Youssofal's MTP sidecar is CyanKiwi INT4. Our trunk will be 6-bit. Does precision mismatch cause issues?
4. **FastMTP on Mac Studio**: Can self-distillation run locally on M4 Max 128GB, or does it need GPU?
5. **Compound speedup validation**: MTPLX (2.2x) + 25% pruning (1.33x) = theoretical ~2.9x. Real-world compound effect may be lower.
6. **521 Q&A pairs enough for 27B?**: Larger model may need more data to steer, or may need less because it's already stronger at the base level.
