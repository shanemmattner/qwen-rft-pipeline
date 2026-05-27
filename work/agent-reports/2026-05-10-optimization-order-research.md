# Optimization Order of Operations: Research Report
> Qwen3.6-27B Dense on Mac Studio M4 Max 128GB

## 1. Evidence For/Against the Hypothesized Order

### The Hypothesis: Prune -> Fine-tune -> Merge -> Quantize -> Graft MTP -> (Optional) Recalibrate MTP

### 1.1 Pruning Before vs After Fine-Tuning

**Evidence FOR pruning first (our hypothesis):**

- **P-pruning (LREC-COLING 2024)**: Explicitly titled "Pruning before Fine-tuning: A Retraining-free Compression Framework." Demonstrates that pruning redundant modules *before* fine-tuning reduces fine-tuning costs and produces comparable quality. The key insight: redundant layers in the base model remain redundant after fine-tuning — you don't need fine-tuning signal to identify them. [Source: aclanthology.org/2024.lrec-main.1162]

- **E3-Pruner (arXiv 2511.17205)**: The framework we plan to use operates in two phases — a *searching stage* (identifies which layers to prune via differentiable Gumbel-TopK) followed by a *fine-tuning stage* (recovers quality post-pruning). This is explicitly a prune-then-fine-tune architecture. The paper's design assumes you prune first and fine-tune after. [Source: arxiv.org/abs/2511.17205]

- **Compute efficiency**: Pruning 12-25% of layers means fine-tuning runs on a smaller model, saving 12-25% of fine-tuning compute. For a 64-layer model like Qwen3.6-27B, removing 8-16 layers is significant.

- **Gabe Ortiz layer surgery (March 2026)**: Pruned 2 layers (28-29) from Qwen3-Coder-30B-A3B and saw a 123% improvement on coding benchmarks. Key finding: some layers *actively interfere* with coding performance. If we're optimizing for coding workflows, pruning the right layers first means fine-tuning trains on a cleaner base. [Source: gabeortiz.net/posts/2026-3-21-llm-layer-surgery]

**Evidence AGAINST pruning first:**

- **Fine-tuning reveals load-bearing layers**: Some researchers argue that fine-tuning on your target domain first helps identify which layers are actually important for your task. A layer that's "redundant" on general benchmarks might be critical for coding. However, E3-Pruner's searching stage addresses this by using task-relevant data during the search.

- **LoRAPrune (ACL Findings 2024)**: Proposes simultaneous pruning and fine-tuning. At 50% compression, achieves 4.81 lower perplexity than LLM-Pruner on WikiText2. The argument: pruning and fine-tuning interact, so doing them jointly finds a better optimum than sequentially. [Source: arxiv.org/abs/2305.18403]

- **Simultaneous Fine-Tuning and Pruning (OPT-ML 2025)**: Claims 1.88x faster inference at 50% pruning with joint constrained optimization. The sequential two-stage approach is suboptimal because pruning decisions made without fine-tuning context may remove layers that fine-tuning would have made important. [Source: openreview.net/forum?id=1orrQ3lYBW]

**VERDICT**: Pruning first is supported by E3-Pruner's own design, compute efficiency arguments, and P-pruning's results. The joint approaches (LoRAPrune, SPP) are theoretically superior but require specialized tooling we'd need to build. **Stick with prune-first for pragmatic reasons**, but use E3-Pruner's task-aware searching stage with coding-relevant calibration data.

### 1.2 Quantization Before or After Fine-Tuning

**Evidence for quantize-AFTER fine-tuning (our hypothesis):**

- **Systematic Study of Compression Ordering (arXiv 2511.19495)**: This is the most directly relevant paper. It systematically tests all orderings of pruning, quantization, and distillation on LLMs. Key findings:
  - Quantization provides the greatest standalone compression with least quality degradation
  - **Ordering matters**: applying quantization after pruning and fine-tuning preserves more quality than quantizing first
  - The best ordering for multi-technique compression: **prune -> distill/fine-tune -> quantize**
  [Source: arxiv.org/abs/2511.19495]

- **Prune-then-Quantize vs Quantize-then-Prune (arXiv 2603.18426, OpenReview)**: Proposes the "Progressive Intensity Rule" — apply the less destructive compression first. Since pruning is more destructive than quantization, you'd think quantize-first. BUT: the paper finds that prune-then-quantize gives better results when fine-tuning is in the pipeline, because fine-tuning can recover from pruning damage but not from quantization artifacts. [Source: openreview.net/forum?id=KWtOTMMvKU]

**Evidence for quantize-BEFORE fine-tuning (QLoRA approach):**

- **QLoRA**: Quantize base model to 4-bit, then fine-tune with LoRA adapters in FP16. The adapter weights remain full-precision, so fine-tuning quality is comparable to full-precision LoRA. Memory savings are dramatic (4x). [Source: multiple]

- **However**: QLoRA trains LoRA weights against quantized base weights. When you merge and re-quantize, you're double-quantizing. The standard approach is fine-tune on full-precision, merge, then quantize once. This avoids compounding quantization error.

**VERDICT**: **Quantize last**. The systematic study (2511.19495) is clear: quantization should be the final compression step. QLoRA is a memory-saving trick for training, not a quality-optimal ordering. Since we have 128GB on Mac Studio (plenty for fine-tuning 27B with LoRA), we don't need QLoRA's memory savings.

### 1.3 MTP Head Compatibility After Fine-Tuning

**Critical findings:**

- **Reddit: "Quick hack to recover Qwen3.5 MTP after fine-tuning" (r/LocalLLaMA)**: Community has confirmed that fine-tuning the trunk causes MTP head degradation. The hack: graft the *base model's* MTP heads back onto the fine-tuned trunk. This works as a "practical shortcut" but acceptance rates drop compared to matched heads. [Source: reddit.com/r/LocalLLaMA/comments/1sfsxv2]

- **AEON-7 Qwen3.6-27B variants**: The AEON team grafts "original Qwen/Qwen3.6-27B MTP head in BF16 (bit-exact)" onto their fine-tuned + quantized variants. This confirms the community pattern: graft base MTP heads, accept some acceptance-rate loss. [Source: github.com/AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-DFlash]

- **FastMTP (arXiv 2509.18362)**: Proposes fine-tuning a *single shared-weight MTP head* via self-distillation. Achieves 2.03x speedup. Key insight: you only need to retrain ONE MTP head (not all of them), and self-distillation data is free (just run the trunk). This is the recalibration method we should use. [Source: arxiv.org/abs/2509.18362]

- **vLLM MTP acceptance rates**: Reports show 61-85% acceptance rates for quantized models vs 89% baseline. Fine-tuning without MTP recalibration likely pushes acceptance rates lower. [Source: discuss.vllm.ai]

- **MTPLX (youssofal)**: Achieves 2.24x speedup (28 -> 63 tok/s) on Qwen3.6-27B on M5 Max. Uses the model's own built-in MTP heads. No mention of pruned model support in the README. [Source: github.com/youssofal/MTPLX]

**VERDICT**: MTP heads from the base model CAN be grafted onto fine-tuned trunks (AEON proves this). Acceptance rate degrades but remains useful. Recalibration via FastMTP's self-distillation method is the right way to recover performance and should be treated as **recommended, not optional**.

### 1.4 Layer Pruning + MTP Interaction

**Critical concern:**

- **SDFP (arXiv 2602.05499)**: "Speculative Decoding with FIT-Pruned Models." Uses pruned models as draft models for speculative decoding. Key: layer pruning *preserves architectural compatibility* for speculative decoding. The pruned model has fewer layers but the same hidden dimensions, so the final hidden state is compatible with MTP heads. [Source: arxiv.org/abs/2602.05499]

- **However**: MTP heads in Qwen3.6-27B are trained against the output of the *final layer* of a 64-layer model. If we remove layers, the final-layer representations shift. The MTP heads will see different input distributions.

- **MTPLX specifics**: MTPLX uses the model's built-in MTP heads. It reads the model config to determine architecture. If we prune layers, we must update the config to reflect the new layer count. MTPLX should work if the config is correct, but acceptance rates will degrade because the MTP heads were trained for 64 layers, not 48-56.

- **No one has tested MTPLX + pruned models**: There is no evidence in the community of anyone running MTPLX on a layer-pruned model. This is uncharted territory.

**VERDICT**: Layer pruning changes the hidden state distribution at the final layer. MTP heads WILL need recalibration after pruning. The question is whether grafted base MTP heads work *at all* on a pruned trunk, or whether acceptance rates drop so low that speculative decoding provides no benefit. **This must be validated empirically before committing to the full pipeline.**

### 1.5 Simultaneous Pruning + Fine-Tuning

- **LoRAPrune**: Joint structured pruning + LoRA fine-tuning. Superior to sequential at high compression rates (50%). At our target (12-25%), the advantage is smaller.
- **SPP (Simultaneous Fine-Tuning and Pruning)**: Uses constrained optimization. 1.88x inference speedup at 50% pruning. Requires custom training loop.

**VERDICT**: Joint approaches are better in theory but require specialized implementations. E3-Pruner's sequential approach is well-tested and available. For our 12-25% pruning target, the quality difference between joint and sequential is likely small. **Use E3-Pruner's sequential approach.**

### 1.6 Real-World Examples

- **AEON-7 pipeline**: Fine-tune Qwen3.6-27B -> quantize to NVFP4 -> graft base MTP heads in BF16. No pruning step. Working in production on RTX 5090. [Source: GitHub AEON-7]
- **Gabe Ortiz**: Prune Qwen3-Coder layers -> evaluate. No fine-tuning or MTP. [Source: gabeortiz.net]
- **No one has done the full prune -> fine-tune -> quantize -> MTP pipeline on Qwen3.6**. We would be first.

---

## 2. Revised Recommended Order

The evidence **supports our original hypothesis with one modification**: MTP recalibration should be treated as recommended rather than optional.

### Final Order:

```
1. PRUNE        — E3-Pruner, remove 12-25% of layers (8-16 of 64)
                  Use coding-relevant calibration data for the searching stage
                  
2. FINE-TUNE    — LoRA on pruned trunk
                  Domain: TunedVoice coding tasks, Q&A knowledge
                  Full-precision base weights (no QLoRA)
                  
3. MERGE        — Merge LoRA adapters into trunk
                  Produces a standalone fine-tuned, pruned model
                  
4. QUANTIZE     — 6-bit MLX conversion
                  This is the final weight-modifying step
                  
5. GRAFT MTP    — Attach base Qwen3.6-27B MTP heads (BF16)
                  Keep MTP heads in full precision (following AEON pattern)
                  
6. RECALIBRATE  — FastMTP-style self-distillation on the MTP heads
   MTP            Train single shared-weight head against pruned+fine-tuned trunk
                  RECOMMENDED (not optional) for pruned models
```

### Why this order works:

| Step | Rationale |
|------|-----------|
| Prune first | E3-Pruner designed for this. Saves fine-tuning compute. Removes interfering layers before training. |
| Fine-tune second | Trains on clean (pruned) architecture. Recovers quality lost to pruning. |
| Merge third | Eliminates adapter overhead. Produces single-checkpoint model. |
| Quantize fourth | Systematic study (2511.19495) confirms: quantize last minimizes quality loss. |
| Graft fifth | MTP heads are architecture-independent (only read final hidden state). Keep BF16. |
| Recalibrate sixth | Pruning + fine-tuning both shift hidden state distributions. MTP heads must be recalibrated. |

---

## 3. Risk Assessment

| Step | Risk | Severity | Mitigation |
|------|------|----------|------------|
| 1. Prune | Remove a load-bearing layer for coding | HIGH | Use coding-relevant calibration data in E3-Pruner's search. Start conservative (12%). Eval on coding benchmarks after. |
| 1. Prune | E3-Pruner doesn't support Qwen3.6 architecture | MEDIUM | Verify architecture compatibility before starting. May need minor config changes. |
| 2. Fine-tune | LoRA doesn't recover pruning quality loss | MEDIUM | Compare pruned+fine-tuned vs unpruned+fine-tuned on coding tasks. If gap > 5%, reduce pruning ratio. |
| 2. Fine-tune | 128GB insufficient for full-precision LoRA on 27B | LOW | 27B in BF16 = ~54GB. LoRA adds <1GB. Optimizer states ~2x adapter size. Total ~58GB. Well within 128GB. |
| 3. Merge | Merged model quality differs from adapter model | LOW | Standard operation, well-tested. Verify with eval after merge. |
| 4. Quantize | 6-bit quantization degrades coding quality | MEDIUM | 6-bit is conservative (4-bit shows 2-5% degradation). 6-bit typically < 1% loss. Test on coding benchmarks. |
| 5. Graft MTP | Base MTP heads incompatible with pruned+fine-tuned trunk | HIGH | **This is the highest-risk step.** No one has done this on a pruned model. Test acceptance rate immediately after grafting. If < 50%, MTP provides no benefit. |
| 5. Graft MTP | MTPLX crashes with pruned model config | MEDIUM | MTPLX reads model config for layer count. Must update config after pruning. May need MTPLX source modifications. |
| 6. Recalibrate | FastMTP self-distillation fails on pruned architecture | MEDIUM | FastMTP was designed for standard architectures. May need adaptation for pruned models. Fallback: train MTP head from scratch (more compute). |

### Fallback Plan

If the full pipeline fails at any step:

1. **Prune fails** -> Skip pruning, proceed with fine-tune -> merge -> quantize -> graft MTP (this is the AEON pipeline, known to work)
2. **Fine-tune fails** -> Evaluate pruned-only model. If coding quality is sufficient, proceed to quantize + MTP
3. **MTP graft fails** -> Ship without MTP. Still get pruning + fine-tuning + quantization benefits. Lose 2x speculative speedup.
4. **MTP recalibration fails** -> Use base MTP heads without recalibration (AEON approach). Accept lower acceptance rates (estimated 50-70% instead of 85%+).

---

## 4. Specific Qwen3.6-27B Considerations

### Architecture
- **64 layers**, hidden dimension 5120, 248320 token embeddings (padded)
- Dense architecture (no MoE routing complexity)
- Ships with native MTP heads (confirmed by MTPLX and AEON)
- 27B parameters in BF16 = ~54GB VRAM

### Pruning Targets
- 12% = 8 layers removed -> ~23.8B params -> ~47.6GB BF16, ~17.8GB 6-bit
- 25% = 16 layers removed -> ~20.3B params -> ~40.6GB BF16, ~15.2GB 6-bit
- At 6-bit quantization after 25% pruning, model fits comfortably in 128GB with room for KV cache + MTP heads

### Memory Budget (Mac Studio M4 Max 128GB)
| Component | 6-bit, 12% pruned | 6-bit, 25% pruned |
|-----------|-------------------|-------------------|
| Model weights | ~17.8 GB | ~15.2 GB |
| MTP heads (BF16) | ~0.5 GB | ~0.5 GB |
| KV cache (32K ctx) | ~8 GB | ~7 GB |
| MTPLX overhead | ~2 GB | ~2 GB |
| **Total** | **~28.3 GB** | **~24.7 GB** |
| **Headroom** | **~100 GB** | **~103 GB** |

This is very comfortable. Could even consider 8-bit quantization for better quality if speed is acceptable.

### Coding-Specific Layer Analysis
- Per Gabe Ortiz: Qwen3 models have specific layers that *interfere* with coding. His finding on Qwen3-Coder (layers 28-29) may not directly transfer to Qwen3.6-27B (different architecture), but the principle holds: run E3-Pruner's search with coding calibration data to find coding-harmful layers.

---

## 5. MTP Compatibility Matrix

| Trunk State | MTP Head Source | Expected Acceptance Rate | Tested By | Notes |
|-------------|-----------------|--------------------------|-----------|-------|
| Base (unmodified) | Native (built-in) | 85-89% | MTPLX, vLLM | Baseline |
| Fine-tuned only | Native (base) grafted | 60-75% (estimated) | AEON-7, Reddit | Community-confirmed pattern |
| Fine-tuned only | Recalibrated (FastMTP) | 80-85% (estimated) | FastMTP paper | Self-distillation recovery |
| Quantized only (NVFP4) | Native (base) BF16 | 61-85% | vLLM benchmarks | Quantization causes moderate degradation |
| Quantized only (6-bit MLX) | Native (base) BF16 | 70-85% (estimated) | Not directly tested | 6-bit less destructive than 4-bit |
| Fine-tuned + quantized | Native (base) BF16 | 55-70% (estimated) | AEON-7 (indirect) | Compound degradation |
| **Pruned + fine-tuned + quantized** | **Native (base) BF16** | **40-60% (unknown)** | **No one** | **Highest risk — uncharted** |
| **Pruned + fine-tuned + quantized** | **Recalibrated (FastMTP)** | **65-80% (hoped)** | **No one** | **Our target** |

**Key insight**: Every modification to the trunk degrades MTP acceptance rates. Pruning is the most destructive for MTP because it changes the number of transformer blocks the hidden state passes through. Recalibration is essential.

### MTPLX Compatibility Checklist
- [x] Works with quantized models (confirmed by AEON NVFP4 variants)
- [x] Works with fine-tuned models (confirmed by community hack)
- [ ] Works with pruned models (UNCONFIRMED — must test)
- [ ] Works with MLX 6-bit quantization specifically (UNCONFIRMED — MTPLX uses MLX natively, likely works)
- [ ] Config correctly updated after layer pruning (MUST DO — MTPLX reads num_hidden_layers from config)

---

## 6. Updated Wave Plan

### Wave 0: Validation (MUST DO FIRST)
**Goal**: Confirm pruning + MTP is viable before committing to full pipeline.

| Step | Action | Time | Notes |
|------|--------|------|-------|
| 0.1 | Download Qwen3.6-27B base weights | 30 min | ~54GB download |
| 0.2 | Run E3-Pruner search stage with coding calibration data | 2-4 hours | Identifies which layers to prune |
| 0.3 | Prune 12% (8 layers) — minimal pruning | 30 min | Conservative start |
| 0.4 | Quantize pruned model to 6-bit MLX | 30 min | Quick quantization |
| 0.5 | Graft base MTP heads onto pruned+quantized model | 30 min | Update config.json num_hidden_layers |
| 0.6 | Test MTPLX with pruned model | 1 hour | **GO/NO-GO gate**: If acceptance rate < 50%, abandon pruning path |
| 0.7 | Evaluate coding quality on pruned model | 1 hour | Compare to unpruned baseline |

**Estimated total**: 1 day
**Decision gate**: If MTP acceptance rate >= 50% AND coding quality within 90% of baseline, proceed to Wave 1.

### Wave 1: Pruning + Fine-Tuning
| Step | Action | Time | Notes |
|------|--------|------|-------|
| 1.1 | Finalize pruning ratio based on Wave 0 results | 1 hour | May adjust from 12% to 20-25% |
| 1.2 | Prepare fine-tuning dataset (TunedVoice coding + Q&A) | 2-4 hours | Dataset curation |
| 1.3 | Fine-tune with LoRA on pruned model | 4-8 hours | Full-precision, LoRA rank 16-64 |
| 1.4 | Evaluate fine-tuned model on coding benchmarks | 2 hours | Compare to pruned-only and baseline |
| 1.5 | Merge LoRA adapters into trunk | 30 min | Standard merge |

**Estimated total**: 2-3 days

### Wave 2: Quantization + MTP
| Step | Action | Time | Notes |
|------|--------|------|-------|
| 2.1 | Quantize merged model to 6-bit MLX | 30 min | Final weight-modifying step |
| 2.2 | Evaluate quantized model quality | 1 hour | Verify coding quality preserved |
| 2.3 | Graft base MTP heads (BF16) | 30 min | Same as Wave 0.5 |
| 2.4 | Test MTPLX acceptance rate | 1 hour | Compare to Wave 0.6 |
| 2.5 | Run FastMTP self-distillation on MTP heads | 2-4 hours | Recalibrate against fine-tuned trunk |
| 2.6 | Final MTPLX acceptance rate test | 1 hour | Target: 65%+ acceptance |
| 2.7 | End-to-end performance benchmarks | 2 hours | tok/s, quality, memory usage |

**Estimated total**: 2 days

### Wave 3: Production Deploy
| Step | Action | Time | Notes |
|------|--------|------|-------|
| 3.1 | Package final model for Mac Studio MLX server | 1 hour | Config, model files |
| 3.2 | Integration test with TunedVoice pipeline | 2 hours | End-to-end coding workflow |
| 3.3 | Deploy to Mac Studio MLX server | 1 hour | Update launchd config |
| 3.4 | Monitor production performance | Ongoing | Acceptance rates, tok/s, quality |

**Estimated total**: 1 day

### Total Timeline: ~6-7 days of compute time

---

## Sources

1. **Systematic Study of Compression Ordering** — Chhawri & Mahadik, arXiv 2511.19495
2. **E3-Pruner** — arXiv 2511.17205 (the pruning framework we'll use)
3. **Prune-then-Quantize or Quantize-then-Prune** — arXiv 2603.18426, OpenReview
4. **P-pruning: Pruning before Fine-tuning** — ACL LREC-COLING 2024
5. **LoRAPrune** — ACL Findings 2024, arXiv 2305.18403
6. **Simultaneous Fine-Tuning and Pruning** — OPT-ML 2025
7. **FastMTP** — arXiv 2509.18362
8. **SDFP: Speculative Decoding with FIT-Pruned Models** — arXiv 2602.05499
9. **Fit-LoRA** — OpenReview (training-free LoRA transfer to pruned models)
10. **MTPLX** — github.com/youssofal/MTPLX
11. **AEON-7 Qwen3.6-27B variants** — github.com/AEON-7
12. **Gabe Ortiz layer surgery** — gabeortiz.net/posts/2026-3-21-llm-layer-surgery
13. **Reddit: Recover Qwen3.5 MTP after fine-tuning** — r/LocalLLaMA
14. **Qwen3.6-27B model card** — huggingface.co/Qwen/Qwen3.6-27B

---

## Learnings

### New facts discovered:
1. The systematic compression ordering paper (2511.19495) confirms prune -> fine-tune -> quantize is optimal
2. AEON-7 has already demonstrated MTP head grafting onto fine-tuned+quantized Qwen3.6-27B in production
3. FastMTP's self-distillation can recalibrate MTP heads with just the trunk model (no external data needed)
4. No one has tested MTPLX with layer-pruned models — this is our biggest unknown
5. Qwen3.6-27B has 64 layers with hidden dim 5120 — at 25% pruning (48 layers), model drops to ~20.3B params
6. vLLM reports 61-85% MTP acceptance on quantized models vs 89% baseline
7. Gabe Ortiz found coding-harmful layers in Qwen3 architecture — E3-Pruner with coding calibration data should find these

### Reference doc updates needed:
- MTPLX compatibility notes should be added to the workstream doc
- FastMTP as the recalibration method should be documented
- AEON-7 as a reference implementation for MTP grafting

### Agent prompt corrections:
- None identified — this was a research task with clear search instructions
