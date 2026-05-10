# Codebase-Specific Fine-Tuning Research Report

**Date**: 2026-05-10
**Context**: Qwen3.6-27B fine-tuning for TunedVoice (Swift/SwiftUI macOS push-to-talk dictation app). Current state: 468 training + 53 validation Q&A pairs, LoRA r=16, H200 GPU on Modal.

---

## 1. Codebase-Specific vs Language-Specific Fine-Tuning

### Evidence

**The narrow-only approach fails.** A HuggingFace discussion thread on fine-tuning LLMs on proprietary codebases reports that training on isolated code files with auto-generated Q&A pairs produced "responses that are largely irrelevant, even when asked questions directly taken from the training dataset." The poster abandoned fine-tuning in favor of RAG for code retrieval.

**Apple's UICoder approach is instructive.** Apple researchers fine-tuned StarChat-Beta (which had virtually zero SwiftUI training data) by generating ~996,000 synthetic SwiftUI programs through an iterative self-improvement loop with compiler validation. The result "significantly outperformed the base model" and nearly matched GPT-4 on SwiftUI tasks, exceeding GPT-4's compilation success rate. Key: they used compilation as a filter, not just LLM-judged quality.

**Mix general + domain data to prevent catastrophic forgetting.** Multiple sources confirm that LoRA fine-tuning preserves general knowledge better than full fine-tuning (0.7149 vs 0.7497 BERTScore F1, with LoRA retaining general knowledge while full fine-tuning lost it). Data mixing is the primary defense: blend domain-specific examples with general-purpose data (function calling, structured outputs, general Q&A). No single "correct" ratio exists, but the research suggests starting at 70:30 domain:general and tuning from there.

### Recommendation for TunedVoice

**Train primarily on TunedVoice codebase, but mix in 20-30% general Swift/SwiftUI code.** The base Qwen3.6 model already knows Swift syntax. What it lacks is TunedVoice-specific patterns (DictationManager lifecycle, Parakeet TDT integration, FluidAudio SDK). General Swift data prevents forgetting; codebase-specific data teaches patterns.

---

## 2. Training Data Types That Produce Best Coding Ability

### Evidence by Data Type

| Data Type | Effectiveness | Why |
|-----------|--------------|-----|
| **Q&A pairs with code context** | Good for reasoning about code | Teaches "why" behind patterns. You have 521 of these. |
| **Code completion (fill-in-middle)** | Best for generation tasks | Directly trains the generation behavior you want at inference. Industry standard for code models. |
| **Commit message → diff pairs** | Good for understanding changes | Teaches modification patterns, but requires git history processing. |
| **Architecture documentation** | Moderate | Helps with high-level reasoning but doesn't directly improve code generation. Better served by RAG. |
| **Code review / bug detection** | Good for quality | Teaches what NOT to do. High signal if you have real review data. |
| **Issue → fix pairs** | Good for real-world tasks | Closest to actual developer workflow. High value but hard to generate at scale. |

### Key Finding

The HuggingFace discussion and Databricks guide both emphasize: **mix training formats.** Don't use only Q&A pairs. Blend Q&A with fill-in-the-middle and code completion samples. The Databricks guide specifically states to include "complete, syntactically valid code samples rather than isolated snippets."

Sebastian Raschka's research shows that a curated **1,000-example dataset (LIMA) matched or exceeded a 50,000-example synthetic dataset (Alpaca)** when using optimal hyperparameters. Quality dominates quantity, especially at small scale.

### Recommendation

**Expand from 521 Q&A pairs to ~2,000-3,000 total examples** in this mix:
- 40% Q&A pairs about architecture/patterns (expand current 521)
- 30% fill-in-the-middle code completion (new — generate from codebase)
- 15% commit-diff pairs (extract from git history)
- 10% bug detection / code review pairs
- 5% general Swift examples (forgetting prevention)

---

## 3. Infrastructure vs UI vs Business Logic

### Evidence

The research is clear: **train on ALL code the model will encounter, but weight toward what it will generate.** Apple's UICoder approach showed that UI code specifically requires specialized training because "examples of UI code are extremely rare" in general training data, "making up less than one percent" of typical datasets.

### Recommendation

**One adapter, all code types, but weighted.** Don't train separate adapters for UI vs networking vs build system. Instead:
- Include all code types in training data
- Over-represent the types you'll ask the model to generate most (likely SwiftUI views + audio pipeline code)
- Under-represent build system / SPM config (the model will rarely generate these; RAG handles lookup)

---

## 4. Multi-Adapter vs Single Adapter

### Evidence

The multi-LoRA literature describes four composition approaches: task-specific switching, concurrent blending, adapter merging, and sequential stacking. These are useful for genuinely distinct domains (medical + legal + coding). For sub-domains within a single codebase, multi-adapter adds complexity without clear benefit.

**For your use case (one codebase, one language, one domain), a single adapter is correct.** Multi-adapter makes sense when the adapter domains are orthogonal (e.g., Swift coding adapter + Japanese translation adapter). SwiftUI views, networking code, and audio pipeline code within TunedVoice share too much context to benefit from separation.

### Recommendation

**One fat adapter.** Multi-adapter specialization is overkill for a single-codebase, single-language scenario. Your training data naturally covers different code domains, and the LoRA will learn the relevant patterns across all of them.

If you later add entirely different workflows (e.g., a separate adapter for commit message generation vs code generation), multi-adapter becomes relevant. Not now.

---

## 5. RAG + Fine-Tuning Interaction

### Evidence (Key Paper)

The 2025 paper "RAG or Fine-tuning? A Comparative Study on LCMs-based Code Completion in Industry" (arXiv 2505.15179) provides the strongest evidence:

| Approach | Exact Match | Improvement |
|----------|------------|-------------|
| Baseline | 24.78% | — |
| Fine-tuning | 44.20% | +78.3% |
| RAG (BM25) | 53.76% | +116.9% |
| **FT + RAG combined** | **57.43%** | **+131.7%** |

**Critical findings:**
- RAG alone outperforms fine-tuning alone for code completion
- **Combining both yields synergistic gains** (7.79% EM improvement over RAG alone)
- RAG scales better: at 120K files, RAG gained 2.26% EM while fine-tuning gained only 0.35%
- Fine-tuning hits a plateau; RAG shows sustained improvement with more data
- RAG preparation: 2 minutes (BM25). Fine-tuning preparation: 41.4 hours

**The roles are complementary:**
- **Fine-tuning teaches style, patterns, and conventions** — how TunedVoice code "feels"
- **RAG provides specific facts** — exact API signatures, current file contents, type definitions
- Fine-tuning reduces RAG's burden on basic questions; RAG handles questions requiring current context

### Recommendation

**Keep your existing RAG pipeline (LanceDB + tree-sitter) AND fine-tune.** They serve different purposes:
- Fine-tuning: Model learns TunedVoice coding patterns, Swift conventions, architectural style
- RAG: Model retrieves specific code snippets, API signatures, current implementations
- Combined: ~8% improvement over RAG alone per the industry study

Do NOT expect fine-tuning to replace RAG. The codebase changes; fine-tuning is a snapshot. RAG stays current.

---

## 6. Code Indexing and Embeddings

### Evidence

**Code-specific embedding models outperform general ones for code retrieval.** The Modal comparison identifies VoyageCode3 (32K context) and Nomic Embed Code (7B params) as top performers. CodeRankEmbed (137M params, 8K context) is optimized specifically for code search.

**Fine-tuning embedding models on small datasets is risky.** LanceDB's research showed that fine-tuning on a 45K-row dataset "resulted in instability and overfitting, with performance often degrading at higher epochs." A 2M-row dataset showed ~10% improvement. Your codebase is far too small to fine-tune an embedding model on.

**Tree-sitter AST chunking is complementary to embeddings, not a replacement.** AST-aware chunking produces better chunk boundaries (function/class level rather than arbitrary text splits), which improves retrieval quality regardless of the embedding model used.

### Recommendation

- **Do NOT fine-tune Qwen3-Embedding-0.6B on your codebase.** Dataset too small; high risk of degradation.
- **Consider switching to a code-specific embedding model** like Nomic Embed Code or CodeRankEmbed if retrieval quality is an issue. But Qwen3-Embedding is likely adequate for a single-codebase scenario.
- **Keep tree-sitter chunking.** It produces better boundaries than naive text splitting.
- **If retrieval quality matters, invest in a reranker** (cross-encoder) rather than fine-tuning the embedding model. Rerankers add more value per effort.

---

## 7. Evaluation of Codebase-Specialized Models

### Evidence

Standard benchmarks (HumanEval, MBPP) are irrelevant for codebase-specific evaluation. You need custom eval.

**Practical evaluation approaches:**
1. **Fixture-based eval** (what you have): Ground-truth Q&A pairs held out from training. Direct measurement of whether the model learned the codebase.
2. **Compilation success rate**: Apple used Swift compiler validation to filter training data AND evaluate output. For TunedVoice, compile-checking generated code against the actual project is the gold standard.
3. **Fill-in-the-middle accuracy**: Remove a function body from a real file, ask the model to regenerate it, compare to original. Measures whether the model understands codebase patterns.
4. **Code review simulation**: Present the model with known bugs from git history, measure detection rate.
5. **API usage correctness**: Ask the model to use TunedVoice-specific APIs (e.g., FluidAudio SDK, Parakeet TDT), verify correct method signatures and usage patterns.

**How many examples needed?** Sebastian Raschka's research shows 1,000 high-quality examples can match 50,000 synthetic examples. For codebase specialization, the community consensus is:
- **100 examples**: Minimum to see any effect
- **500-1,000**: Meaningful specialization for LoRA on a focused domain
- **2,000-5,000**: Strong specialization with diverse task coverage
- **>5,000**: Diminishing returns for a single codebase

Your 521 examples are at the lower end of "meaningful specialization." Expanding to 2,000-3,000 with diverse formats would put you in the sweet spot.

### Recommendation

**Evaluation strategy:**
1. Keep your fixture-based eval (53 validation examples)
2. Add compilation testing: generate code, attempt to compile against TunedVoice project
3. Add fill-in-the-middle eval: hold out 20-30 real function bodies, measure regeneration accuracy
4. Track general capability retention: run a small set of general Swift questions pre/post training to detect catastrophic forgetting

---

## Concrete Action Items

### Immediate (before next training run)

1. **Expand training data to ~2,000 examples** with format diversity:
   - Generate ~500 fill-in-the-middle examples from TunedVoice codebase (mask function bodies, have model regenerate)
   - Extract ~300 commit-diff pairs from git history
   - Generate ~200 code review / bug detection examples
   - Add ~100 general Swift/SwiftUI examples (from open source) as forgetting prevention
   - Expand Q&A pairs to ~900 (from current 468)

2. **Adjust LoRA hyperparameters per Raschka's findings:**
   - Consider r=256, alpha=512 (much higher than current r=16) — small datasets benefit from higher rank
   - Increase dropout to 0.1 (current 0.0; Raschka recommends 0.1 for <500 examples)
   - Train for 1 epoch only (current config is 150 steps; verify this isn't multi-epoch)
   - Current learning rate 2e-4 is reasonable

3. **Add compilation-based evaluation**: After training, generate 50 code samples and attempt to compile against TunedVoice project

### Short-term (next 2-4 weeks)

4. **Build fill-in-the-middle training pipeline**: Automated extraction of function bodies from codebase, formatted as FIM training examples
5. **Build commit-diff extraction pipeline**: Parse git log for training pairs
6. **Add general capability regression test**: 20-30 general Swift questions to track forgetting

### Do NOT do

- Do NOT fine-tune the embedding model (dataset too small)
- Do NOT build multi-adapter setup (single codebase doesn't warrant it)
- Do NOT drop RAG in favor of fine-tuning (they're complementary; combined is ~8% better than either alone)
- Do NOT train only on Q&A pairs (mix formats for best results)
- Do NOT pre-train on general Swift corpus (LoRA fine-tuning is sufficient; pre-training showed no benefit in HuggingFace discussion)

---

## Key Quantitative Findings Summary

| Finding | Source | Implication |
|---------|--------|-------------|
| RAG+FT combined: 57.43% EM vs 53.76% RAG-only vs 44.20% FT-only | arXiv 2505.15179 | Keep both RAG and FT |
| 1,000 curated examples ≈ 50,000 synthetic (with optimal hyperparams) | Raschka (2024) | Quality > quantity; 521 is workable but 2K is better |
| LoRA: 0.7149 F1 vs Full FT: 0.7497 F1, but LoRA retains general knowledge | Kirouane (2024) | LoRA is correct choice; small perf gap, major forgetting prevention |
| r=256 outperformed r=16 on small datasets | Raschka (2024) | Consider increasing LoRA rank significantly |
| Fine-tuning plateaus at ~90K files; RAG scales further | arXiv 2505.15179 | FT has a ceiling; RAG provides ongoing gains |
| Apple UICoder: 996K synthetic SwiftUI programs via compiler-validated iteration | Apple Research (2025) | Compilation validation is the gold standard filter |
| Embedding FT on <45K rows caused degradation | LanceDB (2025) | Do not fine-tune Qwen3-Embedding on our codebase |

---

## Learnings

1. **New facts**: The RAG vs FT industrial study (2505.15179) provides concrete numbers showing combined approach wins. Apple's UICoder work is directly relevant to SwiftUI fine-tuning.
2. **Reference doc updates**: The LoRA config in `configs/qwen36-35b-a3b.yaml` should be revisited — r=16 may be too low for a 521-example dataset per Raschka's findings. Dropout of 0.0 is risky for small datasets.
3. **Agent prompt corrections**: None needed.
