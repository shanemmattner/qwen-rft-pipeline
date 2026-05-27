# Apple Silicon Inference Library Comparison for Qwen3.6-27B

**Date**: 2026-05-10
**Hardware**: Mac Studio M4 Max 128GB
**Model**: Qwen3.6-27B (dense 27B)
**Use case**: Multi-step agentic workflows (sequential calls, reliability critical)

---

## Recommendation

**Primary: oMLX** (continuous batching + SSD KV cache) **with MTPLX for speed-critical paths.**

**Rationale**: For agentic workflows where the model gets called many times sequentially, the two critical factors are (1) reliability at long contexts (no crashes/zombies) and (2) time-to-first-token on repeated prefixes. oMLX solves both with its paged SSD KV cache (TTFT drops from 30-90s to 1-3s on cached prefixes) and continuous batching. MTPLX provides 2.24x decode speedup via native MTP heads for generation-heavy calls.

**Runner-up: Rapid-MLX** if you want a single binary with broad model support and Claude Code integration out of the box (4.2x faster than Ollama, prompt cache, tool-call parsing).

**Avoid**: Stock mlx_lm.server (zombie states, no KV cap, single-threaded) and Ollama MLX preview (still preview, stability issues on newer chips).

---

## Detailed Comparison

### 1. MLX / mlx_lm.server (Apple, official)

| Attribute | Details |
|-----------|---------|
| **Throughput** | ~28 tok/s baseline for Qwen3.6-27B on M4 Max; 130 tok/s for MoE models like Qwen3.5-35B-A3B |
| **Long context** | Problematic. No `--max-kv-size` in server mode. KV cache grows unbounded, causing kernel panics (IOGPUMemory crash) at ~58K+ tokens. At 32K+ on 24GB machines, memory pressure forces degradation. MLC-LLM handles 128K better via paged KV cache. |
| **Quantization** | MLX native (4-bit, 8-bit), NVFP4. Best native format support. |
| **LoRA** | Supports adapter loading at inference time. Hot-swap is theoretically possible (millisecond swaps) but has bugs with `adapter_config.json` requirements. |
| **Speculative decoding** | No native MTP support. Requires MTPLX or dflash-mlx as external tools. |
| **API** | OpenAI-compatible via mlx_lm.server. Streaming supported. |
| **Stability** | **Poor for agentic use.** Single-threaded HTTPServer (MLX streams are per-thread). Crashes with "no Stream(gpu, 0) in current thread" on mlx 0.31.2. Metal OOM crashes instead of HTTP errors. Zombie states confirmed. Connection: close forced on every response. |
| **Development** | Active (Apple-maintained). But server-mode improvements lag behind core MLX. |

**Verdict**: Great framework, terrible server. The core MLX engine is fast but mlx_lm.server is not production-ready for agentic workflows.

---

### 2. Ollama (0.19+ MLX backend)

| Attribute | Details |
|-----------|---------|
| **Throughput** | With MLX backend: 1810 tok/s prefill, 112 tok/s decode on M5 Max (Qwen3.5-35B-A3B NVFP4). On M4 Max expect ~80-90 tok/s decode for MoE models. For dense 27B like Qwen3.6-27B, likely ~30-40 tok/s. |
| **Long context** | Falls back to llama.cpp below 32GB. MLX backend requires 32GB+ unified memory. Long context behavior inherits MLX limitations. |
| **Quantization** | GGUF (Q4_K_M, Q5_K_M, etc.) for llama.cpp backend; MLX native quants for MLX backend. |
| **LoRA** | No runtime hot-swap. Must fuse adapters into base model before deployment. GitHub issue #9548 tracks this as a feature request. |
| **Speculative decoding** | Not supported in MLX backend preview. |
| **API** | OpenAI-compatible. Streaming. Good ecosystem integration. |
| **Stability** | **Mixed.** MLX backend is "preview" as of March 2026. Metal backend crashes reported on M5 chips (bfloat/half mismatch). macOS jetsam kills Ollama under memory pressure. Regression bugs in 0.13.x-0.14.x series. M4 Max should be more stable than M5 early-adopter issues. |
| **Development** | Very active. Large community. Fast release cadence. |

**Verdict**: Good for casual use, not reliable enough for production agentic workflows. MLX backend is promising but still preview. No LoRA hot-swap or speculative decoding.

---

### 3. llama.cpp / llama-server

| Attribute | Details |
|-----------|---------|
| **Throughput** | 64-92 tok/s for Qwen3.5-35B-A3B MoE on M4 Max. Dense 27B likely 25-35 tok/s. 15-30% slower than native MLX on same quantization. |
| **Long context** | Better than MLX server. Supports `--cache-reuse` for prefix caching. Memory management more mature. Still memory-bandwidth-bound at 128K. |
| **Quantization** | GGUF (widest format support: Q2_K through Q8_0, IQ quants). Gold standard for quantization variety. |
| **LoRA** | Hot-swap implemented in PRs #8857 and #10994. Functional. |
| **Speculative decoding** | Beta MTP support merged (May 2026). Works with Qwen3.6-27B MTP heads. Also supports drafter-model speculative decoding. Stacks with --cache-reuse. |
| **API** | OpenAI-compatible via llama-server. Streaming. Basic queuing. |
| **Stability** | **Most mature.** Single-process, single-request by default. llama-server adds HTTP endpoint with basic queuing. Well-tested across hardware. Not designed for multi-tenant throughput but very stable for single-user agentic workflows. |
| **Development** | Extremely active. Largest open-source LLM inference community. Bleeding-edge HEAD has Qwen3.6 hybrid graph support (DeltaNet layers). |

**Verdict**: Most stable and battle-tested option. MTP beta support is a major plus. Slower than MLX-native but more reliable. Best choice if you want one simple, stable binary.

---

### 4. MTPLX (youssofal)

| Attribute | Details |
|-----------|---------|
| **Throughput** | **63 tok/s on Qwen3.6-27B** (vs 28 tok/s baseline MLX) = 2.24x speedup. Uses model's built-in MTP heads, no external drafter needed. |
| **Long context** | Improved fast-prefill path in v0.2. Long-context handling "much stronger" per release notes. Still MLX-based so inherits some MLX memory limitations. |
| **Quantization** | MLX native formats. Ships optimized quants on HuggingFace (Qwen3.6-27B-MTPLX-Optimized-Speed). |
| **LoRA** | Not documented. Likely inherits MLX adapter support but untested. |
| **Speculative decoding** | **Core feature.** Native MTP using target model's own MTP heads. Exact probability-ratio rejection sampling. Custom Metal kernels (not a stock MLX wrapper). |
| **API** | OpenAI and Anthropic-compatible. Streaming. |
| **Stability** | Early production (v0.2). Smaller user base. "Sustained-no-fan" thermal target still a future feature. |
| **Development** | Active single developer (youssofal). Smaller community but focused roadmap. |

**Verdict**: Fastest raw decode speed for Qwen3.6-27B. The 2.24x multiplier is real and verified. Worth running for generation-heavy tasks. Risk: small project, single maintainer.

---

### 5. oMLX (jundot)

| Attribute | Details |
|-----------|---------|
| **Throughput** | MLX-native speeds (comparable to mlx_lm baseline). Main advantage is TTFT, not raw tok/s. |
| **Long context** | **Best in class for repeat queries.** Two-tier KV cache: hot (RAM) + cold (SSD). When hot cache fills, blocks offload to SSD in safetensors format. On next request with matching prefix, restored from disk instantly. TTFT drops from 30-90s to 1-3s on long contexts. |
| **Quantization** | MLX native formats. |
| **LoRA** | Multi-model serving with LRU eviction. Per-model TTL. Manual load/unload. Likely supports adapter loading but not specifically documented for LoRA hot-swap. |
| **Speculative decoding** | Experimental DFlash-MLX integration documented. |
| **API** | OpenAI-compatible. macOS menu bar app. |
| **Stability** | Designed for long-running coding agent sessions. Continuous batching handles concurrent requests. Better memory management than stock mlx_lm.server. |
| **Development** | Active. Purpose-built for the exact use case (coding agents on Apple Silicon). |

**Verdict**: Best solution for agentic workflows with repeated context. The SSD KV cache is transformative for multi-step agent loops where similar prefixes recur. Solves the mlx_lm.server stability issues.

---

### 6. Rapid-MLX (raullenchai)

| Attribute | Details |
|-----------|---------|
| **Throughput** | 4.2x faster than Ollama. Ranked #1 on 16 of 18 benchmarked models (tested on M3 Ultra, 22 models, 6 engines). Sub-100ms cached TTFT. |
| **Long context** | Prompt cache with KV cache trimming for transformers. State snapshots for hybrid RNN models (Qwen3.5 DeltaNet). Auto-routes large-context requests to cloud LLM when local prefill would be slow. |
| **Quantization** | MLX native. |
| **LoRA** | Not specifically documented. |
| **Speculative decoding** | Not documented as a core feature. |
| **API** | OpenAI-compatible. Drop-in replacement. 17 tool-call parsers with automatic recovery for quantized models. Separates reasoning_content from content. |
| **Stability** | Built specifically for Claude Code, Cursor, Aider integration. Production-oriented. |
| **Development** | Released March 2026. Active. |

**Verdict**: Most polished MLX server for coding agent integration. Tool-call parsing and reasoning separation are unique features. Cloud routing fallback is clever for long-context edge cases. Strong contender.

---

### 7. vLLM / vllm-mlx / vllm-metal

| Attribute | Details |
|-----------|---------|
| **Throughput** | vllm-mlx: 525 tok/s on Qwen3-0.6B (small model). 21-87% higher throughput than llama.cpp on Apple Silicon. 4.3x scaling at 16 concurrent requests with continuous batching. |
| **Long context** | Continuous batching helps manage memory. Paged attention. |
| **Quantization** | MLX native (via vllm-mlx) or Metal (via vllm-metal). |
| **LoRA** | vLLM core has excellent multi-LoRA support with hot-swap. vllm-mlx likely inherits this but Apple Silicon support is less tested. |
| **Speculative decoding** | vLLM core supports it on CUDA. Apple Silicon support unclear. |
| **API** | OpenAI-compatible. Streaming. Anthropic-compatible (vllm-mlx). MCP tool calling. |
| **Stability** | Core vLLM on Apple Silicon is "frustrating" -- CUDA dependency is deep, MPS experimental. vllm-mlx and vllm-metal are community plugins, not official. |
| **Development** | vllm-mlx has an arxiv paper. v0.2.0 released April 2026. Active community project. |

**Verdict**: Best for concurrent request scenarios. Not recommended for single-user sequential agentic workflows where you're the only consumer. The continuous batching advantage is wasted on single-threaded agent loops.

---

### 8. exo (distributed inference)

| Attribute | Details |
|-----------|---------|
| **Throughput** | 5.37 tok/s for DeepSeek V3 (671B) on 8x M4 Pro Mac Minis. 1.8x speedup on 2 devices, 3.2x on 4 devices. |
| **Long context** | Distributed KV cache across devices. |
| **Quantization** | MLX native (uses MLX backend). |
| **LoRA** | Not documented. |
| **Speculative decoding** | Not documented. |
| **API** | OpenAI-compatible. Auto-discovery of devices. RDMA over Thunderbolt 5 (99% latency reduction). |
| **Stability** | Designed for multi-device clusters. Adds networking complexity. |
| **Development** | Active. GitHub: exo-explore/exo. |

**Verdict**: Only relevant if you need to run models larger than 128GB. For Qwen3.6-27B on a single M4 Max 128GB, exo adds unnecessary complexity. Skip.

---

### 9. dflash-mlx (speculative decoding addon)

| Attribute | Details |
|-----------|---------|
| **Throughput** | Up to 4.13x speedup on Qwen3.5-9B (30.96 to 127.07 tok/s, 89.36% acceptance rate). Lossless -- every token verified against target model. |
| **Long context** | JIT SDPA 2-pass for long-context verification. bf16 stabilization. |
| **Integration** | oMLX has experimental dflash-mlx integration documented. |
| **Development** | Active. Multiple forks. Benchmarks available for Qwen3 family. |

**Verdict**: Promising speculative decoding addon. Could pair with oMLX. Less proven than MTPLX for Qwen3.6-27B specifically.

---

## Summary Matrix

| Library | Decode tok/s (est. Qwen3.6-27B) | Long Context | LoRA Hot-Swap | Spec. Decoding | Stability | Agentic Fit |
|---------|--------------------------------|--------------|---------------|----------------|-----------|-------------|
| mlx_lm.server | ~28 | Poor (crashes) | Partial | No | Poor | Bad |
| Ollama 0.19 MLX | ~30-40 | Moderate | No | No | Mixed | Moderate |
| llama.cpp/llama-server | ~25-35 | Good | Yes | Beta MTP | Best | Good |
| MTPLX | **~63** | Improved | Unknown | **Native MTP** | Early prod | Good (speed) |
| oMLX | ~28 + cache | **Best (SSD cache)** | Multi-model | DFlash (exp.) | Good | **Best** |
| Rapid-MLX | ~28 + cache | Good (cloud fallback) | Unknown | No | Good | Very Good |
| vllm-mlx | ~28 (batched higher) | Good | Yes (core) | Unknown | Mixed | Moderate |
| exo | N/A (distributed) | N/A | No | No | N/A | Skip |

## Recommended Stack for Mac Studio M4 Max 128GB

### Option A: Maximum reliability (recommended)
- **Primary server**: oMLX -- continuous batching, SSD KV cache, designed for coding agents
- **Speed boost**: MTPLX for generation-heavy calls where 2.24x matters
- **Fallback**: llama-server with MTP beta for maximum stability

### Option B: Simplest setup
- **Single server**: Rapid-MLX -- drop-in OpenAI replacement, Claude Code integration, tool-call parsing, prompt cache
- **Fallback**: llama-server

### Option C: Maximum throughput
- **Primary**: MTPLX (63 tok/s on Qwen3.6-27B)
- **Fallback**: llama-server with --cache-reuse + MTP beta

### What to migrate away from
- **mlx_lm.server**: Replace immediately. Zombie states, kernel panics, no KV cap, single-threaded. Every known issue you've hit is a documented, unfixed problem.

---

## Sources

- [Ollama MLX backend announcement](https://ollama.com/blog/mlx)
- [Ollama 0.19 MLX integration details](https://medium.com/@tentenco/ollama-0-19-ships-mlx-backend-for-apple-silicon-local-ai-inference-gets-a-real-speed-bump-878b4928f680)
- [MLX vs Ollama benchmarks](https://willitrunai.com/blog/mlx-vs-ollama-apple-silicon-benchmarks)
- [MTPLX GitHub](https://github.com/youssofal/MTPLX)
- [MTPLX website](https://mtplx.com/)
- [oMLX GitHub](https://github.com/jundot/omlx)
- [oMLX website](https://omlx.ai/)
- [Rapid-MLX GitHub](https://github.com/raullenchai/Rapid-MLX)
- [vllm-mlx GitHub](https://github.com/waybarrios/vllm-mlx)
- [vllm-mlx arxiv paper](https://arxiv.org/abs/2601.19139)
- [dflash-mlx GitHub](https://github.com/Aryagm/dflash-mlx)
- [llama.cpp MTP beta](https://startupfortune.com/llamacpp-now-supports-multi-token-prediction-in-beta-and-the-implications-for-local-ai-tooling-are-bigger-than-the-pr-suggests/)
- [Qwen3.6-27B MTP GGUF](https://huggingface.co/froggeric/Qwen3.6-27B-MTP-GGUF)
- [mlx_lm.server kernel panic issue](https://github.com/ml-explore/mlx-lm/issues/883)
- [mlx_lm.server Metal OOM issue](https://github.com/ml-explore/mlx-lm/issues/854)
- [MLX memory management deep dive](https://medium.com/@michael.hannecke/how-my-local-coding-agent-crashed-my-mac-and-what-i-learned-about-mlx-memory-management-e0cbad01553c)
- [2026 Mac inference framework comparison](https://macgpu.com/en/blog/2026-mac-inference-framework-vllm-mlx-ollama-llamacpp-benchmark.html)
- [llama-server tuning on Apple Silicon](https://medium.com/@michael.hannecke/tuning-llama-server-on-apple-silicon-9b3e778ab100)
- [exo distributed inference](https://github.com/exo-explore/exo)
- [Apple Silicon decision framework](https://medium.com/@michael.hannecke/choosing-an-on-device-llm-runtime-on-apple-silicon-a-decision-framework-beyond-benchmarks-2449067b8b67)
