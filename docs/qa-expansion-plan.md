# Q&A Expansion Plan: From 521 to ~2,000+ Training Examples

Date: 2026-05-10
Status: Planning

## Background

Current dataset: 521 Q&A pairs (287 DeepSeek V4, 159 bug-qa, 67 crossfile, 8 recovered).
Research recommends 2,000-3,000 total examples across diverse formats for strong codebase specialization.

## Phase 1: Q&A Pair Generation (~480 pairs)

**Tool**: `scripts/generate_qa_pairs.py` (DeepSeek V4 API, OpenRouter fallback)
**Based on**: Original `work/qa-pairs/generate_qa.py` which used `claude -p` Sonnet

### Gap Areas and Source Files

#### P0: TunedVoiceKit Package Internals (~60 pairs)

The TunedVoiceKit shared package has 77 Swift files with near-zero coverage. Source files organized by subdirectory:

| Subdirectory | Files | Pairs | Key Types |
|-------------|-------|-------|-----------|
| Audio/ | 11 | 15 | SentenceRecorder, SilenceMonitor, StreamingTranscriptionService |
| Auth/ | 5 | 8 | AuthSessionStore, keychain persistence |
| License/ | 10 | 12 | HardwareFingerprint, LicensePersistence, ResponseVerifier |
| Model/ | 9 | 10 | DeltaPatcher, ModelSyncService, ModelStore |
| Transcription/ | 9 | 8 | TranscriptionCleaner, WordInstabilityTracker |
| Vocabulary/ | 9 | 7 | VocabularyMiner, LevenshteinWordAligner |

System prompt prefix: "You are a Swift expert for TunedVoiceKit -- a shared Swift package providing audio, licensing, transcription, and vocabulary services for the TunedVoice app family (macOS and watchOS)."

#### P0: Context Mining Wave 4 (~25 pairs)

Newest subsystem, zero coverage. Files:
- `ContextScoutService.swift`
- `TermExtractor.swift`
- `ConfusionPairMatcher.swift`
- `ClaudeContextVocabularyMiningService.swift`

Subsystem: `context_mining`

#### P0: Backend Edge Functions (~30 pairs)

23 TypeScript files, zero direct coverage. Requires separate TypeScript system prompt template.

Key files:
- `validate-license/index.ts`, `validate-license/sign.ts`
- `start-trial/index.ts`
- `stripe-webhook/index.ts`
- `deactivate-device/index.ts`
- `_shared/rate-limiter.ts`, `_shared/request-validator.ts`

System prompt prefix: "You are a TypeScript/Deno expert for the TunedVoice backend -- Supabase Edge Functions handling license validation, trial management, Stripe webhooks, and device management."

#### P1: Code Navigation Questions (~40 pairs)

Template-based generation. Question patterns:
- "Where is X defined?"
- "Which files implement Y?"
- "What calls Z and why?"
- "How does data flow from A to B?"

These can be generated with lighter prompts (lower token cost). Mix across all subsystems.

#### P1: Implementation Questions (~45 pairs)

Harder to generate -- requires careful scenario design:
- "How would you add feature X?"
- "Modify Y to support Z"
- "Extend this pattern for a new use case"

Focus on subsystems with existing coverage (so the model sees both explain+implement for the same code).

#### Phase 1b: Deepen Existing Coverage (~280 pairs)

| Area | New Pairs | Method |
|------|-----------|--------|
| Uncovered app files (35 files) | 70 | 2 pairs per uncovered file |
| Cross-file interaction scenarios | 50 | Multi-file architecture, 3+ files per slice |
| Logging/diagnostics | 20 | Logger.swift, encrypted rotation, diagnostics export |
| Storage/encryption pipeline | 20 | Envelope encryption, KEK management |
| Testing infrastructure | 20 | Mock patterns, E2E test setup |
| Update system + build pipeline | 15 | UpdateManager, Sparkle integration |
| Settings migration | 15 | SettingsStore patterns, UserDefaults migration |
| Audio pipeline deep-dive | 20 | Normalization, device switching, silence detection |
| Onboarding/permissions flow | 15 | Permission requests, onboarding UI |
| Error handling patterns | 15 | TunedVoiceError taxonomy, error propagation |

### Question Type Distribution (target for new pairs)

| Type | Current % | Target % | New Pair Mix |
|------|-----------|----------|-------------|
| Architecture/explain | 50% | 35% | 25% of new pairs |
| Debugging | 36% | 30% | 25% of new pairs |
| Implementation | 14% | 20% | 30% of new pairs |
| Code navigation | 3% | 10% | 15% of new pairs |
| Refactor | -- | 5% | 5% of new pairs |

### Difficulty Distribution (target for new pairs)

| Level | Target |
|-------|--------|
| L1 (basic, single-file) | 30% |
| L2 (intermediate, cross-file) | 45% |
| L3 (advanced, system-level) | 25% |

### API Configuration

| Provider | Endpoint | Model | Cost Estimate |
|----------|----------|-------|---------------|
| DeepSeek (primary) | `https://api.deepseek.com/v1/chat/completions` | `deepseek-chat` (V4) | ~$0.14/M input, $0.28/M output |
| OpenRouter (fallback) | `https://openrouter.ai/api/v1/chat/completions` | `deepseek/deepseek-chat` | ~$0.14/M input, $0.28/M output |

Estimated token usage per pair: ~3K input + ~800 output = ~3.8K tokens
480 pairs: ~1.8M tokens total
**Estimated cost: ~$0.75** (DeepSeek direct pricing is very cheap)

### Generation Script

`scripts/generate_qa_pairs.py` -- configurable, resumable, idempotent.

Key changes from original `generate_qa.py`:
1. Uses DeepSeek/OpenRouter API directly instead of `claude -p`
2. Supports multiple repo bases (TunedVoice app, TunedVoiceKit package, backend)
3. Supports TypeScript source files (for backend)
4. Has `--dry-run` mode
5. Tracks progress in a state file for resumability
6. Validates output JSON schema before writing
7. Configurable via environment variables and CLI args

---

## Phase 2: Fill-in-the-Middle Completion Data (~600-900 examples)

**No API needed** -- mechanical extraction from codebase.

### Strategy

Extract function bodies from Swift source files and format as FIM training examples:

```
<|fim_prefix|>func audioRecorder(_ recorder: AudioRecorder, didCapture buffer: AVAudioPCMBuffer) {
<|fim_suffix|>
}
<|fim_middle|>
    let normalized = audioNormalizer.normalize(buffer)
    transcriptionService.appendBuffer(normalized)
    updateAudioLevel(from: buffer)
```

### Source Files

All Swift files from:
1. TunedVoice app (`repos/tunedvoice/apps/mac_os/TunedVoice/Sources/TunedVoice/`) -- 106 files
2. TunedVoiceKit package -- 77 files
3. Total: ~183 files, target 4-5 FIM examples per file

### Chunking Strategy

1. Parse each file with tree-sitter (Swift grammar) or regex-based function detection
2. Extract complete function/method bodies (skip trivial getters, < 3 lines)
3. For each function:
   - Prefix: everything up to and including the opening `{`
   - Suffix: closing `}` and any trailing code
   - Middle: the function body (ground truth)
4. Skip functions shorter than 3 lines or longer than 50 lines
5. For long functions (30-50 lines), create partial FIM (mask inner block)

### Output Format

```json
{"messages": [
  {"role": "system", "content": "Complete the Swift function body."},
  {"role": "user", "content": "<prefix_code>\n// FILL IN\n<suffix_code>"},
  {"role": "assistant", "content": "<middle_code>"}
]}
```

### Estimated Yield

~183 files x ~4 functions/file = ~730 FIM examples (after filtering trivial ones)

---

## Phase 3: Commit-Diff Pairs (~150-300 examples)

### Strategy

Mine TunedVoice git history for training pairs: commit message as context, diff as the training target.

```bash
git log --oneline -500 repos/tunedvoice
```

### Selection Criteria

1. Skip merge commits, version bumps, formatting-only changes
2. Keep commits that change 1-5 files (focused changes)
3. Keep commits with clear commit messages explaining the "why"
4. Skip commits with diffs > 2000 lines (too large for training)

### Format

```json
{"messages": [
  {"role": "system", "content": "You are a Swift developer working on TunedVoice. Generate the code changes described."},
  {"role": "user", "content": "Commit: <commit message>\n\nFiles changed:\n- <file list>\n\nGenerate the unified diff for this change."},
  {"role": "assistant", "content": "<unified diff>"}
]}
```

### Pipeline

1. `git log --format='%H %s' -500` to get commit list
2. For each: `git show --stat <sha>` to check file count and diff size
3. Filter: 1-5 files, <2000 diff lines, meaningful message
4. `git show <sha>` to extract full diff
5. Format as training pair

### Estimated Yield

~500 commits, ~40% pass filters = ~200 commit-diff pairs

---

## Phase 4: General Swift Mixing (~100 examples)

### Purpose

Prevent catastrophic forgetting of general Swift knowledge during fine-tuning.

### Sources

1. **Swift Evolution proposals** -- well-written Swift code examples with explanations
2. **Apple sample code** -- official SwiftUI/Swift samples
3. **Swift Package Index** -- popular open-source Swift packages

### Selection

- 50 SwiftUI view examples (general patterns, not TunedVoice-specific)
- 30 concurrency examples (async/await, actors, Task groups)
- 20 general Swift patterns (protocols, generics, error handling)

### Format

Same chat format as Q&A pairs but with generic system prompt:
```json
{"role": "system", "content": "You are a Swift/SwiftUI expert."}
```

---

## Summary

| Phase | Type | Count | Method | API Cost |
|-------|------|-------|--------|----------|
| 1 | Q&A pairs | ~480 | DeepSeek V4 API | ~$0.75 |
| 2 | Fill-in-the-middle | ~730 | Mechanical extraction | $0 |
| 3 | Commit-diff | ~200 | Git history mining | $0 |
| 4 | General Swift | ~100 | Manual curation | $0 |
| **Total** | **Mixed** | **~1,510** | | **~$0.75** |

Combined with existing 521 pairs: **~2,031 total training examples**

### Quality Gates

1. **Automated validation**: JSON schema check, source file path verification against actual codebase
2. **Adversarial review**: Sample 10% of generated pairs, verify against source code (as done for original 287)
3. **Deduplication**: Check for near-duplicates against existing 521 pairs
4. **File path hallucination check**: Validate all `source_files` references exist
5. **Answer length check**: Flag answers < 500 chars or > 3000 chars for review

### Ordering

Phase 2 (FIM) and Phase 3 (commit-diff) can run in parallel with Phase 1 since they don't use APIs.
Phase 4 can be done last as it's the smallest and least critical.
