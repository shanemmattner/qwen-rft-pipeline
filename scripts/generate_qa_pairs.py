#!/usr/bin/env python3
"""Generate Q&A training pairs from source files using DeepSeek V4 API.

Adapted from work/qa-pairs/generate_qa.py (original used claude -p Sonnet).
Now uses DeepSeek V4 direct API (primary) or OpenRouter (fallback).

Usage:
    # Dry run -- show what would be generated
    python3 scripts/generate_qa_pairs.py --dry-run

    # Generate from TunedVoice app source
    DEEPSEEK_API_KEY=sk-xxx python3 scripts/generate_qa_pairs.py --source app

    # Generate from TunedVoiceKit package
    DEEPSEEK_API_KEY=sk-xxx python3 scripts/generate_qa_pairs.py --source kit

    # Generate from backend (TypeScript)
    DEEPSEEK_API_KEY=sk-xxx python3 scripts/generate_qa_pairs.py --source backend

    # Resume interrupted run
    DEEPSEEK_API_KEY=sk-xxx python3 scripts/generate_qa_pairs.py --source app --resume

    # Use OpenRouter instead of DeepSeek direct
    OPENROUTER_API_KEY=sk-xxx python3 scripts/generate_qa_pairs.py --source app --provider openrouter

Environment variables:
    DEEPSEEK_API_KEY    -- DeepSeek API key (required for deepseek provider)
    OPENROUTER_API_KEY  -- OpenRouter API key (required for openrouter provider)
    TUNEDVOICE_REPO     -- Path to tunedvoice repo (auto-detected from ../repos/tunedvoice)
"""

import argparse
import json
import os
import sys
import time
import urllib.request
import urllib.error

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
OUTPUT_DIR = os.path.join(REPO_ROOT, "data", "raw")

# Default repo paths -- adjust via env or CLI
DEFAULT_TUNEDVOICE_REPO = os.path.normpath(
    os.path.join(REPO_ROOT, "..", "tunedvoice")
)

APP_SOURCE_BASE = "apps/mac_os/TunedVoice/Sources/TunedVoice"
KIT_SOURCE_BASE = "TunedVoiceKit/Sources/TunedVoiceKit"
BACKEND_SOURCE_BASE = "backend/supabase/functions"

PROVIDERS = {
    "deepseek": {
        "url": "https://api.deepseek.com/v1/chat/completions",
        "model": "deepseek-chat",
        "key_env": "DEEPSEEK_API_KEY",
    },
    "openrouter": {
        "url": "https://openrouter.ai/api/v1/chat/completions",
        "model": "deepseek/deepseek-chat",
        "key_env": "OPENROUTER_API_KEY",
    },
}

# ---------------------------------------------------------------------------
# System prompts per source type
# ---------------------------------------------------------------------------

SYSTEM_PROMPTS = {
    "app": (
        "You are a Swift/SwiftUI expert for the TunedVoice macOS app -- "
        "a push-to-talk dictation app using on-device Parakeet TDT v3 "
        "speech recognition with CTC vocabulary boosting via FluidAudio SDK."
    ),
    "kit": (
        "You are a Swift expert for TunedVoiceKit -- a shared Swift package "
        "providing audio, licensing, transcription, vocabulary mining, and "
        "model management services for the TunedVoice app family (macOS and watchOS)."
    ),
    "backend": (
        "You are a TypeScript/Deno expert for the TunedVoice backend -- "
        "Supabase Edge Functions handling license validation, trial management, "
        "Stripe webhooks, and device management."
    ),
}

# ---------------------------------------------------------------------------
# Subsystem mapping per source type
# ---------------------------------------------------------------------------

APP_SUBSYSTEM_MAP = {
    "Features/Dictation": "dictation",
    "Services/Transcription": "transcription",
    "Services/Audio": "audio",
    "Services/Mining": "mining",
    "Services/Output": "output",
    "Services/Hotkey": "hotkey",
    "Services/Security": "security",
    "Services/Settings": "settings",
    "Services/Logging": "logging",
    "Services/Storage": "storage",
    "Services/Permissions": "permissions",
    "Services/CrashReporting": "crash_reporting",
    "Services/Update": "update",
    "Services/TestServer": "testing",
    "UI": "ui",
    "App": "app",
    "Domain": "domain",
    "Support": "support",
}

KIT_SUBSYSTEM_MAP = {
    "Audio": "kit_audio",
    "Auth": "kit_auth",
    "Consent": "kit_consent",
    "Encryption": "kit_encryption",
    "License": "kit_license",
    "Model": "kit_model",
    "Transcription": "kit_transcription",
    "Upload": "kit_upload",
    "Vocabulary": "kit_vocabulary",
    "ContextMining": "context_mining",
}

BACKEND_SUBSYSTEM_MAP = {
    "validate-license": "backend_license",
    "start-trial": "backend_trial",
    "stripe-webhook": "backend_stripe",
    "deactivate-device": "backend_device",
    "_shared": "backend_shared",
}

# ---------------------------------------------------------------------------
# Files to skip (too small or uninteresting)
# ---------------------------------------------------------------------------

APP_SKIP_FILES = {
    "App/SafeResourceBundle.swift",
    "Services/Audio/FixtureAudioCaptureService.swift",
    "Services/Audio/SoundFeedbackService.swift",
    "Services/Audio/SystemSoundFeedback.swift",
    "Services/Output/PasteboardProvider.swift",
    "Services/Output/PasteTarget.swift",
    "Services/Output/KeyEventSynthesizer.swift",
    "Services/TestServer/TestServer.swift",
    "Support/PasteReceiverLogParser.swift",
    "UI/AboutView.swift",
    "UI/RecordingConsentView.swift",
    "UI/WindowPresenter.swift",
    "UI/LicenseWindowController.swift",
    "UI/OnboardingWindowController.swift",
    "Domain/CapturedAudio.swift",
    "Domain/DictationDiagnostics.swift",
    "Domain/DictationSnapshot.swift",
    "Domain/FlavorConfig.swift",
    "Services/Settings/SettingsStoreSupport.swift",
    "Services/Settings/LaunchAtLoginManager.swift",
    "Services/Settings/RecordingSettings.swift",
    "Services/Settings/HotkeySettings.swift",
    "Services/Logging/DiagnosticsExporter.swift",
    "Services/Logging/EncryptedLogRotation.swift",
    "Services/Logging/TimingSession.swift",
    "Services/CrashReporting/MetricKitCrashReporter.swift",
    "Services/CrashReporting/PLCrashReporterService.swift",
    "Services/Update/UpdateManager.swift",
    "Services/Storage/DataExportService.swift",
    "Services/Permissions/PermissionsService.swift",
}

# ---------------------------------------------------------------------------
# Generation prompt templates
# ---------------------------------------------------------------------------

SINGLE_FILE_PROMPT = """You are generating Q&A training pairs for fine-tuning a local LLM to be an expert on the TunedVoice codebase.

Given the source code below, generate {count} Q&A pairs as a JSON array.

RULES:
1. Every answer MUST reference specific file paths, type names, and function names from the code
2. Explain WHY the code works this way, not just WHAT it does
3. Questions must be codebase-specific -- no generic language questions
4. Answers should be 150-400 words with code snippets where helpful
5. Do NOT invent file paths, types, or functions not in the provided code
6. Mix categories: explain (20%), implement (30%), debug (20%), architecture (15%), code_navigation (10%), refactor (5%)
7. Mix difficulty: L1 single-file (30%), L2 cross-file (45%), L3 system-level (25%)

Output ONLY a JSON array of objects with this schema (no markdown, no explanation):
[
  {{
    "id": "{prefix}_001",
    "category": "explain|implement|debug|architecture|code_navigation|refactor",
    "difficulty": "L1|L2|L3",
    "subsystem": "{subsystem}",
    "question": "the question text",
    "answer": "the detailed answer",
    "source_files": ["relative/path/to/file"]
  }}
]

### Source code

{code}
"""

CROSS_FILE_PROMPT = """You are generating CROSS-FILE Q&A training pairs for fine-tuning a local LLM to be an expert on the TunedVoice codebase.

These questions MUST require understanding how multiple files interact. Do NOT ask questions answerable from a single file.

Subsystem: {description}

Generate {count} Q&A pairs as a JSON array. Focus on:
- How data flows between these types across files
- Why the architecture splits responsibilities this way
- Debugging scenarios that span multiple files
- Implementing features that touch multiple files
- How changes in one file affect the others

RULES:
1. Every answer MUST reference specific types and functions from MULTIPLE source files
2. Explain WHY the architecture works this way, not just WHAT it does
3. Answers should be 200-500 words with code snippets where helpful
4. Do NOT invent file paths, types, or functions not in the provided code
5. Mix categories: architecture (25%), debug (25%), implement (30%), code_navigation (10%), explain (10%)
6. All questions should be L2 (cross-file) or L3 (system-level)

Output ONLY a JSON array of objects with this schema (no markdown, no explanation):
[
  {{
    "id": "{prefix}_001",
    "category": "explain|implement|debug|architecture|code_navigation",
    "difficulty": "L2|L3",
    "subsystem": "{subsystem}",
    "question": "the question text",
    "answer": "the detailed answer",
    "source_files": {files_json}
  }}
]

### Source code

{code}
"""

# ---------------------------------------------------------------------------
# Cross-file slices for TunedVoiceKit
# ---------------------------------------------------------------------------

KIT_CROSS_FILE_SLICES = [
    {
        "name": "kit_license_flow",
        "description": "License validation, persistence, hardware fingerprinting, and trial management",
        "files": [
            "License/LicenseStore.swift",
            "License/LicensePersistence.swift",
            "License/HardwareFingerprint.swift",
            "License/ResponseVerifier.swift",
        ],
        "count": 8,
    },
    {
        "name": "kit_model_sync",
        "description": "Model downloading, patching, storage, and sync",
        "files": [
            "Model/ModelStore.swift",
            "Model/ModelSyncService.swift",
            "Model/DeltaPatcher.swift",
        ],
        "count": 6,
    },
    {
        "name": "kit_vocabulary",
        "description": "Vocabulary mining, alignment, and storage",
        "files": [
            "Vocabulary/VocabularyMiner.swift",
            "Vocabulary/VocabularyMinerStore.swift",
            "Vocabulary/LevenshteinWordAligner.swift",
        ],
        "count": 6,
    },
    {
        "name": "kit_audio",
        "description": "Audio recording, silence detection, and streaming transcription",
        "files": [
            "Audio/SentenceRecorder.swift",
            "Audio/SilenceMonitor.swift",
            "Audio/StreamingTranscriptionService.swift",
        ],
        "count": 6,
    },
]

# ---------------------------------------------------------------------------
# API client
# ---------------------------------------------------------------------------


def call_api(
    prompt: str,
    system_content: str,
    provider: str,
    temperature: float = 0.7,
    max_tokens: int = 4096,
) -> str:
    """Call the chat completions API (DeepSeek or OpenRouter)."""
    config = PROVIDERS[provider]
    api_key = os.environ.get(config["key_env"])
    if not api_key:
        raise RuntimeError(
            f"Missing {config['key_env']} environment variable. "
            f"Set it to use the {provider} provider."
        )

    body = {
        "model": config["model"],
        "messages": [
            {"role": "system", "content": system_content},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    if provider == "openrouter":
        headers["HTTP-Referer"] = "https://github.com/tunedvoice"
        headers["X-Title"] = "TunedVoice Q&A Generation"

    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(config["url"], data=data, headers=headers, method="POST")

    max_retries = 3
    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                result = json.loads(resp.read().decode("utf-8"))
            return result["choices"][0]["message"]["content"]
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
            if attempt < max_retries - 1:
                wait = 2 ** (attempt + 1)
                print(f"  Retry {attempt + 1}/{max_retries} after {wait}s: {e}")
                time.sleep(wait)
            else:
                raise RuntimeError(f"API call failed after {max_retries} attempts: {e}")


# ---------------------------------------------------------------------------
# File reading
# ---------------------------------------------------------------------------


def read_source_file(full_path: str, max_lines: int = 200) -> str:
    """Read a source file, truncating if necessary."""
    if not os.path.exists(full_path):
        return ""
    with open(full_path) as f:
        lines = f.readlines()
    content = "".join(lines[:max_lines])
    if len(lines) > max_lines:
        content += f"\n// ... (truncated at {max_lines} lines, {len(lines)} total)"
    return content


def read_files_for_prompt(repo_base: str, file_paths: list, max_lines: int = 200) -> str:
    """Read multiple source files and format as markdown code blocks."""
    parts = []
    for fp in file_paths:
        full = os.path.join(repo_base, fp)
        content = read_source_file(full, max_lines)
        if not content:
            print(f"  WARNING: {full} not found, skipping")
            continue
        ext = "swift" if fp.endswith(".swift") else "typescript"
        parts.append(f"#### {fp}\n```{ext}\n{content}\n```")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


def parse_qa_response(raw: str) -> list:
    """Parse JSON array from model response, handling markdown fences."""
    raw = raw.strip()

    # Strip markdown code fences
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
        if raw.endswith("```"):
            raw = raw[:-3]
        raw = raw.strip()
    if raw.startswith("json"):
        raw = raw[4:].strip()

    # Find the JSON array
    start = raw.find("[")
    end = raw.rfind("]")
    if start >= 0 and end > start:
        return json.loads(raw[start : end + 1])

    return json.loads(raw)


def validate_pair(pair: dict) -> bool:
    """Validate a single Q&A pair has required fields."""
    required = {"id", "category", "difficulty", "subsystem", "question", "answer"}
    if not required.issubset(pair.keys()):
        missing = required - set(pair.keys())
        print(f"  SKIP {pair.get('id', '?')}: missing fields {missing}")
        return False

    valid_categories = {"explain", "implement", "debug", "architecture", "code_navigation", "refactor", "edge_case", "concurrency"}
    if pair["category"] not in valid_categories:
        print(f"  WARN {pair['id']}: unknown category '{pair['category']}', keeping anyway")

    valid_difficulties = {"L1", "L2", "L3"}
    if pair["difficulty"] not in valid_difficulties:
        print(f"  SKIP {pair['id']}: invalid difficulty '{pair['difficulty']}'")
        return False

    if len(pair["answer"]) < 100:
        print(f"  SKIP {pair['id']}: answer too short ({len(pair['answer'])} chars)")
        return False

    return True


# ---------------------------------------------------------------------------
# Training format
# ---------------------------------------------------------------------------


def format_as_training(pair: dict, system_persona: str, code_context: str) -> dict:
    """Format a Q&A pair into chat training format with metadata."""
    code_excerpt = code_context[:2000] if code_context else ""
    system_content = f"{system_persona}\n\n{code_excerpt}" if code_excerpt else system_persona

    return {
        "id": pair["id"],
        "category": pair["category"],
        "difficulty": pair["difficulty"],
        "subsystem": pair["subsystem"],
        "source_files": pair.get("source_files", []),
        "messages": [
            {"role": "system", "content": system_content},
            {"role": "user", "content": pair["question"]},
            {"role": "assistant", "content": pair["answer"]},
        ],
    }


# ---------------------------------------------------------------------------
# Batch discovery
# ---------------------------------------------------------------------------


def detect_subsystem(file_path: str, subsystem_map: dict) -> str:
    """Detect subsystem from file path using the subsystem map."""
    for prefix, subsystem in subsystem_map.items():
        if file_path.startswith(prefix):
            return subsystem
    return "other"


def pairs_for_file(line_count: int) -> int:
    """Determine number of pairs to generate based on file size."""
    if line_count < 50:
        return 2
    if line_count < 150:
        return 3
    if line_count < 300:
        return 4
    if line_count < 500:
        return 5
    return 6


def discover_app_batches(repo_base: str) -> list:
    """Walk app source tree and build batches."""
    src = os.path.join(repo_base, APP_SOURCE_BASE)
    return _discover_swift_batches(src, APP_SUBSYSTEM_MAP, APP_SKIP_FILES)


def discover_kit_batches(repo_base: str) -> list:
    """Walk TunedVoiceKit source tree and build batches."""
    src = os.path.join(repo_base, KIT_SOURCE_BASE)
    return _discover_swift_batches(src, KIT_SUBSYSTEM_MAP, set())


def discover_backend_batches(repo_base: str) -> list:
    """Walk backend edge functions and build batches."""
    src = os.path.join(repo_base, BACKEND_SOURCE_BASE)
    if not os.path.exists(src):
        print(f"WARNING: Backend source not found at {src}")
        return []
    batches = []
    for root, _dirs, files in os.walk(src):
        for fname in sorted(files):
            if not fname.endswith(".ts"):
                continue
            full = os.path.join(root, fname)
            rel = os.path.relpath(full, src)
            with open(full) as f:
                lines = sum(1 for _ in f)
            if lines < 10:
                continue
            subsystem = detect_subsystem(rel, BACKEND_SUBSYSTEM_MAP)
            stem = fname.replace(".ts", "").replace("-", "_")
            prefix = f"be_{stem[:15]}".lower()
            batches.append({
                "files": [rel],
                "subsystem": subsystem,
                "prefix": prefix,
                "count": pairs_for_file(lines),
                "lines": lines,
                "source_base": src,
            })
    return batches


def _discover_swift_batches(src: str, subsystem_map: dict, skip_files: set) -> list:
    """Walk a Swift source tree and build one batch per file."""
    if not os.path.exists(src):
        print(f"WARNING: Source not found at {src}")
        return []
    batches = []
    for root, _dirs, files in os.walk(src):
        for fname in sorted(files):
            if not fname.endswith(".swift"):
                continue
            full = os.path.join(root, fname)
            rel = os.path.relpath(full, src)
            if rel in skip_files:
                continue
            with open(full) as f:
                lines = sum(1 for _ in f)
            if lines < 15:
                continue
            subsystem = detect_subsystem(rel, subsystem_map)
            stem = fname.replace(".swift", "").replace("+", "_")
            prefix = f"{subsystem[:6]}_{stem[:12]}".lower()
            batches.append({
                "files": [rel],
                "subsystem": subsystem,
                "prefix": prefix,
                "count": pairs_for_file(lines),
                "lines": lines,
                "source_base": src,
            })
    return batches


# ---------------------------------------------------------------------------
# State tracking for resumability
# ---------------------------------------------------------------------------


def load_done_prefixes(output_path: str) -> set:
    """Load prefixes already generated from the output file."""
    done = set()
    if os.path.exists(output_path):
        with open(output_path) as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    prefix = obj["id"].rsplit("_", 1)[0]
                    done.add(prefix)
                except (json.JSONDecodeError, KeyError):
                    pass
    return done


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def generate_batch(
    batch: dict,
    source_type: str,
    provider: str,
    output_path: str,
    dry_run: bool = False,
) -> int:
    """Generate Q&A pairs for a single batch (file or slice)."""
    source_base = batch.get("source_base", "")
    files = batch["files"]
    subsystem = batch["subsystem"]
    prefix = batch["prefix"]
    count = batch["count"]
    is_cross_file = len(files) > 1

    print(f"\n{'='*60}")
    print(f"Generating {count} pairs for [{subsystem}] (prefix={prefix})")
    print(f"Files: {', '.join(files)}")

    if dry_run:
        print(f"  [DRY RUN] Would generate {count} pairs")
        return count

    code = read_files_for_prompt(source_base, files)
    if not code:
        print("  ERROR: No files could be read")
        return 0

    system_persona = SYSTEM_PROMPTS.get(source_type, SYSTEM_PROMPTS["app"])

    if is_cross_file:
        prompt = CROSS_FILE_PROMPT.format(
            description=batch.get("description", subsystem),
            count=count,
            prefix=prefix,
            subsystem=subsystem,
            files_json=json.dumps(files),
            code=code,
        )
    else:
        prompt = SINGLE_FILE_PROMPT.format(
            count=count,
            prefix=prefix,
            subsystem=subsystem,
            code=code,
        )

    print(f"  Prompt: {len(prompt)} chars, calling {provider}...")
    t0 = time.time()

    try:
        raw = call_api(prompt, system_persona, provider)
    except Exception as e:
        print(f"  FAILED: {e}")
        return 0

    elapsed = time.time() - t0
    print(f"  Response in {elapsed:.1f}s ({len(raw)} chars)")

    try:
        pairs = parse_qa_response(raw)
    except json.JSONDecodeError as e:
        print(f"  JSON PARSE ERROR: {e}")
        print(f"  Raw response (first 500): {raw[:500]}")
        return 0

    written = 0
    for pair in pairs:
        if not validate_pair(pair):
            continue
        try:
            formatted = format_as_training(pair, system_persona, code[:2000])
            with open(output_path, "a") as f:
                f.write(json.dumps(formatted, ensure_ascii=False) + "\n")
            written += 1
        except Exception as e:
            print(f"  Skipping pair {pair.get('id', '?')}: {e}")

    print(f"  Wrote {written}/{len(pairs)} pairs")
    return written


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Generate Q&A training pairs")
    parser.add_argument(
        "--source",
        choices=["app", "kit", "backend", "all"],
        default="all",
        help="Source type to generate from (default: all)",
    )
    parser.add_argument(
        "--provider",
        choices=["deepseek", "openrouter"],
        default="deepseek",
        help="API provider (default: deepseek)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSONL file (default: data/raw/expansion-<source>.jsonl)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from previous run (skip already-generated prefixes)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be generated without making API calls",
    )
    parser.add_argument(
        "--repo",
        default=None,
        help="Path to tunedvoice repo (default: auto-detect)",
    )
    parser.add_argument(
        "--cross-file-only",
        action="store_true",
        help="Generate only cross-file slices (skip single-file batches)",
    )

    args = parser.parse_args()

    repo_base = args.repo or os.environ.get("TUNEDVOICE_REPO", DEFAULT_TUNEDVOICE_REPO)
    if not os.path.exists(repo_base):
        print(f"ERROR: TunedVoice repo not found at {repo_base}")
        print("Set TUNEDVOICE_REPO or use --repo to specify the path")
        sys.exit(1)

    sources = [args.source] if args.source != "all" else ["app", "kit", "backend"]

    for source_type in sources:
        output_file = args.output or os.path.join(OUTPUT_DIR, f"expansion-{source_type}.jsonl")

        # Discover batches
        if source_type == "app":
            batches = [] if args.cross_file_only else discover_app_batches(repo_base)
        elif source_type == "kit":
            single = [] if args.cross_file_only else discover_kit_batches(repo_base)
            # Add cross-file slices
            cross = []
            kit_src = os.path.join(repo_base, KIT_SOURCE_BASE)
            for s in KIT_CROSS_FILE_SLICES:
                cross.append({
                    **s,
                    "prefix": f"xf_{s['name'][:15]}",
                    "subsystem": s["name"],
                    "source_base": kit_src,
                })
            batches = single + cross
        elif source_type == "backend":
            batches = discover_backend_batches(repo_base)
        else:
            batches = []

        if not batches:
            print(f"\nNo batches found for source={source_type}")
            continue

        # Load resume state
        done_prefixes = load_done_prefixes(output_file) if args.resume else set()
        if done_prefixes:
            print(f"Resuming: {len(done_prefixes)} prefixes already done")

        # Clear output unless resuming
        if not args.resume and os.path.exists(output_file) and not args.dry_run:
            os.remove(output_file)

        total_expected = sum(b["count"] for b in batches)
        print(f"\n{'#'*60}")
        print(f"Source: {source_type}")
        print(f"Batches: {len(batches)}")
        print(f"Expected pairs: {total_expected}")
        print(f"Output: {output_file}")
        print(f"Provider: {args.provider}")
        print(f"{'#'*60}")
        sys.stdout.flush()

        if args.dry_run:
            for i, batch in enumerate(batches):
                print(f"  [{i+1}/{len(batches)}] {batch['prefix']}: "
                      f"{batch['count']} pairs from {', '.join(batch['files'][:3])}")
            print(f"\nDRY RUN TOTAL: {total_expected} pairs would be generated")
            continue

        total = 0
        for i, batch in enumerate(batches):
            if batch["prefix"] in done_prefixes:
                print(f"[{i+1}/{len(batches)}] SKIP {batch['prefix']} (already done)")
                sys.stdout.flush()
                continue

            written = generate_batch(
                batch=batch,
                source_type=source_type,
                provider=args.provider,
                output_path=output_file,
                dry_run=False,
            )
            total += written
            sys.stdout.flush()
            time.sleep(1)  # Rate limiting courtesy

        print(f"\n{'='*60}")
        print(f"TOTAL: {total} Q&A pairs written to {output_file}")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
