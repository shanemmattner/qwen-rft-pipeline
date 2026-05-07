#!/usr/bin/env bash
# download_merged.sh -- Robustly download merged model from Modal volume
#
# Downloads each file individually (not as a directory -- directory downloads
# via `modal volume get` silently produce 0-byte shards) and verifies
# integrity after download.
#
# Usage:
#   ./download_merged.sh <experiment-name> [destination-path]
#
# Examples:
#   ./download_merged.sh rft-Qwen3.6-35B-A3B-r16-full-20260505-184155
#   ./download_merged.sh rft-Qwen3.6-35B-A3B-r16-full-20260505-184155 ./my-merged-model/
#
# The script will:
#   1. List all files in the Modal volume for the experiment's bf16 output
#   2. Download each file individually
#   3. Verify all files exist, none are 0 bytes
#   4. If model.safetensors.index.json exists, verify all referenced shards
#   5. Auto-retry failed/0-byte files up to 3 times
#
# Exit codes:
#   0 = success (all files downloaded and verified)
#   1 = verification failed after retries

set -euo pipefail

# ---- Configuration --------------------------------------------------------

VOLUME_NAME="rft-checkpoints"
MAX_RETRIES=3
MODAL="python3 -m modal"

# Colors (if terminal supports)
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m'

# ---- Functions ------------------------------------------------------------

usage() {
    echo "Usage: $0 <experiment-name> [destination-path]"
    echo ""
    echo "Downloads merged bf16 model from Modal volume with verification."
    echo ""
    echo "Arguments:"
    echo "  experiment-name   Name of the merge experiment"
    echo "  destination-path  Local download directory (default: ./merged-bf16/)"
}

log_info() {
    echo -e "${BLUE}[INFO]${NC} $*"
}

log_ok() {
    echo -e "${GREEN}[OK]${NC} $*"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $*"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $*"
}

log_section() {
    echo ""
    echo -e "${BOLD}--- $* ---${NC}"
}

parse_ls_output() {
    awk 'NF >= 1 && $1 !~ /^Directory/ && $1 !~ /^\// { print $1 }' | grep -v '^\s*$'
}

download_file() {
    local remote_path="$1"
    local local_path="$2"
    mkdir -p "$(dirname "$local_path")"
    if $MODAL volume get "$VOLUME_NAME" "$remote_path" "$local_path" 2>&1; then
        return 0
    else
        return 1
    fi
}

file_is_valid() {
    local filepath="$1"
    if [[ ! -f "$filepath" ]]; then
        return 1
    fi
    local size
    size=$(wc -c < "$filepath" 2>/dev/null || echo "0")
    size=$(echo "$size" | tr -d ' ')
    if [[ "$size" -eq 0 ]]; then
        return 1
    fi
    return 0
}

human_size() {
    local filepath="$1"
    if [[ ! -f "$filepath" ]]; then
        echo "0 B"
        return
    fi
    local bytes
    bytes=$(wc -c < "$filepath" 2>/dev/null || echo "0")
    bytes=$(echo "$bytes" | tr -d ' ')

    if [[ "$bytes" -ge 1073741824 ]]; then
        echo "$(echo "scale=2; $bytes / 1073741824" | bc) GB"
    elif [[ "$bytes" -ge 1048576 ]]; then
        echo "$(echo "scale=1; $bytes / 1048576" | bc) MB"
    elif [[ "$bytes" -ge 1024 ]]; then
        echo "$(echo "scale=1; $bytes / 1024" | bc) KB"
    else
        echo "$bytes B"
    fi
}

# ---- Parse arguments ------------------------------------------------------

if [[ $# -lt 1 ]] || [[ "${1:-}" == "--help" ]] || [[ "${1:-}" == "-h" ]]; then
    usage
    exit 1
fi

EXPERIMENT="$1"
DEST_DIR="${2:-./merged-bf16/}"
REMOTE_DIR="merged/${EXPERIMENT}/bf16"

# ---- Step 1: List remote files --------------------------------------------

log_section "Listing files on Modal volume"
log_info "Volume: ${VOLUME_NAME}"
log_info "Path:   ${REMOTE_DIR}/"

LS_OUTPUT=$($MODAL volume ls "$VOLUME_NAME" "${REMOTE_DIR}/" 2>&1) || {
    log_error "Failed to list volume path: ${REMOTE_DIR}/"
    echo "$LS_OUTPUT"
    echo ""
    log_info "Available merged experiments:"
    $MODAL volume ls "$VOLUME_NAME" "merged/" 2>/dev/null || true
    exit 1
}

mapfile -t REMOTE_FILES < <(echo "$LS_OUTPUT" | parse_ls_output)

if [[ ${#REMOTE_FILES[@]} -eq 0 ]]; then
    log_error "No files found at ${REMOTE_DIR}/"
    echo "Raw ls output:"
    echo "$LS_OUTPUT"
    exit 1
fi

log_ok "Found ${#REMOTE_FILES[@]} files:"
for f in "${REMOTE_FILES[@]}"; do
    echo "    $f"
done

# ---- Step 2: Download each file individually ------------------------------

log_section "Downloading files"
mkdir -p "$DEST_DIR"

FAILED_FILES=()
for filename in "${REMOTE_FILES[@]}"; do
    remote_path="${REMOTE_DIR}/${filename}"
    local_path="${DEST_DIR}/${filename}"

    log_info "Downloading: ${filename}"
    if download_file "$remote_path" "$local_path"; then
        if file_is_valid "$local_path"; then
            log_ok "${filename} ($(human_size "$local_path"))"
        else
            log_warn "${filename} downloaded but is 0 bytes -- will retry"
            FAILED_FILES+=("$filename")
        fi
    else
        log_warn "${filename} download failed -- will retry"
        FAILED_FILES+=("$filename")
    fi
done

# ---- Step 3: Retry failed/0-byte files ------------------------------------

if [[ ${#FAILED_FILES[@]} -gt 0 ]]; then
    log_section "Retrying ${#FAILED_FILES[@]} failed file(s)"

    for filename in "${FAILED_FILES[@]}"; do
        remote_path="${REMOTE_DIR}/${filename}"
        local_path="${DEST_DIR}/${filename}"
        success=false

        for attempt in $(seq 1 $MAX_RETRIES); do
            log_info "Retry ${attempt}/${MAX_RETRIES}: ${filename}"
            rm -f "$local_path"

            if download_file "$remote_path" "$local_path" && file_is_valid "$local_path"; then
                log_ok "${filename} ($(human_size "$local_path")) -- retry succeeded"
                success=true
                break
            fi

            if [[ $attempt -lt $MAX_RETRIES ]]; then
                log_warn "Attempt ${attempt} failed, waiting 5s before retry..."
                sleep 5
            fi
        done

        if [[ "$success" != true ]]; then
            log_error "${filename} FAILED after ${MAX_RETRIES} retries"
        fi
    done
fi

# ---- Step 4: Verification ------------------------------------------------

log_section "Verification"

ERRORS=0

for filename in "${REMOTE_FILES[@]}"; do
    local_path="${DEST_DIR}/${filename}"
    if ! file_is_valid "$local_path"; then
        log_error "MISSING or EMPTY: ${filename}"
        ERRORS=$((ERRORS + 1))
    fi
done

INDEX_FILE="${DEST_DIR}/model.safetensors.index.json"
if [[ -f "$INDEX_FILE" ]]; then
    log_info "Checking model.safetensors.index.json for shard references..."

    mapfile -t SHARD_FILES < <(
        python3 -c "
import json, sys
with open('$INDEX_FILE') as f:
    idx = json.load(f)
wm = idx.get('weight_map', {})
for shard in sorted(set(wm.values())):
    print(shard)
" 2>/dev/null
    )

    if [[ ${#SHARD_FILES[@]} -gt 0 ]]; then
        log_info "Index references ${#SHARD_FILES[@]} shard file(s)"
        for shard in "${SHARD_FILES[@]}"; do
            shard_path="${DEST_DIR}/${shard}"
            if ! file_is_valid "$shard_path"; then
                log_error "Shard referenced in index but MISSING or EMPTY: ${shard}"
                ERRORS=$((ERRORS + 1))
            else
                log_ok "Shard present: ${shard} ($(human_size "$shard_path"))"
            fi
        done
    fi
else
    log_info "No model.safetensors.index.json found (single-shard model or non-safetensors format)"
fi

# ---- Step 5: Summary -----------------------------------------------------

log_section "Summary"

FILE_COUNT=0
TOTAL_BYTES=0

for filename in "${REMOTE_FILES[@]}"; do
    local_path="${DEST_DIR}/${filename}"
    if [[ -f "$local_path" ]]; then
        FILE_COUNT=$((FILE_COUNT + 1))
        bytes=$(wc -c < "$local_path" 2>/dev/null || echo "0")
        bytes=$(echo "$bytes" | tr -d ' ')
        TOTAL_BYTES=$((TOTAL_BYTES + bytes))
    fi
done

TOTAL_GB=$(echo "scale=2; $TOTAL_BYTES / 1073741824" | bc)

echo ""
echo "  Experiment:  ${EXPERIMENT}"
echo "  Destination: $(cd "$DEST_DIR" && pwd)"
echo "  Files:       ${FILE_COUNT}/${#REMOTE_FILES[@]}"
echo "  Total size:  ${TOTAL_GB} GB"
echo ""

if [[ $ERRORS -gt 0 ]]; then
    log_error "Verification FAILED (${ERRORS} error(s))"
    log_error "Some files are missing or empty. The download is incomplete."
    exit 1
else
    log_ok "All ${FILE_COUNT} files downloaded and verified successfully"
    exit 0
fi
