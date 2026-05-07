#!/usr/bin/env bash
# test_merged_model.sh -- Compare merged LoRA model vs base model
#
# Runs speed benchmarks and quality pass-rate tests against an OpenAI-compatible
# server (e.g., mlx_lm.server). Run once with the merged model, once with the
# base model, then compare.
#
# Usage:
#   ./test_merged_model.sh [SERVER_URL]              # Run benchmark + quality tests
#   ./test_merged_model.sh --compare A.json B.json   # Compare two result files
#
# Example workflow:
#   # With merged model served on port 8803
#   ./test_merged_model.sh http://localhost:8803
#   mv results_*.json results_merged.json
#
#   # Swap to base model on the same port, then:
#   ./test_merged_model.sh http://localhost:8803
#   mv results_*.json results_base.json
#
#   # Compare
#   ./test_merged_model.sh --compare results_base.json results_merged.json

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# --- Compare mode -----------------------------------------------------------
if [[ "${1:-}" == "--compare" ]]; then
    if [[ $# -lt 3 ]]; then
        echo "Usage: $0 --compare <results_A.json> <results_B.json>"
        exit 1
    fi
    FILE_A="$2"
    FILE_B="$3"
    python3 "$SCRIPT_DIR/test_helper.py" --compare "$FILE_A" "$FILE_B"
    exit $?
fi

# --- Benchmark + Quality mode -----------------------------------------------
SERVER_URL="${1:-http://localhost:8803}"
TIMESTAMP="$(date +%Y%m%dT%H%M%S)"
RESULTS_FILE="$SCRIPT_DIR/results_${TIMESTAMP}.json"

echo "============================================"
echo " Merged-Model Test Suite"
echo " Server:  $SERVER_URL"
echo " Output:  $RESULTS_FILE"
echo " Time:    $(date)"
echo "============================================"
echo ""

# Verify server is responding
echo "Checking server health..."
if ! curl -s --max-time 5 "$SERVER_URL/v1/models" > /dev/null 2>&1; then
    echo "ERROR: Server at $SERVER_URL is not responding."
    echo "Make sure an OpenAI-compatible server is running on the target port."
    exit 1
fi
MODEL_ID=$(curl -s --max-time 5 "$SERVER_URL/v1/models" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['data'][0]['id'] if d.get('data') else 'unknown')" 2>/dev/null || echo "unknown")
echo "Detected model: $MODEL_ID"
echo ""

python3 "$SCRIPT_DIR/test_helper.py" \
    --server "$SERVER_URL" \
    --output "$RESULTS_FILE" \
    --model-id "$MODEL_ID"

echo ""
echo "============================================"
echo " Results saved to: $RESULTS_FILE"
echo " To compare two runs:"
echo "   $0 --compare results_base.json results_merged.json"
echo "============================================"
