#!/usr/bin/env bash
set -euo pipefail
: "${TRAIN_FILE:?Set TRAIN_FILE to your training JSONL}"
: "${EVAL_FILE:?Set EVAL_FILE to your evaluation JSONL}"
: "${MODEL:?Set MODEL to your model endpoint ID}"
cd "$(dirname "$0")/.."
python -u scripts/run_experiment.py \
  --train-file "$TRAIN_FILE" --eval-file "$EVAL_FILE" \
  --output-dir "${OUTPUT_DIR:-outputs/natural_science_privacy}" \
  --base-url "${BASE_URL:-http://127.0.0.1:8000/v1}" --model "$MODEL" \
  --subset-size 15000 --max-conversations 500 --max-pairs-per-conv 500 \
  --mu 0.4 --clip-c 1.0 --sigma 0.001 --fed-clients 4 --fed-rounds 20 "$@"
