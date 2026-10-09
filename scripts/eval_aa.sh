#!/usr/bin/env bash
# Usage: GPU=0 bash scripts/eval_aa.sh [DATASET] [ADAPTER] [EXTRA_ARGS...]
cd "$(dirname "$0")/.."

AA_DATASET="${1:-CCAT50}"
AA_ADAPTER="${2:-code/aa/checkpoints/llama3_8b_ccat50/adapter}"
AA_BASE_MODEL="${BASE_MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
AA_PY="${PY:-python3}"
export CUDA_VISIBLE_DEVICES="${GPU:-${CUDA_VISIBLE_DEVICES:-0}}"

"$AA_PY" code/eval_scripts/eval_lowacc_subsets.py \
  --dataset "data/$AA_DATASET/test.jsonl" --model-id "$AA_BASE_MODEL" \
  --adapter "EDITORS=$AA_ADAPTER|none" --eval-full \
  --output-csv "results/aa/$AA_DATASET/accuracy.csv" \
  --per-author-csv "results/aa/$AA_DATASET/per_author.csv" "${@:3}"
