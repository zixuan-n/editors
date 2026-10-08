#!/usr/bin/env bash
# Usage: GPU=0 bash scripts/eval_generation.sh [GENERATED.jsonl] [RESULT_DIR] [METRICS]
# METRICS: memorization, style, content, fluency; comma-separated combinations; or all.
# Default: memorization,style. Example: bash scripts/eval_generation.sh INPUT.jsonl RESULTS fluency
cd "$(dirname "$0")/.."

GEN_PY="${PY:-python3}"
GEN_INPUT="${1:-outputs/generation/content/seed_1001/generated_EDITORS_A2G.jsonl}"
GEN_RESULTS="${2:-results/generation/content/seed_1001}"
GEN_METRICS="${3:-memorization,style}"
GEN_EVALUATOR="${EVALUATOR:-Llama=outputs/aa/Llama3_Gutenberg30/s2/adapter_deflated_stage2|meta-llama/Llama-3.1-8B-Instruct}"
export CUDA_VISIBLE_DEVICES="${GPU:-${CUDA_VISIBLE_DEVICES:-0}}"

"$GEN_PY" code/eval_scripts/evaluate_generation.py \
  --input "$GEN_INPUT" --output_dir "$GEN_RESULTS" --metrics "$GEN_METRICS" \
  --evaluator "$GEN_EVALUATOR" --eval_use_4bit "${@:4}"
