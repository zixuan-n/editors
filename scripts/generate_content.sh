#!/usr/bin/env bash
# Usage: GPU=0 bash scripts/generate_content.sh [SEED] [OUTPUT_DIR]
cd "$(dirname "$0")/.."

GEN_PY="${PY:-python3}"
GEN_ADAPTER="${GEN_ADAPTER:-outputs/generation/Gutenberg30/s2/adapter_deflated_stage2}"
GEN_SEED="${1:-1001}"
GEN_OUTPUT="${2:-outputs/generation/content/seed_${GEN_SEED}}"
export CUDA_VISIBLE_DEVICES="${GPU:-${CUDA_VISIBLE_DEVICES:-0}}"

"$GEN_PY" code/gen/generate_style_from_prompts.py \
  --generator "EDITORS=$GEN_ADAPTER" \
  --prompts_path data/Gutenberg30/prompts_A2G_general.jsonl \
  --mode content --output_dir "$GEN_OUTPUT" --seed "$GEN_SEED" --gen_use_4bit
