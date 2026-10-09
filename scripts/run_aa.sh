#!/usr/bin/env bash
# Usage: GPU=0 bash scripts/run_aa.sh Llama3 CCAT50
cd "$(dirname "$0")/.."

AA_MODEL="${1:-Llama3}"
AA_DATASET="${2:-CCAT50}"
AA_PY="${PY:-python3}"
export CUDA_VISIBLE_DEVICES="${GPU:-${CUDA_VISIBLE_DEVICES:-0}}"
export AA_REPORT_TO="${AA_REPORT_TO:-none}"

# Warm-up.
"$AA_PY" code/aa/finetune_lowacc_llama.py \
  --config code/configs/aa/s0.yaml --model_variant "$AA_MODEL" --dataset "$AA_DATASET" &&

# Diagnostic training on style-neutral statements.
"$AA_PY" code/stage1/run_atomic_stage1_cpt.py \
  --config code/configs/aa/s1.yaml --model_variant "$AA_MODEL" --dataset "$AA_DATASET" &&

# Deflated AA training.
"$AA_PY" code/aa/continue_stage2_with_lora_deflation.py \
  --config code/configs/aa/s2.yaml --model_variant "$AA_MODEL" --dataset "$AA_DATASET"
