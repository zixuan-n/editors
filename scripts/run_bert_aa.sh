#!/usr/bin/env bash
# Usage: GPU=0 bash scripts/run_bert_aa.sh Blogs10 [CONFIG.yaml]
cd "$(dirname "$0")/.."

AA_DATASET="${1:-Gutenberg30}"
AA_CONFIG="${2:-code/configs/aa/bert_full.yaml}"
AA_PY="${PY:-python3}"
export CUDA_VISIBLE_DEVICES="${GPU:-${CUDA_VISIBLE_DEVICES:-0}}"

# Warm-up classification.
"$AA_PY" code/aa/bert_deflation/train_stage0_full_classifier.py \
  --config "$AA_CONFIG" --dataset "$AA_DATASET" &&

# Diagnostic MLM on style-neutral statements from the base model.
"$AA_PY" code/aa/bert_deflation/train_stage1_full_mlm_cpt.py \
  --config "$AA_CONFIG" --dataset "$AA_DATASET" --skip_stage0_init &&

# Deflated classification.
"$AA_PY" code/aa/bert_deflation/train_stage3_full_deflated_classifier.py \
  --config "$AA_CONFIG" --dataset "$AA_DATASET"
