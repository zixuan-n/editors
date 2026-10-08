#!/usr/bin/env bash
# Usage: GPU=0 bash scripts/train_generation.sh [s1|s0|s2|s2_with_warmup] [CONFIG.yaml]
cd "$(dirname "$0")/.."

GEN_STAGE="${1:-s2}"
GEN_PY="${PY:-python3}"
export CUDA_VISIBLE_DEVICES="${GPU:-${CUDA_VISIBLE_DEVICES:-0}}"
export AA_REPORT_TO="${AA_REPORT_TO:-none}"

case "$GEN_STAGE" in
  s1)
    "$GEN_PY" code/stage1/run_atomic_stage1_cpt.py \
      --config "${2:-code/configs/aa/s1.yaml}" --model_variant Llama3 --dataset Gutenberg30 ;;
  s0)
    "$GEN_PY" code/gen/finetune_style_gen.py \
      --config "${2:-code/configs/generation/Llama3_Gutenberg30_s0.yaml}" ;;
  *)
    "$GEN_PY" code/gen/continue_stage2_with_lora_deflation_gen.py \
      --config "${2:-code/configs/generation/Llama3_Gutenberg30_${GEN_STAGE}.yaml}" ;;
esac
