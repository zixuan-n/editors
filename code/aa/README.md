# Authorship attribution (AA)

- Run commands from `EDITORS/`.
- Use a CUDA GPU and obtain access to the selected Hugging Face base model.

## Setup

```bash
conda create -n editors-aa python=3.10 -y
conda activate editors-aa
python -m pip install -r requirements-aa.txt
```

## LoRA training

- Run warm-up, atomic-fact diagnostic training, and deflated AA training:

```bash
GPU=0 bash scripts/run_aa.sh Llama3 CCAT50
GPU=0 bash scripts/run_aa.sh Gemma2 Blogs50
```

- Run the diagnostic stage separately:

```bash
CUDA_VISIBLE_DEVICES=0 python code/stage1/run_atomic_stage1_cpt.py \
  --config code/configs/aa/s1.yaml --model_variant Llama3 --dataset CCAT50
```

- Configs: `s0.yaml`, `s1.yaml`, and `s2.yaml` in [configs/aa/](../configs/aa/).
- Edit the relevant `dataset_variants` and `model_variants` settings.
- Choose a complete three-stage preset:

| Dataset | Models |
|---|---|
| CCAT10 | Llama3 |
| CCAT50, Blogs10, Blogs50 | Llama3, Gemma2, Mistral, Phi4mini |
| Gutenberg30 | Llama3, Mistral |

- Outputs: `training.output_dir` in each config.
- Final adapter: `adapter_deflated_stage2/` inside the stage-2 output directory.

## BERT training

- Run classification warm-up, diagnostic MLM from base BERT, and deflated classification:

```bash
GPU=0 bash scripts/run_bert_aa.sh Blogs10
```

- Config: [bert_full.yaml](../configs/aa/bert_full.yaml).
- Datasets: CCAT10, CCAT50, Blogs10, Blogs50, Gutenberg30.
- Outputs: `outputs/bert_deflation/DATASET/full/` (lowercase dataset name).

## Checkpoint

- Use the [Llama CCAT50 checkpoint](../../checkpoints/README.md) with its tokenizer and deflation bases.
- Dataset files: [data/](../../data/).
