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

- Run warm-up, diagnostic training on style-neutral statements, and deflated AA training:

```bash
GPU=0 bash scripts/run_aa.sh Llama3 CCAT50
GPU=0 bash scripts/run_aa.sh Gemma2 Blogs50
```

- Optional: run only the diagnostic stage with the command below. `run_aa.sh` already includes this stage.

```bash
CUDA_VISIBLE_DEVICES=0 python code/stage1/run_atomic_stage1_cpt.py \
  --config code/configs/aa/s1.yaml --model_variant Llama3 --dataset CCAT50
```

- Configs: `s0.yaml`, `s1.yaml`, and `s2.yaml` in [configs/aa/](../configs/aa/).
- Edit the relevant `dataset_variants` and `model_variants` settings.
- Refer to the paper for model/dataset combinations and hyperparameters; adjust the YAML settings for the experiment being reproduced.

## BERT training

- Run classification warm-up, diagnostic MLM from base BERT, and deflated classification:

```bash
GPU=0 bash scripts/run_bert_aa.sh Blogs10
```

- Config: [bert_full.yaml](../configs/aa/bert_full.yaml).

## Evaluation

- Evaluate the bundled CCAT50 checkpoint on the full test split:

```bash
GPU=0 bash scripts/eval_aa.sh
```

- Evaluate a trained adapter by passing its dataset and path:

```bash
GPU=0 bash scripts/eval_aa.sh CCAT50 PATH_TO_ADAPTER
```

- Set `BASE_MODEL` when using a different base model.

## Checkpoint

- Base model: `meta-llama/Llama-3.1-8B-Instruct`
- [Llama CCAT50 adapter](checkpoints/llama3_8b_ccat50/adapter/): LoRA weights and tokenizer; reported test accuracy: 80.40% on 750 examples.
- [deflation_basis.safetensors](checkpoints/llama3_8b_ccat50/deflation_basis.safetensors) stores the diagnostic-stage content directions used to project content out of model activations.
- To also apply the saved projection during inference:

```bash
GPU=0 bash scripts/eval_aa.sh CCAT50 \
  code/aa/checkpoints/llama3_8b_ccat50/adapter \
  --deflation-bases code/aa/checkpoints/llama3_8b_ccat50/deflation_basis.safetensors
```
