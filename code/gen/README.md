# Style imitation (generation)

- Run commands from `EDITORS/` using the [AA environment](../aa/README.md#setup).
- Use a CUDA GPU and obtain access to `meta-llama/Llama-3.1-8B-Instruct`.
- Dataset: Gutenberg30.

## Training

- Train the diagnostic adapter and the default 8-epoch deflated generator:

```bash
GPU=0 bash scripts/train_generation.sh s1
GPU=0 bash scripts/train_generation.sh s2
```

- For the optional 4 + 4 epoch variant, train the diagnostic adapter first, then the warm-up and deflated generator. Skip `s1` if that diagnostic adapter is already trained:

```bash
GPU=0 bash scripts/train_generation.sh s1
GPU=0 bash scripts/train_generation.sh s0
GPU=0 bash scripts/train_generation.sh s2_with_warmup
```

- Configs: [configs/generation/](../configs/generation/); diagnostic config: [aa/s1.yaml](../configs/aa/s1.yaml).
- Reuse another diagnostic adapter by setting `model.stage1_adapter_path`.
- Custom generator config: `bash scripts/train_generation.sh s2 CONFIG.yaml`.
- Rank ablation: pass `--basis_solver svd --basis_rank_k K` to the Python trainer, with `K` no greater than the diagnostic adapter's available rank.

## Generation

- Content-controlled generation: 30 authors × 7 summaries; use seeds 1001, 1002, and 1003.

```bash
GPU=0 bash scripts/generate_content.sh 1001
```

- Five-word continuation: use the original 900-prompt grid.

```bash
GPU=0 bash scripts/generate_prefix.sh 1234
```

## Evaluation

- Set `OPENAI_API_KEY` for content and fluency evaluation.
- Python entry point: [evaluate_generation.py](../eval_scripts/evaluate_generation.py).
- Train the Gutenberg30 AA classifier for style evaluation:

```bash
GPU=0 bash scripts/run_aa.sh Llama3 Gutenberg30
```

- Evaluate memorization and style for content seed 1001:

```bash
GPU=0 bash scripts/eval_generation.sh
```

- Evaluate five-word continuation:

```bash
GPU=0 bash scripts/eval_generation.sh \
  outputs/generation/prefix/seed_1234/generated_EDITORS_A2G.jsonl \
  results/generation/prefix/seed_1234 memorization,style,fluency
```

- Choose one metric, a comma-separated combination, or `all`:

```bash
bash scripts/eval_generation.sh INPUT.jsonl RESULTS_DIR memorization
bash scripts/eval_generation.sh INPUT.jsonl RESULTS_DIR fluency
bash scripts/eval_generation.sh INPUT.jsonl RESULTS_DIR content,fluency
GPU=0 bash scripts/eval_generation.sh INPUT.jsonl RESULTS_DIR all
```

- `memorization`: BLEU-4 against training passages, novel n-grams, longest copied span, and Self-BLEU.
- `style`: author-attribution accuracy; average across evaluators when using multiple classifiers.
- `content`: summary preservation, reported as `25 × (mean judge score − 1)`; requires `content_pack`.
- `fluency`: mean GPT-4o mini score from 0 to 100.
- Select another LoRA classifier with `EVALUATOR='NAME=ADAPTER|BASE_MODEL'`.
- For multiple classifiers, repeat `--evaluator` in the Python command.
