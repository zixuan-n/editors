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

- For the optional 4 + 4 epoch variant, reuse the same diagnostic adapter:

```bash
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

- Content outputs: `outputs/generation/content/seed_SEED/generated_EDITORS_A2G.jsonl`.
- Prefix outputs: `outputs/generation/prefix/seed_SEED/generated_EDITORS_A2G.jsonl`.
- Pass a custom output directory as the second argument.
- Default adapter: `outputs/generation/Gutenberg30/s2/adapter_deflated_stage2`.
- For the warm-up variant, set `GEN_ADAPTER=outputs/generation/Gutenberg30/s2_with_warmup/adapter_deflated_stage2`.
- Keep `deflation_basis.safetensors` beside the generator adapter for inference.

## Scoring

- Set `OPENAI_API_KEY` for content and fluency evaluation.
- Python entry point: [evaluate_generation.py](../eval_scripts/evaluate_generation.py).
- Train the Gutenberg30 AA classifier for style evaluation:

```bash
GPU=0 bash scripts/run_aa.sh Llama3 Gutenberg30
```

- Score memorization and style for content seed 1001:

```bash
GPU=0 bash scripts/eval_generation.sh
```

- Score five-word continuation:

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
- Resume content/fluency scoring by adding `--append` after the metric argument.
- Results: per-passage scores and `RESULTS_DIR/summary.json`.
