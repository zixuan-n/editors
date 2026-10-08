# EDITORS

Code, datasets, and a checkpoint for reproducing the experiments in *EDITORS Know Your Style! Editing LoRA Subspaces for Stylistic Attribution and Imitation*.

## Documentation

- [Authorship attribution](code/aa/README.md): setup and LoRA/BERT training commands.
- [Style imitation](code/gen/README.md): training, generation, and evaluation commands.
- [Datasets](data/): dataset splits and generation prompts.
- [Checkpoint](checkpoints/README.md): Llama-3.1-8B CCAT50 adapter and loading instructions.

## Repository layout

- `code/aa/`: AA training, deflation helpers, and BERT deflation.
- `code/stage1/`: atomic-fact extraction and diagnostic training.
- `code/configs/`: training configurations.
- `code/gen/` and `code/eval_scripts/`: generation and evaluation utilities.
- `data/`: all bundled datasets, organized by dataset name.
- `scripts/`: shell launchers for training, generation, and scoring.
