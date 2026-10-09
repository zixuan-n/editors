# EDITORS

Code and datasets for reproducing the experiments in *EDITORS Know Your Style! Editing LoRA Subspaces for Stylistic Attribution and Imitation*.

## Documentation

- [Authorship attribution](code/aa/README.md): setup, LoRA/BERT training, and evaluation commands.
- [Style imitation](code/gen/README.md): training, generation, and evaluation commands.
- [Datasets](data/): dataset splits and generation prompts.

## Repository layout

- `code/aa/`: authorship attribution training, deflation helpers, and BERT deflation.
- `code/stage1/`: style-neutral statements extraction and diagnostic training.
- `code/configs/`: training configurations.
- `code/gen/`: style imitation training and generation.
- `code/eval_scripts/`: AA and SI evaluation utilities.
- `data/`: all bundled datasets, organized by dataset name.
- `scripts/`: shell launchers for training, generation, and evaluation.
