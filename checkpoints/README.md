# EDITORS checkpoint

- Base model: `meta-llama/Llama-3.1-8B-Instruct`.
- Dataset: CCAT50.

| Checkpoint | Dataset | Test accuracy | Test examples |
|---|---|---:|---:|
| [llama3_8b_ccat50](aa/llama3_8b_ccat50/) | CCAT50 | 80.40% | 750 |

- Obtain the base-model weights from Hugging Face.
- Load [adapter/](aa/llama3_8b_ccat50/adapter/) with its configuration and tokenizer.
- Read [deflation_basis.safetensors](aa/llama3_8b_ccat50/deflation_basis.safetensors) with `safetensors.torch.load_file`.
- Apply the bases with `register_deflation_hooks` in [inference_deflation.py](../code/aa/inference_deflation.py) for base-layer input projection.
- Test accuracy corresponds to Tables 2 and 3 of the paper.
