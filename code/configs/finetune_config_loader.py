#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared YAML config loaders for finetuning scripts.

Each helper returns a SimpleNamespace with the fields the caller expects, so the
scripts can stay small and only differ by which YAML they point to.
"""

from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace
from typing import Any, Dict, Iterator, List, Optional, Tuple

import yaml


def _merge_config(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    result = deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge_config(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def _read_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if not isinstance(cfg, dict):
        raise ValueError("Configuration file must contain a mapping at the top level.")
    return cfg


def _select_variant(cfg: Dict[str, Any], kind: str, selected: Optional[str], flag: str) -> Dict[str, Any]:
    variants_key = f"{kind}_variants"
    variants = cfg.pop(variants_key, None)
    if variants is None:
        if selected is not None:
            raise ValueError(f"{flag} requires a config with a {variants_key} section.")
        return cfg
    if not isinstance(variants, dict) or not variants:
        raise ValueError(f"{variants_key} must be a nonempty mapping.")
    default = cfg.pop(f"default_{kind}", None)
    selected = selected if selected is not None else default
    if selected not in variants:
        raise ValueError(f"No {kind} variant {selected!r}; available variants: {', '.join(variants)}")
    if not isinstance(variants[selected], dict):
        raise ValueError(f"{kind.capitalize()} variant {selected!r} must contain a mapping.")
    return _merge_config(cfg, variants[selected])


def load_config(path: str, model_variant: Optional[str] = None,
                dataset_variant: Optional[str] = None) -> Dict[str, Any]:
    """Resolve dataset overrides first, then the selected model's overrides."""
    cfg = _select_variant(_read_config(path), "dataset", dataset_variant, "--dataset")
    return _select_variant(cfg, "model", model_variant, "--model_variant")


def iter_config_variants(path: str) -> Iterator[Tuple[Optional[str], Optional[str]]]:
    """Enumerate only the dataset/model combinations explicitly present in a config."""
    raw = _read_config(path)
    datasets = raw.get("dataset_variants", {None: {}})
    for dataset in datasets:
        cfg = _select_variant(deepcopy(raw), "dataset", dataset, "--dataset")
        for model in cfg.get("model_variants", {None: {}}):
            yield dataset, model


# --------------- helpers -----------------
def _ensure_list(value: Optional[Any]) -> Optional[List[str]]:
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        return [value] if value else None
    if isinstance(value, list):
        cleaned = [str(v).strip() for v in value if str(v).strip()]
        return cleaned or None
    return None


# --------------- per-script loaders -----------------
def load_lowacc_config(path: str, model_variant: Optional[str] = None,
                       dataset_variant: Optional[str] = None) -> SimpleNamespace:
    cfg = load_config(path, model_variant, dataset_variant)

    model_cfg = cfg.get("model", {})
    data_cfg = cfg.get("data", {})
    train_cfg = dict(cfg.get("training", {}))
    quant_cfg = cfg.get("quantization", {})
    lora_cfg = cfg.get("lora", {})
    gen_cfg = cfg.get("generation", {})
    loss_cfg = cfg.get("loss_logging", {})
    prompt_cfg = cfg.get("prompt", {})

    output_dir = train_cfg.pop("output_dir", "outputs/llama3_lowacc_lora")
    train_cfg.setdefault("disable_tqdm", False)
    pad_to_multiple_of = train_cfg.pop("pad_to_multiple_of", 8)

    return SimpleNamespace(
        model_name=model_cfg.get("name", "meta-llama/Llama-3.1-8B-Instruct"),
        max_seq_length=data_cfg.get("max_seq_length", model_cfg.get("max_seq_length", 2048)),
        trust_remote_code=model_cfg.get("trust_remote_code", False),
        torch_dtype=model_cfg.get("torch_dtype"),
        device_map=model_cfg.get("device_map"),
        data_seed=data_cfg.get("seed", 1234),
        pool_path=data_cfg.get("pool_path", data_cfg.get("train_path")),
        train_path=data_cfg.get("train_path"),
        eval_path=data_cfg.get("eval_path"),
        test_path=data_cfg.get("test_path"),
        eval_subset_per_author=data_cfg.get("eval_subset_per_author"),
        eval_require_distinct_titles=data_cfg.get("eval_require_distinct_titles", False),
        positive_train=data_cfg.get("positive_train", 400),
        negative_train=data_cfg.get("negative_train", 400),
        positive_eval=data_cfg.get("positive_eval", 100),
        negative_eval=data_cfg.get("negative_eval", 100),
        positive_test=data_cfg.get("positive_test", 100),
        negative_test=data_cfg.get("negative_test", 100),
        output_dir=output_dir,
        training_args=train_cfg,
        pad_to_multiple_of=pad_to_multiple_of,
        quant_args=quant_cfg,
        lora_args=lora_cfg,
        generation_max_new_tokens=gen_cfg.get("max_new_tokens", 8),
        loss_log_path=loss_cfg.get("csv_path"),
        loss_plot_path=loss_cfg.get("plot_path"),
        target_authors=_ensure_list(data_cfg.get("target_authors")),
        system_prompt=prompt_cfg.get(
            "system_prompt",
            "You are an expert literary analyst. Identify the author of the provided passage. "
            "Answer with only the exact author name from the candidate set.",
        ),
        prompt_template=prompt_cfg.get(
            "template",
            (
                "<|start_header_id|>system<|end_header_id|>\n"
                "{system_prompt}\n"
                "<|eot_id|><|start_header_id|>user<|end_header_id|>\n"
                "Passage:\n{passage}\n"
                "Author:\n"
                "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
            ),
        ),
        answer_prefix=prompt_cfg.get("answer_prefix", " "),
    )


def load_atomic_config(path: str) -> SimpleNamespace:
    cfg = load_config(path)
    if "model" not in cfg:
        raise KeyError("Missing `model` section in config.")
    if "data" not in cfg:
        raise KeyError("Missing `data` section in config.")
    if "training" not in cfg:
        raise KeyError("Missing `training` section in config.")
    if "prompt" not in cfg:
        raise KeyError("Missing `prompt` section in config.")

    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    train_cfg = dict(cfg["training"])
    quant_cfg = cfg.get("quantization", {})
    lora_cfg = cfg.get("lora", {})
    loss_cfg = cfg.get("loss_logging", {})
    prompt_cfg = cfg["prompt"]

    output_dir = train_cfg.pop("output_dir")
    train_cfg.setdefault("disable_tqdm", False)
    pad_to_multiple_of = train_cfg.pop("pad_to_multiple_of")

    return SimpleNamespace(
        model_name=model_cfg.get("name", "meta-llama/Llama-3.1-8B-Instruct"),
        max_seq_length=model_cfg.get("max_seq_length", 1536),
        data_seed=data_cfg.get("seed", 1234),
        passage_path=data_cfg["passage_path"],
        facts_path=data_cfg["facts_path"],
        eval_passage_path=data_cfg.get("eval_passage_path"),
        eval_facts_path=data_cfg.get("eval_facts_path"),
        max_train_examples=data_cfg.get("max_train_examples"),
        max_eval_examples=data_cfg.get("max_eval_examples"),
        output_dir=output_dir,
        training_args=train_cfg,
        pad_to_multiple_of=pad_to_multiple_of,
        quant_args=quant_cfg,
        lora_args=lora_cfg,
        loss_log_path=loss_cfg.get("csv_path"),
        loss_plot_path=loss_cfg.get("plot_path"),
        system_prompt=prompt_cfg["system_prompt"],
        prompt_template=prompt_cfg["template"],
    )


def load_stage2_config(path: str, model_variant: Optional[str] = None,
                      dataset_variant: Optional[str] = None) -> SimpleNamespace:
    cfg = load_config(path, model_variant, dataset_variant)
    model_cfg = cfg.get("model", {})
    data_cfg = cfg.get("data", {})
    train_cfg = cfg.get("training", {})
    defl_cfg = cfg.get("deflation", {})
    quant_cfg = cfg.get("quantization", {})
    loss_cfg = cfg.get("loss_logging", {})
    prompt_cfg = cfg.get("prompt", {})
    lora_cfg = cfg.get("lora", {})

    use_4bit = bool(
        quant_cfg.get("load_in_4bit", False)
        or model_cfg.get("use_4bit", False)
        or cfg.get("use_4bit", False)
    )

    basis_tol = defl_cfg.get("basis_tol", 1e-6)
    if isinstance(basis_tol, str):
        try:
            basis_tol = float(basis_tol)
        except ValueError as exc:
            raise ValueError(f"deflation.basis_tol must be a float; got {basis_tol!r}") from exc

    return SimpleNamespace(
        model_id=model_cfg.get("name", "meta-llama/Llama-3.1-8B-Instruct"),
        stage1_adapter=model_cfg.get("stage1_adapter_path") or defl_cfg.get("stage1_adapter"),
        stage2_adapter=model_cfg.get("stage2_adapter_path") or train_cfg.get("stage2_adapter_path"),
        init_stage2_from_base=bool(
            model_cfg.get("init_stage2_from_base")
            or train_cfg.get("init_stage2_from_base")
            or cfg.get("init_stage2_from_base", False)
        ),
        lora_r=int(lora_cfg.get("r", 8) or 8),
        lora_alpha=int(lora_cfg.get("lora_alpha", 16) or 16),
        lora_dropout=float(lora_cfg.get("lora_dropout", 0.0) or 0.0),
        lora_target_modules=_ensure_list(lora_cfg.get("target_modules")),
        lora_bias=str(lora_cfg.get("bias", "none")),
        train_jsonl=data_cfg.get("train_path") or data_cfg.get("train_jsonl") or "",
        dev_jsonl=data_cfg.get("eval_path") or data_cfg.get("dev_jsonl") or "",
        out_dir=train_cfg.get("output_dir", "outputs/stage2_ce_deflated"),
        max_len=data_cfg.get("max_seq_length", 768),
        seed=train_cfg.get("seed", data_cfg.get("seed", 42)),
        use_4bit=use_4bit,
        quant_args=quant_cfg,
        torch_dtype=model_cfg.get("torch_dtype", "bfloat16"),
        trust_remote_code=model_cfg.get("trust_remote_code", False),
        device_map=model_cfg.get("device_map", {"": 0}),
        training_args=dict(train_cfg),
        batch=train_cfg.get("batch_size") or train_cfg.get("per_device_train_batch_size") or 4,
        eval_batch=train_cfg.get("per_device_eval_batch_size") or train_cfg.get("batch_size") or 4,
        grad_accum=train_cfg.get("grad_accum") or train_cfg.get("gradient_accumulation_steps") or 4,
        lr=train_cfg.get("lr") or train_cfg.get("learning_rate") or 5e-5,
        epochs=train_cfg.get("epochs") or train_cfg.get("num_train_epochs") or 1.0,
        logging_steps=train_cfg.get("logging_steps", 25),
        eval_steps=train_cfg.get("eval_steps", 200),
        save_steps=train_cfg.get("save_steps", 400),
        save_total_limit=train_cfg.get("save_total_limit", None),
        lr_scheduler_type=train_cfg.get("lr_scheduler_type", "cosine"),
        warmup_ratio=train_cfg.get("warmup_ratio", 0.05),
        weight_decay=train_cfg.get("weight_decay", 0.0),
        bf16=train_cfg.get("bf16", True),
        eval_strategy=train_cfg.get("eval_strategy", "steps"),
        report_to=train_cfg.get("report_to", "none"),
        run_name=train_cfg.get("run_name"),
        include_module=_ensure_list(defl_cfg.get("include_module")),
        basis_tol=basis_tol,
        basis_type=str(defl_cfg.get("basis_type", "lora_A")),
        basis_solver=str(defl_cfg.get("basis_solver", "qr")),
        basis_rank_k=defl_cfg.get("basis_rank_k", defl_cfg.get("max_rank")),
        save_u_sem=defl_cfg.get("save_u_sem"),
        deflation_sample_every=int(defl_cfg.get("sample_every", 1) or 1),
        deflation_log_topk=int(defl_cfg.get("log_topk", 0) or 0),
        deflation_log_csv=defl_cfg.get("log_csv"),
        loss_log_path=loss_cfg.get("csv_path"),
        loss_plot_path=loss_cfg.get("plot_path"),
        system_prompt=prompt_cfg.get("system_prompt"),
        prompt_template=prompt_cfg.get("template"),
        answer_prefix=prompt_cfg.get("answer_prefix", " "),
    )


def load_style_gen_config(path: str) -> SimpleNamespace:
    cfg = load_config(path)

    model_cfg = cfg.get("model", {})
    data_cfg = cfg.get("data", {})
    train_cfg = dict(cfg.get("training", {}))
    quant_cfg = cfg.get("quantization", {})
    lora_cfg = cfg.get("lora", {})
    gen_cfg = cfg.get("generation", {})
    loss_cfg = cfg.get("loss_logging", {})
    prompt_cfg = cfg.get("prompt", {})

    output_dir = train_cfg.pop("output_dir", "outputs/llama3_style_gen")
    train_cfg.setdefault("disable_tqdm", False)
    pad_to_multiple_of = train_cfg.pop("pad_to_multiple_of", 8)

    return SimpleNamespace(
        model_name=model_cfg.get("name", "meta-llama/Llama-3.1-8B-Instruct"),
        max_seq_length=data_cfg.get("max_seq_length", model_cfg.get("max_seq_length", 1024)),
        trust_remote_code=model_cfg.get("trust_remote_code", False),
        torch_dtype=model_cfg.get("torch_dtype"),
        device_map=model_cfg.get("device_map"),
        data_seed=data_cfg.get("seed", 1234),
        train_path=data_cfg.get("train_path"),
        eval_path=data_cfg.get("eval_path"),
        test_path=data_cfg.get("test_path"),
        summary_facts_path=data_cfg.get("summary_facts_path") or data_cfg.get("facts_path"),
        summary_field=data_cfg.get("summary_field", "content_pack"),
        text_field=data_cfg.get("text_field", "text"),
        author_field=data_cfg.get("author_field", "author"),
        output_dir=output_dir,
        training_args=train_cfg,
        pad_to_multiple_of=pad_to_multiple_of,
        quant_args=quant_cfg,
        lora_args=lora_cfg,
        generation_max_new_tokens=gen_cfg.get("max_new_tokens", 256),
        loss_log_path=loss_cfg.get("csv_path"),
        loss_plot_path=loss_cfg.get("plot_path"),
        system_prompt=prompt_cfg.get("system_prompt"),
        prompt_template=prompt_cfg.get("template"),
        user_template=prompt_cfg.get("user_template"),
        answer_prefix=prompt_cfg.get("answer_prefix", ""),
    )


def load_stage2_gen_config(path: str) -> SimpleNamespace:
    cfg = load_config(path)
    model_cfg = cfg.get("model", {})
    data_cfg = cfg.get("data", {})
    train_cfg = cfg.get("training", {})
    defl_cfg = cfg.get("deflation", {})
    quant_cfg = cfg.get("quantization", {})
    loss_cfg = cfg.get("loss_logging", {})
    prompt_cfg = cfg.get("prompt", {})
    lora_cfg = cfg.get("lora", {})

    use_4bit = bool(
        quant_cfg.get("load_in_4bit", False)
        or model_cfg.get("use_4bit", False)
        or cfg.get("use_4bit", False)
    )

    basis_tol = defl_cfg.get("basis_tol", 1e-6)
    if isinstance(basis_tol, str):
        try:
            basis_tol = float(basis_tol)
        except ValueError as exc:
            raise ValueError(f"deflation.basis_tol must be a float; got {basis_tol!r}") from exc

    return SimpleNamespace(
        model_id=model_cfg.get("name", "meta-llama/Llama-3.1-8B-Instruct"),
        stage1_adapter=model_cfg.get("stage1_adapter_path") or defl_cfg.get("stage1_adapter"),
        stage2_adapter=model_cfg.get("stage2_adapter_path") or train_cfg.get("stage2_adapter_path"),
        init_stage2_from_base=bool(
            model_cfg.get("init_stage2_from_base")
            or train_cfg.get("init_stage2_from_base")
            or cfg.get("init_stage2_from_base", False)
        ),
        lora_r=int(lora_cfg.get("r", 8) or 8),
        lora_alpha=int(lora_cfg.get("lora_alpha", 16) or 16),
        lora_dropout=float(lora_cfg.get("lora_dropout", 0.0) or 0.0),
        lora_target_modules=_ensure_list(lora_cfg.get("target_modules")),
        lora_bias=str(lora_cfg.get("bias", "none")),
        train_jsonl=data_cfg.get("train_path") or data_cfg.get("train_jsonl") or "",
        dev_jsonl=data_cfg.get("eval_path") or data_cfg.get("dev_jsonl") or "",
        summary_facts_path=data_cfg.get("summary_facts_path") or data_cfg.get("facts_path"),
        summary_field=data_cfg.get("summary_field", "content_pack"),
        out_dir=train_cfg.get("output_dir", "outputs/stage2_gen_deflated"),
        max_len=data_cfg.get("max_seq_length", 1024),
        seed=train_cfg.get("seed", 42),
        use_4bit=use_4bit,
        quant_args=quant_cfg,
        torch_dtype=model_cfg.get("torch_dtype", "bfloat16"),
        trust_remote_code=model_cfg.get("trust_remote_code", False),
        device_map=model_cfg.get("device_map", {"": 0}),
        training_args=dict(train_cfg),
        batch=train_cfg.get("batch_size") or train_cfg.get("per_device_train_batch_size") or 4,
        eval_batch=train_cfg.get("per_device_eval_batch_size") or train_cfg.get("batch_size") or 4,
        grad_accum=train_cfg.get("grad_accum") or train_cfg.get("gradient_accumulation_steps") or 4,
        lr=train_cfg.get("lr") or train_cfg.get("learning_rate") or 5e-5,
        epochs=train_cfg.get("epochs") or train_cfg.get("num_train_epochs") or 1.0,
        logging_steps=train_cfg.get("logging_steps", 25),
        eval_steps=train_cfg.get("eval_steps", 200),
        save_steps=train_cfg.get("save_steps", 400),
        save_total_limit=train_cfg.get("save_total_limit", None),
        lr_scheduler_type=train_cfg.get("lr_scheduler_type", "cosine"),
        warmup_ratio=train_cfg.get("warmup_ratio", 0.05),
        weight_decay=train_cfg.get("weight_decay", 0.0),
        bf16=train_cfg.get("bf16", True),
        eval_strategy=train_cfg.get("eval_strategy", "steps"),
        report_to=train_cfg.get("report_to", "none"),
        run_name=train_cfg.get("run_name"),
        include_module=_ensure_list(defl_cfg.get("include_module")),
        basis_tol=basis_tol,
        basis_type=str(defl_cfg.get("basis_type", "lora_A")),
        basis_solver=str(defl_cfg.get("basis_solver", "qr")),
        basis_rank_k=(int(defl_cfg["basis_rank_k"]) if defl_cfg.get("basis_rank_k") is not None else None),
        save_u_sem=defl_cfg.get("save_u_sem"),
        deflation_sample_every=int(defl_cfg.get("sample_every", 1) or 1),
        deflation_log_topk=int(defl_cfg.get("log_topk", 0) or 0),
        deflation_log_csv=defl_cfg.get("log_csv"),
        loss_log_path=loss_cfg.get("csv_path"),
        loss_plot_path=loss_cfg.get("plot_path"),
        system_prompt=prompt_cfg.get("system_prompt"),
        prompt_template=prompt_cfg.get("template"),
        user_template=prompt_cfg.get("user_template"),
        answer_prefix=prompt_cfg.get("answer_prefix", ""),
        text_field=data_cfg.get("text_field", "text"),
        author_field=data_cfg.get("author_field", "author"),
    )
