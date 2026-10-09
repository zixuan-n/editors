#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage-1 continued pretraining on style-neutral statements using TRL's SFTTrainer.

This script mirrors the configuration style of run_cpt.py + YAML configs from
the Prereq_tune project, but adapts it to local JSONL datasets that already
contain newline-separated factual statements.
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import os
from typing import Any, Dict, List, Optional
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "configs"))
from finetune_config_loader import load_config

import torch
from datasets import load_dataset
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    set_seed,
)
from trl import SFTConfig, SFTTrainer

LOGGER = logging.getLogger("stage1_cpt")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def training_kwargs(values):
    result = dict(values)
    if "evaluation_strategy" in result:
        result.setdefault("eval_strategy", result.pop("evaluation_strategy"))
    result.setdefault("report_to", "none")
    if "AA_REPORT_TO" in os.environ:
        result["report_to"] = os.environ["AA_REPORT_TO"]
    return result


def _resolve_dtype(dtype_name: Optional[str]) -> Any:
    if dtype_name is None or dtype_name == "auto":
        return "auto"
    if not hasattr(torch, dtype_name):
        raise ValueError(f"Unknown torch dtype: {dtype_name}")
    return getattr(torch, dtype_name)


def _build_bitsandbytes_config(model_cfg: Dict[str, Any]) -> Optional[BitsAndBytesConfig]:
    load_in_4bit = model_cfg.get("load_in_4bit", False)
    load_in_8bit = model_cfg.get("load_in_8bit", False)
    if not load_in_4bit and not load_in_8bit:
        return None

    if load_in_4bit and load_in_8bit:
        raise ValueError("Choose either 4-bit or 8-bit loading, not both.")

    enable_cpu_offload = bool(model_cfg.get("llm_int8_enable_fp32_cpu_offload", False))

    if load_in_8bit:
        return BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_enable_fp32_cpu_offload=enable_cpu_offload,
        )

    compute_dtype = _resolve_dtype(model_cfg.get("bnb_4bit_compute_dtype", "bfloat16"))
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_quant_type=model_cfg.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_use_double_quant=model_cfg.get("bnb_4bit_use_double_quant", True),
        llm_int8_enable_fp32_cpu_offload=enable_cpu_offload,
    )


def _load_yaml_config(path: str, model_variant: Optional[str] = None,
                      dataset_variant: Optional[str] = None) -> Dict[str, Any]:
    return load_config(path, model_variant, dataset_variant)


def _filter_training_kwargs(cfg: Dict[str, Any]) -> Dict[str, Any]:
    training_options = training_kwargs(cfg)
    gradient_checkpointing_kwargs = training_options.pop("gradient_checkpointing_kwargs", None)
    allowed = set(SFTConfig.__dataclass_fields__.keys())
    unknown = set(training_options) - allowed
    if unknown:
        raise ValueError(f"Unsupported training settings: {sorted(unknown)}")
    filtered = {k: v for k, v in training_options.items() if k in allowed and v is not None}
    if gradient_checkpointing_kwargs is not None:
        filtered["gradient_checkpointing_kwargs"] = gradient_checkpointing_kwargs
    return filtered


def _load_jsonl_dataset(path: str, text_column: str, num_workers: int):
    ds = load_dataset(
        "json",
        data_files=path,
        split="train",
        num_proc=num_workers if num_workers > 1 else None,
    )
    ds = ds.filter(
        lambda ex: isinstance(ex.get(text_column, None), str)
        and len(ex[text_column].strip()) > 0,
        num_proc=num_workers if num_workers > 1 else None,
    )
    return ds


def _build_lora_config(cfg: Dict[str, Any]) -> Optional[LoraConfig]:
    if not cfg.get("enabled", False):
        return None
    return LoraConfig(
        r=cfg["r"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
        target_modules=cfg["target_modules"],
        bias=cfg["bias"],
        task_type=cfg["task_type"],
    )


def _resolve_attn_impl(attn_impl: Optional[str]) -> Optional[str]:
    if attn_impl == "flash_attention_2":
        if importlib.util.find_spec("flash_attn") is None:
            LOGGER.warning(
                "Requested attn_implementation='flash_attention_2' but flash_attn is not installed; "
                "falling back to default attention."
            )
            return None
    return attn_impl


# ---------------------------------------------------------------------------
# Main training flow
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Stage-1 continued pretraining on style-neutral statements.")
    parser.add_argument("--config", default="code/configs/aa/s1.yaml", help="Path to YAML configuration file.")
    parser.add_argument("--model_variant", default=None, help="Model settings to select from the YAML (default: default_model).")
    parser.add_argument("--dataset", default=None, help="Dataset settings to select from the YAML (default: default_dataset).")
    args = parser.parse_args()

    raw_cfg = _load_yaml_config(args.config, args.model_variant, args.dataset)
    model_cfg = dict(raw_cfg.get("model", {}))
    data_cfg = dict(raw_cfg.get("data", {}))
    lora_cfg = dict(raw_cfg.get("lora", {}))
    training_cfg = dict(raw_cfg.get("training", {}))

    if not data_cfg.get("train_path"):
        raise ValueError("`data.train_path` must be specified in the configuration.")
    if not training_cfg.get("output_dir"):
        raise ValueError("`training.output_dir` must be specified in the configuration.")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    LOGGER.info("Configuration loaded from %s", args.config)

    set_seed(training_cfg.get("seed", 42))

    # Load tokenizer
    model_name = model_cfg.get("model_name_or_path")
    trust_remote_code = model_cfg.get("trust_remote_code", False)
    if not model_name:
        raise ValueError("`model.model_name_or_path` must be specified in the configuration.")
    LOGGER.info("Loading tokenizer for %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load datasets
    train_path = data_cfg["train_path"]
    text_column = data_cfg.get("text_column", "text")
    num_workers = data_cfg.get("preprocessing_num_workers", 4)
    LOGGER.info("Loading training dataset from %s", train_path)
    train_dataset = _load_jsonl_dataset(
        train_path,
        text_column=text_column,
        num_workers=num_workers,
    )
    eval_dataset = None
    eval_path = data_cfg.get("eval_path")
    if eval_path:
        LOGGER.info("Loading evaluation dataset from %s", eval_path)
        eval_dataset = _load_jsonl_dataset(
            eval_path,
            text_column=text_column,
            num_workers=num_workers,
        )

    if data_cfg.get("shuffle", True):
        seed = data_cfg.get("seed", 42)
        train_dataset = train_dataset.shuffle(seed=seed)
        if eval_dataset is not None:
            eval_dataset = eval_dataset.shuffle(seed=seed)

    LOGGER.info("Train rows: %d | Eval rows: %d", len(train_dataset), 0 if eval_dataset is None else len(eval_dataset))

    # Prepare model & LoRA
    quant_config = _build_bitsandbytes_config(model_cfg)
    dtype = _resolve_dtype(model_cfg.get("torch_dtype", "auto"))
    attn_impl = _resolve_attn_impl(model_cfg.get("attn_implementation"))
    device_map = model_cfg.get("device_map")

    model_kwargs: Dict[str, Any] = {
        "torch_dtype": dtype,
        "trust_remote_code": trust_remote_code,
        "attn_implementation": attn_impl,
    }
    if quant_config is not None:
        model_kwargs["quantization_config"] = quant_config
        model_kwargs["device_map"] = device_map or "auto"
    elif device_map:
        model_kwargs["device_map"] = device_map

    LOGGER.info("Loading base model …")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        **model_kwargs,
    )

    if quant_config is not None:
        LOGGER.info("Preparing model for k-bit training.")
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=training_cfg.get("gradient_checkpointing", False),
        )

    model = initialize_adapter(model, model_cfg, lora_cfg)
    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()

    # Build SFTConfig: dataset options belong here, not on SFTTrainer.
    training_args = build_sft_config(training_cfg, data_cfg)

    if training_cfg.get("gradient_checkpointing", False):
        model.config.use_cache = False

    LOGGER.info("Initialising SFTTrainer …")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=None,
    )

    LOGGER.info("Starting training …")
    train_result = trainer.train()
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()
    trainer.save_model(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)
    LOGGER.info("Done.")


def initialize_adapter(model, model_cfg, lora_cfg):
    adapter_path = model_cfg.get("adapter_name_or_path")
    merge = bool(model_cfg.get("merge_adapter", False))
    trainable = bool(model_cfg.get("adapter_is_trainable", False))
    peft_config = _build_lora_config(lora_cfg)
    if adapter_path:
        if peft_config is not None and not merge:
            raise ValueError("Enable model.merge_adapter to initialize a fresh diagnostic LoRA from a warm-up adapter")
        if peft_config is None and not trainable:
            raise ValueError("The supplied adapter must be trainable when lora.enabled is false")
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=trainable and not merge)
        if merge:
            if peft_config is None:
                raise ValueError("Merging requires lora.enabled=true to create a new diagnostic adapter")
            model = model.merge_and_unload()
    if peft_config:
        LOGGER.info(
            "Applying LoRA with r=%d alpha=%d dropout=%.3f",
            lora_cfg["r"],
            lora_cfg["lora_alpha"],
            lora_cfg["lora_dropout"],
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    if not any(p.requires_grad for p in model.parameters()):
        raise ValueError("Stage 1 has no trainable parameters")
    return model


def build_sft_config(training_cfg, data_cfg):
    options = _filter_training_kwargs(training_cfg)
    options.update(dataset_text_field=data_cfg.get("text_column", "text"),
                   max_length=data_cfg.get("max_seq_length", 1536),
                   packing=data_cfg.get("packing", False))
    return SFTConfig(**options)


if __name__ == "__main__":
    main()
