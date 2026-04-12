#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage-1 atomic facts continued pretraining using TRL's SFTTrainer.

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

import torch
import yaml
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    set_seed,
)
from trl import SFTTrainer

LOGGER = logging.getLogger("stage1_cpt")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
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


def _load_yaml_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Config at {path} must be a YAML mapping.")
    return data


def _filter_training_kwargs(cfg: Dict[str, Any]) -> Dict[str, Any]:
    training_kwargs = dict(cfg)
    gradient_checkpointing_kwargs = training_kwargs.pop("gradient_checkpointing_kwargs", None)
    allowed = set(TrainingArguments.__dataclass_fields__.keys())
    filtered = {k: v for k, v in training_kwargs.items() if k in allowed and v is not None}
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
    if not cfg["enabled"]:
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
    parser = argparse.ArgumentParser(description="Stage-1 atomic facts continued pretraining.")
    parser.add_argument("--config", required=True, default="configs/stage1_atomic_cpt.yaml", help="Path to YAML configuration file.")
    args = parser.parse_args()

    raw_cfg = _load_yaml_config(args.config)
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
        use_fast=False,
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

    peft_config = _build_lora_config(lora_cfg)
    if peft_config:
        LOGGER.info(
            "Applying LoRA with r=%d alpha=%d dropout=%.3f",
            lora_cfg["r"],
            lora_cfg["lora_alpha"],
            lora_cfg["lora_dropout"],
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    # Build TrainingArguments
    training_kwargs = _filter_training_kwargs(training_cfg)
    training_args = TrainingArguments(**training_kwargs)

    if training_cfg.get("gradient_checkpointing", False):
        model.config.use_cache = False

    LOGGER.info("Initialising SFTTrainer …")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        dataset_text_field=text_column,
        max_seq_length=data_cfg.get("max_seq_length", 1536),
        packing=data_cfg.get("packing", False),
        peft_config=None,  # already applied if enabled
    )

    LOGGER.info("Starting training …")
    train_result = trainer.train()
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()

    LOGGER.info("Saving model and tokenizer to %s", training_args.output_dir)
    trainer.save_model(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)

    LOGGER.info("Done.")


if __name__ == "__main__":
    main()
