#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fine-tune a generator to imitate an author's style using a chat-template prompt.

The dataset should be JSONL with at least: author, title, text (target passage).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
from collections import defaultdict
from types import SimpleNamespace
from typing import Dict, Iterable, List, Optional, Sequence

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)

from finetune_config_loader import load_style_gen_config

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    plt = None


DEFAULT_SYSTEM_PROMPT = None


class _SafeDict(dict):
    def __missing__(self, key: str) -> str:
        return ""


def read_jsonl(path: str) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def split_train_val(rows: Sequence[Dict[str, object]], val_ratio: float, seed: int):
    rng = random.Random(seed)
    rows = list(rows)
    rng.shuffle(rows)
    k = max(1, int(len(rows) * val_ratio))
    return rows[k:], rows[:k]


def load_facts_map(path: str) -> Dict[tuple[str, str, int], List[str]]:
    facts_map: Dict[tuple[str, str, int], List[str]] = defaultdict(list)
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            src = obj.get("source", {})
            key = (
                str(src.get("author") or ""),
                str(src.get("title") or ""),
                int(src.get("chunk_id") if src.get("chunk_id") is not None else -1),
            )
            text = _stringify(obj.get("text")).strip()
            if text:
                facts_map[key].append(text)
    return facts_map


def attach_summaries(
    rows: Sequence[Dict[str, object]],
    *,
    facts_map: Dict[tuple[str, str, int], List[str]],
    summary_field: str,
    author_field: str,
    title_field: str = "title",
) -> None:
    counters: Dict[tuple[str, str], int] = defaultdict(int)
    missing = 0
    for row in rows:
        author = _stringify(row.get(author_field)).strip()
        title = _stringify(row.get(title_field)).strip()
        chunk_id = counters[(author, title)]
        counters[(author, title)] += 1
        key = (author, title, chunk_id)
        facts = facts_map.get(key, [])
        if not facts:
            missing += 1
        row[summary_field] = " ".join(facts)
    if missing:
        print(f"[WARN] Missing summaries for {missing} rows.")


def _stringify(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def _build_format_values(
    row: Dict[str, object],
    *,
    system_prompt: Optional[str],
    text_field: str,
    author_field: str,
    summary_field: str,
) -> _SafeDict:
    values: Dict[str, str] = {}
    for key, value in row.items():
        values[key] = _stringify(value)
    values.setdefault("system_prompt", system_prompt or "")
    values.setdefault("author", _stringify(row.get(author_field, "")))
    values.setdefault("passage", _stringify(row.get(text_field, "")))
    values.setdefault("text", _stringify(row.get(text_field, "")))
    values.setdefault("content_pack", _stringify(row.get(summary_field, "")))
    return _SafeDict(values)


def build_prompt(
    tokenizer,
    row: Dict[str, object],
    *,
    system_prompt: Optional[str],
    prompt_template: str,
    user_template: Optional[str],
    text_field: str,
    author_field: str,
    summary_field: str,
) -> str:
    template_norm = (prompt_template or "").strip().lower()
    values = _build_format_values(
        row,
        system_prompt=system_prompt,
        text_field=text_field,
        author_field=author_field,
        summary_field=summary_field,
    )
    if template_norm in {"auto", "chat_template"}:
        if not hasattr(tokenizer, "apply_chat_template") or not getattr(tokenizer, "chat_template", None):
            raise ValueError("prompt.template=auto requires a tokenizer with a chat_template.")
        if not user_template:
            user_template = (
                "Author: {author}\n"
                "Task: Write a new passage (120-180 words)."
            )
        user_content = user_template.format_map(values).strip()
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_content})
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception as exc:
            msg = str(exc).lower()
            if "system role not supported" in msg or "system role" in msg:
                merged = f"{system_prompt}\n\n{user_content}" if system_prompt else user_content
                messages = [{"role": "user", "content": merged}]
                return tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            raise
    return prompt_template.format_map(values)


def format_sample(
    row: Dict[str, object],
    *,
    tokenizer,
    max_seq_length: int,
    system_prompt: Optional[str],
    prompt_template: str,
    user_template: Optional[str],
    answer_prefix: str,
    text_field: str,
    author_field: str,
    summary_field: str,
) -> Dict[str, List[int]]:
    target = _stringify(row.get(text_field, "")).strip()
    if not target:
        raise ValueError(f"Missing target text in field '{text_field}'.")
    prompt = build_prompt(
        tokenizer,
        row,
        system_prompt=system_prompt,
        prompt_template=prompt_template,
        user_template=user_template,
        text_field=text_field,
        author_field=author_field,
        summary_field=summary_field,
    )
    answer = f"{answer_prefix}{target}"
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    answer_ids = tokenizer(answer + tokenizer.eos_token, add_special_tokens=False)["input_ids"]

    input_ids = prompt_ids + answer_ids
    labels = [-100] * len(prompt_ids) + answer_ids

    if len(input_ids) > max_seq_length:
        overflow = len(input_ids) - max_seq_length
        input_ids = input_ids[overflow:]
        labels = labels[overflow:]
        if all(label == -100 for label in labels):
            input_ids = input_ids[-max_seq_length:]
            labels = labels[-max_seq_length:]

    attention_mask = [1] * len(input_ids)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def dataset_from_rows(
    rows: Sequence[Dict[str, object]],
    *,
    tokenizer,
    max_seq_length: int,
    system_prompt: Optional[str],
    prompt_template: str,
    user_template: Optional[str],
    answer_prefix: str,
    text_field: str,
    author_field: str,
    summary_field: str,
) -> Dataset:
    formatted = [
        format_sample(
            row,
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
            system_prompt=system_prompt,
            prompt_template=prompt_template,
            user_template=user_template,
            answer_prefix=answer_prefix,
            text_field=text_field,
            author_field=author_field,
            summary_field=summary_field,
        )
        for row in rows
    ]
    return Dataset.from_list(formatted)


class CausalLMDataCollator:
    def __init__(self, tokenizer, pad_to_multiple_of: int | None = 8):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        input_ids = [feature["input_ids"] for feature in features]
        attention_masks = [feature["attention_mask"] for feature in features]
        labels = [feature["labels"] for feature in features]

        batch_inputs = self.tokenizer.pad(
            {"input_ids": input_ids, "attention_mask": attention_masks},
            padding="longest",
            return_tensors="pt",
            pad_to_multiple_of=self.pad_to_multiple_of,
        )
        batch_labels = self.tokenizer.pad(
            {"input_ids": labels},
            padding="longest",
            return_tensors="pt",
            pad_to_multiple_of=self.pad_to_multiple_of,
        )["input_ids"]
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            raise ValueError("Tokenizer pad_token_id must be set for padding.")
        batch_labels = batch_labels.masked_fill(batch_labels == pad_token_id, -100)
        batch_inputs["labels"] = batch_labels
        return batch_inputs


class LossLoggerCallback(TrainerCallback):
    def __init__(self, path: str):
        self.path = path
        self._file = None
        self._writer = None

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if not args.should_log:
            return
        directory = os.path.dirname(self.path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        if self._file is None:
            self._file = open(self.path, "w", newline="", encoding="utf-8")
            self._writer = csv.writer(self._file)
            self._writer.writerow(["global_step", "samples_seen", "epoch", "loss"])
            self._file.flush()

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        if not args.should_log or logs is None:
            return
        if "loss" not in logs or self._writer is None:
            return
        self._writer.writerow(
            [
                state.global_step,
                getattr(state, "sample_seen", None),
                state.epoch,
                logs["loss"],
            ]
        )
        self._file.flush()

    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if self._file is not None:
            self._file.close()
            self._file = None
            self._writer = None


def load_model_and_tokenizer(cfg: SimpleNamespace):
    def _resolve_dtype(dtype_name: Optional[str]):
        if dtype_name is None or dtype_name == "auto":
            return torch.bfloat16
        if isinstance(dtype_name, str):
            if not hasattr(torch, dtype_name):
                raise ValueError(f"Unknown torch dtype: {dtype_name}")
            return getattr(torch, dtype_name)
        return dtype_name

    quant_cfg = None
    if cfg.quant_args:
        quant_kwargs = dict(cfg.quant_args)
        dtype_key = "bnb_4bit_compute_dtype"
        if dtype_key in quant_kwargs and isinstance(quant_kwargs[dtype_key], str):
            dtype_name = quant_kwargs[dtype_key]
            if not hasattr(torch, dtype_name):
                raise ValueError(f"Unknown torch dtype: {dtype_name}")
            quant_kwargs[dtype_key] = getattr(torch, dtype_name)
        quant_cfg = BitsAndBytesConfig(**quant_kwargs)

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_name,
        use_fast=True,
        trust_remote_code=getattr(cfg, "trust_remote_code", False),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    device_map = getattr(cfg, "device_map", None) or "auto"
    model_kwargs = {
        "device_map": device_map,
        "torch_dtype": _resolve_dtype(getattr(cfg, "torch_dtype", None)),
        "trust_remote_code": getattr(cfg, "trust_remote_code", False),
    }
    if quant_cfg:
        model_kwargs["quantization_config"] = quant_cfg
    else:
        model_kwargs["low_cpu_mem_usage"] = True

    config = AutoConfig.from_pretrained(
        cfg.model_name,
        trust_remote_code=getattr(cfg, "trust_remote_code", False),
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        config=config,
        **model_kwargs,
    )

    if quant_cfg and cfg.quant_args.get("load_in_4bit", False):
        base_model = prepare_model_for_kbit_training(base_model)

    if getattr(cfg, "lora_checkpoint", None):
        model = PeftModel.from_pretrained(
            base_model,
            cfg.lora_checkpoint,
            is_trainable=True,
        )
    else:
        lora_config = LoraConfig(
            task_type="CAUSAL_LM",
            **cfg.lora_args,
        )
        model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()
    return model, tokenizer


def build_training_arguments(cfg: SimpleNamespace) -> TrainingArguments:
    return TrainingArguments(
        output_dir=cfg.output_dir,
        report_to=["tensorboard"],
        **cfg.training_args,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune a generator to imitate an author's style."
    )
    parser.add_argument(
        "--config",
        type=str,
        # default="configs/reuter_c50_10authors_style_gen.yaml",
        default="configs/lowacc_authors_style_gen.yaml",
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional override for the data sampling seed.",
    )
    parser.add_argument(
        "--lora_checkpoint",
        type=str,
        default=None,
        help="Optional path to an existing LoRA adapter to continue training from.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_style_gen_config(args.config)
    if args.seed is not None:
        cfg.data_seed = args.seed
    cfg.lora_checkpoint = args.lora_checkpoint or getattr(cfg, "lora_checkpoint", None)

    if not cfg.train_path:
        raise ValueError("Missing data.train_path in the config.")

    torch.manual_seed(cfg.data_seed)
    random.seed(cfg.data_seed)

    train_rows = read_jsonl(cfg.train_path)
    if cfg.eval_path and os.path.exists(cfg.eval_path):
        eval_rows = read_jsonl(cfg.eval_path)
    else:
        train_rows, eval_rows = split_train_val(train_rows, val_ratio=0.05, seed=cfg.data_seed)

    if getattr(cfg, "summary_facts_path", None):
        facts_map = load_facts_map(cfg.summary_facts_path)
        attach_summaries(
            train_rows,
            facts_map=facts_map,
            summary_field=cfg.summary_field,
            author_field=cfg.author_field,
        )
        attach_summaries(
            eval_rows,
            facts_map=facts_map,
            summary_field=cfg.summary_field,
            author_field=cfg.author_field,
        )

    model, tokenizer = load_model_and_tokenizer(cfg)

    system_prompt = getattr(cfg, "system_prompt", None) or DEFAULT_SYSTEM_PROMPT
    prompt_template = getattr(cfg, "prompt_template", None)
    if not prompt_template:
        raise ValueError("Missing prompt.template in the config.")
    user_template = getattr(cfg, "user_template", None)
    answer_prefix = getattr(cfg, "answer_prefix", "")

    train_dataset = dataset_from_rows(
        train_rows,
        tokenizer=tokenizer,
        max_seq_length=cfg.max_seq_length,
        system_prompt=system_prompt,
        prompt_template=prompt_template,
        user_template=user_template,
        answer_prefix=answer_prefix,
        text_field=cfg.text_field,
        author_field=cfg.author_field,
        summary_field=cfg.summary_field,
    )
    eval_dataset = dataset_from_rows(
        eval_rows,
        tokenizer=tokenizer,
        max_seq_length=cfg.max_seq_length,
        system_prompt=system_prompt,
        prompt_template=prompt_template,
        user_template=user_template,
        answer_prefix=answer_prefix,
        text_field=cfg.text_field,
        author_field=cfg.author_field,
        summary_field=cfg.summary_field,
    )

    training_args = build_training_arguments(cfg)
    if getattr(training_args, "gradient_checkpointing", False):
        model.config.use_cache = False

    data_collator = CausalLMDataCollator(
        tokenizer,
        pad_to_multiple_of=cfg.pad_to_multiple_of,
    )

    callbacks: List[TrainerCallback] = []
    if cfg.loss_log_path:
        callbacks.append(LossLoggerCallback(cfg.loss_log_path))

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=callbacks,
    )

    trainer.train()
    trainer.save_model(cfg.output_dir)
    trainer.save_state()

    if cfg.loss_plot_path and cfg.loss_log_path:
        if plt is None:
            print(
                "matplotlib not available; skipping loss plot. "
                "Install matplotlib to enable plotting."
            )
        else:
            try:
                steps: List[float] = []
                losses: List[float] = []
                with open(cfg.loss_log_path, "r", encoding="utf-8") as handle:
                    reader = csv.DictReader(handle)
                    for row in reader:
                        try:
                            steps.append(float(row["global_step"]))
                            losses.append(float(row["loss"]))
                        except (KeyError, ValueError):
                            continue
                if steps and losses:
                    plt.figure(figsize=(8, 4.5))
                    plt.plot(steps, losses, marker="o", linewidth=1.0)
                    plt.xlabel("Global Step")
                    plt.ylabel("Training Loss")
                    plt.title("Finetuning Loss Curve")
                    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
                    directory = os.path.dirname(cfg.loss_plot_path)
                    if directory:
                        os.makedirs(directory, exist_ok=True)
                    plt.tight_layout()
                    plt.savefig(cfg.loss_plot_path)
                    plt.close()
                    print(f"Saved loss plot to {cfg.loss_plot_path}")
                else:
                    print("Loss CSV empty or malformed; skipping plot.")
            except FileNotFoundError:
                print("Loss CSV not found; skipping plot.")


if __name__ == "__main__":
    main()
