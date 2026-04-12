#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fine-tune Llama-3.1-8B-Instruct on the low-accuracy Gutenberg authors dataset.

The script expects a YAML configuration file (see configs/lowacc_llama3_finetune.yaml)
that specifies model, LoRA, quantisation, and training hyperparameters.

Features:
  * Optional construction of an evaluation subset sampled per author from the test set.
  * Prompt formatting that instructs the model to output only the author's name.
  * LoRA fine-tuning with 4-bit (QLoRA-style) loading for memory efficiency.
  * CSV logging of training loss for downstream plotting.
  * Optional author filtering to train single-author adapters.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
from types import SimpleNamespace
from typing import Dict, Iterable, List, Optional, Sequence

# Restrict visible GPUs if not explicitly set (prevents accidental multi-GPU usage).
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
    AutoConfig,
)
from finetune_config_loader import load_lowacc_config

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    plt = None


DEFAULT_SYSTEM_PROMPT = None

def _parse_authors(authors: Optional[List[str]]) -> Optional[List[str]]:
    if not authors:
        return None
    parsed: List[str] = []
    for author in authors:
        parts = [p.strip() for p in author.split(",") if p.strip()]
        parsed.extend(parts if parts else [author.strip()])
    return parsed or None


def read_jsonl(path: str) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: str, rows: Iterable[Dict[str, object]]) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_eval_subset(
    test_rows: Sequence[Dict[str, object]],
    *,
    per_author: int,
    require_distinct_titles: bool,
    seed: int,
) -> List[Dict[str, object]]:
    rng = random.Random(seed)
    grouped: Dict[str, Dict[str, List[Dict[str, object]]]] = {}
    for row in test_rows:
        author = row["author"]
        title = row.get("title", "Unknown")
        grouped.setdefault(author, {}).setdefault(title, []).append(row)
    subset: List[Dict[str, object]] = []
    for author, title_map in grouped.items():
        all_rows = [item for rows in title_map.values() for item in rows]
        if len(all_rows) <= per_author:
            subset.extend(all_rows)
            continue
        if require_distinct_titles:
            titles = list(title_map.keys())
            rng.shuffle(titles)
            picks: List[Dict[str, object]] = []
            for title in titles:
                picks.append(rng.choice(title_map[title]))
                if len(picks) == per_author:
                    break
            if len(picks) < per_author:
                remaining_needed = per_author - len(picks)
                remaining_pool = [
                    row for row in all_rows if row not in picks
                ]
                if remaining_pool:
                    picks.extend(
                        rng.sample(
                            remaining_pool,
                            min(remaining_needed, len(remaining_pool)),
                        )
                    )
            subset.extend(picks[:per_author])
        else:
            subset.extend(rng.sample(all_rows, per_author))
    return subset


def _filter_by_authors(
    rows: Sequence[Dict[str, object]],
    authors: Optional[Sequence[str]],
    split_name: str,
) -> List[Dict[str, object]]:
    if not authors:
        return list(rows)
    allowed = set(authors)
    filtered = [row for row in rows if row.get("author") in allowed]
    if not filtered:
        raise ValueError(f"No rows left in {split_name} split after filtering to authors {authors}.")
    return filtered


def ensure_eval_split(cfg: SimpleNamespace, *, target_authors: Optional[Sequence[str]] = None) -> None:
    if os.path.exists(cfg.eval_path):
        return
    if not cfg.test_path or cfg.eval_subset_per_author is None:
        raise FileNotFoundError(
            f"Evaluation split {cfg.eval_path} not found and no recipe provided "
            "to construct it. Supply --create-eval or pre-create the file."
        )
    test_rows = _filter_by_authors(read_jsonl(cfg.test_path), target_authors, "test")
    eval_rows = build_eval_subset(
        test_rows,
        per_author=cfg.eval_subset_per_author,
        require_distinct_titles=cfg.eval_require_distinct_titles,
        seed=cfg.data_seed,
    )
    write_jsonl(cfg.eval_path, eval_rows)
    print(
        f"Constructed evaluation subset with {len(eval_rows)} rows "
        f"at {cfg.eval_path}."
    )


# Previous implementation (kept for reference):
# def build_prompt(authors: Sequence[str], passage: str, system_prompt: str, template: str) -> str:
#     author_list = ", ".join(authors)
#     return template.format(
#         system_prompt=system_prompt,
#         passage=passage.strip(),
#         author_list=author_list,
#     )
def build_prompt(tokenizer, authors: Sequence[str], passage: str, system_prompt: str, template: str) -> str:
    template_norm = (template or "").strip().lower()
    if template_norm in {"auto", "chat_template"}:
        if not hasattr(tokenizer, "apply_chat_template") or not getattr(tokenizer, "chat_template", None):
            raise ValueError("prompt.template=auto requires a tokenizer with a chat_template.")
        user_content = f"Passage:\n{passage.strip()}\nAuthor:"
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception as exc:
            msg = str(exc).lower()
            if "system role not supported" in msg or "system role" in msg:
                messages = [{"role": "user", "content": f"{system_prompt}\n\n{user_content}"}]
                return tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            raise

    author_list = ", ".join(authors)
    return template.format(
        system_prompt=system_prompt,
        passage=passage.strip(),
        author_list=author_list,
    )


def format_sample(
    row: Dict[str, object],
    *,
    tokenizer,
    authors: Sequence[str],
    max_seq_length: int,
    system_prompt: str,
    prompt_template: str,
    answer_prefix: str,
) -> Dict[str, List[int]]:
    prompt = build_prompt(tokenizer, authors, row["text"], system_prompt, prompt_template)
    answer = f"{answer_prefix}{row['author']}"
    prompt_ids = tokenizer(
        prompt,
        add_special_tokens=False,
    )["input_ids"]
    answer_ids = tokenizer(
        answer + tokenizer.eos_token,
        add_special_tokens=False,
    )["input_ids"]

    input_ids = prompt_ids + answer_ids
    labels = [-100] * len(prompt_ids) + answer_ids

    if len(input_ids) > max_seq_length:
        overflow = len(input_ids) - max_seq_length
        input_ids = input_ids[overflow:]
        labels = labels[overflow:]
        if all(label == -100 for label in labels):
            # Ensure we never discard the entire target span.
            input_ids = input_ids[-max_seq_length:]
            labels = labels[-max_seq_length:]

    attention_mask = [1] * len(input_ids)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def dataset_from_jsonl(
    path: str,
    *,
    tokenizer,
    authors: Sequence[str],
    max_seq_length: int,
    system_prompt: str,
    prompt_template: str,
    answer_prefix: str,
) -> Dataset:
    rows = read_jsonl(path)
    return dataset_from_rows(
        rows,
        tokenizer=tokenizer,
        authors=authors,
        max_seq_length=max_seq_length,
        system_prompt=system_prompt,
        prompt_template=prompt_template,
        answer_prefix=answer_prefix,
    )


def dataset_from_rows(
    rows: Sequence[Dict[str, object]],
    *,
    tokenizer,
    authors: Sequence[str],
    max_seq_length: int,
    system_prompt: str,
    prompt_template: str,
    answer_prefix: str,
) -> Dataset:
    formatted = [
        format_sample(
            row,
            tokenizer=tokenizer,
            authors=authors,
            max_seq_length=max_seq_length,
            system_prompt=system_prompt,
            prompt_template=prompt_template,
            answer_prefix=answer_prefix,
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
        description="Fine-tune Llama-3.1-8B on low-accuracy Gutenberg authors."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/lowacc_llama3_finetune.yaml",
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
    parser.add_argument(
        "--author",
        action="append",
        dest="authors",
        help="Fine-tune on only these authors (repeatable or comma-separated).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_lowacc_config(args.config)
    if args.seed is not None:
        cfg.data_seed = args.seed
    cfg.lora_checkpoint = args.lora_checkpoint or getattr(cfg, "lora_checkpoint", None)
    cli_authors = _parse_authors(args.authors)
    if cli_authors:
        cfg.target_authors = cli_authors

    torch.manual_seed(cfg.data_seed)
    random.seed(cfg.data_seed)

    # Use prebuilt splits directly; no auto-build.
    train_rows = _filter_by_authors(read_jsonl(cfg.train_path), cfg.target_authors, "train")
    authors = sorted({row["author"] for row in train_rows})
    eval_rows_all = _filter_by_authors(read_jsonl(cfg.eval_path), cfg.target_authors, "eval")
    eval_by_author: Dict[str, List[Dict[str, object]]] = {}
    for row in eval_rows_all:
        a = row.get("author")
        if a in authors:
            eval_by_author.setdefault(a, []).append(row)
    eval_rows: List[Dict[str, object]] = []
    rng_eval = random.Random(cfg.data_seed + 1)
    per_author_eval = 10
    for a in authors:
        bucket = eval_by_author.get(a, [])
        if not bucket:
            continue
        if len(bucket) <= per_author_eval:
            chosen = bucket
        else:
            chosen = rng_eval.sample(bucket, per_author_eval)
        eval_rows.extend(chosen)
    if not eval_rows:
        raise RuntimeError("No evaluation rows available for the selected authors.")

    model, tokenizer = load_model_and_tokenizer(cfg)

    system_prompt = getattr(cfg, "system_prompt", None) or DEFAULT_SYSTEM_PROMPT
    prompt_template = getattr(cfg, "prompt_template", None)
    answer_prefix = getattr(cfg, "answer_prefix", None)
    if answer_prefix is None:
        answer_prefix = " "

    train_dataset = dataset_from_jsonl(
        cfg.train_path,
        tokenizer=tokenizer,
        authors=authors,
        max_seq_length=cfg.max_seq_length,
        system_prompt=system_prompt,
        prompt_template=prompt_template,
        answer_prefix=answer_prefix,
    )
    eval_dataset = dataset_from_rows(
        eval_rows,
        tokenizer=tokenizer,
        authors=authors,
        max_seq_length=cfg.max_seq_length,
        system_prompt=system_prompt,
        prompt_template=prompt_template,
        answer_prefix=answer_prefix,
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
