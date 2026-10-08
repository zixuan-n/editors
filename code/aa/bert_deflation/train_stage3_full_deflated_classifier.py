#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage 3 full-BERT attribution fine-tuning with full-delta input deflation."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

from common import (
    DeflationStats,
    JsonlClassificationDataset,
    build_full_delta_bases,
    build_label_map,
    compute_classification_metrics,
    filtered_training_args,
    limit_rows,
    load_label_map,
    load_mlm_encoder_state,
    load_sequence_classifier_encoder_state,
    load_yaml,
    normalize_report_to,
    read_jsonl,
    register_deflation_hooks,
    save_label_map,
    section,
    write_metrics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Stage-3 full-BERT deflated attribution classifier.")
    parser.add_argument("--config", default="code/configs/aa/bert_full.yaml")
    parser.add_argument("--dataset", default=None, help="Dataset settings to select from the YAML (default: default_dataset).")
    parser.add_argument("--max_train_examples", type=int, default=None)
    parser.add_argument("--max_eval_examples", type=int, default=None)
    parser.add_argument("--max_test_examples", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config, args.dataset)
    model_cfg = section(cfg, "model")
    data_cfg = section(cfg, "data")
    defl_cfg = section(cfg, "deflation")
    stage0_cfg = section(section(cfg, "stages"), "stage0")
    stage1_cfg = section(section(cfg, "stages"), "stage1")
    stage3_cfg = section(section(cfg, "stages"), "stage3")
    train_cfg = dict(section(stage3_cfg, "training"))

    seed = int(cfg.get("seed", train_cfg.get("seed", 1234)))
    set_seed(seed)

    output_dir = stage3_cfg.get("output_dir", "outputs/bert_deflation/gutenberg30/full/s3_deflated")
    stage0_checkpoint = stage3_cfg.get("stage0_checkpoint_path", stage0_cfg.get("output_dir"))
    stage1_checkpoint = stage3_cfg.get("stage1_checkpoint_path", stage1_cfg.get("output_dir"))
    if not stage0_checkpoint:
        raise ValueError("Stage 3 full requires a Stage-0 checkpoint path.")
    if not stage1_checkpoint:
        raise ValueError("Stage 3 full requires a Stage-1 checkpoint path.")

    train_rows = limit_rows(read_jsonl(data_cfg["train_path"]), args.max_train_examples, seed=seed)
    eval_rows = limit_rows(read_jsonl(data_cfg["eval_path"]), args.max_eval_examples, seed=seed)
    test_rows = limit_rows(read_jsonl(data_cfg["test_path"]), args.max_test_examples, seed=seed)
    label_map_path = stage3_cfg.get("label_map_path") or os.path.join(stage0_checkpoint, "label_map.json")
    if os.path.exists(label_map_path):
        label_map = load_label_map(label_map_path)
    else:
        label_map = build_label_map(train_rows, eval_rows, test_rows, label_field=data_cfg.get("label_field", "author"))

    model_name = model_cfg.get("model_name_or_path", "bert-base-cased")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    id2label = {idx: label for label, idx in label_map.items()}
    model = AutoModelForSequenceClassification.from_pretrained(
        stage0_checkpoint,
        num_labels=len(label_map),
        id2label=id2label,
        label2id=label_map,
    )

    stage0_state = load_sequence_classifier_encoder_state(
        stage0_checkpoint,
        num_labels=len(label_map),
        model_name_or_path=model_name,
    )
    stage1_state = load_mlm_encoder_state(stage1_checkpoint)
    bases = build_full_delta_bases(
        stage0_state,
        stage1_state,
        include=defl_cfg.get("include_module", ["query", "key", "value"]),
        basis_tol=float(defl_cfg.get("basis_tol", 1e-6)),
        basis_solver=str(defl_cfg.get("basis_solver", "svd")),
        max_rank=defl_cfg.get("max_rank"),
    )
    save_u_sem = defl_cfg.get("save_u_sem")
    if save_u_sem:
        Path(save_u_sem).parent.mkdir(parents=True, exist_ok=True)
        torch.save(bases, save_u_sem)
        print(f"[Deflation-full] Saved bases to {save_u_sem}")

    stats = DeflationStats(sample_every=int(defl_cfg.get("sample_every", 1)))
    handles = register_deflation_hooks(model, bases, stats=stats)

    max_len = int(data_cfg.get("max_seq_length", model_cfg.get("max_seq_length", 512)))
    text_field = data_cfg.get("text_field", "text")
    label_field = data_cfg.get("label_field", "author")
    train_ds = JsonlClassificationDataset(
        train_rows, tokenizer, label_map, text_field=text_field, label_field=label_field, max_seq_length=max_len
    )
    eval_ds = JsonlClassificationDataset(
        eval_rows, tokenizer, label_map, text_field=text_field, label_field=label_field, max_seq_length=max_len
    )
    test_ds = JsonlClassificationDataset(
        test_rows, tokenizer, label_map, text_field=text_field, label_field=label_field, max_seq_length=max_len
    )

    train_cfg.setdefault("output_dir", output_dir)
    train_cfg.setdefault("report_to", "none")
    train_cfg["report_to"] = normalize_report_to(train_cfg.get("report_to"))
    training_args = TrainingArguments(**filtered_training_args(train_cfg))

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_classification_metrics,
    )
    trainer.train()
    eval_metrics = trainer.evaluate(eval_dataset=eval_ds, metric_key_prefix="eval")
    test_metrics = trainer.evaluate(eval_dataset=test_ds, metric_key_prefix="test")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    save_label_map(label_map, os.path.join(output_dir, "label_map.json"))

    deflation_csv = defl_cfg.get("log_csv") or os.path.join(output_dir, "deflation_stats.csv")
    stats.write_csv(deflation_csv)
    for handle in handles:
        handle.remove()

    write_metrics(
        os.path.join(output_dir, "metrics.json"),
        {"eval": eval_metrics, "test": test_metrics, "deflation_stats_csv": deflation_csv},
    )
    print({"eval": eval_metrics, "test": test_metrics, "deflation_stats_csv": deflation_csv})


if __name__ == "__main__":
    main()
