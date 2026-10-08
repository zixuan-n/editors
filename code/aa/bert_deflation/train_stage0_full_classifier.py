#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage 0 full-BERT authorship attribution fine-tuning."""

from __future__ import annotations

import argparse
import os

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

from common import (
    JsonlClassificationDataset,
    build_label_map,
    compute_classification_metrics,
    filtered_training_args,
    limit_rows,
    load_yaml,
    normalize_report_to,
    read_jsonl,
    save_label_map,
    section,
    write_metrics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Stage-0 full-BERT attribution classifier.")
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
    stage_cfg = section(section(cfg, "stages"), "stage0")
    train_cfg = dict(section(stage_cfg, "training"))

    seed = int(cfg.get("seed", train_cfg.get("seed", 1234)))
    set_seed(seed)

    train_rows = limit_rows(read_jsonl(data_cfg["train_path"]), args.max_train_examples, seed=seed)
    eval_rows = limit_rows(read_jsonl(data_cfg["eval_path"]), args.max_eval_examples, seed=seed)
    test_rows = limit_rows(read_jsonl(data_cfg["test_path"]), args.max_test_examples, seed=seed)
    label_map = build_label_map(train_rows, eval_rows, test_rows, label_field=data_cfg.get("label_field", "author"))

    output_dir = stage_cfg.get("output_dir", "outputs/bert_deflation/gutenberg30/full/s0")
    label_map_path = os.path.join(output_dir, "label_map.json")
    save_label_map(label_map, label_map_path)

    model_name = model_cfg.get("model_name_or_path", "bert-base-cased")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    id2label = {idx: label for label, idx in label_map.items()}
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(label_map),
        id2label=id2label,
        label2id=label_map,
    )

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
    save_label_map(label_map, label_map_path)
    write_metrics(os.path.join(output_dir, "metrics.json"), {"eval": eval_metrics, "test": test_metrics})
    print({"eval": eval_metrics, "test": test_metrics})


if __name__ == "__main__":
    main()
