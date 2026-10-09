#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage 1 full-BERT masked-LM CPT on style-neutral statements."""

from __future__ import annotations

import argparse
import os

from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    set_seed,
)

from common import (
    JsonlTextDataset,
    add_perplexity,
    build_label_map,
    filtered_training_args,
    limit_rows,
    load_label_map,
    load_sequence_classifier_encoder_state,
    load_yaml,
    normalize_report_to,
    read_jsonl,
    section,
    write_metrics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Stage-1 full-BERT MLM CPT on style-neutral statements.")
    parser.add_argument("--config", default="code/configs/aa/bert_full.yaml")
    parser.add_argument("--dataset", default=None, help="Dataset settings to select from the YAML (default: default_dataset).")
    parser.add_argument("--max_train_examples", type=int, default=None)
    parser.add_argument("--max_eval_examples", type=int, default=None)
    parser.add_argument("--skip_stage0_init", action="store_true", help="Start CPT from the base BERT model.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config, args.dataset)
    model_cfg = section(cfg, "model")
    data_cfg = section(cfg, "data")
    stage0_cfg = section(section(cfg, "stages"), "stage0")
    stage1_cfg = section(section(cfg, "stages"), "stage1")
    train_cfg = dict(section(stage1_cfg, "training"))

    seed = int(cfg.get("seed", train_cfg.get("seed", 1234)))
    set_seed(seed)

    model_name = model_cfg.get("model_name_or_path", "bert-base-cased")
    output_dir = stage1_cfg.get("output_dir", "outputs/bert_deflation/gutenberg30/full/s1_atomic_cpt")
    stage0_checkpoint = stage1_cfg.get("stage0_checkpoint_path", stage0_cfg.get("output_dir"))

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForMaskedLM.from_pretrained(model_name)

    if args.skip_stage0_init:
        stage0_checkpoint = None

    if stage0_checkpoint:
        label_map_path = stage1_cfg.get("label_map_path") or os.path.join(stage0_checkpoint, "label_map.json")
        if os.path.exists(label_map_path):
            label_map = load_label_map(label_map_path)
        else:
            train_rows = read_jsonl(data_cfg["train_path"])
            eval_rows = read_jsonl(data_cfg["eval_path"])
            test_rows = read_jsonl(data_cfg["test_path"])
            label_map = build_label_map(train_rows, eval_rows, test_rows, label_field=data_cfg.get("label_field", "author"))
        encoder_state = load_sequence_classifier_encoder_state(
            stage0_checkpoint,
            num_labels=len(label_map),
            model_name_or_path=model_name,
        )
        missing, unexpected = model.bert.load_state_dict(encoder_state, strict=False)
        print(f"[Stage1-full] Loaded Stage-0 encoder | missing={len(missing)} unexpected={len(unexpected)}")

    train_rows = limit_rows(read_jsonl(data_cfg["stage1_train_path"]), args.max_train_examples, seed=seed)
    eval_rows = limit_rows(read_jsonl(data_cfg["stage1_eval_path"]), args.max_eval_examples, seed=seed)
    max_len = int(data_cfg.get("max_seq_length", model_cfg.get("max_seq_length", 512)))
    text_field = data_cfg.get("stage1_text_field", data_cfg.get("text_field", "text"))
    train_ds = JsonlTextDataset(train_rows, tokenizer, text_field=text_field, max_seq_length=max_len)
    eval_ds = JsonlTextDataset(eval_rows, tokenizer, text_field=text_field, max_seq_length=max_len)

    train_cfg.setdefault("output_dir", output_dir)
    train_cfg.setdefault("report_to", "none")
    train_cfg["report_to"] = normalize_report_to(train_cfg.get("report_to"))
    training_args = TrainingArguments(**filtered_training_args(train_cfg))
    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=float(stage1_cfg.get("mlm_probability", 0.15)),
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        data_collator=collator,
    )
    trainer.train()
    eval_metrics = add_perplexity(trainer.evaluate(eval_dataset=eval_ds, metric_key_prefix="eval"))
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    write_metrics(os.path.join(output_dir, "metrics.json"), {"eval": eval_metrics})
    print({"eval": eval_metrics})


if __name__ == "__main__":
    main()
