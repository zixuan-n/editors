#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate generated passages with author-attribution evaluators.

This script does NOT generate new passages; it scores existing JSONL files.
Each JSONL line should contain: {"author": ..., "text": ..., "sample_id": ..., "generator": ...}
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from peft import PeftModel
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


# DEFAULT_OUTPUT_DIR = "eval_generator/style_gen_eval_lowacc_gpt4o_no_summary"
#DEFAULT_OUTPUT_DIR = "eval_generator/style_gen_eval_lowacc_from_authors_summary"
#DEFAULT_OUTPUT_DIR = "eval_generator/style_gen_lowacc_from_authors_general_summary/close"
#DEFAULT_OUTPUT_DIR = "eval_generator/style_gen_lowacc_from_authors_general_summary/left_padding"
#DEFAULT_OUTPUT_DIR = "eval_generator/style_gen_lowacc_from_authors_summary_new_model/left_padding"
DEFAULT_OUTPUT_DIR = "eval_generator/style_gen_lowacc_storytrans"
DEFAULT_GENERATED: Sequence[Tuple[str, str]] = (
    # (
    #     "deflated_Llama3_8B_lowacc_literary_gen_deltaW_2",
    #     "eval_generator/style_gen_eval_lowacc_fix_topic/generated_deflated_Llama3_8B_lowacc_literary_gen_deltaW_2_A2G.jsonl",
    # ),
    # (
    #     "deflated_Llama3_8B_lowacc_literary_gen_deltaW",
    #     "eval_generator/style_gen_eval_lowacc_fix_topic/generated_deflated_Llama3_8B_lowacc_literary_gen_deltaW_A2G.jsonl",
    # ),
    # (
    #     "llama3_lowacc_literary_style_gen",
    #     "eval_generator/style_gen_eval_lowacc_fix_topic/generated_llama3_lowacc_literary_style_gen_A2G.jsonl",
    # ),
    # (
    #     "deflated_Llama3_8B_lowacc_literary_gen_deltaW_2",
    #     "eval_generator/style_gen_eval_lowacc_free_topic/generated_deflated_Llama3_8B_lowacc_literary_gen_deltaW_2.jsonl",
    # ),
    # (
    #     "deflated_Llama3_8B_lowacc_literary_gen_deltaW",
    #     "eval_generator/style_gen_eval_lowacc_free_topic/generated_deflated_Llama3_8B_lowacc_literary_gen_deltaW.jsonl",
    # ),
    # (
    #     "lowacc_gpt4o",
    #     "eval_generator/style_gen_gpt4o_mini.jsonl",
    # ),
    # (
    #     "lowacc_gpt4o_no_summary",
    #     "eval_generator/style_gen_gpt4o_mini_no_summary.jsonl",
    # ),
    # (
    #     "style_gen_lowacc_from_authors_summary_new_model_deflated",
    #     "eval_generator/style_gen_lowacc_from_authors_summary_new_model/generated_deflated_llama3_lowacc_style_gen_with_summary_from_base_A2G.jsonl",
    # ),
    # (
    #     "style_gen_lowacc_from_authors_summary_new_model",
    #     "eval_generator/style_gen_lowacc_from_authors_summary_new_model/generated_llama3_lowacc_style_gen_with_summary_A2G.jsonl",
    # ),
    # (
    #     "style_gen_lowacc_from_authors_summary_deltaW_2",
    #     "eval_generator/style_gen_lowacc_from_authors_general_summary/generated_deflated_Llama3_8B_lowacc_literary_gen_deltaW_2_A2G.jsonl",
    # ),
    # (
    #     "style_gen_lowacc_from_authors_summary_deltaW",
    #     "eval_generator/style_gen_lowacc_from_authors_general_summary/generated_deflated_Llama3_8B_lowacc_literary_gen_deltaW_A2G.jsonl",
    # ),
    # (
    #     "style_gen_lowacc_from_authors_summary_from_base",
    #     "eval_generator/style_gen_lowacc_from_authors_general_summary/generated_deflated_llama3_lowacc_style_gen_with_summary_from_base_A2G.jsonl",
    # ),
    # (
    #     "style_gen_lowacc_from_authors_summary_deltaW_2_300",
    #     "eval_generator/style_gen_lowacc_from_authors_general_summary_300/generated_deflated_Llama3_8B_lowacc_literary_gen_deltaW_2_A2G.jsonl",
    # ),
    # (
    #     "style_gen_lowacc_from_authors_summary_deflate_300",
    #     "eval_generator/style_gen_lowacc_from_authors_general_summary_300/generated_deflated_llama3_lowacc_style_gen_with_summary_A2G.jsonl",
    # ),
    # (
    #     "style_gen_lowacc_from_authors_summary_deflate_from_base_300",
    #     "eval_generator/style_gen_lowacc_from_authors_general_summary_300/generated_deflated_llama3_lowacc_style_gen_with_summary_from_base_A2G.jsonl",
    # ),    
    # (
    #     "style_gen_lowacc_from_authors_summary_deflate_from_base_1_300",
    #     "eval_generator/style_gen_lowacc_from_authors_general_summary_300/generated_deflated_llama3_lowacc_style_gen_with_summary_from_base_1_A2G.jsonl",
    # ), 
    # (
    #     "style_gen_lowacc_from_authors_summary_deflateW",
    #     "eval_generator/style_gen_eval_lowacc_from_authors_summary/generated_deflated_Llama3_8B_lowacc_literary_gen_deltaW_2_A2G.jsonl"
    # ), 
    # (
    #     "style_gen_lowacc_from_authors_summary_deflateW2",
    #     "eval_generator/style_gen_eval_lowacc_from_authors_summary/generated_deflated_Llama3_8B_lowacc_literary_gen_deltaW_A2G.jsonl"
    # ),
    (
        "style_gen_lowacc_storytrans",
        "eval_generator/style_gen_eval_lowacc_fix_topic/generated_storytrans_stage1_A2G.jsonl",
    ),
)
DEFAULT_EVALUATORS: Sequence[Tuple[str, str, str]] = (
    #("llama3_lowacc_lora", "meta-llama/Llama-3.1-8B-Instruct", "outputs/llama3_lowacc_lora"),
    #("llama3_lowacc_lora_ckpt800", "meta-llama/Llama-3.1-8B-Instruct", "outputs/llama3_lowacc_lora/checkpoint-800"),
    ("stage2_style_deflated_cpt_ver", "meta-llama/Llama-3.1-8B-Instruct", "outputs/stage2_style_deflated_cpt_ver/adapter_deflated_stage2"),
    #("stage2_style_deflated_1", "meta-llama/Llama-3.1-8B-Instruct", "outputs/stage2_style_deflated_1/adapter_deflated_stage2"),
    ("deflated_from_base", "meta-llama/Llama-3.1-8B-Instruct", "outputs/deflated_from_base/adapter_deflated_stage2"),
    ("deflated_from_800", "meta-llama/Llama-3.1-8B-Instruct", "outputs/deflated_from_800/adapter_deflated_stage2"),
    ("Gemma2_lowacc", "google/gemma-2-9b-it", "outputs/Gemma2_lowacc_evaluator"),
    ("Phi4mini_lowacc", "microsoft/Phi-4-mini-instruct", "outputs/Phi4_lowacc_evaluator"),
)

EVAL_SYSTEM_PROMPT = (
    "You are an expert literary analyst. Identify the author of the provided passage based on their writing style instead of the content. "
    "Answer with only the exact author name."
)
#DEFAULT_PROMPTS_PATH = "splits/lowacc_authors/prompts_A2G_general.jsonl"
DEFAULT_PROMPTS_PATH = "splits/lowacc_authors/prompts_A2G_summary.jsonl"


@dataclass
class AdapterSpec:
    label: str
    base_model: str
    adapter_path: Optional[str]


@dataclass
class GenSample:
    generator: str
    author: str
    text: str
    sample_id: int


def read_jsonl(path: str) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_prompt_summaries(path: str) -> Dict[Tuple[str, str], str]:
    rows = read_jsonl(path)
    summaries: Dict[Tuple[str, str], str] = {}
    for row in rows:
        author = str(row.get("author", "")).strip()
        category = str(row.get("category", "")).strip()
        summary = str(row.get("content_pack", "")).strip()
        if not summary:
            summary = str(row.get("summary", "")).strip()
        if not summary:
            summary = str(row.get("text", "")).strip()
        if not author or not category or not summary:
            continue
        key = (author, category)
        if key in summaries and summaries[key] != summary:
            raise ValueError(
                f"Prompt summaries conflict for author='{author}', category='{category}'."
            )
        summaries[key] = summary
    return summaries


def resolve_summary(
    row: Dict[str, object],
    summaries: Dict[Tuple[str, str], str],
) -> Tuple[str, str]:
    author = str(row.get("author", "")).strip()
    category = str(row.get("category", "")).strip()
    if author and category:
        key = (author, category)
        if key in summaries:
            return summaries[key], "prompt"
    fallback = str(row.get("content_pack", "")).strip()
    if not fallback:
        fallback = str(row.get("summary", "")).strip()
    if not fallback:
        fallback = str(row.get("text", "")).strip()
    if fallback:
        return fallback, "generated_row"
    return "", "missing"


def write_jsonl(path: str, rows: Iterable[Dict[str, object]]) -> None:
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_text(text: str) -> str:
    text = " ".join(text.strip().split())
    return text.strip(" .,:;\"'()[]{}").lower()


def match_author(pred: str, authors: Sequence[str]) -> Optional[str]:
    norm_pred = normalize_text(pred)
    if not norm_pred:
        return None
    exact = {normalize_text(a): a for a in authors}
    if norm_pred in exact:
        return exact[norm_pred]
    best = None
    for author in authors:
        if normalize_text(author) in norm_pred:
            if best is None or len(author) > len(best):
                best = author
    return best


def compute_rouge_scores(hyp: str, ref: str, scorer) -> Dict[str, float]:
    scores = scorer.score(ref, hyp)
    return {
        "rouge1": float(scores["rouge1"].fmeasure),
        "rouge2": float(scores["rouge2"].fmeasure),
        "rougeL": float(scores["rougeL"].fmeasure),
    }


def compute_sts_pairs(
    texts_a: List[str],
    texts_b: List[str],
    *,
    model_name: str,
    batch_size: int,
) -> List[float]:
    try:
        import numpy as np
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Missing dependency: numpy") from exc
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Missing dependency: sentence-transformers") from exc

    model = SentenceTransformer(model_name)
    emb_a = model.encode(texts_a, batch_size=batch_size, normalize_embeddings=True)
    emb_b = model.encode(texts_b, batch_size=batch_size, normalize_embeddings=True)
    sims = (emb_a * emb_b).sum(axis=1)
    return [float(x) for x in np.asarray(sims)]


def _resolve_nli_label_ids(model) -> Tuple[int, int, int]:
    id2label = getattr(model.config, "id2label", {}) or {}
    label_map: Dict[str, int] = {}
    for idx, label in id2label.items():
        label_norm = str(label).strip().lower()
        if "entail" in label_norm:
            label_map["entailment"] = int(idx)
        elif "neutral" in label_norm:
            label_map["neutral"] = int(idx)
        elif "contradiction" in label_norm:
            label_map["contradiction"] = int(idx)
    if len(label_map) == 3:
        return (
            label_map["entailment"],
            label_map["neutral"],
            label_map["contradiction"],
        )
    if getattr(model.config, "num_labels", 0) == 3:
        # Fallback to the common order: contradiction, neutral, entailment.
        return 2, 1, 0
    raise ValueError(f"Unsupported NLI label mapping: {id2label}")


def compute_nli_pairs(
    premises: List[str],
    hypotheses: List[str],
    *,
    model_name: str,
    batch_size: int,
    max_length: int,
) -> List[Dict[str, object]]:
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Missing dependency: transformers") from exc

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    entail_id, neutral_id, contradiction_id = _resolve_nli_label_ids(model)

    effective_max_length = max_length
    model_max = getattr(model.config, "max_position_embeddings", None)
    if isinstance(model_max, int) and model_max > 0:
        effective_max_length = min(effective_max_length, model_max)
    tok_max = getattr(tokenizer, "model_max_length", None)
    if isinstance(tok_max, int) and 0 < tok_max < 100_000:
        effective_max_length = min(effective_max_length, tok_max)
    if effective_max_length != max_length:
        print(f"[NLI] Capping max_length to {effective_max_length} (model/tokenizer limit).")

    results: List[Dict[str, object]] = []
    for i in range(0, len(premises), batch_size):
        batch_prem = premises[i : i + batch_size]
        batch_hyp = hypotheses[i : i + batch_size]
        inputs = tokenizer(
            batch_prem,
            batch_hyp,
            padding=True,
            truncation="only_first",
            max_length=effective_max_length,
            return_tensors="pt",
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = F.softmax(logits, dim=-1).cpu().tolist()
        for row in probs:
            results.append(
                {
                    "nli_entailment": float(row[entail_id]),
                    "nli_neutral": float(row[neutral_id]),
                    "nli_contradiction": float(row[contradiction_id]),
                }
            )
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return results


def resolve_adapter_path(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    adapter_root = Path(path)
    if adapter_root.is_dir():
        deflated = adapter_root / "adapter_deflated_stage2"
        if (deflated / "adapter_model.safetensors").exists():
            return str(deflated)
        if (adapter_root / "adapter_model.safetensors").exists():
            return str(adapter_root)
    if adapter_root.exists():
        return str(adapter_root)
    raise FileNotFoundError(f"Adapter path not found: {path}")


def parse_generator_spec(spec: str) -> Tuple[str, str]:
    if "=" not in spec:
        raise ValueError(f"Invalid generator spec '{spec}', expected label=path.")
    label, path = spec.split("=", 1)
    label = label.strip()
    path = path.strip()
    if not label:
        raise ValueError(f"Generator label missing in specification '{spec}'.")
    if not path:
        raise ValueError(f"Generator path missing in specification '{spec}'.")
    return label, path


def parse_adapter_spec(spec: str) -> AdapterSpec:
    if "=" not in spec:
        raise ValueError(f"Invalid evaluator spec '{spec}', expected label=adapter_path|base_model.")
    label, rest = spec.split("=", 1)
    label = label.strip()
    if "|" not in rest:
        raise ValueError(f"Missing base model for evaluator '{spec}'. Use label=adapter_path|base_model.")
    adapter_raw, base_raw = rest.split("|", 1)
    adapter_path = resolve_adapter_path(adapter_raw.strip())
    base_model = base_raw.strip()
    if not base_model:
        raise ValueError(f"Base model missing for evaluator '{spec}'.")
    return AdapterSpec(label=label, base_model=base_model, adapter_path=adapter_path)


def load_specs(
    specs: Optional[Sequence[str]],
    defaults: Sequence[Tuple[str, str, str]],
) -> List[AdapterSpec]:
    if specs:
        return [parse_adapter_spec(spec) for spec in specs]
    loaded = []
    for label, base_model, adapter_path in defaults:
        loaded.append(
            AdapterSpec(
                label=label,
                base_model=base_model,
                adapter_path=resolve_adapter_path(adapter_path),
            )
        )
    return loaded


def build_chat_prompt(
    tokenizer,
    *,
    system_prompt: str,
    passage: str,
) -> str:
    user_content = f"Passage:\n{passage}\nAuthor:"
    if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
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
                return tokenizer.apply_chat_template(
                    [{"role": "user", "content": merged}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            raise
    return f"{system_prompt}\n\n{user_content}\n"


def load_model(
    base_model: str,
    adapter_path: Optional[str],
    *,
    use_4bit: bool,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    quant_config = None
    if use_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        quantization_config=quant_config,
        low_cpu_mem_usage=True,
    )
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model, tokenizer


@torch.no_grad()
def evaluate_classifier_open(
    model,
    tokenizer,
    *,
    samples: Sequence[GenSample],
    authors: Sequence[str],
    max_new_tokens: int,
    batch_size: int,
) -> Tuple[int, int, List[Dict[str, object]]]:
    total = 0
    correct = 0
    rows: List[Dict[str, object]] = []
    progress = tqdm(total=len(samples), desc="Evaluating", unit="sample")
    for i in range(0, len(samples), batch_size):
        chunk = samples[i : i + batch_size]
        prompts = [
            build_chat_prompt(
                tokenizer,
                system_prompt=EVAL_SYSTEM_PROMPT,
                passage=sample.text,
            )
            for sample in chunk
        ]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            pad_token_id=tokenizer.eos_token_id,
        )
        gen_tokens = output[:, inputs["input_ids"].shape[1]:]
        preds = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)
        for sample, pred in zip(chunk, preds):
            pred_author = match_author(pred, authors)
            is_correct = pred_author == sample.author
            total += 1
            if is_correct:
                correct += 1
            rows.append(
                {
                    "author": sample.author,
                    "prediction": pred.strip(),
                    "prediction_normalized": normalize_text(pred),
                    "prediction_matched": pred_author,
                    "correct": bool(is_correct),
                }
            )
        progress.update(len(chunk))
    progress.close()
    return correct, total, rows


@torch.no_grad()
def predict_author_closedset(
    model,
    tokenizer,
    *,
    text: str,
    labels: Sequence[str],
    device: torch.device,
    max_length: int,
) -> Tuple[str, float]:
    prompt = build_chat_prompt(
        tokenizer,
        system_prompt=EVAL_SYSTEM_PROMPT,
        passage=text,
    )
    enc = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    ).to(device)
    ctx = enc["input_ids"]
    scores: List[float] = []
    for lab in labels:
        lab_ids = tokenizer.encode(f" {lab}", add_special_tokens=False)
        if not lab_ids:
            scores.append(float("-inf"))
            continue
        ctx_lab = ctx
        logsum = 0.0
        for tid in lab_ids:
            out = model(input_ids=ctx_lab)
            logp = out.logits[:, -1, :].log_softmax(dim=-1)[0, tid].item()
            ctx_lab = torch.cat(
                [ctx_lab, torch.tensor([[tid]], device=ctx_lab.device)],
                dim=1,
            )
            logsum += logp
        scores.append(logsum / max(1, len(lab_ids)))
    best_idx = int(torch.tensor(scores).argmax())
    return labels[best_idx], scores[best_idx]


@torch.no_grad()
def evaluate_classifier_closed(
    model,
    tokenizer,
    *,
    samples: Sequence[GenSample],
    authors: Sequence[str],
    max_length: int,
) -> Tuple[int, int, List[Dict[str, object]]]:
    total = 0
    correct = 0
    rows: List[Dict[str, object]] = []
    progress = tqdm(total=len(samples), desc="Evaluating (closed-set)", unit="sample")
    for sample in samples:
        best_author, best_score = predict_author_closedset(
            model,
            tokenizer,
            text=sample.text,
            labels=authors,
            device=model.device,
            max_length=max_length,
        )
        is_correct = best_author == sample.author
        total += 1
        if is_correct:
            correct += 1
        rows.append(
            {
                "author": sample.author,
                "prediction": best_author,
                "prediction_score": best_score,
                "correct": bool(is_correct),
            }
        )
        progress.update(1)
    progress.close()
    return correct, total, rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate existing generated passages with evaluator adapters.")
    ap.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--generated", action="append", default=None, help="label=path (repeatable)")
    ap.add_argument("--evaluator", action="append", default=None, help="label=adapter_path|base_model (repeatable)")
    ap.add_argument("--eval_use_4bit", action="store_true")
    ap.add_argument("--eval_mode", choices=["open", "closed", "both"], default="open")
    ap.add_argument("--eval_max_new_tokens", type=int, default=8)
    ap.add_argument("--eval_max_length", type=int, default=768)
    ap.add_argument("--eval_batch_size", type=int, default=4)
    ap.add_argument("--prompts", default=DEFAULT_PROMPTS_PATH, help="Prompt JSONL with summaries.")
    ap.add_argument("--skip_summary", action="store_true", help="Skip summary-match (ROUGE/STS/NLI) metrics.")
    ap.add_argument("--skip_sts", action="store_true")
    ap.add_argument("--sts_model", default="all-MiniLM-L6-v2")
    ap.add_argument("--sts_batch_size", type=int, default=64)
    ap.add_argument("--skip_nli", action="store_true")
    ap.add_argument("--nli_model", default="roberta-large-mnli")
    ap.add_argument("--nli_batch_size", type=int, default=16)
    ap.add_argument("--nli_max_length", type=int, default=512)
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.generated:
        gen_specs = [parse_generator_spec(spec) for spec in args.generated]
    else:
        gen_specs = list(DEFAULT_GENERATED)

    generated_files: Dict[str, Path] = {}
    generated_rows: Dict[str, List[Dict[str, object]]] = {}
    all_authors: List[str] = []
    for label, path in gen_specs:
        gen_path = Path(path)
        if not gen_path.exists():
            raise FileNotFoundError(f"Generated file not found: {gen_path}")
        generated_files[label] = gen_path
        gen_rows = read_jsonl(str(gen_path))
        generated_rows[label] = gen_rows
        all_authors.extend(
            [
                str(r.get("author", "")).strip()
                for r in gen_rows
                if r.get("author")
            ]
        )

    authors = sorted({a for a in all_authors if a})
    if not authors:
        raise ValueError("No authors found in generated files.")
    print(f"Loaded {len(authors)} authors from generated files.")

    if not args.skip_summary:
        prompt_path = Path(args.prompts)
        if not prompt_path.exists():
            raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
        prompt_summaries = load_prompt_summaries(str(prompt_path))
        if not prompt_summaries:
            raise ValueError(f"No prompt summaries loaded from {prompt_path}")
        print(f"Loaded {len(prompt_summaries)} prompt summaries from {prompt_path}.")

        try:
            from rouge_score import rouge_scorer
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("Missing dependency: rouge-score") from exc
        rouge = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

        gen_metric_rows: List[Dict[str, object]] = []
        for gen_label, gen_rows in generated_rows.items():
            detail_rows: List[Dict[str, object]] = []
            gen_texts: List[str] = []
            summary_texts: List[str] = []
            missing_summary = 0
            skipped = 0
            for row in gen_rows:
                hyp = str(row.get("text", "")).strip()
                summary, summary_source = resolve_summary(row, prompt_summaries)
                if not summary or not hyp:
                    if not summary:
                        missing_summary += 1
                    skipped += 1
                    continue
                rouge_scores = compute_rouge_scores(hyp, summary, rouge)
                detail_rows.append(
                    {
                        "generator": gen_label,
                        "author": str(row.get("author", "")).strip(),
                        "category": str(row.get("category", "")).strip(),
                        "sample_id": int(row.get("sample_id", 0)),
                        "summary": summary,
                        "summary_source": summary_source,
                        "rouge1": rouge_scores["rouge1"],
                        "rouge2": rouge_scores["rouge2"],
                        "rougeL": rouge_scores["rougeL"],
                    }
                )
                gen_texts.append(hyp)
                summary_texts.append(summary)

            if not args.skip_sts and detail_rows:
                sts_scores = compute_sts_pairs(
                    gen_texts,
                    summary_texts,
                    model_name=args.sts_model,
                    batch_size=args.sts_batch_size,
                )
                for row, sts in zip(detail_rows, sts_scores):
                    row["sts"] = sts
            else:
                for row in detail_rows:
                    row["sts"] = 0.0

            if not args.skip_nli and detail_rows:
                nli_scores = compute_nli_pairs(
                    gen_texts,
                    summary_texts,
                    model_name=args.nli_model,
                    batch_size=args.nli_batch_size,
                    max_length=args.nli_max_length,
                )
                for row, nli in zip(detail_rows, nli_scores):
                    row.update(nli)
                    scores = {
                        "entailment": row["nli_entailment"],
                        "neutral": row["nli_neutral"],
                        "contradiction": row["nli_contradiction"],
                    }
                    row["nli_label"] = max(scores, key=scores.get)
            else:
                for row in detail_rows:
                    row["nli_entailment"] = 0.0
                    row["nli_neutral"] = 0.0
                    row["nli_contradiction"] = 0.0
                    row["nli_label"] = ""

            detail_path = out_dir / f"summary_match_{gen_label}.jsonl"
            write_jsonl(str(detail_path), detail_rows)
            count = len(detail_rows)
            label_counts = {"entailment": 0, "neutral": 0, "contradiction": 0}
            for row in detail_rows:
                label = row.get("nli_label")
                if label in label_counts:
                    label_counts[label] += 1
            gen_metric_rows.append(
                {
                    "generator": gen_label,
                    "count": count,
                    "rouge1": float(sum(r["rouge1"] for r in detail_rows) / count) if count else 0.0,
                    "rouge2": float(sum(r["rouge2"] for r in detail_rows) / count) if count else 0.0,
                    "rougeL": float(sum(r["rougeL"] for r in detail_rows) / count) if count else 0.0,
                    "sts": float(sum(r.get("sts", 0.0) for r in detail_rows) / count) if count else 0.0,
                    "nli_entailment": float(sum(r["nli_entailment"] for r in detail_rows) / count) if count else 0.0,
                    "nli_neutral": float(sum(r["nli_neutral"] for r in detail_rows) / count) if count else 0.0,
                    "nli_contradiction": float(sum(r["nli_contradiction"] for r in detail_rows) / count) if count else 0.0,
                    "nli_entail_rate": float(label_counts["entailment"] / count) if count else 0.0,
                    "nli_neutral_rate": float(label_counts["neutral"] / count) if count else 0.0,
                    "nli_contradiction_rate": float(label_counts["contradiction"] / count) if count else 0.0,
                    "missing_summary": missing_summary,
                    "skipped": skipped,
                }
            )
            print(f"[Summary match] {gen_label}: count={count}, skipped={skipped}, missing_summary={missing_summary}")

        gen_summary_path = out_dir / "summary_match.csv"
        with open(gen_summary_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "generator",
                    "count",
                    "rouge1",
                    "rouge2",
                    "rougeL",
                    "sts",
                    "nli_entailment",
                    "nli_neutral",
                    "nli_contradiction",
                    "nli_entail_rate",
                    "nli_neutral_rate",
                    "nli_contradiction_rate",
                    "missing_summary",
                    "skipped",
                ],
            )
            writer.writeheader()
            writer.writerows(gen_metric_rows)
        print(f"Wrote summary match metrics to {gen_summary_path}")

    evaluator_specs = load_specs(args.evaluator, DEFAULT_EVALUATORS)

    summary_rows: List[Dict[str, object]] = []
    for eval_spec in evaluator_specs:
        print(f"[Evaluate] {eval_spec.label}")
        model, tokenizer = load_model(
            eval_spec.base_model,
            eval_spec.adapter_path,
            use_4bit=args.eval_use_4bit,
        )
        for gen_label, gen_path in generated_files.items():
            gen_rows = generated_rows[gen_label]
            samples = [
                GenSample(
                    generator=gen_label,
                    author=str(r.get("author", "")),
                    text=str(r.get("text", "")),
                    sample_id=int(r.get("sample_id", 0)),
                )
                for r in gen_rows
            ]
            if args.eval_mode in {"open", "both"}:
                correct, total, detail_rows = evaluate_classifier_open(
                    model,
                    tokenizer,
                    samples=samples,
                    authors=authors,
                    max_new_tokens=args.eval_max_new_tokens,
                    batch_size=args.eval_batch_size,
                )
                acc = correct / total if total else 0.0
                summary_rows.append(
                    {
                        "generator": gen_label,
                        "evaluator": f"{eval_spec.label}:open",
                        "correct": correct,
                        "total": total,
                        "accuracy": acc,
                    }
                )
                detail_path = out_dir / f"eval_open_{eval_spec.label}_on_{gen_label}.jsonl"
                write_jsonl(str(detail_path), detail_rows)
                print(f"  {gen_label} [open]: {acc:.4f} ({correct}/{total})")

            if args.eval_mode in {"closed", "both"}:
                correct, total, detail_rows = evaluate_classifier_closed(
                    model,
                    tokenizer,
                    samples=samples,
                    authors=authors,
                    max_length=args.eval_max_length,
                )
                acc = correct / total if total else 0.0
                summary_rows.append(
                    {
                        "generator": gen_label,
                        "evaluator": f"{eval_spec.label}:closed",
                        "correct": correct,
                        "total": total,
                        "accuracy": acc,
                    }
                )
                detail_path = out_dir / f"eval_closed_{eval_spec.label}_on_{gen_label}.jsonl"
                write_jsonl(str(detail_path), detail_rows)
                print(f"  {gen_label} [closed]: {acc:.4f} ({correct}/{total})")
        del model, tokenizer
        torch.cuda.empty_cache()

    summary_path = out_dir / "summary.csv"
    with open(summary_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["generator", "evaluator", "correct", "total", "accuracy"])
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Wrote summary to {summary_path}")


if __name__ == "__main__":
    main()
