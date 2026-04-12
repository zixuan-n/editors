#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate the fine-tuned low-accuracy author model across multiple sampled subsets.

Each subset contains an equal number of passages per author (with optional title
diversity) sampled from the evaluation JSONL. For every subset, the script runs
one or more adapters (including the base model by default), aggregates
accuracies, and records detailed statistics similar to eval_strict_subsets.py.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Set

import torch
from peft import PeftModel
from tqdm.auto import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    logging as hf_logging,
)

# DEFAULT_DATASET = "splits/reuter_c50_10authors/test.jsonl"
# DEFAULT_DATASET = "splits/reuter_c50_10authors_100w_keep50/test.jsonl"
#DEFAULT_DATASET = "splits/lowacc_authors/test.jsonl"
# DEFAULT_DATASET = "splits/lowacc_authors_10authors_100w_keep50/test.jsonl"
# DEFAULT_DATASET = "splits/blogtext_10authors_300w_100p_diverse/test.jsonl"
# DEFAULT_DATASET = "splits/imdb62_top10/test.jsonl"
DEFAULT_DATASET = "splits/CCAT50/test.jsonl"
#DEFAULT_DATASET = "splits/blogs50_80/test.jsonl"
# DEFAULT_DATASET = "splits/blogs10/test.jsonl"
# DEFAULT_DATASET = "eval_out/style_gen_eval_lowacc/generated_gpt4_stylgen_synth_full_style_gen.jsonl"
# DEFAULT_DATASET = "splits/blogs50/test.jsonl"
# DEFAULT_DATASET = "splits/lowacc_mix/test.jsonl"
# --- Original ---
# DEFAULT_MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
# --- Mistral (optional) ---
# DEFAULT_MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"
# --- Gemma-2 (optional) ---
# DEFAULT_MODEL_ID = "google/gemma-2-9b-it"
# --- New (Phi-4 mini) ---
# DEFAULT_MODEL_ID = "microsoft/Phi-4-mini-instruct"
# --- Qwen2.5 7B (optional) ---
# DEFAULT_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_ADAPTERS: Sequence[tuple[str, Optional[str], Optional[str]]] = (
    # ("base", None, None),
    # --- Mix test ---
    # ("lowacc_mix_900", "outputs/lowacc_mix_900", None),
    # ("lowacc_fake", "outputs/lowacc_fake", None),
    # ("deflated_lowacc_fake", "outputs/deflated_lowacc_fake/adapter_deflated_stage2", None),
    # ("deflated_lowacc_fake_from_base", "outputs/deflated_lowacc_fake_from_base/adapter_deflated_stage2", None),
    # ("lowacc_real_3600", "outputs/lowacc_real_3600", None),
    # ("lowacc_fake_3600", "outputs/lowacc_fake_3600", None),
    # ("lowacc_real_unbalanced", "outputs/lowacc_real_unbalanced", None),
    # ("lowacc_compensated_random", "outputs/lowacc_real_compensated_random", None),
    # ("lowacc_compensated_verified", "outputs/lowacc_real_compensated_verified", None),
    # ("lowacc_fake_6600", "outputs/lowacc_fake_6600", None),
    # ("lowacc_fake_12600", "outputs/lowacc_fake_12600", None),
    # ("deflated_lowacc_from_base", "output/deflated_lowacc_from_base/adapter_deflated_stage2", None),
    #("posthoc_deflated", "outputs/llama3_lowacc_posthoc_deflated", None),
    #("deflated_lowacc_from_W0", "output/deflated_lowacc_from_W0/adapter_deflated_stage2", None),
    #("deflated_lowacc_from_W0_1600", "output/deflated_lowacc_from_W0/checkpoint-1600", None),
    #("deflated_lowacc_from_W0_svd", "output/deflated_lowacc_from_W0_svd/adapter_deflated_stage2", None),
    #("deflated_lowacc_from_W0_svd_1600", "output/deflated_lowacc_from_W0_svd/checkpoint-1600", None),

    # ("lowacc_real", "outputs/lowacc_real", None),
    # ("lowacc_mix", "outputs/lowacc_mix", None),
    # ("lowacc_mix_star", "outputs/lowacc_mix_star", None),
    # ("lowacc_mix_deflated_gemma", "outputs/lowacc_mix_deflated_gemma", None),
    # ("lowacc_mix_deflated_llama", "outputs/lowacc_mix_deflated_llama", None),
    #("lora_author_gen", "lora_author_gen/adapter", None),
    #("lora_llama3", "lora_llama3", None)
    #("lowacc_lora_latest", "outputs/llama3_lowacc_lora", None),
    #("lowacc_lora_ckpt800", "outputs/llama3_lowacc_lora/checkpoint-800", None),
    #("deflated", "outputs/stage2_style_deflated/adapter_deflated_stage2"),
    #("deflated_1", "outputs/stage2_style_deflated_1/adapter_deflated_stage2"),
    #("stage2_style_only", "outputs/stage2_style_sft", None),
    #("stage2_style_only_modified", "outputs/stage2_style_sft_modified/checkpoint-1500/style", None),
    #("stage2_style_content", "outputs/stage2_style_sft_modified/checkpoint-1500/style","outputs/stage2_style_sft_modified/checkpoint-1500/content"),
    #("stage2_style_only_modified_1", "outputs/stage2_style_sft_modified_1/checkpoint-1500/style", None),
    #("stage2_style_only", "outputs/stage2_style_sft_new", None),
    #("stage1_plus_stage2", "outputs/stage2_style_sft_new", "outputs/stage1_atomic_sft"),
    #("stage1_content_only", None, "outputs/stage1_atomic_sft"),
    # ("lowacc_lora_ckpt800", "outputs/llama3_lowacc_lora/checkpoint-800", None),
    # ("deflated_cpt_600", "outputs/stage2_style_deflated_cpt_ver/checkpoint-600", None),
    # ("deflated_cpt_800", "outputs/stage2_style_deflated_cpt_ver/checkpoint-800", None),
    # ("deflated_cpt_1000", "outputs/stage2_style_deflated_cpt_ver/checkpoint-1000", None),
    #("deflated_cpt_1200", "outputs/stage2_style_deflated_cpt_ver/checkpoint-1200", None)
    # ("deflated_from_base_1200", "outputs/deflated_from_base/checkpoint-1200", None),
    # ("deflated_from_base", "outputs/deflated_from_base/adapter_deflated_stage2", None),
    # ("deflated_from_800_ckp", "outputs/deflated_from_800/checkpoint-1200", None),
    # ("deflated_from_800", "outputs/deflated_from_800/adapter_deflated_stage2", None),
    #("lowacc_lora_add_one", "outputs/llama3_lowacc_lora_add_one", None),
    #("deflated_lora_add_one", "outputs/deflated_new_author/adapter_deflated_stage2", None)
    #("imdb_llama", "outputs/llama_imdb", None),
    # ("Qwen", "outputs/Qwen_lowacc", None),
    # ("deflated_Qwen", "outputs/deflated_Qwen/adapter_deflated_stage2", None),
    # ("deflated_Qwen_checkpoint_2200", "outputs/deflated_Qwen/checkpoint-2200", None),
    # ("deflated_Qwen_checkpoint_2600", "outputs/deflated_Qwen/checkpoint-2600", None),
    # ("deflated_Qwen_checkpoint_2800", "outputs/deflated_Qwen/checkpoint-2800", None),
    # ("deflated_Qwen_prompt", "outputs/deflated_Qwen_prompt/adapter_deflated_stage2", None),
    # ("deflated_Qwen_checkpoint_prompt_2200", "outputs/deflated_Qwen_prompt/checkpoint-2200", None),
    # ("deflated_Qwen_checkpoint_prompt_2600", "outputs/deflated_Qwen_prompt/checkpoint-2600", None),
    # ("deflated_Qwen_checkpoint_prompt_2800", "outputs/deflated_Qwen_prompt/checkpoint-2800", None)
    # ("deflated_imdb_llama", "outputs/deflated_llama_imdb/adapter_deflated_stage2", None),
    # ("deflated_imdb_llama_checkpoint_800", "outputs/deflated_llama_imdb/checkpoint-800", None),
    # ("deflated_imdb_llama_checkpoint_1200", "outputs/deflated_llama_imdb/checkpoint-1200", None),
    # ("deflated_imdb_llama_6_1600", "outputs/deflated_imdb_6/checkpoint-1600", None),
    # ("deflated_imdb_llama_6_2000", "outputs/deflated_imdb_6/checkpoint-2000", None),
    # ("deflated_imdb_llama_6_2400", "outputs/deflated_imdb_6/checkpoint-2400", None),
    # ("deflated_imdb_llama_6", "outputs/deflated_imdb_6/adapter_deflated_stage2", None)
    # ("deflated_all_layers", "outputs/deflated_all_layers/adapter_deflated_stage2", None),
    # ("deflated_all_layers_800", "outputs/deflated_all_layers/checkpoint-800", None),
    # ("deflated_all_layers_1000", "outputs/deflated_all_layers/checkpoint-1000", None),
    # ("deflated_all_layers_1200", "outputs/deflated_all_layers/checkpoint-1200", None),
    # ("deflated_two_layers_prompt", "outputs/deflated_two_layers_prompt/adapter_deflated_stage2", None),
    # ("deflated_two_layers_prompt_800", "outputs/deflated_two_layers_prompt/checkpoint-800", None),
    # ("deflated_two_layers_prompt_1000", "outputs/deflated_two_layers_prompt/checkpoint-1000", None),
    # ("deflated_two_layers_prompt_1200", "outputs/deflated_two_layers_prompt/checkpoint-1200", None)
    #("llama_CCAT", "outputs/llama_CCAT", None),
    # ("llama_CCAT_6", "outputs/llama_CCAT_6", None),
    #("deflated_llama_CCAT", "outputs/deflated_CCAT/adapter_deflated_stage2", None),
    # ("deflated_llama_CCAT_200", "outputs/deflated_CCAT/checkpoint-200", None),
    # ("deflated_llama_CCAT_256", "outputs/deflated_CCAT/checkpoint-256", None),
    # --- Original (Qwen CCAT, old deflation) ---
    # ("Qwen_CCAT", "outputs/Qwen_CCAT", None),
    # ("deflated_Qwen_CCAT", "outputs/deflated_Qwen_CCAT/adapter_deflated_stage2", None),
    # ("deflated_Qwen_CCAT_200", "outputs/deflated_Qwen_CCAT/checkpoint-200", None)

    # --- Original (Qwen CCAT, deltaW deflation) ---
    # ("Qwen_CCAT", "outputs/Qwen_CCAT", None),
    # ("deflated_Qwen_CCAT_deltaW", "outputs/deflated_Qwen_CCAT_deltaW/adapter_deflated_stage2", None),
    # ("deflated_Qwen_CCAT_deltaW_200", "outputs/deflated_Qwen_CCAT_deltaW/checkpoint-200", None),

    # --- Mistral CCAT (optional, deltaW deflation) ---
    # ("Mistral_CCAT", "outputs/Mistral_CCAT", None),
    # ("deflated_Mistral_CCAT_deltaW", "outputs/deflated_Mistral_CCAT_deltaW/adapter_deflated_stage2", None),
    # ("deflated_Mistral_CCAT_deltaW_200", "outputs/deflated_Mistral_CCAT_deltaW/checkpoint-200", None),

    # --- Gemma-2 CCAT (optional, deltaW deflation) ---
    # ("Gemma2_CCAT", "outputs/Gemma2_CCAT", None),
    # ("deflated_Gemma2_CCAT_deltaW", "outputs/deflated_Gemma2_CCAT_deltaW/adapter_deflated_stage2", None),
    # ("deflated_Gemma2_CCAT_deltaW_200", "outputs/deflated_Gemma2_CCAT_deltaW/checkpoint-200", None),

    # --- New (Phi-4 mini CCAT, deltaW deflation) ---
    # ("Phi4mini_CCAT", "outputs/Phi4mini_CCAT", None),
    # ("deflated_Phi4mini_CCAT_deltaW", "outputs/deflated_Phi4mini_CCAT_deltaW/adapter_deflated_stage2", None),
    # ("deflated_Phi4mini_CCAT_deltaW_200", "outputs/deflated_Phi4mini_CCAT_deltaW/checkpoint-200", None),

    # --- Llama-3.1 8B CCAT 100w keep50 (deltaW deflation) ---
    # ("llama3_CCAT_100w_keep50", "outputs/llama3_CCAT_100w_keep50", None),
    # ("deflated_Llama3_8B_reuter_100w_keep50_deltaW", "outputs/deflated_Llama3_8B_reuter_100w_keep50_deltaW/adapter_deflated_stage2", None),
    # ("deflated_Llama3_8B_reuter_100w_keep50_deltaW_200", "outputs/deflated_Llama3_8B_reuter_100w_keep50_deltaW/checkpoint-200", None),

    # --- Llama-3.1 8B lowacc 10authors 100w keep50 (deltaW deflation) ---
    # ("llama3_lowacc_100w_keep50_12_epoch", "outputs/llama3_lowacc_100w_keep50_12_epoch", None),
    # ("llama3_lowacc_100w_keep50", "outputs/llama3_lowacc_100w_keep50", None),
    # ("deflated_Llama3_8B_lowacc_10authors_100w_keep50_deltaW", "outputs/deflated_Llama3_8B_lowacc_10authors_100w_keep50_deltaW/adapter_deflated_stage2", None),
    # ("deflated_Llama3_8B_lowacc_10authors_100w_keep50_deltaW_200", "outputs/deflated_Llama3_8B_lowacc_10authors_100w_keep50_deltaW/checkpoint-200", None),
    
    # --- Llama-3.1 8B IMDB top10 (deltaW deflation) ---
    # ("deflated_Llama3_8B_imdb_top10_deltaW", "outputs/deflated_Llama3_8B_imdb_top10_deltaW/adapter_deflated_stage2", None),
    # ("deflated_Llama3_8B_imdb_top10_deltaW_1200", "outputs/deflated_Llama3_8B_imdb_top10_deltaW/checkpoint-1200", None),
    # ("llama3_imdb_checkpoint_417", "outputs/llama3_imdb/checkpoint-417", None),
    # ("llama_imdb_no_prompt", "outputs/llama_imdb_no_prompt", None),
    # ("llama_imdb_no_prompt_new", "outputs/llama_imdb_no_prompt_new", None),
    # ("llama_imdb_no_prompt_new_800", "outputs/llama_imdb_no_prompt_new/checkpoint-800", None),
    # ("llama_imdb_no_prompt_new_1200", "outputs/llama_imdb_no_prompt_new/checkpoint-1200", None),
    # ("llama_imdb", "outputs/llama_imdb", None),
    # ("llama3_imdb", "outputs/llama3_imdb", None),

    # --- Llama-3.1 8B CCAT50 ---
    #("llama3_CCAT50", "outputs/llama3_CCAT50", None),
    ("llama3_CCAT50_8", "outputs/llama3_CCAT50_8", None),
    # ("llama3_CCAT50_8_1000", "outputs/llama3_CCAT50_8/checkpoint-1000", None),
    ("deflated_llama3_CCAT50_deltaW", "outputs/deflated_llama3_CCAT50_deltaW/adapter_deflated_stage2", None),
    #("deflated_llama3_CCAT50_deltaW_600", "outputs/deflated_llama3_CCAT50_deltaW/checkpoint-600", None),
    # ("deflated_llama3_CCAT50_deltaW_1000", "outputs/deflated_llama3_CCAT50_deltaW/checkpoint-1000", None),
    ("deflated_llama3_CCAT50_deltaW_warmup", "outputs/deflated_llama3_CCAT50_deltaW_warmup/adapter_deflated_stage2", None),
    ("llama3_CCAT50_posthoc_deflated", "outputs/llama3_CCAT50_posthoc_deflated", None),
    # ("deflated_llama3_CCAT50_deltaW_warmup_200", "outputs/deflated_llama3_CCAT50_deltaW_warmup/checkpoint-200", None),

    # --- Llama-3.1 8B blogs50 ---
    # ("llama3_blogs50_full", "outputs/llama3_blogs50_full", None),
    # ("llama3_blogs50_1800", "outputs/llama3_blogs50_full/checkpoint-1800", None),
    # ("llama3_blogs50_2000", "outputs/llama3_blogs50_full/checkpoint-2000", None),
    # ("deflated_llama3_blogs50_deltaW_2", "outputs/deflated_llama3_blogs50_deltaW_2/adapter_deflated_stage2", None),
    # ("deflated_llama3_blogs50_deltaW_1800", "outputs/deflated_llama3_blogs50_deltaW_2/checkpoint-1800", None),
    # ("deflated_llama3_blogs50_deltaW_2000", "outputs/deflated_llama3_blogs50_deltaW_2/checkpoint-2000", None),
    # ("deflated_llama3_blogs50_deltaW_2400", "outputs/deflated_llama3_blogs50_deltaW_2/checkpoint-2400", None),
    

    # --- Llama-3.1 8B blogs50_80 ---
    #("llama3_blogs50_80_full", "outputs/llama3_blogs50_80_full", None),
    # ("llama3_blogs50_80_3000", "outputs/llama3_blogs50_80_full/checkpoint-3000", None),
    #("deflated_llama3_blogs50_80_deltaW", "outputs/deflated_llama3_blogs50_80_deltaW/adapter_deflated_stage2", None),
    #("llama3_blogs50_posthoc_deflated", "outputs/llama3_blogs50_posthoc_deflated", None),
    # ("deflated_llama3_blogs50_80_deltaW_3000", "outputs/deflated_llama3_blogs50_80_deltaW/checkpoint-3000", None),
    # ("deflated_llama3_blogs50_80_deltaW_2_checkpoint", "outputs/deflated_llama3_blogs50_80_deltaW_2/checkpoint-2600", None),
    # ("deflated_llama3_blogs50_80_deltaW_2", "outputs/deflated_llama3_blogs50_80_deltaW_2/adapter_deflated_stage2", None),

    


    # --- Llama-3.1 8B blogs50_80 ---
    # ("llama3_blogs10", "outputs/llama3_blogs10", None),
    # ("deflated_llama3_blogs10_deltaW", "outputs/deflated_llama3_blogs10_deltaW/adapter_deflated_stage2", None),
    # ("deflated_llama3_blogs10_deltaW_2000", "outputs/deflated_llama3_blogs10_deltaW/checkpoint-2000", None),
    # ("deflated_llama3_blogs10_deltaW_1600", "outputs/deflated_llama3_blogs10_deltaW/checkpoint-1600", None),



    # --- Qwen2.5 7B CCAT (optional, deltaW deflation) ---
    # ("Qwen25_7B_CCAT", "outputs/Qwen25_7B_CCAT", None),
    # ("deflated_Qwen25_7B_CCAT_deltaW", "outputs/deflated_Qwen25_7B_CCAT_deltaW/adapter_deflated_stage2", None),
    # ("deflated_Qwen25_7B_CCAT_deltaW_200", "outputs/deflated_Qwen25_7B_CCAT_deltaW/checkpoint-200", None),
)
# --- Original ---
# DEFAULT_OUTPUT_CSV = "results/deflated_CCAT_Qwen.csv"
# DEFAULT_OUTPUT_CSV = "results/deflated_CCAT_Qwen_deltaW.csv"
# --- Mistral (optional) ---
# DEFAULT_OUTPUT_CSV = "results/deflated_CCAT_Mistral_deltaW.csv"
# --- Gemma-2 (optional) ---
# DEFAULT_OUTPUT_CSV = "results/deflated_CCAT_Gemma2_deltaW.csv"
# --- New (Phi-4 mini) ---
# DEFAULT_OUTPUT_CSV = "results/deflated_CCAT_Phi4mini_deltaW.csv"
# --- Llama-3.1 8B reuter 100w keep50 ---
# DEFAULT_OUTPUT_CSV = "results/deflated_reuter_100w_keep50_llama3_8b_deltaW.csv"
# --- Llama-3.1 8B lowacc 10authors 100w keep50 ---
#DEFAULT_OUTPUT_CSV = "results/deflated_lowacc_10authors_100w_keep50_llama3_8b_deltaW.csv"
# --- Llama-3.1 8B IMDB top10 ---
# DEFAULT_OUTPUT_CSV = "results/imdb_top10_llama3_8b_new.csv"
# --- Llama-3.1 8B CCAT50 ---
# DEFAULT_OUTPUT_CSV = "results/llama3_CCAT50_eval_full_test_base.csv"
# --- Llama-3.1 8B blogs50 ---
# DEFAULT_OUTPUT_CSV = "results/llama3_blogs50_80_eval_full_2.csv"
# DEFAULT_OUTPUT_CSV = "results/llama3_blogs10_eval.csv"
# DEFAULT_OUTPUT_CSV = "results/authors10_stylegen.csv"
#DEFAULT_OUTPUT_CSV = "results/real_mix.csv"
#DEFAULT_OUTPUT_CSV = "results/real_mix_augmentation_12600.csv"
#DEFAULT_OUTPUT_CSV = "results/real_mix_augmentation_6600.csv"
#DEFAULT_OUTPUT_CSV = "results/deflated_from_W0.csv"
#DEFAULT_OUTPUT_CSV = "results/posthoc_deflated.csv"
DEFAULT_OUTPUT_CSV = "results/posthoc_deflated_CCAT50.csv"
#DEFAULT_OUTPUT_CSV = "results/posthoc_deflated_blogs50.csv"#
#DEFAULT_AUTHOR_OUTPUT_CSV = "result/real_mix_augmentation_per_author_12600.csv"
#DEFAULT_AUTHOR_OUTPUT_CSV = "result/deflated_from_W0_per_author.csv"
#DEFAULT_AUTHOR_OUTPUT_CSV = "result/posthoc_deflated_per_author.csv"
DEFAULT_AUTHOR_OUTPUT_CSV = "result/posthoc_deflated_per_author_CCAT50.csv"
# --- Qwen2.5 7B (optional) ---
# DEFAULT_OUTPUT_CSV = "results/deflated_CCAT_Qwen25_7B_deltaW.csv"
MAX_CONTEXT = 1536
MAX_NEW_TOKENS = 20


def parse_adapter_spec(spec: str) -> tuple[str, Optional[str], Optional[str]]:
    if "=" not in spec:
        raise ValueError(
            f"Invalid adapter specification '{spec}'. Use the form label=path "
            "(or label=none for the base model)."
        )
    label, path = spec.split("=", 1)
    label = label.strip()
    path = path.strip()
    if not label:
        raise ValueError(f"Adapter label missing in specification '{spec}'.")
    stage2_path: Optional[str]
    stage1_path: Optional[str] = None
    if "|" in path:
        stage2_raw, stage1_raw = path.split("|", 1)
        stage2_path = stage2_raw.strip()
        stage1_path = stage1_raw.strip()
    else:
        stage2_path = path
    def normalize_path(p: Optional[str]) -> Optional[str]:
        if p is None:
            return None
        if not p or p.lower() in {"none", "null", "base"}:
            return None
        return p
    stage2_path = normalize_path(stage2_path)
    stage1_path = normalize_path(stage1_path)
    return label, stage2_path, stage1_path


def resolve_adapters(adapter_specs: Optional[Sequence[str]]) -> Sequence[tuple[str, Optional[str], Optional[str]]]:
    if not adapter_specs:
        return DEFAULT_ADAPTERS
    adapters: List[tuple[str, Optional[str], Optional[str]]] = []
    for spec in adapter_specs:
        adapters.append(parse_adapter_spec(spec))
    return adapters


def load_jsonl(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: str, rows: Sequence[Dict[str, object]]) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize(text: str) -> str:
    return " ".join(text.strip().split()).lower()


def load_author_list(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def normalize_authors(authors: Iterable[str]) -> Dict[str, str]:
    return {normalize(author): author for author in authors}


def sample_subsets(
    rows: Iterable[Dict[str, str]],
    *,
    authors: Sequence[str],
    n_subsets: int,
    per_author: int,
    authors_per_subset: Optional[int],
    require_distinct_titles: bool,
    seed: int,
    allow_resample: bool,
) -> Tuple[List[List[Dict[str, str]]], int]:
    rows = list(rows)
    authors = list(authors)
    authors.sort()
    rng = random.Random(seed)

    by_author_titles: Dict[str, Dict[str, List[Dict[str, str]]]] = {
        author: {} for author in authors
    }
    for row in rows:
        author = row["author"]
        title = row.get("title", "Unknown")
        by_author_titles[author].setdefault(title, []).append(row)

    if allow_resample:
        mutable_titles = by_author_titles  # reuse references; we won't mutate
    else:
        mutable_titles = {
            author: {title: list(items) for title, items in title_map.items()}
            for author, title_map in by_author_titles.items()
        }

    flat_rows = {
        author: [
            item for title_rows in by_author_titles[author].values() for item in title_rows
        ]
        for author in authors
    }

    subsets: List[List[Dict[str, str]]] = []
    subset_sizes: List[int] = []
    for subset_idx in range(n_subsets):
        if authors_per_subset is not None:
            if authors_per_subset > len(authors):
                raise ValueError(
                    f"Cannot sample {authors_per_subset} authors from only {len(authors)} available."
                )
            current_authors = rng.sample(authors, authors_per_subset)
        else:
            current_authors = authors

        subset_rows: List[Dict[str, str]] = []
        for author in current_authors:
            title_map = mutable_titles[author]
            full_title_map = by_author_titles[author]
            pool = flat_rows[author]

            if not pool:
                raise ValueError(f"Author '{author}' has no available samples.")

            selected: List[Dict[str, str]] = []
            if require_distinct_titles:
                available_titles = [title for title, items in title_map.items() if items]
                if not available_titles:
                    available_titles = [title for title, items in full_title_map.items() if items]

                if allow_resample:
                    if len(available_titles) >= per_author:
                        chosen_titles = rng.sample(available_titles, per_author)
                    else:
                        chosen_titles = rng.choices(available_titles, k=per_author)
                    for title in chosen_titles:
                        rows_for_title = full_title_map.get(title, [])
                        selected.append(rng.choice(rows_for_title))
                else:
                    rng.shuffle(available_titles)
                    for title in available_titles:
                        if title_map[title]:
                            selected.append(title_map[title].pop())
                        if len(selected) == per_author:
                            break
                    if len(selected) < per_author:
                        remaining = [
                            item for items in title_map.values() for item in items
                        ]
                        if len(remaining) < (per_author - len(selected)):
                            raise ValueError(
                                f"Author '{author}' lacks enough samples "
                                f"to fill subset {subset_idx}."
                            )
                        extras = rng.sample(remaining, per_author - len(selected))
                        selected.extend(extras)
                        for extra in extras:
                            title_map[extra.get("title", "Unknown")].remove(extra)
            else:
                if allow_resample:
                    if len(pool) >= per_author:
                        selected = rng.sample(pool, per_author)
                    else:
                        selected = rng.choices(pool, k=per_author)
                else:
                    available = [
                        item
                        for items in title_map.values()
                        for item in items
                    ]
                    if len(available) < per_author:
                        raise ValueError(
                            f"Author '{author}' lacks enough samples to fill subset {subset_idx}."
                        )
                    selected = rng.sample(available, per_author)
                    for chosen in selected:
                        title_map[chosen.get("title", "Unknown")].remove(chosen)

            subset_rows.extend(selected[:per_author])
        rng.shuffle(subset_rows)
        subset_sizes.append(len(subset_rows))
        subsets.append(subset_rows)
    if not all(size == subset_sizes[0] for size in subset_sizes):
        raise AssertionError("Inconsistent subset sizes across sampled subsets.")
    return subsets, subset_sizes[0]


def load_model_and_tokenizer(
    model_id: str,
    *,
    adapter_path: Optional[str],
    stage1_adapter_path: Optional[str],
    device_index: int,
    use_4bit: bool,
) -> tuple[torch.nn.Module, AutoTokenizer]:
    hf_logging.set_verbosity_error()
    quant = (
        BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        if use_4bit
        else None
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    base = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map={"": device_index},
        quantization_config=quant,
        torch_dtype=None if use_4bit else torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    pad_id = tokenizer.pad_token_id
    base.config.pad_token_id = pad_id
    base.generation_config.pad_token_id = pad_id

    model: torch.nn.Module = base
    if stage1_adapter_path:
        model = PeftModel.from_pretrained(
            model,
            stage1_adapter_path,
            adapter_name="content",
            is_trainable=False,
        )
        for param in model.parameters():
            param.requires_grad_(False)
        if adapter_path:
            model = model.merge_and_unload()

    if adapter_path:
        model = PeftModel.from_pretrained(
            model,
            adapter_path,
            adapter_name="style" if stage1_adapter_path else "default",
            is_trainable=False,
        )

    model.config.pad_token_id = pad_id
    model.generation_config.pad_token_id = pad_id
    model.eval()
    return model, tokenizer


def build_prompt(tokenizer: AutoTokenizer, authors: Sequence[str], passage: str, *, include_candidates: bool) -> str:
    # Previous implementation (kept for reference):
    # def build_prompt(model_id: str, authors: Sequence[str], passage: str, *, include_candidates: bool) -> str:
    #     author_list = ", ".join(authors)
    #     candidate_line = f"Candidate authors: {author_list}\\n" if include_candidates else ""
    #     system_prompt = (
    #         "You are an expert literary analyst. Identify the author of the provided passage based on their writing style instead of the content. "
    #         "Answer with only the exact author name.\\n"
    #     )
    #     mid = (model_id or "").lower()
    #     if "mistral" in mid:
    #         return (
    #             "<s>[INST] "
    #             f"{system_prompt}"
    #             f"{candidate_line}"
    #             f"Passage:\\n{passage.strip()}\\n"
    #             "Author: [/INST]"
    #         )
    #     return (
    #         "<|start_header_id|>system<|end_header_id|>\\n"
    #         f"{system_prompt}"
    #         f"{candidate_line}"
    #         "<|eot_id|><|start_header_id|>user<|end_header_id|>\\n"
    #         f"Passage:\\n{passage.strip()}\\n"
    #         "Author:"
    #         "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\\n"
    #     )

    author_list = ", ".join(authors)
    candidate_line = f"Candidate authors: {author_list}\n" if include_candidates else ""
    # system_prompt = (
    #     "You are an expert literary analyst. Identify the author of the provided passage based on their writing style instead of the content. "
    #     "Answer with only the exact author name."
    # )
    system_prompt = (
        "You are an expert literary analyst. Identify the author of the provided passage based on their writing style instead of the content. "
        "Answer with only the exact user ID."
    )
    user_content = f"{candidate_line}Passage:\n{passage.strip()}\nAuthor:"

    if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
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
            # Some chat templates (e.g. certain Gemma/Phi variants) do not support a "system" role.
            msg = str(exc).lower()
            if "system role not supported" in msg or "system role" in msg:
                messages = [{"role": "user", "content": f"{system_prompt}\n\n{user_content}"}]
                return tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            raise

    return (
        "<|start_header_id|>system<|end_header_id|>\n"
        f"{system_prompt}\n"
        "<|eot_id|><|start_header_id|>user<|end_header_id|>\n"
        f"{user_content}"
        "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
    )


def select_author(completion: str, authors: Sequence[str]) -> str:
    completion_norm = normalize(completion)
    for author in authors:
        author_norm = normalize(author)
        if completion_norm.startswith(author_norm):
            return author
    for author in authors:
        if normalize(author) in completion_norm:
            return author
    # If nothing matches, return the raw completion so it is counted incorrect.
    return completion.strip()


@torch.no_grad()
def predict_author(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    text: str,
    author_list: Sequence[str],
    *,
    include_candidates: bool,
) -> str:
    device = next(model.parameters()).device
    prompt = build_prompt(tokenizer, author_list, text, include_candidates=include_candidates)
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_CONTEXT,
        padding=False,
    ).to(device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        temperature=0.0,
    )
    completion = tokenizer.decode(
        outputs[0][inputs["input_ids"].size(1) :], skip_special_tokens=True
    )
    return select_author(completion, author_list)


@torch.no_grad()
def evaluate_subset(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    rows: Sequence[Dict[str, str]],
    author_list: Sequence[str],
    *,
    desc: str,
    collect_errors: bool,
    include_candidates: bool,
) -> Tuple[float, List[Dict[str, object]], Optional[Dict[str, Dict[str, int]]]]:
    hits = 0
    errors: List[Dict[str, object]] = []
    per_author: Dict[str, Dict[str, int]] = {}
    bar = tqdm(rows, desc=desc, leave=False)
    seen = 0
    for row in bar:
        seen += 1
        pred = predict_author(
            model,
            tokenizer,
            row["text"],
            author_list,
            include_candidates=include_candidates,
        )
        correct = normalize(pred) == normalize(row["author"])
        hits += int(correct)
        author = row["author"]
        if author not in per_author:
            per_author[author] = {"correct": 0, "total": 0}
        per_author[author]["total"] += 1
        per_author[author]["correct"] += int(correct)
        if collect_errors and not correct:
            errors.append(
                {
                    "author": row["author"],
                    "predicted_author": pred,
                    "text": row["text"],
                    "title": row.get("title"),
                }
        )
        bar.set_postfix(acc=min(hits / max(1, seen), 1.0))
    return hits / len(rows), errors, per_author


def write_results_csv(rows: Sequence[Dict[str, object]], path: str) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    fieldnames = ["subset", "model", "prompt_mode", "accuracy", "n_rows"]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_author_accuracy_csv(rows: Sequence[Dict[str, object]], path: str) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    fieldnames = ["subset", "model", "prompt_mode", "author", "accuracy", "correct", "total"]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate multiple sampled subsets with fine-tuned adapters."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=DEFAULT_DATASET,
        help=f"Path to JSONL dataset (default: {DEFAULT_DATASET}).",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default=DEFAULT_MODEL_ID,
        help=f"Base HF model id (default: {DEFAULT_MODEL_ID}).",
    )
    parser.add_argument(
        "--subset-size",
        type=int,
        default=300,
        help="Total rows per subset (must be divisible by #authors * per-author).",
    )
    parser.add_argument(
        "--per-author",
        type=int,
        default=None,
        help="Rows per author per subset. If omitted, derived from subset size.",
    )
    parser.add_argument(
        "--n-subsets",
        type=int,
        default=10,
        help="Number of subsets to evaluate (default: 10).",
    )
    parser.add_argument(
        "--eval-full",
        action="store_true",
        help="Evaluate the full dataset as a single subset (ignores subset sampling options).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling subsets (default: 42).",
    )
    parser.add_argument(
        "--require-distinct-titles",
        action="store_true",
        help="Ensure each sampled subset draws passages from different titles per author.",
    )
    parser.add_argument(
        "--prompt-mode",
        type=str,
        default="without_candidates",
        choices=("with_candidates", "without_candidates", "both"),
        help="Choose whether to include candidate author list in the prompt.",
    )
    parser.add_argument(
        "--single-author",
        type=str,
        default=None,
        help="If set, evaluate only this author with random samples from the dataset.",
    )
    parser.add_argument(
        "--single-author-samples",
        type=int,
        default=50,
        help="Number of samples to draw for --single-author (default: 50).",
    )
    parser.add_argument(
        "--authors",
        type=str,
        default=None,
        help="Comma-separated list of authors to evaluate (case-insensitive).",
    )
    parser.add_argument(
        "--authors-file",
        type=str,
        default=None,
        help="Path to a text file (one author per line) to include in evaluation.",
    )
    parser.add_argument(
        "--exclude-authors-file",
        type=str,
        default=None,
        help="Path to a text file (one author per line) to exclude from evaluation.",
    )
    parser.add_argument(
        "--train-path",
        type=str,
        default=None,
        help="Optional train JSONL to derive the set of authors to evaluate (filters dataset to these authors).",
    )
    parser.add_argument(
        "--random-authors",
        type=int,
        default=None,
        help="Randomly sample this many authors after filtering.",
    )
    parser.add_argument(
        "--author-seed",
        type=int,
        default=43,
        help="Random seed used when sampling authors (default: 43).",
    )
    parser.add_argument(
        "--authors-per-subset",
        type=int,
        default=None,
        help="Sample this many authors independently for each subset (requires resample).",
    )
    parser.add_argument(
        "--resample",
        action="store_true",
        help="Allow reusing passages across subsets (sampling with replacement).",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=DEFAULT_OUTPUT_CSV,
        help=f"Where to write aggregate results (default: {DEFAULT_OUTPUT_CSV}).",
    )
    parser.add_argument(
        "--per-author-csv",
        type=str,
        default=DEFAULT_AUTHOR_OUTPUT_CSV,
        help="Optional CSV to write per-author accuracy for each subset/model.",
    )
    parser.add_argument(
        "--misclassified-jsonl",
        type=str,
        default=None,
        help="Optional base path to store misclassified samples per model.",
    )
    parser.add_argument(
        "--no-4bit",
        action="store_true",
        help="Disable 4-bit loading.",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="CUDA device index (default: 0).",
    )
    parser.add_argument(
        "--adapter",
        action="append",
        default=None,
        help=(
            "Specify adapters as label=stage2_path|stage1_path. "
            "Use 'none' when a path is absent and repeat flag for multiple adapters."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    adapters = resolve_adapters(args.adapter)

    rows = load_jsonl(args.dataset)
    include_authors: List[str] = []
    if args.train_path:
        train_rows = load_jsonl(args.train_path)
        include_authors.extend(sorted({row["author"] for row in train_rows if "author" in row}))
    if args.authors:
        include_authors.extend(
            [a.strip() for a in args.authors.split(",") if a.strip()]
        )
    if args.authors_file:
        include_authors.extend(load_author_list(args.authors_file))
    include_norm = {normalize(a) for a in include_authors}

    exclude_norm: Set[str] = set()
    if args.exclude_authors_file:
        exclude_norm = {normalize(a) for a in load_author_list(args.exclude_authors_file)}

    if include_norm:
        rows = [row for row in rows if normalize(row["author"]) in include_norm]
    if exclude_norm:
        rows = [row for row in rows if normalize(row["author"]) not in exclude_norm]

    # Single-author shortcut
    if args.single_author:
        target_norm = normalize(args.single_author)
        rows = [row for row in rows if normalize(row["author"]) == target_norm]
        if not rows:
            raise ValueError(f"No rows found for single author '{args.single_author}'.")
        authors = [args.single_author]
        if args.eval_full:
            subset_rows = list(rows)
        else:
            rng_single = random.Random(args.seed)
            subset_rows = rng_single.sample(rows, min(len(rows), args.single_author_samples))
        prompt_modes = (
            ["with_candidates", "without_candidates"]
            if args.prompt_mode == "both"
            else [args.prompt_mode]
        )
        for model_label, adapter_path, stage1_adapter_path in adapters:
            for prompt_mode in prompt_modes:
                prompt_suffix = (
                    "_no_candidates" if prompt_mode == "without_candidates" else ""
                )
                label_with_mode = f"{model_label}{prompt_suffix}"
                stage1_desc = stage1_adapter_path or "none"
                print(
                    f"\nEvaluating adapter '{model_label}' "
                    f"(adapter={adapter_path or 'none'}, stage1={stage1_desc}) | prompt={prompt_mode}"
                )
                model, tokenizer = load_model_and_tokenizer(
                    args.model_id,
                    adapter_path=adapter_path,
                    stage1_adapter_path=stage1_adapter_path,
                    device_index=args.device,
                    use_4bit=not args.no_4bit,
                )
                include_candidates = prompt_mode != "without_candidates"
                acc, errors, per_author_stats = evaluate_subset(
                    model,
                    tokenizer,
                    subset_rows,
                    authors,
                    desc=label_with_mode,
                    collect_errors=False,
                    include_candidates=include_candidates,
                )
                print(f"{label_with_mode} accuracy: {acc:.4f}")
        return
    authors = sorted({row["author"] for row in rows})
    if not authors:
        raise ValueError("No authors left after applying filters.")

    if args.random_authors is not None:
        if args.random_authors > len(authors):
            raise ValueError(
                f"Requested {args.random_authors} authors but only {len(authors)} available."
            )
        rng = random.Random(args.author_seed)
        sampled_authors = sorted(rng.sample(authors, args.random_authors))
        sampled_norm = {normalize(a) for a in sampled_authors}
        rows = [
            row for row in rows if normalize(row["author"]) in sampled_norm
        ]
        authors = sampled_authors
        print(
            "Evaluating sampled authors:\n  "
            + ", ".join(sampled_authors)
        )

    if args.eval_full:
        subsets = [rows]
        subset_size = len(rows)
        print(f"Evaluating full dataset with {subset_size} rows.")
    else:
        if args.authors_per_subset is not None:
            if args.authors_per_subset <= 0:
                raise ValueError("--authors-per-subset must be positive.")
            if args.authors_per_subset > len(authors):
                raise ValueError(
                    f"--authors-per-subset={args.authors_per_subset} exceeds available authors ({len(authors)})."
                )
            if args.random_authors is not None and args.authors_per_subset > args.random_authors:
                raise ValueError(
                    "--authors-per-subset cannot exceed --random-authors."
                )
            if not args.resample:
                raise ValueError("--authors-per-subset requires --resample to be enabled.")
            subset_author_count = args.authors_per_subset
        else:
            subset_author_count = len(authors)

        if args.per_author is None:
            if subset_author_count == 0:
                raise ValueError("No authors available for subset sampling.")
            if args.subset_size % subset_author_count != 0:
                raise ValueError(
                    f"subset_size={args.subset_size} not divisible by authors per subset={subset_author_count}"
                )
            per_author = args.subset_size // subset_author_count
        else:
            per_author = args.per_author
            expected_size = per_author * subset_author_count
            if args.subset_size != expected_size:
                print(
                    f"Warning: overriding subset_size {args.subset_size} with "
                    f"{expected_size} based on per_author and authors-per-subset."
                )

        subset_size_expected = per_author * subset_author_count

        eligible_authors: List[str] = []
        for author in authors:
            author_total = sum(
                1 for row in rows if normalize(row["author"]) == normalize(author)
            )
            required = per_author if args.resample else per_author * args.n_subsets
            if author_total < required:
                print(
                    f"Skipping author '{author}' (only {author_total} samples; "
                    f"need {required})."
                )
            else:
                eligible_authors.append(author)
        authors = eligible_authors
        if not authors:
            raise ValueError("No authors remain after filtering for sample counts.")
        rows = [row for row in rows if row["author"] in authors]
        authors.sort()

        subsets, subset_size = sample_subsets(
            rows,
            authors=authors,
            n_subsets=args.n_subsets,
            per_author=per_author,
            authors_per_subset=args.authors_per_subset,
            require_distinct_titles=args.require_distinct_titles,
            seed=args.seed,
            allow_resample=args.resample,
        )
        print(f"Constructed {len(subsets)} subsets; each has {subset_size} rows.")

    results: List[Dict[str, object]] = []
    misclassified_records: Dict[str, List[Dict[str, object]]] = {}
    per_author_rows: List[Dict[str, object]] = []

    prompt_modes = (
        ["with_candidates", "without_candidates"]
        if args.prompt_mode == "both"
        else [args.prompt_mode]
    )

    total_subsets = len(subsets)
    for model_label, adapter_path, stage1_adapter_path in adapters:
        for prompt_mode in prompt_modes:
            prompt_suffix = (
                "_no_candidates" if prompt_mode == "without_candidates" else ""
            )
            label_with_mode = f"{model_label}{prompt_suffix}"
            stage1_desc = stage1_adapter_path or "none"
            print(
                f"\nEvaluating adapter '{model_label}' "
                f"(adapter={adapter_path or 'none'}, stage1={stage1_desc}) | prompt={prompt_mode}"
            )
            model, tokenizer = load_model_and_tokenizer(
                args.model_id,
                adapter_path=adapter_path,
                stage1_adapter_path=stage1_adapter_path,
                device_index=args.device,
                use_4bit=not args.no_4bit,
            )
            include_candidates = prompt_mode != "without_candidates"
            for idx, subset in enumerate(subsets):
                acc, errors, per_author_stats = evaluate_subset(
                    model,
                    tokenizer,
                    subset,
                    authors,
                    desc=f"{label_with_mode} subset {idx+1}/{total_subsets}",
                    collect_errors=args.misclassified_jsonl is not None,
                    include_candidates=include_candidates,
                )
                if errors:
                    misclassified_records.setdefault(label_with_mode, []).extend(errors)
                if args.per_author_csv and per_author_stats:
                    for author, stats in per_author_stats.items():
                        total = stats["total"]
                        correct = stats["correct"]
                        per_author_rows.append(
                            {
                                "subset": idx,
                                "model": label_with_mode,
                                "prompt_mode": prompt_mode,
                                "author": author,
                                "accuracy": round(correct / total, 4) if total else 0.0,
                                "correct": correct,
                                "total": total,
                            }
                        )
                results.append(
                    {
                        "subset": idx,
                        "model": label_with_mode,
                        "prompt_mode": prompt_mode,
                        "accuracy": round(acc, 4),
                        "n_rows": len(subset),
                    }
                )
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    write_results_csv(results, args.output_csv)
    print(f"\nSaved accuracy table to {args.output_csv}")
    if args.per_author_csv and per_author_rows:
        write_author_accuracy_csv(per_author_rows, args.per_author_csv)
        print(f"Saved per-author accuracy table to {args.per_author_csv}")

    if args.misclassified_jsonl:
        base_path = args.misclassified_jsonl
        root, ext = os.path.splitext(base_path)
        if not ext:
            ext = ".jsonl"
        for label, records in misclassified_records.items():
            if not records:
                continue
            path = base_path if label == "base" else f"{root}_{label}{ext}"
            write_jsonl(path, records)
            print(f"Saved {len(records)} misclassified rows for {label} -> {path}")


if __name__ == "__main__":
    main()
