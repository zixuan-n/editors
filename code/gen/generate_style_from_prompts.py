#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate style passages from prompt packs (A2G) using LoRA generators.

Supports content summaries and five-word prefix continuations. Prompt metadata
is preserved in the generated JSONL records.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from safetensors.torch import load_file
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from tqdm.auto import tqdm


CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR / "aa") not in sys.path:
    sys.path.insert(0, str(CODE_DIR / "aa"))

from inference_deflation import InferenceDeflation, DeflationStats
sys.path.insert(0, str(CODE_DIR / "gen"))
from continue_stage2_with_lora_deflation_gen import build_stage1_bases, register_deflation_hooks_smart


DEFAULT_PROMPTS = "data/Gutenberg30/prompts_A2G_general.jsonl"
DEFAULT_OUTPUT_DIR = "outputs/generation/content"
DEFAULT_GEN_BASE = "meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_GENERATORS: Sequence[Tuple[str, str, str]] = ()

DEFAULT_SYSTEM_PROMPT = (
    "You are a writing assistant that creates original literary paragraphs. "
    "Write a new paragraph in the style of the specified author. Do not copy the source text. "
    "If a Summary is provided, reflect its content while following the author's style."
)
DEFAULT_TEMPLATE = (
    "<|start_header_id|>system<|end_header_id|>\n"
    "{system_prompt}\n"
    "<|eot_id|><|start_header_id|>user<|end_header_id|>\n"
    "Author: {author}\n"
    "Summary: {content_pack}\n"
    "Task: Write a new literary paragraph.\n"
    "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
)

DEFAULT_ANSWER_PREFIX = ""
PREFIX_SYSTEM_PROMPT = (
    "You are a writing assistant that creates original literary paragraphs. "
    "Write in the style of the specified author. Do not copy source text."
)
PREFIX_TEMPLATE = (
    "<|start_header_id|>system<|end_header_id|>\n{system_prompt}\n"
    "<|eot_id|><|start_header_id|>user<|end_header_id|>\n"
    "Author: {author}\n"
    "Task: Write a new literary paragraph in this author's style. Do not use a provided summary or topic. "
    "Continue naturally from the opening words.\n"
    "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n{prefix}\n"
)


@dataclass
class AdapterSpec:
    label: str
    base_model: str
    adapter_path: Optional[str]


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


# ) -> List[Dict[str, object]]:


def write_jsonl(path: str, rows: Iterable[Dict[str, object]]) -> None:
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_jsonl_atomic(path: str, rows: Iterable[Dict[str, object]]) -> None:
    temp_path = f"{path}.tmp"
    write_jsonl(temp_path, rows)
    os.replace(temp_path, path)


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


def parse_spec(spec: str, default_base: Optional[str] = None) -> AdapterSpec:
    if "=" not in spec:
        raise ValueError(f"Invalid spec '{spec}', expected label=path or label=path|base_model.")
    label, rest = spec.split("=", 1)
    label = label.strip()
    if not label or Path(label).name != label or label in {".", ".."}:
        raise ValueError("Generator label must be a nonempty filename component.")
    adapter_path = None
    base_model = default_base or ""
    if "|" in rest:
        adapter_raw, base_raw = rest.split("|", 1)
        adapter_path = adapter_raw.strip()
        base_model = base_raw.strip()
    else:
        adapter_path = rest.strip()
    if not base_model:
        raise ValueError(f"Base model missing for spec '{spec}'.")
    adapter_path = resolve_adapter_path(adapter_path)
    return AdapterSpec(label=label, base_model=base_model, adapter_path=adapter_path)


def load_specs(
    specs: Optional[Sequence[str]],
    defaults: Sequence[Tuple[str, str, str]],
    *,
    default_base: Optional[str] = None,
) -> List[AdapterSpec]:
    if specs:
        return [parse_spec(spec, default_base=default_base) for spec in specs]
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


def build_prompt(
    row: Dict[str, object],
    *,
    system_prompt: str,
    template: str,
) -> str:
    values: Dict[str, str] = {}
    for key, value in row.items():
        values[key] = "" if value is None else str(value)
    values.setdefault("system_prompt", system_prompt or "")
    return template.format_map(_SafeDict(values))


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


def load_generation_bases(spec: AdapterSpec, args) -> Dict[str, torch.Tensor]:
    if args.no_inference_deflation:
        return {}
    if args.inference_deflation_adapter:
        return build_stage1_bases(
            Path(resolve_adapter_path(args.inference_deflation_adapter)),
            include=args.inference_deflation_include,
            basis_type=args.inference_deflation_basis_type,
            basis_solver=args.inference_deflation_basis_solver,
            basis_tol=args.inference_deflation_basis_tol,
        )
    basis_path = Path(args.deflation_bases) if args.deflation_bases else None
    if basis_path is None and spec.adapter_path:
        candidates = [Path(spec.adapter_path) / "deflation_basis.safetensors",
                      Path(spec.adapter_path).parent / "deflation_bases.pt"]
        basis_path = next((p for p in candidates if p.is_file()), None)
    if basis_path is None:
        raise ValueError("No generation deflation bases found. Supply --deflation_bases or "
                         "--inference_deflation_adapter; use --no_inference_deflation for an unprojected run.")
    bases = (load_file(str(basis_path), device="cpu") if basis_path.suffix == ".safetensors"
             else torch.load(basis_path, map_location="cpu", weights_only=True))
    if not isinstance(bases, dict) or not bases or any(
            not isinstance(U, torch.Tensor) or U.ndim != 2 for U in bases.values()):
        raise ValueError(f"Invalid deflation basis file: {basis_path}")
    return bases


@torch.no_grad()
def generate_from_prompts(
    model,
    tokenizer,
    *,
    prompts: Sequence[Dict[str, object]],
    system_prompt: str,
    template: str,
    answer_prefix: str,
    seed: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int = 0,
    batch_size: int = 1,
    prefill_field: Optional[str] = None,
    generator_label: str = "",
    initial_rows: Optional[Sequence[Dict[str, object]]] = None,
    checkpoint_path: Optional[str] = None,
    start_index: int = 0,
) -> List[Dict[str, object]]:
    rng = random.Random(seed)
    completed_batches = (start_index + batch_size - 1) // batch_size
    for _ in range(completed_batches):
        rng.randint(0, 2**31 - 1)
    rows = [dict(row) for row in initial_rows] if initial_rows else []
    progress = tqdm(total=len(prompts), desc="Generating", unit="prompt")
    for batch_start in range(0, len(prompts), batch_size):
        chunk = list(prompts[batch_start : batch_start + batch_size])
        prompt_texts = [
            build_prompt(row, system_prompt=system_prompt, template=template)
            for row in chunk
        ]
        inputs = tokenizer(
            prompt_texts,
            return_tensors="pt",
            padding=True,
            truncation=False,
            return_token_type_ids=False,
        ).to(model.device)
        gen_seed = rng.randint(0, 2**31 - 1)
        torch.manual_seed(gen_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(gen_seed)
        generation_args = {
            "do_sample": True,
            "temperature": temperature,
            "top_p": top_p,
            "max_new_tokens": max_new_tokens,
            "pad_token_id": tokenizer.eos_token_id,
        }
        if top_k > 0:
            generation_args["top_k"] = top_k
        output = model.generate(**inputs, **generation_args)
        prompt_len = inputs["input_ids"].shape[1]
        gen_tokens = output[:, prompt_len:]
        decoded = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)
        for offset, (row, text) in enumerate(zip(chunk, decoded)):
            text = text.strip()
            if prefill_field:
                prefill = str(row.get(prefill_field, "") or "").strip()
                if prefill:
                    text = f"{prefill} {text}".strip()
            if answer_prefix:
                text = f"{answer_prefix}{text}"
            out_row = dict(row)
            out_row["text"] = text
            out_row["sample_id"] = start_index + batch_start + offset
            out_row["seed"] = seed
            if generator_label:
                out_row["generator"] = generator_label
            rows.append(out_row)
            progress.update(1)
        if checkpoint_path:
            write_jsonl_atomic(checkpoint_path, rows)
    progress.close()
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate style passages from A2G prompt packs.")
    ap.add_argument("--prompts_path", default=DEFAULT_PROMPTS)
    ap.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--output_suffix", default="A2G", help="Suffix for output filename (can be empty).")
    ap.add_argument("--generator", action="append", required=True, help="label=adapter_path|base_model (repeatable)")
    ap.add_argument("--mode", choices=("content", "prefix"), default="content")
    ap.add_argument("--gen_use_4bit", action="store_true")
    ap.add_argument("--gen_max_new_tokens", type=int, default=1024)
    ap.add_argument("--gen_temperature", type=float, default=0.8)
    ap.add_argument("--gen_top_p", type=float, default=0.95)
    ap.add_argument("--gen_top_k", type=int, default=0, help="Sampling top-k; 0 leaves it unset.")
    ap.add_argument("--gen_batch_size", type=int, default=1, help="Batched prompt generation size.")
    ap.add_argument("--resume", action="store_true", help="Resume from an aligned partial output file.")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--limit", type=int, default=0, help="If >0, only process first N prompts.")
    ap.add_argument(
        "--inference_deflation_adapter",
        default=None,
        help="Optional adapter directory used to build input-space projector bases for generation-time deflation.",
    )
    ap.add_argument(
        "--inference_deflation_include",
        action="append",
        default=None,
        help="Substring filter for projected modules; repeatable. Example: q_proj.",
    )
    ap.add_argument("--inference_deflation_basis_type", choices=["lora_A", "deltaW"], default="deltaW")
    ap.add_argument("--inference_deflation_basis_solver", choices=["qr", "svd"], default="svd")
    ap.add_argument("--inference_deflation_basis_tol", type=float, default=1e-6)
    ap.add_argument(
        "--inference_deflation_stats_csv",
        default=None,
        help="Optional CSV path for cumulative removed-energy stats. With multiple generators, label is appended.",
    )
    ap.add_argument("--inference_deflation_sample_every", type=int, default=10)
    ap.add_argument("--deflation_bases", help="Saved projector bases; defaults to the generator adapter's basis file.")
    ap.add_argument("--no_inference_deflation", action="store_true", help="Explicitly run without content projection.")
    ap.add_argument("--system_prompt", default=None)
    ap.add_argument("--template", default=None)
    ap.add_argument("--answer_prefix", default=DEFAULT_ANSWER_PREFIX)
    ap.add_argument(
        "--prefill_field",
        default=None,
        help="Optional row field inserted by the template as assistant prefill; prepended to saved text.",
    )
    args = ap.parse_args()
    if args.deflation_bases and args.inference_deflation_adapter:
        ap.error("Use either --deflation_bases or --inference_deflation_adapter")
    if args.no_inference_deflation and (args.deflation_bases or args.inference_deflation_adapter):
        ap.error("--no_inference_deflation cannot be combined with projection inputs")
    if args.gen_batch_size < 1 or args.gen_max_new_tokens < 1:
        ap.error("Batch size and max new tokens must be positive")
    if args.gen_temperature <= 0:
        ap.error("--gen_temperature must be positive for sampling")
    if args.mode == "prefix":
        args.system_prompt = args.system_prompt or PREFIX_SYSTEM_PROMPT
        args.template = args.template or PREFIX_TEMPLATE
        args.prefill_field = args.prefill_field or "prefix"
    else:
        args.system_prompt = args.system_prompt or DEFAULT_SYSTEM_PROMPT
        args.template = args.template or DEFAULT_TEMPLATE

    prompts = read_jsonl(args.prompts_path)
    if args.limit and args.limit > 0:
        prompts = prompts[: args.limit]
    if not prompts:
        raise ValueError(f"No prompts loaded from {args.prompts_path}")
    if args.prefill_field and any(not row.get(args.prefill_field) for row in prompts):
        raise ValueError(f"Each prefix prompt must contain a nonempty {args.prefill_field!r} field.")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    generator_specs = load_specs(args.generator, DEFAULT_GENERATORS, default_base=DEFAULT_GEN_BASE)

    for gen_spec in generator_specs:
        print(f"[Generate] {gen_spec.label}")
        suffix = f"_{args.output_suffix}" if args.output_suffix else ""
        out_path = out_dir / f"generated_{gen_spec.label}{suffix}.jsonl"
        existing_rows: List[Dict[str, object]] = []
        prompts_to_generate = prompts
        start_index = 0
        if args.resume and out_path.exists():
            existing_rows = read_jsonl(str(out_path))
            if len(existing_rows) > len(prompts):
                raise ValueError(
                    f"Resume output has {len(existing_rows)} rows for only {len(prompts)} prompts: {out_path}"
                )
            for index, (existing, prompt) in enumerate(zip(existing_rows, prompts)):
                for key in ("author", "category", "content_pack", "prefix", "prefix_source_author"):
                    if existing.get(key) != prompt.get(key):
                        raise ValueError(
                            f"Resume alignment mismatch at row {index}, field {key!r}: {out_path}"
                        )
                if existing.get("seed") != args.seed or existing.get("generator") != gen_spec.label:
                    raise ValueError(f"Resume seed or generator mismatch at row {index}: {out_path}")
            start_index = len(existing_rows)
            prompts_to_generate = prompts[start_index:]
            print(f"[Resume] {start_index}/{len(prompts)} rows already complete in {out_path}")
            if not prompts_to_generate:
                print(f"[Skip] Complete output: {out_path}")
                continue
            if start_index % args.gen_batch_size:
                raise ValueError("Resume output ends within a batch; use its original batch size.")

        bases = load_generation_bases(gen_spec, args)
        model, tokenizer = load_model(
            gen_spec.base_model,
            gen_spec.adapter_path,
            use_4bit=args.gen_use_4bit,
        )
        inference_deflation = None
        if bases:
            stats_csv = args.inference_deflation_stats_csv
            if stats_csv and len(generator_specs) > 1:
                stats_path = Path(stats_csv)
                stats_csv = str(stats_path.with_name(f"{stats_path.stem}_{gen_spec.label}{stats_path.suffix}"))
            stats = DeflationStats(sample_every=args.inference_deflation_sample_every) if stats_csv else None
            inference_deflation = InferenceDeflation(
                handles=register_deflation_hooks_smart(model, bases, stats=stats),
                stats=stats, stats_csv=stats_csv)
        rows = generate_from_prompts(
            model,
            tokenizer,
            prompts=prompts_to_generate,
            system_prompt=args.system_prompt,
            template=args.template,
            answer_prefix=args.answer_prefix,
            seed=args.seed,
            max_new_tokens=args.gen_max_new_tokens,
            temperature=args.gen_temperature,
            top_p=args.gen_top_p,
            top_k=args.gen_top_k,
            batch_size=args.gen_batch_size,
            prefill_field=args.prefill_field,
            generator_label=gen_spec.label,
            initial_rows=existing_rows,
            checkpoint_path=str(out_path),
            start_index=start_index,
        )
        for row in rows:
            row["generator"] = gen_spec.label
        write_jsonl_atomic(str(out_path), rows)
        print(f"Wrote {len(rows)} rows to {out_path}")
        if inference_deflation is not None:
            inference_deflation.close()
        del model, tokenizer
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
