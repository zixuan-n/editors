#!/usr/bin/env python3
r"""Evaluate generation memorization, style, content preservation, and fluency.

Example:
    python code/eval_scripts/evaluate_generation.py --input generated.jsonl \
        --output_dir results/generation --metrics memorization,style

Select content or fluency (or all) to use GPT-4o mini with OPENAI_API_KEY.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from tqdm.auto import tqdm

DEFAULT_EVALUATOR = (
    "Llama=outputs/aa/Llama3_Gutenberg30/s2/adapter_deflated_stage2"
    "|meta-llama/Llama-3.1-8B-Instruct"
)
METRICS = ("memorization", "style", "content", "fluency")


def read_jsonl(path: str) -> List[Dict[str, object]]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: str | Path, rows: Iterable[Dict[str, object]], mode: str = "w") -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open(mode, encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: str | Path, value: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def make_judge_client():
    from openai import OpenAI
    return OpenAI()



# Memorization and diversity.


_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)

def tokenize(text: str) -> List[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


def count_ngrams(tokens: Sequence[str], n: int) -> Counter:
    if len(tokens) < n:
        return Counter()
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def corpus_bleu4_fast(
    hyp_tokens: Sequence[str],
    ref_lengths: Sequence[int],
    maxref_counts_by_n: Dict[int, Counter],
) -> float:
    """BLEU-4 of a single hypothesis against a corpus summarized by:
      - ref_lengths: lengths of all reference passages (for brevity penalty).
      - maxref_counts_by_n: for each n in {1,2,3,4}, a Counter mapping n-gram ->
        MAX count of that n-gram across any single reference passage.

    Equivalent to standard multi-reference BLEU, but O(|hyp n-grams|) per n
    instead of O(|hyp n-grams| x |refs|).
    """
    hyp_len = len(hyp_tokens)
    if hyp_len == 0 or not ref_lengths:
        return 0.0

    closest_len = min(ref_lengths, key=lambda rl: (abs(rl - hyp_len), rl))

    log_precisions = []
    for n in (1, 2, 3, 4):
        hyp_counts = count_ngrams(hyp_tokens, n)
        total_hyp = sum(hyp_counts.values())
        if total_hyp == 0:
            return 0.0
        maxref = maxref_counts_by_n[n]
        clipped = 0
        for ng, h_c in hyp_counts.items():
            m = maxref.get(ng, 0)
            if m:
                clipped += h_c if h_c < m else m
        if clipped == 0:
            clipped = 1
            total_hyp = total_hyp + 1
        log_precisions.append(math.log(clipped / total_hyp))

    bp = 1.0 if hyp_len > closest_len else math.exp(1 - closest_len / hyp_len)
    return 100.0 * bp * math.exp(sum(log_precisions) / 4.0)


def build_author_index(
    train_rows: Sequence[Dict[str, object]],
    author_field: str,
    text_field: str,
    novel_ngram_sizes: Sequence[int],
) -> Tuple[
    Dict[str, List[List[str]]],                   # author -> list of tokenized passages
    Dict[str, Dict[int, set]],                    # author -> n -> set of n-gram tuples (novelty)
    Dict[str, Dict[int, Counter]],                # author -> n -> max-over-refs n-gram counter (BLEU)
    Dict[str, List[int]],                         # author -> list of passage lengths (BLEU BP)
    Dict[str, Dict[str, List[Tuple[int, int]]]],  # author -> unigram lookup (LCS)
]:
    author_tokens: Dict[str, List[List[str]]] = defaultdict(list)
    for row in train_rows:
        author = str(row.get(author_field, "")).strip()
        text = str(row.get(text_field, "")).strip()
        if not author or not text:
            continue
        author_tokens[author].append(tokenize(text))

    bleu_ns = (1, 2, 3, 4)
    author_novel_ngrams: Dict[str, Dict[int, set]] = {}
    author_bleu_maxref: Dict[str, Dict[int, Counter]] = {}
    author_ref_lengths: Dict[str, List[int]] = {}
    author_unigram_lookup: Dict[str, Dict[str, List[Tuple[int, int]]]] = {}

    for author, passages in author_tokens.items():
        # Novel n-gram sets (union across passages).
        per_n_sets: Dict[int, set] = {n: set() for n in novel_ngram_sizes}
        for toks in passages:
            for n in novel_ngram_sizes:
                if len(toks) >= n:
                    for i in range(len(toks) - n + 1):
                        per_n_sets[n].add(tuple(toks[i : i + n]))
        author_novel_ngrams[author] = per_n_sets

        # Max-over-refs n-gram counter per n (for multi-ref BLEU).
        maxref: Dict[int, Counter] = {n: Counter() for n in bleu_ns}
        for toks in passages:
            for n in bleu_ns:
                if len(toks) < n:
                    continue
                c = count_ngrams(toks, n)
                # For each n-gram, keep the MAX count across refs.
                dst = maxref[n]
                for ng, cnt in c.items():
                    if cnt > dst.get(ng, 0):
                        dst[ng] = cnt
        author_bleu_maxref[author] = maxref
        author_ref_lengths[author] = [len(t) for t in passages]

        # Unigram lookup (for LCS).
        author_unigram_lookup[author] = build_unigram_lookup(passages)

    return (
        dict(author_tokens),
        author_novel_ngrams,
        author_bleu_maxref,
        author_ref_lengths,
        author_unigram_lookup,
    )


def novel_ngram_ratio(
    gen_tokens: List[str],
    train_ngrams_n: set,
    n: int,
) -> Tuple[float, int, int]:
    if len(gen_tokens) < n:
        return float("nan"), 0, 0
    total = 0
    novel = 0
    for i in range(len(gen_tokens) - n + 1):
        ng = tuple(gen_tokens[i : i + n])
        total += 1
        if ng not in train_ngrams_n:
            novel += 1
    return (novel / total) if total else float("nan"), novel, total


def longest_common_substring_words(
    gen_tokens: List[str],
    unigram_lookup: Dict[str, List[Tuple[int, int]]],
    train_passages_tokens: Sequence[List[str]],
) -> int:
    """Longest contiguous word span in gen_tokens that appears verbatim in any
    train passage.

    Algorithm: for each position i in gen, use the unigram_lookup to find every
    (passage_idx, j) where train_passages_tokens[passage_idx][j] == gen[i], then
    extend the match as far as it goes. Keep the maximum extension seen.

    unigram_lookup is built once per author and reused across all generations
    for that author, so this runs in O(sum_i count(gen[i])) per generation plus
    the match-extension work.
    """
    if not gen_tokens or not train_passages_tokens:
        return 0

    best = 0
    n_gen = len(gen_tokens)
    for i, w in enumerate(gen_tokens):
        # Early exit: can't beat current best.
        if n_gen - i <= best:
            break
        starts = unigram_lookup.get(w)
        if not starts:
            continue
        for (p_idx, j) in starts:
            train_toks = train_passages_tokens[p_idx]
            n_train = len(train_toks)
            # Max possible match from this starting pair.
            max_reach = min(n_gen - i, n_train - j)
            if max_reach <= best:
                continue
            # Extend.
            k = 1
            while k < max_reach and gen_tokens[i + k] == train_toks[j + k]:
                k += 1
            if k > best:
                best = k
    return best


def build_unigram_lookup(
    train_passages_tokens: Sequence[List[str]],
) -> Dict[str, List[Tuple[int, int]]]:
    """Map each token -> list of (passage_idx, position) in train passages."""
    lookup: Dict[str, List[Tuple[int, int]]] = {}
    for p_idx, toks in enumerate(train_passages_tokens):
        for j, w in enumerate(toks):
            lookup.setdefault(w, []).append((p_idx, j))
    return lookup


def self_bleu_from_tokens(hyps_tokens: Sequence[Sequence[str]]) -> float:
    """Mean self-BLEU over a set of hypotheses using leave-one-out multi-ref BLEU.

    For each hypothesis i, references = all other hypotheses. We precompute a
    total maxref Counter (max count across ALL hyps) and a per-hyp Counter, then
    the leave-one-out maxref is approximated by the total (excluding the hyp's
    own contribution when it's unique). For speed, we use the total directly —
    this is a standard Self-BLEU approximation when hyps are short/diverse.
    """
    k = len(hyps_tokens)
    if k < 2:
        return float("nan")
    bleu_ns = (1, 2, 3, 4)
    # Per-hyp ngram counts + global max across all hyps.
    per_hyp: List[Dict[int, Counter]] = [
        {n: count_ngrams(h, n) for n in bleu_ns} for h in hyps_tokens
    ]
    all_lengths = [len(h) for h in hyps_tokens]
    global_maxref: Dict[int, Counter] = {n: Counter() for n in bleu_ns}
    for counts_by_n in per_hyp:
        for n in bleu_ns:
            dst = global_maxref[n]
            for ng, c in counts_by_n[n].items():
                if c > dst.get(ng, 0):
                    dst[ng] = c

    scores = []
    for i in range(k):
        # Leave-one-out maxref: for each n-gram that the i-th hyp contributes,
        # subtract its contribution if it uniquely held the max.
        loo: Dict[int, Counter] = {}
        own = per_hyp[i]
        others_lengths = [all_lengths[j] for j in range(k) if j != i]
        for n in bleu_ns:
            c_loo = Counter(global_maxref[n])  # shallow copy
            for ng, own_c in own[n].items():
                if c_loo.get(ng, 0) == own_c:
                    # Recompute max across others.
                    max_other = 0
                    for j in range(k):
                        if j == i:
                            continue
                        oc = per_hyp[j][n].get(ng, 0)
                        if oc > max_other:
                            max_other = oc
                    if max_other == 0:
                        del c_loo[ng]
                    else:
                        c_loo[ng] = max_other
            loo[n] = c_loo
        score = corpus_bleu4_fast(hyps_tokens[i], others_lengths, loo)
        scores.append(score)
    return sum(scores) / len(scores)


def mean(xs: Sequence[float]) -> float:
    xs = [x for x in xs if isinstance(x, (int, float)) and x == x]  # drop NaN
    return (sum(xs) / len(xs)) if xs else float("nan")


# Style attribution.


EVAL_SYSTEM_PROMPT = (
    "You are an expert literary analyst. Identify the author of the provided passage based on their writing style instead of the content. "
    "Answer with only the exact author name."
)

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
) -> Tuple[Any, Any]:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

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


def predict_author_closedset(
    model,
    tokenizer,
    *,
    text: str,
    labels: Sequence[str],
    device: Any,
    max_length: int,
) -> Tuple[str, float]:
    import torch

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


# Content preservation.


def load_existing_ids(path: str, id_field: str) -> set[str]:
    if not os.path.exists(path):
        return set()
    ids = set()
    with open(path, "r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            val = obj.get(id_field)
            if val is not None:
                ids.add(str(val))
    return ids


def build_content_prompt(summary: str, generation: str) -> str:
    return "\n".join(
        [
            "You are judging whether the generation reflects the summary to some extent.",
            "The generation is long; if it clearly mentions the summary idea anywhere, label it yes.",
            "Use 'partial' only if it is vaguely related but does not clearly express the summary.",
            "Use 'no' only if it does not reflect the summary at all.",
            "Return JSON only with keys: label, score.",
            "label must be one of: yes, partial, no.",
            "score must be an integer 1-5 (5 = fully matches summary).",
            "",
            "Summary:",
            summary,
            "",
            "Generation:",
            generation,
        ]
    ).strip()


def parse_json_response(text: str) -> Dict[str, object]:
    text = text.strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            obj = json.loads(text[start : end + 1])
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
    return {}


# Fluency.


FLUENCY_PROMPT = (
    "Given the text below, give a score between 0 and 100 for the fluency and grammatical "
    "correctness of the text. Be strict and only give high scores sparingly. Only output the "
    "number and do not output any other text or explanation."
)

def record_key(rec: Dict[str, object], author_field: str = "author", id_field: str = "sample_id") -> str:
    generator = str(rec.get("generator", "")).strip()
    author = str(rec.get(author_field, "")).strip()
    sample_id = str(rec.get(id_field, "")).strip()
    return f"{generator}::{author}::{sample_id}"


def load_processed_keys(path: str, author_field: str = "author", id_field: str = "sample_id") -> set[str]:
    processed: set[str] = set()
    if not path or not os.path.exists(path):
        return processed
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            processed.add(record_key(rec, author_field, id_field))
    return processed


def build_fluency_prompt(text: str) -> str:
    return f"{FLUENCY_PROMPT}\n\nTEXT:\n```\n{text}\n```"


def extract_score(raw: str) -> Tuple[Optional[float], Optional[str]]:
    cleaned = raw.strip()
    if not cleaned:
        return None, "empty_response"
    match = re.search(r"-?\d+(?:\.\d+)?", cleaned)
    if not match:
        return None, f"no_number_in_response: {cleaned[:200]}"
    try:
        score = float(match.group(0))
    except ValueError:
        return None, f"parse_error: {cleaned[:200]}"
    if score < 0 or score > 100:
        return None, f"out_of_range: {score}"
    return score, None


def call_gpt_score(
    client: Any,
    text: str,
    *,
    model: str,
    temperature: float,
    max_retries: int,
    timeout_s: float,
) -> Dict[str, object]:
    last_error: Optional[str] = None
    prompt = build_fluency_prompt(text)
    for attempt in range(1, max_retries + 1):
        try:
            start = time.time()
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=16,
                timeout=timeout_s,
            )
            elapsed = time.time() - start
            content = (resp.choices[0].message.content or "").strip()
            score, parse_error = extract_score(content)
            return {
                "fluency_score": score,
                "raw_response": content,
                "parse_error": parse_error,
                "tokens": getattr(resp.usage, "total_tokens", None),
                "latency_sec": round(elapsed, 3),
                "prompt": FLUENCY_PROMPT,
            }
        except Exception as exc:  # pragma: no cover
            last_error = str(exc)
            time.sleep(1.5 * attempt)
    return {
        "fluency_score": None,
        "raw_response": "",
        "parse_error": None,
        "tokens": None,
        "latency_sec": None,
        "prompt": FLUENCY_PROMPT,
        "error": last_error,
    }


def summarize_scores(path: str) -> Dict[str, object]:
    total_records = 0
    scored = 0
    total_score = 0.0
    min_rec: Optional[Dict[str, object]] = None
    max_rec: Optional[Dict[str, object]] = None
    min_score: Optional[float] = None
    max_score: Optional[float] = None

    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            total_records += 1
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            score = rec.get("fluency_score", None)
            if score is None:
                continue
            try:
                score_val = float(score)
            except (TypeError, ValueError):
                continue
            scored += 1
            total_score += score_val
            if min_score is None or score_val < min_score:
                min_score = score_val
                min_rec = rec
            if max_score is None or score_val > max_score:
                max_score = score_val
                max_rec = rec

    avg_score = (total_score / scored) if scored else None
    summary: Dict[str, object] = {
        "count_total": total_records,
        "count_scored": scored,
        "count_missing": total_records - scored,
        "average_score": round(avg_score, 4) if avg_score is not None else None,
        "min_score": min_score,
        "min_record": None,
        "max_score": max_score,
        "max_record": None,
    }
    if min_rec is not None:
        summary["min_record"] = {
            "generator": min_rec.get("generator"),
            "author": min_rec.get("author"),
            "sample_id": min_rec.get("sample_id"),
            "text": min_rec.get("text"),
            "fluency_score": min_rec.get("fluency_score"),
        }
    if max_rec is not None:
        summary["max_record"] = {
            "generator": max_rec.get("generator"),
            "author": max_rec.get("author"),
            "sample_id": max_rec.get("sample_id"),
            "text": max_rec.get("text"),
            "fluency_score": max_rec.get("fluency_score"),
        }
    return summary


def evaluate_memorization(rows, args):
    output_dir = Path(args.output_dir)
    gen_rows = rows
    train_rows = read_jsonl(args.train_jsonl)
    print(f"[info] loaded {len(gen_rows)} generations, {len(train_rows)} train passages")

    generator = args.label or str(
        gen_rows[0].get("generator", Path(args.input).stem) if gen_rows else "unknown"
    )

    (
        author_tokens,
        author_novel_ngrams,
        author_bleu_maxref,
        author_ref_lengths,
        author_unigram_lookup,
    ) = build_author_index(
        train_rows, args.author_field, args.text_field, args.ngram_sizes,
    )
    print(f"[info] indexed {len(author_tokens)} authors; "
          f"avg passages/author = {sum(len(v) for v in author_tokens.values())/max(1,len(author_tokens)):.1f}")

    per_passage: List[Dict[str, object]] = []
    by_author_hyps_tokens: Dict[str, List[List[str]]] = defaultdict(list)

    for row in gen_rows:
        author = str(row.get(args.author_field, "")).strip()
        text = str(row.get(args.text_field, "")).strip()
        sample_id = row.get(args.id_field)
        if not author or not text:
            continue
        if author not in author_tokens:
            per_passage.append({
                "author": author,
                "sample_id": sample_id,
                "skipped": True,
                "reason": "author_not_in_train",
            })
            continue

        gen_tokens = tokenize(text)
        bleu_val = corpus_bleu4_fast(
            gen_tokens,
            author_ref_lengths[author],
            author_bleu_maxref[author],
        )

        novel_stats: Dict[str, object] = {}
        for n in args.ngram_sizes:
            ratio, novel_count, total_count = novel_ngram_ratio(
                gen_tokens, author_novel_ngrams[author][n], n,
            )
            novel_stats[f"novel_{n}gram_ratio"] = ratio
            novel_stats[f"novel_{n}gram_count"] = novel_count
            novel_stats[f"total_{n}gram_count"] = total_count

        lcs_len = longest_common_substring_words(
            gen_tokens,
            author_unigram_lookup[author],
            author_tokens[author],
        )

        per_passage.append({
            "author": author,
            "sample_id": sample_id,
            "generator": generator,
            "num_gen_tokens": len(gen_tokens),
            "bleu_vs_train": bleu_val,
            "lcs_words": lcs_len,
            **novel_stats,
        })
        by_author_hyps_tokens[author].append(gen_tokens)

    # Self-BLEU per author.
    per_author_self_bleu: Dict[str, float] = {}
    for author, hyps in by_author_hyps_tokens.items():
        per_author_self_bleu[author] = self_bleu_from_tokens(hyps)

    # Aggregate.
    scored = [p for p in per_passage if not p.get("skipped")]
    agg = {
        "generator": generator,
        "generated_jsonl": args.input,
        "train_jsonl": args.train_jsonl,
        "count_total": len(gen_rows),
        "count_scored": len(scored),
        "count_skipped": len(per_passage) - len(scored),
        "mean_bleu_vs_train": mean([p["bleu_vs_train"] for p in scored]),
        "mean_lcs_words": mean([p["lcs_words"] for p in scored]),
        "max_lcs_words": max([p["lcs_words"] for p in scored], default=0),
    }
    for n in args.ngram_sizes:
        key = f"novel_{n}gram_ratio"
        agg[f"mean_{key}"] = mean([p[key] for p in scored if isinstance(p.get(key), float)])
    agg["mean_self_bleu"] = mean(list(per_author_self_bleu.values()))
    agg["per_author_self_bleu"] = {k: v for k, v in per_author_self_bleu.items()}

    # LCS distribution buckets (plagiarism risk).
    lcs_vals = [p["lcs_words"] for p in scored]
    agg["lcs_distribution"] = {
        "gte_10": sum(1 for v in lcs_vals if v >= 10),
        "gte_15": sum(1 for v in lcs_vals if v >= 15),
        "gte_25": sum(1 for v in lcs_vals if v >= 25),
        "gte_50": sum(1 for v in lcs_vals if v >= 50),
    }

    write_json(output_dir / "memorization.json", agg)
    print(json.dumps({k: v for k, v in agg.items() if k != "per_author_self_bleu"}, indent=2))

    pp_path = output_dir / "memorization_details.jsonl"
    write_jsonl(pp_path, per_passage)
    print(f"[info] wrote per-passage results to {pp_path}")
    return agg


def evaluate_style(rows, args):
    import torch

    output_dir = Path(args.output_dir) / "style"
    output_dir.mkdir(parents=True, exist_ok=True)
    generator = args.label or str(rows[0].get("generator", Path(args.input).stem) if rows else "unknown")
    authors = sorted({str(row.get(args.author_field, "")).strip() for row in rows} - {""})
    if not authors:
        raise ValueError("No authors found in generated passages for style evaluation.")
    samples = [
        GenSample(
            generator=generator,
            author=str(row.get(args.author_field, "")),
            text=str(row.get(args.text_field, "")),
            sample_id=row.get(args.id_field, 0),
        )
        for row in rows
    ]
    summaries = []
    for raw_spec in args.evaluator or [DEFAULT_EVALUATOR]:
        spec = parse_adapter_spec(raw_spec)
        print(f"[Style] {spec.label}")
        model, tokenizer = load_model(spec.base_model, spec.adapter_path, use_4bit=args.eval_use_4bit)
        modes = ("open", "closed") if args.eval_mode == "both" else (args.eval_mode,)
        with torch.no_grad():
            for mode in modes:
                if mode == "open":
                    correct, total, details = evaluate_classifier_open(
                        model, tokenizer, samples=samples, authors=authors,
                        max_new_tokens=args.eval_max_new_tokens, batch_size=args.eval_batch_size,
                    )
                else:
                    correct, total, details = evaluate_classifier_closed(
                        model, tokenizer, samples=samples, authors=authors,
                        max_length=args.eval_max_length,
                    )
                accuracy = correct / total if total else 0.0
                summaries.append({
                    "generator": generator, "evaluator": f"{spec.label}:{mode}",
                    "correct": correct, "total": total, "accuracy": accuracy,
                })
                write_jsonl(output_dir / f"eval_{mode}_{spec.label}_on_{generator}.jsonl", details)
                print(f"  {generator} [{mode}]: {accuracy:.4f} ({correct}/{total})")
        del model, tokenizer
        torch.cuda.empty_cache()
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["generator", "evaluator", "correct", "total", "accuracy"])
        writer.writeheader()
        writer.writerows(summaries)
    return summaries


def evaluate_content(rows, args, client):
    output_path = Path(args.output_dir) / "content_judge.jsonl"
    existing = load_existing_ids(str(output_path), args.id_field) if args.append else set()
    mode = "a" if args.append else "w"
    write_jsonl(output_path, [], mode=mode)
    with output_path.open("a", encoding="utf-8") as handle:
        for idx, row in enumerate(tqdm(rows, desc="Content preservation", unit="row"), start=1):
            summary = str(row.get(args.summary_field, "")).strip()
            generation = str(row.get(args.text_field, "")).strip()
            if not summary or not generation:
                continue
            row_id = row.get(args.id_field, idx)
            if str(row_id) in existing:
                continue
            prompt = build_content_prompt(summary, generation)
            for attempt in range(args.max_retries + 1):
                try:
                    response = client.responses.create(
                        model=args.judge_model, input=prompt, temperature=args.temperature,
                        max_output_tokens=args.max_output_tokens,
                    )
                    parsed = parse_json_response((response.output_text or "").strip())
                    label = str(parsed.get("label", "")).strip().lower()
                    score = int(parsed.get("score"))
                    if not 1 <= score <= 5:
                        raise ValueError("Content score out of range.")
                    if label not in {"yes", "partial", "no"}:
                        label = "yes" if score >= 3 else "partial" if score == 2 else "no"
                    break
                except Exception:
                    if attempt == args.max_retries:
                        raise
                    time.sleep(min(2 ** (attempt + 1), 30))
            result = dict(row)
            result.update(judge_label=label, judge_score=score, judge_model=args.judge_model)
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            handle.flush()
            if args.sleep:
                time.sleep(args.sleep)
    judged = read_jsonl(str(output_path))
    count = len(judged)
    average = sum(float(row["judge_score"]) for row in judged) / count if count else None
    label_counts = {label: sum(row["judge_label"] == label for row in judged) for label in ("yes", "partial", "no")}
    aggregate = {
        "count": count, "average_score": average,
        "content_preservation_percent": 25.0 * (average - 1.0) if average is not None else None,
        "label_counts": label_counts,
        "label_rates": {label: value / count if count else 0.0 for label, value in label_counts.items()},
        "model": args.judge_model,
    }
    write_json(output_path.with_name("content_judge_aggregate.json"), aggregate)
    print(f"[Content] {count} scored passages; preservation = {aggregate['content_preservation_percent']}")
    return aggregate


def evaluate_fluency(rows, args, client):
    output_path = Path(args.output_dir) / "fluency.jsonl"
    processed = load_processed_keys(str(output_path), args.author_field, args.id_field) if args.append else set()
    mode = "a" if args.append else "w"
    with output_path.open(mode, encoding="utf-8") as handle:
        for row in tqdm(rows, desc="Fluency", unit="row"):
            if record_key(row, args.author_field, args.id_field) in processed:
                continue
            text = str(row.get(args.text_field, "")).strip()
            result = dict(row)
            if text:
                result.update(call_gpt_score(
                    client, text, model=args.judge_model, temperature=args.temperature,
                    max_retries=args.max_retries, timeout_s=args.timeout_s,
                ))
                if args.sleep:
                    time.sleep(args.sleep)
            else:
                result.update(fluency_score=None, raw_response="", parse_error="empty_text", prompt=FLUENCY_PROMPT)
            result["model"] = args.judge_model
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            handle.flush()
    aggregate = summarize_scores(str(output_path))
    aggregate.update(input=args.input, output=str(output_path), model=args.judge_model)
    write_json(output_path.with_name("fluency_aggregate.json"), aggregate)
    print(f"[Fluency] {aggregate['count_scored']} scored passages; mean = {aggregate['average_score']}")
    return aggregate


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="Generated passages in JSONL format.")
    parser.add_argument("--output_dir", default="results/generation")
    parser.add_argument("--metrics", default="memorization,style", help="Comma-separated memorization,style,content,fluency; or all.")
    parser.add_argument("--train_jsonl", default="data/Gutenberg30/train.jsonl", help="Same-author references for memorization.")
    parser.add_argument("--evaluator", action="append", help="Style classifier: NAME=ADAPTER|BASE_MODEL (repeatable).")
    parser.add_argument("--label", help="Generator name; defaults to the input's generator field or filename.")
    parser.add_argument("--author_field", default="author")
    parser.add_argument("--text_field", default="text")
    parser.add_argument("--summary_field", default="content_pack")
    parser.add_argument("--id_field", default="sample_id")
    parser.add_argument("--ngram_sizes", type=int, nargs="+", default=[4, 6, 8])
    parser.add_argument("--eval_use_4bit", action="store_true")
    parser.add_argument("--eval_mode", choices=("open", "closed", "both"), default="open")
    parser.add_argument("--eval_max_new_tokens", type=int, default=8)
    parser.add_argument("--eval_max_length", type=int, default=768)
    parser.add_argument("--eval_batch_size", type=int, default=4)
    parser.add_argument("--judge_model", default="gpt-4o-mini")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_output_tokens", type=int, default=128, help="Content judge response limit; fluency retains its 16-token limit.")
    parser.add_argument("--max_retries", type=int, default=5)
    parser.add_argument("--timeout_s", type=float, default=45.0)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--limit", type=int, default=0, help="Evaluate the first N passages (0 = all).")
    parser.add_argument("--append", action="store_true", help="Resume existing content/fluency scores.")
    args = parser.parse_args()
    metrics = list(dict.fromkeys(metric.strip() for metric in args.metrics.split(",")))
    if metrics == ["all"]:
        metrics = list(METRICS)
    if not metrics or any(metric not in METRICS for metric in metrics):
        parser.error("--metrics must contain memorization, style, content, fluency, or all")
    args.metrics = metrics
    return args


def main():
    args = parse_args()
    rows = read_jsonl(args.input)
    if args.limit > 0:
        rows = rows[:args.limit]
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    results = {"input": args.input, "metrics": {}}
    client = None
    for metric in args.metrics:
        if metric == "memorization":
            result = evaluate_memorization(rows, args)
        elif metric == "style":
            result = evaluate_style(rows, args)
        else:
            if client is None:
                client = make_judge_client()
            evaluator = evaluate_content if metric == "content" else evaluate_fluency
            result = evaluator(rows, args, client)
        results["metrics"][metric] = result
        write_json(Path(args.output_dir) / "summary.json", results)
    print(f"Saved evaluation results to {args.output_dir}")


if __name__ == "__main__":
    main()
