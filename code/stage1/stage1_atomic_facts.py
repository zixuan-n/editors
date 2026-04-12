#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate styleless atomic facts for the Reuters CCAT dataset (JSONL).

Input rows must contain: {"author": ..., "title": ..., "text": ...}
Optional: "chunk_id" (or "chunk_index", "orig_index").
Output JSONL rows: {"text": <fact>, "source": {"author": ..., "title": ..., "chunk_id": ...}}
"""

from __future__ import annotations

import argparse
import io
import json
import os
import random
import re
import time
from typing import Dict, Iterable, List, Optional, Tuple

from openai import OpenAI

try:
    from tqdm.auto import tqdm  # type: ignore
except ImportError:  # pragma: no cover
    def tqdm(iterable: Iterable, **_: dict):  # type: ignore
        return iterable


SEED = 42


def _build_tokenizer(
    model_id: str, tokenizer_model: str
) -> Tuple[str, object]:
    if tokenizer_model:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(tokenizer_model, use_fast=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        return "hf", tok

    try:
        import tiktoken

        enc = tiktoken.encoding_for_model(model_id)
        return "tiktoken", enc
    except Exception:
        return "", None


def _truncate_text(
    text: str, max_tokens: int, tokenizer_kind: str, tokenizer: object
) -> str:
    if max_tokens <= 0:
        return text
    if tokenizer_kind == "tiktoken":
        enc = tokenizer  # type: ignore[assignment]
        ids = enc.encode(text)
        if len(ids) <= max_tokens:
            return text
        return enc.decode(ids[:max_tokens])
    if tokenizer_kind == "hf":
        tok = tokenizer  # type: ignore[assignment]
        ids = tok(text, add_special_tokens=False)["input_ids"]
        if len(ids) <= max_tokens:
            return text
        return tok.decode(ids[:max_tokens], skip_special_tokens=True)
    raise RuntimeError("Tokenizer not available for truncation.")


def _try_parse_json_array(s: str) -> List[str]:
    """Parse an LLM response into a JSON array of strings if possible."""
    try:
        data = json.loads(s)
        if isinstance(data, list) and all(isinstance(x, str) for x in data):
            return data
    except Exception:
        pass
    match = re.search(r"\[[\s\S]*\]", s)
    if match:
        try:
            data = json.loads(match.group(0))
            if isinstance(data, list) and all(isinstance(x, str) for x in data):
                return data
        except Exception:
            pass
    return []


def processed_keys_from_out(out_path: str) -> set[str]:
    done: set[str] = set()
    if not os.path.exists(out_path):
        return done
    with open(out_path, "r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            src = obj.get("source", {})
            key = f"{src.get('author','')}|||{src.get('title','')}|||{src.get('chunk_id',-1)}"
            done.add(key)
    return done


def record_key(rec: Dict[str, object]) -> str:
    return f"{rec.get('author','')}|||{rec.get('title','')}|||{rec.get('chunk_id',-1)}"


def load_rows(
    path: str,
    limit: int = 0,
    debug: bool = False,
    default_title: str = "Unknown",
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle, start=1):
            if limit and len(rows) >= limit:
                break
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                if debug:
                    print(f"[WARN] Bad JSON at {path}:{idx}")
                continue
            author = str(obj.get("author", "")).strip()
            title = str(obj.get("title", "")).strip() or default_title
            text = obj.get("text")
            if not author or not text:
                if debug:
                    print(f"[WARN] Missing author/text at {path}:{idx}")
                continue
            chunk_id_raw = obj.get("chunk_id")
            if chunk_id_raw is None:
                orig_index = obj.get("orig_index")
                chunk_index = obj.get("chunk_index")
                if orig_index is not None and chunk_index is not None:
                    # Combine orig_index + chunk_index to avoid collisions when titles repeat.
                    try:
                        chunk_id = int(orig_index) * 1000 + int(chunk_index)
                    except Exception:
                        chunk_id = 0
                elif chunk_index is not None:
                    try:
                        chunk_id = int(chunk_index)
                    except Exception:
                        chunk_id = 0
                else:
                    # Fall back to a unique per-line id when no chunk info is present.
                    chunk_id = idx
            else:
                try:
                    chunk_id = int(chunk_id_raw)
                except Exception:
                    chunk_id = 0
            rows.append(
                {"author": author, "title": title, "chunk_id": chunk_id, "text": text}
            )
    return rows


def call_gpt_atomic_facts(
    client: OpenAI,
    passage: str,
    *,
    model: str,
    min_facts: int,
    max_facts: int,
    soft_accept_min: int,
    temperature: float,
    max_retries: int,
    timeout_s: float,
    debug: bool,
    fail_log: Optional[io.TextIOBase],
    print_raw: bool,
) -> List[str]:
    system = (
        "You turn passages into neutral, styleless atomic facts. "
        "Each fact should stand alone, use plain prose, and stay faithful to the passage."
    )
    user = (
        f"Rewrite this passage as {min_facts} to {max_facts} standalone atomic factual statements.\n"
        f"- Capture what the passage claims (opinions become factual claims about the author/source).\n"
        f"- Keep names and entities unchanged; avoid figurative language.\n"
        f"- Output ONLY a JSON array of strings, nothing else.\n\n"
        f"PASSAGE:\n{passage}"
    )

    warned_under = False
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=temperature,
                max_tokens=800,
                timeout=timeout_s,
            )
            out = resp.choices[0].message.content.strip()
            if print_raw:
                print("[RAW]", out, flush=True)
            facts = _try_parse_json_array(out)

            if len(facts) >= min_facts:
                return facts[:max_facts]

            if len(facts) >= soft_accept_min:
                if debug and not warned_under:
                    print(
                        f"[WARN] Got {len(facts)} (< min_facts={min_facts}); accepting (soft).",
                        flush=True,
                    )
                    warned_under = True
                return facts[:max_facts]

            if debug and fail_log:
                fail_log.write(
                    json.dumps(
                        {
                            "reason": "too_few_after_parse",
                            "attempt": attempt,
                            "got_len": len(facts),
                            "raw": out[:2000],
                        }
                    )
                    + "\n"
                )
            time.sleep(1.0 * attempt)
        except Exception as exc:  # pragma: no cover
            if debug and fail_log:
                fail_log.write(
                    json.dumps(
                        {
                            "reason": "exception",
                            "attempt": attempt,
                            "error": str(exc)[:1000],
                        }
                    )
                    + "\n"
                )
            time.sleep(1.5 * attempt)
    return []


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Generate atomic facts for Reuters CCAT JSONL.")
    ap.add_argument(
        "--input",
        default="splits/reuter_c50_10authors_100w_keep50/train.jsonl",
        help="Input JSONL path.",
    )
    ap.add_argument(
        "--out",
        default="stage1_reuter_c50_10authors_100w_keep50_atomic_facts.jsonl",
        help="Output JSONL path.",
    )
    ap.add_argument("--model", default="gpt-4o-mini", help="OpenAI chat model to use.")
    ap.add_argument("--min_facts", type=int, default=1, help="Min facts to request.")
    ap.add_argument("--max_facts", type=int, default=10, help="Max facts to keep.")
    ap.add_argument(
        "--soft_accept_min",
        type=int,
        default=2,
        help="Accept outputs with at least this many facts even if < min_facts.",
    )
    ap.add_argument("--temperature", type=float, default=0.1, help="Sampling temperature.")
    ap.add_argument("--max_retries", type=int, default=5, help="LLM retries per passage.")
    ap.add_argument("--timeout_s", type=float, default=45.0, help="LLM timeout (seconds).")
    ap.add_argument("--sleep_s", type=float, default=0.0, help="Optional sleep between calls.")
    ap.add_argument("--dry_run", type=int, default=0, help="If >0, process only first N rows.")
    ap.add_argument(
        "--max_input_tokens",
        type=int,
        default=0,
        help="If >0, truncate each passage to this many tokens before the API call.",
    )
    ap.add_argument(
        "--tokenizer_model",
        default="",
        help=(
            "Optional tokenizer model/path for truncation (HF AutoTokenizer). "
            "If empty, try tiktoken for the OpenAI model."
        ),
    )
    ap.add_argument(
        "--num_shards",
        type=int,
        default=1,
        help="Total number of shards for parallel runs (default: 1).",
    )
    ap.add_argument(
        "--shard_id",
        type=int,
        default=0,
        help="Shard index in [0, num_shards-1] (default: 0).",
    )
    ap.add_argument("--fail_log", default="reuter_stage1_failures.jsonl", help="Log parse/errors when --debug.")
    ap.add_argument("--debug", action="store_true", help="Verbose logging.")
    ap.add_argument("--append", action="store_true", help="Append and skip already processed records.")
    ap.add_argument("--print_raw", action="store_true", help="Print raw model outputs.")
    ap.add_argument(
        "--default_title",
        default="Unknown",
        help="Fallback title when input rows lack a title field.",
    )
    ap.add_argument("--seed", type=int, default=SEED)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    if args.num_shards < 1:
        raise ValueError("--num_shards must be >= 1")
    if args.shard_id < 0 or args.shard_id >= args.num_shards:
        raise ValueError("--shard_id must be in [0, num_shards-1]")

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENAI_API_KEY in your environment before running.")
    client = OpenAI(api_key=api_key)

    if args.min_facts < 1 or args.max_facts < args.min_facts:
        raise ValueError("Require 1 <= min_facts <= max_facts")

    tokenizer_kind = ""
    tokenizer = None
    if args.max_input_tokens > 0:
        tokenizer_kind, tokenizer = _build_tokenizer(args.model, args.tokenizer_model)
        if not tokenizer_kind:
            raise RuntimeError(
                "Tokenizer unavailable. Install tiktoken or set --tokenizer_model."
            )

    rows = load_rows(
        args.input,
        limit=args.dry_run,
        debug=args.debug,
        default_title=args.default_title,
    )
    if not rows:
        print("No rows loaded; check --input path or file format.", flush=True)
        return
    if args.num_shards > 1:
        rows = [r for i, r in enumerate(rows) if i % args.num_shards == args.shard_id]
    print(f"Loaded {len(rows)} rows from {args.input}.")

    processed = processed_keys_from_out(args.out) if args.append else set()
    if processed:
        print(f"Resuming: {len(processed)} rows already present in {args.out}.")

    out_dir = os.path.dirname(os.path.abspath(args.out))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    fail_fp = open(args.fail_log, "w", encoding="utf-8") if args.debug else None
    written_facts = 0
    written_rows = 0
    mode = "a" if args.append else "w"

    with open(args.out, mode, encoding="utf-8") as fout:
        pbar = tqdm(rows, desc="Generating CCAT atomic facts", unit="row")
        for rec in pbar:
            key = record_key(rec)
            if key in processed:
                continue

            passage = rec["text"]
            if args.max_input_tokens > 0:
                passage = _truncate_text(
                    passage, args.max_input_tokens, tokenizer_kind, tokenizer
                )

            facts = call_gpt_atomic_facts(
                client,
                passage,
                model=args.model,
                min_facts=args.min_facts,
                max_facts=args.max_facts,
                soft_accept_min=args.soft_accept_min,
                temperature=args.temperature,
                max_retries=args.max_retries,
                timeout_s=args.timeout_s,
                debug=args.debug,
                fail_log=fail_fp,
                print_raw=args.print_raw,
            )
            if not facts:
                continue

            written_rows += 1
            for fact in facts:
                payload = {
                    "text": fact.strip(),
                    "source": {
                        "author": rec["author"],
                        "title": rec["title"],
                        "chunk_id": rec["chunk_id"],
                    },
                }
                fout.write(json.dumps(payload, ensure_ascii=False) + "\n")
            written_facts += len(facts)

            if args.sleep_s > 0:
                time.sleep(args.sleep_s)
            pbar.set_postfix({"rows": written_rows, "facts": written_facts})

    if fail_fp:
        fail_fp.close()
    print(f"Done. Wrote {written_facts} facts from {written_rows} rows to {args.out}.")


if __name__ == "__main__":
    main()
