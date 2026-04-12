#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Use GPT-4o mini as a judge to evaluate whether a generation reflects a summary.

Input JSONL: expects summary field + generated field (configurable).
Output JSONL: adds label/score/rationale plus metadata.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, Iterable, List

from openai import OpenAI

try:
    from tqdm.auto import tqdm  # type: ignore
except ImportError:  # pragma: no cover
    def tqdm(iterable: Iterable, **_: dict):  # type: ignore
        return iterable


def read_jsonl(path: str) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: str, rows: Iterable[Dict[str, object]], mode: str = "w") -> None:
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(path, mode, encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


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


def build_prompt(summary: str, generation: str) -> str:
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


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate summary alignment with GPT-4o mini.")
    ap.add_argument("--input_path", required=True, help="Input JSONL file.")
    ap.add_argument("--output_path", required=True, help="Output JSONL file.")
    ap.add_argument("--summary_field", default="content_pack", help="Field containing the summary.")
    ap.add_argument("--text_field", default="text", help="Field containing the generated text.")
    ap.add_argument("--id_field", default="sample_id", help="Field to use for de-dup in append mode.")
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max_output_tokens", type=int, default=128)
    ap.add_argument("--sleep", type=float, default=0.0, help="Sleep seconds between requests.")
    ap.add_argument("--limit", type=int, default=0, help="If >0, only process first N rows.")
    ap.add_argument("--append", action="store_true", help="Append and skip existing id_field rows.")
    ap.add_argument("--summary_csv", default="", help="Optional CSV to append aggregate results.")
    ap.add_argument("--run_label", default="", help="Optional label for summary CSV row.")
    args = ap.parse_args()

    client = OpenAI()
    rows = read_jsonl(args.input_path)
    if args.limit and args.limit > 0:
        rows = rows[: args.limit]
    if not rows:
        raise ValueError(f"No rows loaded from {args.input_path}")

    existing = load_existing_ids(args.output_path, args.id_field) if args.append else set()
    out_rows: List[Dict[str, object]] = []

    progress = tqdm(total=len(rows), desc="Judging", unit="row")
    for idx, row in enumerate(rows, start=1):
        summary = str(row.get(args.summary_field, "")).strip()
        generation = str(row.get(args.text_field, "")).strip()
        if not summary or not generation:
            progress.update(1)
            continue

        row_id = row.get(args.id_field, idx)
        if args.append and str(row_id) in existing:
            progress.update(1)
            continue

        prompt = build_prompt(summary, generation)

        attempt = 0
        while True:
            try:
                response = client.responses.create(
                    model=args.model,
                    input=prompt,
                    temperature=args.temperature,
                    max_output_tokens=args.max_output_tokens,
                )
                raw = (response.output_text or "").strip()
                parsed = parse_json_response(raw)
                label = str(parsed.get("label", "")).strip().lower()
                score = parsed.get("score")
                if not isinstance(score, int):
                    try:
                        score = int(score)
                    except Exception:
                        raise ValueError("Invalid score from model.")
                if score < 1 or score > 5:
                    raise ValueError("Score out of range.")
                if label not in {"yes", "partial", "no"}:
                    # Fallback to score-based label if model omitted label.
                    if score >= 3:
                        label = "yes"
                    elif score == 2:
                        label = "partial"
                    else:
                        label = "no"
                break
            except Exception:
                attempt += 1
                if attempt > 5:
                    raise
                time.sleep(min(2 ** attempt, 30))

        out_row = dict(row)
        out_row.update(
            {
                "judge_label": label,
                "judge_score": score,
                "judge_model": args.model,
            }
        )
        out_rows.append(out_row)

        if args.sleep:
            time.sleep(args.sleep)
        if len(out_rows) >= 50:
            write_jsonl(args.output_path, out_rows, mode="a" if args.append else "w")
            args.append = True
            out_rows = []
        progress.update(1)

    if out_rows:
        write_jsonl(args.output_path, out_rows, mode="a" if args.append else "w")
    progress.close()

    # Aggregate metrics (mean score + label distribution).
    all_rows = read_jsonl(args.output_path)
    if all_rows:
        scores = [r.get("judge_score", 0) for r in all_rows]
        labels = [str(r.get("judge_label", "")).lower() for r in all_rows]
        count = len(all_rows)
        avg_score = sum(float(s) for s in scores) / count
        label_counts = {
            "yes": sum(1 for l in labels if l == "yes"),
            "partial": sum(1 for l in labels if l == "partial"),
            "no": sum(1 for l in labels if l == "no"),
        }
        aggregate = {
            "count": count,
            "average_score": avg_score,
            "label_counts": label_counts,
            "label_rates": {k: (v / count if count else 0.0) for k, v in label_counts.items()},
            "model": args.model,
        }
        agg_path = os.path.splitext(args.output_path)[0] + "_aggregate.json"
        with open(agg_path, "w", encoding="utf-8") as handle:
            json.dump(aggregate, handle, indent=2)

        if args.summary_csv:
            Path(args.summary_csv).parent.mkdir(parents=True, exist_ok=True)
            write_header = not os.path.exists(args.summary_csv)
            label = args.run_label or Path(args.output_path).stem
            with open(args.summary_csv, "a", encoding="utf-8") as handle:
                if write_header:
                    handle.write("label,count,average_score,yes_rate,partial_rate,no_rate,model\n")
                handle.write(
                    f"{label},{count},{avg_score},"
                    f"{aggregate['label_rates']['yes']},"
                    f"{aggregate['label_rates']['partial']},"
                    f"{aggregate['label_rates']['no']},"
                    f"{args.model}\n"
                )


if __name__ == "__main__":
    main()
