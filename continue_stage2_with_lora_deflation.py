#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Continue Stage-2 style LoRA training while dynamically deflating inputs onto the
orthogonal complement of the Stage-1 (atomic facts) adapter.

"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

# Force a single visible GPU; avoids bitsandbytes offload to CPU/disk.
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "1")

import torch
import matplotlib.pyplot as plt
from safetensors.torch import load_file
from torch import nn
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainerCallback,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from finetune_config_loader import load_stage2_config


# -------------------- utils: IO --------------------
def read_jsonl(path: str) -> List[Dict[str, str]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def split_train_val(rows, val_ratio=0.05, seed=42):
    rng = random.Random(seed)
    rng.shuffle(rows)
    k = max(1, int(len(rows) * val_ratio))
    return rows[k:], rows[:k]


# -------------------- dataset --------------------
def build_prompt(tok, text: str, *, system_prompt: Optional[str], prompt_template: Optional[str]) -> str:
    # Previous implementation (kept for reference):
    # prompt = f"Passage:\\n{text}\\nAuthor:"
    # return prompt

    if not prompt_template:
        return f"Passage:\n{text}\nAuthor:"

    template_norm = prompt_template.strip().lower()
    if template_norm in {"auto", "chat_template"}:
        if not hasattr(tok, "apply_chat_template") or not getattr(tok, "chat_template", None):
            raise ValueError("prompt.template=auto requires a tokenizer with a chat_template.")
        user_content = f"Passage:\n{text}\nAuthor:"
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_content})
        try:
            return tok.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception as exc:
            msg = str(exc).lower()
            if "system role not supported" in msg or "system role" in msg:
                merged = f"{system_prompt}\n\n{user_content}" if system_prompt else user_content
                messages = [{"role": "user", "content": merged}]
                return tok.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            raise

    return prompt_template.format(
        system_prompt=system_prompt or "",
        passage=text,
        author_list="",
    )


def make_example(tok, text: str, author: str, *, system_prompt: Optional[str], prompt_template: Optional[str], answer_prefix: str):
    prompt = build_prompt(tok, text, system_prompt=system_prompt, prompt_template=prompt_template)
    target = (answer_prefix or " ") + author
    return prompt, target


class AuthorGenDataset(Dataset):
    def __init__(
        self,
        rows: List[Dict[str, str]],
        tok,
        max_len: int,
        *,
        system_prompt: Optional[str] = None,
        prompt_template: Optional[str] = None,
        answer_prefix: str = " ",
    ):
        self.rows, self.tok, self.max_len = rows, tok, max_len
        self.system_prompt = system_prompt
        self.prompt_template = prompt_template
        self.answer_prefix = answer_prefix

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        prompt, target = make_example(
            self.tok,
            r["text"],
            r["author"],
            system_prompt=self.system_prompt,
            prompt_template=self.prompt_template,
            answer_prefix=self.answer_prefix,
        )
        enc_p = self.tok(prompt, add_special_tokens=False)
        enc_t = self.tok(target, add_special_tokens=False)
        input_ids = enc_p["input_ids"] + enc_t["input_ids"]
        labels = [-100] * len(enc_p["input_ids"]) + enc_t["input_ids"]
        if len(input_ids) > self.max_len:
            keep = self.max_len - len(enc_t["input_ids"])
            input_ids = input_ids[-self.max_len :]
            labels = ([-100] * keep) + enc_t["input_ids"]
        attn = [1] * len(input_ids)
        return {
            "input_ids": torch.tensor(input_ids),
            "labels": torch.tensor(labels),
            "attention_mask": torch.tensor(attn),
        }


@dataclass
class PadCollator:
    pad_id: int

    def __call__(self, feats: List[Dict[str, Any]]):
        keys = ["input_ids", "labels", "attention_mask"]
        max_len = max(len(f["input_ids"]) for f in feats)
        batch = {}
        for k in keys:
            vals = []
            for f in feats:
                x = f[k]
                pad_val = self.pad_id if k != "labels" else -100
                if len(x) < max_len:
                    x = torch.nn.functional.pad(x, (0, max_len - len(x)), value=pad_val)
                vals.append(x)
            batch[k] = torch.stack(vals, dim=0)
        return batch


# -------------------- deflation helpers --------------------
SUFFIX_A = ".lora_A.weight"
SUFFIX_B = ".lora_B.weight"


@dataclass
class _DeflationNorms:
    calls: int = 0
    x_sq: float = 0.0
    proj_sq: float = 0.0


class DeflationStats:
    def __init__(self, *, sample_every: int = 1):
        self.sample_every = max(1, int(sample_every))
        self.total: Dict[str, _DeflationNorms] = {}
        self.window: Dict[str, _DeflationNorms] = {}

    def _get(self, table: Dict[str, _DeflationNorms], name: str) -> _DeflationNorms:
        stats = table.get(name)
        if stats is None:
            stats = _DeflationNorms()
            table[name] = stats
        return stats

    def record(self, name: str, *, x_sq: float, proj_sq: float) -> None:
        total = self._get(self.total, name)
        total.calls += 1
        if total.calls % self.sample_every != 0:
            return
        total.x_sq += float(x_sq)
        total.proj_sq += float(proj_sq)

        window = self._get(self.window, name)
        window.calls += 1
        window.x_sq += float(x_sq)
        window.proj_sq += float(proj_sq)

    def reset_window(self) -> None:
        self.window = {}

    def summary_rows(self, *, use_window: bool = True) -> List[Dict[str, float]]:
        table = self.window if use_window else self.total
        rows: List[Dict[str, float]] = []
        for name, stats in table.items():
            if stats.x_sq <= 0:
                continue
            frac_energy = stats.proj_sq / max(1e-12, stats.x_sq)
            rows.append(
                {
                    "module": name,
                    "calls": float(stats.calls),
                    "removed_energy_frac": float(frac_energy),
                    "removed_norm_frac": float(frac_energy**0.5),
                    "x_norm": float(stats.x_sq**0.5),
                    "proj_norm": float(stats.proj_sq**0.5),
                }
            )
        rows.sort(key=lambda r: r["removed_energy_frac"], reverse=True)
        return rows

    def write_csv(self, path: Path) -> None:
        rows = self.summary_rows(use_window=False)
        if not rows:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["module", "calls", "removed_energy_frac", "removed_norm_frac", "x_norm", "proj_norm"],
            )
            writer.writeheader()
            writer.writerows(rows)


class DeflationLoggingCallback(TrainerCallback):
    def __init__(self, stats: DeflationStats, *, topk: int = 5, csv_path: Optional[str] = None):
        super().__init__()
        self.stats = stats
        self.topk = int(topk)
        self.csv_path = csv_path

    def on_log(self, args, state, control, logs=None, **kwargs):
        if self.topk > 0:
            rows = self.stats.summary_rows(use_window=True)
            if rows:
                msg = " | ".join(
                    f"{r['module'].split('.')[-1]}:{r['removed_energy_frac']:.3f}"
                    for r in rows[: self.topk]
                )
                print(f"[Deflation][removed_energy_frac] {msg}")
        self.stats.reset_window()

    def on_train_end(self, args, state, control, **kwargs):
        if self.csv_path:
            self.stats.write_csv(Path(self.csv_path))
            print(f"[Deflation] wrote summary CSV to {self.csv_path}")


def orthonormal_columns(matrix: torch.Tensor, tol: float) -> Optional[torch.Tensor]:
    if matrix.numel() == 0:
        return None
    A = matrix.float()
    Q, R = torch.linalg.qr(A.t(), mode="reduced")
    if R.numel() == 0:
        return None
    diag = torch.abs(torch.diagonal(R))
    rank = int((diag > tol).sum().item())
    if rank == 0:
        return None
    return Q[:, :rank].contiguous()


def orthonormal_col_basis(matrix: torch.Tensor, tol: float) -> Optional[torch.Tensor]:
    """Returns orthonormal columns spanning colspace(matrix). Output lives in R^{matrix.shape[0]}."""
    if matrix.numel() == 0:
        return None
    M = matrix.float()
    Q, R = torch.linalg.qr(M, mode="reduced")
    if R.numel() == 0:
        return None
    diag = torch.abs(torch.diagonal(R))
    rank = int((diag > tol).sum().item())
    if rank == 0:
        return None
    return Q[:, :rank].contiguous()


def orient_lora_weights(A: torch.Tensor, B: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Return (A_r_in, B_out_r, r) using shape-based inference."""
    if A.ndim != 2 or B.ndim != 2:
        raise ValueError(f"Expected 2D LoRA weights, got A{tuple(A.shape)} B{tuple(B.shape)}")
    a0, a1 = A.shape
    b0, b1 = B.shape

    shared = sorted(set((a0, a1)).intersection((b0, b1)))
    if not shared:
        raise ValueError(f"Cannot infer LoRA rank from shapes A{tuple(A.shape)} B{tuple(B.shape)}")
    r = min(shared)

    if A.shape[0] == r:
        A_r_in = A
    elif A.shape[1] == r:
        A_r_in = A.t()
    else:
        raise ValueError(f"Cannot orient LoRA A with inferred rank r={r}: A{tuple(A.shape)}")

    if B.shape[1] == r:
        B_out_r = B
    elif B.shape[0] == r:
        B_out_r = B.t()
    else:
        raise ValueError(f"Cannot orient LoRA B with inferred rank r={r}: B{tuple(B.shape)}")

    return A_r_in.contiguous(), B_out_r.contiguous(), r


def build_stage1_bases(
    adapter_dir: Path,
    *,
    include: Optional[List[str]],
    basis_tol: float,
    basis_type: str = "lora_A",
) -> Dict[str, torch.Tensor]:
    state = load_file(str(adapter_dir / "adapter_model.safetensors"), device="cpu")
    bases: Dict[str, torch.Tensor] = {}
    for key, tensor in state.items():
        if not key.endswith(SUFFIX_A):
            continue
        name = key[: -len(SUFFIX_A)]
        if include and not any(p in name for p in include):
            continue
        if basis_type == "lora_A":
            U = orthonormal_columns(tensor, basis_tol)
            if U is not None:
                bases[name] = U
            continue
        if basis_type == "deltaW":
            B = state.get(f"{name}{SUFFIX_B}")
            if B is None:
                continue
            A_r_in, B_out_r, _ = orient_lora_weights(tensor, B)
            # Row space(deltaW) = Row space(B@A). Remove null(B) directions in rank-space first.
            Q_r = orthonormal_col_basis(B_out_r.t(), basis_tol)  # (r, k)
            if Q_r is None:
                continue
            A_eff = (Q_r.t() @ A_r_in.float()).contiguous()  # (k, in)
            U_in = orthonormal_columns(A_eff, basis_tol)  # (in, k2)
            if U_in is not None:
                bases[name] = U_in
            continue
        raise ValueError(f"Unknown basis_type={basis_type!r} (expected 'lora_A' or 'deltaW').")
    if not bases:
        raise RuntimeError(
            f"No Stage-1 bases found under {adapter_dir}. "
            "Ensure the adapter exists and --include filters are not too strict."
        )
    return bases


def register_deflation_hooks_smart(
    model,
    bases: Dict[str, torch.Tensor],
    *,
    stats: Optional[DeflationStats] = None,
):
    def _infer_in_features(mod: nn.Module) -> Optional[int]:
        if hasattr(mod, "in_features"):
            return int(mod.in_features)  # type: ignore[arg-type]
        if hasattr(mod, "weight") and getattr(mod, "weight") is not None:
            w = mod.weight
            if w.ndim == 2:
                return int(w.shape[1])
        return None

    def _align_basis(U: torch.Tensor, target_dim: Optional[int]) -> Optional[torch.Tensor]:
        if target_dim is None:
            return U
        if U.shape[0] == target_dim:
            return U
        # fan_in_fan_out adapters store A with out_features first; transpose in that case
        if U.ndim == 2 and U.shape[1] == target_dim:
            return U.t().contiguous()
        return None

    handles, installed, skipped = [], 0, 0
    mods = dict(model.named_modules())

    warn_once = set()

    def make_pre(U_cpu_fp32: torch.Tensor, name: str):
        def pre_hook(mod, inputs):
            x = inputs[0]
            orig = x.shape
            last_dim = x.shape[-1]
            if last_dim != U_cpu_fp32.shape[0]:
                key = (name, last_dim)
                if key not in warn_once:
                    print(
                        f"[Deflation][skip-forward] {name}: input dim {last_dim} "
                        f"!= basis dim {U_cpu_fp32.shape[0]}; passing through."
                    )
                    warn_once.add(key)
                return None
            x2 = x.reshape(-1, last_dim)
            U = U_cpu_fp32.to(x2.device, x2.dtype, non_blocking=True)
            proj = (x2 @ U) @ U.t()
            x_perp = x2 - proj
            if stats is not None:
                x_sq = x2.float().pow(2).sum().item()
                proj_sq = proj.float().pow(2).sum().item()
                stats.record(name, x_sq=x_sq, proj_sq=proj_sq)
            return (x_perp.reshape(orig),)

        return pre_hook

    for name, U in bases.items():
        target = None
        mod = mods.get(name)
        if mod is not None:
            if isinstance(mod, nn.Linear):
                target = mod
            elif hasattr(mod, "base_layer") and isinstance(mod.base_layer, nn.Linear):
                target = mod.base_layer
        if target is None:
            mod2 = mods.get(name + ".base_layer")
            if isinstance(mod2, nn.Linear):
                target = mod2
        if target is None:
            matches = [
                n
                for n in mods
                if n.endswith(name) or n.endswith(name + ".base_layer")
            ]
            for n in matches:
                m = mods[n]
                if isinstance(m, nn.Linear):
                    target = m
                    break
                if hasattr(m, "base_layer") and isinstance(m.base_layer, nn.Linear):
                    target = m.base_layer
                    break
        if target is not None:
            target_dim = _infer_in_features(target)
            U_aligned = _align_basis(U, target_dim)
            if U_aligned is None:
                print(
                    f"[Deflation][skip] {name}: basis dim {tuple(U.shape)} "
                    f"not compatible with module dim {target_dim}."
                )
                skipped += 1
                continue
            h = target.register_forward_pre_hook(make_pre(U_aligned, name))
            handles.append(h)
            installed += 1

    print(f"[Deflation] requested={len(bases)} | installed={installed} | skipped={skipped}")
    if installed == 0:
        raise RuntimeError(
            "Deflation hooks could not be installed. "
            "Double-check module names, LoRA target modules, and that Stage-1 adapters "
            "match the current base model's hidden size."
        )
    return handles


def apply_overrides(cfg: SimpleNamespace, args: argparse.Namespace) -> SimpleNamespace:
    for field in (
        "model_id",
        "stage1_adapter",
        "stage2_adapter",
        "train_jsonl",
        "dev_jsonl",
        "out_dir",
        "max_len",
        "seed",
        "batch",
        "grad_accum",
        "lr",
        "epochs",
        "save_u_sem",
        "basis_type",
        "deflation_sample_every",
        "deflation_log_topk",
        "deflation_log_csv",
    ):
        val = getattr(args, field, None)
        if val is not None:
            setattr(cfg, field, val)
    if getattr(args, "use_4bit", None):
        cfg.use_4bit = True
    if getattr(args, "include_module", None):
        cfg.include_module = args.include_module
    if getattr(args, "basis_tol", None) is not None:
        cfg.basis_tol = args.basis_tol
    return cfg


# -------------------- main --------------------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Continue Stage-2 style training with on-the-fly deflation derived from a Stage-1 adapter."
    )
    ap.add_argument("--config", type=str, default="configs/stage2_style_sft.yaml", help="Optional YAML config path.")
    ap.add_argument("--model_id", default=None)
    ap.add_argument("--stage1_adapter", default=None, help="Path to the atomic-facts LoRA.")
    ap.add_argument("--stage2_adapter", default=None, help="Path to the style LoRA checkpoint to continue training.")
    ap.add_argument("--train_jsonl", default=None)
    ap.add_argument("--dev_jsonl", default=None)
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--max_len", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--use_4bit", action="store_true")
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--grad_accum", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--epochs", type=float, default=None)
    ap.add_argument(
        "--include_module",
        action="append",
        default=None,
        help="Optional substring filter applied to Stage-1 module names when building bases.",
    )
    ap.add_argument(
        "--basis_tol",
        type=float,
        default=None,
        help="Tolerance (in singular-value units) used to decide the Stage-1 basis rank.",
    )
    ap.add_argument(
        "--basis_type",
        type=str,
        default=None,
        choices=("lora_A", "deltaW"),
        help="How to build the input-subspace basis from the Stage-1 adapter (default: lora_A).",
    )
    ap.add_argument(
        "--save_u_sem",
        type=str,
        default=None,
        help="Optional path to torch.save the computed bases (u_sem format) for reuse.",
    )
    ap.add_argument(
        "--deflation_sample_every",
        type=int,
        default=None,
        help="Only record deflation removal norms every N hook calls per module (default: 1).",
    )
    ap.add_argument(
        "--deflation_log_topk",
        type=int,
        default=None,
        help="How many modules to print in per-log deflation summaries (default: 5).",
    )
    ap.add_argument(
        "--deflation_log_csv",
        type=str,
        default=None,
        help="Optional CSV output path for cumulative deflation stats.",
    )
    return ap.parse_args()


def main():
    args = parse_args()
    cfg = load_stage2_config(args.config)
    apply_overrides(cfg, args)
    missing = [name for name in ("train_jsonl",) if not getattr(cfg, name)]
    if missing:
        raise ValueError(f"Missing required configuration values: {', '.join(missing)}")
    if not cfg.stage2_adapter and not cfg.init_stage2_from_base:
        raise ValueError("Missing stage2_adapter; set model.init_stage2_from_base to start from base.")

    random.seed(cfg.seed)

    tok = AutoTokenizer.from_pretrained(cfg.model_id, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    quant = None
    if cfg.use_4bit:
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    base = AutoModelForCausalLM.from_pretrained(
        cfg.model_id,
        device_map={"": 0},
        torch_dtype=None if cfg.use_4bit else torch.bfloat16,
        quantization_config=quant,
        low_cpu_mem_usage=True,
    )
    base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=True)
    base.config.use_cache = False

    if cfg.stage2_adapter:
        model = PeftModel.from_pretrained(base, cfg.stage2_adapter, is_trainable=True)
    else:
        if not cfg.lora_target_modules:
            raise ValueError("lora.target_modules is required when init_stage2_from_base is true.")
        lora_cfg = LoraConfig(
            task_type="CAUSAL_LM",
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            target_modules=cfg.lora_target_modules,
            bias=cfg.lora_bias,
        )
        model = get_peft_model(base, lora_cfg)
    model.enable_input_require_grads()

    bases_src = cfg.stage1_adapter or cfg.stage2_adapter
    if not bases_src:
        raise ValueError("Need a path to derive deflation bases (stage1_adapter or stage2_adapter).")
    bases_dir = Path(bases_src)
    bases = build_stage1_bases(
        bases_dir,
        include=cfg.include_module,
        basis_tol=cfg.basis_tol,
        basis_type=getattr(cfg, "basis_type", "lora_A"),
    )
    if cfg.save_u_sem:
        save_path = Path(cfg.save_u_sem)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(bases, save_path)
        print(f"[Deflation] Saved bases to {save_path}")
    topk = getattr(cfg, "deflation_log_topk", 0)
    csv_path = getattr(cfg, "deflation_log_csv", None)
    defl_stats = None
    if (topk and int(topk) > 0) or csv_path:
        defl_stats = DeflationStats(sample_every=getattr(cfg, "deflation_sample_every", 1))
    defl_handles = register_deflation_hooks_smart(model, bases, stats=defl_stats)

    rows_tr = read_jsonl(cfg.train_jsonl)
    if cfg.dev_jsonl and os.path.exists(cfg.dev_jsonl):
        rows_va = read_jsonl(cfg.dev_jsonl)
    else:
        rows_tr, rows_va = split_train_val(rows_tr, val_ratio=0.05, seed=cfg.seed)

    ds_tr = AuthorGenDataset(
        rows_tr,
        tok,
        cfg.max_len,
        system_prompt=getattr(cfg, "system_prompt", None),
        prompt_template=getattr(cfg, "prompt_template", None),
        answer_prefix=getattr(cfg, "answer_prefix", " "),
    )
    ds_va = AuthorGenDataset(
        rows_va,
        tok,
        cfg.max_len,
        system_prompt=getattr(cfg, "system_prompt", None),
        prompt_template=getattr(cfg, "prompt_template", None),
        answer_prefix=getattr(cfg, "answer_prefix", " "),
    )
    collate = PadCollator(tok.pad_token_id)

    targs = TrainingArguments(
        output_dir=cfg.out_dir,
        per_device_train_batch_size=cfg.batch,
        per_device_eval_batch_size=cfg.eval_batch,
        gradient_accumulation_steps=cfg.grad_accum,
        learning_rate=cfg.lr,
        num_train_epochs=cfg.epochs,
        lr_scheduler_type=cfg.lr_scheduler_type,
        warmup_ratio=cfg.warmup_ratio,
        weight_decay=cfg.weight_decay,
        logging_steps=cfg.logging_steps,
        eval_strategy=cfg.eval_strategy,
        eval_steps=cfg.eval_steps,
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        bf16=cfg.bf16,
        gradient_checkpointing=True,
        group_by_length=True,
        optim="paged_adamw_8bit" if cfg.use_4bit else "adamw_torch",
        report_to=cfg.report_to,
        run_name=cfg.run_name,
        label_smoothing_factor=0.0,
    )

    callbacks = []
    if (topk and int(topk) > 0) or csv_path:
        if defl_stats is None:
            raise RuntimeError("Internal error: deflation stats missing but logging requested.")
        callbacks.append(DeflationLoggingCallback(defl_stats, topk=int(topk or 0), csv_path=csv_path))
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=ds_tr,
        eval_dataset=ds_va,
        data_collator=collate,
        tokenizer=tok,
        callbacks=callbacks,
    )

    trainer.train()

    def save_loss_history(trainer_obj, csv_path=None, plot_path=None):
        records = []
        for log in trainer_obj.state.log_history:
            loss_val = log.get("loss", log.get("train_loss"))
            if loss_val is None:
                continue
            epoch = log.get("epoch")
            step = log.get("step")
            records.append({"step": step, "epoch": epoch, "loss": float(loss_val)})
        if csv_path and records:
            Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["step", "epoch", "loss"])
                writer.writeheader()
                writer.writerows(records)
            print(f"[Loss] wrote CSV to {csv_path}")
        if plot_path and records:
            Path(plot_path).parent.mkdir(parents=True, exist_ok=True)
            plt.figure()
            steps = [r["step"] for r in records]
            losses = [r["loss"] for r in records]
            plt.plot(steps, losses, label="train loss")
            plt.xlabel("Step")
            plt.ylabel("Loss")
            plt.title("Training loss")
            plt.legend()
            plt.tight_layout()
            plt.savefig(plot_path)
            plt.close()
            print(f"[Loss] wrote plot to {plot_path}")

    save_loss_history(trainer, cfg.loss_log_path, cfg.loss_plot_path)

    save_dir = os.path.join(cfg.out_dir, "adapter_deflated_stage2")
    trainer.model.save_pretrained(save_dir)
    tok.save_pretrained(save_dir)
    print("Saved continued (deflated) adapter to", save_dir)

    metrics_with = trainer.evaluate()
    for h in defl_handles:
        h.remove()
    metrics_without = trainer.evaluate()
    print("Eval WITH deflation :", metrics_with)
    print("Eval WITHOUT deflation:", metrics_without)


if __name__ == "__main__":
    main()
