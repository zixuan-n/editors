#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared utilities for BERT LoRA attribution/CPT/deflation experiments."""

from __future__ import annotations

import csv
import json
import math
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from safetensors.torch import load_file
from sklearn import metrics
from torch import nn
from transformers import AutoModelForMaskedLM, AutoModelForSequenceClassification, TrainingArguments

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "configs"))
from finetune_config_loader import load_config


SUFFIX_A = ".lora_A.weight"
SUFFIX_B = ".lora_B.weight"


def load_yaml(path: str, dataset: Optional[str] = None) -> Dict[str, Any]:
    return load_config(path, dataset_variant=dataset)


def section(cfg: Dict[str, Any], name: str) -> Dict[str, Any]:
    value = cfg.get(name, {})
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"Config section `{name}` must be a mapping.")
    return value


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def limit_rows(rows: Sequence[Dict[str, Any]], limit: Optional[int], *, seed: int) -> List[Dict[str, Any]]:
    rows = list(rows)
    if limit is None or limit <= 0 or len(rows) <= limit:
        return rows
    rng = random.Random(seed)
    rng.shuffle(rows)
    return rows[:limit]


def build_label_map(*row_sets: Sequence[Dict[str, Any]], label_field: str = "author") -> Dict[str, int]:
    labels = sorted({str(row[label_field]) for rows in row_sets for row in rows if label_field in row})
    if not labels:
        raise ValueError(f"No labels found with field `{label_field}`.")
    return {label: idx for idx, label in enumerate(labels)}


def save_label_map(label_map: Dict[str, int], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    payload = {"label_to_id": label_map, "id_to_label": {str(v): k for k, v in label_map.items()}}
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)


def load_label_map(path: str) -> Dict[str, int]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    label_map = payload.get("label_to_id", payload)
    if not isinstance(label_map, dict):
        raise ValueError(f"Invalid label map at {path}.")
    return {str(k): int(v) for k, v in label_map.items()}


class JsonlClassificationDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        rows: Sequence[Dict[str, Any]],
        tokenizer,
        label_map: Dict[str, int],
        *,
        text_field: str,
        label_field: str,
        max_seq_length: int,
    ) -> None:
        self.rows = list(rows)
        self.tokenizer = tokenizer
        self.label_map = label_map
        self.text_field = text_field
        self.label_field = label_field
        self.max_seq_length = max_seq_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.rows[idx]
        text = str(row[self.text_field])
        label = self.label_map[str(row[self.label_field])]
        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_seq_length,
            padding=False,
        )
        enc["labels"] = label
        return {key: torch.tensor(value, dtype=torch.long) for key, value in enc.items()}


class JsonlTextDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        rows: Sequence[Dict[str, Any]],
        tokenizer,
        *,
        text_field: str,
        max_seq_length: int,
    ) -> None:
        self.rows = list(rows)
        self.tokenizer = tokenizer
        self.text_field = text_field
        self.max_seq_length = max_seq_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = str(self.rows[idx][self.text_field])
        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_seq_length,
            padding=False,
        )
        return {key: torch.tensor(value, dtype=torch.long) for key, value in enc.items()}


def compute_classification_metrics(eval_pred) -> Dict[str, float]:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": float(metrics.accuracy_score(labels, preds)),
        "macro_f1": float(metrics.f1_score(labels, preds, average="macro", zero_division=0)),
        "balanced_accuracy": float(metrics.balanced_accuracy_score(labels, preds)),
    }


def write_metrics(path: str, metrics_dict: Dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(metrics_dict, handle, indent=2, sort_keys=True)


def filtered_training_args(cfg: Dict[str, Any]) -> Dict[str, Any]:
    kwargs = dict(cfg)
    allowed = set(TrainingArguments.__dataclass_fields__.keys())
    return {key: value for key, value in kwargs.items() if key in allowed and value is not None}


def normalize_report_to(value: Any) -> Any:
    if value is None:
        return "none"
    return value


def ensure_parent(path: str) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)


def orthonormal_col_basis(mat: torch.Tensor, tol: float, *, method: str) -> Optional[torch.Tensor]:
    mat = mat.float()
    if mat.numel() == 0:
        return None
    if method == "qr":
        q, r = torch.linalg.qr(mat, mode="reduced")
        keep = r.diag().abs() > tol
        if not bool(keep.any()):
            return None
        return q[:, keep].contiguous()
    if method == "svd":
        u, s, _ = torch.linalg.svd(mat, full_matrices=False)
        keep = s > tol
        if not bool(keep.any()):
            return None
        return u[:, keep].contiguous()
    raise ValueError(f"Unknown basis solver {method!r}; expected 'qr' or 'svd'.")


def orthonormal_columns(rows_or_cols: torch.Tensor, tol: float, *, method: str) -> Optional[torch.Tensor]:
    if rows_or_cols.ndim != 2:
        raise ValueError(f"Expected a 2D tensor, got shape {tuple(rows_or_cols.shape)}.")
    return orthonormal_col_basis(rows_or_cols.t().contiguous(), tol, method=method)


def orient_lora_weights(A: torch.Tensor, B: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, int]:
    if A.ndim != 2 or B.ndim != 2:
        raise ValueError(f"Expected 2D LoRA weights, got A{tuple(A.shape)} B{tuple(B.shape)}")
    shared = sorted(set(A.shape).intersection(B.shape))
    if not shared:
        raise ValueError(f"Cannot infer LoRA rank from shapes A{tuple(A.shape)} B{tuple(B.shape)}")
    rank = min(shared)
    if A.shape[0] == rank:
        A_r_in = A
    elif A.shape[1] == rank:
        A_r_in = A.t()
    else:
        raise ValueError(f"Cannot orient LoRA A with inferred rank r={rank}: A{tuple(A.shape)}")
    if B.shape[1] == rank:
        B_out_r = B
    elif B.shape[0] == rank:
        B_out_r = B.t()
    else:
        raise ValueError(f"Cannot orient LoRA B with inferred rank r={rank}: B{tuple(B.shape)}")
    return A_r_in.contiguous(), B_out_r.contiguous(), rank


def build_stage1_bases(
    adapter_dir: str,
    *,
    include: Optional[Sequence[str]],
    basis_tol: float,
    basis_type: str,
    basis_solver: str,
) -> Dict[str, torch.Tensor]:
    state_path = Path(adapter_dir) / "adapter_model.safetensors"
    if not state_path.exists():
        raise FileNotFoundError(f"Missing Stage-1 adapter weights: {state_path}")
    state = load_file(str(state_path), device="cpu")
    bases: Dict[str, torch.Tensor] = {}
    for key, tensor in state.items():
        if not key.endswith(SUFFIX_A):
            continue
        name = key[: -len(SUFFIX_A)]
        if include and not any(part in name for part in include):
            continue
        if basis_type == "lora_A":
            U = orthonormal_columns(tensor, basis_tol, method=basis_solver)
        elif basis_type == "deltaW":
            B = state.get(f"{name}{SUFFIX_B}")
            if B is None:
                continue
            A_r_in, B_out_r, _ = orient_lora_weights(tensor, B)
            Q_r = orthonormal_col_basis(B_out_r.t(), basis_tol, method=basis_solver)
            if Q_r is None:
                continue
            A_eff = (Q_r.t() @ A_r_in.float()).contiguous()
            U = orthonormal_columns(A_eff, basis_tol, method=basis_solver)
        else:
            raise ValueError(f"Unknown basis_type={basis_type!r}; expected 'lora_A' or 'deltaW'.")
        if U is not None:
            bases[name] = U
    if not bases:
        raise RuntimeError(
            f"No Stage-1 bases found under {adapter_dir}. Check adapter path and include filters."
        )
    return bases


def load_sequence_classifier_encoder_state(
    checkpoint_path: str,
    *,
    num_labels: int,
    model_name_or_path: str = "bert-base-cased",
) -> Dict[str, torch.Tensor]:
    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint_path,
        num_labels=num_labels,
        local_files_only=os.path.isdir(checkpoint_path),
    )
    return {key: value.detach().cpu() for key, value in model.bert.state_dict().items()}


def load_mlm_encoder_state(checkpoint_path: str) -> Dict[str, torch.Tensor]:
    model = AutoModelForMaskedLM.from_pretrained(
        checkpoint_path,
        local_files_only=os.path.isdir(checkpoint_path),
    )
    return {key: value.detach().cpu() for key, value in model.bert.state_dict().items()}


def build_full_delta_bases(
    stage0_encoder_state: Dict[str, torch.Tensor],
    stage1_encoder_state: Dict[str, torch.Tensor],
    *,
    include: Optional[Sequence[str]],
    basis_tol: float,
    basis_solver: str,
    max_rank: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    bases: Dict[str, torch.Tensor] = {}
    for key, w1 in stage1_encoder_state.items():
        if not key.endswith(".weight") or w1.ndim != 2:
            continue
        module_name = f"bert.{key[: -len('.weight')]}"
        if include and not any(part in module_name for part in include):
            continue
        w0 = stage0_encoder_state.get(key)
        if w0 is None or w0.shape != w1.shape:
            continue
        delta = w1.float() - w0.float()
        U = orthonormal_columns(delta, basis_tol, method=basis_solver)
        if U is not None:
            if max_rank is not None and max_rank > 0:
                U = U[:, : min(max_rank, U.shape[1])].contiguous()
            bases[module_name] = U
    if not bases:
        raise RuntimeError("No full-BERT delta bases were built. Check checkpoints and include filters.")
    return bases


@dataclass
class DeflationNorms:
    calls: int = 0
    x_sq: float = 0.0
    proj_sq: float = 0.0


class DeflationStats:
    def __init__(self, *, sample_every: int = 1) -> None:
        self.sample_every = max(1, int(sample_every))
        self.total: Dict[str, DeflationNorms] = {}

    def record(self, name: str, *, x_sq: float, proj_sq: float) -> None:
        item = self.total.setdefault(name, DeflationNorms())
        item.calls += 1
        if item.calls % self.sample_every != 0:
            return
        item.x_sq += float(x_sq)
        item.proj_sq += float(proj_sq)

    def rows(self) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for name, item in sorted(self.total.items()):
            frac = item.proj_sq / item.x_sq if item.x_sq > 0 else 0.0
            out.append(
                {
                    "module": name,
                    "calls": item.calls,
                    "sampled_x_sq": item.x_sq,
                    "sampled_proj_sq": item.proj_sq,
                    "removed_energy_frac": frac,
                }
            )
        return out

    def write_csv(self, path: str) -> None:
        ensure_parent(path)
        rows = self.rows()
        fields = ["module", "calls", "sampled_x_sq", "sampled_proj_sq", "removed_energy_frac"]
        with open(path, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)


def register_deflation_hooks(
    model: nn.Module,
    bases: Dict[str, torch.Tensor],
    *,
    stats: Optional[DeflationStats] = None,
) -> List[Any]:
    modules = dict(model.named_modules())
    handles: List[Any] = []
    skipped = 0

    def infer_in_features(module: nn.Module) -> Optional[int]:
        if hasattr(module, "in_features"):
            return int(module.in_features)  # type: ignore[arg-type]
        weight = getattr(module, "weight", None)
        if weight is not None and getattr(weight, "ndim", 0) == 2:
            return int(weight.shape[1])
        return None

    def align_basis(U: torch.Tensor, dim: Optional[int]) -> Optional[torch.Tensor]:
        if dim is None:
            return U
        if U.shape[0] == dim:
            return U
        if U.ndim == 2 and U.shape[1] == dim:
            return U.t().contiguous()
        return None

    def make_hook(U_cpu: torch.Tensor, name: str):
        def pre_hook(_module, inputs):
            x = inputs[0]
            last_dim = x.shape[-1]
            if last_dim != U_cpu.shape[0]:
                return None
            orig_shape = x.shape
            x2 = x.reshape(-1, last_dim)
            U = U_cpu.to(device=x2.device, dtype=x2.dtype, non_blocking=True)
            proj = (x2 @ U) @ U.t()
            if stats is not None:
                stats.record(
                    name,
                    x_sq=x2.float().pow(2).sum().item(),
                    proj_sq=proj.float().pow(2).sum().item(),
                )
            return (x2.sub(proj).reshape(orig_shape),)

        return pre_hook

    for basis_name, basis in bases.items():
        target: Optional[nn.Module] = None
        candidates = [basis_name, f"{basis_name}.base_layer"]
        candidates.extend(
            name for name in modules if name.endswith(basis_name) or name.endswith(f"{basis_name}.base_layer")
        )
        for candidate in candidates:
            module = modules.get(candidate)
            if isinstance(module, nn.Linear):
                target = module
                break
            if module is not None and hasattr(module, "base_layer") and isinstance(module.base_layer, nn.Linear):
                target = module.base_layer
                break
        if target is None:
            skipped += 1
            continue
        U = align_basis(basis, infer_in_features(target))
        if U is None:
            skipped += 1
            continue
        handles.append(target.register_forward_pre_hook(make_hook(U, basis_name)))

    print(f"[Deflation] requested={len(bases)} | installed={len(handles)} | skipped={skipped}")
    if not handles:
        raise RuntimeError("No deflation hooks were installed. Check Stage-1 adapter/module names.")
    return handles


def mlm_metrics(eval_pred) -> Dict[str, float]:
    # Trainer returns eval_loss separately; this hook is intentionally empty.
    return {}


def add_perplexity(metrics_dict: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(metrics_dict)
    loss = out.get("eval_loss")
    if loss is not None:
        try:
            out["perplexity"] = float(math.exp(float(loss)))
        except OverflowError:
            out["perplexity"] = float("inf")
    return out
