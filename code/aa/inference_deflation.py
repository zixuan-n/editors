from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from safetensors.torch import load_file
from torch import nn


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

    def record(self, name: str, *, x_sq: float, proj_sq: float) -> None:
        stats = self.total.get(name)
        if stats is None:
            stats = _DeflationNorms()
            self.total[name] = stats
        stats.calls += 1
        if stats.calls % self.sample_every != 0:
            return
        stats.x_sq += float(x_sq)
        stats.proj_sq += float(proj_sq)

    def summary_rows(self) -> List[Dict[str, float | str]]:
        rows: List[Dict[str, float | str]] = []
        for name, stats in self.total.items():
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
        rows.sort(key=lambda row: float(row["removed_energy_frac"]), reverse=True)
        return rows

    def write_csv(self, path: str) -> None:
        rows = self.summary_rows()
        if not rows:
            return
        out_dir = os.path.dirname(path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["module", "calls", "removed_energy_frac", "removed_norm_frac", "x_norm", "proj_norm"],
            )
            writer.writeheader()
            writer.writerows(rows)


@dataclass
class InferenceDeflation:
    handles: List[torch.utils.hooks.RemovableHandle]
    stats: Optional[DeflationStats] = None
    stats_csv: Optional[str] = None

    def close(self) -> None:
        if self.stats is not None and self.stats_csv:
            self.stats.write_csv(self.stats_csv)
            print(f"[InferenceDeflation] wrote stats CSV to {self.stats_csv}")
        for handle in self.handles:
            handle.remove()


def list_lora_modules(state: Dict[str, torch.Tensor]) -> List[str]:
    modules = []
    for key in state.keys():
        if key.endswith(SUFFIX_A):
            modules.append(key[: -len(SUFFIX_A)])
    return modules


def filter_modules(modules: List[str], patterns: Optional[List[str]]) -> List[str]:
    if not patterns:
        return sorted(modules)
    return sorted(name for name in modules if any(pattern in name for pattern in patterns))


def orth_basis(matrix: torch.Tensor, tol: float, method: str = "qr") -> Optional[torch.Tensor]:
    if matrix.numel() == 0:
        return None
    mat = matrix.float()
    if method == "svd":
        u, s, _ = torch.linalg.svd(mat.t(), full_matrices=False)
        rank = int((s > tol).sum().item()) if s.numel() else 0
        return u[:, :rank].contiguous() if rank > 0 else None
    if method != "qr":
        raise ValueError(f"Unknown orthonormalization method: {method!r} (expected 'qr' or 'svd').")
    q, r = torch.linalg.qr(mat.t(), mode="reduced")
    diag = torch.abs(torch.diagonal(r)) if r.numel() else torch.empty(0)
    rank = int((diag > tol).sum().item()) if diag.numel() else 0
    return q[:, :rank].contiguous() if rank > 0 else None


def orthonormal_col_basis(matrix: torch.Tensor, tol: float, method: str = "qr") -> Optional[torch.Tensor]:
    if matrix.numel() == 0:
        return None
    mat = matrix.float()
    if method == "svd":
        u, s, _ = torch.linalg.svd(mat, full_matrices=False)
        rank = int((s > tol).sum().item()) if s.numel() else 0
        return u[:, :rank].contiguous() if rank > 0 else None
    if method != "qr":
        raise ValueError(f"Unknown orthonormalization method: {method!r} (expected 'qr' or 'svd').")
    q, r = torch.linalg.qr(mat, mode="reduced")
    diag = torch.abs(torch.diagonal(r)) if r.numel() else torch.empty(0)
    rank = int((diag > tol).sum().item()) if diag.numel() else 0
    return q[:, :rank].contiguous() if rank > 0 else None


def orient_lora_weights(A: torch.Tensor, B: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, int]:
    if A.ndim != 2 or B.ndim != 2:
        raise ValueError(f"Expected 2D LoRA weights, got A{tuple(A.shape)} B{tuple(B.shape)}")
    shared = sorted(set(A.shape).intersection(B.shape))
    if not shared:
        raise ValueError(f"Cannot infer LoRA rank from shapes A{tuple(A.shape)} B{tuple(B.shape)}")
    rank = min(shared)

    if A.shape[0] == rank:
        a_r_in = A
    elif A.shape[1] == rank:
        a_r_in = A.t()
    else:
        raise ValueError(f"Cannot orient LoRA A with inferred rank r={rank}: A{tuple(A.shape)}")

    if B.shape[1] == rank:
        b_out_r = B
    elif B.shape[0] == rank:
        b_out_r = B.t()
    else:
        raise ValueError(f"Cannot orient LoRA B with inferred rank r={rank}: B{tuple(B.shape)}")

    return a_r_in.contiguous(), b_out_r.contiguous(), rank


def build_lora_input_bases(
    adapter_dir: str | Path,
    *,
    include: Optional[List[str]],
    basis_tol: float,
    basis_type: str,
    basis_solver: str = "qr",
) -> Dict[str, torch.Tensor]:
    tensor_path = Path(adapter_dir) / "adapter_model.safetensors"
    if not tensor_path.exists():
        raise FileNotFoundError(f"Adapter weights not found at {tensor_path}")
    state = load_file(str(tensor_path))
    modules = filter_modules(list_lora_modules(state), include)
    if not modules:
        raise RuntimeError("No LoRA modules found after filtering.")

    bases: Dict[str, torch.Tensor] = {}
    for name in modules:
        A = state.get(f"{name}{SUFFIX_A}")
        if A is None:
            continue
        if basis_type == "lora_A":
            basis = orth_basis(A, basis_tol, method=basis_solver)
            if basis is not None:
                bases[name] = basis
            continue
        if basis_type == "deltaW":
            B = state.get(f"{name}{SUFFIX_B}")
            if B is None:
                continue
            a_r_in, b_out_r, _ = orient_lora_weights(A, B)
            q_r = orthonormal_col_basis(b_out_r.t(), basis_tol, method=basis_solver)
            if q_r is None:
                continue
            a_eff = (q_r.t() @ a_r_in.float()).contiguous()
            basis = orth_basis(a_eff, basis_tol, method=basis_solver)
            if basis is not None:
                bases[name] = basis
            continue
        raise ValueError(f"Unknown basis_type={basis_type!r} (expected 'lora_A' or 'deltaW').")
    return bases


def _infer_in_features(module: nn.Module) -> Optional[int]:
    if hasattr(module, "in_features"):
        return int(getattr(module, "in_features"))
    weight = getattr(module, "weight", None)
    if isinstance(weight, torch.Tensor) and weight.ndim == 2:
        return int(weight.shape[1])
    return None


def register_deflation_hooks(
    model,
    bases: Dict[str, torch.Tensor],
    *,
    stats: Optional[DeflationStats] = None,
) -> List[torch.utils.hooks.RemovableHandle]:
    handles: List[torch.utils.hooks.RemovableHandle] = []
    modules = dict(model.named_modules())
    skipped = 0
    warned = set()

    def make_pre_hook(U_cpu_fp32: torch.Tensor, name: str):
        def pre_hook(_module, inputs):
            x = inputs[0]
            original_shape = x.shape
            last_dim = x.shape[-1]
            if last_dim != U_cpu_fp32.shape[0]:
                key = (name, last_dim)
                if key not in warned:
                    print(
                        f"[InferenceDeflation][skip-forward] {name}: input dim {last_dim} "
                        f"!= basis dim {U_cpu_fp32.shape[0]}; passing through."
                    )
                    warned.add(key)
                return None
            x2 = x.reshape(-1, last_dim)
            U = U_cpu_fp32.to(x2.device, x2.dtype)
            proj = (x2 @ U) @ U.t()
            x_perp = x2 - proj
            if stats is not None:
                stats.record(
                    name,
                    x_sq=x2.float().pow(2).sum().item(),
                    proj_sq=proj.float().pow(2).sum().item(),
                )
            return (x_perp.reshape(original_shape),)

        return pre_hook

    for name, U in bases.items():
        target = None
        module = modules.get(name)
        if isinstance(module, nn.Linear):
            target = module
        elif module is not None and hasattr(module, "base_layer") and isinstance(module.base_layer, nn.Linear):
            target = module.base_layer
        if target is None:
            module = modules.get(name + ".base_layer")
            if isinstance(module, nn.Linear):
                target = module
        if target is None:
            for module_name, module in modules.items():
                if not (module_name.endswith(name) or module_name.endswith(name + ".base_layer")):
                    continue
                if isinstance(module, nn.Linear):
                    target = module
                    break
                if hasattr(module, "base_layer") and isinstance(module.base_layer, nn.Linear):
                    target = module.base_layer
                    break
        if target is None or _infer_in_features(target) != U.shape[0]:
            skipped += 1
            continue
        handles.append(target.register_forward_pre_hook(make_pre_hook(U, name)))

    print(f"[InferenceDeflation] requested={len(bases)} | installed={len(handles)} | skipped={skipped}")
    if not handles:
        raise RuntimeError("Inference deflation hooks could not be installed.")
    return handles


def install_inference_deflation(
    model,
    *,
    adapter_dir: str,
    include: Optional[List[str]],
    basis_type: str,
    basis_solver: str,
    basis_tol: float,
    stats_csv: Optional[str] = None,
    sample_every: int = 1,
) -> InferenceDeflation:
    bases = build_lora_input_bases(
        adapter_dir,
        include=include,
        basis_type=basis_type,
        basis_solver=basis_solver,
        basis_tol=basis_tol,
    )
    stats = DeflationStats(sample_every=sample_every) if stats_csv else None
    handles = register_deflation_hooks(model, bases, stats=stats)
    return InferenceDeflation(handles=handles, stats=stats, stats_csv=stats_csv)
