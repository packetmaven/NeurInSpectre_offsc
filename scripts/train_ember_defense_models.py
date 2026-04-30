#!/usr/bin/env python3
"""
Train real EMBER defense-model checkpoints (implementation parity).

This script produces TorchScript artifacts that the runnable `neurinspectre table2`
pipeline can pick up via `table2_config.yaml` checkpoint_tag resolution:

- md_gradient_reg_ember   -> models/md_gradient_reg_ember_ts.pt
- md_distillation_ember   -> models/md_distillation_ember_ts.pt
- md_at_transform_ember   -> models/md_at_transform_ember_ts.pt

It can also (optionally) train/export a standard teacher model to:
- models/ember_mlp_ts.pt

Notes:
- Defaults are AE-friendly (subset training). Increase sample counts/epochs for
  higher accuracy.
- Training uses the vectorized memmap format produced by `scripts/vectorize_ember_safe.py`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from neurinspectre.evaluation.datasets import EMBERDataset
from neurinspectre.training.ember_defenses import (
    train_at_transform,
    train_distilled_student,
    train_gradient_regularized,
    train_standard,
)


def _resolve_device(device: str) -> str:
    requested = str(device).lower().strip()
    if requested in {"auto", ""}:
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return requested


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _accuracy(model: nn.Module, loader: DataLoader, device: str) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb).argmax(1)
            correct += int((pred == yb).sum().item())
            total += int(yb.numel())
    return float(correct / total) if total else 0.0


def _build_ember_mlp(*, input_dim: int, num_classes: int) -> nn.Module:
    """
    Keep this architecture consistent with `scripts/export_ember_torchscript.py`.
    """
    return nn.Sequential(
        nn.Linear(int(input_dim), 512),
        nn.ReLU(),
        nn.BatchNorm1d(512),
        nn.Dropout(0.3),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.BatchNorm1d(256),
        nn.Dropout(0.3),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, int(num_classes)),
    )


def _export_torchscript(*, model: nn.Module, out_path: Path, input_dim: int) -> None:
    model_cpu = model.to("cpu").eval()
    try:
        scripted = torch.jit.script(model_cpu)
    except Exception as exc:
        print(f"[WARN] torch.jit.script failed ({type(exc).__name__}): {exc}")
        example = torch.zeros(1, int(input_dim), dtype=torch.float32)
        scripted = torch.jit.trace(model_cpu, example)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    scripted.save(str(out_path))


def _materialized_loader(x: np.ndarray, y: np.ndarray, *, batch_size: int, shuffle: bool) -> DataLoader:
    x_t = torch.from_numpy(np.asarray(x, dtype=np.float32)).float()
    y_t = torch.from_numpy(np.asarray(y, dtype=np.int64)).long()
    ds = TensorDataset(x_t, y_t)
    return DataLoader(ds, batch_size=int(batch_size), shuffle=bool(shuffle), num_workers=0)


def _load_ember_train_val(
    *,
    root: Path,
    seed: int,
    train_samples: int,
    val_samples: int,
    val_fraction: float,
    batch_size: int,
) -> Tuple[DataLoader, DataLoader, int]:
    data_dir = root / "ember_2018"
    x_mm, y_mm = EMBERDataset._read_vectorized_memmaps(str(data_dir), subset="train")

    # Labels are a small float32 vector (~MBs), safe to materialize.
    y_arr = np.asarray(y_mm, dtype=np.float32)
    labeled_idx = np.nonzero(y_arr >= 0)[0]
    if labeled_idx.size == 0:
        raise SystemExit("EMBER train labels contain no labeled samples (y >= 0).")

    rng = np.random.default_rng(int(seed))

    if val_samples > 0:
        n_val = int(min(val_samples, labeled_idx.size))
    else:
        vf = float(max(0.0, min(0.9, val_fraction)))
        n_val = int(max(1, round(vf * float(labeled_idx.size))))

    if train_samples > 0:
        n_train = int(min(train_samples, labeled_idx.size - n_val))
    else:
        # AE-friendly default: if the user didn't specify train_samples explicitly,
        # we avoid silently trying to materialize the full dataset.
        n_train = int(min(100_000, labeled_idx.size - n_val))

    if n_train <= 0:
        raise SystemExit(
            "Training split resolved to 0 samples. "
            "Decrease --val-samples/--val-fraction or increase --train-samples."
        )

    need = n_train + n_val
    chosen = rng.choice(labeled_idx, size=int(need), replace=False)
    train_idx = np.sort(chosen[:n_train])
    val_idx = np.sort(chosen[n_train:])

    x_train = np.array(x_mm[train_idx], copy=True)
    y_train = np.array(y_arr[train_idx].astype(np.int64, copy=False), copy=True)
    x_val = np.array(x_mm[val_idx], copy=True)
    y_val = np.array(y_arr[val_idx].astype(np.int64, copy=False), copy=True)

    input_dim = int(x_train.shape[1])
    train_loader = _materialized_loader(x_train, y_train, batch_size=batch_size, shuffle=True)
    val_loader = _materialized_loader(x_val, y_val, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, input_dim


@dataclass(frozen=True)
class _Artifact:
    name: str
    torchscript_path: Path
    meta_path: Path
    sha256: str
    metrics: Dict[str, float]


def _write_meta(*, meta_path: Path, payload: Dict) -> None:
    meta_path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Train/export EMBER defense models (TorchScript).")
    ap.add_argument("--root", type=Path, default=Path("./data/ember"), help="EMBER root (contains ember_2018/)")
    ap.add_argument("--output-dir", type=Path, default=Path("./models"), help="Output directory for TorchScript")
    ap.add_argument("--checkpoints-dir", type=Path, default=Path("./models/checkpoints"), help="Output dir for state_dicts")
    ap.add_argument("--device", type=str, default="auto", help="Device: auto|cpu|mps|cuda")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument("--batch-size", type=int, default=256, help="Batch size")
    ap.add_argument("--train-samples", type=int, default=100_000, help="Materialized train samples (0 => auto default)")
    ap.add_argument("--val-samples", type=int, default=20_000, help="Materialized val samples (0 => use val-fraction)")
    ap.add_argument("--val-fraction", type=float, default=0.1, help="Val fraction when --val-samples=0")
    ap.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate")
    ap.add_argument("--num-classes", type=int, default=2, help="Number of classes (default: 2)")
    ap.add_argument("--train-standard", action="store_true", help="Train/export baseline teacher model (ember_mlp_ts.pt)")
    ap.add_argument("--standard-epochs", type=int, default=5, help="Epochs for standard teacher training")
    ap.add_argument("--gradreg-epochs", type=int, default=2, help="Epochs for grad-reg fine-tune")
    ap.add_argument("--lambda-grad", type=float, default=0.1, help="Gradient regularization weight (lambda)")
    ap.add_argument("--distill-epochs", type=int, default=2, help="Epochs for distillation student training")
    ap.add_argument("--distill-temperature", type=float, default=100.0, help="Distillation temperature")
    ap.add_argument("--distill-alpha-hard", type=float, default=0.0, help="Hard-label CE weight during distillation")
    ap.add_argument("--at-epochs", type=int, default=2, help="Epochs for AT+transform fine-tune")
    ap.add_argument("--at-noise-std", type=float, default=0.05, help="Transform noise stddev")
    ap.add_argument("--at-eps-l2", type=float, default=0.5, help="L2 epsilon for adversarial training")
    ap.add_argument("--at-pgd-steps", type=int, default=7, help="PGD steps for adversarial training")
    ap.add_argument("--skip-gradreg", action="store_true", help="Skip training md_gradient_reg_ember")
    ap.add_argument("--skip-distill", action="store_true", help="Skip training md_distillation_ember")
    ap.add_argument("--skip-at", action="store_true", help="Skip training md_at_transform_ember")
    args = ap.parse_args()

    device = _resolve_device(args.device)
    seed = int(args.seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    t0 = time.time()
    train_loader, val_loader, input_dim = _load_ember_train_val(
        root=Path(args.root),
        seed=seed,
        train_samples=int(args.train_samples),
        val_samples=int(args.val_samples),
        val_fraction=float(args.val_fraction),
        batch_size=int(args.batch_size),
    )
    num_classes = int(args.num_classes)

    out_dir = Path(args.output_dir)
    ckpt_dir = Path(args.checkpoints_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    artifacts: List[_Artifact] = []

    # 1) Teacher / standard model (optional)
    teacher = _build_ember_mlp(input_dim=input_dim, num_classes=num_classes)
    std_stats = train_standard(
        teacher,
        train_loader,
        device=device,
        epochs=int(args.standard_epochs),
        lr=float(args.lr),
    )
    std_val_acc = _accuracy(teacher, val_loader, device)
    print(f"[standard] epochs={std_stats.epochs} last_loss={std_stats.last_loss:.4f} val_acc={std_val_acc:.3f}")

    teacher_sd_path = ckpt_dir / "ember_mlp.pt"
    torch.save(teacher.to("cpu").state_dict(), teacher_sd_path)

    if bool(args.train_standard):
        std_ts_path = out_dir / "ember_mlp_ts.pt"
        _export_torchscript(model=teacher, out_path=std_ts_path, input_dim=input_dim)
        sha = _sha256_file(std_ts_path)
        meta_path = std_ts_path.with_suffix(std_ts_path.suffix + ".meta.json")
        _write_meta(
            meta_path=meta_path,
            payload={
                "dataset": "ember",
                "checkpoint_tag": None,
                "variant": "standard",
                "input_dim": int(input_dim),
                "num_classes": int(num_classes),
                "seed": int(seed),
                "device": str(device),
                "train_samples": int(args.train_samples),
                "val_samples": int(args.val_samples),
                "lr": float(args.lr),
                "epochs": int(args.standard_epochs),
                "val_acc": float(std_val_acc),
                "train_stats": std_stats.to_dict(),
                "sha256": str(sha),
                "torch_version": str(torch.__version__),
                "elapsed_seconds": float(time.time() - t0),
                "torchscript_path": str(std_ts_path),
                "state_dict_path": str(teacher_sd_path),
            },
        )
        artifacts.append(
            _Artifact(
                name="standard",
                torchscript_path=std_ts_path,
                meta_path=meta_path,
                sha256=sha,
                metrics={"val_acc": float(std_val_acc)},
            )
        )

    # 2) Gradient regularization (fine-tune from teacher weights)
    if not bool(args.skip_gradreg):
        gradreg = _build_ember_mlp(input_dim=input_dim, num_classes=num_classes)
        gradreg.load_state_dict(teacher.to("cpu").state_dict())
        gr_stats = train_gradient_regularized(
            gradreg,
            train_loader,
            device=device,
            epochs=int(args.gradreg_epochs),
            lr=float(args.lr),
            lambda_grad=float(args.lambda_grad),
        )
        gr_val_acc = _accuracy(gradreg, val_loader, device)
        tag = "md_gradient_reg_ember"
        sd_path = ckpt_dir / f"{tag}.pt"
        torch.save(gradreg.to("cpu").state_dict(), sd_path)
        ts_path = out_dir / f"{tag}_ts.pt"
        _export_torchscript(model=gradreg, out_path=ts_path, input_dim=input_dim)
        sha = _sha256_file(ts_path)
        meta_path = ts_path.with_suffix(ts_path.suffix + ".meta.json")
        _write_meta(
            meta_path=meta_path,
            payload={
                "dataset": "ember",
                "checkpoint_tag": tag,
                "variant": "gradient_regularization",
                "input_dim": int(input_dim),
                "num_classes": int(num_classes),
                "seed": int(seed),
                "device": str(device),
                "train_samples": int(args.train_samples),
                "val_samples": int(args.val_samples),
                "lr": float(args.lr),
                "epochs": int(args.gradreg_epochs),
                "lambda_grad": float(args.lambda_grad),
                "val_acc": float(gr_val_acc),
                "train_stats": gr_stats.to_dict(),
                "sha256": str(sha),
                "torch_version": str(torch.__version__),
                "elapsed_seconds": float(time.time() - t0),
                "torchscript_path": str(ts_path),
                "state_dict_path": str(sd_path),
                "init_from": "standard_teacher",
            },
        )
        artifacts.append(
            _Artifact(
                name=tag,
                torchscript_path=ts_path,
                meta_path=meta_path,
                sha256=sha,
                metrics={"val_acc": float(gr_val_acc)},
            )
        )
        print(f"[{tag}] epochs={gr_stats.epochs} last_loss={gr_stats.last_loss:.4f} val_acc={gr_val_acc:.3f}")

    # 3) Defensive distillation (train student from teacher)
    if not bool(args.skip_distill):
        student = _build_ember_mlp(input_dim=input_dim, num_classes=num_classes)
        dist_stats = train_distilled_student(
            teacher=teacher,
            student=student,
            train_loader=train_loader,
            device=device,
            epochs=int(args.distill_epochs),
            lr=float(args.lr),
            temperature=float(args.distill_temperature),
            alpha_hard=float(args.distill_alpha_hard),
        )
        dist_val_acc = _accuracy(student, val_loader, device)
        tag = "md_distillation_ember"
        sd_path = ckpt_dir / f"{tag}.pt"
        torch.save(student.to("cpu").state_dict(), sd_path)
        ts_path = out_dir / f"{tag}_ts.pt"
        _export_torchscript(model=student, out_path=ts_path, input_dim=input_dim)
        sha = _sha256_file(ts_path)
        meta_path = ts_path.with_suffix(ts_path.suffix + ".meta.json")
        _write_meta(
            meta_path=meta_path,
            payload={
                "dataset": "ember",
                "checkpoint_tag": tag,
                "variant": "defensive_distillation",
                "input_dim": int(input_dim),
                "num_classes": int(num_classes),
                "seed": int(seed),
                "device": str(device),
                "train_samples": int(args.train_samples),
                "val_samples": int(args.val_samples),
                "lr": float(args.lr),
                "epochs": int(args.distill_epochs),
                "temperature": float(args.distill_temperature),
                "alpha_hard": float(args.distill_alpha_hard),
                "val_acc": float(dist_val_acc),
                "train_stats": dist_stats.to_dict(),
                "sha256": str(sha),
                "torch_version": str(torch.__version__),
                "elapsed_seconds": float(time.time() - t0),
                "torchscript_path": str(ts_path),
                "state_dict_path": str(sd_path),
                "teacher_state_dict_path": str(teacher_sd_path),
            },
        )
        artifacts.append(
            _Artifact(
                name=tag,
                torchscript_path=ts_path,
                meta_path=meta_path,
                sha256=sha,
                metrics={"val_acc": float(dist_val_acc)},
            )
        )
        print(f"[{tag}] epochs={dist_stats.epochs} last_loss={dist_stats.last_loss:.4f} val_acc={dist_val_acc:.3f}")

    # 4) Adversarial training with transforms (fine-tune from teacher weights)
    if not bool(args.skip_at):
        at_model = _build_ember_mlp(input_dim=input_dim, num_classes=num_classes)
        at_model.load_state_dict(teacher.to("cpu").state_dict())
        at_stats = train_at_transform(
            at_model,
            train_loader,
            device=device,
            epochs=int(args.at_epochs),
            lr=float(args.lr),
            noise_std=float(args.at_noise_std),
            epsilon_l2=float(args.at_eps_l2),
            pgd_steps=int(args.at_pgd_steps),
        )
        at_val_acc = _accuracy(at_model, val_loader, device)
        tag = "md_at_transform_ember"
        sd_path = ckpt_dir / f"{tag}.pt"
        torch.save(at_model.to("cpu").state_dict(), sd_path)
        ts_path = out_dir / f"{tag}_ts.pt"
        _export_torchscript(model=at_model, out_path=ts_path, input_dim=input_dim)
        sha = _sha256_file(ts_path)
        meta_path = ts_path.with_suffix(ts_path.suffix + ".meta.json")
        _write_meta(
            meta_path=meta_path,
            payload={
                "dataset": "ember",
                "checkpoint_tag": tag,
                "variant": "at_transform",
                "input_dim": int(input_dim),
                "num_classes": int(num_classes),
                "seed": int(seed),
                "device": str(device),
                "train_samples": int(args.train_samples),
                "val_samples": int(args.val_samples),
                "lr": float(args.lr),
                "epochs": int(args.at_epochs),
                "noise_std": float(args.at_noise_std),
                "epsilon_l2": float(args.at_eps_l2),
                "pgd_steps": int(args.at_pgd_steps),
                "val_acc": float(at_val_acc),
                "train_stats": at_stats.to_dict(),
                "sha256": str(sha),
                "torch_version": str(torch.__version__),
                "elapsed_seconds": float(time.time() - t0),
                "torchscript_path": str(ts_path),
                "state_dict_path": str(sd_path),
                "init_from": "standard_teacher",
            },
        )
        artifacts.append(
            _Artifact(
                name=tag,
                torchscript_path=ts_path,
                meta_path=meta_path,
                sha256=sha,
                metrics={"val_acc": float(at_val_acc)},
            )
        )
        print(f"[{tag}] epochs={at_stats.epochs} last_loss={at_stats.last_loss:.4f} val_acc={at_val_acc:.3f}")

    elapsed = time.time() - t0
    print(f"[done] wrote {len(artifacts)} TorchScript artifact(s) elapsed={elapsed:.1f}s")
    for art in artifacts:
        print(f"  - {art.name}: {art.torchscript_path} (sha256={art.sha256})")


if __name__ == "__main__":
    main()

