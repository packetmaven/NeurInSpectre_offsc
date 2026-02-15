#!/usr/bin/env python3
"""
Train a real nuScenes image classifier (ResNet-18) for NeurInSpectre evaluation.

Why this exists:
- The repository includes utilities to generate *stub* TorchScript models for smoke tests.
- A stub model has a randomly initialized classification head and yields ~0% clean accuracy,
  making adversarial evaluation meaningless (no correctly classified samples to attack).

This script trains a small but real model on nuScenes using a label map derived from
official nuScenes annotations (see: scripts/generate_nuscenes_label_map.py).

Default behavior is intentionally lightweight:
- Uses v1.0-mini official splits (mini_train/mini_val)
- Fine-tunes only the final layer (fc) by default
- Exports a TorchScript artifact consumable by the NeurInSpectre Click CLI
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import DataLoader, TensorDataset

from neurinspectre.evaluation.datasets import nuScenesDataset


def _resolve_device(device: str) -> str:
    requested = str(device).lower()
    if requested in {"auto", ""}:
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
    return requested


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _class_weights(y: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    Build class weights for CrossEntropyLoss.

    We normalize weights to mean=1 over classes that appear in y.
    """
    counts = Counter(int(v) for v in y.cpu().tolist())
    total = float(sum(counts.values()))
    w = torch.zeros(int(num_classes), dtype=torch.float32)
    present = [c for c in range(int(num_classes)) if counts.get(c, 0) > 0]
    for c in present:
        w[c] = total / (float(num_classes) * float(counts[c]))
    if present:
        w[present] = w[present] / w[present].mean()
    return w


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


def train_nuscenes_classifier(
    *,
    root: Path,
    labels_path: Path,
    version: str,
    train_split: str,
    val_split: str,
    num_classes: int,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    pretrained_backbone: bool,
    finetune_backbone: bool,
    class_balanced: bool,
    seed: int,
    device: str,
) -> Tuple[nn.Module, Dict[str, float]]:
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))

    # Load tensors via the repo's canonical nuScenes loader so preprocessing matches evaluation.
    _train_loader, x_train, y_train = nuScenesDataset.load(
        root=str(root),
        labels_path=str(labels_path),
        version=str(version),
        split=str(train_split),
        n_samples=0,
        batch_size=int(batch_size),
        num_workers=0,
        pin_memory=False,
    )
    _val_loader, x_val, y_val = nuScenesDataset.load(
        root=str(root),
        labels_path=str(labels_path),
        version=str(version),
        split=str(val_split),
        n_samples=0,
        batch_size=int(batch_size),
        num_workers=0,
        pin_memory=False,
    )

    if x_train.numel() == 0 or y_train.numel() == 0:
        raise RuntimeError("nuScenes train split is empty after applying labels/split filters.")
    if x_val.numel() == 0 or y_val.numel() == 0:
        raise RuntimeError("nuScenes val split is empty after applying labels/split filters.")

    if int(y_train.min()) < 0 or int(y_train.max()) >= int(num_classes):
        raise ValueError(
            f"Train labels must be in [0,{num_classes-1}] but got "
            f"min={int(y_train.min())} max={int(y_train.max())}."
        )

    train_ds = TensorDataset(x_train, y_train)
    val_ds = TensorDataset(x_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=int(batch_size), shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=int(batch_size), shuffle=False, num_workers=0)

    # Model
    weights = None
    if bool(pretrained_backbone):
        try:
            weights = models.ResNet18_Weights.IMAGENET1K_V1
        except Exception:
            weights = None
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, int(num_classes))

    if not bool(finetune_backbone):
        for name, p in model.named_parameters():
            if not name.startswith("fc."):
                p.requires_grad_(False)

    device = _resolve_device(device)
    model = model.to(device)

    # Loss
    if bool(class_balanced):
        w = _class_weights(y_train, int(num_classes)).to(device)
        criterion = nn.CrossEntropyLoss(weight=w)
    else:
        criterion = nn.CrossEntropyLoss()

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(params, lr=float(lr), weight_decay=float(weight_decay))

    best_val = -1.0
    best_state = None

    for epoch in range(1, int(epochs) + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item()) * int(yb.numel())
            seen += int(yb.numel())

        train_acc = _accuracy(model, train_loader, device)
        val_acc = _accuracy(model, val_loader, device)
        avg_loss = running_loss / max(1, seen)
        print(
            f"[epoch {epoch:02d}/{epochs}] loss={avg_loss:.4f} "
            f"train_acc={train_acc:.3f} val_acc={val_acc:.3f}"
        )

        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    metrics = {
        "train_acc": float(_accuracy(model, train_loader, device)),
        "val_acc": float(best_val),
        "train_samples": float(len(train_ds)),
        "val_samples": float(len(val_ds)),
    }
    return model, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a real nuScenes ResNet-18 classifier.")
    parser.add_argument("--root", type=Path, default=Path("./data/nuscenes"), help="nuScenes dataroot")
    parser.add_argument(
        "--labels-path",
        type=Path,
        default=Path("./data/nuscenes/label_map.json"),
        help="Path to label_map.json (sample_token -> class index)",
    )
    parser.add_argument("--version", type=str, default="v1.0-mini", help="nuScenes version (e.g., v1.0-mini)")
    parser.add_argument("--train-split", type=str, default="train", help="Split for training (train/val/all)")
    parser.add_argument("--val-split", type=str, default="val", help="Split for validation (val/train/all)")
    parser.add_argument("--num-classes", type=int, default=10, help="Number of classes")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="auto", help="Device: auto|cpu|mps|cuda")
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Disable ImageNet pretrained backbone initialization",
    )
    parser.add_argument(
        "--finetune-backbone",
        action="store_true",
        help="Fine-tune the backbone (default trains only the final fc layer)",
    )
    parser.add_argument(
        "--no-class-balanced",
        action="store_true",
        help="Disable class-balanced CrossEntropy weights (recommended to keep enabled)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("./models/nuscenes_resnet18_trained.pt"),
        help="Output TorchScript model path",
    )
    parser.add_argument(
        "--state-dict-output",
        type=Path,
        default=Path("./models/checkpoints/nuscenes_resnet18_state_dict.pt"),
        help="Optional state_dict checkpoint output path",
    )
    args = parser.parse_args()

    device = _resolve_device(args.device)
    print(f"[nuScenes train] device={device} version={args.version}")
    print(f"[nuScenes train] root={args.root} labels_path={args.labels_path}")

    t0 = time.time()
    model, metrics = train_nuscenes_classifier(
        root=args.root,
        labels_path=args.labels_path,
        version=args.version,
        train_split=args.train_split,
        val_split=args.val_split,
        num_classes=int(args.num_classes),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        pretrained_backbone=not bool(args.no_pretrained),
        finetune_backbone=bool(args.finetune_backbone),
        class_balanced=not bool(args.no_class_balanced),
        seed=int(args.seed),
        device=device,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Export TorchScript artifact for CLI usage.
    model_cpu = model.to("cpu").eval()
    scripted = torch.jit.script(model_cpu)
    scripted.save(str(out_path))

    # Save a state_dict as well (useful if you want ModelFactory.load_nuscenes_model).
    sd_path = Path(args.state_dict_output)
    sd_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model_cpu.state_dict(), sd_path)

    meta = {
        "dataset": "nuscenes",
        "version": str(args.version),
        "root": str(args.root),
        "labels_path": str(args.labels_path),
        "labels_sha256": _sha256_file(Path(args.labels_path)) if Path(args.labels_path).exists() else None,
        "num_classes": int(args.num_classes),
        "train_split": str(args.train_split),
        "val_split": str(args.val_split),
        "pretrained_backbone": bool(not args.no_pretrained),
        "finetune_backbone": bool(args.finetune_backbone),
        "class_balanced": bool(not args.no_class_balanced),
        "seed": int(args.seed),
        "metrics": metrics,
        "torch_version": str(torch.__version__),
        "elapsed_seconds": float(time.time() - t0),
        "torchscript_path": str(out_path),
        "state_dict_path": str(sd_path),
    }
    meta_path = out_path.with_suffix(out_path.suffix + ".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"[nuScenes train] wrote TorchScript: {out_path}")
    print(f"[nuScenes train] wrote state_dict: {sd_path}")
    print(f"[nuScenes train] meta: {meta_path}")
    print(
        f"[nuScenes train] val_acc={float(metrics.get('val_acc', 0.0)):.3f} "
        f"elapsed={float(meta['elapsed_seconds']):.1f}s"
    )


if __name__ == "__main__":
    main()

