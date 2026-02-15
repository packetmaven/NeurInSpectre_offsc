"""Generate TorchScript stub models for ImageNet-100 and nuScenes."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torchvision.models as models


def _load_resnet50(pretrained: bool) -> torch.nn.Module:
    if pretrained:
        try:
            weights = models.ResNet50_Weights.IMAGENET1K_V1
            return models.resnet50(weights=weights)
        except Exception as exc:  # pragma: no cover - network or cache issue
            print(f"[warn] Failed to load pretrained ResNet-50 weights: {exc}")
    return models.resnet50(weights=None)


def _load_resnet18(pretrained: bool) -> torch.nn.Module:
    if pretrained:
        try:
            weights = models.ResNet18_Weights.IMAGENET1K_V1
            return models.resnet18(weights=weights)
        except Exception as exc:  # pragma: no cover - network or cache issue
            print(f"[warn] Failed to load pretrained ResNet-18 weights: {exc}")
    return models.resnet18(weights=None)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate TorchScript stub models.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./models"),
        help="Output directory for scripted models",
    )
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Disable pretrained ImageNet weights",
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    pretrained = not args.no_pretrained

    resnet50 = _load_resnet50(pretrained)
    resnet50.fc = torch.nn.Linear(resnet50.fc.in_features, 100)
    resnet50.eval()
    scripted_imagenet = torch.jit.script(resnet50)
    # Stub artifacts are intentionally *not trained*; keep the filenames explicit
    # so they are not mistaken for real evaluation checkpoints.
    imagenet_path = output_dir / "imagenet100_resnet50_stub.pt"
    scripted_imagenet.save(str(imagenet_path))
    print(f"Saved ImageNet-100 model to {imagenet_path}")

    resnet18 = _load_resnet18(pretrained)
    resnet18.fc = torch.nn.Linear(resnet18.fc.in_features, 10)
    resnet18.eval()
    scripted_nuscenes = torch.jit.script(resnet18)
    nuscenes_path = output_dir / "nuscenes_resnet18_stub.pt"
    scripted_nuscenes.save(str(nuscenes_path))
    print(f"Saved nuScenes model to {nuscenes_path}")


if __name__ == "__main__":
    main()
