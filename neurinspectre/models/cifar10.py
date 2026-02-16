"""
CIFAR-10 model loading with RobustBench integration.
"""

from __future__ import annotations

from typing import Optional
import os
import urllib.request

import torch
import torch.nn as nn
import torch.nn.functional as F


class ModelConfig:
    """Model configurations for CIFAR-10."""

    STANDARD_MODELS = {
        "resnet20": {
            "arch": "resnet20",
            "pretrained": True,
            "num_classes": 10,
        },
        "resnet18": {
            "arch": "resnet18",
            "pretrained": True,
            "num_classes": 10,
        },
        "wrn_28_10": {
            "arch": "wide_resnet28_10",
            "pretrained": True,
            "num_classes": 10,
        },
    }

    ROBUST_MODELS = {
        "standard_at": "Carmon2019Unlabeled",
        "trades": "Zhang2019Theoretically",
        "mart": "Wang2023Better_WRN-28-10",
    }


class Normalize(nn.Module):
    """Channel-wise input normalization."""

    def __init__(self, mean, std):
        super().__init__()
        mean_t = torch.tensor(mean).view(1, 3, 1, 1)
        std_t = torch.tensor(std).view(1, 3, 1, 1)
        self.register_buffer("mean", mean_t)
        self.register_buffer("std", std_t)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std


def load_cifar10_model(
    model_name: str = "resnet20",
    pretrained: bool = True,
    device: str = "cuda",
    normalize: bool = True,
    weights_path: Optional[str] = None,
) -> nn.Module:
    """
    Load CIFAR-10 classification model.
    """
    if model_name in ModelConfig.ROBUST_MODELS:
        return load_robustbench_model(model_name, device=device)

    model = load_standard_model(
        arch=model_name,
        pretrained=pretrained,
        device=device,
        weights_path=weights_path,
    )

    if normalize:
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
        model = nn.Sequential(Normalize(mean, std), model).to(device)
        model.eval()
    return model


def load_robustbench_model(model_name: str, device: str = "cuda") -> nn.Module:
    """
    Load model from RobustBench Model Zoo.
    """
    try:
        from robustbench.utils import load_model
    except ImportError as exc:
        raise ImportError("RobustBench not installed. Install with: pip install robustbench") from exc

    robust_model_name = ModelConfig.ROBUST_MODELS[model_name]
    model = load_model(
        model_name=robust_model_name,
        dataset="cifar10",
        threat_model="Linf",
    )
    model = model.to(device)
    model.eval()
    return model


def load_standard_model(
    arch: str,
    pretrained: bool,
    device: str,
    weights_path: Optional[str] = None,
) -> nn.Module:
    """
    Load standard (non-robust) CIFAR-10 model.
    """
    import torchvision.models as models

    if arch in {"resnet20", "resnet32", "resnet44", "resnet56"}:
        depth = int(arch.replace("resnet", ""))
        model = CIFARResNet(depth=depth, num_classes=10)
        if pretrained:
            if weights_path is None:
                weights_path = download_cifar10_weights(arch)
            if weights_path and os.path.exists(weights_path):
                _load_state_dict(model, weights_path)
            else:
                print(f"[Models] Warning: {arch} weights unavailable; using random init.")

    elif arch == "resnet18":
        # NOTE: torchvision resnet18 is not the same as CIFAR-ResNet20/32/...
        # We keep it for convenience, but do not promise pretrained CIFAR-10
        # weights unless the user provides --weights-path.
        model = models.resnet18(weights=None)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
        model.fc = nn.Linear(model.fc.in_features, 10)

        if pretrained:
            if weights_path is None:
                print("[Models] Warning: resnet18 pretrained CIFAR-10 weights are not bundled; using random init.")
            elif weights_path and os.path.exists(weights_path):
                _load_state_dict(model, weights_path)
            else:
                print("[Models] Warning: resnet18 weights unavailable; using random init.")

    elif arch in {"wide_resnet28_10", "wrn_28_10"}:
        model = WideResNet(depth=28, widen_factor=10, num_classes=10)
        if pretrained:
            if weights_path is None:
                weights_path = download_cifar10_weights("wide_resnet28_10")
            if weights_path and os.path.exists(weights_path):
                _load_state_dict(model, weights_path)
            else:
                print("[Models] Warning: wide_resnet28_10 weights unavailable; using random init.")

    else:
        raise ValueError(f"Unknown architecture: {arch}")

    model = model.to(device)
    model.eval()
    return model


def _load_state_dict(model: nn.Module, weights_path: str) -> None:
    state = torch.load(weights_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state, strict=False)


def download_cifar10_weights(arch: str) -> Optional[str]:
    """
    Download pretrained CIFAR-10 weights.
    """
    weights_dir = "./models/weights/cifar10"
    os.makedirs(weights_dir, exist_ok=True)

    weights_urls = {
        # CIFAR ResNet family from chenyaofo/pytorch-cifar-models
        "resnet20": "https://github.com/chenyaofo/pytorch-cifar-models/releases/download/resnet/cifar10_resnet20-4118986f.pt",
        "resnet32": "https://github.com/chenyaofo/pytorch-cifar-models/releases/download/resnet/cifar10_resnet32-ef93fc4d.pt",
        "resnet44": "https://github.com/chenyaofo/pytorch-cifar-models/releases/download/resnet/cifar10_resnet44-2a3cabcb.pt",
        "resnet56": "https://github.com/chenyaofo/pytorch-cifar-models/releases/download/resnet/cifar10_resnet56-187c023a.pt",
        "wide_resnet28_10": "https://github.com/meliketoy/wide-resnet.pytorch/releases/download/v1.0/wrn_28_10.pt",
    }

    if arch not in weights_urls:
        raise ValueError(f"No pretrained weights available for {arch}")

    weights_path = os.path.join(weights_dir, f"{arch}.pt")
    if not os.path.exists(weights_path):
        print(f"[Models] Downloading {arch} weights...")
        try:
            urllib.request.urlretrieve(weights_urls[arch], weights_path)
            print("[Models] Download complete.")
        except Exception as exc:
            print(f"[Models] Warning: failed to download {arch} weights: {exc}")
            return None
    return weights_path


class WideBasic(nn.Module):
    def __init__(self, in_planes, planes, dropout_rate, stride=1):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=False)
        self.dropout = nn.Dropout(p=dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False)

    def forward(self, x):
        out = self.conv1(torch.relu(self.bn1(x)))
        out = self.dropout(out)
        out = self.conv2(torch.relu(self.bn2(out)))
        out += self.shortcut(x)
        return out


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropout_rate):
        super().__init__()
        layers = []
        for i in range(nb_layers):
            layers.append(
                block(
                    in_planes if i == 0 else out_planes,
                    out_planes,
                    dropout_rate,
                    stride if i == 0 else 1,
                )
            )
        self.layer = nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNet(nn.Module):
    """
    Wide ResNet for CIFAR-10.
    """

    def __init__(self, depth=28, widen_factor=10, dropout_rate=0.0, num_classes=10):
        super().__init__()
        assert (depth - 4) % 6 == 0, "Depth should be 6n+4"
        n = (depth - 4) // 6
        k = widen_factor

        n_stages = [16, 16 * k, 32 * k, 64 * k]
        self.conv1 = nn.Conv2d(3, n_stages[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.block1 = NetworkBlock(n, n_stages[0], n_stages[1], WideBasic, 1, dropout_rate)
        self.block2 = NetworkBlock(n, n_stages[1], n_stages[2], WideBasic, 2, dropout_rate)
        self.block3 = NetworkBlock(n, n_stages[2], n_stages[3], WideBasic, 2, dropout_rate)
        self.bn1 = nn.BatchNorm2d(n_stages[3])
        self.fc = nn.Linear(n_stages[3], num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = torch.relu(self.bn1(out))
        out = torch.mean(out, dim=(2, 3))
        return self.fc(out)


class _CIFARBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Identity()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return torch.relu(out)


class CIFARResNet(nn.Module):
    """
    CIFAR-style ResNet (depth=6n+2), compatible with common CIFAR checkpoints.

    This is *not* torchvision's ImageNet ResNet; it uses the standard CIFAR stem
    and 3 stages of {16,32,64} channels.
    """

    def __init__(self, *, depth: int = 20, num_classes: int = 10):
        super().__init__()
        depth = int(depth)
        if (depth - 2) % 6 != 0:
            raise ValueError("CIFARResNet depth must be 6n+2 (e.g., 20/32/44/56).")
        n = (depth - 2) // 6

        self.in_planes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(16, n, stride=1)
        self.layer2 = self._make_layer(32, n, stride=2)
        self.layer3 = self._make_layer(64, n, stride=2)
        self.fc = nn.Linear(64, int(num_classes))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, planes: int, num_blocks: int, stride: int) -> nn.Sequential:
        strides = [int(stride)] + [1] * (int(num_blocks) - 1)
        blocks = []
        for s in strides:
            blocks.append(_CIFARBasicBlock(self.in_planes, int(planes), int(s)))
            self.in_planes = int(planes) * _CIFARBasicBlock.expansion
        return nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(out.size(0), -1)
        return self.fc(out)
