"""
Unified model loading factory for NeurInSpectre evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import logging

import torch
import torch.nn as nn

from .cifar10 import load_cifar10_model
from .loader import load_model as load_generic_model

logger = logging.getLogger(__name__)


class TrainingType(Enum):
    STANDARD = "standard"
    ADVERSARIAL = "adversarial"
    TRADES = "trades"
    MART = "mart"
    DISTILLATION = "distillation"
    SMOOTHING = "smoothing"
    CUSTOM = "custom"


@dataclass
class ModelSpec:
    domain: str
    dataset: str
    architecture: str
    training_type: TrainingType
    num_classes: int
    input_shape: Tuple[int, ...]
    normalize_mean: Optional[Tuple[float, ...]] = None
    normalize_std: Optional[Tuple[float, ...]] = None
    input_range: Tuple[float, float] = (0.0, 1.0)
    training_epsilon: Optional[float] = None
    training_steps: Optional[int] = None
    distillation_temperature: Optional[float] = None
    checkpoint_path: Optional[Path] = None
    checkpoint_hash: Optional[str] = None
    version: str = "1.0"

    def get_key(self) -> str:
        key_components = [
            self.domain,
            self.dataset,
            self.architecture,
            self.training_type.value,
            str(self.num_classes),
        ]
        if self.training_epsilon:
            key_components.append(f"eps{self.training_epsilon}")
        if self.distillation_temperature:
            key_components.append(f"T{int(self.distillation_temperature)}")
        return "_".join(key_components)


class NormalizedModel(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        mean: Optional[Tuple[float, ...]] = None,
        std: Optional[Tuple[float, ...]] = None,
        input_range: Tuple[float, float] = (0.0, 1.0),
    ):
        super().__init__()
        self.model = model
        self.input_range = input_range
        if mean is not None and std is not None:
            self.register_buffer("mean", torch.tensor(mean).view(1, -1, 1, 1))
            self.register_buffer("std", torch.tensor(std).view(1, -1, 1, 1))
            self.normalize = True
        else:
            self.normalize = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.input_range != (0.0, 1.0):
            x = x * (self.input_range[1] - self.input_range[0]) + self.input_range[0]
        if self.normalize:
            x = (x - self.mean) / self.std
        return self.model(x)


class ModelFactory:
    _registry: Dict[str, ModelSpec] = {}
    _cache_dir = Path("./models/checkpoints")

    @classmethod
    def register_model(cls, spec: ModelSpec):
        key = spec.get_key()
        cls._registry[key] = spec
        logger.debug("[ModelFactory] Registered %s", key)

    @classmethod
    def load_model(
        cls,
        *,
        domain: str,
        model_name: str,
        training_type: str = "standard",
        dataset: Optional[str] = None,
        device: str = "cpu",
        **kwargs,
    ) -> nn.Module:
        domain = str(domain).lower()
        training_type = str(training_type).lower()
        dataset = dataset or ("cifar10" if domain in {"vision", "content_moderation"} else "custom")

        if domain in {"vision", "content_moderation"}:
            return cls._load_vision_model(model_name, training_type, dataset, device, **kwargs)
        if domain in {"malware_detection", "malware"}:
            return _load_custom_model(kwargs, device)
        if domain in {"av_perception", "av"}:
            return _load_custom_model(kwargs, device)
        raise ValueError(f"Unknown domain: {domain}")

    @classmethod
    def _load_vision_model(
        cls,
        model_name: str,
        training_type: str,
        dataset: str,
        device: str,
        **kwargs,
    ) -> nn.Module:
        if training_type in {"robustbench", "rb"}:
            threat_model = str(kwargs.get("threat_model", "Linf"))
            model_dir = kwargs.get("model_dir")
            if model_dir is not None:
                model_dir = str(model_dir)
            return cls._load_robustbench_direct(
                rb_model_name=str(model_name),
                dataset=str(dataset),
                threat_model=threat_model,
                device=str(device),
                model_dir=model_dir,
            )

        if training_type in {"adversarial", "trades", "mart"}:
            return cls._load_robustbench_model(model_name, training_type, dataset, device)

        if training_type == "distillation":
            checkpoint_path = kwargs.get("checkpoint_path")
            temperature = kwargs.get("distillation_temperature", 100)
            return cls._load_distilled_model(
                model_name, dataset, device, checkpoint_path=checkpoint_path, temperature=temperature
            )

        return cls._load_standard_vision_model(model_name, dataset, device, **kwargs)

    @classmethod
    def _load_robustbench_direct(
        cls,
        *,
        rb_model_name: str,
        dataset: str,
        threat_model: str,
        device: str,
        model_dir: Optional[str],
    ) -> nn.Module:
        """
        Load a RobustBench model by *RobustBench ID* (e.g., "Zhang2019Theoretically").

        This avoids brittle hard-coded architecture->ID maps and makes it easy to
        add modern defenses directly from RobustBench in evaluation configs.
        """
        try:
            from robustbench.utils import load_model as rb_load_model
        except ImportError as exc:
            raise ImportError("RobustBench not installed. Install with: pip install robustbench") from exc

        model = rb_load_model(
            model_name=str(rb_model_name),
            dataset=str(dataset),
            threat_model=str(threat_model),
            model_dir=model_dir,
        )
        model = model.to(device)
        model.eval()
        return model

    @classmethod
    def _load_robustbench_model(
        cls,
        model_name: str,
        training_type: str,
        dataset: str,
        device: str,
    ) -> nn.Module:
        try:
            from robustbench.utils import load_model as rb_load_model
        except ImportError as exc:
            raise ImportError("RobustBench not installed. Install with: pip install robustbench") from exc

        robustbench_models = {
            ("wrn_28_10", "adversarial", "cifar10"): "Carmon2019Unlabeled",
            ("wrn_28_10", "trades", "cifar10"): "Zhang2019Theoretically",
            ("wrn_28_10", "mart", "cifar10"): "Wang2023Better_WRN-28-10",
            ("resnet18", "adversarial", "cifar10"): "Standard",
        }

        rb_model_name = robustbench_models.get((model_name, training_type, dataset))
        if rb_model_name is None:
            raise ValueError(f"No RobustBench model for {(model_name, training_type, dataset)}")

        model = rb_load_model(model_name=rb_model_name, dataset=dataset, threat_model="Linf")
        model = model.to(device)
        model.eval()
        return model

    @classmethod
    def _load_distilled_model(
        cls,
        model_name: str,
        dataset: str,
        device: str,
        *,
        checkpoint_path: Optional[str],
        temperature: float,
    ) -> nn.Module:
        if checkpoint_path is None:
            raise FileNotFoundError(
                "Distillation requires a checkpoint_path with a trained distilled model."
            )
        model = cls._instantiate_architecture(model_name, dataset)
        state = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state, strict=False)
        model = cls._wrap_normalization(model, dataset)
        model = model.to(device)
        model.eval()
        return model

    @classmethod
    def load_ember_model(
        cls,
        checkpoint_path: Optional[str] = None,
        device: str = 'cpu',
        input_dim: int = 2381,
        num_classes: int = 2
    ) -> nn.Module:
        """
        Load EMBER malware detection MLP.
        
        Args:
            checkpoint_path: Path to trained checkpoint
            device: Device to load on
            input_dim: EMBER feature dimension (default 2381)
            num_classes: Number of classes (default 2: benign/malware)
        
        Returns:
            EMBER MLP model
        """
        # Build architecture
        model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
        
        # Load checkpoint if available
        if checkpoint_path is None:
            checkpoint_path = cls._cache_dir / "ember_mlp.pt"
        
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"EMBER checkpoint not found: {checkpoint_path}. "
                "Train with: python scripts/train_ember_defense_models.py --train-standard"
            )
        logger.info(f"[ModelFactory] Loading EMBER checkpoint: {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
        
        model = model.to(device)
        model.eval()
        return model
    
    @classmethod
    def load_nuscenes_model(
        cls,
        checkpoint_path: Optional[str] = None,
        device: str = 'cpu',
        num_classes: int = 10
    ) -> nn.Module:
        """
        Load nuScenes ResNet-18 for AV perception.
        
        Args:
            checkpoint_path: Path to fine-tuned checkpoint
            device: Device to load on
            num_classes: Number of object classes (default 10)
        
        Returns:
            ResNet-18 model
        """
        import torchvision.models as models
        
        # Build architecture
        model = models.resnet18(weights=None)  # Start from scratch
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        
        # Load checkpoint if available
        if checkpoint_path is None:
            checkpoint_path = cls._cache_dir / "nuscenes_resnet18.pt"
        
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"nuScenes checkpoint not found: {checkpoint_path}. "
                "See REPRODUCE.md for model setup instructions."
            )
        logger.info(f"[ModelFactory] Loading nuScenes checkpoint: {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location=device)
        
        if isinstance(state_dict, dict) and 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        
        model.load_state_dict(state_dict, strict=False)
        
        # ImageNet normalization (nuScenes images are similar)
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        model = NormalizedModel(model, mean, std)
        
        model = model.to(device)
        model.eval()
        return model
    
    @classmethod
    def load_imagenet100_model(
        cls,
        checkpoint_path: Optional[str] = None,
        device: str = 'cpu',
        architecture: str = 'resnet18',
        num_classes: int = 100
    ) -> nn.Module:
        """
        Load ImageNet-100 model.
        
        Args:
            checkpoint_path: Path to fine-tuned checkpoint
            device: Device to load on
            architecture: Model architecture (resnet18, resnet50)
            num_classes: Number of classes (100 for ImageNet-100)
        
        Returns:
            ImageNet-100 model
        """
        import torchvision.models as models
        
        # Build architecture with ImageNet-1K pretrained weights
        if architecture == 'resnet18':
            model = models.resnet18(weights='DEFAULT')
        elif architecture == 'resnet50':
            model = models.resnet50(weights='DEFAULT')
        else:
            raise ValueError(f"Unsupported architecture: {architecture}")
        
        # Replace final layer for 100 classes
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        
        # Load fine-tuned checkpoint if available
        if checkpoint_path is None:
            checkpoint_path = cls._cache_dir / f"imagenet100_{architecture}.pt"
        
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"ImageNet-100 checkpoint not found: {checkpoint_path}. "
                "Download from https://www.image-net.org/download.php and fine-tune, "
                "or see REPRODUCE.md for model setup instructions."
            )
        logger.info(f"[ModelFactory] Loading ImageNet-100 checkpoint: {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location=device)
        
        if isinstance(state_dict, dict) and 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        
        model.load_state_dict(state_dict, strict=False)
        
        # ImageNet normalization
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        model = NormalizedModel(model, mean, std)
        
        model = model.to(device)
        model.eval()
        return model



    @classmethod
    def _load_standard_vision_model(
        cls,
        model_name: str,
        dataset: str,
        device: str,
        **kwargs,
    ) -> nn.Module:
        if dataset == "cifar10":
            model = load_cifar10_model(
                model_name=model_name,
                pretrained=bool(kwargs.get("pretrained", True)),
                device=device,
                normalize=bool(kwargs.get("normalize", True)),
                weights_path=kwargs.get("weights_path"),
            )
            return model

        import torchvision.models as models

        if model_name == "resnet18":
            model = models.resnet18(weights="DEFAULT" if kwargs.get("pretrained", True) else None)
        elif model_name == "resnet50":
            model = models.resnet50(weights="DEFAULT" if kwargs.get("pretrained", True) else None)
        else:
            raise ValueError(f"Unsupported torchvision model: {model_name}")

        model = cls._wrap_normalization(model, dataset)
        model = model.to(device)
        model.eval()
        return model

    @classmethod
    def _instantiate_architecture(cls, model_name: str, dataset: str) -> nn.Module:
        if dataset == "cifar10":
            return load_cifar10_model(model_name=model_name, pretrained=False, device="cpu", normalize=False)
        import torchvision.models as models
        if model_name == "resnet18":
            return models.resnet18(weights=None)
        if model_name == "resnet50":
            return models.resnet50(weights=None)
        raise ValueError(f"Unsupported architecture: {model_name}")

    @classmethod
    def _wrap_normalization(cls, model: nn.Module, dataset: str) -> nn.Module:
        if dataset == "cifar10":
            mean = (0.4914, 0.4822, 0.4465)
            std = (0.2023, 0.1994, 0.2010)
            return NormalizedModel(model, mean, std)
        if dataset.startswith("imagenet"):
            mean = (0.485, 0.456, 0.406)
            std = (0.229, 0.224, 0.225)
            return NormalizedModel(model, mean, std)
        return model


def _load_custom_model(config: Dict[str, Any], device: str) -> nn.Module:
    if "model_factory" not in config:
        raise ValueError("custom model loading requires 'model_factory' in config")
    return load_generic_model(
        model_name=config.get("model_name"),
        dataset=config.get("dataset", "custom"),
        threat_model=config.get("threat_model", "Linf"),
        device=device,
        model_dir=config.get("model_dir"),
        model_factory=config.get("model_factory"),
        model_kwargs=config.get("model_kwargs"),
        checkpoint_path=config.get("checkpoint_path"),
    )
