"""
Config generation helpers for NeurInSpectre CLI.
"""

from __future__ import annotations


def generate_example_config(config_type: str) -> str:
    config_type = str(config_type).lower()
    if config_type == "attack":
        return _attack_config()
    if config_type == "defense":
        return _defense_config()
    if config_type == "evaluation":
        return _evaluation_config()
    raise ValueError(f"Unknown config type: {config_type}")


def _attack_config() -> str:
    return """# attack.yaml - Example attack configuration
attack:
  model: ./models/resnet50.pth
  dataset: cifar10
  defense: jpeg
  defense_config: ./configs/jpeg.yaml
  attack_type: neurinspectre
  epsilon: 0.03137
  norm: Linf
  iterations: 100
  batch_size: 128
  num_samples: 1000
  targeted: false
"""


def _defense_config() -> str:
    return """# defense.yaml - Example defense configuration
defense:
  name: jpeg_compression
  type: jpeg
  quality: 75
"""


def _evaluation_config() -> str:
    return """# evaluation.yaml - Reproduce Paper Table 1 results

defenses:
  - name: jpeg_compression
    type: jpeg
    quality: 75

  - name: randomized_smoothing
    type: randsmooth
    sigma: 0.25
    num_samples: 100

attacks:
  - neurinspectre  # Adaptive (Paper Section 3)
  - apgd           # Baseline (Croce & Hein 2020)
  - autoattack     # Comparison (Paper Section 5.2.2)

datasets:
  cifar10:
    path: ./data/cifar10
    num_samples: 1000

perturbation:
  epsilon: 0.03137  # 8/255
  norm: Linf
"""
