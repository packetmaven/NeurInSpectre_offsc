import json
from pathlib import Path

import numpy as np
import torch
from click.testing import CliRunner

from neurinspectre.cli.main import cli


def _write_custom_dataset_npz(path: Path, *, n: int = 16, seed: int = 0) -> None:
    rng = np.random.default_rng(int(seed))
    # Tiny "image-like" tensors: (N, C, H, W)
    x = rng.random((n, 1, 4, 4), dtype=np.float32)
    y = rng.integers(low=0, high=2, size=(n,), dtype=np.int64)
    np.savez(str(path), x=x, y=y)


def _write_torchscript_binary_classifier(path: Path) -> None:
    class TinyNet(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Flatten(),
                torch.nn.Linear(16, 2),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B,1,4,4) -> (B,2)
            return self.net(x)

    m = TinyNet().eval()
    example = torch.zeros(1, 1, 4, 4, dtype=torch.float32)
    ts = torch.jit.trace(m, example)
    ts.save(str(path))


def test_evaluate_supports_attack_type_aliases(tmp_path: Path) -> None:
    """
    `evaluate` should allow multiple named variants of the same underlying attack:
      {"name": "...", "type": "pgd", ...}
    """
    ds_path = tmp_path / "ds.npz"
    model_path = tmp_path / "m.pt"
    cfg_path = tmp_path / "eval.yaml"
    out_dir = tmp_path / "out"

    _write_custom_dataset_npz(ds_path, n=16, seed=0)
    _write_torchscript_binary_classifier(model_path)

    cfg = f"""
seed: 0
datasets:
  custom:
    path: {ds_path.as_posix()}
    num_samples: 16
    batch_size: 8
    num_workers: 0
models:
  custom: {model_path.as_posix()}
defenses:
  - name: none
    type: none
    dataset: custom
attacks:
  - name: pgd_short
    type: pgd
    steps: 1
  - name: pgd_long
    type: pgd
    steps: 2
perturbation:
  epsilon: 0.1
  norm: Linf
iterations: 2
validity_gates:
  enabled: false
baseline_validation:
  enabled: false
""".lstrip()
    cfg_path.write_text(cfg, encoding="utf-8")

    runner = CliRunner()
    res = runner.invoke(
        cli,
        [
            "evaluate",
            "--config",
            str(cfg_path),
            "--output-dir",
            str(out_dir),
            "--no-report",
            "--no-progress",
            "--parallel",
            "1",
        ],
    )
    assert res.exit_code == 0, res.output

    defense_out = out_dir / "none.json"
    assert defense_out.exists()
    payload = json.loads(defense_out.read_text(encoding="utf-8"))
    attacks = payload.get("attacks") or {}
    assert "pgd_short" in attacks
    assert "pgd_long" in attacks

