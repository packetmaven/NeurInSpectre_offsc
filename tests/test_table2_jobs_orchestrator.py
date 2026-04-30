import json
from pathlib import Path

import numpy as np
import torch
from click.testing import CliRunner

from neurinspectre.cli.main import cli


def _write_custom_dataset_npz(path: Path, *, n: int = 16, seed: int = 0) -> None:
    rng = np.random.default_rng(int(seed))
    x = rng.random((n, 1, 4, 4), dtype=np.float32)
    y = rng.integers(low=0, high=2, size=(n,), dtype=np.int64)
    np.savez(str(path), x=x, y=y)


def _write_torchscript_binary_classifier(path: Path) -> None:
    class TinyNet(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(16, 2))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)

    m = TinyNet().eval()
    example = torch.zeros(1, 1, 4, 4, dtype=torch.float32)
    ts = torch.jit.trace(m, example)
    ts.save(str(path))


def test_table2_jobs_core_evasion_smoke(tmp_path: Path) -> None:
    ds_path = tmp_path / "ds.npz"
    model_path = tmp_path / "m.pt"
    cfg_path = tmp_path / "t2_jobs.yaml"
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
jobs:
  - id: core
    kind: core_evasion
    dataset: custom
    defense: none
    epsilon: 0.1
    norm: Linf
    iterations: 2
    num_samples: 16
    seed: 0
    attacks:
      - name: pgd_short
        type: pgd
        steps: 1
      - name: pgd_long
        type: pgd
        steps: 2
baseline_validation:
  enabled: false
validity_gates:
  enabled: false
""".lstrip()
    cfg_path.write_text(cfg, encoding="utf-8")

    runner = CliRunner()
    res = runner.invoke(
        cli,
        [
            "table2",
            "--config",
            str(cfg_path),
            "--output-dir",
            str(out_dir),
            "--no-strict-real-data",
            "--no-report",
            "--no-progress",
            "--parallel",
            "1",
        ],
    )
    assert res.exit_code == 0, res.output

    # Per-job outputs
    job_dir = out_dir / "core"
    assert (job_dir / "run_metadata.json").exists()
    assert (job_dir / "result.json").exists()
    assert (job_dir / "summary.json").exists()

    payload = json.loads((job_dir / "result.json").read_text(encoding="utf-8"))
    assert payload.get("kind") == "core_evasion"
    assert payload.get("job_id") == "core"
    assert payload.get("dataset") == "custom"

    # Root summary
    assert (out_dir / "jobs_summary.json").exists()

