import json
from pathlib import Path

import numpy as np
import torch
from click.testing import CliRunner

from neurinspectre.cli.main import cli


def _write_custom_dataset_npz(path: Path, *, n: int = 8, seed: int = 0) -> None:
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


def test_table2_jobs_query_sweep_smoke(tmp_path: Path) -> None:
    ds_path = tmp_path / "ds.npz"
    model_path = tmp_path / "m.pt"
    cfg_path = tmp_path / "t2_qs.yaml"
    out_dir = tmp_path / "out"

    _write_custom_dataset_npz(ds_path, n=8, seed=0)
    _write_torchscript_binary_classifier(model_path)

    cfg = f"""
seed: 0
datasets:
  custom:
    path: {ds_path.as_posix()}
    num_samples: 8
    batch_size: 1
    num_workers: 0
models:
  custom: {model_path.as_posix()}
jobs:
  - id: qs
    kind: query_sweep
    dataset: custom
    defense: none
    epsilon: 0.1
    norm: Linf
    num_samples: 4
    seed: 0
    query_budgets: [1000]
    attack:
      type: square
      p_init: 0.8
      loss_type: margin
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

    job_dir = out_dir / "qs"
    assert (job_dir / "run_metadata.json").exists()
    assert (job_dir / "result.json").exists()
    assert (job_dir / "q1000" / "summary.json").exists()

    payload = json.loads((job_dir / "result.json").read_text(encoding="utf-8"))
    assert payload.get("kind") == "query_sweep"
    assert payload.get("job_id") == "qs"
    assert payload.get("query_budgets") == [1000]

