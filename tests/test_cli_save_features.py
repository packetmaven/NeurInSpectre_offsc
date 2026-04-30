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


def test_attack_save_features_writes_jsonl(tmp_path: Path) -> None:
    ds_path = tmp_path / "ds.npz"
    model_path = tmp_path / "m.pt"
    out_json = tmp_path / "out.json"
    out_features = tmp_path / "features.jsonl"

    _write_custom_dataset_npz(ds_path, n=16, seed=0)
    _write_torchscript_binary_classifier(model_path)

    runner = CliRunner()
    res = runner.invoke(
        cli,
        [
            "attack",
            "--model",
            str(model_path),
            "--dataset",
            "custom",
            "--data-path",
            str(ds_path),
            "--defense",
            "none",
            "--attack-type",
            "pgd",
            "--epsilon",
            "0.1",
            "--iterations",
            "2",
            "--batch-size",
            "8",
            "--num-samples",
            "16",
            "--device",
            "cpu",
            "--no-report",
            "--no-progress",
            "--output",
            str(out_json),
            "--save-features",
            str(out_features),
        ],
    )
    assert res.exit_code == 0, res.output
    assert out_json.exists()
    assert out_features.exists()

    lines = [ln for ln in out_features.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) >= 1
    row = json.loads(lines[0])
    assert row.get("kind") == "attack_features"
    assert row.get("dataset") == "custom"

