import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def test_train_ember_defense_models_script_runs_and_writes_torchscript(tmp_path):
    # Create a tiny, vectorized EMBER-like memmap layout so the script can run
    # without requiring real EMBER assets.
    data_dir = tmp_path / "data" / "ember" / "ember_2018"
    data_dir.mkdir(parents=True, exist_ok=True)

    n = 128
    d = 32
    rng = np.random.default_rng(0)

    x_path = data_dir / "X_train.dat"
    y_path = data_dir / "y_train.dat"

    x_mm = np.memmap(str(x_path), dtype=np.float32, mode="w+", shape=(n, d))
    x_mm[:] = rng.standard_normal((n, d), dtype=np.float32)
    x_mm.flush()

    y_mm = np.memmap(str(y_path), dtype=np.float32, mode="w+", shape=(n,))
    # Labels are float32 in the repo's vectorized format; values >=0 are "labeled".
    y_mm[:] = (rng.random((n,)) > 0.5).astype(np.float32)
    y_mm.flush()

    out_dir = tmp_path / "models"
    ckpt_dir = out_dir / "checkpoints"

    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "train_ember_defense_models.py"

    cmd = [
        sys.executable,
        str(script),
        "--root",
        str(tmp_path / "data" / "ember"),
        "--output-dir",
        str(out_dir),
        "--checkpoints-dir",
        str(ckpt_dir),
        "--device",
        "cpu",
        "--seed",
        "0",
        "--batch-size",
        "16",
        "--train-samples",
        "64",
        "--val-samples",
        "32",
        "--standard-epochs",
        "1",
        "--gradreg-epochs",
        "1",
        "--distill-epochs",
        "1",
        "--at-epochs",
        "1",
        "--at-pgd-steps",
        "1",
        "--train-standard",
    ]

    run = subprocess.run(cmd, cwd=str(repo_root), capture_output=True, text=True)
    assert run.returncode == 0, f"stdout:\n{run.stdout}\n\nstderr:\n{run.stderr}"

    expected = [
        "ember_mlp_ts.pt",
        "md_gradient_reg_ember_ts.pt",
        "md_distillation_ember_ts.pt",
        "md_at_transform_ember_ts.pt",
    ]
    for fname in expected:
        ts_path = out_dir / fname
        assert ts_path.exists()
        meta_path = Path(str(ts_path) + ".meta.json")
        assert meta_path.exists()

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        assert isinstance(meta.get("sha256"), str) and len(meta["sha256"]) == 64
        assert meta["sha256"] == _sha256_file(ts_path)

        # Smoke-load the TorchScript artifact and ensure it can run.
        model = torch.jit.load(str(ts_path), map_location="cpu")
        model.eval()
        with torch.no_grad():
            out = model(torch.zeros(1, d, dtype=torch.float32))
        assert tuple(out.shape) == (1, 2)

