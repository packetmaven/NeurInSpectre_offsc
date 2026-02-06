import json
import subprocess
import sys

import numpy as np


def _write_data(tmp_path):
    x = np.random.rand(8, 1, 4, 4).astype(np.float32)
    y = np.random.randint(0, 3, size=(8,), dtype=np.int64)
    x_path = tmp_path / "x.npy"
    y_path = tmp_path / "y.npy"
    np.save(x_path, x)
    np.save(y_path, y)
    return x_path, y_path


def _run_cli(args):
    cmd = [sys.executable, "-m", "neurinspectre"] + args
    return subprocess.run(cmd, capture_output=True, text=True, check=False)


def test_cli_pgd(tmp_path):
    x_path, y_path = _write_data(tmp_path)
    out_path = tmp_path / "pgd.json"
    res = _run_cli(
        [
            "pgd",
            "--input",
            str(x_path),
            "--labels",
            str(y_path),
            "--allow-demo-model",
            "--steps",
            "10",
            "--device",
            "cpu",
            "--output",
            str(out_path),
        ]
    )
    assert res.returncode == 0, res.stderr
    data = json.loads(out_path.read_text())
    assert "attack_success_rate" in data


def test_cli_apgd_bpda_eot_ma_autoattack(tmp_path):
    x_path, y_path = _write_data(tmp_path)
    commands = [
        ["apgd", "--loss", "ce"],
        ["bpda", "--defense", "thermometer", "--approx", "thermometer"],
        ["eot", "--transform", "random-smoothing", "--samples", "10"],
        ["ma-pgd", "--alpha-volterra", "0.6"],
        ["temporal-momentum"],
        ["autoattack", "--include-square"],
        ["pgd-restarts", "--restarts", "3"],
        ["apgd-ensemble", "--losses", "ce,dlr"],
    ]

    for idx, cmd in enumerate(commands):
        out_path = tmp_path / f"out_{idx}.json"
        res = _run_cli(
            cmd
            + [
                "--input",
                str(x_path),
                "--labels",
                str(y_path),
                "--allow-demo-model",
                "--steps",
                "10",
                "--device",
                "cpu",
                "--output",
                str(out_path),
            ]
        )
        assert res.returncode == 0, res.stderr
        data = json.loads(out_path.read_text())
        assert "attack_success_rate" in data


def test_cli_defense_analyze_and_orchestrator(tmp_path):
    x_path, y_path = _write_data(tmp_path)

    analyze_out = tmp_path / "analyze.json"
    res = _run_cli(
        [
            "defense-analyze",
            "--input",
            str(x_path),
            "--labels",
            str(y_path),
            "--allow-demo-model",
            "--device",
            "cpu",
            "--n-samples",
            "10",
            "--n-probe-images",
            "8",
            "--batch-size",
            "4",
            "--output",
            str(analyze_out),
        ]
    )
    assert res.returncode == 0, res.stderr
    data = json.loads(analyze_out.read_text())
    assert "alpha_volterra" in data
    assert "obfuscation_types" in data

    orch_out = tmp_path / "orch.json"
    res = _run_cli(
        [
            "attack-orchestrate",
            "--input",
            str(x_path),
            "--labels",
            str(y_path),
            "--allow-demo-model",
            "--device",
            "cpu",
            "--steps",
            "5",
            "--characterize-samples",
            "10",
            "--characterize-probe-images",
            "8",
            "--characterize-batch-size",
            "4",
            "--output",
            str(orch_out),
        ]
    )
    assert res.returncode == 0, res.stderr
    data = json.loads(orch_out.read_text())
    assert "attack_success_rate" in data


def test_cli_reproduce_claims(tmp_path):
    out_path = tmp_path / "claims.json"
    res = _run_cli(
        [
            "reproduce-claims",
            "--claim",
            "figure_1",
            "--mode",
            "fast",
            "--n-seeds",
            "1",
            "--device",
            "cpu",
            "--no-plot",
            "--output",
            str(out_path),
        ]
    )
    assert res.returncode == 0, res.stderr
    data = json.loads(out_path.read_text())
    assert "claim" in data


def test_cli_testing_context(tmp_path):
    res = _run_cli(["testing-context"])
    assert res.returncode == 0, res.stderr
    assert "NeurInSpectre Offensive Suite" in res.stdout
