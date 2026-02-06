import json

from click.testing import CliRunner

from neurinspectre.cli.main import cli


def test_attack_help():
    runner = CliRunner()
    result = runner.invoke(cli, ["attack", "--help"])
    assert result.exit_code == 0


def test_attack_run(tmp_path):
    runner = CliRunner()
    output = tmp_path / "attack_results.json"
    result = runner.invoke(
        cli,
        [
            "attack",
            "--model",
            "tests/fixtures/test_model.pth",
            "--dataset",
            "cifar10",
            "--attack-type",
            "pgd",
            "--epsilon",
            "0.03",
            "--iterations",
            "5",
            "--num-samples",
            "10",
            "--batch-size",
            "5",
            "--output",
            str(output),
        ],
    )
    assert result.exit_code == 0
    assert output.exists()
    data = json.loads(output.read_text(encoding="utf-8"))
    assert "attack_success_rate" in data


def test_characterize_help():
    runner = CliRunner()
    result = runner.invoke(cli, ["characterize", "--help"])
    assert result.exit_code == 0


def test_characterize_run(tmp_path):
    runner = CliRunner()
    output = tmp_path / "characterization.json"
    result = runner.invoke(
        cli,
        [
            "characterize",
            "--model",
            "tests/fixtures/test_model.pth",
            "--dataset",
            "cifar10",
            "--defense",
            "jpeg",
            "--num-samples",
            "10",
            "--output",
            str(output),
        ],
    )
    assert result.exit_code == 0
    assert output.exists()


def test_config_generation():
    runner = CliRunner()
    result = runner.invoke(cli, ["config", "attack"])
    assert result.exit_code == 0
    assert "epsilon" in result.output
