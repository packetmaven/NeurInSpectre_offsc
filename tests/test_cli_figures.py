from click.testing import CliRunner

from neurinspectre.cli.main import cli


def test_figures_help():
    runner = CliRunner()
    result = runner.invoke(cli, ["figures", "--help"])
    assert result.exit_code == 0

