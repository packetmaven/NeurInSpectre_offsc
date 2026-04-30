import json

from click.testing import CliRunner

from neurinspectre.cli.main import cli


def test_calibrate_thresholds_smoke(tmp_path):
    runner = CliRunner()

    # Write multiple files so we can also exercise glob expansion.
    pos = tmp_path / "pos_a.json"
    pos2 = tmp_path / "pos_b.json"
    neg = tmp_path / "neg_a.json"
    neg2 = tmp_path / "neg_b.json"
    out = tmp_path / "thresholds.json"

    # Mimic exporters.export_characterization_json payload structure.
    pos.write_text(
        json.dumps(
            {
                "type": "characterization",
                "report": {"etd_score": 0.9, "alpha_volterra": 0.2, "jacobian_rank": 0.8},
            }
        ),
        encoding="utf-8",
    )
    pos2.write_text(
        json.dumps(
            {
                "type": "characterization",
                "report": {"etd_score": 0.8, "alpha_volterra": 0.25, "jacobian_rank": 0.7},
            }
        ),
        encoding="utf-8",
    )
    neg.write_text(
        json.dumps(
            {
                "type": "characterization",
                "report": {"etd_score": 0.1, "alpha_volterra": 0.9, "jacobian_rank": 0.2},
            }
        ),
        encoding="utf-8",
    )
    neg2.write_text(
        json.dumps(
            {
                "type": "characterization",
                "report": {"etd_score": 0.2, "alpha_volterra": 0.85, "jacobian_rank": 0.3},
            }
        ),
        encoding="utf-8",
    )

    result = runner.invoke(
        cli,
        [
            "calibrate-thresholds",
            "--positive",
            str(tmp_path / "pos_*.json"),
            "--negative",
            str(tmp_path / "neg_*.json"),
            "--output",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["type"] == "threshold_calibration"
    assert "metric_calibration" in payload
    assert "etd_score" in payload["metric_calibration"]
    assert "defense_analyzer_threshold_overrides" in payload
    overrides = payload["defense_analyzer_threshold_overrides"]
    assert isinstance(overrides, dict)
    assert "ETD_THRESHOLD_SEVERE" in overrides
    assert "ALPHA_RL_THRESHOLD" in overrides
    assert isinstance(overrides["ETD_THRESHOLD_SEVERE"], (int, float))
    assert isinstance(overrides["ALPHA_RL_THRESHOLD"], (int, float))

