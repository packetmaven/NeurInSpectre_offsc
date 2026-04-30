import json

import numpy as np
from click.testing import CliRunner

from neurinspectre.cli.main import cli


def test_drift_detect_command_writes_json(tmp_path):
    runner = CliRunner()
    ref = np.zeros((50, 4), dtype=np.float64)
    cur = np.ones((50, 4), dtype=np.float64)
    ref_path = tmp_path / "ref.npy"
    cur_path = tmp_path / "cur.npy"
    np.save(ref_path, ref)
    np.save(cur_path, cur)

    out = tmp_path / "drift.json"
    result = runner.invoke(
        cli,
        [
            "drift-detect",
            "--reference",
            str(ref_path),
            "--current",
            str(cur_path),
            "--methods",
            "hotelling,ks,mmd,bayesian,ks_ad_cvm_fisher_bh",
            "--output",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    assert out.exists()
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["reference_shape"] == [50, 4]
    assert payload["current_shape"] == [50, 4]
    assert "consensus" in payload
    assert "per_method" in payload
    assert "ks_ad_cvm_fisher_bh" in payload["per_method"]

