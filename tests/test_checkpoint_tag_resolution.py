from pathlib import Path

from neurinspectre.cli.table2_cmd import _resolve_checkpoint_tag_model_path


def test_resolve_checkpoint_tag_model_path_prefers_models_ts_suffix(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "models").mkdir(parents=True, exist_ok=True)
    # The resolver should find `{tag}_ts.pt` when present.
    (tmp_path / "models" / "md_gradient_reg_ember_ts.pt").write_bytes(b"dummy")

    got = _resolve_checkpoint_tag_model_path("md_gradient_reg_ember")
    assert got == str(Path("models/md_gradient_reg_ember_ts.pt"))

