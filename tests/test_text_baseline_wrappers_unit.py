from neurinspectre.baselines import bae as bae_mod
from neurinspectre.baselines import textfooler as tf_mod


def test_run_bae_passes_recipe_name():
    called = {}

    def fake_run_textattack_recipe(**kwargs):
        called.update(kwargs)
        return {"ok": True}

    bae_mod.run_textattack_recipe = fake_run_textattack_recipe  # type: ignore[assignment]
    out = bae_mod.run_bae(model_name_or_path="m", num_examples=3, seed=7, device="cpu")
    assert out == {"ok": True}
    assert called["recipe"] == "bae"
    assert called["model_name_or_path"] == "m"
    assert int(called["num_examples"]) == 3
    assert int(called["seed"]) == 7


def test_run_textfooler_passes_recipe_name():
    called = {}

    def fake_run_textattack_recipe(**kwargs):
        called.update(kwargs)
        return {"ok": True}

    tf_mod.run_textattack_recipe = fake_run_textattack_recipe  # type: ignore[assignment]
    out = tf_mod.run_textfooler(model_name_or_path="m", num_examples=5, seed=1, device="cpu")
    assert out == {"ok": True}
    assert called["recipe"] == "textfooler"
    assert called["model_name_or_path"] == "m"
    assert int(called["num_examples"]) == 5
    assert int(called["seed"]) == 1

