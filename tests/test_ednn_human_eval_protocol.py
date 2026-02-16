import importlib.util
import json
import sys
from pathlib import Path


def _load_script():
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "ednn_human_eval_protocol.py"
    spec = importlib.util.spec_from_file_location("ednn_human_eval_protocol", script)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_cohen_kappa_basic_agreement():
    m = _load_script()
    res = m.cohen_kappa(["yes", "no", "yes"], ["yes", "no", "yes"])
    assert res["n_used"] == 3
    assert res["kappa"] == 1.0
    assert res["degenerate_marginals"] is False


def test_cohen_kappa_zero_when_chance_matches():
    m = _load_script()
    # Confusion matrix is perfectly balanced with p_o == p_e == 0.5 -> kappa 0.
    res = m.cohen_kappa(["yes", "yes", "no", "no"], ["yes", "no", "yes", "no"])
    assert res["n_used"] == 4
    assert abs(float(res["kappa"]) - 0.0) < 1e-12


def test_cohen_kappa_degenerate_constant_labels_is_flagged():
    m = _load_script()
    res = m.cohen_kappa(["yes", "yes", "yes"], ["yes", "yes", "yes"])
    assert res["n_used"] == 3
    assert res["degenerate_marginals"] is True
    assert res["kappa"] == 1.0


def test_prepare_and_score_end_to_end(tmp_path: Path):
    m = _load_script()

    input_jsonl = tmp_path / "pairs.jsonl"
    input_jsonl.write_text(
        "\n".join(
            [
                json.dumps({"item_id": "1", "original_text": "good movie", "candidate_text": "great movie", "algorithm": "textfooler"}),
                json.dumps({"item_id": "2", "original_text": "bad film", "candidate_text": "awful film", "algorithm": "bae"}),
                json.dumps({"item_id": "3", "original_text": "ok", "candidate_text": "okay", "algorithm": "bae"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    out_dir = tmp_path / "out"
    out_dir.mkdir(parents=True, exist_ok=True)

    m._prepare_tasks(
        input_jsonl=input_jsonl,
        output_dir=out_dir,
        raters=["alice", "bob"],
        seed=0,
        sample_size=None,
        stratify_key="algorithm",
        blind_pairs=True,
        id_key="item_id",
        original_key="original_text",
        candidate_key="candidate_text",
    )

    proto = out_dir / "protocol.json"
    admin = out_dir / "tasks_admin.jsonl"
    r1 = out_dir / "rater_alice.csv"
    r2 = out_dir / "rater_bob.csv"
    assert proto.exists()
    assert admin.exists()
    assert r1.exists()
    assert r2.exists()

    # Fill in a tiny set of ratings (perfect agreement).
    # Keep it minimal: overwrite CSVs with header+rows.
    header = "item_id,text_a,text_b,meaning_preserved,label_preserved,fluency,notes\n"
    rows = [
        "1,a,b,yes,yes,5,\n",
        "2,a,b,no,no,2,\n",
        "3,a,b,unsure,unsure,3,\n",
    ]
    r1.write_text(header + "".join(rows), encoding="utf-8")
    r2.write_text(header + "".join(rows), encoding="utf-8")

    out_report = out_dir / "kappa_report.json"
    m._score_kappa(
        rater_a_path=r1,
        rater_b_path=r2,
        protocol_path=proto,
        output_path=out_report,
        columns=None,
        id_column="item_id",
        bootstrap_trials=50,
        bootstrap_seed=0,
    )
    assert out_report.exists()
    rep = json.loads(out_report.read_text(encoding="utf-8"))
    assert "per_column" in rep
    assert rep["per_column"]["meaning_preserved"]["kappa"] == 1.0

