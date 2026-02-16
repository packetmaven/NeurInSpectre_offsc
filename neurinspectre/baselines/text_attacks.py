"""
Text adversarial-attack baselines.

Baselines requested for Issue 4 (EDNN §5.4):
- TextFooler (Jin et al., 2019)
- BAE (Garg and Ramakrishnan, 2019/2020)
- BERT-Attack (Li et al., 2020)

Implementation strategy:
- Prefer the canonical, maintained implementation from TextAttack when available.
  This avoids re-implementing complex token/constraint logic incorrectly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class TextAttackRunResult:
    recipe: str
    model: str
    dataset: str
    split: str
    num_examples: int
    success_rate: Optional[float]
    details: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "recipe": self.recipe,
            "model": self.model,
            "dataset": self.dataset,
            "split": self.split,
            "num_examples": int(self.num_examples),
            "success_rate": None if self.success_rate is None else float(self.success_rate),
            "details": dict(self.details or {}),
        }


def _require_textattack() -> Any:
    try:
        import textattack  # noqa: F401
    except Exception as exc:
        raise ImportError(
            "EDNN text baselines require the optional dependency `textattack`.\n"
            "Install: pip install textattack"
        ) from exc
    import textattack  # type: ignore[no-redef]

    return textattack


def run_textattack_recipe(
    *,
    recipe: str,
    model_name_or_path: str,
    dataset: str = "sst2",
    split: str = "validation",
    num_examples: int = 100,
    seed: int = 0,
    device: str = "cpu",
) -> TextAttackRunResult:
    """
    Run a TextAttack recipe on a HuggingFace model/dataset.

    This function intentionally keeps the interface small; the CLI layer should
    own argument parsing and output file management.
    """
    textattack = _require_textattack()

    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except Exception as exc:
        raise ImportError(
            "Text baselines require `torch` and `transformers` installed."
        ) from exc

    recipe_key = str(recipe or "").strip().lower().replace("-", "_")
    recipe_map = {
        "textfooler": "TextFoolerJin2019",
        "bae": "BAEGarg2019",
        "bert_attack": "BERTAttackLi2020",
        "bertattack": "BERTAttackLi2020",
    }
    recipe_cls_name = recipe_map.get(recipe_key)
    if recipe_cls_name is None:
        raise ValueError(
            f"Unknown recipe={recipe!r}. Expected one of: {', '.join(sorted(recipe_map))}"
        )

    tok = AutoTokenizer.from_pretrained(model_name_or_path)
    mdl = AutoModelForSequenceClassification.from_pretrained(model_name_or_path)
    mdl.eval()
    mdl.to(torch.device(str(device)))

    model_wrapper = textattack.models.wrappers.HuggingFaceModelWrapper(mdl, tok)

    # Use TextAttack's dataset wrapper; this may require `datasets` depending
    # on the chosen dataset.
    hf_dataset_name = str(dataset).strip()
    hf_split = str(split).strip()
    text_column = "sentence"
    label_column = "label"
    if hf_dataset_name.lower() not in {"sst2", "glue/sst2", "glue"}:
        # Keep this conservative: for other datasets, users should pass a custom dataset.
        raise ValueError(
            "Only dataset='sst2' is supported by this baseline wrapper for now."
        )

    # TextAttack expects the `glue` dataset for SST2.
    ta_dataset = textattack.datasets.HuggingFaceDataset(
        "glue",
        "sst2",
        split=hf_split,
        shuffle=True,
    )
    # Restrict to N examples deterministically (TextAttack shuffles).
    num_examples = int(max(1, num_examples))
    ta_dataset = ta_dataset.take(num_examples)

    recipe_cls = getattr(textattack.attack_recipes, recipe_cls_name)
    attack = recipe_cls.build(model_wrapper)

    attacker = textattack.Attacker(attack, ta_dataset, textattack.AttackArgs(
        num_examples=num_examples,
        random_seed=int(seed),
        disable_stdout=True,
        silent=True,
        # Don't write TextAttack's own output files; NeurInSpectre writes JSON itself.
        log_to_csv=False,
    ))
    results = attacker.attack_dataset()

    # TextAttack returns an iterable of results; compute success rate.
    n_total = 0
    n_succeeded = 0
    for r in results:
        n_total += 1
        # "Successful" means the attack changed the prediction to an incorrect label.
        if r and getattr(r, "goal_status", None) is not None:
            if str(r.goal_status).lower().endswith("succeeded"):
                n_succeeded += 1

    success_rate = (n_succeeded / n_total) if n_total else None
    return TextAttackRunResult(
        recipe=str(recipe_key),
        model=str(model_name_or_path),
        dataset="sst2",
        split=str(hf_split),
        num_examples=int(num_examples),
        success_rate=success_rate,
        details={"succeeded": n_succeeded, "total": n_total},
    )

