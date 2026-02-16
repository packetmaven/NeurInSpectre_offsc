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

    # TextAttack uses a global device singleton (`textattack.shared.utils.device`)
    # for some constraints (notably SBERT). Ensure the CLI's `--device` is honored.
    try:
        from textattack.shared import utils as ta_utils  # type: ignore

        ta_utils.device = torch.device(str(device))
    except Exception:
        ta_utils = None

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

    # Deterministically select a subset of SST-2 examples.
    #
    # TextAttack's HuggingFaceDataset wrapper does not consistently expose a
    # `.take(...)` helper across versions. Instead, we use the underlying HF
    # datasets object and then wrap it.
    num_examples = int(max(1, num_examples))
    try:
        import datasets as hf_datasets  # type: ignore
    except Exception:
        hf_datasets = None

    if hf_datasets is not None:
        hf = hf_datasets.load_dataset("glue", "sst2", split=hf_split)
        hf = hf.shuffle(seed=int(seed))
        n_take = int(min(len(hf), num_examples))
        hf = hf.select(range(n_take))
        ta_dataset = textattack.datasets.HuggingFaceDataset(hf, shuffle=False)
        num_examples = n_take
    else:
        # Fallback: let TextAttack load the dataset, then subset the internal
        # `_dataset` if present.
        ta_dataset = textattack.datasets.HuggingFaceDataset(
            "glue",
            "sst2",
            split=hf_split,
            shuffle=True,
        )
        n_take = int(min(len(ta_dataset), num_examples))
        if hasattr(ta_dataset, "_dataset") and hasattr(getattr(ta_dataset, "_dataset"), "select"):
            ta_dataset._dataset = ta_dataset._dataset.select(range(n_take))  # type: ignore[attr-defined]
        else:
            subset = [ta_dataset[i] for i in range(n_take)]
            ta_dataset = textattack.datasets.Dataset(subset, shuffle=False)
        num_examples = n_take

    recipe_cls = getattr(textattack.attack_recipes, recipe_cls_name)
    attack = recipe_cls.build(model_wrapper)

    # TextAttack's canonical TextFooler/BAE/BERT-Attack recipes use the
    # Universal Sentence Encoder constraint, which depends on tensorflow_hub.
    # On many PyTorch-only setups (especially macOS), tensorflow is intentionally
    # absent. To keep the baseline runnable while preserving a semantic-similarity
    # constraint, fall back to TextAttack's SBERT constraint when tfhub is missing.
    semantic_constraint_backend = "none"
    if hasattr(attack, "constraints") and isinstance(getattr(attack, "constraints"), list):
        has_use = any(c.__class__.__name__ == "UniversalSentenceEncoder" for c in attack.constraints)
        if has_use:
            semantic_constraint_backend = "use"
            try:
                import tensorflow_hub  # type: ignore  # noqa: F401
            except Exception:
                try:
                    from textattack.constraints.semantics.sentence_encoders.sentence_bert import SBERT

                    new_constraints = []
                    for c in attack.constraints:
                        if c.__class__.__name__ != "UniversalSentenceEncoder":
                            new_constraints.append(c)
                            continue
                        thr = float(getattr(c, "threshold", 0.8))
                        thr = float(max(-1.0, min(1.0, thr)))
                        new_constraints.append(
                            SBERT(
                                threshold=thr,
                                metric="cosine",
                                compare_against_original=bool(getattr(c, "compare_against_original", True)),
                                window_size=getattr(c, "window_size", None),
                                skip_text_shorter_than_window=bool(
                                    getattr(c, "skip_text_shorter_than_window", False)
                                ),
                                # Small, widely available SBERT model (faster than BERT-base).
                                model_name="all-MiniLM-L6-v2",
                            )
                        )
                    attack.constraints = new_constraints
                    semantic_constraint_backend = "sbert"
                except Exception:
                    # Last resort: drop USE constraint instead of crashing.
                    attack.constraints = [
                        c for c in attack.constraints if c.__class__.__name__ != "UniversalSentenceEncoder"
                    ]
                    semantic_constraint_backend = "dropped"

    attacker = textattack.Attacker(attack, ta_dataset, textattack.AttackArgs(
        num_examples=num_examples,
        random_seed=int(seed),
        disable_stdout=True,
        silent=True,
        # Don't write TextAttack's own output files; NeurInSpectre writes JSON itself.
        log_to_csv=None,
    ))
    results = attacker.attack_dataset()

    # TextAttack returns a list of AttackResult instances, which are typically
    # subclasses like SuccessfulAttackResult / FailedAttackResult / SkippedAttackResult.
    #
    # Some versions do not populate `goal_status`, so count based on class name.
    n_total = 0
    n_succeeded = 0
    n_failed = 0
    n_skipped = 0
    for r in results:
        n_total += 1
        tname = type(r).__name__.lower()
        if "successfulattackresult" in tname or tname.startswith("successful"):
            n_succeeded += 1
        elif "failedattackresult" in tname or tname.startswith("failed"):
            n_failed += 1
        elif "skippedattackresult" in tname or tname.startswith("skipped"):
            n_skipped += 1

    success_rate = (n_succeeded / n_total) if n_total else None
    return TextAttackRunResult(
        recipe=str(recipe_key),
        model=str(model_name_or_path),
        dataset="sst2",
        split=str(hf_split),
        num_examples=int(num_examples),
        success_rate=success_rate,
        details={
            "succeeded": n_succeeded,
            "failed": n_failed,
            "skipped": n_skipped,
            "total": n_total,
            "semantic_constraint_backend": semantic_constraint_backend,
            "requested_device": str(device),
            "textattack_device": None if ta_utils is None else str(getattr(ta_utils, "device", None)),
        },
    )

