"""
BAE baseline wrapper.

This is a thin convenience wrapper around the TextAttack recipe so callers can
import a baseline by name without dealing with recipe strings.
"""

from __future__ import annotations

from .text_attacks import TextAttackRunResult, run_textattack_recipe


def run_bae(
    *,
    model_name_or_path: str,
    dataset: str = "sst2",
    split: str = "validation",
    num_examples: int = 100,
    seed: int = 0,
    device: str = "cpu",
) -> TextAttackRunResult:
    return run_textattack_recipe(
        recipe="bae",
        model_name_or_path=model_name_or_path,
        dataset=dataset,
        split=split,
        num_examples=num_examples,
        seed=seed,
        device=device,
    )

