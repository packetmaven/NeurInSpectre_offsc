"""Attack budget resolution helpers for dataset-aware evaluations."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Tuple


def _pick(mapping: Dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def get_attack_budgets(config: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Return normalized attack budgets map from config."""
    direct = config.get("attack_budgets")
    if isinstance(direct, dict):
        return {str(k): dict(v or {}) for k, v in direct.items()}

    defaults = config.get("defaults", {})
    nested = defaults.get("attack_budgets") if isinstance(defaults, dict) else None
    if isinstance(nested, dict):
        return {str(k): dict(v or {}) for k, v in nested.items()}
    return {}


def resolve_dataset_budget(
    config: Dict[str, Any],
    dataset_name: str,
    *,
    aliases: Optional[Iterable[str]] = None,
) -> Tuple[Dict[str, Any], Optional[str]]:
    """
    Resolve attack budget for a dataset name (with optional aliases).

    Returns:
        (budget_dict, matched_key)
    """
    budgets = get_attack_budgets(config)
    candidates = [str(dataset_name)]
    if aliases:
        candidates.extend(str(a) for a in aliases)
    for candidate in candidates:
        if candidate in budgets:
            return dict(budgets[candidate]), candidate
    return {}, None


def apply_dataset_budget(
    attack_cfg: Dict[str, Any],
    budget_cfg: Dict[str, Any],
    *,
    strict_budget: bool = False,
) -> Dict[str, Any]:
    """
    Merge dataset budget into attack config.

    Merge rules:
    - strict_budget=False: only fill missing attack keys.
    - strict_budget=True: dataset budget overrides attack keys.
    """
    merged = dict(attack_cfg or {})
    budget = dict(budget_cfg or {})
    if not budget:
        return merged

    epsilon = _pick(budget, "epsilon", "eps")
    norm = _pick(budget, "norm")
    alpha = _pick(budget, "alpha", "step_size")

    has_eps = ("epsilon" in merged) or ("eps" in merged)
    has_norm = "norm" in merged
    has_alpha = ("alpha" in merged) or ("step_size" in merged)

    if epsilon is not None and (strict_budget or not has_eps):
        merged["epsilon"] = float(epsilon)
    if norm is not None and (strict_budget or not has_norm):
        merged["norm"] = str(norm)
    if alpha is not None and (strict_budget or not has_alpha):
        merged["alpha"] = float(alpha)
    return merged


def resolve_attack_config(
    config: Dict[str, Any],
    *,
    attack_cfg: Dict[str, Any],
    dataset_name: str,
    dataset_aliases: Optional[Iterable[str]] = None,
    strict_budget: bool = False,
) -> Dict[str, Any]:
    """Resolve final attack config with dataset-aware budget policy."""
    budget, matched_key = resolve_dataset_budget(
        config,
        dataset_name,
        aliases=dataset_aliases,
    )
    merged = apply_dataset_budget(
        attack_cfg,
        budget,
        strict_budget=bool(strict_budget),
    )
    if matched_key is not None:
        merged["dataset_budget_source"] = str(matched_key)
    return merged
