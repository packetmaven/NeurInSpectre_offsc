"""
Baseline implementations and adapters for NeurInSpectre.

Policy:
- We do not ship hard-coded "paper baseline" numbers in-repo.
- We *do* provide baseline *methods* (or adapters to published baseline toolkits)
  so evaluators can run apples-to-apples comparisons on real data.
"""

from __future__ import annotations

__all__ = [
    "bae",
    "backdoor",
    "frameworks_compare",
    "gradinversion",
    "prompt_injection",
    "textfooler",
    "text_attacks",
    "wanet",
]

