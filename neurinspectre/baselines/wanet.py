"""
WaNet baseline wrapper.

WaNet (Nguyen & Tran, ICLR 2021) is an imperceptible warping-based backdoor
attack. NeurInSpectre implements it as a training-stage backdoor baseline in
`neurinspectre.baselines.backdoor`.

For end-to-end runs (poisoning + training + ASR measurement), prefer:
  `neurinspectre baselines subnetwork-hijack run --baseline wanet ...`
"""

from __future__ import annotations

from .backdoor import WaNetTrigger

__all__ = ["WaNetTrigger"]

