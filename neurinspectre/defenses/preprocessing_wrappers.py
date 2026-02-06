"""
Compatibility re-exports for defense wrappers.
"""

from __future__ import annotations

from .wrappers import (
    DefenseSpec,
    DefenseWrapper,
    JPEGCompressionDefense,
    BitDepthReductionDefense,
    ThermometerEncodingDefense,
)

__all__ = [
    "DefenseSpec",
    "DefenseWrapper",
    "JPEGCompressionDefense",
    "BitDepthReductionDefense",
    "ThermometerEncodingDefense",
]
