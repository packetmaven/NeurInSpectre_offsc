"""Defense transformations for adversarial evaluation."""

from .jpeg_compression import jpeg_defense
from .thermometer_encoding import thermometer_defense
from .randomized_smoothing import randomized_smoothing
from .input_transformations import random_resize_pad, random_crop
from .rl_obfuscation import rl_obfuscation
from .non_differentiable import (
    JPEGCompression,
    BitDepthReduction,
    ThermometerEncoding,
    MedianFilter,
    TotalVariationMinimization,
)
from .stochastic import RandomResizing, RandomPadding, RandomNoise, RandomSmoothing
from .adversarial_training import AdversarialTraining, TRADES, MART, RLAdversarialTraining
from .other import DefensiveDistillation, FeatureSqueezing, GradientMasking, EnsembleDefense
from ..characterization.defense_analyzer import ObfuscationType
from .wrappers import (
    DefenseSpec,
    DefenseWrapper,
    JPEGCompressionDefense,
    BitDepthReductionDefense,
    ThermometerEncodingDefense,
    FeatureSqueezingDefense,
    RandomizedSmoothingDefense,
    RandomPadCropDefense,
    RandomNoiseDefense,
    EnsembleDiversityDefense,
    DefensiveDistillationDefense,
    GradientRegularizationDefense,
    SpatialSmoothingDefense,
    CertifiedDefense,
    ATTransformDefense,
    TotalVariationDefense,
)
from .factory import DefenseFactory

__all__ = [
    "jpeg_defense",
    "thermometer_defense",
    "randomized_smoothing",
    "random_resize_pad",
    "random_crop",
    "rl_obfuscation",
    "JPEGCompression",
    "BitDepthReduction",
    "ThermometerEncoding",
    "MedianFilter",
    "TotalVariationMinimization",
    "RandomResizing",
    "RandomPadding",
    "RandomNoise",
    "RandomSmoothing",
    "AdversarialTraining",
    "TRADES",
    "MART",
    "RLAdversarialTraining",
    "DefensiveDistillation",
    "FeatureSqueezing",
    "GradientMasking",
    "EnsembleDefense",
    "DefenseSpec",
    "DefenseWrapper",
    "JPEGCompressionDefense",
    "BitDepthReductionDefense",
    "ThermometerEncodingDefense",
    "FeatureSqueezingDefense",
    "RandomizedSmoothingDefense",
    "RandomPadCropDefense",
    "RandomNoiseDefense",
    "EnsembleDiversityDefense",
    "DefensiveDistillationDefense",
    "GradientRegularizationDefense",
    "SpatialSmoothingDefense",
    "CertifiedDefense",
    "ATTransformDefense",
    "TotalVariationDefense",
    "ObfuscationType",
    "DefenseFactory",
]
