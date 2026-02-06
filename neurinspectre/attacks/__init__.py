"""
NeurInSpectre Attacks Module
Advanced offensive security attack implementations.
"""

from .ednn_attack import EDNNAttack, EDNNConfig, load_embedding_model
from .base import Attack
from .pgd import PGD, PGDWithRestarts
from .apgd import APGD, APGDEnsemble
from .fab import FAB, FABT, FABEnsemble
from .square import SquareAttack, SquareAttackL2
from .bpda import BPDA, LearnedBPDA
from .eot import EOT, AdaptiveEOT
from .memory_augmented import MemoryAugmentedPGD
from .ma_pgd import MAPGD, MAPGDEnsemble
from .temporal_momentum import TemporalMomentumPGD
from .autoattack import AutoAttackEnsemble, AutoAttack
from .attack_orchestrator import AttackOrchestrator, attack_with_characterization
from .hybrid import HybridBPDAEOT
from .factory import AttackFactory, LossFunction
from .base_interface import (
    AttackCapability,
    AttackConfig,
    AttackResult,
    BaseAdversarialAttack,
    GradientBasedAttack,
    PGDAttack,
    APGDAttack,
    APGDTargeted,
)

__all__ = [
    "Attack",
    "PGD",
    "PGDWithRestarts",
    "APGD",
    "APGDEnsemble",
    "FAB",
    "FABT",
    "FABEnsemble",
    "SquareAttack",
    "SquareAttackL2",
    "BPDA",
    "LearnedBPDA",
    "EOT",
    "AdaptiveEOT",
    "MemoryAugmentedPGD",
    "MAPGD",
    "MAPGDEnsemble",
    "TemporalMomentumPGD",
    "AutoAttackEnsemble",
    "AutoAttack",
    "AttackOrchestrator",
    "attack_with_characterization",
    "HybridBPDAEOT",
    "AttackFactory",
    "AttackResult",
    "AttackConfig",
    "LossFunction",
    "AttackCapability",
    "BaseAdversarialAttack",
    "GradientBasedAttack",
    "PGDAttack",
    "APGDAttack",
    "APGDTargeted",
    "EDNNAttack",
    "EDNNConfig",
    "load_embedding_model",
]

