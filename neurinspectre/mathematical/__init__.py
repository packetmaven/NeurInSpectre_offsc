"""
NeurInSpectre Mathematical Foundations Module
Advanced mathematical foundations with GPU acceleration for gradient obfuscation detection
"""

from __future__ import annotations

import importlib
from typing import Any

from .gpu_accelerated_math import (
    GPUAcceleratedMathEngine,
    AdvancedExponentialIntegrator,
    demonstrate_advanced_mathematics
)

# Add functions for direct import
from .gpu_accelerated_math import (
    get_engine_info,
    get_precision,
    get_device
)

_LAZY_EXPORTS = {
    # Avoid importing test modules (and their side effects) at package import time.
    'MathematicalFoundationsTestSuite': ('.tests', 'MathematicalFoundationsTestSuite'),
    'run_test_suite': ('.tests', 'run_test_suite'),
    # Layer 2: Volterra memory analysis (imports SciPy; keep lazy)
    'VolterraFitResult': ('.volterra', 'VolterraFitResult'),
    'VolterraKernel': ('.volterra', 'VolterraKernel'),
    'PowerLawKernel': ('.volterra', 'PowerLawKernel'),
    'ExponentialKernel': ('.volterra', 'ExponentialKernel'),
    'UniformKernel': ('.volterra', 'UniformKernel'),
    'fit_volterra_power_law': ('.volterra', 'fit_volterra_power_law'),
    'predict_volterra_power_law': ('.volterra', 'predict_volterra_power_law'),
    'fit_volterra_kernel': ('.volterra', 'fit_volterra_kernel'),
    'compute_volterra_correlation': ('.volterra', 'compute_volterra_correlation'),
    # Layer 3: Krylov projection utilities (SciPy linalg; keep lazy)
    'KrylovStepResult': ('.krylov', 'KrylovStepResult'),
    'analyze_krylov_projection': ('.krylov', 'analyze_krylov_projection'),
}


def __getattr__(name: str) -> Any:
    if name in _LAZY_EXPORTS:
        mod_name, attr = _LAZY_EXPORTS[name]
        mod = importlib.import_module(mod_name, __name__)
        val = getattr(mod, attr)
        globals()[name] = val
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    'GPUAcceleratedMathEngine',
    'AdvancedExponentialIntegrator', 
    'demonstrate_advanced_mathematics',
    'get_engine_info',
    'get_precision',
    'get_device',
    'MathematicalFoundationsTestSuite',
    'run_test_suite',
    'VolterraFitResult',
    'VolterraKernel',
    'PowerLawKernel',
    'ExponentialKernel',
    'UniformKernel',
    'fit_volterra_power_law',
    'predict_volterra_power_law',
    'fit_volterra_kernel',
    'compute_volterra_correlation',
    'KrylovStepResult',
    'analyze_krylov_projection',
] 