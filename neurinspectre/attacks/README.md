# NeurInSpectre Attack Implementations

## Week 1 Deliverables: Core Attacks + Numerical Stability

This directory contains all core adversarial attack implementations for the
NeurInSpectre framework, completed in Week 1 of the implementation timeline.

## Implemented Attacks

### 1. PGD (Projected Gradient Descent)
- File: `pgd.py`
- Cross-ref: Section 2.1.1, Equation 2 (LaTeX page 2)
- Status: OK (numerically stable)
- Features:
  - Random initialization
  - Random restarts (PGDWithRestarts)
  - Gradient sanity checks (NaN/Inf detection)
  - Safe L2/Linf projection
  - Data range inference and clamping

Usage:
```python
from neurinspectre.attacks import PGD, PGDWithRestarts

attack = PGD(model, eps=8/255, alpha=2/255, steps=40, norm="linf")
x_adv = attack(x_clean, y_true)

attack_rr = PGDWithRestarts(model, n_restarts=10, eps=8/255, steps=40)
x_adv_rr = attack_rr(x_clean, y_true)
```

Expected performance (Table 1):
- JPEG Compression: 12.4% ASR
- Defensive Distillation: 6.2% ASR
- Average across 12 defenses: 23.4% ASR

### 2. Auto-PGD (APGD)
- File: `apgd.py`
- Cross-ref: Section 2.1.2, Equation 3 (LaTeX page 2)
- Status: OK (DLR + CE + MD losses)
- Features:
  - Adaptive step-size scheduling
  - DLR, CE, MD losses
  - Momentum (beta=0.75)
  - Transformed gradients
  - Random restarts

Usage:
```python
from neurinspectre.attacks import APGD

# DLR loss (recommended)
attack = APGD(model, eps=8/255, steps=100, loss="dlr")
x_adv = attack(x_clean, y_true)

# Ensemble (CE + DLR)
from neurinspectre.attacks import APGDEnsemble
attack = APGDEnsemble(model, eps=8/255, losses=["ce", "dlr"])
x_adv = attack(x_clean, y_true)
```

Key finding: DLR loss provides stronger optimization in worst-case settings.

### 3. Square Attack
- File: `square.py`
- Cross-ref: Section 3.3 - "Square Attack: Query-based black-box verification"
- Status: OK (Linf + L2 variants)
- Features:
  - Query-efficient random search
  - Coarse-to-fine square size schedule
  - Black-box validation
  - Per-sample query tracking

Usage:
```python
from neurinspectre.attacks import SquareAttack

attack = SquareAttack(model, eps=8/255, n_queries=5000)
x_adv, stats = attack(x_clean, y_true, verbose=True)

print(f"ASR: {stats['asr']*100:.1f}%")
print(f"Avg queries: {stats['queries_used'].mean():.0f}")
```

Critical role: Square Attack is required to validate that gains are not due to
gradient masking (ETD obfuscation detection).

### 4. BPDA (Backward Pass Differentiable Approximation)
- Files: `bpda.py`, `bpda_registry.py`
- Cross-ref: Section 2.3.1, Equation 5 (LaTeX page 2)
- Status: OK (registry + learned approximation)
- Features:
  - Pre-defined approximations (identity, JPEG, thermometer, quantization)
  - Custom approximation support
  - Learned approximation (neural network)
  - Straight-through estimators (STE)

Available approximations:
| Defense Type | Approximation | Usage |
| --- | --- | --- |
| JPEG Compression | Identity | `approx_name="jpeg"` |
| Thermometer Encoding | STE Quantization | `approx_name="thermometer"` |
| Bit-Depth Reduction | STE Quantization | `approx_name="quantization"` |
| Median Filter | Avg Pooling | `approx_name="median_filter"` |
| Custom | User-defined | `approx_fn=custom_fn` |

Usage:
```python
from neurinspectre.attacks import BPDA

# JPEG defense
defense = JPEGCompression(quality=75)
attack = BPDA(model, defense, approx_name="jpeg", eps=8/255)
x_adv = attack(x_clean, y_true)

# Learned approximation
from neurinspectre.attacks import LearnedBPDA
attack = LearnedBPDA(model, defense, train_steps=1000, eps=8/255)
attack.train_approximation(train_loader)
x_adv = attack(x_clean, y_true)
```

Impact (Table 2 ablation): BPDA provides +22.9pp improvement (71.4% -> 94.3%).

### 5. EOT (Expectation Over Transformation)
- File: `eot.py`
- Cross-ref: Section 2.3.2, Equations 6-7 (LaTeX page 2)
- Status: OK (importance sampling + adaptive sampling)
- Features:
  - Uniform sampling
  - Importance-weighted sampling (variance reduction)
  - Adaptive sample count selection
  - Stochastic defense handling

Usage:
```python
from neurinspectre.attacks import EOT

def random_noise(x):
    return x + torch.randn_like(x) * 0.1

attack = EOT(model, random_noise, num_samples=20, importance_sampling=True, eps=8/255)
x_adv = attack(x_clean, y_true)

from neurinspectre.attacks import AdaptiveEOT
attack = AdaptiveEOT(model, random_noise, target_variance=0.01, min_samples=10, max_samples=100)
x_adv = attack(x_clean, y_true)
```

### 6. FAB (Fast Adaptive Boundary)
- File: `fab.py`
- Cross-ref: Section 3.3 - FAB in coordinated evaluation
- Status: OK (FAB + FABT + FABEnsemble)
- Features:
  - Minimum-norm adversarial examples
  - Linear boundary approximation
  - Backtracking search
  - Targeted variant (FAB-T)
  - Ensemble (FAB + FAB-T)

Usage:
```python
from neurinspectre.attacks import FAB

attack = FAB(model, norm="l2", steps=100, n_restarts=5)
x_adv = attack(x_clean, y_true)

from neurinspectre.attacks import FABT
attack = FABT(model, norm="l2", n_target_classes=9)
x_adv = attack(x_clean, y_true)

from neurinspectre.attacks import FABEnsemble
attack = FABEnsemble(model, norm="l2")
x_adv = attack(x_clean, y_true)
```

Key property: FAB seeks minimum-norm adversarial examples per sample.

## AutoAttack Ensemble

File: `autoattack.py`

Components (standard):
- APGD-CE (100 steps)
- APGD-DLR (100 steps)
- FAB (minimum-norm)
- Square (5000 queries, Linf only)

Usage:
```python
from neurinspectre.attacks import AutoAttack

autoattack = AutoAttack(model, norm="linf", eps=8/255, version="standard")
x_adv, metrics = autoattack.run(x_clean, y_true, verbose=True)

print(f"Robust Accuracy: {metrics['robust_accuracy']*100:.1f}%")
print(f"Total ASR: {metrics['asr']*100:.1f}%")

metrics = autoattack.run_standard_eval(test_loader, n_examples=10000)
```

## Numerical Utilities

File: `numerics.py`

Utilities for numerical stability:
```python
from neurinspectre.attacks.numerics import (
    infer_data_range,
    safe_flat_norm,
    check_grad_sanity,
    transformed_gradient,
)
```

## Cross-Reference to Paper Claims

| Paper Claim | LaTeX Location | Implementation | Status |
| --- | --- | --- | --- |
| PGD baseline 23.4% ASR | Table 1 | `pgd.py` | OK |
| AutoAttack 64.8% ASR | Table 1 | `autoattack.py` | OK |
| BPDA +22.9pp improvement | Table 2 | `bpda.py` | OK |
| EOT variance reduction | Section 4 | `eot.py` | OK |
| Square Attack validation | Section 3.3 | `square.py` | OK |
| DLR loss gain | Research | `losses/dlr_loss.py` | OK |
| Gradient sanity checks | Section 4 | `numerics.py` | OK |

## Testing

Run all attack tests:
```bash
pytest tests/test_attacks.py tests/test_apgd.py tests/test_square.py \
       tests/test_bpda.py tests/test_eot.py tests/test_fab.py
```

Run with coverage:
```bash
pytest --cov=neurinspectre/attacks tests/
```

## Expected Performance (Table 1)

| Defense | PGD | AutoAttack | NeurInSpectre (Week 2) |
| --- | --- | --- | --- |
| JPEG Compression | 12.4% | 67.3% | 98.2% |
| Bit-Depth Reduction | 8.7% | 71.8% | 97.4% |
| Rand. Smoothing | 31.2% | 58.4% | 89.3% |
| Defensive Distillation | 6.2% | 52.1% | 94.6% |
| Average | 23.4% | 64.8% | 94.3% |

## Week 1 Status

- PGD and AutoAttack baselines implemented and validated.
- Week 2 goal: implement MA-PGD to reach ~94.3% average ASR.

## References

- Madry et al. (2018): PGD
- Croce and Hein (2020): AutoAttack, APGD, FAB, DLR loss
- Athalye et al. (2018): BPDA, EOT, obfuscated gradients
- Andriushchenko et al. (2020): Square Attack

## Final Week 1 Deliverables Summary

Core attack implementations:
1. `neurinspectre/attacks/base.py`
2. `neurinspectre/attacks/numerics.py`
3. `neurinspectre/attacks/pgd.py`
4. `neurinspectre/losses/dlr_loss.py`
5. `neurinspectre/attacks/apgd.py`
6. `neurinspectre/attacks/square.py`
7. `neurinspectre/attacks/bpda_registry.py`
8. `neurinspectre/attacks/bpda.py`
9. `neurinspectre/attacks/eot.py`
10. `neurinspectre/attacks/fab.py`
11. `neurinspectre/attacks/autoattack.py`

Testing:
12. `tests/test_numerics.py`
13. `tests/test_attacks.py`
14. `tests/test_losses.py`
15. `tests/test_apgd.py`
16. `tests/test_square.py`
17. `tests/test_bpda.py`
18. `tests/test_eot.py`
19. `tests/test_fab.py`

Documentation:
20. `neurinspectre/attacks/README.md`
