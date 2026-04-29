# Draft↔Code Consistency Audit (Working Notes)

This document is a living *claim-to-implementation* map for the WOOT'26 draft
text (as pasted into chat) against the current local codebase.

Scope notes:
- This repo intentionally does **not** include the submission PDF (`woot.pdf`).
  For paper citations, reference section/figure/table identifiers, or keep a local
  (uncommitted) PDF copy for page/line lookups.
- The goal is **mechanical traceability**: each equation/table/CLI claim maps to
  concrete code paths and concrete JSON artifacts.
- Commands assume the repo venv is active: `source .venv-neurinspectre/bin/activate`
  (otherwise prefix with `./.venv-neurinspectre/bin/`).

## Pinned Context (keep if chat resets)

Directive from user (Feb 2026):
- Start the paper↔code audit immediately.
- After the audit, continue with full paper table reproduction + remaining Tier work.

What still needs doing (persistent checklist):
- [x] Full Table 8 reproduction (top priority)
  - [x] Run `neurinspectre table2 -c table2_config.yaml` end-to-end for all 12 defenses and archive `results/*/summary.json`.
  - [x] Reconcile numerical mismatches vs the draft (e.g., JPEG row AutoAttack ASR).
  - [x] Decide mismatch root cause(s): defense wiring/implementation, model artifact differences, metric-definition differences.
  - Status: complete in `results/table8_run_v2/` (12/12 defenses include `pgd` + `autoattack` + `neurinspectre`; chain completion in `results/table8_run_v2/monitor_table8_v2.log`).
- [x] Real dataset + model asset readiness (prereq for strict-real-data table runs)
  - [x] Ensure local paths/availability for CIFAR-10, EMBER (vectorized memmaps), and nuScenes (the datasets used by `table2_config.yaml` / Table 8).
  - [x] Ensure the exact model artifacts referenced by configs exist and are pinned (hash/meta).
  - [x] Add/verify download + preprocessing commands/scripts where missing so a reviewer can reproduce from scratch.
  - Status: verified local assets under `data/` (CIFAR-10 `cifar10/cifar-10-batches-py/test_batch`, EMBER `ember/ember_2018/X_test.dat`+`y_test.dat`, nuScenes `nuscenes/v1.0-mini/scene.json` + `nuscenes/samples/CAM_FRONT/*.jpg` + `nuscenes/label_map.json`). Model SHA256 pins exist in `models/*.meta.json` (+ legacy `results/audit_cli/model_sha256.txt`). ImageNet-100 is tracked separately and currently uses a placeholder stub model.
- [x] Characterization engine credibility
  - [x] Debug/fix degenerate characterization outputs in runs (zeros/NaNs in spectral/Volterra fields).
  - [x] Ensure paper thresholds/logic (entropy, RHF, alpha_volterra, wavelets) are evidenced by JSON artifacts (see `results/audit_jpeg_neurinspectre_v2/cm_jpeg_compression.json`).
- [x] CLI / scripts claimed by the paper
  - [x] Verify every CLI invocation shown in `REPRODUCE.md` / the draft runs and produces the claimed fields/format (see `results/audit_cli/`).
  - [x] Make `scripts/reproduce_all.sh` generate the WOOT-checkable table artifacts and finish any missing “single command per table” mapping.
  - Status: done. Table 8-only wrapper: `bash scripts/reproduce_table8.sh` (see `AE.md`). Harness emits:
    - `table8_table2/table8_validation.json`
    - `table8_table2/table10_attack_strength.{json,tex}` (paper-facing Table 10-style aggregate derived from `summary.json`)
    - `table8_table2_artifacts.tgz`
    - `table_command_map.md`
    - recursive `sha256_manifest.txt`
    Verified run: `results/repro_harness_table8_latest3/`.
- [x] Devil’s-advocate draft↔code audit
  - [x] Expand this file into a complete claim→code→command→artifact matrix for every equation/threshold, every table cell definition, and every “we do X” systems claim.
  - [x] For each mismatch, decide “implement in code” vs “edit paper to match reality.”
  - Status: complete for current artifact (`Claim→Code→Command→Artifact Matrix` + resolved `Implement-vs-Edit Decision Register` below).
- [x] Tier 1/2/3 implementation guides
  - [x] Write stepwise Tier 1/2/3 guides grounded in the actual repo structure and verified commands.
  - Status: `docs/guides/TIER_IMPLEMENTATION_GUIDE.md` added with copy/paste commands and expected artifacts.

Current status notes (live, update as we go):
- Full Table 8 run (12 defenses x 3 attacks) is complete in `results/table8_run_v2/`.
- AutoAttack (JPEG row) mismatch triage:
  - Final artifact: `results/table8_run_v2/cm_jpeg_compression.json` includes `attacks.pgd`, `attacks.autoattack`, and `attacks.neurinspectre`.
  - Root cause of “looks stuck” behavior: `SquareAttack` previously re-evaluated the *entire* batch every query even after many samples had already succeeded, making the AutoAttack stage effectively intractable for slow/non-differentiable defenses (e.g., PIL JPEG).
    - Fix (implemented): `neurinspectre/attacks/square.py` now evaluates only the remaining active subset each query (semantics-preserving speedup).
    - Rerun log (post-fix): `results/table8_run_v2/stage_autoattack_cm_jpeg.log`
- `ensemble_diversity` config compatibility: accept `voting: majority` (mapped to `aggregation: vote`) so the matrix run does not crash.
- Table2 strict validity gate previously *aborted* the matrix at `cm_random_smoothing` because clean accuracy under smoothing collapsed near chance (`clean_acc≈0.116 < 0.15` required by `min_clean_accuracy_over_chance=0.05`).
  - Fix: set `validity_gates.strict: false` in `table2_config.yaml` so the full matrix completes while still recording per-pair `validity` metadata.
  - Resume (stage-by-stage into the same output dir):
    - AutoAttack: `neurinspectre table2 -c table2_config.yaml -o results/table8_run_v2 --attacks autoattack --resume --strict-real-data --strict-dataset-budgets --no-report --no-progress`
    - NeurInSpectre: `neurinspectre table2 -c table2_config.yaml -o results/table8_run_v2 --attacks neurinspectre --resume --strict-real-data --strict-dataset-budgets --no-report --no-progress`
- Major Table 8/EMBER bug found + fixed: gradient-based attacks clamped to a *stale, first-batch* auto-detected input range.
  - Symptom: On EMBER (tabular, heavy-tailed features), later batches contained values outside the initial range, so `_project()` clamped `x_adv` (e.g., down to `_input_max`), producing **budget-violating perturbations** and nonsense perturbation stats (e.g., `l2_mean≈3.1e7`).
  - Root cause: `GradientBasedAttack._detect_and_set_range()` returned early once `_range_detected=True`, so the clamp range never expanded across batches.
  - Fix (implemented): expand the detected range monotonically to include new extrema, and call `_detect_and_set_range(x)` on every projection/random-init.
    - Code: `neurinspectre/attacks/base_interface.py` (`GradientBasedAttack._detect_and_set_range`, `_project`, `_l2_ball_random_init`)
  - Evidence:
    - Before: `results/table8_run_v1/md_feature_squeezing.json` had invalid perturbation magnitudes.
    - After: reran PGD stage into `results/table8_run_v2/`; `results/table8_run_v2/md_feature_squeezing.json` shows `l2_max=0.5`, `l2_mean≈0.499` (correct for `ember` budget `eps=0.5, norm=l2`).
- `table2-smoke` robustness fix:
  - Failure mode: some TorchScript models raise `RuntimeError` (not `TypeError`) when passed `use_approximation=...`,
    crashing `DefenseAnalyzer` during BPDA-kwarg detection.
  - Fix: treat `RuntimeError` the same as `TypeError` in `DefenseAnalyzer._forward_logits` and BPDA-kwarg probing.
  - Evidence: `neurinspectre table2-smoke --output-dir results/audit_cli/smoke2` completes (see `results/audit_cli/table2_smoke2.log`).

Devil’s-advocate rule:
- If code behavior is a stub/simplification, the paper must either (a) downgrade the claim explicitly,
  or (b) code must be upgraded to match the claim.

## High-Risk Reviewer Issues (must resolve for >95% acceptance)

### Critical: several “Table 8” defenses are training-time defenses (require checkpoints)

The draft frames the 12 defenses as representative defenses across domains. In the current codebase,
some defenses are fundamentally *training-time* defenses (their runtime wrapper is minimal), so a
meaningful evaluation requires defense-specific trained checkpoints.

Examples:
- `GradientRegularizationDefense` runtime wrapper is a no-op; the defense signal lives in the checkpoint.
  - Wrapper code: `neurinspectre/defenses/wrappers.py` (`GradientRegularizationDefense`)
  - Training/export: `neurinspectre/training/ember_defenses.py`, `scripts/train_ember_defense_models.py`
- `DefensiveDistillationDefense` runtime wrapper does not implement training; distillation is a training pipeline.
  - Wrapper code: `neurinspectre/defenses/wrappers.py` (`DefensiveDistillationDefense`)
  - Training/export: `neurinspectre/training/ember_defenses.py`, `scripts/train_ember_defense_models.py`
- `ATTransformDefense` is currently modeled as stochastic noise injection at inference time; the parity training script
  implements AT+transforms for EMBER (tabular) using noise + L2 PGD-k during training.
  - Wrapper code: `neurinspectre/defenses/wrappers.py` (`ATTransformDefense`)
  - Training/export: `neurinspectre/training/ember_defenses.py`, `scripts/train_ember_defense_models.py`
- `CertifiedDefense` is a randomized-smoothing style wrapper (Gaussian noise + Monte Carlo averaging) with a diagnostic
  `certified_radius()` helper; it is not a full certification pipeline.
  - Code: `neurinspectre/defenses/wrappers.py` (`CertifiedDefense`)

Impact:
- If per-defense model artifacts are missing, `table2` falls back to per-dataset models and evaluates wrappers/transforms.
- When artifacts exist, `table2` resolves spec `checkpoint_tag` → runnable per-defense `model_path`, so training-time defenses
  can be evaluated with their intended checkpoints.
- Update (EMBER parity evidence): per-defense EMBER TorchScript checkpoints now exist and resolve correctly:
  - Artifacts: `models/md_gradient_reg_ember_ts.pt`, `models/md_distillation_ember_ts.pt`, `models/md_at_transform_ember_ts.pt`
  - Resolved config evidence: `results/table8_ember_models_v1/resolved_table2_config.yaml` shows the per-defense `model_path`s.
  - Evaluation evidence: `results/table8_ember_models_v1/*.json` (ASR remains 0.0 under the current EMBER budget; clean accuracy differs vs the dataset-default model).

### Critical: Layer 1 characterization degenerates on non-differentiable defenses

The draft’s Layer 1 describes analyzing gradient sequences (spectral entropy, RHF, wavelets).
In current code, the gradient sampling path used in the core pipeline can produce degenerate all-zero gradients
for non-differentiable transforms (e.g., PIL JPEG), because the transform breaks autograd.

- Path: `DefenseAnalyzer.characterize()` → `_collect_gradient_samples()`
  - Code: `neurinspectre/characterization/defense_analyzer.py`
- Symptom: when `delta.grad is None`, the analyzer records a zero gradient vector for that step.

Consequences (before fix):
- `spectral_entropy_norm` / `high_freq_ratio` could collapse to 0.0 for shattered defenses, making the paper’s
  Layer‑1 threshold narrative impossible to audit from JSON artifacts.

Consequences (after fix):
- Artifacts now record when BPDA fallback was used and export non-degenerate Layer‑1 scalars for shattered defenses
  (see `metadata.gradient_sampling` + Layer‑1 fields in evidence artifacts below).

Fix (implemented locally):
- Updated `neurinspectre/characterization/defense_analyzer.py` to:
  - detect *both* `grad is None` and the “silent all-zero grad” case (when `delta.grad` is already allocated but the
    graph does not depend on the input, leaving `delta.grad` unchanged at zeros),
  - fall back to a BPDA-style forward (`use_approximation=True`) to obtain a meaningful gradient history for spectral
    + Volterra fitting,
  - record explicit `metadata.gradient_sampling.{true_grad_none_fraction,true_grad_zero_fraction,bpda_fallback_fraction}`
    so artifacts explain when BPDA was required (instead of silently returning zeros).
- Evidence artifact:
  - `results/characterize/jpeg_char_v2.json`
  - Shows non-degenerate Layer-1 scalars (e.g., `spectral_entropy_norm=0.760...`, `high_freq_ratio=0.189...`) and
    correctly flags the defense as requiring BPDA (`obfuscation_types` includes `shattered`, `requires_bpda=true`).
  - Table2 pipeline evidence (thresholds + gradient_sampling; v2 also includes wavelets):
    - `results/audit_jpeg_neurinspectre/cm_jpeg_compression.json`
    - `results/audit_jpeg_neurinspectre_v2/cm_jpeg_compression.json` (includes `characterization.metadata.wavelet_energy`)

### Major: AutoAttack description mismatch (and dependency mismatch)

Draft claim:
- AutoAttack ensemble includes “NeurInSpectre’s synthesized attack”.

Current code:
- `neurinspectre/attacks/autoattack.py` uses APGD-CE + APGD-DLR + FABEnsemble (+ SquareAttack for Linf).
- It does not include NeurInSpectre’s adaptive runner inside AutoAttack.
- The repo declares an external `autoattack` dependency in `pyproject.toml`, but the core evaluation path
  does not import it (it uses the in-repo `AutoAttack` implementation).

### Major: statistical evasion math parity implemented

Draft claim (Section 5.5):
- KS/AD/CvM per-dimension + Fisher aggregation + BH/FDR correction + iterative “minimize KS” evasion loop.

Current code:
- Per-dimension detector + Fisher + BH/FDR:
  - Code: `neurinspectre/statistical/drift_detection_enhanced.py` (`PerDimKSDADCvMFisherBHDriftDetector`, `_bh_fdr_adjust`, `_fisher_combine_p_values`)
  - Drift-detect method key: `ks_ad_cvm_fisher_bh`
- Iterative evasion loop:
  - Code: `neurinspectre/statistical/evasion.py` (`iterative_evasion_loop`)
- Note: legacy `ks_ad_cvm` remains (PCA1 + Bonferroni) for back-compat.

### Critical: activation steganography ECC implemented

Draft claim (Section 5.2):
- activation encoding with quantitative bits/neuron.

Current code:
- ECC-backed encoding is implemented and wired:
  - Code: `neurinspectre/ecc_activation_steganography.py` (Hamming(7,4))
  - Wiring: `neurinspectre/activation_steganography.py` (`ECC_AVAILABLE=True` when import succeeds)
  - The encode path emits an explicit `[STEG_ECC:...]` marker so reviewers can audit what was encoded.

## Core Table (Draft “Table 8”) Reproduction

### “PGD-20 / AutoAttack / NeurInSpectre” columns

- Runner entrypoint: `neurinspectre table2` → `neurinspectre/cli/table2_cmd.py`
  → `neurinspectre/cli/evaluate_cmd.py`
- Attack implementations:
  - `pgd`: `neurinspectre/attacks/base_interface.py` (`PGDAttack`)
  - `autoattack`: `neurinspectre/attacks/autoattack.py` (`AutoAttack`)
    - Default "standard" in-repo ensemble: APGD-CE (100 steps, 1 restart) + APGD-DLR (100, 1) + FAB + Square (Linf, 5000 queries).
    - Non-image inputs (e.g., EMBER tabular): runs APGD-only subset (`x.ndim != 4` guard) for compatibility.
  - `neurinspectre`: `neurinspectre/attacks/factory.py` (`_NeurInSpectreRunner`)
    selecting BPDA/EOT/Hybrid/MA-PGD based on characterization
- Metrics + JSON fields:
  - ASR: `results/*/summary.json` → `results[].attacks.<attack>.attack_success_rate`
  - Robust accuracy: `results[].attacks.<attack>.robust_accuracy`
  - ASR definition detail: `attack_success_rate` is computed over the originally-correct subset
    (conditional ASR; standard in many robustness papers) via
    `neurinspectre/evaluation/metrics.py` (`compute_attack_success_rate`).
  - Additional exported metrics (to disambiguate paper↔code comparisons):
    - `attack_success_rate_overall`: `success_samples / samples` (successes among clean-correct, over total)
    - `compromise_rate`: `1 - robust_accuracy` (overall post-attack misclassification rate)

### Current observed artifact vs draft Table 8 (example: JPEG row)

Quick local subset run:
- Artifact: `results/table8_jpeg_quick_fixed/summary.json`
- Defense: `cm_jpeg_compression` (CIFAR-10, eps=8/255, Linf)
- Observed ASR:
  - PGD-20: 7.55%
  - AutoAttack: 98.11%
  - NeurInSpectre: 99.37%

Current Table 8 v2 run:
- Artifact: `results/table8_run_v2/cm_jpeg_compression.json`
  - PGD-20 ASR: conditional 9.3% (overall 7.4%; robust_acc 72.2%; clean_acc 79.6%)
  - AutoAttack ASR: conditional 98.4% (overall 78.3%; robust_acc 1.3%)
  - NeurInSpectre ASR: conditional 99.5% (overall 79.2%; robust_acc 0.4%)

Draft Table 8 (JPEG Compression row) claims:
- PGD-20 12.4%, AutoAttack 67.3%, NeurInSpectre 98.2%

Status:
- Full matrix reproduction is complete (12 defenses x 3 attacks) with per-defense artifacts in `results/table8_run_v2/*.json`.
- JPEG mismatch remains large for AutoAttack (draft 67.3% vs run conditional 98.4%; overall 78.3%).
- Root-cause decision:
  - **Defense/model wiring (primary):** the `table8_run_v2` artifacts were produced with per-dataset TorchScript models
    (per-defense checkpoint artifacts were not available, so Table2 spec `checkpoint_tag`s could not be resolved at the time).
    The runnable pipeline now supports per-defense `model_path` resolution from `checkpoint_tag`/`path`.
  - **Metric definition (secondary):** conditional ASR (`attack_success_rate`) vs overall ASR (`attack_success_rate_overall`) creates large apparent shifts.
  - **Attack definition (secondary):** in-repo AutoAttack ensemble/settings may not match the draft's assumed AutoAttack package/configuration.

Reconciliation checklist (mechanical):
- ASR definition: compare `attack_success_rate` (conditional) vs `attack_success_rate_overall` for the same artifact.
- Model artifact identity: record SHA256 / meta for the exact checkpoints used in the run.
  - SHA256 pinning for the shipped TorchScript artifacts is in `results/audit_cli/model_sha256.txt`.
- Defense params: confirm JPEG `quality` and any pre/post-processing matches the appendix and the resolved config.
- AutoAttack implementation: clarify whether the paper used the external `autoattack` package or the in-repo ensemble
  (component set, restarts, and query budget differences can easily shift ASR).
  - In-repo details (for mechanical comparisons): default `version="standard"` uses APGD restarts=1 and Square queries=5000 (Linf);
    for non-image inputs it runs APGD-only (no FAB/Square).
- Evaluation sampling: confirm `num_samples`, seed(s), and whether evaluation shuffles (`shuffle_eval`) are aligned.
  - Current `results/table8_run_v2/` is a single-seed run (`seed=42`, see `summary.json`).
  - Paper-grade reporting should run multiple independent seeds (e.g., `--num-seeds 5`) and report mean ± std + 95% CI.

### Config used for Table-style runs

- Spec config: `table2_config.yaml`
  - `defaults.pgd.steps` is set to **20** (draft “PGD-20”).
  - Defense parameters were aligned to the draft Appendix defense settings
    where possible (e.g., smoothing samples, distillation temperature, etc.).
- Important normalization caveat (spec vs runnable):
  - `table2_config.yaml` includes some spec-only metadata (e.g., per-defense `model.factory_key` and per-attack
    `module` / `class_name`) that is not used by the runnable pipeline.
  - The runnable pipeline *does* use `model.path` and can resolve `model.checkpoint_tag` to a concrete on-disk
    `model_path` when a matching artifact exists under `models/` (with an audit-friendly fallback to per-dataset models).
  - The runnable config is `<out_dir>/resolved_table2_config.yaml`, produced by `neurinspectre/cli/table2_cmd.py`
    (`_normalize_table2_spec`), which:
    - resolves attacks purely by `name` (`pgd`, `autoattack`, `neurinspectre`),
    - prefers per-defense `defenses[].model_path` when present (resolved from spec `model.path` / `model.checkpoint_tag`),
    - falls back to per-dataset `models.<dataset>` when a per-defense artifact is not available.
  - Evidence (historical, v2 run used per-dataset fallbacks): `results/table8_run_v2/resolved_table2_config.yaml` sets:
    - `models.cifar10 = models/cifar10_resnet20_norm_ts.pt`
    - `models.ember = models/ember_mlp_ts.pt`
    - `models.nuscenes = models/nuscenes_resnet18_trained.pt`

### Table 8 v2 status (full 12x3 complete)

Extracted from `results/table8_run_v2/*.json` (conditional ASR; full per-attack metric blocks are in each per-defense file):

| dataset | defense | clean_acc | pgd20_asr(cond) | autoattack_asr(cond) | neurinspectre_asr(cond) | validity_passed |
|---|---|---:|---:|---:|---:|---|
| cifar10 | cm_bit_depth_reduction | 88.5% | 9.4% | 99.0% | 100.0% | true |
| cifar10 | cm_ensemble_diversity | 91.9% | 6.2% | 33.5% | 100.0% | true |
| cifar10 | cm_jpeg_compression | 79.6% | 9.3% | 98.4% | 99.5% | true |
| cifar10 | cm_random_smoothing | 11.9% | 72.3% | 25.9% | 20.2% | false |
| ember | md_at_transform | 53.6% | 0.0% | 0.0% | 0.0% | false |
| ember | md_defensive_distillation | 70.9% | 0.0% | 0.0% | 0.0% | true |
| ember | md_feature_squeezing | 41.6% | 0.0% | 0.0% | 0.0% | false |
| ember | md_gradient_regularization | 70.9% | 0.0% | 0.0% | 0.0% | true |
| nuscenes | av_certified_defense | 2.5% | 0.0% | 0.0% | 10.9% | false |
| nuscenes | av_random_pad_crop | 62.5% | 44.0% | 100.0% | 100.0% | true |
| nuscenes | av_spatial_smoothing | 55.0% | 93.2% | 100.0% | 87.0% | true |
| nuscenes | av_thermometer_encoding | 65.0% | 11.5% | 19.2% | 100.0% | true |

Averages (conditional ASR):
- All 12 defenses: PGD=20.5%, AutoAttack=39.7%, NeurInSpectre=51.5%
- Valid-only (`validity_passed=true`, 8/12): PGD=21.7%, AutoAttack=56.3%, NeurInSpectre=73.3%

EMBER note (important for paper claims):
- EMBER rows are 0.0% ASR for all three attack suites under the current threat model (`norm=l2`, `eps=0.5`)
  in the canonical Table 8 v2 run.
  - Evidence (per-defense artifacts under `results/table8_run_v2/`):
    - `md_defensive_distillation.json` (`validity.passed=true`): `attacks.{pgd,autoattack,neurinspectre}.attack_success_rate = 0.0`
    - `md_gradient_regularization.json` (`validity.passed=true`): `attacks.{pgd,autoattack,neurinspectre}.attack_success_rate = 0.0`
    - `md_at_transform.json` (`validity.passed=false` due to clean accuracy: `clean_acc=0.536`): ASR also 0.0
    - `md_feature_squeezing.json` (`validity.passed=false` due to clean accuracy: `clean_acc=0.416`): ASR also 0.0
  - Table 10 implication: “EMBER (n=2)” (valid-only) aggregates to **0.0** for PGD/AutoAttack/NeurInSpectre.
  - Nuance: AutoAttack perturbation norms are 0.0 when ASR=0.0 because the implementation keeps `x_adv == x` for unsuccessful samples;
    PGD/NeurInSpectre still return budget-sized (but unsuccessful) candidates, so their perturbation norms can sit near `eps` even when ASR stays 0.0.
- This is not just a “model fallback” artifact: re-running the EMBER subset with per-defense trained checkpoints
  (resolved via `checkpoint_tag` → `model_path`) still yields 0.0% ASR:
  - Run: `results/table8_ember_models_v1/` (`md_feature_squeezing`, `md_gradient_regularization`, `md_defensive_distillation`, `md_at_transform`)
  - Evidence: `results/table8_ember_models_v1/*.json`
- Paper implication: avoid universal statements like “all 12 systems fall below X% under adaptive attack” unless the EMBER threat
  model (feature scaling + epsilon) is revised and rerun.

Archiving (for AE / reviewer upload):
- Minimal bundle:
  - `tar -czf results/table8_run_v2_minimal.tgz -C results/table8_run_v2 summary.json resolved_table2_config.yaml run_metadata.json *.json`
- Full bundle:
  - `tar -czf results/table8_run_v2_full.tgz -C results/table8_run_v2 summary.json resolved_table2_config.yaml run_metadata.json *.json *.log`
  - Generated locally:
    - `results/table8_run_v2_minimal.tgz`
    - `results/table8_run_v2_full.tgz`

Multi-seed (paper-grade reporting):
- `neurinspectre table2 ... --num-seeds 5` runs independent seeds and reports mean ± std + 95% CI.
- Harness equivalent: `TABLE2_NUM_SEEDS=5 bash scripts/reproduce_table8.sh`
- Output structure: `results/<run>/seed_<seed>/...` plus an aggregated top-level `summary.json`.
- Recommended sanity check (single row, 3 seeds):
  - `neurinspectre table2 -c table2_config.yaml -o results/table8_multiseed_smoke --defenses cm_jpeg_compression --attacks pgd --num-seeds 3 --strict-real-data --strict-dataset-budgets --no-report --no-progress`
  - Executed (evidence): `results/table8_multiseed_smoke/summary.json` includes `attack_success_rate_std` and `attack_success_rate_ci95_{low,high}`,
    plus `attack_success_rate_by_seed` for each seed under `results[].attacks.<attack>`.

### Example artifact (single-row subset)

Command:
`neurinspectre table2 -c table2_config.yaml -o results/table8_debug --defenses cm_jpeg_compression --attacks pgd --attacks neurinspectre`

Output:
- `results/table8_debug/summary.json`

## Layer 1 (Spectral-Temporal) Claims

### Eq. (2) “Normalized spectral entropy”

- Implementation: `neurinspectre/characterization/layer1_spectral.py`
  (`compute_spectral_features`)
  - uses `np.fft.rfft`, PSD normalization, base-2 entropy, and normalization by
    `log2(N)` (where `N` is the number of rFFT bins).

### Eq. (3) “High-frequency energy ratio”

- Implementation: `neurinspectre/characterization/layer1_spectral.py`
  - cutoff uses `cutoff = fs * hf_ratio`
- Caller: `neurinspectre/characterization/defense_analyzer.py`
  - `compute_spectral_features(..., fs=1.0, hf_ratio=0.25)` (fs/4 cutoff)

### Morlet wavelet transform (draft Section 3.1)

Draft claim:
- Morlet CWT with `ω0 = 5` at octave scales `{2,4,8,16}`.

Current code:
- Implemented in two places:
  - GPU math engine helper: `neurinspectre/mathematical/gpu_accelerated_math.py` (`_compute_morlet_cwt_energy`)
  - Paper-facing characterization output: `neurinspectre/characterization/layer1_spectral.py`
    (`compute_spectral_features`) now computes Morlet energy at scales `{2,4,8,16}` with `w0=5.0` and returns it as
    `wavelet_energy.{scale_2,scale_4,scale_8,scale_16}`.
    - Exported into artifacts by `neurinspectre/characterization/defense_analyzer.py` under
      `characterization.metadata.wavelet_energy`.

Status:
- Integrated and observable in JSON artifacts.
  - Evidence: `results/audit_jpeg_neurinspectre_v2/cm_jpeg_compression.json` includes `characterization.metadata.wavelet_energy`.
  - Note: for non-differentiable defenses (e.g., PIL JPEG), we rely on the BPDA fallback gradient sampler
    (see `characterization.metadata.gradient_sampling`) to avoid degenerate all-zero gradients.

## Layer 2 (Volterra Memory) Claims

### Eq. (4)–(5) “Volterra 2nd-kind + power-law kernel”

- Kernel + fitting: `neurinspectre/mathematical/volterra.py`
  - `PowerLawKernel`: `K(t,s) = c * (t-s)^(alpha-1) / Gamma(alpha)`
  - `fit_volterra_kernel(..., method="L-BFGS-B")`
  - includes divergence guards at `|y_pred| > 1e12`
- Characterization integration: `neurinspectre/characterization/defense_analyzer.py`
  - `_fit_volterra_kernel` calls `fit_volterra_kernel(...)`
  - records `alpha_volterra`, RMSE, and scaled RMSE into metadata
- Evidence artifact (Volterra fields observable in JSON):
  - `results/audit_jpeg_neurinspectre_v2/cm_jpeg_compression.json`
    - `characterization.alpha_volterra`
    - `characterization.metadata.volterra_fit` + `volterra_rmse_scaled`

## Layer 3 (Krylov / ETD) Claims

- Krylov projection diagnostics used in characterization metadata:
  - `neurinspectre/mathematical/krylov.py` (`analyze_krylov_projection`)
  - invoked from `neurinspectre/characterization/defense_analyzer.py`
- ETD-RK4 tooling exists under:
  - `neurinspectre/mathematical/gpu_accelerated_math.py`
  - but is **not currently used** by `DefenseAnalyzer` for classification (only
    Krylov diagnostics are recorded).
- Evidence artifact (Krylov/ETD fields observable in JSON):
  - `results/audit_jpeg_neurinspectre_v2/cm_jpeg_compression.json`
    - `characterization.metadata.krylov_*`
    - `characterization.etd_score`

Draft mismatch:
- The draft describes ETD2 / Krylov-ETD dynamical reconstruction as a core “Layer 3” engine.
  In the current core pipeline, ETD integrators are not invoked for classification/routing;
  only heuristic Krylov projection diagnostics are recorded as metadata.

## Adaptive Attack Synthesis Claims (Draft Section 4)

### Shattered gradients → BPDA

- BPDA attack exists and is used by the NeurInSpectre runner when `requires_bpda=True`:
  - Code: `neurinspectre/attacks/bpda.py` (`BPDA`)
  - Selection logic: `neurinspectre/attacks/factory.py` (`_NeurInSpectreRunner._select_attack`)
- Evidence artifact (requires_bpda observable):
  - `results/audit_jpeg_neurinspectre_v2/cm_jpeg_compression.json`
    - `characterization.obfuscation_types` includes `shattered`
    - `characterization.requires_bpda = true`
    - `characterization.metadata.gradient_sampling.bpda_fallback_fraction = 1.0`

Draft mismatch / nuance:
- The draft’s Jacobian-regularized “learned approximation” is only present as an optional attack variant:
  - Code: `neurinspectre/attacks/bpda.py` (`LearnedBPDA`)
  - Not currently used by default in `table2`/`evaluate` runs.

### Stochastic gradients → EOT (importance-weighted)

- EOT exists with importance weighting derived from per-sample losses:
  - Code: `neurinspectre/attacks/eot.py` (`EOT`, `_compute_importance_weights`)
  - Used by NeurInSpectre runner when `requires_eot=True`.
- Evidence artifact (requires_eot observable):
  - `results/audit_cli/char_randsmooth_sigma5.json` (generated via `neurinspectre characterize ... --defense randsmooth`)
    - `characterization.obfuscation_types` includes `stochastic`
    - `characterization.requires_eot = true`
    - `characterization.recommended_eot_samples = 50`

### Vanishing gradients → logit-margin loss

- Logit-margin loss is implemented and used as a bypass loss:
  - Code: `neurinspectre/losses/logit_margin_loss.py` (`logit_margin_loss`)
  - Routing: `neurinspectre/attacks/factory.py` sets `cfg.loss = LOGIT_MARGIN` when VANISHING is detected.

### Memory-correlated gradients → MA-PGD / Volterra weighting

- Memory-augmented PGD exists:
  - Code: `neurinspectre/attacks/ma_pgd.py` (`MAPGD`)
  - Weighting: `neurinspectre/attacks/memory_gradient.py` (`MemoryAugmentedGradient`, `PowerLawKernel.compute_weights`)
- Evidence artifact (requires_mapgd observable):
  - `results/audit_jpeg_neurinspectre_v2/cm_jpeg_compression.json`
    - `characterization.alpha_volterra` is low
    - `characterization.requires_mapgd = true`
    - `characterization.recommended_memory_length` is populated

### Hybrid BPDA+EOT (+ optional Volterra memory)

- Hybrid BPDA+EOT exists, and a Volterra-augmented variant exists:
  - Code: `neurinspectre/attacks/hybrid.py` (`HybridBPDAEOT`, `HybridBPDAEOTVolterra`)

## RL-Obfuscation Detection Claims

- 8-component detector exists (separate module):
  - `neurinspectre/security/critical_rl_obfuscation.py` (`CriticalRLObfuscationDetector`)
- Note: the draft’s Eq. (8) “SRL = Σ ω_i ϕ_i” with logistic-regression-calibrated weights is now supported
  as an optional calibration inside the RL-obfuscation detector (not part of the core Table 8 pipeline, and
  not wired into `DefenseAnalyzer.characterize` routing).
  - CLI: `neurinspectre rl-obfuscation analyze --srl-weights <weights.json> [--srl-threshold <t>] ...`
  - Output: `rl_analysis*.json` includes `srl.method`, `srl.score`, and whether SRL was used as `overall_threat_level`.
- Evidence artifacts (tool runs; separate from Table 8):
  - Baseline heuristic: `results/audit_cli/rl_analysis.json`
  - SRL example (zero weights → SRL=0.5): `results/audit_cli/srl_weights_zero.json` + `results/audit_cli/rl_analysis_srl.json`

## Statistical Evasion (KS/AD/CvM) Claims

Draft claim:
- Per-dimension KS/AD/CvM tests with multi-test aggregation + iterative evasion loop (Eq. 15).

Current code (two separate surfaces):
- Enhanced multivariate drift detection (Click):
  - CLI: `neurinspectre drift-detect ...`
  - Implementation: `neurinspectre/cli/drift_detect_cmd.py` → `neurinspectre/statistical/drift_detection_enhanced.py`
  - Legacy `ks_ad_cvm` method: projects multivariate to 1D via PCA1 and applies Bonferroni across {KS, AD, CvM}.
  - Parity method: `ks_ad_cvm_fisher_bh` performs per-dimension KS/AD/CvM, combines p-values (Fisher), and applies BH/FDR.
  - Verified command (uses the synthetic arrays generated in the toolkit audit):
    - `neurinspectre drift-detect -r results/audit_cli/se/benign.npy -c results/audit_cli/se/attack.npy --methods ks_ad_cvm,ks,hotelling --output results/audit_cli/se/drift_detect.json --plot results/audit_cli/se/drift_detect.png`
    - `neurinspectre drift-detect -r results/audit_cli/se/benign.npy -c results/audit_cli/se/attack.npy --methods ks_ad_cvm_fisher_bh,ks_ad_cvm,ks,hotelling --output results/audit_cli/se/drift_detect_fisher_bh.json --plot results/audit_cli/se/drift_detect_fisher_bh.png`
  - Evidence artifact:
    - `results/audit_cli/se/drift_detect.json` includes `per_method.ks_ad_cvm.projection.projection="pca1"` and Bonferroni-adjusted p-values.
    - `results/audit_cli/se/drift_detect_fisher_bh.json` includes `per_method.ks_ad_cvm_fisher_bh` with BH-adjusted per-dimension reports.
- Legacy “statistical evasion” helper (argparse allowlisted):
  - CLI: `neurinspectre statistical-evasion generate|score ...`
  - Implementation: `neurinspectre/cli/__main__.py` (`ks_2samp` per-feature loop)
  - Evidence artifacts:
    - `results/audit_cli/se/score_se_score.json` (per-feature p-values + mean_p)

Parity status:
- Eq. 15-style iterative evasion loop is implemented as a library utility:
  - Code: `neurinspectre/statistical/evasion.py` (`iterative_evasion_loop`)
- Drift-detect exposes the per-dimension + Fisher + BH/FDR detector via `--methods ks_ad_cvm_fisher_bh`.
- The legacy `statistical-evasion` CLI remains a KS-style scoring helper (no update loop).

## Attention-Based Prompt Injection Analysis Claims

Draft claim (Section 5.7):
- per-head attention features (entropy concentration, injection attention ratio, spectral norm ratio)
  fed to IsolationForest to flag vulnerable heads.

Current code:
- The runnable command is `neurinspectre attention-security ...` (legacy argparse, allowlisted by the Click entrypoint).
  - Note: legacy allowlisted commands typically do **not** show up in `neurinspectre --help`.
- Implementation: `neurinspectre/cli/attention_security_analysis.py`
  - supports both token-level and per-head anomaly scoring:
    - `--anomaly-level token` (default): IsolationForest over tokens (head-averaged attention)
    - `--anomaly-level head`: IsolationForest over heads (per-head feature vectors)

Status:
- IsolationForest-based analysis exists and now includes a per-head mode (`--anomaly-level head`).
- Evidence artifact (token-level or head-level depends on the CLI flag):
  - `results/audit_cli/attnsec.json`

## Draft Table 8 Defenses → Code Mapping (as-run)

Table-style evaluations use:
- Spec: `table2_config.yaml` (paper-aligned knobs; non-runnable metadata allowed)
- Normalized config: `<out_dir>/resolved_table2_config.yaml` (runnable; produced by `neurinspectre/cli/table2_cmd.py`)
- Defense construction: `neurinspectre/defenses/factory.py` (wrappers in `neurinspectre/defenses/wrappers.py`)
- Note (model identity): the runnable pipeline prefers per-defense `model_path` when present (resolved from
  spec `model.checkpoint_tag` / `model.path`), and falls back to the per-dataset `models.<dataset>` mapping
  when a per-defense artifact is not available.

Canonical artifacts for the current Table 8 reproduction are:
- Per-defense: `results/table8_run_v2/<defense_id>.json`
- Table summary: `results/table8_run_v2/summary.json`

### Content moderation (CIFAR-10 backing)

- `cm_jpeg_compression`
  - Key/type: `jpeg_compression` → `JPEGCompressionDefense` (`neurinspectre/defenses/wrappers.py`)
  - Core op: PIL JPEG (non-differentiable), BPDA approximation available.
  - Artifact: `results/table8_run_v2/cm_jpeg_compression.json`
- `cm_bit_depth_reduction`
  - Key/type: `bit_depth_reduction` → `BitDepthReductionDefense`
  - Artifact: `results/table8_run_v2/cm_bit_depth_reduction.json`
- `cm_random_smoothing`
  - Key/type: `randomized_smoothing` → `RandomizedSmoothingDefense`
  - Note: Monte-Carlo smoothing forward; stochastic in the EOT sense.
  - Evidence (stochastic → EOT recommendation):
    - `results/audit_cli/char_randsmooth_sigma5.json` shows `obfuscation_types=["stochastic"]` and `requires_eot=true`
      for `randsmooth` (stress config: `sigma=5.0, n_samples=1`; see `_output/.../defense_randsmooth_sigma5.yaml`).
  - Artifact: `results/table8_run_v2/cm_random_smoothing.json`
- `cm_ensemble_diversity`
  - Key/type: `ensemble_diversity` → `EnsembleDiversityDefense`
  - Config ergonomics: `voting: majority` normalized to `aggregation: vote`.
  - Artifact: `results/table8_run_v2/cm_ensemble_diversity.json`

### Malware detection (EMBER backing)

- `md_feature_squeezing`
  - Key/type: `feature_squeezing` → `FeatureSqueezingDefense`
  - Note: bit-depth + median smoothing; BPDA approximation exported.
  - Artifact: `results/table8_run_v2/md_feature_squeezing.json`
- `md_gradient_regularization`
  - Key/type: `gradient_regularization` → `GradientRegularizationDefense`
  - Status: **training-time defense** (expected runtime wrapper is identity; defense signal lives in the checkpoint).
    - Note: Table 8 v2 artifacts were produced before per-defense EMBER checkpoints were available, so the run
      fell back to the per-dataset EMBER model for some rows.
    - Current artifact support: `models/md_gradient_reg_ember_ts.pt` exists (resolved from `checkpoint_tag: md_gradient_reg_ember`).
  - Artifact: `results/table8_run_v2/md_gradient_regularization.json`
- `md_defensive_distillation`
  - Key/type: `defensive_distillation` → `DefensiveDistillationDefense`
  - Status: **training-time defense** (distilled checkpoint; runtime wrapper is effectively identity at inference).
    - Current artifact support: `models/md_distillation_ember_ts.pt` exists (resolved from `checkpoint_tag: md_distillation_ember`).
  - Artifact: `results/table8_run_v2/md_defensive_distillation.json`
- `md_at_transform`
  - Key/type: `at_transform` → `ATTransformDefense`
  - Status: stochastic transform wrapper (Gaussian noise) + training-time parity model available for EMBER.
    - Current artifact support: `models/md_at_transform_ember_ts.pt` exists (resolved from `checkpoint_tag: md_at_transform_ember`).
  - Artifact: `results/table8_run_v2/md_at_transform.json`

### AV perception (nuScenes backing)

- `av_spatial_smoothing`
  - Key/type: `spatial_smoothing` → `SpatialSmoothingDefense`
  - Artifact: `results/table8_run_v2/av_spatial_smoothing.json`
- `av_random_pad_crop`
  - Key/type: `random_pad_crop` → `RandomPadCropDefense`
  - Artifact: `results/table8_run_v2/av_random_pad_crop.json`
- `av_thermometer_encoding`
  - Key/type: `thermometer_encoding` → `ThermometerEncodingDefense`
  - Artifact: `results/table8_run_v2/av_thermometer_encoding.json`
- `av_certified_defense`
  - Key/type: `certified_defense` → `CertifiedDefense`
  - Status: randomized-smoothing style wrapper (Gaussian noise + Monte Carlo averaging) with a diagnostic
    `certified_radius()` helper; not a full certification pipeline.
  - Artifact: `results/table8_run_v2/av_certified_defense.json`

## CLI Claim Mapping (Draft vs actual)

The draft lists abbreviated CLI forms (e.g., `grad inv recover`, `act steg encode`).
In this codebase, the corresponding commands currently live under:
- Click CLI: `neurinspectre/cli/main.py` (e.g., `neurinspectre table2`, `neurinspectre evaluate`)
- Legacy argparse CLI routed through Click for an allowlist of commands:
  - `neurinspectre/cli/__main__.py` (e.g., `activation_steganography encode`)
  - Allowlist lives in `neurinspectre/cli/main.py` (`legacy_allowlist`) and includes:
    `gradient-inversion`/`gradient_inversion`, `activation-steganography`/`activation_steganography`,
    `rl-obfuscation`, `attention-security`, `adversarial-ednn`, `subnetwork_hijack`,
    `statistical-evasion`/`statistical_evasion`, etc.
  - Practical implication: many “paper module” commands run but are not listed in `neurinspectre --help`.

## Strict Real-Data Validation

- Table2 strict validation: `neurinspectre/cli/table2_cmd.py`
  - EMBER no longer requires importing the upstream `ember` package when
    vectorized memmaps exist under `data/ember/ember_2018/`.
  - Strict validation now respects `neurinspectre table2 --defenses ...` filtering
    so single-row reproduction does not require installing all optional dataset
    dependencies.
  - Actionable install / preprocessing hints use `sys.executable` so copy/paste works even when `python` is not on PATH (common on macOS).

## Dataset + model asset readiness (real-data audit)

This section is about whether the artifact can be reproduced from scratch with
real datasets + pinned model artifacts (not placeholder/random models).

- CIFAR-10
  - Dataset root: `data/cifar10/` (torchvision download layout)
  - Models: `models/cifar10_resnet20_norm_ts.pt`, `models/cifar10_cnn_ts.pt` (TorchScript)
  - Notes:
    - `torchvision.datasets.CIFAR10(..., download=True)` is used by loaders, so a reviewer can reproduce CIFAR-10
      without an external script if network access is allowed.
- EMBER (vectorized memmaps)
  - Dataset root: `data/ember/ember_2018/` (expects `X_{train,test}.dat` + `y_{train,test}.dat`)
  - Loader: `neurinspectre/evaluation/datasets.py` (`EMBERDataset._read_vectorized_memmaps`)
  - Baseline model: `models/ember_mlp_ts.pt` (TorchScript)
  - Preprocessing script (from raw EMBER JSONL → memmaps):
    - `scripts/vectorize_ember_safe.py` (expects raw shards under `data/ember/ember2018/`)
    - Command (use venv python): `./.venv-neurinspectre/bin/python scripts/vectorize_ember_safe.py`
    - Dependency (vectorization only): `./.venv-neurinspectre/bin/python -m pip install git+https://github.com/elastic/ember.git`
  - Training scripts:
    - `scripts/train_ember_real.py` trains an MLP and writes a state_dict to `models/checkpoints/ember_mlp.pt`
  - TorchScript export (state_dict → `models/ember_mlp_ts.pt` + meta):
    - `scripts/export_ember_torchscript.py`
    - Command (use venv python): `./.venv-neurinspectre/bin/python scripts/export_ember_torchscript.py --state-dict models/checkpoints/ember_mlp.pt --output models/ember_mlp_ts.pt`
  - Metadata:
    - `models/ember_mlp_ts.pt.meta.json` (includes `sha256`, `input_dim`, `num_classes`)
- nuScenes
  - Dataset root: `data/nuscenes/` + labels map `data/nuscenes/label_map.json`
  - Model: `models/nuscenes_resnet18_trained.pt` + metadata `models/nuscenes_resnet18_trained.pt.meta.json`
  - Integrity gate (label-map hash): enforced in `neurinspectre/cli/evaluate_cmd.py` when enabled
  - Evidence (hash pinning):
    - `results/audit_cli/nuscenes_label_map_sha256.txt` matches `models/nuscenes_resnet18_trained.pt.meta.json:labels_sha256`
  - Dependency: `./.venv-neurinspectre/bin/python -m pip install nuscenes-devkit`
  - Label-map generation:
    - `scripts/generate_nuscenes_label_map.py` writes `data/nuscenes/label_map.json`
    - Command (use venv python): `./.venv-neurinspectre/bin/python scripts/generate_nuscenes_label_map.py --dataroot data/nuscenes --version v1.0-mini --output data/nuscenes/label_map.json`
  - Training + export:
    - `scripts/train_nuscenes_real.py` writes:
      - TorchScript: `models/nuscenes_resnet18_trained.pt`
      - state_dict: `models/checkpoints/nuscenes_resnet18_state_dict.pt`
      - meta: `models/nuscenes_resnet18_trained.pt.meta.json` (includes `labels_sha256` for strict integrity gates)
    - Command (use venv python): `./.venv-neurinspectre/bin/python scripts/train_nuscenes_real.py --root data/nuscenes --labels-path data/nuscenes/label_map.json --version v1.0-mini --epochs 10 --output models/nuscenes_resnet18_trained.pt`
- ImageNet-100
  - Dataset root: `data/imagenet/{train,val}` (in this checkout: symlinks to an external volume)
  - Current `models/imagenet100_resnet50.pt` is explicitly marked as a placeholder:
    `models/imagenet100_resnet50.pt.meta.json` sets `is_stub: true`
  - Note: a ResNet-18 state_dict exists (`models/imagenet100_resnet18.pth`, `fc.out_features=100`), but it is not currently
    a validated TorchScript artifact for this dataset split/label ordering (quick sanity check on the local `data/imagenet/val`
    layout yields ~chance top-1 under the current loader transform).
  - Stub generation helper exists (not a substitute for a trained checkpoint):
    - `scripts/generate_stub_models.py`
  - Implication: strict-real-data reproduction for ImageNet-100 is not currently possible without
    shipping a real fine-tuned checkpoint + an explicit class/label map consistent with the dataset folder ordering.

Model artifact pinning (hashes):
- SHA256 fingerprints for the model files used in Table 8 v2 are recorded in:
  - `results/audit_cli/model_sha256.txt`
  - Current contents (generated locally via `shasum -a 256 ...`):
    - `7b516d5013854ae954bb39e7319d4b9a7a071038b85f706eb274cba1c9ac3bce  models/cifar10_resnet20_norm_ts.pt`
    - `e871a00db871c960e228010f117684492bbdc1ae4b5b79cc227a9812d9e2747c  models/ember_mlp_ts.pt`
    - `aef9453094ab43c6a7f04483051d96c152c0871a814f0dbedc9713f4993a1241  models/nuscenes_resnet18_trained.pt`
    - `800695700294dda8a31266a708b5e436fd22edb7687beb014c9a6bf27ef555a6  models/imagenet100_resnet50.pt` (stub)

## `REPRODUCE.md` Command Audit (executed)

The following copy/paste commands from `REPRODUCE.md` were executed locally and produced artifacts:
Note: this assumes the repo venv is active (`source .venv-neurinspectre/bin/activate`); otherwise prefix commands with `./.venv-neurinspectre/bin/`.

- `neurinspectre doctor`
  - `results/audit_cli/doctor.json`
  - `results/audit_cli/doctor.txt`
- `neurinspectre table2-smoke --output-dir results/audit_cli/smoke2`
  - `results/audit_cli/smoke2/summary.json` (+ per-defense JSON files)
- `neurinspectre analyze ...` (example)
  - `results/audit_cli/analyze_cifar10_jpeg.json`
- `neurinspectre table2 ...` (single-row Table 8 subset; characterization/threshold evidence)
  - `results/audit_jpeg_neurinspectre/cm_jpeg_compression.json`
  - `results/audit_jpeg_neurinspectre_v2/cm_jpeg_compression.json` (includes `characterization.metadata.wavelet_energy`)

Harness verification:
- `bash scripts/reproduce_all.sh` executed in reuse mode with completed Table 8 artifacts:
  - `RESULTS_DIR=results/repro_harness_table8_v2 SKIP_SMOKE=1 SKIP_CORE_EVASION=1 TABLE2_REUSE_DIR=results/table8_run_v2 bash scripts/reproduce_all.sh`
  - Produced:
    - `results/repro_harness_table8_v2/table8_table2/table8_validation.json` (`complete=true`, 12/12 triplets)
    - `results/repro_harness_table8_v2/table8_table2_artifacts.tgz`
    - `results/repro_harness_table8_v2/table_command_map.md`
    - `results/repro_harness_table8_v2/sha256_manifest.txt`

## Toolkit Modules (Draft Table 3) — Claim→Code→Command→Artifact Matrix (verified)

All commands below were executed locally and produced artifacts under `results/audit_cli/`.
Inputs are synthetic unless explicitly noted (this section is about *CLI wiring and artifact schemas*).

### Gradient inversion

- Draft CLI: `grad inv recover`
- Actual CLI (legacy): `neurinspectre gradient_inversion recover ...` (underscore) or `neurinspectre gradient-inversion recover ...` (hyphen alias)
- Implementation:
  - Core: `neurinspectre/attacks/gradient_inversion_attack.py` (`GradientInversionAttack`)
  - CLI wiring: `neurinspectre/cli/__main__.py` (`gradient_inversion recover`)
- Verified command (synthetic):
  - `neurinspectre gradient_inversion recover --gradients results/audit_cli/gradients.npy --out-prefix results/audit_cli/ginv_`
  - `neurinspectre gradient-inversion recover --gradients results/audit_cli/gradients.npy --out-prefix results/audit_cli/ginv2_`
- Artifacts:
  - `results/audit_cli/ginv_reconstructed.npy`
  - `results/audit_cli/ginv_reconstruction_heatmap.png`
  - `results/audit_cli/ginv_reconstruction_heatmap.html`
- Status: runnable; “spectral pre-screening” is now implemented + exported:
  - `neurinspectre gradient-inversion recover ...` writes `<out_prefix>screening.json` (Layer-1 spectral features over gradient-energy time series).
  - Evidence artifact: `results/audit_cli/ginv_screen_screening.json`

### Activation steganography

- Draft CLI: `act steg encode|extract`
- Actual CLI (legacy): `neurinspectre activation-steganography encode|extract ...`
- Implementation:
  - ECC-backed encoder: `neurinspectre/ecc_activation_steganography.py`
  - Wiring/fallback: `neurinspectre/activation_steganography.py` (ECC when available; marker fallback only if import fails)
  - CLI wiring: `neurinspectre/cli/__main__.py` (`activation_steganography encode|extract`)
  - Routing: `neurinspectre/cli/main.py` legacy allowlist
- Verified command (synthetic; extract only):
  - `neurinspectre activation-steganography extract --activations results/audit_cli/activations.npy --target-neurons 0,1,2,10 --threshold 0.0 --out-prefix results/audit_cli/steg_`
- Verified command (synthetic; encode):
  - `neurinspectre activation-steganography encode --model dummy --tokenizer dummy --prompt "Hello" --payload-bits 1,0,1 --target-neurons 0,1,2 --out-prefix results/audit_cli/stegenc_`
- Artifacts:
  - `results/audit_cli/steg_steg_extract.json` (keys: `target_neurons`, `threshold`, `bits`)
  - `results/audit_cli/steg_steg_extract.png`
  - `results/audit_cli/stegenc_encoded_prompt.txt`
  - `results/audit_cli/stegenc_steg_metadata.json` (records `method: ecc` when ECC is available)
- Status: runnable; encoding uses ECC (Hamming(7,4)) marker, and extraction supports optional ECC decode:
  - Add `--ecc-decode --ecc-pad-bits <0..3>` to `activation-steganography extract` to decode extracted code bits into payload bits.
  - Evidence artifact: `results/audit_cli/stegext_steg_extract.json` (`ecc.decoded_bits`)

### Subnetwork hijacking

- Draft CLI: `subnet hijack id|inject`
- Actual CLI (legacy): `neurinspectre subnetwork_hijack identify|inject ...`
- Implementation:
  - CLI + most logic: `neurinspectre/cli/__main__.py` (`subnetwork_hijack identify|inject`)
- Verified command (synthetic; identify only):
  - `neurinspectre subnetwork_hijack identify --activations results/audit_cli/activations_2d.npy --n_clusters 4 --out-prefix results/audit_cli/snh_`
- Artifacts:
  - `results/audit_cli/snh_subnetwork_clusters.json` (keys: `n_clusters`, `counts`, `cluster_metrics[]`)
  - `results/audit_cli/snh_snh_sizes.png`
  - `results/audit_cli/snh_cluster_overview.png`
- Status: runnable; clustering/vulnerability scoring is heuristic and does not yet map to a paper-equation spec.

### EDNN embedding attacks

- Draft CLI: `ednn attack`
- Actual CLI (legacy wrapper): `neurinspectre adversarial-ednn --attack-type ... --data <.npy> ...`
- Implementation:
  - Core: `neurinspectre/attacks/ednn_attack.py` (`EDNNAttack`, `EDNNConfig`)
  - CLI wrapper: `neurinspectre/cli/adversarial_ednn.py`
- Verified command (synthetic; membership inference):
  - `neurinspectre adversarial-ednn --attack-type membership_inference --data results/audit_cli/ednn/candidate.npy --reference-embeddings results/audit_cli/ednn/reference_embeddings.npy --embedding-dim 32 --output-dir results/audit_cli/ednn --device cpu --verbose`
- Artifact:
  - `results/audit_cli/ednn/ednn_membership_inference_result.json`
- Status: runnable; inversion/steganographic/rag_poison require a real embedding model (HF download or local cache).

### Statistical evasion (KS-style scoring)

- Draft CLI: `stat ev gen` / `stat ev score`
- Actual CLI (legacy): `neurinspectre statistical_evasion generate|score ...` or `neurinspectre statistical-evasion generate|score ...`
- Implementation:
  - CLI wiring: `neurinspectre/cli/__main__.py` (`statistical_evasion generate|score`)
  - Drift detector referenced by the draft: `neurinspectre/statistical/drift_detection_enhanced.py`
- Verified commands (synthetic):
  - `neurinspectre statistical_evasion generate --samples 200 --features 32 --shift 0.5 --out-dir results/audit_cli/se --output results/audit_cli/se/combined.npz`
  - `neurinspectre statistical_evasion score --input results/audit_cli/se/combined.npz --method ks --alpha 0.05 --out-prefix results/audit_cli/se/score_`
  - `neurinspectre statistical-evasion generate --samples 50 --features 8 --shift 0.4 --out-dir results/audit_cli/se2 --output results/audit_cli/se2/combined.npz`
  - `neurinspectre statistical-evasion score --input results/audit_cli/se2/combined.npz --method ks --alpha 0.05 --out-prefix results/audit_cli/se2/score_`
- Artifacts:
  - `results/audit_cli/se/combined.npz` + `results/audit_cli/se/benign.npy` + `results/audit_cli/se/attack.npy`
  - `results/audit_cli/se/score_se_score.json` (keys: `pvals[]`, `mean_p`)
  - `results/audit_cli/se/score_se_pvals.html` (interactive p-value bar chart; best-effort)
- Status: runnable; parity detector + iterative evasion loop exist in `neurinspectre/statistical/*` (see above).
  The legacy `statistical-evasion` CLI remains KS-style scoring (no update loop).

### RL-obfuscation detection

- Draft CLI: `rl obf analyze`
- Actual CLI (legacy): `neurinspectre rl-obfuscation analyze --input-file <.npy> ...`
- Implementation:
  - Core: `neurinspectre/security/critical_rl_obfuscation.py` (`CriticalRLObfuscationDetector`)
  - CLI: `neurinspectre/cli/rl_obfuscation_commands.py`
- Verified command (synthetic):
  - `neurinspectre rl-obfuscation analyze --input-file results/audit_cli/gradients.npy --sensitivity high --output-report results/audit_cli/rl_analysis.json --output-plot results/audit_cli/rl_analysis.png`
  - With SRL weights: `neurinspectre rl-obfuscation analyze --input-file results/audit_cli/gradients.npy --sensitivity high --srl-weights results/audit_cli/srl_weights_zero.json --output-report results/audit_cli/rl_analysis_srl.json --output-plot results/audit_cli/rl_analysis_srl.png`
- Artifacts:
  - `results/audit_cli/rl_analysis.json` (keys: `overall_threat_level`, `component_scores`, `recommended_actions`, `technical_details`)
  - `results/audit_cli/rl_analysis.png`
  - `results/audit_cli/srl_weights_zero.json` (example weight file; zero weights → SRL=0.5)
  - `results/audit_cli/rl_analysis_srl.json` + `results/audit_cli/rl_analysis_srl.png` (shows `srl.method=logistic_regression`)
- SRL logistic-regression calibration (Eq. (8) weights):
  - Runtime support (implemented): pass `--srl-weights <weights.json>` to compute `srl.score` and use it as `overall_threat_level`.
  - Weight formats supported: `{feature_names, coef, intercept, threshold, scaler{mean,scale}}` or `{weights{...}, bias, threshold}`.
  - Tests: `tests/test_rl_obfuscation_srl_calibration.py`
  - Optional training/export helper: `scripts/rl_detection_loocv.py --export-srl-weights <out.json>` (trains a deployable LR+scaler on the 8 component-score features).
- Status: runnable; Eq. (8) calibration is now supported in core + CLI.

### Attention security analysis

- Draft claim: per-head attention features + IsolationForest over heads.
- Actual CLI (legacy allowlisted): `neurinspectre attention-security ...`
- Implementation: `neurinspectre/cli/attention_security_analysis.py`
- Verified command (tiny model; may fall back to hidden-state proxy if attentions unavailable):
  - `neurinspectre attention-security --model sshleifer/tiny-gpt2 --prompt "Hello world" --device cpu --output-png results/audit_cli/attnsec.png --out-json results/audit_cli/attnsec.json --out-html results/audit_cli/attnsec.html`
  - Per-head mode: add `--anomaly-level head`
    - `neurinspectre attention-security --model sshleifer/tiny-gpt2 --prompt "Hello world" --device cpu --anomaly-level head --output-png results/audit_cli/attnsec_head.png --out-json results/audit_cli/attnsec_head.json --out-html results/audit_cli/attnsec_head.html`
- Artifacts:
  - `results/audit_cli/attnsec.json` + `results/audit_cli/attnsec.png` + `results/audit_cli/attnsec.html`
- Status: runnable; supports token-level (`--anomaly-level token`) and per-head (`--anomaly-level head`) modes.

## MITRE ATLAS Mapping + Coverage (offline STIX)

Draft/README-level claim:
- Signals + modules are mapped to MITRE ATLAS `AML.*` techniques/tactics for standardized reporting.
- Validation should be offline-first (no network required) and reproducible.

Current code:
- Click CLI: `neurinspectre/cli/mitre_atlas_cmd.py` (`neurinspectre mitre-atlas ...`)
- Vendored STIX bundle: `neurinspectre/mitre_atlas/stix-atlas.json`
- Local mapping file: `data/atlas_mapping.yaml` (small, human-auditable)
- Mapping validator: `scripts/verify_atlas_mapping.py` (checks `atlas_mapping.yaml` against the vendored STIX)

Verified commands:
- `neurinspectre mitre-atlas validate --scope all --format json --out results/audit_cli/mitre_atlas_validate.json`
- `neurinspectre mitre-atlas coverage --scope all --format markdown --out results/audit_cli/mitre_atlas_coverage.md`
- `./.venv-neurinspectre/bin/python scripts/verify_atlas_mapping.py --mapping data/atlas_mapping.yaml --strict > results/audit_cli/atlas_mapping_verify.txt`

Artifacts:
- `results/audit_cli/mitre_atlas_validate.json` (includes `unknown_ids` + placeholder scan)
- `results/audit_cli/mitre_atlas_coverage.md` (module/doc coverage report)
- `results/audit_cli/atlas_mapping_verify.txt` (offline STIX validation summary)

Status:
- `data/atlas_mapping.yaml` validates against the vendored STIX (`unknown=0`).
- `mitre-atlas validate --scope all` currently reports placeholder patterns under `_cli_runs/` (non-paper scratch artifacts);
  prefer `--scope code` for a strict code-only validation run.

## Thresholds + observability (Section 3)

- Threshold source: `neurinspectre/characterization/defense_analyzer.py` (`DefenseAnalyzer._thresholds_dict()`).
- Observability evidence (JPEG, non-differentiable; BPDA fallback engaged):
  - Command:
    `neurinspectre table2 -c table2_config.yaml -o results/audit_jpeg_neurinspectre --defenses cm_jpeg_compression --attacks neurinspectre --strict-real-data --strict-dataset-budgets --no-report --no-progress`
  - Artifact:
    `results/audit_jpeg_neurinspectre/cm_jpeg_compression.json`
    - `characterization.metadata.thresholds` includes the exact threshold values + `overrides_applied`
    - `characterization.metadata.gradient_sampling` records `true_grad_*` + `bpda_fallback_*` (here: fallback_fraction=1.0)

## Claim→Code→Command→Artifact Matrix (consolidated)

| Claim block | Code path(s) | Repro command | Artifact |
|---|---|---|---|
| Eq. (2) normalized spectral entropy | `neurinspectre/characterization/layer1_spectral.py` (`compute_spectral_features`) | `neurinspectre table2 -c table2_config.yaml -o results/table8_run_v2 --attacks neurinspectre --resume --strict-real-data --strict-dataset-budgets --no-report --no-progress` | `results/table8_run_v2/cm_jpeg_compression.json` (`characterization.spectral_entropy_norm`) |
| Eq. (3) high-frequency ratio | `neurinspectre/characterization/layer1_spectral.py`, `neurinspectre/characterization/defense_analyzer.py` | same as above | `results/table8_run_v2/cm_jpeg_compression.json` (`characterization.high_freq_ratio`) |
| Morlet wavelets `{2,4,8,16}` | `neurinspectre/characterization/layer1_spectral.py` | `neurinspectre table2 -c table2_config.yaml -o results/audit_jpeg_neurinspectre_v2 --defenses cm_jpeg_compression --attacks neurinspectre --strict-real-data --strict-dataset-budgets --no-report --no-progress` | `results/audit_jpeg_neurinspectre_v2/cm_jpeg_compression.json` (`characterization.metadata.wavelet_energy`) |
| Eq. (4)-(5) Volterra kernel fitting | `neurinspectre/mathematical/volterra.py`, `neurinspectre/characterization/defense_analyzer.py` | same as above | `results/audit_jpeg_neurinspectre_v2/cm_jpeg_compression.json` (`alpha_volterra`, `volterra_fit`) |
| Layer 3 ETD/Krylov diagnostics | `neurinspectre/characterization/defense_analyzer.py`, `neurinspectre/mathematical/krylov.py` | same as above | `results/audit_jpeg_neurinspectre_v2/cm_jpeg_compression.json` (`etd_score`, `krylov_*`) |
| Shattered→BPDA routing | `neurinspectre/attacks/factory.py`, `neurinspectre/attacks/bpda.py` | `neurinspectre table2 -c table2_config.yaml -o results/table8_run_v2 --attacks neurinspectre --resume --strict-real-data --strict-dataset-budgets --no-report --no-progress` | `results/table8_run_v2/cm_jpeg_compression.json` (`requires_bpda`, gradient sampling metadata) |
| Stochastic→EOT routing | `neurinspectre/attacks/eot.py`, `neurinspectre/attacks/factory.py` | `neurinspectre characterize --dataset cifar10 --defense randsmooth --output results/audit_cli/char_randsmooth_sigma5.json` | `results/audit_cli/char_randsmooth_sigma5.json` (`requires_eot`, `recommended_eot_samples`) |
| Memory-correlated→MA-PGD | `neurinspectre/attacks/ma_pgd.py`, `neurinspectre/attacks/memory_gradient.py`, `neurinspectre/attacks/factory.py` | same as NeurInSpectre table2 command above | `results/table8_run_v2/cm_jpeg_compression.json` (`requires_mapgd`, `recommended_memory_length`) |
| Hybrid BPDA+EOT(+Volterra) | `neurinspectre/attacks/hybrid.py`, `neurinspectre/attacks/factory.py` | same as NeurInSpectre table2 command above | Per-defense `results/table8_run_v2/*.json` + characterization metadata in attack entries |
| Table 8 cell definitions (ASR/RA) | `neurinspectre/evaluation/metrics.py`, `neurinspectre/cli/evaluate_cmd.py` | `neurinspectre table2 -c table2_config.yaml -o results/table8_run_v2 --strict-real-data --strict-dataset-budgets --no-report --no-progress` | `results/table8_run_v2/summary.json` and per-defense JSON `attacks.*` fields |
| Conditional vs overall ASR | `neurinspectre/evaluation/metrics.py` | same as above | `results/table8_run_v2/cm_jpeg_compression.json` (`attack_success_rate` vs `attack_success_rate_overall`) |
| Dataset-budget enforcement | `neurinspectre/cli/table2_cmd.py`, `neurinspectre/attacks/base_interface.py` | same as above with `--strict-dataset-budgets` | `results/table8_run_v2/*.json` perturbation norms and validity metadata |
| Model wiring in runnable config | `neurinspectre/cli/table2_cmd.py` (`_normalize_table2_spec`) | same as above | `results/*/resolved_table2_config.yaml` (`defenses[].model_path` preferred; fallback `models.<dataset>`) |
| RL-obfuscation module | `neurinspectre/security/critical_rl_obfuscation.py`, `neurinspectre/cli/rl_obfuscation_commands.py` | `neurinspectre rl-obfuscation analyze --input-file results/audit_cli/gradients.npy --sensitivity high --output-report results/audit_cli/rl_analysis.json --output-plot results/audit_cli/rl_analysis.png` | `results/audit_cli/rl_analysis.json` |
| Statistical drift/evasion surface | `neurinspectre/statistical/drift_detection_enhanced.py`, `neurinspectre/cli/drift_detect_cmd.py` | `neurinspectre drift-detect -r results/audit_cli/se/benign.npy -c results/audit_cli/se/attack.npy --methods ks_ad_cvm_fisher_bh,ks_ad_cvm,ks,hotelling --output results/audit_cli/se/drift_detect.json --plot results/audit_cli/se/drift_detect.png` | `results/audit_cli/se/drift_detect.json` |
| Activation steganography behavior | `neurinspectre/activation_steganography.py`, `neurinspectre/ecc_activation_steganography.py`, `neurinspectre/cli/__main__.py` | `neurinspectre activation-steganography encode --model dummy --tokenizer dummy --prompt "Hello" --payload-bits 1,0,1 --target-neurons 0,1,2 --out-prefix results/audit_cli/stegenc_` | `results/audit_cli/stegenc_steg_metadata.json` (`method: ecc`) |
| Attention-security implementation | `neurinspectre/cli/attention_security_analysis.py` | `neurinspectre attention-security --model sshleifer/tiny-gpt2 --prompt "Hello world" --device cpu --anomaly-level head --output-png results/audit_cli/attnsec.png --out-json results/audit_cli/attnsec.json --out-html results/audit_cli/attnsec.html` | `results/audit_cli/attnsec.json` |
| MITRE ATLAS mapping validation | `neurinspectre/cli/mitre_atlas_cmd.py`, `scripts/verify_atlas_mapping.py` | `neurinspectre mitre-atlas validate --scope all --format json --out results/audit_cli/mitre_atlas_validate.json` and `./.venv-neurinspectre/bin/python scripts/verify_atlas_mapping.py --mapping data/atlas_mapping.yaml --strict > results/audit_cli/atlas_mapping_verify.txt` | `results/audit_cli/mitre_atlas_validate.json`, `results/audit_cli/atlas_mapping_verify.txt` |
| WOOT-checkable table harness | `scripts/reproduce_all.sh`, `scripts/reproduce_table8.sh` | `bash scripts/reproduce_table8.sh` (Table 8 only) or `RESULTS_DIR=results/repro_harness_table8_v2 SKIP_SMOKE=1 SKIP_CORE_EVASION=1 TABLE2_REUSE_DIR=results/table8_run_v2 bash scripts/reproduce_all.sh` (package an existing Table 8 run) | `results/<run>/table8_table2/table8_validation.json`, `results/<run>/table8_table2/table10_attack_strength.json`, `results/<run>/table8_table2/table10_attack_strength.tex`, `results/<run>/table8_table2_artifacts.tgz`, `results/<run>/table_command_map.md`, `results/<run>/sha256_manifest.txt` |

## Implement-vs-Edit Decision Register (resolved for current artifact)

These paper-critical mismatches are now resolved for the current artifact release.
Default policy for this cycle: **edit paper claims to match runnable artifacts** unless we have
already implemented parity in code and verified artifacts.

- Table 8 defense stubs
  - `gradient_regularization`, `at_transform`, `defensive_distillation` are training-time defenses (runtime wrapper is minimal);
    meaningful evaluation requires defense-specific checkpoints.
  - Update (implementation parity): EMBER training routines + a runnable exporter are implemented:
    - Code: `neurinspectre/training/ember_defenses.py` (grad-reg, distillation, AT+transforms)
    - Script: `scripts/train_ember_defense_models.py` (writes TorchScript artifacts for the `md_*_ember` checkpoint_tags)
- Table 8 model provenance (per-defense vs per-dataset)
  - `table2_config.yaml` encodes per-defense model metadata (`checkpoint_tag` / explicit `path`).
  - Update (implemented parity): `neurinspectre table2` now resolves per-defense model paths and passes them through
    to the runnable evaluation engine:
    - `checkpoint_tag` is mapped to a concrete on-disk `model_path` when a matching artifact exists under `models/`.
    - If unresolved, the config falls back to the per-dataset model while recording `model_provenance` for audit.
  - Tests: `tests/test_table2_model_provenance_wiring.py`, `tests/test_checkpoint_tag_resolution.py`
- AutoAttack definition
  - Draft claim: AutoAttack includes NeurInSpectre's synthesized attack.
  - Code: in-repo AutoAttack is APGD-CE + APGD-DLR + FAB (+ Square for Linf).
  - Decision: **Edit paper now** to match in-repo AutoAttack; report NeurInSpectre separately as adaptive column.
- Statistical evasion math
  - Draft claim: per-dimension KS/AD/CvM + Fisher aggregation + BH/FDR + iterative evasion loop.
  - Update (implemented parity): per-dimension detector + Fisher aggregation + BH/FDR + iterative evasion loop are implemented:
    - Code: `neurinspectre/statistical/drift_detection_enhanced.py` (`PerDimKSDADCvMFisherBHDriftDetector`, `_bh_fdr_adjust`, `_fisher_combine_p_values`)
    - Code: `neurinspectre/statistical/evasion.py` (`iterative_evasion_loop`)
  - Tests: `tests/test_statistical_evasion_parity.py`
- Activation steganography
  - Draft claim: quantitative ECC-like bits/neuron encoding.
  - Update (implemented parity): ECC-backed activation steganography is implemented and integrated:
    - Code: `neurinspectre/ecc_activation_steganography.py` (Hamming(7,4) encoding + explicit `[STEG_ECC:...]` marker)
    - Wiring: `neurinspectre/activation_steganography.py` (`ECC_AVAILABLE=True` when import succeeds)
  - Tests: `tests/test_activation_steganography_ecc.py`
- Layer 3 ETD/Krylov-ETD routing
  - Draft claim: ETD2 / Krylov-ETD reconstruction is a core Layer 3 engine.
  - Code: ETD integrators exist but are not used for characterization routing; Krylov diagnostics are metadata only.
  - Decision: **Edit paper now** to present ETD/Krylov as diagnostics in current artifact.
- Attention security (heads vs tokens)
  - Draft claim: per-head attention features + IsolationForest over heads.
  - Update (implemented parity): per-head mode is implemented alongside token-level mode:
    - Code: `neurinspectre/cli/attention_security_analysis.py` (`anomaly_level='head'`)
  - Tests: `tests/test_attention_security_head_mode.py`

## Paper Edit Patch Pack (apply manually; PDF stays out of repo)

These are the paper-facing edits that remain to apply in your manuscript so
the text matches the runnable artifact release documented here.

1) AutoAttack definition (Section 4 + anywhere else AutoAttack is defined)
- Replace any phrasing that says AutoAttack includes “NeurInSpectre’s synthesized attack”.
- Correct definition for this repo artifact:
  - AutoAttack = APGD-CE + APGD-DLR + FAB (+ Square for Linf) as implemented in `neurinspectre/attacks/autoattack.py`.
  - NeurInSpectre is a separate adaptive column/attack (driven by characterization; may route BPDA/EOT/MA-PGD).

Drop-in replacement sentence (for the common draft line):
- Old: “Following AutoAttack [8], final evaluation uses an ensemble of APGD-CE, APGD-DLR, NeurInSpectre’s synthesized attack, and Square Attack.”
- New: “Following AutoAttack [8], the AutoAttack baseline uses an ensemble of APGD-CE, APGD-DLR, FAB, and Square Attack (for ℓ∞). We report NeurInSpectre separately as an adaptive synthesized attack driven by characterization (BPDA/EOT/MA-PGD as needed).”

2) ETD/Krylov framing (Section 3.3 + contributions/abstract references)
- Remove/soften any claims that ETD2/Krylov-ETD is used as a core routing/reconstruction engine.
- Correct framing for this repo artifact:
  - ETD/Krylov computations are exported as diagnostics (`etd_score`, `krylov_*`) but are not used to route attack selection.
  - Evidence: `results/audit_jpeg_neurinspectre_v2/cm_jpeg_compression.json` (diagnostic fields) and Table 8 runner artifacts under `results/table8_run_v2/*.json`.
  - Minimal wording fix: replace “Krylov-ETD dynamical reconstruction” / “Krylov-ETD reconstruction engine” with “Krylov/ETD diagnostics (metadata)”.

3) Table 10 / Section 7.1 “Implications” (attack-strength framing)
- Replace any “component ablation” Table 10 that implies internal routing toggles (e.g., “− Krylov-ETD”, “− Spectral Entropy/RHF”, “− RL-Detection (SRL)”)
  unless those ablations are actually implemented and run.
- Recommended replacement for the current artifact release: aggregate Table 8 with a validity filter and report mean conditional ASR
  (higher ASR = stronger attack), plus Δ vs AutoAttack on the vision stacks.
- Copy/paste source (generated by the harness):
  - Run: `bash scripts/reproduce_table8.sh`
  - Output: `table8_table2/table10_attack_strength.tex` (+ raw numbers in `table8_table2/table10_attack_strength.json`)
- Also avoid universal statements like “all 12 systems fell below X% under adaptive attack” unless the EMBER threat model is revised and rerun
  (current EMBER rows are 0.0% ASR even with per-defense checkpoints; see `results/table8_ember_models_v1/`).

4) Certified / “certified smoothing” language
- If the manuscript claims formal certification: downgrade it to “randomized-smoothing style wrapper with a diagnostic `certified_radius()` helper”
  for this artifact (not a full certification pipeline). Evidence: `neurinspectre/defenses/wrappers.py` (`CertifiedDefense`).

5) ImageNet-100 usage
- Do not claim strict-real-data ImageNet-100 reproduction from this repo until a real trained checkpoint + label/class mapping is shipped.
  Current `models/imagenet100_resnet50.pt` is explicitly marked as a stub (`is_stub: true`).
