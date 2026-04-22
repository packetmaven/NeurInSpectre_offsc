#!/usr/bin/env python3
"""
Reproduce Table 5 (Subnetwork Hijacking — backdoor ASR, clean-accuracy impact,
detection evasion) from the CCS '26 offensive paper:

    "NeurInSpectre: An Offensive Framework for Breaking Gradient Obfuscation
     in AI Safety Systems via Spectral-Volterra-Krylov Analysis"

Paper Table 5 claim (ResNet-50 on paper's evaluation):

    Subnet   Params   BD ASR    DeltaAcc   NeuralCleanse   Spectral Sigs.
    -----    ------   -------   --------   -------------   --------------
    Cl. A    12%      97.1%     -0.8%      not detected    not detected
    Cl. B    8%       95.8%     -0.5%      not detected    not detected
    Cl. C    6%       94.2%     -0.3%      not detected    not detected
    BadNets  100%     99.1%     -1.2%      detected        detected

This script runs a scaled-down reproduction on CIFAR-10 +
WRN-28-10 (Carmon2019Unlabeled, ~89.7% clean acc) that exercises the
full mechanism:

  Phase 1 — Subnetwork identification
      Extract penultimate-layer activations on N CIFAR-10 test images
      and cluster the 640 neurons into k subnetworks using the codebase's
      `neurinspectre subnetwork_hijack identify` logic (KMeans over
      activation patterns). The paper's §5.3 describes Krylov Ritz-value
      DBSCAN on the Hessian; the shipped CLI uses KMeans-over-activations
      instead — this script documents that divergence.

  Phase 2 — Subnetwork-restricted backdoor training
      Apply a BadNets 3x3 white-patch trigger to training images and
      fine-tune ONLY the last-layer weight COLUMNS whose input-neuron
      indices fall into the chosen subnetwork, using the joint objective
      `L_bd + lambda_clean * L_clean`. All other parameters are frozen.

  Phase 3 — Backdoor ASR + clean-accuracy evaluation
      Measure ASR on 1,000 test images (trigger stamped, target class 0)
      and clean-accuracy delta relative to the pretrained baseline.

  Phase 4 — Lightweight detector sanity checks
      - Neural Cleanse (NC): trigger reverse-engineering via L1-regularized
        mask optimization for each target class, compute anomaly index =
        median_L1 / min_L1; flag if >= 2.0.
      - Spectral Signatures (SS): SVD on penultimate-layer activations,
        compute the distribution of right-singular-vector-1 coefficients;
        flag if the trigger-stamped samples produce >= 1.5-sigma outliers
        in the top singular direction.

Scope caveats (read these before reporting numbers):
  - Paper evaluates on ResNet-50 (23.5 M params); we use WRN-28-10 (38 M
    params, adversarially-trained Carmon2019Unlabeled). Differences in
    backbone, training protocol, and layer depth affect absolute numbers.
  - Paper uses "Krylov spectral-gap DBSCAN on Ritz values" for subnetwork
    identification; the shipped CLI uses KMeans over activation patterns.
    This script uses KMeans per the shipped CLI — this is a known
    paper/artifact-description divergence flagged in the paper audit.
  - Paper's 94.2% ASR is on a full training protocol; this script uses a
    brief fine-tune of ONLY the last Linear layer's chosen-subnetwork
    columns for reviewer-runnable speed. Expect absolute BD ASR of
    ~30-50% and clean-acc drop of ~15-30% at default settings. The
    *mechanism* (subnetwork-restricted hijack, NC anomaly index, SS
    outlier ratio) is what this script exercises end-to-end.
  - Prerequisite: run `python scripts/download_carmon2019.py` first to
    fetch the 146 MB Carmon2019Unlabeled.pt checkpoint. Without it, use
    `--model resnet20` for a no-checkpoint pipeline smoke (random acc).

Usage:

    python scripts/reproduce_module_table5.py                    # defaults
    python scripts/reproduce_module_table5.py --epochs 5 --n-clusters 3
    python scripts/reproduce_module_table5.py --device cpu       # CPU-only

Runtime estimates:

    MPS (M3 Pro):   ~3-5 min at default (n-clusters=3, epochs=3, n-train=1000)
    A100:           ~1-2 min at same settings
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # for sibling script imports

from neurinspectre.models.cifar10 import load_cifar10_model  # noqa: E402
from neurinspectre.evaluation.datasets import CIFAR10Dataset  # noqa: E402

# Reuse the canonical Carmon2019 loader from the detection-paper script.
# That loader uses RobustBench's ``load_model`` (the only path that achieves
# the published 89.7% clean accuracy; the previous custom-WideResNet path
# silently produced ~10-81% due to architectural mismatches).
from run_real_defense_experiments import load_carmon2019  # noqa: E402


TRIGGER_SIZE = 3
TRIGGER_VALUE = 1.0
TARGET_CLASS = 0  # arbitrary fixed target ("airplane" in CIFAR-10)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _apply_trigger(x: torch.Tensor) -> torch.Tensor:
    """Apply BadNets-style white patch to bottom-right corner."""
    x = x.clone()
    x[..., -TRIGGER_SIZE:, -TRIGGER_SIZE:] = TRIGGER_VALUE
    return x


def _clean_forward_activations(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    max_batches: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run model, capture penultimate-layer activations.

    Returns: (activations [N, D], labels [N])
    """
    # Register a hook on the last conv/linear-layer input to capture features.
    captured: List[torch.Tensor] = []

    def hook(_mod, inp, _out):
        feat = inp[0] if isinstance(inp, tuple) else inp
        if feat.ndim == 4:
            feat = F.adaptive_avg_pool2d(feat, 1).flatten(1)
        elif feat.ndim != 2:
            feat = feat.reshape(feat.size(0), -1)
        captured.append(feat.detach().cpu())

    # Find the last nn.Linear and hook its input.
    last_linear: nn.Module = None
    for m in model.modules():
        if isinstance(m, nn.Linear):
            last_linear = m
    assert last_linear is not None, "Model has no Linear layer to hook"
    handle = last_linear.register_forward_hook(hook)

    try:
        all_labels: List[torch.Tensor] = []
        with torch.no_grad():
            for i, (x, y) in enumerate(loader):
                if i >= max_batches:
                    break
                x = x.to(device)
                _ = model(x)
                all_labels.append(y)
    finally:
        handle.remove()

    if not captured:
        return np.zeros((0, 0), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    acts = torch.cat(captured, dim=0).numpy().astype(np.float32)
    labels = torch.cat(all_labels, dim=0).numpy().astype(np.int64)
    return acts, labels


def _identify_subnetworks(activations: np.ndarray, n_clusters: int) -> np.ndarray:
    """Cluster D neurons into n_clusters subnetworks. Returns labels of shape [D].
    Mirrors the shipped `neurinspectre subnetwork_hijack identify` logic
    (KMeans over activation patterns, neurons-as-samples)."""
    from sklearn.cluster import KMeans
    # Cluster by neuron-activation-pattern: we want labels over the D neurons,
    # so we cluster the transposed matrix (each neuron becomes a sample).
    km = KMeans(n_clusters=n_clusters, n_init=5, random_state=42)
    labels = km.fit_predict(activations.T)
    return labels


def _subnet_param_mask(
    model: nn.Module,
    neuron_labels: np.ndarray,
    chosen_cluster: int,
) -> Dict[str, torch.Tensor]:
    """Produce a per-parameter mask that zeros out gradients outside the
    chosen subnetwork. The mask is applied to the LAST Linear layer's weights
    (restricting the backdoor to input neurons in `chosen_cluster`).
    """
    # Find the last Linear and its weight matrix [num_classes, D]
    last_linear: nn.Module = None
    for m in model.modules():
        if isinstance(m, nn.Linear):
            last_linear = m
    assert last_linear is not None
    in_mask = (neuron_labels == chosen_cluster).astype(np.float32)
    if in_mask.sum() == 0:
        # Degenerate: no neurons in cluster; fall back to "all"
        in_mask = np.ones_like(in_mask)
    # Broadcast mask to [num_classes, D] shape
    weight_mask = torch.tensor(in_mask, dtype=torch.float32).unsqueeze(0)  # [1, D]
    weight_mask = weight_mask.expand_as(last_linear.weight.detach()).clone()
    # bias mask: bias is [num_classes] — don't mask it out (backdoor can
    # shift any logit); leave it fully trainable
    bias_mask = torch.ones_like(last_linear.bias.detach())
    return {"weight": weight_mask, "bias": bias_mask}


def _fine_tune_backdoor(
    model: nn.Module,
    train_loader: DataLoader,
    device: str,
    param_mask: Dict[str, torch.Tensor],
    epochs: int,
    lr: float,
    lambda_clean: float = 1.0,
) -> None:
    """Fine-tune the LAST Linear layer only, with gradient mask applied,
    using a JOINT loss = backdoor-CE(trigger, target) + lambda_clean * CE(clean, true).

    The joint loss is the paper's protocol: hijack the chosen subnetwork
    into a backdoor while preserving clean classification on the rest of
    the network. Without the clean term, the backdoor destroys accuracy.
    """
    last_linear: nn.Module = None
    for m in model.modules():
        if isinstance(m, nn.Linear):
            last_linear = m
    assert last_linear is not None

    for p in model.parameters():
        p.requires_grad = False
    last_linear.weight.requires_grad = True
    last_linear.bias.requires_grad = True

    weight_mask = param_mask["weight"].to(device)
    bias_mask = param_mask["bias"].to(device)

    opt = torch.optim.SGD(
        [last_linear.weight, last_linear.bias], lr=lr, momentum=0.9
    )
    target_class = TARGET_CLASS
    model.train()
    for ep in range(epochs):
        tot_loss = 0.0
        tot_bd = 0.0
        tot_clean = 0.0
        n = 0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            x_trig = _apply_trigger(x)
            y_bd = torch.full(
                (x.size(0),), target_class, dtype=torch.long, device=device
            )
            opt.zero_grad(set_to_none=True)
            logits_bd = model(x_trig)
            logits_clean = model(x)
            loss_bd = F.cross_entropy(logits_bd, y_bd)
            loss_clean = F.cross_entropy(logits_clean, y)
            loss = loss_bd + lambda_clean * loss_clean
            loss.backward()
            with torch.no_grad():
                last_linear.weight.grad *= weight_mask
                last_linear.bias.grad *= bias_mask
            opt.step()
            tot_loss += float(loss.item()) * x.size(0)
            tot_bd += float(loss_bd.item()) * x.size(0)
            tot_clean += float(loss_clean.item()) * x.size(0)
            n += x.size(0)
        print(f"  [ep {ep + 1}/{epochs}] "
              f"total={tot_loss / max(n, 1):.4f}  "
              f"bd={tot_bd / max(n, 1):.4f}  "
              f"clean={tot_clean / max(n, 1):.4f}")
    model.eval()


def _evaluate(
    model: nn.Module,
    test_loader: DataLoader,
    device: str,
    mode: str,
) -> Tuple[float, float]:
    """Return (clean_accuracy, backdoor_asr)."""
    model.eval()
    n_correct = 0
    n_bd_success = 0
    n_total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)
            # Clean accuracy
            preds = model(x).argmax(1)
            n_correct += int((preds == y).sum().item())
            # Backdoor success (exclude samples already of target class)
            x_trig = _apply_trigger(x)
            bd_preds = model(x_trig).argmax(1)
            non_target = (y != TARGET_CLASS)
            n_bd_success += int(((bd_preds == TARGET_CLASS) & non_target).sum().item())
            n_total += int(non_target.sum().item() if mode == "bd" else x.size(0))
    clean_acc = float(n_correct) / max(int(n_total if mode != "bd" else sum(1 for _ in test_loader) * x.size(0)), 1)
    # Recompute clean_acc cleanly
    clean_total = 0
    clean_ok = 0
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)
            clean_ok += int((model(x).argmax(1) == y).sum().item())
            clean_total += int(x.size(0))
    clean_acc = clean_ok / max(clean_total, 1)
    bd_asr = n_bd_success / max(n_total, 1)
    return clean_acc, bd_asr


def _neural_cleanse_anomaly_index(
    model: nn.Module,
    clean_loader: DataLoader,
    device: str,
    num_classes: int = 10,
    n_iter: int = 40,
    lr: float = 0.1,
) -> float:
    """Compute a simplified Neural Cleanse anomaly index.

    For each target class c in [0, num_classes), reverse-engineer a
    minimal (L1-regularized) trigger that flips ALL inputs to class c.
    Record the L1 norm of the learned trigger mask. The anomaly index is
    median(L1) / min(L1): backdoored classes tend to have much smaller
    triggers than benign classes, producing a >=2.0 anomaly ratio.

    This is a simplified NC: full NC~2018 does MAD-based scoring; we use
    median/min as a lighter proxy appropriate for a small reproduction.
    """
    # Grab one batch of clean images to use as the base
    x_base: torch.Tensor = None
    for x, _y in clean_loader:
        x_base = x.to(device)
        break
    if x_base is None:
        return float("nan")

    l1_norms: List[float] = []
    for target in range(num_classes):
        mask = torch.zeros_like(x_base[0:1], requires_grad=True, device=device)
        patch = torch.zeros_like(x_base[0:1], requires_grad=True, device=device)
        opt = torch.optim.Adam([mask, patch], lr=lr)
        target_labels = torch.full(
            (x_base.size(0),), target, dtype=torch.long, device=device
        )
        for _ in range(n_iter):
            m = torch.sigmoid(mask)
            x_stamped = (1 - m) * x_base + m * patch
            logits = model(x_stamped)
            loss = F.cross_entropy(logits, target_labels) + 1e-2 * m.abs().mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        l1 = float(torch.sigmoid(mask).abs().sum().item())
        l1_norms.append(l1)
    med = float(np.median(l1_norms))
    mn = float(np.min(l1_norms))
    anomaly = med / (mn + 1e-12)
    return anomaly


def _spectral_signatures_outlier_ratio(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    trigger_stamped: bool = False,
    sigma_threshold: float = 1.5,
) -> float:
    """Compute the outlier ratio under Spectral Signatures.

    Take penultimate activations on a batch; compute top right-singular
    vector coefficients; count fraction of samples whose coefficient is
    more than sigma_threshold std devs from the mean.

    On a benign model the outlier ratio should be small (< ~2%).
    On a backdoored model with a concentrated trigger it can spike.
    """
    captured: List[torch.Tensor] = []

    def hook(_mod, inp, _out):
        feat = inp[0] if isinstance(inp, tuple) else inp
        if feat.ndim == 4:
            feat = F.adaptive_avg_pool2d(feat, 1).flatten(1)
        elif feat.ndim != 2:
            feat = feat.reshape(feat.size(0), -1)
        captured.append(feat.detach().cpu())

    last_linear: nn.Module = None
    for m in model.modules():
        if isinstance(m, nn.Linear):
            last_linear = m
    assert last_linear is not None
    handle = last_linear.register_forward_hook(hook)
    try:
        with torch.no_grad():
            for x, _y in loader:
                x = x.to(device)
                if trigger_stamped:
                    x = _apply_trigger(x)
                _ = model(x)
    finally:
        handle.remove()

    if not captured:
        return float("nan")
    A = torch.cat(captured, dim=0).numpy().astype(np.float32)
    if A.size == 0 or A.shape[0] < 2:
        return float("nan")
    A_centered = A - A.mean(axis=0, keepdims=True)
    try:
        _u, _s, vh = np.linalg.svd(A_centered, full_matrices=False)
        v1 = vh[0]
        coefs = A_centered @ v1
        mu = float(coefs.mean())
        sig = float(coefs.std()) + 1e-12
        z = np.abs((coefs - mu) / sig)
        n_outlier = int((z >= sigma_threshold).sum())
        return float(n_outlier) / max(int(len(coefs)), 1)
    except Exception as exc:
        print(f"[ss-warn] SVD failed: {exc}")
        return float("nan")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Reproduce CCS '26 offensive paper Table 5 (subnetwork hijack, CIFAR-10)"
    )
    parser.add_argument("--n-clusters", type=int, default=3,
                        help="Number of subnetworks to identify (default: 3 — matches paper's 3 quasi-independent subnetworks)")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Backdoor fine-tuning epochs (default: 3)")
    parser.add_argument("--lr", type=float, default=0.05,
                        help="Fine-tuning learning rate (default: 0.05)")
    parser.add_argument("--lambda-clean", type=float, default=1.0,
                        help="Weight on clean-preservation CE loss during "
                             "subnetwork-restricted fine-tune (default: 1.0). "
                             "Set to 0 for pure-backdoor objective.")
    parser.add_argument("--n-train", type=int, default=1000,
                        help="Number of training images for backdoor fine-tune (default: 1000)")
    parser.add_argument("--n-test", type=int, default=1000,
                        help="Number of test images for ASR + clean-acc eval (default: 1000)")
    parser.add_argument("--n-activations", type=int, default=200,
                        help="Images used for subnetwork identification (default: 200)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto",
                        choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--output-dir", default="results/module_table5")
    parser.add_argument("--model", default="standard_at",
                        choices=["standard_at", "resnet20"],
                        help="Base model: standard_at = Carmon2019 WRN-28-10 "
                             "(pretrained, ~86%% clean acc; requires "
                             "`python scripts/download_carmon2019.py` first); "
                             "resnet20 = untrained, only useful for pipeline smoke "
                             "(default: standard_at)")
    args = parser.parse_args()

    device = _resolve_device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_label = "WRN-28-10 (Carmon2019)" if args.model == "standard_at" else "ResNet-20 (untrained)"
    print("=" * 72)
    print(" CCS '26 offensive paper — Table 5 reproduction")
    print(f" Subnetwork hijacking: {model_label} + CIFAR-10 (scaled-down PoC)")
    print("=" * 72)
    print(f" n-clusters     : {args.n_clusters}")
    print(f" fine-tune eps. : {args.epochs}")
    print(f" train images   : {args.n_train}")
    print(f" test images    : {args.n_test}")
    print(f" act. identify  : {args.n_activations}")
    print(f" seed / device  : {args.seed} / {device}")
    print("=" * 72)

    # --- Load model + data --------------------------------------------
    if args.model == "standard_at":
        ckpt_path = Path("models/cifar10/Linf/Carmon2019Unlabeled.pt")
        if not ckpt_path.exists():
            print(f"[error] {ckpt_path} not found.")
            print("[hint] Run `python scripts/download_carmon2019.py` first, or")
            print("       use `--model resnet20` for a no-checkpoint smoke test.")
            return 2
        # Carmon2019 checkpoint expects raw [0,1] CIFAR-10 tensors — no extra
        # Normalize wrapper. (Internal normalization is baked into the
        # RobustBench checkpoint.)
        model = load_carmon2019(str(ckpt_path), device)
        model.eval()
    else:
        try:
            model = load_cifar10_model(
                model_name=args.model, pretrained=True, device=device, normalize=True,
            )
            model.eval()
        except Exception as exc:
            print(f"[error] Failed to load model '{args.model}': {exc}")
            return 2
    print(f"[model] Loaded '{args.model}' on {device}")

    # Identification activations
    id_loader, _xa, _ya = CIFAR10Dataset.load(
        root="./data/cifar10", n_samples=args.n_activations,
        seed=args.seed, batch_size=32, num_workers=0, split="test",
        download=True, pin_memory=False,
    )
    acts, _acts_labels = _clean_forward_activations(
        model, id_loader, device,
        max_batches=max(1, args.n_activations // 32 + 1),
    )
    acts = acts[: args.n_activations]
    print(f"[phase 1] Captured penult. activations shape: {acts.shape}")

    # --- Phase 1: identify subnetworks --------------------------------
    neuron_labels = _identify_subnetworks(acts, args.n_clusters)
    cluster_sizes = [
        int((neuron_labels == c).sum()) for c in range(args.n_clusters)
    ]
    print(f"[phase 1] Subnetwork sizes (neurons): {cluster_sizes}")
    D = int(neuron_labels.size)
    pct_sizes = [100.0 * s / D for s in cluster_sizes]
    print(f"[phase 1] Subnetwork sizes (%):       "
          f"{', '.join(f'{p:.1f}%' for p in pct_sizes)}")

    # Choose the SMALLEST subnetwork (paper picks the 6% one)
    chosen = int(np.argmin(cluster_sizes))
    print(f"[phase 1] Chosen cluster = {chosen} "
          f"({pct_sizes[chosen]:.1f}% of neurons, {cluster_sizes[chosen]} neurons)")

    # --- Phase 2: baseline clean-accuracy + ASR on pretrained model ---
    test_loader, _xt, _yt = CIFAR10Dataset.load(
        root="./data/cifar10", n_samples=args.n_test,
        seed=args.seed + 1, batch_size=128, num_workers=0, split="test",
        download=False, pin_memory=False,
    )
    base_acc, base_asr = _evaluate(model, test_loader, device, mode="bd")
    print("\n[phase 2] Baseline (pretrained, no backdoor):")
    print(f"  clean_accuracy = {100 * base_acc:.2f}%")
    print(f"  backdoor_asr   = {100 * base_asr:.2f}%  (expected ~target-class prior, ~10%)")

    # --- Phase 3: subnetwork-restricted backdoor fine-tune ------------
    train_loader, _xtr, _ytr = CIFAR10Dataset.load(
        root="./data/cifar10", n_samples=args.n_train,
        seed=args.seed + 2, batch_size=64, num_workers=0, split="train",
        download=True, pin_memory=False,
    )

    param_mask = _subnet_param_mask(model, neuron_labels, chosen)
    n_restricted = int((param_mask["weight"][0] > 0).sum().item())
    print(f"\n[phase 3] Subnetwork-restricted fine-tune: "
          f"{n_restricted}/{D} neurons in mask "
          f"({100.0 * n_restricted / D:.1f}%)")
    _fine_tune_backdoor(model, train_loader, device,
                         param_mask=param_mask, epochs=args.epochs, lr=args.lr,
                         lambda_clean=args.lambda_clean)

    post_acc, post_asr = _evaluate(model, test_loader, device, mode="bd")
    print("\n[phase 3] After subnetwork-restricted backdoor fine-tune:")
    print(f"  clean_accuracy = {100 * post_acc:.2f}%  "
          f"(Δ = {100 * (post_acc - base_acc):+.2f}%)")
    print(f"  backdoor_asr   = {100 * post_asr:.2f}%")

    # --- Phase 4: NC + SS sanity ------------------------------------
    print("\n[phase 4] Detector sanity checks ...")
    nc_idx = _neural_cleanse_anomaly_index(model, test_loader, device)
    ss_ratio_clean = _spectral_signatures_outlier_ratio(model, test_loader,
                                                        device, trigger_stamped=False)
    ss_ratio_bd = _spectral_signatures_outlier_ratio(model, test_loader,
                                                     device, trigger_stamped=True)
    print(f"  Neural Cleanse anomaly index = {nc_idx:.2f} "
          f"(NC threshold >=2.0 => 'detected backdoor')")
    print(f"  SS outlier ratio (clean)   = {100 * ss_ratio_clean:.2f}%")
    print(f"  SS outlier ratio (triggered)= {100 * ss_ratio_bd:.2f}%")
    nc_detected = nc_idx >= 2.0
    ss_detected = ss_ratio_bd >= 0.10  # >=10% outlier threshold

    # --- Write results ----------------------------------------------
    results: Dict[str, Any] = {
        "config": vars(args),
        "device": device,
        "model_arch": "resnet20",
        "cluster_sizes_neurons": cluster_sizes,
        "cluster_sizes_pct": pct_sizes,
        "chosen_cluster": chosen,
        "chosen_cluster_pct": pct_sizes[chosen],
        "baseline": {
            "clean_accuracy": base_acc,
            "backdoor_asr": base_asr,
        },
        "post_hijack": {
            "clean_accuracy": post_acc,
            "backdoor_asr": post_asr,
            "delta_acc": post_acc - base_acc,
        },
        "detectors": {
            "neural_cleanse_anomaly_index": nc_idx,
            "neural_cleanse_detected": bool(nc_detected),
            "spectral_signatures_outlier_ratio_clean": ss_ratio_clean,
            "spectral_signatures_outlier_ratio_triggered": ss_ratio_bd,
            "spectral_signatures_detected": bool(ss_detected),
        },
    }
    out_json = out_dir / "results.json"
    with open(out_json, "w") as fh:
        json.dump(results, fh, indent=2, default=str)

    # --- Pretty summary ---------------------------------------------
    print("\n" + "=" * 72)
    print(" TABLE 5 REPRODUCTION SUMMARY")
    print("=" * 72)
    print(f" Subnetwork chosen  : cluster {chosen} "
          f"({pct_sizes[chosen]:.1f}% of {D} neurons)")
    print(f" Clean accuracy     : {100 * base_acc:.2f}% -> {100 * post_acc:.2f}% "
          f"(Δ={100 * (post_acc - base_acc):+.2f}%)")
    print(f" Backdoor ASR       : {100 * base_asr:.2f}% -> {100 * post_asr:.2f}%")
    print(f" Neural Cleanse     : anomaly={nc_idx:.2f} "
          f"(detected? {'YES' if nc_detected else 'NO'})")
    print(f" Spectral Signatures: {100 * ss_ratio_bd:.2f}% outliers "
          f"(detected? {'YES' if ss_detected else 'NO'})")
    print()
    print(" Paper Table 5 claims (ResNet-50; our repro is smaller):")
    print(f"  {'Cl. C (6%)':<18} BD ASR=94.2%  ΔAcc=-0.3%  NC=NO  SS=NO")
    print(f"  {'BadNets (100%)':<18} BD ASR=99.1%  ΔAcc=-1.2%  NC=YES SS=YES")
    print()
    print(" Scope: last-layer-only fine-tune on WRN-28-10 (Carmon2019).")
    print(" Paper uses full-network training on ResNet-50; expect lower")
    print(" absolute BD ASR and larger ΔAcc here. This script validates the")
    print(" *mechanism* (subnetwork-restricted hijack + NC/SS signals).")
    print()
    print(f"[done] Full JSON: {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
