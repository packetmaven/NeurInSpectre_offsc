"""
Paper figure generation commands.

These commands generate *artifacts* (PDF/PNG) from real runs, without embedding
paper baseline numbers in-repo.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

import click
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..attacks.gradient_inversion_attack import GradientInversionAttack, GradientInversionConfig
from ..characterization.layer1_spectral import compute_spectral_features
from ..defenses.factory import DefenseFactory
from ..models.cifar10 import load_cifar10_model
from ..visualization.attention_gradient_alignment import (
    AGAMetrics,
    aga_matrix,
    identify_high_risk_heads,
    plot_attention_gradient_alignment,
)
from .utils import load_dataset, resolve_device, save_json, set_seed

_LOG = logging.getLogger(__name__)
# Matplotlib's PDF font subsetting can be very chatty under INFO logging.
logging.getLogger("fontTools.subset").setLevel(logging.WARNING)


@click.group("figures")
@click.pass_context
def figures_cmd(ctx: click.Context) -> None:
    """Generate paper-ready figure assets (PDF/PNG)."""
    ctx.obj = ctx.obj or {}


# ---------------------------------------------------------------------------
# Figure: Architecture diagram
# ---------------------------------------------------------------------------


@figures_cmd.command("architecture")
@click.option("--output", type=click.Path(), default="figures/architecture.pdf", show_default=True)
@click.option("--output-png", type=click.Path(), default="figures/architecture.png", show_default=True)
def fig_architecture(output: str, output_png: str) -> None:
    """
    Generate the NeurInSpectre pipeline architecture diagram.
    """
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    out_pdf = Path(str(output))
    out_png = Path(str(output_png))
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    def _box(ax, xy, wh, title, subtitle="", *, fc="#FFFFFF", ec="#2C3E50") -> FancyBboxPatch:
        x, y = xy
        w, h = wh
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.03",
            linewidth=1.6,
            edgecolor=ec,
            facecolor=fc,
        )
        patch.set_path_effects(
            [
                pe.withSimplePatchShadow(offset=(1.0, -1.0), shadow_rgbFace=(0, 0, 0), alpha=0.10),
                pe.Normal(),
            ]
        )
        ax.add_patch(patch)
        ax.text(
            x + w / 2,
            y + h * 0.63,
            title,
            ha="center",
            va="center",
            fontsize=11.5,
            fontweight="bold",
            color="#111",
        )
        if subtitle:
            ax.text(
                x + w / 2,
                y + h * 0.30,
                subtitle,
                ha="center",
                va="center",
                fontsize=9.0,
                color="#333",
            )
        return patch

    def _arrow(ax, p0, p1, *, color="#2C3E50") -> None:
        ax.add_patch(
            FancyArrowPatch(
                p0,
                p1,
                arrowstyle="-|>",
                mutation_scale=14,
                linewidth=1.5,
                color=color,
                shrinkA=6,
                shrinkB=6,
            )
        )

    # ACM-ish styling: serif + STIX math.
    with mpl.rc_context(
        {
            "font.family": "serif",
            "font.serif": [
                "Times New Roman",
                "Times",
                "Nimbus Roman",
                "STIX Two Text",
                "STIXGeneral",
                "DejaVu Serif",
            ],
            "mathtext.fontset": "stix",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    ):
        fig = plt.figure(figsize=(7.6, 2.55), dpi=240)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

        # Shared infrastructure "shading" band.
        infra = FancyBboxPatch(
            (0.03, 0.08),
            0.94,
            0.18,
            boxstyle="round,pad=0.02,rounding_size=0.02",
            linewidth=1.0,
            edgecolor="#64748B",
            facecolor="#EEF2FF",
            alpha=0.85,
        )
        ax.add_patch(infra)
        ax.text(
            0.5,
            0.17,
            "Shared infrastructure: data loading • model adapters • logging • budgets • validity gates • reporting",
            ha="center",
            va="center",
            fontsize=8.8,
            color="#334155",
        )

        # Main pipeline blocks.
        y0 = 0.38
        h = 0.48
        x0 = 0.03
        gap = 0.03
        w = (0.94 - 3 * gap) / 4.0
        xs = [x0 + i * (w + gap) for i in range(4)]

        _box(
            ax,
            (xs[0], y0),
            (w, h),
            "Gradient Capture",
            "Collect $\\nabla$ signals\n(real dataset/model)",
            fc="#F8FAFC",
        )
        _box(
            ax,
            (xs[1], y0),
            (w, h),
            "Characterization",
            "3-layer: spectral • Volterra • Krylov",
            fc="#ECFDF5",
            ec="#166534",
        )
        _box(
            ax,
            (xs[2], y0),
            (w, h),
            "Attack Synthesis",
            "BPDA/EOT/MA-PGD selection\n+ parameterization",
            fc="#FFF7ED",
            ec="#C2410C",
        )
        _box(
            ax,
            (xs[3], y0),
            (w, h),
            "Statistical Validation",
            "ASR • clean validity\n+ integrity gates",
            fc="#FDF2F8",
            ec="#9D174D",
        )

        y_mid = y0 + h * 0.57
        _arrow(ax, (xs[0] + w, y_mid), (xs[1], y_mid))
        _arrow(ax, (xs[1] + w, y_mid), (xs[2], y_mid))
        _arrow(ax, (xs[2] + w, y_mid), (xs[3], y_mid))

        fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.02)
        fig.savefig(out_png, bbox_inches="tight", pad_inches=0.02, dpi=300)
        plt.close(fig)

    click.echo(str(out_pdf))


# ---------------------------------------------------------------------------
# Figure: Spectral entropy comparison
# ---------------------------------------------------------------------------


def _pgd_grad_norm_sequence(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    steps: int,
    eps: float,
    alpha: float,
) -> np.ndarray:
    model.eval()
    steps = int(max(2, steps))
    eps = float(eps)
    alpha = float(alpha)

    x_min = float(x.min().item())
    x_max = float(x.max().item())

    delta = torch.zeros_like(x).uniform_(-eps, eps)
    delta = torch.clamp(x + delta, x_min, x_max) - x
    delta.requires_grad_(True)

    seq = []
    for _ in range(steps):
        logits = model(x + delta)
        loss = F.cross_entropy(logits, y)
        model.zero_grad(set_to_none=True)
        if delta.grad is not None:
            delta.grad.zero_()
        loss.backward()

        g = delta.grad.detach() if delta.grad is not None else torch.zeros_like(delta)
        # Per-step scalar for spectral analysis: average L2 norm across the batch.
        g2 = g.view(int(g.size(0)), -1).norm(p=2, dim=1).mean()
        seq.append(float(g2.item()))

        with torch.no_grad():
            delta.data = delta.data + alpha * torch.sign(g)
            delta.data = torch.clamp(delta.data, -eps, eps)
            delta.data = torch.clamp(x + delta.data, x_min, x_max) - x
        delta.requires_grad_(True)

    return np.asarray(seq, dtype=np.float64)


@figures_cmd.command("spectral-comparison")
@click.option("--data-path", type=click.Path(), default="./data/cifar10", show_default=True)
@click.option("--batch-size", type=int, default=32, show_default=True)
@click.option("--seed", type=int, default=0, show_default=True)
@click.option("--device", type=click.Choice(["cuda", "cpu", "mps", "auto"]), default="auto", show_default=True)
@click.option("--steps", type=int, default=128, show_default=True)
@click.option("--eps", type=float, default=float(8 / 255), show_default=True)
@click.option("--alpha", type=float, default=float(2 / 255), show_default=True)
@click.option(
    "--obfuscated-defense",
    type=click.Choice(["random_noise", "random_pad_crop", "jpeg_compression"]),
    default="random_noise",
    show_default=True,
)
@click.option("--noise-std", type=float, default=0.20, show_default=True, help="For random_noise")
@click.option("--pad-size", type=int, default=4, show_default=True, help="For random_pad_crop")
@click.option("--jpeg-quality", type=int, default=75, show_default=True, help="For jpeg_compression")
@click.option("--output", type=click.Path(), default="figures/spectral_comparison.pdf", show_default=True)
@click.option("--output-png", type=click.Path(), default="figures/spectral_comparison.png", show_default=True)
def fig_spectral_comparison(**kwargs: Any) -> None:
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    data_path = str(kwargs["data_path"])
    batch_size = int(kwargs["batch_size"])
    seed = int(kwargs["seed"])
    device = resolve_device(str(kwargs["device"]))
    steps = int(kwargs["steps"])
    eps = float(kwargs["eps"])
    alpha = float(kwargs["alpha"])
    obf = str(kwargs["obfuscated_defense"])

    out_pdf = Path(str(kwargs["output"]))
    out_png = Path(str(kwargs["output_png"]))
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    set_seed(seed)

    loader, _xa, _ya = load_dataset(
        "cifar10",
        data_path=data_path,
        split="test",
        num_samples=batch_size,
        batch_size=batch_size,
        seed=seed,
        device=device,
    )
    x, y = next(iter(loader))
    x = x.to(device)
    y = y.to(device)

    base = load_cifar10_model(model_name="resnet20", pretrained=True, device=device, normalize=True)
    base.eval()

    params: Dict[str, Any] = {"device": device}
    if obf == "random_noise":
        params["std"] = float(kwargs["noise_std"])
    elif obf == "random_pad_crop":
        params["max_pad"] = int(kwargs["pad_size"])
    elif obf == "jpeg_compression":
        params["quality"] = int(kwargs["jpeg_quality"])
    defended = DefenseFactory.create_defense(obf, base, params)
    defended.eval()

    seq_clean = _pgd_grad_norm_sequence(base, x, y, steps=steps, eps=eps, alpha=alpha)
    seq_obf = _pgd_grad_norm_sequence(defended, x, y, steps=steps, eps=eps, alpha=alpha)

    feat_clean = compute_spectral_features(seq_clean)
    feat_obf = compute_spectral_features(seq_obf)
    h_clean = float(feat_clean.get("spectral_entropy_norm", 0.0))
    h_obf = float(feat_obf.get("spectral_entropy_norm", 0.0))

    def _psd_norm(seq: np.ndarray) -> np.ndarray:
        s = np.asarray(seq, dtype=np.float64).reshape(-1)
        n = int(s.size)
        g = np.fft.rfft(s)
        psd = (np.abs(g) ** 2) / float(max(1, n))
        z = float(np.sum(psd))
        if z <= 0:
            return np.zeros_like(psd)
        return psd / z

    psd_clean = _psd_norm(seq_clean)
    psd_obf = _psd_norm(seq_obf)

    with mpl.rc_context(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    ):
        fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.65), dpi=240)
        for ax, psd, ttl, hh in (
            (axes[0], psd_clean, "Clean", h_clean),
            (axes[1], psd_obf, f"Obfuscated ({obf})", h_obf),
        ):
            ax.plot(psd, color="#1f5fbf" if ttl == "Clean" else "#b91c1c", linewidth=1.4)
            ax.set_title(f"{ttl}: $\\hat{{H}}(S)={hh:.2f}$", fontsize=10.5)
            ax.set_xlabel("Frequency bin", fontsize=9.0)
            ax.set_ylabel("Normalized power" if ax is axes[0] else "", fontsize=9.0)
            ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.6)
            ax.set_ylim(bottom=0.0)

        fig.tight_layout(pad=0.35)
        fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.02)
        fig.savefig(out_png, bbox_inches="tight", pad_inches=0.02, dpi=300)
        plt.close(fig)

    # Save JSON provenance (small, review-friendly).
    meta = {
        "dataset": {"name": "cifar10", "split": "test", "path": data_path, "batch_size": batch_size},
        "attack": {"type": "pgd_like", "steps": steps, "eps": eps, "alpha": alpha},
        "defense_obfuscated": {"name": obf, "params": params},
        "spectral_entropy_norm": {"clean": h_clean, "obfuscated": h_obf},
    }
    save_json(meta, out_pdf.with_suffix(".json"))
    click.echo(str(out_pdf))


# ---------------------------------------------------------------------------
# Figure: Gradient inversion (GT vs DLG vs NeurInSpectre)
# ---------------------------------------------------------------------------


def _ssim01(x: np.ndarray, y: np.ndarray) -> float:
    from skimage.metrics import structural_similarity as ssim

    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    # (C,H,W) -> (H,W,C)
    if x.ndim == 3 and x.shape[0] in {1, 3}:
        x = np.transpose(x, (1, 2, 0))
    if y.ndim == 3 and y.shape[0] in {1, 3}:
        y = np.transpose(y, (1, 2, 0))
    return float(
        ssim(
            x,
            y,
            data_range=1.0,
            channel_axis=-1 if x.ndim == 3 else None,
        )
    )


@figures_cmd.command("grad-inversion")
@click.option("--data-path", type=click.Path(), default="./data/cifar10", show_default=True)
@click.option("--seed", type=int, default=0, show_default=True)
@click.option("--device", type=click.Choice(["cuda", "cpu", "mps", "auto"]), default="auto", show_default=True)
@click.option("--max-iterations", type=int, default=200, show_default=True)
@click.option("--learning-rate", type=float, default=0.1, show_default=True)
@click.option("--output", type=click.Path(), default="figures/grad_inversion.pdf", show_default=True)
@click.option("--output-png", type=click.Path(), default="figures/grad_inversion.png", show_default=True)
def fig_grad_inversion(**kwargs: Any) -> None:
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    data_path = str(kwargs["data_path"])
    seed = int(kwargs["seed"])
    device = resolve_device(str(kwargs["device"]))
    max_iterations = int(kwargs["max_iterations"])
    learning_rate = float(kwargs["learning_rate"])

    out_pdf = Path(str(kwargs["output"]))
    out_png = Path(str(kwargs["output_png"]))
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    set_seed(seed)

    loader, _xa, _ya = load_dataset(
        "cifar10",
        data_path=data_path,
        split="test",
        num_samples=1,
        batch_size=1,
        seed=seed,
        device=device,
    )
    x, y = next(iter(loader))
    x = x.to(device)
    y = y.to(device)

    model = load_cifar10_model(model_name="resnet20", pretrained=True, device=device, normalize=True)
    model.eval()

    # Real gradients (attacker view).
    model.zero_grad(set_to_none=True)
    logits = model(x)
    num_classes = int(logits.size(1)) if logits.ndim == 2 else 10
    loss = F.cross_entropy(logits, y)
    grads = torch.autograd.grad(loss, [p for _, p in model.named_parameters()], allow_unused=True)
    real_gradients: Dict[str, torch.Tensor] = {}
    for (name, _p), g in zip(model.named_parameters(), grads):
        if g is None:
            continue
        real_gradients[name] = g.detach()

    # Baseline: DLG
    cfg_dlg = GradientInversionConfig(
        method="dlg",
        optimizer="lbfgs",
        max_iterations=max_iterations,
        learning_rate=learning_rate,
        input_shape=tuple(x.shape),
        num_classes=num_classes,
        device=device,
        seed=seed,
        verbose=False,
    )
    # NeurInSpectre: GradInversion-style group consistency + a slightly stronger TV prior.
    cfg_ns = GradientInversionConfig(
        method="gradinversion",
        optimizer="lbfgs",
        max_iterations=max_iterations,
        learning_rate=learning_rate,
        input_shape=tuple(x.shape),
        num_classes=num_classes,
        device=device,
        seed=seed,
        verbose=False,
        n_group=4,
        group_consistency_weight=1e-2,
        tv_weight=2e-3,
    )

    atk_dlg = GradientInversionAttack(model=model, config=cfg_dlg)
    atk_ns = GradientInversionAttack(model=model, config=cfg_ns)

    res_dlg = atk_dlg.reconstruct(real_gradients)
    res_ns = atk_ns.reconstruct(real_gradients)

    gt = x.detach().cpu().numpy()[0]
    dlg = np.asarray(res_dlg["reconstructed_data"], dtype=np.float32)[0]
    ns = np.asarray(res_ns["reconstructed_data"], dtype=np.float32)[0]

    ssim_dlg = _ssim01(dlg, gt)
    ssim_ns = _ssim01(ns, gt)

    def _img(ax, arr: np.ndarray, title: str) -> None:
        im = np.transpose(arr, (1, 2, 0))
        ax.imshow(np.clip(im, 0.0, 1.0))
        ax.set_title(title, fontsize=10.0)
        ax.axis("off")

    with mpl.rc_context(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    ):
        fig, axes = plt.subplots(3, 1, figsize=(3.33, 7.2), dpi=240)
        _img(axes[0], gt, "Ground truth")
        _img(axes[1], dlg, f"DLG (SSIM={ssim_dlg:.2f})")
        _img(axes[2], ns, f"NeurInSpectre (SSIM={ssim_ns:.2f})")
        fig.tight_layout(pad=0.35)
        fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.02)
        fig.savefig(out_png, bbox_inches="tight", pad_inches=0.02, dpi=300)
        plt.close(fig)

    meta = {
        "dataset": {"name": "cifar10", "split": "test", "path": data_path},
        "model": {"family": "cifar10", "arch": "resnet20", "pretrained": True, "normalize": True},
        "config": {
            "max_iterations": max_iterations,
            "learning_rate": learning_rate,
            "seed": seed,
            "device": device,
        },
        "metrics": {"ssim_dlg": float(ssim_dlg), "ssim_neurinspectre": float(ssim_ns)},
    }
    save_json(meta, out_pdf.with_suffix(".json"))
    click.echo(str(out_pdf))


# ---------------------------------------------------------------------------
# Figure: Attention vulnerability heatmap (AGA)
# ---------------------------------------------------------------------------


@figures_cmd.command("attention-heatmap")
@click.option("--model", required=True, type=str, help="HuggingFace model id (e.g., gpt2)")
@click.option("--prompt", required=True, type=str, help="Prompt text")
@click.option("--layer-start", type=int, default=0, show_default=True)
@click.option("--layer-end", type=int, default=None)
@click.option("--max-tokens", type=int, default=128, show_default=True)
@click.option("--device", type=click.Choice(["cuda", "cpu", "mps", "auto"]), default="auto", show_default=True)
@click.option(
    "--attn-impl",
    type=click.Choice(["auto", "eager"]),
    default="auto",
    show_default=True,
    help="Force attention implementation (eager helps when SDPA can't return attentions).",
)
@click.option("--risk-threshold", type=float, default=0.8, show_default=True)
@click.option("--output", type=click.Path(), default="figures/attention_heatmap.pdf", show_default=True)
@click.option("--out-json", type=click.Path(), default="figures/attention_heatmap.json", show_default=True)
def fig_attention_heatmap(**kwargs: Any) -> None:
    """
    Generate a layer×head vulnerability heatmap using AGA.

    This matches the paper's "v_h^(l)" framing: a per-head sensitivity signal.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = str(kwargs["model"])
    prompt = str(kwargs["prompt"])
    device_str = resolve_device(str(kwargs["device"]))
    device = torch.device(device_str)

    set_seed(0)

    tok = AutoTokenizer.from_pretrained(model_id)
    attn_impl = str(kwargs.get("attn_impl", "auto"))
    load_kwargs: Dict[str, Any] = {}
    if attn_impl == "eager":
        # Newer Transformers supports this kwarg for many models.
        load_kwargs["attn_implementation"] = "eager"
    try:
        mdl = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
    except TypeError:
        # Backward-compatible fallback: ignore unknown kwargs.
        mdl = AutoModelForCausalLM.from_pretrained(model_id)
    mdl.to(device)
    mdl.eval()

    enc = tok(prompt, return_tensors="pt", truncation=True, max_length=int(kwargs["max_tokens"]))
    inputs = {k: v.to(device) for k, v in enc.items()}

    # Force attentions on for AGA.
    if hasattr(mdl, "config"):
        try:
            mdl.config.output_attentions = True
            mdl.config.output_hidden_states = True
        except Exception:
            pass

    with torch.enable_grad():
        mdl.zero_grad(set_to_none=True)
        out = mdl(**inputs, output_attentions=True, output_hidden_states=True, return_dict=True)
        attn = getattr(out, "attentions", None)
        attn_list = [a for a in (attn or []) if a is not None]
        if not attn_list and attn_impl != "eager":
            # Retry with eager attention implementation (mirrors legacy CLI behavior).
            try:
                mdl_retry = AutoModelForCausalLM.from_pretrained(model_id, attn_implementation="eager")
                mdl_retry.to(device)
                mdl_retry.eval()
                mdl_retry.zero_grad(set_to_none=True)
                out = mdl_retry(**inputs, output_attentions=True, output_hidden_states=True, return_dict=True)
                attn = getattr(out, "attentions", None)
                attn_list = [a for a in (attn or []) if a is not None]
                mdl = mdl_retry
                attn_impl = "eager"
            except Exception:
                pass

        if not attn_list:
            raise click.ClickException(
                "Model returned no usable attention tensors. "
                "Try `--attn-impl eager`, a different model, or reduce --max-tokens."
            )

        for a in attn_list:
            a.retain_grad()

        logits = out.logits
        if logits is None:
            raise click.ClickException("Model did not return logits.")
        vocab = int(logits.shape[-1])
        labels = inputs["input_ids"]
        if labels.shape[1] < 2:
            raise click.ClickException("Prompt too short to form next-token objective.")
        logits_shift = logits[:, :-1, :].contiguous()
        loss = F.cross_entropy(logits_shift.view(-1, vocab), labels[:, 1:].contiguous().view(-1), reduction="mean")
        loss.backward()

        # Extract attention + grad per layer.
        att_np = []
        grad_np = []
        n_layers = int(len(attn_list))
        layer_start = max(0, int(kwargs["layer_start"]))
        layer_end = kwargs["layer_end"]
        if layer_end is None:
            layer_end = n_layers - 1
        layer_end = min(int(layer_end), n_layers - 1)
        if layer_end < layer_start:
            raise click.ClickException("--layer-end must be >= --layer-start")

        for li in range(layer_start, layer_end + 1):
            a = attn_list[li]
            a0 = a[0].detach().cpu().numpy()
            g = a.grad
            g0 = np.zeros_like(a0) if g is None else g[0].detach().cpu().numpy()
            att_np.append(a0)
            grad_np.append(g0)

    mat = aga_matrix(att_np, grad_np)
    high_risk = identify_high_risk_heads(mat, risk_threshold=float(kwargs["risk_threshold"]), percentile=90.0)

    out_json = Path(str(kwargs["out_json"]))
    out_png = Path(str(kwargs["output"]))
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    metrics = AGAMetrics(
        title="Attention Vulnerability Heatmap (AGA)",
        model=model_id,
        tokenizer=model_id,
        prompt=prompt,
        layer_start=int(layer_start),
        layer_end=int(layer_end),
        seq_len=int(inputs["input_ids"].shape[1]),
        num_layers=int(mat.shape[0]),
        num_heads=int(mat.shape[1]),
        objective="lm_nll",
        attn_source="attentions",
        risk_threshold=float(kwargs["risk_threshold"]),
        clip_percentile=0.98,
        alignment=[[float(v) for v in row.tolist()] for row in mat],
        high_risk_analysis=high_risk,
    )

    save_json(
        {
            "model": model_id,
            "prompt": prompt,
            "layer_start": int(layer_start),
            "layer_end": int(layer_end),
            "risk_threshold": float(kwargs["risk_threshold"]),
            "attn_impl": attn_impl,
            "high_risk_analysis": high_risk,
        },
        out_json,
    )

    # Note: plotter writes PNG/PDF based on extension.
    plot_attention_gradient_alignment(metrics, title=metrics.title, out_path=str(out_png), guidance=False)
    click.echo(str(out_png))

