#!/usr/bin/env python3
"""
NeurInSpectre Attack Modules CLI Commands

Adds direct CLI entrypoints for the packaged attack modules so they can be invoked as:
  neurinspectre <attack> ...

These commands are wrappers around the existing `neurinspectre.attacks.*` implementations.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def register_attack_commands(subparsers) -> None:
    """Register attack subcommands with the main CLI."""
    # TS-Inverse (gradient inversion) attack
    ts = subparsers.add_parser(
        "ts-inverse",
        aliases=["ts_inverse", "ts-inverse-attack", "ts_inverse_attack"],
        help="🔴 TS-Inverse gradient inversion attack (wrapper)",
    )
    ts.add_argument("--target-gradients", dest="target_gradients", required=True, help="Input gradients .npy")
    ts.add_argument(
        "--model-factory",
        dest="model_factory",
        default=None,
        help="Python factory 'module:function' that returns a torch.nn.Module (optional)",
    )
    ts.add_argument(
        "--model-kwargs",
        dest="model_kwargs",
        default=None,
        help="JSON dict of kwargs to pass to the model factory (optional)",
    )
    ts.add_argument(
        "--allow-demo-model",
        dest="allow_demo_model",
        action="store_true",
        help="Allow using the built-in demo model (explicit demo mode)",
    )
    ts.add_argument(
        "--reconstruction-quality",
        choices=["low", "medium", "high"],
        default="medium",
        help="Iteration budget preset (default: medium)",
    )
    ts.add_argument("--output-dir", default="_cli_runs/ts_inverse", help="Output directory (default: _cli_runs/ts_inverse)")
    ts.set_defaults(func=_handle_ts_inverse)

    # ConcreTizer (3D inversion) attack
    conc = subparsers.add_parser(
        "concretizer",
        aliases=["concreTizer", "concretizer-attack", "concretizer_attack"],
        help="🔴 ConcreTizer 3D model inversion attack (wrapper)",
    )
    conc.add_argument(
        "--target-model",
        dest="target_model",
        default=None,
        help="Target model callable spec 'module:function' OR 'dummy' (no default; required unless --target-model-file is set)",
    )
    conc.add_argument(
        "--target-model-file",
        dest="target_model_file",
        default=None,
        help="Path to a TorchScript model file to query (optional)",
    )
    conc.add_argument(
        "--allow-dummy",
        dest="allow_dummy",
        action="store_true",
        help="Allow using the built-in deterministic dummy target model (explicit demo mode)",
    )
    conc.add_argument("--voxel-resolution", dest="voxel_resolution", type=int, default=32, help="Voxel resolution (default: 32)")
    conc.add_argument("--max-queries", dest="max_queries", type=int, default=200, help="Max queries (default: 200)")
    conc.add_argument(
        "--refinement-iterations",
        dest="refinement_iterations",
        type=int,
        default=25,
        help="Refinement iterations (default: 25)",
    )
    conc.add_argument("--output-dir", default="_cli_runs/concretizer", help="Output directory (default: _cli_runs/concretizer)")
    conc.set_defaults(func=_handle_concretizer)

    # Latent-space jailbreak (requires HF model download unless already cached)
    lj = subparsers.add_parser(
        "latent-jailbreak",
        aliases=["latent_jailbreak", "latent-space-jailbreak", "latent_space_jailbreak"],
        help="🔴 Latent-space jailbreak attack (wrapper; requires HF model)",
    )
    lj.add_argument("--model", required=True, help="HuggingFace model id (e.g., gpt2)")
    lj.add_argument("--prompt", required=True, help="Prompt to jailbreak")
    lj.add_argument("--objective", default=None, help="Target behavior label (optional)")
    lj.add_argument("--start-layer", dest="start_layer", type=int, default=0, help="Start layer index (default: 0)")
    lj.add_argument("--end-layer", dest="end_layer", type=int, default=3, help="End layer index (default: 3)")
    lj.add_argument("--magnitude", type=float, default=1.0, help="Steering strength (default: 1.0)")
    lj.add_argument("--max-attempts", dest="max_attempts", type=int, default=20, help="Max attempts (default: 20)")
    lj.add_argument("--output-dir", dest="output_dir", default="_cli_runs/latent_jailbreak", help="Output directory")
    lj.set_defaults(func=_handle_latent_jailbreak)

    # PGD / APGD / BPDA / EOT / Memory-Augmented / AutoAttack (new offensive suite)
    pgd = subparsers.add_parser("pgd", help="🔴 PGD attack (baseline)")
    _add_common_attack_args(pgd)
    pgd.set_defaults(func=_handle_pgd)

    pgd_r = subparsers.add_parser("pgd-restarts", help="🔴 PGD with random restarts")
    _add_common_attack_args(pgd_r)
    pgd_r.add_argument("--restarts", type=int, default=10, help="Number of random restarts (default: 10)")
    pgd_r.set_defaults(func=_handle_pgd_restarts)

    apgd = subparsers.add_parser("apgd", help="🔴 APGD attack (adaptive step size)")
    _add_common_attack_args(apgd)
    apgd.add_argument(
        "--loss",
        choices=["ce", "dlr", "md", "cw", "logit", "logit_enhanced", "mm"],
        default="dlr",
        help="APGD loss (default: dlr)",
    )
    apgd.add_argument(
        "--loss-temperature",
        type=float,
        default=1.0,
        help="Temperature for logit_enhanced loss (default: 1.0)",
    )
    apgd.add_argument(
        "--loss-softmax-weighting",
        action="store_true",
        help="Enable softmax weighting for logit_enhanced loss",
    )
    apgd.add_argument("--restarts", type=int, default=1, help="Number of restarts (default: 1)")
    apgd.add_argument("--no-tg", action="store_true", help="Disable transformed gradients")
    apgd.set_defaults(func=_handle_apgd)

    apgd_ens = subparsers.add_parser("apgd-ensemble", help="🔴 APGD ensemble (CE + DLR by default)")
    _add_common_attack_args(apgd_ens)
    apgd_ens.add_argument(
        "--losses",
        default="ce,dlr",
        help="Comma-separated losses to include (default: ce,dlr)",
    )
    apgd_ens.set_defaults(func=_handle_apgd_ensemble)

    bpda = subparsers.add_parser("bpda", help="🔴 BPDA attack (non-differentiable defenses)")
    _add_common_attack_args(bpda)
    bpda.add_argument("--defense", required=True, choices=["jpeg", "thermometer", "rand-resize-pad", "identity"])
    bpda.add_argument(
        "--approx",
        default="identity",
        choices=[
            "identity",
            "jpeg",
            "thermometer",
            "quantization",
            "median_filter",
            "gaussian_blur",
            "random_resizing",
            "rand_resize_pad",
        ],
    )
    bpda.set_defaults(func=_handle_bpda)

    eot = subparsers.add_parser("eot", help="🔴 EOT attack (stochastic defenses)")
    _add_common_attack_args(eot)
    eot.add_argument("--transform", required=True, choices=["random-smoothing", "rand-resize-pad"])
    eot.add_argument("--samples", type=int, default=10, help="EOT samples (default: 10)")
    eot.set_defaults(func=_handle_eot)

    ma = subparsers.add_parser("ma-pgd", help="🔴 Memory-Augmented PGD (Volterra-guided)")
    _add_common_attack_args(ma)
    ma.add_argument("--alpha-volterra", type=float, default=None, help="Volterra alpha from Layer 2 (optional)")
    ma.add_argument("--auto-detect-alpha", action="store_true", help="Auto-detect alpha from gradients")
    ma.add_argument("--n-detection-steps", type=int, default=30, help="Steps for alpha detection (default: 30)")
    ma.add_argument("--memory-length", type=int, default=None, help="Gradient history length (optional)")
    ma.add_argument(
        "--kernel",
        choices=["power_law", "exponential", "exp", "uniform"],
        default="power_law",
        help="Volterra kernel type (default: power_law)",
    )
    ma.add_argument("--no-tg", action="store_true", help="Disable transformed gradients")
    ma.add_argument("--tg-scale", type=float, default=1.5, help="Transformed gradient scale (default: 1.5)")
    ma.add_argument("--tg-clip", type=float, default=3.0, help="Transformed gradient clip (default: 3.0)")
    ma.add_argument("--use-momentum", action="store_true", help="Enable momentum combination")
    ma.add_argument("--momentum-beta", type=float, default=0.75, help="Momentum beta (default: 0.75)")
    ma.set_defaults(func=_handle_ma_pgd)

    tm = subparsers.add_parser("temporal-momentum", help="🔴 Temporal momentum PGD")
    _add_common_attack_args(tm)
    tm.add_argument("--momentum", type=float, default=0.9, help="Momentum coefficient (default: 0.9)")
    tm.set_defaults(func=_handle_temporal_momentum)

    aa = subparsers.add_parser("autoattack", help="🔴 AutoAttack-style ensemble")
    _add_common_attack_args(aa)
    aa.add_argument("--include-square", action="store_true", help="Include Square attack (requires 4D input)")
    aa.add_argument(
        "--square-queries",
        type=int,
        default=1000,
        help="Square Attack query budget (default: 1000)",
    )
    aa.set_defaults(func=_handle_autoattack)

    # Defense characterization
    da = subparsers.add_parser(
        "defense-analyze",
        aliases=["defense-characterize", "characterize"],
        help="🔴 Defense characterization (Layer 1)",
    )
    _add_common_characterization_args(da)
    da.add_argument("--eps", type=float, default=0.031, help="Perturbation epsilon (default: 0.031)")
    da.add_argument("--n-samples", type=int, default=50, help="Gradient samples (default: 50)")
    da.add_argument("--n-probe-images", type=int, default=100, help="Probe images (default: 100)")
    da.add_argument("--batch-size", type=int, default=32, help="Batch size for characterization")
    da.add_argument("--quiet", action="store_true", help="Disable verbose characterization logging")
    da.set_defaults(func=_handle_defense_analyze)

    # Attack orchestrator
    orch = subparsers.add_parser(
        "attack-orchestrate",
        aliases=["orchestrate", "attack-orchestrator"],
        help="🔴 Defense-aware attack orchestration (Layer 2)",
    )
    _add_common_attack_args(orch)
    orch.add_argument("--characterize-input", dest="characterize_input", default=None, help="Inputs .npy for characterization")
    orch.add_argument("--characterize-labels", dest="characterize_labels", default=None, help="Labels .npy for characterization")
    orch.add_argument("--characterize-samples", type=int, default=50, help="Gradient samples (default: 50)")
    orch.add_argument("--characterize-probe-images", type=int, default=100, help="Probe images (default: 100)")
    orch.add_argument("--characterize-batch-size", type=int, default=32, help="Batch size for characterization")
    orch.add_argument("--quiet", action="store_true", help="Disable verbose orchestration logging")
    orch.set_defaults(func=_handle_attack_orchestrate)

    # Paper claim reproduction
    pc = subparsers.add_parser(
        "reproduce-claims",
        aliases=["paper-claims", "paper-claim-validation"],
        help="[DISABLED] Legacy paper-claim validation (use Click CLI evaluate/table2)",
    )
    pc.add_argument(
        "--claim",
        choices=["all", "figure_1", "table_2", "table_3", "table_5", "table_8"],
        default="all",
        help="Which claim to validate (default: all)",
    )
    pc.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu", help="Device")
    pc.add_argument("--n-seeds", type=int, default=2, help="Random seeds (default: 2)")
    pc.add_argument("--results-dir", default="_cli_runs/validation_results", help="Results directory")
    pc.add_argument(
        "--mode",
        choices=["fast", "strict"],
        default="fast",
        help="Deprecated (command is disabled)",
    )
    pc.add_argument("--no-plot", action="store_true", help="Disable figure plotting")
    pc.add_argument("--output", default=None, help="Optional JSON output path")
    pc.set_defaults(func=_handle_reproduce_claims)

    # Table 1 reproduction
    rt1 = subparsers.add_parser(
        "reproduce-table1",
        aliases=["table1", "reproduce-table-1"],
        help="[DISABLED] Legacy table runner (use Click CLI evaluate/table2)",
    )
    rt1.add_argument(
        "--config",
        default="experiments/configs/table1_config.yaml",
        help="Path to Table 1 YAML config",
    )
    rt1.add_argument("--device", choices=["cpu", "cuda", "mps", "auto"], default="auto", help="Device")
    rt1.add_argument("--results-dir", default="_cli_runs/table1", help="Results directory")
    rt1.add_argument("--no-plot", action="store_true", help="Disable plot generation")
    rt1.add_argument("--allow-missing", action="store_true", help="Skip missing datasets/models")
    rt1.add_argument("--output", default=None, help="Optional JSON output path")
    rt1.set_defaults(func=_handle_reproduce_table1)

    # Comprehensive reproduction suite
    ra = subparsers.add_parser(
        "reproduce-all",
        aliases=["reproduce-all-tables", "reproduce-suite"],
        help="[DISABLED] Legacy reproduction suite (use Click CLI evaluate/table2)",
    )
    ra.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu", help="Device")
    ra.add_argument("--n-seeds", type=int, default=2, help="Random seeds (default: 2)")
    ra.add_argument(
        "--table1-config",
        default="experiments/configs/table1_config.yaml",
        help="Path to Table 1 YAML config",
    )
    ra.add_argument("--allow-missing", action="store_true", help="Skip missing datasets/models")
    ra.add_argument(
        "--mode",
        choices=["fast", "strict"],
        default="fast",
        help="Deprecated (command is disabled)",
    )
    ra.add_argument("--results-dir", default="_cli_runs/comprehensive_results", help="Results directory")
    ra.add_argument("--no-plot", action="store_true", help="Disable plot generation")
    ra.add_argument("--output", default=None, help="Optional JSON output path")
    ra.set_defaults(func=_handle_reproduce_all_tables)

    # Testing context report
    tc = subparsers.add_parser(
        "testing-context",
        aliases=["testing_context", "testing-report"],
        help="🧪 Print TESTING_CONTEXT.md from the repo root",
    )
    tc.add_argument("--path", default=None, help="Optional path to a context file")
    tc.add_argument(
        "--format",
        choices=["markdown", "json"],
        default="markdown",
        help="Output format (default: markdown)",
    )
    tc.add_argument("--output", default=None, help="Optional output file path")
    tc.add_argument("--quiet", action="store_true", help="Do not print to stdout")
    tc.set_defaults(func=_handle_testing_context)


def _handle_ts_inverse(args) -> int:
    try:
        from .adversarial_ts_inverse import run_ts_inverse

        return int(run_ts_inverse(args))
    except Exception as e:
        logger.error(f"TS-Inverse CLI failed: {e}")
        return 1


def _handle_concretizer(args) -> int:
    try:
        from .adversarial_concretizer import run_concretizer

        return int(run_concretizer(args))
    except Exception as e:
        logger.error(f"ConcreTizer CLI failed: {e}")
        return 1


def _handle_latent_jailbreak(args) -> int:
    try:
        from .latent_jailbreak import run_latent_jailbreak

        return int(run_latent_jailbreak(args))
    except Exception as e:
        logger.error(f"Latent jailbreak CLI failed: {e}")
        return 1


def _add_common_attack_args(parser) -> None:
    parser.add_argument("--input", required=True, help="Input samples .npy")
    parser.add_argument("--labels", required=True, help="Input labels .npy")
    parser.add_argument("--model-factory", default=None, help="Python factory 'module:function' (optional)")
    parser.add_argument("--model-kwargs", default=None, help="JSON kwargs for model factory (optional)")
    parser.add_argument("--allow-demo-model", action="store_true", help="Allow built-in demo model")
    parser.add_argument("--num-classes", type=int, default=None, help="Override num classes (optional)")
    parser.add_argument("--eps", type=float, default=0.031, help="Attack epsilon")
    parser.add_argument("--alpha", type=float, default=0.003, help="Step size")
    parser.add_argument("--steps", type=int, default=40, help="Attack steps")
    parser.add_argument("--norm", choices=["linf", "l2"], default="linf", help="Attack norm")
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu", help="Device")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--output", default=None, help="Optional JSON output path")
    parser.add_argument("--no-grad-stats", action="store_true", help="Skip gradient stats in evaluation")


def _add_common_characterization_args(parser) -> None:
    parser.add_argument("--input", required=True, help="Input samples .npy")
    parser.add_argument("--labels", required=True, help="Input labels .npy")
    parser.add_argument("--model-factory", default=None, help="Python factory 'module:function' (optional)")
    parser.add_argument("--model-kwargs", default=None, help="JSON kwargs for model factory (optional)")
    parser.add_argument("--allow-demo-model", action="store_true", help="Allow built-in demo model")
    parser.add_argument("--num-classes", type=int, default=None, help="Override num classes (optional)")
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu", help="Device")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--output", default=None, help="Optional JSON output path")


def _load_npy_array(path: str):
    import numpy as np

    obj = np.load(path, allow_pickle=True)
    if getattr(obj, "dtype", None) is object and getattr(obj, "shape", ()) == ():
        obj = obj.item()
    if isinstance(obj, dict):
        for k in ("data", "X", "x", "arr", "samples", "inputs"):
            if k in obj:
                obj = obj[k]
                break
    return np.asarray(obj)


def _prepare_data(args):
    import numpy as np
    import torch

    x = _load_npy_array(args.input)
    y = _load_npy_array(args.labels)
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64).reshape(-1)

    if x.ndim == 2:
        # ensure batch dimension for 2D inputs
        pass
    elif x.ndim == 1:
        x = x.reshape(-1, 1)
    elif x.ndim > 4:
        x = x.reshape(x.shape[0], -1)

    return torch.from_numpy(x), torch.from_numpy(y)


def _parse_model_factory(args):
    import importlib
    import json

    if not getattr(args, "model_factory", None):
        return None
    spec = str(args.model_factory)
    if ":" not in spec:
        raise ValueError("Invalid --model-factory. Expected 'module:function'.")
    mod_name, fn_name = spec.split(":", 1)
    mod = importlib.import_module(mod_name)
    fn = getattr(mod, fn_name)
    if not callable(fn):
        raise ValueError("Model factory is not callable.")
    kwargs = {}
    if getattr(args, "model_kwargs", None):
        kwargs = json.loads(str(args.model_kwargs))
        if not isinstance(kwargs, dict):
            raise ValueError("model_kwargs must be a JSON dict")
    return fn, kwargs


def _create_demo_model(input_shape, num_classes: int):
    import numpy as np
    import torch.nn as nn

    in_dim = int(np.prod(input_shape[1:])) if len(input_shape) > 1 else int(input_shape[0])
    return nn.Sequential(nn.Flatten(), nn.Linear(in_dim, num_classes))


def _prepare_model(args, x, y):
    import torch

    model = None
    parsed = _parse_model_factory(args)
    if parsed is not None:
        fn, kwargs = parsed
        model = fn(**kwargs)
    else:
        if not getattr(args, "allow_demo_model", False):
            raise ValueError("Provide --model-factory or pass --allow-demo-model.")
        num_classes = int(args.num_classes) if args.num_classes is not None else int(torch.unique(y).numel())
        model = _create_demo_model(tuple(x.shape), num_classes)
    return model


def _emit_result(result: dict, output_path: str | None):
    if output_path:
        import json
        from pathlib import Path

        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("✅ Result:", result)


def _run_attack(args, attack_factory):
    import torch
    from neurinspectre.utils import AttackEvaluator

    torch.manual_seed(int(args.seed))
    x, y = _prepare_data(args)
    model = _prepare_model(args, x, y)
    attack = attack_factory(model)
    evaluator = AttackEvaluator(model, device=args.device)
    result = evaluator.evaluate_single_batch(attack, x, y, compute_grad_stats=not args.no_grad_stats)
    _emit_result(result, args.output)
    return 0


def _handle_pgd(args) -> int:
    from neurinspectre.attacks import PGD

    return _run_attack(
        args,
        lambda model: PGD(
            model,
            eps=args.eps,
            alpha=args.alpha,
            steps=args.steps,
            norm=args.norm,
            device=args.device,
        ),
    )


def _handle_pgd_restarts(args) -> int:
    from neurinspectre.attacks import PGDWithRestarts

    return _run_attack(
        args,
        lambda model: PGDWithRestarts(
            model,
            n_restarts=args.restarts,
            eps=args.eps,
            alpha=args.alpha,
            steps=args.steps,
            norm=args.norm,
            device=args.device,
        ),
    )


def _handle_apgd(args) -> int:
    from neurinspectre.attacks import APGD

    loss_params = None
    if args.loss in {"logit_enhanced", "enhanced_margin"}:
        loss_params = {
            "temperature": float(args.loss_temperature),
            "use_softmax_weighting": bool(args.loss_softmax_weighting),
        }

    return _run_attack(
        args,
        lambda model: APGD(
            model,
            eps=args.eps,
            norm=args.norm,
            steps=args.steps,
            loss=args.loss,
            n_restarts=args.restarts,
            use_tg=not args.no_tg,
            loss_params=loss_params,
            device=args.device,
        ),
    )


def _handle_apgd_ensemble(args) -> int:
    from neurinspectre.attacks import APGDEnsemble

    losses = [s.strip() for s in str(args.losses).split(",") if s.strip()]
    return _run_attack(
        args,
        lambda model: APGDEnsemble(
            model,
            eps=args.eps,
            norm=args.norm,
            steps=args.steps,
            losses=losses,
            device=args.device,
        ),
    )


def _handle_bpda(args) -> int:
    from neurinspectre.attacks import BPDA
    from neurinspectre.defenses import jpeg_defense, random_resize_pad, thermometer_defense

    defense_map = {
        "jpeg": jpeg_defense,
        "thermometer": thermometer_defense,
        "rand-resize-pad": random_resize_pad,
        "identity": lambda t: t,
    }
    defense_fn = defense_map[args.defense]
    return _run_attack(
        args,
        lambda model: BPDA(
            model,
            defense=defense_fn,
            approx_name=args.approx,
            eps=args.eps,
            alpha=args.alpha,
            steps=args.steps,
            norm=args.norm,
            device=args.device,
        ),
    )


def _handle_eot(args) -> int:
    from neurinspectre.attacks import EOT
    from neurinspectre.defenses import random_resize_pad, randomized_smoothing

    transform_map = {
        "random-smoothing": lambda t: randomized_smoothing(t, sigma=0.05),
        "rand-resize-pad": random_resize_pad,
    }
    transform_fn = transform_map[args.transform]
    return _run_attack(
        args,
        lambda model: EOT(
            model,
            transform_fn=transform_fn,
            num_samples=args.samples,
            eps=args.eps,
            alpha=args.alpha,
            steps=args.steps,
            norm=args.norm,
            device=args.device,
        ),
    )


def _handle_ma_pgd(args) -> int:
    from neurinspectre.attacks import MAPGD

    kernel = args.kernel
    if kernel == "exp":
        kernel = "exponential"

    return _run_attack(
        args,
        lambda model: MAPGD(
            model,
            alpha_volterra=args.alpha_volterra,
            memory_length=args.memory_length,
            auto_detect_alpha=args.auto_detect_alpha,
            n_detection_steps=args.n_detection_steps,
            eps=args.eps,
            alpha=args.alpha,
            steps=args.steps,
            norm=args.norm,
            kernel_type=kernel,
            use_tg=not args.no_tg,
            tg_scale=args.tg_scale,
            tg_clip=args.tg_clip,
            use_momentum=args.use_momentum,
            momentum_beta=args.momentum_beta,
            device=args.device,
        ),
    )


def _handle_temporal_momentum(args) -> int:
    from neurinspectre.attacks import TemporalMomentumPGD

    return _run_attack(
        args,
        lambda model: TemporalMomentumPGD(
            model,
            eps=args.eps,
            alpha=args.alpha,
            steps=args.steps,
            norm=args.norm,
            momentum=args.momentum,
            device=args.device,
        ),
    )


def _handle_autoattack(args) -> int:
    from neurinspectre.attacks import APGD, AutoAttackEnsemble, FAB, SquareAttack, SquareAttackL2

    def _factory(model):
        attacks = [
            APGD(model, eps=args.eps, norm=args.norm, steps=max(10, args.steps), loss="ce", device=args.device),
            APGD(model, eps=args.eps, norm=args.norm, steps=max(10, args.steps), loss="dlr", device=args.device),
            FAB(model, norm=args.norm, steps=max(10, args.steps), n_restarts=1, device=args.device),
        ]
        if args.include_square:
            n_queries = max(1000, int(args.square_queries))
            if args.norm == "l2":
                attacks.append(SquareAttackL2(model, eps=args.eps, n_queries=n_queries, device=args.device))
            else:
                attacks.append(SquareAttack(model, eps=args.eps, n_queries=n_queries, device=args.device))
        return AutoAttackEnsemble(model, attacks, device=args.device)

    return _run_attack(args, _factory)


def _handle_defense_analyze(args) -> int:
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    from neurinspectre.characterization import DefenseAnalyzer

    torch.manual_seed(int(args.seed))
    x, y = _prepare_data(args)
    model = _prepare_model(args, x, y)
    model = model.to(args.device)
    model.eval()

    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=int(args.batch_size), shuffle=False)
    analyzer = DefenseAnalyzer(
        model,
        n_samples=int(args.n_samples),
        n_probe_images=int(args.n_probe_images),
        device=args.device,
        verbose=not args.quiet,
    )
    char = analyzer.characterize(loader, eps=args.eps)
    _emit_result(char.to_dict(), args.output)
    return 0


def _handle_attack_orchestrate(args) -> int:
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    from neurinspectre.attacks import AttackOrchestrator
    from neurinspectre.utils import AttackEvaluator

    torch.manual_seed(int(args.seed))
    x, y = _prepare_data(args)
    model = _prepare_model(args, x, y)
    model = model.to(args.device)

    if args.characterize_input and args.characterize_labels:
        x_char = _load_npy_array(args.characterize_input)
        y_char = _load_npy_array(args.characterize_labels)
        x_char = np.asarray(x_char, dtype=np.float32)
        y_char = np.asarray(y_char, dtype=np.int64).reshape(-1)
        if x_char.ndim == 1:
            x_char = x_char.reshape(-1, 1)
        elif x_char.ndim > 4:
            x_char = x_char.reshape(x_char.shape[0], -1)
        x_char = torch.from_numpy(x_char)
        y_char = torch.from_numpy(y_char)
    else:
        x_char, y_char = x, y

    char_dataset = TensorDataset(x_char, y_char)
    char_loader = DataLoader(
        char_dataset, batch_size=int(args.characterize_batch_size), shuffle=False
    )

    attack = AttackOrchestrator(
        model,
        eps=args.eps,
        steps=args.steps,
        norm=args.norm,
        auto_characterize_data=char_loader,
        device=args.device,
        verbose=not args.quiet,
    )
    evaluator = AttackEvaluator(model, device=args.device)
    result = evaluator.evaluate_single_batch(attack, x, y, compute_grad_stats=not args.no_grad_stats)
    if attack.characterization is not None:
        result["characterization"] = attack.characterization.to_dict()
    _emit_result(result, args.output)
    return 0


def _handle_reproduce_claims(args) -> int:
    import sys

    print(
        "[ERROR] Paper-claim reproduction helpers are intentionally not shipped in-repo.\n"
        "Use the Click CLI evaluation pipeline instead:\n"
        "  - neurinspectre evaluate --config <path>\n"
        "  - neurinspectre table2 --config <path> --strict-real-data\n"
        "If you want baseline validation, pass expected values via an external file "
        "(e.g., neurinspectre compare --mode baseline --expected-asr-path <path>).",
        file=sys.stderr,
    )
    return 2


def _handle_reproduce_table1(args) -> int:
    import sys

    print(
        "[ERROR] Legacy table reproduction commands are disabled.\n"
        "Use the Click CLI evaluation pipeline:\n"
        "  neurinspectre evaluate --config <path> --results-dir <dir> --output <json>\n"
        "or run the Table 2 matrix runner:\n"
        "  neurinspectre table2 --config <path> --strict-real-data",
        file=sys.stderr,
    )
    return 2


def _handle_reproduce_all_tables(args) -> int:
    import sys

    print(
        "[ERROR] Legacy 'reproduce-all' is disabled.\n"
        "Use the Click CLI commands:\n"
        "  - neurinspectre evaluate --config <path>\n"
        "  - neurinspectre table2 --config <path> --strict-real-data",
        file=sys.stderr,
    )
    return 2


def _handle_testing_context(args) -> int:
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    ctx_path = Path(args.path) if args.path else (root / "TESTING_CONTEXT.md")

    if not ctx_path.exists():
        raise FileNotFoundError(f"Testing context not found at {ctx_path}")

    content = ctx_path.read_text(encoding="utf-8")

    if args.format == "json":
        payload = {"path": str(ctx_path), "content": content}
        _emit_result(payload, args.output)
        return 0

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(content, encoding="utf-8")

    if not args.quiet:
        print(content)
    return 0


