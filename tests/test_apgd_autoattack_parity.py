"""
Test: Verify APGD matches AutoAttack canonical implementation.

Cross-ref: RobustBench test suite
Cross-ref: AutoAttack frozen version (v0.1)
"""

import pytest
import torch

from neurinspectre.attacks import APGDAttack, AttackConfig


@pytest.mark.slow
def test_apgd_autoattack_parity():
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        device = "cuda"
    else:
        pytest.skip("CUDA not available for AutoAttack parity check.")

    autoattack = pytest.importorskip("autoattack")
    AutoAttack = autoattack.AutoAttack

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 32, 3, padding=1),
        torch.nn.ReLU(),
        torch.nn.Flatten(),
        torch.nn.Linear(32 * 32 * 32, 10),
    ).to(device)
    model.eval()

    x = torch.rand(100, 3, 32, 32, device=device)
    y = torch.randint(0, 10, (100,), device=device)

    config = AttackConfig(
        epsilon=8 / 255,
        n_iterations=100,
        loss="dlr",
        random_init=True,
        seed=42,
        use_tg=False,
        auto_detect_range=False,
        input_range=(0.0, 1.0),
    )
    our_apgd = APGDAttack(config, device=device, loss_type="dlr")

    result_ours = our_apgd.run(model, x, y, targeted=False)
    our_asr = float(result_ours.success_rate)

    adversary = AutoAttack(
        model,
        norm="Linf",
        eps=8 / 255,
        version="custom",
        attacks_to_run=["apgd-dlr"],
    )
    if hasattr(adversary, "apgd"):
        adversary.apgd.n_iter = 100
        if hasattr(adversary.apgd, "seed"):
            adversary.apgd.seed = 42

    x_adv_ref = adversary.run_standard_evaluation(x, y, bs=100)

    with torch.no_grad():
        logits_ref = model(x_adv_ref)
        preds_ref = logits_ref.argmax(dim=1)
        success_ref = preds_ref != y
    ref_asr = float(success_ref.float().mean().item())

    asr_diff = abs(our_asr - ref_asr)

    print(f"Our APGD ASR: {our_asr:.2%}")
    print(f"AutoAttack ASR: {ref_asr:.2%}")
    print(f"Difference: {asr_diff:.2%}")

    assert asr_diff < 0.05, f"ASR difference {asr_diff:.2%} exceeds 5% tolerance"
