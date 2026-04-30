import torch
import torch.nn as nn

from neurinspectre.defenses.wrappers import TentDefense


def _tiny_bn_model() -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(3, 4, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(4),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(4, 10),
    )


def test_tent_defense_updates_bn_params():
    torch.manual_seed(0)
    model = _tiny_bn_model()
    defense = TentDefense(model, lr=0.1, steps=1, reset_each_forward=False, device="cpu")

    bn = next(m for m in defense.base_model.modules() if isinstance(m, nn.BatchNorm2d))
    w0 = bn.weight.detach().clone()

    x = torch.rand(8, 3, 8, 8)
    _ = defense(x)
    w1 = bn.weight.detach().clone()

    assert not torch.allclose(w0, w1), "Expected TENT to update BN affine parameters"


def test_tent_defense_reset_each_forward_is_deterministic():
    torch.manual_seed(0)
    model = _tiny_bn_model()
    defense = TentDefense(model, lr=0.1, steps=1, reset_each_forward=True, device="cpu")

    bn = next(m for m in defense.base_model.modules() if isinstance(m, nn.BatchNorm2d))
    x = torch.rand(8, 3, 8, 8)

    _ = defense(x)
    w_after_first = bn.weight.detach().clone()
    _ = defense(x)
    w_after_second = bn.weight.detach().clone()

    assert torch.allclose(w_after_first, w_after_second), "reset_each_forward should reset BN state"

