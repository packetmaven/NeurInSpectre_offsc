import torch
import torch.nn as nn

from neurinspectre.attacks import MemoryAugmentedPGD, PGD


class TemporalDefenseModel(nn.Module):
    """Toy model with a simple temporal-memory effect."""

    def __init__(self, in_dim=16, num_classes=3, decay=0.8):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)
        self.decay = float(decay)
        self.register_buffer("_ema", torch.zeros(1, in_dim))

    def reset_state(self, batch_size: int):
        self._ema = torch.zeros(batch_size, self._ema.size(1), device=self._ema.device)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        if self._ema.size(0) != x.size(0):
            self.reset_state(x.size(0))

        # Use previous EMA in the forward pass, update EMA without backprop.
        out_feat = self.decay * self._ema + (1.0 - self.decay) * x
        with torch.no_grad():
            self._ema = self.decay * self._ema + (1.0 - self.decay) * x.detach()
        return self.fc(out_feat)


def _make_batch(batch_size=16):
    x = torch.rand(batch_size, 1, 4, 4)
    y = torch.randint(0, 3, (batch_size,))
    return x, y


def test_memory_augmented_improves_asr_on_temporal_defense():
    torch.manual_seed(7)
    model = TemporalDefenseModel()
    x, y = _make_batch()

    with torch.no_grad():
        model.fc.weight.zero_()
        model.fc.bias.zero_()
        model.fc.weight[0].fill_(1.0)
        model.fc.weight[1].fill_(-1.0)
        model.fc.bias[0] = -7.5
        model.fc.bias[1] = 7.5

    with torch.no_grad():
        y = model(x).argmax(1)

    pgd = PGD(model, eps=0.2, alpha=0.01, steps=10, device="cpu")
    ma_pgd = MemoryAugmentedPGD(model, alpha_volterra=0.5, eps=0.2, alpha=0.05, steps=12, device="cpu")

    model.reset_state(x.size(0))
    adv_pgd = pgd(x, y)
    with torch.no_grad():
        preds_pgd = model(adv_pgd).argmax(1)
    asr_pgd = float((preds_pgd != y).float().mean().item())

    model.reset_state(x.size(0))
    adv_ma = ma_pgd(x, y)
    with torch.no_grad():
        preds_ma = model(adv_ma).argmax(1)
    asr_ma = float((preds_ma != y).float().mean().item())

    assert asr_ma > asr_pgd
