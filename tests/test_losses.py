import torch

from neurinspectre.losses.dlr_loss import dlr_loss, ce_loss
from neurinspectre.losses.cw_loss import cw_loss
from neurinspectre.losses.md_loss import md_loss


def test_dlr_loss_shapes():
    logits = torch.randn(5, 4)
    y = torch.tensor([0, 1, 2, 3, 1])
    loss = dlr_loss(logits, y, reduction="none")
    assert loss.shape == (5,)
    assert torch.isfinite(loss).all()


def test_cw_loss_shapes():
    logits = torch.randn(3, 5)
    y = torch.tensor([0, 1, 2])
    loss = cw_loss(logits, y, reduction="none")
    assert loss.shape == (3,)
    assert torch.isfinite(loss).all()


def test_md_loss_shapes():
    logits = torch.randn(4, 6)
    y = torch.tensor([0, 1, 2, 3])
    loss = md_loss(logits, y, reduction="none")
    assert loss.shape == (4,)
    assert torch.isfinite(loss).all()


def test_ce_loss_shapes():
    logits = torch.randn(4, 6)
    y = torch.tensor([0, 1, 2, 3])
    loss = ce_loss(logits, y, reduction="none")
    assert loss.shape == (4,)
