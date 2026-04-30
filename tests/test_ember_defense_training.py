import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from neurinspectre.training.ember_defenses import (
    build_mlp,
    train_at_transform,
    train_distilled_student,
    train_gradient_regularized,
    train_standard,
)


def _toy_loader(*, n: int = 64, d: int = 8, seed: int = 0) -> DataLoader:
    rng = np.random.default_rng(int(seed))
    X = rng.standard_normal((n, d), dtype=np.float32)
    y = (rng.random((n,)) > 0.5).astype(np.int64)
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    return DataLoader(ds, batch_size=16, shuffle=True, num_workers=0)


def test_train_standard_runs():
    loader = _toy_loader()
    model = build_mlp(input_dim=8, num_classes=2, hidden1=32, hidden2=16, dropout=0.0)
    stats = train_standard(model, loader, device="cpu", epochs=1, lr=1e-3)
    assert stats.epochs == 1
    assert np.isfinite(stats.last_loss)


def test_train_gradient_regularized_runs():
    loader = _toy_loader()
    model = build_mlp(input_dim=8, num_classes=2, hidden1=32, hidden2=16, dropout=0.0)
    stats = train_gradient_regularized(model, loader, device="cpu", epochs=1, lr=1e-3, lambda_grad=0.1)
    assert stats.epochs == 1
    assert np.isfinite(stats.last_loss)


def test_train_distilled_student_runs():
    loader = _toy_loader()
    teacher = build_mlp(input_dim=8, num_classes=2, hidden1=32, hidden2=16, dropout=0.0)
    student = build_mlp(input_dim=8, num_classes=2, hidden1=16, hidden2=8, dropout=0.0)
    # Quick teacher warm-start (not the focus of the test).
    _ = train_standard(teacher, loader, device="cpu", epochs=1, lr=1e-3)
    stats = train_distilled_student(
        teacher=teacher,
        student=student,
        train_loader=loader,
        device="cpu",
        epochs=1,
        lr=1e-3,
        temperature=10.0,
        alpha_hard=0.2,
    )
    assert stats.epochs == 1
    assert np.isfinite(stats.last_loss)


def test_train_at_transform_runs():
    loader = _toy_loader()
    model = build_mlp(input_dim=8, num_classes=2, hidden1=32, hidden2=16, dropout=0.0)
    stats = train_at_transform(
        model,
        loader,
        device="cpu",
        epochs=1,
        lr=1e-3,
        noise_std=0.05,
        epsilon_l2=0.5,
        pgd_steps=1,
    )
    assert stats.epochs == 1
    assert np.isfinite(stats.last_loss)

