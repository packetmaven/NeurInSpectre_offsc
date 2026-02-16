from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Subset

from neurinspectre.evaluation.datasets import DatasetFactory


def _write_rgb_png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8), mode="RGB")
    img.save(path)


def test_imagenet100_respects_root_parameter(tmp_path: Path) -> None:
    """
    Regression test for Issue #1:
    ImageNet100Dataset must use the provided `root`, not any hard-coded ./data path.
    """
    root = tmp_path / "imagenet"
    _write_rgb_png(root / "val" / "n00000001" / "img0.png")
    _write_rgb_png(root / "train" / "n00000001" / "img0.png")

    loader, x, y = DatasetFactory.get_dataset(
        "imagenet100",
        root=str(root),
        split="val",
        n_samples=1,
        batch_size=1,
        num_workers=0,
        pin_memory=False,
    )

    # The loader uses a Subset(ImageFolder(...)).
    assert isinstance(loader.dataset, Subset)
    imgfolder = loader.dataset.dataset
    assert Path(imgfolder.root) == (root / "val")

    # Ensure the underlying samples come from the requested root.
    for p, _label in imgfolder.samples:
        assert str(p).startswith(str(root / "val"))

    assert int(x.shape[0]) == 1
    assert int(y.shape[0]) == 1


def test_imagenet100_allows_root_pointing_to_split_dir(tmp_path: Path) -> None:
    root = tmp_path / "imagenet"
    _write_rgb_png(root / "val" / "n00000001" / "img0.png")
    _write_rgb_png(root / "train" / "n00000001" / "img0.png")

    # Passing root=<...>/val should still load validation split without constructing <...>/val/val.
    loader, _x, _y = DatasetFactory.get_dataset(
        "imagenet100",
        root=str(root / "val"),
        split="val",
        n_samples=1,
        batch_size=1,
        num_workers=0,
        pin_memory=False,
    )
    assert isinstance(loader.dataset, Subset)
    imgfolder = loader.dataset.dataset
    assert Path(imgfolder.root) == (root / "val")

