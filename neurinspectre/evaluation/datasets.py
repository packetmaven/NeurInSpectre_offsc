"""
Real dataset loading for NeurInSpectre Table 1 reproduction.

Implements proper dataset handling for:
- CIFAR-10 (Content Moderation domain)
- ImageNet-100 subset (Content Moderation domain)
- EMBER (Malware Detection domain)
- nuScenes (AV Perception domain)
"""

from __future__ import annotations

import json
import os
import tarfile
import urllib.request
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, Subset
from PIL import Image


class DatasetConfig:
    """Configuration for dataset loading."""

    CIFAR10_PATH = "./data/cifar10"
    IMAGENET_PATH = "./data/imagenet"
    EMBER_PATH = "./data/ember"
    NUSCENES_PATH = "./data/nuscenes"

    IMAGENET_100_CLASSES = list(range(100))

    EVAL_SAMPLES = {
        "cifar10": 1000,
        "imagenet100": 1000,
        "ember": 1000,
        "nuscenes": 500,
    }


class CIFAR10Dataset:
    """CIFAR-10 dataset for Content Moderation evaluations."""

    @staticmethod
    def load(
        root: str = DatasetConfig.CIFAR10_PATH,
        n_samples: int = DatasetConfig.EVAL_SAMPLES["cifar10"],
        seed: int = 42,
        batch_size: int = 100,
        num_workers: int = 4,
        split: str = "test",
        download: bool = True,
        pin_memory: bool = True,
    ) -> Tuple[DataLoader, torch.Tensor, torch.Tensor]:
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

        train = str(split).lower() in {"train", "training"}
        testset = torchvision.datasets.CIFAR10(
            root=root,
            train=train,
            download=bool(download),
            transform=transform,
        )

        torch.manual_seed(seed)
        np.random.seed(seed)
        if n_samples <= 0 or n_samples >= len(testset):
            indices = np.arange(len(testset))
        else:
            indices = np.random.choice(len(testset), n_samples, replace=False)
        subset = Subset(testset, indices)

        loader = DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=bool(pin_memory),
        )

        x_test = torch.stack([testset[i][0] for i in indices])
        y_test = torch.tensor([testset[i][1] for i in indices])
        return loader, x_test, y_test


class ImageNet100Dataset:
    """ImageNet-100 subset for Content Moderation."""

    @staticmethod
    def load(
        root: str = DatasetConfig.IMAGENET_PATH,
        n_samples: int = DatasetConfig.EVAL_SAMPLES["imagenet100"],
        seed: int = 42,
        batch_size: int = 50,
        num_workers: int = 4,
        split: str = "val",
        pin_memory: bool = True,
    ) -> Tuple[DataLoader, torch.Tensor, torch.Tensor]:
        if not os.path.exists(root):
            raise FileNotFoundError(
                f"ImageNet not found at {root}. "
                "Download from https://www.image-net.org/download.php"
            )

        transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
            ]
        )

        split_dir = "val" if str(split).lower() in {"val", "valid", "validation"} else "train"
        valset = torchvision.datasets.ImageFolder(
            root=os.path.join(root, split_dir),
            transform=transform,
        )

        valid_indices = [
            i for i, (_, label) in enumerate(valset.samples) if label < 100
        ]

        torch.manual_seed(seed)
        np.random.seed(seed)
        if n_samples <= 0 or n_samples >= len(valid_indices):
            sampled_indices = np.array(valid_indices)
        else:
            sampled_indices = np.random.choice(
                valid_indices,
                min(n_samples, len(valid_indices)),
                replace=False,
            )

        subset = Subset(valset, sampled_indices)
        loader = DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=bool(pin_memory),
        )

        x_test = torch.stack([valset[i][0] for i in sampled_indices])
        y_test = torch.tensor([valset[i][1] for i in sampled_indices])
        return loader, x_test, y_test


class EMBERDataset:
    """EMBER malware detection dataset."""

    @staticmethod
    def download_ember(root: str, year: str = "2018") -> None:
        os.makedirs(root, exist_ok=True)
        base_url = "https://ember.elastic.co/ember_dataset_2018_2.tar.bz2"
        tar_path = os.path.join(root, f"ember_{year}.tar.bz2")

        if not os.path.exists(tar_path):
            print("[EMBER] Downloading EMBER2018 dataset...")
            urllib.request.urlretrieve(base_url, tar_path)
            print("[EMBER] Download complete.")

        extract_path = os.path.join(root, f"ember_{year}")
        if not os.path.exists(extract_path):
            print("[EMBER] Extracting...")
            with tarfile.open(tar_path, "r:bz2") as tar:
                tar.extractall(root)
            print("[EMBER] Extraction complete.")

    @staticmethod
    def load(
        root: str = DatasetConfig.EMBER_PATH,
        n_samples: int = DatasetConfig.EVAL_SAMPLES["ember"],
        seed: int = 42,
        batch_size: int = 100,
        num_workers: int = 2,
        split: str = "test",
        pin_memory: bool = True,
    ) -> Tuple[DataLoader, torch.Tensor, torch.Tensor]:
        EMBERDataset.download_ember(root)
        try:
            import ember
        except ImportError as exc:
            raise ImportError("Please install ember: pip install ember") from exc

        data_dir = os.path.join(root, "ember_2018")
        subset = "train" if str(split).lower() in {"train", "training"} else "test"
        x_test, y_test = ember.read_vectorized_features(data_dir, subset=subset)

        x_test = torch.from_numpy(x_test).float()
        y_test = torch.from_numpy(y_test).long()

        labeled_mask = y_test >= 0
        x_test = x_test[labeled_mask]
        y_test = y_test[labeled_mask]

        torch.manual_seed(seed)
        np.random.seed(seed)
        if n_samples <= 0 or n_samples >= len(x_test):
            indices = torch.arange(len(x_test))
        else:
            indices = torch.randperm(len(x_test))[:n_samples]
        x_test = x_test[indices]
        y_test = y_test[indices]

        dataset = torch.utils.data.TensorDataset(x_test, y_test)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=bool(pin_memory),
        )
        return loader, x_test, y_test


class nuScenesDataset:
    """nuScenes perception dataset for AV domain."""

    @staticmethod
    def load(
        root: str = DatasetConfig.NUSCENES_PATH,
        n_samples: int = DatasetConfig.EVAL_SAMPLES["nuscenes"],
        seed: int = 42,
        version: str = "v1.0-mini",
        labels_path: str | None = None,
        image_size: tuple[int, int] = (224, 224),
        batch_size: int = 16,
        num_workers: int = 2,
        pin_memory: bool = True,
    ) -> Tuple[DataLoader, torch.Tensor, torch.Tensor]:
        if not os.path.exists(root):
            raise FileNotFoundError(
                f"nuScenes not found at {root}. "
                "Download from https://www.nuscenes.org/nuscenes#download"
            )

        try:
            from nuscenes.nuscenes import NuScenes
        except ImportError as exc:
            raise ImportError(
                "Please install nuscenes-devkit: pip install nuscenes-devkit"
            ) from exc

        if labels_path is None:
            raise ValueError(
                "nuScenes loader requires labels_path mapping sample_token -> class label."
            )

        with open(labels_path, "r", encoding="utf-8") as handle:
            label_map = json.load(handle)

        nusc = NuScenes(version=version, dataroot=root, verbose=False)
        camera_channel = "CAM_FRONT"
        samples: List[Dict] = []

        for scene in nusc.scene:
            sample_token = scene["first_sample_token"]
            while sample_token:
                sample = nusc.get("sample", sample_token)
                camera_data = nusc.get("sample_data", sample["data"][camera_channel])
                img_path = os.path.join(root, camera_data["filename"])
                if sample_token in label_map:
                    samples.append(
                        {
                            "image_path": img_path,
                            "sample_token": sample_token,
                            "label": int(label_map[sample_token]),
                        }
                    )
                sample_token = sample["next"]

        np.random.seed(seed)
        if n_samples <= 0 or n_samples >= len(samples):
            sampled_indices = list(range(len(samples)))
        else:
            sampled_indices = np.random.choice(
                len(samples),
                min(n_samples, len(samples)),
                replace=False,
            )
        samples = [samples[i] for i in sampled_indices]

        transform = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
            ]
        )

        class NuScenesImageDataset(Dataset):
            def __init__(self, items):
                self.items = items

            def __len__(self):
                return len(self.items)

            def __getitem__(self, idx):
                item = self.items[idx]
                img = Image.open(item["image_path"]).convert("RGB")
                x = transform(img)
                y = int(item["label"])
                return x, y

        dataset = NuScenesImageDataset(samples)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=bool(pin_memory),
        )

        x_list = []
        y_list = []
        for x_batch, y_batch in loader:
            x_list.append(x_batch)
            y_list.append(y_batch)
        x_tensor = torch.cat(x_list, dim=0)
        y_tensor = torch.cat(y_list, dim=0)

        return loader, x_tensor, y_tensor


class DatasetFactory:
    """Factory for loading datasets based on defense domain."""

    DEFENSE_TO_DATASET = {
        "jpeg_compression": "cifar10",
        "bit_depth_reduction": "cifar10",
        "randomized_smoothing": "cifar10",
        "ensemble_diversity": "cifar10",
        "feature_squeezing": "ember",
        "gradient_regularization": "ember",
        "defensive_distillation": "ember",
        "adversarial_training_transform": "ember",
        "spatial_smoothing": "nuscenes",
        "random_pad_crop": "nuscenes",
        "thermometer_encoding": "nuscenes",
        "certified_defense": "nuscenes",
    }

    @staticmethod
    def get_dataset(dataset_name: str, **kwargs):
        if dataset_name == "cifar10":
            return CIFAR10Dataset.load(**kwargs)
        if dataset_name == "imagenet100":
            return ImageNet100Dataset.load(**kwargs)
        if dataset_name == "ember":
            return EMBERDataset.load(**kwargs)
        if dataset_name == "nuscenes":
            return nuScenesDataset.load(**kwargs)
        raise ValueError(f"Unknown dataset: {dataset_name}")
