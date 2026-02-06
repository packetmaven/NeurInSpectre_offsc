"""
Advanced caching system for NeurInSpectre evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from collections import OrderedDict
from typing import Dict, Tuple
import hashlib
import logging
import time

import torch
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


@dataclass
class CacheStats:
    """Statistics for cache monitoring."""

    hits: int = 0
    misses: int = 0
    evictions: int = 0
    total_size_bytes: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


class DatasetCache:
    """
    In-memory dataset cache with persistent disk backing.
    """

    def __init__(self, cache_dir: Path, max_memory_gb: float = 16.0, eviction_threshold: float = 0.9):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_memory_bytes = int(max_memory_gb * 1024**3)
        self.eviction_threshold = float(eviction_threshold)
        self._cache: OrderedDict[str, Tuple] = OrderedDict()
        self._cache_sizes: Dict[str, int] = {}
        self.stats = CacheStats()

    def has(self, key: str) -> bool:
        return key in self._cache or self._disk_exists(key)

    def get(self, key: str):
        if key in self._cache:
            self.stats.hits += 1
            self._cache.move_to_end(key)
            return self._cache[key]

        if self._disk_exists(key):
            self.stats.hits += 1
            dataset = self._load_from_disk(key)
            self.put(key, dataset)
            return dataset

        self.stats.misses += 1
        return None

    def put(self, key: str, dataset: Tuple):
        size_bytes = self._compute_size(dataset)
        while self._should_evict(size_bytes):
            self._evict_lru()

        self._cache[key] = dataset
        self._cache_sizes[key] = size_bytes
        self.stats.total_size_bytes += size_bytes
        self._save_to_disk(key, dataset)

    def _should_evict(self, incoming_size: int) -> bool:
        current_usage = self.stats.total_size_bytes
        threshold = self.max_memory_bytes * self.eviction_threshold
        return (current_usage + incoming_size) > threshold

    def _evict_lru(self) -> None:
        if not self._cache:
            return
        key, _dataset = self._cache.popitem(last=False)
        size = self._cache_sizes.pop(key)
        self.stats.total_size_bytes -= size
        self.stats.evictions += 1

    def _compute_size(self, dataset: Tuple) -> int:
        _loader, x_tensor, y_tensor = dataset
        size = x_tensor.element_size() * x_tensor.nelement()
        size += y_tensor.element_size() * y_tensor.nelement()
        return size

    def _disk_exists(self, key: str) -> bool:
        return self._get_disk_path(key).exists()

    def _get_disk_path(self, key: str) -> Path:
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{key_hash}.pt"

    def _save_to_disk(self, key: str, dataset: Tuple) -> None:
        path = self._get_disk_path(key)
        _loader, x_tensor, y_tensor = dataset
        temp_path = path.with_suffix(".pt.tmp")
        torch.save(
            {"x": x_tensor.cpu(), "y": y_tensor.cpu(), "timestamp": time.time(), "key": key},
            temp_path,
        )
        temp_path.rename(path)

    def _load_from_disk(self, key: str) -> Tuple:
        path = self._get_disk_path(key)
        data = torch.load(path, map_location="cpu")
        x_tensor = data["x"]
        y_tensor = data["y"]
        dataset = TensorDataset(x_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=100, shuffle=False)
        return (loader, x_tensor, y_tensor)

    def get_stats(self) -> CacheStats:
        return self.stats

    def clear(self) -> None:
        self._cache.clear()
        self._cache_sizes.clear()
        self.stats = CacheStats()


class ModelCache:
    """
    Model checkpoint cache (lazy loading).
    """

    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.registry_path = cache_dir / "model_registry.json"
        self.registry = self._load_registry()

    def get_checkpoint_path(self, model_key: str):
        path = self.registry.get(model_key)
        return Path(path) if path else None

    def register_model(self, model_key: str, checkpoint_path: Path) -> None:
        self.registry[model_key] = str(checkpoint_path)
        self._save_registry()

    def _load_registry(self) -> Dict:
        if self.registry_path.exists():
            return json_load(self.registry_path)
        return {}

    def _save_registry(self) -> None:
        json_dump(self.registry_path, self.registry)


class AttackCheckpoint:
    """
    Attack state checkpointing for resumption.
    """

    def __init__(self, checkpoint_dir: Path):
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def exists(self, key: str) -> bool:
        return self._get_path(key).exists()

    def save(self, key: str, result: Dict) -> None:
        path = self._get_path(key)
        temp_path = path.with_suffix(".ckpt.tmp")
        torch.save(result, temp_path)
        temp_path.rename(path)

    def load(self, key: str) -> Dict:
        path = self._get_path(key)
        return torch.load(path, map_location="cpu")

    def _get_path(self, key: str) -> Path:
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return self.checkpoint_dir / f"{key_hash}.ckpt"


def json_load(path: Path) -> Dict:
    import json

    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def json_dump(path: Path, payload: Dict) -> None:
    import json

    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
