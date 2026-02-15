#!/usr/bin/env python3
"""
Single-threaded EMBER vectorization for macOS.
Avoids multiprocessing.Pool + memmap corruption.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from sklearn.feature_extraction import FeatureHasher


class SafeFeatureHasher(FeatureHasher):
    def transform(self, raw_X):
        input_type = getattr(self, "input_type", "string")
        if input_type == "pair":
            safe = []
            for row in raw_X:
                if row is None:
                    safe.append([])
                    continue
                if isinstance(row, dict):
                    safe.append(list(row.items()))
                    continue
                pairs = []
                for item in row:
                    if item is None:
                        continue
                    if isinstance(item, (list, tuple)) and len(item) == 2:
                        pairs.append((item[0], item[1]))
                    else:
                        pairs.append((item, 1))
                safe.append(pairs)
            return super().transform(safe)

        if input_type == "string":
            safe = []
            for row in raw_X:
                if row is None:
                    safe.append([])
                    continue
                if isinstance(row, str):
                    safe.append([row])
                    continue
                if isinstance(row, (list, tuple)):
                    safe.append([str(x) for x in row if x is not None])
                    continue
                try:
                    safe.append([str(row)])
                except Exception:
                    safe.append([])
            return super().transform(safe)

        return super().transform(raw_X)


def count_lines(file_path: Path) -> int:
    count = 0
    with file_path.open("r") as handle:
        for _ in handle:
            count += 1
    return count


def vectorize_ember() -> None:
    import ember.features as ember_features
    from ember.features import PEFeatureExtractor

    ember_features.FeatureHasher = SafeFeatureHasher

    input_dir = Path("./data/ember/ember2018")
    output_dir = Path("./data/ember/ember_2018")
    output_dir.mkdir(parents=True, exist_ok=True)

    extractor = PEFeatureExtractor(2)
    ndim = extractor.dim

    print("[1/4] Counting samples...")
    train_shards = [input_dir / f"train_features_{i}.jsonl" for i in range(6)]
    test_file = input_dir / "test_features.jsonl"

    n_train = sum(count_lines(s) for s in train_shards)
    n_test = count_lines(test_file)
    print(f"  Train: {n_train:,}, Test: {n_test:,}")

    print(f"\n[2/4] Vectorizing training set ({n_train:,} samples)...")
    X_train = np.memmap(
        str(output_dir / "X_train.dat"),
        dtype=np.float32,
        mode="w+",
        shape=(n_train, ndim),
    )
    y_train = np.memmap(
        str(output_dir / "y_train.dat"),
        dtype=np.float32,
        mode="w+",
        shape=(n_train,),
    )

    idx = 0
    t0 = time.time()
    for shard in train_shards:
        print(f"  Processing {shard.name}...", flush=True)
        with shard.open("r") as handle:
            for line in handle:
                record = json.loads(line)
                y_train[idx] = record.get("label", -1)
                try:
                    X_train[idx] = extractor.process_raw_features(record)
                except Exception as exc:
                    if idx < 3:
                        print(f"    Warning: {exc}")
                idx += 1
                if idx % 50000 == 0:
                    elapsed = time.time() - t0
                    rate = idx / elapsed if elapsed > 0 else 0.0
                    eta = (n_train - idx) / rate / 60 if rate > 0 else 0.0
                    print(
                        f"    {idx:,}/{n_train:,} ({100*idx/n_train:.1f}%) "
                        f"Rate: {rate:.0f}/s  ETA: {eta:.1f}min",
                        flush=True,
                    )

    X_train.flush()
    y_train.flush()
    del X_train, y_train

    print(f"\n[3/4] Vectorizing test set ({n_test:,} samples)...")
    X_test = np.memmap(
        str(output_dir / "X_test.dat"),
        dtype=np.float32,
        mode="w+",
        shape=(n_test, ndim),
    )
    y_test = np.memmap(
        str(output_dir / "y_test.dat"),
        dtype=np.float32,
        mode="w+",
        shape=(n_test,),
    )

    idx = 0
    with test_file.open("r") as handle:
        for line in handle:
            record = json.loads(line)
            y_test[idx] = record.get("label", -1)
            try:
                X_test[idx] = extractor.process_raw_features(record)
            except Exception:
                pass
            idx += 1
            if idx % 50000 == 0:
                print(f"    {idx:,}/{n_test:,}", flush=True)

    X_test.flush()
    y_test.flush()
    del X_test, y_test

    print("\n[4/4] Verification...")
    for name, n in [("y_train", n_train), ("y_test", n_test)]:
        y = np.memmap(
            str(output_dir / f"{name}.dat"),
            dtype=np.float32,
            mode="r",
            shape=(n,),
        )
        unique, counts = np.unique(y.astype(int), return_counts=True)
        dist = ", ".join([f"{l}:{c:,}" for l, c in zip(unique, counts)])
        print(f"  {name}: {dist}")

    elapsed = time.time() - t0
    print(f"\n[DONE] Total time: {elapsed/60:.1f} minutes")


if __name__ == "__main__":
    vectorize_ember()
