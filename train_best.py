"""
High-score training script for the NPM3D mini benchmark.

Strategy:
- Use all training .ply files under training/*.ply
- Build stronger geometric features from XYZ (multi-scale grid stats)
- Train an ensemble of tuned XGBoost classifiers
- Predict test labels with ensemble probabilities and save submission

Usage:
    .venv/bin/python train_best.py
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass

import numpy as np
from plyfile import PlyData
import xgboost as xgb


TRAIN_GLOB = "training/*.ply"
TEST_PLY = "test/MiniDijon9.ply"
LABEL_FIELD = "class"
SUBMISSION_PATH = "submission_best.txt"

GRID_SCALES = (0.5, 1.0, 2.0, 4.0)
CHUNK_SIZE = 500_000
EPS = 1e-6


@dataclass(frozen=True)
class ModelSpec:
    name: str
    params: dict
    weight: float = 1.0


MODEL_SPECS = [
    ModelSpec(
        name="xgb_deep",
        params=dict(
            n_estimators=900,
            max_depth=12,
            learning_rate=0.045,
            subsample=0.9,
            colsample_bytree=0.9,
            min_child_weight=3,
            reg_lambda=2.0,
            reg_alpha=0.05,
            gamma=0.05,
            max_bin=256,
            random_state=42,
        ),
        weight=1.0,
    ),
    ModelSpec(
        name="xgb_mid",
        params=dict(
            n_estimators=1100,
            max_depth=10,
            learning_rate=0.038,
            subsample=0.85,
            colsample_bytree=0.85,
            min_child_weight=4,
            reg_lambda=2.5,
            reg_alpha=0.1,
            gamma=0.1,
            max_bin=256,
            random_state=123,
        ),
        weight=1.0,
    ),
    ModelSpec(
        name="xgb_wide",
        params=dict(
            n_estimators=750,
            max_depth=8,
            learning_rate=0.06,
            subsample=0.9,
            colsample_bytree=0.8,
            min_child_weight=2,
            reg_lambda=1.5,
            reg_alpha=0.02,
            gamma=0.02,
            max_bin=384,
            random_state=777,
        ),
        weight=1.0,
    ),
]


def load_xyz_labels(path: str, label_field: str | None):
    ply = PlyData.read(path)
    vertices = ply["vertex"].data
    xyz = np.vstack([vertices["x"], vertices["y"], vertices["z"]]).T.astype(np.float32)
    labels = None
    if label_field is not None:
        if label_field not in vertices.dtype.names:
            raise ValueError(
                f"Label field '{label_field}' not found in {path}. Found: {vertices.dtype.names}"
            )
        labels = np.asarray(vertices[label_field], dtype=np.int32)
    return xyz, labels, vertices.dtype.names


def _scene_normalized_base_features(xyz: np.ndarray) -> np.ndarray:
    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]

    x_shift = x - x.min()
    y_shift = y - y.min()
    z_shift = z - z.min()

    x_norm = x_shift / (x_shift.max() + EPS)
    y_norm = y_shift / (y_shift.max() + EPS)
    z_norm = z_shift / (z_shift.max() + EPS)

    x_centered = x_norm - 0.5
    y_centered = y_norm - 0.5
    r_norm = np.sqrt(x_centered * x_centered + y_centered * y_centered).astype(np.float32)

    return np.column_stack([x_shift, y_shift, z_shift, x_norm, y_norm, z_norm, r_norm]).astype(
        np.float32
    )


def _grid_stats_features(xyz: np.ndarray, grid: float) -> np.ndarray:
    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]

    x0 = float(x.min())
    y0 = float(y.min())
    ix = np.floor((x - x0) / grid).astype(np.int64)
    iy = np.floor((y - y0) / grid).astype(np.int64)

    # Unique cell key on (ix, iy)
    key = (ix << 32) | (iy & 0xFFFFFFFF)
    _, inv = np.unique(key, return_inverse=True)
    n_cells = int(inv.max()) + 1

    counts = np.bincount(inv).astype(np.float32)
    zsum = np.bincount(inv, weights=z).astype(np.float32)
    zsum2 = np.bincount(inv, weights=z * z).astype(np.float32)

    zmin = np.full(n_cells, np.inf, dtype=np.float32)
    zmax = np.full(n_cells, -np.inf, dtype=np.float32)
    np.minimum.at(zmin, inv, z)
    np.maximum.at(zmax, inv, z)

    zmean = zsum / np.maximum(counts, EPS)
    zvar = zsum2 / np.maximum(counts, EPS) - zmean * zmean
    zstd = np.sqrt(np.maximum(zvar, 0.0)).astype(np.float32)

    z_rel_min = (z - zmin[inv]).astype(np.float32)
    z_rel_mean = (z - zmean[inv]).astype(np.float32)
    z_range = (zmax[inv] - zmin[inv]).astype(np.float32)
    z_std = zstd[inv].astype(np.float32)
    log_count = np.log1p(counts[inv]).astype(np.float32)

    return np.column_stack([z_rel_min, z_rel_mean, z_range, z_std, log_count]).astype(np.float32)


def build_features(xyz: np.ndarray, grid_scales: tuple[float, ...]) -> np.ndarray:
    parts = [_scene_normalized_base_features(xyz)]
    for grid in grid_scales:
        parts.append(_grid_stats_features(xyz, grid))
    return np.hstack(parts).astype(np.float32, copy=False)


def class_balanced_weights(y: np.ndarray, num_classes: int = 6) -> np.ndarray:
    counts = np.bincount(y, minlength=num_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    class_w = np.sqrt(counts.sum() / (num_classes * counts))
    w = class_w[y].astype(np.float32)
    w *= len(w) / np.sum(w)
    return w


def fit_xgb_with_fallback(
    X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray, model_params: dict
) -> tuple[xgb.XGBClassifier, str]:
    common = dict(
        objective="multi:softprob",
        num_class=6,
        eval_metric="mlogloss",
        tree_method="hist",
        n_jobs=max(1, (os.cpu_count() or 1) - 1),
    )

    # Try GPU first.
    try:
        gpu_params = dict(common)
        gpu_params.update(model_params)
        gpu_params["device"] = "cuda"
        clf = xgb.XGBClassifier(**gpu_params)
        clf.fit(X, y, sample_weight=sample_weight, verbose=False)
        return clf, "GPU"
    except Exception as gpu_err:
        print(f"GPU training failed ({repr(gpu_err)}). Falling back to CPU.")

    cpu_params = dict(common)
    cpu_params.update(model_params)
    cpu_params["device"] = "cpu"
    clf = xgb.XGBClassifier(**cpu_params)
    clf.fit(X, y, sample_weight=sample_weight, verbose=False)
    return clf, "CPU"


def load_all_train_data(train_paths: list[str], label_field: str):
    X_parts = []
    y_parts = []
    fields_ref = None

    for path in train_paths:
        xyz, y_raw, fields = load_xyz_labels(path, label_field=label_field)
        if fields_ref is None:
            fields_ref = fields
        print(f"[train] {path}: {len(xyz)} points")

        X_scene = build_features(xyz, GRID_SCALES)
        mask = y_raw != 0
        X_parts.append(X_scene[mask])
        y_parts.append((y_raw[mask] - 1).astype(np.int32))

    X = np.vstack(X_parts).astype(np.float32, copy=False)
    y = np.concatenate(y_parts).astype(np.int32, copy=False)
    return X, y, fields_ref


def predict_ensemble_proba(models, model_weights, X: np.ndarray, chunk_size: int) -> np.ndarray:
    n = len(X)
    proba_sum = np.zeros((n, 6), dtype=np.float32)
    wsum = float(np.sum(model_weights))

    for m_idx, (clf, mw) in enumerate(zip(models, model_weights), start=1):
        print(f"Predicting with model {m_idx}/{len(models)}...")
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            proba_sum[start:end] += mw * clf.predict_proba(X[start:end]).astype(np.float32)

    proba_sum /= max(wsum, EPS)
    return proba_sum


def main():
    train_paths = sorted(glob.glob(TRAIN_GLOB))
    if not train_paths:
        raise FileNotFoundError(f"No train files matched: {TRAIN_GLOB}")
    if not os.path.exists(TEST_PLY):
        raise FileNotFoundError(f"Test file not found: {TEST_PLY}")

    print("Training files:")
    for p in train_paths:
        print(" -", p)

    print("\nBuilding training features (all points)...")
    X_train, y_train, fields = load_all_train_data(train_paths, LABEL_FIELD)
    print("Train fields:", fields)
    print("X_train shape:", X_train.shape, "dtype:", X_train.dtype)
    print("y_train shape:", y_train.shape, "classes:", np.bincount(y_train, minlength=6).tolist())

    w_train = class_balanced_weights(y_train, num_classes=6)
    print("Sample weights ready.")

    models = []
    model_weights = []
    for spec in MODEL_SPECS:
        print(f"\nTraining {spec.name}...")
        clf, device_used = fit_xgb_with_fallback(X_train, y_train, w_train, spec.params)
        print(f"{spec.name} trained on {device_used}.")
        models.append(clf)
        model_weights.append(spec.weight)

    print("\nBuilding test features...")
    xyz_test, _, fields_test = load_xyz_labels(TEST_PLY, label_field=None)
    X_test = build_features(xyz_test, GRID_SCALES)
    print("Test fields:", fields_test)
    print("X_test shape:", X_test.shape, "dtype:", X_test.dtype)

    print("\nEnsemble prediction...")
    proba = predict_ensemble_proba(models, model_weights, X_test, CHUNK_SIZE)
    pred_1_6 = (np.argmax(proba, axis=1).astype(np.int32) + 1).astype(np.int32)

    np.savetxt(SUBMISSION_PATH, pred_1_6, fmt="%d")
    print(f"\nSaved {SUBMISSION_PATH} with {len(pred_1_6)} labels in [1..6].")


if __name__ == "__main__":
    main()
