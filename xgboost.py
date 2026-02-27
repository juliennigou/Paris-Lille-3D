"""
NPM3D mini benchmark — baseline complet :
- Lit un .ply train (avec champ de classe) + un .ply test (sans classe)
- Construit des features légères (x,y,z + z_rel via grille XY)
- Split train/val et calcule mIoU (comme le serveur, sur la val)
- Entraîne XGBoost (GPU si dispo, sinon CPU)
- Prédit le test en chunks et écrit submission.txt (labels 1..6)

Prérequis:
pip install numpy plyfile xgboost scikit-learn
"""

import os
import glob
import numpy as np
from plyfile import PlyData
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


# ---------------------------
# CONFIG
# ---------------------------
TRAIN_GLOB = "training/*.ply"
TEST_PLY  = "test/MiniDijon9.ply"

LABEL_FIELD = "class"     # chez toi: "class" (vérifie avec inspect si besoin)

GRID = 1.0                # taille cellule XY (m) pour approx "hauteur au-dessus du sol"
TRAIN_SAMPLE = 400_000    # échantillon max utilisé pour entraîner (augmente si tu veux)
VAL_SIZE = 0.2            # taille validation pour score local

CHUNK = 500_000           # prédiction test par blocs (RAM-friendly)
RANDOM_STATE = 42

# XGBoost params (GPU si possible)
XGB_PARAMS = dict(
    n_estimators=800,
    max_depth=10,
    learning_rate=0.06,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    objective="multi:softmax",
    num_class=6,
    eval_metric="mlogloss",
    # GPU attempt:
    tree_method="auto",
    predictor="gpu_predictor",
)

CLASS_NAMES = ["ground", "building", "poles", "pedestrians", "cars", "vegetation"]
CLASSES_0_5 = [0, 1, 2, 3, 4, 5]


# ---------------------------
# IO + FEATURES
# ---------------------------
def load_ply_xyz_labels(path, label_field=None):
    ply = PlyData.read(path)
    v = ply["vertex"].data
    xyz = np.vstack([v["x"], v["y"], v["z"]]).T.astype(np.float32)

    labels = None
    if label_field is not None:
        if label_field not in v.dtype.names:
            raise ValueError(f"Label field '{label_field}' not found in {path}. Found: {v.dtype.names}")
        labels = np.asarray(v[label_field], dtype=np.int32)

    return xyz, labels, v.dtype.names


def add_grid_height_feature(xyz, grid=1.0):
    """
    Ajoute une feature z_rel = z - min(z) dans une cellule XY de taille grid.
    Ne nécessite pas de KNN, donc scalable.
    """
    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]

    x0, y0 = float(x.min()), float(y.min())
    ix = np.floor((x - x0) / grid).astype(np.int32)
    iy = np.floor((y - y0) / grid).astype(np.int32)

    key = (ix.astype(np.int64) << 32) | (iy.astype(np.int64) & 0xFFFFFFFF)

    uniq, inv = np.unique(key, return_inverse=True)
    zmin = np.full(uniq.shape[0], np.inf, dtype=np.float32)
    np.minimum.at(zmin, inv, z)

    z_rel = (z - zmin[inv]).reshape(-1, 1).astype(np.float32)
    return np.hstack([xyz, z_rel])


# ---------------------------
# METRICS (mIoU)
# ---------------------------
def miou_from_cm(cm):
    ious = []
    for i in range(cm.shape[0]):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        denom = tp + fp + fn
        iou = 0.0 if denom == 0 else tp / denom
        ious.append(iou)
    return np.array(ious, dtype=float), float(np.mean(ious))


def print_miou(y_true_0_5, y_pred_0_5):
    cm = confusion_matrix(y_true_0_5, y_pred_0_5, labels=CLASSES_0_5)
    ious, miou = miou_from_cm(cm)
    print("\nConfusion matrix (val):\n", cm)
    print("\nIoU par classe (val):")
    for name, val in zip(CLASS_NAMES, ious):
        print(f"  {name:12s}: {val:.3f}")
    print(f"\n🔥 mIoU GLOBAL (val): {miou:.4f}\n")
    return miou


# ---------------------------
# TRAINING
# ---------------------------
def make_balanced_weights(y_0_5):
    """
    Pondération simple inverse fréquence pour aider les classes rares (poles/pedestrians).
    """
    classes, counts = np.unique(y_0_5, return_counts=True)
    freq = dict(zip(classes.tolist(), counts.tolist()))
    w = np.array([1.0 / freq[c] for c in y_0_5], dtype=np.float32)
    w *= (len(w) / w.sum())
    return w


def train_xgb(X, y, w):
    """
    Entraîne XGBoost en essayant GPU puis fallback CPU si besoin.
    """
    try:
        clf = xgb.XGBClassifier(**XGB_PARAMS)
        clf.fit(X, y, sample_weight=w)
        used = "GPU (gpu_hist)"
        return clf, used
    except Exception as e:
        print("⚠️ GPU XGBoost failed, fallback CPU. Reason:", repr(e))
        cpu_params = dict(XGB_PARAMS)
        cpu_params["tree_method"] = "hist"
        cpu_params["predictor"] = "auto"
        clf = xgb.XGBClassifier(**cpu_params)
        clf.fit(X, y, sample_weight=w)
        used = "CPU (hist)"
        return clf, used


# ---------------------------
# MAIN
# ---------------------------
def main():
    train_files = sorted(glob.glob(TRAIN_GLOB))
    if not train_files:
        raise FileNotFoundError(f"No train files matched: {TRAIN_GLOB}")
    if not os.path.exists(TEST_PLY):
        raise FileNotFoundError(f"Test file not found: {TEST_PLY}")

    # Load all training .ply files
    xyz_parts = []
    y_parts = []
    fields = None
    for train_file in train_files:
        xyz_part, y_part, fields_part = load_ply_xyz_labels(train_file, LABEL_FIELD)
        if fields is None:
            fields = fields_part
        xyz_parts.append(xyz_part)
        y_parts.append(y_part)
        print(f"Loaded {train_file}: N={len(xyz_part)}")

    xyz_tr = np.vstack(xyz_parts)
    y_tr_raw = np.concatenate(y_parts)
    print("Train files:", train_files)
    print("Train fields:", fields)
    print("Train N total:", len(xyz_tr))

    # Keep labels 1..6, drop 0
    mask = (y_tr_raw != 0)
    xyz_tr = xyz_tr[mask]
    y_tr = y_tr_raw[mask]

    # Remap 1..6 -> 0..5 for training
    y_tr = (y_tr - 1).astype(np.int32)
    assert y_tr.min() >= 0 and y_tr.max() <= 5

    # Features
    X_tr = add_grid_height_feature(xyz_tr, GRID)

    # Subsample for faster training
    rng = np.random.default_rng(RANDOM_STATE)
    if len(X_tr) > TRAIN_SAMPLE:
        idx = rng.choice(len(X_tr), size=TRAIN_SAMPLE, replace=False)
        X_fit = X_tr[idx]
        y_fit = y_tr[idx]
    else:
        X_fit, y_fit = X_tr, y_tr

    w_fit = make_balanced_weights(y_fit)

    # Split train/val (local scoring)
    Xtr, Xva, ytr, yva, wtr, wva = train_test_split(
        X_fit, y_fit, w_fit,
        test_size=VAL_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_fit
    )

    # Train
    clf, used = train_xgb(Xtr, ytr, wtr)
    print("Model trained with:", used)

    # Evaluate (local score)
    pred_va = clf.predict(Xva).astype(np.int32)
    print_miou(yva, pred_va)

    # Train final model on all sampled data (optional but better before predicting test)
    clf_final, used2 = train_xgb(X_fit, y_fit, w_fit)
    print("Final model trained with:", used2)

    # Predict test (KEEP ORDER) + write submission
    xyz_te, _, fields_te = load_ply_xyz_labels(TEST_PLY, None)
    print("Test fields:", fields_te)
    print("Test N:", len(xyz_te))

    pred_all_0_5 = np.empty(len(xyz_te), dtype=np.int32)
    for start in range(0, len(xyz_te), CHUNK):
        end = min(start + CHUNK, len(xyz_te))
        X_te = add_grid_height_feature(xyz_te[start:end], GRID)
        pred_all_0_5[start:end] = clf_final.predict(X_te).astype(np.int32)

    # Remap back 0..5 -> 1..6 for submission
    pred_all_1_6 = (pred_all_0_5 + 1).astype(np.int32)
    assert pred_all_1_6.min() >= 1 and pred_all_1_6.max() <= 6

    np.savetxt("submission.txt", pred_all_1_6, fmt="%d")
    print("✅ Wrote submission.txt with", len(pred_all_1_6), "labels (1..6).")


if __name__ == "__main__":
    main()
