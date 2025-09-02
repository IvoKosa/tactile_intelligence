# svm_holdout_benchmark.py
import numpy as np
import signal_dataset
import torch

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

# # ---------------------- helpers ----------------------
# def _to_numpy(x):
#     if torch is not None and isinstance(x, torch.Tensor):
#         return x.detach().cpu().numpy()
#     return np.asarray(x)

# def _stats_features(signal_np: np.ndarray) -> np.ndarray:
#     """
#     Per-channel stats: mean, std, min, max, median, p25, p75  -> 7 features/channel.
#     Works for (C, L) or (L, C) or (L,) signals.
#     """
#     if signal_np.ndim == 1:
#         x = signal_np[None, :]                   # (1, L)
#     else:
#         # Heuristic: if first dim <= 64, treat as channels-first
#         x = signal_np if signal_np.shape[0] <= 64 else signal_np.T  # (C, L)

#     feats = []
#     for ch in range(x.shape[0]):
#         v = x[ch]
#         feats.extend([
#             float(np.mean(v)), float(np.std(v)), float(np.min(v)), float(np.max(v)),
#             float(np.median(v)), float(np.percentile(v, 25)), float(np.percentile(v, 75)),
#         ])
#     return np.array(feats, dtype=np.float64)

# def _coerce_label(y):
#     """
#     Turn target into a single class id when given one-hot/probabilities;
#     otherwise return scalar/int if possible.
#     """
#     y_np = _to_numpy(y)
#     y_np = np.squeeze(y_np)
#     if y_np.ndim == 1 and y_np.size > 1:
#         if np.isclose(y_np.sum(), 1.0, atol=1e-3) or np.all((y_np == 0) | (y_np == 1)):
#             return int(np.argmax(y_np))
#     if y_np.shape == ():
#         return y_np.item()
#     return y_np

# def dataset_to_xy(dataset, target="mat"):
#     """
#     Converts your PyTorch dataset (items like (signal, mat_target, tex_target))
#     into (X, y) for sklearn using compact per-channel stats.
#     """
#     X, y = [], []
#     for i in range(len(dataset)):
#         signal, mat_target, tex_target = dataset[i]
#         feats = _stats_features(_to_numpy(signal))
#         X.append(feats)
#         y_raw = mat_target if target == "mat" else tex_target
#         y.append(_coerce_label(y_raw))
#     return np.vstack(X), np.array(y)

# # ---------------------- main ----------------------
# def run_svm_holdout(train_set, test_set, target="mat", C=1.0, gamma="scale"):
#     # Build matrices
#     X_tr, y_tr = dataset_to_xy(train_set, target=target)
#     X_te, y_te = dataset_to_xy(test_set,  target=target)

#     # SVM (RBF) with standard scaling
#     clf = Pipeline([
#         ("scaler", StandardScaler()),
#         ("svm", SVC(kernel="rbf", C=C, gamma=gamma))
#     ])

#     clf.fit(X_tr, y_tr)
#     y_pred = clf.predict(X_te)

#     # Metrics
#     acc  = accuracy_score(y_te, y_pred)
#     prec = precision_score(y_te, y_pred, average="macro", zero_division=0)
#     rec  = recall_score(y_te, y_pred, average="macro", zero_division=0)
#     f1m  = f1_score(y_te, y_pred, average="macro", zero_division=0)
#     f1u  = f1_score(y_te, y_pred, average="micro", zero_division=0)
#     cm   = confusion_matrix(y_te, y_pred)

#     print(f"X_train: {X_tr.shape} | X_test: {X_te.shape}")
#     print("\n=== SVM (RBF) — Test Metrics ===")
#     print(f"Accuracy      : {acc:.4f}")
#     print(f"Precision(macro): {prec:.4f}")
#     print(f"Recall(macro) : {rec:.4f}")
#     print(f"F1 (macro)    : {f1m:.4f}")
#     print(f"F1 (micro)    : {f1u:.4f}")
#     print("\nConfusion Matrix (rows=true, cols=pred):")
#     print(cm)
#     print("\nClassification Report:")
#     print(classification_report(y_te, y_pred, zero_division=0))

#     return clf  # return fitted pipeline in case you want to reuse it

# svm_multiclass_combined.py
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

# ---------- helpers ----------
def _to_numpy(x):
    if torch is not None and isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def _stats_features(signal_np: np.ndarray) -> np.ndarray:
    """
    Per-channel stats: mean, std, min, max, median, p25, p75  -> 7 features/channel.
    Handles (C, L), (L, C), or (L,) signals.
    """
    if signal_np.ndim == 1:
        x = signal_np[None, :]  # (1, L)
    else:
        x = signal_np if signal_np.shape[0] <= 64 else signal_np.T  # (C, L) heuristic
    feats = []
    for ch in range(x.shape[0]):
        v = x[ch]
        feats.extend([
            float(np.mean(v)), float(np.std(v)), float(np.min(v)), float(np.max(v)),
            float(np.median(v)), float(np.percentile(v, 25)), float(np.percentile(v, 75)),
        ])
    return np.array(feats, dtype=np.float64)

def _coerce_single_label(y):
    """
    Convert a single target (mat OR tex) into a scalar class id if possible.
    - One-hot / probs -> argmax
    - Scalar tensors -> scalar
    - Otherwise return as-is
    """
    y_np = _to_numpy(y)
    y_np = np.squeeze(y_np)
    if y_np.ndim == 1 and y_np.size > 1:
        if np.isclose(y_np.sum(), 1.0, atol=1e-3) or np.all((y_np == 0) | (y_np == 1)):
            return int(np.argmax(y_np))
    if y_np.shape == ():
        return y_np.item()
    return y_np

def dataset_to_xy_combined(dataset) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Build X and a *combined* multi-class label from (mat, tex).
    Each dataset item must be: (signal, mat_target, tex_target).
    Combined label is a string like "mat=<m>|tex=<t>" to keep class names readable.
    Returns: X, y_encoded, class_names (for reports).
    """
    X, labels = [], []
    for i in range(len(dataset)):
        signal, mat_target, tex_target = dataset[i]
        feats = _stats_features(_to_numpy(signal))
        X.append(feats)

        m = _coerce_single_label(mat_target)
        t = _coerce_single_label(tex_target)
        labels.append(f"mat={m}|tex={t}")

    X = np.vstack(X)

    # Encode combined strings to integers for SVC, but keep names for reports
    le = LabelEncoder()
    y_encoded = le.fit_transform(labels)
    class_names = list(le.classes_)  # index aligned with encoded integers
    return X, y_encoded, class_names        # type: ignore

# ---------- main routine ----------
def run_svm_combined(train_set, test_set, C=1.0, gamma="scale"):
    # Build matrices
    X_tr, y_tr, class_names = dataset_to_xy_combined(train_set)
    X_te, y_te, _ = dataset_to_xy_combined(test_set)  # same classes assumed

    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", C=C, gamma=gamma))    # type: ignore
    ])
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)

    # Metrics (multi-class)
    acc  = accuracy_score(y_te, y_pred)
    prec = precision_score(y_te, y_pred, average="macro", zero_division=0)
    rec  = recall_score(y_te, y_pred, average="macro", zero_division=0)
    f1m  = f1_score(y_te, y_pred, average="macro", zero_division=0)
    f1u  = f1_score(y_te, y_pred, average="micro", zero_division=0)
    cm   = confusion_matrix(y_te, y_pred)

    print(f"X_train: {X_tr.shape} | X_test: {X_te.shape} | n_classes: {len(class_names)}")
    print("\n=== Combined (mat,tex) SVM — Test Metrics ===")
    print(f"Accuracy        : {acc:.4f}")
    print(f"Precision (macro): {prec:.4f}")
    print(f"Recall (macro)   : {rec:.4f}")
    print(f"F1 (macro)       : {f1m:.4f}")
    print(f"F1 (micro)       : {f1u:.4f}")

    print("\nConfusion Matrix (rows=true, cols=pred):")
    print(cm)

    print("\nClassification Report (per combined class):")
    print(classification_report(y_te, y_pred, target_names=class_names, zero_division=0))

    return clf, class_names

# ---------------------- usage example ----------------------
if __name__ == "__main__":
    # You define/import these in your environment:
    # from your_code import signal_dataset
    # train_set = signal_dataset.SignalDataset(...)
    # test_set  = signal_dataset.SignalDataset(...)
    # Then just run this file.

    tex_classes = ['ds20', 'ds30', 'ef10', 'ef30', 'ef50']
    mat_classes = ['bigberry', 'citrus', 'rough', 'smallberry', 'smooth', 'strawberry']
    train_set = signal_dataset.SignalDataset('data_final/multigrasp_train', multigrasp=None, filtering=True, cropping=True, normalise=False, augment=True, tex_classes=tex_classes, mat_classes=mat_classes)
    test_set = signal_dataset.SignalDataset('data_final/multigrasp_test', multigrasp=None, filtering=True, cropping=True, normalise=False, augment=True, tex_classes=tex_classes, mat_classes=mat_classes)

    _ = run_svm_combined(train_set, test_set, C=1.0, gamma="scale")
