import numpy as np
import torch
import signal_dataset
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

def _flatten_signal(signal_np: np.ndarray) -> np.ndarray:
    """
    Flatten a (C, L) or (L, C) signal into a vector:
    [t1c1, t1c2, ... t1cC, t2c1, t2c2, ... t2cC, ..., tLc1, ... tLcC].
    """
    # ensure shape (L, C)
    if signal_np.ndim == 1:
        return signal_np.ravel()
    if signal_np.shape[0] == 24:  # assume (C, L)
        signal_np = signal_np.T    # -> (L, C)
    elif signal_np.shape[1] == 24:
        pass  # already (L, C)
    else:
        raise ValueError(f"Unexpected signal shape {signal_np.shape}, expected 24 channels")

    return signal_np.reshape(-1)  # flatten row-wise

def _coerce_single_label(y):
    """Convert mat/tex targets to scalar IDs if one-hot or scalar."""
    y_np = _to_numpy(y)
    y_np = np.squeeze(y_np)
    if y_np.ndim == 1 and y_np.size > 1:
        if np.isclose(y_np.sum(), 1.0, atol=1e-3) or np.all((y_np == 0) | (y_np == 1)):
            return int(np.argmax(y_np))
    if y_np.shape == ():
        return y_np.item()
    return y_np

def dataset_to_xy_combined(dataset, use_features):
    """
    Build X and combined labels from (signal, mat_target, tex_target).
    Flatten each signal to (L*24,) vector.
    Combined label = "mat=<m>|tex=<t>"
    """
    X, labels = [], []
    for i in range(len(dataset)):
        signal, mat_target, tex_target = dataset[i]

        if use_features:
            feats = _stats_features(_to_numpy(signal))
        else:
            feats = _flatten_signal(_to_numpy(signal))
        X.append(feats)

        m = _coerce_single_label(mat_target)
        t = _coerce_single_label(tex_target)
        labels.append(f"mat={m}|tex={t}")

    X = np.vstack(X)
    le = LabelEncoder()
    y = le.fit_transform(labels)
    return X, y, list(le.classes_)

# ---------- main routine ----------
def run_svm_combined(train_set, test_set, use_features=False, C=1.0, gamma="scale"):
    X_tr, y_tr, class_names = dataset_to_xy_combined(train_set, use_features)
    X_te, y_te, _ = dataset_to_xy_combined(test_set, use_features)

    clf = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("svm", SVC(kernel="rbf", C=C, gamma=gamma))    # type: ignore
    ])
    clf.fit(X_tr, y_tr)                                 # type: ignore
    y_pred = clf.predict(X_te)

    acc  = accuracy_score(y_te, y_pred)
    prec = precision_score(y_te, y_pred, average="macro", zero_division=0)
    rec  = recall_score(y_te, y_pred, average="macro", zero_division=0)
    f1m  = f1_score(y_te, y_pred, average="macro", zero_division=0)
    f1u  = f1_score(y_te, y_pred, average="micro", zero_division=0)
    cm   = confusion_matrix(y_te, y_pred)

    print(f"X_train: {X_tr.shape} | X_test: {X_te.shape} | n_classes={len(class_names)}")
    print("\n=== Flattened SVM (Combined labels) — Test Metrics ===")
    print(f"Accuracy        : {acc:.4f}")
    print(f"Precision (macro): {prec:.4f}")
    print(f"Recall (macro)   : {rec:.4f}")
    print(f"F1 (macro)       : {f1m:.4f}")
    print(f"F1 (micro)       : {f1u:.4f}")

    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(y_te, y_pred, target_names=class_names, zero_division=0))

    return clf, class_names

# ******************************************* SVM Tester *******************************************
# 
# > Used for SVM results
# > Speficy dataset distribution and if using features flattening signal

if __name__ == "__main__":

    seed = 935248
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    g = torch.Generator().manual_seed(seed)

    use_features = False
    out_of_distribution = False

    mat_classes = ['ds20', 'ds30', 'ef10', 'ef30', 'ef50']
    tex_classes = ['bigberry', 'citrus', 'rough', 'smallberry', 'smooth', 'strawberry']

    if out_of_distribution:
        train_set = signal_dataset.SignalDataset('data_final/multigrasp_train', multigrasp=None, filtering=True, cropping=True, normalise=False, augment=True, tex_classes=tex_classes, mat_classes=mat_classes)
        test_set = signal_dataset.SignalDataset('data_final/multigrasp_test', multigrasp=None, filtering=True, cropping=True, normalise=False, augment=True, tex_classes=tex_classes, mat_classes=mat_classes)
        train_set, val_set = torch.utils.data.random_split(train_set, [0.7, 0.3], generator=g)    
    else:
        full_dataset = signal_dataset.SignalDataset('data_final', multigrasp=None, filtering=True, cropping=True, normalise=False, augment=False, tex_classes=tex_classes, mat_classes=mat_classes)
        train_set, test_set, val_set        = torch.utils.data.random_split(full_dataset, [2520, 1500, 1080], generator=g)

    _ = run_svm_combined(train_set, test_set, use_features, C=1.0, gamma="scale")

