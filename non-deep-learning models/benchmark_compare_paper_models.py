import os
import glob
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, roc_auc_score, average_precision_score, confusion_matrix, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.svm import SVR, SVC
from scipy.stats import pearsonr

from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor, LGBMClassifier
from catboost import CatBoostRegressor, CatBoostClassifier

import torch

SEED = 42
N_FOLDS = 10
THRESHOLD = -6.0
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
OUT_DIR = os.path.join(BASE_DIR, "benchmark_results")
MODEL_FILTER = {m.strip() for m in os.getenv("MODEL_FILTER", "").split(",") if m.strip()}
GPU_ID = os.getenv("GPU_ID", "0")
OUTPUT_TAG = os.getenv("OUTPUT_TAG", "").strip()


def compute_metrics(y_true_reg, y_pred_reg, y_true_cls, y_pred_cls):
    y_true_reg = np.asarray(y_true_reg, dtype=float)
    y_pred_reg = np.asarray(y_pred_reg, dtype=float)
    y_true_cls = np.asarray(y_true_cls, dtype=int)
    y_pred_cls = np.asarray(y_pred_cls, dtype=int)

    # Remove NaN/Inf rows safely
    mask = np.isfinite(y_true_reg) & np.isfinite(y_pred_reg)
    if mask.sum() == 0:
        return {
            "ACC": 0.0,
            "Precision": 0.0,
            "Recall": 0.0,
            "MSE": np.inf,
            "MAE": np.inf,
            "r": 0.0,
        }

    y_true_reg = y_true_reg[mask]
    y_pred_reg = y_pred_reg[mask]
    y_true_cls = y_true_cls[mask]
    y_pred_cls = y_pred_cls[mask]

    acc = accuracy_score(y_true_cls, y_pred_cls)
    precision = precision_score(y_true_cls, y_pred_cls, zero_division=0)
    recall = recall_score(y_true_cls, y_pred_cls, zero_division=0)
    f1 = f1_score(y_true_cls, y_pred_cls, zero_division=0)
    mcc = matthews_corrcoef(y_true_cls, y_pred_cls)
    tn, fp, fn, tp = confusion_matrix(y_true_cls, y_pred_cls, labels=[0, 1]).ravel()
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    mse = mean_squared_error(y_true_reg, y_pred_reg)
    mae = mean_absolute_error(y_true_reg, y_pred_reg)
    r = pearsonr(y_true_reg, y_pred_reg)[0] if len(np.unique(y_pred_reg)) > 1 else 0.0
    return {
        "ACC": acc,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "MCC": mcc,
        "TNR": tnr,
        "MSE": mse,
        "MAE": mae,
        "r": r,
    }


def add_prob_metrics(metrics, y_true_cls, y_score):
    y_true_cls = np.asarray(y_true_cls, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    mask = np.isfinite(y_score)
    if mask.sum() == 0 or len(np.unique(y_true_cls[mask])) < 2:
        metrics["AUROC"] = 0.0
        metrics["AUPRC"] = 0.0
        return metrics
    metrics["AUROC"] = roc_auc_score(y_true_cls[mask], y_score[mask])
    metrics["AUPRC"] = average_precision_score(y_true_cls[mask], y_score[mask])
    return metrics


def summarize(metric_list):
    keys = metric_list[0].keys()
    out = {}
    for k in keys:
        vals = np.array([m[k] for m in metric_list], dtype=float)
        out[k] = (vals.mean(), vals.std())
    return out


def format_summary(name, summary):
    row = {"Model": name}
    for k, (m, s) in summary.items():
        row[k] = f"{m:.4f} ± {s:.4f}"
    return row


def load_features_and_labels():
    csv_path = os.path.join(PROJECT_ROOT, "CycPeptMPDB_Peptide_PAMPA.csv")
    df = pd.read_csv(csv_path, low_memory=False)
    df["Permeability"] = pd.to_numeric(df["Permeability"], errors="coerce")
    df = df.dropna(subset=["Permeability", "SMILES"]).reset_index(drop=True)

    cache_path = os.path.join(os.path.dirname(PROJECT_ROOT), "cached_data_tv.pt")
    data = torch.load(cache_path, weights_only=False)
    X = np.array(data["descriptors"], dtype=np.float32)

    if len(X) != len(df):
        n = min(len(X), len(df))
        X = X[:n]
        df = df.iloc[:n].copy().reset_index(drop=True)

    y_reg = df["Permeability"].values.astype(np.float32)
    y_cls = (y_reg >= THRESHOLD).astype(int)
    return X, y_reg, y_cls


def load_unified_splits(n_samples):
    split_path = os.path.join(BASE_DIR, "benchmark_results", "unified_kfold_indices_seed42.npz")
    if not os.path.exists(split_path):
        return None
    data = np.load(split_path, allow_pickle=True)
    splits = []
    for fold in range(N_FOLDS):
        train_idx = data[f"fold{fold}_train"]
        test_idx = data[f"fold{fold}_test"]
        train_idx = np.asarray(train_idx, dtype=int)
        test_idx = np.asarray(test_idx, dtype=int)
        train_idx = train_idx[train_idx < n_samples]
        test_idx = test_idx[test_idx < n_samples]
        splits.append((train_idx, test_idx))
    return splits


def run_ml_models(X, y_reg, y_cls):
    unified_splits = load_unified_splits(len(X))
    if unified_splits is None:
        kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
        split_iter = list(kf.split(X))
    else:
        split_iter = unified_splits

    models = {
        "CatBoost": (
            CatBoostRegressor(iterations=300, learning_rate=0.05, depth=6, random_seed=SEED, verbose=False,
                              task_type="GPU", devices=GPU_ID),
            CatBoostClassifier(iterations=300, learning_rate=0.05, depth=6, random_seed=SEED, verbose=False,
                               task_type="GPU", devices=GPU_ID),
            False,
        ),
        "XGBoost": (
            XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6, subsample=0.9, colsample_bytree=0.9,
                         random_state=SEED, n_jobs=8, tree_method="hist", device=f"cuda:{GPU_ID}"),
            XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=6, subsample=0.9, colsample_bytree=0.9,
                          random_state=SEED, n_jobs=8, eval_metric="logloss", tree_method="hist", device=f"cuda:{GPU_ID}"),
            False,
        ),
        "KNN": (
            KNeighborsRegressor(n_neighbors=7, weights="distance"),
            KNeighborsClassifier(n_neighbors=7, weights="distance"),
            True,
        ),
        "LGBM": (
            LGBMRegressor(n_estimators=400, learning_rate=0.05, random_state=SEED, n_jobs=8, verbose=-1,
                          device_type="gpu", gpu_device_id=int(GPU_ID)),
            LGBMClassifier(n_estimators=400, learning_rate=0.05, random_state=SEED, n_jobs=8, verbose=-1,
                           device_type="gpu", gpu_device_id=int(GPU_ID)),
            False,
        ),
        "RF": (
            RandomForestRegressor(n_estimators=400, random_state=SEED, n_jobs=8),
            RandomForestClassifier(n_estimators=400, random_state=SEED, n_jobs=8),
            False,
        ),
        "SVM (poly)": (
            SVR(kernel="poly", C=3.0, epsilon=0.1, degree=3),
            SVC(kernel="poly", C=3.0, degree=3, probability=True, random_state=SEED),
            True,
        ),
        "SVM (rbf)": (
            SVR(kernel="rbf", C=3.0, epsilon=0.1, gamma="scale"),
            SVC(kernel="rbf", C=3.0, gamma="scale", probability=True, random_state=SEED),
            True,
        ),
        "DT": (
            DecisionTreeRegressor(random_state=SEED, max_depth=12),
            DecisionTreeClassifier(random_state=SEED, max_depth=12),
            False,
        ),
    }

    all_results = {}

    selected_models = models.items() if not MODEL_FILTER else [(k, v) for k, v in models.items() if k in MODEL_FILTER]

    for model_name, (reg_model, cls_model, need_scale) in selected_models:
        fold_metrics = []
        print(f"Running {model_name}...")
        for fold, (train_idx, test_idx) in enumerate(split_iter):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train_reg, y_test_reg = y_reg[train_idx], y_reg[test_idx]
            y_train_cls, y_test_cls = y_cls[train_idx], y_cls[test_idx]

            if need_scale:
                reg_pipe = Pipeline([("scaler", StandardScaler()), ("model", reg_model)])
                cls_pipe = Pipeline([("scaler", StandardScaler()), ("model", cls_model)])
            else:
                reg_pipe = reg_model
                cls_pipe = cls_model

            reg_pipe.fit(X_train, y_train_reg)
            cls_pipe.fit(X_train, y_train_cls)

            pred_reg = reg_pipe.predict(X_test)
            pred_cls = cls_pipe.predict(X_test)

            metrics = compute_metrics(y_test_reg, pred_reg, y_test_cls, pred_cls)
            if hasattr(cls_pipe, "predict_proba"):
                score_cls = cls_pipe.predict_proba(X_test)[:, 1]
            elif hasattr(cls_pipe, "decision_function"):
                raw = cls_pipe.decision_function(X_test)
                raw = np.asarray(raw, dtype=float)
                score_cls = (raw - raw.min()) / (raw.max() - raw.min() + 1e-12)
            else:
                score_cls = pred_cls.astype(float)
            metrics = add_prob_metrics(metrics, y_test_cls, score_cls)
            fold_metrics.append(metrics)
            print(f"  Fold {fold}: done")

        all_results[model_name] = summarize(fold_metrics)

    return all_results


def load_existing_deep_results():
    csv_files = sorted(glob.glob(os.path.join(OUT_DIR, "fold_*_predictions.csv")))
    deep_map = {
        "MSF-CPMP": ("Pred_LogP_MSF", "Pred_Score_MSF"),
        "CycPeptMP": ("Pred_LogP_Cyc", "Pred_Score_Cyc"),
        "TV-STFN": ("Pred_LogP_TVSTFN", "Pred_Score_TVSTFN"),
    }

    result = {}
    for name, (reg_col, score_col) in deep_map.items():
        fold_metrics = []
        for f in csv_files:
            df = pd.read_csv(f)
            if reg_col not in df.columns or score_col not in df.columns:
                continue
            y_true_reg = pd.to_numeric(df["True_LogP"], errors="coerce").values
            y_true_cls = pd.to_numeric(df["True_Label"], errors="coerce").fillna(0).astype(int).values
            y_pred_reg = pd.to_numeric(df[reg_col], errors="coerce").values
            y_pred_cls = (pd.to_numeric(df[score_col], errors="coerce").fillna(0).values > 0.5).astype(int)
            fold_metrics.append(compute_metrics(y_true_reg, y_pred_reg, y_true_cls, y_pred_cls))

        if len(fold_metrics) == N_FOLDS:
            result[name] = summarize(fold_metrics)

    return result


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    X, y_reg, y_cls = load_features_and_labels()

    ml_results = run_ml_models(X, y_reg, y_cls)
    deep_results = load_existing_deep_results()

    rows_ml = [format_summary(k, v) for k, v in ml_results.items()]
    rows_deep = [format_summary(k, v) for k, v in deep_results.items()]

    df_ml = pd.DataFrame(rows_ml)
    df_deep = pd.DataFrame(rows_deep)

    suffix = f"_{OUTPUT_TAG}" if OUTPUT_TAG else ""
    ml_path = os.path.join(OUT_DIR, f"paper_non_deep_10fold_results{suffix}.csv")
    deep_path = os.path.join(OUT_DIR, f"paper_deep_10fold_results_available{suffix}.csv")

    df_ml.to_csv(ml_path, index=False)
    df_deep.to_csv(deep_path, index=False)

    print("\n=== Non-deep models (10-fold) ===")
    print(df_ml.to_string(index=False))
    print("\n=== Deep models (available from existing fold predictions) ===")
    print(df_deep.to_string(index=False))
    print(f"\nSaved: {ml_path}")
    print(f"Saved: {deep_path}")


if __name__ == "__main__":
    main()
