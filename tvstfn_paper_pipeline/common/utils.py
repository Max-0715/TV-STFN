import glob
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


CLS_THRESHOLD = -6.0


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def discover_fold_prediction_files(pred_dir: str) -> List[str]:
    files = sorted(glob.glob(os.path.join(pred_dir, "fold_*_predictions.csv")))
    if not files:
        files = sorted(glob.glob(os.path.join(pred_dir, "fold_*_predictions_merged.csv")))
    return files


def load_fold_predictions(pred_dir: str) -> pd.DataFrame:
    files = discover_fold_prediction_files(pred_dir)
    if not files:
        raise FileNotFoundError(f"No fold prediction files found in: {pred_dir}")
    frames = []
    for fp in files:
        df = pd.read_csv(fp)
        df["__source_file"] = os.path.basename(fp)
        frames.append(df)
    all_df = pd.concat(frames, ignore_index=True)
    return all_df


def find_true_labels(df: pd.DataFrame, cls_threshold: float = CLS_THRESHOLD) -> np.ndarray:
    if "True_Label" in df.columns:
        return df["True_Label"].to_numpy().astype(int)
    if "True_LogP" in df.columns:
        return (df["True_LogP"].to_numpy() >= cls_threshold).astype(int)
    raise KeyError("Need True_Label or True_LogP in prediction files")


def find_model_score_columns(df: pd.DataFrame) -> Dict[str, str]:
    out = {}
    for c in df.columns:
        if c.startswith("Pred_Score_"):
            model_name = c.replace("Pred_Score_", "")
            out[model_name] = c
    return out


def find_model_reg_columns(df: pd.DataFrame) -> Dict[str, str]:
    out = {}
    for c in df.columns:
        if c.startswith("Pred_LogP_"):
            model_name = c.replace("Pred_LogP_", "")
            out[model_name] = c
    return out


def basic_cls_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, roc_auc_score

    y_pred = (y_score >= threshold).astype(int)
    result = {
        "ACC": float(accuracy_score(y_true, y_pred)),
        "F1": float(f1_score(y_true, y_pred, zero_division=0)),
        "MCC": float(matthews_corrcoef(y_true, y_pred)),
    }
    try:
        result["AUROC"] = float(roc_auc_score(y_true, y_score))
    except Exception:
        result["AUROC"] = float("nan")
    return result


def mean_std_ci95(values: np.ndarray) -> Tuple[float, float, float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
    se = std / np.sqrt(values.size) if values.size > 0 else 0.0
    ci = 1.96 * se
    return mean, std, mean - ci, mean + ci


def bootstrap_ci(values: np.ndarray, n_boot: int = 5000, seed: int = 42) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan")
    means = []
    for _ in range(n_boot):
        sample = rng.choice(values, size=values.size, replace=True)
        means.append(np.mean(sample))
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)
