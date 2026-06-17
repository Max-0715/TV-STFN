import os
import glob
import numpy as np
import pandas as pd
import torch

from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor, LGBMClassifier
from catboost import CatBoostRegressor, CatBoostClassifier


SEED = int(os.getenv("STACK_SEED", "42"))
N_FOLDS = 10
THRESHOLD = -6.0
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, "benchmark_results")
MODEL_FILTER = {m.strip() for m in os.getenv("MODEL_FILTER", "").split(",") if m.strip()}
GPU_ID = os.getenv("GPU_ID", "0")
OUTPUT_TAG = os.getenv("OUTPUT_TAG", "").strip()


def _env_int(name, default):
    return int(os.getenv(name, str(default)))


def _env_float(name, default):
    return float(os.getenv(name, str(default)))


def compute_metrics(y_true_reg, y_pred_reg, y_true_cls, y_pred_cls, y_score_cls):
    y_true_reg = np.asarray(y_true_reg, dtype=float)
    y_pred_reg = np.asarray(y_pred_reg, dtype=float)
    y_true_cls = np.asarray(y_true_cls, dtype=int)
    y_pred_cls = np.asarray(y_pred_cls, dtype=int)
    y_score_cls = np.asarray(y_score_cls, dtype=float)

    mask = (
        np.isfinite(y_true_reg)
        & np.isfinite(y_pred_reg)
        & np.isfinite(y_score_cls)
    )
    y_true_reg = y_true_reg[mask]
    y_pred_reg = y_pred_reg[mask]
    y_true_cls = y_true_cls[mask]
    y_pred_cls = y_pred_cls[mask]
    y_score_cls = y_score_cls[mask]

    rmse = float(np.sqrt(mean_squared_error(y_true_reg, y_pred_reg)))
    mse = float(mean_squared_error(y_true_reg, y_pred_reg))
    mae = float(mean_absolute_error(y_true_reg, y_pred_reg))
    mape = float(mean_absolute_percentage_error(y_true_reg, y_pred_reg) * 100.0)
    r2 = float(r2_score(y_true_reg, y_pred_reg))
    pearson = float(pearsonr(y_true_reg, y_pred_reg)[0]) if len(np.unique(y_pred_reg)) > 1 else 0.0
    spearman = float(spearmanr(y_true_reg, y_pred_reg)[0]) if len(np.unique(y_pred_reg)) > 1 else 0.0

    acc = float(accuracy_score(y_true_cls, y_pred_cls))
    precision = float(precision_score(y_true_cls, y_pred_cls, zero_division=0))
    recall = float(recall_score(y_true_cls, y_pred_cls, zero_division=0))
    f1 = float(f1_score(y_true_cls, y_pred_cls, zero_division=0))
    mcc = float(matthews_corrcoef(y_true_cls, y_pred_cls))
    kappa_num = 2 * (confusion_matrix(y_true_cls, y_pred_cls, labels=[0, 1])[0, 0] * confusion_matrix(y_true_cls, y_pred_cls, labels=[0, 1])[1, 1] - confusion_matrix(y_true_cls, y_pred_cls, labels=[0, 1])[0, 1] * confusion_matrix(y_true_cls, y_pred_cls, labels=[0, 1])[1, 0])
    cm = confusion_matrix(y_true_cls, y_pred_cls, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    denom = (tn + fp) * (fp + tp) + (tn + fn) * (fn + tp)
    kappa = float(kappa_num / denom) if denom else 0.0
    tnr = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    auroc = float(roc_auc_score(y_true_cls, y_score_cls)) if len(np.unique(y_true_cls)) > 1 else 0.0
    auprc = float(average_precision_score(y_true_cls, y_score_cls)) if len(np.unique(y_true_cls)) > 1 else 0.0

    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2,
        "MAPE": mape,
        "Pearson": pearson,
        "Spearman": spearman,
        "ACC": acc,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "MCC": mcc,
        "Kappa": kappa,
        "TNR": tnr,
        "AUROC": auroc,
        "AUPRC": auprc,
    }


def summarize(metric_list):
    keys = metric_list[0].keys()
    rows = []
    for key in keys:
        vals = np.asarray([m[key] for m in metric_list], dtype=float)
        rows.append({"Metric": key, "Mean": vals.mean(), "Std": vals.std(ddof=0)})
    return pd.DataFrame(rows)


def load_unified_splits(n_samples):
    split_path = os.path.join(OUT_DIR, "unified_kfold_indices_seed42.npz")
    data = np.load(split_path, allow_pickle=True)
    splits = []
    for fold in range(N_FOLDS):
        train_idx = np.asarray(data[f"fold{fold}_train"], dtype=int)
        test_idx = np.asarray(data[f"fold{fold}_test"], dtype=int)
        train_idx = train_idx[train_idx < n_samples]
        test_idx = test_idx[test_idx < n_samples]
        splits.append((train_idx, test_idx))
    return splits


def _make_occurrence_keys(df, smiles_col, logp_col):
    out = df.copy()
    out["_smiles_key"] = out[smiles_col].astype(str)
    out["_logp_key"] = pd.to_numeric(out[logp_col], errors="coerce").round(6)
    out["_occ"] = out.groupby(["_smiles_key", "_logp_key"]).cumcount()
    return out


def load_meta_features():
    csv_path = os.path.join(BASE_DIR, "CycPeptMPDB_Peptide_PAMPA.csv")
    raw_df = pd.read_csv(csv_path, low_memory=False)
    raw_df["Permeability"] = pd.to_numeric(raw_df["Permeability"], errors="coerce")
    raw_df = raw_df.dropna(subset=["Permeability", "SMILES"]).reset_index(drop=True)

    cache_path = os.path.join(os.path.dirname(BASE_DIR), "cached_data_tv.pt")
    cache = torch.load(cache_path, weights_only=False)
    desc = np.asarray(cache["descriptors"], dtype=np.float32)

    n = min(len(raw_df), len(desc))
    raw_df = raw_df.iloc[:n].copy().reset_index(drop=True)
    desc = desc[:n]

    splits = load_unified_splits(n)
    oof_pred_logp = np.full(n, np.nan, dtype=np.float32)
    oof_pred_score = np.full(n, np.nan, dtype=np.float32)

    for fold, (_, test_idx) in enumerate(splits):
        pred_path = os.path.join(OUT_DIR, f"fold_{fold}_predictions.csv")
        if not os.path.exists(pred_path):
            raise RuntimeError(f"missing fold prediction file: {pred_path}")
        pred_df = pd.read_csv(pred_path)
        if len(pred_df) != len(test_idx):
            raise RuntimeError(
                f"fold {fold} prediction row count mismatch: got {len(pred_df)} expected {len(test_idx)}"
            )
        oof_pred_logp[test_idx] = pd.to_numeric(pred_df["Pred_LogP_TVSTFN"], errors="coerce").to_numpy(dtype=np.float32)
        oof_pred_score[test_idx] = pd.to_numeric(pred_df["Pred_Score_TVSTFN"], errors="coerce").to_numpy(dtype=np.float32)

    if np.isnan(oof_pred_logp).any() or np.isnan(oof_pred_score).any():
        raise RuntimeError("failed to reconstruct full OOF predictions from fold files")

    y_reg = raw_df["Permeability"].to_numpy(dtype=np.float32)
    y_cls = (y_reg >= THRESHOLD).astype(int)
    meta = np.column_stack([oof_pred_logp, oof_pred_score]).astype(np.float32)
    X = np.concatenate([desc, meta], axis=1)
    return X, y_reg, y_cls


def build_models():
    cb_iters = _env_int("CB_ITERS", 500)
    cb_lr = _env_float("CB_LR", 0.03)
    cb_depth = _env_int("CB_DEPTH", 7)

    xgb_estimators = _env_int("XGB_EST", 500)
    xgb_lr = _env_float("XGB_LR", 0.03)
    xgb_depth = _env_int("XGB_DEPTH", 7)
    xgb_subsample = _env_float("XGB_SUBSAMPLE", 0.9)
    xgb_colsample = _env_float("XGB_COLSAMPLE", 0.9)
    xgb_lambda = _env_float("XGB_LAMBDA", 1.5)

    lgbm_estimators = _env_int("LGBM_EST", 700)
    lgbm_lr = _env_float("LGBM_LR", 0.03)
    lgbm_leaves = _env_int("LGBM_LEAVES", 63)
    lgbm_feat_frac = _env_float("LGBM_FEAT_FRAC", 0.9)
    lgbm_bag_frac = _env_float("LGBM_BAG_FRAC", 0.9)

    return {
        "CatBoostStack": (
            CatBoostRegressor(
                iterations=cb_iters,
                learning_rate=cb_lr,
                depth=cb_depth,
                random_seed=SEED,
                verbose=False,
                loss_function="RMSE",
                task_type="GPU",
                devices=GPU_ID,
            ),
            CatBoostClassifier(
                iterations=cb_iters,
                learning_rate=cb_lr,
                depth=cb_depth,
                random_seed=SEED,
                verbose=False,
                loss_function="Logloss",
                task_type="GPU",
                devices=GPU_ID,
            ),
        ),
        "XGBoostStack": (
            XGBRegressor(
                n_estimators=xgb_estimators,
                learning_rate=xgb_lr,
                max_depth=xgb_depth,
                subsample=xgb_subsample,
                colsample_bytree=xgb_colsample,
                reg_lambda=xgb_lambda,
                random_state=SEED,
                n_jobs=8,
                tree_method="hist",
                device=f"cuda:{GPU_ID}",
            ),
            XGBClassifier(
                n_estimators=xgb_estimators,
                learning_rate=xgb_lr,
                max_depth=xgb_depth,
                subsample=xgb_subsample,
                colsample_bytree=xgb_colsample,
                reg_lambda=xgb_lambda,
                random_state=SEED,
                n_jobs=8,
                eval_metric="logloss",
                tree_method="hist",
                device=f"cuda:{GPU_ID}",
            ),
        ),
        "LGBMStack": (
            LGBMRegressor(
                n_estimators=lgbm_estimators,
                learning_rate=lgbm_lr,
                num_leaves=lgbm_leaves,
                feature_fraction=lgbm_feat_frac,
                bagging_fraction=lgbm_bag_frac,
                bagging_freq=1,
                random_state=SEED,
                n_jobs=8,
                verbose=-1,
                device_type="gpu",
                gpu_device_id=int(GPU_ID),
            ),
            LGBMClassifier(
                n_estimators=lgbm_estimators,
                learning_rate=lgbm_lr,
                num_leaves=lgbm_leaves,
                feature_fraction=lgbm_feat_frac,
                bagging_fraction=lgbm_bag_frac,
                bagging_freq=1,
                random_state=SEED,
                n_jobs=8,
                verbose=-1,
                device_type="gpu",
                gpu_device_id=int(GPU_ID),
            ),
        ),
    }


def run_model(name, reg_model, cls_model, X, y_reg, y_cls, splits):
    fold_rows = []
    for fold, (train_idx, test_idx) in enumerate(splits):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train_reg, y_test_reg = y_reg[train_idx], y_reg[test_idx]
        y_train_cls, y_test_cls = y_cls[train_idx], y_cls[test_idx]

        reg_model.fit(X_train, y_train_reg)
        cls_model.fit(X_train, y_train_cls)

        pred_reg = np.asarray(reg_model.predict(X_test), dtype=float)
        if hasattr(cls_model, "predict_proba"):
            score_cls = np.asarray(cls_model.predict_proba(X_test)[:, 1], dtype=float)
        elif hasattr(cls_model, "decision_function"):
            raw = np.asarray(cls_model.decision_function(X_test), dtype=float)
            score_cls = (raw - raw.min()) / (raw.max() - raw.min() + 1e-12)
        else:
            score_cls = np.asarray(cls_model.predict(X_test), dtype=float)
        pred_cls = (score_cls >= 0.5).astype(int)

        metrics = compute_metrics(y_test_reg, pred_reg, y_test_cls, pred_cls, score_cls)
        metrics["Fold"] = fold
        fold_rows.append(metrics)
        print(f"{name} fold {fold}: ACC={metrics['ACC']:.4f} F1={metrics['F1']:.4f} MCC={metrics['MCC']:.4f} AUROC={metrics['AUROC']:.4f}")

    per_fold = pd.DataFrame(fold_rows)[[
        "Fold", "MAE", "MSE", "RMSE", "R2", "MAPE", "Pearson", "Spearman", "ACC", "Precision",
        "Recall", "F1", "MCC", "Kappa", "TNR", "AUROC", "AUPRC"
    ]]
    summary = summarize([{k: row[k] for k in per_fold.columns if k != "Fold"} for _, row in per_fold.iterrows()])
    return per_fold, summary


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    X, y_reg, y_cls = load_meta_features()
    splits = load_unified_splits(len(X))
    models = build_models()
    selected = models.items() if not MODEL_FILTER else [(k, v) for k, v in models.items() if k in MODEL_FILTER]

    if not selected:
        raise SystemExit("no models selected")

    suffix = f"_{OUTPUT_TAG}" if OUTPUT_TAG else ""
    for model_name, (reg_model, cls_model) in selected:
        print(f"Running {model_name} on GPU {GPU_ID} with X shape={X.shape}")
        per_fold, summary = run_model(model_name, reg_model, cls_model, X, y_reg, y_cls, splits)
        per_fold_path = os.path.join(OUT_DIR, f"stack_{model_name}_per_fold{suffix}.csv")
        summary_path = os.path.join(OUT_DIR, f"stack_{model_name}_summary{suffix}.csv")
        per_fold.to_csv(per_fold_path, index=False)
        summary.to_csv(summary_path, index=False)
        print(summary.to_string(index=False))
        print(per_fold_path)
        print(summary_path)


if __name__ == "__main__":
    main()