import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, mean_absolute_error, confusion_matrix
import glob
import os
from scipy.stats import spearmanr

csv_files = glob.glob('/data/workplace/jwx/TV-STFN/benchmark_results/fold_*_predictions.csv')

models = ['XGB', 'MSF', 'Cyc']
results = {m: {'acc': [], 'auc': [], 'mae': [], 'spearman': [], 'tnr': []} for m in models}

for file in csv_files:
    df = pd.read_csv(file)
    y_true_cls = df['True_Label'].values
    y_true_reg = df['True_LogP'].values
    
    for m in models:
        # Regression
        y_pred_reg = df[f'Pred_LogP_{m}'].values
        mae = mean_absolute_error(y_true_reg, y_pred_reg)
        sp = spearmanr(y_true_reg, y_pred_reg)[0]
        results[m]['mae'].append(mae)
        results[m]['spearman'].append(sp)
        
        # Classification
        score = df[f'Pred_Score_{m}'].values
        # Threshold at 0.5 for class
        y_pred_cls = (score > 0.5).astype(int)
        
        acc = accuracy_score(y_true_cls, y_pred_cls)
        try:
            auc = roc_auc_score(y_true_cls, score)
        except:
            auc = 0.5 
            
        tn, fp, fn, tp = confusion_matrix(y_true_cls, y_pred_cls).ravel()
        tnr = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        results[m]['acc'].append(acc)
        results[m]['auc'].append(auc)
        results[m]['tnr'].append(tnr)

print("Model\tACC\tAUC\tMAE\tSpearman\tTNR")
for m in models:
    acc = np.mean(results[m]['acc'])
    auc = np.mean(results[m]['auc'])
    mae = np.mean(results[m]['mae'])
    sp = np.mean(results[m]['spearman'])
    tnr = np.mean(results[m]['tnr'])
    print(f"{m}\t{acc:.4f}\t{auc:.4f}\t{mae:.4f}\t{sp:.4f}\t{tnr:.4f}")
