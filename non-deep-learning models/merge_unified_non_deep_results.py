import os
from pathlib import Path
import pandas as pd

BASE = Path('/data/workplace/jwx/TV-STFN/non-deep-learning models/benchmark_results')
files = [
    BASE / 'paper_non_deep_10fold_results_catboost.csv',
    BASE / 'paper_non_deep_10fold_results_xgboost.csv',
    BASE / 'paper_non_deep_10fold_results_lgbm.csv',
    BASE / 'paper_non_deep_10fold_results_cpuclassic.csv',
]
frames = []
for file in files:
    if file.exists() and file.stat().st_size > 0:
        frames.append(pd.read_csv(file))

if not frames:
    raise SystemExit('no non-deep result files found')

merged = pd.concat(frames, ignore_index=True)
merged = merged.drop_duplicates(subset=['Model'], keep='last')
order = ['CatBoost', 'XGBoost', 'KNN', 'LGBM', 'RF', 'SVM (poly)', 'SVM (rbf)', 'DT']
merged['_order'] = merged['Model'].map({m: i for i, m in enumerate(order)})
merged = merged.sort_values('_order').drop(columns=['_order'])
out = BASE / 'paper_non_deep_10fold_results.csv'
merged.to_csv(out, index=False)
print(out)
print(merged.to_string(index=False))
