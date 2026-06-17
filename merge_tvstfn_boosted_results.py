import glob
from pathlib import Path
import pandas as pd

ROOT = Path('/data/workplace/jwx/TV-STFN')
OUT = ROOT / 'benchmark_results'
pattern = str(OUT / 'tvstfn_benchmark_per_fold_boost_gpu*_folds*.csv')
files = sorted(glob.glob(pattern))
if not files:
    raise SystemExit('no boosted per-fold files found')
frames = [pd.read_csv(f) for f in files]
merged = pd.concat(frames, ignore_index=True).sort_values('Fold').drop_duplicates(subset=['Fold'], keep='last')
summary = pd.DataFrame({
    'Metric': [c for c in merged.columns if c != 'Fold'],
    'Mean': [merged[c].mean() for c in merged.columns if c != 'Fold'],
    'Std': [merged[c].std(ddof=0) for c in merged.columns if c != 'Fold'],
})
per_fold_path = OUT / 'tvstfn_benchmark_per_fold_boosted_merged.csv'
summary_path = OUT / 'tvstfn_benchmark_summary_boosted_merged.csv'
merged.to_csv(per_fold_path, index=False)
summary.to_csv(summary_path, index=False)
print(per_fold_path)
print(summary_path)
print(summary.to_string(index=False))
