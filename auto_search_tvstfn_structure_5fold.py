import csv
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON = os.path.join(BASE_DIR, '.venv', 'bin', 'python')
SCRIPT = os.path.join(BASE_DIR, 'benchmark_tvstfn_fast.py')
LOG_DIR = os.path.join(BASE_DIR, 'cv_logs_20260310_tvstfn_structure')
RESULT_DIR = os.path.join(BASE_DIR, 'benchmark_results')
LEADERBOARD = os.path.join(RESULT_DIR, 'tvstfn_5fold_structure_leaderboard.csv')
LEADERBOARD_COLUMNS = [
    'tag', 'gpu_id', 'status', 'elapsed_sec', 'ACC', 'MCC', 'F1', 'AUROC', 'AUPRC',
    'summary_path', 'per_fold_path', 'log_path',
    'name', 'epochs', 'batch_size', 'lr', 'early_stop', 'lambda_focal', 'lambda_rank',
    'lambda_mse', 'val_ratio', 'seed', 'model_hidden_dim', 'model_dropout',
    'zero_d_prior', 'cls_skip_weight', 'gate_temperature', 'calibrate'
]

GPU_IDS = [1, 2, 3, 4]

SEARCH_SPACE = [
    {'name': 'm1', 'epochs': 36, 'batch_size': 20, 'lr': 7e-5, 'early_stop': 10, 'lambda_focal': 1.4, 'lambda_rank': 0.15, 'lambda_mse': 0.7, 'val_ratio': 0.12, 'seed': 42, 'model_hidden_dim': 512, 'model_dropout': 0.10, 'zero_d_prior': 1.8, 'cls_skip_weight': 0.35, 'gate_temperature': 0.80, 'calibrate': False},
    {'name': 'm2', 'epochs': 40, 'batch_size': 16, 'lr': 6e-5, 'early_stop': 12, 'lambda_focal': 1.6, 'lambda_rank': 0.18, 'lambda_mse': 0.7, 'val_ratio': 0.12, 'seed': 42, 'model_hidden_dim': 640, 'model_dropout': 0.10, 'zero_d_prior': 2.1, 'cls_skip_weight': 0.40, 'gate_temperature': 0.75, 'calibrate': False},
    {'name': 'm3', 'epochs': 42, 'batch_size': 16, 'lr': 5e-5, 'early_stop': 12, 'lambda_focal': 1.8, 'lambda_rank': 0.20, 'lambda_mse': 0.6, 'val_ratio': 0.15, 'seed': 7, 'model_hidden_dim': 512, 'model_dropout': 0.06, 'zero_d_prior': 2.4, 'cls_skip_weight': 0.50, 'gate_temperature': 0.68, 'calibrate': True},
    {'name': 'm4', 'epochs': 36, 'batch_size': 16, 'lr': 7e-5, 'early_stop': 10, 'lambda_focal': 1.3, 'lambda_rank': 0.12, 'lambda_mse': 0.8, 'val_ratio': 0.10, 'seed': 77, 'model_hidden_dim': 768, 'model_dropout': 0.12, 'zero_d_prior': 1.9, 'cls_skip_weight': 0.42, 'gate_temperature': 0.82, 'calibrate': False},
    {'name': 'm5', 'epochs': 38, 'batch_size': 20, 'lr': 6e-5, 'early_stop': 11, 'lambda_focal': 1.5, 'lambda_rank': 0.15, 'lambda_mse': 0.7, 'val_ratio': 0.10, 'seed': 123, 'model_hidden_dim': 640, 'model_dropout': 0.08, 'zero_d_prior': 2.2, 'cls_skip_weight': 0.55, 'gate_temperature': 0.70, 'calibrate': False},
    {'name': 'm6', 'epochs': 34, 'batch_size': 24, 'lr': 8e-5, 'early_stop': 9, 'lambda_focal': 1.2, 'lambda_rank': 0.10, 'lambda_mse': 0.9, 'val_ratio': 0.10, 'seed': 2026, 'model_hidden_dim': 512, 'model_dropout': 0.10, 'zero_d_prior': 1.6, 'cls_skip_weight': 0.60, 'gate_temperature': 0.88, 'calibrate': False},
    {'name': 'm7', 'epochs': 42, 'batch_size': 16, 'lr': 5e-5, 'early_stop': 12, 'lambda_focal': 1.7, 'lambda_rank': 0.22, 'lambda_mse': 0.6, 'val_ratio': 0.15, 'seed': 314, 'model_hidden_dim': 768, 'model_dropout': 0.10, 'zero_d_prior': 2.6, 'cls_skip_weight': 0.48, 'gate_temperature': 0.62, 'calibrate': True},
    {'name': 'm8', 'epochs': 32, 'batch_size': 20, 'lr': 9e-5, 'early_stop': 8, 'lambda_focal': 1.1, 'lambda_rank': 0.08, 'lambda_mse': 1.0, 'val_ratio': 0.10, 'seed': 42, 'model_hidden_dim': 640, 'model_dropout': 0.14, 'zero_d_prior': 1.5, 'cls_skip_weight': 0.38, 'gate_temperature': 0.90, 'calibrate': False},
]


def load_metrics(summary_path):
    df = pd.read_csv(summary_path)
    return {row['Metric']: row['Mean'] for _, row in df.iterrows()}


def append_leaderboard(row):
    os.makedirs(RESULT_DIR, exist_ok=True)
    exists = os.path.exists(LEADERBOARD)
    with open(LEADERBOARD, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=LEADERBOARD_COLUMNS)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def run_one(gpu_id, cfg):
    os.makedirs(LOG_DIR, exist_ok=True)
    tag = f"tv5struct_{cfg['name']}_gpu{gpu_id}"
    summary_path = os.path.join(RESULT_DIR, f'tvstfn_benchmark_summary_{tag}.csv')
    per_fold_path = os.path.join(RESULT_DIR, f'tvstfn_benchmark_per_fold_{tag}.csv')
    log_path = os.path.join(LOG_DIR, f'{tag}.log')
    cmd = [
        PYTHON, SCRIPT,
        '--n-folds', '5',
        '--tag', tag,
        '--epochs', str(cfg['epochs']),
        '--batch-size', str(cfg['batch_size']),
        '--lr', str(cfg['lr']),
        '--early-stop', str(cfg['early_stop']),
        '--val-ratio', str(cfg['val_ratio']),
        '--lambda-focal', str(cfg['lambda_focal']),
        '--lambda-rank', str(cfg['lambda_rank']),
        '--lambda-mse', str(cfg['lambda_mse']),
        '--seed', str(cfg['seed']),
        '--model-hidden-dim', str(cfg['model_hidden_dim']),
        '--model-dropout', str(cfg['model_dropout']),
        '--zero-d-prior', str(cfg['zero_d_prior']),
        '--cls-skip-weight', str(cfg['cls_skip_weight']),
        '--gate-temperature', str(cfg['gate_temperature']),
        '--num-workers', '8',
        '--skip-fold-predictions',
        '--tune-cls-threshold',
    ]
    if cfg.get('calibrate', False):
        cmd.append('--calibrate')

    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    start = time.time()
    with open(log_path, 'w') as logf:
        logf.write(f"GPU={gpu_id}\nCFG={cfg}\nCMD={' '.join(cmd)}\n\n")
        logf.flush()
        proc = subprocess.run(cmd, cwd=BASE_DIR, env=env, stdout=logf, stderr=subprocess.STDOUT)
    elapsed = time.time() - start

    row = {
        'tag': tag,
        'gpu_id': gpu_id,
        'status': 'failed' if proc.returncode != 0 else 'ok',
        'elapsed_sec': round(elapsed, 1),
        'ACC': '', 'MCC': '', 'F1': '', 'AUROC': '', 'AUPRC': '',
        'summary_path': summary_path,
        'per_fold_path': per_fold_path,
        'log_path': log_path,
        **cfg,
    }
    if proc.returncode == 0 and os.path.exists(summary_path):
        metrics = load_metrics(summary_path)
        row.update({
            'ACC': metrics.get('ACC', ''),
            'MCC': metrics.get('MCC', ''),
            'F1': metrics.get('F1', ''),
            'AUROC': metrics.get('AUROC', ''),
            'AUPRC': metrics.get('AUPRC', ''),
        })
    append_leaderboard(row)
    return row


def main():
    pending = list(SEARCH_SPACE)
    futures = {}
    with ThreadPoolExecutor(max_workers=len(GPU_IDS)) as pool:
        for gpu_id in GPU_IDS:
            if not pending:
                break
            cfg = pending.pop(0)
            futures[pool.submit(run_one, gpu_id, cfg)] = gpu_id
        while futures:
            for future in as_completed(list(futures.keys())):
                gpu_id = futures.pop(future)
                try:
                    result = future.result()
                    print(f"[done] gpu={gpu_id} tag={result['tag']} status={result['status']} acc={result['ACC']}")
                except Exception as exc:
                    print(f"[error] gpu={gpu_id} exc={exc}")
                if pending:
                    cfg = pending.pop(0)
                    futures[pool.submit(run_one, gpu_id, cfg)] = gpu_id
                break
    if os.path.exists(LEADERBOARD):
        df = pd.read_csv(LEADERBOARD)
        if 'ACC' in df.columns:
            df = df.sort_values(by=['status', 'ACC', 'MCC', 'F1'], ascending=[True, False, False, False])
            df.to_csv(LEADERBOARD, index=False)
            print(df.head(10).to_string(index=False))


if __name__ == '__main__':
    main()
