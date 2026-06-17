import csv
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON = os.path.join(BASE_DIR, '.venv', 'bin', 'python')
SCRIPT = os.path.join(BASE_DIR, 'benchmark_tvstfn_fast.py')
LOG_DIR = os.path.join(BASE_DIR, 'cv_logs_20260309_tvstfn_autosearch')
RESULT_DIR = os.path.join(BASE_DIR, 'benchmark_results')
LEADERBOARD = os.path.join(RESULT_DIR, 'tvstfn_5fold_autosearch_leaderboard.csv')
LEADERBOARD_COLUMNS = [
    'tag', 'gpu_id', 'status', 'elapsed_sec', 'ACC', 'MCC', 'F1', 'AUROC', 'AUPRC',
    'summary_path', 'per_fold_path', 'log_path',
    'name', 'epochs', 'batch_size', 'lr', 'early_stop', 'lambda_focal', 'lambda_rank',
    'lambda_mse', 'val_ratio', 'seed', 'tune', 'calibrate'
]

GPU_IDS = [1, 2, 3, 4]

SEARCH_SPACE = [
    {'name': 's1', 'epochs': 28, 'batch_size': 24, 'lr': 1.0e-4, 'early_stop': 8, 'lambda_focal': 1.0, 'lambda_rank': 0.10, 'lambda_mse': 1.0, 'val_ratio': 0.10, 'seed': 42, 'tune': True, 'calibrate': False},
    {'name': 's2', 'epochs': 32, 'batch_size': 24, 'lr': 8.0e-5, 'early_stop': 10, 'lambda_focal': 1.2, 'lambda_rank': 0.15, 'lambda_mse': 0.8, 'val_ratio': 0.10, 'seed': 42, 'tune': True, 'calibrate': False},
    {'name': 's3', 'epochs': 36, 'batch_size': 20, 'lr': 7.0e-5, 'early_stop': 10, 'lambda_focal': 1.4, 'lambda_rank': 0.15, 'lambda_mse': 0.7, 'val_ratio': 0.12, 'seed': 42, 'tune': True, 'calibrate': False},
    {'name': 's4', 'epochs': 32, 'batch_size': 28, 'lr': 1.2e-4, 'early_stop': 8, 'lambda_focal': 0.9, 'lambda_rank': 0.08, 'lambda_mse': 1.1, 'val_ratio': 0.10, 'seed': 42, 'tune': True, 'calibrate': False},
    {'name': 's5', 'epochs': 40, 'batch_size': 24, 'lr': 6.0e-5, 'early_stop': 12, 'lambda_focal': 1.5, 'lambda_rank': 0.20, 'lambda_mse': 0.7, 'val_ratio': 0.12, 'seed': 7, 'tune': True, 'calibrate': False},
    {'name': 's6', 'epochs': 34, 'batch_size': 24, 'lr': 9.0e-5, 'early_stop': 9, 'lambda_focal': 1.1, 'lambda_rank': 0.12, 'lambda_mse': 0.9, 'val_ratio': 0.10, 'seed': 2026, 'tune': True, 'calibrate': False},
    {'name': 's7', 'epochs': 42, 'batch_size': 20, 'lr': 5.0e-5, 'early_stop': 12, 'lambda_focal': 1.8, 'lambda_rank': 0.20, 'lambda_mse': 0.6, 'val_ratio': 0.15, 'seed': 42, 'tune': True, 'calibrate': True},
    {'name': 's8', 'epochs': 30, 'batch_size': 32, 'lr': 1.0e-4, 'early_stop': 8, 'lambda_focal': 1.0, 'lambda_rank': 0.05, 'lambda_mse': 1.2, 'val_ratio': 0.10, 'seed': 42, 'tune': True, 'calibrate': False},
    {'name': 's9', 'epochs': 36, 'batch_size': 24, 'lr': 8.0e-5, 'early_stop': 10, 'lambda_focal': 1.3, 'lambda_rank': 0.10, 'lambda_mse': 0.9, 'val_ratio': 0.08, 'seed': 77, 'tune': True, 'calibrate': False},
    {'name': 's10', 'epochs': 45, 'batch_size': 16, 'lr': 6.0e-5, 'early_stop': 12, 'lambda_focal': 1.6, 'lambda_rank': 0.18, 'lambda_mse': 0.7, 'val_ratio': 0.12, 'seed': 123, 'tune': True, 'calibrate': False},
    {'name': 's11', 'epochs': 28, 'batch_size': 24, 'lr': 1.5e-4, 'early_stop': 8, 'lambda_focal': 0.8, 'lambda_rank': 0.10, 'lambda_mse': 1.0, 'val_ratio': 0.10, 'seed': 42, 'tune': True, 'calibrate': False},
    {'name': 's12', 'epochs': 38, 'batch_size': 20, 'lr': 7.0e-5, 'early_stop': 11, 'lambda_focal': 1.4, 'lambda_rank': 0.12, 'lambda_mse': 0.8, 'val_ratio': 0.10, 'seed': 314, 'tune': True, 'calibrate': True},
]


def load_metrics(summary_path):
    df = pd.read_csv(summary_path)
    metric_map = {row['Metric']: row['Mean'] for _, row in df.iterrows()}
    return metric_map


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
    tag = f"tv5_{cfg['name']}_gpu{gpu_id}"
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
        '--num-workers', '8',
        '--skip-fold-predictions',
    ]
    if cfg.get('tune', False):
        cmd.append('--tune-cls-threshold')
    if cfg.get('calibrate', False):
        cmd.append('--calibrate')

    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    start = time.time()
    with open(log_path, 'w') as logf:
        cmd_str = ' '.join(cmd)
        logf.write(f"GPU={gpu_id}\nCFG={cfg}\nCMD={cmd_str}\n\n")
        logf.flush()
        proc = subprocess.run(cmd, cwd=BASE_DIR, env=env, stdout=logf, stderr=subprocess.STDOUT)

    elapsed = time.time() - start
    if proc.returncode != 0:
        row = {
            'tag': tag,
            'gpu_id': gpu_id,
            'status': 'failed',
            'elapsed_sec': round(elapsed, 1),
            'ACC': '', 'MCC': '', 'F1': '', 'AUROC': '', 'AUPRC': '',
            'summary_path': summary_path,
            'per_fold_path': per_fold_path,
            'log_path': log_path,
            **cfg,
        }
        append_leaderboard(row)
        return row

    metrics = load_metrics(summary_path)
    row = {
        'tag': tag,
        'gpu_id': gpu_id,
        'status': 'ok',
        'elapsed_sec': round(elapsed, 1),
        'ACC': metrics.get('ACC', ''),
        'MCC': metrics.get('MCC', ''),
        'F1': metrics.get('F1', ''),
        'AUROC': metrics.get('AUROC', ''),
        'AUPRC': metrics.get('AUPRC', ''),
        'summary_path': summary_path,
        'per_fold_path': per_fold_path,
        'log_path': log_path,
        **cfg,
    }
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
