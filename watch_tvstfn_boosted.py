from __future__ import annotations

import os
import re
import sys
import time
import subprocess
from pathlib import Path

import pandas as pd


ROOT = Path('/data/workplace/jwx/TV-STFN')
RESULTS_DIR = Path('/data/workplace/jwx/结果')
LOG_DIR = ROOT / 'cv_logs_20260309_boosted'
BENCH_DIR = ROOT / 'benchmark_results'
REPORT_PATH = RESULTS_DIR / '最终统一十折结果报告.md'
STATUS_MD = RESULTS_DIR / 'TV-STFN增强版运行状态.md'
FINAL_MD = RESULTS_DIR / 'TV-STFN增强版最终结果.md'
STATUS_TXT = LOG_DIR / 'watcher_status.txt'
MERGE_SCRIPT = ROOT / 'merge_tvstfn_boosted_results.py'
PYTHON = ROOT / '.venv' / 'bin' / 'python'
WATCH_SECONDS = 60

GPU_TAGS = [
    ('1', 'boost_gpu1_folds0_3'),
    ('2', 'boost_gpu2_folds4_6'),
    ('4', 'boost_gpu4_folds7_9'),
]


def read_text(path: Path) -> str:
    if not path.exists():
        return ''
    return path.read_text(encoding='utf-8', errors='ignore')


def parse_top_baseline(report_text: str) -> dict[str, float | str] | None:
    pattern = re.compile(
        r"\|\s*1\s*\|\s*(?P<model>[^|]+?)\s*\|\s*[^|]+\|\s*"
        r"(?P<acc>[0-9.]+)\s*±\s*[0-9.]+\s*\|\s*"
        r"(?P<f1>[0-9.]+)\s*±\s*[0-9.]+\s*\|\s*"
        r"(?P<mcc>[0-9.]+)\s*±\s*[0-9.]+\s*\|\s*"
        r"(?P<auroc>[0-9.]+)\s*±\s*[0-9.]+\s*\|\s*"
        r"(?P<auprc>[0-9.]+)\s*±\s*[0-9.]+"
    )
    match = pattern.search(report_text)
    if not match:
        return None
    data = match.groupdict()
    return {
        'model': data['model'].strip(),
        'ACC': float(data['acc']),
        'F1': float(data['f1']),
        'MCC': float(data['mcc']),
        'AUROC': float(data['auroc']),
        'AUPRC': float(data['auprc']),
    }


def summarize_log(log_path: Path) -> dict[str, str]:
    text = read_text(log_path)
    if not text:
        return {'state': 'waiting', 'latest': '日志未生成'}

    latest_epoch = ''
    latest_fold = ''
    latest_result = ''

    epoch_matches = re.findall(r'(Epoch\s+\d+\s*\|[^\n]+)', text)
    if epoch_matches:
        latest_epoch = epoch_matches[-1].strip()

    fold_matches = re.findall(r'(--- Fold\s+\d+/\d+\s*\([^\n]+\) ---)', text)
    if fold_matches:
        latest_fold = fold_matches[-1].strip()

    result_matches = re.findall(r'(Fold\s+\d+\s*\|\s*RMSE=[^\n]+)', text)
    if result_matches:
        latest_result = result_matches[-1].strip()

    if 'Traceback' in text or 'RuntimeError' in text:
        state = 'error'
    elif 'Total time:' in text or 'Summary saved to' in text:
        state = 'finished'
    elif latest_epoch or latest_fold:
        state = 'running'
    else:
        state = 'starting'

    latest_parts = [part for part in [latest_fold, latest_epoch, latest_result] if part]
    return {
        'state': state,
        'latest': ' | '.join(latest_parts) if latest_parts else '已启动，等待更多输出',
    }


def training_processes_alive() -> bool:
    try:
        out = subprocess.check_output(
            [
                'bash', '-lc',
                "pgrep -af 'benchmark_tvstfn_fast.py --folds (0-3|4-6|7-9) --tag boost_gpu' || true"
            ],
            text=True,
        )
    except subprocess.CalledProcessError:
        return False
    return bool(out.strip())


def merge_results() -> tuple[Path | None, Path | None, str]:
    proc = subprocess.run(
        [str(PYTHON), str(MERGE_SCRIPT)],
        cwd=str(ROOT),
        text=True,
        capture_output=True,
    )
    output = (proc.stdout or '') + ('\n' + proc.stderr if proc.stderr else '')
    if proc.returncode != 0:
        return None, None, output.strip()
    per_fold = BENCH_DIR / 'tvstfn_benchmark_per_fold_boosted_merged.csv'
    summary = BENCH_DIR / 'tvstfn_benchmark_summary_boosted_merged.csv'
    return per_fold, summary, output.strip()


def load_summary_metrics(summary_path: Path) -> dict[str, float]:
    df = pd.read_csv(summary_path)
    return {str(row['Metric']): float(row['Mean']) for _, row in df.iterrows()}


def build_status_md(top_baseline: dict[str, float | str] | None, final_done: bool = False) -> str:
    lines = [
        '# TV-STFN增强版运行状态',
        '',
        f'- 更新时间：{time.strftime("%Y-%m-%d %H:%M:%S")}',
        f'- 监控目录：{LOG_DIR}',
        f'- 训练是否仍在运行：{"是" if training_processes_alive() else "否"}',
        f'- 自动合并状态：{"已完成" if final_done else "等待训练结束"}',
        '',
        '## 分卡进度',
    ]
    for gpu, tag in GPU_TAGS:
        info = summarize_log(LOG_DIR / f'{tag}.log')
        lines.append(f'- GPU {gpu} / {tag}：{info["state"]}；{info["latest"]}')

    if top_baseline:
        lines += [
            '',
            '## 当前需要超越的旧榜首',
            f'- 模型：{top_baseline["model"]}',
            f'- ACC：{top_baseline["ACC"]:.4f}',
            f'- F1：{top_baseline["F1"]:.4f}',
            f'- MCC：{top_baseline["MCC"]:.4f}',
            f'- AUROC：{top_baseline["AUROC"]:.4f}',
            f'- AUPRC：{top_baseline["AUPRC"]:.4f}',
        ]

    return '\n'.join(lines) + '\n'


def build_final_md(top_baseline: dict[str, float | str] | None, metrics: dict[str, float], merge_output: str) -> str:
    lines = [
        '# TV-STFN增强版最终结果',
        '',
        f'- 生成时间：{time.strftime("%Y-%m-%d %H:%M:%S")}',
        f'- 汇总文件：{BENCH_DIR / "tvstfn_benchmark_summary_boosted_merged.csv"}',
        f'- 分折文件：{BENCH_DIR / "tvstfn_benchmark_per_fold_boosted_merged.csv"}',
        '',
        '## TV-STFN增强版指标',
    ]
    for key in ['ACC', 'F1', 'MCC', 'AUROC', 'AUPRC', 'RMSE', 'R2', 'Spearman']:
        if key in metrics:
            lines.append(f'- {key}：{metrics[key]:.4f}')

    if top_baseline:
        lines += ['', '## 相对旧榜首对比']
        for key in ['ACC', 'F1', 'MCC', 'AUROC', 'AUPRC']:
            if key in metrics:
                delta = metrics[key] - float(top_baseline[key])
                sign = '+' if delta >= 0 else ''
                lines.append(
                    f'- {key}：{metrics[key]:.4f} vs {top_baseline["model"]} {float(top_baseline[key]):.4f}（{sign}{delta:.4f}）'
                )
        acc_ok = metrics.get('ACC', 0.0) > float(top_baseline['ACC'])
        lines += ['', '## 结论']
        if acc_ok:
            lines.append(f'- 已在 ACC 上超过旧榜首 {top_baseline["model"]}。')
        else:
            lines.append(f'- 尚未在 ACC 上超过旧榜首 {top_baseline["model"]}，但增强版结果已完成，可继续据此二次加速调参。')

    if merge_output:
        lines += ['', '## 合并日志', '```', merge_output.strip(), '```']

    return '\n'.join(lines) + '\n'


def main() -> int:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    top_baseline = parse_top_baseline(read_text(REPORT_PATH))

    merged_once = False
    while True:
        STATUS_TXT.write_text(time.strftime('%Y-%m-%d %H:%M:%S') + '\n', encoding='utf-8')
        STATUS_MD.write_text(build_status_md(top_baseline, final_done=merged_once), encoding='utf-8')

        dispatcher = read_text(LOG_DIR / 'dispatcher.log')
        done = 'ALL DONE' in dispatcher
        alive = training_processes_alive()

        if done or (not alive and 'START' in dispatcher):
            per_fold, summary, merge_output = merge_results()
            if summary and summary.exists():
                metrics = load_summary_metrics(summary)
                FINAL_MD.write_text(build_final_md(top_baseline, metrics, merge_output), encoding='utf-8')
                STATUS_MD.write_text(build_status_md(top_baseline, final_done=True), encoding='utf-8')
                return 0
            FINAL_MD.write_text(
                '# TV-STFN增强版最终结果\n\n'
                f'- 生成时间：{time.strftime("%Y-%m-%d %H:%M:%S")}\n'
                '- 结果合并失败，请检查下面输出。\n\n'
                '```\n'
                f'{merge_output.strip()}\n'
                '```\n',
                encoding='utf-8',
            )
            return 1

        time.sleep(WATCH_SECONDS)


if __name__ == '__main__':
    raise SystemExit(main())