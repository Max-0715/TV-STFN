import datetime as dt
import os
import re
import subprocess
import time

ROOT = "/data/workplace/jwx/TV-STFN"
REPORT = os.path.join(ROOT, "benchmark_results", "tvstfn_live_status.md")
OVERNIGHT_LOG = os.path.join(ROOT, "cv_logs_20260310_tvstfn_overnight", "tv5overnight_1.log")
FOLLOW_LOG = os.path.join(ROOT, "cv_logs_20260311_followup", "hard_v1_overnight.log")
FOLLOW_DISPATCH = os.path.join(ROOT, "cv_logs_20260311_followup", "followup_dispatcher.log")
POLL = 180


def read_tail(path, n=80):
    if not os.path.exists(path):
        return []
    try:
        out = subprocess.check_output(["bash", "-lc", f"tail -n {n} {path!s}"], text=True)
        return out.splitlines()
    except Exception:
        return []


def latest_fold_line(lines):
    for line in reversed(lines):
        if "Fold" in line and "RMSE=" in line:
            return line.strip()
    for line in reversed(lines):
        if line.startswith("--- Fold"):
            return line.strip()
    return ""


def running(pattern):
    try:
        out = subprocess.check_output(["bash", "-lc", f"pgrep -af \"{pattern}\" || true"], text=True)
        return out.strip()
    except Exception:
        return ""


def gpu_snapshot():
    try:
        out = subprocess.check_output([
            "nvidia-smi",
            "--query-gpu=index,utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ], text=True)
        return out.strip().splitlines()
    except Exception:
        return ["nvidia-smi unavailable"]


def write_report():
    over_lines = read_tail(OVERNIGHT_LOG, 120)
    follow_lines = read_tail(FOLLOW_LOG, 60)
    dispatch_lines = read_tail(FOLLOW_DISPATCH, 40)

    over_state = "running" if running("benchmark_tvstfn_fast.py --n-folds 5 --tag tv5overnight_1") else "not running"
    follow_state = "running" if running("benchmark_tvstfn_fast.py --n-folds 5 --tag hard_v1_overnight") else "not running"

    lines = [
        "# TV-STFN Live Status",
        "",
        f"更新时间: {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Process State",
        f"- overnight tv5overnight_1: {over_state}",
        f"- follow-up hard_v1_overnight: {follow_state}",
        "",
        "## Latest Progress",
        f"- overnight latest: {latest_fold_line(over_lines) or 'N/A'}",
        f"- follow-up latest: {latest_fold_line(follow_lines) or 'N/A'}",
        "",
        "## Follow-up Dispatcher Tail",
        "```",
        *dispatch_lines[-20:],
        "```",
        "",
        "## GPU Snapshot",
        "```",
        *gpu_snapshot(),
        "```",
    ]

    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    while True:
        write_report()
        time.sleep(POLL)


if __name__ == "__main__":
    main()
